"""One person picture in, a locked walk (or box-carry) cycle out.

For each frame: the gait's pose -> depth mannequin -> FLUX.2 [klein] draws the person in it -> BiRefNet cuts the
person out -> measured checks and (optional) the vision judge. A frame that fails is drawn again with another seed.
Then the frames are locked together and written as a sheet, GIFs, a JSON and a review sheet.

When every frame is there, the two steps are compared: frame i and frame i + half must lead with different legs.
A frame drawn on the same foot as its partner is drawn again with another seed, the same way.

Not every frame comes out right: the checks catch extra limbs, a turning box, two steps on the same foot and (with
the judge) a third hand, but not a hand drawn oddly, or a hat or pocket that changes. A person looks at the review
sheet and draws the frames they do not like again by number (--only); those get seeds they have not had.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .checks import (FACING_SURE, ma_alternate_reason, ma_alternates, ma_clothes_change, ma_cutout, ma_feet_detail,
                     ma_stray_arm,
                     ma_facing, ma_frame_ok)
from .comfy import (ARMS_NEAR_BACK, ARMS_NEAR_BACK_HALF, CARRY, EXTRA_FIX, FEET_FIX, FEET_HIGH, FEET_LOW, FEET_MIN,
                    HAND_FIX,
                    HEAD_FIX, STRAY_BAND, STRAY_FIX, STYLE, VIDEO_H, VIDEO_W, WALK,
                    MaComfy, ma_arm_words,
                    ma_cutout_graph, ma_hand_fix_graph, ma_klein_depth_graph, ma_klein_text_graph, ma_prompt,
                    ma_sharpen_graph, ma_video_graph)
from .gait import LSHO, RSHO, H, W, MaGait, ma_frame_phases, ma_stride_px, ma_walk_points
from .judge import HEAD_FLOOR, HEAD_MIN, HEAD_SLACK, ma_hand_box, ma_hand_boxes
from .lock import ma_colour_match, ma_head_match, ma_head_once, ma_leg_masks, ma_leg_tones, ma_lock, ma_marks
from .mannequin import ma_arms_on_body, ma_box_rect, ma_depth, ma_fit, ma_skirt_at, ma_skirt_flare, ma_skirt_hem, ma_skirt_hem_of, ma_skirt_of
from .sheet import ma_loop_gif, ma_meta, ma_review_sheet, ma_sheet, ma_sprites, ma_strip_gif

NEIGHBOUR_TRIES = 3     # an older man's arm half way back: 12 seeds failed from his picture, 1 in 3 from frame 1
SEEDS = (0, 4, 16, 23, 42, 61)          # offsets from the seed; about one frame in twenty needs a second try
# A head drawn again must not turn away from frame 0: two women's video heads, scored 0.90-0.91 against the picture
# (a bar of 0.92), drawn again came out turned, a tic in the walk; their pixels matched frame 0's head at 0.80-0.84
# where the video's heads around them matched at 0.93-0.99 (11-06). Kept only if it matches frame 0 at least as well
# as the head it replaces, less this.
HEAD_TURN = 0.02


def ma_reference(im: Image.Image) -> Image.Image:
    """The person standing, side view facing right, on white at the frame's size and shape (padded, not stretched)."""
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        rgba = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    im = im.convert("RGB")
    w, h = im.size
    side = int(max(h, w * H / W) * 1.08)            # a little white round the figure
    can = Image.new("RGB", (int(side * W / H), side), "white")
    can.paste(im, ((can.width - w) // 2, (side - h) // 2))
    return can.resize((W, H), Image.LANCZOS)


def ma_face_right(im: Image.Image, mask: np.ndarray | None = None, facing: str = "auto", judge=None,
                  log=print) -> Image.Image:
    """The person picture turned to face right, as the mannequin walks.

    A picture facing left drawn over a mannequin walking right comes out with the head and coat facing left on legs
    walking right, a third hand and steps on the same foot. facing: "right" or "left" says it; "auto" asks the
    judge if there is one, else reads the feet (mask: the person; without it, anything not white)."""
    way = facing
    if way == "auto":
        way = judge.facing(im) if judge is not None else ""
        if not way:
            a = np.asarray(im.convert("RGB")).astype(int)
            m = mask if mask is not None else ((a < 235).any(-1) & ((a.max(-1) - a.min(-1) > 40) | (a.max(-1) < 150)))
            v = ma_facing(m)
            way = "right" if v >= FACING_SURE else "left" if v <= -FACING_SURE else ""
        if not way:
            log("the picture's facing is unclear: taken as facing right (say --facing left if it faces left)")
            way = "right"
    if way == "left":
        log("the picture faces left: mirrored to face right")
        return im.transpose(Image.FLIP_LEFT_RIGHT)
    return im


@dataclass
class MaRequest:
    reference: Path
    out: Path
    name: str = "person"
    kind: str = "walk"                   # walk | carry
    look: str = ""                       # the person in words: hair, clothes, shoes
    change: str = ""                     # the one thing to draw differently from the picture
    gait: MaGait = field(default_factory=MaGait)
    frames: int = 12                   # 8 read as jumps (a hand 112 px a frame); 12 walked smoothly (11-03)
    seed: int = 7
    # the mannequin's shape (ma_depth), each None: measured on the person's picture (ma_shape)
    head: float | None = None            # head width / height of the person in the picture
    bust: float | None = None
    body: float | None = None            # how thick the body is, front to back
    skirt: float | None = None           # where a skirt ends along the leg; whether there is one from the look
    height: int | None = 256             # sprite height in px (None: full size)
    fps: int = 12                      # one cycle a second at 12 frames
    far_shade: float = 0.9               # the far leg's brightness against the near one's (1: legs as drawn)
    colour: float = 0.5                  # how far each frame's colours go to the picture's (0: as drawn)
    head_once: bool = True               # frame 0's head on every frame (ma_head_once); False: each frame's own
    bob: bool = False                    # the head rides the walk's small wave; False: one height (ma_lock)
    # draw again a frame on the same foot as the one half a cycle away. Off: the model draws the leg behind darker
    # whichever it is, so new seeds rarely help (0 of 18 on Rin's walk); the leg tones fix what shows.
    alternate: bool = False
    facing: str = "auto"                 # which way the person picture faces: auto | right | left (mirrored)
    # video: frame 0 drawn, the whole cycle drawn from it by the video model, every frame made sharp by the frame
    # model (ma_video_graph, ma_sharpen_graph); frames: every frame drawn on its own from the picture
    engine: str = "video"
    video_model: str = "14b"             # 14b | 5b (ma_video_graph)


def ma_draw_cycle(req: MaRequest, comfy: MaComfy, judge=None, only: list[int] | None = None, log=print,
                  use_seed: int | None = None) -> dict:
    """Draw, check and lock one cycle. Returns a report of every frame (seed, what was measured, the judge).

    use_seed: draw the `only` frames with exactly this seed (the same seed gives the same picture), e.g. to go back
    to an earlier try that looked better than the new ones."""
    out = Path(req.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    ref = ma_reference(Image.open(req.reference))
    if req.facing != "right":
        mask = None
        if req.facing == "auto" and judge is None:
            m = comfy.run(ma_cutout_graph(comfy.upload(ref, f"{req.name}_ref_in.png")), raw / "reference_mask.png")
            mask = np.asarray(Image.open(m).convert("L").resize(ref.size)) > 128
        ref = ma_face_right(ref, mask, req.facing, judge, log)
    ref.save(raw / "reference.png")
    if not (raw / "shape.json").exists():
        shape = ma_measure_shape(req, ref, judge)
        log("shape from the picture: " + ", ".join(f"{k} {v}" for k, v in shape.items()))
        (raw / "shape.json").write_text(json.dumps(shape) + "\n", encoding="utf-8")
    shape = ma_shape(req)
    words = ma_look_words(req, ref, judge, raw, log)
    ref_im = ref                                        # the person, for the head likeness
    ref_name = comfy.upload(ref, f"{req.name}_ref.png")
    carry = req.kind == "carry"
    phases = ma_frame_phases(req.frames, req.gait)
    pts = [ma_arms_on_body(ma_walk_points(t, req.gait, carry=carry), shape["torso"]) for t in phases]
    depths = [ma_depth(p, box=carry, **shape) for p in pts]
    widest = max(abs(p[LSHO + 1][0] - p[LSHO][0]) for p in pts)
    prompts = [ma_prompt(words, req.kind, change=req.change, arms="" if carry else ma_arm_words(p, widest))
               for p in pts]
    # where a hand must be drawn: the near one always, the far one when it swings out in front of the body
    # and only where the mannequin shows it: half behind the body or a skirt the video rightly drew none, and every
    # seed failed it (a girl's far hand at the half swing, 66-79 % behind her skirt, two frames of twelve, 11-05)
    hands = [{} for _ in pts]
    for h, p, d in zip([] if carry else hands, pts, depths):
        h["near"] = ma_hand_box(p, LSHO + 2)
        far = ma_hand_box(p, RSHO + 2)
        if ma_arm_words(p, widest) in (ARMS_NEAR_BACK, ARMS_NEAR_BACK_HALF) and ma_hidden(d, far) < 0.5:
            h["far"] = far
    tries = len(SEEDS) if judge is not None else 3
    path = out / f"{req.kind}_report.json"
    report = {r["frame"]: r for r in (json.loads(path.read_text(encoding="utf-8")) if path.exists() else [])}
    dnames = {}
    head_min = [HEAD_MIN]
    seen = {}                            # frame: [(head likeness, seed)] of the seeds that passed every other check

    def draw(i, sd, ref=None, start=None, mask=None, denoise=HAND_FIX):
        """Frame i at seed sd, drawn (or, given a start picture, made sharp from it; given a mask too, drawn again
        inside it only), cut out and checked: (passes, what was measured, the judge's answer)."""
        if i not in dnames:
            dnames[i] = comfy.upload(depths[i], f"{req.name}_{req.kind}_depth_{i}.png")
            depths[i].save(raw / f"{req.kind}_{i}_depth.png")
        if mask:
            graph = ma_hand_fix_graph(dnames[i], ref or ref_name, start, mask, prompts[i], seed=sd, denoise=denoise)
        elif start:
            graph = ma_sharpen_graph(dnames[i], ref or ref_name, start, prompts[i], seed=sd)
        else:
            graph = ma_klein_depth_graph(dnames[i], ref or ref_name, prompts[i], seed=sd)
        frame = comfy.run(graph, raw / f"{req.kind}_{i}.png")
        fname = comfy.upload(Image.open(frame).convert("RGB"), f"{req.name}_{req.kind}_{i}_frame.png")
        mask = comfy.run(ma_cutout_graph(fname), raw / f"{req.kind}_{i}_mask.png")
        rgb = np.asarray(Image.open(frame).convert("RGB"))
        m = np.asarray(Image.open(mask).convert("L").resize((rgb.shape[1], rgb.shape[0]))) > 128
        ok, said = ma_frame_ok(rgb, m, depths[i], pts[i], req.kind)
        verdict = ""
        if ok and judge is not None:
            ok, verdict = judge.arms_ok(Image.open(frame), req.kind, hands[i])
            if ok:
                ok, feet = judge.feet_ok(Image.open(frame))
                verdict += f"; shoes {(re.findall(r'[0-9]+', feet) or ['?'])[-1]}"
            if ok:
                likeness = getattr(judge, "head_likeness", None)      # a judge of one's own may not have it
                like = likeness(Image.open(frame), ref_im) if likeness else None
                if like is not None:
                    seen.setdefault(i, []).append((like, sd))
                    ok = like >= head_min[0]
                    verdict += f"; head {like:.2f}" + ("" if ok else " (another person)")
        log(f"frame {i} seed {sd}: {said}" + (f", judge {verdict!r}" if verdict else "") + (" ok" if ok else " again"))
        return ok, said, verdict

    def cut(i):
        return ma_cutout(Image.open(raw / f"{req.kind}_{i}.png"), Image.open(raw / f"{req.kind}_{i}_mask.png"))

    def feet(i):
        """Frame i's fine lines on the feet (ma_feet_detail) and its person mask."""
        g = np.asarray(Image.open(raw / f"{req.kind}_{i}.png").convert("L"))
        m = np.asarray(Image.open(raw / f"{req.kind}_{i}_mask.png").convert("L").resize(g.shape[::-1])) > 128
        return ma_feet_detail(g, m), m

    def feet_pass():
        """Feet smeared or printed scaly against the cycle's own (under FEET_LOW or over FEET_HIGH of the median):
        drawn again in a box round the feet from frame 0's look, kept only if they come out like the rest and the
        frame still passes."""
        detail = [feet(i)[0] for i in range(req.frames)]
        mid = float(np.median(detail))
        if mid < FEET_MIN:
            return
        for i in drawn:
            r = detail[i] / mid
            if FEET_LOW <= r <= FEET_HIGH:
                continue
            row = report[i]
            keep = {p: p.read_bytes() for p in (raw / f"{req.kind}_{i}.png", raw / f"{req.kind}_{i}_mask.png")}
            sharp = comfy.upload(Image.open(raw / f"{req.kind}_{i}.png").convert("RGB"), f"{req.name}_{req.kind}_sharp_{i}.png")
            mask = comfy.upload(ma_feet_mask(feet(i)[1]), f"{req.name}_{req.kind}_feet_{i}.png")
            ok = False
            for sd in [x for x in ma_seeds(req.seed) if x not in row["tried"]][:tries]:
                row["tried"].append(sd)
                ok, said, verdict = draw(i, sd, look or ref_name, start=sharp, mask=mask, denoise=FEET_FIX)
                after = feet(i)[0] / mid
                # the head is asked last: failed on it alone, every other check passed. The feet drawn again do
                # not touch it (the video's own head, 0.90 against a bar of 0.91, sent a girl's mended feet back
                # four times, 11-07) and ma_head_once puts frame 0's head on the frame in the end
                head_only = "(another person)" in verdict
                ok = (ok or head_only) and FEET_LOW <= after <= FEET_HIGH
                log(f"frame {i} seed {sd}: feet {r:.2f} of the cycle's, drawn again {after:.2f}" + (", ok" if ok else ", again"))
                if ok:
                    row.update(seed=sd, passed=not head_only, checks=not head_only, measured=said, judge=verdict,
                               feet=round(after, 2))
                    break
            if not ok:
                for p, data in keep.items():
                    p.write_bytes(data)
                row["feet"] = round(r, 2)
                log(f"frame {i}: feet kept as drawn ({r:.2f} of the cycle's); look at them")
        path.write_text(json.dumps([report[k] for k in sorted(report)], indent=2) + "\n", encoding="utf-8")

    def face0(i):
        """How well frame i's head, as drawn now, matches frame 0's (pixels, ma_head_match)."""
        from .checks import ma_person
        a, b = np.asarray(cut(i)), np.asarray(cut(0))
        pa, pb = ma_person(a[..., 3] > 128), ma_person(b[..., 3] > 128)
        M = ma_marks(pb)
        return ma_head_match(a, pa, b, pb, M["neck"] - M["top"])[0]

    def draw_frame(i):
        """Frame i through its seeds (and the ways back), its report row written."""
        t0 = time.time()
        # drawn again (a person did not like it): never a seed this frame already had, or the same picture comes back
        tried = report.get(i, {}).get("tried", [])
        seeds = [use_seed] if use_seed is not None else [s for s in ma_seeds(req.seed) if s not in tried][:tries]
        ok, frm = False, None
        if i in starts:
            sd = seeds[0]
            tried = tried + ([sd] if sd not in tried else [])
            ok, said, verdict = draw(i, sd, look, start=starts[i])
            # A frame of the video is only ever mended where it failed, never drawn again whole: drawn on its own from
            # the picture, frames came out in another skirt, shorts for a skirt or another cardigan, 4 to 12 frames
            # of twelve on four people of eight (11-05). The video lost a hand: drawn again where the mannequin has
            # it; drew one too many: drawn over where that one is; drew another person's head: the head drawn again.
            # A lost hand is drawn in the mannequin's hand crop alone: grown by 30 % it took in the hem and the pocket
            # by the hip, and an older man's frame came out with another hem than its neighbours (11-04)
            fix, dz, grow = ma_video_fix(verdict, hands[i], pts[i], shape["head"]) if not ok else ({}, HAND_FIX, 0.0)
            frame, fref = Image.open(raw / f"{req.kind}_{i}.png").convert("RGB"), look
            if "stray arm" in said:     # the far arm swung forward where it is behind the body: wiped, edges drawn
                m = np.asarray(Image.open(raw / f"{req.kind}_{i}_mask.png").convert("L").resize(frame.size)) > 128
                frame, band = ma_stray_wipe(frame, m, depths[i], ma_stray_arm(m, depths[i], pts[i]))
                fix, dz, grow, fref = {"stray": band}, STRAY_FIX, 0.0, ref_name
            if fix:
                keep = {p: p.read_bytes() for p in (raw / f"{req.kind}_{i}.png", raw / f"{req.kind}_{i}_mask.png")}
                kept = (sd, said, verdict)
                face = face0(i) if "head" in fix else None
                sharp = comfy.upload(frame, f"{req.name}_{req.kind}_sharp_{i}.png")
                mask = comfy.upload(fix["stray"] if "stray" in fix else ma_hands_mask(fix, grow),
                                    f"{req.name}_{req.kind}_fix_{i}.png")
                for sd in [x for x in ma_seeds(req.seed) if x not in tried][:tries]:
                    tried = tried + [sd]
                    ok, said, verdict = draw(i, sd, fref, start=sharp, mask=mask, denoise=dz)
                    if ok and face is not None and face0(i) < face - HEAD_TURN:
                        ok = False
                        log(f"frame {i} seed {sd}: the head drawn again turned away from frame 0 "
                            f"({face0(i):.2f} < {face:.2f}), again")
                    if ok:
                        log(f"frame {i} seed {sd}: mended ({', '.join(fix)}), ok")
                        break
                    # the stray arm gone and only the head left, the video's own head (the arm made the frame fail
                    # before its head was looked at; a girl's far arm wiped in six seeds of seven, each one then
                    # failed at head 0.90 against her bar of 0.91, 11-06): that head drawn again on the mended frame
                    if "stray" in fix and "stray arm" not in said and "another person" in verdict:
                        log(f"frame {i} seed {sd}: arm mended, the head next")
                        sharp = comfy.upload(Image.open(raw / f"{req.kind}_{i}.png").convert("RGB"),
                                             f"{req.name}_{req.kind}_sharp_{i}.png")
                        fix, dz, fref = {"head": ma_head_box(pts[i], shape["head"])}, HEAD_FIX, look
                        face = face0(i)
                        mask = comfy.upload(ma_hands_mask(fix, 0.0), f"{req.name}_{req.kind}_fix_{i}.png")
                if not ok:              # the video's own frame, made sharp, for a person's look
                    for p, data in keep.items():
                        p.write_bytes(data)
                    sd, said, verdict = kept
            if not ok:
                log(f"frame {i}: kept the video's frame (seed {sd}); look at it")
        else:
            for sd in seeds:
                tried = tried + ([sd] if sd not in tried else [])
                ok, said, verdict = draw(i, sd)
                if ok:
                    break
        # every seed drew the hand where the standing picture has it (an arm half way back): the frame is drawn from
        # a neighbour that passed instead, whose arm is already near; one step only, so the look does not drift
        if not ok and judge is not None and use_seed is None and i not in starts:
            for nb in ((i - 1) % req.frames, (i + 1) % req.frames):
                row = report.get(nb, {})
                if not row.get("passed") or row.get("from_frame") is not None or nb == i:
                    continue
                nref = comfy.upload(Image.open(raw / f"{req.kind}_{nb}.png").convert("RGB"),
                                    f"{req.name}_{req.kind}_ref_{nb}.png")
                for sd in [x for x in ma_seeds(req.seed) if x not in tried][:NEIGHBOUR_TRIES]:
                    tried = tried + [sd]
                    ok, said, verdict = draw(i, sd, nref)
                    if ok:
                        frm = nb
                        log(f"frame {i} seed {sd}: drawn from frame {nb}, ok")
                        break
                if ok:
                    break
        if not ok:
            log(f"frame {i}: every seed failed, look at {raw / f'{req.kind}_{i}.png'}")
        report[i] = dict(frame=i, seed=sd, tried=tried, passed=ok, checks=ok, measured=said, judge=verdict,
                         seconds=round(time.time() - t0),
                         engine="video" if i in starts else "frames",
                         **({"from_frame": frm} if frm is not None else {}))

    drawn = list(only) if only is not None else list(range(req.frames))
    starts, look = {}, None
    if req.engine == "video" and only is None and use_seed is None:
        draw_frame(0)
        # how like the picture this person's own drawn frame 0 came out sets the bar for the rest: a man with a shaved
        # head, the same man in every frame by eye, scored 0.85-0.91 against his picture (a bald crown says little)
        # and failed every seed; a face turned into another person scores 0.51-0.65, hair over the face 0.83-0.87
        # The bar is the middle of every seed of frame 0 that passed all else, not the one seed that got over it: a
        # girl drawn as herself in every frame by eye scored 0.83-0.94 against her picture, frame 0 passed on its
        # fourth seed at 0.94, the bar stayed 0.91 and 42 of 70 tries of her walk went to heads at 0.88-0.91; with
        # the box no seed got over 0.88 and her carry was drawn frame by frame without the video (11-08). Under the
        # bar every time, frame 0 is drawn again at its best seed when that one is over the person's own bar.
        got = [s for s, _ in seen.get(0, [])]
        if got:
            head_min[0] = max(HEAD_FLOOR, min(HEAD_MIN, float(np.median(got)) - HEAD_SLACK))
            best, bsd = max(seen[0])
            if not report[0]["passed"] and best >= head_min[0]:
                ok, said, verdict = draw(0, bsd)
                if ok:
                    report[0].update(seed=bsd, passed=True, checks=True, measured=said, judge=verdict)
                    log(f"frame 0 seed {bsd}: its best head {best:.2f}, over this person's bar {head_min[0]:.2f}, ok")
        if report[0]["passed"]:
            starts = ma_video_starts(req, comfy, depths, raw, log, words)
            # the cycle again from the video's own frame 0 made sharp: from frame 0 as drawn alone the video drew a
            # round pouch on an older man's hip in every frame (his drawing had none), and each frame made sharp
            # read it as a pocket, a flap or the hem; from its own frame 0 it kept his jacket as it was (11-04)
            draw_frame(0)
            if report[0]["passed"]:
                starts = ma_video_starts(req, comfy, depths, raw, log, words)
            # the look of the whole cycle is frame 0's from here on: made sharp, or a hand drawn again, a frame is
            # drawn from frame 0 and not from the standing picture, or each one reads the hem, pockets and cuffs its
            # own way. An older man's walk changed at the hem by 24 between frames drawn from his picture, 22 from
            # frame 0 (the video alone 20); frames with a hand drawn again from his picture by 45-66 (11-04)
            look = comfy.upload(Image.open(raw / f"{req.kind}_0.png").convert("RGB"), f"{req.name}_{req.kind}_look.png")
    for i in drawn:
        draw_frame(i)
    missing = [i for i in range(req.frames) if not (raw / f"{req.kind}_{i}.png").exists()]
    if not missing and judge is not None and use_seed is None:
        feet_pass()
    if not missing and req.alternate:
        cuts = {i: cut(i) for i in range(req.frames)}
        alt = ma_alternates([cuts[i] for i in range(req.frames)], depths)
        # a frame on the same foot as its partner: drawn again with seeds it has not had (not with --use-seed, where
        # the person chose the picture); only frames drawn in this run
        for i in [k for k in alt["fail"] if k in drawn] if use_seed is None else []:
            t0, r = time.time(), report[i]
            keep = {p: p.read_bytes() for p in (raw / f"{req.kind}_{i}.png", raw / f"{req.kind}_{i}_mask.png")}
            pair = next(p for p in alt["pairs"] if i in p["fail"])
            for sd in [s for s in ma_seeds(req.seed) if s not in r["tried"]][:tries]:
                r["tried"].append(sd)
                ok, said, verdict = draw(i, sd)
                if not ok:
                    continue
                cuts[i] = cut(i)
                two = ma_alternates([cuts[k] for k in pair["frames"]], [depths[k] for k in pair["frames"]])
                if pair["frames"].index(i) not in two["fail"]:
                    log(f"frame {i} seed {sd}: leads with the other leg now, ok")
                    report[i].update(seed=sd, passed=True, checks=True, measured=said, judge=verdict)
                    break
                log(f"frame {i} seed {sd}: still on the same foot, again")
            else:
                log(f"frame {i}: no seed led with the other leg, kept seed {r['seed']}; look at it")
                for p, data in keep.items():
                    p.write_bytes(data)
                cuts[i] = cut(i)
            r["seconds"] = r.get("seconds", 0) + round(time.time() - t0)
    rows = [report[k] for k in sorted(report)]
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if missing:
        log(f"frames {missing} not drawn yet: no lock or sheet")
        return dict(report=rows, files={})
    files = ma_finish(req, phases, pts, depths)
    return dict(report=json.loads(path.read_text(encoding="utf-8")), files=files)   # with the two-step check


def ma_extra_hands(boxes: list, pts: dict) -> dict:
    """The hands past the two the mannequin has (boxes in the frame's pixels, from the judge): each wrist takes the
    nearest box, the rest are too many. Each one's box reaches up its own height, over the sleeve it hangs from."""
    if len(boxes) <= 2:
        return {}
    left = list(boxes)
    for wrist in (LSHO + 2, RSHO + 2):
        x0, y0, x1, y1 = ma_hand_box(pts, wrist)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        left.remove(min(left, key=lambda b: ((b[0] + b[2]) / 2 - cx) ** 2 + ((b[1] + b[3]) / 2 - cy) ** 2))
    return {f"extra{k}": (b[0], b[1] - (b[3] - b[1]), b[2], b[3]) for k, b in enumerate(left)}


def ma_hidden(depth: Image.Image, box: tuple[int, int, int, int]) -> float:
    """How much of a box on the mannequin is covered by something nearer than the far limbs (body, skirt, box, the
    near arm and leg), 0 to 1."""
    x0, y0, x1, y1 = box
    d = np.asarray(depth.convert("L"))[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    return float((d > 130).mean()) if d.size else 0.0


def ma_head_box(pts: dict, head: float = 0.19) -> tuple[int, int, int, int]:
    """The mannequin's head with room for the hair around it (frame pixels)."""
    from .gait import FIG, NOSE
    hx, hy, hw = pts[NOSE][0] - 0.05 * FIG, pts[NOSE][1] - 0.01 * FIG, head * FIG
    r, b = hx + 0.065 * FIG, hy + 0.07 * FIG
    return int(r - 1.2 * hw), int(b - 1.25 * hw), int(r + 0.05 * FIG), int(b + 0.02 * FIG)


def ma_video_fix(verdict: str, hands: dict, pts: dict, head: float = 0.19) -> tuple[dict, float, float]:
    """Where a failed video frame is drawn again, how strongly and how far the boxes grow: ({name: box}, denoise,
    grow); no boxes when the judge named nothing to mend (the video's frame is kept then, for a person's look)."""
    if "another person" in verdict:
        return {"head": ma_head_box(pts, head)}, HEAD_FIX, 0.0
    if hands and "missing" in verdict:
        return hands, HAND_FIX, 0.0
    return ma_extra_hands(ma_hand_boxes(verdict), pts), EXTRA_FIX, 0.3


def ma_stray_wipe(frame: Image.Image, mask: np.ndarray, depth: Image.Image,
                  box: tuple[int, int, int, int]) -> tuple[Image.Image, Image.Image]:
    """A stray arm wiped white (the part of its box in front of the mannequin, grown 3 px) and the band round it to
    draw again (white, edges soft): (the wiped frame, the band)."""
    from PIL import ImageFilter
    from scipy import ndimage
    from .checks import ma_person
    p = ma_person(mask)
    man = np.asarray(depth.convert("L").resize(p.shape[::-1])) > 20
    man = ndimage.binary_dilation(man, iterations=max(1, int(0.01 * np.ptp(np.nonzero(p.any(1))[0]))))
    x0, y0, x1, y1 = box
    wipe = np.zeros_like(p)
    wipe[y0:y1, x0:x1] = True
    wipe = ndimage.binary_dilation(wipe & ~man, iterations=3) & ~man
    rgb = np.asarray(frame.convert("RGB")).copy()
    rgb[wipe] = 255
    band = ndimage.binary_dilation(wipe, iterations=STRAY_BAND)
    return (Image.fromarray(rgb),
            Image.fromarray((band * 255).astype(np.uint8)).convert("RGB").filter(ImageFilter.GaussianBlur(4)))


def ma_feet_mask(person: np.ndarray, share: float = 0.15, grow: int = 18) -> Image.Image:
    """White round the feet (the lowest `share` of the figure, the person grown `grow` px), edges soft."""
    from PIL import ImageFilter
    from scipy import ndimage
    from .checks import ma_person
    p = ma_person(person)
    ys = np.nonzero(p.any(1))[0]
    box = np.zeros_like(p)
    box[int(ys.max() - share * (ys.max() - ys.min())):] = True
    box &= ndimage.binary_dilation(p, iterations=grow)
    return Image.fromarray((box * 255).astype(np.uint8)).convert("RGB").filter(ImageFilter.GaussianBlur(4))


def ma_hands_mask(boxes: dict, grow: float = 0.3) -> Image.Image:
    """White where the hands must be (the judge's crops, grown by `grow` each way, edges soft), black elsewhere."""
    from PIL import ImageDraw, ImageFilter
    m = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(m)
    for x0, y0, x1, y1 in boxes.values():
        gx, gy = (x1 - x0) * grow, (y1 - y0) * grow
        d.rectangle([x0 - gx, y0 - gy, x1 + gx, y1 + gy], fill=(255, 255, 255))
    return m.filter(ImageFilter.GaussianBlur(8))


def ma_video_prompt(look: str, kind: str) -> str:
    """The video model's words: the person, walking in place seen from the side (the mannequin walks on the spot)."""
    action = CARRY if kind == "carry" else WALK + ", the arms swinging gently with the walk"
    return (f"{look + ', ' if look else ''}{action}, walking in place, seen exactly from the side facing right, the "
            f"whole figure, the same person in every frame, plain white background, {STYLE}")


def ma_video_starts(req: MaRequest, comfy: MaComfy, depths: list, raw: Path, log=print, look: str | None = None) -> dict:
    """The cycle drawn at once by the video model from frame 0 (drawn and checked), two cycles over the mannequins
    (the first settles in), the second cycle's frames uploaded at full size as each frame's start picture."""
    n = req.frames
    length = 2 * n + 1
    while (length - 1) % 4:                              # the video VAE takes 4k + 1 frames
        length += 1
    names = [comfy.upload(depths[j].resize((VIDEO_W, VIDEO_H), Image.LANCZOS), f"{req.name}_{req.kind}_vdepth_{j}.png")
             for j in range(n)]
    first = comfy.upload(Image.open(raw / f"{req.kind}_0.png").convert("RGB").resize((VIDEO_W, VIDEO_H), Image.LANCZOS),
                         f"{req.name}_{req.kind}_vfirst.png")
    t0 = time.time()
    comfy.free()
    graph = ma_video_graph([names[j % n] for j in range(length)], first, ma_video_prompt(req.look if look is None else look, req.kind),
                           seed=req.seed, model=req.video_model)
    frames = comfy.run_all(graph, raw / "video", stem=req.kind)
    comfy.free()                                         # the frame model again next
    log(f"the cycle drawn as a video: {len(frames)} frames, {round(time.time() - t0)} s")
    return {i: comfy.upload(Image.open(frames[n + i]).convert("RGB").resize((W, H), Image.LANCZOS),
                            f"{req.name}_{req.kind}_vstart_{i}.png") for i in range(n) if n + i < len(frames)}


def ma_new_person(comfy: MaComfy, look: str, out: Path, count: int = 4, seed: int = 7, log=print) -> list[Path]:
    """Candidates for a new character from words, each kept only if it is one whole standing figure (BiRefNet finds
    one piece, head to feet inside the frame). A person picks one and draws from it."""
    from .checks import ma_person
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    kept = []
    for k in range(count):
        p = comfy.run(ma_klein_text_graph(look, seed=seed + k), out / f"person_{seed + k}.png")
        name = comfy.upload(Image.open(p).convert("RGB"), f"person_{seed + k}.png")
        m = np.asarray(Image.open(comfy.run(ma_cutout_graph(name), out / f"person_{seed + k}_mask.png")).convert("L"))
        fig = ma_person(m > 128)
        ys, xs = np.nonzero(fig)
        whole = len(ys) and ys.min() > 4 and ys.max() < m.shape[0] - 4 and fig.sum() > 0.8 * (m > 128).sum()
        tall = len(ys) and (ys.max() - ys.min()) > 1.6 * (xs.max() - xs.min())
        log(f"{p.name}: " + ("ok" if whole and tall else "not one whole standing figure"))
        if whole and tall and ma_facing(fig) <= -FACING_SURE:     # asked for facing right, drawn facing left
            for q in (p, out / f"person_{seed + k}_mask.png"):
                Image.open(q).transpose(Image.FLIP_LEFT_RIGHT).save(q)
            log(f"{p.name}: faced left, mirrored")
        if whole and tall:
            kept.append(p)
    return kept


def ma_seeds(seed: int) -> list[int]:
    """The seeds a frame tries, in order: the usual few first, then fresh ones for frames drawn again."""
    return [seed + off for off in SEEDS] + [seed + 100 + 37 * k for k in range(40)]


def ma_measure_shape(req: MaRequest, ref: Image.Image, judge=None) -> dict:
    """The mannequin's shape from the person's own picture (ma_fit: head, body, bust, torso), what the request does
    not set. A skirt where the words name one and a judge, when there is one, finds it (the words said a skirt where
    the picture has jeans; a judge alone took wide trousers for a skirt); where it ends and how wide it stands
    measured on the picture's pixels (ma_skirt_hem_of, ma_skirt_flare). Heels where the judge sees them on the feet,
    or the words say so (11-05)."""
    from scipy import ndimage
    from .checks import ma_person
    rgb = np.asarray(ref.convert("RGB"))
    person = ndimage.binary_fill_holes(ma_person((rgb < 235).any(-1)))    # a white skirt or shirt is the person
    shape = ma_fit(person)
    skirt, flare = ma_skirt_of(req.look), None
    if skirt and judge is not None and hasattr(judge, "skirt_hem") and judge.skirt_hem(ref) is None:
        skirt = 0.0
    if skirt and person.any():
        hem = ma_skirt_hem_of(person, rgb)
        skirt, flare = ma_skirt_at(hem), round(ma_skirt_flare(person, hem), 3)
    heels = judge.heels(ref) if judge is not None and hasattr(judge, "heels") else None
    if heels is None:                                      # no judge to look: the words
        heels = bool(re.search(r"\b(heels?|heeled|pumps|stilettos?)\b", req.look.lower()))
    shape.update(skirt=round(skirt, 3), flare=flare, heels=bool(heels))
    return shape


def ma_look_words(req: MaRequest, ref: Image.Image, judge=None, raw: Path | None = None, log=print) -> str:
    """The words every frame and the video are drawn with: the request's look and the judge's head-to-toe account of
    the picture (ASK_LOOK), asked once and kept in raw/look.txt so the walk and the box are drawn with the same words.
    No judge, or one that cannot describe: the request's look alone."""
    path = raw / "look.txt" if raw is not None else None
    if path is not None and path.exists():
        seen = path.read_text(encoding="utf-8").strip()
    else:
        describe = getattr(judge, "describe", None)
        seen = describe(ref) if describe else ""
        if seen and path is not None:
            path.write_text(seen + "\n", encoding="utf-8")
            log(f"the person from head to toe: {seen}")
    return "; ".join(filter(None, [req.look, f"from head to toe: {seen}" if seen else ""]))


def ma_shape(req: MaRequest) -> dict:
    """The mannequin's shape for ma_depth (head, body, bust, torso, skirt, flare, heels): as the request sets it (a body or a
    bust asked for drops the measured outline), else as measured
    on the picture when the cycle was first drawn (raw/shape.json), else the defaults (a skirt from the words)."""
    saved = Path(req.out) / "raw" / "shape.json"
    shape = dict(head=0.19, body=0.16, bust=0.0, torso=None, skirt=ma_skirt_of(req.look), flare=None, heels=False)
    if saved.exists():
        shape.update(json.loads(saved.read_text(encoding="utf-8")))
    shape.update({k: v for k in ("head", "body", "bust", "skirt") if (v := getattr(req, k)) is not None})
    if req.body is not None or req.bust is not None:
        shape["torso"] = None
    return shape


def ma_finish(req: MaRequest, phases=None, pts=None, depths=None) -> dict:
    """Cut out, lock and write the outputs from the raw frames already in req.out/raw."""
    out, raw = Path(req.out), Path(req.out) / "raw"
    carry = req.kind == "carry"
    phases = phases or ma_frame_phases(req.frames, req.gait)
    shape = ma_shape(req)
    pts = pts or [ma_arms_on_body(ma_walk_points(t, req.gait, carry=carry), shape["torso"]) for t in phases]
    depths = depths or [ma_depth(p, box=carry, **shape) for p in pts]
    cut = [ma_cutout(Image.open(raw / f"{req.kind}_{i}.png"), Image.open(raw / f"{req.kind}_{i}_mask.png"))
           for i in range(req.frames)]
    locked = ma_lock(cut, phases, pts, depths, bob=req.bob)
    if req.head_once:
        locked = ma_head_once(locked)
    # the legs told apart on the mannequin without the skirt, from its hem down
    skirt = shape["skirt"]
    bare = [ma_depth(p, box=carry, **{**shape, "skirt": 0}) for p in pts] if skirt else depths
    hems = [ma_skirt_hem(p, skirt, shape["flare"])[2] for p in pts] if skirt else None
    if req.far_shade < 1:
        locked = ma_leg_tones(locked, bare, req.far_shade, pts, hems)
    ref_path = raw / "reference.png"
    if req.colour > 0 and ref_path.exists():
        from scipy import ndimage
        from .checks import ma_person
        refa = np.asarray(Image.open(ref_path).convert("RGB"))
        figure = ndimage.binary_fill_holes(ma_person((refa < 235).any(-1)))
        box = None
        if carry:                                       # the box keeps its own colours (it is not in the picture)
            box = []
            for f, p in zip(locked, pts):
                b = np.zeros(f.shape[:2], bool)
                x0, y0, x1, y1 = (int(v) for v in ma_box_rect(p))
                b[max(0, y0 - 20):y1 + 20, max(0, x0 - 20):x1 + 20] = True
                box.append(b)
        locked = ma_colour_match(locked, refa[figure], req.colour, box)
    for i, f in enumerate(locked):
        Image.fromarray(f).save(out / f"{req.kind}_{i}.png")
    path = out / f"{req.kind}_report.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    # the two steps, checked on the frames as they are written (after the leg tones), so the sheet says what shows:
    # the light on each leg, on the same leg masks the tones used
    alt = ma_alternates(locked, depths, [ma_leg_masks(f, d, p, e) for f, d, p, e
                                         in zip(locked, bare, pts, hems or [None] * len(locked))])
    # the clothes from one frame to the next, on the frames as drawn, written down for a look. It does not fail a
    # frame: the drawn figure never lies exactly on its mannequin, and a girl's wide sleeves swinging over her coat
    # read as a jump in a video the eye saw as one look (11-04)
    change = ma_clothes_change([np.asarray(Image.open(raw / f"{req.kind}_{i}.png").convert("RGB"))
                                for i in range(req.frames)], pts)
    for r in report:
        i = r["frame"]
        if i < req.frames:
            r["lead"] = alt["lead"][i]
            r["clothes_change"] = round(change[i], 1)
            r["passed"] = bool(r.get("checks", r.get("passed", True))) and i not in alt["fail"]
            r["reason"] = ma_alternate_reason(alt, i)
    if report:
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    review = ma_review_sheet(locked, report, out / f"{req.kind}_review.png", req.kind)
    sprites, k = ma_sprites(locked, req.height)
    stride = ma_stride_px(req.frames, req.gait) * k
    cols = 4 if req.frames % 4 == 0 else req.frames
    return {
        "sheet": str(ma_sheet(sprites, out / f"{req.kind}_sheet.png", cols)),
        "loop": str(ma_loop_gif(sprites, out / f"{req.kind}_loop.gif", req.fps)),
        "strip": str(ma_strip_gif(sprites, out / f"{req.kind}_strip.gif", stride, req.fps)),
        "meta": str(ma_meta(out / f"{req.kind}_sheet.json", sprites, stride, req.fps, req.kind, cols)),
        "review": str(review),
    }

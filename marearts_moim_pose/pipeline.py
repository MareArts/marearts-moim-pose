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

from .checks import (FACING_SURE, ma_alternate_reason, ma_alternates, ma_clothes_change, ma_cutout,
                     ma_facing, ma_frame_ok)
from .comfy import (ARMS_NEAR_BACK, ARMS_NEAR_BACK_HALF, CARRY, EXTRA_FIX, HAND_FIX, STYLE, VIDEO_H, VIDEO_W, WALK,
                    MaComfy, ma_arm_words,
                    ma_cutout_graph, ma_hand_fix_graph, ma_klein_depth_graph, ma_klein_text_graph, ma_prompt,
                    ma_sharpen_graph, ma_video_graph)
from .gait import LSHO, RSHO, H, W, MaGait, ma_frame_phases, ma_stride_px, ma_walk_points
from .judge import HEAD_MIN, ma_hand_box, ma_hand_boxes
from .lock import ma_leg_tones, ma_lock
from .mannequin import ma_depth
from .sheet import ma_loop_gif, ma_meta, ma_review_sheet, ma_sheet, ma_sprites, ma_strip_gif

NEIGHBOUR_TRIES = 3     # an older man's arm half way back: 12 seeds failed from his picture, 1 in 3 from frame 1
SEEDS = (0, 4, 16, 23, 42, 61)          # offsets from the seed; about one frame in twenty needs a second try


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
    head: float = 0.19                   # head width / height of the person in the picture
    bust: float = 0.0
    height: int | None = 256             # sprite height in px (None: full size)
    fps: int = 12                      # one cycle a second at 12 frames
    far_shade: float = 0.55              # the far leg's brightness against the near one's (1: legs as drawn)
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
    ref_im = ref                                        # the person, for the head likeness
    ref_name = comfy.upload(ref, f"{req.name}_ref.png")
    carry = req.kind == "carry"
    phases = ma_frame_phases(req.frames, req.gait)
    pts = [ma_walk_points(t, req.gait, carry=carry) for t in phases]
    depths = [ma_depth(p, head=req.head, box=carry, bust=req.bust) for p in pts]
    widest = max(abs(p[LSHO + 1][0] - p[LSHO][0]) for p in pts)
    prompts = [ma_prompt(req.look, req.kind, change=req.change, arms="" if carry else ma_arm_words(p, widest))
               for p in pts]
    # where a hand must be drawn: the near one always, the far one when it swings out in front of the body
    hands = [{} for _ in pts]
    for h, p in zip([] if carry else hands, pts):
        h["near"] = ma_hand_box(p, LSHO + 2)
        if ma_arm_words(p, widest) in (ARMS_NEAR_BACK, ARMS_NEAR_BACK_HALF):
            h["far"] = ma_hand_box(p, RSHO + 2)
    tries = len(SEEDS) if judge is not None else 3
    path = out / f"{req.kind}_report.json"
    report = {r["frame"]: r for r in (json.loads(path.read_text(encoding="utf-8")) if path.exists() else [])}
    dnames = {}

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
                    ok = like >= HEAD_MIN
                    verdict += f"; head {like:.2f}" + ("" if ok else " (another person)")
        log(f"frame {i} seed {sd}: {said}" + (f", judge {verdict!r}" if verdict else "") + (" ok" if ok else " again"))
        return ok, said, verdict

    def cut(i):
        return ma_cutout(Image.open(raw / f"{req.kind}_{i}.png"), Image.open(raw / f"{req.kind}_{i}_mask.png"))

    def draw_frame(i):
        """Frame i through its seeds (and the ways back), its report row written."""
        t0 = time.time()
        # drawn again (a person did not like it): never a seed this frame already had, or the same picture comes back
        tried = report.get(i, {}).get("tried", [])
        seeds = [use_seed] if use_seed is not None else [s for s in ma_seeds(req.seed) if s not in tried][:tries]
        ok, frm, kept, alone = False, None, None, False
        if i in starts:
            sd = seeds[0]
            tried = tried + ([sd] if sd not in tried else [])
            ok, said, verdict = draw(i, sd, start=starts[i])
            # the video lost a hand: drawn again where the mannequin has the hands; it drew one too many: drawn over
            # where that one is. The rest of the frame is kept either way.
            fix, dz = ({}, HAND_FIX) if ok else (hands[i], HAND_FIX) if hands[i] and "missing" in verdict else \
                (ma_extra_hands(ma_hand_boxes(verdict), pts[i]), EXTRA_FIX)
            if fix:
                sharp = comfy.upload(Image.open(raw / f"{req.kind}_{i}.png").convert("RGB"),
                                     f"{req.name}_{req.kind}_sharp_{i}.png")
                mask = comfy.upload(ma_hands_mask(fix), f"{req.name}_{req.kind}_hands_{i}.png")
                for sd in [x for x in ma_seeds(req.seed) if x not in tried][:tries]:
                    tried = tried + [sd]
                    ok, said, verdict = draw(i, sd, start=sharp, mask=mask, denoise=dz)
                    if ok:
                        log(f"frame {i} seed {sd}: the hands drawn again, ok")
                        break
        else:
            for sd in seeds:
                tried = tried + ([sd] if sd not in tried else [])
                ok, said, verdict = draw(i, sd)
                if ok:
                    break
        if not ok and i in starts:
            lost = "missing" in verdict          # only a hand the video lost is worth keeping it for
            # drawn on its own only if that passes: an arm half way back failed both ways on every seed, and a frame
            # drawn on its own broke the one look of the cycle (clothes change 6.9 -> 11.1, 11-03); the video's
            # frame is kept then, failed, for a person's look
            keep = {p: p.read_bytes() for p in (raw / f"{req.kind}_{i}.png", raw / f"{req.kind}_{i}_mask.png")}
            kept = (sd, said, verdict)
            log(f"frame {i}: no sharp frame from the video passed, drawn on its own")
            for sd in [x for x in ma_seeds(req.seed) if x not in tried][:tries]:
                tried = tried + [sd]
                ok, said, verdict = draw(i, sd)
                if ok:
                    kept, alone = None, True
                    break
            if not ok and lost:
                for p, data in keep.items():
                    p.write_bytes(data)
                sd, said, verdict = kept
                log(f"frame {i}: kept the video's frame (seed {sd}); look at it")
            elif not ok:
                kept = None
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
                         engine="video" if i in starts and not alone else "frames",
                         **({"from_frame": frm} if frm is not None else {}))

    drawn = list(only) if only is not None else list(range(req.frames))
    starts = {}
    if req.engine == "video" and only is None and use_seed is None:
        draw_frame(0)
        if report[0]["passed"]:
            starts = ma_video_starts(req, comfy, depths, raw, log)
    for i in drawn:
        draw_frame(i)
    missing = [i for i in range(req.frames) if not (raw / f"{req.kind}_{i}.png").exists()]
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


def ma_video_starts(req: MaRequest, comfy: MaComfy, depths: list, raw: Path, log=print) -> dict:
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
    graph = ma_video_graph([names[j % n] for j in range(length)], first, ma_video_prompt(req.look, req.kind),
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


def ma_finish(req: MaRequest, phases=None, pts=None, depths=None) -> dict:
    """Cut out, lock and write the outputs from the raw frames already in req.out/raw."""
    out, raw = Path(req.out), Path(req.out) / "raw"
    carry = req.kind == "carry"
    phases = phases or ma_frame_phases(req.frames, req.gait)
    pts = pts or [ma_walk_points(t, req.gait, carry=carry) for t in phases]
    depths = depths or [ma_depth(p, head=req.head, box=carry, bust=req.bust) for p in pts]
    cut = [ma_cutout(Image.open(raw / f"{req.kind}_{i}.png"), Image.open(raw / f"{req.kind}_{i}_mask.png"))
           for i in range(req.frames)]
    locked = ma_lock(cut, phases, pts, depths)
    if req.far_shade < 1:
        locked = ma_leg_tones(locked, depths, req.far_shade)
    for i, f in enumerate(locked):
        Image.fromarray(f).save(out / f"{req.kind}_{i}.png")
    path = out / f"{req.kind}_report.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    # the two steps, checked on the frames as they are written (after the leg tones), so the sheet says what shows
    alt = ma_alternates(locked, depths)
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

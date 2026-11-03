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
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .checks import FACING_SURE, ma_alternate_reason, ma_alternates, ma_cutout, ma_facing, ma_frame_ok
from .comfy import MaComfy, ma_cutout_graph, ma_klein_depth_graph, ma_klein_text_graph, ma_prompt
from .gait import H, W, MaGait, ma_frame_phases, ma_stride_px, ma_walk_points
from .lock import ma_leg_tones, ma_lock
from .mannequin import ma_depth
from .sheet import ma_loop_gif, ma_meta, ma_review_sheet, ma_sheet, ma_sprites, ma_strip_gif

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
    frames: int = 8
    seed: int = 7
    head: float = 0.19                   # head width / height of the person in the picture
    bust: float = 0.0
    height: int | None = 256             # sprite height in px (None: full size)
    fps: int = 8
    far_shade: float = 0.55              # the far leg's brightness against the near one's (1: legs as drawn)
    # draw again a frame on the same foot as the one half a cycle away. Off: the model draws the leg behind darker
    # whichever it is, so new seeds rarely help (0 of 18 on Rin's walk); the leg tones fix what shows.
    alternate: bool = False
    facing: str = "auto"                 # which way the person picture faces: auto | right | left (mirrored)


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
    ref_name = comfy.upload(ref, f"{req.name}_ref.png")
    carry = req.kind == "carry"
    phases = ma_frame_phases(req.frames, req.gait)
    pts = [ma_walk_points(t, req.gait, carry=carry) for t in phases]
    depths = [ma_depth(p, head=req.head, box=carry, bust=req.bust) for p in pts]
    prompt = ma_prompt(req.look, req.kind, change=req.change)
    tries = len(SEEDS) if judge is not None else 3
    path = out / f"{req.kind}_report.json"
    report = {r["frame"]: r for r in (json.loads(path.read_text(encoding="utf-8")) if path.exists() else [])}
    dnames = {}

    def draw(i, sd):
        """Frame i at seed sd, drawn, cut out and checked: (passes, what was measured, the judge's answer)."""
        if i not in dnames:
            dnames[i] = comfy.upload(depths[i], f"{req.name}_{req.kind}_depth_{i}.png")
            depths[i].save(raw / f"{req.kind}_{i}_depth.png")
        frame = comfy.run(ma_klein_depth_graph(dnames[i], ref_name, prompt, seed=sd), raw / f"{req.kind}_{i}.png")
        fname = comfy.upload(Image.open(frame).convert("RGB"), f"{req.name}_{req.kind}_{i}_frame.png")
        mask = comfy.run(ma_cutout_graph(fname), raw / f"{req.kind}_{i}_mask.png")
        rgb = np.asarray(Image.open(frame).convert("RGB"))
        m = np.asarray(Image.open(mask).convert("L").resize((rgb.shape[1], rgb.shape[0]))) > 128
        ok, said = ma_frame_ok(rgb, m, depths[i], pts[i], req.kind)
        verdict = ""
        if ok and judge is not None:
            ok, verdict = judge.arms_ok(Image.open(frame), req.kind)
        log(f"frame {i} seed {sd}: {said}" + (f", judge {verdict!r}" if verdict else "") + (" ok" if ok else " again"))
        return ok, said, verdict

    def cut(i):
        return ma_cutout(Image.open(raw / f"{req.kind}_{i}.png"), Image.open(raw / f"{req.kind}_{i}_mask.png"))

    drawn = list(only) if only is not None else list(range(req.frames))
    for i in drawn:
        t0 = time.time()
        # drawn again (a person did not like it): never a seed this frame already had, or the same picture comes back
        tried = report.get(i, {}).get("tried", [])
        seeds = [use_seed] if use_seed is not None else [s for s in ma_seeds(req.seed) if s not in tried][:tries]
        for sd in seeds:
            tried = tried + ([sd] if sd not in tried else [])
            ok, said, verdict = draw(i, sd)
            if ok:
                break
        else:
            log(f"frame {i}: every seed failed, look at {raw / f'{req.kind}_{i}.png'}")
        report[i] = dict(frame=i, seed=sd, tried=tried, passed=ok, checks=ok, measured=said, judge=verdict,
                         seconds=round(time.time() - t0))
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
    for r in report:
        i = r["frame"]
        if i < req.frames:
            r["lead"] = alt["lead"][i]
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

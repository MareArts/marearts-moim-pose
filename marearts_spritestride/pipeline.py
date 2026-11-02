"""One person picture in, a locked walk (or box-carry) cycle out.

For each frame: the gait's pose -> depth mannequin -> FLUX.2 [klein] draws the person in it -> BiRefNet cuts the
person out -> measured checks and (optional) the vision judge. A frame that fails is drawn again with another seed.
Then the frames are locked together and written as a sheet, GIFs, a JSON and a review sheet.

Not every frame comes out right: the checks catch extra limbs, a turning box and (with the judge) a third hand,
but not which leg is in front, a hand drawn oddly, or a hat or pocket that changes. A person looks at the review
sheet and draws the frames they do not like again by number (--only); those get seeds they have not had.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .checks import ma_cutout, ma_frame_ok
from .comfy import MaComfy, ma_cutout_graph, ma_klein_depth_graph, ma_klein_text_graph, ma_prompt
from .gait import H, W, MaGait, ma_frame_phases, ma_stride_px, ma_walk_points
from .lock import ma_lock
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


def ma_draw_cycle(req: MaRequest, comfy: MaComfy, judge=None, only: list[int] | None = None, log=print,
                  use_seed: int | None = None) -> dict:
    """Draw, check and lock one cycle. Returns a report of every frame (seed, what was measured, the judge).

    use_seed: draw the `only` frames with exactly this seed (the same seed gives the same picture), e.g. to go back
    to an earlier try that looked better than the new ones."""
    out = Path(req.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    ref_name = comfy.upload(ma_reference(Image.open(req.reference)), f"{req.name}_ref.png")
    carry = req.kind == "carry"
    phases = ma_frame_phases(req.frames, req.gait)
    pts = [ma_walk_points(t, req.gait, carry=carry) for t in phases]
    depths = [ma_depth(p, head=req.head, box=carry, bust=req.bust) for p in pts]
    prompt = ma_prompt(req.look, req.kind, change=req.change)
    tries = len(SEEDS) if judge is not None else 3
    path = out / f"{req.kind}_report.json"
    report = {r["frame"]: r for r in (json.loads(path.read_text(encoding="utf-8")) if path.exists() else [])}
    for i in only if only is not None else range(req.frames):
        t0 = time.time()
        dname = comfy.upload(depths[i], f"{req.name}_{req.kind}_depth_{i}.png")
        depths[i].save(raw / f"{req.kind}_{i}_depth.png")
        # drawn again (a person did not like it): never a seed this frame already had, or the same picture comes back
        tried = report.get(i, {}).get("tried", [])
        seeds = [use_seed] if use_seed is not None else [s for s in ma_seeds(req.seed) if s not in tried][:tries]
        for sd in seeds:
            tried = tried + ([sd] if sd not in tried else [])
            frame = comfy.run(ma_klein_depth_graph(dname, ref_name, prompt, seed=sd), raw / f"{req.kind}_{i}.png")
            fname = comfy.upload(Image.open(frame).convert("RGB"), f"{req.name}_{req.kind}_{i}_frame.png")
            mask = comfy.run(ma_cutout_graph(fname), raw / f"{req.kind}_{i}_mask.png")
            rgb = np.asarray(Image.open(frame).convert("RGB"))
            m = np.asarray(Image.open(mask).convert("L").resize((rgb.shape[1], rgb.shape[0]))) > 128
            ok, said = ma_frame_ok(rgb, m, depths[i], pts[i], req.kind)
            verdict = ""
            if ok and judge is not None:
                ok, verdict = judge.arms_ok(Image.open(frame), req.kind)
            log(f"frame {i} seed {sd}: {said}" + (f", judge {verdict!r}" if verdict else "") + (" ok" if ok else " again"))
            if ok:
                break
        else:
            log(f"frame {i}: every seed failed, look at {frame}")
        report[i] = dict(frame=i, seed=sd, tried=tried, passed=ok, measured=said, judge=verdict,
                         seconds=round(time.time() - t0))
    rows = [report[k] for k in sorted(report)]
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    missing = [i for i in range(req.frames) if not (raw / f"{req.kind}_{i}.png").exists()]
    if missing:
        log(f"frames {missing} not drawn yet: no lock or sheet")
        return dict(report=rows, files={})
    return dict(report=rows, files=ma_finish(req, phases, pts, depths))


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
    for i, f in enumerate(locked):
        Image.fromarray(f).save(out / f"{req.kind}_{i}.png")
    path = out / f"{req.kind}_report.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
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

"""Frames out: a sprite sheet, a loop GIF, a GIF of the sprite walking across a strip, and a small JSON a game reads."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def ma_crop_box(frames: list[np.ndarray], pad: int = 8) -> tuple[int, int, int, int]:
    """One box round every frame's figure, so the frames stay registered."""
    al = np.max([f[..., 3] for f in frames], axis=0) > 8
    ys, xs = np.nonzero(al)
    h, w = al.shape
    return max(0, xs.min() - pad), max(0, ys.min() - pad), min(w, xs.max() + 1 + pad), min(h, ys.max() + 1 + pad)


def ma_sprites(frames: list[np.ndarray], height: int | None = None) -> tuple[list[Image.Image], float]:
    """The frames cropped to one box and scaled to `height` px (None: as drawn). Returns the sprites and the scale."""
    box = ma_crop_box(frames)
    ims = [Image.fromarray(f).crop(box) for f in frames]
    k = 1.0 if height is None else height / ims[0].height
    if k != 1.0:
        ims = [im.resize((round(im.width * k), height), Image.LANCZOS) for im in ims]
    return ims, k


def ma_sheet(sprites: list[Image.Image], path: Path, cols: int = 4) -> Path:
    w, h = sprites[0].size
    rows = -(-len(sprites) // cols)
    sheet = Image.new("RGBA", (w * cols, h * rows), (0, 0, 0, 0))
    for i, im in enumerate(sprites):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(path)
    return path


def ma_loop_gif(sprites: list[Image.Image], path: Path, fps: int = 8, bg=(255, 255, 255)) -> Path:
    flat = [Image.alpha_composite(Image.new("RGBA", im.size, bg + (255,)), im).convert("RGB") for im in sprites]
    flat[0].save(path, save_all=True, append_images=flat[1:], duration=round(1000 / fps), loop=0, optimize=True)
    return path


def ma_strip_gif(sprites: list[Image.Image], path: Path, stride: float, fps: int = 8, loops: int = 3,
                 bg=(246, 244, 239), ground=(205, 199, 188)) -> Path:
    """The sprite walking across a strip at its own stride per frame: if the feet slide, it shows."""
    w, h = sprites[0].size
    n = len(sprites) * loops
    width = int(w + stride * n + w)
    out = []
    for j in range(n):
        can = Image.new("RGBA", (width, h + 12), bg + (255,))
        can.paste(Image.new("RGBA", (width, 4), ground + (255,)), (0, h - 6))
        can.alpha_composite(sprites[j % len(sprites)], (int(stride * j), 0))
        out.append(can.convert("RGB"))
    out[0].save(path, save_all=True, append_images=out[1:], duration=round(1000 / fps), loop=0, optimize=True)
    return path


LOOK_AT = {"walk": "legs: the near one in front every other frame · hands: two at most · clothes, face, hair the same",
           "carry": "legs · no hand by the hip · the box level, one flat face · clothes, face, hair the same"}


def ma_review_sheet(frames: list[np.ndarray], report: list[dict], path: Path, kind: str = "walk",
                    cell: tuple[int, int] = (240, 350), cols: int = 4) -> Path:
    """Every frame with its number, seed and what was measured; a frame the checks failed in red. What the checks
    cannot see is written on top, and how to draw a frame again."""
    from PIL import ImageDraw, ImageFont
    font = ImageFont.load_default()
    rep = {r["frame"]: r for r in report}
    rows = -(-len(frames) // cols)
    top, foot = 54, 46
    cw, ch = cell
    sheet = Image.new("RGB", (cw * cols, top + rows * (ch + foot)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((8, 6), f"Look at: {LOOK_AT.get(kind, LOOK_AT['walk'])}", fill=(30, 30, 30), font=font)
    d.text((8, 24), f"Draw again: spritestride draw <picture> --out <same folder> --kind {kind} --only <frame numbers>",
           fill=(30, 30, 30), font=font)
    box = ma_crop_box(frames)
    for i, f in enumerate(frames):
        im = Image.fromarray(f).crop(box)
        im.thumbnail((cw - 12, ch - 12))
        x, y = (i % cols) * cw, top + (i // cols) * (ch + foot)
        flat = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im).convert("RGB")
        sheet.paste(flat, (x + (cw - im.width) // 2, y + (ch - im.height) // 2))
        r = rep.get(i, {})
        bad = r and not r.get("passed", True)
        d.rectangle([x + 2, y + 2, x + cw - 3, y + ch - 3], outline=(214, 40, 40) if bad else (210, 210, 210),
                    width=4 if bad else 1)
        d.text((x + 8, y + 8), f"#{i}", fill=(214, 40, 40) if bad else (30, 30, 30), font=font)
        if r:
            tries = len(r.get("tried", [r.get("seed")]))
            why = r.get("reason") or "CHECKS FAILED"
            d.text((x + 6, y + ch + 2), f"seed {r.get('seed')} · try {tries}" + (f" · {why}" if bad else ""),
                   fill=(214, 40, 40) if bad else (60, 60, 60), font=font)
            said = r.get("judge", "")
            count = re.findall(r"\d+", said)                # the judge lists the hands, then counts them
            said = f"hands {count[-1]}" if count else said[:12]
            measured = str(r.get("measured", "")) + (f", judge {said}" if said else "")
            measured += f", lead {r['lead']:+.0f}" if "lead" in r else ""
            head, _, tail = measured.partition(", box")      # carry: the box numbers on their own line
            d.text((x + 6, y + ch + 16), head, fill=(90, 90, 90), font=font)
            if tail:
                d.text((x + 6, y + ch + 30), "box" + tail, fill=(90, 90, 90), font=font)
    sheet.save(path)
    return path


def ma_meta(path: Path, sprites: list[Image.Image], stride: float, fps: int, kind: str, cols: int) -> Path:
    w, h = sprites[0].size
    meta = {"kind": kind, "frames": len(sprites), "cols": cols, "frame_width": w, "frame_height": h, "fps": fps,
            "facing": "right", "stride_px_per_frame": round(stride, 2),
            "note": "move the sprite stride_px_per_frame each frame and the planted foot stays put"}
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path

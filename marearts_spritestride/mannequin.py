"""Pictures of the pose that the image model follows.

In profile the two legs of an OpenPose skeleton lie on one line, so the model cannot tell which leg is in front and
draws the near leg forward on both steps ("walking on the same foot"). A depth mannequin says it: rounded limbs,
the far side drawn first and dark, the body, then the near side over it and brightest.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .gait import FIG, H, LHIP, LSHO, NECK_I, NOSE, RHIP, RSHO, W

LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
         (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLOURS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
           (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
           (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]
FAR, BODY, HEAD, CHEST, BOX, NEAR = 70, 150, 165, 158, 200, 225     # depth values: brighter is nearer


def ma_skeleton(pts: dict) -> Image.Image:
    """The skeleton drawn as OpenPose detectors draw it (limbs at 60 % over black, then the joints). For previews."""
    base = np.zeros((H, W, 3), np.float32)
    stick = int(round(4 * H / 512))
    for i, (a, b) in enumerate(LIMBS):
        if a not in pts or b not in pts:
            continue
        layer = Image.new("RGB", (W, H))
        (x0, y0), (x1, y1) = pts[a], pts[b]
        mx, my, ln = (x0 + x1) / 2, (y0 + y1) / 2, math.hypot(x1 - x0, y1 - y0) / 2
        ang = math.atan2(y1 - y0, x1 - x0)
        poly = [(mx + ln * math.cos(u) * math.cos(ang) - stick * math.sin(u) * math.sin(ang),
                 my + ln * math.cos(u) * math.sin(ang) + stick * math.sin(u) * math.cos(ang))
                for u in np.linspace(0, 2 * math.pi, 36)]
        ImageDraw.Draw(layer).polygon(poly, fill=COLOURS[i])
        lay = np.asarray(layer, np.float32)
        base = np.where(lay.sum(axis=2, keepdims=True) > 0, base * 0.4 + lay * 0.6, base)
    im = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for k, (x, y) in pts.items():
        d.ellipse([x - stick, y - stick, x + stick, y + stick], fill=COLOURS[k])
    return im


def ma_depth(pts: dict, head: float = 0.19, box: bool = False, bust: float = 0.0) -> Image.Image:
    """The depth mannequin of a pose (RGB, near bright, far dark).

    head: head width as a share of the height. Use the person's own (a big anime head with hair is about 0.19), or
    the model shrinks the head toward its default. The chin stays put; the head grows back and up.
    box: a carried box before the chest, nearer than the body. bust: how far the chest stands out (share of the
    height, 0 for none, about 0.13 for a fuller figure); the depth picture sets the silhouette, so say it here.
    """
    from scipy import ndimage
    im = Image.new("L", (W, H))
    d = ImageDraw.Draw(im)
    s = FIG

    def cap(a, b, w, v):
        d.line([a, b], fill=v, width=round(w * s))
        for p in (a, b):
            r = w * s / 2
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=v)

    def side(hip_i, sho_i, v):
        hp, kn, an = pts[hip_i], pts[hip_i + 1], pts[hip_i + 2]
        cap(hp, kn, 0.1, v)
        cap(kn, an, 0.072, v)
        cap(an, (an[0] + 0.075 * s, an[1] + 0.02 * s), 0.035, v)          # the foot, pointing right
        sh, el, wr = pts[sho_i], pts[sho_i + 1], pts[sho_i + 2]
        cap(sh, el, 0.055, v)
        cap(el, wr, 0.045, v)

    side(RHIP, RSHO, FAR)
    neck, hip = pts[NECK_I], ((pts[LHIP][0] + pts[RHIP][0]) / 2, pts[LHIP][1])
    cap((neck[0], neck[1] + 0.04 * s), hip, 0.16, BODY)
    hx, hy, hw = pts[NOSE][0] - 0.05 * s, pts[NOSE][1] - 0.01 * s, head * s
    d.ellipse([hx + 0.065 * s - hw, hy + 0.07 * s - 1.115 * hw, hx + 0.065 * s, hy + 0.07 * s], fill=HEAD)
    if bust:
        cx, cy = neck[0] + (hip[0] - neck[0]) * 0.24, neck[1] + (hip[1] - neck[1]) * 0.24
        d.ellipse([cx + 0.01 * s, cy - 0.065 * s, cx + bust * s, cy + 0.055 * s], fill=CHEST)
    if box:
        d.rectangle(ma_box_rect(pts), fill=BOX)
    side(LHIP, LSHO, NEAR)
    a = ndimage.gaussian_filter(np.asarray(im, np.float32), 3)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8)).convert("RGB")


def ma_box_rect(pts: dict) -> tuple[float, float, float, float]:
    """Where the carried box is in the mannequin: (left, top, right, bottom), on the forearms before the chest."""
    neck, wr = pts[NECK_I], pts[LSHO + 2]
    return (neck[0] + 0.03 * FIG, wr[1] - 0.17 * FIG, wr[0] + 0.02 * FIG, wr[1] + 0.03 * FIG)


def ma_legs_mask(depth: Image.Image) -> np.ndarray:
    """The mannequin's figure as a boolean mask (anything brighter than the background)."""
    return np.asarray(depth.convert("L")) > 30


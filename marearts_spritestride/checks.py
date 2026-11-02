"""Measured checks on a drawn frame, and its cut-out. They catch what the model gets wrong now and then: a third
arm or leg, a carried box that turns or bobs, white paper left inside the figure."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .gait import H, W
from .mannequin import ma_box_rect

# Limits found on 192 frames checked by eye: a real arm outside the mannequin is ~15-20 (thousandths of height
# squared), passed frames at most 2.4; legs across the shins at most 2 (3 at one height where a shoe and a cuff part).
OUTSIDE_MAX, LEGS_MAX = 6.0, 3
BOX_END_MAX, BOX_TOP_MAX = 0.15, 10.0


def ma_person(mask: np.ndarray) -> np.ndarray:
    """The largest piece of the mask: the person, without stray specks."""
    from scipy import ndimage
    lab, k = ndimage.label(mask)
    if k <= 1:
        return mask
    return lab == 1 + int(np.argmax(ndimage.sum(mask, lab, range(1, k + 1))))


def ma_extra_limbs(mask: np.ndarray, depth: Image.Image) -> tuple[float, int]:
    """(outside, legs) of a drawn person against the mannequin it was drawn over.

    outside: the biggest piece of the person farther than 4 % of its height from the mannequin, below the head and
    hair, in thousandths of height squared. legs: separate legs across the shins, the second most of four heights
    (one height alone does not count).
    """
    from scipy import ndimage
    m = ma_person(mask)
    ys = np.nonzero(m.any(1))[0]
    top, h = ys.min(), ys.max() - ys.min()
    d = np.asarray(depth.convert("L").resize(m.shape[::-1])) > 20
    out = m & ~ndimage.binary_dilation(d, iterations=max(1, int(0.04 * h)))
    out[:top + int(0.25 * h)] = False
    ol, ok = ndimage.label(out)
    big = max((int((ol == j).sum()) for j in range(1, ok + 1)), default=0) / (h * h) * 1000
    counts = []
    for fr in (0.74, 0.80, 0.86, 0.92):
        row = ndimage.binary_opening(m[top + int(fr * h)][None], structure=np.ones((1, 5)))[0]
        counts.append(int((row[1:] & ~row[:-1]).sum() + row[0]))
    return big, sorted(counts)[-2]


def ma_box_drift(rgb: np.ndarray, mask: np.ndarray, pts: dict) -> tuple[float, float]:
    """A carried cardboard box as drawn: (its top against the mannequin's box top in px, the share of it that is the
    darker end face). Seen square from the side the end face is 0.01-0.10 of the box; turned, 0.24-0.36."""
    from scipy import ndimage
    from skimage import color
    hsv = color.rgb2hsv(rgb.astype(float) / 255)
    hue = hsv[..., 0] * 360
    m = mask & (hue > 22) & (hue < 42) & (hsv[..., 1] > 0.25) & (hsv[..., 1] < 0.65) & (hsv[..., 2] > 0.45)
    m[:int(0.164 * H)], m[int(0.576 * H):], m[:, :int(0.505 * W)] = False, False, False   # before the chest
    lab, k = ndimage.label(m)
    if not k:
        return 999.0, 1.0
    box = lab == 1 + int(np.argmax(ndimage.sum(m, lab, range(1, k + 1))))
    rows = np.nonzero(box.sum(1) > 0.5 * box.sum(1).max())[0]
    return float(rows.min() - ma_box_rect(pts)[1]), float(((hsv[..., 2] < 0.62) & box).sum() / box.sum())


def ma_frame_ok(rgb: np.ndarray, mask: np.ndarray, depth: Image.Image, pts: dict, kind: str) -> tuple[bool, str]:
    """All measured checks of one frame: (passes, what was measured)."""
    outside, legs = ma_extra_limbs(mask, depth)
    said = f"outside {outside:.1f}, legs {legs}"
    ok = outside < OUTSIDE_MAX and legs < LEGS_MAX
    if kind == "carry":
        top, end = ma_box_drift(rgb, mask, pts)
        said += f", box top {top:+.0f} px, end face {end:.2f}"
        ok = ok and end <= BOX_END_MAX and abs(top) <= BOX_TOP_MAX
    return ok, said


def ma_eye_holes(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Holes a background remover cuts in the face (an eye white read as paper): a hole closed in by the person, in
    the top fifth, touching the face's own skin or never fully clear. Gaps between hair strands and between the legs
    are real background and stay cut."""
    from scipy import ndimage
    from skimage import color
    per = alpha > 0.5
    hole = ndimage.binary_fill_holes(per) & ~per
    if not hole.any():
        return hole
    ys = np.nonzero(per.any(1))[0]
    top, h = ys.min(), ys.max() - ys.min()
    hsv, lab = color.rgb2hsv(rgb / 255), color.rgb2lab(rgb / 255)
    face = np.zeros_like(per)
    face[top + int(0.06 * h):top + int(0.14 * h)] = True
    face &= per & (hsv[..., 0] * 360 < 40) & (hsv[..., 1] > 0.12) & (hsv[..., 1] < 0.6) & (hsv[..., 2] > 0.6)
    if face.sum() < 50:
        return np.zeros_like(per)
    skin = np.linalg.norm(lab - np.median(lab[face], axis=0), axis=-1) < 20
    holes, k = ndimage.label(hole)
    out = np.zeros_like(per)
    for j in range(1, k + 1):
        c = holes == j
        if np.nonzero(c)[0].mean() > top + 0.2 * h:
            continue
        ring = ndimage.binary_dilation(c, iterations=3) & ~c & per
        if ring.any() and (skin[ring].mean() > 0.1 or alpha[c].min() > 0.2):
            out |= c
    return out


def ma_cutout(frame: Image.Image, mask: Image.Image) -> np.ndarray:
    """A frame drawn on white as RGBA: the person by the mask, the edge colour freed of the white it was mixed with."""
    from scipy import ndimage
    a = np.asarray(frame.convert("RGB")).astype(np.float32)
    al = np.asarray(mask.convert("L").resize(frame.size)).astype(np.float32) / 255
    big = ma_person(al > 0.5)
    al = np.where(ndimage.binary_dilation(big, iterations=6), al, 0)
    al = np.where(ma_eye_holes(a, al), 1.0, al)
    c = (a - 255 * (1 - al[..., None])) / np.maximum(al[..., None], 1e-3)
    c = np.where(al[..., None] > 0.02, c, 0).clip(0, 255)
    return np.dstack([c, al * 255]).round().astype(np.uint8)


"""What must not change from frame to frame is set by code, not left to the drawing.

Drawn one by one, a walk's frames still differ a few percent in height and head size, and the figure wanders
sideways. One frame is the master; every frame gets
  1. its height from the walk: the feet on the master's floor, the top of the head on the walk's own smooth wave,
     the head-to-neck length the master's (scaled, easing back over the shoulders),
  2. its place from its own mannequin: the drawn legs laid on the mannequin's legs, so the planted foot goes back
     evenly as the gait planned and does not slide.
"""
from __future__ import annotations

import numpy as np

from .checks import ma_person
from .gait import LHIP, ma_height_wave
from .mannequin import ma_legs_mask


def ma_marks(person: np.ndarray) -> dict:
    """top, floor, neck rows and the body's middle column of a person mask.

    The neck is read from the front edge (the figure faces right, so long hair or a ponytail swinging behind does
    not count): the row where the front edge goes in most under the chin."""
    ys = np.nonzero(person.any(1))[0]
    top, floor = int(ys.min()), int(ys.max())
    h = floor - top
    r0, r1 = top + int(0.10 * h), top + int(0.26 * h)
    front = np.array([np.nonzero(r)[0].max() if r.any() else 0 for r in person[r0:r1]])
    neck = r0 + int(np.argmin(front))
    body = person[neck:neck + int(0.25 * h)]
    return dict(top=top, floor=floor, neck=neck, cx=float(np.nonzero(body)[1].mean()))


def ma_legs_dx(person: np.ndarray, floor: int, pts: dict, depth) -> int:
    """How far right of its mannequin's legs the drawing's legs stand (px), floors put together, below the hip."""
    man = ma_legs_mask(depth)
    mfloor = int(np.nonzero(man.any(1))[0].max())
    hip = pts[LHIP][1]
    r0, r1 = int(hip + 0.1 * (mfloor - hip)), mfloor
    a = np.roll(person, mfloor - floor, 0)[r0:r1]
    b = man[r0:r1]
    return -max(range(-60, 61), key=lambda x: (np.roll(a, x, 1) & b).sum() - 0.5 * (np.roll(a, x, 1) ^ b).sum())


def ma_far_shade(frame: np.ndarray, depth, k: float = 0.55) -> np.ndarray:
    """The far leg darker, as sprite artists draw it.

    Both legs of a side view wear the same trousers, so a correct cycle still reads as "the same foot" at a glance:
    frames 0 and 4 look alike. The mannequin knows which leg is far (its dark one); a locked frame stands on its
    mannequin's legs, so that leg is darkened to k, from below the hem of a coat down to the shoe."""
    from scipy import ndimage
    d = np.asarray(depth.convert("L")).astype(np.float64)
    man = ma_legs_mask(depth)
    ys = np.nonzero(man.any(1))[0]
    far = ndimage.binary_dilation((d > 35) & (d < 115), iterations=6) & ~(d > 130)    # widened to the drawn leg
    soft = ndimage.gaussian_filter(far.astype(np.float64), 3)
    t0, t1 = (ys.min() + r * (ys.max() - ys.min()) for r in (0.60, 0.68))            # mid-thigh: legs only
    soft *= np.clip((np.arange(len(d)) - t0) / (t1 - t0), 0, 1)[:, None]
    floor = np.nonzero((frame[..., 3] > 128).any(1))[0].max()
    soft = np.roll(soft, int(floor - ys.max()), 0)
    out = frame.copy()
    out[..., :3] = (frame[..., :3] * (1 - (1 - k) * soft[..., None])).clip(0, 255).round().astype(np.uint8)
    return out


def ma_leg_masks(frame: np.ndarray, depth) -> tuple[np.ndarray, np.ndarray]:
    """Soft masks (0..1) of the near and the far leg of a locked frame, from its mannequin, below mid-thigh."""
    from scipy import ndimage
    d = np.asarray(depth.convert("L")).astype(np.float64)
    ys = np.nonzero(ma_legs_mask(depth).any(1))[0]
    near = ndimage.binary_dilation(d > 130, iterations=6)
    far = ndimage.binary_dilation((d > 35) & (d < 115), iterations=6) & ~near    # where they cross: the near one
    t0, t1 = (ys.min() + r * (ys.max() - ys.min()) for r in (0.60, 0.68))
    ramp = np.clip((np.arange(len(d)) - t0) / (t1 - t0), 0, 1)[:, None]
    shift = int(np.nonzero((frame[..., 3] > 128).any(1))[0].max() - ys.max())
    return tuple(np.roll(ndimage.gaussian_filter(m.astype(np.float64), 3) * ramp, shift, 0) for m in (near, far))


def ma_leg_tones(frames: list[np.ndarray], depths: list, k: float = 0.55) -> list[np.ndarray]:
    """The near leg lit and the far leg dark in every frame, by the mannequin, as sprite artists draw a side walk.

    The model draws the leg behind darker, whichever it is: when the far leg should lead, the near leg comes out
    dark behind it, and the two steps look alike ("the same foot"); darkening the far leg alone (ma_far_shade) does
    not undo that. So each leg is brought to one brightness for the whole cycle: the near leg to the lit leg's
    usual brightness, the far leg to k of it."""
    lum = [f[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114]) for f in frames]
    masks = [ma_leg_masks(f, d) for f, d in zip(frames, depths)]
    means = []
    for f, L, (n, fa) in zip(frames, lum, masks):
        a = f[..., 3] > 128
        means.append(tuple(float(L[(m > 0.8) & a].mean()) if ((m > 0.8) & a).any() else 0.0 for m in (n, fa)))
    lit = float(np.median([max(m) for m in means]))
    out = []
    for f, (n, fa), (ln, lf) in zip(frames, masks, means):
        if not ln or not lf:
            out.append(f.copy())
            continue
        g = 1 + n * (np.clip(lit / ln, 0.5, 2.2) - 1) + fa * (np.clip(k * lit / lf, 0.3, 2.0) - 1)
        o = f.copy()
        o[..., :3] = (f[..., :3] * g[..., None]).clip(0, 255).round().astype(np.uint8)
        out.append(o)
    return out


def ma_lock(frames: list[np.ndarray], phases: list[float], pts: list[dict], depths: list, master: int = 0,
            align_legs: bool = True) -> list[np.ndarray]:
    """RGBA frames (same size) -> RGBA frames with one height, one head size and an even step."""
    from scipy import ndimage
    people = [ma_person(f[..., 3] > 128) for f in frames]
    M = ma_marks(people[master])
    torso, height = M["neck"] - M["top"], M["floor"] - M["top"]
    tall = [ma_height_wave(t) for t in phases]
    H, W = frames[0].shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    out = []
    for i, (f, p) in enumerate(zip(frames, people)):
        m = ma_marks(p)
        top = M["floor"] - height * tall[i] / tall[master]
        s = torso / max(1, m["neck"] - m["top"])
        yo = np.array([0, top, top + torso, M["floor"], H - 1], float)      # output row -> source row
        ys = np.array([0, m["top"], m["neck"], m["floor"], H - 1], float)
        if yo[1] <= 0:
            yo[0], ys[0] = yo[1] - 1, ys[1] - 1
        Y = np.interp(np.arange(H), yo, ys)
        ease = np.clip((np.arange(H) - (top + torso)) / (0.06 * height), 0, 1)
        sx = s + (1 - s) * ease
        dx = ma_legs_dx(p, m["floor"], pts[i], depths[i]) if align_legs else 0
        X = m["cx"] + (xx - m["cx"]) / sx[:, None] + dx
        a = f.astype(np.float64)
        w = np.stack([ndimage.map_coordinates(a[..., c], [Y[:, None] + 0 * xx, X], order=1, cval=0.0)
                      for c in range(4)], -1)
        out.append(w.clip(0, 255).round().astype(np.uint8))
    return out

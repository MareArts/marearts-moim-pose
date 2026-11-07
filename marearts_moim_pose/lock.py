"""What must not change from frame to frame is set by code, not left to the drawing.

Drawn one by one, a walk's frames still differ a few percent in height and head size, and the figure wanders
sideways. One frame is the master; every frame gets
  1. its height from the walk: the feet on the master's floor, the top of the head on the walk's own smooth wave,
     the head the master's size (found by matching the master's head in it, scaled, easing back over the shoulders),
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


def ma_leg_masks(frame: np.ndarray, depth, pts: dict | None = None,
                 hem: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Soft masks (0..1) of the near and the far leg of a locked frame, from its mannequin, below mid-thigh (and
    below a skirt's hem: its row in the mannequin)."""
    from scipy import ndimage
    d = np.asarray(depth.convert("L")).astype(np.float64)
    ys = np.nonzero(ma_legs_mask(depth).any(1))[0]
    near = ndimage.binary_dilation(d > 130, iterations=6)
    far = ndimage.binary_dilation((d > 35) & (d < 115), iterations=6) & ~near    # where they cross: the near one
    t0, t1 = (ys.min() + r * (ys.max() - ys.min()) for r in (0.60, 0.68))
    ramp = np.clip((np.arange(len(d)) - t0) / (t1 - t0), 0, 1)[:, None]
    shift = int(np.nonzero((frame[..., 3] > 128).any(1))[0].max() - ys.max())
    soft = [np.roll(ndimage.gaussian_filter(m.astype(np.float64), 3) * ramp, shift, 0) for m in (near, far)]
    if hem is not None:                         # the skirt is cloth over both legs, neither leg's
        for m in soft:
            m[:max(0, int(hem) + shift)] = 0
    # Above mid-thigh the masks faded out, and a far leg shaded below the knee only showed as a dark smudge on bare
    # skin (a girl in shorts, 11-04). Up to the hip each leg takes, inside its own mannequin limb, the pixels of the
    # colour of its own lower part (the same cloth or skin), so the shade runs evenly to the hem and stops there.
    rgb = ndimage.uniform_filter(frame[..., :3].astype(np.float64), size=(3, 3, 1))
    lum = rgb.sum(-1) + 1
    chroma = rgb[..., :2] / lum[..., None]
    alpha = frame[..., 3] > 128
    hip = int(ys.min() + 0.40 * (ys.max() - ys.min())) + shift
    rows = np.zeros(len(d), bool)
    rows[max(0, hip, int(hem or 0) + shift):int(t1) + shift] = True
    thighs = (near, far)
    if pts is not None:                         # the thighs alone: the mannequin's near arm and body are as bright
        from PIL import Image, ImageDraw
        from .gait import FIG, LHIP, RHIP
        thighs = []
        for hip_i in (LHIP, RHIP):
            im = Image.new("L", (d.shape[1], d.shape[0]))
            ImageDraw.Draw(im).line([pts[hip_i], pts[hip_i + 1]], fill=255, width=int(0.13 * FIG))
            thighs.append(np.asarray(im) > 0)
        thighs[1] = thighs[1] & ~thighs[0]
    for k, m in enumerate(thighs):
        sure = (soft[k] > 0.8) & alpha
        if sure.sum() < 50:
            continue
        c0, l0 = np.median(chroma[sure], 0), np.median(lum[sure])
        same = (np.abs(chroma - c0).max(-1) < 0.035) & (lum > 0.45 * l0) & (lum < 1.6 * l0)
        up = np.roll(m, shift, 0) & alpha & rows[:, None] & same
        if k == 1:
            up &= ~np.roll(thighs[0], shift, 0)
        # only what joins the lower leg without crossing a line: a white skirt matched pale skin in colour and took a
        # grey wedge of shade (11-05); its hem's outline keeps it apart
        lab, _ = ndimage.label(up | sure)
        keep = np.unique(lab[sure])
        up &= np.isin(lab, keep[keep > 0])
        soft[k] = np.maximum(soft[k], ndimage.gaussian_filter(up.astype(np.float64), 1.5))
    return tuple(soft)


def ma_leg_tones(frames: list[np.ndarray], depths: list, k: float = 0.9, pts: list | None = None,
                 hems: list | None = None) -> list[np.ndarray]:
    """The near leg lit and the far leg dark in every frame, by the mannequin, as sprite artists draw a side walk.

    The model draws the leg behind darker, whichever it is: when the far leg should lead, the near leg comes out
    dark behind it, and the two steps look alike ("the same foot"); darkening the far leg alone (ma_far_shade) does
    not undo that. So each leg is brought to one brightness for the whole cycle: the near leg to the lit leg's
    usual brightness, the far leg to k of it.

    The far leg is darkened as a shade is painted, deeper in its own colour (each channel to a power), not by one
    factor: pale skin and white shoes scaled by 0.75 came out grey, a stocking and another shoe to the eye, on four
    girls of eight (11-05); to a power, skin goes a warmer skin and white stays white."""
    lum = [f[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114]) for f in frames]
    masks = [ma_leg_masks(f, d, p, e) for f, d, p, e in zip(frames, depths, pts or [None] * len(frames),
                                                             hems or [None] * len(frames))]
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
        # the near leg brought up only where the model drew it darker than the far one, and only to the far one's
        # light: lit to the cycle's usual brightness up to the hip, a near leg on bare skin blew out to white (11-04)
        # both to a power of their own colour, the power found so the leg's mean lands where it should: scaled up, a
        # pale patch on a girl's near shin went white (11-05)
        c = f[..., :3] / 255
        a = f[..., 3] > 128
        qn = _ma_power_to(c[(n > 0.8) & a], min(lf, 2.2 * ln)) if lf > ln else 0.0  # the near leg's mean to lf
        q = _ma_power_to(c[(fa > 0.8) & a], float(np.clip(k * lit, 0.3 * lf, 2.0 * lf)))  # the far's to k * lit
        o = f.copy()
        o[..., :3] = (255 * c ** (1 + qn * n[..., None] + q * fa[..., None])).clip(0, 255).round().astype(np.uint8)
        out.append(o)
    return out


def _ma_power_to(px: np.ndarray, target: float) -> float:
    """q such that the pixels (RGB 0..1, N x 3) to the power 1 + q have the mean grey `target` (0..255)."""
    if len(px) < 20:
        return 0.0
    w = np.array([0.299, 0.587, 0.114])
    lo, hi = -0.8, 4.0
    for _ in range(30):
        q = (lo + hi) / 2
        if float((np.clip(px, 1e-4, 1) ** (1 + q) @ w).mean()) * 255 > target:
            lo = q
        else:
            hi = q
    return (lo + hi) / 2


def ma_head_match(frame: np.ndarray, person: np.ndarray, master: np.ndarray, mperson: np.ndarray,
                  rows: int) -> tuple[float, float]:
    """The master's head (its top `rows` rows, grey on white) found in an RGBA frame at scales 0.85-1.18: (how well
    it matches at the best scale, 0-1, that scale)."""
    from skimage.feature import match_template
    from skimage.transform import rescale

    def grey(f):
        a = f[..., 3:4] / 255.0
        return (f[..., :3] * a + 255 * (1 - a)).mean(-1) / 255.0

    M = ma_marks(mperson)
    xs = np.nonzero(mperson[M["top"]:M["top"] + rows].any(0))[0]
    tpl = grey(master)[M["top"]:M["top"] + rows, xs.min():xs.max() + 1]
    m = ma_marks(person)
    cols = np.nonzero(person[m["top"]:m["top"] + rows].any(0))[0]
    g = grey(frame)
    best = (0.0, 1.0)
    for s in np.arange(0.85, 1.185, 0.01):
        t = rescale(tpl, s, anti_aliasing=True)
        y0, x0 = max(0, m["top"] - 20), max(0, cols.min() - 40)
        win = g[y0:m["top"] + t.shape[0] + 40, x0:cols.max() + 41]
        if win.shape[0] < t.shape[0] or win.shape[1] < t.shape[1]:
            continue
        r = float(match_template(win, t).max())
        if r > best[0]:
            best = (r, float(s))
    return best


def ma_head_sizes(frames: list[np.ndarray], people: list[np.ndarray], master: int, rows: int) -> list[float]:
    """How big each frame's head is against the master's (1.0 = the same): the scale its head matches best at
    (ma_head_match). A woman's long hair over her neck moved the neck read from the front edge 23 px in four frames
    of twelve, and her head drawn the same size was made 13 % bigger in those (11-06); the head itself does not move
    with the hair. A match under 0.5 says nothing: 1.0 then."""
    out = []
    for i, (f, p) in enumerate(zip(frames, people)):
        r, s = (1.0, 1.0) if i == master else ma_head_match(f, p, frames[master], people[master], rows)
        out.append(s if r > 0.5 else 1.0)
    return out


def ma_head_once(frames: list[np.ndarray], master: int = 0, feather: float = 0.12) -> list[np.ndarray]:
    """The head drawn once: the master's head, from the top of the hair to the chin, put on every frame where that
    frame's head matches it best (scale 0.94-1.06 and place), fading into the frame below the chin over `feather`
    of the head's height. Each frame of a video made sharp drew the face again: a girl's glasses rode higher or
    lower, her smile came and went and her parting moved from frame to frame, a face that changed as she walked
    (11-07); hand-drawn walks draw the head once and move it."""
    from skimage.feature import match_template
    from skimage.transform import rescale

    def grey(f):
        a = f[..., 3:4] / 255.0
        return (f[..., :3] * a + 255 * (1 - a)).mean(-1) / 255.0

    people = [ma_person(f[..., 3] > 128) for f in frames]
    marks = [ma_marks(p) for p in people]
    # the head's height from all the frames: frame 0's neck read low on a girl's collar over a carried box, and its
    # head band took the top of the box with it into every frame (11-07)
    rows = int(np.median([m["neck"] - m["top"] for m in marks]))
    p0, M = people[master], marks[master]
    ext = int(rows * (1 + feather))
    xs = np.nonzero(p0[M["top"]:M["top"] + rows].any(0))[0]
    x0, y0 = max(0, xs.min() - 6), max(0, M["top"] - 6)
    patch = frames[master][y0:M["top"] + ext, x0:xs.max() + 7].astype(np.float64)
    tpl = grey(frames[master])[M["top"]:M["top"] + rows, xs.min():xs.max() + 1]
    out = []
    for i, (f, p, m) in enumerate(zip(frames, people, marks)):
        if i == master:
            out.append(f)
            continue
        cols = np.nonzero(p[m["top"]:m["top"] + rows].any(0))[0]
        wy, wx = max(0, m["top"] - 30), max(0, cols.min() - 50)
        win = grey(f)[wy:m["top"] + rows + 40, wx:cols.max() + 51]
        best = (-1.0, 1.0, m["top"], cols.min())
        for s in np.arange(0.94, 1.065, 0.01):
            t = rescale(tpl, s, anti_aliasing=True)
            if win.shape[0] < t.shape[0] or win.shape[1] < t.shape[1]:
                continue
            r = match_template(win, t)
            j = np.unravel_index(r.argmax(), r.shape)
            if r[j] > best[0]:
                best = (float(r[j]), float(s), wy + j[0], wx + j[1])
        _, s, ty, tx = best
        P = np.stack([rescale(patch[..., c], s, anti_aliasing=True) for c in range(4)], -1)
        oy, ox = int(round(ty - (M["top"] - y0) * s)), int(round(tx - (xs.min() - x0) * s))
        h, w = P.shape[:2]
        k = np.ones(h)
        chin = int((M["top"] + rows - y0) * s)
        k[chin:] = np.linspace(1, 0, h - chin)
        o = f.astype(np.float64)
        ya, yb, xa, xb = max(0, oy), min(o.shape[0], oy + h), max(0, ox), min(o.shape[1], ox + w)
        Pc, kc = P[ya - oy:yb - oy, xa - ox:xb - ox], k[ya - oy:yb - oy, None, None]
        a1, a2 = Pc[..., 3:4] / 255, o[ya:yb, xa:xb, 3:4] / 255
        y = np.arange(ya, yb)[:, None, None]
        own = y < m["top"] + rows                       # this frame's own head: replaced, hair drawn wider too
        head = y - oy < chin                            # the rest of frame 0's head: laid over what is there
        # under the chin the frame keeps its own outline and only its colours fade from the head's (frame 0's
        # shoulder edge laid where this frame had none left a grey line down a girl's back)
        kc = np.where(head, kc, kc * (a1 > 0.5) * (a2 > 0.5))
        A = np.where(own, a1, np.where(head, a1 + a2 * (1 - a1), a2))
        top_w = np.where(own, 1.0, np.where(head, 1.0, kc))
        low_w = np.where(own, 0.0, np.where(head, 1 - a1, 1 - kc))
        C = (top_w * a1 * Pc[..., :3] + low_w * a2 * o[ya:yb, xa:xb, :3]) / np.maximum(top_w * a1 + low_w * a2, 1e-6)
        o[ya:yb, xa:xb, :3] = np.where(A > 0, C, o[ya:yb, xa:xb, :3])
        o[ya:yb, xa:xb, 3:4] = A * 255
        out.append(o.clip(0, 255).round().astype(np.uint8))
    return out


def ma_lock(frames: list[np.ndarray], phases: list[float], pts: list[dict], depths: list, master: int = 0,
            align_legs: bool = True) -> list[np.ndarray]:
    """RGBA frames (same size) -> RGBA frames with one height, one head size and an even step."""
    from scipy import ndimage
    people = [ma_person(f[..., 3] > 128) for f in frames]
    M = ma_marks(people[master])
    torso, height = M["neck"] - M["top"], M["floor"] - M["top"]
    heads = ma_head_sizes(frames, people, master, torso)
    tall = [ma_height_wave(t) for t in phases]
    H, W = frames[0].shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    out = []
    for i, (f, p) in enumerate(zip(frames, people)):
        m = ma_marks(p)
        top = M["floor"] - height * tall[i] / tall[master]
        s = 1 / heads[i]
        yo = np.array([0, top, top + torso, M["floor"], H - 1], float)      # output row -> source row
        ys = np.array([0, m["top"], m["top"] + torso * heads[i], m["floor"], H - 1], float)
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


def ma_colour_match(frames: list[np.ndarray], ref: np.ndarray, k: float = 0.5,
                    keep: list | None = None, groups: int = 8) -> list[np.ndarray]:
    """Each frame's colours brought k of the way to the person's picture, colour by colour: the picture's colours in
    `groups` groups (k-means in Lab), each pixel moved by its nearest groups' differences between the frame and the
    picture, softly. A video drew a yellow cardigan pale (saturation 0.52-0.59 for 0.71) and made sharp it stayed so;
    all the colours matched together by percentile put a red blouse's red into her legs (11-05). keep: per frame, a
    mask of pixels left as drawn (a carried box)."""
    from scipy.cluster.vq import kmeans2
    from skimage import color
    if len(ref) < 100:                                     # no figure in the picture to match to
        return [f.copy() for f in frames]
    r = color.rgb2lab(ref.reshape(-1, 1, 3)[..., :3] / 255.0).reshape(-1, 3)
    rng = np.random.default_rng(0)
    sample = r[rng.choice(len(r), min(len(r), 20000), replace=False)]
    groups = min(groups, len(np.unique(np.round(sample), axis=0)))          # a picture of few colours: fewer
    centres, _ = kmeans2(sample, groups, minit="++", seed=0)
    centres = centres[~np.isnan(centres).any(1)]
    groups = len(centres)
    out = []
    for i, f in enumerate(frames):
        m = f[..., 3] > 128
        if keep is not None and keep[i] is not None:
            m &= ~keep[i]
        if m.sum() < 100:
            out.append(f.copy())
            continue
        lab = color.rgb2lab(f[..., :3] / 255.0)
        px = lab[m]
        d = np.linalg.norm(px[:, None, :] - centres[None], axis=-1)
        near = d.argmin(1)
        shift = np.zeros_like(centres)
        for j in range(groups):                            # the frame's own mean of each group, to the picture's
            if (near == j).sum() > 50:
                shift[j] = centres[j] - px[near == j].mean(0)
        w = np.exp(-((d - d.min(1, keepdims=True)) ** 2) / (2 * 8.0 ** 2))
        w /= w.sum(1, keepdims=True)
        lab[m] = px + k * (w @ shift)
        o = f.copy()
        o[..., :3][m] = (np.clip(color.lab2rgb(lab), 0, 1) * 255).round().astype(np.uint8)[m]
        out.append(o)
    return out

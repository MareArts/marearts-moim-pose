"""Measured checks on a drawn frame, and its cut-out. They catch what the model gets wrong now and then: a third
arm or leg, a carried box that turns or bobs, white paper left inside the figure, and two steps half a cycle apart
drawn on the same foot."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .gait import H, W
from .mannequin import ma_box_rect

# Limits found on 192 frames checked by eye: a real arm outside the mannequin is ~15-20 (thousandths of height
# squared), passed frames at most 2.4; legs across the shins at most 2 (3 at one height where a shoe and a cuff part).
OUTSIDE_MAX, LEGS_MAX = 6.0, 3
BOX_END_MAX, BOX_TOP_MAX = 0.15, 10.0
# Two steps half a cycle apart must lead with different legs. On 32 frames of two sample characters, pairs that read
# apart to the eye had 12 grey levels of lead or more each way; same-footed pairs had the same sign, or one under 5.
SEEN = 10.0
APART, SPREAD = 0.12, 0.30            # the mannequin's feet apart (share of height); without it, the legs' width
FACING_SURE = 0.006                   # toes past the ankles by less than this share of the height: cannot tell
# An arm the model swung on its own (a hand reaching forward where the mannequin's arms hang): on 24 walk frames of
# two characters, arms as the mannequin had them reached at most 4.4 % of the height past it, swung ones 6.8-12.5.
REACH_MAX = 5.0


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


def ma_arm_reach(mask: np.ndarray, depth: Image.Image, pts: dict) -> float:
    """How far the drawn arms reach past the mannequin's, in % of the person's height: in the rows from below the
    shoulders to just under the lower hand, the person's front edge past the mannequin's, and its back edge, the
    90th percentile of each, the larger."""
    from .gait import LSHO, RSHO
    m = ma_person(mask)
    man = np.asarray(depth.convert("L").resize(m.shape[::-1])) > 20
    ys = np.nonzero(m.any(1))[0]
    h = ys.max() - ys.min()
    r0 = int(pts[LSHO][1] + 0.08 * h)
    r1 = int(max(pts[LSHO + 2][1], pts[RSHO + 2][1]) + 0.03 * h)
    ahead, behind = [], []
    for r in range(r0, min(r1, len(m))):
        a, b = np.nonzero(m[r])[0], np.nonzero(man[r])[0]
        if len(a) and len(b):
            ahead.append(a.max() - b.max())
            behind.append(b.min() - a.min())
    if not ahead:
        return 0.0
    return float(max(np.percentile(ahead, 90), np.percentile(behind, 90)) / h * 100)


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
    if kind == "walk":
        reach = ma_arm_reach(mask, depth, pts)
        said += f", arms {reach:.1f}"
        ok = ok and reach <= REACH_MAX
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


def ma_facing(mask: np.ndarray) -> float:
    """Which way a standing person faces, from the feet: the shoes reach past the ankles toward the toes.

    + right, - left, as a share of the height; near 0 (under FACING_SURE) it cannot tell. On 8 pictures it was
    right on 7 and unsure on the eighth (feet set one behind the other)."""
    m = ma_person(mask)
    ys = np.nonzero(m.any(1))[0]
    if len(ys) < 20:
        return 0.0
    top, floor = int(ys.min()), int(ys.max())
    h = floor - top

    def cx(a, b):
        xs = np.nonzero(m[int(floor - a * h):int(floor - b * h) + 1])[1]
        return xs.mean() if len(xs) else 0.0
    return float((cx(0.03, 0.0) - cx(0.10, 0.06)) / max(1, h))


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



def ma_leg_lead(frame: np.ndarray) -> tuple[float, float]:
    """(lead, spread) of the legs of an RGBA frame facing right.

    In profile both legs wear the same trousers, and what tells them apart is the light: the near leg is drawn
    lighter. lead: how much lighter the front third of the legs is than the back third (grey levels), > 0 when the
    near leg is in front. Measured from mid-thigh to the ankles, so a coat's hem and the shoes stay out.
    spread: the legs' width as a share of the height.
    """
    m = ma_person(frame[..., 3] > 128)
    ys = np.nonzero(m.any(1))[0]
    if not len(ys):
        return 0.0, 0.0
    top, h = ys.min(), ys.max() - ys.min()
    r0, r1 = top + int(0.62 * h), top + int(0.94 * h)
    legs = m[r0:r1]
    grey = frame[r0:r1, :, :3].astype(np.float64).mean(-1)
    cols = np.nonzero(legs.any(0))[0]
    if len(cols) < 6:
        return 0.0, 0.0
    x0, x1 = cols.min(), cols.max()
    third = (x1 - x0) / 3
    front, back = legs.copy(), legs.copy()
    front[:, :int(x1 - third)] = False
    back[:, int(x0 + third) + 1:] = False
    if front.sum() < 20 or back.sum() < 20:
        return 0.0, float((x1 - x0) / max(1, h))
    return float(grey[front].mean() - grey[back].mean()), float((x1 - x0) / max(1, h))


def ma_near_ahead(depth: Image.Image) -> float:
    """How far the mannequin's near foot is ahead of its far foot, as a share of its height (< 0: behind). 0 when
    one foot is off the bottom rows (mid-swing)."""
    d = np.asarray(depth.convert("L"))
    ys = np.nonzero((d > 30).any(1))[0]
    if not len(ys):
        return 0.0
    top, bot = ys.min(), ys.max()
    low = d[int(bot - 0.15 * (bot - top)):bot + 1]
    near = np.nonzero(low > 190)[1]
    far = np.nonzero((low > 45) & (low < 115))[1]
    if not len(near) or not len(far):
        return 0.0
    return float((near.mean() - far.mean()) / max(1, bot - top))


def ma_alternates(frames: list[np.ndarray], depths: list | None = None) -> dict:
    """Whether the two steps of a cycle lead with different legs ("walking on the same foot" when not).

    Frame i and frame i + half are half a cycle apart: one has the near leg in front, the other the far one, so
    their leads (ma_leg_lead) must have opposite signs, each at least SEEN. Only pairs with the legs well apart are
    measured; at mid-stance the legs cross and the light says nothing.

    depths: the frames' mannequins. They say which leg should be in front, so a failed pair names the frame drawn
    the wrong way (or too faint); without them both frames of a failed pair fail.
    Returns {"ok", "fail": frame numbers, "pairs": [{frames, lead, measured, ok, fail}], "lead": per frame}.
    """
    half = len(frames) // 2
    got = [ma_leg_lead(f) for f in frames]
    ahead = [ma_near_ahead(d) for d in depths] if depths is not None else None
    pairs, fail = [], set()
    for i in range(half):
        j = i + half
        (a, sa), (b, sb) = got[i], got[j]
        row = dict(frames=[i, j], lead=[round(a, 1), round(b, 1)])
        if ahead is not None:
            measured = min(abs(ahead[i]), abs(ahead[j])) >= APART and ahead[i] * ahead[j] < 0
        else:
            measured = min(sa, sb) >= SPREAD
        ok = not measured or (a * b < 0 and min(abs(a), abs(b)) >= SEEN)
        bad = []
        if not ok:
            if ahead is not None:      # the wrong way round, or too faint to see
                bad = [k for k, v in ((i, a), (j, b)) if v * np.sign(ahead[k]) < SEEN]
            bad = bad or [i, j]
        row.update(measured=bool(measured), ok=bool(ok), fail=bad)
        pairs.append(row)
        fail |= set(bad)
    return {"ok": not fail, "fail": sorted(fail), "pairs": pairs, "lead": [round(g[0], 1) for g in got]}


def ma_alternate_reason(alt: dict, i: int) -> str:
    """A few words for the review sheet on why frame i failed ma_alternates ("" if it did not)."""
    for p in alt["pairs"]:
        if i in p["fail"]:
            a, b = p["lead"]
            j = p["frames"][1] if p["frames"][0] == i else p["frames"][0]
            return f"same leg as #{j}" if a * b > 0 else f"legs faint vs #{j}"
    return ""

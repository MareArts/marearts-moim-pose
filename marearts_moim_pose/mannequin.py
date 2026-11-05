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
SKIRT = 190                                                          # a skirt: over both thighs, under the near arm


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


def ma_depth(pts: dict, head: float = 0.19, box: bool = False, bust: float = 0.0,
             seat: float = 0.06, skirt: float = 0.0, flare: float | None = None, body: float = 0.16,
             torso: dict | None = None) -> Image.Image:
    """The depth mannequin of a pose (RGB, near bright, far dark).

    head: head width as a share of the height. Use the person's own (a big anime head with hair is about 0.19), or
    the model shrinks the head toward its default. The chin stays put; the head grows back and up.
    box: a carried box before the chest, nearer than the body. bust: how far the chest stands out (share of the
    height, 0 for none, about 0.13 for a fuller figure); the depth picture sets the silhouette, so say it here.
    seat: how far below the hips the body covers the top of the near thigh (share of the height, 0 for the thigh
    over it). With the thigh drawn over the body, a video drew the trousers' seat where an older man's jacket hangs
    behind his hips, in four frames of twelve with the near leg back; with the seat over it, his jacket covered his
    hips in all twelve (11-04).
    skirt: where a skirt or dress ends, along the leg (0 none, 1 the knee, 2 the ankle; ma_skirt_of reads it from the
    words). The legs alone, split at the hip, made a video draw shorts or trousers for a skirt to the knee, a pleated
    skirt cut to mid-thigh, and the near thigh showing through the cloth (11-05); the skirt over both thighs keeps it.
    flare: how far the skirt stands off the legs at its hem (share of the height; ma_skirt_flare measures it on the
    picture). One width for all made a pencil skirt an A-line one (11-05).
    body: how thick the body is from front to back (share of the height; ma_fit measures it with head and bust).
    torso: the body's own outline from the neck past the hips (ma_fit, from the person's picture): drawn in place of
    the tube and the chest, so a slim waist, a chest and the hips come through to the drawing.
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

    def leg(hip_i, v):
        hp, kn, an = pts[hip_i], pts[hip_i + 1], pts[hip_i + 2]
        cap(hp, kn, 0.1, v)
        cap(kn, an, 0.072, v)
        cap(an, (an[0] + 0.075 * s, an[1] + 0.02 * s), 0.035, v)          # the foot, pointing right

    def arm(sho_i, v):
        sh, el, wr = pts[sho_i], pts[sho_i + 1], pts[sho_i + 2]
        cap(sh, el, 0.055, v)
        cap(el, wr, 0.045, v)

    def side(hip_i, sho_i, v):
        leg(hip_i, v)
        arm(sho_i, v)

    side(RHIP, RSHO, FAR)
    neck, hip = pts[NECK_I], ((pts[LHIP][0] + pts[RHIP][0]) / 2, pts[LHIP][1])
    if torso:
        d.polygon(ma_torso_outline(pts, torso), fill=BODY)
    else:
        cap((neck[0], neck[1] + 0.04 * s), hip, body, BODY)
    hx, hy, hw = pts[NOSE][0] - 0.05 * s, pts[NOSE][1] - 0.01 * s, head * s
    d.ellipse([hx + 0.065 * s - hw, hy + 0.07 * s - 1.115 * hw, hx + 0.065 * s, hy + 0.07 * s], fill=HEAD)
    # a nose: the face in profile, facing right. A plain oval said nothing of where the face looks, and a video
    # model turned an older man's face to the viewer and drew another man (head likeness 0.51-0.65); with the nose
    # he stayed himself (0.87-0.95), 11-04
    fx, cy = hx + 0.065 * s, hy + 0.07 * s - 0.45 * 1.115 * hw
    d.polygon([(fx - 0.035 * s, cy - 0.022 * s), (fx + 0.018 * s, cy + 0.008 * s), (fx - 0.035 * s, cy + 0.022 * s)],
              fill=HEAD + 25)
    if bust and not torso:
        cx, cy = neck[0] + (hip[0] - neck[0]) * 0.24, neck[1] + (hip[1] - neck[1]) * 0.24
        d.ellipse([cx + 0.01 * s, cy - 0.065 * s, cx + bust * s, cy + 0.055 * s], fill=CHEST)
    if box:
        d.rectangle(ma_box_rect(pts), fill=BOX)
    if seat:                                    # the seat over the top of the near thigh, then the near arm
        leg(LHIP, NEAR)
        if torso:                               # its outline from just above the hips down over the seat
            d.polygon(ma_torso_outline(pts, torso, start=0.9), fill=BODY)
        else:
            r = body / 2 * s
            d.ellipse([hip[0] - r, hip[1] - r, hip[0] + r, hip[1] + seat * s], fill=BODY)
    else:
        leg(LHIP, NEAR)
    if skirt:
        d.polygon(ma_skirt_outline(pts, skirt, flare), fill=SKIRT)
    arm(LSHO, NEAR)
    a = ndimage.gaussian_filter(np.asarray(im, np.float32), 3)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8)).convert("RGB")


SKIRT_WORDS = ((("maxi", "floor", "ankle-length", "long"), 1.85), (("midi", "calf", "below the knee"), 1.4),
               (("mini", "short", "mid-thigh", "above the knee"), 0.75))


def ma_skirt_of(look: str) -> float:
    """Where the skirt or dress in a person's words ends (ma_depth's skirt): 0 with none, the knee unless the words
    of that garment say. Read in the garment's own phrase: "tan ankle boots" after a midi skirt made it a maxi, and a
    dress shirt or dress shoes are no dress."""
    import re
    for part in look.lower().split(","):
        if re.search(r"\b(skirt|dress|gown|kilt)\b(?!\s*(shirt|shoes|trousers|pants|boots))", part):
            return next((v for keys, v in SKIRT_WORDS if any(re.search(rf"\b{k}\b", part) for k in keys)), 1.05)
    return 0.0


def ma_skirt_at(hem: float) -> float:
    """A hem measured on a standing picture (share of the height from the top of the head) as ma_depth's skirt."""
    from .gait import HIP, SHIN, THIGH
    hip, knee = 1 - HIP, 1 - HIP + THIGH
    return float(np.clip((hem - hip) / THIGH if hem <= knee else 1 + (hem - knee) / SHIN, 0.4, 2.0))


TORSO_ROWS = 12                                        # the torso's outline: rows from the neck to the hips
TORSO_SEAT = 0.06                                      # and on past the hips over the seat (share of the height)
BACK_MAX, BACK_HIP = 0.62, 0.78                        # how far a back stands behind the waist's middle (of its
                                                       # thickness): above the waist, and the hips and seat below


def ma_fit(person: np.ndarray) -> dict:
    """The mannequin's shape from a standing side picture's person mask (facing right), as shares of its height:
    head the widest row of the top eighth (hair too); body the thickness at the waist (the narrowest row from the
    waist to the hips: long hair down the back stops above it); torso the body's outline from the neck to the hips at
    TORSO_ROWS rows, front and back from the waist's middle: the front as drawn (chest, waist, belly), the back as
    drawn but no further out than a back stands (BACK_MAX of the waist above it, BACK_HIP at the hips), so hair
    hanging behind is not body. A tube of one thickness for everyone drew slim women with the waist, chest and hips
    gone, the body 0.16 thick where they are 0.11-0.13, a neck covered by its top (0.174 of the height for 0.194),
    a chest 0.19 for 0.14 where a table said "full", and a head 0.19 for 0.156 (11-05)."""
    from .gait import HIP
    from .lock import ma_marks
    ys = np.nonzero(person.any(1))[0]
    if len(ys) < 20:                                       # no figure to measure: the usual anime proportions
        return {"head": 0.19, "body": 0.16, "bust": 0.0, "torso": None}
    top, h = ys.min(), max(1, ys.max() - ys.min())

    def row(f):
        xs = np.nonzero(person[int(top + f * h)])[0]
        return (int(xs.min()), int(xs.max())) if len(xs) else None
    neck = (ma_marks(person)["neck"] - top) / h
    # the head as wide as drawn (hair round it too), but no bigger than its height to the neck makes it: a ponytail
    # standing out behind measured a head 0.23 wide and the mannequin's round head grew up and back with it
    head = max((r[1] - r[0] for r in (row(f) for f in np.linspace(0.03, 0.12, 10)) if r), default=0) / h
    head = min(head, (neck - 0.015) / 1.115)
    waist = [(f, r) for f in np.linspace(0.38, 0.50, 13) if (r := row(f))]
    if not waist:
        return {"head": 0.19, "body": 0.16, "bust": 0.0, "torso": None}
    fw, thin = min(waist, key=lambda x: x[1][1] - x[1][0])
    t = thin[1] - thin[0]
    body, axis = t / h, (thin[0] + thin[1]) / 2
    rows = []
    for f in np.linspace(neck, 1 - HIP + TORSO_SEAT, TORSO_ROWS):
        r = row(f) or thin
        most = (BACK_MAX if f < fw else BACK_HIP) * t
        rows.append([round((r[1] - axis) / h, 4), round(min(axis - r[0], most) / h, 4)])
    torso = {"neck": round(float(neck), 4), "rows": rows}
    out = (np.median([r[1] for f in np.linspace(0.26, 0.32, 7) if (r := row(f))]) - thin[1]) / h
    bust = body / 2 + out if out > 0.01 else 0.0
    return {"head": round(float(np.clip(head, 0.12, 0.26)), 3), "body": round(float(np.clip(body, 0.09, 0.22)), 3),
            "bust": round(float(bust), 3), "torso": torso}


def ma_skirt_flare(person: np.ndarray, hem: float) -> float:
    """How far a skirt stands off the legs (ma_depth's flare), from a standing picture's person mask and its hem
    (share of the height): the figure's width six tenths of the way from the hips down to the hem. On 12 the game skirts
    the pencil and straight ones were 0.11-0.13 of the height wide there, the A-line and pleated ones 0.16-0.19."""
    from .gait import HIP
    ys = np.nonzero(person.any(1))[0]
    top, h = ys.min(), ys.max() - ys.min()
    row = person[int(top + h * ((1 - HIP) + 0.6 * (hem - (1 - HIP))))]
    xs = np.nonzero(row)[0]
    width = (xs.max() - xs.min()) / max(1, h) if len(xs) else 0.13
    return float(np.clip(width - 0.07, 0.03, 0.13))


def ma_skirt_hem(pts: dict, skirt: float, flare: float | None = None) -> tuple[float, float, float]:
    """A skirt's hem: (back x, front x, y). The hem spans both legs where they are `skirt` along their length."""
    def along(hip_i):
        hp, kn, an = pts[hip_i], pts[hip_i + 1], pts[hip_i + 2]
        a, b, f = (hp, kn, skirt) if skirt <= 1 else (kn, an, skirt - 1)
        return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
    (x0, y0), (x1, y1) = along(LHIP), along(RHIP)
    # the cloth stands off the legs: as measured, or more the longer it is
    room = FIG * (flare if flare is not None else 0.055 + 0.03 * min(skirt, 1.5))
    return min(x0, x1) - room, max(x0, x1) + room, max(y0, y1)


def ma_skirt_outline(pts: dict, skirt: float, flare: float | None = None) -> list[tuple[float, float]]:
    """The skirt from the waist (the body's width) to its hem, as a polygon."""
    hip = ((pts[LHIP][0] + pts[RHIP][0]) / 2, pts[LHIP][1])
    back, front, y = ma_skirt_hem(pts, skirt, flare)
    w = 0.08 * FIG
    top = hip[1] - 0.06 * FIG
    return [(hip[0] - w, top), (hip[0] + w, top), (front, y), (back, y)]


def ma_torso_outline(pts: dict, torso: dict, start: float = 0.0) -> list[tuple[float, float]]:
    """The torso outline (ma_fit) laid along the mannequin's body, neck to hips and past them, leaning as it leans:
    a polygon, from `start` of the way from the neck to the hips."""
    from .gait import HIP
    neck = np.array(pts[NECK_I], float)
    hip = np.array(((pts[LHIP][0] + pts[RHIP][0]) / 2, pts[LHIP][1]), float)
    a = hip - neck
    n = np.array([-a[1], a[0]]) / max(1e-6, np.hypot(*a)) * -1       # the front, to the right when upright
    rows = torso["rows"]
    end = 1 + TORSO_SEAT / max(1e-3, (1 - HIP) - torso["neck"])
    us = np.linspace(0, end, len(rows))
    keep = [k for k, u in enumerate(us) if u >= start]
    front = [tuple(neck + a * us[k] + n * rows[k][0] * FIG) for k in keep]
    back = [tuple(neck + a * us[k] - n * rows[k][1] * FIG) for k in reversed(keep)]
    return front + back


def ma_box_rect(pts: dict) -> tuple[float, float, float, float]:
    """Where the carried box is in the mannequin: (left, top, right, bottom), on the forearms before the chest."""
    neck, wr = pts[NECK_I], pts[LSHO + 2]
    return (neck[0] + 0.03 * FIG, wr[1] - 0.17 * FIG, wr[0] + 0.02 * FIG, wr[1] + 0.03 * FIG)


def ma_legs_mask(depth: Image.Image) -> np.ndarray:
    """The mannequin's figure as a boolean mask (anything brighter than the background)."""
    return np.asarray(depth.convert("L")) > 30


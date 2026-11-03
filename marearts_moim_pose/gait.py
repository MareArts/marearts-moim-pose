"""A side-view walk cycle: where each joint is in each frame of a walk facing right.

The figure is seen from its left, so the left limbs are the near ones. Segment lengths are shares of body height
(Drillis & Contini). Points use the 18-point OpenPose body order so the skeleton can also be drawn the way pose
detectors draw it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

W, H = 832, 1216                     # one frame; the figure stands about 84 % of it
FIG, GROUND = 0.84 * H, H - 0.07 * H
UPPER, FORE, THIGH, SHIN = 0.186, 0.146, 0.245, 0.246
ANKLE, HIP, NECK = 0.039, 0.53, 0.83
BOB = 0.012                          # the body rises and falls 1.2 % of its height each step

NOSE, NECK_I, RSHO, RELB, RWRI, LSHO, LELB, LWRI, RHIP, RKNE, RANK, LHIP, LKNE, LANK, REYE, LEYE, REAR, LEAR = range(18)


@dataclass(frozen=True)
class MaGait:
    """How one person walks.

    step: stride as a share of the base stride · knee: how far the swinging knee bends · arm: arm swing in degrees
    each way · lean: body lean forward in degrees · back: extra reach of the backward arm swing (edit models draw
    it short, so the skeleton asks for a little more) · elbow: how much the elbow bends (1: 12 degrees, more on the
    forward swing; less keeps the hand nearer the hip).
    """
    step: float = 0.85
    knee: float = 0.85
    arm: float = 17.0
    lean: float = 0.5
    back: float = 1.6
    elbow: float = 1.0


# Walks that read as different people. Arms below ~14 degrees look stuck to the hip once drawn; under a jacket 14
# was not enough either (the hand stayed by the hip in 3 of 8 frames), 19 was.
PRESETS = {
    "neat": MaGait(step=0.80, knee=0.85, arm=15, lean=0.5),
    "relaxed": MaGait(step=0.85, knee=0.80, arm=15, lean=1.5),
    "lively": MaGait(step=0.85, knee=1.00, arm=19, lean=0.5),
    "upright": MaGait(step=0.92, knee=0.85, arm=17, lean=0.0),
    "calm": MaGait(step=0.88, knee=0.85, arm=16, lean=0.5),
    "graceful": MaGait(step=0.72, knee=0.80, arm=15, lean=0.0),
    "heavy": MaGait(step=0.95, knee=0.90, arm=20, lean=1.5),
    "older": MaGait(step=0.62, knee=0.70, arm=19, lean=3.5),
    "energetic": MaGait(step=0.88, knee=1.00, arm=21, lean=0.5),
    "loose": MaGait(step=1.03, knee=0.95, arm=24, lean=1.0),
}


def ma_gait(name_or_gait: str | MaGait | None = None, **changes) -> MaGait:
    """A preset by name (or a MaGait), with any field changed."""
    g = PRESETS[name_or_gait] if isinstance(name_or_gait, str) else (name_or_gait or MaGait())
    return replace(g, **changes) if changes else g


def ma_leg_angles(t: float) -> tuple[float, float]:
    """Thigh angle from the vertical (+ forward) and knee bend, in degrees, for a leg at phase t.

    0 to 0.5 the leg carries the body: heel strike in front, the thigh going back evenly. 0.5 to 1 it swings: the
    knee bends most just after the toes leave, so at mid-swing the foot is already under the body and clearly
    passes the other leg (a symmetric swing makes the legs look like they never cross).
    """
    if t < 0.5:
        u = t / 0.5
        thigh = 24 - 48 * u
        knee = 14 * math.sin(math.pi * u) ** 2
        if t >= 0.375:                                  # the back knee gives as the heel rises
            knee += 15 * (t - 0.375) / 0.125
        return thigh, knee
    u = (t - 0.5) / 0.5
    thigh = -24 + 48 * (1 - (1 - u) ** 3)
    knee = 15 + 47 * math.sin(math.pi / 2 * u / 0.3) if u < 0.3 else 62 * math.cos(math.pi / 2 * (u - 0.3) / 0.7) ** 2
    return thigh, knee


def _ankle_ahead(th: float, kn: float) -> float:
    """The ankle ahead of the hip (share of the height) for a thigh angle and knee bend in degrees."""
    return THIGH * math.sin(math.radians(th)) + SHIN * math.sin(math.radians(th - kn))


def ma_stance_thigh(phase: float, kn: float, g: MaGait) -> float:
    """Thigh angle of the standing leg at phase (0 to 0.5): the one that puts the ankle where a steady walk has it.

    A body walks at a steady speed, so the planted foot goes back the same distance in each equal time. A thigh
    turning evenly does not do that (the foot went back 108, 109, then 58 px), and frames picked at uneven times to
    make up for it moved the arms and the swinging leg in jumps (a hand 28 to 170 px between frames). The knee bends
    as before; the thigh is solved for the ankle, the same as before at heel strike and toe-off."""
    k0, k1 = ma_leg_angles(0.0)[1] * g.knee, ma_leg_angles(0.4999)[1] * g.knee
    x0, x1 = _ankle_ahead(24 * g.step, k0), _ankle_ahead(-24 * g.step, k1)
    want = x0 + (x1 - x0) * phase / 0.5
    lo, hi = -60.0, 60.0                                 # the ankle goes forward as the thigh does
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _ankle_ahead(mid, kn) < want else (lo, mid)
    return (lo + hi) / 2


def ma_walk_points(t: float, gait: MaGait | None = None, carry: bool = False) -> dict[int, tuple[float, float]]:
    """Joint positions (px on the W x H frame) at phase t of the walk.

    The left leg strikes at t = 0, the right at 0.5; each arm swings with the other side's leg. Carrying, both
    forearms come forward under a box and the upper body rides a smooth wave instead of the hip's small hop, so a
    carried box does not jump.
    """
    g = gait or MaGait()
    s = FIG
    hip = (W / 2 - 0.02 * s, 0.0)
    pts, low = {}, []
    for side, phase, hip_i in (("L", t % 1, LHIP), ("R", (t + 0.5) % 1, RHIP)):
        th, kn = ma_leg_angles(phase)
        th, kn = th * g.step, kn * g.knee
        if phase < 0.5:
            th = ma_stance_thigh(phase, kn, g)
        a, b = math.radians(th), math.radians(th - kn)
        top = (hip[0] - (0.012 * s if side == "R" else 0), 0.0)
        knee = (top[0] + THIGH * s * math.sin(a), top[1] + THIGH * s * math.cos(a))
        ank = (knee[0] + SHIN * s * math.sin(b), knee[1] + SHIN * s * math.cos(b))
        pts[hip_i], pts[hip_i + 1], pts[hip_i + 2] = top, knee, ank
        low.append(ank[1])
    dy = GROUND - ANKLE * s - max(low)                  # the lower ankle stands on the ground
    pts = {k: (x, y + dy) for k, (x, y) in pts.items()}
    body_y = dy
    if carry:
        top = min(ma_walk_points(u, g)[LHIP][1] for u in np.linspace(0, 1, 41))
        body_y = top + s * BOB * (1 + math.cos(4 * math.pi * t)) / 2
    neck = (hip[0] + math.tan(math.radians(g.lean)) * (NECK - HIP) * s, body_y - (NECK - HIP) * s)
    pts[NECK_I] = neck
    for side, sho_i, sign in (("L", LSHO, -1), ("R", RSHO, 1)):
        if carry:
            up, fo = math.radians(20), math.radians(92)
        else:
            # one smooth wave from arm forward to arm * back behind: scaling only the back half made the arm speed
            # up as it passed the hip, and the elbow kinked there too
            c = math.cos(2 * math.pi * t) * sign
            sw = g.arm * ((1 + g.back) / 2 * c + (1 - g.back) / 2)
            up, fo = math.radians(sw), math.radians(sw + g.elbow * (12 + 0.6 * g.arm * (1 + c) / 2))
        sh = (neck[0] - (0.014 * s if side == "R" else 0), neck[1] + 0.025 * s)
        el = (sh[0] + UPPER * s * math.sin(up), sh[1] + UPPER * s * math.cos(up))
        wr = (el[0] + FORE * s * math.sin(fo), el[1] + FORE * s * math.cos(fo))
        pts[sho_i], pts[sho_i + 1], pts[sho_i + 2] = sh, el, wr
    pts[NOSE] = (neck[0] + 0.07 * s, neck[1] - 0.075 * s)
    pts[LEYE] = (neck[0] + 0.055 * s, neck[1] - 0.095 * s)
    pts[LEAR] = (neck[0], neck[1] - 0.085 * s)
    return pts


def ma_frame_phases(frames: int = 8, gait: MaGait | None = None) -> list[float]:
    """Walk phases for the frames, chosen so the planted foot goes back the same distance every frame (a game moves
    the sprite the same distance each frame, or the foot slides). The standing leg is solved for a steady foot
    (ma_stance_thigh), so these come out even (i / frames) and the arms and the swinging leg move evenly too.
    Half the frames per step.
    """
    half = frames // 2
    ts = np.linspace(0, 0.5, 201)
    xs = np.minimum.accumulate([ma_walk_points(t, gait)[LANK][0] for t in ts])     # the planted foot only goes back
    want = np.linspace(xs[0], xs[-1], half + 1)[:half]
    first = [float(np.interp(-w, -xs, ts)) for w in want]
    return first + [t + 0.5 for t in first]


def ma_stride_px(frames: int = 8, gait: MaGait | None = None) -> float:
    """How far (px on the W x H frame) the sprite must move each frame for the planted foot to stay put."""
    ts = np.linspace(0, 0.5, 201)
    xs = np.minimum.accumulate([ma_walk_points(t, gait)[LANK][0] for t in ts])
    return float((xs[0] - xs[-1]) / (frames // 2))


def ma_height_wave(t: float) -> float:
    """Figure height against its tallest at phase t: lowest with both feet down (0, 0.5), tallest over the standing
    foot (0.25, 0.75). One smooth wave a step, so the drawn head never hops."""
    return 1 - BOB * (1 + math.cos(4 * math.pi * t)) / 2

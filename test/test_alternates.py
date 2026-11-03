"""Two steps half a cycle apart must lead with different legs. The frames are depth mannequins painted as a model
draws them, near limbs lighter; "on the same foot" is the pair drawn with the same leg lit in front."""
import numpy as np

from marearts_moim_pose.checks import ma_alternate_reason, ma_alternates, ma_leg_lead, ma_near_ahead
from marearts_moim_pose.gait import ma_frame_phases, ma_walk_points
from marearts_moim_pose.mannequin import ma_depth


def _paint(depth, swap=False, flat=False):
    """RGBA: the mannequin painted in one trouser colour, lighter where nearer (or the far leg lit, or no light)."""
    d = np.asarray(depth.convert("L")).astype(np.float64)
    if swap:
        d = np.where(d > 30, 295 - d, d)
    shade = np.full_like(d, 0.7) if flat else 0.35 + 0.65 * d / 255
    rgb = shade[..., None] * np.array([90, 120, 200])
    return np.dstack([rgb, (d > 30) * 255]).astype(np.uint8)


def _cycle():
    depths = [ma_depth(ma_walk_points(t)) for t in ma_frame_phases(8)]
    return [_paint(d) for d in depths], depths


def test_cycle_drawn_right_alternates():
    frames, depths = _cycle()
    a = ma_alternates(frames, depths)
    assert a["ok"] and a["fail"] == [], a
    assert [p["measured"] for p in a["pairs"]] == [True, True, False, True]      # mid-stance not measured
    assert a["lead"][0] > 10 and a["lead"][4] < -10
    assert ma_alternates(frames)["ok"]                                            # without the mannequins too


def test_same_foot_fails_the_wrong_frame():
    frames, depths = _cycle()
    frames[4] = _paint(depths[4], swap=True)            # frame 4 with the far leg lit: the same leg leads as in 0
    a = ma_alternates(frames, depths)
    assert not a["ok"] and a["fail"] == [4], a
    assert ma_alternate_reason(a, 4) == "same leg as #0" and ma_alternate_reason(a, 0) == ""
    assert ma_alternates(frames)["fail"] == [0, 4]      # without the mannequins both frames of the pair fail
    frames[4] = frames[0].copy()                        # the same picture twice
    assert ma_alternates(frames, depths)["fail"] == [4]


def test_faint_legs_fail():
    frames, depths = _cycle()
    frames[5] = _paint(depths[5], flat=True)             # both legs one shade: nothing tells them apart
    lead, spread = ma_leg_lead(frames[5])
    assert abs(lead) < 1 and spread > 0.25
    a = ma_alternates(frames, depths)
    assert a["fail"] == [5] and ma_alternate_reason(a, 5) == "legs faint vs #1"


def test_mannequin_says_which_leg_leads():
    p0, p5 = ma_walk_points(0.0), ma_walk_points(0.5)
    assert ma_near_ahead(ma_depth(p0)) > 0.2 and ma_near_ahead(ma_depth(p5)) < -0.2

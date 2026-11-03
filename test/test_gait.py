import numpy as np

from marearts_moim_pose.gait import (LANK, LHIP, LWRI, NECK_I, PRESETS, RANK, MaGait, ma_frame_phases, ma_gait,
                                        ma_height_wave, ma_stride_px, ma_walk_points)


def planted_x(phases, gait):
    """x of the standing ankle in each frame (the left leg stands for t in [0, 0.5))."""
    return [ma_walk_points(t, gait)[LANK if t % 1 < 0.5 else RANK][0] for t in phases]


def test_planted_foot_goes_back_evenly():
    for name, g in PRESETS.items():
        ph = ma_frame_phases(8, g)
        xs = planted_x(ph[:4], g)
        steps = -np.diff(xs)
        assert (steps > 0).all(), name
        assert steps.std() < 0.08 * steps.mean(), (name, steps)


def test_frames_even_in_time_so_arms_do_not_jump():
    for name, g in PRESETS.items():
        ph = ma_frame_phases(8, g)
        assert np.allclose(ph, [i / 8 for i in range(8)], atol=0.01), (name, ph)
    g = ma_gait("lively")
    pts = [ma_walk_points(t, g) for t in ma_frame_phases(8, g)]
    hand = [np.hypot(*np.subtract(pts[(i + 1) % 8][LWRI], pts[i][LWRI])) for i in range(8)]
    assert max(hand) < 3 * min(hand)                       # was 28 to 170 px: a jump every half cycle


def test_stride_matches_frames():
    g = MaGait()
    ph = ma_frame_phases(8, g)
    xs = planted_x(ph[:4], g)
    assert abs(-np.diff(xs).mean() - ma_stride_px(8, g)) < 0.1 * ma_stride_px(8, g)


def test_legs_swap_each_step():
    g = MaGait()
    a, b = ma_walk_points(0.0, g), ma_walk_points(0.5, g)
    assert a[LANK][0] > a[RANK][0]          # left (near) foot in front at its strike
    assert b[RANK][0] > b[LANK][0]          # right foot in front half a cycle later


def test_feet_on_the_ground():
    g = MaGait()
    for t in np.linspace(0, 1, 17, endpoint=False):
        p = ma_walk_points(t, g)
        assert abs(max(p[LANK][1], p[RANK][1]) - max(ma_walk_points(0, g)[LANK][1], ma_walk_points(0, g)[RANK][1])) < 1e-6


def test_carry_upper_body_rides_smooth_wave():
    g = MaGait()
    ys = np.array([ma_walk_points(t, g, carry=True)[NECK_I][1] for t in np.linspace(0, 1, 33)])
    assert np.abs(np.diff(ys, 2)).max() < 2.0               # no hop
    assert ma_walk_points(0.25, g, carry=True)[LHIP] == ma_walk_points(0.25, g)[LHIP]


def test_height_wave_and_presets():
    assert ma_height_wave(0.0) < ma_height_wave(0.25)
    assert ma_gait("older").step < ma_gait("loose").step
    assert ma_gait("calm", arm=30).arm == 30


def test_old_name_still_imports():
    import marearts_moim_pose
    import marearts_spritestride
    from marearts_spritestride.gait import ma_gait as old
    assert old is ma_gait and marearts_spritestride.__version__ == marearts_moim_pose.__version__

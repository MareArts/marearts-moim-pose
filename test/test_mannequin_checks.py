import numpy as np
from PIL import Image

from marearts_spritestride.checks import REACH_MAX, ma_arm_reach, ma_box_drift, ma_cutout, ma_extra_limbs, ma_frame_ok
from marearts_spritestride.comfy import ARMS_HANG, ARMS_NEAR_BACK, ARMS_NEAR_FORWARD, ma_arm_words
from marearts_spritestride.gait import LANK, LELB, LSHO, LWRI, RANK, ma_frame_phases, ma_gait, ma_walk_points
from marearts_spritestride.mannequin import NEAR, ma_box_rect, ma_depth, ma_legs_mask, ma_skeleton


def test_near_leg_is_brighter_than_far():
    pts = ma_walk_points(0.0)
    d = np.asarray(ma_depth(pts).convert("L"))
    near = d[int(pts[LANK][1]), int(pts[LANK][0])]
    far = d[int(pts[RANK][1]), int(pts[RANK][0])]
    assert near > far + 60
    assert near > NEAR - 40


def test_carry_box_is_in_the_mannequin():
    pts = ma_walk_points(0.1, carry=True)
    d = np.asarray(ma_depth(pts, box=True).convert("L"))
    x0, y0, x1, y1 = (int(v) for v in ma_box_rect(pts))
    assert np.median(d[y0 + 5:y1 - 5, x0 + 5:x1 - 5]) >= 190


def test_skeleton_draws():
    im = ma_skeleton(ma_walk_points(0.3))
    assert np.asarray(im).sum() > 0


def test_limbs_clean_and_extra():
    pts = ma_walk_points(0.2)
    dep = ma_depth(pts)
    person = ma_legs_mask(dep)
    outside, legs = ma_extra_limbs(person, dep)
    assert outside < 1 and legs <= 2
    third = person.copy()
    ys = np.nonzero(person.any(1))[0]
    y = int(ys.min() + 0.5 * (ys.max() - ys.min()))
    third[y:y + 250, 20:90] = True                          # an arm-sized piece where none can be, joined by a bar
    third[y + 50:y + 54, 90:int(np.nonzero(person[y + 52])[0].min()) + 1] = True
    assert ma_extra_limbs(third, dep)[0] > 6


def test_arm_words_follow_the_swing():
    for name in ("lively", "older"):
        g = ma_gait(name)
        pts = [ma_walk_points(t, g) for t in ma_frame_phases(8, g)]
        widest = max(abs(p[LELB][0] - p[LSHO][0]) for p in pts)
        words = [ma_arm_words(p, widest) for p in pts]
        assert words[2] == words[6] == ARMS_HANG                 # the legs pass: the arms hang
        assert words[0] == words[7] == ARMS_NEAR_BACK            # near leg ahead, near arm back
        assert words[4] == ARMS_NEAR_FORWARD


def test_arm_reach_catches_a_hand_swung_forward():
    pts = ma_walk_points(0.21)
    dep = ma_depth(pts)
    person = ma_legs_mask(dep)
    assert ma_arm_reach(person, dep, pts) < 1
    swung = person.copy()
    y, x = int(pts[LWRI][1]), int(np.nonzero(person[int(pts[LWRI][1])])[0].max())
    swung[y - 160:y, x - 5:x + 90] = True                    # a forearm and hand out in front of the body
    assert ma_arm_reach(swung, dep, pts) > REACH_MAX
    ok, said = ma_frame_ok(np.zeros((*swung.shape, 3), np.uint8), swung, dep, pts, "walk")
    assert not ok and "arms" in said


def test_box_drift_and_frame_ok():
    pts = ma_walk_points(0.1, carry=True)
    dep = ma_depth(pts, box=True)
    rgb = np.full((*np.asarray(dep).shape[:2], 3), 255, np.uint8)
    x0, y0, x1, y1 = (int(v) for v in ma_box_rect(pts))
    rgb[y0:y1, x0:x1] = (196, 150, 90)                     # cardboard, seen square
    mask = ma_legs_mask(dep)
    top, end = ma_box_drift(rgb, mask, pts)
    assert abs(top) <= 2 and end < 0.05
    ok, said = ma_frame_ok(rgb, mask, dep, pts, "carry")
    assert ok, said


def test_cutout_unmixes_white_and_keeps_eye():
    frame = Image.new("RGB", (100, 200), "white")
    px = frame.load()
    for y in range(20, 190):
        for x in range(30, 70):
            px[x, y] = (230, 180, 150)                       # skin-ish figure
    mask = Image.new("L", (100, 200), 0)
    mp = mask.load()
    for y in range(20, 190):
        for x in range(30, 70):
            mp[x, y] = 255
    for y in range(35, 40):
        for x in range(50, 55):
            mp[x, y] = 0                                     # the eye white read as paper
    rgba = ma_cutout(frame, mask)
    assert rgba[37, 52, 3] == 255
    assert rgba[5, 5, 3] == 0


def test_far_leg_shaded_near_leg_and_body_kept():
    from marearts_spritestride.gait import NECK_I
    from marearts_spritestride.lock import ma_far_shade
    pts = ma_walk_points(0.0)
    dep = ma_depth(pts)
    frame = np.zeros(np.asarray(dep).shape[:2] + (4,), np.uint8)
    frame[ma_legs_mask(dep)] = (200, 200, 200, 255)              # a grey figure standing on its mannequin
    out = ma_far_shade(frame, dep, 0.5)
    at = lambda p, dy=0: out[int(p[1]) + dy, int(p[0]), 0]
    assert at(pts[RANK], -20) < 120                              # far shin dark
    assert at(pts[LANK], -20) == 200                             # near shin as drawn
    assert at(pts[NECK_I], 40) == 200                            # body as drawn
    assert ma_far_shade(frame, dep, 1.0)[..., :3].sum() == frame[..., :3].sum()


def test_judge_asks_where_the_hands_must_be():
    from marearts_spritestride.judge import ASK_HAND_HERE, MaJudge, ma_hand_box
    pts = ma_walk_points(0.0)
    box = ma_hand_box(pts, LWRI)
    x0, y0, x1, y1 = box
    assert x0 < pts[LWRI][0] < x1 and y0 < pts[LWRI][1] < y1

    class Judge(MaJudge):
        def __init__(self, here):
            self.here, self.asked = here, []

        def ask(self, im, text):
            self.asked.append(text)
            return self.here if text == ASK_HAND_HERE else "hand behind the body\nhand in front of the coat\n2"
    im = Image.new("RGB", (832, 1216), "white")
    ok, said = Judge("Yes.").arms_ok(im, "walk", {"near": box})
    assert ok and "near hand there" in said
    j = Judge("No.")
    ok, said = j.arms_ok(im, "walk", {"near": box, "far": box})
    assert not ok and "near hand missing" in said and len(j.asked) == 2      # stops at the first missing hand
    assert Judge("No.").arms_ok(im, "walk")[0]                                # no boxes: the count alone

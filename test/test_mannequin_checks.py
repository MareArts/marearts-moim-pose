import numpy as np
from PIL import Image

from marearts_moim_pose.checks import REACH_MAX, ma_arm_reach, ma_box_drift, ma_cutout, ma_extra_limbs, ma_frame_ok
from marearts_moim_pose.comfy import ARMS_HANG, ARMS_NEAR_BACK, ARMS_NEAR_FORWARD, ma_arm_words
from marearts_moim_pose.gait import LANK, LELB, LSHO, LWRI, RANK, ma_frame_phases, ma_gait, ma_walk_points
from marearts_moim_pose.mannequin import NEAR, ma_box_rect, ma_depth, ma_legs_mask, ma_skeleton


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
    from marearts_moim_pose.gait import NECK_I
    from marearts_moim_pose.lock import ma_far_shade
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
    from marearts_moim_pose.judge import ASK_CARRY, ASK_HAND_HERE, ASK_HANDS, MaJudge, ma_hand_box
    pts = ma_walk_points(0.0)
    box = ma_hand_box(pts, LWRI)
    x0, y0, x1, y1 = box
    assert x0 < pts[LWRI][0] < x1 and y0 < pts[LWRI][1] < y1

    class Judge(MaJudge):
        def __init__(self, here):
            self.here, self.asked = here, []

        def ask(self, im, text):
            self.asked.append(text)
            return self.here if text == ASK_HAND_HERE else '[{"bbox_2d": [1, 1, 9, 9]}, {"bbox_2d": [50, 50, 60, 60]}]'
    im = Image.new("RGB", (832, 1216), "white")
    ok, said = Judge("Yes.").arms_ok(im, "walk", {"near": box})
    assert ok and "near hand there" in said
    j = Judge("No.")
    ok, said = j.arms_ok(im, "walk", {"near": box, "far": box})
    assert not ok and "near hand missing" in said and len(j.asked) == 2      # stops at the first missing hand
    assert Judge("No.").arms_ok(im, "walk")[0]                                # no boxes: the count alone

    class Carry(MaJudge):
        def __init__(self, hanging, boxes):
            self.hanging, self.boxes, self.asked = hanging, boxes, []

        def ask(self, im, text):
            self.asked.append(text)
            if text == ASK_CARRY:
                return self.hanging
            return str([{"bbox_2d": [10 * k, 0, 10 * k + 9, 9]} for k in range(self.boxes)])
    assert Carry("No.", 2).arms_ok(im, "carry")[0]
    j = Carry("No.", 3)                                                       # a third hand somewhere else
    ok, said = j.arms_ok(im, "carry")
    assert not ok and "hands 3" in said and j.asked == [ASK_CARRY, ASK_HANDS]
    j = Carry("Yes.", 2)                                                      # one hangs by the hip: no more asked
    assert not j.arms_ok(im, "carry")[0] and j.asked == [ASK_CARRY]


def test_judge_counts_shoes():
    from marearts_moim_pose.judge import ASK_FEET, MaJudge

    class Judge(MaJudge):
        def __init__(self, n):
            self.n, self.boxes = n, []

        def ask(self, im, text):
            assert text == ASK_FEET
            self.boxes.append(im.size)
            return "red sneaker\n" * self.n + str(self.n)
    im = Image.new("RGB", (832, 1216), "white")
    assert Judge(2).feet_ok(im)[0] and not Judge(3).feet_ok(im)[0]
    j = Judge(2)
    j.feet_ok(im)
    assert j.boxes == [(832, 1216 - int(1216 * 0.62))]                    # the legs and feet only


def test_clothes_change_between_frames():
    from marearts_moim_pose.checks import CHANGE_MAX, ma_clothes_change
    pts = [ma_walk_points(t) for t in ma_frame_phases(4, ma_gait())]
    same = [np.asarray(ma_depth(pts[0]))] * len(pts)            # the same drawing in every frame
    assert max(ma_clothes_change(same, pts)) < CHANGE_MAX
    jumped = [f.copy() for f in same]
    jumped[2][...] = 255                                        # frame 2 drawn in other clothes: white
    c = ma_clothes_change(jumped, pts)
    assert c[1] > CHANGE_MAX and c[2] > CHANGE_MAX              # the pairs into and out of frame 2
    assert c[0] < CHANGE_MAX


def test_head_likeness_fails_another_person():
    from marearts_moim_pose.judge import HEAD_MIN, MaJudge, ma_head
    im = Image.new("RGB", (832, 1216), "white")
    assert ma_head(im).size == (int(832 * 0.85) - int(832 * 0.25), int(1216 * 0.30) - int(1216 * 0.03))   # blank
    from PIL import ImageDraw
    fig = im.copy()
    ImageDraw.Draw(fig).ellipse([380, 100, 460, 180], fill="black")            # a head
    ImageDraw.Draw(fig).rectangle([360, 180, 480, 900], fill="grey")           # a body
    ImageDraw.Draw(fig).rectangle([480, 400, 600, 500], fill="brown")          # a box, below the head
    x0, y0, x1, y1 = ma_head(fig).getbbox() and (0, 0, 0, 0)
    head = ma_head(fig)
    assert head.size[0] == head.size[1] and head.size[0] < 200                  # the head only, not the box
    assert MaJudge().head_likeness(im, im) is None                 # a judge without it says nothing
    assert 0.8 < HEAD_MIN < 0.95


def test_mannequin_head_has_a_nose_facing_right():
    from marearts_moim_pose.gait import FIG, NOSE
    pts = ma_walk_points(0.0)
    d = np.asarray(ma_depth(pts).convert("L")).astype(int)
    hx, hy = pts[NOSE][0] - 0.05 * FIG, pts[NOSE][1] - 0.01 * FIG
    fx, cy = int(hx + 0.065 * FIG), int(hy + 0.07 * FIG - 0.45 * 1.115 * 0.19 * FIG)
    assert d[cy, fx + 4] > 40                                      # the nose stands out in front of the head
    assert d[cy, fx + 4] > d[cy - 40, fx + 4] + 30                  # and only there


def test_mannequin_seat_covers_the_top_of_the_near_thigh():
    from marearts_moim_pose.gait import FIG, LHIP
    from marearts_moim_pose.mannequin import BODY, NEAR
    pts = ma_walk_points(0.0)
    x, y = int(pts[LHIP][0]), int(pts[LHIP][1] + 0.03 * FIG)            # just under the hip, on the near thigh
    seat = np.asarray(ma_depth(pts).convert("L")).astype(int)[y, x]
    thigh = np.asarray(ma_depth(pts, seat=0).convert("L")).astype(int)[y, x]
    assert abs(seat - BODY) < 10 and abs(thigh - NEAR) < 10              # the jacket's place, not the trousers'


def test_far_leg_shaded_evenly_to_the_hem_and_nothing_else():
    from marearts_moim_pose.gait import FIG, LHIP, LSHO, RHIP, RKNE
    from marearts_moim_pose.lock import ma_leg_tones
    pts = ma_walk_points(0.0)
    d = ma_depth(pts)
    a = np.asarray(d.convert("L")) > 20
    f = np.zeros(a.shape + (4,), np.uint8)
    f[a] = (235, 195, 175, 255)                                   # bare skin everywhere
    hem = int(pts[LHIP][1] + 0.06 * FIG)
    f[:hem][a[:hem]] = (60, 80, 150, 255)                          # shorts and top down to the hem
    toned = ma_leg_tones([f], [d], 0.55, [pts])[0]
    lum = lambda im, x, y: float(im[y - 3:y + 3, x - 3:x + 3, :3].mean())
    rh, rk = pts[RHIP], pts[RKNE]
    y = int(hem + 0.03 * FIG)                                       # just under the hem, the back edge: the far
    upper = (int(np.nonzero(a[y])[0].min()) + 12, y)               # leg is the one behind at this step
    shin = (int(rk[0] + 0.5 * (pts[RKNE + 1][0] - rk[0])), int(rk[1] + 0.5 * (pts[RKNE + 1][1] - rk[1])))
    assert lum(toned, *upper) < 0.8 * lum(f, *upper)               # shaded up to the hem, not only below the knee
    assert abs(lum(toned, *upper) - lum(toned, *shin)) < 0.15 * lum(f, *shin)
    x, y = int(pts[LHIP][0]), int(pts[LHIP][1])                     # the shorts at the hip: as drawn
    assert lum(toned, x, y) == lum(f, x, y)
    sx, sy = (int(v) for v in pts[LSHO + 1])                        # the near elbow (skin colour too): as drawn
    assert lum(toned, sx, sy) == lum(f, sx, sy)


def test_far_leg_shaded_in_its_own_colour_and_white_stays_white():
    from marearts_moim_pose.gait import FIG, RKNE
    from marearts_moim_pose.lock import ma_leg_tones
    pts = ma_walk_points(0.0)
    d = ma_depth(pts)
    a = np.asarray(d.convert("L")) > 20
    f = np.zeros(a.shape + (4,), np.uint8)
    f[a] = (245, 222, 210, 255)                                   # pale skin
    floor = np.nonzero(a.any(1))[0].max()
    f[floor - int(0.04 * FIG):][a[floor - int(0.04 * FIG):]] = (250, 250, 250, 255)      # white shoes
    toned = ma_leg_tones([f], [d], 0.9, [pts])[0]
    rk, ra = pts[RKNE], pts[RKNE + 1]
    x, y = int((rk[0] + ra[0]) / 2), int((rk[1] + ra[1]) / 2)    # the far shin
    was, now = f[y, x, :3].astype(float), toned[y, x, :3].astype(float)
    sat = lambda c: (c.max() - c.min()) / c.max()
    assert now.mean() < was.mean() - 5                            # shaded
    assert sat(now) > sat(was)                                    # deeper skin, not grey (scaled, 0.75 read grey)
    shoes = toned[..., :3][f[..., 0] == 250].astype(float)
    assert len(shoes) and shoes.min() > 235                       # a white shoe stays white, the far one too


def test_skirt_read_from_the_words_of_that_garment():
    from marearts_moim_pose.mannequin import ma_skirt_of
    assert ma_skirt_of("a white pencil skirt to the knee, red pumps") == ma_skirt_of("a navy pleated skirt") == 1.05
    assert ma_skirt_of("khaki trousers, brown boots") == 0 == ma_skirt_of("denim shorts")
    assert ma_skirt_of("a long maxi dress") > 1.5 and ma_skirt_of("a short black pencil skirt") < 1
    assert ma_skirt_of("a rust orange midi skirt, tan ankle boots") == 1.4         # the boots are not the skirt
    assert ma_skirt_of("a white dress shirt, grey trousers, black dress shoes") == 0


def test_skirt_measured_on_the_picture():
    from marearts_moim_pose.gait import HIP, THIGH
    from marearts_moim_pose.judge import MaJudge
    from marearts_moim_pose.mannequin import ma_skirt_at
    assert abs(ma_skirt_at(1 - HIP + THIGH) - 1) < 1e-6 and ma_skirt_at(0.67) < 0.9 and ma_skirt_at(0.92) > 1.7

    class Judge(MaJudge):
        def ask(self, im, text):
            assert "skirt" in text
            return '[{"bbox_2d": [100, 400, 900, 750], "label": "skirt"}]'
    im = Image.new("RGB", (300, 700), "white")
    im.paste(Image.new("RGB", (100, 600), (40, 50, 110)), (100, 50))     # a figure from row 50 to 649
    hem = Judge().skirt_hem(im)
    fig_h = 600 + 20                                                     # the crop: 10 px round the figure
    assert abs(hem - (0.75 * fig_h - 10) / 599) < 0.01


def test_mannequin_skirt_covers_both_thighs_to_the_hem():
    from marearts_moim_pose.gait import FIG, LHIP, LKNE, RHIP
    from marearts_moim_pose.mannequin import SKIRT, ma_skirt_hem
    pts = ma_walk_points(0.0)                                     # legs apart
    d = np.asarray(ma_depth(pts, skirt=1.05).convert("L")).astype(int)
    bare = np.asarray(ma_depth(pts).convert("L")).astype(int)
    f = 0.85                                                      # low on the thigh: one piece of cloth, front leg
    y = int(pts[LHIP][1] + f * (pts[LKNE][1] - pts[LHIP][1]))      # to back
    xs = sorted(int(pts[h][0] + f * (pts[h + 1][0] - pts[h][0])) for h in (LHIP, RHIP))
    assert (abs(d[y, xs[0]:xs[1] + 1] - SKIRT) < 12).all()
    assert (bare[y, xs[0]:xs[1]] < 30).sum() > 20                 # without it the legs split there
    hem = ma_skirt_hem(pts, 1.05)[2]
    below = int(hem + 0.04 * FIG)
    assert (d[below] < 30).any() and abs(d[below] - bare[below]).max() < 5        # under the hem: the legs alone


def test_leg_tones_leave_the_skirt_as_drawn():
    from marearts_moim_pose.gait import FIG, LHIP, LKNE
    from marearts_moim_pose.lock import ma_leg_tones
    from marearts_moim_pose.mannequin import ma_skirt_hem
    pts = ma_walk_points(0.0)
    d = ma_depth(pts, skirt=1.05)
    a = np.asarray(d.convert("L")) > 20
    hem = int(ma_skirt_hem(pts, 1.05)[2])
    f = np.zeros(a.shape + (4,), np.uint8)
    f[a] = (235, 195, 175, 255)
    f[:hem][a[:hem]] = (40, 50, 110, 255)                           # a navy skirt to the hem
    toned = ma_leg_tones([f], [ma_depth(pts)], 0.9, [pts], [hem])[0]
    assert (toned[:hem] == f[:hem]).all()                          # nothing shaded or lit through the skirt


def test_skirt_flare_measured_on_the_picture():
    from marearts_moim_pose.gait import FIG, HIP, LHIP, LKNE
    from marearts_moim_pose.mannequin import SKIRT, ma_skirt_flare
    h, hem = 1000, 0.72

    def standing(width):                         # a figure 1000 px tall, its skirt `width` of the height wide
        m = np.zeros((1100, 600), bool)
        m[50:1050, 250:350] = True
        m[50 + int(h * (1 - HIP)):50 + int(h * hem), 300 - int(width * h / 2):300 + int(width * h / 2)] = True
        return m
    pencil, aline = ma_skirt_flare(standing(0.11), hem), ma_skirt_flare(standing(0.18), hem)
    assert pencil < 0.06 and aline > 0.09
    pts = ma_walk_points(0.0)
    y = int(pts[LHIP][1] + 0.9 * (pts[LKNE][1] - pts[LHIP][1]))
    wide = [int((np.abs(np.asarray(ma_depth(pts, skirt=1.0, flare=f).convert("L")).astype(int)[y] - SKIRT) < 12).sum())
            for f in (pencil, aline)]
    assert wide[1] > wide[0] + 0.05 * FIG                  # an A-line skirt stands wider than a pencil one

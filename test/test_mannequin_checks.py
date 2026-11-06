import numpy as np
from PIL import Image

from marearts_moim_pose.checks import (REACH_MAX, ma_arm_reach, ma_box_drift, ma_cutout, ma_extra_limbs, ma_frame_ok,
                                       ma_stray_arm)
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
    assert abs(pencil - 0.11) < 0.01 and abs(aline - 0.18) < 0.01
    pts = ma_walk_points(0.25)                            # the legs together: the skirt's own width shows
    y = int(pts[LHIP][1] + 0.9 * (pts[LKNE][1] - pts[LHIP][1]))
    wide = [int((np.abs(np.asarray(ma_depth(pts, skirt=1.0, flare=f).convert("L")).astype(int)[y] - SKIRT) < 12).sum())
            for f in (pencil, aline)]
    assert wide[1] > wide[0] + 0.05 * FIG                  # an A-line skirt stands wider than a pencil one


def test_mannequin_fitted_to_the_picture():
    from marearts_moim_pose.gait import FIG, HIP, LHIP, NECK_I
    from marearts_moim_pose.mannequin import BODY, ma_fit, ma_torso_outline
    m = np.zeros((1100, 600), bool)                       # a slim figure 1000 px tall facing right
    m[50:250, 230:390] = True                              # head with hair: 0.16 wide, the neck at 0.2
    m[250:270, 290:330] = True                             # the neck
    m[270:1050, 250:370] = True                            # body 0.12 thick
    m[300:480, 100:250] = True                             # long hair down the back, to 0.43
    m[310:370, 370:420] = True                             # a chest standing out 0.05 at the front
    f = ma_fit(m)
    assert abs(f["head"] - 0.16) < 0.01 and abs(f["body"] - 0.12) < 0.01 and abs(f["bust"] - 0.11) < 0.015
    rows = f["torso"]["rows"]                                          # x from the face's front (389), of 1000
    assert min(r[1] for r in rows) >= (310 - 0.78 * 120 - 389) / 1000 - 1e-3   # the hair behind is not body
    assert max(r[0] for r in rows) - min(r[0] for r in rows[6:]) > 0.04        # the chest is
    flat = m.copy()
    flat[310:370, 370:420] = False
    assert ma_fit(flat)["bust"] == 0                       # nothing stands out: no chest drawn
    assert ma_fit(np.zeros((50, 50), bool))["body"] == 0.16
    pony = m.copy()
    pony[60:120, 60:230] = True                            # a ponytail behind the head
    assert ma_fit(pony)["head"] < 0.17                     # the head stays the size of the head
    pts = ma_walk_points(0.25)
    y = int((pts[NECK_I][1] + pts[LHIP][1]) / 2)
    wide = [int((np.asarray(ma_depth(pts, body=b).convert("L"))[y] > 100).sum()) for b in (0.12, 0.16)]
    assert abs(wide[1] - wide[0] - 0.04 * FIG) < 12        # the tube drawn as thick as measured
    d = np.asarray(ma_depth(pts, torso=f["torso"]).convert("L")).astype(int)
    out = ma_torso_outline(pts, f["torso"])
    chest_y = int(pts[NECK_I][1] + (pts[LHIP][1] - pts[NECK_I][1]) * 0.35)
    xs = np.nonzero(np.abs(d[chest_y] - BODY) < 15)[0]
    assert xs.max() - pts[NECK_I][0] > 0.08 * FIG          # the chest stands out in the mannequin too
    assert max(p[1] for p in out) > pts[LHIP][1]           # and the seat goes on past the hips


def test_heels_in_the_mannequin_and_from_the_judge():
    from marearts_moim_pose.gait import LANK
    from marearts_moim_pose.judge import MaJudge
    pts = ma_walk_points(0.0)
    flat = np.asarray(ma_depth(pts).convert("L")) > 30
    heel = np.asarray(ma_depth(pts, heels=True).convert("L")) > 30
    ax, ay = (int(v) for v in pts[LANK])
    below = slice(ay + 25, ay + 40)
    arch = slice(ax + 8, ax + 30)                            # under the foot, between the heel and the ball
    assert flat[below, arch].mean() > 0.5 and heel[below, arch].mean() < flat[below, arch].mean() - 0.2

    class Judge(MaJudge):
        def ask(self, im, text):
            return "Heels"
    im = Image.new("RGB", (200, 600), "white")
    im.paste(Image.new("RGB", (60, 500), (200, 30, 30)), (70, 50))
    assert Judge().heels(im) is True


def test_skirt_hem_measured_on_the_pixels():
    from marearts_moim_pose.mannequin import ma_skirt_hem_of
    rgb = np.full((1100, 600, 3), 255, np.uint8)
    rgb[50:1050, 260:340] = (240, 200, 180)                 # a figure 1000 px tall, bare legs
    rgb[500:600, 285:300] = (240, 200, 180)
    rgb[480:700, 230:370] = (40, 40, 40)                    # a black skirt to 0.65
    rgb[520:560, 330:345] = (240, 200, 180)                 # a hand over it
    rgb[700:1050, 270:330] = (40, 40, 40)                   # black tights below, narrower than the skirt
    person = (rgb < 235).any(-1)
    assert abs(ma_skirt_hem_of(person, rgb) - 0.65) < 0.01


def test_skirt_hem_as_wide_as_measured_unless_the_stride_needs_more():
    from marearts_moim_pose.gait import FIG
    from marearts_moim_pose.mannequin import ma_skirt_hem
    apart, together = ma_walk_points(0.0), ma_walk_points(0.25)
    b, f, _ = ma_skirt_hem(together, 1.0, 0.18)
    assert abs((f - b) - 0.18 * FIG) < 1                  # legs together: the hem as wide as measured
    b, f, _ = ma_skirt_hem(apart, 1.0, 0.18)
    b2, f2, _ = ma_skirt_hem(apart, 1.0, 0.36)
    assert f - b < f2 - b2 and f - b >= 0.18 * FIG - 1     # a stride never narrows it, a wider skirt stays wider


def test_skirt_follows_the_hips_of_the_body():
    from marearts_moim_pose.gait import FIG, LHIP
    from marearts_moim_pose.mannequin import ma_skirt_outline
    pts = ma_walk_points(0.25)
    rows = [[0.05, 0.05]] * 6 + [[0.05, 0.05], [0.06, 0.08], [0.06, 0.09], [0.06, 0.1], [0.06, 0.1], [0.05, 0.09]]
    torso = {"neck": 0.18, "rows": rows}                 # a slim waist, the seat standing out behind at the hips
    plain = ma_skirt_outline(pts, 1.0, 0.12)
    curved = ma_skirt_outline(pts, 1.0, 0.12, torso)
    hem_y = plain[2][1]
    seat = min(x for x, y in curved if y < hem_y - 1)
    assert seat < plain[0][0] - 0.01 * FIG                 # the seat's curve in the skirt, past the plain wedge
    assert len(curved) > len(plain) and pts[LHIP][1] > 0


def test_arms_hang_from_the_middle_of_the_body():
    from marearts_moim_pose.gait import LSHO, RSHO
    from marearts_moim_pose.mannequin import ma_arms_on_body, ma_torso_sides
    pts = ma_walk_points(0.0)
    rows = [[-0.02, -0.14]] * 12                            # a body standing well forward of the skeleton
    torso = {"neck": 0.18, "rows": rows, "from": "face"}
    moved = ma_arms_on_body(pts, torso)
    front, back = ma_torso_sides(pts, torso)
    mid = (front[1][0] + back[1][0]) / 2
    assert abs(moved[LSHO][0] - mid) < 25 and abs(moved[RSHO][0] - mid) < 25
    assert moved[LSHO + 2][0] - pts[LSHO + 2][0] == moved[LSHO][0] - pts[LSHO][0]     # the whole arm moves
    assert ma_arms_on_body(pts, None) is pts


def test_colours_brought_to_the_picture_and_the_box_kept():
    from marearts_moim_pose.lock import ma_colour_match
    ref = np.tile(np.array([[235, 180, 20]], np.uint8), (2000, 1))         # the picture: a deep yellow
    f = np.zeros((100, 100, 4), np.uint8)
    f[20:80, 20:80] = (240, 220, 140, 255)                                  # drawn pale
    f[20:40, 20:40] = (150, 110, 70, 255)                                   # a box
    box = np.zeros((100, 100), bool)
    box[20:40, 20:40] = True
    out = ma_colour_match([f], ref, 0.5, [box])[0]
    sat = lambda c: (int(c.max()) - int(c.min())) / max(1, int(c.max()))
    assert sat(out[60, 60, :3]) > sat(f[60, 60, :3]) + 0.1                  # deeper, toward the picture
    assert (out[25, 25] == f[25, 25]).all() and (out[5, 5] == f[5, 5]).all()      # the box and the background kept


def test_near_leg_lit_without_blowing_out():
    from marearts_moim_pose.lock import ma_leg_tones
    pts = ma_walk_points(0.0)
    d = ma_depth(pts)
    a = np.asarray(d.convert("L")) > 20
    f = np.zeros(a.shape + (4,), np.uint8)
    f[a] = (200, 160, 140, 255)                                             # both legs, the near one drawn dark
    dd = np.asarray(d.convert("L"))
    f[(dd > 130) & a] = (150, 120, 105, 255)
    f[(dd > 130) & a & (np.arange(a.shape[0])[:, None] % 40 < 6)] = (245, 225, 215, 255)   # pale patches on it
    toned = ma_leg_tones([f], [d], 0.9, [pts])[0]
    patch = (dd > 130) & a & (np.arange(a.shape[0])[:, None] % 40 < 6)
    assert toned[patch][:, :3].min() < 250                                  # lit, not white


def test_stray_arm_only_where_the_far_arm_is_behind_the_body():
    for t, hidden in ((0.25, True), (0.0, False)):          # legs passing: far hand behind the body; swung: it shows
        pts = ma_walk_points(t)
        dep = ma_depth(pts)
        person = ma_legs_mask(dep)
        assert ma_stray_arm(person, dep, pts) is None
        swung = person.copy()
        y = int(pts[LSHO][1] + 0.25 * (pts[LWRI][1] - pts[LSHO][1]) + 120)
        x = int(np.nonzero(person[y])[0].max())
        swung[y - 100:y + 20, x + 4:x + 90] = True           # a forearm and hand out in front of the body
        box = ma_stray_arm(swung, dep, pts)
        if hidden:
            assert box is not None and box[0] <= x + 30 and box[2] >= x + 85
            ok, said = ma_frame_ok(np.zeros((*swung.shape, 3), np.uint8), swung, dep, pts, "walk")
            assert not ok and "stray arm" in said
        else:
            assert box is None


def test_lock_keeps_the_head_size_when_hair_hides_the_neck():
    from scipy import ndimage
    from marearts_moim_pose.lock import ma_head_sizes, ma_lock
    rng = np.random.default_rng(3)
    face = ndimage.gaussian_filter(rng.random((120, 120)), 3)
    face = ((face - face.min()) / np.ptp(face) * 200).astype(np.uint8)

    def frame(r, hair=False):
        f = np.zeros((600, 400, 4), np.uint8)
        f[..., :3] = 255
        yy, xx = np.mgrid[0:600, 0:400]
        disc = (yy - (40 + r)) ** 2 + (xx - 200) ** 2 < r * r
        tex = np.asarray(Image.fromarray(face).resize((2 * r, 2 * r)))
        sub = np.zeros((600, 400), np.uint8)
        sub[40:40 + 2 * r, 200 - r:200 + r] = tex
        f[disc, :3] = sub[disc, None]
        f[disc, 3] = 255
        f[40 + 2 * r:560, 150:250] = (90, 90, 160, 255)    # the body from under the chin
        f[40 + 2 * r - 5:560, 180:250, 3] = 255            # the neck
        if hair:                                            # long hair falling in front of the neck and chest
            f[40 + r:40 + 2 * r + 60, 240:262] = (230, 220, 180, 255)
        return f
    frames = [frame(50), frame(55), frame(50, hair=True)]
    from marearts_moim_pose.checks import ma_person
    people = [ma_person(f[..., 3] > 128) for f in frames]
    sizes = ma_head_sizes(frames, people, 0, 100)
    assert sizes[0] == 1.0 and abs(sizes[1] - 1.1) <= 0.02 and abs(sizes[2] - 1.0) <= 0.02
    out = ma_lock(frames, [0.0, 0.0, 0.0], [None] * 3, [None] * 3, align_legs=False)

    def head_rows(f):                                       # the face down its middle column, to the body's blue
        c = f[:, 200].astype(int)
        face = (c[:, 3] > 128) & (c[:, 2] - c[:, 0] < 40)
        return int(face[:np.nonzero((c[:, 3] > 128) & (c[:, 2] - c[:, 0] >= 40))[0].min()].sum())
    rows = [head_rows(f) for f in out]
    assert abs(rows[1] - rows[0]) <= 3 and abs(rows[2] - rows[0]) <= 3, rows


def test_stray_arm_wiped_and_only_a_band_round_it_drawn_again():
    from marearts_moim_pose.pipeline import ma_stray_wipe
    pts = ma_walk_points(0.25)
    dep = ma_depth(pts)
    person = ma_legs_mask(dep)
    y = int(pts[LSHO][1] + 0.25 * (pts[LWRI][1] - pts[LSHO][1]) + 120)
    x = int(np.nonzero(person[y])[0].max())
    person[y - 100:y + 20, x + 4:x + 90] = True               # the far forearm and hand out in front
    rgb = np.full((*person.shape, 3), 255, np.uint8)
    rgb[person] = (200, 30, 30)
    wiped, band = ma_stray_wipe(Image.fromarray(rgb), person, dep, ma_stray_arm(person, dep, pts))
    w, b = np.asarray(wiped), np.asarray(band.convert("L")) > 128
    assert (w[y - 90:y + 10, x + 20:x + 85] == 255).all()        # the hand is gone
    assert (w[y - 50, x - 30] == (200, 30, 30)).all()            # the body stays as drawn
    edge = int(np.nonzero((w[y - 50] != 255).any(-1))[0].max())  # where the body is cut
    assert b[y - 50, x + 50] and b[y - 50, edge - 3:edge + 1].all()   # the band takes in the wiped part and the cut
    assert not b[y - 50, x - 60] and b.mean() < 0.02             # and nothing much else

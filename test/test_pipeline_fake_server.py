"""The whole draw -> check -> lock -> sheet path against a fake ComfyUI server (no GPU, no models).

The fake server "draws" each frame as the uploaded depth mannequin painted on white, near limbs lighter as a model
draws them, scaled a little differently per frame, and "cuts it out" by thresholding. The lock must bring the
heights back together. Frames listed in `same_foot` (frame, seed) are painted with the far leg lit instead."""
import io
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from marearts_moim_pose.comfy import EXTRA_FIX, HAND_FIX, HEAD_FIX, MaComfy, ma_ui_workflow
from marearts_moim_pose.judge import ma_frame_box, ma_hand_boxes, ma_verdict
from marearts_moim_pose.pipeline import MaRequest, ma_draw_cycle


class Fake(BaseHTTPRequestHandler):
    uploads: dict = {}
    outputs: dict = {}
    count = [0]
    same_foot: set = set()
    calls: list = []

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, kind="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        data = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/free":
            self.calls.append("free")
            return self._send(b"{}")
        if self.path == "/upload/image":
            name = re.search(rb'filename="([^"]+)"', data).group(1).decode()
            png = data[data.index(b"\x89PNG"):data.rindex(b"\r\n--")]
            self.uploads[name] = Image.open(io.BytesIO(png)).convert("RGB")
            return self._send(json.dumps({"name": name}).encode())
        g = json.loads(data)["prompt"]
        self.count[0] += 1
        pid = f"p{self.count[0]}"
        if g.get("20", {}).get("class_type") == "Wan22FunControlToVideo":   # a video: each mannequin painted
            self.calls.append("video")
            frames = []
            for k in range(g["20"]["inputs"]["length"]):
                d = np.asarray(self.uploads[g[f"d{k}"]["inputs"]["image"]].convert("L")).astype(np.float64)
                shade = (0.35 + 0.65 * d / 255)[..., None] * np.array([90, 120, 200])
                buf = io.BytesIO()
                Image.fromarray(np.where((d > 30)[..., None], shade, 255).astype(np.uint8)).save(buf, "PNG")
                frames.append(buf.getvalue())
            self.outputs[pid] = frames
            return self._send(json.dumps({"prompt_id": pid}).encode())
        if "10" not in g and "2" not in g:                # a new character from words: a tall figure on white
            out = Image.new("RGB", (832, 1216), "white")
            out.paste(Image.new("RGB", (220, 1000), (200, 90, 60)), (300, 100))
        elif "10" in g:                                   # a frame: the mannequin painted, scaled per frame
            self.calls.append("sharpen" if "40" in g else "frame")
            if "40" in g:                                 # what the frame is drawn from
                self.calls.append(("ref", g["11"]["inputs"]["image"]))
            if "45" in g:                                 # inside a mask: how much of the noise
                self.calls.append(("fix", next(v["inputs"]["denoise"] for v in g.values()
                                               if "denoise" in v.get("inputs", {}))))
            name = g["10"]["inputs"]["image"]
            d = np.asarray(self.uploads[name].convert("L")).astype(np.float64)
            frame = int(re.search(r"_depth_(\d+)", name).group(1))
            k = 1 + 0.01 * ((frame % 5) - 3)              # by frame, not by call; over 1.03 the head leaves the canvas
            if (frame, g["22"]["inputs"]["noise_seed"]) in self.same_foot:
                d = np.where(d > 30, 295 - d, d)          # far leg lit, near leg dark: drawn on the other foot
            shade = (0.35 + 0.65 * d / 255)[..., None] * np.array([90, 120, 200])
            im = Image.fromarray(np.where((d > 30)[..., None], shade, 255).astype(np.uint8))
            w, h = im.size
            big = im.resize((int(w * k), int(h * k)))
            can = Image.new("RGB", (w, h), "white")
            can.paste(big, ((w - big.width) // 2, h - big.height))
            out = can
        else:                                             # a cut-out mask
            a = np.asarray(self.uploads[g["2"]["inputs"]["image"]].convert("L"))
            out = Image.fromarray(((a < 250) * 255).astype(np.uint8))
        buf = io.BytesIO()
        out.save(buf, "PNG")
        self.outputs[pid] = buf.getvalue()
        return self._send(json.dumps({"prompt_id": pid}).encode())

    def do_GET(self):
        if self.path.startswith("/history/"):
            pid = self.path.split("/")[-1]
            n = len(self.outputs[pid]) if isinstance(self.outputs[pid], list) else 0
            names = [f"{pid}.{k}" for k in range(n)] if n else [pid]
            body = {pid: {"outputs": {"17": {"images": [{"filename": f, "subfolder": "", "type": "output"}
                                                        for f in names]}}}}
            return self._send(json.dumps(body).encode())
        if self.path.startswith("/view"):
            name = re.search(r"filename=([^&]+)", self.path).group(1)
            pid, _, k = name.partition(".")
            return self._send(self.outputs[pid][int(k)] if k else self.outputs[pid], "image/png")
        self.send_error(404)


def test_cycle_end_to_end(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ref = tmp_path / "person.png"
        Image.new("RGBA", (300, 700), (0, 0, 0, 0)).save(ref)
        # the fake draws the legs in the mannequin's light only: told apart for the same-foot check by a strong shade
        req = MaRequest(reference=ref, out=tmp_path / "out", name="test", height=200, far_shade=0.55)
        lines = []
        res = ma_draw_cycle(req, MaComfy(f"http://127.0.0.1:{srv.server_port}"), log=lines.append)
    finally:
        srv.shutdown()
    assert all(r["passed"] for r in res["report"]), lines
    assert len(res["report"]) == 12                         # the default cycle
    tops = []
    for i in range(12):
        al = np.asarray(Image.open(tmp_path / "out" / f"walk_{i}.png"))[..., 3] > 128
        ys = np.nonzero(al.any(1))[0]
        tops.append(ys.max() - ys.min())
    assert max(tops) - min(tops) <= 0.02 * max(tops)       # heights within the walk's own 1.2 % wave
    meta = json.loads(open(res["files"]["meta"]).read())
    assert meta["frames"] == 12 and meta["frame_height"] == 200 and meta["stride_px_per_frame"] > 0
    assert Image.open(res["files"]["loop"]).n_frames == 12
    assert Image.open(res["files"]["review"]).width == 960


def test_redraw_gets_new_seeds_and_keeps_report(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ref = tmp_path / "person.png"
        Image.new("RGB", (300, 700), "white").save(ref)
        req = MaRequest(reference=ref, frames=8, out=tmp_path / "out", name="test", height=120)
        comfy = MaComfy(f"http://127.0.0.1:{srv.server_port}")
        first = ma_draw_cycle(req, comfy, log=lambda s: None)
        again = ma_draw_cycle(req, comfy, only=[3], log=lambda s: None)
    finally:
        srv.shutdown()
    old = {r["frame"]: r for r in first["report"]}
    new = {r["frame"]: r for r in again["report"]}
    assert len(new) == 8 and new[0] == old[0]                       # the other frames' rows stay
    assert new[3]["seed"] not in old[3]["tried"] and old[3]["tried"][0] in new[3]["tried"]
    assert again["files"]["review"]


def test_partial_draw_skips_finish(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ref = tmp_path / "person.png"
        Image.new("RGB", (300, 700), "white").save(ref)
        lines = []
        res = ma_draw_cycle(MaRequest(reference=ref, frames=8, out=tmp_path / "o", name="t"),
                            MaComfy(f"http://127.0.0.1:{srv.server_port}"), only=[0, 4], log=lines.append)
    finally:
        srv.shutdown()
    assert res["files"] == {} and any("not drawn yet" in s for s in lines)


def test_ui_workflow_links():
    api = {"1": {"class_type": "A", "inputs": {"x": 3}}, "2": {"class_type": "B", "inputs": {"a": ["1", 0], "s": "hi"}}}
    info = {"A": {"input": {"required": {"x": ["INT", {"default": 1}]}}, "output": ["IMAGE"]},
            "B": {"input": {"required": {"a": ["IMAGE"], "s": ["STRING", {}]}}, "output": ["IMAGE"]}}
    ui = ma_ui_workflow(api, info)
    assert ui["links"] == [[1, 1, 0, 2, 0, "IMAGE"]]
    assert ui["nodes"][1]["widgets_values"] == ["hi"] and ui["nodes"][0]["outputs"][0]["links"] == [1]


def test_new_person(tmp_path):
    from marearts_moim_pose.comfy import ma_klein_text_graph
    from marearts_moim_pose.pipeline import ma_new_person
    g = ma_klein_text_graph("a man in a yellow raincoat")
    assert "yellow raincoat" in g["5"]["inputs"]["text"] and g["24"]["inputs"]["model"] == ["1", 0]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        kept = ma_new_person(MaComfy(f"http://127.0.0.1:{srv.server_port}"), "x", tmp_path, count=2, log=lambda s: None)
    finally:
        srv.shutdown()
    assert len(kept) == 2


def test_verdict():
    assert ma_verdict("2", "walk") and not ma_verdict("3", "walk") and not ma_verdict("", "walk")
    assert not ma_verdict("hand behind the body, hand 2 in front of the coat, hand in front\n3", "walk")
    assert ma_verdict("No.", "carry") and not ma_verdict("Yes", "carry")
    two = '[{"bbox_2d": [216, 672, 300, 788], "label": "hand"}, {"bbox_2d": [688, 600, 796, 750], "label": "hand"}]'
    three = two[:-1] + ', {"bbox_2d": [438, 795, 528, 878], "label": "hand"}]'
    assert ma_verdict(two, "walk") and not ma_verdict(three, "walk")    # Otto's frame 8 (11-04): a hand between
    again = two[:-1] + ', {"bbox_2d": [220, 676, 298, 790], "label": "hand"}]'
    assert ma_verdict(again, "walk") and len(ma_hand_boxes(again)) == 2  # the same hand boxed twice counts once


def _serve(same_foot):
    handler = type("SameFoot", (Fake,), {"uploads": {}, "outputs": {}, "count": [0], "same_foot": set(same_foot)})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_same_foot_frame_drawn_again(tmp_path):
    srv = _serve({(4, 7)})                                  # frame 4 at the first seed: on the same foot as frame 0
    try:
        ref = tmp_path / "person.png"
        Image.new("RGB", (300, 700), "white").save(ref)
        lines = []
        res = ma_draw_cycle(MaRequest(reference=ref, far_shade=0.55, frames=8, out=tmp_path / "out", name="t", height=120,
                                      alternate=True),
                            MaComfy(f"http://127.0.0.1:{srv.server_port}"), log=lines.append)
    finally:
        srv.shutdown()
    rows = {r["frame"]: r for r in res["report"]}
    assert rows[4]["tried"][0] == 7 and rows[4]["seed"] != 7 and len(rows[4]["tried"]) == 2, lines
    assert all(r["passed"] and not r["reason"] for r in rows.values()), lines
    assert rows[0]["lead"] > 10 and rows[4]["lead"] < -10
    assert any("other leg now" in s for s in lines)


def test_same_foot_every_seed_fails_in_red(tmp_path):
    from marearts_moim_pose.pipeline import ma_seeds
    srv = _serve({(4, s) for s in ma_seeds(7)})             # frame 4 never comes out right
    try:
        ref = tmp_path / "person.png"
        Image.new("RGB", (300, 700), "white").save(ref)
        lines = []
        res = ma_draw_cycle(MaRequest(reference=ref, frames=8, out=tmp_path / "out", name="t", height=120,
                                      alternate=True,
                                      far_shade=1.0),           # legs as drawn: the same foot shows
                            MaComfy(f"http://127.0.0.1:{srv.server_port}"), log=lines.append)
    finally:
        srv.shutdown()
    rows = {r["frame"]: r for r in res["report"]}
    assert not rows[4]["passed"] and rows[4]["reason"] == "same leg as #0" and rows[4]["seed"] == 7
    assert len(rows[4]["tried"]) == 4                       # the first seed, then 3 more for the other leg
    assert all(rows[i]["passed"] for i in rows if i != 4)
    saved = json.loads((tmp_path / "out" / "walk_report.json").read_text())
    assert saved[4]["reason"] == "same leg as #0" and not saved[4]["passed"]
    sheet = np.asarray(Image.open(res["files"]["review"]).convert("RGB"))
    assert (np.abs(sheet[54 + 350 + 46 + 2, 4:236] - [214, 40, 40]).sum(-1) < 10).mean() > 0.9   # frame 4 boxed red


def test_same_foot_fixed_by_leg_tones_without_redraw(tmp_path):
    srv = _serve({(4, 7)})                                  # frame 4 drawn with the near leg dark behind
    try:
        ref = tmp_path / "person.png"
        Image.new("RGB", (300, 700), "white").save(ref)
        lines = []
        res = ma_draw_cycle(MaRequest(reference=ref, far_shade=0.55, frames=8, out=tmp_path / "out", name="t", height=120),
                            MaComfy(f"http://127.0.0.1:{srv.server_port}"), log=lines.append)
    finally:
        srv.shutdown()
    rows = {r["frame"]: r for r in res["report"]}
    assert rows[4]["tried"] == [7], lines                   # not drawn again
    assert all(r["passed"] and not r["reason"] for r in rows.values()), lines
    assert rows[0]["lead"] > 10 and rows[4]["lead"] < -10   # the near leg lit in both: the steps read apart


def test_picture_facing_left_is_mirrored(tmp_path):
    from marearts_moim_pose.checks import FACING_SURE, ma_facing
    from marearts_moim_pose.pipeline import ma_face_right
    im = Image.new("RGB", (300, 700), "white")
    im.paste(Image.new("RGB", (60, 600), (40, 60, 120)), (120, 40))      # a standing figure
    im.paste(Image.new("RGB", (110, 25), (180, 30, 30)), (70, 615))       # its shoes reach left: it faces left
    m = (np.asarray(im) < 235).any(-1)
    assert ma_facing(m) <= -FACING_SURE
    assert ma_facing(m[:, ::-1]) >= FACING_SURE
    lines = []
    out = ma_face_right(im, log=lines.append)
    assert np.array_equal(np.asarray(out), np.asarray(im)[:, ::-1]) and "mirrored" in lines[0]
    assert ma_face_right(im, facing="right") is im

    class Judge:
        def facing(self, im):
            return "right"
    assert ma_face_right(im, judge=Judge()) is im                        # the judge's answer wins over the feet


def test_frame_failing_every_seed_is_drawn_from_a_neighbour(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    class Judge:
        """Frame 2's hand is never where the mannequin has it, unless frame 1 is the picture it was drawn from."""
        def __init__(self):
            self.calls = []

        def arms_ok(self, im, kind, hands=None):
            self.calls.append(im.size)
            n = len(self.calls)
            bad = 3 <= n <= 8                                    # frame 2's six seeds from the standing picture
            return (not bad), ("near hand missing" if bad else "2; near hand there")

        def feet_ok(self, im):
            return True, "2"

        def facing(self, im):
            return "right"
    try:
        ref = tmp_path / "person.png"
        Image.new("RGBA", (300, 700), (0, 0, 0, 0)).save(ref)
        lines = []
        res = ma_draw_cycle(MaRequest(reference=ref, frames=8, out=tmp_path / "out", name="t", height=120),
                            MaComfy(f"http://127.0.0.1:{srv.server_port}"), judge=Judge(), only=[0, 1, 2],
                            log=lines.append)
    finally:
        srv.shutdown()
    row = next(r for r in res["report"] if r["frame"] == 2)
    assert row["passed"] and row["from_frame"] == 1 and len(row["tried"]) == 7, lines
    assert any("drawn from frame 1" in s for s in lines)
    assert "from_frame" not in next(r for r in res["report"] if r["frame"] == 1)


def test_cycle_drawn_as_a_video_then_made_sharp(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Fake.calls.clear()
    try:
        ref = tmp_path / "person.png"
        Image.new("RGBA", (300, 700), (0, 0, 0, 0)).save(ref)
        res = ma_draw_cycle(MaRequest(reference=ref, out=tmp_path / "out", name="t", height=120, far_shade=0.55),
                            MaComfy(f"http://127.0.0.1:{srv.server_port}"), log=lambda s: None)
        frames = MaRequest(reference=ref, out=tmp_path).frames
        assert Fake.calls.count("video") == 2                            # again from its own frame 0 made sharp
        assert Fake.calls.count("sharpen") == frames + 1
        assert Fake.calls[:4] == ["frame", "free", "video", "free"]      # frame 0, then the video on a freed card
        assert Fake.calls[4:7] == ["sharpen", ("ref", "t_ref.png"), "free"]
        assert all(r["engine"] == "video" and r["passed"] for r in res["report"])
        after = Fake.calls[len(Fake.calls) - Fake.calls[::-1].index("video"):]
        refs = {c[1] for c in after if isinstance(c, tuple) and c[0] == "ref"}
        assert refs == {"t_walk_look.png"}                               # every frame from frame 0's look
        assert len(list((tmp_path / "out" / "raw" / "video").glob("walk_*.png"))) == 2 * frames + 1
        Fake.calls.clear()
        ma_draw_cycle(MaRequest(reference=ref, out=tmp_path / "f", name="t", height=120, engine="frames"),
                      MaComfy(f"http://127.0.0.1:{srv.server_port}"), log=lambda s: None)
        assert "video" not in Fake.calls and "sharpen" not in Fake.calls
    finally:
        srv.shutdown()


def test_video_frame_that_lost_a_hand_gets_the_hands_drawn_again(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    class Judge:
        """Frame 2 made sharp from the video has no near hand; drawn again inside the hand crops it has."""
        def __init__(self):
            self.n = 0

        def arms_ok(self, im, kind, hands=None):
            self.n += 1
            return (self.n != 5), ("near hand missing" if self.n == 5 else "2; near hand there")

        def feet_ok(self, im):
            return True, "2"

        def facing(self, im):
            return "right"
    try:
        ref = tmp_path / "person.png"
        Image.new("RGBA", (300, 700), (0, 0, 0, 0)).save(ref)
        lines = []
        res = ma_draw_cycle(MaRequest(reference=ref, frames=8, out=tmp_path / "out", name="t", height=120),
                            MaComfy(f"http://127.0.0.1:{srv.server_port}"), judge=Judge(), log=lines.append)
    finally:
        srv.shutdown()
    row = next(r for r in res["report"] if r["frame"] == 2)
    assert row["passed"] and row["engine"] == "video" and len(row["tried"]) == 2, lines
    assert any("frame 2" in s and "mended (near)" in s for s in lines)


def test_extra_hand_is_the_box_away_from_both_wrists():
    from marearts_moim_pose.gait import ma_gait, ma_frame_phases, ma_walk_points
    from marearts_moim_pose.pipeline import ma_extra_hands
    g = ma_gait("older")
    pts = ma_walk_points(ma_frame_phases(12, g)[8], g)
    # the judge's boxes on Otto's walk frame 8 (11-04): behind, under the coat between the two, in front
    boxes = [ma_frame_box(b, (832, 1216)) for b in ([328, 738, 388, 818], [438, 795, 528, 878], [600, 675, 695, 798])]
    extra = ma_extra_hands(boxes, pts)
    x0, y0, x1, y1 = boxes[1]
    assert list(extra.values()) == [(x0, y0 - (y1 - y0), x1, y1)]           # the middle one, up over its sleeve
    assert ma_extra_hands(boxes[::2], pts) == {}


def test_video_frame_with_a_hand_too_many_gets_it_drawn_over(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Fake.calls.clear()

    class Judge:
        """Frame 2 made sharp from the video has three hands; drawn over where the one too many is."""
        def __init__(self):
            self.n = 0

        def arms_ok(self, im, kind, hands=None):
            self.n += 1
            if self.n == 5:
                return False, "hands 3 at [150, 600, 200, 650], [380, 640, 430, 700], [600, 600, 650, 650]"
            return True, "hands 2; near hand there"

        def feet_ok(self, im):
            return True, "2"

        def facing(self, im):
            return "right"
    try:
        ref = tmp_path / "person.png"
        Image.new("RGBA", (300, 700), (0, 0, 0, 0)).save(ref)
        lines = []
        res = ma_draw_cycle(MaRequest(reference=ref, frames=8, out=tmp_path / "out", name="t", height=120),
                            MaComfy(f"http://127.0.0.1:{srv.server_port}"), judge=Judge(), log=lines.append)
    finally:
        srv.shutdown()
    row = next(r for r in res["report"] if r["frame"] == 2)
    assert row["passed"] and row["engine"] == "video" and len(row["tried"]) == 2, lines
    assert ("fix", EXTRA_FIX) in Fake.calls and ("fix", HAND_FIX) not in Fake.calls


def test_video_frame_kept_when_its_mend_fails_too(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Fake.calls.clear()

    class Judge:
        """Frame 2 fails every way: made sharp (1), its hands drawn again (6). Never drawn on its own."""
        def __init__(self):
            self.n = 0

        def arms_ok(self, im, kind, hands=None):
            self.n += 1
            bad = 5 <= self.n <= 11
            return (not bad), ("near hand missing" if bad else "2; near hand there")

        def feet_ok(self, im):
            return True, "2"

        def facing(self, im):
            return "right"
    try:
        ref = tmp_path / "person.png"
        Image.new("RGBA", (300, 700), (0, 0, 0, 0)).save(ref)
        lines = []
        res = ma_draw_cycle(MaRequest(reference=ref, far_shade=0.55, frames=8, out=tmp_path / "out", name="t", height=120),
                            MaComfy(f"http://127.0.0.1:{srv.server_port}"), judge=Judge(), log=lines.append)
    finally:
        srv.shutdown()
    row = next(r for r in res["report"] if r["frame"] == 2)
    assert not row["passed"] and row["engine"] == "video" and len(row["tried"]) == 7, lines    # the video's, kept
    assert any("kept the video's frame" in s for s in lines)
    assert all(r["passed"] for r in res["report"] if r["frame"] != 2)
    assert Fake.calls.count("frame") == 1                       # frame 0 before the video; nothing else on its own


def test_video_frame_of_another_person_gets_the_head_drawn_again(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Fake.calls.clear()

    class Judge:
        """Frame 2 from the video is someone else (its head unlike the picture's); its head drawn again, it is not."""
        def __init__(self):
            self.n = 0

        def arms_ok(self, im, kind, hands=None):
            return True, "2; near hand there"

        def feet_ok(self, im):
            return True, "2"

        def head_likeness(self, im, ref):
            self.n += 1
            return 0.8 if self.n == 5 else 0.97                 # frame 0, 0 sharp twice, 1, then frame 2 sharp

        def facing(self, im):
            return "right"
    try:
        ref = tmp_path / "person.png"
        Image.new("RGBA", (300, 700), (0, 0, 0, 0)).save(ref)
        lines = []
        res = ma_draw_cycle(MaRequest(reference=ref, frames=8, out=tmp_path / "out", name="t", height=120),
                            MaComfy(f"http://127.0.0.1:{srv.server_port}"), judge=Judge(), log=lines.append)
    finally:
        srv.shutdown()
    row = next(r for r in res["report"] if r["frame"] == 2)
    assert row["passed"] and row["engine"] == "video" and len(row["tried"]) == 2, lines
    assert any("another person" in s for s in lines)
    assert any("mended (head)" in s for s in lines)                     # the head, in the clothes the video drew
    assert ("fix", HEAD_FIX) in Fake.calls and Fake.calls.count("frame") == 1


def test_head_bar_follows_the_persons_own_frame_0(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    class Judge:
        """Frame 0 scores 0.92 against the picture, every frame after it 0.90 (a shaved head: little to compare)."""
        def __init__(self):
            self.n = 0

        def arms_ok(self, im, kind, hands=None):
            return True, "hands 2; near hand there"

        def feet_ok(self, im):
            return True, "2"

        def head_likeness(self, im, ref):
            self.n += 1
            return 0.92 if self.n == 1 else 0.90

        def facing(self, im):
            return "right"
    try:
        ref = tmp_path / "person.png"
        Image.new("RGBA", (300, 700), (0, 0, 0, 0)).save(ref)
        lines = []
        res = ma_draw_cycle(MaRequest(reference=ref, far_shade=0.55, frames=8, out=tmp_path / "out", name="t", height=120),
                            MaComfy(f"http://127.0.0.1:{srv.server_port}"), judge=Judge(), log=lines.append)
    finally:
        srv.shutdown()
    assert all(r["passed"] and r["engine"] == "video" for r in res["report"]), lines
    assert not any("another person" in s for s in lines)


def test_head_box_holds_the_mannequins_head_and_hair():
    from marearts_moim_pose.gait import FIG, NECK_I, NOSE, ma_walk_points
    from marearts_moim_pose.mannequin import ma_depth
    from marearts_moim_pose.pipeline import ma_head_box
    pts = ma_walk_points(0.0)
    x0, y0, x1, y1 = ma_head_box(pts)
    d = np.asarray(ma_depth(pts).convert("L"))
    head = d[:int(pts[NECK_I][1])] > 30
    ys, xs = np.nonzero(head)
    assert x0 <= xs.min() and xs.max() <= x1 and y0 <= ys.min() and ys.max() <= y1          # all of the head
    assert y1 < pts[NECK_I][1] + 0.05 * FIG and x0 < pts[NOSE][0] < x1                       # and not the body


def test_far_hand_asked_for_only_where_the_mannequin_shows_it():
    from marearts_moim_pose.comfy import ARMS_NEAR_BACK, ARMS_NEAR_BACK_HALF, ma_arm_words
    from marearts_moim_pose.gait import LSHO, RSHO, ma_frame_phases, ma_walk_points
    from marearts_moim_pose.judge import ma_hand_box
    from marearts_moim_pose.mannequin import ma_depth
    from marearts_moim_pose.pipeline import ma_hidden
    pts = [ma_walk_points(t) for t in ma_frame_phases(12)]
    widest = max(abs(p[LSHO + 1][0] - p[LSHO][0]) for p in pts)
    swung = [i for i, p in enumerate(pts) if ma_arm_words(p, widest) in (ARMS_NEAR_BACK, ARMS_NEAR_BACK_HALF)]
    hid = {i: ma_hidden(ma_depth(pts[i], skirt=1.05), ma_hand_box(pts[i], RSHO + 2)) for i in swung}
    assert [i for i in swung if hid[i] < 0.5] == [0, 1, 11]      # well out in front: asked for
    assert all(hid[i] > 0.5 for i in (2, 10))                      # at the half swing, behind her skirt: not

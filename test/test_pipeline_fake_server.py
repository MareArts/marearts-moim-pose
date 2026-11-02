"""The whole draw -> check -> lock -> sheet path against a fake ComfyUI server (no GPU, no models).

The fake server "draws" each frame as the uploaded depth mannequin painted on white, scaled a little differently
per frame, and "cuts it out" by thresholding. The lock must bring the heights back together."""
import io
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from marearts_spritestride.comfy import MaComfy, ma_ui_workflow
from marearts_spritestride.judge import ma_verdict
from marearts_spritestride.pipeline import MaRequest, ma_draw_cycle


class Fake(BaseHTTPRequestHandler):
    uploads: dict = {}
    outputs: dict = {}
    count = [0]

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
        if self.path == "/upload/image":
            name = re.search(rb'filename="([^"]+)"', data).group(1).decode()
            png = data[data.index(b"\x89PNG"):data.rindex(b"\r\n--")]
            self.uploads[name] = Image.open(io.BytesIO(png)).convert("RGB")
            return self._send(json.dumps({"name": name}).encode())
        g = json.loads(data)["prompt"]
        self.count[0] += 1
        pid = f"p{self.count[0]}"
        if "10" not in g and "2" not in g:                # a new character from words: a tall figure on white
            out = Image.new("RGB", (832, 1216), "white")
            out.paste(Image.new("RGB", (220, 1000), (200, 90, 60)), (300, 100))
        elif "10" in g:                                   # a frame: the mannequin painted, scaled per frame
            d = np.asarray(self.uploads[g["10"]["inputs"]["image"]].convert("L"))
            k = 1 + 0.03 * ((self.count[0] % 5) - 2)
            im = Image.fromarray(np.where((d > 30)[..., None], np.array([90, 120, 200], np.uint8), 255).astype(np.uint8))
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
            body = {pid: {"outputs": {"17": {"images": [{"filename": pid, "subfolder": "", "type": "output"}]}}}}
            return self._send(json.dumps(body).encode())
        if self.path.startswith("/view"):
            pid = re.search(r"filename=([^&]+)", self.path).group(1)
            return self._send(self.outputs[pid], "image/png")
        self.send_error(404)


def test_cycle_end_to_end(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ref = tmp_path / "person.png"
        Image.new("RGBA", (300, 700), (0, 0, 0, 0)).save(ref)
        req = MaRequest(reference=ref, out=tmp_path / "out", name="test", height=200)
        lines = []
        res = ma_draw_cycle(req, MaComfy(f"http://127.0.0.1:{srv.server_port}"), log=lines.append)
    finally:
        srv.shutdown()
    assert all(r["passed"] for r in res["report"]), lines
    tops = []
    for i in range(8):
        al = np.asarray(Image.open(tmp_path / "out" / f"walk_{i}.png"))[..., 3] > 128
        ys = np.nonzero(al.any(1))[0]
        tops.append(ys.max() - ys.min())
    assert max(tops) - min(tops) <= 0.02 * max(tops)       # heights within the walk's own 1.2 % wave
    meta = json.loads(open(res["files"]["meta"]).read())
    assert meta["frames"] == 8 and meta["frame_height"] == 200 and meta["stride_px_per_frame"] > 0
    assert Image.open(res["files"]["loop"]).n_frames == 8
    assert Image.open(res["files"]["review"]).width == 960


def test_redraw_gets_new_seeds_and_keeps_report(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ref = tmp_path / "person.png"
        Image.new("RGB", (300, 700), "white").save(ref)
        req = MaRequest(reference=ref, out=tmp_path / "out", name="test", height=120)
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
        res = ma_draw_cycle(MaRequest(reference=ref, out=tmp_path / "o", name="t"),
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
    from marearts_spritestride.comfy import ma_klein_text_graph
    from marearts_spritestride.pipeline import ma_new_person
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
    assert ma_verdict("No.", "carry") and not ma_verdict("Yes", "carry")

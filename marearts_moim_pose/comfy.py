"""Talking to a ComfyUI server, and the graphs it runs.

Only ComfyUI's own nodes are used (0.4x): UNETLoader, CLIPLoader (flux2), ReferenceLatent, Flux2Scheduler,
SamplerCustomAdvanced, and the background-removal nodes. Models (put them in ComfyUI/models/...):

  diffusion_models/flux-2-klein-4b-fp8.safetensors       black-forest-labs/FLUX.2-klein-4b-fp8          Apache-2.0
  text_encoders/qwen_3_4b.safetensors                     Comfy-Org/Flux-2-Klein-4B (split_files)        Apache-2.0
  vae/flux2-vae.safetensors                               Comfy-Org/Flux-2-Klein-4B (split_files)        Apache-2.0
  loras/flux2_klein_4b_refcontrol_depth.safetensors       thedeoxen/refcontrol-FLUX.2-klein-4B-reference-depth-lora
  background_removal/birefnet.safetensors                 Comfy-Org/BiRefNet                             MIT

Runs on an 8 GB card (tested on an RTX 2080 Super; fp8 weights are upcast on cards without fp8 compute).
"""
from __future__ import annotations

import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

from .gait import H, W

MODELS = dict(unet="flux-2-klein-4b-fp8.safetensors", clip="qwen_3_4b.safetensors", vae="flux2-vae.safetensors",
              lora="flux2_klein_4b_refcontrol_depth.safetensors", cutout="birefnet.safetensors")

STYLE = ("highly detailed anime illustration: fine hair strands with glossy highlights, cloth folds, seams and buttons, "
         "two-tone anime shading with soft highlights, crisp clean line art; the eye seen in profile large, open and "
         "clearly drawn — a dark iris with a white highlight — never hidden by the hair or the glasses")

# Words that fixed things the pictures alone did not: a third hand behind the hip, the mouth open as if talking,
# the box turning so its end face showed. Name in `look` any piece of clothing that changes between frames.
WALK = "walking, full body"
CARRY = ("walking, carrying a closed brown cardboard box in both hands in front of the chest, the box seen exactly "
         "from the side so only one flat face of it shows, held level, full body")
# "the far arm is hidden behind the body" left the picture's own hanging near arm in: a third hand before the coat on
# every seed; saying how each arm swings, as image 1 shows, drew two (11-03).
ARMS_WALK = ("The arms swing in the walk: the near arm swings forward or back exactly as the bright arm of image 1, the "
             "far arm shows only where it swings out past the body. No arm hangs straight down at the side. Two hands "
             "at most.")
ARMS_CARRY = ("Both forearms hold the box; the near arm is in front of the box, the far hand hidden behind it. Two "
              "hands at most.")
# The same words on every frame drew one hand in front of the coat and one behind it on all eight: the arms stood
# still while the legs walked, and where the mannequin's arms hang (legs passing) a third hand came. Each frame is
# told where its arms are (11-03).
ARMS_HANG = ("Both arms hang straight down at the sides, the near hand beside the hip, the far arm hidden behind the "
             "body; no hand in front of the body or behind it. Two hands at most.")
ARMS_NEAR_FORWARD = ("The near arm swings forward, its hand just in front of the hip; the far arm swings back, its "
                     "hand just behind the back. Two hands at most.")
# Swung back, the near hand stayed by the hip as in image 2 while the sleeve went back (three of eight frames of an
# older man): the hand is placed in words, and the hip named as where it is not.
ARMS_NEAR_BACK = ("The near arm swings back, the near hand behind the back of the body, past the back edge of the "
                  "clothes, nothing beside the hip; the far arm swings forward, its hand just in front of the body. Two "
                  "hands at most.")
# Drawn one by one, a hand was a fist in one frame and open in the next, and the loop flickered at the hands.
HANDS = "Both hands relaxed, the fingers loosely curled, the wrists straight."
HANG = 0.35                           # a swing under this share of the widest one: the arms hang


def ma_arm_words(pts: dict, widest: float) -> str:
    """Where a walk frame's arms are, in words: from the near elbow ahead of its shoulder (px) against the widest
    such swing of the cycle (the wrist is always a little ahead: the elbow bends)."""
    from .gait import LELB, LSHO
    ahead = pts[LELB][0] - pts[LSHO][0]
    if abs(ahead) < HANG * widest:
        return ARMS_HANG
    return ARMS_NEAR_FORWARD if ahead > 0 else ARMS_NEAR_BACK


def ma_prompt(look: str = "", kind: str = "walk", style: str = STYLE, change: str = "", arms: str = "") -> str:
    """The words for one frame. image 1 is the depth mannequin, image 2 the person (the LoRA's trained wording).

    look: the person in words as well as in image 2 (from the picture alone a piece that changes between frames
    drifts). change: the one thing to draw differently from image 2, e.g. "a fuller figure than in image 2".
    arms: this frame's arm words (ma_arm_words) in place of the general ones."""
    who = f"The same person as image 2, {look} —" if look else "The same person as image 2 —"
    same = f"the same face, hair, clothes and shoes; {change} —" if change else \
        "the same face, hair, clothes and shoes, nothing added —"
    action, arms = (CARRY, ARMS_CARRY) if kind == "carry" else (WALK, f"{arms or ARMS_WALK} {HANDS}")
    return (f"refcontrol, {who} {same} {action}, the mouth closed in a calm, gentle smile, side view facing right, "
            f"plain white background, {style}. {arms}")


def ma_klein_depth_graph(depth_image: str, ref_image: str, prompt: str, seed: int = 7, steps: int = 4,
                         cfg: float = 1.0, width: int = W, height: int = H, prefix: str = "moim-pose/frame",
                         models: dict | None = None) -> dict:
    """One frame: FLUX.2 [klein] 4B edit with the RefControl depth LoRA. The person picture goes in as a reference
    latent, so the face and clothes carry over; the pose comes from the depth mannequin. Distilled model: 4 steps,
    cfg 1. The negative carries the same reference latents with empty words. The output node is "17"."""
    m = dict(MODELS, **(models or {}))
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": m["lora"],
                                                              "strength_model": 1.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["clip"], "type": "flux2", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": m["vae"]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": ""}},
        "10": {"class_type": "LoadImage", "inputs": {"image": depth_image}},
        "11": {"class_type": "LoadImage", "inputs": {"image": ref_image}},
        "12": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["4", 0]}},
        "13": {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["4", 0]}},
        "14": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["5", 0], "latent": ["12", 0]}},   # image 1
        "15": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["14", 0], "latent": ["13", 0]}},  # image 2
        "7": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["6", 0], "latent": ["12", 0]}},
        "16": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["13", 0]}},
        "20": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "21": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
        "22": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "23": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "24": {"class_type": "CFGGuider", "inputs": {"model": ["2", 0], "positive": ["15", 0], "negative": ["16", 0],
                                                     "cfg": cfg}},
        "25": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["22", 0], "guider": ["24", 0],
                                                                 "sampler": ["23", 0], "sigmas": ["21", 0],
                                                                 "latent_image": ["20", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["4", 0]}},
        "17": {"class_type": "SaveImage", "inputs": {"images": ["26", 0], "filename_prefix": prefix}},
    }


PERSON = ("One character, the whole figure from the top of the head to the shoes, standing upright, seen exactly from "
          "the side facing right, arms relaxed at the sides, {look}, plain white background, nothing else in the "
          "picture, {style}")


def ma_klein_text_graph(look: str, seed: int = 7, steps: int = 4, width: int = W, height: int = H,
                        style: str = STYLE, prefix: str = "moim-pose/person", models: dict | None = None) -> dict:
    """A new character from words: FLUX.2 [klein] text-to-image, standing side view facing right, the picture
    `moim-pose draw` starts from. Output node "17"."""
    m = dict(MODELS, **(models or {}))
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["clip"], "type": "flux2", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": m["vae"]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0],
                                                         "text": PERSON.format(look=look, style=style)}},
        "16": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "20": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "21": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}},
        "22": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "23": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "24": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["16", 0],
                                                     "cfg": 1.0}},
        "25": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["22", 0], "guider": ["24", 0],
                                                                 "sampler": ["23", 0], "sigmas": ["21", 0],
                                                                 "latent_image": ["20", 0]}},
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["4", 0]}},
        "17": {"class_type": "SaveImage", "inputs": {"images": ["26", 0], "filename_prefix": prefix}},
    }


def ma_cutout_graph(image: str, model: str = MODELS["cutout"], prefix: str = "moim-pose/mask") -> dict:
    """The person's mask of a frame by BiRefNet (ComfyUI's own background-removal nodes). Output node "17"."""
    return {"1": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": model}},
            "2": {"class_type": "LoadImage", "inputs": {"image": image}},
            "3": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["1", 0], "image": ["2", 0]}},
            "4": {"class_type": "MaskToImage", "inputs": {"mask": ["3", 0]}},
            "17": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": prefix}}}


class MaComfy:
    """A ComfyUI server (default the local one). Pictures go up by /upload/image and come back by /view."""

    def __init__(self, url: str = "http://127.0.0.1:8188", timeout: float = 60):
        self.url, self.timeout = url.rstrip("/"), timeout

    def call(self, path: str, data: dict | None = None):
        req = urllib.request.Request(self.url + path, data=json.dumps(data).encode() if data is not None else None,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def upload(self, im: Image.Image, name: str) -> str:
        buf = io.BytesIO()
        im.save(buf, "PNG")
        boundary = "----moim-pose" + str(time.time_ns())
        body = b"".join([
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            "Content-Type: image/png\r\n\r\n".encode(), buf.getvalue(), f"\r\n--{boundary}--\r\n".encode()])
        req = urllib.request.Request(self.url + "/upload/image", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["name"]

    def run(self, graph: dict, out: Path, node: str = "17", retries: int = 1) -> Path:
        """Queue the graph, wait, save the output picture of `node` to `out`. Once more if the server fails mid-run
        (a memory fault on a small card often clears on the next call)."""
        for attempt in range(retries + 1):
            try:
                return self._run(graph, Path(out), node)
            except RuntimeError:
                if attempt == retries:
                    raise
                time.sleep(5)

    def _run(self, graph: dict, out: Path, node: str) -> Path:
        pid = self.call("/prompt", {"prompt": graph})["prompt_id"]
        while True:
            h = self.call(f"/history/{pid}").get(pid, {})
            if h.get("outputs"):
                img = h["outputs"][node]["images"][0]
                break
            if h.get("status", {}).get("status_str") == "error":
                raise RuntimeError(json.dumps(h["status"])[:2000])
            time.sleep(1)
        q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": "output"})
        with urllib.request.urlopen(f"{self.url}/view?{q}", timeout=self.timeout) as r:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(r.read())
        return out

    def object_info(self) -> dict:
        return self.call("/object_info")


WIDGET_TYPES = ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")


def ma_ui_workflow(api: dict, info: dict) -> dict:
    """An API graph as a UI workflow (format 0.4) you can drag into ComfyUI: nodes laid out in columns by depth,
    wires and widgets in the order the server's /object_info gives."""
    ids = {k: i + 1 for i, k in enumerate(api)}
    depth: dict = {}

    def d(k):
        if k not in depth:
            ups = [v[0] for v in api[k]["inputs"].values() if isinstance(v, list) and len(v) == 2 and v[0] in api]
            depth[k] = 1 + max((d(u) for u in ups), default=-1)
        return depth[k]

    for k in api:
        d(k)
    links, nodes, rows = [], [], {}
    outs: dict = {k: {} for k in api}
    for k, node in api.items():
        spec = info[node["class_type"]]
        order = list(spec["input"].get("required", {}).items()) + list(spec["input"].get("optional", {}).items())
        inputs, widgets = [], []
        for name, s in order:
            typ = s[0] if isinstance(s[0], str) else "COMBO"
            opts = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
            v = node["inputs"].get(name)
            if isinstance(v, list) and len(v) == 2 and v[0] in api:
                lid = len(links) + 1
                links.append([lid, ids[v[0]], v[1], ids[k], len(inputs), info[api[v[0]]["class_type"]]["output"][v[1]]])
                outs[v[0]].setdefault(v[1], []).append(lid)
                inputs.append({"name": name, "type": typ, "link": lid})
            elif typ in WIDGET_TYPES or isinstance(s[0], list):
                widgets.append(v if v is not None else opts.get("default"))
                if opts.get("control_after_generate"):
                    widgets.append("fixed")
            else:
                inputs.append({"name": name, "type": typ, "link": None})
        col = depth[k]
        rows[col] = rows.get(col, 0) + 1
        nodes.append({"id": ids[k], "type": node["class_type"], "pos": [40 + 360 * col, 40 + 230 * (rows[col] - 1)],
                      "size": [320, 160], "flags": {}, "order": col, "mode": 0, "inputs": inputs,
                      "outputs": [{"name": n, "type": t, "links": [], "slot_index": i}
                                  for i, (n, t) in enumerate(zip(spec.get("output_name", spec["output"]),
                                                                 spec["output"]))],
                      "properties": {"Node name for S&R": node["class_type"]}, "widgets_values": widgets})
    for n, k in zip(nodes, api):
        for slot, lids in outs[k].items():
            n["outputs"][slot]["links"] = lids
    return {"last_node_id": len(nodes), "last_link_id": len(links), "nodes": nodes, "links": links, "groups": [],
            "config": {}, "extra": {}, "version": 0.4}

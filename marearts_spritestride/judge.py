"""A vision-language model looks at each frame and fails the ones with a hand too many.

Measured checks miss a third hand drawn over the body; a small question to a local VLM on a crop of the torso
catches it. On our test set (every frame a person had flagged, plus the good ones) it caught all of them and failed
no good frame (40/40), with Qwen3.5-9B (Apache-2.0). Any model that answers the two questions will do.

Two ways to run it:
  MaLocalJudge("Qwen/Qwen3.5-9B")          transformers on this machine (about 20 GB of GPU memory in bf16)
  MaHttpJudge("http://127.0.0.1:8080")     any OpenAI-compatible server (llama.cpp, vLLM, Ollama) with a vision model
"""
from __future__ import annotations

import base64
import io
import json
import re
import urllib.request

from PIL import Image

ASK_WALK = ("Look at this cartoon drawing closely. How many separate hands (fists or palms with fingers) can you see? "
            "Count every hand, also one that is partly behind the body or the trousers. Answer with one number only.")
# Carrying, a third arm hangs by the hip while two hands hold the box, and a hand count sees only those two.
ASK_CARRY = ("Look at this cartoon drawing closely. Is there a hand or arm hanging down beside the hip or the trousers, "
             "below the box and not touching the box? Answer yes or no.")
TORSO = (0.15, 0.25, 0.85, 0.62)          # shoulders to thighs: hands are small in the whole frame


def ma_torso(im: Image.Image) -> Image.Image:
    w, h = im.size
    x0, y0, x1, y1 = TORSO
    return im.convert("RGB").crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))


def ma_verdict(answer: str, kind: str) -> bool:
    """Carrying: no hand hangs by the hip. Walking: at most two hands."""
    if kind == "carry":
        return not answer.strip().lower().startswith("yes")
    m = re.search(r"\d+", answer)
    return bool(m) and int(m.group()) <= 2


class MaJudge:
    def ask(self, im: Image.Image, text: str) -> str:
        raise NotImplementedError

    def arms_ok(self, im: Image.Image, kind: str) -> tuple[bool, str]:
        """(frame passes, the model's answer)."""
        ans = self.ask(ma_torso(im), ASK_CARRY if kind == "carry" else ASK_WALK)
        return ma_verdict(ans, kind), ans


class MaLocalJudge(MaJudge):
    def __init__(self, model_id: str = "Qwen/Qwen3.5-9B", device: str = "cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16,
                                                                 device_map=device).eval()

    def ask(self, im: Image.Image, text: str) -> str:
        msg = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": text}]}]
        x = self.proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True,
                                          return_tensors="pt", enable_thinking=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**x, max_new_tokens=8, do_sample=False)
        return self.proc.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class MaHttpJudge(MaJudge):
    def __init__(self, url: str = "http://127.0.0.1:8080", model: str = "local", timeout: float = 120):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def ask(self, im: Image.Image, text: str) -> str:
        buf = io.BytesIO()
        im.save(buf, "PNG")
        image = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        body = {"model": self.model, "temperature": 0, "max_tokens": 8,
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}},
                                                          {"type": "text", "text": text}]}]}
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()

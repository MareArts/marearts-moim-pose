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

# Asked for a number alone it said 2 on frames with a third hand hanging before the coat (5 of 9 wrong); made to
# name each hand first, it counted all 9 right (Rin's walk, 11-03).
ASK_WALK = ("Look at this cartoon drawing closely. List every hand you can see, from left to right, one short phrase "
            "each (for example: hand behind the body, hand in front of the coat, hand in front of the body). Then on "
            "the last line write the number of hands.")
# Carrying, a third arm hangs by the hip while two hands hold the box, and a hand count sees only those two.
ASK_CARRY = ("Look at this cartoon drawing closely. Is there a hand or arm hanging down beside the hip or the trousers, "
             "below the box and not touching the box? Answer yes or no.")
TORSO = (0.15, 0.25, 0.85, 0.62)          # shoulders to thighs: hands are small in the whole frame
# A third shoe between the two (a leg drawn twice below the knee) passes the leg count across the shins. Asked to name
# each shoe of the feet alone, it found all three on 48 frames of two characters and failed no good one (11-03).
ASK_FEET = ("Look at this cartoon drawing of a person's legs closely. List every shoe you can see, from left to right, "
            "one short phrase each. Then on the last line write the number of shoes.")
FEET = (0.0, 0.62, 1.0, 1.0)
# The reference picture's way: on 16 pictures (8 and the same mirrored) it answered all 16 right, on the head alone.
ASK_FACING = ("This is a side view of one person. Which way is the person's face pointing: to the left side of the "
              "picture or to the right side? Answer with one word: left or right.")
HEAD = (0.0, 0.0, 1.0, 0.3)
# A count says nothing of where the hands are: an older man's near arm stayed by the hip in five of eight frames
# while the mannequin swung it back, and two hands were still two. Asked yes/no on a small crop where the mannequin
# has the hand, it says no where the hand did not follow (11-03). The crop sits just past the wrist and is 10 % of
# the figure across: a wider one, further down the forearm, let a hand drawn in front of the hip through when half
# of it reached in (two frames of a 12-frame walk, the arm half way back: the hand jumped forward and back); on 24
# frames this one failed exactly those two.
ASK_HAND_HERE = ("Look at the middle of this picture. Is a human hand (fingers, a fist or the back of a hand) there? "
                 "Answer yes or no.")
HAND_BOX, HAND_PAST = 0.05, 0.015        # half the crop's side, its centre past the wrist; shares of the figure


def ma_hand_box(pts: dict, wrist: int) -> tuple[int, int, int, int]:
    """The crop where the mannequin has a hand: just past the wrist along the forearm."""
    from .gait import FIG
    (ex, ey), (x, y) = pts[wrist - 1], pts[wrist]
    n = max(1e-6, ((x - ex) ** 2 + (y - ey) ** 2) ** 0.5)
    x, y, r = x + (x - ex) / n * HAND_PAST * FIG, y + (y - ey) / n * HAND_PAST * FIG, HAND_BOX * FIG
    return int(x - r), int(y - r), int(x + r), int(y + r)


def ma_torso(im: Image.Image) -> Image.Image:
    w, h = im.size
    x0, y0, x1, y1 = TORSO
    return im.convert("RGB").crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))


def ma_verdict(answer: str, kind: str) -> bool:
    """Carrying: no hand hangs by the hip. Walking: at most two hands."""
    if kind == "carry":
        return not answer.strip().lower().startswith("yes")
    m = re.findall(r"\d+", answer)                  # the last number: the count after the list
    return bool(m) and int(m[-1]) <= 2


class MaJudge:
    def ask(self, im: Image.Image, text: str) -> str:
        raise NotImplementedError

    def arms_ok(self, im: Image.Image, kind: str, hands: dict | None = None) -> tuple[bool, str]:
        """(frame passes, the model's answers). hands: {name: crop box} where the mannequin has a hand that must be
        drawn there (ma_hand_box)."""
        ans = self.ask(ma_torso(im), ASK_CARRY if kind == "carry" else ASK_WALK)
        ok, said = ma_verdict(ans, kind), ans
        for name, box in (hands or {}).items():
            if not ok:
                break
            here = self.ask(im.convert("RGB").crop(box), ASK_HAND_HERE).strip().lower()
            ok = here.startswith("yes")
            said += f"; {name} hand {'there' if ok else 'missing'}"
        return ok, said

    def feet_ok(self, im: Image.Image) -> tuple[bool, str]:
        """(two shoes at most, the model's answer) on the legs and feet."""
        w, h = im.size
        x0, y0, x1, y1 = FEET
        ans = self.ask(im.convert("RGB").crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1))), ASK_FEET)
        m = re.findall(r"\d+", ans)
        return bool(m) and int(m[-1]) <= 2, ans

    def facing(self, im: Image.Image) -> str:
        """'left', 'right' or '' (no clear answer): which way the person in a standing picture faces."""
        w, h = im.size
        x0, y0, x1, y1 = HEAD
        ans = self.ask(im.convert("RGB").crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1))), ASK_FACING)
        ans = ans.strip().lower()
        return "left" if ans.startswith("left") else "right" if ans.startswith("right") else ""


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
            out = self.model.generate(**x, max_new_tokens=96, do_sample=False)
        return self.proc.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class MaHttpJudge(MaJudge):
    def __init__(self, url: str = "http://127.0.0.1:8080", model: str = "local", timeout: float = 120):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def ask(self, im: Image.Image, text: str) -> str:
        buf = io.BytesIO()
        im.save(buf, "PNG")
        image = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        body = {"model": self.model, "temperature": 0, "max_tokens": 96,
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}},
                                                          {"type": "text", "text": text}]}]}
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()

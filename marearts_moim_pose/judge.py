"""A vision-language model looks at each frame and fails the ones with a hand too many.

Measured checks miss a third hand drawn over the body; a small question to a local VLM on a crop of the torso
catches it. On our test set (every frame a person had flagged, plus the good ones) it caught all of them and failed
no good frame (40/40), with Qwen3.5-9B (Apache-2.0). The hands are boxed one by one and the boxes counted; a model
that answers in words instead is read by its last number.

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

import numpy as np
from PIL import Image

# Asked for a number alone it said 2 on frames with a third hand hanging before the coat (5 of 9 wrong); made to
# name each hand first, it counted all 9 right (Rin's walk, 11-03). Naming still missed a hand drawn under the coat
# between the two (an older man's video frame a person saw at once, and 6 of 12 such frames in all). Asked for a box
# around every hand it boxed three on all 12 and two or fewer on 56 frames, the 36 of the last two walks and carries
# among them (11-04): a box is counted where a list was skimmed.
ASK_WALK = ("Find every human hand in this cartoon drawing (fingers, a fist or the back of a hand), also a hand partly "
            "hidden or drawn in front of the coat or trousers. Output a JSON list, one entry per hand: "
            '[{"bbox_2d": [x1, y1, x2, y2], "label": "hand"}]')
ASK_HANDS = ASK_WALK                     # walking or carrying, the same question
# Carrying, a third arm hangs by the hip while two hands hold the box, and a hand count sees only those two.
ASK_CARRY = ("Look at this cartoon drawing closely. Is there a hand or arm hanging down beside the hip or the trousers, "
             "below the box and not touching the box? Answer yes or no.")
TORSO = (0.15, 0.25, 0.85, 0.62)          # shoulders to thighs: hands are small in the whole frame
# A third shoe between the two (a leg drawn twice below the knee) passes the leg count across the shins. Asked to name
# each shoe of the feet alone, it found all three on 48 frames of two characters and failed no good one (11-03).
ASK_FEET = ("Look at this cartoon drawing of a person's legs closely. List every shoe you can see, from left to right, "
            "one short phrase each. Then on the last line write the number of shoes.")
# On 204 walk frames of 17 game characters it said yes on 29: about 14 ghost or smeared shoes, the rest mostly feet
# crossing at mid-stance (11-09). A yes only sends the feet to be drawn again, kept if they come out clean.
ASK_FEET_CLEAN = ("Look closely at the shoes and feet in this cartoon drawing of a walking person. Is any shoe drawn "
                  "twice (a ghost or second copy of the shoe), smeared or streaked, bent at an impossible angle at the "
                  "ankle, or with a stray blob or white streak next to it? Answer yes or no on the first line, then "
                  "name the problem in a few words.")
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
# Heeled shoes or flat, on the feet of the person's picture: on 50 game characters it named the three in pumps or heels
# and no one else (11-05). The mannequin's feet were flat for all, and the video drew pumps as low shoes.
ASK_HEELS = ("Look at the shoes in this cartoon drawing. Do they have high heels (the heel raised well above the toes, "
             "like pumps, stilettos or heeled boots), or are they flat (sneakers, flats, loafers, boots with a low heel)? "
             "Answer with one word: heels or flat.")
# The person from head to toe, for the words every frame and the video are drawn with: the words a person wrote for
# a girl said "brown shoes" and not her white socks with a folded grey cuff; frame 0, drawn from her picture, had
# the socks, the video drew bare skin under the cuff and every frame made sharp from it a band round a bare ankle
# (11-07). What the words leave out, the video does not keep.
ASK_LOOK = ("Describe this cartoon character from head to toe for an artist who must draw the same person again "
            "exactly: hair (colour, length, style, clips or bands), face (glasses, earrings, make-up), every layer "
            "worn on top (each garment, its colour, sleeves, collar, buttons), the skirt or trousers (colour, "
            "length, cut), the legs (bare, socks with their colour, length and any folded cuff, tights or "
            "stockings), the shoes (colour, kind, laces, heel, sole colour) and anything carried or worn. One line "
            "of short phrases separated by commas, from the head down; nothing about the pose, the background or "
            "the drawing style.")
HAND_BOX, HAND_PAST = 0.05, 0.015        # half the crop's side, its centre past the wrist; shares of the figure
# Whether the person in the picture wears a skirt: with the words (ma_skirt_of) it found all 12 skirts of 50 the game
# people and none of the jeans the words of one called a skirt; alone it boxed wide trousers as a skirt on 3. Its
# box's bottom moved between asks (0.71 then 0.78 of the height for one hem), so the hem is measured on the pixels.
ASK_SKIRT = ('Find the skirt or dress this person wears. A skirt or dress is one piece of cloth hanging around both '
             'legs; trousers, jeans, joggers, leggings and shorts are not. Output a JSON list: '
             '[{"bbox_2d": [x1, y1, x2, y2], "label": "skirt"}], or [] if the person wears none.')


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


def ma_hand_boxes(answer: str) -> list[tuple[int, ...]]:
    """The boxes in a hand answer, one per hand: a box mostly inside another (the same hand twice) counts once."""
    boxes = []
    for b in (tuple(map(int, m)) for m in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", answer)):
        x0, y0, x1, y1 = b
        area = max(1, (x1 - x0) * (y1 - y0))
        if all(max(0, min(x1, c[2]) - max(x0, c[0])) * max(0, min(y1, c[3]) - max(y0, c[1])) < 0.5 * area for c in boxes):
            boxes.append(b)
    return boxes


def ma_frame_box(box: tuple[int, ...], size: tuple[int, int]) -> tuple[int, ...]:
    """A box the model gave on the torso crop (0-1000 each way) in the frame's pixels."""
    w, h = size
    x0, y0, x1, y1 = TORSO
    return tuple(int(round(v)) for v in (w * (x0 + box[0] / 1000 * (x1 - x0)), h * (y0 + box[1] / 1000 * (y1 - y0)),
                                         w * (x0 + box[2] / 1000 * (x1 - x0)), h * (y0 + box[3] / 1000 * (y1 - y0))))


def ma_verdict(answer: str, kind: str) -> bool:
    """Carrying: no hand hangs by the hip. Walking: at most two hands (boxes, or the last number of a model that
    answers in words)."""
    if kind == "carry":
        return not answer.strip().lower().startswith("yes")
    boxes = ma_hand_boxes(answer)
    if boxes:
        return len(boxes) <= 2
    m = re.findall(r"\d+", answer)                  # the last number: the count after the list
    return bool(m) and int(m[-1]) <= 2


class MaJudge:
    def ask(self, im: Image.Image, text: str) -> str:
        raise NotImplementedError

    def arms_ok(self, im: Image.Image, kind: str, hands: dict | None = None) -> tuple[bool, str]:
        """(frame passes, the model's answers). hands: {name: crop box} where the mannequin has a hand that must be
        drawn there (ma_hand_box)."""
        ok, said = True, ""
        if kind == "carry":
            said = self.ask(ma_torso(im), ASK_CARRY).strip()
            ok = ma_verdict(said, "carry")
        if ok:                                       # carrying too: no hand past the two on the box
            ans = self.ask(ma_torso(im), ASK_HANDS)
            boxes = ma_hand_boxes(ans)
            ok, n = ma_verdict(ans, "walk"), len(boxes)
            # too many: where each one is, in the frame's pixels, so the one too many can be drawn over
            at = " at " + ", ".join(str(list(ma_frame_box(b, im.size))) for b in boxes) if n > 2 else ""
            said = "; ".join(filter(None, [said, f"hands {n}{at}" if n else ans.strip()]))
        for name, box in (hands or {}).items():
            if not ok:
                break
            here = self.ask(im.convert("RGB").crop(box), ASK_HAND_HERE).strip().lower()
            ok = here.startswith("yes")
            said += f"; {name} hand {'there' if ok else 'missing'}"
        return ok, said

    def head_likeness(self, im: Image.Image, ref: Image.Image) -> float | None:
        """How like the picture's head a frame's head is (cosine of DINOv2 features, 1 = same), or None when this
        judge cannot tell. MaLikeness does it on the CPU; a judge may hand it to one."""
        return None

    def feet_ok(self, im: Image.Image) -> tuple[bool, str]:
        """(two shoes at most, the model's answer) on the legs and feet."""
        w, h = im.size
        x0, y0, x1, y1 = FEET
        ans = self.ask(im.convert("RGB").crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1))), ASK_FEET)
        m = re.findall(r"\d+", ans)
        return bool(m) and int(m[-1]) <= 2, ans

    def feet_clean(self, im: Image.Image) -> tuple[bool, str]:
        """(no shoe drawn twice, smeared or with a stray piece, the model's answer) on the legs and feet. The video
        draws a back foot lifting off as a ghost shoe or a white streak, and two shoes are still counted (11-09)."""
        w, h = im.size
        x0, y0, x1, y1 = FEET
        ans = self.ask(im.convert("RGB").crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1))), ASK_FEET_CLEAN)
        return not ans.strip().lower().startswith("yes"), ans

    def skirt_hem(self, im: Image.Image) -> float | None:
        """Where the skirt in a standing picture ends, as a share of the figure's height from the top of the head
        (None: no skirt found). The picture on a light background."""
        a = (np.asarray(im.convert("RGB")) < 235).any(-1)
        ys, xs = np.nonzero(a)
        if not len(ys):
            return None
        top, bot = ys.min(), ys.max()
        fig = im.convert("RGB").crop((max(0, xs.min() - 10), max(0, top - 10), xs.max() + 10, bot + 10))
        boxes = ma_hand_boxes(self.ask(fig, ASK_SKIRT))
        if not boxes:
            return None
        hem = max(b[3] for b in boxes) / 1000 * fig.size[1] + max(0, top - 10)
        return float((hem - top) / max(1, bot - top))

    def heels(self, im: Image.Image) -> bool | None:
        """Whether the person in a standing picture (on a light background) wears heeled shoes; None: no answer."""
        a = (np.asarray(im.convert("RGB")) < 235).any(-1)
        ys, xs = np.nonzero(a)
        if not len(ys):
            return None
        top, bot = ys.min(), ys.max()
        feet = im.convert("RGB").crop((max(0, xs.min() - 10), int(bot - 0.2 * (bot - top)), xs.max() + 10, bot + 10))
        ans = self.ask(feet, ASK_HEELS).strip().lower()
        return True if ans.startswith("heel") else False if ans.startswith("flat") else None

    def describe(self, im: Image.Image) -> str:
        """The person in a standing picture (on a light background) in words, head to toe, one line (ASK_LOOK)."""
        a = (np.asarray(im.convert("RGB")) < 235).any(-1)
        ys, xs = np.nonzero(a)
        fig = im.convert("RGB") if not len(ys) else im.convert("RGB").crop(
            (max(0, xs.min() - 10), max(0, ys.min() - 10), xs.max() + 10, ys.max() + 10))
        ans = " ".join(self.ask(fig, ASK_LOOK).split())
        return ans.strip(" .")

    def facing(self, im: Image.Image) -> str:
        """'left', 'right' or '' (no clear answer): which way the person in a standing picture faces."""
        w, h = im.size
        x0, y0, x1, y1 = HEAD
        ans = self.ask(im.convert("RGB").crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1))), ASK_FACING)
        ans = ans.strip().lower()
        return "left" if ans.startswith("left") else "right" if ans.startswith("right") else ""


# A video model can change the person while every hand and shoe is right: an older man's flat cap lost its brim
# and his short hair grew to the neck in 9 frames of 12. The VLM saw a flat cap and a profile in all of them; head
# features did not. On the top 14 % of the figure (hat, hair, the top of the face) against his picture those 9 came
# to 0.87-0.90 and 63 frames that were the person (him and a girl, walking and with a box, drawn either way) to
# 0.93-0.98; a bigger crop took in the collar or a carried box and the two overlapped (11-04).
HEAD_MIN = 0.915
HEAD_SLACK, HEAD_FLOOR = 0.03, 0.85        # the bar from the person's own frame 0 (pipeline): its score less this, not under
HEAD_CROP = (0.25, 0.03, 0.85, 0.30)


def ma_head(im: Image.Image, share: float = 0.14) -> Image.Image:
    """The head of the person on a light background: the top `share` of the figure, square, around the drawn
    pixels in those rows (a fixed crop of the frame took in a carried box and missed a head drawn lower)."""
    import numpy as np
    rgb = im.convert("RGB")
    a = np.asarray(rgb).astype(int)
    ink = (a < 235).any(-1)
    rows = np.nonzero(ink.any(1))[0]
    if not len(rows):
        w, h = im.size
        x0, y0, x1, y1 = HEAD_CROP
        return rgb.crop((int(w * x0), int(h * y0), int(w * x1), int(h * y1)))
    top, h = rows.min(), rows.max() - rows.min()
    y1 = top + max(1, int(share * h))
    cols = np.nonzero(ink[top:y1].any(0))[0]
    cx, side = (cols.min() + cols.max()) / 2, max(cols.max() - cols.min(), y1 - top)
    return rgb.crop((int(cx - side / 2), int(top), int(cx + side / 2), int(top + side)))


class MaLikeness:
    """Head likeness by DINOv2 small (Apache-2.0) on the CPU, loaded on first use."""

    def __init__(self, model_id: str = "facebook/dinov2-small"):
        self.model_id, self.model = model_id, None

    def __call__(self, im: Image.Image, ref: Image.Image) -> float:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        if self.model is None:
            self.proc = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id).eval()
        with torch.no_grad():
            e = self.model(**self.proc(images=[ma_head(im), ma_head(ref)], return_tensors="pt")).last_hidden_state[:, 0]
        e = torch.nn.functional.normalize(e, dim=-1)
        return float(e[0] @ e[1])


class MaLocalJudge(MaJudge):
    def __init__(self, model_id: str = "Qwen/Qwen3.5-9B", device: str = "cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16,
                                                                 device_map=device).eval()

    def head_likeness(self, im: Image.Image, ref: Image.Image) -> float:
        if not hasattr(self, "likeness"):
            self.likeness = MaLikeness()
        return self.likeness(im, ref)

    def ask(self, im: Image.Image, text: str) -> str:
        msg = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": text}]}]
        x = self.proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True,
                                          return_tensors="pt", enable_thinking=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**x, max_new_tokens=256, do_sample=False)
        return self.proc.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class MaHttpJudge(MaJudge):
    def __init__(self, url: str = "http://127.0.0.1:8080", model: str = "local", timeout: float = 120):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def ask(self, im: Image.Image, text: str) -> str:
        buf = io.BytesIO()
        im.save(buf, "PNG")
        image = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        body = {"model": self.model, "temperature": 0, "max_tokens": 256,
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}},
                                                          {"type": "text", "text": text}]}]}
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()

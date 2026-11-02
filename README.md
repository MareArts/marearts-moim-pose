# marearts-spritestride

<p align="center">
  <img src="docs/walk-loops.gif" alt="Rin and Otto, each drawn from one sentence, walking and carrying a box in 8-frame loops" width="100%">
</p>

Walk-cycle sprites of one person from one picture. Give it a side view of a character; it draws the same
character walking (or carrying a box) in 8 frames that loop, checks every frame, locks height and head size
across the cycle, and writes a sprite sheet, GIFs and a small JSON a game can read.

It runs on an 8 GB GPU through ComfyUI with FLUX.2 [klein] 4B, and every model it uses is Apache-2.0 or MIT.

**ComfyUI workflows: [download on MareArts Open Lab](https://www.marearts.com/blogs/open-lab/marearts-spritestride-walk-cycle-sprites-from-one-picture?utm_source=github&utm_medium=readme&utm_campaign=spritestride)**

## Why frames drift, and what fixes it

| Problem | Why | What spritestride does |
|---|---|---|
| The character changes from frame to frame (sleeves, hem, colours) | Each frame is drawn on its own | An **edit** model: the character's own picture goes in as a reference latent, and the words name each piece of clothing |
| It walks "on the same foot" | In profile the two legs of a pose skeleton lie on one line, so the model cannot tell which is in front | A **depth mannequin** instead of a skeleton: far limbs dark, near limbs bright. In the finished frames the **far leg is drawn darker**, as sprite artists do, so the left and right steps read apart even in the same trousers |
| A third hand appears now and then | The model guesses where the far arm is | Measured checks against the mannequin, and an optional **vision-model judge** on a crop of the torso; a failed frame is drawn again with another seed |
| Height and head size wobble a few percent, and the feet slide | Drawn frames are never exactly the size of the pose | A **lock**: one master frame sets the head and the floor, the height follows the walk's own smooth wave, the legs are laid on the mannequin's legs |
| The sprite skates across the ground in a game | Even time steps move the planted foot unevenly | Frame phases chosen so the planted foot goes back the **same distance every frame**; that distance is written to the JSON |

<p align="center"><img src="docs/mannequin.png" alt="Two pose skeletons half a step apart look almost the same; two depth mannequins show the near leg in front, then behind" width="100%"></p>
<p align="center"><sub>Left: the pose skeleton at two steps half a cycle apart — almost the same shape. Right: the depth mannequin at the same steps — near limbs bright, far ones dark.</sub></p>

## Examples

Both characters were made from words only (`spritestride person`), then walked and given a box. All 32 frames
passed the automatic checks at the first seed; four of Rin's frames were drawn again after a look (see below).

| | made with |
|---|---|
| **Rin** | `--gait lively --look "a young woman with a short black bob, a yellow raincoat, dark blue jeans, red sneakers"` |
| **Otto** | `--gait older --look "an older man with a short grey beard and a brown flat cap, a brown corduroy jacket over a cream shirt, olive trousers, brown leather boots"` |

The loops at the top are the four cycles. Rin's walk as the sprite sheet a game reads:

<p align="center"><img src="docs/rin-walk-sheet.png" alt="Rin's walk: the eight frames of the sprite sheet, the far leg darker" width="70%"></p>

## Install

```bash
pip install git+https://github.com/MareArts/marearts-spritestride            # Python 3.10+
pip install "marearts-spritestride[judge] @ git+https://github.com/MareArts/marearts-spritestride"   # + the judge
```

You also need a running [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (0.4x, its own nodes only) with these
models:

| Put in `ComfyUI/models/` | From | Licence |
|---|---|---|
| `diffusion_models/flux-2-klein-4b-fp8.safetensors` | [black-forest-labs/FLUX.2-klein-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) | Apache-2.0 |
| `text_encoders/qwen_3_4b.safetensors` | [Comfy-Org/Flux-2-Klein-4B](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B) `split_files/` | Apache-2.0 |
| `vae/flux2-vae.safetensors` | [Comfy-Org/Flux-2-Klein-4B](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B) `split_files/` | Apache-2.0 |
| `loras/flux2_klein_4b_refcontrol_depth.safetensors` | [thedeoxen/refcontrol-FLUX.2-klein-4B-reference-depth-lora](https://huggingface.co/thedeoxen/refcontrol-FLUX.2-klein-4B-reference-depth-lora) | Apache-2.0 |
| `background_removal/birefnet.safetensors` | [Comfy-Org/BiRefNet](https://huggingface.co/Comfy-Org/BiRefNet) `background_removal/` | MIT |

Judge (optional): [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (Apache-2.0) through transformers, or any
vision model behind an OpenAI-compatible server (llama.cpp, vLLM, Ollama).

## Use

The reference picture: the whole character standing, side view, **facing right**, on a plain background.

```bash
# no picture yet? candidates for a new character from words (pick one you like)
spritestride person --out people/alice \
    --look "a woman with a red-orange bob and round glasses, a mustard knit sweater, rolled blue jeans, black sneakers"

# see the poses first (no GPU, no server)
spritestride mannequin --out preview/ --gait lively

# draw a walk; describe the character in words too, it keeps clothes from drifting
spritestride draw alice.png --out out/alice --gait neat \
    --look "a woman with a red-orange bob and round glasses, a mustard knit sweater, rolled blue jeans, black sneakers"

# carrying a box, with the judge on a local OpenAI-compatible server
spritestride draw alice.png --out out/alice --kind carry --judge http://127.0.0.1:8080

# draw frames 2 and 6 again, then lock and sheet the whole cycle
spritestride draw alice.png --out out/alice --only 2 6
spritestride finish --out out/alice
```

| Option | Meaning |
|---|---|
| `--gait` | `neat relaxed lively upright calm graceful heavy older energetic loose`; or `--step --knee --arm --lean` |
| `--head` | head width / height of the character (0.19 for a big anime head with hair); the mannequin's head follows it |
| `--bust` | how far the chest stands out in the mannequin (0.13 for a fuller figure); the silhouette comes from the depth |
| `--change` | the one thing to draw differently from the picture |
| `--height` | sprite height in px (default 256, 0 = full 832 x 1216 frame) |
| `--far-shade` | how bright the far leg stays (default 0.55; 1 = as drawn) |

Out, in `--out`:

| File | |
|---|---|
| `walk_sheet.png` | 4 x 2 sprite sheet, transparent |
| `walk_sheet.json` | frame size, fps, `stride_px_per_frame`: move the sprite that far each frame and the feet stay put |
| `walk_loop.gif` · `walk_strip.gif` | the loop in place · walking across a strip at its own stride (sliding feet show here) |
| `walk_0.png` … | full-size locked frames |
| `walk_review.png` | every frame with its number, seed and checks; failed ones in red — look at this first |
| `walk_report.json` | per frame: seed, every seed tried, measured checks, judge answer, seconds |
| `raw/` | the drawn frames, masks and mannequins before the lock |

On an RTX 2080 Super (8 GB) a frame takes about half a minute; a cycle of 8 about 4-6 minutes with retries.

## It is not 100 % — plan on a look

An image model draws every frame, and now and then it draws something wrong. Measured over 507 frames drawn on an
8 GB card:

| | |
|---|---|
| Frames that passed the automatic checks at the first seed | 84 % — the rest at the 2nd to 4th seed |
| Walk cycles a person passed at the first look (one review of 12) | 4 of 12 — the others had a leg or a hand drawn oddly, a hat, beard or pocket that changed, a mouth open |
| What the checks catch | a third arm or leg, a carried box that turns or bobs, a third hand (with the judge) |
| What they do not catch yet | a leg the model drew on the wrong side of its mannequin, an odd hand pose, small changes in clothes or face. The far leg is darkened from the mannequin (`--far-shade`, 1 = off), so a cycle drawn right reads left-right; a leg drawn wrong still needs your eye |

<p align="center"><img src="docs/rin-walk-review.png" alt="The review sheet: every frame with its number, seed and what was measured" width="100%"></p>
<p align="center"><sub>Rin's walk review sheet after the redraws. Frames 3 and 7 first came out with the face turned toward the viewer; the next seeds turned her further, so frame 3 went back to seed 7 (<code>--use-seed 7</code>).</sub></p>

So the tool gives you **drafts that are checked**, not finished art. After every cycle open `walk_review.png`: every
frame with its number, seed and what was measured, frames that failed the checks boxed in red, and what to look at
written on top. Draw the frames you do not like again by number; they get seeds they have not had, and the report
keeps every frame's history:

```bash
spritestride draw alice.png --out out/alice --only 3 5      # frames 3 and 5 again, then lock and sheets again
spritestride draw alice.png --out out/alice --only 3 --use-seed 7   # a new try was worse: back to seed 7
```

A new seed is not always better. Each frame's report row keeps every seed it has tried, so you can go back.

Most fixes that stick are words, not retries: name in `--look` the piece that changed (a beanie "pulled down over the
whole top of the head", "bare legs of smooth skin"), and the next draw keeps it.

## The graphs in ComfyUI

**Get the ready-made ComfyUI workflows on the [spritestride page of MareArts Open Lab](https://www.marearts.com/blogs/open-lab/marearts-spritestride-walk-cycle-sprites-from-one-picture?utm_source=github&utm_medium=readme&utm_campaign=spritestride)** (free sign-up)
— drag a `.json` into ComfyUI: a new character from words, one walk frame
(image 1 = depth mannequin, image 2 = the character), and the cut-out. `spritestride workflows --out workflows/`
also writes them from your own server's node list.

## In Python

```python
from marearts_spritestride import ma_gait, ma_frame_phases, ma_walk_points, ma_depth
from marearts_spritestride.comfy import MaComfy
from marearts_spritestride.pipeline import MaRequest, ma_draw_cycle

req = MaRequest(reference="alice.png", out="out/alice", gait=ma_gait("neat"), look="a woman with a red bob, ...")
ma_draw_cycle(req, MaComfy("http://127.0.0.1:8188"))
```

## Tests

```bash
pip install -e ".[test]" && pytest -q test
```

They run on CPU: the gait (even steps, legs swap, feet on the ground), the mannequin and checks, and the whole
draw -> check -> lock -> sheet path against a fake ComfyUI server.

## Licence

Apache-2.0. Made at [MareArts Open Lab](https://www.marearts.com/blogs/open-lab). If you make something with it,
we would like to see it.

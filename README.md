# marearts-moim-pose

<p align="center">
  <img src="docs/walk-loops.gif" alt="Rin and Otto, each drawn from one sentence, walking and carrying a box in 12-frame loops" width="100%">
</p>

Walk-cycle sprites of one person from one picture. Give it a side view of a character; it draws the same
character walking (or carrying a box) in 12 frames that loop, checks every frame, locks height and head size
across the cycle, and writes a sprite sheet, GIFs and a small JSON a game can read.

It runs on an 8 GB GPU through ComfyUI with FLUX.2 [klein] 4B, and every model it uses is Apache-2.0 or MIT.

Formerly **marearts-spritestride** (renamed in 0.2.0): the `spritestride` command and `import marearts_spritestride`
still work.

**ComfyUI workflows: [download on MareArts Open Lab](https://www.marearts.com/blogs/open-lab/marearts-spritestride-walk-cycle-sprites-from-one-picture?utm_source=github&utm_medium=readme&utm_campaign=spritestride)**

## Why frames drift, and what fixes it

| Problem | Why | What moim-pose does |
|---|---|---|
| The character changes from frame to frame (sleeves, hem, colours) | Each frame is drawn on its own | An **edit** model: the character's own picture goes in as a reference latent, and the words name each piece of clothing |
| It walks "on the same foot" | In profile the two legs of a pose skeleton lie on one line, so the model cannot tell which is in front | A **depth mannequin** instead of a skeleton: far limbs dark, near limbs bright. Frames half a cycle apart must lead with **different legs** (the lighter leg in front in one, behind in the other); one drawn on the same foot is drawn again. In the finished frames the far leg is drawn darker, as sprite artists do |
| A third hand or foot appears, or the arms stand still while the legs walk | The model guesses where the far arm is, and given the same words for every frame it draws one hand before the body and one behind in all of them | Each frame is **told where its arms are** (hanging, near arm forward, near arm back). Measured checks against the mannequin (an arm swung past it fails), and an optional **vision-model judge**: it names each hand and counts them, says yes or no on a small crop **where the mannequin has the hand**, and counts the shoes on the feet. A failed frame is drawn again with another seed; one whose hand fails every seed (an arm half way back, where the model puts the hand back by the hip as in the picture) is drawn from its neighbour that passed |
| Height and head size wobble a few percent, and the feet slide | Drawn frames are never exactly the size of the pose | A **lock**: one master frame sets the head and the floor, the height follows the walk's own smooth wave, the legs are laid on the mannequin's legs |
| The sprite skates across the ground, or the arms move in jumps | A leg turning at an even rate moves the planted foot unevenly, and frames picked to make up for it come at uneven times | The standing leg is solved so the planted foot goes back the **same distance every frame** at even time steps (a body walks at a steady speed), and each arm swings on one smooth wave; that distance is written to the JSON |

<p align="center"><img src="docs/mannequin.png" alt="Two pose skeletons half a step apart look almost the same; two depth mannequins show the near leg in front, then behind" width="100%"></p>
<p align="center"><sub>Left: the pose skeleton at two steps half a cycle apart — almost the same shape. Right: the depth mannequin at the same steps — near limbs bright, far ones dark.</sub></p>

## Examples

Both characters were made from words only (`moim-pose person`), then walked and given a box, 12 frames a cycle.
Every frame passed the checks; the judge sent back third hands, hands left by the hip where the mannequin swung the
arm back, and three frames with a third shoe, and those were drawn again with other seeds. Two frames of Otto's walk,
his arm half way back, failed 12 seeds and passed drawn from the frame before or after. One pair of frames in
each cycle still leads with a faint leg (marked in the review sheet).

| | made with |
|---|---|
| **Rin** | `--gait lively --look "a young woman with a short black bob, a yellow raincoat, dark blue jeans, red sneakers"` |
| **Otto** | `--gait older --look "an older man with a short grey beard and a brown flat cap, a brown corduroy jacket over a cream shirt, olive trousers, brown leather boots"` |

The loops at the top are the four cycles. Rin's walk as the sprite sheet a game reads:

<p align="center"><img src="docs/rin-walk-sheet.png" alt="Rin's walk: the twelve frames of the sprite sheet, the far leg darker" width="70%"></p>

## Install

```bash
pip install git+https://github.com/MareArts/marearts-moim-pose            # Python 3.10+
pip install "marearts-moim-pose[judge] @ git+https://github.com/MareArts/marearts-moim-pose"   # + the judge
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

The reference picture: the whole character standing, side view, **facing right**, on a plain background. A picture
facing left is mirrored first: the judge says which way it faces, or without the judge the feet do (the shoes reach
past the ankles toward the toes); `--facing left|right` says it yourself. `moim-pose person` mirrors candidates
drawn facing left.

```bash
# no picture yet? candidates for a new character from words (pick one you like)
moim-pose person --out people/alice \
    --look "a woman with a red-orange bob and round glasses, a mustard knit sweater, rolled blue jeans, black sneakers"

# see the poses first (no GPU, no server)
moim-pose mannequin --out preview/ --gait lively

# draw a walk; describe the character in words too, it keeps clothes from drifting
moim-pose draw alice.png --out out/alice --gait neat \
    --look "a woman with a red-orange bob and round glasses, a mustard knit sweater, rolled blue jeans, black sneakers"

# carrying a box, with the judge on a local OpenAI-compatible server
moim-pose draw alice.png --out out/alice --kind carry --judge http://127.0.0.1:8080

# draw frames 2 and 6 again, then lock and sheet the whole cycle
moim-pose draw alice.png --out out/alice --only 2 6
moim-pose finish --out out/alice
```

| Option | Meaning |
|---|---|
| `--gait` | `neat relaxed lively upright calm graceful heavy older energetic loose`; or `--step --knee --arm --lean --elbow` |
| `--frames` | frames a cycle (default 12; 8 makes a smaller sheet but the arms move in bigger jumps) |
| `--fps` | GIF speed (default 12: one cycle a second at 12 frames) |
| `--head` | head width / height of the character (0.19 for a big anime head with hair); the mannequin's head follows it |
| `--bust` | how far the chest stands out in the mannequin (0.13 for a fuller figure); the silhouette comes from the depth |
| `--change` | the one thing to draw differently from the picture |
| `--height` | sprite height in px (default 256, 0 = full 832 x 1216 frame) |
| `--far-shade` | how bright the far leg is against the near one (default 0.55; 1 = legs as drawn) |
| `--facing` | `auto` (default), `right`, or `left` (the picture is mirrored to face right) |
| `--alternate` | draw again a frame that leads with the same leg as the frame half a cycle away (off by default: the leg tones fix it, new seeds rarely do) |

Out, in `--out`:

| File | |
|---|---|
| `walk_sheet.png` | sprite sheet, 4 frames a row (4 x 3 at 12 frames), transparent |
| `walk_sheet.json` | frame size, fps, `stride_px_per_frame`: move the sprite that far each frame and the feet stay put |
| `walk_loop.gif` · `walk_strip.gif` | the loop in place · walking across a strip at its own stride (sliding feet show here) |
| `walk_0.png` … | full-size locked frames |
| `walk_review.png` | every frame with its number, seed and checks; failed ones in red — look at this first |
| `walk_report.json` | per frame: seed, every seed tried, measured checks, leg lead, judge answer, why it failed, seconds |
| `raw/` | the drawn frames, masks and mannequins before the lock |

On an RTX 2080 Super (8 GB) a try takes about half a minute to a minute; a cycle of 12 about 10 minutes with retries.

## It is not 100 % — plan on a look

An image model draws every frame, and now and then it draws something wrong. Measured over 507 frames drawn on an
8 GB card:

| | |
|---|---|
| Frames that passed the automatic checks at the first seed | 84 % — the rest at the 2nd to 4th seed. With the 0.2.0 checks (hands where the mannequin has them, shoes) the 48 example frames: 69 % at the first seed, most of the rest by the 5th; two frames of the older man's walk, his arm half way back, took 10 |
| Walk cycles a person passed at the first look (one review of 12) | 4 of 12 — the others had a leg or a hand drawn oddly, a hat, beard or pocket that changed, a mouth open |
| What the checks catch | a third arm or leg, an arm swung past the mannequin, a carried box that turns or bobs; with the judge a third hand, a hand missing where the mannequin has it, a third shoe; two steps on the same foot (by the legs' light, at the contact frames) |
| What they do not catch yet | the same foot at mid-stance (legs crossed), legs of one flat colour with no light on them, an odd hand pose, small changes in clothes or face |

<p align="center"><img src="docs/rin-walk-review.png" alt="The review sheet: every frame with its number, seed and what was measured" width="100%"></p>
<p align="center"><sub>Rin's walk review sheet, 12 frames. Frame 2 is boxed: its lead leg reads faint against frame 8, half a cycle away. The seed line shows the frames the judge sent back (try 2 to 5).</sub></p>

So the tool gives you **drafts that are checked**, not finished art. After every cycle open `walk_review.png`: every
frame with its number, seed and what was measured, frames that failed the checks boxed in red, and what to look at
written on top. Draw the frames you do not like again by number; they get seeds they have not had, and the report
keeps every frame's history:

```bash
moim-pose draw alice.png --out out/alice --only 3 5      # frames 3 and 5 again, then lock and sheets again
moim-pose draw alice.png --out out/alice --only 3 --use-seed 7   # a new try was worse: back to seed 7
```

A new seed is not always better. Each frame's report row keeps every seed it has tried, so you can go back.

Most fixes that stick are words, not retries: name in `--look` the piece that changed (a beanie "pulled down over the
whole top of the head", "bare legs of smooth skin"), and the next draw keeps it.

## The graphs in ComfyUI

**Get the ready-made ComfyUI workflows on the [moim-pose page of MareArts Open Lab](https://www.marearts.com/blogs/open-lab/marearts-spritestride-walk-cycle-sprites-from-one-picture?utm_source=github&utm_medium=readme&utm_campaign=spritestride)** (free sign-up)
— drag a `.json` into ComfyUI: a new character from words, one walk frame
(image 1 = depth mannequin, image 2 = the character), and the cut-out. `moim-pose workflows --out workflows/`
also writes them from your own server's node list.

## In Python

```python
from marearts_moim_pose import ma_gait, ma_frame_phases, ma_walk_points, ma_depth
from marearts_moim_pose.comfy import MaComfy
from marearts_moim_pose.pipeline import MaRequest, ma_draw_cycle

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

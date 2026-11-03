"""spritestride — command line.

  spritestride mannequin --out preview/ --gait lively --kind carry      # the poses only, no server, CPU
  spritestride person --look "a woman with a red bob, a mustard sweater, blue jeans" --out people/alice   # no picture yet
  spritestride draw person.png --out out/alice --look "a woman with a red bob, a mustard sweater, blue jeans"
  spritestride draw person.png --out out/alice --kind carry --judge http://127.0.0.1:8080
  spritestride finish --out out/alice                                   # lock and sheet again from out/alice/raw
  spritestride workflows --out workflows/                               # the graphs as UI workflows for ComfyUI
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from . import __version__
from .gait import PRESETS, ma_frame_phases, ma_gait, ma_walk_points
from .mannequin import ma_depth, ma_skeleton


def _gait(args):
    g = ma_gait(args.gait)
    changes = {k: getattr(args, k) for k in ("step", "knee", "arm", "lean", "elbow") if getattr(args, k) is not None}
    return ma_gait(g, **changes)


def _common(p):
    p.add_argument("--kind", choices=("walk", "carry"), default="walk")
    p.add_argument("--gait", choices=sorted(PRESETS), default=None, help="a preset walk")
    for k in ("step", "knee", "arm", "lean", "elbow"):
        p.add_argument(f"--{k}", type=float, default=None)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--head", type=float, default=0.19, help="head width / height of the person")
    p.add_argument("--bust", type=float, default=0.0)


def cmd_mannequin(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = _gait(args)
    for i, t in enumerate(ma_frame_phases(args.frames, g)):
        pts = ma_walk_points(t, g, carry=args.kind == "carry")
        ma_depth(pts, head=args.head, box=args.kind == "carry", bust=args.bust).save(out / f"{args.kind}_{i}_depth.png")
        ma_skeleton(pts).save(out / f"{args.kind}_{i}_pose.png")
    print(out)


def _request(args):
    from .pipeline import MaRequest
    return MaRequest(reference=Path(args.reference) if getattr(args, "reference", None) else Path(),
                     out=Path(args.out), name=Path(args.out).name, kind=args.kind, look=getattr(args, "look", ""),
                     change=getattr(args, "change", ""), gait=_gait(args), frames=args.frames,
                     seed=getattr(args, "seed", 7), head=args.head, bust=args.bust,
                     height=None if args.height == 0 else args.height, fps=args.fps,
                     far_shade=args.far_shade, alternate=getattr(args, "alternate", False),
                     facing=getattr(args, "facing", "auto"))


def cmd_draw(args):
    from .comfy import MaComfy
    from .pipeline import ma_draw_cycle
    judge = None
    if args.judge:
        from .judge import MaHttpJudge, MaLocalJudge
        judge = MaHttpJudge(args.judge, args.judge_model) if args.judge.startswith("http") else MaLocalJudge(args.judge)
    res = ma_draw_cycle(_request(args), MaComfy(args.server), judge=judge, only=args.only, use_seed=args.use_seed)
    print(json.dumps(res["files"], indent=2))


def cmd_person(args):
    from .comfy import MaComfy
    from .pipeline import ma_new_person
    for p in ma_new_person(MaComfy(args.server), args.look, Path(args.out), args.count, args.seed):
        print(p)


def cmd_finish(args):
    from .pipeline import ma_finish
    print(json.dumps(ma_finish(_request(args)), indent=2))


def cmd_workflows(args):
    from .comfy import MaComfy, ma_cutout_graph, ma_klein_depth_graph, ma_klein_text_graph, ma_prompt, ma_ui_workflow
    info = MaComfy(args.server).object_info()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    graphs = {"spritestride_1_frame_klein_depth": ma_klein_depth_graph("depth.png", "person.png", ma_prompt()),
              "spritestride_2_cutout_birefnet": ma_cutout_graph("frame.png"),
              "spritestride_0_new_character_klein": ma_klein_text_graph("a young man with short black hair, a "
                                                                         "yellow raincoat, dark jeans, red sneakers")}
    for name, g in graphs.items():
        (out / f"{name}.json").write_text(json.dumps(ma_ui_workflow(g, info), indent=1) + "\n", encoding="utf-8")
        (out / f"{name}_api.json").write_text(json.dumps(g, indent=1) + "\n", encoding="utf-8")
        print(out / f"{name}.json")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spritestride", description="Walk-cycle sprites of one person from one picture.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("mannequin", help="write the depth mannequins and skeletons of a cycle (no server)")
    p.add_argument("--out", required=True)
    _common(p)
    p.set_defaults(fn=cmd_mannequin)

    for name, fn in (("draw", cmd_draw), ("finish", cmd_finish)):
        p = sub.add_parser(name, help="draw a cycle with ComfyUI" if name == "draw" else "lock and sheet raw frames")
        if name == "draw":
            p.add_argument("reference", help="the person standing, side view facing right")
            p.add_argument("--look", default="", help="the person in words: hair, clothes, shoes")
            p.add_argument("--change", default="", help="one thing to draw differently from the picture")
            p.add_argument("--seed", type=int, default=7)
            p.add_argument("--server", default="http://127.0.0.1:8188")
            p.add_argument("--judge", default="", help="an OpenAI-compatible URL, or a transformers model id")
            p.add_argument("--judge-model", default="local")
            p.add_argument("--only", type=int, nargs="*", default=None, help="draw only these frames again")
            p.add_argument("--use-seed", type=int, default=None,
                           help="with --only: this exact seed (go back to an earlier try; see the report's tried list)")
            p.add_argument("--facing", choices=("auto", "right", "left"), default="auto",
                           help="which way the picture faces (left: mirrored first); auto: the judge, else the feet")
            p.add_argument("--alternate", action="store_true",
                           help="draw again a frame that leads with the same leg as the frame half a cycle away")
        p.add_argument("--out", required=True)
        p.add_argument("--height", type=int, default=256, help="sprite height in px, 0 for full size")
        p.add_argument("--fps", type=int, default=8)
        p.add_argument("--far-shade", type=float, default=0.55,
                       help="far leg brightness against the near leg (1: legs as drawn)")
        _common(p)
        p.set_defaults(fn=fn)

    p = sub.add_parser("person", help="candidates for a new character from words (side view, facing right)")
    p.add_argument("--look", required=True, help="the character in words: hair, clothes, shoes")
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.set_defaults(fn=cmd_person)

    p = sub.add_parser("workflows", help="export the graphs as ComfyUI UI workflows (needs the server's node list)")
    p.add_argument("--out", required=True)
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.set_defaults(fn=cmd_workflows)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

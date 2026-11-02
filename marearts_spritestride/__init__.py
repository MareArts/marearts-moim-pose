"""marearts-spritestride: walk-cycle sprites of one person from one picture, with ComfyUI and FLUX.2 [klein]."""
__version__ = "0.1.0"

from .gait import PRESETS, MaGait, ma_frame_phases, ma_gait, ma_stride_px, ma_walk_points  # noqa: E402,F401
from .mannequin import ma_depth, ma_skeleton  # noqa: E402,F401

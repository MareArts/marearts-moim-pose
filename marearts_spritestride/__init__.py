"""The old name of marearts_moim_pose (renamed in 0.2.0). `import marearts_spritestride` and its modules still work and
are the new ones; the `spritestride` command too."""
import importlib
import sys

import marearts_moim_pose
from marearts_moim_pose import *  # noqa: F401,F403
from marearts_moim_pose import __version__  # noqa: F401

for _name in ("checks", "cli", "comfy", "gait", "judge", "lock", "mannequin", "pipeline", "sheet"):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"marearts_moim_pose.{_name}")

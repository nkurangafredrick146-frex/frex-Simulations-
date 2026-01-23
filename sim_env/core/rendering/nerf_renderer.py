import importlib.util
import os
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
_orig = os.path.join(str(_root), "NeRF_Integration.py")
if os.path.exists(_orig):
    spec = importlib.util.spec_from_file_location("sim_env._nerf", _orig)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    globals().update({k: v for k, v in mod.__dict__.items() if not k.startswith("_")})

__all__ = [n for n in globals().keys() if not n.startswith("_")]

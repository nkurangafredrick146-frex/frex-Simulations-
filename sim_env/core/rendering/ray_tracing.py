"""Ray tracing wrapper that loads original ray & path tracing implementation."""
import importlib.util
import os
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
_orig = os.path.join(str(_root), "ray_&_Path_Tracing.py")
if not os.path.exists(_orig):
    # fallback to similarly named file
    _orig = os.path.join(str(_root), "ray_&_Path_Tracing.py")

if os.path.exists(_orig):
    spec = importlib.util.spec_from_file_location("sim_env._ray_tracing", _orig)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    globals().update({k: v for k, v in mod.__dict__.items() if not k.startswith("_")})

__all__ = [n for n in globals().keys() if not n.startswith("_")]

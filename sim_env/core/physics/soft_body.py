"""Soft body physics wrapper (placeholder)."""
from pathlib import Path
import importlib.util
import os

_root = Path(__file__).resolve().parents[2]
_p = os.path.join(str(_root), "Physics_simulation_module.py")
if os.path.exists(_p):
    spec = importlib.util.spec_from_file_location("sim_env._physics_sb", _p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        from mod import *  # type: ignore
    except Exception:
        pass

__all__ = [n for n in globals().keys() if not n.startswith("_")]

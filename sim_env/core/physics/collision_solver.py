"""Collision solver wrapper (placeholder).

This module re-exports any collision-related helpers found in the original root modules.
"""
from pathlib import Path
import importlib.util
import os

_root = Path(__file__).resolve().parents[2]
_candidates = ["Physics_simulation_module.py", "physics_api.py", "particle_system.py"]
for cand in _candidates:
    _p = os.path.join(str(_root), cand)
    if os.path.exists(_p):
        spec = importlib.util.spec_from_file_location(f"sim_env._{cand}", _p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        break

try:
    from mod import *  # type: ignore
except Exception:
    pass

__all__ = [n for n in globals().keys() if not n.startswith("_")]

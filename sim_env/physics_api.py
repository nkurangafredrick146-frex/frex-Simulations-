"""Simple Physics API wrapper to allow lightweight imports.
This module tries to import the heavy `Physics_simulation_module` and exposes a minimal interface.
If the heavy module isn't available, provides informative stubs.
"""
from typing import Any
import importlib

_mod_name = ".Physics_simulation_module"

def _load_module():
    try:
        return importlib.import_module(f"sim_env{_mod_name}")
    except Exception as e:
        raise ImportError("Physics module cannot be imported: %s" % e)


def is_available() -> bool:
    try:
        _load_module()
        return True
    except Exception:
        return False


def create_engine(*args, **kwargs) -> Any:
    mod = _load_module()
    return getattr(mod, "PhysicsEngine")(*args, **kwargs)


def make_particle(position=(0, 0, 0), velocity=(0, 0, 0), mass=1.0, radius=0.1, **kwargs):
    mod = _load_module()
    return getattr(mod, "Particle")(position=position, velocity=velocity, mass=mass, radius=radius, **kwargs)


def get_settings(*args, **kwargs):
    mod = _load_module()
    return getattr(mod, "PhysicsSettings" )(*args, **kwargs)


__all__ = ["is_available", "create_engine", "make_particle", "get_settings"]

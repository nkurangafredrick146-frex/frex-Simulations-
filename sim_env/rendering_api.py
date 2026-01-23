"""Rendering API wrapper for lightweight imports.
Exposes core renderer creation functions without pulling heavy rendering subsystems during import time.
"""
from typing import Any
import importlib

_mod_name = ".Rendering_engine"

def _load_module():
    try:
        return importlib.import_module(f"sim_env{_mod_name}")
    except Exception as e:
        raise ImportError("Rendering module cannot be imported: %s" % e)


def is_available() -> bool:
    try:
        _load_module()
        return True
    except Exception:
        return False


def create_renderer(*args, **kwargs) -> Any:
    mod = _load_module()
    return getattr(mod, "Renderer")(*args, **kwargs)


def create_camera(*args, **kwargs) -> Any:
    mod = _load_module()
    return getattr(mod, "Camera")(*args, **kwargs)


__all__ = ["is_available", "create_renderer", "create_camera"]

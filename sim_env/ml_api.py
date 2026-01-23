"""ML API wrapper to provide safe imports and simple entrypoints for model training/inference."""
from typing import Any
import importlib

_mod_ml = ".ml_pipeline"
_mod_neural = ".neural_Physics"

def _load_ml():
    try:
        return importlib.import_module(f"sim_env{_mod_ml}")
    except Exception as e:
        raise ImportError("ML pipeline cannot be imported: %s" % e)

def _load_neural():
    try:
        return importlib.import_module(f"sim_env{_mod_neural}")
    except Exception as e:
        raise ImportError("neural_Physics cannot be imported: %s" % e)


def is_available() -> bool:
    try:
        _load_ml()
        _load_neural()
        return True
    except Exception:
        return False


def create_pipeline(*args, **kwargs) -> Any:
    mod = _load_ml()
    return getattr(mod, "MLPipeline")(*args, **kwargs)


def create_neural_engine(*args, **kwargs) -> Any:
    mod = _load_neural()
    return getattr(mod, "NeuralPhysicsEngine")(*args, **kwargs)


__all__ = ["is_available", "create_pipeline", "create_neural_engine"]

"""Quantum API wrapper to provide safe access to quantum simulation components."""
from typing import Any
import importlib

_mod_circuit = ".quantum_computing_Hybrid"
_mod_system = ".Quantum_Physics_simulations"

def _load_circuit():
    try:
        return importlib.import_module(f"sim_env{_mod_circuit}")
    except Exception as e:
        raise ImportError("Quantum circuit module cannot be imported: %s" % e)

def _load_system():
    try:
        return importlib.import_module(f"sim_env{_mod_system}")
    except Exception as e:
        raise ImportError("Quantum system module cannot be imported: %s" % e)


def is_available() -> bool:
    try:
        _load_circuit()
        _load_system()
        return True
    except Exception:
        return False


def create_circuit(*args, **kwargs) -> Any:
    mod = _load_circuit()
    return getattr(mod, "QuantumCircuit")(*args, **kwargs)


def create_system(*args, **kwargs) -> Any:
    mod = _load_system()
    return getattr(mod, "QuantumSystem")(*args, **kwargs)


__all__ = ["is_available", "create_circuit", "create_system"]

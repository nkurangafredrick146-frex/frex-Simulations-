"""
frex Simulations - sim_env module
Complete simulation environment with physics, rendering, ML, and quantum capabilities.
"""

"""Lightweight sim_env package initializer.

This package avoids importing heavy subsystems at import-time. Use the `*_api` wrappers
or import modules directly when you need full functionality.
"""

from importlib import import_module

__version__ = "0.1.0"

# Expose lightweight helpers and APIs
from .core_types import Vec3, ParticleSpec, SimulationConfig
from .physics_api import is_available as physics_available, create_engine as create_physics_engine
from .rendering_api import is_available as rendering_available, create_renderer
from .ml_api import is_available as ml_available, create_pipeline
from .quantum_api import is_available as quantum_available, create_circuit

__all__ = [
    "Vec3",
    "ParticleSpec",
    "SimulationConfig",
    "physics_available",
    "create_physics_engine",
    "rendering_available",
    "create_renderer",
    "ml_available",
    "create_pipeline",
    "quantum_available",
    "create_circuit",
]
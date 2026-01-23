"""Core lightweight types shared across modules.
Provides simple dataclasses and helpers so modules can interoperate without heavy imports.
"""
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

Vec3 = Tuple[float, float, float]

@dataclass
class ParticleSpec:
    position: Vec3 = (0.0, 0.0, 0.0)
    velocity: Vec3 = (0.0, 0.0, 0.0)
    mass: float = 1.0
    radius: float = 0.1
    properties: Optional[Dict[str, Any]] = None

@dataclass
class SimulationConfig:
    window_width: int = 1200
    window_height: int = 800
    max_particles: int = 10000
    physics_steps_per_frame: int = 1
    enable_audio: bool = True
    enable_ml: bool = False
    render_quality: str = "high"
    simulation_speed: float = 1.0


__all__ = ["Vec3", "ParticleSpec", "SimulationConfig"]

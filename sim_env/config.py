"""Configuration for frex Simulations environment variables."""

import os
from pathlib import Path
from typing import Optional

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Environment
ENV = os.getenv("ENV", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# Paths
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "output"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", PROJECT_ROOT / "logs"))

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Simulation settings
RESOLUTION = tuple(map(int, os.getenv("RESOLUTION", "1920,1080").split(",")))
FPS = int(os.getenv("FPS", "60"))
QUALITY = os.getenv("QUALITY", "high")

# Physics settings
PHYSICS_ENABLED = os.getenv("PHYSICS_ENABLED", "true").lower() == "true"
PHYSICS_TIMESTEP = float(os.getenv("PHYSICS_TIMESTEP", "0.016"))
GRAVITY = float(os.getenv("GRAVITY", "9.81"))

# Rendering settings
RENDERING_ENABLED = os.getenv("RENDERING_ENABLED", "true").lower() == "true"
RENDER_QUALITY = os.getenv("RENDER_QUALITY", "high")
ENABLE_SHADOWS = os.getenv("ENABLE_SHADOWS", "true").lower() == "true"
ENABLE_REFLECTIONS = os.getenv("ENABLE_REFLECTIONS", "true").lower() == "true"

# ML settings
ML_ENABLED = os.getenv("ML_ENABLED", "true").lower() == "true"
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models"))
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Quantum settings
QUANTUM_ENABLED = os.getenv("QUANTUM_ENABLED", "false").lower() == "true"
QUANTUM_BACKEND = os.getenv("QUANTUM_BACKEND", "qiskit")

# Web settings
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "5000"))
WEB_DEBUG = os.getenv("WEB_DEBUG", "true").lower() == "true" if ENV == "development" else False

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# GPU settings
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
GPU_ID = int(os.getenv("GPU_ID", "0"))

# Database settings (if needed)
DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL", None)

# API settings
API_KEY: Optional[str] = os.getenv("API_KEY", None)
API_SECRET: Optional[str] = os.getenv("API_SECRET", None)

# Export settings
EXPORT_FORMATS = os.getenv("EXPORT_FORMATS", "mp4,png,hdr").split(",")
EXPORT_QUALITY = os.getenv("EXPORT_QUALITY", "high")

__all__ = [
    "PROJECT_ROOT",
    "ENV",
    "DEBUG",
    "DATA_DIR",
    "OUTPUT_DIR",
    "LOGS_DIR",
    "RESOLUTION",
    "FPS",
    "QUALITY",
    "PHYSICS_ENABLED",
    "PHYSICS_TIMESTEP",
    "GRAVITY",
    "RENDERING_ENABLED",
    "RENDER_QUALITY",
    "ENABLE_SHADOWS",
    "ENABLE_REFLECTIONS",
    "ML_ENABLED",
    "MODEL_PATH",
    "QUANTUM_ENABLED",
    "QUANTUM_BACKEND",
    "WEB_HOST",
    "WEB_PORT",
    "WEB_DEBUG",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "USE_GPU",
    "GPU_ID",
    "DATABASE_URL",
    "API_KEY",
    "API_SECRET",
    "EXPORT_FORMATS",
    "EXPORT_QUALITY",
]

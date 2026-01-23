"""Project constants and configuration values."""

# Version
__version__ = "0.1.0"

# Physics constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
SPEED_OF_LIGHT = 299792458  # m/s
PLANCK_CONSTANT = 6.62607015e-34  # J·s
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K

# Default simulation parameters
DEFAULT_TIMESTEP = 0.016  # seconds (for 60 FPS)
DEFAULT_GRAVITY = 9.81  # m/s^2
DEFAULT_RESOLUTION = (1920, 1080)
DEFAULT_FPS = 60
DEFAULT_QUALITY = "high"

# Rendering parameters
RENDER_DISTANCES = {
    "ultra": 1000.0,
    "high": 500.0,
    "medium": 250.0,
    "low": 100.0,
}

# Physics simulation parameters
MAX_PARTICLES = 1000000
MAX_PARTICLE_SPEED = 1000.0  # m/s
DAMPING_COEFFICIENT = 0.99

# ML parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100

# Quantum parameters
NUM_QUBITS = 8
NUM_SHOTS = 1024

# File paths
DATA_FORMATS = ["hdf5", "npy", "csv"]
VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv"]
IMAGE_FORMATS = ["png", "jpg", "hdr", "exr"]

# Performance thresholds
MIN_FPS_WARNING = 30
TARGET_FPS = 60
MAX_CPU_USAGE = 80  # percentage
MAX_MEMORY_USAGE = 90  # percentage

# Error codes
ERROR_NO_GPU = 1001
ERROR_INVALID_CONFIG = 1002
ERROR_FILE_NOT_FOUND = 1003
ERROR_SIMULATION_FAILED = 1004

__all__ = [
    "__version__",
    "G",
    "SPEED_OF_LIGHT",
    "PLANCK_CONSTANT",
    "BOLTZMANN_CONSTANT",
    "DEFAULT_TIMESTEP",
    "DEFAULT_GRAVITY",
    "DEFAULT_RESOLUTION",
    "DEFAULT_FPS",
    "DEFAULT_QUALITY",
    "RENDER_DISTANCES",
    "MAX_PARTICLES",
    "MAX_PARTICLE_SPEED",
    "DAMPING_COEFFICIENT",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "EPOCHS",
    "NUM_QUBITS",
    "NUM_SHOTS",
    "DATA_FORMATS",
    "VIDEO_FORMATS",
    "IMAGE_FORMATS",
    "MIN_FPS_WARNING",
    "TARGET_FPS",
    "MAX_CPU_USAGE",
    "MAX_MEMORY_USAGE",
    "ERROR_NO_GPU",
    "ERROR_INVALID_CONFIG",
    "ERROR_FILE_NOT_FOUND",
    "ERROR_SIMULATION_FAILED",
]

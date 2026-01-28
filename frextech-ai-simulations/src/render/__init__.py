"""
FrexTech AI Simulations - Rendering Module
Provides 3D rendering capabilities for world visualization and interaction.
"""

from .cameras.camera_controller import (
    CameraController, CameraState, CameraIntrinsics,
    CameraProjection, CameraControlMode
)
from .cameras.camera_path import (
    CameraPath, Keyframe, InterpolationMode, EasingFunction
)
from .cameras.cinematic_recorder import (
    CinematicRecorder, RecordingSettings, RecordingFormat,
    RecordingQuality, RecordingMetadata
)

from .engines.raster_engine import (
    RasterEngine, RenderMode, ShadingModel,
    Vertex, Material, Mesh, Light
)
from .engines.ray_tracing_engine import (
    RayTracingEngine, RayTracingMode,
    Ray, HitRecord, RTMaterial, MaterialType,
    RTObject, RTSphere, RTMesh, RTTriangle,
    Camera as RTCamera, AABB, BVHNode
)
from .engines.webgl_engine import (
    WebGLEngine, WebGLRenderMode
)

# Shader utilities
from .shaders import (
    load_shader_source,
    compile_shader_program,
    create_shader_from_files
)

__version__ = "1.0.0"
__author__ = "FrexTech AI Simulations Team"
__all__ = [
    # Cameras
    "CameraController", "CameraState", "CameraIntrinsics",
    "CameraProjection", "CameraControlMode",
    "CameraPath", "Keyframe", "InterpolationMode", "EasingFunction",
    "CinematicRecorder", "RecordingSettings", "RecordingFormat",
    "RecordingQuality", "RecordingMetadata",
    
    # Rendering Engines
    "RasterEngine", "RenderMode", "ShadingModel",
    "Vertex", "Material", "Mesh", "Light",
    "RayTracingEngine", "RayTracingMode",
    "Ray", "HitRecord", "RTMaterial", "MaterialType",
    "RTObject", "RTSphere", "RTMesh", "RTTriangle",
    "RTCamera", "AABB", "BVNNode",
    "WebGLEngine", "WebGLRenderMode",
    
    # Version
    "__version__", "__author__"
]

# Export shader paths
import os
SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")
GAUSSIAN_SPLAT_SHADER = os.path.join(SHADER_DIR, "gaussian_splat.glsl")
MESH_SHADING_SHADER = os.path.join(SHADER_DIR, "mesh_shading.glsl")
POST_PROCESSING_SHADER = os.path.join(SHADER_DIR, "post_processing.glsl")

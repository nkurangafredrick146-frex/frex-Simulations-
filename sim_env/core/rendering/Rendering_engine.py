"""
Rendering engine core module
Provides a minimal, dependency-light rendering engine API used by the project.
This implementation favours clarity and import-time safety: it will run basic CPU-based
rasterization and a simple path-tracer stub without requiring full GPU/OpenGL.
"""
import os
import math
import time
from typing import Any, Dict, List, Tuple, Optional
import numpy as np


class RenderSettings:
    def __init__(self):
        self.width = 640
        self.height = 360
        self.samples = 4
        self.max_bounces = 2
        self.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.tone_map = True
        self.exposure = 1.0


class Scene:
    """Lightweight scene container"""
    def __init__(self):
        self.objects = []
        self.lights = []
        self.camera = None


class Camera:
    def __init__(self, position=(0, 0, 5), target=(0, 0, 0), fov_deg=60.0):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.fov = math.radians(fov_deg)


class SimpleMaterial:
    def __init__(self, color=(1, 1, 1), emission=(0, 0, 0)):
        self.color = np.array(color, dtype=np.float32)
        self.emission = np.array(emission, dtype=np.float32)


class Sphere:
    def __init__(self, center=(0, 0, 0), radius=1.0, material: SimpleMaterial = None):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self.material = material if material else SimpleMaterial()


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


class RenderingEngine:
    """A simple, portable rendering engine used for offline preview and testing."""

    def __init__(self, settings: RenderSettings = None):
        self.settings = settings if settings else RenderSettings()
        self.scene = Scene()
        self.last_frame = None
        self._frame_time = 0.0

    def load_scene(self, scene: Scene):
        self.scene = scene

    def render(self) -> np.ndarray:
        """Render a single frame and return an HxWx3 float32 image in [0,1]."""
        start = time.time()
        w = self.settings.width
        h = self.settings.height
        img = np.zeros((h, w, 3), dtype=np.float32)

        if self.scene.camera is None:
            # quick fallback camera
            self.scene.camera = Camera()
"""
Rendering engine core module
Provides a minimal, dependency-light rendering engine API used by the project.
This implementation favours clarity and import-time safety: it will run basic CPU-based
ray-tracing for previews and a simple helper API without requiring full GPU/OpenGL.
"""
import math
import time
from typing import Any, Dict, List, Tuple, Optional
import numpy as np


class RenderSettings:
    def __init__(self):
        self.width = 640
        self.height = 360
        self.samples = 4
        self.max_bounces = 2
        self.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.tone_map = True
        self.exposure = 1.0


class Scene:
    """Lightweight scene container"""
    def __init__(self):
        self.objects = []
        self.lights = []
        self.camera = None


class Camera:
    def __init__(self, position=(0, 0, 5), target=(0, 0, 0), fov_deg=60.0):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.fov = math.radians(fov_deg)


class SimpleMaterial:
    def __init__(self, color=(1, 1, 1), emission=(0, 0, 0)):
        self.color = np.array(color, dtype=np.float32)
        self.emission = np.array(emission, dtype=np.float32)


class Sphere:
    def __init__(self, center=(0, 0, 0), radius=1.0, material: SimpleMaterial = None):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self.material = material if material else SimpleMaterial()


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


class RenderingEngine:
    """A simple, portable rendering engine used for offline preview and testing."""

    def __init__(self, settings: RenderSettings = None):
        self.settings = settings if settings else RenderSettings()
        self.scene = Scene()
        self.last_frame = None
        self._frame_time = 0.0

    def load_scene(self, scene: Scene):
        self.scene = scene

    def render(self) -> np.ndarray:
        """Render a single frame and return an HxWx3 float32 image in [0,1]."""
        start = time.time()
        w = self.settings.width
        h = self.settings.height
        img = np.zeros((h, w, 3), dtype=np.float32)

        if self.scene.camera is None:
            # quick fallback camera
            self.scene.camera = Camera()

        # very simple ray marching for spheres in the scene
        aspect = float(w) / float(h)
        cam = self.scene.camera
        eye = cam.position
        forward = normalize(cam.target - cam.position)
        right = normalize(np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
        up = normalize(np.cross(right, forward))
        scale = math.tan(cam.fov * 0.5)

        for y in range(h):
            v = (1 - 2 * ((y + 0.5) / h)) * scale
            for x in range(w):
                u = (2 * ((x + 0.5) / w) - 1) * scale * aspect
                dir = normalize(forward + right * u + up * v)
                color = self._trace_ray(eye, dir)
                img[y, x, :] = color

        if self.settings.tone_map:
            img = self._apply_tonemap(img)

        self._frame_time = time.time() - start
        self.last_frame = img
        return img

    def _trace_ray(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        # simple nearest-sphere shading
        closest_t = float('inf')
        hit = None
        for obj in self.scene.objects:
            if isinstance(obj, Sphere):
                oc = origin - obj.center
                a = np.dot(direction, direction)
                b = 2.0 * np.dot(oc, direction)
                c = np.dot(oc, oc) - obj.radius * obj.radius
                disc = b * b - 4 * a * c
                if disc < 0:
                    continue
                sqrt_d = math.sqrt(disc)
                t0 = (-b - sqrt_d) / (2 * a)
                t1 = (-b + sqrt_d) / (2 * a)
                t = t0 if t0 > 1e-4 else (t1 if t1 > 1e-4 else None)
                if t and t < closest_t:
                    closest_t = t
                    hit = obj
        if hit is None:
            return self.settings.background_color

        # compute simple Lambertian shading using the first light
        position = origin + direction * closest_t
        normal = normalize(position - hit.center)
        light_dir = normalize(np.array([1.0, 1.0, 0.5], dtype=np.float32))
        lambert = max(0.0, np.dot(normal, light_dir))
        base = hit.material.color
        emission = hit.material.emission
        return np.clip(base * lambert + emission, 0.0, 1.0)

    def _apply_tonemap(self, img: np.ndarray) -> np.ndarray:
        # simple Reinhard tonemap + exposure
        img = 1.0 - np.exp(-img * self.settings.exposure)
        return np.clip(img, 0.0, 1.0)

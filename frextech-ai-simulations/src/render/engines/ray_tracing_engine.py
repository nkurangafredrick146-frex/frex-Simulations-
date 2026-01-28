"""
Ray tracing rendering engine with support for path tracing and global illumination.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import struct


class RayTracingMode(Enum):
    """Ray tracing modes."""
    WHITTED = "whitted"          # Classic Whitted-style ray tracing
    PATH_TRACING = "path_tracing"  # Path tracing with global illumination
    AMBIENT_OCCLUSION = "ao"     # Ambient occlusion only
    DIRECT_LIGHTING = "direct"   # Direct lighting only
    HYBRID = "hybrid"            # Hybrid raster/ray tracing


class MaterialType(Enum):
    """Material types for ray tracing."""
    DIFFUSE = "diffuse"
    SPECULAR = "specular"
    GLASS = "glass"
    METAL = "metal"
    EMISSIVE = "emissive"
    MIX = "mix"


@dataclass
class Ray:
    """Ray data structure."""
    origin: np.ndarray      # vec3
    direction: np.ndarray   # vec3 (normalized)
    time: float = 0.0       # Time for motion blur
    wavelength: float = 0.5  # Wavelength in microns (for dispersion)
    
    def __post_init__(self):
        """Normalize direction."""
        if not isinstance(self.origin, np.ndarray):
            self.origin = np.array(self.origin, dtype=np.float32)
        if not isinstance(self.direction, np.ndarray):
            self.direction = np.array(self.direction, dtype=np.float32)
        
        # Normalize direction
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm
    
    def at(self, t: float) -> np.ndarray:
        """Get point at distance t along ray.
        
        Args:
            t: Distance along ray
            
        Returns:
            Point at distance t
        """
        return self.origin + self.direction * t
    
    def reflect(self, normal: np.ndarray) -> "Ray":
        """Reflect ray off surface with normal.
        
        Args:
            normal: Surface normal (normalized)
            
        Returns:
            Reflected ray
        """
        reflected = self.direction - 2.0 * np.dot(self.direction, normal) * normal
        return Ray(origin=self.at(1e-4), direction=reflected, time=self.time)
    
    def refract(self, normal: np.ndarray, ior_ratio: float) -> Optional["Ray"]:
        """Refract ray through surface.
        
        Args:
            normal: Surface normal (normalized)
            ior_ratio: Ratio of refraction indices (n1/n2)
            
        Returns:
            Refracted ray, or None for total internal reflection
        """
        cos_theta = min(np.dot(-self.direction, normal), 1.0)
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
        
        # Total internal reflection
        if ior_ratio * sin_theta > 1.0:
            return None
        
        r_out_perp = ior_ratio * (self.direction + cos_theta * normal)
        r_out_parallel = -math.sqrt(abs(1.0 - np.dot(r_out_perp, r_out_perp))) * normal
        
        refracted = r_out_perp + r_out_parallel
        return Ray(origin=self.at(1e-4), direction=refracted, time=self.time)
    
    def transform(self, matrix: np.ndarray) -> "Ray":
        """Transform ray by matrix.
        
        Args:
            matrix: 4x4 transformation matrix
            
        Returns:
            Transformed ray
        """
        origin_hom = np.append(self.origin, 1.0)
        direction_hom = np.append(self.direction, 0.0)
        
        new_origin = matrix @ origin_hom
        new_direction = matrix @ direction_hom
        
        return Ray(
            origin=new_origin[:3] / new_origin[3],
            direction=new_direction[:3],
            time=self.time,
            wavelength=self.wavelength
        )


@dataclass
class HitRecord:
    """Hit record for ray-object intersection."""
    t: float = float('inf')         # Distance along ray
    point: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    front_face: bool = False        # Whether ray hit from outside
    uv: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))  # Texture coordinates
    material_id: int = 0
    object_id: int = 0
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        if not isinstance(self.point, np.ndarray):
            self.point = np.array(self.point, dtype=np.float32)
        if not isinstance(self.normal, np.ndarray):
            self.normal = np.array(self.normal, dtype=np.float32)
        if not isinstance(self.uv, np.ndarray):
            self.uv = np.array(self.uv, dtype=np.float32)
    
    def set_face_normal(self, ray: Ray, outward_normal: np.ndarray):
        """Set normal based on ray direction.
        
        Args:
            ray: Incident ray
            outward_normal: Outward-facing normal
        """
        self.front_face = np.dot(ray.direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal


@dataclass
class RTMaterial:
    """Ray tracing material."""
    name: str = "default"
    material_type: MaterialType = MaterialType.DIFFUSE
    albedo: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8], dtype=np.float32))
    emission: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    roughness: float = 0.0          # 0: perfect specular, 1: diffuse
    ior: float = 1.5                # Index of refraction
    metallic: float = 0.0           # 0: dielectric, 1: metal
    
    # Texture indices (for texture atlas)
    albedo_texture_id: int = -1
    normal_texture_id: int = -1
    roughness_texture_id: int = -1
    metallic_texture_id: int = -1
    
    # Procedural parameters
    procedural_type: str = ""       # "checker", "noise", "marble", "wood"
    procedural_scale: float = 1.0
    procedural_color1: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32))
    procedural_color2: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        if not isinstance(self.albedo, np.ndarray):
            self.albedo = np.array(self.albedo, dtype=np.float32)
        if not isinstance(self.emission, np.ndarray):
            self.emission = np.array(self.emission, dtype=np.float32)
        if not isinstance(self.procedural_color1, np.ndarray):
            self.procedural_color1 = np.array(self.procedural_color1, dtype=np.float32)
        if not isinstance(self.procedural_color2, np.ndarray):
            self.procedural_color2 = np.array(self.procedural_color2, dtype=np.float32)
    
    def evaluate(self, hit_record: HitRecord, textures: List[np.ndarray] = None) -> np.ndarray:
        """Evaluate material at hit point.
        
        Args:
            hit_record: Hit record
            textures: List of texture images
            
        Returns:
            Material color at hit point
        """
        if self.procedural_type:
            return self._evaluate_procedural(hit_record)
        elif self.albedo_texture_id >= 0 and textures is not None:
            return self._sample_texture(hit_record, textures)
        else:
            return self.albedo
    
    def _evaluate_procedural(self, hit_record: HitRecord) -> np.ndarray:
        """Evaluate procedural texture.
        
        Args:
            hit_record: Hit record
            
        Returns:
            Procedural texture color
        """
        if self.procedural_type == "checker":
            # Checkerboard pattern
            scale = self.procedural_scale
            sines = math.sin(scale * hit_record.point[0]) * \
                    math.sin(scale * hit_record.point[1]) * \
                    math.sin(scale * hit_record.point[2])
            if sines < 0:
                return self.procedural_color1
            else:
                return self.procedural_color2
        
        elif self.procedural_type == "noise":
            # Simple noise pattern
            import noise
            scale = self.procedural_scale
            x, y, z = hit_record.point * scale
            
            # Use Perlin noise
            n = noise.pnoise3(x, y, z, octaves=1, persistence=0.5, lacunarity=2.0)
            n = (n + 1.0) * 0.5  # Normalize to 0-1
            
            return self.procedural_color1 * n + self.procedural_color2 * (1.0 - n)
        
        elif self.procedural_type == "marble":
            # Marble pattern
            import noise
            scale = self.procedural_scale
            x, y, z = hit_record.point * scale
            
            n = noise.pnoise3(x, y, z, octaves=6, persistence=0.5, lacunarity=2.0)
            t = math.sin((x + n * 4.0) * math.pi * 2.0)
            t = (t + 1.0) * 0.5
            
            return self.procedural_color1 * t + self.procedural_color2 * (1.0 - t)
        
        else:
            return self.albedo
    
    def _sample_texture(self, hit_record: HitRecord, textures: List[np.ndarray]) -> np.ndarray:
        """Sample texture at UV coordinates.
        
        Args:
            hit_record: Hit record with UV coordinates
            textures: List of texture images
            
        Returns:
            Sampled texture color
        """
        if self.albedo_texture_id < 0 or self.albedo_texture_id >= len(textures):
            return self.albedo
        
        texture = textures[self.albedo_texture_id]
        u = hit_record.uv[0]
        v = hit_record.uv[1]
        
        # Wrap UV coordinates
        u = u - math.floor(u)
        v = v - math.floor(v)
        
        # Bilinear sampling
        width, height = texture.shape[1], texture.shape[0]
        x = u * (width - 1)
        y = (1.0 - v) * (height - 1)  # Flip V coordinate
        
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = min(x0 + 1, width - 1)
        y1 = min(y0 + 1, height - 1)
        
        tx = x - x0
        ty = y - y0
        
        # Sample four texels
        c00 = texture[y0, x0]
        c10 = texture[y0, x1]
        c01 = texture[y1, x0]
        c11 = texture[y1, x1]
        
        # Bilinear interpolation
        c0 = c00 * (1.0 - tx) + c10 * tx
        c1 = c01 * (1.0 - tx) + c11 * tx
        color = c0 * (1.0 - ty) + c1 * ty
        
        return color[:3]  # RGB only
    
    def scatter(self, ray: Ray, hit_record: HitRecord, rng: random.Random) -> Tuple[Ray, np.ndarray, bool]:
        """Scatter ray based on material properties.
        
        Args:
            ray: Incident ray
            hit_record: Hit record
            rng: Random number generator
            
        Returns:
            Tuple of (scattered_ray, attenuation, should_scatter)
        """
        if self.material_type == MaterialType.DIFFUSE:
            # Diffuse reflection
            scatter_direction = hit_record.normal + self._random_unit_vector(rng)
            
            # Catch degenerate scatter direction
            if np.allclose(scatter_direction, 0):
                scatter_direction = hit_record.normal
            
            scattered = Ray(origin=hit_record.point, direction=scatter_direction)
            return scattered, self.albedo, True
            
        elif self.material_type == MaterialType.SPECULAR:
            # Perfect specular reflection
            reflected = ray.reflect(hit_record.normal)
            return reflected, self.albedo, True
            
        elif self.material_type == MaterialType.METAL:
            # Metallic reflection with roughness
            reflected = ray.reflect(hit_record.normal)
            
            # Add roughness
            if self.roughness > 0:
                reflected.direction += self.roughness * self._random_in_unit_sphere(rng)
                reflected.direction = reflected.direction / np.linalg.norm(reflected.direction)
            
            scattered = Ray(origin=hit_record.point, direction=reflected.direction)
            return scattered, self.albedo, np.dot(scattered.direction, hit_record.normal) > 0
            
        elif self.material_type == MaterialType.GLASS:
            # Dielectric material (glass)
            refraction_ratio = 1.0 / self.ior if hit_record.front_face else self.ior
            
            # Schlick's approximation for reflectance
            cos_theta = min(np.dot(-ray.direction, hit_record.normal), 1.0)
            sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
            
            cannot_refract = refraction_ratio * sin_theta > 1.0
            reflectance = self._schlick(cos_theta, refraction_ratio)
            
            if cannot_refract or reflectance > rng.random():
                # Reflect
                reflected = ray.reflect(hit_record.normal)
                return reflected, np.ones(3), True
            else:
                # Refract
                refracted = ray.refract(hit_record.normal, refraction_ratio)
                if refracted is not None:
                    return refracted, np.ones(3), True
                else:
                    # Total internal reflection
                    reflected = ray.reflect(hit_record.normal)
                    return reflected, np.ones(3), True
            
        elif self.material_type == MaterialType.EMISSIVE:
            # Emissive material - doesn't scatter, just emits
            return ray, self.emission, False
            
        else:
            # Default: diffuse
            scatter_direction = hit_record.normal + self._random_unit_vector(rng)
            scattered = Ray(origin=hit_record.point, direction=scatter_direction)
            return scattered, self.albedo, True
    
    def _random_unit_vector(self, rng: random.Random) -> np.ndarray:
        """Generate random unit vector.
        
        Args:
            rng: Random number generator
            
        Returns:
            Random unit vector
        """
        while True:
            p = np.array([
                rng.uniform(-1, 1),
                rng.uniform(-1, 1),
                rng.uniform(-1, 1)
            ], dtype=np.float32)
            
            length_squared = np.dot(p, p)
            if 0.0001 < length_squared < 1.0:
                return p / math.sqrt(length_squared)
    
    def _random_in_unit_sphere(self, rng: random.Random) -> np.ndarray:
        """Generate random point in unit sphere.
        
        Args:
            rng: Random number generator
            
        Returns:
            Random point in unit sphere
        """
        while True:
            p = np.array([
                rng.uniform(-1, 1),
                rng.uniform(-1, 1),
                rng.uniform(-1, 1)
            ], dtype=np.float32)
            
            if np.dot(p, p) < 1.0:
                return p
    
    def _schlick(self, cosine: float, refraction_ratio: float) -> float:
        """Schlick's approximation for reflectance.
        
        Args:
            cosine: Cosine of incident angle
            refraction_ratio: Ratio of refraction indices
            
        Returns:
            Reflectance
        """
        r0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio)
        r0 = r0 * r0
        return r0 + (1.0 - r0) * math.pow(1.0 - cosine, 5)


class AABB:
    """Axis-Aligned Bounding Box for acceleration."""
    
    def __init__(self, min_point: np.ndarray = None, max_point: np.ndarray = None):
        """Initialize AABB.
        
        Args:
            min_point: Minimum corner
            max_point: Maximum corner
        """
        if min_point is None:
            self.min_point = np.array([float('inf'), float('inf'), float('inf')], dtype=np.float32)
        else:
            self.min_point = np.array(min_point, dtype=np.float32)
        
        if max_point is None:
            self.max_point = np.array([float('-inf'), float('-inf'), float('-inf')], dtype=np.float32)
        else:
            self.max_point = np.array(max_point, dtype=np.float32)
    
    def expand(self, point: np.ndarray):
        """Expand bounding box to include point.
        
        Args:
            point: Point to include
        """
        self.min_point = np.minimum(self.min_point, point)
        self.max_point = np.maximum(self.max_point, point)
    
    def expand_box(self, other: "AABB"):
        """Expand bounding box to include other box.
        
        Args:
            other: Other AABB
        """
        self.min_point = np.minimum(self.min_point, other.min_point)
        self.max_point = np.maximum(self.max_point, other.max_point)
    
    def centroid(self) -> np.ndarray:
        """Get box centroid.
        
        Returns:
            Centroid point
        """
        return (self.min_point + self.max_point) * 0.5
    
    def surface_area(self) -> float:
        """Calculate surface area.
        
        Returns:
            Surface area
        """
        size = self.max_point - self.min_point
        return 2.0 * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0])
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> bool:
        """Check if ray hits bounding box.
        
        Args:
            ray: Ray to test
            t_min: Minimum hit distance
            t_max: Maximum hit distance
            
        Returns:
            True if ray hits box
        """
        # Optimized slab method
        for i in range(3):
            inv_d = 1.0 / ray.direction[i] if ray.direction[i] != 0 else float('inf')
            t0 = (self.min_point[i] - ray.origin[i]) * inv_d
            t1 = (self.max_point[i] - ray.origin[i]) * inv_d
            
            if inv_d < 0:
                t0, t1 = t1, t0
            
            t_min = max(t_min, t0)
            t_max = min(t_max, t1)
            
            if t_max <= t_min:
                return False
        
        return True
    
    @staticmethod
    def surrounding_box(box0: "AABB", box1: "AABB") -> "AABB":
        """Create box surrounding two boxes.
        
        Args:
            box0: First box
            box1: Second box
            
        Returns:
            Surrounding box
        """
        min_point = np.minimum(box0.min_point, box1.min_point)
        max_point = np.maximum(box0.max_point, box1.max_point)
        return AABB(min_point, max_point)


class BVHNode:
    """Bounding Volume Hierarchy node."""
    
    def __init__(self, objects: List["RTObject"], start: int, end: int, 
                 time0: float = 0.0, time1: float = 1.0):
        """Initialize BVH node.
        
        Args:
            objects: List of objects
            start: Start index
            end: End index
            time0: Start time for motion blur
            time1: End time for motion blur
        """
        self.left: Optional[BVHNode] = None
        self.right: Optional[BVHNode] = None
        self.object: Optional[RTObject] = None
        self.box: AABB = AABB()
        
        # Build BVH
        axis = random.randint(0, 2)  # Random axis for splitting
        object_span = end - start
        
        if object_span == 1:
            # Leaf node with single object
            self.object = objects[start]
            self.box = self.object.bounding_box(time0, time1)
        elif object_span == 2:
            # Leaf node with two objects
            self.left = BVHNode(objects, start, start + 1, time0, time1)
            self.right = BVHNode(objects, start + 1, end, time0, time1)
            self.box = AABB.surrounding_box(self.left.box, self.right.box)
        else:
            # Sort objects along chosen axis
            objects_slice = objects[start:end]
            objects_slice.sort(key=lambda obj: obj.bounding_box(time0, time1).centroid()[axis])
            
            # Update original list
            for i, obj in enumerate(objects_slice):
                objects[start + i] = obj
            
            # Split at midpoint
            mid = start + object_span // 2
            self.left = BVHNode(objects, start, mid, time0, time1)
            self.right = BVHNode(objects, mid, end, time0, time1)
            self.box = AABB.surrounding_box(self.left.box, self.right.box)
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Check if ray hits any object in BVH.
        
        Args:
            ray: Ray to test
            t_min: Minimum hit distance
            t_max: Maximum hit distance
            
        Returns:
            Hit record if hit, None otherwise
        """
        if not self.box.hit(ray, t_min, t_max):
            return None
        
        # Leaf node
        if self.object is not None:
            return self.object.hit(ray, t_min, t_max)
        
        # Internal node
        hit_left = self.left.hit(ray, t_min, t_max) if self.left else None
        hit_right = self.right.hit(ray, t_min, t_max) if self.right else None
        
        if hit_left and hit_right:
            return hit_left if hit_left.t < hit_right.t else hit_right
        elif hit_left:
            return hit_left
        else:
            return hit_right


class RTObject:
    """Base class for ray tracing objects."""
    
    def __init__(self, material_id: int = 0, transform: np.ndarray = None):
        """Initialize object.
        
        Args:
            material_id: Material ID
            transform: Transformation matrix
        """
        self.material_id = material_id
        self.transform = transform if transform is not None else np.eye(4, dtype=np.float32)
        self.inverse_transform = np.linalg.inv(self.transform)
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Check if ray hits object.
        
        Args:
            ray: Ray to test
            t_min: Minimum hit distance
            t_max: Maximum hit distance
            
        Returns:
            Hit record if hit, None otherwise
        """
        raise NotImplementedError
    
    def bounding_box(self, time0: float, time1: float) -> AABB:
        """Get bounding box.
        
        Args:
            time0: Start time
            time1: End time
            
        Returns:
            Bounding box
        """
        raise NotImplementedError


class RTSphere(RTObject):
    """Sphere object for ray tracing."""
    
    def __init__(self, center: np.ndarray, radius: float, 
                 material_id: int = 0, transform: np.ndarray = None):
        """Initialize sphere.
        
        Args:
            center: Sphere center
            radius: Sphere radius
            material_id: Material ID
            transform: Transformation matrix
        """
        super().__init__(material_id, transform)
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Check if ray hits sphere.
        
        Args:
            ray: Ray to test
            t_min: Minimum hit distance
            t_max: Maximum hit distance
            
        Returns:
            Hit record if hit, None otherwise
        """
        # Transform ray to object space
        ray_local = ray.transform(self.inverse_transform)
        
        oc = ray_local.origin - self.center
        a = np.dot(ray_local.direction, ray_local.direction)
        half_b = np.dot(oc, ray_local.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        
        discriminant = half_b * half_b - a * c
        
        if discriminant < 0:
            return None
        
        sqrt_d = math.sqrt(discriminant)
        
        # Find nearest root in acceptable range
        root = (-half_b - sqrt_d) / a
        if root < t_min or t_max < root:
            root = (-half_b + sqrt_d) / a
            if root < t_min or t_max < root:
                return None
        
        hit_record = HitRecord()
        hit_record.t = root
        hit_record.point = ray_local.at(root)
        
        # Calculate normal
        outward_normal = (hit_record.point - self.center) / self.radius
        hit_record.set_face_normal(ray_local, outward_normal)
        
        # Calculate UV coordinates (spherical mapping)
        phi = math.atan2(hit_record.point[2], hit_record.point[0])
        theta = math.asin(hit_record.point[1] / self.radius)
        
        hit_record.uv = np.array([
            0.5 + phi / (2 * math.pi),
            0.5 + theta / math.pi
        ], dtype=np.float32)
        
        hit_record.material_id = self.material_id
        
        # Transform back to world space
        point_hom = np.append(hit_record.point, 1.0)
        normal_hom = np.append(outward_normal, 0.0)
        
        hit_record.point = (self.transform @ point_hom)[:3]
        hit_record.normal = (self.transform @ normal_hom)[:3]
        hit_record.normal = hit_record.normal / np.linalg.norm(hit_record.normal)
        
        return hit_record
    
    def bounding_box(self, time0: float, time1: float) -> AABB:
        """Get sphere bounding box.
        
        Args:
            time0: Start time
            time1: End time
            
        Returns:
            Bounding box
        """
        # Transform center
        center_hom = np.append(self.center, 1.0)
        center_world = (self.transform @ center_hom)[:3]
        
        # Account for radius
        radius_vec = np.array([self.radius, self.radius, self.radius], dtype=np.float32)
        
        return AABB(center_world - radius_vec, center_world + radius_vec)


class RTMesh(RTObject):
    """Triangle mesh object for ray tracing."""
    
    def __init__(self, vertices: List[np.ndarray], indices: List[int],
                 normals: Optional[List[np.ndarray]] = None,
                 uvs: Optional[List[np.ndarray]] = None,
                 material_id: int = 0, transform: np.ndarray = None):
        """Initialize mesh.
        
        Args:
            vertices: List of vertex positions
            indices: Triangle indices (triplets)
            normals: Optional vertex normals
            uvs: Optional texture coordinates
            material_id: Material ID
            transform: Transformation matrix
        """
        super().__init__(material_id, transform)
        self.vertices = [np.array(v, dtype=np.float32) for v in vertices]
        self.indices = indices
        self.normals = [np.array(n, dtype=np.float32) for n in normals] if normals else None
        self.uvs = [np.array(uv, dtype=np.float32) for uv in uvs] if uvs else None
        
        # Build acceleration structure
        self.triangles: List["RTTriangle"] = []
        self.bvh: Optional[BVHNode] = None
        
        # Create triangles
        for i in range(0, len(indices), 3):
            if i + 2 >= len(indices):
                break
            
            v0_idx = indices[i]
            v1_idx = indices[i + 1]
            v2_idx = indices[i + 2]
            
            triangle = RTTriangle(
                self.vertices[v0_idx],
                self.vertices[v1_idx],
                self.vertices[v2_idx],
                material_id,
                self.transform
            )
            
            # Add normals if available
            if self.normals and v0_idx < len(self.normals):
                triangle.normals = [
                    self.normals[v0_idx],
                    self.normals[v1_idx],
                    self.normals[v2_idx]
                ]
            
            # Add UVs if available
            if self.uvs and v0_idx < len(self.uvs):
                triangle.uvs = [
                    self.uvs[v0_idx],
                    self.uvs[v1_idx],
                    self.uvs[v2_idx]
                ]
            
            self.triangles.append(triangle)
        
        # Build BVH
        if self.triangles:
            self.bvh = BVHNode(self.triangles, 0, len(self.triangles))
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Check if ray hits mesh.
        
        Args:
            ray: Ray to test
            t_min: Minimum hit distance
            t_max: Maximum hit distance
            
        Returns:
            Hit record if hit, None otherwise
        """
        if self.bvh:
            return self.bvh.hit(ray, t_min, t_max)
        else:
            # Brute force if no BVH
            closest_hit = None
            closest_t = t_max
            
            for triangle in self.triangles:
                hit = triangle.hit(ray, t_min, closest_t)
                if hit and hit.t < closest_t:
                    closest_hit = hit
                    closest_t = hit.t
            
            return closest_hit
    
    def bounding_box(self, time0: float, time1: float) -> AABB:
        """Get mesh bounding box.
        
        Args:
            time0: Start time
            time1: End time
            
        Returns:
            Bounding box
        """
        if self.bvh:
            return self.bvh.box
        else:
            box = AABB()
            for triangle in self.triangles:
                box.expand_box(triangle.bounding_box(time0, time1))
            return box


class RTTriangle(RTObject):
    """Triangle object for ray tracing."""
    
    def __init__(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                 material_id: int = 0, transform: np.ndarray = None,
                 normals: List[np.ndarray] = None, uvs: List[np.ndarray] = None):
        """Initialize triangle.
        
        Args:
            v0, v1, v2: Triangle vertices
            material_id: Material ID
            transform: Transformation matrix
            normals: Optional vertex normals
            uvs: Optional texture coordinates
        """
        super().__init__(material_id, transform)
        self.v0 = np.array(v0, dtype=np.float32)
        self.v1 = np.array(v1, dtype=np.float32)
        self.v2 = np.array(v2, dtype=np.float32)
        self.normals = normals
        self.uvs = uvs
        
        # Precompute edges and normal
        self.edge1 = self.v1 - self.v0
        self.edge2 = self.v2 - self.v0
        self.normal = np.cross(self.edge1, self.edge2)
        self.normal_length = np.linalg.norm(self.normal)
        
        if self.normal_length > 0:
            self.normal = self.normal / self.normal_length
        
        # Transform vertices
        if transform is not None:
            v0_hom = np.append(self.v0, 1.0)
            v1_hom = np.append(self.v1, 1.0)
            v2_hom = np.append(self.v2, 1.0)
            
            self.v0 = (transform @ v0_hom)[:3]
            self.v1 = (transform @ v1_hom)[:3]
            self.v2 = (transform @ v2_hom)[:3]
            
            # Recompute edges and normal
            self.edge1 = self.v1 - self.v0
            self.edge2 = self.v2 - self.v0
            self.normal = np.cross(self.edge1, self.edge2)
            self.normal_length = np.linalg.norm(self.normal)
            
            if self.normal_length > 0:
                self.normal = self.normal / self.normal_length
    
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Check if ray hits triangle using Möller-Trumbore algorithm.
        
        Args:
            ray: Ray to test
            t_min: Minimum hit distance
            t_max: Maximum hit distance
            
        Returns:
            Hit record if hit, None otherwise
        """
        # Möller-Trumbore algorithm
        h = np.cross(ray.direction, self.edge2)
        a = np.dot(self.edge1, h)
        
        # Ray parallel to triangle
        if abs(a) < 1e-8:
            return None
        
        f = 1.0 / a
        s = ray.origin - self.v0
        u = f * np.dot(s, h)
        
        if u < 0.0 or u > 1.0:
            return None
        
        q = np.cross(s, self.edge1)
        v = f * np.dot(ray.direction, q)
        
        if v < 0.0 or u + v > 1.0:
            return None
        
        t = f * np.dot(self.edge2, q)
        
        if t < t_min or t > t_max:
            return None
        
        # Intersection found
        hit_record = HitRecord()
        hit_record.t = t
        hit_record.point = ray.at(t)
        hit_record.set_face_normal(ray, self.normal)
        hit_record.material_id = self.material_id
        
        # Calculate barycentric coordinates for interpolation
        w = 1.0 - u - v
        
        # Interpolate UV coordinates if available
        if self.uvs is not None and len(self.uvs) == 3:
            hit_record.uv = w * self.uvs[0] + u * self.uvs[1] + v * self.uvs[2]
        else:
            # Default UVs based on barycentric coordinates
            hit_record.uv = np.array([u, v], dtype=np.float32)
        
        # Interpolate normals if available
        if self.normals is not None and len(self.normals) == 3:
            interp_normal = w * self.normals[0] + u * self.normals[1] + v * self.normals[2]
            hit_record.normal = interp_normal / np.linalg.norm(interp_normal)
        
        return hit_record
    
    def bounding_box(self, time0: float, time1: float) -> AABB:
        """Get triangle bounding box.
        
        Args:
            time0: Start time
            time1: End time
            
        Returns:
            Bounding box
        """
        box = AABB()
        box.expand(self.v0)
        box.expand(self.v1)
        box.expand(self.v2)
        return box


class Camera:
    """Camera for ray tracing."""
    
    def __init__(self, look_from: np.ndarray, look_at: np.ndarray, 
                 vup: np.ndarray, vfov: float, aspect_ratio: float,
                 aperture: float = 0.0, focus_dist: float = 1.0,
                 time0: float = 0.0, time1: float = 1.0):
        """Initialize camera.
        
        Args:
            look_from: Camera position
            look_at: Look-at point
            vup: Up vector
            vfov: Vertical field of view in degrees
            aspect_ratio: Image aspect ratio
            aperture: Lens aperture (0 = pinhole)
            focus_dist: Focus distance
            time0: Shutter open time
            time1: Shutter close time
        """
        self.look_from = np.array(look_from, dtype=np.float32)
        self.look_at = np.array(look_at, dtype=np.float32)
        self.vup = np.array(vup, dtype=np.float32)
        self.vfov = vfov
        self.aspect_ratio = aspect_ratio
        self.aperture = aperture
        self.focus_dist = focus_dist
        self.time0 = time0
        self.time1 = time1
        
        # Calculate camera basis
        theta = math.radians(vfov)
        h = math.tan(theta / 2)
        self.viewport_height = 2.0 * h
        self.viewport_width = aspect_ratio * self.viewport_height
        
        self.w = (look_from - look_at) / np.linalg.norm(look_from - look_at)
        self.u = np.cross(vup, self.w) / np.linalg.norm(np.cross(vup, self.w))
        self.v = np.cross(self.w, self.u)
        
        # Calculate viewport and camera parameters
        self.horizontal = self.focus_dist * self.viewport_width * self.u
        self.vertical = self.focus_dist * self.viewport_height * self.v
        self.lower_left_corner = (
            self.look_from - 
            self.horizontal / 2 - 
            self.vertical / 2 - 
            self.focus_dist * self.w
        )
        
        self.lens_radius = aperture / 2
    
    def get_ray(self, s: float, t: float, rng: random.Random) -> Ray:
        """Generate ray for given image coordinates.
        
        Args:
            s: Horizontal coordinate (0-1)
            t: Vertical coordinate (0-1)
            rng: Random number generator
            
        Returns:
            Ray for pixel
        """
        # Depth of field
        if self.lens_radius > 0:
            rd = self.lens_radius * self._random_in_unit_disk(rng)
            offset = self.u * rd[0] + self.v * rd[1]
        else:
            offset = np.zeros(3, dtype=np.float32)
        
        origin = self.look_from + offset
        direction = (
            self.lower_left_corner + 
            s * self.horizontal + 
            t * self.vertical - 
            origin
        )
        
        time = self.time0 + rng.random() * (self.time1 - self.time0)
        
        return Ray(origin=origin, direction=direction, time=time)
    
    def _random_in_unit_disk(self, rng: random.Random) -> np.ndarray:
        """Generate random point in unit disk.
        
        Args:
            rng: Random number generator
            
        Returns:
            Random point in unit disk
        """
        while True:
            p = np.array([
                rng.uniform(-1, 1),
                rng.uniform(-1, 1),
                0.0
            ], dtype=np.float32)
            
            if np.dot(p, p) < 1.0:
                return p


class RayTracingEngine:
    """Main ray tracing engine."""
    
    def __init__(self, width: int = 800, height: int = 600,
                 mode: RayTracingMode = RayTracingMode.PATH_TRACING):
        """Initialize ray tracing engine.
        
        Args:
            width: Image width
            height: Image height
            mode: Ray tracing mode
        """
        self.width = width
        self.height = height
        self.mode = mode
        
        # Scene data
        self.objects: List[RTObject] = []
        self.materials: List[RTMaterial] = []
        self.textures: List[np.ndarray] = []
        self.lights: List[RTObject] = []  # Emissive objects
        
        # Acceleration structure
        self.bvh: Optional[BVHNode] = None
        
        # Camera
        self.camera: Optional[Camera] = None
        
        # Rendering parameters
        self.samples_per_pixel = 16
        self.max_depth = 8
        self.russian_roulette_depth = 5
        self.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.enable_denoising = True
        self.enable_adaptive_sampling = True
        
        # Progress tracking
        self.progress_callback: Optional[Callable[[float], None]] = None
        
        # Threading
        self.num_threads = max(1, (os.cpu_count() or 4) - 1)
        
        # Image buffer
        self.image_buffer = np.zeros((height, width, 3), dtype=np.float32)
        self.sample_count_buffer = np.zeros((height, width), dtype=np.int32)
        
    def add_object(self, obj: RTObject):
        """Add object to scene.
        
        Args:
            obj: Object to add
        """
        self.objects.append(obj)
        
        # If object is emissive, add to lights
        if obj.material_id < len(self.materials):
            material = self.materials[obj.material_id]
            if material.material_type == MaterialType.EMISSIVE:
                self.lights.append(obj)
    
    def add_material(self, material: RTMaterial):
        """Add material to scene.
        
        Args:
            material: Material to add
        """
        self.materials.append(material)
    
    def add_texture(self, texture: np.ndarray):
        """Add texture to scene.
        
        Args:
            texture: Texture image (HxWx3 or HxWx4)
        """
        self.textures.append(texture)
    
    def set_camera(self, camera: Camera):
        """Set camera.
        
        Args:
            camera: Camera to use
        """
        self.camera = camera
    
    def build_acceleration_structure(self):
        """Build acceleration structure (BVH)."""
        if self.objects:
            self.bvh = BVHNode(self.objects, 0, len(self.objects))
        else:
            self.bvh = None
    
    def ray_color(self, ray: Ray, depth: int, rng: random.Random) -> np.ndarray:
        """Calculate color for ray.
        
        Args:
            ray: Ray to trace
            depth: Current recursion depth
            rng: Random number generator
            
        Returns:
            Ray color
        """
        if depth <= 0:
            return np.zeros(3, dtype=np.float32)
        
        # Find closest hit
        hit_record = self._hit_world(ray, 0.001, float('inf'))
        
        if hit_record is None:
            # Background
            if self.mode == RayTracingMode.PATH_TRACING:
                # Sky gradient
                unit_direction = ray.direction / np.linalg.norm(ray.direction)
                t = 0.5 * (unit_direction[1] + 1.0)
                return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])
            else:
                return self.background_color
        
        # Get material
        material = self.materials[hit_record.material_id]
        
        # Emissive material
        if material.material_type == MaterialType.EMISSIVE:
            return material.emission
        
        # Scatter ray
        scattered, attenuation, should_scatter = material.scatter(ray, hit_record, rng)
        
        if not should_scatter:
            return np.zeros(3, dtype=np.float32)
        
        if self.mode == RayTracingMode.PATH_TRACING:
            # Path tracing with multiple importance sampling
            color = np.zeros(3, dtype=np.float32)
            
            # Direct lighting (next event estimation)
            if self.lights and depth < self.max_depth:
                light_color = self._sample_lights(hit_record, rng)
                color += attenuation * light_color
            
            # Russian roulette for path termination
            if depth < self.russian_roulette_depth:
                # Continue path
                color += attenuation * self.ray_color(scattered, depth - 1, rng)
            else:
                # Russian roulette
                max_component = max(attenuation[0], attenuation[1], attenuation[2])
                if rng.random() < max_component:
                    color += (attenuation / max_component) * self.ray_color(scattered, depth - 1, rng)
            
            return color
            
        elif self.mode == RayTracingMode.WHITTED:
            # Classic Whitted-style ray tracing
            color = np.zeros(3, dtype=np.float32)
            
            # Ambient term
            ambient = 0.1 * material.albedo
            color += ambient
            
            # Recursive reflection/refraction
            color += attenuation * self.ray_color(scattered, depth - 1, rng)
            
            return color
            
        else:
            # Default: simple scattering
            return attenuation * self.ray_color(scattered, depth - 1, rng)
    
    def _hit_world(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        """Find closest hit in world.
        
        Args:
            ray: Ray to test
            t_min: Minimum hit distance
            t_max: Maximum hit distance
            
        Returns:
            Closest hit record, or None
        """
        if self.bvh:
            return self.bvh.hit(ray, t_min, t_max)
        else:
            # Brute force
            closest_hit = None
            closest_t = t_max
            
            for obj in self.objects:
                hit = obj.hit(ray, t_min, closest_t)
                if hit and hit.t < closest_t:
                    closest_hit = hit
                    closest_t = hit.t
            
            return closest_hit
    
    def _sample_lights(self, hit_record: HitRecord, rng: random.Random) -> np.ndarray:
        """Sample lights for direct lighting.
        
        Args:
            hit_record: Surface hit record
            rng: Random number generator
            
        Returns:
            Direct lighting contribution
        """
        if not self.lights:
            return np.zeros(3, dtype=np.float32)
        
        # Randomly select a light
        light_idx = rng.randint(0, len(self.lights))
        light = self.lights[light_idx]
        
        # Sample point on light
        light_sample = self._sample_light(light, rng)
        if light_sample is None:
            return np.zeros(3, dtype=np.float32)
        
        light_point, light_normal, light_pdf = light_sample
        
        # Check visibility
        to_light = light_point - hit_record.point
        distance = np.linalg.norm(to_light)
        direction = to_light / distance
        
        shadow_ray = Ray(origin=hit_record.point, direction=direction)
        
        # Offset origin to avoid self-intersection
        shadow_ray.origin += hit_record.normal * 0.001
        
        # Check if light is visible
        shadow_hit = self._hit_world(shadow_ray, 0.001, distance - 0.001)
        if shadow_hit is not None:
            return np.zeros(3, dtype=np.float32)
        
        # Calculate lighting
        material = self.materials[hit_record.material_id]
        light_material = self.materials[light.material_id]
        
        # Cosine terms
        cos_theta = max(0.0, np.dot(hit_record.normal, direction))
        cos_theta_prime = max(0.0, np.dot(light_normal, -direction))
        
        if cos_theta <= 0 or cos_theta_prime <= 0:
            return np.zeros(3, dtype=np.float32)
        
        # BRDF (simplified Lambertian)
        brdf = material.albedo / math.pi
        
        # Light contribution
        geometry_term = (cos_theta * cos_theta_prime) / (distance * distance)
        light_power = light_material.emission * light_material.albedo
        
        # Multiple importance sampling weight
        weight = self._balance_heuristic(1, light_pdf, 1, self._brdf_pdf(material, hit_record, direction))
        
        return brdf * light_power * geometry_term * weight / light_pdf
    
    def _sample_light(self, light: RTObject, rng: random.Random) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Sample point on light surface.
        
        Args:
            light: Light object
            rng: Random number generator
            
        Returns:
            Tuple of (point, normal, pdf), or None
        """
        if isinstance(light, RTSphere):
            # Sample sphere uniformly
            center = light.center
            radius = light.radius
            
            # Generate random point on sphere
            z = rng.uniform(-1, 1)
            phi = rng.uniform(0, 2 * math.pi)
            
            sin_theta = math.sqrt(1.0 - z * z)
            x = sin_theta * math.cos(phi)
            y = sin_theta * math.sin(phi)
            
            point = center + radius * np.array([x, y, z], dtype=np.float32)
            normal = (point - center) / radius
            
            # PDF for uniform sphere sampling
            pdf = 1.0 / (4 * math.pi * radius * radius)
            
            return point, normal, pdf
        
        elif isinstance(light, RTTriangle):
            # Sample triangle uniformly
            v0, v1, v2 = light.v0, light.v1, light.v2
            
            # Barycentric coordinates
            u = rng.random()
            v = rng.random()
            
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            
            w = 1 - u - v
            
            point = w * v0 + u * v1 + v * v2
            normal = light.normal
            
            # PDF for uniform triangle sampling
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            pdf = 1.0 / area if area > 0 else 0.0
            
            return point, normal, pdf
        
        return None
    
    def _brdf_pdf(self, material: RTMaterial, hit_record: HitRecord, 
                 direction: np.ndarray) -> float:
        """Calculate PDF for BRDF sampling.
        
        Args:
            material: Material
            hit_record: Surface hit
            direction: Sampling direction
            
        Returns:
            PDF value
        """
        if material.material_type == MaterialType.DIFFUSE:
            # Cosine-weighted hemisphere sampling
            cos_theta = np.dot(hit_record.normal, direction)
            return max(0.0, cos_theta) / math.pi
        else:
            # Default uniform sampling
            return 1.0 / (2 * math.pi)
    
    def _balance_heuristic(self, nf: int, f_pdf: float, ng: int, g_pdf: float) -> float:
        """Balance heuristic for multiple importance sampling.
        
        Args:
            nf: Number of samples from f
            f_pdf: PDF from f
            ng: Number of samples from g
            g_pdf: PDF from g
            
        Returns:
            Weight
        """
        f = nf * f_pdf
        g = ng * g_pdf
        return f / (f + g) if f + g > 0 else 0.0
    
    def render_tile(self, x0: int, x1: int, y0: int, y1: int, 
                    samples: int, seed: int = 0) -> np.ndarray:
        """Render tile of image.
        
        Args:
            x0, x1: X range
            y0, y1: Y range
            samples: Samples per pixel
            seed: Random seed
            
        Returns:
            Rendered tile
        """
        rng = random.Random(seed)
        tile = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
        
        for y in range(y0, y1):
            for x in range(x0, x1):
                pixel_color = np.zeros(3, dtype=np.float32)
                
                # Anti-aliasing
                for s in range(samples):
                    u = (x + rng.random()) / (self.width - 1)
                    v = (y + rng.random()) / (self.height - 1)
                    
                    ray = self.camera.get_ray(u, v, rng)
                    pixel_color += self.ray_color(ray, self.max_depth, rng)
                
                # Average samples
                tile[y - y0, x - x0] = pixel_color / samples
        
        return tile
    
    def render(self) -> np.ndarray:
        """Render complete image.
        
        Returns:
            Rendered image (HxWx3)
        """
        if self.camera is None:
            raise ValueError("Camera not set")
        
        if not self.bvh:
            self.build_acceleration_structure()
        
        print(f"Starting rendering: {self.width}x{self.height}")
        print(f"Mode: {self.mode.value}")
        print(f"Samples per pixel: {self.samples_per_pixel}")
        print(f"Max depth: {self.max_depth}")
        print(f"Threads: {self.num_threads}")
        
        # Initialize image
        image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        # Divide image into tiles
        tile_size = 32
        tiles = []
        
        for y in range(0, self.height, tile_size):
            for x in range(0, self.width, tile_size):
                x1 = min(x + tile_size, self.width)
                y1 = min(y + tile_size, self.height)
                tiles.append((x, x1, y, y1))
        
        total_tiles = len(tiles)
        
        # Render tiles in parallel
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            
            for i, (x0, x1, y0, y1) in enumerate(tiles):
                seed = i * 12345  # Deterministic seeds
                future = executor.submit(
                    self.render_tile, x0, x1, y0, y1, 
                    self.samples_per_pixel, seed
                )
                futures.append((future, x0, x1, y0, y1))
            
            # Collect results
            completed = 0
            for future, x0, x1, y0, y1 in futures:
                tile = future.result()
                image[y0:y1, x0:x1] = tile
                
                completed += 1
                progress = completed / total_tiles
                
                if self.progress_callback:
                    self.progress_callback(progress)
                
                print(f"Progress: {progress:.1%}", end='\r')
        
        print("\nRendering complete!")
        
        # Apply tone mapping and gamma correction
        image = self._post_process(image)
        
        return image
    
    def _post_process(self, image: np.ndarray) -> np.ndarray:
        """Apply post-processing to image.
        
        Args:
            image: Linear color image
            
        Returns:
            Processed image
        """
        # Tone mapping (Reinhard)
        image = image / (image + 1.0)
        
        # Gamma correction
        image = np.power(image, 1.0 / 2.2)
        
        # Clamp to [0, 1]
        image = np.clip(image, 0.0, 1.0)
        
        return image
    
    def save_image(self, image: np.ndarray, filepath: Union[str, Path]):
        """Save image to file.
        
        Args:
            image: Image to save (HxWx3, 0-1)
            filepath: Output file path
        """
        filepath = Path(filepath)
        
        # Convert to 8-bit
        image_8bit = (image * 255).astype(np.uint8)
        
        # Save based on extension
        if filepath.suffix.lower() == '.png':
            from PIL import Image
            img = Image.fromarray(image_8bit, 'RGB')
            img.save(filepath)
        elif filepath.suffix.lower() == '.jpg' or filepath.suffix.lower() == '.jpeg':
            from PIL import Image
            img = Image.fromarray(image_8bit, 'RGB')
            img.save(filepath, quality=95)
        elif filepath.suffix.lower() == '.hdr':
            self._save_hdr(image, filepath)
        else:
            # Default to PNG
            from PIL import Image
            img = Image.fromarray(image_8bit, 'RGB')
            img.save(filepath.with_suffix('.png'))
    
    def _save_hdr(self, image: np.ndarray, filepath: Path):
        """Save HDR image.
        
        Args:
            image: HDR image
            filepath: Output file path
        """
        # Simple RGBE encoding
        height, width = image.shape[:2]
        
        with open(filepath, 'wb') as f:
            # Write header
            f.write(b"#?RGBE\n")
            f.write(b"FORMAT=32-bit_rle_rgbe\n")
            f.write(b"\n")
            f.write(f"-Y {height} +X {width}\n".encode())
            
            # Write pixels
            for y in range(height):
                for x in range(width):
                    rgb = image[y, x]
                    
                    # Find maximum component
                    max_val = max(rgb[0], rgb[1], rgb[2])
                    
                    if max_val < 1e-32:
                        # Black pixel
                        f.write(b"\x00\x00\x00\x00")
                    else:
                        # Normalize and encode
                        mantissa, exponent = math.frexp(max_val)
                        scaled = rgb * 256.0 / max_val
                        
                        r = min(255, int(scaled[0]))
                        g = min(255, int(scaled[1]))
                        b = min(255, int(scaled[2]))
                        e = exponent + 128
                        
                        f.write(bytes([r, g, b, e]))
    
    def create_cornell_box(self):
        """Create Cornell box test scene."""
        # Materials
        red_material = RTMaterial(
            name="red",
            material_type=MaterialType.DIFFUSE,
            albedo=np.array([0.65, 0.05, 0.05])
        )
        
        green_material = RTMaterial(
            name="green",
            material_type=MaterialType.DIFFUSE,
            albedo=np.array([0.12, 0.45, 0.15])
        )
        
        white_material = RTMaterial(
            name="white",
            material_type=MaterialType.DIFFUSE,
            albedo=np.array([0.73, 0.73, 0.73])
        )
        
        light_material = RTMaterial(
            name="light",
            material_type=MaterialType.EMISSIVE,
            albedo=np.array([1.0, 1.0, 1.0]),
            emission=np.array([15.0, 15.0, 15.0])
        )
        
        # Add materials
        self.add_material(red_material)
        self.add_material(green_material)
        self.add_material(white_material)
        self.add_material(light_material)
        
        # Cornell box dimensions
        box_size = 5.0
        half_size = box_size * 0.5
        
        # Walls
        # Left wall (red)
        left_wall = RTMesh(
            vertices=[
                [-half_size, -half_size, -half_size],
                [-half_size, -half_size, half_size],
                [-half_size, half_size, half_size],
                [-half_size, half_size, -half_size]
            ],
            indices=[0, 1, 2, 0, 2, 3],
            material_id=0  # red
        )
        
        # Right wall (green)
        right_wall = RTMesh(
            vertices=[
                [half_size, -half_size, -half_size],
                [half_size, half_size, -half_size],
                [half_size, half_size, half_size],
                [half_size, -half_size, half_size]
            ],
            indices=[0, 1, 2, 0, 2, 3],
            material_id=1  # green
        )
        
        # Floor (white)
        floor = RTMesh(
            vertices=[
                [-half_size, -half_size, -half_size],
                [half_size, -half_size, -half_size],
                [half_size, -half_size, half_size],
                [-half_size, -half_size, half_size]
            ],
            indices=[0, 1, 2, 0, 2, 3],
            material_id=2  # white
        )
        
        # Ceiling (white)
        ceiling = RTMesh(
            vertices=[
                [-half_size, half_size, -half_size],
                [-half_size, half_size, half_size],
                [half_size, half_size, half_size],
                [half_size, half_size, -half_size]
            ],
            indices=[0, 1, 2, 0, 2, 3],
            material_id=2  # white
        )
        
        # Back wall (white)
        back_wall = RTMesh(
            vertices=[
                [-half_size, -half_size, -half_size],
                [-half_size, half_size, -half_size],
                [half_size, half_size, -half_size],
                [half_size, -half_size, -half_size]
            ],
            indices=[0, 1, 2, 0, 2, 3],
            material_id=2  # white
        )
        
        # Light (on ceiling)
        light_size = box_size * 0.4
        light = RTMesh(
            vertices=[
                [-light_size/2, half_size - 0.01, -light_size/2],
                [-light_size/2, half_size - 0.01, light_size/2],
                [light_size/2, half_size - 0.01, light_size/2],
                [light_size/2, half_size - 0.01, -light_size/2]
            ],
            indices=[0, 1, 2, 0, 2, 3],
            material_id=3  # light
        )
        
        # Add objects
        self.add_object(left_wall)
        self.add_object(right_wall)
        self.add_object(floor)
        self.add_object(ceiling)
        self.add_object(back_wall)
        self.add_object(light)
        
        # Add boxes
        box1 = RTMesh.create_box([-1.0, -half_size, -0.5], [0.0, 0.0, 0.5], material_id=2)
        box2 = RTMesh.create_box([0.5, -half_size, 0.0], [1.5, 1.0, 1.0], material_id=2)
        
        self.add_object(box1)
        self.add_object(box2)
        
        # Set camera
        look_from = np.array([0.0, 0.0, 9.0])
        look_at = np.array([0.0, 0.0, 0.0])
        vup = np.array([0.0, 1.0, 0.0])
        
        self.camera = Camera(
            look_from=look_from,
            look_at=look_at,
            vup=vup,
            vfov=40.0,
            aspect_ratio=self.width / self.height
        )
        
        print("Cornell box scene created")

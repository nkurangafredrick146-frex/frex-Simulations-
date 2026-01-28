"""
Raster-based rendering engine for real-time 3D graphics.
Supports mesh rendering with various shading techniques.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import moderngl
import moderngl_window as mglw
from moderngl_window.opengl.vao import VAO
from moderngl_window.scene.camera import KeyboardCamera
import pygame
import glm
from dataclasses import dataclass, field
from enum import Enum
import time
from pathlib import Path
import json


class RenderMode(Enum):
    """Rendering modes."""
    SOLID = "solid"
    WIREFRAME = "wireframe"
    POINTS = "points"
    NORMALS = "normals"
    DEPTH = "depth"
    UV = "uv"
    SHADED = "shaded"


class ShadingModel(Enum):
    """Shading models."""
    FLAT = "flat"
    GOURAUD = "gouraud"
    PHONG = "phong"
    BLINN_PHONG = "blinn_phong"
    PBR = "pbr"
    CEL = "cel"


@dataclass
class Vertex:
    """Vertex data structure."""
    position: np.ndarray  # vec3
    normal: np.ndarray    # vec3
    texcoord: np.ndarray  # vec2
    color: np.ndarray     # vec4
    tangent: np.ndarray   # vec3
    bitangent: np.ndarray # vec3
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)
        if not isinstance(self.normal, np.ndarray):
            self.normal = np.array(self.normal, dtype=np.float32)
        if not isinstance(self.texcoord, np.ndarray):
            self.texcoord = np.array(self.texcoord, dtype=np.float32)
        if not isinstance(self.color, np.ndarray):
            self.color = np.array(self.color, dtype=np.float32)
        if not isinstance(self.tangent, np.ndarray):
            self.tangent = np.array(self.tangent, dtype=np.float32)
        if not isinstance(self.bitangent, np.ndarray):
            self.bitangent = np.array(self.bitangent, dtype=np.float32)
    
    def to_array(self) -> np.ndarray:
        """Convert to flat array for buffer."""
        return np.concatenate([
            self.position,
            self.normal,
            self.texcoord,
            self.color,
            self.tangent,
            self.bitangent
        ])


@dataclass
class Material:
    """Material properties."""
    name: str = "default"
    diffuse_color: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32))
    specular_color: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32))
    emissive_color: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    ambient_color: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.2, 1.0], dtype=np.float32))
    
    shininess: float = 32.0
    roughness: float = 0.5
    metallic: float = 0.0
    ior: float = 1.5  # Index of refraction
    
    # Texture paths
    diffuse_texture: Optional[str] = None
    normal_texture: Optional[str] = None
    specular_texture: Optional[str] = None
    roughness_texture: Optional[str] = None
    metallic_texture: Optional[str] = None
    ao_texture: Optional[str] = None  # Ambient occlusion
    emissive_texture: Optional[str] = None
    
    # Opacity
    opacity: float = 1.0
    alpha_cutoff: float = 0.5
    
    # Rendering flags
    double_sided: bool = False
    wireframe: bool = False
    cast_shadows: bool = True
    receive_shadows: bool = True
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        if not isinstance(self.diffuse_color, np.ndarray):
            self.diffuse_color = np.array(self.diffuse_color, dtype=np.float32)
        if not isinstance(self.specular_color, np.ndarray):
            self.specular_color = np.array(self.specular_color, dtype=np.float32)
        if not isinstance(self.emissive_color, np.ndarray):
            self.emissive_color = np.array(self.emissive_color, dtype=np.float32)
        if not isinstance(self.ambient_color, np.ndarray):
            self.ambient_color = np.array(self.ambient_color, dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "diffuse_color": self.diffuse_color.tolist(),
            "specular_color": self.specular_color.tolist(),
            "emissive_color": self.emissive_color.tolist(),
            "ambient_color": self.ambient_color.tolist(),
            "shininess": self.shininess,
            "roughness": self.roughness,
            "metallic": self.metallic,
            "ior": self.ior,
            "diffuse_texture": self.diffuse_texture,
            "normal_texture": self.normal_texture,
            "specular_texture": self.specular_texture,
            "roughness_texture": self.roughness_texture,
            "metallic_texture": self.metallic_texture,
            "ao_texture": self.ao_texture,
            "emissive_texture": self.emissive_texture,
            "opacity": self.opacity,
            "alpha_cutoff": self.alpha_cutoff,
            "double_sided": self.double_sided,
            "wireframe": self.wireframe,
            "cast_shadows": self.cast_shadows,
            "receive_shadows": self.receive_shadows
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Material":
        """Create from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            Material instance
        """
        return cls(
            name=data.get("name", "default"),
            diffuse_color=np.array(data.get("diffuse_color", [0.8, 0.8, 0.8, 1.0]), dtype=np.float32),
            specular_color=np.array(data.get("specular_color", [0.5, 0.5, 0.5, 1.0]), dtype=np.float32),
            emissive_color=np.array(data.get("emissive_color", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32),
            ambient_color=np.array(data.get("ambient_color", [0.2, 0.2, 0.2, 1.0]), dtype=np.float32),
            shininess=data.get("shininess", 32.0),
            roughness=data.get("roughness", 0.5),
            metallic=data.get("metallic", 0.0),
            ior=data.get("ior", 1.5),
            diffuse_texture=data.get("diffuse_texture"),
            normal_texture=data.get("normal_texture"),
            specular_texture=data.get("specular_texture"),
            roughness_texture=data.get("roughness_texture"),
            metallic_texture=data.get("metallic_texture"),
            ao_texture=data.get("ao_texture"),
            emissive_texture=data.get("emissive_texture"),
            opacity=data.get("opacity", 1.0),
            alpha_cutoff=data.get("alpha_cutoff", 0.5),
            double_sided=data.get("double_sided", False),
            wireframe=data.get("wireframe", False),
            cast_shadows=data.get("cast_shadows", True),
            receive_shadows=data.get("receive_shadows", True)
        )


@dataclass
class Mesh:
    """Mesh data structure."""
    name: str = "mesh"
    vertices: List[Vertex] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    material: Optional[Material] = None
    transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    
    # OpenGL resources
    _vao: Optional[VAO] = None
    _vbo: Optional[Any] = None
    _ibo: Optional[Any] = None
    
    def __post_init__(self):
        """Ensure transform is numpy array."""
        if not isinstance(self.transform, np.ndarray):
            self.transform = np.array(self.transform, dtype=np.float32)
    
    def calculate_normals(self):
        """Calculate smooth vertex normals."""
        if len(self.vertices) < 3 or len(self.indices) < 3:
            return
        
        # Reset normals
        for vertex in self.vertices:
            vertex.normal = np.zeros(3, dtype=np.float32)
        
        # Calculate face normals and accumulate
        for i in range(0, len(self.indices), 3):
            if i + 2 >= len(self.indices):
                break
            
            idx0 = self.indices[i]
            idx1 = self.indices[i + 1]
            idx2 = self.indices[i + 2]
            
            v0 = self.vertices[idx0].position
            v1 = self.vertices[idx1].position
            v2 = self.vertices[idx2].position
            
            # Calculate face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal_length = np.linalg.norm(normal)
            
            if normal_length > 0:
                normal = normal / normal_length
                
                # Accumulate to vertices
                self.vertices[idx0].normal += normal
                self.vertices[idx1].normal += normal
                self.vertices[idx2].normal += normal
        
        # Normalize vertex normals
        for vertex in self.vertices:
            normal_length = np.linalg.norm(vertex.normal)
            if normal_length > 0:
                vertex.normal = vertex.normal / normal_length
    
    def calculate_tangents(self):
        """Calculate tangent and bitangent vectors."""
        if len(self.vertices) < 3 or len(self.indices) < 3:
            return
        
        # Reset tangents and bitangents
        for vertex in self.vertices:
            vertex.tangent = np.zeros(3, dtype=np.float32)
            vertex.bitangent = np.zeros(3, dtype=np.float32)
        
        # Accumulate tangent and bitangent for each triangle
        for i in range(0, len(self.indices), 3):
            if i + 2 >= len(self.indices):
                break
            
            idx0 = self.indices[i]
            idx1 = self.indices[i + 1]
            idx2 = self.indices[i + 2]
            
            v0 = self.vertices[idx0]
            v1 = self.vertices[idx1]
            v2 = self.vertices[idx2]
            
            # Calculate edges
            pos1 = v1.position - v0.position
            pos2 = v2.position - v0.position
            
            # Calculate UV differences
            uv1 = v1.texcoord - v0.texcoord
            uv2 = v2.texcoord - v0.texcoord
            
            # Calculate tangent and bitangent
            r = 1.0 / (uv1[0] * uv2[1] - uv1[1] * uv2[0])
            tangent = (pos1 * uv2[1] - pos2 * uv1[1]) * r
            bitangent = (pos2 * uv1[0] - pos1 * uv2[0]) * r
            
            # Accumulate to vertices
            v0.tangent += tangent
            v1.tangent += tangent
            v2.tangent += tangent
            
            v0.bitangent += bitangent
            v1.bitangent += bitangent
            v2.bitangent += bitangent
        
        # Orthonormalize and normalize
        for vertex in self.vertices:
            # Gram-Schmidt orthogonalize
            n = vertex.normal
            t = vertex.tangent
            
            # Orthogonalize tangent with respect to normal
            t = t - n * np.dot(n, t)
            t_length = np.linalg.norm(t)
            if t_length > 0:
                vertex.tangent = t / t_length
            
            # Calculate bitangent
            b = np.cross(n, vertex.tangent)
            b_length = np.linalg.norm(b)
            if b_length > 0:
                vertex.bitangent = b / b_length
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box.
        
        Returns:
            Tuple of (min_bound, max_bound)
        """
        if not self.vertices:
            return np.zeros(3), np.zeros(3)
        
        min_bound = np.array([float('inf'), float('inf'), float('inf')])
        max_bound = np.array([float('-inf'), float('-inf'), float('-inf')])
        
        for vertex in self.vertices:
            pos = vertex.position
            min_bound = np.minimum(min_bound, pos)
            max_bound = np.maximum(max_bound, pos)
        
        return min_bound, max_bound
    
    def get_center(self) -> np.ndarray:
        """Get mesh center.
        
        Returns:
            Center point
        """
        min_bound, max_bound = self.get_bounds()
        return (min_bound + max_bound) * 0.5
    
    def get_diameter(self) -> float:
        """Get bounding sphere diameter.
        
        Returns:
            Diameter
        """
        min_bound, max_bound = self.get_bounds()
        return np.linalg.norm(max_bound - min_bound)


@dataclass
class Light:
    """Light source."""
    name: str = "light"
    type: str = "directional"  # "directional", "point", "spot"
    position: np.ndarray = field(default_factory=lambda: np.array([0, 5, 0], dtype=np.float32))
    direction: np.ndarray = field(default_factory=lambda: np.array([0, -1, 0], dtype=np.float32))
    color: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32))
    intensity: float = 1.0
    
    # Point/spot light properties
    range: float = 100.0
    attenuation: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # constant, linear, quadratic
    
    # Spot light properties
    spot_angle: float = 45.0  # degrees
    spot_blend: float = 0.5   # 0-1
    
    # Shadow properties
    cast_shadows: bool = True
    shadow_map_size: int = 1024
    shadow_bias: float = 0.001
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)
        if not isinstance(self.direction, np.ndarray):
            self.direction = np.array(self.direction, dtype=np.float32)
        if not isinstance(self.color, np.ndarray):
            self.color = np.array(self.color, dtype=np.float32)
    
    def get_view_projection_matrix(self) -> np.ndarray:
        """Get view-projection matrix for shadow mapping.
        
        Returns:
            View-projection matrix
        """
        if self.type == "directional":
            # Orthographic projection for directional light
            size = 50.0
            near = 0.1
            far = 100.0
            
            light_view = glm.lookAt(
                glm.vec3(self.position[0], self.position[1], self.position[2]),
                glm.vec3(self.position[0] + self.direction[0],
                        self.position[1] + self.direction[1],
                        self.position[2] + self.direction[2]),
                glm.vec3(0, 1, 0)
            )
            
            light_proj = glm.ortho(-size, size, -size, size, near, far)
            return light_proj * light_view
            
        else:
            # Perspective projection for point/spot lights
            fov = glm.radians(90.0)
            aspect = 1.0
            near = 0.1
            far = self.range
            
            light_view = glm.lookAt(
                glm.vec3(self.position[0], self.position[1], self.position[2]),
                glm.vec3(self.position[0] + self.direction[0],
                        self.position[1] + self.direction[1],
                        self.position[2] + self.direction[2]),
                glm.vec3(0, 1, 0)
            )
            
            light_proj = glm.perspective(fov, aspect, near, far)
            return light_proj * light_view


class RasterEngine:
    """Main raster rendering engine."""
    
    def __init__(self, width: int = 1280, height: int = 720, 
                 window_title: str = "FrexTech Raster Engine"):
        """Initialize raster engine.
        
        Args:
            width: Window width
            height: Window height
            window_title: Window title
        """
        self.width = width
        self.height = height
        self.window_title = window_title
        
        # OpenGL context
        self.ctx: Optional[moderngl.Context] = None
        self.window: Optional[mglw.Window] = None
        
        # Scene data
        self.meshes: List[Mesh] = []
        self.lights: List[Light] = []
        self.materials: Dict[str, Material] = {}
        
        # Rendering state
        self.render_mode = RenderMode.SHADED
        self.shading_model = ShadingModel.PBR
        self.background_color = (0.1, 0.1, 0.1, 1.0)
        self.wireframe_color = (0.8, 0.8, 0.8, 1.0)
        
        # Camera
        self.camera_pos = glm.vec3(0, 0, 5)
        self.camera_target = glm.vec3(0, 0, 0)
        self.camera_up = glm.vec3(0, 1, 0)
        self.fov = 60.0
        self.near_plane = 0.1
        self.far_plane = 1000.0
        
        # Matrices
        self.view_matrix = glm.mat4(1.0)
        self.projection_matrix = glm.mat4(1.0)
        self.view_projection_matrix = glm.mat4(1.0)
        
        # Shaders
        self.shader_programs: Dict[str, moderngl.Program] = {}
        self.current_shader: Optional[moderngl.Program] = None
        
        # Textures
        self.textures: Dict[str, moderngl.Texture] = {}
        self.default_texture: Optional[moderngl.Texture] = None
        
        # Framebuffers for effects
        self.main_fbo: Optional[moderngl.Framebuffer] = None
        self.shadow_fbo: Optional[moderngl.Framebuffer] = None
        
        # Uniform buffers
        self.camera_ubo: Optional[moderngl.Buffer] = None
        self.lights_ubo: Optional[moderngl.Buffer] = None
        
        # Performance
        self.frame_time = 0.0
        self.fps = 0.0
        self.triangle_count = 0
        self.draw_call_count = 0
        
        # Initialization flags
        self.is_initialized = False
        self.should_close = False
        
        # Input handling
        self.keyboard_state = {}
        self.mouse_position = (0, 0)
        self.mouse_buttons = [False, False, False]
        
    def initialize(self):
        """Initialize the rendering engine."""
        try:
            # Configure ModernGL window
            config = mglw.WindowConfig(
                title=self.window_title,
                size=(self.width, self.height),
                gl_version=(3, 3),
                aspect_ratio=self.width / self.height,
                vsync=True,
                resizable=True
            )
            
            # Create window
            self.window = mglw.create_window_from_config(config)
            self.ctx = self.window.ctx
            
            # Enable OpenGL features
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.CULL_FACE)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            
            # Create default texture
            self._create_default_texture()
            
            # Create shaders
            self._create_shaders()
            
            # Create framebuffers
            self._create_framebuffers()
            
            # Create uniform buffers
            self._create_uniform_buffers()
            
            # Set up camera
            self._update_camera_matrices()
            
            # Add default light
            self._add_default_light()
            
            self.is_initialized = True
            print("Raster engine initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize raster engine: {e}")
            raise
    
    def _create_default_texture(self):
        """Create default white texture."""
        # 1x1 white texture
        data = np.array([[[255, 255, 255, 255]]], dtype=np.uint8)
        self.default_texture = self.ctx.texture((1, 1), 4, data.tobytes())
        self.default_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.default_texture.repeat_x = False
        self.default_texture.repeat_y = False
        
        # Normal map texture (neutral blue)
        normal_data = np.array([[[128, 128, 255, 255]]], dtype=np.uint8)
        self.default_normal_texture = self.ctx.texture((1, 1), 4, normal_data.tobytes())
        self.default_normal_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    
    def _create_shaders(self):
        """Create shader programs."""
        # Basic shader for solid/wireframe rendering
        basic_vert = """
        #version 330
        
        layout(location = 0) in vec3 in_position;
        layout(location = 1) in vec3 in_normal;
        layout(location = 2) in vec2 in_texcoord;
        layout(location = 3) in vec4 in_color;
        
        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;
        uniform mat4 u_normal_matrix;
        
        out vec3 v_position;
        out vec3 v_normal;
        out vec2 v_texcoord;
        out vec4 v_color;
        
        void main() {
            vec4 world_position = u_model * vec4(in_position, 1.0);
            v_position = world_position.xyz;
            v_normal = mat3(u_normal_matrix) * in_normal;
            v_texcoord = in_texcoord;
            v_color = in_color;
            
            gl_Position = u_projection * u_view * world_position;
        }
        """
        
        basic_frag = """
        #version 330
        
        in vec3 v_position;
        in vec3 v_normal;
        in vec2 v_texcoord;
        in vec4 v_color;
        
        uniform vec4 u_color;
        uniform int u_render_mode;
        
        out vec4 f_color;
        
        void main() {
            if (u_render_mode == 0) { // SOLID
                f_color = u_color;
            } else if (u_render_mode == 1) { // WIREFRAME
                f_color = vec4(0.8, 0.8, 0.8, 1.0);
            } else if (u_render_mode == 2) { // NORMALS
                f_color = vec4(normalize(v_normal) * 0.5 + 0.5, 1.0);
            } else if (u_render_mode == 3) { // DEPTH
                float depth = gl_FragCoord.z;
                f_color = vec4(vec3(depth), 1.0);
            } else if (u_render_mode == 4) { // UV
                f_color = vec4(v_texcoord, 0.0, 1.0);
            } else {
                f_color = v_color;
            }
        }
        """
        
        # PBR shader
        pbr_vert = """
        #version 330
        
        layout(location = 0) in vec3 in_position;
        layout(location = 1) in vec3 in_normal;
        layout(location = 2) in vec2 in_texcoord;
        layout(location = 3) in vec4 in_color;
        layout(location = 4) in vec3 in_tangent;
        layout(location = 5) in vec3 in_bitangent;
        
        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;
        uniform mat3 u_normal_matrix;
        
        out vec3 v_position;
        out vec3 v_normal;
        out vec2 v_texcoord;
        out vec4 v_color;
        out mat3 v_tbn;
        
        void main() {
            vec4 world_position = u_model * vec4(in_position, 1.0);
            v_position = world_position.xyz;
            v_normal = normalize(u_normal_matrix * in_normal);
            v_texcoord = in_texcoord;
            v_color = in_color;
            
            // Calculate TBN matrix for normal mapping
            vec3 T = normalize(u_normal_matrix * in_tangent);
            vec3 B = normalize(u_normal_matrix * in_bitangent);
            vec3 N = v_normal;
            
            // Re-orthogonalize T with respect to N
            T = normalize(T - dot(T, N) * N);
            
            // Calculate B again to ensure orthonormal basis
            B = cross(N, T);
            
            v_tbn = mat3(T, B, N);
            
            gl_Position = u_projection * u_view * world_position;
        }
        """
        
        pbr_frag = """
        #version 330
        
        // Material properties
        struct Material {
            vec4 diffuse_color;
            vec4 specular_color;
            vec4 emissive_color;
            float roughness;
            float metallic;
            float ior;
            float opacity;
        };
        
        // Light properties
        struct Light {
            vec4 position;
            vec4 direction;
            vec4 color;
            float intensity;
            float range;
            float spot_angle;
            float spot_blend;
            int type; // 0: directional, 1: point, 2: spot
        };
        
        // Camera
        uniform vec3 u_camera_position;
        
        // Material
        uniform Material u_material;
        uniform sampler2D u_diffuse_texture;
        uniform sampler2D u_normal_texture;
        uniform sampler2D u_roughness_texture;
        uniform sampler2D u_metallic_texture;
        uniform sampler2D u_ao_texture;
        uniform sampler2D u_emissive_texture;
        
        // Lights (max 8 lights)
        uniform int u_light_count;
        uniform Light u_lights[8];
        
        // Inputs from vertex shader
        in vec3 v_position;
        in vec3 v_normal;
        in vec2 v_texcoord;
        in vec4 v_color;
        in mat3 v_tbn;
        
        // Output
        out vec4 f_color;
        
        // Constants
        const float PI = 3.14159265359;
        
        // PBR functions
        float distribution_ggx(vec3 N, vec3 H, float roughness) {
            float a = roughness * roughness;
            float a2 = a * a;
            float NdotH = max(dot(N, H), 0.0);
            float NdotH2 = NdotH * NdotH;
            
            float denom = (NdotH2 * (a2 - 1.0) + 1.0);
            denom = PI * denom * denom;
            
            return a2 / denom;
        }
        
        float geometry_schlick_ggx(float NdotV, float roughness) {
            float r = (roughness + 1.0);
            float k = (r * r) / 8.0;
            
            return NdotV / (NdotV * (1.0 - k) + k);
        }
        
        float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
            float NdotV = max(dot(N, V), 0.0);
            float NdotL = max(dot(N, L), 0.0);
            float ggx1 = geometry_schlick_ggx(NdotV, roughness);
            float ggx2 = geometry_schlick_ggx(NdotL, roughness);
            
            return ggx1 * ggx2;
        }
        
        vec3 fresnel_schlick(float cos_theta, vec3 F0) {
            return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
        }
        
        void main() {
            // Sample textures
            vec4 diffuse_sample = texture(u_diffuse_texture, v_texcoord);
            vec4 normal_sample = texture(u_normal_texture, v_texcoord);
            float roughness_sample = texture(u_roughness_texture, v_texcoord).r;
            float metallic_sample = texture(u_metallic_texture, v_texcoord).r;
            float ao_sample = texture(u_ao_texture, v_texcoord).r;
            vec4 emissive_sample = texture(u_emissive_texture, v_texcoord);
            
            // Material properties
            vec4 albedo = u_material.diffuse_color * diffuse_sample * v_color;
            float roughness = u_material.roughness * roughness_sample;
            float metallic = u_material.metallic * metallic_sample;
            vec3 emissive = u_material.emissive_color.rgb * emissive_sample.rgb;
            
            // Normal mapping
            vec3 N = normalize(v_normal);
            if (normal_sample.a > 0.5) { // Only use normal map if it exists
                vec3 tangent_normal = normal_sample.rgb * 2.0 - 1.0;
                N = normalize(v_tbn * tangent_normal);
            }
            
            // View direction
            vec3 V = normalize(u_camera_position - v_position);
            
            // Reflectance at normal incidence
            vec3 F0 = vec3(0.04);
            F0 = mix(F0, albedo.rgb, metallic);
            
            // Result
            vec3 Lo = vec3(0.0);
            
            // Calculate lighting for each light
            for (int i = 0; i < u_light_count; ++i) {
                Light light = u_lights[i];
                vec3 light_color = light.color.rgb * light.intensity;
                
                // Light direction and distance
                vec3 L;
                float attenuation = 1.0;
                
                if (light.type == 0) { // Directional
                    L = normalize(-light.direction.xyz);
                } else { // Point or spot
                    vec3 light_vec = light.position.xyz - v_position;
                    float distance = length(light_vec);
                    L = normalize(light_vec);
                    
                    // Attenuation
                    float atten = 1.0 / (1.0 + 0.01 * distance + 0.0001 * distance * distance);
                    attenuation *= atten;
                    
                    // Spot light cone
                    if (light.type == 2) { // Spot
                        float theta = dot(L, normalize(-light.direction.xyz));
                        float epsilon = light.spot_blend;
                        float intensity = clamp((theta - cos(radians(light.spot_angle))) / epsilon, 0.0, 1.0);
                        attenuation *= intensity;
                    }
                }
                
                // Half vector
                vec3 H = normalize(V + L);
                
                // Cook-Torrance BRDF
                float NDF = distribution_ggx(N, H, roughness);
                float G = geometry_smith(N, V, L, roughness);
                vec3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);
                
                vec3 kS = F;
                vec3 kD = vec3(1.0) - kS;
                kD *= 1.0 - metallic;
                
                vec3 numerator = NDF * G * F;
                float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
                vec3 specular = numerator / denominator;
                
                // Lambertian diffuse
                float NdotL = max(dot(N, L), 0.0);
                vec3 radiance = light_color * attenuation * NdotL;
                
                Lo += (kD * albedo.rgb / PI + specular) * radiance;
            }
            
            // Ambient lighting
            vec3 ambient = vec3(0.03) * albedo.rgb * ao_sample;
            
            // Emissive
            vec3 emissive_contribution = emissive;
            
            // Final color
            vec3 color = ambient + Lo + emissive_contribution;
            
            // Gamma correction
            color = color / (color + vec3(1.0));
            color = pow(color, vec3(1.0 / 2.2));
            
            f_color = vec4(color, albedo.a * u_material.opacity);
            
            // Alpha discard
            if (f_color.a < u_material.opacity * 0.5) {
                discard;
            }
        }
        """
        
        try:
            # Compile basic shader
            self.shader_programs["basic"] = self.ctx.program(
                vertex_shader=basic_vert,
                fragment_shader=basic_frag
            )
            
            # Compile PBR shader
            self.shader_programs["pbr"] = self.ctx.program(
                vertex_shader=pbr_vert,
                fragment_shader=pbr_frag
            )
            
            self.current_shader = self.shader_programs["basic"]
            
        except Exception as e:
            print(f"Failed to compile shaders: {e}")
            raise
    
    def _create_framebuffers(self):
        """Create framebuffers for rendering."""
        # Main framebuffer (offscreen rendering for post-processing)
        self.main_fbo = self.ctx.framebuffer(
            color_attachments=[
                self.ctx.texture((self.width, self.height), 4, samples=4)
            ],
            depth_attachment=self.ctx.depth_texture((self.width, self.height), samples=4)
        )
        
        # Shadow map framebuffer
        shadow_size = 2048
        self.shadow_fbo = self.ctx.framebuffer(
            depth_attachment=self.ctx.depth_texture((shadow_size, shadow_size))
        )
    
    def _create_uniform_buffers(self):
        """Create uniform buffers."""
        # Camera uniform buffer
        camera_data = np.zeros(16 * 4 + 4, dtype=np.float32)  # 4x4 matrices + vec3 position
        self.camera_ubo = self.ctx.buffer(data=camera_data)
        
        # Lights uniform buffer
        max_lights = 8
        light_size = 16 * 4 + 4  # vec4 * 4 + float * 3 + int
        lights_data = np.zeros(max_lights * light_size, dtype=np.float32)
        self.lights_ubo = self.ctx.buffer(data=lights_data)
    
    def _add_default_light(self):
        """Add default directional light."""
        default_light = Light(
            name="default_light",
            type="directional",
            position=np.array([5, 10, 5], dtype=np.float32),
            direction=np.array([-0.5, -1.0, -0.5], dtype=np.float32),
            color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            intensity=1.0
        )
        self.lights.append(default_light)
    
    def _update_camera_matrices(self):
        """Update camera matrices."""
        # View matrix
        self.view_matrix = glm.lookAt(
            self.camera_pos,
            self.camera_target,
            self.camera_up
        )
        
        # Projection matrix
        aspect = self.width / self.height
        self.projection_matrix = glm.perspective(
            glm.radians(self.fov),
            aspect,
            self.near_plane,
            self.far_plane
        )
        
        # Combined matrix
        self.view_projection_matrix = self.projection_matrix * self.view_matrix
        
        # Update camera uniform buffer
        if self.camera_ubo:
            # Flatten matrices
            view_flat = np.array(self.view_matrix).flatten('F')
            proj_flat = np.array(self.projection_matrix).flatten('F')
            vp_flat = np.array(self.view_projection_matrix).flatten('F')
            camera_pos_flat = np.array([self.camera_pos.x, self.camera_pos.y, self.camera_pos.z, 1.0])
            
            # Combine into single array
            camera_data = np.concatenate([view_flat, proj_flat, vp_flat, camera_pos_flat])
            self.camera_ubo.write(camera_data.astype('f4').tobytes())
    
    def add_mesh(self, mesh: Mesh):
        """Add mesh to scene.
        
        Args:
            mesh: Mesh to add
        """
        # Calculate normals and tangents if needed
        if len(mesh.vertices) > 0 and np.all(mesh.vertices[0].normal == 0):
            mesh.calculate_normals()
        
        # Calculate tangents if needed (for normal mapping)
        if len(mesh.vertices) > 0 and np.all(mesh.vertices[0].tangent == 0):
            mesh.calculate_tangents()
        
        # Create OpenGL buffers
        self._create_mesh_buffers(mesh)
        
        self.meshes.append(mesh)
        self.triangle_count += len(mesh.indices) // 3
    
    def _create_mesh_buffers(self, mesh: Mesh):
        """Create OpenGL buffers for mesh.
        
        Args:
            mesh: Mesh to create buffers for
        """
        # Convert vertices to array
        vertex_data = []
        for vertex in mesh.vertices:
            vertex_data.extend(vertex.to_array())
        
        vertex_array = np.array(vertex_data, dtype=np.float32)
        index_array = np.array(mesh.indices, dtype=np.uint32)
        
        # Create vertex buffer
        mesh._vbo = self.ctx.buffer(vertex_array.tobytes())
        
        # Create index buffer
        mesh._ibo = self.ctx.buffer(index_array.tobytes())
        
        # Create vertex array object
        mesh._vao = self.ctx.vertex_array(
            self.current_shader,
            [
                (mesh._vbo, '3f 3f 2f 4f 3f 3f', 'in_position', 'in_normal', 'in_texcoord', 
                 'in_color', 'in_tangent', 'in_bitangent')
            ],
            mesh._ibo,
            mode=moderngl.TRIANGLES
        )
    
    def remove_mesh(self, mesh_name: str):
        """Remove mesh by name.
        
        Args:
            mesh_name: Name of mesh to remove
        """
        for i, mesh in enumerate(self.meshes):
            if mesh.name == mesh_name:
                # Clean up OpenGL resources
                if mesh._vbo:
                    mesh._vbo.release()
                if mesh._ibo:
                    mesh._ibo.release()
                if mesh._vao:
                    mesh._vao.release()
                
                self.meshes.pop(i)
                self.triangle_count -= len(mesh.indices) // 3
                break
    
    def add_light(self, light: Light):
        """Add light to scene.
        
        Args:
            light: Light to add
        """
        self.lights.append(light)
    
    def remove_light(self, light_name: str):
        """Remove light by name.
        
        Args:
            light_name: Name of light to remove
        """
        self.lights = [light for light in self.lights if light.name != light_name]
    
    def set_camera(self, position: np.ndarray, target: np.ndarray, up: np.ndarray = None):
        """Set camera position and orientation.
        
        Args:
            position: Camera position
            target: Camera look-at target
            up: Camera up vector
        """
        self.camera_pos = glm.vec3(position[0], position[1], position[2])
        self.camera_target = glm.vec3(target[0], target[1], target[2])
        if up is not None:
            self.camera_up = glm.vec3(up[0], up[1], up[2])
        
        self._update_camera_matrices()
    
    def render_frame(self):
        """Render a single frame."""
        if not self.is_initialized:
            return
        
        # Clear buffers
        self.ctx.clear(*self.background_color)
        
        # Bind main framebuffer
        if self.main_fbo:
            self.main_fbo.use()
            self.main_fbo.clear(*self.background_color)
        
        # Update camera
        self._update_camera_matrices()
        
        # Update lights uniform buffer
        self._update_lights_ubo()
        
        # Set render mode
        if self.render_mode == RenderMode.SHADED:
            self.current_shader = self.shader_programs.get("pbr", self.shader_programs["basic"])
        else:
            self.current_shader = self.shader_programs["basic"]
        
        # Render meshes
        self.draw_call_count = 0
        for mesh in self.meshes:
            self._render_mesh(mesh)
            self.draw_call_count += 1
        
        # Copy to main framebuffer if using offscreen rendering
        if self.main_fbo:
            self.ctx.screen.use()
            self._blit_framebuffer(self.main_fbo, self.ctx.screen)
        
        # Swap buffers
        self.window.swap_buffers()
        
        # Update performance metrics
        self._update_performance_metrics()
    
    def _render_mesh(self, mesh: Mesh):
        """Render a single mesh.
        
        Args:
            mesh: Mesh to render
        """
        if mesh._vao is None:
            return
        
        # Set shader uniforms
        if self.current_shader == self.shader_programs["basic"]:
            # Basic shader uniforms
            self.current_shader['u_model'].write(mesh.transform.astype('f4').tobytes())
            self.current_shader['u_view'].write(np.array(self.view_matrix).flatten('F').astype('f4').tobytes())
            self.current_shader['u_projection'].write(np.array(self.projection_matrix).flatten('F').astype('f4').tobytes())
            
            # Normal matrix (transpose of inverse of model matrix)
            model_mat = glm.mat4(*mesh.transform.flatten())
            normal_mat = glm.transpose(glm.inverse(model_mat))
            self.current_shader['u_normal_matrix'].write(np.array(normal_mat).flatten('F').astype('f4').tobytes())
            
            # Render mode
            render_mode_value = {
                RenderMode.SOLID: 0,
                RenderMode.WIREFRAME: 1,
                RenderMode.NORMALS: 2,
                RenderMode.DEPTH: 3,
                RenderMode.UV: 4,
                RenderMode.SHADED: 5
            }.get(self.render_mode, 0)
            self.current_shader['u_render_mode'].value = render_mode_value
            
            # Color
            if mesh.material:
                color = mesh.material.diffuse_color
            else:
                color = np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)
            self.current_shader['u_color'].write(color.astype('f4').tobytes())
            
        elif self.current_shader == self.shader_programs["pbr"]:
            # PBR shader uniforms
            self.current_shader['u_model'].write(mesh.transform.astype('f4').tobytes())
            self.current_shader['u_view'].write(np.array(self.view_matrix).flatten('F').astype('f4').tobytes())
            self.current_shader['u_projection'].write(np.array(self.projection_matrix).flatten('F').astype('f4').tobytes())
            
            # Normal matrix
            model_mat = glm.mat4(*mesh.transform.flatten())
            normal_mat = glm.mat3(glm.transpose(glm.inverse(model_mat)))
            self.current_shader['u_normal_matrix'].write(np.array(normal_mat).flatten('F').astype('f4').tobytes())
            
            # Camera position
            self.current_shader['u_camera_position'].value = (self.camera_pos.x, self.camera_pos.y, self.camera_pos.z)
            
            # Material properties
            if mesh.material:
                material = mesh.material
                self.current_shader['u_material.diffuse_color'].write(material.diffuse_color.astype('f4').tobytes())
                self.current_shader['u_material.specular_color'].write(material.specular_color.astype('f4').tobytes())
                self.current_shader['u_material.emissive_color'].write(material.emissive_color.astype('f4').tobytes())
                self.current_shader['u_material.roughness'].value = material.roughness
                self.current_shader['u_material.metallic'].value = material.metallic
                self.current_shader['u_material.ior'].value = material.ior
                self.current_shader['u_material.opacity'].value = material.opacity
                
                # Bind textures
                self._bind_material_textures(material)
            else:
                # Default material
                default_color = np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)
                self.current_shader['u_material.diffuse_color'].write(default_color.astype('f4').tobytes())
                self.current_shader['u_material.roughness'].value = 0.5
                self.current_shader['u_material.metallic'].value = 0.0
                self.current_shader['u_material.opacity'].value = 1.0
                
                # Default textures
                self._bind_default_textures()
        
        # Set polygon mode for wireframe
        if (self.render_mode == RenderMode.WIREFRAME or 
            (mesh.material and mesh.material.wireframe)):
            self.ctx.wireframe = True
        else:
            self.ctx.wireframe = False
        
        # Render mesh
        mesh._vao.render()
        
        # Reset wireframe mode
        self.ctx.wireframe = False
    
    def _bind_material_textures(self, material: Material):
        """Bind material textures to shader.
        
        Args:
            material: Material with textures
        """
        # Diffuse texture
        if material.diffuse_texture and material.diffuse_texture in self.textures:
            self.textures[material.diffuse_texture].use(0)
        else:
            self.default_texture.use(0)
        
        # Normal texture
        if material.normal_texture and material.normal_texture in self.textures:
            self.textures[material.normal_texture].use(1)
        else:
            self.default_normal_texture.use(1)
        
        # Other textures (simplified - would need more slots in practice)
        # For now, use default textures
        
        # Set texture uniforms
        self.current_shader['u_diffuse_texture'].value = 0
        self.current_shader['u_normal_texture'].value = 1
        self.current_shader['u_roughness_texture'].value = 2
        self.current_shader['u_metallic_texture'].value = 3
        self.current_shader['u_ao_texture'].value = 4
        self.current_shader['u_emissive_texture'].value = 5
    
    def _bind_default_textures(self):
        """Bind default textures."""
        self.default_texture.use(0)
        self.default_normal_texture.use(1)
        self.default_texture.use(2)  # Roughness
        self.default_texture.use(3)  # Metallic
        self.default_texture.use(4)  # AO
        self.default_texture.use(5)  # Emissive
    
    def _update_lights_ubo(self):
        """Update lights uniform buffer."""
        if not self.lights_ubo:
            return
        
        max_lights = 8
        light_size = 16 * 4 + 4  # vec4 * 4 + float * 3 + int
        lights_data = np.zeros(max_lights * light_size, dtype=np.float32)
        
        for i, light in enumerate(self.lights[:max_lights]):
            offset = i * light_size
            
            # Position (vec4)
            lights_data[offset:offset+4] = np.array([light.position[0], light.position[1], 
                                                   light.position[2], 1.0], dtype=np.float32)
            offset += 4
            
            # Direction (vec4)
            lights_data[offset:offset+4] = np.array([light.direction[0], light.direction[1],
                                                   light.direction[2], 0.0], dtype=np.float32)
            offset += 4
            
            # Color (vec4)
            lights_data[offset:offset+4] = np.array([light.color[0], light.color[1],
                                                   light.color[2], 1.0], dtype=np.float32)
            offset += 4
            
            # Other properties (packed in vec4)
            lights_data[offset] = light.intensity
            lights_data[offset+1] = light.range
            lights_data[offset+2] = light.spot_angle
            lights_data[offset+3] = light.spot_blend
            offset += 4
            
            # Light type
            light_type = {
                "directional": 0,
                "point": 1,
                "spot": 2
            }.get(light.type, 0)
            
            # Pack type into float (for simplicity)
            lights_data[offset] = float(light_type)
        
        # Write to buffer
        self.lights_ubo.write(lights_data.astype('f4').tobytes())
        
        # Set light count in shader
        if self.current_shader == self.shader_programs["pbr"]:
            light_count = min(len(self.lights), max_lights)
            self.current_shader['u_light_count'].value = light_count
    
    def _blit_framebuffer(self, src_fbo: moderngl.Framebuffer, dst_fbo: moderngl.Framebuffer):
        """Copy framebuffer contents.
        
        Args:
            src_fbo: Source framebuffer
            dst_fbo: Destination framebuffer
        """
        src_fbo.color_attachments[0].use(0)
        
        # Simple fullscreen quad shader for blitting
        if not hasattr(self, '_blit_program'):
            blit_vert = """
            #version 330
            
            in vec2 in_position;
            out vec2 v_texcoord;
            
            void main() {
                v_texcoord = in_position * 0.5 + 0.5;
                gl_Position = vec4(in_position, 0.0, 1.0);
            }
            """
            
            blit_frag = """
            #version 330
            
            in vec2 v_texcoord;
            uniform sampler2D u_texture;
            out vec4 f_color;
            
            void main() {
                f_color = texture(u_texture, v_texcoord);
            }
            """
            
            self._blit_program = self.ctx.program(
                vertex_shader=blit_vert,
                fragment_shader=blit_frag
            )
            
            # Fullscreen quad vertices
            quad_vertices = np.array([
                -1.0, -1.0,
                1.0, -1.0,
                -1.0, 1.0,
                1.0, 1.0
            ], dtype=np.float32)
            
            self._blit_vbo = self.ctx.buffer(quad_vertices.tobytes())
            self._blit_vao = self.ctx.vertex_array(
                self._blit_program,
                [(self._blit_vbo, '2f', 'in_position')]
            )
        
        # Render fullscreen quad
        self._blit_program['u_texture'].value = 0
        self._blit_vao.render(moderngl.TRIANGLE_STRIP)
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        current_time = time.time()
        if hasattr(self, '_last_frame_time'):
            self.frame_time = current_time - self._last_frame_time
            if self.frame_time > 0:
                self.fps = 1.0 / self.frame_time
        self._last_frame_time = current_time
    
    def load_texture(self, filepath: Union[str, Path], texture_name: str):
        """Load texture from file.
        
        Args:
            filepath: Path to texture file
            texture_name: Name to assign to texture
        """
        try:
            # Load image using PIL
            from PIL import Image
            img = Image.open(filepath)
            img = img.convert('RGBA')
            
            # Create texture
            texture = self.ctx.texture(img.size, 4, img.tobytes())
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = True
            texture.repeat_y = True
            
            self.textures[texture_name] = texture
            print(f"Loaded texture: {texture_name}")
            
        except Exception as e:
            print(f"Failed to load texture {filepath}: {e}")
    
    def create_simple_geometry(self, geometry_type: str, **kwargs) -> Mesh:
        """Create simple geometry.
        
        Args:
            geometry_type: Type of geometry ('cube', 'sphere', 'plane', 'cylinder')
            **kwargs: Geometry parameters
            
        Returns:
            Created mesh
        """
        if geometry_type == 'cube':
            return self._create_cube_mesh(**kwargs)
        elif geometry_type == 'sphere':
            return self._create_sphere_mesh(**kwargs)
        elif geometry_type == 'plane':
            return self._create_plane_mesh(**kwargs)
        elif geometry_type == 'cylinder':
            return self._create_cylinder_mesh(**kwargs)
        else:
            raise ValueError(f"Unknown geometry type: {geometry_type}")
    
    def _create_cube_mesh(self, size: float = 1.0, 
                         center: bool = True) -> Mesh:
        """Create cube mesh.
        
        Args:
            size: Cube size
            center: Whether to center at origin
            
        Returns:
            Cube mesh
        """
        half_size = size * 0.5
        offset = -half_size if center else 0.0
        
        # Cube vertices (positions, normals, texcoords, colors)
        vertices = []
        
        # Front face
        vertices.extend([
            # Position           Normal          TexCoord    Color
            [-half_size+offset, -half_size+offset, half_size+offset,  0, 0, 1,  0, 0,  1, 1, 1, 1,  1, 0, 0,  0, 1, 0],
            [half_size+offset, -half_size+offset, half_size+offset,   0, 0, 1,  1, 0,  1, 1, 1, 1,  1, 0, 0,  0, 1, 0],
            [half_size+offset, half_size+offset, half_size+offset,    0, 0, 1,  1, 1,  1, 1, 1, 1,  1, 0, 0,  0, 1, 0],
            [-half_size+offset, half_size+offset, half_size+offset,   0, 0, 1,  0, 1,  1, 1, 1, 1,  1, 0, 0,  0, 1, 0],
        ])
        
        # Back face
        vertices.extend([
            [-half_size+offset, -half_size+offset, -half_size+offset,  0, 0, -1,  1, 0,  1, 1, 1, 1,  -1, 0, 0,  0, -1, 0],
            [-half_size+offset, half_size+offset, -half_size+offset,   0, 0, -1,  1, 1,  1, 1, 1, 1,  -1, 0, 0,  0, -1, 0],
            [half_size+offset, half_size+offset, -half_size+offset,    0, 0, -1,  0, 1,  1, 1, 1, 1,  -1, 0, 0,  0, -1, 0],
            [half_size+offset, -half_size+offset, -half_size+offset,   0, 0, -1,  0, 0,  1, 1, 1, 1,  -1, 0, 0,  0, -1, 0],
        ])
        
        # Top face
        vertices.extend([
            [-half_size+offset, half_size+offset, -half_size+offset,  0, 1, 0,  0, 1,  1, 1, 1, 1,  1, 0, 0,  0, 0, 1],
            [-half_size+offset, half_size+offset, half_size+offset,   0, 1, 0,  0, 0,  1, 1, 1, 1,  1, 0, 0,  0, 0, 1],
            [half_size+offset, half_size+offset, half_size+offset,    0, 1, 0,  1, 0,  1, 1, 1, 1,  1, 0, 0,  0, 0, 1],
            [half_size+offset, half_size+offset, -half_size+offset,   0, 1, 0,  1, 1,  1, 1, 1, 1,  1, 0, 0,  0, 0, 1],
        ])
        
        # Bottom face
        vertices.extend([
            [-half_size+offset, -half_size+offset, -half_size+offset,  0, -1, 0,  1, 1,  1, 1, 1, 1,  1, 0, 0,  0, 0, -1],
            [half_size+offset, -half_size+offset, -half_size+offset,   0, -1, 0,  0, 1,  1, 1, 1, 1,  1, 0, 0,  0, 0, -1],
            [half_size+offset, -half_size+offset, half_size+offset,    0, -1, 0,  0, 0,  1, 1, 1, 1,  1, 0, 0,  0, 0, -1],
            [-half_size+offset, -half_size+offset, half_size+offset,   0, -1, 0,  1, 0,  1, 1, 1, 1,  1, 0, 0,  0, 0, -1],
        ])
        
        # Right face
        vertices.extend([
            [half_size+offset, -half_size+offset, -half_size+offset,  1, 0, 0,  0, 0,  1, 1, 1, 1,  0, 0, 1,  0, 1, 0],
            [half_size+offset, half_size+offset, -half_size+offset,   1, 0, 0,  0, 1,  1, 1, 1, 1,  0, 0, 1,  0, 1, 0],
            [half_size+offset, half_size+offset, half_size+offset,    1, 0, 0,  1, 1,  1, 1, 1, 1,  0, 0, 1,  0, 1, 0],
            [half_size+offset, -half_size+offset, half_size+offset,   1, 0, 0,  1, 0,  1, 1, 1, 1,  0, 0, 1,  0, 1, 0],
        ])
        
        # Left face
        vertices.extend([
            [-half_size+offset, -half_size+offset, -half_size+offset,  -1, 0, 0,  1, 0,  1, 1, 1, 1,  0, 0, -1,  0, 1, 0],
            [-half_size+offset, -half_size+offset, half_size+offset,   -1, 0, 0,  0, 0,  1, 1, 1, 1,  0, 0, -1,  0, 1, 0],
            [-half_size+offset, half_size+offset, half_size+offset,    -1, 0, 0,  0, 1,  1, 1, 1, 1,  0, 0, -1,  0, 1, 0],
            [-half_size+offset, half_size+offset, -half_size+offset,   -1, 0, 0,  1, 1,  1, 1, 1, 1,  0, 0, -1,  0, 1, 0],
        ])
        
        # Convert to Vertex objects
        vertex_objects = []
        for v in vertices:
            vertex = Vertex(
                position=np.array(v[0:3], dtype=np.float32),
                normal=np.array(v[3:6], dtype=np.float32),
                texcoord=np.array(v[6:8], dtype=np.float32),
                color=np.array(v[8:12], dtype=np.float32),
                tangent=np.array(v[12:15], dtype=np.float32),
                bitangent=np.array(v[15:18], dtype=np.float32)
            )
            vertex_objects.append(vertex)
        
        # Cube indices (2 triangles per face)
        indices = []
        for i in range(6):
            base = i * 4
            indices.extend([base, base+1, base+2, base, base+2, base+3])
        
        # Create mesh
        mesh = Mesh(
            name=f"cube_{size}",
            vertices=vertex_objects,
            indices=indices
        )
        
        return mesh
    
    def _create_sphere_mesh(self, radius: float = 1.0, 
                           segments: int = 32) -> Mesh:
        """Create sphere mesh.
        
        Args:
            radius: Sphere radius
            segments: Number of segments (more = smoother)
            
        Returns:
            Sphere mesh
        """
        vertices = []
        indices = []
        
        # Generate sphere vertices
        for i in range(segments + 1):
            lat = math.pi * i / segments
            sin_lat = math.sin(lat)
            cos_lat = math.cos(lat)
            
            for j in range(segments + 1):
                lon = 2 * math.pi * j / segments
                sin_lon = math.sin(lon)
                cos_lon = math.cos(lon)
                
                # Position
                x = cos_lon * sin_lat * radius
                y = cos_lat * radius
                z = sin_lon * sin_lat * radius
                
                # Normal (normalized position)
                normal = np.array([x, y, z], dtype=np.float32)
                if radius > 0:
                    normal = normal / radius
                
                # Texcoord
                u = j / segments
                v = i / segments
                
                # Tangent (derivative of position with respect to u)
                tangent_x = -sin_lon * sin_lat * radius
                tangent_y = 0
                tangent_z = cos_lon * sin_lat * radius
                tangent = np.array([tangent_x, tangent_y, tangent_z], dtype=np.float32)
                tangent_len = np.linalg.norm(tangent)
                if tangent_len > 0:
                    tangent = tangent / tangent_len
                
                # Bitangent (cross product of normal and tangent)
                bitangent = np.cross(normal, tangent)
                
                # Color (white by default)
                color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
                
                vertex = Vertex(
                    position=np.array([x, y, z], dtype=np.float32),
                    normal=normal,
                    texcoord=np.array([u, v], dtype=np.float32),
                    color=color,
                    tangent=tangent,
                    bitangent=bitangent
                )
                vertices.append(vertex)
        
        # Generate indices
        for i in range(segments):
            for j in range(segments):
                first = i * (segments + 1) + j
                second = first + segments + 1
                
                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])
        
        # Create mesh
        mesh = Mesh(
            name=f"sphere_{radius}",
            vertices=vertices,
            indices=indices
        )
        
        return mesh
    
    def _create_plane_mesh(self, width: float = 10.0, 
                          height: float = 10.0,
                          segments_x: int = 10,
                          segments_y: int = 10) -> Mesh:
        """Create plane mesh.
        
        Args:
            width: Plane width
            height: Plane height
            segments_x: Number of segments along X
            segments_y: Number of segments along Y
            
        Returns:
            Plane mesh
        """
        vertices = []
        indices = []
        
        half_width = width * 0.5
        half_height = height * 0.5
        
        # Generate vertices
        for i in range(segments_y + 1):
            v = i / segments_y
            y = (v - 0.5) * height
            
            for j in range(segments_x + 1):
                u = j / segments_x
                x = (u - 0.5) * width
                
                # Position
                position = np.array([x, 0.0, y], dtype=np.float32)
                
                # Normal (up)
                normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                
                # Texcoord
                texcoord = np.array([u, v], dtype=np.float32)
                
                # Tangent (along X)
                tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                
                # Bitangent (along Z, since normal is up)
                bitangent = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                
                # Color
                color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
                
                vertex = Vertex(
                    position=position,
                    normal=normal,
                    texcoord=texcoord,
                    color=color,
                    tangent=tangent,
                    bitangent=bitangent
                )
                vertices.append(vertex)
        
        # Generate indices
        for i in range(segments_y):
            for j in range(segments_x):
                top_left = i * (segments_x + 1) + j
                top_right = top_left + 1
                bottom_left = (i + 1) * (segments_x + 1) + j
                bottom_right = bottom_left + 1
                
                # Two triangles per quad
                indices.extend([top_left, bottom_left, top_right])
                indices.extend([top_right, bottom_left, bottom_right])
        
        # Create mesh
        mesh = Mesh(
            name=f"plane_{width}x{height}",
            vertices=vertices,
            indices=indices
        )
        
        return mesh
    
    def _create_cylinder_mesh(self, radius: float = 0.5, 
                             height: float = 2.0,
                             segments: int = 32) -> Mesh:
        """Create cylinder mesh.
        
        Args:
            radius: Cylinder radius
            height: Cylinder height
            segments: Number of segments around circumference
            
        Returns:
            Cylinder mesh
        """
        vertices = []
        indices = []
        
        half_height = height * 0.5
        
        # Create side vertices
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            
            # Position
            x = cos_angle * radius
            z = sin_angle * radius
            
            # Normal (outward from center)
            normal = np.array([cos_angle, 0.0, sin_angle], dtype=np.float32)
            
            # Tangent (perpendicular to normal, along circumference)
            tangent = np.array([-sin_angle, 0.0, cos_angle], dtype=np.float32)
            
            # Bitangent (up)
            bitangent = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            
            # Texcoord
            u = i / segments
            v_top = 0.0
            v_bottom = 1.0
            
            # Color
            color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            
            # Top vertex
            vertex_top = Vertex(
                position=np.array([x, half_height, z], dtype=np.float32),
                normal=normal,
                texcoord=np.array([u, v_top], dtype=np.float32),
                color=color,
                tangent=tangent,
                bitangent=bitangent
            )
            vertices.append(vertex_top)
            
            # Bottom vertex
            vertex_bottom = Vertex(
                position=np.array([x, -half_height, z], dtype=np.float32),
                normal=normal,
                texcoord=np.array([u, v_bottom], dtype=np.float32),
                color=color,
                tangent=tangent,
                bitangent=bitangent
            )
            vertices.append(vertex_bottom)
        
        # Create side indices
        for i in range(segments):
            top_left = i * 2
            top_right = (i + 1) * 2
            bottom_left = i * 2 + 1
            bottom_right = (i + 1) * 2 + 1
            
            indices.extend([top_left, bottom_left, top_right])
            indices.extend([top_right, bottom_left, bottom_right])
        
        # Create top and bottom caps
        # Top center vertex
        top_center = Vertex(
            position=np.array([0.0, half_height, 0.0], dtype=np.float32),
            normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            texcoord=np.array([0.5, 0.5], dtype=np.float32),
            color=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            tangent=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            bitangent=np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )
        top_center_idx = len(vertices)
        vertices.append(top_center)
        
        # Bottom center vertex
        bottom_center = Vertex(
            position=np.array([0.0, -half_height, 0.0], dtype=np.float32),
            normal=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            texcoord=np.array([0.5, 0.5], dtype=np.float32),
            color=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            tangent=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            bitangent=np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )
        bottom_center_idx = len(vertices)
        vertices.append(bottom_center)
        
        # Create top and bottom triangles
        for i in range(segments):
            # Top triangles
            current_top = i * 2
            next_top = ((i + 1) % segments) * 2
            indices.extend([top_center_idx, current_top, next_top])
            
            # Bottom triangles
            current_bottom = i * 2 + 1
            next_bottom = ((i + 1) % segments) * 2 + 1
            indices.extend([bottom_center_idx, next_bottom, current_bottom])
        
        # Create mesh
        mesh = Mesh(
            name=f"cylinder_{radius}x{height}",
            vertices=vertices,
            indices=indices
        )
        
        return mesh
    
    def handle_input(self):
        """Handle input events."""
        if not self.window:
            return
        
        # Poll events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_close = True
            elif event.type == pygame.KEYDOWN:
                self.keyboard_state[event.key] = True
            elif event.type == pygame.KEYUP:
                self.keyboard_state[event.key] = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button <= 3:
                    self.mouse_buttons[event.button - 1] = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button <= 3:
                    self.mouse_buttons[event.button - 1] = False
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_position = event.pos
        
        # Camera controls
        speed = 0.1
        if pygame.K_w in self.keyboard_state and self.keyboard_state[pygame.K_w]:
            self.camera_pos += glm.normalize(self.camera_target - self.camera_pos) * speed
        if pygame.K_s in self.keyboard_state and self.keyboard_state[pygame.K_s]:
            self.camera_pos -= glm.normalize(self.camera_target - self.camera_pos) * speed
        if pygame.K_a in self.keyboard_state and self.keyboard_state[pygame.K_a]:
            self.camera_pos -= glm.normalize(glm.cross(self.camera_target - self.camera_pos, 
                                                     self.camera_up)) * speed
        if pygame.K_d in self.keyboard_state and self.keyboard_state[pygame.K_d]:
            self.camera_pos += glm.normalize(glm.cross(self.camera_target - self.camera_pos, 
                                                     self.camera_up)) * speed
        
        # Mouse look
        if self.mouse_buttons[2]:  # Right mouse button
            dx, dy = pygame.mouse.get_rel()
            sensitivity = 0.1
            
            # Calculate rotation
            forward = glm.normalize(self.camera_target - self.camera_pos)
            right = glm.normalize(glm.cross(forward, self.camera_up))
            up = glm.normalize(glm.cross(right, forward))
            
            # Yaw (around up vector)
            yaw = glm.rotate(glm.mat4(1.0), -dx * sensitivity, up)
            forward = glm.vec3(yaw * glm.vec4(forward, 0.0))
            
            # Pitch (around right vector)
            pitch = glm.rotate(glm.mat4(1.0), -dy * sensitivity, right)
            forward = glm.vec3(pitch * glm.vec4(forward, 0.0))
            
            # Update camera target
            self.camera_target = self.camera_pos + forward
        else:
            pygame.mouse.get_rel()  # Reset relative motion
    
    def run(self):
        """Main render loop."""
        if not self.is_initialized:
            self.initialize()
        
        # Initialize pygame for input
        pygame.init()
        pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.window_title)
        
        # Hide mouse cursor
        pygame.mouse.set_visible(False)
        
        print("Starting render loop...")
        print("Controls:")
        print("  W/S/A/D: Move camera")
        print("  Right mouse button + drag: Look around")
        print("  ESC: Exit")
        
        # Main loop
        while not self.should_close:
            # Handle input
            self.handle_input()
            
            # Render frame
            self.render_frame()
            
            # Print FPS every second
            if hasattr(self, '_last_fps_print') and time.time() - self._last_fps_print > 1.0:
                print(f"FPS: {self.fps:.1f}, Triangles: {self.triangle_count}, Draw calls: {self.draw_call_count}")
                self._last_fps_print = time.time()
            elif not hasattr(self, '_last_fps_print'):
                self._last_fps_print = time.time()
        
        # Cleanup
        self.cleanup()
        pygame.quit()
    
    def cleanup(self):
        """Cleanup resources."""
        print("Cleaning up raster engine...")
        
        # Release mesh resources
        for mesh in self.meshes:
            if mesh._vbo:
                mesh._vbo.release()
            if mesh._ibo:
                mesh._ibo.release()
            if mesh._vao:
                mesh._vao.release()
        
        # Release shaders
        for program in self.shader_programs.values():
            program.release()
        
        # Release textures
        for texture in self.textures.values():
            texture.release()
        
        if self.default_texture:
            self.default_texture.release()
        if self.default_normal_texture:
            self.default_normal_texture.release()
        
        # Release framebuffers
        if self.main_fbo:
            self.main_fbo.release()
        if self.shadow_fbo:
            self.shadow_fbo.release()
        
        # Release uniform buffers
        if self.camera_ubo:
            self.camera_ubo.release()
        if self.lights_ubo:
            self.lights_ubo.release()
        
        # Release blit resources
        if hasattr(self, '_blit_program'):
            self._blit_program.release()
        if hasattr(self, '_blit_vbo'):
            self._blit_vbo.release()
        if hasattr(self, '_blit_vao'):
            self._blit_vao.release()
        
        print("Raster engine cleanup complete")

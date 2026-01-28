"""
WebGL-based rendering engine for browser-based 3D visualization.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
import base64
import zlib

# Import JavaScript/WebGL modules
try:
    from js import WebGL2RenderingContext, Float32Array, Uint16Array, Uint8Array, document
    from pyodide.ffi import create_proxy
    HAS_WEBGL = True
except ImportError:
    HAS_WEBGL = False
    print("Running in non-browser environment, WebGL disabled")


class WebGLRenderMode(Enum):
    """WebGL rendering modes."""
    POINTS = "points"
    LINES = "lines"
    TRIANGLES = "triangles"
    TRIANGLE_STRIP = "triangle_strip"
    TRIANGLE_FAN = "triangle_fan"


@dataclass
class WebGLBuffer:
    """WebGL buffer wrapper."""
    buffer: Any = None
    target: int = 0
    usage: int = 0
    size: int = 0
    
    def __post_init__(self):
        """Set default WebGL constants if available."""
        if HAS_WEBGL:
            if self.target == 0:
                self.target = WebGL2RenderingContext.ARRAY_BUFFER
            if self.usage == 0:
                self.usage = WebGL2RenderingContext.STATIC_DRAW


@dataclass
class WebGLTexture:
    """WebGL texture wrapper."""
    texture: Any = None
    width: int = 0
    height: int = 0
    format: int = 0
    internal_format: int = 0
    type: int = 0
    
    def __post_init__(self):
        """Set default WebGL constants if available."""
        if HAS_WEBGL:
            if self.format == 0:
                self.format = WebGL2RenderingContext.RGBA
            if self.internal_format == 0:
                self.internal_format = WebGL2RenderingContext.RGBA
            if self.type == 0:
                self.type = WebGL2RenderingContext.UNSIGNED_BYTE


@dataclass
class WebGLProgram:
    """WebGL shader program wrapper."""
    program: Any = None
    vertex_shader: Any = None
    fragment_shader: Any = None
    uniforms: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)


class WebGLEngine:
    """Main WebGL rendering engine for browser environments."""
    
    def __init__(self, canvas_id: str = "webgl-canvas", 
                 width: int = 800, height: int = 600):
        """Initialize WebGL engine.
        
        Args:
            canvas_id: HTML canvas element ID
            width: Canvas width
            height: Canvas height
        """
        self.canvas_id = canvas_id
        self.width = width
        self.height = height
        
        # WebGL context
        self.gl: Optional[WebGL2RenderingContext] = None
        self.canvas: Optional[Any] = None
        
        # Resources
        self.buffers: Dict[str, WebGLBuffer] = {}
        self.textures: Dict[str, WebGLTexture] = {}
        self.programs: Dict[str, WebGLProgram] = {}
        self.vaos: Dict[str, Any] = {}
        
        # Rendering state
        self.clear_color = (0.1, 0.1, 0.1, 1.0)
        self.clear_depth = 1.0
        self.enable_depth_test = True
        self.enable_blending = False
        self.enable_culling = True
        self.cull_face = "back"
        
        # Camera
        self.view_matrix = np.eye(4, dtype=np.float32)
        self.projection_matrix = np.eye(4, dtype=np.float32)
        self.view_projection_matrix = np.eye(4, dtype=np.float32)
        
        # Scene data
        self.meshes: List[Dict[str, Any]] = []
        self.lights: List[Dict[str, Any]] = []
        
        # Callbacks
        self.on_init = None
        self.on_frame = None
        self.on_resize = None
        
        # Animation
        self.animation_frame_id = None
        self.last_time = 0
        self.delta_time = 0
        self.frame_count = 0
        
        # Performance
        self.fps = 0
        self.draw_calls = 0
        
    def initialize(self) -> bool:
        """Initialize WebGL context and resources.
        
        Returns:
            True if initialization successful
        """
        if not HAS_WEBGL:
            print("WebGL not available in this environment")
            return False
        
        try:
            # Get canvas element
            self.canvas = document.getElementById(self.canvas_id)
            if not self.canvas:
                print(f"Canvas element '{self.canvas_id}' not found")
                return False
            
            # Set canvas size
            self.canvas.width = self.width
            self.canvas.height = self.height
            
            # Get WebGL2 context
            self.gl = self.canvas.getContext("webgl2")
            if not self.gl:
                print("WebGL2 not supported")
                return False
            
            # Set initial WebGL state
            self._setup_gl_state()
            
            # Create default shaders
            self._create_default_shaders()
            
            # Create default geometry
            self._create_default_geometry()
            
            # Set up event listeners
            self._setup_event_listeners()
            
            print("WebGL engine initialized successfully")
            
            # Call initialization callback
            if self.on_init:
                self.on_init()
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize WebGL engine: {e}")
            return False
    
    def _setup_gl_state(self):
        """Setup initial WebGL state."""
        # Clear color and depth
        self.gl.clearColor(*self.clear_color)
        self.gl.clearDepth(self.clear_depth)
        
        # Enable depth testing
        if self.enable_depth_test:
            self.gl.enable(self.gl.DEPTH_TEST)
            self.gl.depthFunc(self.gl.LEQUAL)
        
        # Enable blending
        if self.enable_blending:
            self.gl.enable(self.gl.BLEND)
            self.gl.blendFunc(self.gl.SRC_ALPHA, self.gl.ONE_MINUS_SRC_ALPHA)
        
        # Enable face culling
        if self.enable_culling:
            self.gl.enable(self.gl.CULL_FACE)
            if self.cull_face == "back":
                self.gl.cullFace(self.gl.BACK)
            elif self.cull_face == "front":
                self.gl.cullFace(self.gl.FRONT)
            else:
                self.gl.cullFace(self.gl.FRONT_AND_BACK)
        
        # Set viewport
        self.gl.viewport(0, 0, self.width, self.height)
    
    def _create_default_shaders(self):
        """Create default shader programs."""
        # Basic vertex shader
        basic_vert = """
        #version 300 es
        precision highp float;
        
        in vec3 aPosition;
        in vec3 aNormal;
        in vec2 aTexCoord;
        in vec4 aColor;
        
        uniform mat4 uModelMatrix;
        uniform mat4 uViewMatrix;
        uniform mat4 uProjectionMatrix;
        uniform mat3 uNormalMatrix;
        
        out vec3 vPosition;
        out vec3 vNormal;
        out vec2 vTexCoord;
        out vec4 vColor;
        
        void main() {
            vec4 worldPosition = uModelMatrix * vec4(aPosition, 1.0);
            vPosition = worldPosition.xyz;
            vNormal = uNormalMatrix * aNormal;
            vTexCoord = aTexCoord;
            vColor = aColor;
            
            gl_Position = uProjectionMatrix * uViewMatrix * worldPosition;
        }
        """
        
        # Basic fragment shader
        basic_frag = """
        #version 300 es
        precision highp float;
        
        in vec3 vPosition;
        in vec3 vNormal;
        in vec2 vTexCoord;
        in vec4 vColor;
        
        uniform vec4 uColor;
        uniform sampler2D uTexture;
        uniform int uRenderMode;
        
        out vec4 fragColor;
        
        void main() {
            if (uRenderMode == 0) { // SOLID_COLOR
                fragColor = uColor;
            } else if (uRenderMode == 1) { // TEXTURED
                fragColor = texture(uTexture, vTexCoord) * uColor;
            } else if (uRenderMode == 2) { // NORMALS
                fragColor = vec4(normalize(vNormal) * 0.5 + 0.5, 1.0);
            } else if (uRenderMode == 3) { // DEPTH
                float depth = gl_FragCoord.z;
                fragColor = vec4(vec3(depth), 1.0);
            } else if (uRenderMode == 4) { // VERTEX_COLOR
                fragColor = vColor;
            } else {
                fragColor = vec4(1.0, 0.0, 1.0, 1.0); // Magenta fallback
            }
        }
        """
        
        # Compile basic shader
        basic_program = self._create_program("basic", basic_vert, basic_frag)
        if basic_program:
            self.programs["basic"] = basic_program
        
        # Gaussian splatting shader (will be loaded from external file)
        # This is a placeholder, actual shader loaded from gaussian_splat.glsl
        gaussian_vert = """
        #version 300 es
        precision highp float;
        
        // Gaussian splatting vertex shader placeholder
        // Actual shader loaded from external file
        
        void main() {
            gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        }
        """
        
        gaussian_frag = """
        #version 300 es
        precision highp float;
        
        out vec4 fragColor;
        
        void main() {
            fragColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
        """
        
        # Compile Gaussian shader (simplified)
        gaussian_program = self._create_program("gaussian", gaussian_vert, gaussian_frag)
        if gaussian_program:
            self.programs["gaussian"] = gaussian_program
    
    def _create_program(self, name: str, vertex_source: str, fragment_source: str) -> Optional[WebGLProgram]:
        """Create WebGL shader program.
        
        Args:
            name: Program name
            vertex_source: Vertex shader source
            fragment_source: Fragment shader source
            
        Returns:
            WebGLProgram if successful, None otherwise
        """
        if not self.gl:
            return None
        
        try:
            # Create shaders
            vertex_shader = self._compile_shader(self.gl.VERTEX_SHADER, vertex_source)
            fragment_shader = self._compile_shader(self.gl.FRAGMENT_SHADER, fragment_source)
            
            if not vertex_shader or not fragment_shader:
                return None
            
            # Create program
            program = self.gl.createProgram()
            self.gl.attachShader(program, vertex_shader)
            self.gl.attachShader(program, fragment_shader)
            self.gl.linkProgram(program)
            
            # Check linking status
            if not self.gl.getProgramParameter(program, self.gl.LINK_STATUS):
                error = self.gl.getProgramInfoLog(program)
                print(f"Failed to link program '{name}': {error}")
                return None
            
            # Get attribute and uniform locations
            program_wrapper = WebGLProgram(
                program=program,
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
            
            # Get active attributes
            num_attrs = self.gl.getProgramParameter(program, self.gl.ACTIVE_ATTRIBUTES)
            for i in range(num_attrs):
                info = self.gl.getActiveAttrib(program, i)
                if info:
                    location = self.gl.getAttribLocation(program, info.name)
                    program_wrapper.attributes[info.name] = location
            
            # Get active uniforms
            num_uniforms = self.gl.getProgramParameter(program, self.gl.ACTIVE_UNIFORMS)
            for i in range(num_uniforms):
                info = self.gl.getActiveUniform(program, i)
                if info:
                    location = self.gl.getUniformLocation(program, info.name)
                    program_wrapper.uniforms[info.name] = location
            
            print(f"Created shader program: {name}")
            return program_wrapper
            
        except Exception as e:
            print(f"Error creating program '{name}': {e}")
            return None
    
    def _compile_shader(self, shader_type: int, source: str) -> Optional[Any]:
        """Compile WebGL shader.
        
        Args:
            shader_type: Shader type (VERTEX_SHADER or FRAGMENT_SHADER)
            source: Shader source code
            
        Returns:
            Compiled shader, or None if failed
        """
        if not self.gl:
            return None
        
        shader = self.gl.createShader(shader_type)
        self.gl.shaderSource(shader, source)
        self.gl.compileShader(shader)
        
        # Check compilation status
        if not self.gl.getShaderParameter(shader, self.gl.COMPILE_STATUS):
            error = self.gl.getShaderInfoLog(shader)
            shader_type_str = "vertex" if shader_type == self.gl.VERTEX_SHADER else "fragment"
            print(f"Failed to compile {shader_type_str} shader: {error}")
            return None
        
        return shader
    
    def _create_default_geometry(self):
        """Create default geometry buffers."""
        # Create unit cube
        self._create_cube_geometry()
        
        # Create unit sphere
        self._create_sphere_geometry()
        
        # Create unit quad
        self._create_quad_geometry()
    
    def _create_cube_geometry(self):
        """Create unit cube geometry."""
        # Cube vertices (positions, normals, texcoords, colors)
        positions = [
            # Front face
            [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5], [0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
            # Back face
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [0.5,  0.5, -0.5], [0.5, -0.5, -0.5],
            # Top face
            [-0.5,  0.5, -0.5], [-0.5,  0.5,  0.5], [0.5,  0.5,  0.5], [0.5,  0.5, -0.5],
            # Bottom face
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5,  0.5], [-0.5, -0.5,  0.5],
            # Right face
            [0.5, -0.5, -0.5], [0.5,  0.5, -0.5], [0.5,  0.5,  0.5], [0.5, -0.5,  0.5],
            # Left face
            [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5]
        ]
        
        normals = [
            # Front
            [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
            # Back
            [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],
            # Top
            [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
            # Bottom
            [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],
            # Right
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            # Left
            [-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0]
        ]
        
        texcoords = [
            [0, 0], [1, 0], [1, 1], [0, 1],
            [1, 0], [1, 1], [0, 1], [0, 0],
            [0, 1], [0, 0], [1, 0], [1, 1],
            [1, 1], [0, 1], [0, 0], [1, 0],
            [1, 0], [1, 1], [0, 1], [0, 0],
            [0, 0], [1, 0], [1, 1], [0, 1]
        ]
        
        colors = [
            [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
            [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
            [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
            [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
            [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
            [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]
        ]
        
        indices = []
        for i in range(6):
            base = i * 4
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
        
        # Create interleaved vertex data
        vertex_data = []
        for i in range(len(positions)):
            vertex_data.extend(positions[i])
            vertex_data.extend(normals[i])
            vertex_data.extend(texcoords[i])
            vertex_data.extend(colors[i])
        
        # Create buffers
        self.create_buffer("cube_vertices", 
                          np.array(vertex_data, dtype=np.float32),
                          self.gl.ARRAY_BUFFER, self.gl.STATIC_DRAW)
        
        self.create_buffer("cube_indices",
                          np.array(indices, dtype=np.uint16),
                          self.gl.ELEMENT_ARRAY_BUFFER, self.gl.STATIC_DRAW)
        
        # Create VAO
        self._create_cube_vao()
    
    def _create_cube_vao(self):
        """Create Vertex Array Object for cube."""
        if not self.gl or "cube_vertices" not in self.buffers:
            return
        
        vao = self.gl.createVertexArray()
        self.gl.bindVertexArray(vao)
        
        # Bind vertex buffer
        vertex_buffer = self.buffers["cube_vertices"]
        self.gl.bindBuffer(self.gl.ARRAY_BUFFER, vertex_buffer.buffer)
        
        # Set up attribute pointers
        stride = 12 * 4  # 12 floats * 4 bytes per float
        
        # Position (3 floats)
        self.gl.vertexAttribPointer(0, 3, self.gl.FLOAT, False, stride, 0)
        self.gl.enableVertexAttribArray(0)
        
        # Normal (3 floats)
        self.gl.vertexAttribPointer(1, 3, self.gl.FLOAT, False, stride, 3 * 4)
        self.gl.enableVertexAttribArray(1)
        
        # TexCoord (2 floats)
        self.gl.vertexAttribPointer(2, 2, self.gl.FLOAT, False, stride, 6 * 4)
        self.gl.enableVertexAttribArray(2)
        
        # Color (4 floats)
        self.gl.vertexAttribPointer(3, 4, self.gl.FLOAT, False, stride, 8 * 4)
        self.gl.enableVertexAttribArray(3)
        
        # Bind index buffer
        index_buffer = self.buffers["cube_indices"]
        self.gl.bindBuffer(self.gl.ELEMENT_ARRAY_BUFFER, index_buffer.buffer)
        
        self.gl.bindVertexArray(None)
        
        self.vaos["cube"] = vao
    
    def _create_sphere_geometry(self, radius: float = 0.5, segments: int = 16):
        """Create sphere geometry.
        
        Args:
            radius: Sphere radius
            segments: Number of segments
        """
        positions = []
        normals = []
        texcoords = []
        colors = []
        
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
                
                positions.append([x, y, z])
                
                # Normal (normalized position)
                normal = np.array([x, y, z], dtype=np.float32)
                if radius > 0:
                    normal = normal / radius
                normals.append(normal.tolist())
                
                # TexCoord
                u = j / segments
                v = i / segments
                texcoords.append([u, v])
                
                # Color (white)
                colors.append([1.0, 1.0, 1.0, 1.0])
        
        # Generate indices
        indices = []
        for i in range(segments):
            for j in range(segments):
                first = i * (segments + 1) + j
                second = first + segments + 1
                
                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])
        
        # Create interleaved vertex data
        vertex_data = []
        for i in range(len(positions)):
            vertex_data.extend(positions[i])
            vertex_data.extend(normals[i])
            vertex_data.extend(texcoords[i])
            vertex_data.extend(colors[i])
        
        # Create buffers
        self.create_buffer("sphere_vertices",
                          np.array(vertex_data, dtype=np.float32),
                          self.gl.ARRAY_BUFFER, self.gl.STATIC_DRAW)
        
        self.create_buffer("sphere_indices",
                          np.array(indices, dtype=np.uint16),
                          self.gl.ELEMENT_ARRAY_BUFFER, self.gl.STATIC_DRAW)
        
        # Create VAO (similar to cube)
        self._create_sphere_vao()
    
    def _create_sphere_vao(self):
        """Create Vertex Array Object for sphere."""
        if not self.gl or "sphere_vertices" not in self.buffers:
            return
        
        vao = self.gl.createVertexArray()
        self.gl.bindVertexArray(vao)
        
        # Bind vertex buffer
        vertex_buffer = self.buffers["sphere_vertices"]
        self.gl.bindBuffer(self.gl.ARRAY_BUFFER, vertex_buffer.buffer)
        
        # Set up attribute pointers (same layout as cube)
        stride = 12 * 4
        
        self.gl.vertexAttribPointer(0, 3, self.gl.FLOAT, False, stride, 0)
        self.gl.enableVertexAttribArray(0)
        
        self.gl.vertexAttribPointer(1, 3, self.gl.FLOAT, False, stride, 3 * 4)
        self.gl.enableVertexAttribArray(1)
        
        self.gl.vertexAttribPointer(2, 2, self.gl.FLOAT, False, stride, 6 * 4)
        self.gl.enableVertexAttribArray(2)
        
        self.gl.vertexAttribPointer(3, 4, self.gl.FLOAT, False, stride, 8 * 4)
        self.gl.enableVertexAttribArray(3)
        
        # Bind index buffer
        index_buffer = self.buffers["sphere_indices"]
        self.gl.bindBuffer(self.gl.ELEMENT_ARRAY_BUFFER, index_buffer.buffer)
        
        self.gl.bindVertexArray(None)
        
        self.vaos["sphere"] = vao
    
    def _create_quad_geometry(self):
        """Create fullscreen quad geometry."""
        # Quad vertices (positions, texcoords)
        vertices = np.array([
            # positions   texcoords
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
            -1.0,  1.0,  0.0, 1.0,
             1.0,  1.0,  1.0, 1.0,
        ], dtype=np.float32)
        
        # Create buffer
        self.create_buffer("quad_vertices", vertices,
                          self.gl.ARRAY_BUFFER, self.gl.STATIC_DRAW)
        
        # Create VAO for quad
        vao = self.gl.createVertexArray()
        self.gl.bindVertexArray(vao)
        
        vertex_buffer = self.buffers["quad_vertices"]
        self.gl.bindBuffer(self.gl.ARRAY_BUFFER, vertex_buffer.buffer)
        
        # Position (2 floats)
        self.gl.vertexAttribPointer(0, 2, self.gl.FLOAT, False, 4 * 4, 0)
        self.gl.enableVertexAttribArray(0)
        
        # TexCoord (2 floats)
        self.gl.vertexAttribPointer(1, 2, self.gl.FLOAT, False, 4 * 4, 2 * 4)
        self.gl.enableVertexAttribArray(1)
        
        self.gl.bindVertexArray(None)
        
        self.vaos["quad"] = vao
    
    def _setup_event_listeners(self):
        """Setup event listeners for canvas."""
        if not self.canvas:
            return
        
        # Resize handler
        def handle_resize(event=None):
            if self.canvas:
                # Get computed style dimensions
                display_width = self.canvas.clientWidth
                display_height = self.canvas.clientHeight
                
                # Check if canvas needs resize
                if (self.canvas.width != display_width or 
                    self.canvas.height != display_height):
                    self.canvas.width = display_width
                    self.canvas.height = display_height
                    self.width = display_width
                    self.height = display_height
                    
                    # Update viewport
                    if self.gl:
                        self.gl.viewport(0, 0, display_width, display_height)
                    
                    # Update projection matrix
                    self._update_projection_matrix()
                    
                    # Call resize callback
                    if self.on_resize:
                        self.on_resize(display_width, display_height)
        
        # Mouse event handlers
        def handle_mouse_down(event):
            # Implement mouse interaction
            pass
        
        def handle_mouse_up(event):
            pass
        
        def handle_mouse_move(event):
            pass
        
        # Keyboard event handlers
        def handle_key_down(event):
            pass
        
        def handle_key_up(event):
            pass
        
        # Add event listeners
        window = document.defaultView
        
        # Use create_proxy to keep references
        self._resize_proxy = create_proxy(handle_resize)
        self._mouse_down_proxy = create_proxy(handle_mouse_down)
        self._mouse_up_proxy = create_proxy(handle_mouse_up)
        self._mouse_move_proxy = create_proxy(handle_mouse_move)
        self._key_down_proxy = create_proxy(handle_key_down)
        self._key_up_proxy = create_proxy(handle_key_up)
        
        window.addEventListener("resize", self._resize_proxy)
        self.canvas.addEventListener("mousedown", self._mouse_down_proxy)
        self.canvas.addEventListener("mouseup", self._mouse_up_proxy)
        self.canvas.addEventListener("mousemove", self._mouse_move_proxy)
        self.canvas.addEventListener("keydown", self._key_down_proxy)
        self.canvas.addEventListener("keyup", self._key_up_proxy)
        
        # Initial resize
        handle_resize()
    
    def _update_projection_matrix(self):
        """Update projection matrix based on canvas size."""
        aspect = self.width / self.height
        fov = math.radians(60.0)
        
        # Perspective projection
        f = 1.0 / math.tan(fov / 2.0)
        near = 0.1
        far = 1000.0
        
        self.projection_matrix = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
        
        self.view_projection_matrix = self.projection_matrix @ self.view_matrix
    
    def create_buffer(self, name: str, data: np.ndarray, 
                     target: int, usage: int) -> bool:
        """Create WebGL buffer.
        
        Args:
            name: Buffer name
            data: Buffer data
            target: Buffer target
            usage: Buffer usage
            
        Returns:
            True if successful
        """
        if not self.gl:
            return False
        
        try:
            buffer = self.gl.createBuffer()
            self.gl.bindBuffer(target, buffer)
            
            # Convert numpy array to JavaScript typed array
            if data.dtype == np.float32:
                js_data = Float32Array.new(data.tobytes())
            elif data.dtype == np.uint16:
                js_data = Uint16Array.new(data.tobytes())
            elif data.dtype == np.uint8:
                js_data = Uint8Array.new(data.tobytes())
            else:
                print(f"Unsupported data type: {data.dtype}")
                return False
            
            self.gl.bufferData(target, js_data, usage)
            self.gl.bindBuffer(target, None)
            
            self.buffers[name] = WebGLBuffer(
                buffer=buffer,
                target=target,
                usage=usage,
                size=data.nbytes
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to create buffer '{name}': {e}")
            return False
    
    def create_texture(self, name: str, width: int, height: int,
                      data: Optional[np.ndarray] = None,
                      format: int = None,
                      internal_format: int = None,
                      type: int = None) -> bool:
        """Create WebGL texture.
        
        Args:
            name: Texture name
            width: Texture width
            height: Texture height
            data: Texture data (optional)
            format: Texture format
            internal_format: Internal format
            type: Data type
            
        Returns:
            True if successful
        """
        if not self.gl:
            return False
        
        try:
            texture = self.gl.createTexture()
            self.gl.bindTexture(self.gl.TEXTURE_2D, texture)
            
            # Set default parameters
            self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_S, self.gl.CLAMP_TO_EDGE)
            self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_WRAP_T, self.gl.CLAMP_TO_EDGE)
            self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MIN_FILTER, self.gl.LINEAR)
            self.gl.texParameteri(self.gl.TEXTURE_2D, self.gl.TEXTURE_MAG_FILTER, self.gl.LINEAR)
            
            # Set format defaults
            if format is None:
                format = self.gl.RGBA
            if internal_format is None:
                internal_format = self.gl.RGBA
            if type is None:
                type = self.gl.UNSIGNED_BYTE
            
            # Upload texture data
            if data is not None:
                # Convert numpy array to JavaScript typed array
                if data.dtype == np.uint8:
                    js_data = Uint8Array.new(data.tobytes())
                elif data.dtype == np.float32:
                    js_data = Float32Array.new(data.tobytes())
                    type = self.gl.FLOAT
                else:
                    print(f"Unsupported texture data type: {data.dtype}")
                    return False
                
                self.gl.texImage2D(self.gl.TEXTURE_2D, 0, internal_format,
                                  width, height, 0, format, type, js_data)
            else:
                # Create empty texture
                self.gl.texImage2D(self.gl.TEXTURE_2D, 0, internal_format,
                                  width, height, 0, format, type, None)
            
            self.gl.bindTexture(self.gl.TEXTURE_2D, None)
            
            self.textures[name] = WebGLTexture(
                texture=texture,
                width=width,
                height=height,
                format=format,
                internal_format=internal_format,
                type=type
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to create texture '{name}': {e}")
            return False
    
    def load_texture_from_url(self, name: str, url: str) -> bool:
        """Load texture from URL.
        
        Args:
            name: Texture name
            url: Image URL
            
        Returns:
            True if successful
        """
        # This would use JavaScript's Image object to load texture
        # Implementation depends on browser environment
        print(f"Texture loading from URL not implemented in this environment: {url}")
        return False
    
    def load_shader_from_file(self, name: str, vertex_path: str, 
                             fragment_path: str) -> bool:
        """Load shader from files.
        
        Args:
            name: Shader name
            vertex_path: Vertex shader file path
            fragment_path: Fragment shader file path
            
        Returns:
            True if successful
        """
        # In browser environment, would fetch shader files
        # For now, implement as placeholder
        print(f"Shader loading from files not implemented: {vertex_path}, {fragment_path}")
        return False
    
    def set_camera(self, position: np.ndarray, target: np.ndarray, 
                  up: np.ndarray = None):
        """Set camera view matrix.
        
        Args:
            position: Camera position
            target: Look-at target
            up: Up vector
        """
        if up is None:
            up = np.array([0, 1, 0], dtype=np.float32)
        
        # Calculate view matrix
        forward = target - position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        new_up = np.cross(right, forward)
        
        # Create view matrix
        self.view_matrix = np.array([
            [right[0], new_up[0], -forward[0], 0],
            [right[1], new_up[1], -forward[1], 0],
            [right[2], new_up[2], -forward[2], 0],
            [-np.dot(right, position), -np.dot(new_up, position), np.dot(forward, position), 1]
        ], dtype=np.float32)
        
        self.view_projection_matrix = self.projection_matrix @ self.view_matrix
    
    def render_frame(self, current_time: float):
        """Render a single frame.
        
        Args:
            current_time: Current time in milliseconds
        """
        if not self.gl:
            return
        
        # Calculate delta time
        if self.last_time > 0:
            self.delta_time = (current_time - self.last_time) / 1000.0
            if self.delta_time > 0:
                self.fps = 1.0 / self.delta_time
        self.last_time = current_time
        self.frame_count += 1
        
        # Clear buffers
        self.gl.clear(self.gl.COLOR_BUFFER_BIT | self.gl.DEPTH_BUFFER_BIT)
        
        # Call frame callback
        if self.on_frame:
            self.on_frame(self.delta_time)
        
        # Reset draw call count
        self.draw_calls = 0
        
        # Render objects
        self._render_scene()
        
        # Update FPS display
        if self.frame_count % 60 == 0:
            self._update_fps_display()
    
    def _render_scene(self):
        """Render the current scene."""
        # Example: render a rotating cube
        if "cube" in self.vaos and "basic" in self.programs:
            program = self.programs["basic"]
            self.gl.useProgram(program.program)
            
            # Set up matrices
            model_matrix = np.eye(4, dtype=np.float32)
            
            # Apply rotation
            angle = self.frame_count * 0.01
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            
            rotation = np.array([
                [cos_a, 0, sin_a, 0],
                [0, 1, 0, 0],
                [-sin_a, 0, cos_a, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            
            model_matrix = rotation @ model_matrix
            
            # Normal matrix (transpose of inverse of model matrix)
            normal_matrix = np.linalg.inv(model_matrix[:3, :3]).T
            
            # Set uniforms
            self._set_uniform_matrix4fv(program, "uModelMatrix", model_matrix)
            self._set_uniform_matrix4fv(program, "uViewMatrix", self.view_matrix)
            self._set_uniform_matrix4fv(program, "uProjectionMatrix", self.projection_matrix)
            self._set_uniform_matrix3fv(program, "uNormalMatrix", normal_matrix)
            self._set_uniform4f(program, "uColor", 1.0, 0.5, 0.2, 1.0)
            self._set_uniform1i(program, "uRenderMode", 0)  # SOLID_COLOR
            
            # Bind texture if available
            if "default" in self.textures:
                self.gl.activeTexture(self.gl.TEXTURE0)
                self.gl.bindTexture(self.gl.TEXTURE_2D, self.textures["default"].texture)
                self._set_uniform1i(program, "uTexture", 0)
            
            # Draw cube
            vao = self.vaos["cube"]
            index_buffer = self.buffers["cube_indices"]
            
            self.gl.bindVertexArray(vao)
            self.gl.drawElements(self.gl.TRIANGLES, 36, self.gl.UNSIGNED_SHORT, 0)
            self.gl.bindVertexArray(None)
            
            self.draw_calls += 1
    
    def _set_uniform_matrix4fv(self, program: WebGLProgram, name: str, matrix: np.ndarray):
        """Set 4x4 matrix uniform.
        
        Args:
            program: Shader program
            name: Uniform name
            matrix: Matrix data
        """
        if name in program.uniforms:
            location = program.uniforms[name]
            # Transpose for WebGL (column-major)
            self.gl.uniformMatrix4fv(location, False, matrix.flatten())
    
    def _set_uniform_matrix3fv(self, program: WebGLProgram, name: str, matrix: np.ndarray):
        """Set 3x3 matrix uniform.
        
        Args:
            program: Shader program
            name: Uniform name
            matrix: Matrix data
        """
        if name in program.uniforms:
            location = program.uniforms[name]
            self.gl.uniformMatrix3fv(location, False, matrix.flatten())
    
    def _set_uniform4f(self, program: WebGLProgram, name: str, 
                      x: float, y: float, z: float, w: float):
        """Set vec4 uniform.
        
        Args:
            program: Shader program
            name: Uniform name
            x, y, z, w: Vector components
        """
        if name in program.uniforms:
            location = program.uniforms[name]
            self.gl.uniform4f(location, x, y, z, w)
    
    def _set_uniform1i(self, program: WebGLProgram, name: str, value: int):
        """Set integer uniform.
        
        Args:
            program: Shader program
            name: Uniform name
            value: Integer value
        """
        if name in program.uniforms:
            location = program.uniforms[name]
            self.gl.uniform1i(location, value)
    
    def _update_fps_display(self):
        """Update FPS display if available."""
        # Look for FPS display element
        fps_element = document.getElementById("fps-display")
        if fps_element:
            fps_element.textContent = f"FPS: {self.fps:.1f}"
    
    def start_animation(self):
        """Start animation loop."""
        if not self.gl:
            return
        
        def animate(timestamp):
            self.render_frame(timestamp)
            self.animation_frame_id = requestAnimationFrame(animate)
        
        # Start animation loop
        self.animation_frame_id = requestAnimationFrame(animate)
    
    def stop_animation(self):
        """Stop animation loop."""
        if self.animation_frame_id:
            cancelAnimationFrame(self.animation_frame_id)
            self.animation_frame_id = None
    
    def render_gaussian_splat(self, splat_data: np.ndarray):
        """Render Gaussian splatting data.
        
        Args:
            splat_data: Gaussian splat data (positions, scales, rotations, colors, opacities)
        """
        if "gaussian" not in self.programs:
            print("Gaussian splatting shader not available")
            return
        
        program = self.programs["gaussian"]
        self.gl.useProgram(program.program)
        
        # TODO: Implement Gaussian splatting rendering
        # This would involve:
        # 1. Creating buffers for splat data
        # 2. Setting up custom shader for splat rendering
        # 3. Using instanced rendering or point sprites
        
        print("Gaussian splatting rendering not fully implemented")
    
    def render_mesh(self, vertices: np.ndarray, indices: np.ndarray,
                   normals: np.ndarray = None, colors: np.ndarray = None,
                   texcoords: np.ndarray = None, material: Dict[str, Any] = None):
        """Render a mesh.
        
        Args:
            vertices: Vertex positions
            indices: Triangle indices
            normals: Vertex normals
            colors: Vertex colors
            texcoords: Texture coordinates
            material: Material properties
        """
        # Create temporary buffers for mesh
        mesh_id = f"mesh_{self.draw_calls}"
        
        # Create interleaved vertex buffer
        vertex_count = len(vertices) // 3
        
        # Prepare vertex data
        vertex_data = []
        for i in range(vertex_count):
            # Position
            vertex_data.extend(vertices[i*3:(i+1)*3])
            
            # Normal
            if normals is not None and i < len(normals) // 3:
                vertex_data.extend(normals[i*3:(i+1)*3])
            else:
                vertex_data.extend([0, 1, 0])  # Default up normal
            
            # TexCoord
            if texcoords is not None and i < len(texcoords) // 2:
                vertex_data.extend(texcoords[i*2:(i+1)*2])
            else:
                vertex_data.extend([0, 0])
            
            # Color
            if colors is not None and i < len(colors) // 4:
                vertex_data.extend(colors[i*4:(i+1)*4])
            else:
                vertex_data.extend([1, 1, 1, 1])
        
        # Create buffers
        self.create_buffer(f"{mesh_id}_vertices",
                          np.array(vertex_data, dtype=np.float32),
                          self.gl.ARRAY_BUFFER, self.gl.STATIC_DRAW)
        
        self.create_buffer(f"{mesh_id}_indices",
                          np.array(indices, dtype=np.uint16),
                          self.gl.ELEMENT_ARRAY_BUFFER, self.gl.STATIC_DRAW)
        
        # Create temporary VAO
        vao = self.gl.createVertexArray()
        self.gl.bindVertexArray(vao)
        
        vertex_buffer = self.buffers[f"{mesh_id}_vertices"]
        self.gl.bindBuffer(self.gl.ARRAY_BUFFER, vertex_buffer.buffer)
        
        # Set up attribute pointers
        stride = 12 * 4  # 3 pos + 3 normal + 2 texcoord + 4 color
        
        self.gl.vertexAttribPointer(0, 3, self.gl.FLOAT, False, stride, 0)
        self.gl.enableVertexAttribArray(0)
        
        self.gl.vertexAttribPointer(1, 3, self.gl.FLOAT, False, stride, 3 * 4)
        self.gl.enableVertexAttribArray(1)
        
        self.gl.vertexAttribPointer(2, 2, self.gl.FLOAT, False, stride, 6 * 4)
        self.gl.enableVertexAttribArray(2)
        
        self.gl.vertexAttribPointer(3, 4, self.gl.FLOAT, False, stride, 8 * 4)
        self.gl.enableVertexAttribArray(3)
        
        index_buffer = self.buffers[f"{mesh_id}_indices"]
        self.gl.bindBuffer(self.gl.ELEMENT_ARRAY_BUFFER, index_buffer.buffer)
        
        self.gl.bindVertexArray(None)
        
        # Render using basic shader
        if "basic" in self.programs:
            program = self.programs["basic"]
            self.gl.useProgram(program.program)
            
            # Set up matrices (identity model matrix for now)
            model_matrix = np.eye(4, dtype=np.float32)
            normal_matrix = np.eye(3, dtype=np.float32)
            
            self._set_uniform_matrix4fv(program, "uModelMatrix", model_matrix)
            self._set_uniform_matrix4fv(program, "uViewMatrix", self.view_matrix)
            self._set_uniform_matrix4fv(program, "uProjectionMatrix", self.projection_matrix)
            self._set_uniform_matrix3fv(program, "uNormalMatrix", normal_matrix)
            
            # Set material properties
            if material and "color" in material:
                color = material["color"]
                self._set_uniform4f(program, "uColor", 
                                  color[0], color[1], color[2], color[3] if len(color) > 3 else 1.0)
            else:
                self._set_uniform4f(program, "uColor", 0.8, 0.8, 0.8, 1.0)
            
            self._set_uniform1i(program, "uRenderMode", 4)  # VERTEX_COLOR
            
            # Draw mesh
            self.gl.bindVertexArray(vao)
            self.gl.drawElements(self.gl.TRIANGLES, len(indices), self.gl.UNSIGNED_SHORT, 0)
            self.gl.bindVertexArray(None)
            
            self.draw_calls += 1
        
        # Cleanup temporary buffers
        self.gl.deleteBuffer(vertex_buffer.buffer)
        self.gl.deleteBuffer(index_buffer.buffer)
        self.gl.deleteVertexArray(vao)
        
        del self.buffers[f"{mesh_id}_vertices"]
        del self.buffers[f"{mesh_id}_indices"]
    
    def apply_post_processing(self, shader_name: str = "post_process"):
        """Apply post-processing effect.
        
        Args:
            shader_name: Post-processing shader name
        """
        # TODO: Implement post-processing pipeline
        # This would involve:
        # 1. Rendering scene to framebuffer
        # 2. Applying post-processing shader to fullscreen quad
        # 3. Composite with final output
        
        print("Post-processing not fully implemented")
    
    def cleanup(self):
        """Cleanup WebGL resources."""
        if not self.gl:
            return
        
        print("Cleaning up WebGL resources...")
        
        # Stop animation
        self.stop_animation()
        
        # Delete buffers
        for name, buffer in self.buffers.items():
            if buffer.buffer:
                self.gl.deleteBuffer(buffer.buffer)
        
        # Delete textures
        for name, texture in self.textures.items():
            if texture.texture:
                self.gl.deleteTexture(texture.texture)
        
        # Delete programs
        for name, program in self.programs.items():
            if program.program:
                self.gl.deleteProgram(program.program)
            if program.vertex_shader:
                self.gl.deleteShader(program.vertex_shader)
            if program.fragment_shader:
                self.gl.deleteShader(program.fragment_shader)
        
        # Delete VAOs
        for name, vao in self.vaos.items():
            if vao:
                self.gl.deleteVertexArray(vao)
        
        # Clear dictionaries
        self.buffers.clear()
        self.textures.clear()
        self.programs.clear()
        self.vaos.clear()
        
        print("WebGL engine cleanup complete")

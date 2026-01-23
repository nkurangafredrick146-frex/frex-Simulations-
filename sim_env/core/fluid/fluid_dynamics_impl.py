"""
Complete Fluid Dynamics Simulation Module
Advanced fluid simulation using Smoothed Particle Hydrodynamics (SPH) and grid-based methods
"""

import numpy as np
import glm

# Guard OpenGL imports so the module can be imported in headless environments.
HAVE_OPENGL = True
try:
    from OpenGL.GL import *
    from OpenGL.GL.shaders import *
except Exception:
    HAVE_OPENGL = False

import numba
from numba import jit, prange
import math
import time
from typing import Dict, List, Any, Optional, Tuple
import random

class SPHParameters:
    """Parameters for Smoothed Particle Hydrodynamics simulation"""
    
    def __init__(self):
        # Fluid properties
        self.rest_density = 1000.0  # kg/m³
        self.gas_constant = 2000.0  # Equation of state constant
        self.viscosity = 0.1       # Dynamic viscosity
        self.surface_tension = 0.072  # Surface tension coefficient
        
        # SPH kernel parameters
        self.kernel_radius = 0.2   # Smoothing kernel radius
        self.particle_radius = 0.08  # Particle rendering radius
        self.particle_mass = 1.0   # Particle mass
        
        # Simulation parameters
        self.time_step = 0.016     # Simulation time step
        self.substeps = 1          # Simulation substeps
        self.gravity = glm.vec3(0, -9.81, 0)
        
        # Boundary conditions
        self.boundary_stiffness = 10000.0
        self.boundary_damping = 256.0
        
        # Advanced parameters
        self.enable_vorticity = True
        self.enable_surface_tension = True
        self.vorticity_confinement = 0.1
        self.artificial_pressure = 0.001
        self.artificial_viscosity = 0.01

@numba.jit(nopython=True, fastmath=True)
def poly6_kernel(r_sq: float, h: float) -> float:
    """Poly6 smoothing kernel for density estimation"""
    if r_sq >= h * h:
        return 0.0
    h_sq = h * h
    return (315.0 / (64.0 * math.pi * h_sq * h_sq * h)) * (h_sq - r_sq) ** 3

@numba.jit(nopython=True, fastmath=True)
def spiky_kernel_gradient(r_vec: np.ndarray, r_len: float, h: float) -> np.ndarray:
    """Spiky kernel gradient for pressure forces"""
    if r_len >= h or r_len < 1e-5:
        return np.zeros(3, dtype=np.float32)
    
    scale = -45.0 / (math.pi * h ** 6) * (h - r_len) ** 2
    return (r_vec / r_len) * scale

@numba.jit(nopython=True, fastmath=True)
def viscosity_kernel_laplacian(r_len: float, h: float) -> float:
    """Viscosity kernel laplacian for viscous forces"""
    if r_len >= h:
        return 0.0
    return 45.0 / (math.pi * h ** 6) * (h - r_len)

@numba.jit(nopython=True, parallel=True)
def calculate_density_pressure(positions: np.ndarray, densities: np.ndarray, 
                              pressures: np.ndarray, mass: float, rest_density: float,
                              gas_constant: float, kernel_radius: float, num_particles: int):
    """Calculate density and pressure for all particles using SPH"""
    h = kernel_radius
    h_sq = h * h
    
    for i in prange(num_particles):
        density = 0.0
        
        for j in range(num_particles):
            if i == j:
                continue
                
            # Calculate distance squared
            dx = positions[i*3] - positions[j*3]
            dy = positions[i*3+1] - positions[j*3+1]
            dz = positions[i*3+2] - positions[j*3+2]
            r_sq = dx*dx + dy*dy + dz*dz
            
            if r_sq < h_sq:
                # Add contribution to density
                density += mass * poly6_kernel(r_sq, h)
        
        # Self-density contribution
        density += mass * poly6_kernel(0.0, h)
        densities[i] = density
        
        # Calculate pressure using equation of state
        pressures[i] = gas_constant * (density - rest_density)

@numba.jit(nopython=True, parallel=True)
def calculate_pressure_forces(positions: np.ndarray, velocities: np.ndarray, 
                             forces: np.ndarray, densities: np.ndarray, 
                             pressures: np.ndarray, mass: float, kernel_radius: float, 
                             num_particles: int, artificial_pressure: float):
    """Calculate pressure forces for all particles"""
    h = kernel_radius
    
    for i in prange(num_particles):
        pressure_force_x = 0.0
        pressure_force_y = 0.0
        pressure_force_z = 0.0
        
        if densities[i] < 1e-5:
            continue
            
        for j in range(num_particles):
            if i == j:
                continue
                
            if densities[j] < 1e-5:
                continue
                
            # Calculate distance vector
            dx = positions[i*3] - positions[j*3]
            dy = positions[i*3+1] - positions[j*3+1]
            dz = positions[i*3+2] - positions[j*3+2]
            r_len = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if 0 < r_len < h:
                r_vec = np.array([dx, dy, dz], dtype=np.float32)
                
                # Calculate pressure force
                pressure_accel = -mass * (pressures[i] + pressures[j]) / (2 * densities[j])
                kernel_grad = spiky_kernel_gradient(r_vec, r_len, h)
                
                pressure_force_x += pressure_accel * kernel_grad[0]
                pressure_force_y += pressure_accel * kernel_grad[1]
                pressure_force_z += pressure_accel * kernel_grad[2]
                
                # Artificial pressure to prevent particle clustering
                if artificial_pressure > 0:
                    r_norm = r_len / h
                    artificial_factor = artificial_pressure * (1 - r_norm) ** 3
                    pressure_force_x += artificial_factor * (dx / r_len)
                    pressure_force_y += artificial_factor * (dy / r_len)
                    pressure_force_z += artificial_factor * (dz / r_len)
        
        forces[i*3] += pressure_force_x
        forces[i*3+1] += pressure_force_y
        forces[i*3+2] += pressure_force_z

@numba.jit(nopython=True, parallel=True)
def calculate_viscosity_forces(positions: np.ndarray, velocities: np.ndarray,
                              forces: np.ndarray, densities: np.ndarray,
                              mass: float, viscosity: float, kernel_radius: float,
                              num_particles: int, artificial_viscosity: float):
    """Calculate viscosity forces for all particles"""
    h = kernel_radius
    
    for i in prange(num_particles):
        viscosity_force_x = 0.0
        viscosity_force_y = 0.0
        viscosity_force_z = 0.0
        
        if densities[i] < 1e-5:
            continue
            
        for j in range(num_particles):
            if i == j:
                continue
                
            if densities[j] < 1e-5:
                continue
                
            # Calculate distance
            dx = positions[i*3] - positions[j*3]
            dy = positions[i*3+1] - positions[j*3+1]
            dz = positions[i*3+2] - positions[j*3+2]
            r_len = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if 0 < r_len < h:
                # Velocity difference
                dvx = velocities[i*3] - velocities[j*3]
                dvy = velocities[i*3+1] - velocities[j*3+1]
                dvz = velocities[i*3+2] - velocities[j*3+2]
                
                # Calculate viscosity force
                viscosity_accel = viscosity * mass / densities[j]
                kernel_lap = viscosity_kernel_laplacian(r_len, h)
                
                viscosity_force_x += viscosity_accel * dvx * kernel_lap
                viscosity_force_y += viscosity_accel * dvy * kernel_lap
                viscosity_force_z += viscosity_accel * dvz * kernel_lap
                
                # Artificial viscosity for stability
                if artificial_viscosity > 0:
                    dot_product = dx*dvx + dy*dvy + dz*dvz
                    if dot_product < 0:  # Approaching particles
                        artificial_factor = artificial_viscosity * dot_product / (r_len * densities[j])
                        viscosity_force_x += artificial_factor * dx
                        viscosity_force_y += artificial_factor * dy
                        viscosity_force_z += artificial_factor * dz
        
        forces[i*3] += viscosity_force_x
        forces[i*3+1] += viscosity_force_y
        forces[i*3+2] += viscosity_force_z

@numba.jit(nopython=True, parallel=True)
def calculate_vorticity_forces(positions: np.ndarray, velocities: np.ndarray,
                              forces: np.ndarray, densities: np.ndarray,
                              mass: float, kernel_radius: float, num_particles: int,
                              vorticity_confinement: float):
    """Calculate vorticity confinement forces for more turbulent flows"""
    h = kernel_radius
    
    for i in prange(num_particles):
        vorticity_x = 0.0
        vorticity_y = 0.0
        vorticity_z = 0.0
        
        if densities[i] < 1e-5:
            continue
            
        for j in range(num_particles):
            if i == j:
                continue
                
            if densities[j] < 1e-5:
                continue
                
            # Calculate distance vector
            dx = positions[i*3] - positions[j*3]
            dy = positions[i*3+1] - positions[j*3+1]
            dz = positions[i*3+2] - positions[j*3+2]
            r_len = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if 0 < r_len < h:
                # Velocity difference
                dvx = velocities[i*3] - velocities[j*3]
                dvy = velocities[i*3+1] - velocities[j*3+1]
                dvz = velocities[i*3+2] - velocities[j*3+2]
                
                # Cross product for vorticity
                vorticity_x += dy * dvz - dz * dvy
                vorticity_y += dz * dvx - dx * dvz
                vorticity_z += dx * dvy - dy * dvx
        
        # Vorticity confinement force
        vorticity_len = math.sqrt(vorticity_x*vorticity_x + vorticity_y*vorticity_y + vorticity_z*vorticity_z)
        if vorticity_len > 1e-5:
            # Normalize vorticity
            vorticity_x /= vorticity_len
            vorticity_y /= vorticity_len
            vorticity_z /= vorticity_len
            
            # Apply confinement force
            confinement_force = vorticity_confinement * vorticity_len
            forces[i*3] += vorticity_x * confinement_force
            forces[i*3+1] += vorticity_y * confinement_force
            forces[i*3+2] += vorticity_z * confinement_force

class SpatialGrid:
    """Spatial grid for efficient neighbor search in SPH"""
    
    def __init__(self, cell_size: float, world_size: Tuple[float, float, float]):
        self.cell_size = cell_size
        self.world_size = world_size
        self.grid = {}
        self.particle_cells = {}
        
    def clear(self):
        """Clear the spatial grid"""
        self.grid.clear()
        self.particle_cells.clear()
        
    def get_cell_index(self, position: glm.vec3) -> Tuple[int, int, int]:
        """Get grid cell index for a position"""
        x = int(position.x / self.cell_size)
        y = int(position.y / self.cell_size)
        z = int(position.z / self.cell_size)
        return (x, y, z)
        
    def add_particle(self, particle_id: int, position: glm.vec3):
        """Add particle to spatial grid"""
        cell_index = self.get_cell_index(position)
        
        if cell_index not in self.grid:
            self.grid[cell_index] = []
        self.grid[cell_index].append(particle_id)
        self.particle_cells[particle_id] = cell_index
        
    def update_particle(self, particle_id: int, old_position: glm.vec3, new_position: glm.vec3):
        """Update particle position in spatial grid"""
        old_cell = self.get_cell_index(old_position)
        new_cell = self.get_cell_index(new_position)
        
        if old_cell != new_cell:
            # Remove from old cell
            if old_cell in self.grid and particle_id in self.grid[old_cell]:
                self.grid[old_cell].remove(particle_id)
                
            # Add to new cell
            if new_cell not in self.grid:
                self.grid[new_cell] = []
            self.grid[new_cell].append(particle_id)
            self.particle_cells[particle_id] = new_cell
            
    def get_neighbors(self, position: glm.vec3, radius: float) -> List[int]:
        """Get all particles within radius of position"""
        neighbors = []
        center_cell = self.get_cell_index(position)
        search_radius = int(math.ceil(radius / self.cell_size))
        
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                for dz in range(-search_radius, search_radius + 1):
                    cell_index = (center_cell[0] + dx, center_cell[1] + dy, center_cell[2] + dz)
                    
                    if cell_index in self.grid:
                        neighbors.extend(self.grid[cell_index])
                        
        return neighbors

class FluidParticle:
    """Extended particle class for fluid simulation"""
    
    def __init__(self, position: glm.vec3, velocity: glm.vec3 = None, 
                 mass: float = 1.0, radius: float = 0.08):
        self.position = glm.vec3(position)
        self.velocity = velocity if velocity else glm.vec3(0, 0, 0)
        self.force = glm.vec3(0, 0, 0)
        self.mass = mass
        self.radius = radius
        
        # Fluid properties
        self.density = 0.0
        self.pressure = 0.0
        self.color = glm.vec3(0.2, 0.4, 1.0)
        
        # Visualization
        self.age = 0.0
        self.lifetime = float('inf')
        
    def update(self, dt: float):
        """Update particle state"""
        # Store previous position for Verlet integration
        self.previous_position = glm.vec3(self.position)
        
        # Integrate using semi-implicit Euler
        acceleration = self.force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Reset forces
        self.force = glm.vec3(0, 0, 0)
        
    def is_alive(self) -> bool:
        """Check if particle is still alive"""
        return self.age < self.lifetime

class FluidRenderer:
    """Advanced fluid rendering system"""
    
    def __init__(self):
        self.shader_program = None
        self.vao = 0
        self.vbo = 0
        self.fluid_texture = 0
        
        self.initialize_opengl()
        
    def initialize_opengl(self):
        """Initialize OpenGL resources for fluid rendering"""
        # Fluid rendering shader
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;
        layout (location = 2) in float aDensity;
        
        out vec3 fragColor;
        out float density;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform vec3 lightPos;
        uniform vec3 viewPos;
        
        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            fragColor = aColor;
            density = aDensity;
            
            // Point size based on density and distance
            vec4 eyePos = view * model * vec4(aPos, 1.0);
            float distance = length(eyePos.xyz);
            gl_PointSize = 8.0 * (1.0 - min(distance / 20.0, 0.8)) * (0.5 + density * 0.5);
        }
        """
        
        fragment_shader = """
        #version 330 core
        in vec3 fragColor;
        in float density;
        out vec4 FragColor;
        
        uniform float time;
        uniform vec3 lightDir;
        
        void main() {
            // Make particles circular with smooth edges
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);
            
            if (dist > 0.5)
                discard;
                
            // Fluid-like coloring based on density
            vec3 waterColor = mix(vec3(0.1, 0.3, 0.8), vec3(0.3, 0.6, 1.0), density);
            
            // Lighting calculation
            vec3 normal = vec3(coord * 2.0, sqrt(1.0 - dot(coord, coord)));
            float diff = max(dot(normal, lightDir), 0.2);
            
            // Foam effect at surface (low density)
            float foam = 1.0 - smoothstep(0.3, 0.7, density);
            waterColor = mix(waterColor, vec3(1.0, 1.0, 1.0), foam * 0.3);
            
            // Alpha based on density and distance from center
            float alpha = (0.5 + density * 0.5) * (1.0 - dist * 1.5);
            
            FragColor = vec4(waterColor * diff, alpha);
        }
        """
        
        self.shader_program = compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )
        
        # Create buffers
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), ctypes.c_void_p(3 * sizeof(GLfloat)))
        glEnableVertexAttribArray(1)
        
        # Density attribute
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), ctypes.c_void_p(6 * sizeof(GLfloat)))
        glEnableVertexAttribArray(2)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        
    def render_fluid(self, particles: List[FluidParticle], view_matrix: glm.mat4, 
                    projection_matrix: glm.mat4, camera_position: glm.vec3, time: float):
        """Render fluid particles"""
        if not particles:
            return
            
        # Prepare particle data
        particle_data = []
        for particle in particles:
            # Normalize density for coloring
            normalized_density = min(particle.density / 2000.0, 1.0)
            particle_data.extend([
                particle.position.x, particle.position.y, particle.position.z,
                particle.color.x, particle.color.y, particle.color.z,
                normalized_density
            ])
            
        particle_data = np.array(particle_data, dtype=np.float32)
        
        # Upload to GPU
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, particle_data.nbytes, particle_data, GL_DYNAMIC_DRAW)
        
        # Render
        glUseProgram(self.shader_program)
        glBindVertexArray(self.vao)
        
        # Set uniforms
        model_matrix = glm.mat4(1.0)
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model_matrix))
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "view"), 1, GL_FALSE, glm.value_ptr(view_matrix))
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "projection"), 1, GL_FALSE, glm.value_ptr(projection_matrix))
        
        glUniform3f(glGetUniformLocation(self.shader_program, "lightDir"), 0.5, -1.0, 0.5)
        glUniform3f(glGetUniformLocation(self.shader_program, "viewPos"), 
                   camera_position.x, camera_position.y, camera_position.z)
        glUniform1f(glGetUniformLocation(self.shader_program, "time"), time)
        
        # Set rendering state for fluid
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_DEPTH_TEST)
        
        # Render particles
        glDrawArrays(GL_POINTS, 0, len(particles))
        
        # Cleanup
        glBindVertexArray(0)
        glUseProgram(0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

class FluidDynamicsSimulation:
    """Complete fluid dynamics simulation using SPH"""
    
    def __init__(self, config: SPHParameters = None):
        self.config = config if config else SPHParameters()
        
        # Simulation state
        self.particles = []
        self.spatial_grid = SpatialGrid(self.config.kernel_radius, (10.0, 10.0, 10.0))
        self.fluid_renderer = FluidRenderer()
        
        # Performance tracking
        self.simulation_time = 0.0
        self.frame_count = 0
        self.performance_stats = {}
        
        # Boundaries
        self.boundaries = self.create_default_boundaries()
        
        # Initialization
        self.initialized = False
        
    def create_default_boundaries(self) -> List[Dict]:
        """Create default simulation boundaries"""
        boundaries = [
            # Ground plane
            {'type': 'plane', 'normal': glm.vec3(0, 1, 0), 'point': glm.vec3(0, -3, 0), 'stiffness': 10000.0},
            # Walls
            {'type': 'plane', 'normal': glm.vec3(1, 0, 0), 'point': glm.vec3(-4, 0, 0), 'stiffness': 10000.0},
            {'type': 'plane', 'normal': glm.vec3(-1, 0, 0), 'point': glm.vec3(4, 0, 0), 'stiffness': 10000.0},
            {'type': 'plane', 'normal': glm.vec3(0, 0, 1), 'point': glm.vec3(0, 0, -4), 'stiffness': 10000.0},
            {'type': 'plane', 'normal': glm.vec3(0, 0, -1), 'point': glm.vec3(0, 0, 4), 'stiffness': 10000.0},
            # Ceiling
            {'type': 'plane', 'normal': glm.vec3(0, -1, 0), 'point': glm.vec3(0, 3, 0), 'stiffness': 10000.0}
        ]
        return boundaries
        
    def initialize(self, num_particles: int = 1000):
        """Initialize fluid simulation with particles"""
        print(f"Initializing fluid simulation with {num_particles} particles...")
        
        # Create fluid volume
        self.create_fluid_volume(glm.vec3(-1, -1, -1), glm.vec3(2, 2, 2), num_particles)
        
        # Initialize spatial grid
        self.update_spatial_grid()
        
        self.initialized = True
        print("Fluid simulation initialized successfully")
        
    def create_fluid_volume(self, min_corner: glm.vec3, size: glm.vec3, num_particles: int):
        """Create a volume of fluid particles"""
        particle_spacing = 0.15
        count_x = max(1, int(size.x / particle_spacing))
        count_y = max(1, int(size.y / particle_spacing))
        count_z = max(1, int(size.z / particle_spacing))
        
        actual_count = count_x * count_y * count_z
        particles_to_create = min(num_particles, actual_count)
        
        print(f"Creating {particles_to_create} particles in {count_x}x{count_y}x{count_z} grid")
        
        # Clear existing particles
        self.particles.clear()
        
        # Create particles in grid pattern
        for i in range(count_x):
            for j in range(count_y):
                for k in range(count_z):
                    if len(self.particles) >= particles_to_create:
                        break
                        
                    x = min_corner.x + i * particle_spacing
                    y = min_corner.y + j * particle_spacing
                    z = min_corner.z + k * particle_spacing
                    
                    position = glm.vec3(x, y, z)
                    
                    # Add some random variation
                    position += glm.vec3(
                        random.uniform(-0.02, 0.02),
                        random.uniform(-0.02, 0.02),
                        random.uniform(-0.02, 0.02)
                    )
                    
                    particle = FluidParticle(
                        position=position,
                        velocity=glm.vec3(0, 0, 0),
                        mass=self.config.particle_mass,
                        radius=self.config.particle_radius
                    )
                    
                    self.particles.append(particle)
                    
        print(f"Created {len(self.particles)} fluid particles")
        
    def update_spatial_grid(self):
        """Update spatial grid with current particle positions"""
        self.spatial_grid.clear()
        for i, particle in enumerate(self.particles):
            self.spatial_grid.add_particle(i, particle.position)
            
    def update(self, dt: float):
        """Update fluid simulation"""
        if not self.initialized or not self.particles:
            return
            
        start_time = time.time()
        self.simulation_time += dt
        self.frame_count += 1
        
        # Multiple substeps for stability
        substep_dt = dt / self.config.substeps
        for substep in range(self.config.substeps):
            self.update_sph(substep_dt)
            
        # Update performance stats
        self.performance_stats = {
            'simulation_time': self.simulation_time,
            'particle_count': len(self.particles),
            'frame_time': time.time() - start_time,
            'substeps': self.config.substeps
        }
        
    def update_sph(self, dt: float):
        """Update SPH simulation for one time step"""
        num_particles = len(self.particles)
        if num_particles == 0:
            return
            
        # Convert to numpy arrays for numba acceleration
        positions = np.zeros(num_particles * 3, dtype=np.float32)
        velocities = np.zeros(num_particles * 3, dtype=np.float32)
        densities = np.zeros(num_particles, dtype=np.float32)
        pressures = np.zeros(num_particles, dtype=np.float32)
        forces = np.zeros(num_particles * 3, dtype=np.float32)
        
        # Copy data to numpy arrays
        for i, particle in enumerate(self.particles):
            positions[i*3] = particle.position.x
            positions[i*3+1] = particle.position.y
            positions[i*3+2] = particle.position.z
            
            velocities[i*3] = particle.velocity.x
            velocities[i*3+1] = particle.velocity.y
            velocities[i*3+2] = particle.velocity.z
            
        # SPH calculations
        calculate_density_pressure(
            positions, densities, pressures,
            self.config.particle_mass, self.config.rest_density,
            self.config.gas_constant, self.config.kernel_radius, num_particles
        )
        
        calculate_pressure_forces(
            positions, velocities, forces, densities, pressures,
            self.config.particle_mass, self.config.kernel_radius, num_particles,
            self.config.artificial_pressure
        )
        
        calculate_viscosity_forces(
            positions, velocities, forces, densities,
            self.config.particle_mass, self.config.viscosity, self.config.kernel_radius,
            num_particles, self.config.artificial_viscosity
        )
        
        if self.config.enable_vorticity:
            calculate_vorticity_forces(
                positions, velocities, forces, densities,
                self.config.particle_mass, self.config.kernel_radius, num_particles,
                self.config.vorticity_confinement
            )
        
        # Apply forces and update particles
        for i, particle in enumerate(self.particles):
            # Apply calculated forces
            particle.force.x += forces[i*3]
            particle.force.y += forces[i*3+1]
            particle.force.z += forces[i*3+2]
            
            # Apply gravity
            particle.force += self.config.gravity * particle.mass
            
            # Apply boundary conditions
            self.apply_boundary_forces(particle)
            
            # Update density and pressure
            particle.density = densities[i]
            particle.pressure = pressures[i]
            
            # Update particle state
            particle.update(dt)
            
        # Update spatial grid
        self.update_spatial_grid()
        
    def apply_boundary_forces(self, particle: FluidParticle):
        """Apply boundary condition forces"""
        for boundary in self.boundaries:
            if boundary['type'] == 'plane':
                self.apply_plane_boundary(particle, boundary)
                
    def apply_plane_boundary(self, particle: FluidParticle, boundary: Dict):
        """Apply plane boundary condition"""
        normal = boundary['normal']
        point = boundary['point']
        stiffness = boundary['stiffness']
        
        # Calculate distance to plane
        distance = glm.dot(normal, particle.position - point)
        
        if distance < particle.radius:
            # Particle is penetrating boundary
            penetration = particle.radius - distance
            if penetration > 0:
                # Apply repulsion force
                force = normal * stiffness * penetration
                particle.force += force
                
                # Damping
                normal_velocity = glm.dot(particle.velocity, normal)
                if normal_velocity < 0:  # Moving toward boundary
                    damping_force = -normal * normal_velocity * self.config.boundary_damping
                    particle.force += damping_force
                    
    def render(self, view_matrix: glm.mat4 = None, projection_matrix: glm.mat4 = None):
        """Render fluid simulation"""
        if not self.initialized:
            return
            
        # Use default matrices if not provided
        if view_matrix is None:
            view_matrix = glm.lookAt(
                glm.vec3(0, 0, 8),
                glm.vec3(0, 0, 0),
                glm.vec3(0, 1, 0)
            )
            
        if projection_matrix is None:
            projection_matrix = glm.perspective(
                glm.radians(45.0), 16/9, 0.1, 100.0
            )
            
        camera_position = glm.vec3(0, 0, 8)  # Simplified
        
        self.fluid_renderer.render_fluid(
            self.particles, view_matrix, projection_matrix, 
            camera_position, self.simulation_time
        )
        
    def add_fluid_volume(self, position: glm.vec3, size: glm.vec3, num_particles: int):
        """Add a new volume of fluid particles"""
        for i in range(num_particles):
            if len(self.particles) >= 5000:  # Limit total particles
                break
                
            # Random position within volume
            pos = position + glm.vec3(
                random.uniform(-size.x/2, size.x/2),
                random.uniform(-size.y/2, size.y/2),
                random.uniform(-size.z/2, size.z/2)
            )
            
            particle = FluidParticle(
                position=pos,
                velocity=glm.vec3(0, 0, 0),
                mass=self.config.particle_mass,
                radius=self.config.particle_radius
            )
            
            self.particles.append(particle)
            
        self.update_spatial_grid()
        
    def apply_force(self, position: glm.vec3, force: glm.vec3, radius: float = 1.0):
        """Apply force to particles within radius"""
        for particle in self.particles:
            distance = glm.distance(particle.position, position)
            if distance < radius:
                # Force decreases with distance
                strength = 1.0 - (distance / radius)
                particle.force += force * strength
                
    def create_splash(self, position: glm.vec3, intensity: float = 1.0):
        """Create a splash effect"""
        splash_particles = 20
        for i in range(splash_particles):
            # Random direction with upward bias
            direction = glm.normalize(glm.vec3(
                random.uniform(-1, 1),
                random.uniform(0.5, 1.5),
                random.uniform(-1, 1)
            ))
            
            velocity = direction * intensity * 5.0
            
            particle = FluidParticle(
                position=position + glm.vec3(0, 0.1, 0),
                velocity=velocity,
                mass=self.config.particle_mass,
                radius=self.config.particle_radius * 0.8
            )
            
            # Splash particles have shorter lifetime
            particle.lifetime = 2.0
            particle.color = glm.vec3(1.0, 1.0, 1.0)  # White for foam
            
            self.particles.append(particle)
            
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return self.performance_stats
        
    def reset(self):
        """Reset simulation"""
        self.particles.clear()
        self.spatial_grid.clear()
        self.simulation_time = 0.0
        self.frame_count = 0
        
    def cleanup(self):
        """Clean up resources"""
        self.particles.clear()
        self.spatial_grid.clear()

class InteractiveFluidSimulation(FluidDynamicsSimulation):
    """Fluid simulation with interactive features"""
    
    def __init__(self, config: SPHParameters = None):
        super().__init__(config)
        self.interactive_forces = []
        self.fluid_emitters = []
        
    def add_fluid_emitter(self, position: glm.vec3, rate: float = 10.0, 
                         velocity: glm.vec3 = None, lifetime: float = 10.0):
        """Add a fluid emitter"""
        if velocity is None:
            velocity = glm.vec3(0, 2.0, 0)
            
        emitter = {
            'position': position,
            'rate': rate,
            'velocity': velocity,
            'lifetime': lifetime,
            'time_since_emission': 0.0,
            'active': True
        }
        self.fluid_emitters.append(emitter)
        
    def update_emitters(self, dt: float):
        """Update fluid emitters"""
        for emitter in self.fluid_emitters:
            if not emitter['active']:
                continue
                
            emitter['time_since_emission'] += dt
            particles_to_emit = int(emitter['rate'] * emitter['time_since_emission'])
            
            for _ in range(particles_to_emit):
                if len(self.particles) >= 5000:  # Particle limit
                    break
                    
                # Add some randomness to position and velocity
                pos = emitter['position'] + glm.vec3(
                    random.uniform(-0.1, 0.1),
                    random.uniform(-0.1, 0.1),
                    random.uniform(-0.1, 0.1)
                )
                
                vel = emitter['velocity'] + glm.vec3(
                    random.uniform(-0.5, 0.5),
                    random.uniform(-0.2, 0.2),
                    random.uniform(-0.5, 0.5)
                )
                
                particle = FluidParticle(
                    position=pos,
                    velocity=vel,
                    mass=self.config.particle_mass,
                    radius=self.config.particle_radius
                )
                particle.lifetime = emitter['lifetime']
                
                self.particles.append(particle)
                
            emitter['time_since_emission'] -= particles_to_emit / emitter['rate']
            
    def update(self, dt: float):
        """Update interactive fluid simulation"""
        # Update emitters first
        self.update_emitters(dt)
        
        # Then update SPH simulation
        super().update(dt)
        
        # Remove dead particles
        self.particles = [p for p in self.particles if p.is_alive()]

# Demo and testing
if __name__ == "__main__":
    print("Testing Fluid Dynamics Simulation...")
    
    # Create fluid simulation
    config = SPHParameters()
    config.rest_density = 800.0
    config.gas_constant = 1000.0
    config.viscosity = 0.05
    
    fluid_sim = FluidDynamicsSimulation(config)
    fluid_sim.initialize(500)
    
    # Test simulation for a few steps
    for i in range(100):
        fluid_sim.update(0.016)
        
    stats = fluid_sim.get_performance_stats()
    print(f"Fluid simulation test completed: {stats}")
    
    # Test interactive simulation
    interactive_sim = InteractiveFluidSimulation(config)
    interactive_sim.initialize(200)
    interactive_sim.add_fluid_emitter(glm.vec3(0, -2, 0), rate=20.0)
    
    for i in range(50):
        interactive_sim.update(0.016)
        
    print("Interactive fluid simulation test completed")

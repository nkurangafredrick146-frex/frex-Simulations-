#!/usr/bin/env python3
"""
Multi-scale Physics & Fractal Dynamics
Simulations spanning quantum to cosmological scales with fractal geometry
"""

import numpy as np
import glm
import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import *
from typing import Dict, List, Optional, Tuple, Any
import math
import random
from dataclasses import dataclass
from enum import Enum
import noise
from scipy import ndimage
from PIL import Image
import json

@dataclass
class ScaleLevel:
    name: str
    scale_factor: float
    time_step: float
    physics_laws: str
    rendering_style: str
    particle_size: float

class FractalDimension:
    """Fractal dimension calculations and generation"""
    
    @staticmethod
    def mandelbrot(x: float, y: float, max_iter: int = 100) -> int:
        """Calculate Mandelbrot set iteration count"""
        c = complex(x, y)
        z = 0.0j
        for i in range(max_iter):
            z = z * z + c
            if abs(z) >= 2.0:
                return i
        return max_iter
    
    @staticmethod
    def julia(x: float, y: float, cx: float = -0.7, cy: float = 0.27015, max_iter: int = 100) -> int:
        """Calculate Julia set iteration count"""
        z = complex(x, y)
        c = complex(cx, cy)
        for i in range(max_iter):
            z = z * z + c
            if abs(z) >= 2.0:
                return i
        return max_iter
    
    @staticmethod
    def generate_fractal_landscape(width: int, height: int, scale: float = 1.0, octaves: int = 6, 
                                 persistence: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
        """Generate fractal landscape using Perlin noise"""
        landscape = np.zeros((height, width))
        
        for y in range(height):
            for x in range(width):
                nx = x / width - 0.5
                ny = y / height - 0.5
                landscape[y][x] = noise.pnoise2(nx * scale, ny * scale, 
                                              octaves=octaves, 
                                              persistence=persistence, 
                                              lacunarity=lacunarity, 
                                              repeatx=1024, 
                                              repeaty=1024, 
                                              base=42)
        
        # Normalize to [0, 1]
        landscape = (landscape - landscape.min()) / (landscape.max() - landscape.min())
        return landscape
    
    @staticmethod
    def calculate_fractal_dimension(data: np.ndarray, box_sizes: List[int] = None) -> float:
        """Calculate fractal dimension using box-counting method"""
        if box_sizes is None:
            box_sizes = [2, 4, 8, 16, 32, 64]
        
        counts = []
        
        for box_size in box_sizes:
            # Downsample the data
            downsampled = data[::box_size, ::box_size]
            # Count non-empty boxes
            non_empty = np.sum(downsampled > 0)
            counts.append(non_empty)
        
        # Linear regression on log-log plot
        log_sizes = np.log(box_sizes)
        log_counts = np.log(counts)
        
        # Calculate slope (fractal dimension = -slope)
        slope = -np.polyfit(log_sizes, log_counts, 1)[0]
        return slope

class MultiScaleParticle:
    """Particle that can exist at multiple scales simultaneously"""
    
    def __init__(self, position: glm.vec3, scale_level: ScaleLevel):
        self.position = position
        self.velocity = glm.vec3(0)
        self.acceleration = glm.vec3(0)
        self.scale_level = scale_level
        self.quantum_state = None  # For quantum scale
        self.cosmic_attributes = None  # For cosmic scale
        self.fractal_connections = []  # Connections to particles at other scales
        
        # Multi-scale properties
        self.mass = 1.0 * scale_level.scale_factor
        self.charge = 0.0
        self.spin = 0.0
        self.color = glm.vec4(1.0, 1.0, 1.0, 1.0)
        self.lifetime = 0.0
        self.quantum_amplitude = 1.0
        self.cosmic_redshift = 0.0
        
        # Fractal properties
        self.fractal_level = 0
        self.self_similarity = 1.0
        self.recursion_depth = 0
        
    def update_quantum_behavior(self, dt: float):
        """Update quantum-scale behavior"""
        # Simplified quantum mechanics
        if self.scale_level.scale_factor < 1e-9:  # Quantum scale
            # Quantum fluctuation
            fluctuation = glm.vec3(
                random.gauss(0, 0.1),
                random.gauss(0, 0.1), 
                random.gauss(0, 0.1)
            ) * math.sqrt(dt)
            
            self.position += fluctuation
            
            # Wave function evolution
            self.quantum_amplitude *= math.exp(-dt * 0.1)  # Decay
            
    def update_cosmic_behavior(self, dt: float):
        """Update cosmic-scale behavior"""
        if self.scale_level.scale_factor > 1e9:  # Cosmic scale
            # Hubble expansion
            expansion_rate = 70.0  # km/s/Mpc in simulation units
            self.velocity += self.position * expansion_rate * dt * 1e-6
            
            # Gravitational attraction to center
            center_force = -self.position * 0.1 / max(glm.length2(self.position), 1.0)
            self.velocity += center_force * dt
            
    def update_fractal_connections(self, all_particles: List['MultiScaleParticle']):
        """Update connections to particles at other fractal levels"""
        self.fractal_connections.clear()
        
        for other in all_particles:
            if other is not self and self.is_fractally_connected(other):
                distance = glm.length(self.position - other.position)
                connection_strength = 1.0 / (1.0 + distance * distance)
                self.fractal_connections.append((other, connection_strength))
    
    def is_fractally_connected(self, other: 'MultiScaleParticle') -> bool:
        """Check if two particles are connected through fractal similarity"""
        if abs(self.fractal_level - other.fractal_level) <= 1:
            position_similarity = 1.0 / (1.0 + glm.length2(self.position - other.position))
            scale_similarity = 1.0 / (1.0 + abs(math.log(self.scale_level.scale_factor / other.scale_level.scale_factor)))
            
            return (position_similarity * scale_similarity) > 0.1
        return False

class MultiScalePhysicsEngine:
    """Physics engine handling multiple scales and fractal dynamics"""
    
    def __init__(self, simulation_app):
        self.simulation_app = simulation_app
        
        # Scale definitions
        self.scale_levels = {
            "quantum": ScaleLevel("quantum", 1e-12, 1e-18, "quantum", "point", 0.001),
            "atomic": ScaleLevel("atomic", 1e-10, 1e-15, "quantum_classical", "sphere", 0.01),
            "molecular": ScaleLevel("molecular", 1e-9, 1e-12, "classical", "ball", 0.1),
            "micro": ScaleLevel("micro", 1e-6, 1e-9, "classical", "sphere", 1.0),
            "meso": ScaleLevel("meso", 1e-3, 1e-6, "continuum", "cube", 10.0),
            "macro": ScaleLevel("macro", 1.0, 1e-3, "continuum", "object", 100.0),
            "cosmic": ScaleLevel("cosmic", 1e9, 1e6, "relativistic", "star", 1000.0),
            "galactic": ScaleLevel("galactic", 1e12, 1e9, "cosmological", "galaxy", 10000.0)
        }
        
        # Active particles
        self.particles: List[MultiScaleParticle] = []
        
        # Fractal systems
        self.fractal_systems = {}
        self.fractal_dimension = 2.0
        self.self_similarity_threshold = 0.8
        
        # Cross-scale interactions
        self.cross_scale_coupling = 0.01
        self.quantum_decoherence_rate = 0.1
        self.cosmic_inflation_factor = 1.0
        
        # Visualization
        self.scale_transition_shader = None
        self.fractal_texture = None
        
        # Initialize systems
        self.initialize_fractal_systems()
        self.initialize_scale_transitions()
        
        print("Multi-scale Physics Engine initialized")
    
    def initialize_fractal_systems(self):
        """Initialize fractal geometry systems"""
        # Generate fractal landscapes for different scales
        for scale_name in ["quantum", "micro", "macro", "cosmic"]:
            landscape = FractalDimension.generate_fractal_landscape(
                256, 256, scale=10.0 if scale_name == "quantum" else 1.0
            )
            self.fractal_systems[scale_name] = landscape
            
            # Calculate fractal dimension
            fractal_dim = FractalDimension.calculate_fractal_dimension(landscape)
            print(f"Fractal dimension for {scale_name} scale: {fractal_dim:.3f}")
    
    def initialize_scale_transitions(self):
        """Initialize scale transition systems"""
        # This would set up shaders and rendering for smooth scale transitions
        try:
            self.scale_transition_shader = self.compile_scale_transition_shader()
        except Exception as e:
            print(f"Could not initialize scale transition shader: {e}")
    
    def compile_scale_transition_shader(self):
        """Compile shader for smooth scale transitions"""
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec4 aColor;
        layout (location = 2) in float aScale;
        
        out vec4 Color;
        out float Scale;
        
        uniform mat4 view;
        uniform mat4 projection;
        uniform float time;
        
        void main() {
            Color = aColor;
            Scale = aScale;
            
            // Fractal perturbation based on position and time
            vec3 fractal_pos = aPos + 0.1 * sin(aPos * 10.0 + time) * exp(-aScale);
            
            gl_Position = projection * view * vec4(fractal_pos, 1.0);
            gl_PointSize = 10.0 * aScale;
        }
        """
        
        fragment_shader = """
        #version 330 core
        in vec4 Color;
        in float Scale;
        
        out vec4 FragColor;
        
        uniform float time;
        uniform float fractal_dimension;
        
        void main() {
            // Fractal-inspired coloring based on scale
            vec3 base_color = Color.rgb;
            
            // Add fractal detail based on screen coordinates
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            float r = length(coord);
            
            if (r > 1.0) discard;
            
            // Fractal pattern based on Mandelbrot
            float fractal_detail = 0.5 + 0.5 * sin(coord.x * 20.0 + time) * 
                                             sin(coord.y * 20.0 + time);
            
            // Scale-dependent transparency
            float alpha = Color.a * (1.0 - r) * fractal_detail;
            
            // Color variation based on scale and fractal dimension
            vec3 final_color = base_color * (0.8 + 0.2 * fractal_detail);
            final_color *= (1.0 + 0.1 * sin(Scale * 10.0 + time));
            
            FragColor = vec4(final_color, alpha);
        }
        """
        
        return compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )
    
    def create_scale_hierarchy(self, center: glm.vec3, base_scale: str, depth: int = 3):
        """Create a hierarchical system of particles across scales"""
        base_level = self.scale_levels[base_scale]
        
        for i in range(depth):
            # Create particles at this fractal level
            scale_factor = math.pow(2, i)  # Each level is 2x larger/smaller
            num_particles = int(100 / scale_factor)  # Fewer particles at larger scales
            
            for j in range(num_particles):
                # Position with fractal distribution
                angle = j * 2 * math.pi / num_particles
                radius = scale_factor * (1.0 + 0.2 * math.sin(angle * 3))
                
                position = center + glm.vec3(
                    radius * math.cos(angle),
                    radius * math.sin(angle) * 0.5,
                    random.uniform(-1, 1) * 0.1
                )
                
                # Determine scale level based on fractal level
                if i == 0:
                    scale_name = base_scale
                else:
                    # Move up or down scale hierarchy
                    scale_names = list(self.scale_levels.keys())
                    base_index = scale_names.index(base_scale)
                    new_index = max(0, min(len(scale_names)-1, base_index + i))
                    scale_name = scale_names[new_index]
                
                particle = MultiScaleParticle(position, self.scale_levels[scale_name])
                particle.fractal_level = i
                particle.self_similarity = 1.0 / (1.0 + i)
                
                self.particles.append(particle)
    
    def update(self, dt: float):
        """Update multi-scale physics"""
        # Update particles at each scale
        for particle in self.particles:
            self.update_particle_physics(particle, dt)
        
        # Update cross-scale interactions
        self.update_cross_scale_interactions()
        
        # Update fractal connections
        for particle in self.particles:
            particle.update_fractal_connections(self.particles)
        
        # Update fractal dimension based on system state
        self.update_fractal_dimension()
    
    def update_particle_physics(self, particle: MultiScaleParticle, dt: float):
        """Update physics for a single particle based on its scale"""
        scale_dt = dt * particle.scale_level.time_step / self.scale_levels["macro"].time_step
        
        # Scale-specific physics
        if particle.scale_level.name in ["quantum", "atomic"]:
            particle.update_quantum_behavior(scale_dt)
        elif particle.scale_level.name in ["cosmic", "galactic"]:
            particle.update_cosmic_behavior(scale_dt)
        else:
            # Classical physics for intermediate scales
            particle.velocity += particle.acceleration * scale_dt
            particle.position += particle.velocity * scale_dt
            particle.acceleration = glm.vec3(0)  # Reset for next frame
        
        # Universal forces (gravity-like)
        self.apply_universal_forces(particle, scale_dt)
        
        # Fractal dynamics
        self.apply_fractal_dynamics(particle, scale_dt)
        
        particle.lifetime += scale_dt
    
    def apply_universal_forces(self, particle: MultiScaleParticle, dt: float):
        """Apply forces that work across all scales"""
        # Central attraction force (simplified gravity)
        center_force = -particle.position * 0.1 / max(glm.length2(particle.position), 0.1)
        particle.velocity += center_force * dt
        
        # Fractal field force (emergent from fractal structure)
        fractal_force = self.calculate_fractal_field_force(particle.position)
        particle.velocity += fractal_force * dt * 0.1
    
    def calculate_fractal_field_force(self, position: glm.vec3) -> glm.vec3:
        """Calculate force from fractal field potential"""
        # Sample fractal landscape at position
        x = int((position.x + 2) * 64) % 256
        y = int((position.y + 2) * 64) % 256
        z = int((position.z + 2) * 64) % 256
        
        # Get gradient from fractal field (simplified)
        force = glm.vec3(0)
        for scale_name, landscape in self.fractal_systems.items():
            if x < 256 and y < 256:
                # Calculate gradient (simplified)
                grad_x = (landscape[y, (x+1)%256] - landscape[y, (x-1)%256]) * 0.5
                grad_y = (landscape[(y+1)%256, x] - landscape[(y-1)%256, x]) * 0.5
                
                force += glm.vec3(grad_x, grad_y, 0) * 0.01
        
        return force
    
    def apply_fractal_dynamics(self, particle: MultiScaleParticle, dt: float):
        """Apply fractal-specific dynamics"""
        # Self-similar motion patterns
        fractal_frequency = math.pow(2, particle.fractal_level)
        fractal_amplitude = 0.1 / fractal_frequency
        
        particle.position += glm.vec3(
            fractal_amplitude * math.sin(particle.lifetime * fractal_frequency),
            fractal_amplitude * math.cos(particle.lifetime * fractal_frequency * 1.618),  # Golden ratio
            fractal_amplitude * math.sin(particle.lifetime * fractal_frequency * 3.141)   # Pi
        ) * dt
    
    def update_cross_scale_interactions(self):
        """Handle interactions between different scales"""
        for i, particle1 in enumerate(self.particles):
            for j, particle2 in enumerate(self.particles[i+1:], i+1):
                if self.are_scales_coupled(particle1.scale_level, particle2.scale_level):
                    self.apply_cross_scale_interaction(particle1, particle2)
    
    def are_scales_coupled(self, level1: ScaleLevel, level2: ScaleLevel) -> bool:
        """Check if two scale levels can interact"""
        scale_ratio = level1.scale_factor / level2.scale_factor
        return abs(math.log10(scale_ratio)) < 6  # Within 6 orders of magnitude
    
    def apply_cross_scale_interaction(self, particle1: MultiScaleParticle, particle2: MultiScaleParticle):
        """Apply interaction between particles at different scales"""
        distance = glm.length(particle1.position - particle2.position)
        max_interaction_distance = 2.0  # In simulation units
        
        if distance < max_interaction_distance:
            # Scale-dependent interaction strength
            scale_factor = min(particle1.scale_level.scale_factor, 
                             particle2.scale_level.scale_factor)
            interaction_strength = self.cross_scale_coupling / (1.0 + distance * distance)
            
            # Repulsive/attractive force based on scale difference
            scale_diff = math.log(particle1.scale_level.scale_factor / 
                                particle2.scale_level.scale_factor)
            
            force_direction = glm.normalize(particle1.position - particle2.position)
            force = force_direction * interaction_strength * math.tanh(scale_diff)
            
            particle1.velocity += force * scale_factor
            particle2.velocity -= force * scale_factor
    
    def update_fractal_dimension(self):
        """Update overall fractal dimension based on system state"""
        if len(self.particles) < 10:
            return
        
        # Calculate spatial distribution fractal dimension
        positions = np.array([[p.position.x, p.position.y, p.position.z] for p in self.particles])
        
        # Simplified box-counting in 2D projection
        hist, x_edges, y_edges = np.histogram2d(
            positions[:, 0], positions[:, 1], bins=8
        )
        
        non_empty_boxes = np.sum(hist > 0)
        total_boxes = hist.size
        
        if total_boxes > 0:
            self.fractal_dimension = math.log(non_empty_boxes) / math.log(total_boxes) * 2.0
    
    def render(self, view_matrix: glm.mat4, projection_matrix: glm.mat4):
        """Render multi-scale system"""
        if self.scale_transition_shader:
            glUseProgram(self.scale_transition_shader)
            
            # Set shader uniforms
            glUniformMatrix4fv(
                glGetUniformLocation(self.scale_transition_shader, "view"),
                1, GL_FALSE, glm.value_ptr(view_matrix)
            )
            glUniformMatrix4fv(
                glGetUniformLocation(self.scale_transition_shader, "projection"), 
                1, GL_FALSE, glm.value_ptr(projection_matrix)
            )
            glUniform1f(
                glGetUniformLocation(self.scale_transition_shader, "time"),
                self.simulation_app.simulation_time
            )
            glUniform1f(
                glGetUniformLocation(self.scale_transition_shader, "fractal_dimension"),
                self.fractal_dimension
            )
        
        # Render particles
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        for particle in self.particles:
            self.render_particle(particle, view_matrix, projection_matrix)
        
        # Render fractal connections
        self.render_fractal_connections()
        
        glUseProgram(0)
    
    def render_particle(self, particle: MultiScaleParticle, view_matrix: glm.mat4, projection_matrix: glm.mat4):
        """Render a single particle with scale-appropriate appearance"""
        # Scale-dependent color
        scale_index = list(self.scale_levels.keys()).index(particle.scale_level.name)
        hue = scale_index / len(self.scale_levels)
        
        # Convert HSV to RGB
        color = self.hsv_to_rgb(hue, 0.8, 1.0)
        alpha = 0.7 * particle.quantum_amplitude
        
        # Set color based on scale
        glColor4f(color[0], color[1], color[2], alpha)
        
        # Render as point
        glBegin(GL_POINTS)
        glVertex3f(particle.position.x, particle.position.y, particle.position.z)
        glEnd()
    
    def render_fractal_connections(self):
        """Render connections between fractally related particles"""
        glColor4f(0.3, 0.3, 0.8, 0.3)
        glBegin(GL_LINES)
        
        for particle in self.particles:
            for other, strength in particle.fractal_connections:
                if strength > 0.1:
                    glVertex3f(particle.position.x, particle.position.y, particle.position.z)
                    glVertex3f(other.position.x, other.position.y, other.position.z)
        
        glEnd()
    
    def hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Convert HSV color to RGB"""
        if s == 0.0:
            return (v, v, v)
        
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        
        i = i % 6
        if i == 0:
            return (v, t, p)
        elif i == 1:
            return (q, v, p)
        elif i == 2:
            return (p, v, t)
        elif i == 3:
            return (p, q, v)
        elif i == 4:
            return (t, p, v)
        else:
            return (v, p, q)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the multi-scale system"""
        scale_counts = {}
        for scale_name in self.scale_levels:
            scale_counts[scale_name] = sum(
                1 for p in self.particles if p.scale_level.name == scale_name
            )
        
        return {
            "total_particles": len(self.particles),
            "scale_distribution": scale_counts,
            "fractal_dimension": self.fractal_dimension,
            "cross_scale_coupling": self.cross_scale_coupling,
            "average_self_similarity": np.mean([p.self_similarity for p in self.particles]) 
                                     if self.particles else 0.0
        }
    
    def create_quantum_fluctuation(self, position: glm.vec3, intensity: float = 1.0):
        """Create quantum-scale fluctuations"""
        num_particles = int(10 * intensity)
        
        for i in range(num_particles):
            quantum_pos = position + glm.vec3(
                random.gauss(0, 0.1),
                random.gauss(0, 0.1),
                random.gauss(0, 0.1)
            )
            
            particle = MultiScaleParticle(quantum_pos, self.scale_levels["quantum"])
            particle.quantum_amplitude = random.uniform(0.5, 1.0)
            particle.velocity = glm.vec3(
                random.gauss(0, 0.1),
                random.gauss(0, 0.1), 
                random.gauss(0, 0.1)
            )
            
            self.particles.append(particle)
    
    def create_cosmic_structure(self, position: glm.vec3, size: float = 10.0):
        """Create cosmic-scale structure"""
        # Create galactic cluster
        for i in range(5):  # Create 5 "galaxies"
            galaxy_pos = position + glm.vec3(
                random.uniform(-size, size),
                random.uniform(-size, size) * 0.1,  # Flatten in y-axis
                random.uniform(-size, size)
            )
            
            # Create spiral arm pattern
            for j in range(20):  # 20 stars per galaxy
                angle = j * 2 * math.pi / 20
                radius = random.uniform(1.0, 3.0)
                
                star_pos = galaxy_pos + glm.vec3(
                    radius * math.cos(angle),
                    0,
                    radius * math.sin(angle)
                )
                
                particle = MultiScaleParticle(star_pos, self.scale_levels["cosmic"])
                particle.cosmic_redshift = random.uniform(0, 0.1)
                particle.color = glm.vec4(1.0, 0.9, 0.8, 1.0)  # Starlight color
                
                # Orbital velocity
                orbital_speed = 1.0 / math.sqrt(radius)
                particle.velocity = glm.vec3(
                    -orbital_speed * math.sin(angle),
                    0,
                    orbital_speed * math.cos(angle)
                )
                
                self.particles.append(particle)

# Example integration
class MultiScaleSimulation:
    """Wrapper for multi-scale simulation"""
    
    def __init__(self, base_simulation, multi_scale_engine):
        self.base_simulation = base_simulation
        self.multi_scale_engine = multi_scale_engine
        self.active_scales = ["macro", "micro"]  # Default active scales
    
    def update(self, dt):
        self.multi_scale_engine.update(dt)
    
    def render(self, view_matrix, projection_matrix):
        self.multi_scale_engine.render(view_matrix, projection_matrix)
    
    def switch_scale_focus(self, new_scale):
        """Switch the primary scale focus"""
        if new_scale in self.multi_scale_engine.scale_levels:
            print(f"Switching scale focus to: {new_scale}")
            # This would adjust camera, time scale, etc.

if __name__ == "__main__":
    # Test the multi-scale physics engine
    engine = MultiScalePhysicsEngine(None)
    
    # Create test systems
    engine.create_scale_hierarchy(glm.vec3(0, 0, 0), "macro", depth=4)
    engine.create_quantum_fluctuation(glm.vec3(1, 0, 0), intensity=2.0)
    engine.create_cosmic_structure(glm.vec3(0, 5, 0), size=8.0)
    
    # Print system info
    info = engine.get_system_info()
    print("Multi-scale System Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("Multi-scale Physics Engine test completed successfully")
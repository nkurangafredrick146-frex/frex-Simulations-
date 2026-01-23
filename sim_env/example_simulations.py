"""
Complete Example Simulations Module
Pre-built simulation configurations with advanced effects and interactive features
"""

import pygame
import numpy as np
import glm
from OpenGL.GL import *
from OpenGL.GL.shaders import *
import math
import time
from typing import Dict, List, Any, Optional, Tuple
import random
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    """Configuration container for simulation parameters"""
    name: str
    particle_count: int = 1000
    gravity: glm.vec3 = None
    time_scale: float = 1.0
    enable_collisions: bool = True
    enable_fluid_dynamics: bool = False
    enable_thermal_dynamics: bool = False
    enable_electromagnetism: bool = False
    render_mode: str = "advanced"
    
    def __post_init__(self):
        if self.gravity is None:
            self.gravity = glm.vec3(0, -9.81, 0)

class BaseSimulation:
    """Base class for all simulations with common functionality"""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig("Base")
        self.initialized = False
        self.paused = False
        self.simulation_time = 0.0
        self.frame_count = 0
        
        # Core systems
        self.physics_module = None
        self.particle_system = None
        self.rendering_engine = None
        
        # Interactive elements
        self.interactive_forces = []
        self.user_particles = []
        self.force_visualizations = []
        
        # Performance tracking
        self.performance_stats = {}
        self.frame_times = []
        
    def initialize(self):
        """Initialize simulation systems"""
        from physics_simulation_module import PhysicsSimulationModule, PhysicsSettings
        from particle_system import ParticleSystem
        from rendering_engine import RenderingEngine
        
        # Initialize physics with custom settings
        physics_settings = PhysicsSettings()
        physics_settings.gravity = self.config.gravity
        physics_settings.time_scale = self.config.time_scale
        physics_settings.collision_enabled = self.config.enable_collisions
        physics_settings.fluid_dynamics_enabled = self.config.enable_fluid_dynamics
        
        self.physics_module = PhysicsSimulationModule(physics_settings)
        
        # Initialize particle system
        self.particle_system = ParticleSystem(max_particles=self.config.particle_count)
        self.particle_system.set_render_mode(self.config.render_mode)
        
        # Initialize rendering
        self.rendering_engine = RenderingEngine()
        
        # Set up simulation-specific elements
        self.setup_simulation()
        
        self.initialized = True
        print(f"Initialized {self.config.name} simulation")
        
    def setup_simulation(self):
        """Override this method to set up simulation-specific elements"""
        pass
        
    def update(self, dt: float):
        """Update simulation state"""
        if self.paused or not self.initialized:
            return
            
        self.simulation_time += dt
        self.frame_count += 1
        
        # Update physics
        self.physics_module.update(dt)
        
        # Update particle system
        self.particle_system.update(dt)
        
        # Update interactive elements
        self.update_interactive_elements(dt)
        
        # Update performance stats
        self.update_performance_stats()
        
    def render(self, view_matrix: glm.mat4 = None, projection_matrix: glm.mat4 = None):
        """Render simulation"""
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
            
        # Render particles
        camera_position = glm.vec3(0, 0, 8)  # Simplified
        self.particle_system.render(view_matrix, projection_matrix, camera_position)
        
        # Render force visualizations
        self.render_force_visualizations(view_matrix, projection_matrix)
        
    def update_interactive_elements(self, dt: float):
        """Update interactive simulation elements"""
        # Update force field visualizations
        for viz in self.force_visualizations:
            viz['time'] += dt
            
        # Remove old user particles
        self.user_particles = [p for p in self.user_particles if p.is_alive()]
        
    def render_force_visualizations(self, view_matrix: glm.mat4, projection_matrix: glm.mat4):
        """Render visualizations for force fields"""
        # This would implement force field visualization
        pass
        
    def update_performance_stats(self):
        """Update performance statistics"""
        physics_stats = self.physics_module.get_performance_stats()
        particle_stats = self.particle_system.get_performance_stats()
        
        self.performance_stats = {
            **physics_stats,
            **particle_stats,
            'simulation_time': self.simulation_time,
            'frame_count': self.frame_count,
            'paused': self.paused
        }
        
    def pause_toggle(self):
        """Toggle simulation pause state"""
        self.paused = not self.paused
        print(f"Simulation {'paused' if self.paused else 'resumed'}")
        
    def reset(self):
        """Reset simulation to initial state"""
        self.physics_module.clear_particles()
        self.physics_module.clear_force_fields()
        self.particle_system.clear_particles()
        self.particle_system.clear_emitters()
        self.interactive_forces.clear()
        self.user_particles.clear()
        self.force_visualizations.clear()
        self.simulation_time = 0.0
        self.frame_count = 0
        
        # Re-setup simulation
        self.setup_simulation()
        
    def add_particle_at_screen_pos(self, screen_pos: Tuple[int, int], screen_size: Tuple[int, int] = (1200, 800)):
        """Add particle at screen coordinates"""
        # Convert screen coordinates to world coordinates
        x = (screen_pos[0] / screen_size[0] - 0.5) * 10.0
        y = -(screen_pos[1] / screen_size[1] - 0.5) * 8.0
        z = 0.0
        
        world_pos = glm.vec3(x, y, z)
        
        # Create particle with upward velocity
        from physics_simulation_module import Particle
        particle = Particle(
            position=world_pos,
            velocity=glm.vec3(0, 3, 0),
            mass=1.0,
            radius=0.1,
            color=glm.vec3(random.random(), random.random(), random.random())
        )
        
        self.physics_module.add_particle(particle)
        self.user_particles.append(particle)
        
    def add_force_field_at_position(self, position: glm.vec3, strength: float = 10.0, radius: float = 2.0):
        """Add interactive force field"""
        force_field = self.physics_module.add_force_field(
            position, strength, radius, "radial"
        )
        
        # Add visualization
        self.force_visualizations.append({
            'position': position,
            'radius': radius,
            'strength': strength,
            'time': 0.0,
            'type': 'radial'
        })
        
        return force_field
        
    def cleanup(self):
        """Clean up simulation resources"""
        self.physics_module = None
        self.particle_system = None
        self.rendering_engine = None

class BasicParticleSimulation(BaseSimulation):
    """Basic particle simulation with gravity and collisions"""
    
    def __init__(self):
        config = SimulationConfig(
            name="Basic Particles",
            particle_count=2000,
            gravity=glm.vec3(0, -9.81, 0),
            enable_collisions=True
        )
        super().__init__(config)
        
    def setup_simulation(self):
        """Set up basic particle simulation"""
        # Create multiple particle emitters
        self.create_fountain_emitter(glm.vec3(-2, -1, 0), glm.vec3(0.2, 0.5, 1.0))
        self.create_fountain_emitter(glm.vec3(2, -1, 0), glm.vec3(1.0, 0.3, 0.2))
        self.create_rain_emitter(glm.vec3(0, 3, 0))
        
        # Add some force fields
        self.physics_module.add_force_field(glm.vec3(0, 1, 0), 5.0, 3.0, "vortex")
        self.physics_module.add_force_field(glm.vec3(-3, 0, 0), 8.0, 2.0, "radial")
        self.physics_module.add_force_field(glm.vec3(3, 0, 0), 8.0, 2.0, "radial")
        
    def create_fountain_emitter(self, position: glm.vec3, color: glm.vec3):
        """Create a fountain-like particle emitter"""
        emitter = self.particle_system.create_emitter(
            position=position,
            emission_rate=30.0,
            particle_lifetime=4.0,
            particle_speed=5.0,
            spread_angle=25.0,
            emitter_type="point"
        )
        emitter.color_start = color
        emitter.color_end = color * 0.5
        emitter.size_start = 0.15
        emitter.size_end = 0.05
        emitter.particle_mass = 1.0
        emitter.particle_radius = 0.08
        
    def create_rain_emitter(self, position: glm.vec3):
        """Create a rain particle emitter"""
        emitter = self.particle_system.create_emitter(
            position=position,
            emission_rate=50.0,
            particle_lifetime=3.0,
            particle_speed=2.0,
            spread_angle=60.0,
            emitter_type="box",
            size=glm.vec3(4, 0, 0)
        )
        emitter.color_start = glm.vec3(0.7, 0.8, 1.0)
        emitter.color_end = glm.vec3(0.4, 0.6, 1.0)
        emitter.size_start = 0.08
        emitter.size_end = 0.08
        emitter.particle_mass = 0.5

class FountainSimulation(BaseSimulation):
    """Elegant fountain simulation with realistic water physics"""
    
    def __init__(self):
        config = SimulationConfig(
            name="Fountain",
            particle_count=3000,
            gravity=glm.vec3(0, -15.0, 0),
            enable_collisions=True
        )
        super().__init__(config)
        
    def setup_simulation(self):
        """Set up fountain simulation"""
        # Create main fountain emitter
        self.create_main_fountain(glm.vec3(0, -1.5, 0))
        
        # Create secondary fountains
        self.create_secondary_fountain(glm.vec3(-1.5, -1.5, 1), glm.vec3(0.3, 0.7, 1.0))
        self.create_secondary_fountain(glm.vec3(1.5, -1.5, -1), glm.vec3(0.8, 0.9, 1.0))
        
        # Add upward force to simulate water pressure
        self.physics_module.add_force_field(glm.vec3(0, -1, 0), 25.0, 1.5, "radial")
        
        # Create splash effect emitters at ground level
        self.create_splash_emitters()
        
    def create_main_fountain(self, position: glm.vec3):
        """Create main fountain emitter"""
        emitter = self.particle_system.create_emitter(
            position=position,
            emission_rate=80.0,
            particle_lifetime=2.5,
            particle_speed=8.0,
            spread_angle=15.0,
            emitter_type="circle",
            size=glm.vec3(0.3, 0, 0.3)
        )
        emitter.color_start = glm.vec3(0.7, 0.8, 1.0)
        emitter.color_end = glm.vec3(0.4, 0.6, 0.9)
        emitter.size_start = 0.12
        emitter.size_end = 0.06
        emitter.particle_mass = 1.0
        emitter.particle_radius = 0.06
        
    def create_secondary_fountain(self, position: glm.vec3, color: glm.vec3):
        """Create secondary fountain emitter"""
        emitter = self.particle_system.create_emitter(
            position=position,
            emission_rate=40.0,
            particle_lifetime=2.0,
            particle_speed=6.0,
            spread_angle=20.0,
            emitter_type="point"
        )
        emitter.color_start = color
        emitter.color_end = color * 0.7
        emitter.size_start = 0.1
        emitter.size_end = 0.05
        
    def create_splash_emitters(self):
        """Create emitters for splash effects"""
        # These would be triggered when particles hit the ground
        # For now, create static splash zones
        for x in np.linspace(-2, 2, 5):
            emitter = self.particle_system.create_emitter(
                position=glm.vec3(x, -1.8, 0),
                emission_rate=5.0,  # Low continuous emission for effect
                particle_lifetime=1.0,
                particle_speed=2.0,
                spread_angle=90.0,
                emitter_type="point"
            )
            emitter.color_start = glm.vec3(1.0, 1.0, 1.0)
            emitter.color_end = glm.vec3(0.8, 0.8, 1.0)
            emitter.size_start = 0.08
            emitter.size_end = 0.02

class FireSimulation(BaseSimulation):
    """Realistic fire simulation with thermal dynamics and turbulence"""
    
    def __init__(self):
        config = SimulationConfig(
            name="Fire",
            particle_count=2500,
            gravity=glm.vec3(0, -2.0, 0),  # Weak gravity for fire
            enable_thermal_dynamics=True,
            enable_collisions=False
        )
        super().__init__(config)
        
    def setup_simulation(self):
        """Set up fire simulation"""
        # Create main fire emitter
        self.create_fire_emitter(glm.vec3(0, -1.8, 0), 1.0)
        
        # Create smaller fire emitters around main fire
        for i in range(3):
            angle = i * 2 * math.pi / 3
            x = math.cos(angle) * 0.8
            z = math.sin(angle) * 0.8
            self.create_fire_emitter(glm.vec3(x, -1.8, z), 0.5)
            
        # Add turbulence force fields for realistic fire movement
        self.create_turbulence_fields()
        
        # Set global temperature for thermal effects
        self.physics_module.global_temperature = 293.15  # Room temperature
        
    def create_fire_emitter(self, position: glm.vec3, intensity: float):
        """Create a fire emitter"""
        emitter = self.particle_system.create_emitter(
            position=position,
            emission_rate=60.0 * intensity,
            particle_lifetime=1.5 * intensity,
            particle_speed=3.0 * intensity,
            spread_angle=8.0,
            emitter_type="circle",
            size=glm.vec3(0.4 * intensity, 0, 0.4 * intensity)
        )
        
        # Fire color gradient from hot to cool
        emitter.color_start = glm.vec3(1.0, 0.3, 0.1)  # Hot core
        emitter.color_end = glm.vec3(1.0, 0.8, 0.1)    # Cool edges
        emitter.size_start = 0.2 * intensity
        emitter.size_end = 0.05
        emitter.particle_mass = 0.3
        
        # Set particle temperature for thermal effects
        def custom_particle_creator(particle_data):
            from physics_simulation_module import Particle
            particle = Particle(
                particle_data['position'],
                particle_data['velocity'],
                particle_data.get('mass', 0.3),
                particle_data.get('radius', 0.1)
            )
            particle.lifetime = particle_data['lifetime']
            particle.color = particle_data.get('color_start', glm.vec3(1, 1, 1))
            particle.temperature = 1200 + random.random() * 300  # 1200-1500K for fire
            
            # Add some randomness to fire particles
            particle.velocity += glm.vec3(
                random.uniform(-0.5, 0.5),
                0,
                random.uniform(-0.5, 0.5)
            )
            
            return particle
            
        # This would require modifying the particle system to support custom particle creation
        
    def create_turbulence_fields(self):
        """Create turbulence force fields for realistic fire movement"""
        # Multiple small turbulence fields around the fire
        for i in range(5):
            x = random.uniform(-1.5, 1.5)
            y = random.uniform(-1.0, 1.0)
            z = random.uniform(-1.5, 1.5)
            
            self.physics_module.add_force_field(
                glm.vec3(x, y, z),
                strength=2.0,
                radius=1.5,
                field_type="turbulence",
                noise_strength=0.3,
                pulse_frequency=2.0
            )

class FluidDynamicsSimulation(BaseSimulation):
    """Advanced fluid dynamics simulation using smoothed particle hydrodynamics (SPH)"""
    
    def __init__(self):
        config = SimulationConfig(
            name="Fluid Dynamics",
            particle_count=1500,
            gravity=glm.vec3(0, -9.81, 0),
            enable_fluid_dynamics=True,
            enable_collisions=True
        )
        super().__init__(config)
        self.fluid_particles = []
        self.sph_parameters = {
            'rest_density': 1000.0,
            'gas_constant': 2000.0,
            'viscosity': 0.1,
            'surface_tension': 0.072,
            'kernel_radius': 0.2
        }
        
    def setup_simulation(self):
        """Set up fluid dynamics simulation"""
        # Create fluid container
        self.create_fluid_container()
        
        # Create initial fluid volume
        self.create_fluid_volume(glm.vec3(0, 0, 0), glm.vec3(2, 3, 2))
        
        # Set fluid dynamics parameters
        self.configure_sph_parameters()
        
    def create_fluid_container(self):
        """Create boundaries for fluid container"""
        # Container walls
        self.physics_module.collision_planes.clear()
        
        # Floor
        self.physics_module.collision_planes.append(
            self.physics_module.CollisionPlane(glm.vec3(0, 1, 0), -2.0, 0.1)
        )
        
        # Walls
        self.physics_module.collision_planes.append(
            self.physics_module.CollisionPlane(glm.vec3(1, 0, 0), -3.0, 0.1)
        )
        self.physics_module.collision_planes.append(
            self.physics_module.CollisionPlane(glm.vec3(-1, 0, 0), -3.0, 0.1)
        )
        self.physics_module.collision_planes.append(
            self.physics_module.CollisionPlane(glm.vec3(0, 0, 1), -3.0, 0.1)
        )
        self.physics_module.collision_planes.append(
            self.physics_module.CollisionPlane(glm.vec3(0, 0, -1), -3.0, 0.1)
        )
        
    def create_fluid_volume(self, center: glm.vec3, size: glm.vec3):
        """Create a volume of fluid particles"""
        particle_spacing = 0.15
        count_x = int(size.x / particle_spacing)
        count_y = int(size.y / particle_spacing)
        count_z = int(size.z / particle_spacing)
        
        for i in range(count_x):
            for j in range(count_y):
                for k in range(count_z):
                    x = center.x - size.x/2 + i * particle_spacing
                    y = center.y - size.y/2 + j * particle_spacing
                    z = center.z - size.z/2 + k * particle_spacing
                    
                    position = glm.vec3(x, y, z)
                    
                    # Create fluid particle
                    from physics_simulation_module import Particle
                    particle = Particle(
                        position=position,
                        velocity=glm.vec3(0, 0, 0),
                        mass=1.0,
                        radius=0.08,
                        color=glm.vec3(0.2, 0.4, 1.0)
                    )
                    particle.density = self.sph_parameters['rest_density']
                    
                    self.physics_module.add_particle(particle)
                    self.fluid_particles.append(particle)
                    
    def configure_sph_parameters(self):
        """Configure SPH simulation parameters"""
        # Enable fluid dynamics in physics module
        self.physics_module.settings.fluid_dynamics_enabled = True
        
        # Set fluid properties
        for particle in self.fluid_particles:
            particle.density = self.sph_parameters['rest_density']
            
    def update(self, dt: float):
        """Update fluid dynamics simulation"""
        super().update(dt)
        
        # Additional SPH calculations
        if self.initialized and not self.paused:
            self.calculate_sph_density_pressure()
            self.calculate_sph_forces(dt)
            
    def calculate_sph_density_pressure(self):
        """Calculate density and pressure for SPH particles"""
        kernel_radius = self.sph_parameters['kernel_radius']
        
        for i, particle in enumerate(self.fluid_particles):
            density = 0.0
            
            # Find neighboring particles
            for j, neighbor in enumerate(self.fluid_particles):
                if i == j:
                    continue
                    
                distance = glm.distance(particle.position, neighbor.position)
                
                if distance < kernel_radius:
                    # Poly6 kernel for density
                    h = kernel_radius
                    r = distance
                    volume = neighbor.mass / neighbor.density
                    
                    if r <= h:
                        kernel_value = (315 / (64 * math.pi * h**9)) * (h**2 - r**2)**3
                        density += neighbor.mass * kernel_value
                        
            # Update particle density and pressure
            if density > 0:
                particle.density = density
                # Calculate pressure using equation of state
                particle.pressure = self.sph_parameters['gas_constant'] * (density - self.sph_parameters['rest_density'])
                
    def calculate_sph_forces(self, dt: float):
        """Calculate SPH forces (pressure, viscosity, surface tension)"""
        kernel_radius = self.sph_parameters['kernel_radius']
        
        for i, particle in enumerate(self.fluid_particles):
            pressure_force = glm.vec3(0, 0, 0)
            viscosity_force = glm.vec3(0, 0, 0)
            
            for j, neighbor in enumerate(self.fluid_particles):
                if i == j:
                    continue
                    
                r = neighbor.position - particle.position
                distance = glm.length(r)
                
                if distance < kernel_radius and distance > 0.001:
                    direction = r / distance
                    
                    # Pressure force
                    if particle.density > 0 and neighbor.density > 0:
                        pressure_accel = -neighbor.mass * (particle.pressure + neighbor.pressure) / (2 * neighbor.density)
                        pressure_force += direction * pressure_accel
                        
                    # Viscosity force
                    velocity_diff = neighbor.velocity - particle.velocity
                    viscosity_force += self.sph_parameters['viscosity'] * neighbor.mass * velocity_diff / neighbor.density
                    
            # Apply forces
            particle.force += pressure_force + viscosity_force

class QuantumPhysicsSimulation(BaseSimulation):
    """Quantum physics simulation showing wave-particle duality and quantum effects"""
    
    def __init__(self):
        config = SimulationConfig(
            name="Quantum Physics",
            particle_count=1000,
            gravity=glm.vec3(0, 0, 0),
            enable_electromagnetism=True,
            enable_collisions=False
        )
        super().__init__(config)
        self.wave_functions = []
        self.quantum_states = []
        
    def setup_simulation(self):
        """Set up quantum physics simulation"""
        # Create quantum particles
        self.create_quantum_particles()
        
        # Set up double-slit experiment
        self.create_double_slit_experiment()
        
        # Add electromagnetic fields
        self.setup_electromagnetic_fields()
        
    def create_quantum_particles(self):
        """Create particles exhibiting quantum behavior"""
        # Create electron-like particles
        for i in range(10):
            emitter = self.particle_system.create_emitter(
                position=glm.vec3(-4, random.uniform(-1, 1), 0),
                emission_rate=2.0,
                particle_lifetime=10.0,
                particle_speed=2.0,
                spread_angle=1.0,
                emitter_type="point"
            )
            emitter.color_start = glm.vec3(0.8, 0.2, 0.8)
            emitter.color_end = glm.vec3(0.4, 0.1, 0.8)
            emitter.size_start = 0.05
            emitter.size_end = 0.05
            emitter.particle_mass = 0.1
            emitter.particle_radius = 0.03
            
            # Set electric charge for electromagnetic interactions
            def custom_particle_creator(particle_data):
                from physics_simulation_module import Particle
                particle = Particle(
                    particle_data['position'],
                    particle_data['velocity'],
                    particle_data.get('mass', 0.1),
                    particle_data.get('radius', 0.03)
                )
                particle.lifetime = particle_data['lifetime']
                particle.color = particle_data.get('color_start', glm.vec3(1, 1, 1))
                particle.charge = -1.0  # Electron charge (simplified)
                return particle
                
    def create_double_slit_experiment(self):
        """Set up a double-slit experiment visualization"""
        # Create barrier with two slits
        barrier_height = 3.0
        slit_width = 0.3
        slit_separation = 1.0
        
        # Left barrier
        self.physics_module.collision_planes.append(
            self.physics_module.CollisionPlane(glm.vec3(1, 0, 0), -1.0, 0.0)
        )
        
        # Right barrier
        self.physics_module.collision_planes.append(
            self.physics_module.CollisionPlane(glm.vec3(-1, 0, 0), -1.0, 0.0)
        )
        
        # Note: In a real quantum simulation, we would implement wave function
        # propagation instead of classical collisions
        
    def setup_electromagnetic_fields(self):
        """Set up electromagnetic fields for quantum effects"""
        # Magnetic field
        self.physics_module.set_magnetic_field(glm.vec3(0, 0.5, 0))
        
        # Electric field for particle acceleration
        self.physics_module.set_electric_field(glm.vec3(2.0, 0, 0))

class AstrophysicsSimulation(BaseSimulation):
    """Astrophysics simulation with gravitational bodies and orbital mechanics"""
    
    def __init__(self):
        config = SimulationConfig(
            name="Astrophysics",
            particle_count=500,
            gravity=glm.vec3(0, 0, 0),  # Custom gravity
            enable_collisions=True
        )
        super().__init__(config)
        self.celestial_bodies = []
        self.gravitational_constant = 6.67430e-11  # m³/kg/s² (scaled down)
        
    def setup_simulation(self):
        """Set up astrophysics simulation"""
        # Disable default gravity (we'll use custom gravitational forces)
        self.physics_module.settings.gravity = glm.vec3(0, 0, 0)
        
        # Create solar system
        self.create_solar_system()
        
        # Create background stars
        self.create_star_field()
        
    def create_solar_system(self):
        """Create a simplified solar system"""
        # Sun (central massive body)
        sun = self.create_celestial_body(
            position=glm.vec3(0, 0, 0),
            velocity=glm.vec3(0, 0, 0),
            mass=1000.0,
            radius=0.5,
            color=glm.vec3(1.0, 0.8, 0.1),
            name="Sun"
        )
        
        # Planets
        planets = [
            {"name": "Mercury", "distance": 1.5, "speed": 3.0, "color": glm.vec3(0.7, 0.7, 0.7), "size": 0.08},
            {"name": "Venus", "distance": 2.2, "speed": 2.2, "color": glm.vec3(0.9, 0.7, 0.3), "size": 0.12},
            {"name": "Earth", "distance": 3.0, "speed": 1.8, "color": glm.vec3(0.2, 0.4, 0.8), "size": 0.13},
            {"name": "Mars", "distance": 3.8, "speed": 1.5, "color": glm.vec3(0.8, 0.3, 0.2), "size": 0.1},
        ]
        
        for i, planet_data in enumerate(planets):
            angle = random.uniform(0, 2 * math.pi)
            distance = planet_data["distance"]
            
            position = glm.vec3(
                math.cos(angle) * distance,
                0,
                math.sin(angle) * distance
            )
            
            # Calculate orbital velocity (tangential to position)
            tangent = glm.normalize(glm.cross(position, glm.vec3(0, 1, 0)))
            velocity = tangent * planet_data["speed"]
            
            planet = self.create_celestial_body(
                position=position,
                velocity=velocity,
                mass=10.0,
                radius=planet_data["size"],
                color=planet_data["color"],
                name=planet_data["name"]
            )
            
    def create_celestial_body(self, position: glm.vec3, velocity: glm.vec3, mass: float, 
                            radius: float, color: glm.vec3, name: str = ""):
        """Create a celestial body with gravitational properties"""
        from physics_simulation_module import Particle
        
        body = Particle(
            position=position,
            velocity=velocity,
            mass=mass,
            radius=radius,
            color=color,
            lifetime=float('inf')
        )
        body.name = name
        body.celestial_body = True
        
        self.physics_module.add_particle(body)
        self.celestial_bodies.append(body)
        
        return body
        
    def create_star_field(self):
        """Create background star field"""
        for _ in range(200):
            # Random position in a large sphere
            theta = random.uniform(0, 2 * math.pi)
            phi = math.acos(2 * random.uniform(0, 1) - 1)
            r = random.uniform(8, 15)
            
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta) * 0.5  # Flatten a bit
            z = r * math.cos(phi)
            
            from physics_simulation_module import Particle
            star = Particle(
                position=glm.vec3(x, y, z),
                velocity=glm.vec3(0, 0, 0),
                mass=0.1,  # Very small mass - mostly visual
                radius=random.uniform(0.01, 0.05),
                color=glm.vec3(random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), random.uniform(0.7, 1.0)),
                lifetime=float('inf')
            )
            star.celestial_body = False
            star.fixed_position = True  # Don't move due to physics
            
            self.physics_module.add_particle(star)
            
    def update(self, dt: float):
        """Update astrophysics simulation with custom gravitational forces"""
        # First update base simulation
        super().update(dt)
        
        # Apply custom gravitational forces between celestial bodies
        self.apply_gravitational_forces()
        
    def apply_gravitational_forces(self):
        """Apply Newtonian gravity between celestial bodies"""
        for i, body1 in enumerate(self.celestial_bodies):
            for j, body2 in enumerate(self.celestial_bodies[i+1:], i+1):
                if body1.fixed_position and body2.fixed_position:
                    continue
                    
                # Calculate gravitational force
                r = body2.position - body1.position
                distance = glm.length(r)
                
                if distance > 0.1:  # Avoid division by zero and extreme forces
                    direction = glm.normalize(r)
                    force_magnitude = self.gravitational_constant * body1.mass * body2.mass / (distance * distance)
                    force = direction * force_magnitude
                    
                    # Apply forces (Newton's third law)
                    if not body1.fixed_position:
                        body1.force += force
                    if not body2.fixed_position:
                        body2.force -= force

class MagneticFieldSimulation(BaseSimulation):
    """Magnetic field simulation showing field lines and particle interactions"""
    
    def __init__(self):
        config = SimulationConfig(
            name="Magnetic Fields",
            particle_count=800,
            gravity=glm.vec3(0, 0, 0),
            enable_electromagnetism=True,
            enable_collisions=False
        )
        super().__init__(config)
        self.magnets = []
        self.field_lines = []
        
    def setup_simulation(self):
        """Set up magnetic field simulation"""
        # Create magnets
        self.create_magnets()
        
        # Create charged particles
        self.create_charged_particles()
        
        # Generate magnetic field lines
        self.generate_field_lines()
        
    def create_magnets(self):
        """Create magnetic poles"""
        # North pole
        north_magnet = self.create_magnetic_pole(
            position=glm.vec3(-2, 0, 0),
            strength=50.0,
            polarity=1.0,
            color=glm.vec3(1.0, 0.2, 0.2)  # Red for north
        )
        
        # South pole
        south_magnet = self.create_magnetic_pole(
            position=glm.vec3(2, 0, 0),
            strength=50.0,
            polarity=-1.0,
            color=glm.vec3(0.2, 0.2, 1.0)  # Blue for south
        )
        
        self.magnets.extend([north_magnet, south_magnet])
        
    def create_magnetic_pole(self, position: glm.vec3, strength: float, polarity: float, color: glm.vec3):
        """Create a magnetic pole"""
        from physics_simulation_module import Particle
        
        magnet = Particle(
            position=position,
            velocity=glm.vec3(0, 0, 0),
            mass=100.0,  # Heavy so they don't move much
            radius=0.3,
            color=color,
            lifetime=float('inf')
        )
        magnet.magnetic_strength = strength
        magnet.polarity = polarity
        magnet.fixed_position = True
        
        self.physics_module.add_particle(magnet)
        return magnet
        
    def create_charged_particles(self):
        """Create charged particles that interact with magnetic fields"""
        # Positive charges (protons)
        for i in range(5):
            emitter = self.particle_system.create_emitter(
                position=glm.vec3(-3, random.uniform(-1, 1), random.uniform(-1, 1)),
                emission_rate=1.0,
                particle_lifetime=8.0,
                particle_speed=2.0,
                spread_angle=10.0,
                emitter_type="point"
            )
            emitter.color_start = glm.vec3(1.0, 0.3, 0.3)  # Red for positive
            emitter.color_end = glm.vec3(0.8, 0.2, 0.2)
            emitter.size_start = 0.08
            emitter.size_end = 0.08
            emitter.particle_mass = 1.0
            
            # Set positive charge
            def positive_particle_creator(particle_data):
                from physics_simulation_module import Particle
                particle = Particle(
                    particle_data['position'],
                    particle_data['velocity'],
                    particle_data.get('mass', 1.0),
                    particle_data.get('radius', 0.08)
                )
                particle.lifetime = particle_data['lifetime']
                particle.color = particle_data.get('color_start', glm.vec3(1, 1, 1))
                particle.charge = 1.0
                return particle
                
        # Negative charges (electrons)
        for i in range(5):
            emitter = self.particle_system.create_emitter(
                position=glm.vec3(3, random.uniform(-1, 1), random.uniform(-1, 1)),
                emission_rate=1.0,
                particle_lifetime=8.0,
                particle_speed=2.0,
                spread_angle=10.0,
                emitter_type="point"
            )
            emitter.color_start = glm.vec3(0.3, 0.3, 1.0)  # Blue for negative
            emitter.color_end = glm.vec3(0.2, 0.2, 0.8)
            emitter.size_start = 0.06
            emitter.size_end = 0.06
            emitter.particle_mass = 0.5
            
            # Set negative charge
            def negative_particle_creator(particle_data):
                from physics_simulation_module import Particle
                particle = Particle(
                    particle_data['position'],
                    particle_data['velocity'],
                    particle_data.get('mass', 0.5),
                    particle_data.get('radius', 0.06)
                )
                particle.lifetime = particle_data['lifetime']
                particle.color = particle_data.get('color_start', glm.vec3(1, 1, 1))
                particle.charge = -1.0
                return particle
                
    def generate_field_lines(self):
        """Generate magnetic field line visualizations"""
        # This would create visual representations of magnetic field lines
        # For now, we'll create some static visual guides
        num_lines = 20
        for i in range(num_lines):
            angle = i * 2 * math.pi / num_lines
            radius = 0.5
            start_pos = glm.vec3(
                math.cos(angle) * radius,
                math.sin(angle) * radius,
                0
            )
            
            # Create a visual particle to represent field line
            from physics_simulation_module import Particle
            field_marker = Particle(
                position=start_pos,
                velocity=glm.vec3(0, 0, 0),
                mass=0.0,  # No mass - purely visual
                radius=0.02,
                color=glm.vec3(0.8, 0.8, 0.2),
                lifetime=float('inf')
            )
            field_marker.fixed_position = True
            
            self.physics_module.add_particle(field_marker)
            self.field_lines.append(field_marker)
            
    def update(self, dt: float):
        """Update magnetic field simulation"""
        super().update(dt)
        
        # Update magnetic fields based on magnet positions
        self.update_magnetic_fields()
        
    def update_magnetic_fields(self):
        """Calculate and update magnetic field strengths"""
        # Simplified magnetic field calculation
        total_field = glm.vec3(0, 0, 0)
        
        for magnet in self.magnets:
            # Each magnet contributes to the global field
            # This is a simplified dipole field calculation
            for particle in self.physics_module.particles:
                if hasattr(particle, 'charge') and abs(particle.charge) > 0:
                    r = particle.position - magnet.position
                    distance = glm.length(r)
                    
                    if distance > 0.1:
                        direction = glm.normalize(r)
                        # Simplified magnetic field: B = (μ₀/4π) * (3r(m·r) - m) / r³
                        # We'll use a simplified version for visualization
                        field_strength = magnet.magnetic_strength * magnet.polarity / (distance * distance)
                        field_component = direction * field_strength
                        total_field += field_component
                        
        # Apply the magnetic field to the physics module
        self.physics_module.set_magnetic_field(total_field * 0.1)  # Scale down for stability

class PlasmaSimulation(BaseSimulation):
    """Plasma physics simulation with ionized gas and electromagnetic effects"""
    
    def __init__(self):
        config = SimulationConfig(
            name="Plasma Physics",
            particle_count=2000,
            gravity=glm.vec3(0, 0, 0),
            enable_electromagnetism=True,
            enable_thermal_dynamics=True
        )
        super().__init__(config)
        self.plasma_temperature = 10000.0  # High temperature for plasma
        self.electric_fields = []
        
    def setup_simulation(self):
        """Set up plasma simulation"""
        # Create plasma chamber boundaries
        self.create_plasma_chamber()
        
        # Create ionized particles
        self.create_plasma_particles()
        
        # Set up electric fields for plasma confinement
        self.setup_confinement_fields()
        
        # Set high global temperature
        self.physics_module.global_temperature = self.plasma_temperature
        
    def create_plasma_chamber(self):
        """Create containment chamber for plasma"""
        # Spherical boundary
        self.physics_module.collision_planes.clear()
        
        # Create a spherical boundary using multiple planes
        num_planes = 12
        for i in range(num_planes):
            angle = i * 2 * math.pi / num_planes
            normal = glm.vec3(math.cos(angle), math.sin(angle), 0)
            self.physics_module.collision_planes.append(
                self.physics_module.CollisionPlane(normal, -2.5, 0.9)
            )
            
        # Add some vertical boundaries
        self.physics_module.collision_planes.append(
            self.physics_module.CollisionPlane(glm.vec3(0, 1, 0), -2.5, 0.9)
        )
        self.physics_module.collision_planes.append(
            self.physics_module.CollisionPlane(glm.vec3(0, -1, 0), -2.5, 0.9)
        )
        
    def create_plasma_particles(self):
        """Create ionized plasma particles"""
        # Ions (positive charge)
        ion_emitter = self.particle_system.create_emitter(
            position=glm.vec3(0, 0, 0),
            emission_rate=40.0,
            particle_lifetime=6.0,
            particle_speed=1.5,
            spread_angle=180.0,
            emitter_type="sphere",
            size=glm.vec3(1.5, 1.5, 1.5)
        )
        ion_emitter.color_start = glm.vec3(1.0, 0.4, 0.4)  # Reddish for ions
        ion_emitter.color_end = glm.vec3(0.8, 0.2, 0.2)
        ion_emitter.size_start = 0.07
        ion_emitter.size_end = 0.07
        ion_emitter.particle_mass = 2.0
        
        # Electrons (negative charge)
        electron_emitter = self.particle_system.create_emitter(
            position=glm.vec3(0, 0, 0),
            emission_rate=60.0,
            particle_lifetime=6.0,
            particle_speed=2.0,  # Electrons move faster
            spread_angle=180.0,
            emitter_type="sphere",
            size=glm.vec3(1.5, 1.5, 1.5)
        )
        electron_emitter.color_start = glm.vec3(0.4, 0.4, 1.0)  # Bluish for electrons
        electron_emitter.color_end = glm.vec3(0.2, 0.2, 0.8)
        electron_emitter.size_start = 0.05
        electron_emitter.size_end = 0.05
        electron_emitter.particle_mass = 0.5
        
        # Set charges for plasma particles
        def ion_particle_creator(particle_data):
            from physics_simulation_module import Particle
            particle = Particle(
                particle_data['position'],
                particle_data['velocity'],
                particle_data.get('mass', 2.0),
                particle_data.get('radius', 0.07)
            )
            particle.lifetime = particle_data['lifetime']
            particle.color = particle_data.get('color_start', glm.vec3(1, 1, 1))
            particle.charge = 1.0  # Positive charge for ions
            particle.temperature = self.plasma_temperature + random.random() * 5000
            return particle
            
        def electron_particle_creator(particle_data):
            from physics_simulation_module import Particle
            particle = Particle(
                particle_data['position'],
                particle_data['velocity'],
                particle_data.get('mass', 0.5),
                particle_data.get('radius', 0.05)
            )
            particle.lifetime = particle_data['lifetime']
            particle.color = particle_data.get('color_start', glm.vec3(1, 1, 1))
            particle.charge = -1.0  # Negative charge for electrons
            particle.temperature = self.plasma_temperature + random.random() * 5000
            return particle
            
    def setup_confinement_fields(self):
        """Set up electromagnetic fields for plasma confinement"""
        # Radial electric field for confinement
        self.physics_module.add_force_field(
            glm.vec3(0, 0, 0),
            strength=15.0,
            radius=3.0,
            field_type="radial"
        )
        
        # Magnetic field for helical motion
        self.physics_module.set_magnetic_field(glm.vec3(0, 0.5, 0))
        
        # Oscillating electric field for plasma heating
        oscillating_field = self.physics_module.add_force_field(
            glm.vec3(0, 0, 0),
            strength=8.0,
            radius=2.5,
            field_type="turbulence"
        )
        oscillating_field.pulse_frequency = 3.0
        oscillating_field.noise_strength = 0.5

class ChemicalReactionSimulation(BaseSimulation):
    """Chemical reaction simulation with different particle types and reactions"""
    
    def __init__(self):
        config = SimulationConfig(
            name="Chemical Reactions",
            particle_count=1500,
            gravity=glm.vec3(0, -2.0, 0),
            enable_collisions=True
        )
        super().__init__(config)
        self.particle_types = {}
        self.reactions = []
        self.reaction_zones = []
        
    def setup_simulation(self):
        """Set up chemical reaction simulation"""
        # Define particle types
        self.define_particle_types()
        
        # Create initial reactants
        self.create_initial_reactants()
        
        # Define chemical reactions
        self.define_reactions()
        
        # Create reaction zones
        self.create_reaction_zones()
        
    def define_particle_types(self):
        """Define different types of chemical particles"""
        self.particle_types = {
            'hydrogen': {
                'color': glm.vec3(1.0, 1.0, 1.0),  # White
                'radius': 0.06,
                'mass': 0.5,
                'reactivity': 0.8
            },
            'oxygen': {
                'color': glm.vec3(1.0, 0.2, 0.2),  # Red
                'radius': 0.08,
                'mass': 1.0,
                'reactivity': 0.6
            },
            'water': {
                'color': glm.vec3(0.2, 0.5, 1.0),  # Blue
                'radius': 0.09,
                'mass': 1.5,
                'reactivity': 0.1
            },
            'carbon': {
                'color': glm.vec3(0.3, 0.3, 0.3),  # Gray
                'radius': 0.1,
                'mass': 2.0,
                'reactivity': 0.4
            },
            'carbon_dioxide': {
                'color': glm.vec3(0.5, 0.5, 0.5),  # Dark gray
                'radius': 0.11,
                'mass': 2.2,
                'reactivity': 0.2
            }
        }
        
    def create_initial_reactants(self):
        """Create initial reactant particles"""
        # Hydrogen particles (left side)
        hydrogen_emitter = self.particle_system.create_emitter(
            position=glm.vec3(-2, 0, 0),
            emission_rate=15.0,
            particle_lifetime=10.0,
            particle_speed=1.0,
            spread_angle=30.0,
            emitter_type="box",
            size=glm.vec3(0.5, 1.0, 0.5)
        )
        hydrogen_emitter.color_start = self.particle_types['hydrogen']['color']
        hydrogen_emitter.color_end = self.particle_types['hydrogen']['color'] * 0.8
        hydrogen_emitter.size_start = self.particle_types['hydrogen']['radius']
        hydrogen_emitter.size_end = self.particle_types['hydrogen']['radius']
        hydrogen_emitter.particle_mass = self.particle_types['hydrogen']['mass']
        
        # Oxygen particles (right side)
        oxygen_emitter = self.particle_system.create_emitter(
            position=glm.vec3(2, 0, 0),
            emission_rate=8.0,
            particle_lifetime=10.0,
            particle_speed=1.0,
            spread_angle=30.0,
            emitter_type="box",
            size=glm.vec3(0.5, 1.0, 0.5)
        )
        oxygen_emitter.color_start = self.particle_types['oxygen']['color']
        oxygen_emitter.color_end = self.particle_types['oxygen']['color'] * 0.8
        oxygen_emitter.size_start = self.particle_types['oxygen']['radius']
        oxygen_emitter.size_end = self.particle_types['oxygen']['radius']
        oxygen_emitter.particle_mass = self.particle_types['oxygen']['mass']
        
        # Set particle type identifiers
        def hydrogen_particle_creator(particle_data):
            from physics_simulation_module import Particle
            particle = Particle(
                particle_data['position'],
                particle_data['velocity'],
                self.particle_types['hydrogen']['mass'],
                self.particle_types['hydrogen']['radius']
            )
            particle.lifetime = particle_data['lifetime']
            particle.color = self.particle_types['hydrogen']['color']
            particle.particle_type = 'hydrogen'
            particle.reactivity = self.particle_types['hydrogen']['reactivity']
            return particle
            
        def oxygen_particle_creator(particle_data):
            from physics_simulation_module import Particle
            particle = Particle(
                particle_data['position'],
                particle_data['velocity'],
                self.particle_types['oxygen']['mass'],
                self.particle_types['oxygen']['radius']
            )
            particle.lifetime = particle_data['lifetime']
            particle.color = self.particle_types['oxygen']['color']
            particle.particle_type = 'oxygen'
            particle.reactivity = self.particle_types['oxygen']['reactivity']
            return particle
            
    def define_reactions(self):
        """Define chemical reactions"""
        self.reactions = [
            {
                'name': 'water_formation',
                'reactants': ['hydrogen', 'oxygen'],
                'products': ['water'],
                'ratio': [2, 1, 2],  # 2H₂ + O₂ → 2H₂O
                'energy_release': 2.0,
                'activation_energy': 0.5
            },
            {
                'name': 'combustion',
                'reactants': ['carbon', 'oxygen'],
                'products': ['carbon_dioxide'],
                'ratio': [1, 1, 1],  # C + O₂ → CO₂
                'energy_release': 3.0,
                'activation_energy': 0.8
            }
        ]
        
    def create_reaction_zones(self):
        """Create zones where reactions are more likely to occur"""
        # Central reaction zone
        reaction_zone = {
            'position': glm.vec3(0, 0, 0),
            'radius': 1.5,
            'catalyst_strength': 2.0
        }
        self.reaction_zones.append(reaction_zone)
        
        # Add a catalyst force field to encourage reactions
        self.physics_module.add_force_field(
            reaction_zone['position'],
            strength=reaction_zone['catalyst_strength'],
            radius=reaction_zone['radius'],
            field_type="radial"
        )
        
    def update(self, dt: float):
        """Update chemical reaction simulation"""
        super().update(dt)
        
        # Check for and process chemical reactions
        self.process_reactions()
        
    def process_reactions(self):
        """Process chemical reactions between particles"""
        # This is a simplified reaction system
        for reaction in self.reactions:
            self.check_reaction(reaction)
            
    def check_reaction(self, reaction: Dict):
        """Check if a specific reaction can occur"""
        reactants = reaction['reactants']
        products = reaction['products']
        
        # Find particles that can react
        potential_reactants = {rtype: [] for rtype in reactants}
        
        for particle in self.physics_module.particles:
            if hasattr(particle, 'particle_type') and particle.particle_type in reactants:
                potential_reactants[particle.particle_type].append(particle)
                
        # Check if we have enough reactants
        can_react = True
        for rtype in reactants:
            if len(potential_reactants[rtype]) < reaction['ratio'][reactants.index(rtype)]:
                can_react = False
                break
                
        if can_react:
            # Perform reaction (simplified - just change particle types)
            self.execute_reaction(reaction, potential_reactants)
            
    def execute_reaction(self, reaction: Dict, reactants: Dict):
        """Execute a chemical reaction"""
        # For simplicity, we'll just change the properties of existing particles
        # In a more advanced system, we'd create new particles and remove old ones
        
        # Take the first available reactants
        for rtype in reaction['reactants']:
            needed = reaction['ratio'][reaction['reactants'].index(rtype)]
            available = reactants[rtype][:needed]
            
            for particle in available:
                # Convert to product (simplified - just change type and properties)
                product_type = reaction['products'][0]  # Take first product for simplicity
                particle.particle_type = product_type
                particle.color = self.particle_types[product_type]['color']
                particle.radius = self.particle_types[product_type]['radius']
                particle.mass = self.particle_types[product_type]['mass']
                
                # Release reaction energy as velocity
                energy_boost = glm.length(particle.velocity) + reaction['energy_release']
                if energy_boost > 0:
                    particle.velocity = glm.normalize(particle.velocity) * energy_boost

class ChaosTheorySimulation(BaseSimulation):
    """Chaos theory demonstration with sensitive dependence on initial conditions"""
    
    def __init__(self):
        config = SimulationConfig(
            name="Chaos Theory",
            particle_count=500,
            gravity=glm.vec3(0, 0, 0),
            enable_collisions=False
        )
        super().__init__(config)
        self.double_pendulums = []
        self.initial_conditions = []
        self.trajectories = []
        
    def setup_simulation(self):
        """Set up chaos theory demonstration"""
        # Create multiple double pendulums with slightly different initial conditions
        self.create_double_pendulums()
        
        # Create Lorenz attractor particles
        self.create_lorenz_attractor()
        
    def create_double_pendulums(self):
        """Create multiple double pendulums to demonstrate chaos"""
        num_pendulums = 5
        base_angle = math.pi / 2  # 90 degrees
        
        for i in range(num_pendulums):
            # Slightly different initial angles
            angle_variation = (i - num_pendulums//2) * 0.01
            initial_angle = base_angle + angle_variation
            
            pendulum = {
                'angle1': initial_angle,
                'angle2': initial_angle,
                'angular_velocity1': 0.0,
                'angular_velocity2': 0.0,
                'length1': 1.0,
                'length2': 1.0,
                'mass1': 1.0,
                'mass2': 1.0,
                'trail': [],
                'color': glm.vec3(
                    i / num_pendulums,
                    1.0 - i / num_pendulums,
                    0.5
                )
            }
            self.double_pendulums.append(pendulum)
            
    def create_lorenz_attractor(self):
        """Create particles following Lorenz attractor equations"""
        # Lorenz system parameters
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        
        for i in range(10):
            # Slightly different initial conditions
            x = 1.0 + i * 0.01
            y = 1.0 + i * 0.01
            z = 1.0 + i * 0.01
            
            lorenz_particle = {
                'position': glm.vec3(x, y, z),
                'velocity': glm.vec3(0, 0, 0),
                'sigma': sigma,
                'rho': rho,
                'beta': beta,
                'trail': [],
                'color': glm.vec3(0.8, 0.3, 0.8)
            }
            self.trajectories.append(lorenz_particle)
            
    def update(self, dt: float):
        """Update chaos theory simulation"""
        super().update(dt)
        
        # Update double pendulums
        self.update_double_pendulums(dt)
        
        # Update Lorenz attractor
        self.update_lorenz_attractor(dt)
        
    def update_double_pendulums(self, dt: float):
        """Update double pendulum physics"""
        g = 9.81  # gravity
        
        for pendulum in self.double_pendulums:
            # Extract parameters
            m1, m2 = pendulum['mass1'], pendulum['mass2']
            l1, l2 = pendulum['length1'], pendulum['length2']
            a1, a2 = pendulum['angle1'], pendulum['angle2']
            a1_v, a2_v = pendulum['angular_velocity1'], pendulum['angular_velocity2']
            
            # Double pendulum equations (simplified)
            num1 = -g * (2 * m1 + m2) * math.sin(a1)
            num2 = -m2 * g * math.sin(a1 - 2 * a2)
            num3 = -2 * math.sin(a1 - a2) * m2
            num4 = a2_v * a2_v * l2 + a1_v * a1_v * l1 * math.cos(a1 - a2)
            den = l1 * (2 * m1 + m2 - m2 * math.cos(2 * a1 - 2 * a2))
            
            a1_a = (num1 + num2 + num3 * num4) / den
            
            num1 = 2 * math.sin(a1 - a2)
            num2 = a1_v * a1_v * l1 * (m1 + m2)
            num3 = g * (m1 + m2) * math.cos(a1)
            num4 = a2_v * a2_v * l2 * m2 * math.cos(a1 - a2)
            den = l2 * (2 * m1 + m2 - m2 * math.cos(2 * a1 - 2 * a2))
            
            a2_a = (num1 * (num2 + num3 + num4)) / den
            
            # Update angular velocities and angles
            pendulum['angular_velocity1'] += a1_a * dt
            pendulum['angular_velocity2'] += a2_a * dt
            pendulum['angle1'] += pendulum['angular_velocity1'] * dt
            pendulum['angle2'] += pendulum['angular_velocity2'] * dt
            
            # Calculate positions
            x1 = l1 * math.sin(pendulum['angle1'])
            y1 = -l1 * math.cos(pendulum['angle1'])
            x2 = x1 + l2 * math.sin(pendulum['angle2'])
            y2 = y1 - l2 * math.cos(pendulum['angle2'])
            
            # Store trail for visualization
            trail_point = glm.vec3(x2, y2, 0)
            pendulum['trail'].append(trail_point)
            if len(pendulum['trail']) > 100:  # Limit trail length
                pendulum['trail'].pop(0)
                
    def update_lorenz_attractor(self, dt: float):
        """Update Lorenz attractor system"""
        for trajectory in self.trajectories:
            x, y, z = trajectory['position'].x, trajectory['position'].y, trajectory['position'].z
            sigma, rho, beta = trajectory['sigma'], trajectory['rho'], trajectory['beta']
            
            # Lorenz equations
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            # Update position
            trajectory['position'] += glm.vec3(dx, dy, dz) * dt * 0.05  # Scale for visualization
            
            # Store trail
            trajectory['trail'].append(glm.vec3(trajectory['position']))
            if len(trajectory['trail']) > 200:
                trajectory['trail'].pop(0)

# Simulation factory for easy creation
class SimulationFactory:
    """Factory class for creating different simulation types"""
    
    @staticmethod
    def create_simulation(simulation_type: str, **kwargs) -> BaseSimulation:
        """Create a simulation of the specified type"""
        simulations = {
            "basic": BasicParticleSimulation,
            "fountain": FountainSimulation,
            "fire": FireSimulation,
            "fluid": FluidDynamicsSimulation,
            "quantum": QuantumPhysicsSimulation,
            "astrophysics": AstrophysicsSimulation,
            "magnetic": MagneticFieldSimulation,
            "plasma": PlasmaSimulation,
            "chemical": ChemicalReactionSimulation,
            "chaos": ChaosTheorySimulation
        }
        
        if simulation_type in simulations:
            return simulations[simulation_type](**kwargs)
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")
            
    @staticmethod
    def get_available_simulations() -> List[str]:
        """Get list of available simulation types"""
        return [
            "basic", "fountain", "fire", "fluid", "quantum",
            "astrophysics", "magnetic", "plasma", "chemical", "chaos"
        ]

# Demo and testing
if __name__ == "__main__":
    # Test each simulation briefly
    factory = SimulationFactory()
    
    print("Testing available simulations:")
    for sim_type in factory.get_available_simulations():
        try:
            print(f"Creating {sim_type} simulation...")
            sim = factory.create_simulation(sim_type)
            sim.initialize()
            sim.update(0.1)  # Quick update
            sim.cleanup()
            print(f"✓ {sim_type} simulation test passed")
        except Exception as e:
            print(f"✗ {sim_type} simulation test failed: {e}")
            
    print("All simulation tests completed!")
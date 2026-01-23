"""
Complete Physics Simulation Module
Advanced physics engine with multiple integration methods, collision detection, and force fields
"""

import numpy as np
import glm
from typing import Dict, List, Any, Optional, Tuple
import numba
from numba import jit, prange
import math
import time

class PhysicsSettings:
    """Configuration settings for physics simulation"""
    def __init__(self):
        self.gravity = glm.vec3(0, -9.81, 0)
        self.air_density = 1.2
        self.drag_coefficient = 0.47
        self.time_scale = 1.0
        self.substeps = 1
        self.collision_enabled = True
        self.fluid_dynamics_enabled = False
        self.soft_body_enabled = False
        self.rigid_body_enabled = True
        self.max_velocity = 100.0
        self.solver_iterations = 3
        
        # Advanced settings
        self.enable_verlet_integration = True
        self.enable_sleeping = True
        self.contact_restitution = 0.8
        self.contact_friction = 0.6
        self.wind_force = glm.vec3(0, 0, 0)

class Particle:
    """Advanced particle with physical properties"""
    def __init__(self, 
                 position: glm.vec3, 
                 velocity: glm.vec3 = None,
                 mass: float = 1.0,
                 radius: float = 0.1,
                 color: glm.vec3 = None,
                 lifetime: float = float('inf'),
                 particle_type: str = "standard"):
        
        self.position = glm.vec3(position)
        self.previous_position = glm.vec3(position)
        self.velocity = velocity if velocity else glm.vec3(0, 0, 0)
        self.force = glm.vec3(0, 0, 0)
        self.mass = mass
        self.inverse_mass = 1.0 / mass if mass > 0 else 0
        self.radius = radius
        self.age = 0.0
        self.lifetime = lifetime
        self.type = particle_type
        
        # Visual properties
        if color:
            self.color = glm.vec3(color)
        else:
            self.color = glm.vec3(1.0, 0.5, 0.2)
            
        # Physical properties
        self.density = 1000.0  # kg/m³
        self.restitution = 0.8
        self.friction = 0.6
        self.drag_area = math.pi * radius * radius
        
        # State tracking
        self.sleeping = False
        self.awake_timer = 0.0
        self.collision_count = 0
        
        # Advanced properties
        self.temperature = 293.15  # Kelvin
        self.charge = 0.0  # Electric charge
        self.vorticity = glm.vec3(0, 0, 0)
        
    def apply_force(self, force: glm.vec3):
        """Apply a force to the particle"""
        if not self.sleeping:
            self.force += force
            
    def apply_impulse(self, impulse: glm.vec3):
        """Apply an impulse (instant change in momentum)"""
        if not self.sleeping:
            self.velocity += impulse * self.inverse_mass
            
    def update_age(self, dt: float):
        """Update particle age"""
        self.age += dt
        
    def is_alive(self) -> bool:
        """Check if particle is still alive"""
        return self.age < self.lifetime
        
    def get_volume(self) -> float:
        """Calculate particle volume"""
        return (4.0/3.0) * math.pi * self.radius * self.radius * self.radius
        
    def get_bounding_sphere(self) -> Tuple[glm.vec3, float]:
        """Get bounding sphere for collision detection"""
        return self.position, self.radius
        
    def wake_up(self):
        """Wake up sleeping particle"""
        self.sleeping = False
        self.awake_timer = 2.0  # Stay awake for 2 seconds
        
@jit(nopython=True, fastmath=True)
def apply_gravity_numba(positions, forces, masses, gravity, dt):
    """Numba-accelerated gravity application"""
    for i in prange(len(positions)):
        if masses[i] > 0:
            forces[i*3] += gravity[0] * masses[i]
            forces[i*3+1] += gravity[1] * masses[i]
            forces[i*3+2] += gravity[2] * masses[i]

@jit(nopython=True, fastmath=True)
def verlet_integration_numba(positions, previous_positions, forces, masses, dt, inverse_masses):
    """Numba-accelerated Verlet integration"""
    dt_sq = dt * dt
    for i in prange(len(positions)):
        if masses[i] > 0:
            # Store current position
            temp_x = positions[i*3]
            temp_y = positions[i*3+1]
            temp_z = positions[i*3+2]
            
            # Verlet integration
            acceleration_x = forces[i*3] * inverse_masses[i]
            acceleration_y = forces[i*3+1] * inverse_masses[i]
            acceleration_z = forces[i*3+2] * inverse_masses[i]
            
            positions[i*3] = (2 * positions[i*3] - previous_positions[i*3] + 
                            acceleration_x * dt_sq)
            positions[i*3+1] = (2 * positions[i*3+1] - previous_positions[i*3+1] + 
                              acceleration_y * dt_sq)
            positions[i*3+2] = (2 * positions[i*3+2] - previous_positions[i*3+2] + 
                              acceleration_z * dt_sq)
            
            # Update previous position
            previous_positions[i*3] = temp_x
            previous_positions[i*3+1] = temp_y
            previous_positions[i*3+2] = temp_z

class ForceField:
    """Advanced force field with various field types"""
    def __init__(self, 
                 position: glm.vec3, 
                 strength: float, 
                 radius: float, 
                 field_type: str = "radial",
                 falloff: str = "inverse_square",
                 direction: glm.vec3 = None):
        
        self.position = glm.vec3(position)
        self.strength = strength
        self.radius = radius
        self.type = field_type
        self.falloff = falloff
        self.direction = direction if direction else glm.vec3(0, 1, 0)
        self.enabled = True
        self.noise_strength = 0.0
        self.pulse_frequency = 0.0
        self.creation_time = time.time()
        
    def get_force_at_position(self, position: glm.vec3, current_time: float = 0.0) -> glm.vec3:
        """Calculate force at given position"""
        if not self.enabled:
            return glm.vec3(0, 0, 0)
            
        direction = self.position - position
        distance = glm.length(direction)
        
        if distance > self.radius or distance < 0.001:
            return glm.vec3(0, 0, 0)
            
        # Calculate falloff
        if self.falloff == "inverse_square":
            falloff = 1.0 / (distance * distance + 0.1)
        elif self.falloff == "linear":
            falloff = 1.0 - (distance / self.radius)
        elif self.falloff == "constant":
            falloff = 1.0
        else:
            falloff = 1.0 / (distance * distance + 0.1)
            
        # Apply noise
        noise = 1.0
        if self.noise_strength > 0:
            import random
            noise = 1.0 + random.uniform(-self.noise_strength, self.noise_strength)
            
        # Apply pulsing
        pulse = 1.0
        if self.pulse_frequency > 0:
            pulse = 0.5 + 0.5 * math.sin(current_time * self.pulse_frequency * 2 * math.pi)
            
        strength = self.strength * falloff * noise * pulse
        
        if self.type == "radial":
            if distance > 0.001:
                direction = glm.normalize(direction)
            return direction * strength
            
        elif self.type == "vortex":
            if distance > 0.001:
                direction = glm.normalize(direction)
            tangent = glm.cross(direction, glm.vec3(0, 1, 0))
            return tangent * strength
            
        elif self.type == "directional":
            return glm.normalize(self.direction) * strength
            
        elif self.type == "turbulence":
            # Simple turbulence using sine waves
            turb_x = math.sin(position.x * 10 + current_time * 5)
            turb_y = math.sin(position.y * 8 + current_time * 4)
            turb_z = math.sin(position.z * 12 + current_time * 6)
            return glm.vec3(turb_x, turb_y, turb_z) * strength * 0.1
            
        elif self.type == "magnetic":
            # Simplified magnetic field
            if distance > 0.001:
                direction = glm.normalize(direction)
            tangent = glm.cross(direction, self.direction)
            return tangent * strength
            
        else:
            return glm.vec3(0, 0, 0)

class CollisionPlane:
    """Infinite collision plane"""
    def __init__(self, normal: glm.vec3, distance: float, restitution: float = 0.8):
        self.normal = glm.normalize(normal)
        self.distance = distance
        self.restitution = restitution
        self.friction = 0.6
        
    def distance_to_point(self, point: glm.vec3) -> float:
        """Calculate signed distance from point to plane"""
        return glm.dot(point, self.normal) - self.distance

class PhysicsSimulationModule:
    """Complete physics simulation engine with advanced features"""
    
    def __init__(self, settings: PhysicsSettings = None):
        self.settings = settings if settings else PhysicsSettings()
        
        # Simulation state
        self.particles = []
        self.rigid_bodies = []
        self.soft_bodies = []
        self.force_fields = []
        self.collision_planes = []
        self.springs = []
        
        # Spatial partitioning
        self.spatial_grid = {}
        self.grid_cell_size = 1.0
        
        # Performance tracking
        self.frame_time = 0.0
        self.collision_time = 0.0
        self.integration_time = 0.0
        self.force_calculation_time = 0.0
        
        # Advanced features
        self.wind_noise_time = 0.0
        self.global_temperature = 293.15
        self.electric_field = glm.vec3(0, 0, 0)
        self.magnetic_field = glm.vec3(0, 0, 0)
        
        # Initialize default collision boundaries
        self.initialize_default_boundaries()
        
    def initialize_default_boundaries(self):
        """Initialize default collision boundaries"""
        # Ground plane
        self.collision_planes.append(CollisionPlane(glm.vec3(0, 1, 0), -2.0))
        
        # Walls
        self.collision_planes.append(CollisionPlane(glm.vec3(1, 0, 0), -3.0))
        self.collision_planes.append(CollisionPlane(glm.vec3(-1, 0, 0), -3.0))
        self.collision_planes.append(CollisionPlane(glm.vec3(0, 0, 1), -3.0))
        self.collision_planes.append(CollisionPlane(glm.vec3(0, 0, -1), -3.0))
        
        # Ceiling
        self.collision_planes.append(CollisionPlane(glm.vec3(0, -1, 0), -3.0))
        
    def add_particle(self, particle: Particle):
        """Add a particle to the simulation"""
        self.particles.append(particle)
        
    def add_force_field(self, position: glm.vec3, strength: float, radius: float, 
                       field_type: str = "radial", **kwargs) -> ForceField:
        """Add a force field to the simulation"""
        force_field = ForceField(position, strength, radius, field_type, **kwargs)
        self.force_fields.append(force_field)
        return force_field
        
    def add_spring(self, particle1: Particle, particle2: Particle, 
                  rest_length: float, stiffness: float, damping: float = 0.1):
        """Add a spring between two particles"""
        spring = {
            'particle1': particle1,
            'particle2': particle2,
            'rest_length': rest_length,
            'stiffness': stiffness,
            'damping': damping
        }
        self.springs.append(spring)
        
    def update(self, dt: float):
        """Update the entire physics simulation"""
        start_time = time.time()
        dt = dt * self.settings.time_scale
        
        # Update wind with noise
        self.update_wind_noise(dt)
        
        # Multiple substeps for stability
        substep_dt = dt / self.settings.substeps
        for _ in range(self.settings.substeps):
            self.update_forces(substep_dt)
            self.update_integration(substep_dt)
            
            if self.settings.collision_enabled:
                self.update_collisions(substep_dt)
                
            self.update_springs(substep_dt)
            self.update_particle_ages(substep_dt)
            
        self.update_spatial_partitioning()
        self.frame_time = time.time() - start_time
        
    def update_wind_noise(self, dt: float):
        """Update wind forces with Perlin-like noise"""
        self.wind_noise_time += dt
        import random
        # Simple pseudo-random wind variation
        wind_x = math.sin(self.wind_noise_time * 0.5) * 0.5
        wind_y = math.cos(self.wind_noise_time * 0.3) * 0.3
        wind_z = math.sin(self.wind_noise_time * 0.7) * 0.4
        
        self.settings.wind_force = glm.vec3(wind_x, wind_y, wind_z) * 2.0
        
    def update_forces(self, dt: float):
        """Calculate and apply all forces"""
        force_start = time.time()
        current_time = time.time()
        
        # Reset all forces
        for particle in self.particles:
            if not particle.sleeping:
                particle.force = glm.vec3(0, 0, 0)
                
        # Apply gravity
        self.apply_gravity()
        
        # Apply force fields
        self.apply_force_fields(current_time)
        
        # Apply aerodynamic forces
        if self.settings.fluid_dynamics_enabled:
            self.apply_aerodynamic_forces()
            
        # Apply electric and magnetic forces
        self.apply_electromagnetic_forces()
        
        # Apply thermal forces (buoyancy)
        self.apply_thermal_forces()
        
        self.force_calculation_time = time.time() - force_start
        
    def apply_gravity(self):
        """Apply gravity to all particles"""
        for particle in self.particles:
            if not particle.sleeping and particle.mass > 0:
                particle.force += self.settings.gravity * particle.mass
                
    def apply_force_fields(self, current_time: float):
        """Apply all force fields to particles"""
        for force_field in self.force_fields:
            for particle in self.particles:
                if not particle.sleeping:
                    force = force_field.get_force_at_position(particle.position, current_time)
                    particle.force += force * particle.mass
                    
    def apply_aerodynamic_forces(self):
        """Apply aerodynamic drag and wind forces"""
        for particle in self.particles:
            if not particle.sleeping:
                # Drag force
                relative_velocity = particle.velocity - self.settings.wind_force
                speed = glm.length(relative_velocity)
                
                if speed > 0.001:
                    drag_direction = -glm.normalize(relative_velocity)
                    drag_magnitude = (0.5 * self.settings.air_density * 
                                    speed * speed * 
                                    self.settings.drag_coefficient * 
                                    particle.drag_area)
                    particle.force += drag_direction * drag_magnitude
                    
    def apply_electromagnetic_forces(self):
        """Apply electric and magnetic forces to charged particles"""
        for particle in self.particles:
            if abs(particle.charge) > 0.001 and not particle.sleeping:
                # Electric force: F = qE
                particle.force += self.electric_field * particle.charge
                
                # Magnetic force: F = q(v × B)
                if glm.length(self.magnetic_field) > 0.001:
                    magnetic_force = particle.charge * glm.cross(particle.velocity, self.magnetic_field)
                    particle.force += magnetic_force
                    
    def apply_thermal_forces(self):
        """Apply thermal buoyancy forces"""
        for particle in self.particles:
            if not particle.sleeping:
                # Simple buoyancy based on temperature difference
                temp_difference = particle.temperature - self.global_temperature
                if abs(temp_difference) > 0.1:
                    buoyancy_force = glm.vec3(0, temp_difference * 0.01, 0)
                    particle.force += buoyancy_force * particle.mass
                    
    def update_integration(self, dt: float):
        """Update particle positions using numerical integration"""
        integration_start = time.time()
        
        if self.settings.enable_verlet_integration:
            self.verlet_integration(dt)
        else:
            self.euler_integration(dt)
            
        self.integration_time = time.time() - integration_start
        
    def verlet_integration(self, dt: float):
        """Verlet integration for better stability"""
        for particle in self.particles:
            if not particle.sleeping and particle.mass > 0:
                # Store current position
                temp_pos = particle.position
                
                # Verlet integration
                acceleration = particle.force * particle.inverse_mass
                particle.position = (2 * particle.position - 
                                   particle.previous_position + 
                                   acceleration * dt * dt)
                particle.previous_position = temp_pos
                
                # Update velocity for other calculations
                particle.velocity = (particle.position - particle.previous_position) / dt
                
                # Clamp maximum velocity
                speed = glm.length(particle.velocity)
                if speed > self.settings.max_velocity:
                    particle.velocity = glm.normalize(particle.velocity) * self.settings.max_velocity
                    
    def euler_integration(self, dt: float):
        """Semi-implicit Euler integration"""
        for particle in self.particles:
            if not particle.sleeping and particle.mass > 0:
                # Update velocity
                acceleration = particle.force * particle.inverse_mass
                particle.velocity += acceleration * dt
                
                # Clamp maximum velocity
                speed = glm.length(particle.velocity)
                if speed > self.settings.max_velocity:
                    particle.velocity = glm.normalize(particle.velocity) * self.settings.max_velocity
                    
                # Update position
                particle.position += particle.velocity * dt
                
    def update_collisions(self, dt: float):
        """Handle all collisions"""
        collision_start = time.time()
        
        # Particle-plane collisions
        for particle in self.particles:
            if not particle.sleeping:
                for plane in self.collision_planes:
                    self.handle_plane_collision(particle, plane)
                    
        # Particle-particle collisions (using spatial partitioning)
        self.handle_particle_collisions()
        
        self.collision_time = time.time() - collision_start
        
    def handle_plane_collision(self, particle: Particle, plane: CollisionPlane):
        """Handle collision between particle and plane"""
        distance = plane.distance_to_point(particle.position)
        
        if distance < particle.radius:
            # Collision detected
            penetration = particle.radius - distance
            particle.position += plane.normal * penetration
            
            # Calculate reflection
            normal_velocity = glm.dot(particle.velocity, plane.normal)
            if normal_velocity < 0:  # Moving toward plane
                # Apply restitution
                particle.velocity -= plane.normal * normal_velocity * (1 + plane.restitution)
                
                # Apply friction
                tangent_velocity = particle.velocity - plane.normal * normal_velocity
                particle.velocity -= tangent_velocity * plane.friction
                
            particle.wake_up()
            particle.collision_count += 1
            
    def handle_particle_collisions(self):
        """Handle collisions between particles using spatial partitioning"""
        # This is a simplified version - full implementation would use spatial hashing
        for i, p1 in enumerate(self.particles):
            if p1.sleeping:
                continue
                
            for j, p2 in enumerate(self.particles[i+1:], i+1):
                if p2.sleeping:
                    continue
                    
                direction = p1.position - p2.position
                distance = glm.length(direction)
                min_distance = p1.radius + p2.radius
                
                if distance < min_distance and distance > 0.001:
                    # Collision response
                    direction = glm.normalize(direction)
                    overlap = min_distance - distance
                    
                    # Move particles apart
                    if p1.mass > 0 and p2.mass > 0:
                        total_mass = p1.mass + p2.mass
                        p1.position += direction * (overlap * p2.mass / total_mass)
                        p2.position -= direction * (overlap * p1.mass / total_mass)
                        
                    # Velocity exchange (simplified)
                    p1.wake_up()
                    p2.wake_up()
                    
    def update_springs(self, dt: float):
        """Update spring forces"""
        for spring in self.springs:
            p1 = spring['particle1']
            p2 = spring['particle2']
            
            if p1.sleeping and p2.sleeping:
                continue
                
            direction = p2.position - p1.position
            distance = glm.length(direction)
            
            if distance > 0.001:
                direction = glm.normalize(direction)
                displacement = distance - spring['rest_length']
                
                # Spring force (Hooke's law)
                spring_force = direction * displacement * spring['stiffness']
                
                # Apply forces
                p1.force += spring_force
                p2.force -= spring_force
                
                # Damping force
                relative_velocity = p2.velocity - p1.velocity
                damping_force = direction * glm.dot(relative_velocity, direction) * spring['damping']
                
                p1.force += damping_force
                p2.force -= damping_force
                
    def update_particle_ages(self, dt: float):
        """Update particle ages and remove dead particles"""
        dead_particles = []
        for particle in self.particles:
            particle.update_age(dt)
            if not particle.is_alive():
                dead_particles.append(particle)
                
        for dead_particle in dead_particles:
            self.particles.remove(dead_particle)
            
    def update_spatial_partitioning(self):
        """Update spatial partitioning grid for collision optimization"""
        self.spatial_grid.clear()
        
        for particle in self.particles:
            grid_pos = (
                int(particle.position.x / self.grid_cell_size),
                int(particle.position.y / self.grid_cell_size),
                int(particle.position.z / self.grid_cell_size)
            )
            
            if grid_pos not in self.spatial_grid:
                self.spatial_grid[grid_pos] = []
            self.spatial_grid[grid_pos].append(particle)
            
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            'total_frame_time': self.frame_time,
            'force_calculation_time': self.force_calculation_time,
            'integration_time': self.integration_time,
            'collision_time': self.collision_time,
            'particle_count': len(self.particles),
            'force_field_count': len(self.force_fields),
            'spring_count': len(self.springs)
        }
        
    def set_gravity(self, gravity: glm.vec3):
        """Set gravity vector"""
        self.settings.gravity = gravity
        
    def set_electric_field(self, field: glm.vec3):
        """Set global electric field"""
        self.electric_field = field
        
    def set_magnetic_field(self, field: glm.vec3):
        """Set global magnetic field"""
        self.magnetic_field = field
        
    def clear_particles(self):
        """Remove all particles"""
        self.particles.clear()
        
    def clear_force_fields(self):
        """Remove all force fields"""
        self.force_fields.clear()
        
    def get_particle_at_position(self, position: glm.vec3, radius: float = 0.5) -> Optional[Particle]:
        """Find particle near given position"""
        for particle in self.particles:
            if glm.distance(particle.position, position) < radius:
                return particle
        return None

# Example usage and testing
if __name__ == "__main__":
    # Test the physics module
    settings = PhysicsSettings()
    physics = PhysicsSimulationModule(settings)
    
    # Add some test particles
    for i in range(10):
        particle = Particle(
            position=glm.vec3(i * 0.5 - 2.5, 5, 0),
            velocity=glm.vec3(0, 0, 0),
            mass=1.0,
            radius=0.2
        )
        physics.add_particle(particle)
        
    # Add a force field
    physics.add_force_field(glm.vec3(0, 0, 0), 5.0, 3.0, "vortex")
    
    # Run simulation for a few steps
    for step in range(100):
        physics.update(0.016)
        
    print("Physics simulation test completed successfully!")
    stats = physics.get_performance_stats()
    print(f"Performance stats: {stats}")

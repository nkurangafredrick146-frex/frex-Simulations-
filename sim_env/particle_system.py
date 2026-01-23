"""
Complete Particle System Module
Advanced particle system with multiple emitter types, rendering optimizations, and special effects
"""

import pygame
import numpy as np
import glm
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numba
from numba import jit, prange
import math
import time
from typing import List, Dict, Any, Optional, Tuple
import random

class ParticleRenderer:
    """Advanced particle rendering system with multiple rendering techniques"""
    
    def __init__(self):
        self.shaders = {}
        self.vaos = {}
        self.vbos = {}
        self.textures = {}
        self.particle_count = 0
        
        # Rendering settings
        self.point_size = 8.0
        self.enable_soft_particles = True
        self.enable_motion_blur = False
        self.blend_mode = GL_SRC_ALPHA
        self.depth_test = True
        
        # Performance optimization
        self.batch_size = 1000
        self.instance_rendering = True
        self.buffer_swap_technique = True
        
    def initialize(self):
        """Initialize all rendering resources"""
        self.compile_shaders()
        self.create_buffers()
        self.load_textures()
        
    def compile_shaders(self):
        """Compile all shader programs for different rendering techniques"""
        
        # Basic particle shader
        basic_vertex = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;
        layout (location = 2) in float aSize;
        layout (location = 3) in float aLife;
        
        out vec3 fragColor;
        out float life;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform float time;
        
        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            fragColor = aColor;
            life = aLife;
            
            // Size attenuation based on life
            gl_PointSize = aSize * (1.0 - life * 0.5);
        }
        """
        
        basic_fragment = """
        #version 330 core
        in vec3 fragColor;
        in float life;
        out vec4 FragColor;
        
        uniform sampler2D particleTexture;
        
        void main() {
            // Circular particles with smooth edges
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);
            
            if (dist > 0.5)
                discard;
                
            // Color based on life
            vec3 finalColor = fragColor;
            finalColor *= (1.0 - life * 0.3); // Fade with age
            
            // Texture sampling
            vec4 texColor = texture(particleTexture, gl_PointCoord);
            FragColor = vec4(finalColor * texColor.rgb, texColor.a * (1.0 - life));
            
            // Add glow effect for dying particles
            if (life > 0.8) {
                float glow = (life - 0.8) * 5.0;
                FragColor.rgb += vec3(1.0, 0.7, 0.3) * glow;
            }
        }
        """
        
        self.shaders['basic'] = self.create_shader_program(basic_vertex, basic_fragment)
        
        # Instanced rendering shader
        instanced_vertex = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;
        layout (location = 2) in float aSize;
        layout (location = 3) in float aLife;
        
        out vec3 fragColor;
        out float life;
        
        uniform mat4 view;
        uniform mat4 projection;
        uniform float time;
        
        void main() {
            // Billboard technique - always face camera
            vec3 right = vec3(view[0][0], view[1][0], view[2][0]);
            vec3 up = vec3(view[0][1], view[1][1], view[2][1]);
            
            vec3 pos = aPos + right * aPos.x * aSize + up * aPos.y * aSize;
            gl_Position = projection * view * vec4(pos, 1.0);
            
            fragColor = aColor;
            life = aLife;
            gl_PointSize = aSize * (1.0 - life * 0.3);
        }
        """
        
        self.shaders['instanced'] = self.create_shader_program(instanced_vertex, basic_fragment)
        
        # Advanced particle shader with lighting
        advanced_vertex = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;
        layout (location = 2) in float aSize;
        layout (location = 3) in float aLife;
        layout (location = 4) in vec3 aVelocity;
        
        out vec3 fragColor;
        out float life;
        out vec3 worldPos;
        out vec3 velocity;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform vec3 cameraPos;
        uniform float time;
        
        void main() {
            worldPos = aPos;
            velocity = aVelocity;
            
            // Advanced billboarding with velocity stretching
            vec3 toCamera = normalize(cameraPos - aPos);
            vec3 right = normalize(cross(toCamera, vec3(0, 1, 0)));
            vec3 up = normalize(cross(right, toCamera));
            
            // Stretch in velocity direction
            vec3 velDir = normalize(aVelocity);
            float speed = length(aVelocity);
            float stretch = min(speed * 0.1, 2.0);
            
            vec3 pos = aPos + right * aPos.x * aSize + up * aPos.y * aSize * stretch;
            gl_Position = projection * view * vec4(pos, 1.0);
            
            fragColor = aColor;
            life = aLife;
            gl_PointSize = aSize * (1.0 - life * 0.2);
        }
        """
        
        advanced_fragment = """
        #version 330 core
        in vec3 fragColor;
        in float life;
        in vec3 worldPos;
        in vec3 velocity;
        out vec4 FragColor;
        
        uniform vec3 lightPos;
        uniform float time;
        uniform sampler2D particleTexture;
        
        void main() {
            // Advanced particle rendering with lighting
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);
            
            if (dist > 0.5)
                discard;
                
            // Normal calculation for lighting
            vec3 normal = vec3(coord * 2.0, sqrt(1.0 - dot(coord, coord)));
            
            // Lighting calculation
            vec3 lightDir = normalize(lightPos - worldPos);
            float diff = max(dot(normal, lightDir), 0.2);
            
            // Color processing
            vec3 finalColor = fragColor * diff;
            
            // Life-based effects
            float lifeEffect = 1.0 - life;
            finalColor *= lifeEffect;
            
            // Velocity-based effects (trails)
            float speed = length(velocity);
            if (speed > 1.0) {
                finalColor += vec3(0.8, 0.6, 0.2) * min(speed * 0.1, 0.5);
            }
            
            // Texture and alpha
            vec4 texColor = texture(particleTexture, gl_PointCoord);
            float alpha = texColor.a * lifeEffect;
            
            // Glow for high-energy particles
            if (speed > 5.0) {
                alpha += 0.3 * min(speed * 0.1, 1.0);
            }
            
            FragColor = vec4(finalColor * texColor.rgb, alpha);
        }
        """
        
        self.shaders['advanced'] = self.create_shader_program(advanced_vertex, advanced_fragment)
        
    def create_shader_program(self, vertex_source, fragment_source):
        """Compile and link shader program"""
        try:
            vertex_shader = compileShader(vertex_source, GL_VERTEX_SHADER)
            fragment_shader = compileShader(fragment_source, GL_FRAGMENT_SHADER)
            return compileProgram(vertex_shader, fragment_shader)
        except Exception as e:
            print(f"Shader compilation error: {e}")
            return None
            
    def create_buffers(self):
        """Create OpenGL buffers for particle rendering"""
        # Basic particle VAO
        self.vaos['basic'] = glGenVertexArrays(1)
        self.vbos['basic'] = glGenBuffers(1)
        
        glBindVertexArray(self.vaos['basic'])
        glBindBuffer(GL_ARRAY_BUFFER, self.vbos['basic'])
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), ctypes.c_void_p(3 * sizeof(GLfloat)))
        glEnableVertexAttribArray(1)
        
        # Size attribute
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), ctypes.c_void_p(6 * sizeof(GLfloat)))
        glEnableVertexAttribArray(2)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        
    def load_textures(self):
        """Load particle textures"""
        # Create a simple procedural texture for particles
        texture_size = 64
        texture_data = np.zeros((texture_size, texture_size, 4), dtype=np.uint8)
        
        for y in range(texture_size):
            for x in range(texture_size):
                dx = x - texture_size // 2
                dy = y - texture_size // 2
                dist = math.sqrt(dx*dx + dy*dy) / (texture_size // 2)
                
                if dist <= 1.0:
                    # Create smooth circular gradient
                    intensity = int(255 * (1.0 - dist))
                    alpha = int(255 * (1.0 - dist * dist))  # Quadratic falloff
                    texture_data[y, x] = [255, 255, 255, alpha]
                else:
                    texture_data[y, x] = [0, 0, 0, 0]
                    
        # Upload texture to GPU
        self.textures['particle'] = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.textures['particle'])
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture_size, texture_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        
    def render_particles(self, particles, view_matrix, projection_matrix, camera_position, render_mode='basic'):
        """Render particles using specified technique"""
        if not particles:
            return
            
        shader = self.shaders.get(render_mode, self.shaders['basic'])
        if not shader:
            return
            
        glUseProgram(shader)
        
        # Prepare particle data
        particle_data = []
        for particle in particles:
            life_ratio = particle.age / particle.lifetime if particle.lifetime < float('inf') else 0.0
            particle_data.extend([
                particle.position.x, particle.position.y, particle.position.z,
                particle.color.x, particle.color.y, particle.color.z,
                particle.radius * self.point_size,
                life_ratio
            ])
            
        particle_data = np.array(particle_data, dtype=np.float32)
        self.particle_count = len(particles)
        
        # Upload to GPU
        glBindVertexArray(self.vaos['basic'])
        glBindBuffer(GL_ARRAY_BUFFER, self.vbos['basic'])
        glBufferData(GL_ARRAY_BUFFER, particle_data.nbytes, particle_data, GL_DYNAMIC_DRAW)
        
        # Set uniforms
        model_matrix = glm.mat4(1.0)
        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm.value_ptr(model_matrix))
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm.value_ptr(view_matrix))
        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection_matrix))
        glUniform1f(glGetUniformLocation(shader, "time"), time.time())
        glUniform3f(glGetUniformLocation(shader, "cameraPos"), camera_position.x, camera_position.y, camera_position.z)
        glUniform3f(glGetUniformLocation(shader, "lightPos"), 2.0, 5.0, 2.0)
        
        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.textures['particle'])
        glUniform1i(glGetUniformLocation(shader, "particleTexture"), 0)
        
        # Set rendering state
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        
        if self.depth_test:
            glEnable(GL_DEPTH_TEST)
        else:
            glDisable(GL_DEPTH_TEST)
            
        # Render particles
        glDrawArrays(GL_POINTS, 0, self.particle_count)
        
        # Cleanup
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glUseProgram(0)
        
        # Restore blend mode
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

class ParticleEmitter:
    """Advanced particle emitter with various emission patterns"""
    
    def __init__(self, 
                 position: glm.vec3 = None,
                 emission_rate: float = 10.0,
                 burst_count: int = 0,
                 particle_lifetime: float = 5.0,
                 particle_speed: float = 1.0,
                 spread_angle: float = 45.0,
                 emitter_type: str = "point",
                 size: glm.vec3 = None):
        
        self.position = position if position else glm.vec3(0, 0, 0)
        self.emission_rate = emission_rate  # particles per second
        self.burst_count = burst_count
        self.particle_lifetime = particle_lifetime
        self.particle_speed = particle_speed
        self.spread_angle = math.radians(spread_angle)
        self.emitter_type = emitter_type
        self.size = size if size else glm.vec3(1, 1, 1)
        
        # Emission state
        self.time_since_emission = 0.0
        self.active = True
        self.emitted_count = 0
        self.burst_emitted = 0
        
        # Particle properties
        self.particle_mass = 1.0
        self.particle_radius = 0.1
        self.color_start = glm.vec3(1.0, 0.5, 0.2)
        self.color_end = glm.vec3(0.2, 0.8, 1.0)
        self.size_start = 1.0
        self.size_end = 0.1
        self.randomize_properties = True
        
        # Advanced emission properties
        self.velocity_variation = 0.3
        self.lifetime_variation = 0.2
        self.angular_velocity = glm.vec3(0, 0, 0)
        self.force_override = None
        
    def get_emission_position(self) -> glm.vec3:
        """Get emission position based on emitter type"""
        if self.emitter_type == "point":
            return self.position
            
        elif self.emitter_type == "sphere":
            # Random point on sphere surface
            theta = random.uniform(0, 2 * math.pi)
            phi = math.acos(2 * random.uniform(0, 1) - 1)
            r = self.size.x
            
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = r * math.cos(phi)
            
            return self.position + glm.vec3(x, y, z)
            
        elif self.emitter_type == "box":
            # Random point in box
            x = random.uniform(-self.size.x/2, self.size.x/2)
            y = random.uniform(-self.size.y/2, self.size.y/2)
            z = random.uniform(-self.size.z/2, self.size.z/2)
            
            return self.position + glm.vec3(x, y, z)
            
        elif self.emitter_type == "circle":
            # Random point on circle (XZ plane)
            angle = random.uniform(0, 2 * math.pi)
            radius = self.size.x
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            
            return self.position + glm.vec3(x, 0, z)
            
        else:
            return self.position
            
    def get_emission_velocity(self, position: glm.vec3) -> glm.vec3:
        """Get emission velocity based on emitter type and spread"""
        base_direction = glm.vec3(0, 1, 0)  # Default upward
        
        if self.emitter_type == "sphere":
            # Outward from sphere center
            base_direction = glm.normalize(position - self.position)
        elif self.emitter_type == "circle":
            # Combination of upward and outward
            horizontal_dir = glm.normalize(glm.vec3(position.x - self.position.x, 0, position.z - self.position.z))
            base_direction = glm.normalize(horizontal_dir + glm.vec3(0, 2, 0))
            
        # Apply spread angle
        if self.spread_angle > 0:
            # Random direction within cone
            angle = random.uniform(0, self.spread_angle)
            rotation_axis = glm.normalize(glm.cross(base_direction, glm.vec3(0, 1, 0)))
            if glm.length(rotation_axis) < 0.001:
                rotation_axis = glm.vec3(1, 0, 0)
                
            rotation = glm.rotate(glm.mat4(1.0), angle, rotation_axis)
            base_direction = glm.vec3(rotation * glm.vec4(base_direction, 1.0))
            
        # Apply speed with variation
        speed = self.particle_speed * random.uniform(1 - self.velocity_variation, 1 + self.velocity_variation)
        
        return base_direction * speed
        
    def get_particle_color(self, age_ratio: float) -> glm.vec3:
        """Get particle color based on age"""
        return glm.mix(self.color_start, self.color_end, age_ratio)
        
    def get_particle_size(self, age_ratio: float) -> float:
        """Get particle size based on age"""
        return glm.mix(self.size_start, self.size_end, age_ratio)
        
    def update(self, dt: float) -> List[Dict]:
        """Update emitter and return new particles to emit"""
        if not self.active:
            return []
            
        particles_to_emit = []
        self.time_since_emission += dt
        
        # Continuous emission
        if self.emission_rate > 0:
            expected_emissions = self.emission_rate * self.time_since_emission
            num_emissions = int(expected_emissions)
            
            for _ in range(num_emissions):
                particles_to_emit.append(self.create_particle_data())
                
            self.time_since_emission -= num_emissions / self.emission_rate
            
        # Burst emission
        if self.burst_count > 0 and self.burst_emitted < self.burst_count:
            burst_now = min(self.burst_count - self.burst_emitted, int(self.burst_count * dt))
            for _ in range(burst_now):
                particles_to_emit.append(self.create_particle_data())
                self.burst_emitted += 1
                
        self.emitted_count += len(particles_to_emit)
        return particles_to_emit
        
    def create_particle_data(self) -> Dict:
        """Create data for a new particle"""
        position = self.get_emission_position()
        velocity = self.get_emission_velocity(position)
        
        # Apply lifetime variation
        lifetime = self.particle_lifetime * random.uniform(1 - self.lifetime_variation, 1 + self.lifetime_variation)
        
        particle_data = {
            'position': position,
            'velocity': velocity,
            'mass': self.particle_mass,
            'radius': self.particle_radius,
            'lifetime': lifetime,
            'color_start': self.color_start,
            'color_end': self.color_end,
            'size_start': self.size_start,
            'size_end': self.size_end,
            'angular_velocity': self.angular_velocity,
            'force_override': self.force_override
        }
        
        return particle_data

class ParticleSystem:
    """Complete particle system with advanced features"""
    
    def __init__(self, max_particles: int = 10000):
        self.max_particles = max_particles
        self.particles = []
        self.emitters = []
        self.force_fields = []
        
        # Rendering system
        self.renderer = ParticleRenderer()
        self.render_mode = 'advanced'
        
        # Performance optimization
        self.particle_pool = []
        self.reuse_particles = True
        self.update_enabled = True
        self.render_enabled = True
        
        # System statistics
        self.total_emitted = 0
        self.total_destroyed = 0
        self.performance_stats = {}
        
        # Initialize systems
        self.renderer.initialize()
        self.initialize_particle_pool()
        
    def initialize_particle_pool(self):
        """Initialize particle object pool for performance"""
        for _ in range(self.max_particles):
            from physics_simulation_module import Particle
            self.particle_pool.append(Particle(glm.vec3(0, 0, 0), glm.vec3(0, 0, 0)))
            
    def add_emitter(self, emitter: ParticleEmitter):
        """Add an emitter to the system"""
        self.emitters.append(emitter)
        
    def create_emitter(self, **kwargs) -> ParticleEmitter:
        """Create and add a new emitter"""
        emitter = ParticleEmitter(**kwargs)
        self.add_emitter(emitter)
        return emitter
        
    def add_particle(self, particle_data: Dict):
        """Add a particle to the system"""
        if len(self.particles) >= self.max_particles:
            if self.reuse_particles and self.particle_pool:
                # Reuse dead particle
                particle = self.particle_pool.pop()
                self.initialize_particle_from_data(particle, particle_data)
                self.particles.append(particle)
            else:
                # Remove oldest particle if pool is empty
                if self.particles:
                    old_particle = self.particles.pop(0)
                    self.initialize_particle_from_data(old_particle, particle_data)
                    self.particles.append(old_particle)
        else:
            # Create new particle
            from physics_simulation_module import Particle
            particle = Particle(
                particle_data['position'],
                particle_data['velocity'],
                particle_data.get('mass', 1.0),
                particle_data.get('radius', 0.1)
            )
            particle.lifetime = particle_data['lifetime']
            particle.color = particle_data.get('color_start', glm.vec3(1, 1, 1))
            self.particles.append(particle)
            
        self.total_emitted += 1
        
    def initialize_particle_from_data(self, particle, particle_data: Dict):
        """Initialize particle from emitter data"""
        particle.position = particle_data['position']
        particle.velocity = particle_data['velocity']
        particle.mass = particle_data.get('mass', 1.0)
        particle.radius = particle_data.get('radius', 0.1)
        particle.lifetime = particle_data['lifetime']
        particle.age = 0.0
        particle.color = particle_data.get('color_start', glm.vec3(1, 1, 1))
        particle.force = glm.vec3(0, 0, 0)
        
    def update(self, dt: float):
        """Update all particles and emitters"""
        if not self.update_enabled:
            return
            
        start_time = time.time()
        
        # Update emitters and add new particles
        for emitter in self.emitters:
            new_particles = emitter.update(dt)
            for particle_data in new_particles:
                self.add_particle(particle_data)
                
        # Update existing particles
        dead_particles = []
        for particle in self.particles:
            particle.update_age(dt)
            
            # Update visual properties based on age
            if particle.lifetime < float('inf'):
                life_ratio = particle.age / particle.lifetime
                # Update color based on age
                if hasattr(particle, 'color_start') and hasattr(particle, 'color_end'):
                    particle.color = glm.mix(particle.color_start, particle.color_end, life_ratio)
                # Update size based on age
                if hasattr(particle, 'size_start') and hasattr(particle, 'size_end'):
                    particle.radius = glm.mix(particle.size_start, particle.size_end, life_ratio)
                    
            if not particle.is_alive():
                dead_particles.append(particle)
                
        # Remove dead particles
        for dead_particle in dead_particles:
            self.particles.remove(dead_particle)
            if self.reuse_particles:
                self.particle_pool.append(dead_particle)
            self.total_destroyed += 1
            
        # Update performance stats
        update_time = time.time() - start_time
        self.performance_stats = {
            'update_time': update_time,
            'active_particles': len(self.particles),
            'available_pool': len(self.particle_pool),
            'active_emitters': len([e for e in self.emitters if e.active]),
            'total_emitted': self.total_emitted,
            'total_destroyed': self.total_destroyed
        }
        
    def render(self, view_matrix: glm.mat4, projection_matrix: glm.mat4, camera_position: glm.vec3):
        """Render all particles"""
        if not self.render_enabled or not self.particles:
            return
            
        self.renderer.render_particles(
            self.particles, 
            view_matrix, 
            projection_matrix, 
            camera_position,
            self.render_mode
        )
        
    def clear_particles(self):
        """Remove all particles"""
        if self.reuse_particles:
            self.particle_pool.extend(self.particles)
        self.particles.clear()
        
    def clear_emitters(self):
        """Remove all emitters"""
        self.emitters.clear()
        
    def set_render_mode(self, mode: str):
        """Set particle rendering mode"""
        if mode in ['basic', 'instanced', 'advanced']:
            self.render_mode = mode
            
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return self.performance_stats
        
    def create_fountain_effect(self, position: glm.vec3, height: float = 5.0) -> ParticleEmitter:
        """Create a fountain particle effect"""
        emitter = self.create_emitter(
            position=position,
            emission_rate=50.0,
            particle_lifetime=3.0,
            particle_speed=height / 2.0,
            spread_angle=30.0,
            emitter_type="point"
        )
        emitter.color_start = glm.vec3(0.2, 0.5, 1.0)
        emitter.color_end = glm.vec3(0.8, 0.9, 1.0)
        emitter.size_start = 0.2
        emitter.size_end = 0.05
        return emitter
        
    def create_fire_effect(self, position: glm.vec3, intensity: float = 1.0) -> ParticleEmitter:
        """Create a fire particle effect"""
        emitter = self.create_emitter(
            position=position,
            emission_rate=100.0 * intensity,
            particle_lifetime=1.5,
            particle_speed=2.0 * intensity,
            spread_angle=10.0,
            emitter_type="circle",
            size=glm.vec3(0.5, 0, 0.5)
        )
        emitter.color_start = glm.vec3(1.0, 0.3, 0.1)
        emitter.color_end = glm.vec3(1.0, 0.8, 0.1)
        emitter.size_start = 0.3
        emitter.size_end = 0.1
        return emitter
        
    def create_explosion_effect(self, position: glm.vec3, power: float = 1.0) -> ParticleEmitter:
        """Create an explosion particle effect"""
        emitter = self.create_emitter(
            position=position,
            emission_rate=0,  # No continuous emission
            burst_count=int(200 * power),
            particle_lifetime=2.0 * power,
            particle_speed=8.0 * power,
            spread_angle=180.0,  # Full sphere
            emitter_type="point"
        )
        emitter.color_start = glm.vec3(1.0, 0.6, 0.1)
        emitter.color_end = glm.vec3(0.8, 0.2, 0.1)
        emitter.size_start = 0.3 * power
        emitter.size_end = 0.05
        emitter.particle_mass = 0.8
        emitter.velocity_variation = 0.5
        emitter.lifetime_variation = 0.4
        
        # Set burst to emit immediately
        emitter.burst_emitted = 0  # Reset to emit full burst
        
        return emitter

    def create_smoke_effect(self, position: glm.vec3, intensity: float = 1.0) -> ParticleEmitter:
        """Create a smoke particle effect"""
        emitter = self.create_emitter(
            position=position,
            emission_rate=30.0 * intensity,
            particle_lifetime=4.0 * intensity,
            particle_speed=1.0 * intensity,
            spread_angle=45.0,
            emitter_type="point"
        )
        emitter.color_start = glm.vec3(0.3, 0.3, 0.3)
        emitter.color_end = glm.vec3(0.1, 0.1, 0.1)
        emitter.size_start = 0.4 * intensity
        emitter.size_end = 0.8 * intensity  # Smoke expands
        emitter.particle_mass = 0.3
        return emitter

    def create_spark_effect(self, position: glm.vec3, count: int = 20) -> ParticleEmitter:
        """Create a spark particle effect"""
        emitter = self.create_emitter(
            position=position,
            emission_rate=0,  # Burst only
            burst_count=count,
            particle_lifetime=1.5,
            particle_speed=6.0,
            spread_angle=60.0,
            emitter_type="point"
        )
        emitter.color_start = glm.vec3(1.0, 0.9, 0.1)
        emitter.color_end = glm.vec3(1.0, 0.5, 0.1)
        emitter.size_start = 0.08
        emitter.size_end = 0.02
        emitter.particle_mass = 0.2
        return emitter

    def create_magic_effect(self, position: glm.vec3, color: glm.vec3 = None) -> ParticleEmitter:
        """Create a magical/sparkly particle effect"""
        if color is None:
            color = glm.vec3(0.8, 0.3, 0.8)
            
        emitter = self.create_emitter(
            position=position,
            emission_rate=25.0,
            particle_lifetime=3.0,
            particle_speed=1.5,
            spread_angle=360.0,  # Full sphere
            emitter_type="sphere",
            size=glm.vec3(0.5, 0.5, 0.5)
        )
        emitter.color_start = color
        emitter.color_end = color * 0.6
        emitter.size_start = 0.1
        emitter.size_end = 0.03
        emitter.particle_mass = 0.4
        emitter.angular_velocity = glm.vec3(0, 2, 0)  # Slow rotation
        
        # Add some randomness
        emitter.velocity_variation = 0.4
        emitter.lifetime_variation = 0.3
        
        return emitter

    def create_black_hole_effect(self, position: glm.vec3, strength: float = 1.0) -> ParticleEmitter:
        """Create a black hole/attractor effect"""
        emitter = self.create_emitter(
            position=position,
            emission_rate=40.0 * strength,
            particle_lifetime=5.0,
            particle_speed=0.5,  # Slow initial speed
            spread_angle=180.0,
            emitter_type="sphere",
            size=glm.vec3(2.0, 2.0, 2.0)
        )
        emitter.color_start = glm.vec3(0.1, 0.1, 0.3)
        emitter.color_end = glm.vec3(0.8, 0.8, 1.0)
        emitter.size_start = 0.07
        emitter.size_end = 0.02
        emitter.particle_mass = 0.5
        
        # Override force to create attraction
        emitter.force_override = glm.vec3(0, 0, 0)  # Will be handled by external force field
        
        return emitter

    def create_lightning_effect(self, start_pos: glm.vec3, end_pos: glm.vec3) -> List[ParticleEmitter]:
        """Create a lightning bolt effect between two points"""
        emitters = []
        direction = end_pos - start_pos
        distance = glm.length(direction)
        segments = max(3, int(distance * 2))
        
        for i in range(segments):
            t = i / (segments - 1)
            base_pos = start_pos + direction * t
            
            # Add some randomness to create jagged lightning
            if i > 0 and i < segments - 1:
                offset = glm.vec3(
                    random.uniform(-0.2, 0.2),
                    random.uniform(-0.2, 0.2),
                    random.uniform(-0.2, 0.2)
                )
                base_pos += offset
            
            emitter = self.create_emitter(
                position=base_pos,
                emission_rate=15.0,
                particle_lifetime=0.8,
                particle_speed=0.3,
                spread_angle=30.0,
                emitter_type="point"
            )
            emitter.color_start = glm.vec3(0.7, 0.8, 1.0)
            emitter.color_end = glm.vec3(0.3, 0.5, 1.0)
            emitter.size_start = 0.15
            emitter.size_end = 0.03
            emitter.particle_mass = 0.1
            emitters.append(emitter)
            
        return emitters

    def create_nebula_effect(self, position: glm.vec3, size: float = 3.0) -> ParticleEmitter:
        """Create a nebula/cloud effect"""
        emitter = self.create_emitter(
            position=position,
            emission_rate=20.0,
            particle_lifetime=8.0,
            particle_speed=0.3,  # Very slow movement
            spread_angle=180.0,
            emitter_type="sphere",
            size=glm.vec3(size, size, size)
        )
        
        # Random nebula colors
        colors = [
            glm.vec3(0.8, 0.3, 0.8),  # Purple
            glm.vec3(0.3, 0.5, 0.9),  # Blue
            glm.vec3(0.9, 0.4, 0.3),  # Orange
            glm.vec3(0.3, 0.8, 0.6)   # Teal
        ]
        
        emitter.color_start = random.choice(colors)
        emitter.color_end = emitter.color_start * 0.5
        emitter.size_start = random.uniform(0.2, 0.4)
        emitter.size_end = random.uniform(0.4, 0.8)  # Clouds expand
        emitter.particle_mass = 0.2
        emitter.velocity_variation = 0.8
        emitter.lifetime_variation = 0.6
        
        return emitter

    def create_rainbow_effect(self, position: glm.vec3) -> List[ParticleEmitter]:
        """Create a rainbow of colored particles"""
        emitters = []
        colors = [
            glm.vec3(1.0, 0.0, 0.0),  # Red
            glm.vec3(1.0, 0.5, 0.0),  # Orange
            glm.vec3(1.0, 1.0, 0.0),  # Yellow
            glm.vec3(0.0, 1.0, 0.0),  # Green
            glm.vec3(0.0, 0.0, 1.0),  # Blue
            glm.vec3(0.3, 0.0, 0.5),  # Indigo
            glm.vec3(0.5, 0.0, 0.5)   # Violet
        ]
        
        for i, color in enumerate(colors):
            angle = i * (2 * math.pi / len(colors))
            offset = glm.vec3(math.cos(angle) * 1.5, 0, math.sin(angle) * 1.5)
            
            emitter = self.create_emitter(
                position=position + offset,
                emission_rate=15.0,
                particle_lifetime=4.0,
                particle_speed=2.0,
                spread_angle=45.0,
                emitter_type="point"
            )
            emitter.color_start = color
            emitter.color_end = color * 0.7
            emitter.size_start = 0.12
            emitter.size_end = 0.04
            emitters.append(emitter)
            
        return emitters

    def trigger_explosion(self, position: glm.vec3, power: float = 1.0):
        """Trigger an explosion at the specified position"""
        # Create explosion particles
        explosion_emitter = self.create_explosion_effect(position, power)
        explosion_particles = explosion_emitter.update(0.1)  # Force immediate emission
        
        for particle_data in explosion_particles:
            self.add_particle(particle_data)
            
        # Create secondary effects
        self.create_explosion_secondary_effects(position, power)
        
    def create_explosion_secondary_effects(self, position: glm.vec3, power: float):
        """Create secondary effects for explosions (smoke, sparks, etc.)"""
        # Smoke after explosion
        smoke_emitter = self.create_smoke_effect(position, power * 0.8)
        smoke_particles = smoke_emitter.update(0.1)
        for particle_data in smoke_particles:
            self.add_particle(particle_data)
            
        # Sparks
        spark_emitter = self.create_spark_effect(position, int(30 * power))
        spark_particles = spark_emitter.update(0.1)
        for particle_data in spark_particles:
            self.add_particle(particle_data)

    def create_particle_beam(self, start_pos: glm.vec3, end_pos: glm.vec3, beam_type: str = "energy"):
        """Create a particle beam between two points"""
        direction = end_pos - start_pos
        distance = glm.length(direction)
        num_emitters = max(2, int(distance * 3))
        
        for i in range(num_emitters):
            t = i / (num_emitters - 1)
            emitter_pos = start_pos + direction * t
            
            if beam_type == "energy":
                emitter = self.create_emitter(
                    position=emitter_pos,
                    emission_rate=10.0,
                    particle_lifetime=1.0,
                    particle_speed=0.5,
                    spread_angle=10.0,
                    emitter_type="point"
                )
                emitter.color_start = glm.vec3(0.2, 0.8, 1.0)
                emitter.color_end = glm.vec3(0.1, 0.4, 1.0)
            elif beam_type == "plasma":
                emitter = self.create_emitter(
                    position=emitter_pos,
                    emission_rate=8.0,
                    particle_lifetime=1.2,
                    particle_speed=0.3,
                    spread_angle=15.0,
                    emitter_type="point"
                )
                emitter.color_start = glm.vec3(1.0, 0.3, 0.3)
                emitter.color_end = glm.vec3(0.8, 0.1, 0.1)
            else:  # laser
                emitter = self.create_emitter(
                    position=emitter_pos,
                    emission_rate=12.0,
                    particle_lifetime=0.8,
                    particle_speed=1.0,
                    spread_angle=5.0,
                    emitter_type="point"
                )
                emitter.color_start = glm.vec3(1.0, 0.1, 0.1)
                emitter.color_end = glm.vec3(0.5, 0.0, 0.0)
                
            emitter.size_start = 0.08
            emitter.size_end = 0.02

    def create_particle_ribbon(self, points: List[glm.vec3], color: glm.vec3 = None):
        """Create a particle ribbon along a path of points"""
        if color is None:
            color = glm.vec3(0.8, 0.6, 0.2)
            
        for i in range(len(points) - 1):
            start_pos = points[i]
            end_pos = points[i + 1]
            
            emitter = self.create_emitter(
                position=start_pos,
                emission_rate=5.0,
                particle_lifetime=2.0,
                particle_speed=0.2,
                spread_angle=5.0,
                emitter_type="point"
            )
            emitter.color_start = color
            emitter.color_end = color * 0.6
            emitter.size_start = 0.1
            emitter.size_end = 0.03
            
            # Set initial velocity toward next point
            direction = glm.normalize(end_pos - start_pos)
            emitter.force_override = direction * 2.0

    def create_weather_effect(self, weather_type: str, intensity: float = 1.0):
        """Create weather-based particle effects"""
        if weather_type == "rain":
            self.create_rain_effect(intensity)
        elif weather_type == "snow":
            self.create_snow_effect(intensity)
        elif weather_type == "fog":
            self.create_fog_effect(intensity)
            
    def create_rain_effect(self, intensity: float = 1.0):
        """Create rain particle effect"""
        emitter = self.create_emitter(
            position=glm.vec3(0, 5, 0),
            emission_rate=100.0 * intensity,
            particle_lifetime=3.0,
            particle_speed=8.0 * intensity,
            spread_angle=10.0,
            emitter_type="box",
            size=glm.vec3(8, 0, 8)
        )
        emitter.color_start = glm.vec3(0.7, 0.8, 1.0)
        emitter.color_end = glm.vec3(0.4, 0.6, 1.0)
        emitter.size_start = 0.05
        emitter.size_end = 0.05
        emitter.particle_mass = 0.3
        
    def create_snow_effect(self, intensity: float = 1.0):
        """Create snow particle effect"""
        emitter = self.create_emitter(
            position=glm.vec3(0, 5, 0),
            emission_rate=80.0 * intensity,
            particle_lifetime=6.0,
            particle_speed=1.5 * intensity,
            spread_angle=30.0,
            emitter_type="box",
            size=glm.vec3(8, 0, 8)
        )
        emitter.color_start = glm.vec3(1.0, 1.0, 1.0)
        emitter.color_end = glm.vec3(0.9, 0.9, 1.0)
        emitter.size_start = 0.08
        emitter.size_end = 0.08
        emitter.particle_mass = 0.1
        emitter.velocity_variation = 0.4
        
    def create_fog_effect(self, intensity: float = 1.0):
        """Create fog particle effect"""
        emitter = self.create_emitter(
            position=glm.vec3(0, 0, 0),
            emission_rate=50.0 * intensity,
            particle_lifetime=10.0,
            particle_speed=0.1 * intensity,
            spread_angle=180.0,
            emitter_type="box",
            size=glm.vec3(6, 2, 6)
        )
        emitter.color_start = glm.vec3(0.8, 0.8, 0.8)
        emitter.color_end = glm.vec3(0.6, 0.6, 0.6)
        emitter.size_start = 0.3
        emitter.size_end = 0.5  # Fog expands
        emitter.particle_mass = 0.05

    def get_emitter_by_id(self, emitter_id: int) -> Optional[ParticleEmitter]:
        """Get emitter by its index in the list"""
        if 0 <= emitter_id < len(self.emitters):
            return self.emitters[emitter_id]
        return None

    def remove_emitter(self, emitter: ParticleEmitter):
        """Remove an emitter from the system"""
        if emitter in self.emitters:
            self.emitters.remove(emitter)

    def remove_emitter_by_id(self, emitter_id: int):
        """Remove emitter by its index"""
        if 0 <= emitter_id < len(self.emitters):
            self.emitters.pop(emitter_id)

    def set_emitter_active(self, emitter: ParticleEmitter, active: bool):
        """Set whether an emitter is active"""
        emitter.active = active

    def set_all_emitters_active(self, active: bool):
        """Set all emitters active or inactive"""
        for emitter in self.emitters:
            emitter.active = active

    def get_emitter_count(self) -> int:
        """Get the number of active emitters"""
        return len(self.emitters)

    def get_active_emitter_count(self) -> int:
        """Get the number of active emitters"""
        return len([e for e in self.emitters if e.active])

    def clear_dead_emitters(self):
        """Remove emitters that are no longer emitting particles"""
        # Keep emitters that are active or have burst particles left to emit
        self.emitters = [
            e for e in self.emitters 
            if e.active or (e.burst_count > 0 and e.burst_emitted < e.burst_count)
        ]

    def optimize_performance(self, target_fps: float = 60.0, current_fps: float = 60.0):
        """Automatically adjust settings for performance"""
        fps_ratio = current_fps / target_fps
        
        if fps_ratio < 0.8:  # If FPS is below 80% of target
            # Reduce particle count
            if len(self.particles) > 1000:
                # Remove oldest particles
                particles_to_remove = min(100, len(self.particles) - 1000)
                self.particles = self.particles[particles_to_remove:]
                
            # Reduce emission rates
            for emitter in self.emitters:
                emitter.emission_rate *= 0.9
                
            # Switch to simpler rendering
            self.render_mode = 'basic'
            
        elif fps_ratio > 1.2:  # If we have performance headroom
            # Increase emission rates slightly
            for emitter in self.emitters:
                emitter.emission_rate *= 1.05
                
            # Use advanced rendering if we have headroom
            if fps_ratio > 1.5:
                self.render_mode = 'advanced'

    def save_particle_state(self) -> Dict:
        """Save current particle system state for serialization"""
        state = {
            'particles': [],
            'emitters': [],
            'max_particles': self.max_particles,
            'render_mode': self.render_mode
        }
        
        # Save particle data
        for particle in self.particles:
            particle_data = {
                'position': [particle.position.x, particle.position.y, particle.position.z],
                'velocity': [particle.velocity.x, particle.velocity.y, particle.velocity.z],
                'mass': particle.mass,
                'radius': particle.radius,
                'color': [particle.color.x, particle.color.y, particle.color.z],
                'age': particle.age,
                'lifetime': particle.lifetime
            }
            state['particles'].append(particle_data)
            
        # Save emitter data
        for emitter in self.emitters:
            emitter_data = {
                'position': [emitter.position.x, emitter.position.y, emitter.position.z],
                'emission_rate': emitter.emission_rate,
                'burst_count': emitter.burst_count,
                'particle_lifetime': emitter.particle_lifetime,
                'particle_speed': emitter.particle_speed,
                'spread_angle': math.degrees(emitter.spread_angle),
                'emitter_type': emitter.emitter_type,
                'size': [emitter.size.x, emitter.size.y, emitter.size.z],
                'active': emitter.active,
                'color_start': [emitter.color_start.x, emitter.color_start.y, emitter.color_start.z],
                'color_end': [emitter.color_end.x, emitter.color_end.y, emitter.color_end.z]
            }
            state['emitters'].append(emitter_data)
            
        return state

    def load_particle_state(self, state: Dict):
        """Load particle system state from serialized data"""
        self.clear_particles()
        self.clear_emitters()
        
        # Load particles
        for particle_data in state['particles']:
            from physics_simulation_module import Particle
            particle = Particle(
                position=glm.vec3(*particle_data['position']),
                velocity=glm.vec3(*particle_data['velocity']),
                mass=particle_data['mass'],
                radius=particle_data['radius'],
                color=glm.vec3(*particle_data['color'])
            )
            particle.age = particle_data['age']
            particle.lifetime = particle_data['lifetime']
            self.particles.append(particle)
            
        # Load emitters
        for emitter_data in state['emitters']:
            emitter = ParticleEmitter(
                position=glm.vec3(*emitter_data['position']),
                emission_rate=emitter_data['emission_rate'],
                burst_count=emitter_data['burst_count'],
                particle_lifetime=emitter_data['particle_lifetime'],
                particle_speed=emitter_data['particle_speed'],
                spread_angle=emitter_data['spread_angle'],
                emitter_type=emitter_data['emitter_type'],
                size=glm.vec3(*emitter_data['size'])
            )
            emitter.active = emitter_data['active']
            emitter.color_start = glm.vec3(*emitter_data['color_start'])
            emitter.color_end = glm.vec3(*emitter_data['color_end'])
            self.emitters.append(emitter)
            
        self.max_particles = state['max_particles']
        self.render_mode = state.get('render_mode', 'advanced')

# Utility functions for particle system management
class ParticleSystemManager:
    """Manager for multiple particle systems"""
    
    def __init__(self):
        self.systems = {}
        self.active_systems = set()
        
    def create_system(self, name: str, max_particles: int = 10000) -> ParticleSystem:
        """Create a new particle system"""
        system = ParticleSystem(max_particles)
        self.systems[name] = system
        self.active_systems.add(name)
        return system
        
    def get_system(self, name: str) -> Optional[ParticleSystem]:
        """Get a particle system by name"""
        return self.systems.get(name)
        
    def remove_system(self, name: str):
        """Remove a particle system"""
        if name in self.systems:
            del self.systems[name]
            self.active_systems.discard(name)
            
    def set_system_active(self, name: str, active: bool):
        """Set a system active or inactive"""
        if active:
            self.active_systems.add(name)
        else:
            self.active_systems.discard(name)
            
    def update_all(self, dt: float):
        """Update all active particle systems"""
        for name in self.active_systems:
            if name in self.systems:
                self.systems[name].update(dt)
                
    def render_all(self, view_matrix: glm.mat4, projection_matrix: glm.mat4, camera_position: glm.vec3):
        """Render all active particle systems"""
        for name in self.active_systems:
            if name in self.systems:
                self.systems[name].render(view_matrix, projection_matrix, camera_position)

# Demo and testing
if __name__ == "__main__":
    # Test the particle system
    print("Testing Particle System...")
    
    # Create a particle system
    ps = ParticleSystem(max_particles=1000)
    
    # Test various effects
    effects = [
        ("Fountain", ps.create_fountain_effect(glm.vec3(-2, 0, 0))),
        ("Fire", ps.create_fire_effect(glm.vec3(0, 0, 0))),
        ("Explosion", ps.create_explosion_effect(glm.vec3(2, 0, 0))),
        ("Magic", ps.create_magic_effect(glm.vec3(0, 2, 0))),
        ("Smoke", ps.create_smoke_effect(glm.vec3(0, -2, 0)))
    ]
    
    print("Created effects:")
    for name, emitter in effects:
        print(f"  ✓ {name}")
        
    # Test explosion triggering
    ps.trigger_explosion(glm.vec3(1, 1, 1), 1.5)
    print("✓ Explosion triggered")
    
    # Test weather effects
    ps.create_weather_effect("rain", 1.0)
    print("✓ Weather effects created")
    
    # Test performance
    stats = ps.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("Particle System test completed successfully!")
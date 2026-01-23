#!/usr/bin/env python3
"""
frex Simulations - Updated Main Application
Enhanced main application with complete ultimate integration
"""

import pygame
import sys
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import *
import glm
import json
import time
from datetime import datetime

class EnhancedVideoSimulationSoftware:
    def __init__(self):
        # Core application state
        self.running = True
        self.simulation_time = 0.0
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
        
        # Enhanced simulation management
        self.simulations = {}
        self.current_simulation = None
        self.simulation_types = {
            "fluid": "FluidDynamicsSimulation",
            "quantum": "QuantumPhysicsSimulation", 
            "astrophysics": "AstrophysicsSimulation",
            "particles": "BasicParticleSimulation",
            "fountain": "FountainSimulation",
            "fire": "FireSimulation",
            "quantum_field": "QuantumFieldSimulation",
            "ultimate_integration": "UltimateIntegrationSimulation"  # NEW
        }
        
        # Ultimate Integration System
        self.ultimate_integration = None
        self.integration_active = False
        
        # Enhanced camera and view
        self.camera_position = glm.vec3(0, 0, 5)
        self.camera_target = glm.vec3(0, 0, 0)
        self.camera_up = glm.vec3(0, 1, 0)
        self.camera_speed = 2.0
        self.camera_modes = ["free", "orbit", "follow", "quantum"]
        self.current_camera_mode = "free"
        
        # Enhanced input state
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        self.keys_pressed = set()
        
        # Performance monitoring
        self.frame_times = []
        self.average_frame_time = 0.016
        self.performance_overlay = True
        
        # Configuration
        self.config = self.load_config()
        
    def load_config(self):
        """Load enhanced application configuration"""
        default_config = {
            "window_width": 1200,
            "window_height": 800,
            "max_particles": 10000,
            "physics_steps_per_frame": 1,
            "enable_audio": True,
            "enable_ml": False,
            "export_quality": "high",
            "enable_ultimate_integration": True,  # NEW
            "integration_level": 0.5,
            "neural_interface": True,
            "quantum_security": True
        }
        
        try:
            with open('config.json', 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except FileNotFoundError:
            print("No config file found, using defaults")
            
        return default_config
        
    def initialize(self):
        """Initialize the enhanced complete application"""
        print("🚀 Initializing frex Simulations...")
        
        # Initialize Pygame and OpenGL
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.config["window_width"], self.config["window_height"]), 
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        )
        pygame.display.set_caption("frex Simulations - Ultimate Integrated Platform")
        
        # Set enhanced window icon
        self.set_window_icon()
        
        # Initialize OpenGL with advanced features
        self.initialize_opengl()
        
        # Load all simulation modules
        self.initialize_simulation_modules()
        
        # Initialize Ultimate Integration System if enabled
        if self.config["enable_ultimate_integration"]:
            self.initialize_ultimate_integration()
            
        # Initialize enhanced GUI system
        self.initialize_gui()
        
        # Initialize audio system if enabled
        if self.config["enable_audio"]:
            self.initialize_audio()
            
        # Initialize export system
        self.initialize_export_system()
        
        print("✅ frex Simulations initialized successfully!")
        print(f"🎯 Available simulations: {list(self.simulation_types.keys())}")
        if self.ultimate_integration:
            print(f"🔗 Ultimate Integration: ACTIVE (Level: {self.config['integration_level']})")
        
    def set_window_icon(self):
        """Create and set an enhanced program icon"""
        icon_surface = pygame.Surface((32, 32), pygame.SRCALPHA)
        
        # Draw quantum-inspired icon
        pygame.draw.circle(icon_surface, (0, 200, 255, 255), (16, 16), 12)
        pygame.draw.circle(icon_surface, (255, 100, 0, 255), (8, 8), 4)
        pygame.draw.circle(icon_surface, (100, 255, 0, 255), (24, 24), 4)
        
        # Add neural network lines
        for i in range(4):
            angle = i * np.pi / 2
            start_x = 16 + 8 * np.cos(angle)
            start_y = 16 + 8 * np.sin(angle)
            end_x = 16 + 12 * np.cos(angle + 0.5)
            end_y = 16 + 12 * np.sin(angle + 0.5)
            pygame.draw.line(icon_surface, (255, 255, 255, 200), 
                           (start_x, start_y), (end_x, end_y), 1)
            
        pygame.display.set_icon(icon_surface)
        
    def initialize_opengl(self):
        """Initialize advanced OpenGL settings"""
        print("🔧 Initializing OpenGL...")
        
        # Basic OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glPointSize(4.0)
        
        # Advanced features
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        
        # Enhanced rendering features
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # Set clear color
        glClearColor(0.08, 0.08, 0.12, 1.0)
        
        # Check OpenGL version
        version = glGetString(GL_VERSION).decode()
        print(f"📊 OpenGL Version: {version}")
        
    def initialize_simulation_modules(self):
        """Initialize all simulation modules"""
        print("🔄 Loading simulation modules...")
        
        try:
            # Import and initialize core modules
            from physics_simulation_module import PhysicsSimulationModule
            from particle_system import ParticleSystem
            from rendering_engine import RenderingEngine
            from example_simulations import (
                BasicParticleSimulation, FountainSimulation, 
                FireSimulation, FluidDynamicsSimulation
            )
            
            # Initialize core systems
            self.physics_module = PhysicsSimulationModule()
            self.particle_system = ParticleSystem(max_particles=self.config["max_particles"])
            self.rendering_engine = RenderingEngine()
            
            # Preload common simulations
            self.simulations["basic"] = BasicParticleSimulation()
            self.simulations["fountain"] = FountainSimulation()
            self.simulations["fire"] = FireSimulation()
            self.simulations["fluid"] = FluidDynamicsSimulation()
            
            # Load default simulation
            self.current_simulation = self.simulations["basic"]
            self.current_simulation.initialize()
            
        except ImportError as e:
            print(f"❌ Error loading simulation modules: {e}")
            sys.exit(1)
            
    def initialize_ultimate_integration(self):
        """Initialize the Ultimate Integration System"""
        print("🔗 Initializing Ultimate Integration System...")
        
        try:
            from complete_ultimate_integration import UnifiedSimulationOrchestrator
            
            self.ultimate_integration = UnifiedSimulationOrchestrator()
            
            # Configure integration based on user settings
            integration_config = {
                "max_integration_level": self.config["integration_level"],
                "enable_neural_interface": self.config["neural_interface"],
                "enable_security_system": self.config["quantum_security"]
            }
            self.ultimate_integration.config.update(integration_config)
            
            # Initialize all integrated systems
            self.ultimate_integration.initialize_all_systems()
            
            self.integration_active = True
            print("✅ Ultimate Integration System initialized!")
            
        except ImportError as e:
            print(f"⚠️ Ultimate Integration not available: {e}")
            self.ultimate_integration = None
            self.integration_active = False
            
    def initialize_gui(self):
        """Initialize the enhanced real-time GUI"""
        try:
            from realtime_gui import RealTimeGUI
            self.gui = RealTimeGUI(self)
            self.gui.initialize()
            
            # Add integration controls to GUI if integration is active
            if self.integration_active:
                self.gui.add_integration_controls(self.ultimate_integration)
                
        except ImportError:
            print("⚠️ GUI module not available, running in headless mode")
            self.gui = None
            
    def initialize_audio(self):
        """Initialize audio system"""
        try:
            from audio_integration import AudioSystem
            self.audio_system = AudioSystem()
            self.audio_system.initialize()
        except ImportError:
            print("⚠️ Audio system not available")
            self.audio_system = None
            
    def initialize_export_system(self):
        """Initialize export capabilities"""
        try:
            from export_options import ExportManager
            self.export_manager = ExportManager()
        except ImportError:
            print("⚠️ Export system not available")
            self.export_manager = None
            
    def run(self):
        """Enhanced main application loop with ultimate integration"""
        clock = pygame.time.Clock()
        print("🎬 Starting enhanced main simulation loop...")
        
        while self.running:
            frame_start_time = time.time()
            
            # Calculate delta time
            dt = clock.tick(60) / 1000.0
            
            # Handle input events
            self.handle_events()
            
            # Update camera based on input
            self.update_camera(dt)
            
            # Update simulation state (including ultimate integration)
            self.update(dt)
            
            # Render everything
            self.render()
            
            # Update performance metrics
            self.update_performance_metrics(frame_start_time)
            
            # Update display
            pygame.display.flip()
            
    def handle_events(self):
        """Handle all input events with enhanced functionality"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.clean_shutdown()
                
            elif event.type == pygame.KEYDOWN:
                self.handle_keydown(event)
                
            elif event.type == pygame.KEYUP:
                self.handle_keyup(event)
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_down(event)
                
            elif event.type == pygame.MOUSEBUTTONUP:
                self.handle_mouse_up(event)
                
            elif event.type == pygame.MOUSEMOTION:
                self.handle_mouse_motion(event)
                
            elif event.type == pygame.VIDEORESIZE:
                self.handle_resize(event)
                
            # Pass event to GUI if available
            if self.gui:
                self.gui.handle_event(event)
                
            # Pass event to neural interface if active
            if self.integration_active and self.ultimate_integration.neural_interface:
                self.ultimate_integration.handle_user_input(event)
    
    def handle_keydown(self, event):
        """Handle keyboard key press with enhanced controls"""
        self.keys_pressed.add(event.key)
        
        # Quick simulation switching
        if event.key == pygame.K_1:
            self.switch_simulation("basic")
        elif event.key == pygame.K_2:
            self.switch_simulation("fountain") 
        elif event.key == pygame.K_3:
            self.switch_simulation("fire")
        elif event.key == pygame.K_4:
            self.switch_simulation("fluid")
        elif event.key == pygame.K_5 and self.integration_active:
            self.activate_ultimate_integration()
            
        # Simulation controls
        elif event.key == pygame.K_SPACE:
            if self.current_simulation:
                self.current_simulation.pause_toggle()
        elif event.key == pygame.K_r:
            if self.current_simulation:
                self.current_simulation.reset()
                
        # Enhanced camera controls
        elif event.key == pygame.K_c:
            self.cycle_camera_mode()
        elif event.key == pygame.K_p:
            self.take_screenshot()
            
        # Ultimate integration controls
        elif event.key == pygame.K_u and self.integration_active:
            self.toggle_integration_display()
        elif event.key == pygame.K_i and self.integration_active:
            self.boost_integration_level()
        elif event.key == pygame.K_o and self.integration_active:
            self.reduce_integration_level()
            
        # Performance overlay
        elif event.key == pygame.K_F1:
            self.performance_overlay = not self.performance_overlay
    
    def handle_keyup(self, event):
        """Handle keyboard key release"""
        if event.key in self.keys_pressed:
            self.keys_pressed.remove(event.key)
    
    def handle_mouse_down(self, event):
        """Handle mouse button press with enhanced interactions"""
        if event.button == 1:  # Left click
            self.mouse_dragging = True
            self.last_mouse_pos = event.pos
            
            # Enhanced particle addition
            if self.current_simulation and not self.is_mouse_in_gui_area(event.pos):
                if self.integration_active:
                    # Use neural interface for enhanced particle creation
                    self.create_neural_enhanced_particle(event.pos)
                else:
                    self.current_simulation.add_particle_at_screen_pos(event.pos)
                
        elif event.button == 3:  # Right click
            # Enhanced force field with integration
            if self.current_simulation:
                world_pos = self.screen_to_world(event.pos)
                field_type = "quantum" if self.integration_active else "radial"
                self.physics_module.add_force_field(world_pos, 10.0, 2.0, field_type)
    
    def handle_mouse_up(self, event):
        """Handle mouse button release"""
        if event.button == 1:
            self.mouse_dragging = False
    
    def handle_mouse_motion(self, event):
        """Handle mouse movement with enhanced camera control"""
        if self.mouse_dragging and event.buttons[0]:  # Left button dragged
            dx = event.pos[0] - self.last_mouse_pos[0]
            dy = event.pos[1] - self.last_mouse_pos[1]
            
            # Enhanced camera rotation based on mode
            if self.current_camera_mode == "quantum" and self.integration_active:
                self.quantum_camera_rotation(dx, dy)
            else:
                self.standard_camera_rotation(dx, dy)
                
            self.last_mouse_pos = event.pos
    
    def handle_resize(self, event):
        """Handle window resize"""
        glViewport(0, 0, event.w, event.h)
        self.config["window_width"] = event.w
        self.config["window_height"] = event.h
        
    def create_neural_enhanced_particle(self, screen_pos):
        """Create particles enhanced by neural interface"""
        if not self.integration_active or not self.ultimate_integration.neural_interface:
            self.current_simulation.add_particle_at_screen_pos(screen_pos)
            return
            
        # Get neural metrics to influence particle creation
        neural_metrics = self.ultimate_integration.neural_interface.get_neural_metrics()
        
        # Create particles based on neural state
        attention = neural_metrics.get('attention', 0.5)
        emotional_valence = neural_metrics.get('emotional_valence', 0)
        
        # More particles when attention is high
        particle_count = max(1, int(attention * 5))
        
        for i in range(particle_count):
            # Modify particle properties based on emotional state
            world_pos = self.screen_to_world(screen_pos)
            
            # Add some randomness based on neural state
            jitter = (1 - attention) * 0.2
            jitter_pos = world_pos + glm.vec3(
                np.random.uniform(-jitter, jitter),
                np.random.uniform(-jitter, jitter),
                0
            )
            
            self.current_simulation.add_particle_at_position(jitter_pos)
            
    def quantum_camera_rotation(self, dx, dy):
        """Quantum-inspired camera rotation"""
        # Use quantum state to influence camera movement
        if self.ultimate_integration.quantum_consciousness:
            quantum_metrics = self.ultimate_integration.quantum_consciousness.get_consciousness_metrics()
            coherence = quantum_metrics.get('interface_coherence', 0.5)
            
            # More stable camera movement with higher coherence
            stability_factor = 0.5 + coherence * 0.5
            dx *= stability_factor
            dy *= stability_factor
            
        self.rotate_camera(dx * 0.01, dy * 0.01)
        
    def standard_camera_rotation(self, dx, dy):
        """Standard camera rotation"""
        self.rotate_camera(dx * 0.01, dy * 0.01)
        
    def rotate_camera(self, dx, dy):
        """Rotate camera around target"""
        # Convert to spherical coordinates
        direction = self.camera_position - self.camera_target
        radius = glm.length(direction)
        
        # Calculate angles
        theta = np.arctan2(direction.x, direction.z) + dx
        phi = np.clip(np.arcsin(direction.y / radius) + dy, -1.5, 1.5)
        
        # Convert back to Cartesian
        new_x = radius * np.sin(phi) * np.sin(theta)
        new_y = radius * np.cos(phi)
        new_z = radius * np.sin(phi) * np.cos(theta)
        
        self.camera_position = self.camera_target + glm.vec3(new_x, new_y, new_z)
    
    def cycle_camera_mode(self):
        """Cycle through available camera modes"""
        current_index = self.camera_modes.index(self.current_camera_mode)
        next_index = (current_index + 1) % len(self.camera_modes)
        self.current_camera_mode = self.camera_modes[next_index]
        print(f"📷 Camera mode: {self.current_camera_mode}")
    
    def update_camera(self, dt):
        """Update camera position based on keyboard input and mode"""
        move_speed = self.camera_speed * dt
        
        # Camera movement with WASD
        if pygame.K_w in self.keys_pressed:
            self.camera_position.z -= move_speed
        if pygame.K_s in self.keys_pressed:
            self.camera_position.z += move_speed
        if pygame.K_a in self.keys_pressed:
            self.camera_position.x -= move_speed
        if pygame.K_d in self.keys_pressed:
            self.camera_position.x += move_speed
        if pygame.K_q in self.keys_pressed:
            self.camera_position.y -= move_speed
        if pygame.K_e in self.keys_pressed:
            self.camera_position.y += move_speed
            
        # Special camera behaviors based on mode
        if self.current_camera_mode == "orbit":
            self.orbit_camera_update(dt)
        elif self.current_camera_mode == "quantum" and self.integration_active:
            self.quantum_camera_update(dt)
    
    def orbit_camera_update(self, dt):
        """Update camera for orbit mode"""
        angle = dt * 0.5  # Slow rotation
        self.rotate_camera(angle, 0)
        
    def quantum_camera_update(self, dt):
        """Update camera for quantum mode"""
        if self.ultimate_integration.quantum_consciousness:
            quantum_metrics = self.ultimate_integration.quantum_consciousness.get_consciousness_metrics()
            neural_entropy = quantum_metrics.get('neural_entropy', 0.5)
            
            # Camera subtly responds to neural entropy
            jitter = neural_entropy * 0.01
            self.camera_position.x += np.random.uniform(-jitter, jitter)
            self.camera_position.y += np.random.uniform(-jitter, jitter)
    
    def update(self, dt):
        """Enhanced update with ultimate integration"""
        self.simulation_time += dt
        self.frame_count += 1
        
        # Update current simulation
        if self.current_simulation:
            for _ in range(self.config["physics_steps_per_frame"]):
                step_dt = dt / self.config["physics_steps_per_frame"]
                self.current_simulation.update(step_dt)
        
        # Update ultimate integration system
        if self.integration_active:
            self.ultimate_integration.update(dt)
            
            # Cross-influence between simulation and integration
            self.apply_integration_influences()
        
        # Update GUI
        if self.gui:
            self.gui.update(dt)
            
        # Update audio based on simulation state
        if self.audio_system and self.current_simulation:
            self.audio_system.update(self.current_simulation)
    
    def apply_integration_influences(self):
        """Apply influences from integration system to current simulation"""
        if not self.integration_active or not self.current_simulation:
            return
            
        # Get integration state
        integration_status = self.ultimate_integration.get_system_status()
        integration_level = integration_status['integration']['level']
        
        # Apply quantum consciousness influences
        if self.ultimate_integration.quantum_consciousness:
            quantum_metrics = self.ultimate_integration.quantum_consciousness.get_consciousness_metrics()
            
            # Influence particle behavior based on consciousness
            consciousness_level = quantum_metrics.get('avg_consciousness', 0.5)
            if hasattr(self.current_simulation, 'set_consciousness_influence'):
                self.current_simulation.set_consciousness_influence(consciousness_level)
                
        # Apply multiversal influences  
        if self.ultimate_integration.multiversal_explorer:
            multiverse_stats = self.ultimate_integration.multiversal_explorer.get_multiverse_statistics()
            
            # Influence physics based on multiverse stress
            reality_stress = multiverse_stats.get('reality_fabric_stress', 0.0)
            if hasattr(self.current_simulation, 'set_reality_stability'):
                stability = 1.0 - reality_stress
                self.current_simulation.set_reality_stability(stability)
    
    def render(self):
        """Enhanced render with ultimate integration"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Calculate matrices
        view_matrix = glm.lookAt(self.camera_position, self.camera_target, self.camera_up)
        projection_matrix = glm.perspective(
            glm.radians(45.0),
            self.config["window_width"] / self.config["window_height"],
            0.1, 100.0
        )
        
        # Render current simulation
        if self.current_simulation:
            self.current_simulation.render(view_matrix, projection_matrix)
        
        # Render ultimate integration systems
        if self.integration_active:
            self.ultimate_integration.render_all_systems(view_matrix, projection_matrix)
        
        # Render GUI on top
        if self.gui:
            self.gui.render()
            
        # Render performance overlay
        if self.performance_overlay:
            self.render_enhanced_performance_overlay()
    
    def render_enhanced_performance_overlay(self):
        """Render enhanced performance information"""
        # Switch to 2D orthographic projection for HUD
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.config["window_width"], self.config["window_height"], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth test for HUD
        glDisable(GL_DEPTH_TEST)
        
        # Switch back to Pygame 2D rendering for text
        pygame.display.flip()  # Ensure OpenGL rendering is complete
        
        # Create surface for text rendering
        overlay_surface = pygame.Surface((400, 200), pygame.SRCALPHA)
        overlay_surface.fill((0, 0, 0, 128))  # Semi-transparent background
        
        # Basic performance info
        font = pygame.font.Font(None, 24)
        fps_text = font.render(f"FPS: {self.fps:.1f}", True, (255, 255, 255))
        frame_time_text = font.render(f"Frame: {self.average_frame_time*1000:.1f}ms", True, (255, 255, 255))
        
        overlay_surface.blit(fps_text, (10, 10))
        overlay_surface.blit(frame_time_text, (10, 40))
        
        # Integration status if active
        if self.integration_active:
            integration_status = self.ultimate_integration.get_system_status()
            integration_text = font.render(
                f"Integration: {integration_status['integration']['level']:.1%}", 
                True, (100, 255, 100)
            )
            coherence_text = font.render(
                f"Coherence: {integration_status['integration']['coherence']:.1%}", 
                True, (100, 255, 100)
            )
            
            overlay_surface.blit(integration_text, (10, 70))
            overlay_surface.blit(coherence_text, (10, 100))
            
        # Camera mode
        camera_text = font.render(f"Camera: {self.current_camera_mode}", True, (255, 200, 100))
        overlay_surface.blit(camera_text, (10, 130))
        
        # Blit overlay to screen
        self.screen.blit(overlay_surface, (10, 10))
        
        # Restore OpenGL state
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def update_performance_metrics(self, frame_start_time):
        """Update performance tracking"""
        frame_time = time.time() - frame_start_time
        self.frame_times.append(frame_time)
        
        # Keep only last 60 frames
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
            
        # Update FPS every second
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            self.fps = len(self.frame_times) / (current_time - self.last_fps_update)
            self.last_fps_update = current_time
            self.average_frame_time = np.mean(self.frame_times)
    
    def switch_simulation(self, sim_type):
        """Switch to a different simulation type"""
        if sim_type in self.simulations:
            print(f"🔄 Switching to {sim_type} simulation")
            self.current_simulation = self.simulations[sim_type]
            if not self.current_simulation.initialized:
                self.current_simulation.initialize()
        else:
            print(f"❌ Simulation type '{sim_type}' not found")
    
    def activate_ultimate_integration(self):
        """Activate or deactivate ultimate integration"""
        if not self.ultimate_integration:
            print("❌ Ultimate Integration not available")
            return
            
        self.integration_active = not self.integration_active
        status = "ACTIVE" if self.integration_active else "INACTIVE"
        print(f"🔗 Ultimate Integration: {status}")
    
    def toggle_integration_display(self):
        """Toggle integration system display"""
        if self.integration_active:
            # This would toggle visibility of integration visualizations
            print("👁️ Toggling integration display")
    
    def boost_integration_level(self):
        """Boost integration level"""
        if self.integration_active:
            current_level = self.ultimate_integration.integration_level
            new_level = min(1.0, current_level + 0.1)
            self.ultimate_integration.integration_level = new_level
            print(f"📈 Integration level boosted to {new_level:.1%}")
    
    def reduce_integration_level(self):
        """Reduce integration level"""
        if self.integration_active:
            current_level = self.ultimate_integration.integration_level
            new_level = max(0.1, current_level - 0.1)
            self.ultimate_integration.integration_level = new_level
            print(f"📉 Integration level reduced to {new_level:.1%}")
    
    def take_screenshot(self):
        """Capture and save a screenshot"""
        if self.export_manager:
            self.export_manager.capture_screenshot()
        else:
            print("⚠️ Export system not available for screenshots")
    
    def is_mouse_in_gui_area(self, pos):
        """Check if mouse is in GUI area to prevent interaction conflicts"""
        if not self.gui:
            return False
        return pos[0] < 300 and pos[1] < 400
    
    def screen_to_world(self, screen_pos):
        """Convert screen coordinates to world coordinates"""
        x = (screen_pos[0] / self.config["window_width"] - 0.5) * 4.0
        y = -(screen_pos[1] / self.config["window_height"] - 0.5) * 4.0
        return glm.vec3(x, y, 0)
    
    def clean_shutdown(self):
        """Perform enhanced clean shutdown of all systems"""
        print("🔴 Shutting down frex Simulations...")
        
        # Stop all simulations
        if self.current_simulation:
            self.current_simulation.cleanup()
            
        # Cleanup ultimate integration
        if self.integration_active:
            # Integration system handles its own cleanup internally
            pass
            
        # Cleanup systems
        if self.audio_system:
            self.audio_system.cleanup()
            
        if self.export_manager:
            self.export_manager.cleanup()
            
        # Save configuration
        self.save_config()
        
        pygame.quit()
        sys.exit()
    
    def save_config(self):
        """Save current configuration"""
        try:
            with open('config.json', 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving config: {e}")

def main():
    """Enhanced main entry point"""
    print("🎯 ENHANCED VIDEO SIMULATION SOFTWARE")
    print("=====================================")
    
    try:
        app = EnhancedVideoSimulationSoftware()
        app.initialize()
        app.run()
    except KeyboardInterrupt:
        print("⏹️ Interrupted by user")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure clean shutdown even on error
        if 'app' in locals():
            app.clean_shutdown()

if __name__ == "__main__":
    main()
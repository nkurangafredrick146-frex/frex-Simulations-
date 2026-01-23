"""
Complete Real-time Editing GUI Module
Advanced graphical user interface for interactive simulation control and editing
"""

import pygame
import pygame.gfxdraw
import numpy as np
import glm
from OpenGL.GL import *
from OpenGL.GL.shaders import *
import math
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import json

class GUITheme:
    """GUI theme and styling configuration"""
    
    def __init__(self):
        # Color scheme
        self.background = (30, 30, 40, 200)
        self.primary = (70, 130, 180, 255)
        self.secondary = (100, 160, 200, 255)
        self.accent = (255, 165, 0, 255)
        self.text = (240, 240, 240, 255)
        self.text_secondary = (180, 180, 180, 255)
        self.border = (60, 60, 80, 255)
        self.highlight = (255, 200, 50, 255)
        self.warning = (220, 80, 60, 255)
        self.success = (80, 200, 120, 255)
        
        # Typography
        self.font_name = "Arial"
        self.font_sizes = {
            'small': 12,
            'medium': 16,
            'large': 20,
            'title': 24
        }
        
        # Spacing and layout
        self.padding = 8
        self.margin = 4
        self.border_radius = 6
        self.control_height = 28
        
        # Animation
        self.transition_speed = 0.2
        self.hover_animation = True

class GUIEventType(Enum):
    """Types of GUI events"""
    BUTTON_CLICK = "button_click"
    SLIDER_CHANGE = "slider_change"
    TOGGLE_CHANGE = "toggle_change"
    TEXT_INPUT = "text_input"
    DROPDOWN_SELECT = "dropdown_select"
    COLOR_PICK = "color_pick"
    FILE_SELECT = "file_select"

@dataclass
class GUIEvent:
    """GUI event data"""
    event_type: GUIEventType
    element_id: str
    value: Any
    timestamp: float

class GUIElement:
    """Base class for all GUI elements"""
    
    def __init__(self, element_id: str, position: Tuple[int, int], size: Tuple[int, int], 
                 theme: GUITheme, parent=None):
        self.element_id = element_id
        self.position = position
        self.size = size
        self.theme = theme
        self.parent = parent
        self.visible = True
        self.enabled = True
        self.hovered = False
        self.focused = False
        self.tooltip = ""
        
        # Animation state
        self.animation_state = 0.0
        self.target_animation = 0.0
        
    def update(self, dt: float, mouse_pos: Tuple[int, int], mouse_click: bool):
        """Update element state"""
        if not self.visible or not self.enabled:
            return None
            
        # Update hover state
        was_hovered = self.hovered
        self.hovered = self.is_point_inside(mouse_pos)
        
        # Handle hover animation
        if self.theme.hover_animation:
            target = 1.0 if self.hovered else 0.0
            self.animation_state += (target - self.animation_state) * self.theme.transition_speed
            
        # Check for click events
        event = None
        if self.hovered and mouse_click:
            event = self.handle_click()
            
        return event
        
    def is_point_inside(self, point: Tuple[int, int]) -> bool:
        """Check if point is inside element bounds"""
        x, y = point
        elem_x, elem_y = self.position
        width, height = self.size
        return (elem_x <= x <= elem_x + width and 
                elem_y <= y <= elem_y + height)
                
    def handle_click(self) -> Optional[GUIEvent]:
        """Handle mouse click - to be implemented by subclasses"""
        return None
        
    def render(self, surface: pygame.Surface):
        """Render element - to be implemented by subclasses"""
        pass
        
    def draw_rounded_rect(self, surface: pygame.Surface, rect: Tuple, color: Tuple, 
                         radius: int = None, border: int = 0, border_color: Tuple = None):
        """Draw a rounded rectangle"""
        if radius is None:
            radius = self.theme.border_radius
            
        x, y, width, height = rect
        
        if border > 0 and border_color:
            # Draw border
            self.draw_rounded_rect(surface, (x, y, width, height), border_color, radius)
            # Adjust inner rect
            x += border
            y += border
            width -= 2 * border
            height -= 2 * border
            
        # Draw rounded rectangle
        pygame.gfxdraw.aacircle(surface, x + radius, y + radius, radius, color)
        pygame.gfxdraw.aacircle(surface, x + width - radius - 1, y + radius, radius, color)
        pygame.gfxdraw.aacircle(surface, x + radius, y + height - radius - 1, radius, color)
        pygame.gfxdraw.aacircle(surface, x + width - radius - 1, y + height - radius - 1, radius, color)
        
        pygame.gfxdraw.filled_circle(surface, x + radius, y + radius, radius, color)
        pygame.gfxdraw.filled_circle(surface, x + width - radius - 1, y + radius, radius, color)
        pygame.gfxdraw.filled_circle(surface, x + radius, y + height - radius - 1, radius, color)
        pygame.gfxdraw.filled_circle(surface, x + width - radius - 1, y + height - radius - 1, radius, color)
        
        pygame.gfxdraw.box(surface, (x + radius, y, width - 2 * radius, height), color)
        pygame.gfxdraw.box(surface, (x, y + radius, width, height - 2 * radius), color)

class Button(GUIElement):
    """Interactive button element"""
    
    def __init__(self, element_id: str, position: Tuple[int, int], size: Tuple[int, int],
                 theme: GUITheme, text: str = "", callback: Callable = None, 
                 icon: str = None, toggle: bool = False):
        super().__init__(element_id, position, size, theme)
        self.text = text
        self.callback = callback
        self.icon = icon
        self.toggle = toggle
        self.toggled = False
        self.font = pygame.font.SysFont(theme.font_name, theme.font_sizes['medium'])
        
    def handle_click(self) -> Optional[GUIEvent]:
        """Handle button click"""
        if self.toggle:
            self.toggled = not self.toggled
            
        if self.callback:
            self.callback(self.element_id, self.toggled if self.toggle else True)
            
        return GUIEvent(
            event_type=GUIEventType.BUTTON_CLICK,
            element_id=self.element_id,
            value=self.toggled if self.toggle else True,
            timestamp=time.time()
        )
        
    def render(self, surface: pygame.Surface):
        """Render button"""
        if not self.visible:
            return
            
        x, y = self.position
        width, height = self.size
        
        # Determine button color based on state
        if not self.enabled:
            base_color = (self.theme.primary[0]//2, self.theme.primary[1]//2, 
                         self.theme.primary[2]//2, self.theme.primary[3])
        elif self.toggled:
            base_color = self.theme.accent
        else:
            base_color = self.theme.primary
            
        # Apply hover effect
        if self.hovered and self.enabled:
            color = tuple(min(c + 30, 255) for c in base_color[:3]) + (base_color[3],)
        else:
            color = base_color
            
        # Draw button background
        self.draw_rounded_rect(surface, (x, y, width, height), color, 
                              border=1, border_color=self.theme.border)
        
        # Draw text
        if self.text:
            text_surface = self.font.render(self.text, True, self.theme.text)
            text_rect = text_surface.get_rect(center=(x + width//2, y + height//2))
            surface.blit(text_surface, text_rect)
            
        # Draw icon if provided
        if self.icon:
            # Icon rendering would be implemented here
            pass

class Slider(GUIElement):
    """Value slider element"""
    
    def __init__(self, element_id: str, position: Tuple[int, int], size: Tuple[int, int],
                 theme: GUITheme, min_value: float, max_value: float, value: float,
                 callback: Callable = None, label: str = "", value_format: str = "{:.2f}"):
        super().__init__(element_id, position, size, theme)
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.callback = callback
        self.label = label
        self.value_format = value_format
        self.dragging = False
        self.font = pygame.font.SysFont(theme.font_name, theme.font_sizes['small'])
        
    def handle_click(self) -> Optional[GUIEvent]:
        """Start dragging slider"""
        self.dragging = True
        self.focused = True
        return self.update_value_from_mouse(pygame.mouse.get_pos())
        
    def update(self, dt: float, mouse_pos: Tuple[int, int], mouse_click: bool):
        """Update slider state"""
        event = super().update(dt, mouse_pos, mouse_click)
        
        # Handle drag continuation
        if self.dragging and pygame.mouse.get_pressed()[0]:
            drag_event = self.update_value_from_mouse(mouse_pos)
            if drag_event:
                event = drag_event
        else:
            self.dragging = False
            self.focused = False
            
        return event
        
    def update_value_from_mouse(self, mouse_pos: Tuple[int, int]) -> Optional[GUIEvent]:
        """Update slider value based on mouse position"""
        if not self.dragging:
            return None
            
        x, y = self.position
        width, height = self.size
        
        # Calculate new value based on mouse X position
        mouse_x = max(x, min(mouse_pos[0], x + width))
        normalized = (mouse_x - x) / width
        new_value = self.min_value + normalized * (self.max_value - self.min_value)
        
        # Apply value change
        if new_value != self.value:
            self.value = new_value
            if self.callback:
                self.callback(self.element_id, self.value)
                
            return GUIEvent(
                event_type=GUIEventType.SLIDER_CHANGE,
                element_id=self.element_id,
                value=self.value,
                timestamp=time.time()
            )
            
        return None
        
    def render(self, surface: pygame.Surface):
        """Render slider"""
        if not self.visible:
            return
            
        x, y = self.position
        width, height = self.size
        track_height = 4
        handle_size = 16
        
        # Calculate handle position
        normalized = (self.value - self.min_value) / (self.max_value - self.min_value)
        handle_x = x + normalized * width
        
        # Draw track
        track_rect = (x, y + height//2 - track_height//2, width, track_height)
        self.draw_rounded_rect(surface, track_rect, self.theme.border, radius=2)
        
        # Draw filled portion
        if normalized > 0:
            filled_rect = (x, y + height//2 - track_height//2, normalized * width, track_height)
            self.draw_rounded_rect(surface, filled_rect, self.theme.primary, radius=2)
            
        # Draw handle
        handle_color = self.theme.accent if self.dragging or self.hovered else self.theme.secondary
        handle_rect = (handle_x - handle_size//2, y + height//2 - handle_size//2, 
                      handle_size, handle_size)
        self.draw_rounded_rect(surface, handle_rect, handle_color)
        
        # Draw label and value
        if self.label:
            label_text = f"{self.label}: {self.value_format.format(self.value)}"
            label_surface = self.font.render(label_text, True, self.theme.text)
            surface.blit(label_surface, (x, y - 20))

class Toggle(GUIElement):
    """Toggle switch element"""
    
    def __init__(self, element_id: str, position: Tuple[int, int], size: Tuple[int, int],
                 theme: GUITheme, value: bool = False, callback: Callable = None,
                 label: str = ""):
        super().__init__(element_id, position, size, theme)
        self.value = value
        self.callback = callback
        self.label = label
        self.font = pygame.font.SysFont(theme.font_name, theme.font_sizes['small'])
        
    def handle_click(self) -> Optional[GUIEvent]:
        """Toggle value"""
        self.value = not self.value
        
        if self.callback:
            self.callback(self.element_id, self.value)
            
        return GUIEvent(
            event_type=GUIEventType.TOGGLE_CHANGE,
            element_id=self.element_id,
            value=self.value,
            timestamp=time.time()
        )
        
    def render(self, surface: pygame.Surface):
        """Render toggle"""
        if not self.visible:
            return
            
        x, y = self.position
        width, height = self.size
        
        # Toggle dimensions
        toggle_width = 40
        toggle_height = 20
        knob_size = 16
        padding = 2
        
        # Calculate toggle position (centered in element)
        toggle_x = x + (width - toggle_width) // 2
        toggle_y = y + (height - toggle_height) // 2
        
        # Background color based on state
        bg_color = self.theme.success if self.value else self.theme.border
        
        # Draw toggle background
        self.draw_rounded_rect(surface, (toggle_x, toggle_y, toggle_width, toggle_height), 
                              bg_color, radius=toggle_height//2)
        
        # Calculate knob position
        if self.value:
            knob_x = toggle_x + toggle_width - knob_size - padding
        else:
            knob_x = toggle_x + padding
            
        knob_y = toggle_y + (toggle_height - knob_size) // 2
        
        # Draw knob
        knob_color = self.theme.text
        self.draw_rounded_rect(surface, (knob_x, knob_y, knob_size, knob_size), 
                              knob_color, radius=knob_size//2)
        
        # Draw label
        if self.label:
            label_surface = self.font.render(self.label, True, self.theme.text)
            surface.blit(label_surface, (x, y - 20))

class Dropdown(GUIElement):
    """Dropdown selection element"""
    
    def __init__(self, element_id: str, position: Tuple[int, int], size: Tuple[int, int],
                 theme: GUITheme, options: List[str], selected_index: int = 0,
                 callback: Callable = None, label: str = ""):
        super().__init__(element_id, position, size, theme)
        self.options = options
        self.selected_index = selected_index
        self.callback = callback
        self.label = label
        self.expanded = False
        self.font = pygame.font.SysFont(theme.font_name, theme.font_sizes['small'])
        
    def handle_click(self) -> Optional[GUIEvent]:
        """Handle dropdown click"""
        if not self.expanded:
            self.expanded = True
            return None
        else:
            # Handle option selection
            mouse_pos = pygame.mouse.get_pos()
            x, y = self.position
            width, height = self.size
            
            # Calculate which option was clicked
            option_height = height
            relative_y = mouse_pos[1] - y - height  # Below the main element
            option_index = int(relative_y // option_height)
            
            if 0 <= option_index < len(self.options):
                self.selected_index = option_index
                self.expanded = False
                
                if self.callback:
                    self.callback(self.element_id, self.selected_index)
                    
                return GUIEvent(
                    event_type=GUIEventType.DROPDOWN_SELECT,
                    element_id=self.element_id,
                    value=self.selected_index,
                    timestamp=time.time()
                )
                
            self.expanded = False
            return None
            
    def render(self, surface: pygame.Surface):
        """Render dropdown"""
        if not self.visible:
            return
            
        x, y = self.position
        width, height = self.size
        
        # Main dropdown background
        main_color = self.theme.primary if self.hovered else self.theme.secondary
        self.draw_rounded_rect(surface, (x, y, width, height), main_color, 
                              border=1, border_color=self.theme.border)
        
        # Selected option text
        if self.selected_index < len(self.options):
            selected_text = self.options[self.selected_index]
            text_surface = self.font.render(selected_text, True, self.theme.text)
            text_rect = text_surface.get_rect(midleft=(x + 10, y + height//2))
            surface.blit(text_surface, text_rect)
            
        # Dropdown arrow
        arrow_size = 6
        arrow_x = x + width - 15
        arrow_y = y + height//2
        
        if self.expanded:
            # Up arrow
            pygame.draw.polygon(surface, self.theme.text, [
                (arrow_x, arrow_y - arrow_size//2),
                (arrow_x - arrow_size, arrow_y + arrow_size//2),
                (arrow_x + arrow_size, arrow_y + arrow_size//2)
            ])
        else:
            # Down arrow
            pygame.draw.polygon(surface, self.theme.text, [
                (arrow_x, arrow_y + arrow_size//2),
                (arrow_x - arrow_size, arrow_y - arrow_size//2),
                (arrow_x + arrow_size, arrow_y - arrow_size//2)
            ])
            
        # Expanded options
        if self.expanded:
            option_height = height
            for i, option in enumerate(self.options):
                option_y = y + height + i * option_height
                option_color = self.theme.highlight if i == self.selected_index else self.theme.secondary
                
                self.draw_rounded_rect(surface, (x, option_y, width, option_height), 
                                      option_color, border=1, border_color=self.theme.border)
                
                option_surface = self.font.render(option, True, self.theme.text)
                option_rect = option_surface.get_rect(midleft=(x + 10, option_y + option_height//2))
                surface.blit(option_surface, option_rect)
                
        # Label
        if self.label:
            label_surface = self.font.render(self.label, True, self.theme.text)
            surface.blit(label_surface, (x, y - 20))

class Panel(GUIElement):
    """Container panel for organizing GUI elements"""
    
    def __init__(self, element_id: str, position: Tuple[int, int], size: Tuple[int, int],
                 theme: GUITheme, title: str = "", collapsible: bool = True):
        super().__init__(element_id, position, size, theme)
        self.title = title
        self.collapsible = collapsible
        self.collapsed = False
        self.elements = []
        self.title_font = pygame.font.SysFont(theme.font_name, theme.font_sizes['medium'])
        self.header_height = 30
        
    def add_element(self, element: GUIElement):
        """Add element to panel"""
        element.parent = self
        self.elements.append(element)
        
    def update(self, dt: float, mouse_pos: Tuple[int, int], mouse_click: bool):
        """Update panel and all elements"""
        if not self.visible:
            return None
            
        events = []
        
        # Check for panel header click (for collapsing)
        header_rect = (self.position[0], self.position[1], self.size[0], self.header_height)
        header_hovered = (header_rect[0] <= mouse_pos[0] <= header_rect[0] + header_rect[2] and
                         header_rect[1] <= mouse_pos[1] <= header_rect[1] + header_rect[3])
                         
        if header_hovered and mouse_click and self.collapsible:
            self.collapsed = not self.collapsed
            return GUIEvent(
                event_type=GUIEventType.BUTTON_CLICK,
                element_id=f"{self.element_id}_toggle",
                value=self.collapsed,
                timestamp=time.time()
            )
            
        # Update elements if not collapsed
        if not self.collapsed:
            for element in self.elements:
                element_event = element.update(dt, mouse_pos, mouse_click)
                if element_event:
                    events.append(element_event)
                    
        return events if events else None
        
    def render(self, surface: pygame.Surface):
        """Render panel and all elements"""
        if not self.visible:
            return
            
        x, y = self.position
        width, height = self.size
        
        # Panel background
        self.draw_rounded_rect(surface, (x, y, width, height), self.theme.background, 
                              border=1, border_color=self.theme.border)
        
        # Panel header
        header_color = tuple(min(c + 20, 255) for c in self.theme.background[:3]) + (255,)
        self.draw_rounded_rect(surface, (x, y, width, self.header_height), header_color, 
                              radius=self.theme.border_radius, border=1, border_color=self.theme.border)
        
        # Title
        if self.title:
            title_surface = self.title_font.render(self.title, True, self.theme.text)
            title_rect = title_surface.get_rect(midleft=(x + 10, y + self.header_height//2))
            surface.blit(title_surface, title_rect)
            
        # Collapse indicator
        if self.collapsible:
            indicator_size = 8
            indicator_x = x + width - 20
            indicator_y = y + self.header_height//2
            
            if self.collapsed:
                # Right-facing triangle
                pygame.draw.polygon(surface, self.theme.text, [
                    (indicator_x, indicator_y - indicator_size//2),
                    (indicator_x + indicator_size, indicator_y),
                    (indicator_x, indicator_y + indicator_size//2)
                ])
            else:
                # Down-facing triangle
                pygame.draw.polygon(surface, self.theme.text, [
                    (indicator_x - indicator_size//2, indicator_y),
                    (indicator_x + indicator_size//2, indicator_y),
                    (indicator_x, indicator_y + indicator_size)
                ])
                
        # Render elements if not collapsed
        if not self.collapsed:
            for element in self.elements:
                element.render(surface)

class RealTimeGUI:
    """Main real-time GUI system for simulation control"""
    
    def __init__(self, application):
        self.application = application
        self.theme = GUITheme()
        self.panels = {}
        self.elements = {}
        self.visible = True
        self.fonts = {}
        self.event_handlers = {}
        
        # Performance tracking
        self.frame_time = 0.0
        self.render_time = 0.0
        
        # Initialize fonts
        self.initialize_fonts()
        
    def initialize_fonts(self):
        """Initialize pygame fonts"""
        self.fonts = {
            'small': pygame.font.SysFont(self.theme.font_name, self.theme.font_sizes['small']),
            'medium': pygame.font.SysFont(self.theme.font_name, self.theme.font_sizes['medium']),
            'large': pygame.font.SysFont(self.theme.font_name, self.theme.font_sizes['large']),
            'title': pygame.font.SysFont(self.theme.font_name, self.theme.font_sizes['title'])
        }
        
    def initialize(self):
        """Initialize GUI panels and elements"""
        print("Initializing Real-time GUI...")
        
        # Create main control panels
        self.create_simulation_controls()
        self.create_physics_controls()
        self.create_particle_controls()
        self.create_visualization_controls()
        self.create_performance_panel()
        
        print("GUI initialized successfully")
        
    def create_simulation_controls(self):
        """Create simulation control panel"""
        panel = Panel("simulation_controls", (10, 10), (300, 200), self.theme, 
                     "Simulation Controls", collapsible=True)
        
        # Play/Pause button
        play_button = Button("play_pause", (20, 50), (80, 30), self.theme,
                           "Pause", self.on_play_pause)
        panel.add_element(play_button)
        
        # Reset button
        reset_button = Button("reset", (110, 50), (80, 30), self.theme,
                            "Reset", self.on_reset)
        panel.add_element(reset_button)
        
        # Step button
        step_button = Button("step", (200, 50), (80, 30), self.theme,
                           "Step", self.on_step)
        panel.add_element(step_button)
        
        # Time scale slider
        time_slider = Slider("time_scale", (20, 100), (260, 40), self.theme,
                           0.1, 5.0, 1.0, self.on_time_scale_change, "Time Scale")
        panel.add_element(time_slider)
        
        # Simulation type dropdown
        sim_types = ["Basic Particles", "Fountain", "Fire", "Fluid", "Quantum"]
        sim_dropdown = Dropdown("simulation_type", (20, 150), (260, 30), self.theme,
                               sim_types, 0, self.on_simulation_change, "Simulation Type")
        panel.add_element(sim_dropdown)
        
        self.panels["simulation_controls"] = panel
        
    def create_physics_controls(self):
        """Create physics control panel"""
        panel = Panel("physics_controls", (10, 220), (300, 250), self.theme,
                     "Physics Controls", collapsible=True)
        
        # Gravity sliders
        gravity_x = Slider("gravity_x", (20, 50), (260, 30), self.theme,
                          -20.0, 20.0, 0.0, self.on_gravity_change, "Gravity X")
        panel.add_element(gravity_x)
        
        gravity_y = Slider("gravity_y", (20, 90), (260, 30), self.theme,
                          -20.0, 20.0, -9.81, self.on_gravity_change, "Gravity Y")
        panel.add_element(gravity_y)
        
        gravity_z = Slider("gravity_z", (20, 130), (260, 30), self.theme,
                          -20.0, 20.0, 0.0, self.on_gravity_change, "Gravity Z")
        panel.add_element(gravity_z)
        
        # Physics toggles
        collisions_toggle = Toggle("enable_collisions", (20, 170), (120, 30), self.theme,
                                  True, self.on_physics_toggle, "Collisions")
        panel.add_element(collisions_toggle)
        
        fluid_toggle = Toggle("enable_fluid", (150, 170), (120, 30), self.theme,
                            False, self.on_physics_toggle, "Fluid Dynamics")
        panel.add_element(fluid_toggle)
        
        # Viscosity slider
        viscosity_slider = Slider("viscosity", (20, 210), (260, 30), self.theme,
                                 0.0, 1.0, 0.1, self.on_physics_param_change, "Viscosity")
        panel.add_element(viscosity_slider)
        
        self.panels["physics_controls"] = panel
        
    def create_particle_controls(self):
        """Create particle system control panel"""
        panel = Panel("particle_controls", (10, 480), (300, 200), self.theme,
                     "Particle Controls", collapsible=True)
        
        # Emission rate slider
        emission_slider = Slider("emission_rate", (20, 50), (260, 30), self.theme,
                                0, 200, 50, self.on_particle_param_change, "Emission Rate")
        panel.add_element(emission_slider)
        
        # Particle lifetime slider
        lifetime_slider = Slider("particle_lifetime", (20, 90), (260, 30), self.theme,
                                0.1, 10.0, 3.0, self.on_particle_param_change, "Lifetime")
        panel.add_element(lifetime_slider)
        
        # Particle size slider
        size_slider = Slider("particle_size", (20, 130), (260, 30), self.theme,
                            0.01, 0.5, 0.1, self.on_particle_param_change, "Size")
        panel.add_element(size_slider)
        
        # Add particle button
        add_particle_btn = Button("add_particle", (20, 170), (260, 30), self.theme,
                                "Add Particle Burst", self.on_add_particles)
        panel.add_element(add_particle_btn)
        
        self.panels["particle_controls"] = panel
        
    def create_visualization_controls(self):
        """Create visualization control panel"""
        panel = Panel("visualization_controls", (320, 10), (300, 200), self.theme,
                     "Visualization", collapsible=True)
        
        # Render mode dropdown
        render_modes = ["Solid", "Wireframe", "Points", "Advanced", "Basic"]
        render_dropdown = Dropdown("render_mode", (20, 50), (260, 30), self.theme,
                                  render_modes, 3, self.on_render_mode_change, "Render Mode")
        panel.add_element(render_dropdown)
        
        # Camera controls
        camera_dropdown = Dropdown("camera_mode", (20, 90), (260, 30), self.theme,
                                  ["Free", "Orbit", "Fixed", "Follow"], 0, 
                                  self.on_camera_mode_change, "Camera Mode")
        panel.add_element(camera_dropdown)
        
        # Visualization toggles
        lighting_toggle = Toggle("enable_lighting", (20, 130), (120, 30), self.theme,
                                True, self.on_visualization_toggle, "Lighting")
        panel.add_element(lighting_toggle)
        
        shadows_toggle = Toggle("enable_shadows", (150, 130), (120, 30), self.theme,
                               False, self.on_visualization_toggle, "Shadows")
        panel.add_element(shadows_toggle)
        
        # Post-processing toggle
        postprocess_toggle = Toggle("enable_postprocess", (20, 170), (260, 30), self.theme,
                                   True, self.on_visualization_toggle, "Post-Processing")
        panel.add_element(postprocess_toggle)
        
        self.panels["visualization_controls"] = panel
        
    def create_performance_panel(self):
        """Create performance monitoring panel"""
        panel = Panel("performance_panel", (320, 220), (300, 150), self.theme,
                     "Performance", collapsible=False)
        
        # This panel will be updated dynamically with performance stats
        self.panels["performance_panel"] = panel
        
    def handle_event(self, event: pygame.event.Event):
        """Handle pygame events"""
        if not self.visible:
            return
            
        # Handle keyboard shortcuts
        if event.type == pygame.KEYDOWN:
            self.handle_keyboard_shortcuts(event)
            
    def handle_keyboard_shortcuts(self, event: pygame.event.Event):
        """Handle keyboard shortcuts for quick controls"""
        if event.key == pygame.K_SPACE:
            self.on_play_pause("play_pause", None)
        elif event.key == pygame.K_r:
            self.on_reset("reset", None)
        elif event.key == pygame.K_t:
            # Toggle GUI visibility
            self.visible = not self.visible
        elif event.key == pygame.K_1:
            self.on_simulation_change("simulation_type", 0)
        elif event.key == pygame.K_2:
            self.on_simulation_change("simulation_type", 1)
        elif event.key == pygame.K_3:
            self.on_simulation_change("simulation_type", 2)
            
    def update(self, dt: float):
        """Update GUI state"""
        if not self.visible:
            return
            
        start_time = time.time()
        mouse_pos = pygame.mouse.get_pos()
        mouse_click = pygame.mouse.get_pressed()[0]
        
        # Update all panels
        for panel in self.panels.values():
            events = panel.update(dt, mouse_pos, mouse_click)
            if events:
                self.handle_gui_events(events)
                
        # Update performance panel dynamically
        self.update_performance_panel()
        
        self.frame_time = time.time() - start_time
        
    def update_performance_panel(self):
        """Update performance panel with current stats"""
        panel = self.panels["performance_panel"]
        
        # Clear existing elements (keep only static ones if any)
        panel.elements.clear()
        
        # Get performance stats from application
        if hasattr(self.application, 'get_performance_stats'):
            stats = self.application.get_performance_stats()
        else:
            stats = {}
            
        y_offset = 40
        line_height = 20
        
        # Add performance metrics as labels
        metrics = [
            ("FPS", f"{stats.get('fps', 0):.1f}"),
            ("Particles", f"{stats.get('particle_count', 0)}"),
            ("Frame Time", f"{stats.get('frame_time', 0)*1000:.1f}ms"),
            ("GUI Time", f"{self.frame_time*1000:.1f}ms"),
            ("Simulation Time", f"{stats.get('simulation_time', 0):.1f}s")
        ]
        
        for label, value in metrics:
            # Create a simple text element (we'd need a Label class for proper rendering)
            # For now, we'll just store the data and render directly in render_performance_panel
            pass
            
    def render_performance_panel(self, surface: pygame.Surface):
        """Render performance panel with dynamic content"""
        panel = self.panels["performance_panel"]
        if not panel.visible:
            return
            
        # Render panel background first
        panel.render(surface)
        
        # Get performance stats
        if hasattr(self.application, 'get_performance_stats'):
            stats = self.application.get_performance_stats()
        else:
            stats = {}
            
        x, y = panel.position
        y_offset = 40
        line_height = 20
        
        # Render performance metrics
        metrics = [
            ("FPS", f"{stats.get('fps', 0):.1f}"),
            ("Particles", f"{stats.get('particle_count', 0)}"),
            ("Frame Time", f"{stats.get('frame_time', 0)*1000:.1f}ms"),
            ("GUI Time", f"{self.frame_time*1000:.1f}ms"),
            ("Simulation Time", f"{stats.get('simulation_time', 0):.1f}s")
        ]
        
        font = self.fonts['small']
        for i, (label, value) in enumerate(metrics):
            text = f"{label}: {value}"
            text_surface = font.render(text, True, self.theme.text)
            surface.blit(text_surface, (x + 10, y + y_offset + i * line_height))
        
    def handle_gui_events(self, events: List[GUIEvent]):
        """Handle GUI events and call appropriate callbacks"""
        for event in events:
            # Global event handlers
            if event.event_type in self.event_handlers:
                for handler in self.event_handlers[event.event_type]:
                    handler(event)
                    
    def register_event_handler(self, event_type: GUIEventType, handler: Callable):
        """Register global event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        
    # Callback methods for GUI elements
    def on_play_pause(self, element_id: str, value: Any):
        """Handle play/pause button click"""
        if self.application and hasattr(self.application, 'current_simulation'):
            self.application.current_simulation.pause_toggle()
            
    def on_reset(self, element_id: str, value: Any):
        """Handle reset button click"""
        if self.application and hasattr(self.application, 'current_simulation'):
            self.application.current_simulation.reset()
            
    def on_step(self, element_id: str, value: Any):
        """Handle step button click"""
        # Step simulation one frame
        pass
        
    def on_time_scale_change(self, element_id: str, value: float):
        """Handle time scale change"""
        if self.application and hasattr(self.application, 'physics_module'):
            self.application.physics_module.settings.time_scale = value
            
    def on_simulation_change(self, element_id: str, value: int):
        """Handle simulation type change"""
        if self.application:
            sim_types = ["basic", "fountain", "fire", "fluid", "quantum"]
            if value < len(sim_types):
                self.application.switch_simulation(sim_types[value])
                
    def on_gravity_change(self, element_id: str, value: float):
        """Handle gravity change"""
        if self.application and hasattr(self.application, 'physics_module'):
            # Update specific gravity component based on which slider changed
            if element_id == "gravity_x":
                self.application.physics_module.settings.gravity.x = value
            elif element_id == "gravity_y":
                self.application.physics_module.settings.gravity.y = value
            elif element_id == "gravity_z":
                self.application.physics_module.settings.gravity.z = value
                
    def on_physics_toggle(self, element_id: str, value: bool):
        """Handle physics toggle changes"""
        if self.application and hasattr(self.application, 'physics_module'):
            if element_id == "enable_collisions":
                self.application.physics_module.settings.collision_enabled = value
            elif element_id == "enable_fluid":
                self.application.physics_module.settings.fluid_dynamics_enabled = value
                
    def on_physics_param_change(self, element_id: str, value: float):
        """Handle physics parameter changes"""
        if self.application and hasattr(self.application, 'physics_module'):
            if element_id == "viscosity":
                self.application.physics_module.settings.viscosity = value
                
    def on_particle_param_change(self, element_id: str, value: float):
        """Handle particle parameter changes"""
        # This would update particle system parameters
        pass
        
    def on_add_particles(self, element_id: str, value: Any):
        """Handle add particle button"""
        if self.application and hasattr(self.application, 'current_simulation'):
            # Add particles at random positions
            for _ in range(50):
                pos = glm.vec3(
                    random.uniform(-2, 2),
                    random.uniform(-2, 2),
                    random.uniform(-2, 2)
                )
                self.application.current_simulation.add_particle_at_screen_pos(
                    (random.randint(0, 1200), random.randint(0, 800))
                )
                
    def on_render_mode_change(self, element_id: str, value: int):
        """Handle render mode change"""
        if self.application and hasattr(self.application, 'rendering_engine'):
            modes = ["solid", "wireframe", "points", "advanced", "basic"]
            if value < len(modes):
                self.application.rendering_engine.render_mode = modes[value]
                
    def on_camera_mode_change(self, element_id: str, value: int):
        """Handle camera mode change"""
        # This would change camera behavior
        pass
        
    def on_visualization_toggle(self, element_id: str, value: bool):
        """Handle visualization toggle changes"""
        if self.application and hasattr(self.application, 'rendering_engine'):
            if element_id == "enable_lighting":
                self.application.rendering_engine.enable_lighting = value
            elif element_id == "enable_shadows":
                self.application.rendering_engine.enable_shadows = value
            elif element_id == "enable_postprocess":
                self.application.rendering_engine.post_processing.enabled = value
                
    def render(self):
        """Render the complete GUI"""
        if not self.visible:
            return
            
        start_time = time.time()
        
        # Create a surface for GUI rendering
        gui_surface = pygame.Surface((self.application.config["window_width"], 
                                    self.application.config["window_height"]), 
                                    pygame.SRCALPHA)
        
        # Render all panels
        for panel in self.panels.values():
            if panel != self.panels["performance_panel"]:  # Performance panel handled separately
                panel.render(gui_surface)
                
        # Render performance panel separately for dynamic content
        self.render_performance_panel(gui_surface)
        
        # Convert to OpenGL texture and render
        self.render_gui_to_screen(gui_surface)
        
        self.render_time = time.time() - start_time
        
    def render_gui_to_screen(self, gui_surface: pygame.Surface):
        """Render GUI surface to OpenGL screen"""
        # Convert pygame surface to OpenGL texture
        gui_string = pygame.image.tostring(gui_surface, "RGBA")
        width, height = gui_surface.get_size()
        
        # Setup orthographic projection for 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth test for GUI
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Create texture
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, gui_string)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Render fullscreen quad with GUI texture
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(width, 0)
        glTexCoord2f(1, 1); glVertex2f(width, height)
        glTexCoord2f(0, 1); glVertex2f(0, height)
        glEnd()
        
        # Cleanup
        glDeleteTextures(1, [texture])
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        
        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
    def get_performance_stats(self) -> Dict:
        """Get GUI performance statistics"""
        return {
            'gui_frame_time': self.frame_time,
            'gui_render_time': self.render_time,
            'visible_panels': len([p for p in self.panels.values() if p.visible]),
            'total_elements': sum(len(p.elements) for p in self.panels.values())
        }

# Demo and testing
if __name__ == "__main__":
    # Initialize pygame for testing
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("GUI Test")
    
    # Create test GUI
    class TestApp:
        def __init__(self):
            self.config = {"window_width": 800, "window_height": 600}
            
    app = TestApp()
    gui = RealTimeGUI(app)
    gui.initialize()
    
    # Test rendering
    running = True
    clock = pygame.time.Clock()
    
    print("Testing Real-time GUI...")
    
    while running:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            gui.handle_event(event)
                
        # Update GUI
        gui.update(dt)
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Render GUI
        gui.render()
        
        pygame.display.flip()
        
    pygame.quit()
    print("GUI test completed successfully!")
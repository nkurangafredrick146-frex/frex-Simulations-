#!/usr/bin/env python3
"""
Biological & Cellular Automata Simulations
Simulation of biological processes, cellular automata, and artificial life systems
"""

import numpy as np
import glm
import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import *
from typing import Dict, List, Optional, Tuple, Any, Callable
import math
import random
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import json
import time

@dataclass
class CellState:
    type: int  # 0=dead, 1=alive, 2=stem, 3=cancer, etc.
    age: int
    energy: float
    metabolism: float
    division_count: int
    genetic_markers: np.ndarray
    position: Tuple[int, int, int]
    neighbors: List[Tuple[int, int, int]]

@dataclass
class BiologicalParameters:
    birth_energy: float = 100.0
    division_threshold: float = 150.0
    death_threshold: float = 10.0
    metabolism_rate: float = 0.1
    mutation_rate: float = 0.01
    diffusion_rate: float = 0.1
    toxin_decay: float = 0.05
    nutrient_production: float = 0.1

class CellularAutomata3D:
    """3D Cellular Automata for biological simulations"""
    
    def __init__(self, width: int = 64, height: int = 64, depth: int = 64):
        self.width = width
        self.height = height
        self.depth = depth
        self.params = BiologicalParameters()
        
        # 3D grid for cell states
        self.cell_grid = np.zeros((depth, height, width), dtype=object)
        self.nutrient_grid = np.ones((depth, height, width), dtype=np.float32)
        self.toxin_grid = np.zeros((depth, height, width), dtype=np.float32)
        self.signal_grid = np.zeros((depth, height, width), dtype=np.float32)
        
        # Cell types and their properties
        self.cell_types = {
            0: {"name": "dead", "color": (0.1, 0.1, 0.1), "metabolism": 0.0},
            1: {"name": "living", "color": (0.2, 0.8, 0.2), "metabolism": 0.1},
            2: {"name": "stem", "color": (0.8, 0.2, 0.8), "metabolism": 0.05},
            3: {"name": "cancer", "color": (0.8, 0.2, 0.2), "metabolism": 0.2},
            4: {"name": "immune", "color": (0.2, 0.2, 0.8), "metabolism": 0.15},
            5: {"name": "neural", "color": (0.8, 0.8, 0.2), "metabolism": 0.3}
        }
        
        # Rule sets for different biological processes
        self.rules = {
            "conway": self.conway_rules,
            "bio_metabolism": self.biological_metabolism_rules,
            "cancer_growth": self.cancer_growth_rules,
            "neural_development": self.neural_development_rules
        }
        
        # Rendering
        self.cell_vao = None
        self.cell_shader = None
        self.initialized_rendering = False
        
        # Statistics
        self.generation = 0
        self.cell_count = 0
        self.average_energy = 0.0
        self.diversity_index = 0.0
        
        # Initialize grid
        self.initialize_grid()
        
        print(f"3D Cellular Automata initialized: {width}x{height}x{depth}")
    
    def initialize_grid(self):
        """Initialize the cellular automata grid with random cells"""
        # Clear grid
        self.cell_grid.fill(None)
        
        # Create initial cell colonies
        num_colonies = 5
        for _ in range(num_colonies):
            center_x = random.randint(10, self.width - 10)
            center_y = random.randint(10, self.height - 10)
            center_z = random.randint(10, self.depth - 10)
            
            colony_radius = random.randint(3, 8)
            cell_type = random.choice([1, 2, 4])  # Living, stem, or immune
            
            self.create_colony(center_x, center_y, center_z, colony_radius, cell_type)
        
        # Initialize nutrient gradient (higher in center)
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    # Distance from center
                    dx = (x - self.width / 2) / self.width
                    dy = (y - self.height / 2) / self.height
                    dz = (z - self.depth / 2) / self.depth
                    distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    self.nutrient_grid[z, y, x] = max(0.1, 1.0 - distance)
        
        self.update_statistics()
    
    def create_colony(self, x: int, y: int, z: int, radius: int, cell_type: int):
        """Create a spherical colony of cells"""
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx*dx + dy*dy + dz*dz <= radius*radius:
                        pos_x, pos_y, pos_z = x + dx, y + dy, z + dz
                        if self.is_valid_position(pos_x, pos_y, pos_z):
                            if random.random() < 0.7:  # 70% density
                                self.create_cell(pos_x, pos_y, pos_z, cell_type)
    
    def create_cell(self, x: int, y: int, z: int, cell_type: int):
        """Create a new cell at the specified position"""
        neighbors = self.get_neighbors(x, y, z)
        
        cell = CellState(
            type=cell_type,
            age=0,
            energy=self.params.birth_energy,
            metabolism=self.cell_types[cell_type]["metabolism"],
            division_count=0,
            genetic_markers=np.random.rand(8),  # Random genetic profile
            position=(x, y, z),
            neighbors=neighbors
        )
        
        self.cell_grid[z, y, x] = cell
        self.cell_count += 1
    
    def is_valid_position(self, x: int, y: int, z: int) -> bool:
        """Check if position is within grid bounds"""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                0 <= z < self.depth)
    
    def get_neighbors(self, x: int, y: int, z: int, radius: int = 1) -> List[Tuple[int, int, int]]:
        """Get Moore neighborhood coordinates (3D)"""
        neighbors = []
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue  # Skip self
                    
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if self.is_valid_position(nx, ny, nz):
                        neighbors.append((nx, ny, nz))
        return neighbors
    
    def update(self, rule_set: str = "bio_metabolism"):
        """Update the cellular automata using specified rule set"""
        if rule_set not in self.rules:
            print(f"Unknown rule set: {rule_set}")
            return
        
        # Apply the selected rule set
        self.rules[rule_set]()
        
        # Update nutrient and toxin diffusion
        self.diffuse_nutrients()
        self.diffuse_toxins()
        
        # Update statistics
        self.generation += 1
        self.update_statistics()
    
    def conway_rules(self):
        """Conway's Game of Life rules extended to 3D"""
        new_grid = np.copy(self.cell_grid)
        
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    cell = self.cell_grid[z, y, x]
                    neighbors = self.get_neighbors(x, y, z)
                    
                    # Count living neighbors
                    living_neighbors = 0
                    for nx, ny, nz in neighbors:
                        neighbor = self.cell_grid[nz, ny, nx]
                        if neighbor and neighbor.type in [1, 2, 3, 4, 5]:  # Any living cell type
                            living_neighbors += 1
                    
                    # Apply Conway's rules
                    if cell and cell.type in [1, 2, 3, 4, 5]:  # Living cell
                        if living_neighbors < 4 or living_neighbors > 9:  # Adjusted for 3D
                            new_grid[z, y, x] = None  # Death
                    else:  # Dead cell
                        if living_neighbors == 6:  # Birth (adjusted for 3D)
                            new_cell_type = 1  # Basic living cell
                            self.create_cell(x, y, z, new_cell_type)
        
        self.cell_grid = new_grid
    
    def biological_metabolism_rules(self):
        """Biological rules with metabolism, energy, and cell division"""
        new_grid = np.copy(self.cell_grid)
        cells_to_add = []
        
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    cell = self.cell_grid[z, y, x]
                    
                    if cell is None:
                        continue
                    
                    # Skip dead cells
                    if cell.type == 0:
                        continue
                    
                    # Age the cell
                    cell.age += 1
                    
                    # Metabolism: consume energy and nutrients
                    energy_consumption = cell.metabolism
                    nutrient_available = self.nutrient_grid[z, y, x]
                    
                    if nutrient_available > 0:
                        # Convert nutrients to energy
                        energy_gain = min(nutrient_available, cell.metabolism * 2)
                        cell.energy += energy_gain
                        self.nutrient_grid[z, y, x] -= energy_gain * 0.5
                    else:
                        # Starvation
                        cell.energy -= energy_consumption * 2
                    
                    # Produce toxins as waste
                    toxin_production = energy_consumption * 0.1
                    self.toxin_grid[z, y, x] += toxin_production
                    
                    # Toxin damage
                    toxin_level = self.toxin_grid[z, y, x]
                    if toxin_level > 50.0:
                        cell.energy -= toxin_level * 0.01
                    
                    # Check for death
                    if cell.energy <= self.params.death_threshold:
                        new_grid[z, y, x] = None
                        self.cell_count -= 1
                        # Release nutrients upon death
                        self.nutrient_grid[z, y, x] += cell.energy * 0.5
                        continue
                    
                    # Cell division
                    if (cell.energy >= self.params.division_threshold and 
                        cell.division_count < 10):  # Limit divisions
                        
                        # Find empty neighbor for division
                        empty_neighbors = []
                        for nx, ny, nz in cell.neighbors:
                            if self.cell_grid[nz, ny, nx] is None:
                                empty_neighbors.append((nx, ny, nz))
                        
                        if empty_neighbors:
                            # Choose random empty neighbor
                            child_x, child_y, child_z = random.choice(empty_neighbors)
                            
                            # Create child cell
                            child_type = cell.type
                            
                            # Mutation
                            if random.random() < self.params.mutation_rate:
                                child_type = random.choice(list(self.cell_types.keys()))
                                if child_type == 0:  # Don't mutate to dead
                                    child_type = cell.type
                            
                            cells_to_add.append((child_x, child_y, child_z, child_type))
                            
                            # Parent loses energy and increments division count
                            cell.energy *= 0.6
                            cell.division_count += 1
                    
                    # Update cell in grid
                    new_grid[z, y, x] = cell
        
        # Add new cells from division
        for x, y, z, cell_type in cells_to_add:
            self.create_cell(x, y, z, cell_type)
        
        self.cell_grid = new_grid
    
    def cancer_growth_rules(self):
        """Cancer-specific growth and invasion rules"""
        new_grid = np.copy(self.cell_grid)
        
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    cell = self.cell_grid[z, y, x]
                    
                    if cell is None or cell.type != 3:  # Only process cancer cells
                        continue
                    
                    # Cancer cells have aggressive metabolism
                    cell.energy -= cell.metabolism
                    self.nutrient_grid[z, y, x] -= cell.metabolism * 3  # Consume more nutrients
                    
                    # Produce more toxins
                    self.toxin_grid[z, y, x] += cell.metabolism * 0.3
                    
                    # Angiogenesis: create nutrient sources
                    if random.random() < 0.01:
                        # Increase nutrients in neighborhood
                        for nx, ny, nz in self.get_neighbors(x, y, z, radius=2):
                            self.nutrient_grid[nz, ny, nx] += 0.5
                    
                    # Invasion: replace neighboring cells
                    if random.random() < 0.1:  # 10% chance per update
                        neighbors = self.get_neighbors(x, y, z)
                        for nx, ny, nz in neighbors:
                            neighbor = self.cell_grid[nz, ny, nx]
                            if neighbor and neighbor.type != 3:  # Not cancer
                                # Cancer invades and replaces
                                if random.random() < 0.3:  # 30% success rate
                                    new_grid[nz, ny, nx] = None
                                    self.create_cell(nx, ny, nz, 3)  # New cancer cell
                    
                    # Check for death (cancer cells are more resilient)
                    if cell.energy <= self.params.death_threshold * 0.5:
                        new_grid[z, y, x] = None
                        self.cell_count -= 1
        
        self.cell_grid = new_grid
    
    def neural_development_rules(self):
        """Neural network development and signal propagation rules"""
        # Clear signal grid
        self.signal_grid.fill(0.0)
        
        # First pass: neural cells generate signals
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    cell = self.cell_grid[z, y, x]
                    
                    if cell and cell.type == 5:  # Neural cell
                        # Neural cells generate signals based on energy and activity
                        signal_strength = cell.energy * 0.1
                        self.signal_grid[z, y, x] += signal_strength
                        
                        # Neural cells consume more energy
                        cell.energy -= cell.metabolism * 1.5
        
        # Second pass: signal propagation
        new_signal_grid = np.copy(self.signal_grid)
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    if self.signal_grid[z, y, x] > 0.1:
                        # Propagate signal to neighbors
                        neighbors = self.get_neighbors(x, y, z)
                        signal_to_propagate = self.signal_grid[z, y, x] * 0.2
                        
                        for nx, ny, nz in neighbors:
                            new_signal_grid[nz, ny, nx] += signal_to_propagate
        
        self.signal_grid = new_signal_grid * 0.95  # Signal decay
        
        # Third pass: signal effects on other cells
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    cell = self.cell_grid[z, y, x]
                    signal = self.signal_grid[z, y, x]
                    
                    if cell and signal > 0.5:
                        # Signals can stimulate or inhibit other cells
                        if cell.type == 1:  # Living cells
                            cell.energy += signal * 0.1  # Stimulation
                        elif cell.type == 4:  # Immune cells
                            # Immune cells activated by neural signals
                            cell.energy += signal * 0.2
    
    def diffuse_nutrients(self):
        """Diffuse nutrients through the grid"""
        new_nutrients = np.copy(self.nutrient_grid)
        
        for z in range(1, self.depth - 1):
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    # Laplacian diffusion
                    diffusion = 0.0
                    neighbors = self.get_neighbors(x, y, z)
                    
                    for nx, ny, nz in neighbors:
                        diffusion += self.nutrient_grid[nz, ny, nx] - self.nutrient_grid[z, y, x]
                    
                    diffusion *= self.params.diffusion_rate / len(neighbors)
                    new_nutrients[z, y, x] += diffusion
        
        # Add nutrient production (simulating external source)
        center_z, center_y, center_x = self.depth//2, self.height//2, self.width//2
        production_radius = 5
        
        for dz in range(-production_radius, production_radius + 1):
            for dy in range(-production_radius, production_radius + 1):
                for dx in range(-production_radius, production_radius + 1):
                    if dz*dz + dy*dy + dx*dx <= production_radius*production_radius:
                        nz, ny, nx = center_z + dz, center_y + dy, center_x + dx
                        if self.is_valid_position(nx, ny, nz):
                            new_nutrients[nz, ny, nx] += self.params.nutrient_production
        
        self.nutrient_grid = np.clip(new_nutrients, 0.0, 10.0)
    
    def diffuse_toxins(self):
        """Diffuse toxins through the grid with decay"""
        new_toxins = np.copy(self.toxin_grid)
        
        for z in range(1, self.depth - 1):
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    # Laplacian diffusion
                    diffusion = 0.0
                    neighbors = self.get_neighbors(x, y, z)
                    
                    for nx, ny, nz in neighbors:
                        diffusion += self.toxin_grid[nz, ny, nx] - self.toxin_grid[z, y, x]
                    
                    diffusion *= self.params.diffusion_rate / len(neighbors)
                    new_toxins[z, y, x] += diffusion
                    
                    # Toxin decay
                    new_toxins[z, y, x] *= (1.0 - self.params.toxin_decay)
        
        self.toxin_grid = np.clip(new_toxins, 0.0, 100.0)
    
    def update_statistics(self):
        """Update simulation statistics"""
        total_energy = 0
        type_counts = defaultdict(int)
        living_cells = 0
        
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    cell = self.cell_grid[z, y, x]
                    if cell and cell.type != 0:
                        total_energy += cell.energy
                        type_counts[cell.type] += 1
                        living_cells += 1
        
        self.cell_count = living_cells
        self.average_energy = total_energy / living_cells if living_cells > 0 else 0
        
        # Calculate diversity index (Simpson's diversity)
        total_cells = sum(type_counts.values())
        if total_cells > 0:
            diversity = 0.0
            for count in type_counts.values():
                proportion = count / total_cells
                diversity += proportion * proportion
            self.diversity_index = 1.0 - diversity
        else:
            self.diversity_index = 0.0
    
    def initialize_rendering(self):
        """Initialize OpenGL rendering for the cellular automata"""
        # Compile shader for cell rendering
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;
        layout (location = 2) in float aSize;
        
        out vec3 Color;
        out float Size;
        
        uniform mat4 view;
        uniform mat4 projection;
        uniform float time;
        
        void main() {
            Color = aColor;
            Size = aSize;
            
            // Animate cells with gentle pulsing
            vec3 animatedPos = aPos + 0.1 * sin(aPos * 3.0 + time) * 0.01;
            
            gl_Position = projection * view * vec4(animatedPos, 1.0);
            gl_PointSize = 8.0 * aSize;
        }
        """
        
        fragment_shader = """
        #version 330 core
        in vec3 Color;
        in float Size;
        
        out vec4 FragColor;
        
        uniform float time;
        
        void main() {
            // Circular point sprite
            vec2 coord = gl_PointCoord * 2.0 - 1.0;
            float r = length(coord);
            
            if (r > 1.0) discard;
            
            // Cell-like appearance with nucleus
            float nucleus = 1.0 - smoothstep(0.0, 0.3, r);
            float membrane = smoothstep(0.7, 1.0, r);
            
            vec3 cellColor = Color;
            
            // Add some biological variation
            cellColor *= (0.9 + 0.2 * sin(Size * 10.0 + time));
            
            // Nucleus is brighter
            vec3 finalColor = mix(cellColor, vec3(1.0), nucleus * 0.5);
            
            // Membrane edge
            finalColor = mix(finalColor, vec3(0.1), membrane);
            
            float alpha = 1.0 - r * r;
            FragColor = vec4(finalColor, alpha);
        }
        """
        
        try:
            self.cell_shader = compileProgram(
                compileShader(vertex_shader, GL_VERTEX_SHADER),
                compileShader(fragment_shader, GL_FRAGMENT_SHADER)
            )
            self.initialized_rendering = True
        except Exception as e:
            print(f"Failed to initialize cellular automata rendering: {e}")
    
    def render(self, view_matrix: glm.mat4, projection_matrix: glm.mat4):
        """Render the cellular automata"""
        if not self.initialized_rendering:
            self.initialize_rendering()
        
        if not self.initialized_rendering or self.cell_count == 0:
            return
        
        glUseProgram(self.cell_shader)
        
        # Set shader uniforms
        glUniformMatrix4fv(
            glGetUniformLocation(self.cell_shader, "view"),
            1, GL_FALSE, glm.value_ptr(view_matrix)
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.cell_shader, "projection"),
            1, GL_FALSE, glm.value_ptr(projection_matrix)
        )
        glUniform1f(
            glGetUniformLocation(self.cell_shader, "time"),
            time.time()
        )
        
        # Prepare vertex data
        vertices = []
        colors = []
        sizes = []
        
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    cell = self.cell_grid[z, y, x]
                    if cell and cell.type != 0:  # Skip dead cells
                        # Convert grid coordinates to world coordinates
                        world_x = (x - self.width / 2) * 0.1
                        world_y = (y - self.height / 2) * 0.1
                        world_z = (z - self.depth / 2) * 0.1
                        
                        vertices.append([world_x, world_y, world_z])
                        
                        # Get color based on cell type
                        cell_info = self.cell_types[cell.type]
                        colors.append(cell_info["color"])
                        
                        # Size based on energy and age
                        size = 0.5 + (cell.energy / 200.0) + (cell.age / 1000.0)
                        sizes.append(min(size, 2.0))
        
        if not vertices:
            return
        
        # Convert to numpy arrays
        vertices_np = np.array(vertices, dtype=np.float32)
        colors_np = np.array(colors, dtype=np.float32)
        sizes_np = np.array(sizes, dtype=np.float32)
        
        # Create and bind VAO
        if self.cell_vao is None:
            self.cell_vao = glGenVertexArrays(1)
        
        glBindVertexArray(self.cell_vao)
        
        # Vertex buffer
        vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, vertices_np.nbytes, vertices_np, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        # Color buffer
        vbo_colors = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors)
        glBufferData(GL_ARRAY_BUFFER, colors_np.nbytes, colors_np, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        
        # Size buffer
        vbo_sizes = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_sizes)
        glBufferData(GL_ARRAY_BUFFER, sizes_np.nbytes, sizes_np, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)
        
        # Enable point sprites and blending
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)
        
        # Draw cells
        glDrawArrays(GL_POINTS, 0, len(vertices))
        
        # Cleanup
        glDeleteBuffers(1, [vbo_vertices])
        glDeleteBuffers(1, [vbo_colors])
        glDeleteBuffers(1, [vbo_sizes])
        
        glDisable(GL_BLEND)
        glUseProgram(0)
    
    def get_simulation_info(self) -> Dict[str, Any]:
        """Get current simulation statistics and information"""
        type_counts = defaultdict(int)
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    cell = self.cell_grid[z, y, x]
                    if cell:
                        type_counts[self.cell_types[cell.type]["name"]] += 1
        
        return {
            "generation": self.generation,
            "total_cells": self.cell_count,
            "average_energy": self.average_energy,
            "diversity_index": self.diversity_index,
            "cell_type_distribution": dict(type_counts),
            "total_nutrients": np.sum(self.nutrient_grid),
            "total_toxins": np.sum(self.toxin_grid),
            "grid_dimensions": f"{self.width}x{self.height}x{self.depth}"
        }
    
    def add_external_stimulus(self, x: int, y: int, z: int, stimulus_type: str, strength: float):
        """Add external stimulus to the simulation"""
        if not self.is_valid_position(x, y, z):
            return
        
        if stimulus_type == "nutrient":
            self.nutrient_grid[z, y, x] += strength
        elif stimulus_type == "toxin":
            self.toxin_grid[z, y, x] += strength
        elif stimulus_type == "signal":
            self.signal_grid[z, y, x] += strength
        elif stimulus_type == "radiation":
            # Radiation can kill cells or cause mutations
            cell = self.cell_grid[z, y, x]
            if cell:
                if random.random() < strength * 0.1:  # Cell death
                    self.cell_grid[z, y, x] = None
                    self.cell_count -= 1
                elif random.random() < strength * 0.05:  # Mutation
                    cell.type = random.choice(list(self.cell_types.keys()))
    
    def save_state(self, filename: str):
        """Save simulation state to file"""
        state = {
            "generation": self.generation,
            "params": self.params.__dict__,
            "cell_data": []
        }
        
        # Save cell data
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    cell = self.cell_grid[z, y, x]
                    if cell:
                        cell_data = {
                            "position": cell.position,
                            "type": cell.type,
                            "age": cell.age,
                            "energy": cell.energy,
                            "metabolism": cell.metabolism,
                            "division_count": cell.division_count,
                            "genetic_markers": cell.genetic_markers.tolist()
                        }
                        state["cell_data"].append(cell_data)
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Simulation state saved to {filename}")
    
    def load_state(self, filename: str):
        """Load simulation state from file"""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            # Clear current state
            self.cell_grid.fill(None)
            self.nutrient_grid.fill(1.0)
            self.toxin_grid.fill(0.0)
            self.signal_grid.fill(0.0)
            
            # Load parameters
            for key, value in state["params"].items():
                setattr(self.params, key, value)
            
            # Load cells
            self.cell_count = 0
            for cell_data in state["cell_data"]:
                x, y, z = cell_data["position"]
                cell = CellState(
                    type=cell_data["type"],
                    age=cell_data["age"],
                    energy=cell_data["energy"],
                    metabolism=cell_data["metabolism"],
                    division_count=cell_data["division_count"],
                    genetic_markers=np.array(cell_data["genetic_markers"]),
                    position=(x, y, z),
                    neighbors=self.get_neighbors(x, y, z)
                )
                self.cell_grid[z, y, x] = cell
                self.cell_count += 1
            
            self.generation = state["generation"]
            self.update_statistics()
            
            print(f"Simulation state loaded from {filename}")
            
        except Exception as e:
            print(f"Error loading simulation state: {e}")

# Example usage and integration
class BiologicalSimulation:
    """Wrapper for biological simulation integration"""
    
    def __init__(self, simulation_app):
        self.simulation_app = simulation_app
        self.cellular_automata = CellularAutomata3D(32, 32, 32)  # Smaller for performance
        self.current_rule_set = "bio_metabolism"
        self.update_interval = 5  # Update every 5 frames
        self.paused = False
        
    def update(self, dt):
        """Update biological simulation"""
        if self.paused:
            return
        
        if self.simulation_app.frame_count % self.update_interval == 0:
            self.cellular_automata.update(self.current_rule_set)
    
    def render(self, view_matrix, projection_matrix):
        """Render biological simulation"""
        self.cellular_automata.render(view_matrix, projection_matrix)
    
    def toggle_pause(self):
        """Toggle simulation pause state"""
        self.paused = not self.paused
        print(f"Biological simulation {'paused' if self.paused else 'resumed'}")
    
    def change_rule_set(self, rule_set):
        """Change the active rule set"""
        if rule_set in self.cellular_automata.rules:
            self.current_rule_set = rule_set
            print(f"Changed rule set to: {rule_set}")
        else:
            print(f"Unknown rule set: {rule_set}")

if __name__ == "__main__":
    # Test the biological cellular automata
    bio_sim = CellularAutomata3D(16, 16, 16)  # Small for testing
    
    print("Initial simulation state:")
    info = bio_sim.get_simulation_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Run a few generations
    for i in range(10):
        bio_sim.update("bio_metabolism")
    
    print("\nAfter 10 generations:")
    info = bio_sim.get_simulation_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("Biological Cellular Automata test completed successfully")
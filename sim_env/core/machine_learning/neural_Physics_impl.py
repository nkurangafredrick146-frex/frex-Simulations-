#!/usr/bin/env python3
"""
Neural Physics & Differentiable Simulations
Differentiable physics engines and neural network-integrated physical models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glm
from typing import Dict, List, Optional, Tuple, Any
import json
import time
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class PhysicalProperties:
    mass: float = 1.0
    elasticity: float = 0.8
    friction: float = 0.1
    density: float = 1.0
    viscosity: float = 0.01
    youngs_modulus: float = 1e6
    poissons_ratio: float = 0.3

class DifferentiableParticleSystem(nn.Module):
    """Differentiable particle system with learnable physics"""
    
    def __init__(self, num_particles: int = 100, device='cuda'):
        super(DifferentiableParticleSystem, self).__init__()
        self.num_particles = num_particles
        self.device = device
        
        # Learnable physical parameters
        self.mass = nn.Parameter(torch.ones(num_particles, device=device))
        self.elasticity = nn.Parameter(torch.full((num_particles,), 0.8, device=device))
        self.friction = nn.Parameter(torch.full((num_particles,), 0.1, device=device))
        self.density = nn.Parameter(torch.ones(num_particles, device=device))
        
        # Particle state (position, velocity, force)
        self.positions = nn.Parameter(torch.randn(num_particles, 3, device=device) * 0.1)
        self.velocities = nn.Parameter(torch.zeros(num_particles, 3, device=device))
        self.forces = torch.zeros(num_particles, 3, device=device)
        
        # Neural network for force prediction
        self.force_network = NeuralForceModel(6, 64, 3).to(device)
        
        # Environment parameters
        self.gravity = torch.tensor([0.0, -9.8, 0.0], device=device)
        self.dt = 0.016
        
    def forward(self, external_forces: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of differentiable physics"""
        # Reset forces
        self.forces = torch.zeros_like(self.positions)
        
        # Apply gravity
        gravity_force = self.gravity.unsqueeze(0) * self.mass.unsqueeze(1)
        self.forces += gravity_force
        
        # Apply external forces
        if external_forces is not None:
            self.forces += external_forces
        
        # Neural network predicted forces (inter-particle interactions)
        neural_forces = self.compute_neural_forces()
        self.forces += neural_forces
        
        # Compute acceleration
        acceleration = self.forces / self.mass.unsqueeze(1)
        
        # Update velocities (Euler integration)
        new_velocities = self.velocities + acceleration * self.dt
        
        # Apply friction
        friction_mask = torch.norm(new_velocities, dim=1) > 0.1
        friction_force = -new_velocities[friction_mask] * self.friction[friction_mask].unsqueeze(1)
        new_velocities[friction_mask] += friction_force * self.dt
        
        # Update positions
        new_positions = self.positions + new_velocities * self.dt
        
        # Handle collisions with ground (z=0 plane)
        ground_collision = new_positions[:, 1] < 0
        if ground_collision.any():
            new_positions[ground_collision, 1] = 0
            new_velocities[ground_collision, 1] = -new_velocities[ground_collision, 1] * self.elasticity[ground_collision].unsqueeze(1)
        
        return new_positions, new_velocities
    
    def compute_neural_forces(self) -> torch.Tensor:
        """Compute inter-particle forces using neural network"""
        forces = torch.zeros_like(self.positions)
        
        # For each particle, consider neighbors within a radius
        for i in range(self.num_particles):
            # Compute distances to all other particles
            distances = torch.norm(self.positions - self.positions[i], dim=1)
            neighbors = distances < 1.0  # Interaction radius
            
            if neighbors.sum() > 1:  # Exclude self
                neighbor_indices = torch.where(neighbors)[0]
                
                for j in neighbor_indices:
                    if i != j:
                        # Prepare input features for force network
                        rel_pos = self.positions[j] - self.positions[i]
                        rel_vel = self.velocities[j] - self.velocities[i]
                        distance = distances[j]
                        
                        features = torch.cat([
                            rel_pos,
                            rel_vel,
                            torch.tensor([distance, self.mass[i] / self.mass[j]], device=self.device)
                        ])
                        
                        # Predict force
                        force = self.force_network(features.unsqueeze(0))
                        forces[i] += force.squeeze(0)
        
        return forces
    
    def step(self, external_forces: Optional[torch.Tensor] = None):
        """Perform one simulation step"""
        with torch.no_grad():
            new_pos, new_vel = self.forward(external_forces)
            
            # Update state
            self.positions.data = new_pos
            self.velocities.data = new_vel
    
    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get current system state"""
        return {
            'positions': self.positions.detach(),
            'velocities': self.velocities.detach(),
            'forces': self.forces.detach(),
            'mass': self.mass.detach(),
            'elasticity': self.elasticity.detach()
        }

class NeuralForceModel(nn.Module):
    """Neural network for predicting physical forces"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(NeuralForceModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()  # Force output bounded
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) * 10.0  # Scale force output

class DifferentiableFluidSimulator(nn.Module):
    """Differentiable fluid dynamics simulator"""
    
    def __init__(self, grid_size: int = 32, device='cuda'):
        super(DifferentiableFluidSimulator, self).__init__()
        self.grid_size = grid_size
        self.device = device
        
        # Fluid properties
        self.density = nn.Parameter(torch.ones(grid_size, grid_size, grid_size, device=device))
        self.velocity = nn.Parameter(torch.zeros(3, grid_size, grid_size, grid_size, device=device))
        self.pressure = nn.Parameter(torch.zeros(grid_size, grid_size, grid_size, device=device))
        
        # Neural network for turbulence modeling
        self.turbulence_model = TurbulenceModel(4, 128, 3).to(device)
        
        # Simulation parameters
        self.dt = 0.01
        self.viscosity = 0.001
        self.ambient_density = 1.0
        
    def forward(self, sources: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of differentiable fluid simulation"""
        # Advection
        self.advect_velocity()
        self.advect_density()
        
        # Apply forces
        if sources is not None:
            self.apply_sources(sources)
        
        # Neural turbulence
        turbulence = self.compute_turbulence()
        self.velocity += turbulence * self.dt
        
        # Pressure projection
        self.project_pressure()
        
        # Diffusion
        self.diffuse_velocity()
        
        return {
            'density': self.density,
            'velocity': self.velocity,
            'pressure': self.pressure
        }
    
    def advect_velocity(self):
        """Advect velocity field using semi-Lagrangian method"""
        # Simplified advection - in practice would use more sophisticated method
        new_velocity = torch.zeros_like(self.velocity)
        for i in range(3):
            new_velocity[i] = self.velocity[i] - torch.einsum('jkl,jkl->jkl', 
                self.velocity[i], torch.stack(torch.gradient(self.velocity[i], dim=[0,1,2])))
        self.velocity.data = new_velocity
        
    def advect_density(self):
        """Advect density field"""
        # Simplified advection
        grad_density = torch.stack(torch.gradient(self.density, dim=[0,1,2]))
        advection = torch.einsum('i,jkl->jkl', torch.ones(3, device=self.device), 
                                self.density.unsqueeze(0) * grad_density)
        self.density.data = self.density - advection * self.dt
        
    def apply_sources(self, sources: torch.Tensor):
        """Apply source terms to fluid"""
        self.density.data += sources[0] * self.dt
        self.velocity.data += sources[1:4] * self.dt
        
    def compute_turbulence(self) -> torch.Tensor:
        """Compute turbulence using neural network"""
        turbulence = torch.zeros_like(self.velocity)
        
        # For each grid point, compute turbulence
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    features = torch.cat([
                        self.velocity[:, i, j, k],
                        torch.tensor([self.density[i, j, k]], device=self.device)
                    ])
                    
                    turb = self.turbulence_model(features.unsqueeze(0))
                    turbulence[:, i, j, k] = turb.squeeze(0)
        
        return turbulence
        
    def project_pressure(self):
        """Project velocity field to be divergence-free"""
        # Simplified pressure projection
        divergence = self.compute_divergence()
        
        # Solve Poisson equation for pressure (simplified)
        pressure_update = divergence * 0.1
        self.pressure.data += pressure_update
        
        # Subtract pressure gradient from velocity
        pressure_grad = torch.stack(torch.gradient(self.pressure, dim=[0,1,2]))
        self.velocity.data -= pressure_grad * self.dt
        
    def compute_divergence(self) -> torch.Tensor:
        """Compute divergence of velocity field"""
        grad_vx = torch.gradient(self.velocity[0], dim=[0,1,2])
        grad_vy = torch.gradient(self.velocity[1], dim=[0,1,2])
        grad_vz = torch.gradient(self.velocity[2], dim=[0,1,2])
        
        divergence = grad_vx[0] + grad_vy[1] + grad_vz[2]
        return divergence
        
    def diffuse_velocity(self):
        """Diffuse velocity field using viscosity"""
        # Simplified diffusion
        for i in range(3):
            laplacian = self.compute_laplacian(self.velocity[i])
            self.velocity[i].data += self.viscosity * laplacian * self.dt
        
    def compute_laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian of a scalar field"""
        second_derivatives = []
        for dim in range(3):
            grad = torch.gradient(field, dim=dim)
            second_grad = torch.gradient(grad[0], dim=dim)
            second_derivatives.append(second_grad[0])
        
        return sum(second_derivatives)

class TurbulenceModel(nn.Module):
    """Neural network for turbulence modeling"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(TurbulenceModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) * 0.1  # Scale turbulence

class NeuralPhysicsEngine:
    """Main neural physics engine integrating differentiable simulations"""
    
    def __init__(self, simulation_app):
        self.simulation_app = simulation_app
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Differentiable simulators
        self.particle_system = DifferentiableParticleSystem(100, self.device)
        self.fluid_simulator = DifferentiableFluidSimulator(32, self.device)
        
        # Training and optimization
        self.optimizer = optim.Adam(
            list(self.particle_system.parameters()) + 
            list(self.fluid_simulator.parameters()),
            lr=0.001
        )
        
        # Loss functions
        self.loss_functions = {
            'position_mse': nn.MSELoss(),
            'velocity_mse': nn.MSELoss(),
            'energy_conservation': self.energy_conservation_loss,
            'momentum_conservation': self.momentum_conservation_loss
        }
        
        # Training data
        self.training_trajectories = []
        self.validation_trajectories = []
        
        # Real-time adaptation
        self.adaptation_enabled = True
        self.adaptation_interval = 10  # steps
        
        print(f"Neural Physics Engine initialized on {self.device}")
        
    def train_step(self, target_positions: torch.Tensor, target_velocities: torch.Tensor) -> Dict[str, float]:
        """Perform one training step to match target behavior"""
        self.optimizer.zero_grad()
        
        # Forward pass
        pred_positions, pred_velocities = self.particle_system()
        
        # Compute losses
        position_loss = self.loss_functions['position_mse'](pred_positions, target_positions)
        velocity_loss = self.loss_functions['velocity_mse'](pred_velocities, target_velocities)
        energy_loss = self.loss_functions['energy_conservation'](pred_positions, pred_velocities)
        momentum_loss = self.loss_functions['momentum_conservation'](pred_velocities)
        
        # Total loss
        total_loss = (position_loss + velocity_loss + 
                     0.1 * energy_loss + 0.1 * momentum_loss)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'position_loss': position_loss.item(),
            'velocity_loss': velocity_loss.item(),
            'energy_loss': energy_loss.item(),
            'momentum_loss': momentum_loss.item()
        }

    def energy_conservation_loss(self, positions: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
        """Loss for energy conservation"""
        kinetic_energy = 0.5 * self.particle_system.mass * torch.norm(velocities, dim=1)**2
        potential_energy = self.particle_system.mass * 9.8 * positions[:, 1]  # gravity potential
        
        total_energy = kinetic_energy.sum() + potential_energy.sum()
        
        # Energy should be conserved (change should be small)
        energy_change = torch.abs(total_energy - getattr(self, 'last_energy', total_energy))
        self.last_energy = total_energy.detach()
        
        return energy_change / len(positions)

    def momentum_conservation_loss(self, velocities: torch.Tensor) -> torch.Tensor:
        """Loss for momentum conservation"""
        momentum = self.particle_system.mass.unsqueeze(1) * velocities
        total_momentum = momentum.sum(dim=0)
        
        # Momentum should be conserved (change should be small)
        momentum_change = torch.norm(total_momentum - getattr(self, 'last_momentum', total_momentum))
        self.last_momentum = total_momentum.detach()
        
        return momentum_change / len(velocities)

    def adapt_to_observation(self, observed_positions: np.ndarray, observed_velocities: np.ndarray):
        """Adapt physics parameters to match observed behavior"""
        if not self.adaptation_enabled:
            return
        
        # Convert to tensors
        target_positions = torch.tensor(observed_positions, dtype=torch.float32, device=self.device)
        target_velocities = torch.tensor(observed_velocities, dtype=torch.float32, device=self.device)
        
        # Perform adaptation
        losses = self.train_step(target_positions, target_velocities)
        
        if self.simulation_app.frame_count % 60 == 0:  # Log every second
            print(f"Physics adaptation - Loss: {losses['total_loss']:.6f}")

    def predict_future_state(self, current_state: Dict[str, Any], steps: int = 10) -> List[Dict[str, Any]]:
        """Predict future states using differentiable physics"""
        predictions = []
        
        # Save current state
        original_positions = self.particle_system.positions.clone()
        original_velocities = self.particle_system.velocities.clone()
        
        # Perform prediction steps
        for step in range(steps):
            with torch.no_grad():
                new_pos, new_vel = self.particle_system()
                
                predictions.append({
                    'positions': new_pos.cpu().numpy(),
                    'velocities': new_vel.cpu().numpy(),
                    'time': current_state['simulation_time'] + step * self.particle_system.dt
                })
                
                # Update for next step
                self.particle_system.positions.data = new_pos
                self.particle_system.velocities.data = new_vel
        
        # Restore original state
        self.particle_system.positions.data = original_positions
        self.particle_system.velocities.data = original_velocities
        
        return predictions

    def compute_sensitivity(self, parameter_name: str, target_metric: str) -> torch.Tensor:
        """Compute sensitivity of target metric to physics parameter"""
        # Enable gradients for all parameters
        for param in self.particle_system.parameters():
            param.requires_grad = True
        
        # Forward pass
        positions, velocities = self.particle_system()
        
        # Compute target metric
        if target_metric == 'spread':
            metric = torch.std(positions, dim=0).mean()
        elif target_metric == 'energy':
            kinetic_energy = 0.5 * self.particle_system.mass * torch.norm(velocities, dim=1)**2
            metric = kinetic_energy.mean()
        else:
            metric = torch.norm(positions, dim=1).mean()
        
        # Compute gradient
        if parameter_name == 'mass':
            parameter = self.particle_system.mass
        elif parameter_name == 'elasticity':
            parameter = self.particle_system.elasticity
        elif parameter_name == 'friction':
            parameter = self.particle_system.friction
        else:
            raise ValueError(f"Unknown parameter: {parameter_name}")
        
        gradient = torch.autograd.grad(metric, parameter, retain_graph=True)[0]
        
        return gradient

    def get_physical_parameters(self) -> Dict[str, np.ndarray]:
        """Get current learned physical parameters"""
        return {
            'mass': self.particle_system.mass.detach().cpu().numpy(),
            'elasticity': self.particle_system.elasticity.detach().cpu().numpy(),
            'friction': self.particle_system.friction.detach().cpu().numpy(),
            'density': self.particle_system.density.detach().cpu().numpy()
        }

    def set_physical_parameters(self, parameters: Dict[str, np.ndarray]):
        """Set physical parameters from external source"""
        with torch.no_grad():
            if 'mass' in parameters:
                self.particle_system.mass.data = torch.tensor(parameters['mass'], device=self.device)
            if 'elasticity' in parameters:
                self.particle_system.elasticity.data = torch.tensor(parameters['elasticity'], device=self.device)
            if 'friction' in parameters:
                self.particle_system.friction.data = torch.tensor(parameters['friction'], device=self.device)

    def render_physics_debug(self, view_matrix: glm.mat4, projection_matrix: glm.mat4):
        """Render debug visualization of neural physics"""
        # This would integrate with the main rendering system
        # For now, just extract positions for rendering
        state = self.particle_system.get_state()
        positions = state['positions'].cpu().numpy()
        
        # In a real implementation, this would pass positions to the rendering system
        print(f"Rendering {len(positions)} differentiable particles")

    def save_model(self, filepath: str):
        """Save trained neural physics models"""
        torch.save({
            'particle_system_state': self.particle_system.state_dict(),
            'fluid_simulator_state': self.fluid_simulator.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, filepath)
        print(f"Neural physics model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained neural physics models"""
        checkpoint = torch.load(filepath)
        self.particle_system.load_state_dict(checkpoint['particle_system_state'])
        self.fluid_simulator.load_state_dict(checkpoint['fluid_simulator_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"Neural physics model loaded from {filepath}")

    def cleanup(self):
        """Cleanup resources"""
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Example integration with main simulation
class NeuralPhysicsSimulation:
    """Wrapper simulation that uses neural physics"""
    
    def __init__(self, base_simulation, neural_engine):
        self.base_simulation = base_simulation
        self.neural_engine = neural_engine
        self.use_neural_physics = True
        
    def update(self, dt):
        if self.use_neural_physics:
            # Use differentiable physics
            self.neural_engine.particle_system.step()
            
            # Extract state for adaptation
            current_state = self.get_current_state()
            if self.should_adapt():
                self.neural_engine.adapt_to_observation(
                    current_state['positions'],
                    current_state['velocities']
                )
        else:
            # Use traditional physics
            self.base_simulation.update(dt)
    
    def get_current_state(self):
        state = self.neural_engine.particle_system.get_state()
        return {
            'positions': state['positions'].cpu().numpy(),
            'velocities': state['velocities'].cpu().numpy(),
            'simulation_time': self.base_simulation.simulation_time
        }
    
    def should_adapt(self):
        # Adapt every N frames
        return self.base_simulation.frame_count % self.neural_engine.adaptation_interval == 0
    
    def render(self, view_matrix, projection_matrix):
        # Render using neural physics state
        self.neural_engine.render_physics_debug(view_matrix, projection_matrix)

# Test and demonstration
if __name__ == "__main__":
    # Test the neural physics engine
    engine = NeuralPhysicsEngine(None)
    
    # Test prediction
    test_state = {
        'positions': np.random.randn(100, 3) * 0.1,
        'velocities': np.zeros((100, 3)),
        'simulation_time': 0.0
    }
    
    predictions = engine.predict_future_state(test_state, steps=5)
    print(f"Generated {len(predictions)} future predictions")
    
    # Test sensitivity analysis
    sensitivity = engine.compute_sensitivity('mass', 'energy')
    print(f"Mass sensitivity: {sensitivity.mean().item():.6f}")
    
    print("Neural Physics Engine test completed successfully")

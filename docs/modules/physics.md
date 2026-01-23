# Physics Module

## Overview
The Physics module provides advanced physics simulation capabilities including particle dynamics, rigid body physics, fluid dynamics, and collision detection.

## Features

### Core Physics Engine
- **Particle System**: Simulate thousands of particles with physical properties (mass, velocity, acceleration)
- **Rigid Body Dynamics**: Support for rigid bodies with rotation, angular velocity, and torque
- **Collision Detection**: Efficient collision detection using spatial partitioning and bounding volumes
- **Force Fields**: Gravity, wind, magnetic forces, and custom force calculations

### Integration Methods
- **Verlet Integration**: Semi-implicit integration for stable particle dynamics
- **Runge-Kutta**: Higher-order integration for complex scenarios
- **Euler Integration**: Fast, simple integration for real-time applications

### Advanced Features
- **Soft Body Simulation**: Cloth, jelly, and deformable object simulation
- **Fluid Dynamics**: SPH-based fluid simulation with viscosity and pressure
- **Constraint Solving**: Distance constraints, angle constraints, and contact constraints
- **Sleeping Bodies**: Performance optimization by deactivating inactive objects

## Usage Example

```python
from sim_env.Physics_simulation_module import PhysicsEngine, Particle

# Initialize physics engine
engine = PhysicsEngine()

# Create a particle
particle = Particle(position=(0, 0, 0), velocity=(1, 0, 0), mass=1.0)
engine.add_particle(particle)

# Update physics
engine.update(dt=0.016)  # 60 FPS
```

## Configuration

Customize physics behavior via `PhysicsSettings`:

```python
from sim_env.Physics_simulation_module import PhysicsSettings

settings = PhysicsSettings()
settings.gravity = (0, -9.81, 0)
settings.air_density = 1.2
settings.collision_enabled = True
```

## Performance
- Optimized for 10,000+ particles in real-time
- Spatial hashing for collision detection
- Numba JIT compilation for critical loops

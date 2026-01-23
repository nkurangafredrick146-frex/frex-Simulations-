# Architecture

## System Overview

The frex Simulations system is a modular, scalable architecture supporting physics, machine learning, quantum computing, and advanced rendering.

```
┌─────────────────────────────────────────┐
│     User Interface & Real-Time GUI      │
├─────────────────────────────────────────┤
│  Rendering Engine    │   ML Pipeline    │
│  (OpenGL/Ray Trace)  │   (Neural Net)   │
├─────────────────────────────────────────┤
│ Physics Engine │ Quantum Sim │ Fluid Dyn │
├─────────────────────────────────────────┤
│         Core Simulation Loop             │
├─────────────────────────────────────────┤
│  File I/O  │  Data Export  │  Config   │
└─────────────────────────────────────────┘
```

## Core Modules

### Physics (`Physics_simulation_module.py`)
- Particle dynamics and rigid body physics
- Collision detection and constraint solving
- Force fields and integrators
- Soft body and cloth simulation

### Rendering (`Rendering_engine.py`)
- Real-time OpenGL rendering
- Ray tracing and path tracing
- Neural radiance fields (NeRF)
- Post-processing and effects

### Machine Learning (`ml_pipeline.py`, `neural_Physics.py`)
- Data loading and preprocessing
- Neural network training
- Physics prediction models
- NeRF-based rendering

### Quantum Computing (`quantum_computing_Hybrid.py`)
- Quantum circuit simulation
- Quantum physics visualization
- Hybrid quantum-classical algorithms

### Fluid Dynamics (`fluid_dynamics.py`)
- SPH-based fluid simulation
- Grid-based solvers
- Interactive fluid effects

## Data Flow

1. **Initialization**: Load configuration from `config.json`
2. **Simulation Loop**:
   - Physics: Update particle positions and velocities
   - Collision: Detect and resolve collisions
   - Forces: Apply gravity, wind, custom forces
   - ML: Use neural models for optimization (optional)
   - Rendering: Draw scene to screen
3. **Output**: Export data, save frames, record metrics

## Design Patterns

### Modular Design
Each simulation type (physics, fluid, quantum) is independent and can be used alone or together.

### Configurable Settings
`PhysicsSettings`, `RenderSettings`, etc. allow runtime customization.

### Real-Time Interaction
GUI enables live parameter adjustment and visualization.

### GPU Acceleration
Critical paths use OpenGL, CUDA, or Numba for performance.

## Performance Characteristics

| Task | Target | Notes |
|------|--------|-------|
| Particle Physics | 10,000+ particles @ 60 FPS | With LOD |
| Fluid Simulation | 64³ grid @ 30 FPS | Real-time |
| Ray Tracing | 1024² @ 4 samples/sec | Interactive |
| Neural Rendering | 1920×1080 @ 30 FPS | NeRF inference |

## Extension Points

- **Custom Physics**: Inherit from `PhysicsEngine`
- **Custom Renderers**: Implement `Renderer` interface
- **Custom Forces**: Add to `PhysicsEngine.forces`
- **Custom Shaders**: Add to `Rendering_engine.shaders`

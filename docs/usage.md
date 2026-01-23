# Usage

## Advanced Examples

### Particle Simulation with Physics

```python
from sim_env.Physics_simulation_module import PhysicsEngine, Particle, PhysicsSettings
import numpy as np

# Configure physics
settings = PhysicsSettings()
settings.gravity = (0, -9.81, 0)
settings.collision_enabled = True

# Create engine
engine = PhysicsEngine(settings=settings)

# Create particles in a grid
for x in range(10):
    for y in range(10):
        particle = Particle(
            position=(x * 0.5, y * 0.5, 0),
            velocity=(np.random.randn() * 2, 0, 0),
            mass=1.0,
            radius=0.1
        )
        engine.add_particle(particle)

# Simulation loop
dt = 0.016  # 60 FPS
for _ in range(6000):  # 100 seconds
    engine.update(dt)
    particles = engine.get_particles()
    # Render or save particles...
```

### Fluid Dynamics

```python
from sim_env.fluid_dynamics import FluidSimulation

# Initialize fluid solver
fluid = FluidSimulation(
    grid_size=(64, 64, 64),
    cell_size=0.1,
    viscosity=0.01
)

# Add fluid source
fluid.add_source(position=(32, 32, 32), velocity=(10, 0, 0), density=1.0)

# Solve
for _ in range(1000):
    fluid.step(dt=0.016)
```

### Machine Learning Integration

```python
from sim_env.ml_pipeline import MLPipeline
from sim_env.neural_Physics import NeuralPhysicsEngine

# Train a neural physics model
pipeline = MLPipeline(model_type='physics_predictor')
pipeline.train(training_data, epochs=50)

# Use it in simulation
neural_engine = NeuralPhysicsEngine(model=pipeline.model)
for _ in range(1000):
    next_state = neural_engine.predict_next_state(current_state)
    current_state = next_state
```

### Quantum Simulation

```python
from sim_env.quantum_computing_Hybrid import QuantumCircuit

# Create and run quantum circuit
circuit = QuantumCircuit(num_qubits=5)
circuit.hadamard(0)
for i in range(4):
    circuit.cnot(i, i+1)

results = circuit.measure(shots=1000)
print(f"Measurement results: {results}")
```

### Real-Time Rendering

```python
from sim_env.Rendering_engine import Renderer, Scene
from sim_env.realtime_gui import RealtimeGUI

# Create renderer
renderer = Renderer(width=1920, height=1080)
scene = Scene()

# Launch GUI
gui = RealtimeGUI(renderer, scene)
gui.run()
```

## Tips & Tricks

1. **Performance**: Reduce particle count or use LOD rendering for large simulations
2. **Stability**: Use smaller time steps (dt < 0.02) for numerical stability
3. **Debugging**: Enable logging to track simulation state
4. **Visualization**: Use the built-in GUI for real-time monitoring
5. **Export**: Save simulation data for post-processing or ML training Guide

## CLI
```bash
frextech-sim --run
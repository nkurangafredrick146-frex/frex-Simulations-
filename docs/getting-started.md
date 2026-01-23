# Getting Started

## Installation

### Prerequisites
- Python 3.10+
- pip or conda
- OpenGL-compatible GPU (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nkurangafredrick146-code/frex-simulations.git
   cd frex-simulations
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running Your First Simulation

### Basic Example

```python
from sim_env.main import Enhancedfrexsimulations 
from sim_env.Physics_simulation_module import PhysicsEngine, Particle

# Initialize simulation
sim = Enhancedfrexsimulations()

# Create physics engine
physics = PhysicsEngine()

# Add particles
for i in range(100):
    particle = Particle(position=(i * 0.1, 0, 0), mass=1.0)
    physics.add_particle(particle)

# Run simulation
sim.run(physics)
```

## Configuration

Edit `config.json` to customize:

```json
{
  "window_width": 1920,
  "window_height": 1080,
  "max_particles": 50000,
  "enable_ml": true,
  "render_quality": "high"
}
```

## Next Steps

- Read the [Architecture](../architecture.md) documentation
- Explore [Physics Module](./modules/physics.md) for simulation details
- Check [Usage](../usage.md) for advanced examples

## Installation
```bash
pip install frextech_simulation

# Quantum Module

## Overview
The Quantum module provides quantum computing simulation, quantum physics visualization, and hybrid quantum-classical algorithms.

## Features

### Quantum Simulation
- **Qubit Systems**: Simulate quantum bits and entanglement
- **Quantum Gates**: Single and multi-qubit gate operations
- **Quantum Circuits**: Build and execute quantum circuits
- **State Tomography**: Measure and reconstruct quantum states

### Quantum Physics
- **Schrödinger Equation**: Wave function evolution
- **Quantum Fields**: Quantum field theory visualizations
- **Particle Interactions**: Photon, electron, and boson dynamics
- **Quantum Tunneling**: Probability-based particle behavior

### Hybrid Quantum-Classical
- **QAOA**: Quantum Approximate Optimization Algorithm
- **VQE**: Variational Quantum Eigensolver
- **Quantum Annealing**: Optimization via quantum tunneling
- **Parameterized Circuits**: Trainable quantum circuits

## Usage Example

```python
from sim_env.quantum_computing_Hybrid import QuantumCircuit, Qubit

# Initialize quantum circuit
circuit = QuantumCircuit(num_qubits=3)

# Apply gates
circuit.hadamard(0)  # Hadamard on qubit 0
circuit.cnot(0, 1)   # CNOT from qubit 0 to 1

# Measure
results = circuit.measure(shots=1000)
print(results)
```

## Quantum Physics Simulation

```python
from sim_env.Quantum_Physics_simulations import QuantumSystem

# Create quantum system
system = QuantumSystem(num_particles=10)
system.add_particle(name='electron', mass=9.109e-31)

# Evolve system
system.evolve(time_steps=1000, dt=1e-15)
```

## Performance
- Classical simulation up to 30 qubits
- Optimized state vector representation
- GPU acceleration for large systems
- Visualization of quantum states and measurements

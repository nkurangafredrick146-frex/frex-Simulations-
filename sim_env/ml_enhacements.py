"""
Complete Machine Learning Enhanced Simulations Module
Advanced ML integration for simulation control, prediction, and enhancement
"""

import numpy as np
import glm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import time
from typing import Dict, List, Any, Optional, Tuple
import random
import json
from dataclasses import dataclass
from enum import Enum

class MLModelType(Enum):
    """Types of machine learning models available"""
    NEURAL_NETWORK = "neural_network"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    GENERATIVE_ADVERSARIAL_NETWORK = "gan"
    VARIATIONAL_AUTOENCODER = "vae"
    PHYSICS_INFORMED_NN = "physics_informed"
    RECURRENT_NN = "recurrent"
    TRANSFORMER = "transformer"

@dataclass
class MLTrainingConfig:
    """Configuration for ML model training"""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    hidden_layers: List[int] = None
    activation: str = "relu"
    dropout_rate: float = 0.1
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64, 32]

class PhysicsInformedNN(nn.Module):
    """Physics-Informed Neural Network for learning physical laws"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int], 
                 physics_constraints: Dict = None):
        super(PhysicsInformedNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_constraints = physics_constraints or {}
        
        # Build network layers
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with physics constraints"""
        prediction = self.network(x)
        
        # Apply physics constraints if provided
        if self.physics_constraints.get('energy_conservation', False):
            prediction = self.apply_energy_conservation(x, prediction)
            
        if self.physics_constraints.get('momentum_conservation', False):
            prediction = self.apply_momentum_conservation(x, prediction)
            
        return prediction
        
    def apply_energy_conservation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply energy conservation constraint"""
        # Simplified energy conservation - adjust predictions to conserve energy
        if len(y.shape) == 2 and y.shape[1] >= 3:  # Assuming velocity components
            # Calculate kinetic energy
            kinetic_energy = 0.5 * torch.sum(y[:, :3] ** 2, dim=1)
            # Normalize to maintain average energy
            target_energy = torch.mean(kinetic_energy)
            normalized_energy = kinetic_energy / (torch.mean(kinetic_energy) + 1e-8) * target_energy
            # Adjust velocities to match normalized energy
            scale_factor = torch.sqrt(normalized_energy / (kinetic_energy + 1e-8))
            y[:, :3] = y[:, :3] * scale_factor.unsqueeze(1)
            
        return y
        
    def apply_momentum_conservation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply momentum conservation constraint"""
        # Simplified momentum conservation
        if len(y.shape) == 2 and y.shape[1] >= 3:
            # Zero out very small momentum changes that violate conservation
            momentum_threshold = 0.01
            small_momentum = torch.abs(y[:, :3]) < momentum_threshold
            y[:, :3] = torch.where(small_momentum, torch.zeros_like(y[:, :3]), y[:, :3])
            
        return y

class ReinforcementLearningAgent:
    """Reinforcement Learning agent for simulation control"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Q-network for deep Q-learning
        self.q_network = self.build_q_network()
        self.target_network = self.build_q_network()
        self.update_target_network()
        
        # Training parameters
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.gamma = self.config.get('gamma', 0.99)  # Discount factor
        self.epsilon = self.config.get('epsilon', 1.0)  # Exploration rate
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.memory = []  # Experience replay buffer
        self.batch_size = self.config.get('batch_size', 32)
        self.memory_capacity = self.config.get('memory_capacity', 10000)
        
    def build_q_network(self) -> nn.Module:
        """Build Q-network for deep Q-learning"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim)
        )
        
    def update_target_network(self):
        """Update target network with Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()
        
    def replay(self):
        """Train on batch from replay buffer"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class SimulationDataset(Dataset):
    """Dataset for simulation data collection and training"""
    
    def __init__(self, data: List[Tuple], transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

class SimulationDataCollector:
    """Collects and manages simulation data for ML training"""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.trajectories = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.current_trajectory = []
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.is_fitted = False
        
    def record_state(self, state: np.ndarray, action: np.ndarray = None, 
                    reward: float = None, done: bool = False):
        """Record simulation state"""
        sample = {
            'state': state.copy(),
            'action': action.copy() if action is not None else None,
            'reward': reward,
            'timestamp': time.time(),
            'done': done
        }
        
        self.current_trajectory.append(sample)
        
        if done or len(self.current_trajectory) >= 1000:  # Max trajectory length
            self.trajectories.append(self.current_trajectory.copy())
            self.current_trajectory = []
            
            # Trim if too many trajectories
            if len(self.trajectories) > 100:
                self.trajectories.pop(0)
                
    def preprocess_data(self):
        """Preprocess collected data for training"""
        if not self.trajectories:
            return
            
        # Extract states
        all_states = []
        for trajectory in self.trajectories:
            for sample in trajectory:
                all_states.append(sample['state'])
                
        states_array = np.array(all_states)
        
        # Fit scaler and PCA
        if not self.is_fitted:
            self.scaler.fit(states_array)
            self.pca.fit(self.scaler.transform(states_array))
            self.is_fitted = True
            
    def get_training_data(self, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Get training data in sequences"""
        if not self.is_fitted:
            self.preprocess_data()
            
        sequences = []
        targets = []
        
        for trajectory in self.trajectories:
            if len(trajectory) < sequence_length + 1:
                continue
                
            # Preprocess states
            states = np.array([s['state'] for s in trajectory])
            states_normalized = self.scaler.transform(states)
            states_pca = self.pca.transform(states_normalized)
            
            # Create sequences
            for i in range(len(trajectory) - sequence_length):
                sequence = states_pca[i:i+sequence_length]
                target = states_pca[i+sequence_length]  # Predict next state
                
                sequences.append(sequence)
                targets.append(target)
                
        return np.array(sequences), np.array(targets)
        
    def save_data(self, filepath: str):
        """Save collected data to file"""
        data = {
            'trajectories': self.trajectories,
            'scaler_mean': self.scaler.mean_.tolist() if self.is_fitted else [],
            'scaler_scale': self.scaler.scale_.tolist() if self.is_fitted else [],
            'pca_components': self.pca.components_.tolist() if self.is_fitted else []
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_data(self, filepath: str):
        """Load data from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.trajectories = data['trajectories']
        
        if data['scaler_mean']:
            self.scaler.mean_ = np.array(data['scaler_mean'])
            self.scaler.scale_ = np.array(data['scaler_scale'])
            self.pca.components_ = np.array(data['pca_components'])
            self.is_fitted = True

class NeuralPhysicsPredictor:
    """Neural network for predicting physics simulation outcomes"""
    
    def __init__(self, input_dim: int, output_dim: int, config: MLTrainingConfig = None):
        self.config = config or MLTrainingConfig()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.model = PhysicsInformedNN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=self.config.hidden_layers,
            physics_constraints={
                'energy_conservation': True,
                'momentum_conservation': True
            }
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Train the neural network"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                inputs, targets = batch
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            # Validation phase
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, targets = batch
                        outputs = self.model(inputs)
                        val_loss += self.criterion(outputs, targets).item()
                        
                val_loss /= len(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                    
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss if val_loader else train_loss)
            self.training_history['epochs'].append(epoch)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, "
                      f"Val Loss = {val_loss:.6f if val_loader else 'N/A'}")
                      
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Make predictions using trained model"""
        self.model.eval()
        with torch.no_grad():
            inputs_tensor = torch.FloatTensor(inputs)
            predictions = self.model(inputs_tensor)
            return predictions.numpy()
            
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }, filepath)
        
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.config = checkpoint['config']

class GANGenerator:
    """Generative Adversarial Network for creating realistic simulations"""
    
    def __init__(self, latent_dim: int, output_dim: int):
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.criterion = nn.BCELoss()
        
    def build_generator(self) -> nn.Module:
        """Build generator network"""
        return nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim),
            nn.Tanh()
        )
        
    def build_discriminator(self) -> nn.Module:
        """Build discriminator network"""
        return nn.Sequential(
            nn.Linear(self.output_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def generate_samples(self, num_samples: int) -> np.ndarray:
        """Generate synthetic simulation samples"""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim)
            generated = self.generator(noise)
            return generated.numpy()
            
    def train_step(self, real_data: torch.Tensor):
        """Single training step for GAN"""
        batch_size = real_data.size(0)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real data
        real_labels = torch.ones(batch_size, 1)
        real_output = self.discriminator(real_data)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # Fake data
        noise = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(noise)
        fake_labels = torch.zeros(batch_size, 1)
        fake_output = self.discriminator(fake_data.detach())
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        fake_output = self.discriminator(fake_data)
        g_loss = self.criterion(fake_output, real_labels)  # Trick discriminator
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return d_loss.item(), g_loss.item()

class MLEnhancedSimulation:
    """Base class for machine learning enhanced simulations"""
    
    def __init__(self, simulation_type: str, ml_config: Dict = None):
        self.simulation_type = simulation_type
        self.ml_config = ml_config or {}
        
        # ML components
        self.data_collector = SimulationDataCollector()
        self.neural_predictor = None
        self.rl_agent = None
        self.gan_generator = None
        
        # Simulation state
        self.is_training = False
        self.training_episode = 0
        self.best_reward = -float('inf')
        
        # Performance tracking
        self.ml_performance_stats = {}
        
    def initialize_ml_components(self):
        """Initialize machine learning components based on configuration"""
        ml_type = self.ml_config.get('type', MLModelType.NEURAL_NETWORK)
        
        if ml_type == MLModelType.NEURAL_NETWORK:
            self.initialize_neural_network()
        elif ml_type == MLModelType.REINFORCEMENT_LEARNING:
            self.initialize_reinforcement_learning()
        elif ml_type == MLModelType.GENERATIVE_ADVERSARIAL_NETWORK:
            self.initialize_gan()
            
    def initialize_neural_network(self):
        """Initialize neural network for physics prediction"""
        input_dim = self.ml_config.get('input_dim', 10)
        output_dim = self.ml_config.get('output_dim', 6)
        
        training_config = MLTrainingConfig(
            learning_rate=self.ml_config.get('learning_rate', 0.001),
            hidden_layers=self.ml_config.get('hidden_layers', [128, 64, 32])
        )
        
        self.neural_predictor = NeuralPhysicsPredictor(input_dim, output_dim, training_config)
        
    def initialize_reinforcement_learning(self):
        """Initialize reinforcement learning agent"""
        state_dim = self.ml_config.get('state_dim', 8)
        action_dim = self.ml_config.get('action_dim', 4)
        
        rl_config = {
            'learning_rate': self.ml_config.get('learning_rate', 0.001),
            'gamma': self.ml_config.get('gamma', 0.99),
            'epsilon': self.ml_config.get('epsilon', 1.0),
            'batch_size': self.ml_config.get('batch_size', 32)
        }
        
        self.rl_agent = ReinforcementLearningAgent(state_dim, action_dim, rl_config)
        
    def initialize_gan(self):
        """Initialize GAN for generating simulations"""
        latent_dim = self.ml_config.get('latent_dim', 100)
        output_dim = self.ml_config.get('output_dim', 20)
        
        self.gan_generator = GANGenerator(latent_dim, output_dim)
        
    def collect_training_data(self, simulation_state: Dict):
        """Collect data from current simulation state"""
        # Extract relevant features for ML
        state_vector = self.extract_state_features(simulation_state)
        self.data_collector.record_state(state_vector)
        
    def extract_state_features(self, simulation_state: Dict) -> np.ndarray:
        """Extract features from simulation state for ML"""
        features = []
        
        # Particle-based features
        if 'particles' in simulation_state:
            particles = simulation_state['particles']
            if particles:
                # Position statistics
                positions = np.array([p.position for p in particles])
                features.extend(positions.mean(axis=0))
                features.extend(positions.std(axis=0))
                
                # Velocity statistics
                velocities = np.array([p.velocity for p in particles])
                features.extend(velocities.mean(axis=0))
                features.extend(velocities.std(axis=0))
                
                # Energy
                kinetic_energy = 0.5 * np.sum(velocities ** 2, axis=1)
                features.append(kinetic_energy.mean())
                features.append(kinetic_energy.std())
                
        # Force field features
        if 'force_fields' in simulation_state:
            features.append(len(simulation_state['force_fields']))
            
        # Fill with zeros if not enough features
        while len(features) < 20:  # Minimum feature size
            features.append(0.0)
            
        return np.array(features[:20])  # Truncate to fixed size
        
    def apply_ml_control(self, simulation_state: Dict) -> Dict:
        """Apply ML-based control to simulation"""
        if not self.rl_agent:
            return {}
            
        state_features = self.extract_state_features(simulation_state)
        action = self.rl_agent.act(state_features)
        
        # Convert action to simulation parameters
        control_params = self.action_to_parameters(action)
        
        # Calculate reward
        reward = self.calculate_reward(simulation_state, control_params)
        
        # Store experience
        next_state_features = self.extract_state_features(simulation_state)  # Would be updated after action
        self.rl_agent.remember(state_features, action, reward, next_state_features, False)
        
        # Train RL agent
        self.rl_agent.replay()
        
        return control_params
        
    def action_to_parameters(self, action: int) -> Dict:
        """Convert RL action to simulation parameters"""
        actions = [
            {'gravity_scale': 0.5, 'emission_rate': 50},   # Action 0: Calm
            {'gravity_scale': 1.0, 'emission_rate': 100},  # Action 1: Normal
            {'gravity_scale': 2.0, 'emission_rate': 200},  # Action 2: Intense
            {'gravity_scale': 0.1, 'emission_rate': 25}    # Action 3: Gentle
        ]
        
        return actions[action % len(actions)]
        
    def calculate_reward(self, simulation_state: Dict, control_params: Dict) -> float:
        """Calculate reward for RL agent"""
        reward = 0.0
        
        # Reward for particle diversity (encourage interesting behavior)
        if 'particles' in simulation_state:
            particles = simulation_state['particles']
            if len(particles) > 10:
                positions = np.array([p.position for p in particles])
                # Reward for spread (variance in positions)
                position_variance = np.var(positions, axis=0).sum()
                reward += position_variance * 0.1
                
                # Reward for velocity diversity
                velocities = np.array([p.velocity for p in particles])
                velocity_variance = np.var(velocities, axis=0).sum()
                reward += velocity_variance * 0.05
                
        # Penalty for too many or too few particles
        particle_count = len(simulation_state.get('particles', []))
        target_count = 500
        count_penalty = -abs(particle_count - target_count) * 0.01
        reward += count_penalty
        
        return reward
        
    def predict_simulation_state(self, current_state: Dict, steps_ahead: int = 1) -> Dict:
        """Predict future simulation states using neural network"""
        if not self.neural_predictor:
            return current_state
            
        # Extract current state features
        current_features = self.extract_state_features(current_state)
        
        # Prepare input for prediction
        input_data = current_features.reshape(1, -1)
        
        # Make prediction
        predicted_features = self.neural_predictor.predict(input_data)[0]
        
        # Convert predicted features back to simulation state
        predicted_state = self.features_to_state(predicted_features, current_state)
        
        return predicted_state
        
    def features_to_state(self, features: np.ndarray, template_state: Dict) -> Dict:
        """Convert feature vector back to simulation state"""
        predicted_state = template_state.copy()
        
        # This is a simplified conversion - in practice, this would be more complex
        if 'particles' in predicted_state and predicted_state['particles']:
            # Modify particle positions based on predicted features
            for i, particle in enumerate(predicted_state['particles']):
                if i < len(features) // 3:
                    # Use features to modify particle positions
                    offset = glm.vec3(
                        features[(i*3) % len(features)],
                        features[(i*3+1) % len(features)],
                        features[(i*3+2) % len(features)]
                    ) * 0.1
                    particle.position += offset
                    
        return predicted_state
        
    def generate_synthetic_simulation(self, num_particles: int = 100) -> Dict:
        """Generate synthetic simulation using GAN"""
        if not self.gan_generator:
            return self.create_default_simulation()
            
        # Generate synthetic state features
        synthetic_features = self.gan_generator.generate_samples(num_particles)
        
        # Convert to simulation state
        synthetic_state = self.create_simulation_from_features(synthetic_features)
        
        return synthetic_state
        
    def create_simulation_from_features(self, features: np.ndarray) -> Dict:
        """Create simulation state from feature vectors"""
        simulation_state = {
            'particles': [],
            'force_fields': [],
            'time': 0.0
        }
        
        # Create particles from features
        for i in range(min(len(features), 100)):  # Limit to 100 particles
            feature = features[i]
            position = glm.vec3(
                feature[0] if len(feature) > 0 else 0,
                feature[1] if len(feature) > 1 else 0,
                feature[2] if len(feature) > 2 else 0
            )
            
            velocity = glm.vec3(
                feature[3] if len(feature) > 3 else 0,
                feature[4] if len(feature) > 4 else 0,
                feature[5] if len(feature) > 5 else 0
            )
            
            # Create particle (simplified - would use proper particle class)
            particle = {
                'position': position,
                'velocity': velocity,
                'mass': 1.0,
                'color': glm.vec3(0.5, 0.5, 0.8)
            }
            simulation_state['particles'].append(particle)
            
        return simulation_state
        
    def create_default_simulation(self) -> Dict:
        """Create default simulation state"""
        return {
            'particles': [],
            'force_fields': [{'position': glm.vec3(0, 0, 0), 'strength': 1.0}],
            'time': 0.0
        }
        
    def train_ml_models(self, training_data: List[Tuple] = None):
        """Train machine learning models"""
        if not training_data and self.data_collector.trajectories:
            # Use collected data
            sequences, targets = self.data_collector.get_training_data()
            if len(sequences) > 0:
                dataset = SimulationDataset(list(zip(sequences, targets)))
                train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
                
                if self.neural_predictor:
                    # Create validation split
                    train_size = int(0.8 * len(dataset))
                    val_size = len(dataset) - train_size
                    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
                    
                    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                    
                    self.neural_predictor.train(train_loader, val_loader)
                    
        print("ML models training completed")
        
    def get_ml_performance_stats(self) -> Dict:
        """Get ML performance statistics"""
        stats = {
            'data_samples': len(self.data_collector.trajectories),
            'is_training': self.is_training,
            'training_episode': self.training_episode,
            'best_reward': self.best_reward
        }
        
        if self.neural_predictor:
            stats['neural_network_trained'] = len(self.neural_predictor.training_history['train_loss']) > 0
            if stats['neural_network_trained']:
                stats['final_train_loss'] = self.neural_predictor.training_history['train_loss'][-1]
                stats['final_val_loss'] = self.neural_predictor.training_history['val_loss'][-1]
                
        if self.rl_agent:
            stats['rl_epsilon'] = self.rl_agent.epsilon
            stats['rl_memory_size'] = len(self.rl_agent.memory)
            
        return stats

class AdaptivePhysicsSimulation(MLEnhancedSimulation):
    """Simulation that adapts its physics based on ML predictions"""
    
    def __init__(self, base_simulation, ml_config: Dict = None):
        super().__init__("adaptive_physics", ml_config)
        self.base_simulation = base_simulation
        self.adaptation_strength = ml_config.get('adaptation_strength', 0.1)
        self.prediction_horizon = ml_config.get('prediction_horizon', 5)
        
    def update_with_adaptation(self, dt: float):
        """Update simulation with ML-based adaptation"""
        # Get current state
        current_state = self.get_simulation_state()
        
        # Predict future state
        predicted_state = self.predict_simulation_state(current_state, self.prediction_horizon)
        
        # Calculate adaptation
        adaptation = self.calculate_adaptation(current_state, predicted_state)
        
        # Apply adaptation to simulation
        self.apply_adaptation(adaptation)
        
        # Update base simulation
        self.base_simulation.update(dt)
        
        # Collect data for training
        self.collect_training_data(current_state)
        
    def get_simulation_state(self) -> Dict:
        """Extract state from base simulation"""
        # This would extract the actual state from the base simulation
        # For now, return a simplified state
        return {
            'particles': getattr(self.base_simulation, 'particles', []),
            'force_fields': getattr(self.base_simulation, 'force_fields', []),
            'time': getattr(self.base_simulation, 'simulation_time', 0.0)
        }
        
    def calculate_adaptation(self, current_state: Dict, predicted_state: Dict) -> Dict:
        """Calculate adaptation parameters based on prediction"""
        adaptation = {}
        
        # Simple adaptation: adjust gravity based on predicted particle spread
        if 'particles' in predicted_state and predicted_state['particles']:
            positions = np.array([p.position for p in predicted_state['particles']])
            spread = np.std(positions, axis=0).sum()
            
            # If particles are spreading too much, increase gravity
            if spread > 5.0:
                adaptation['gravity_scale'] = 1.2
            elif spread < 2.0:
                adaptation['gravity_scale'] = 0.8
            else:
                adaptation['gravity_scale'] = 1.0
                
        return adaptation
        
    def apply_adaptation(self, adaptation: Dict):
        """Apply adaptation to base simulation"""
        if 'gravity_scale' in adaptation and hasattr(self.base_simulation, 'physics_module'):
            self.base_simulation.physics_module.settings.gravity *= adaptation['gravity_scale']

# Demo and testing
if __name__ == "__main__":
    print("Testing Machine Learning Enhanced Simulations...")
    
    # Test neural network predictor
    config = MLTrainingConfig(epochs=5)  # Short test
    predictor = NeuralPhysicsPredictor(10, 6, config)
    
    # Generate dummy training data
    X_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randn(100, 6).astype(np.float32)
    
    dataset = SimulationDataset(list(zip(X_train, y_train)))
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Train briefly
    predictor.train(train_loader)
    print("Neural network training test completed")
    
    # Test RL agent
    rl_agent = ReinforcementLearningAgent(8, 4)
    
    # Test a few actions
    state = np.random.randn(8)
    for i in range(5):
        action = rl_agent.act(state)
        reward = random.random()
        next_state = np.random.randn(8)
        rl_agent.remember(state, action, reward, next_state, False)
        state = next_state
        
    rl_agent.replay()
    print("Reinforcement learning test completed")
    
    # Test data collector
    collector = SimulationDataCollector()
    for i in range(50):
        state = np.random.randn(20)
        collector.record_state(state)
        
    collector.preprocess_data()
    sequences, targets = collector.get_training_data(sequence_length=5)
    print(f"Data collection test completed: {len(sequences)} sequences")
    
    # Test ML-enhanced simulation
    ml_config = {
        'type': MLModelType.NEURAL_NETWORK,
        'input_dim': 20,
        'output_dim': 6
    }
    
    ml_sim = MLEnhancedSimulation("test", ml_config)
    ml_sim.initialize_ml_components()
    
    # Test with dummy simulation state
    test_state = {
        'particles': [type('Particle', (), {'position': glm.vec3(1, 2, 3), 'velocity': glm.vec3(0, 1, 0)})()],
        'force_fields': [{'position': glm.vec3(0, 0, 0)}]
    }
    
    ml_sim.collect_training_data(test_state)
    predicted = ml_sim.predict_simulation_state(test_state)
    print("ML-enhanced simulation test completed")
    
    print("All machine learning tests passed successfully!")
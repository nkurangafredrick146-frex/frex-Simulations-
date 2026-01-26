"""
Alignment trainer for training multimodal alignment models.
Supports various alignment strategies, metrics, and training loops.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from tqdm import tqdm
import json
import os

@dataclass
class AlignmentConfig:
    """Configuration for alignment training."""
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    
    # Alignment parameters
    alignment_method: str = 'contrastive'
    temperature: float = 0.07
    margin: float = 1.0
    similarity_type: str = 'cosine'
    
    # Model parameters
    embedding_dim: int = 512
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    
    # Training strategy
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    patience: int = 10  # For early stopping
    min_delta: float = 1e-4
    
    # Data handling
    validation_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    num_workers: int = 4
    
    # Logging and checkpointing
    log_interval: int = 10
    checkpoint_interval: int = 5
    checkpoint_dir: str = 'checkpoints'
    tensorboard_logging: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AlignmentConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save config to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'AlignmentConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

@dataclass
class AlignmentMetrics:
    """Metrics for alignment evaluation."""
    
    # Loss metrics
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    test_loss: Optional[float] = None
    
    # Alignment metrics
    alignment_score: List[float] = field(default_factory=list)
    val_alignment_score: List[float] = field(default_factory=list)
    
    # Retrieval metrics (if applicable)
    recall_at_k: Dict[int, List[float]] = field(default_factory=dict)
    precision_at_k: Dict[int, List[float]] = field(default_factory=dict)
    map_score: List[float] = field(default_factory=list)
    
    # Timing metrics
    epoch_times: List[float] = field(default_factory=list)
    total_time: float = 0.0
    
    # Convergence metrics
    learning_rates: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                current = getattr(self, key)
                if isinstance(current, list):
                    current.append(value)
                elif isinstance(current, dict):
                    for k, v in value.items():
                        if k not in current:
                            current[k] = []
                        current[k].append(v)
                else:
                    setattr(self, key, value)
            else:
                # Create new attribute
                setattr(self, key, value)
    
    def get_latest(self) -> Dict[str, Any]:
        """Get latest metric values."""
        latest = {}
        for key in self.__dict__.keys():
            if not key.startswith('_'):
                value = getattr(self, key)
                if isinstance(value, list) and value:
                    latest[key] = value[-1]
                elif isinstance(value, dict) and value:
                    latest[key] = {k: v[-1] if v else 0.0 for k, v in value.items()}
                else:
                    latest[key] = value
        return latest
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        metrics_dict = {}
        for key in self.__dict__.keys():
            if not key.startswith('_'):
                value = getattr(self, key)
                metrics_dict[key] = value
        return metrics_dict
    
    def save(self, path: str):
        """Save metrics to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'AlignmentMetrics':
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            metrics_dict = json.load(f)
        
        metrics = cls()
        for key, value in metrics_dict.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        
        return metrics

class AlignmentDataset(Dataset):
    """
    Dataset for multimodal alignment training.
    Supports multiple modalities and flexible pairing strategies.
    """
    
    def __init__(self,
                 data_dict: Dict[str, torch.Tensor],
                 modality_pairs: List[Tuple[str, str]],
                 labels: Optional[torch.Tensor] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize alignment dataset.
        
        Args:
            data_dict: Dictionary mapping modality names to data tensors
            modality_pairs: List of (modality_a, modality_b) pairs for alignment
            labels: Optional labels for supervised alignment
            metadata: Optional metadata for each sample
        """
        self.data_dict = data_dict
        self.modality_pairs = modality_pairs
        self.labels = labels
        self.metadata = metadata or {}
        
        # Validate data
        self._validate_data()
        
        # Get dataset size
        self.size = len(next(iter(data_dict.values())))
        
        # Create index mapping
        self.indices = list(range(self.size))
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Created AlignmentDataset with {self.size} samples")
        self.logger.info(f"Modality pairs: {modality_pairs}")
        self.logger.info(f"Available modalities: {list(data_dict.keys())}")
    
    def _validate_data(self):
        """Validate dataset consistency."""
        if not self.data_dict:
            raise ValueError("Data dictionary cannot be empty")
        
        # Check all modalities have same number of samples
        sizes = [len(data) for data in self.data_dict.values()]
        if len(set(sizes)) > 1:
            raise ValueError(
                f"All modalities must have same number of samples. Got sizes: {sizes}"
            )
        
        # Check modality pairs exist
        for modality_a, modality_b in self.modality_pairs:
            if modality_a not in self.data_dict:
                raise ValueError(f"Modality {modality_a} not found in data_dict")
            if modality_b not in self.data_dict:
                raise ValueError(f"Modality {modality_b} not found in data_dict")
        
        # Check labels if provided
        if self.labels is not None:
            if len(self.labels) != self.size:
                raise ValueError(
                    f"Labels size {len(self.labels)} doesn't match data size {self.size}"
                )
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = {'index': idx}
        
        # Add data from all modalities
        for modality, data in self.data_dict.items():
            sample[modality] = data[idx]
        
        # Add labels if available
        if self.labels is not None:
            sample['label'] = self.labels[idx]
        
        # Add metadata if available
        for key, metadata_list in self.metadata.items():
            if idx < len(metadata_list):
                sample[key] = metadata_list[idx]
        
        return sample
    
    def split(self,
              train_ratio: float = 0.8,
              val_ratio: float = 0.1,
              test_ratio: float = 0.1,
              random_seed: int = 42) -> Tuple['AlignmentDataset', ...]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        
        # Shuffle indices
        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(self.size)
        
        # Calculate split sizes
        train_size = int(train_ratio * self.size)
        val_size = int(val_ratio * self.size)
        test_size = self.size - train_size - val_size
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subsets
        train_data = self._create_subset(train_indices)
        val_data = self._create_subset(val_indices)
        test_data = self._create_subset(test_indices)
        
        return train_data, val_data, test_data
    
    def _create_subset(self, indices: np.ndarray) -> 'AlignmentDataset':
        """Create a subset of the dataset."""
        # Select data for each modality
        subset_data = {}
        for modality, data in self.data_dict.items():
            subset_data[modality] = data[indices]
        
        # Select labels if available
        subset_labels = None
        if self.labels is not None:
            subset_labels = self.labels[indices]
        
        # Select metadata if available
        subset_metadata = {}
        for key, metadata_list in self.metadata.items():
            if isinstance(metadata_list, list) and len(metadata_list) == self.size:
                subset_metadata[key] = [metadata_list[i] for i in indices]
        
        return AlignmentDataset(
            data_dict=subset_data,
            modality_pairs=self.modality_pairs,
            labels=subset_labels,
            metadata=subset_metadata
        )
    
    def get_modality_data(self, modality: str) -> torch.Tensor:
        """Get all data for a specific modality."""
        if modality not in self.data_dict:
            raise ValueError(f"Modality {modality} not found in dataset")
        return self.data_dict[modality]
    
    def get_pair_data(self,
                     modality_a: str,
                     modality_b: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get paired data for two modalities."""
        return self.get_modality_data(modality_a), self.get_modality_data(modality_b)

class AlignmentModel(nn.Module):
    """
    Base model for multimodal alignment.
    Projects different modalities into a shared embedding space.
    """
    
    def __init__(self,
                 input_dims: Dict[str, int],
                 output_dim: int = 512,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize alignment model.
        
        Args:
            input_dims: Dictionary mapping modality names to input dimensions
            output_dim: Dimension of shared embedding space
            hidden_dim: Dimension of hidden layers
            num_layers: Number of hidden layers per modality
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu', 'tanh', 'sigmoid')
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Activation function
        activation_fn = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leaky_relu': nn.LeakyReLU
        }.get(activation, nn.ReLU)
        
        # Create projection networks for each modality
        self.modality_projectors = nn.ModuleDict()
        
        for modality, input_dim in input_dims.items():
            # Build MLP for this modality
            layers = []
            current_dim = input_dim
            
            # Hidden layers
            for i in range(num_layers):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(activation_fn())
                layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
            
            # Output layer to shared space
            layers.append(nn.Linear(current_dim, output_dim))
            
            self.modality_projectors[modality] = nn.Sequential(*layers)
        
        # Optional: Cross-modal attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self,
                modality_inputs: Dict[str, torch.Tensor],
                apply_cross_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through alignment model.
        
        Args:
            modality_inputs: Dictionary mapping modality names to input tensors
            apply_cross_attention: Whether to apply cross-modal attention
            
        Returns:
            Dictionary of aligned embeddings for each modality
        """
        embeddings = {}
        
        # Project each modality to shared space
        for modality, projector in self.modality_projectors.items():
            if modality in modality_inputs:
                x = modality_inputs[modality]
                embeddings[modality] = projector(x)
        
        # Apply cross-modal attention if requested
        if apply_cross_attention and len(embeddings) > 1:
            embeddings = self._apply_cross_attention(embeddings)
        
        return embeddings
    
    def _apply_cross_attention(self,
                              embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply cross-modal attention between all modalities."""
        # Stack embeddings from all modalities
        modality_names = list(embeddings.keys())
        stacked_embeddings = torch.stack(
            [embeddings[name] for name in modality_names],
            dim=1  # [batch_size, num_modalities, embedding_dim]
        )
        
        # Apply self-attention (acts as cross-modal attention)
        attended, _ = self.cross_attention(
            stacked_embeddings,
            stacked_embeddings,
            stacked_embeddings
        )
        
        attended = self.layer_norm(attended)
        
        # Split back to modalities
        attended_embeddings = {}
        for i, name in enumerate(modality_names):
            attended_embeddings[name] = attended[:, i, :]
        
        return attended_embeddings
    
    def align_pair(self,
                  modality_a: str,
                  modality_b: str,
                  input_a: torch.Tensor,
                  input_b: torch.Tensor,
                  normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align a specific pair of modalities.
        
        Args:
            modality_a: Name of first modality
            modality_b: Name of second modality
            input_a: Input for modality A
            input_b: Input for modality B
            normalize: Whether to normalize embeddings
            
        Returns:
            Tuple of aligned embeddings
        """
        # Project to shared space
        emb_a = self.modality_projectors[modality_a](input_a)
        emb_b = self.modality_projectors[modality_b](input_b)
        
        # Normalize if requested
        if normalize:
            emb_a = F.normalize(emb_a, p=2, dim=1)
            emb_b = F.normalize(emb_b, p=2, dim=1)
        
        return emb_a, emb_b
    
    def get_similarity(self,
                      modality_a: str,
                      modality_b: str,
                      input_a: torch.Tensor,
                      input_b: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between two modalities.
        
        Args:
            modality_a: Name of first modality
            modality_b: Name of second modality
            input_a: Input for modality A
            input_b: Input for modality B
            
        Returns:
            Similarity matrix [batch_size, batch_size]
        """
        emb_a, emb_b = self.align_pair(modality_a, modality_b, input_a, input_b)
        similarity = torch.mm(emb_a, emb_b.t())
        return similarity
    
    def save(self, path: str):
        """Save model weights."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """Load model weights."""
        self.load_state_dict(torch.load(path))

class AlignmentTrainer:
    """
    Trainer for multimodal alignment models.
    Handles training loop, validation, checkpointing, and metrics.
    """
    
    def __init__(self,
                 config: Optional[AlignmentConfig] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize alignment trainer.
        
        Args:
            config: Training configuration
            device: Device to train on (None for auto-detection)
        """
        self.config = config or AlignmentConfig()
        
        # Device setup
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Components (will be initialized during training)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.metrics = AlignmentMetrics()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.early_stopping_counter = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"AlignmentTrainer initialized on device: {self.device}")
        
        # TensorBoard (optional)
        self.writer = None
        if self.config.tensorboard_logging:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(
                    log_dir=os.path.join(self.config.checkpoint_dir, 'logs')
                )
            except ImportError:
                self.logger.warning("TensorBoard not available, disabling logging")
                self.config.tensorboard_logging = False
    
    def train(self,
              dataset: AlignmentDataset,
              modality_pairs: List[Tuple[str, str]],
              config: Optional[AlignmentConfig] = None) -> Dict[str, Any]:
        """
        Train alignment model on dataset.
        
        Args:
            dataset: Alignment dataset
            modality_pairs: Modality pairs to align
            config: Optional training configuration
            
        Returns:
            Training results
        """
        # Update config if provided
        if config is not None:
            self.config = config
        
        # Start timing
        start_time = time.time()
        
        # Split dataset
        train_data, val_data, test_data = dataset.split(
            train_ratio=1.0 - self.config.validation_split - self.config.test_split,
            val_ratio=self.config.validation_split,
            test_ratio=self.config.test_split,
            random_seed=self.config.random_seed
        )
        
        self.logger.info(f"Dataset split: {len(train_data)} train, "
                        f"{len(val_data)} val, {len(test_data)} test")
        
        # Create data loaders
        train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_data,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Initialize model
        input_dims = {}
        for modality in dataset.data_dict.keys():
            input_dims[modality] = dataset.data_dict[modality].shape[1]
        
        self.model = AlignmentModel(
            input_dims=input_dims,
            output_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        # Initialize loss function based on alignment method
        self.loss_fn = self._get_loss_function()
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Initialize scheduler
        if self.config.use_scheduler:
            self.scheduler = self._get_scheduler()
        
        # Training loop
        self.logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self._train_epoch(train_loader, modality_pairs)
            
            # Validate
            val_metrics = self._validate(val_loader, modality_pairs)
            
            # Update metrics
            self.metrics.update(
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                alignment_score=train_metrics['alignment_score'],
                val_alignment_score=val_metrics['alignment_score'],
                learning_rates=self.optimizer.param_groups[0]['lr']
            )
            
            # Log metrics
            self._log_epoch(epoch, train_metrics, val_metrics)
            
            # Checkpointing
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch)
            
            # Early stopping check
            if self._check_early_stopping(val_metrics['loss']):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Training complete
        total_time = time.time() - start_time
        self.metrics.total_time = total_time
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("Loaded best model from checkpoint")
        
        # Test on test set
        test_metrics = self._test(test_loader, modality_pairs)
        self.metrics.test_loss = test_metrics['loss']
        
        # Final logging
        self._log_final_results(test_metrics)
        
        # Save final model and metrics
        self._save_final_results()
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
        
        return {
            'model': self.model,
            'metrics': self.metrics,
            'config': self.config,
            'test_metrics': test_metrics
        }
    
    def _train_epoch(self,
                     loader: DataLoader,
                     modality_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_alignment_score = 0.0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch + 1}", leave=False)
        
        for batch in pbar:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.model(batch)
            
            # Compute loss for each modality pair
            batch_loss = 0.0
            batch_alignment_score = 0.0
            
            for modality_a, modality_b in modality_pairs:
                if modality_a in embeddings and modality_b in embeddings:
                    # Get embeddings for this pair
                    emb_a = embeddings[modality_a]
                    emb_b = embeddings[modality_b]
                    
                    # Compute loss
                    if isinstance(self.loss_fn, nn.Module):
                        # For contrastive-based losses
                        loss = self.loss_fn(emb_a, emb_b)
                    else:
                        # For custom loss functions
                        loss = self.loss_fn(emb_a, emb_b)
                    
                    batch_loss += loss
                    
                    # Compute alignment score (cosine similarity)
                    alignment_score = F.cosine_similarity(emb_a, emb_b).mean().item()
                    batch_alignment_score += alignment_score
            
            # Average over pairs
            batch_loss /= len(modality_pairs)
            batch_alignment_score /= len(modality_pairs)
            
            # Backward pass
            batch_loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            total_loss += batch_loss.item()
            total_alignment_score += batch_alignment_score
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': batch_loss.item(),
                'align_score': batch_alignment_score
            })
            
            # Log gradient norms
            if num_batches % self.config.log_interval == 0:
                grad_norm = self._compute_gradient_norm()
                self.metrics.gradient_norms.append(grad_norm)
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {
            'loss': total_loss / num_batches,
            'alignment_score': total_alignment_score / num_batches
        }
    
    def _validate(self,
                  loader: DataLoader,
                  modality_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0.0
        total_alignment_score = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                embeddings = self.model(batch)
                
                # Compute metrics for each modality pair
                batch_loss = 0.0
                batch_alignment_score = 0.0
                
                for modality_a, modality_b in modality_pairs:
                    if modality_a in embeddings and modality_b in embeddings:
                        emb_a = embeddings[modality_a]
                        emb_b = embeddings[modality_b]
                        
                        # Compute loss
                        if isinstance(self.loss_fn, nn.Module):
                            loss = self.loss_fn(emb_a, emb_b)
                        else:
                            loss = self.loss_fn(emb_a, emb_b)
                        
                        batch_loss += loss.item()
                        
                        # Compute alignment score
                        alignment_score = F.cosine_similarity(emb_a, emb_b).mean().item()
                        batch_alignment_score += alignment_score
                
                # Average over pairs
                batch_loss /= len(modality_pairs)
                batch_alignment_score /= len(modality_pairs)
                
                # Update totals
                total_loss += batch_loss
                total_alignment_score += batch_alignment_score
                num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'alignment_score': total_alignment_score / num_batches
        }
    
    def _test(self,
              loader: DataLoader,
              modality_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """Test model."""
        return self._validate(loader, modality_pairs)  # Same as validation for now
    
    def _get_loss_function(self) -> Union[nn.Module, Callable]:
        """Get loss function based on alignment method."""
        from .contrastive_loss import (
            ContrastiveLoss, InfoNCELoss, TripletLoss,
            CircleLoss, SupConLoss, AlignUniformLoss
        )
        
        method = self.config.alignment_method.lower()
        
        if method == 'contrastive':
            return ContrastiveLoss(
                temperature=self.config.temperature,
                margin=self.config.margin,
                similarity_type=self.config.similarity_type
            )
        elif method == 'infonce':
            return InfoNCELoss(temperature=self.config.temperature)
        elif method == 'triplet':
            return TripletLoss(
                margin=self.config.margin,
                distance_type=self.config.similarity_type
            )
        elif method == 'circle':
            return CircleLoss(margin=self.config.margin)
        elif method == 'supcon':
            return SupConLoss(temperature=self.config.temperature)
        elif method == 'align_uniform':
            return AlignUniformLoss()
        else:
            # Default to contrastive
            self.logger.warning(
                f"Unknown alignment method '{method}', using contrastive"
            )
            return ContrastiveLoss(temperature=self.config.temperature)
    
    def _get_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler."""
        if self.config.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif self.config.scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        else:
            self.logger.warning(
                f"Unknown scheduler type '{self.config.scheduler_type}', "
                "disabling scheduler"
            )
            return None
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _compute_gradient_norm(self) -> float:
        """Compute gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _log_epoch(self,
                   epoch: int,
                   train_metrics: Dict[str, float],
                   val_metrics: Dict[str, float]):
        """Log epoch metrics."""
        self.logger.info(
            f"Epoch {epoch + 1}/{self.config.num_epochs}: "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Train Align: {train_metrics['alignment_score']:.4f}, "
            f"Val Align: {val_metrics['alignment_score']:.4f}, "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # TensorBoard logging
        if self.writer is not None:
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Alignment/train', train_metrics['alignment_score'], epoch)
            self.writer.add_scalar('Alignment/val', val_metrics['alignment_score'], epoch)
            self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)
    
    def _log_final_results(self, test_metrics: Dict[str, float]):
        """Log final training results."""
        self.logger.info("=" * 50)
        self.logger.info("Training Complete")
        self.logger.info(f"Total Time: {self.metrics.total_time:.2f}s")
        self.logger.info(f"Best Validation Loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        self.logger.info(f"Test Alignment Score: {test_metrics['alignment_score']:.4f}")
        self.logger.info("=" * 50)
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_epoch_{epoch + 1}.pt"
        )
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'metrics': self.metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _save_final_results(self):
        """Save final model and metrics."""
        # Save final model
        model_path = os.path.join(self.config.checkpoint_dir, 'final_model.pt')
        self.model.save(model_path)
        
        # Save metrics
        metrics_path = os.path.join(self.config.checkpoint_dir, 'metrics.json')
        self.metrics.save(metrics_path)
        
        # Save config
        config_path = os.path.join(self.config.checkpoint_dir, 'config.json')
        self.config.save(config_path)
        
        self.logger.info(f"Final model saved to {model_path}")
        self.logger.info(f"Metrics saved to {metrics_path}")
        self.logger.info(f"Config saved to {config_path}")
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping criteria met."""
        if val_loss < self.best_val_loss - self.config.min_delta:
            self.best_val_loss = val_loss
            self.best_model_state = self.model.state_dict().copy()
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.config.patience
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore state
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.metrics = checkpoint['metrics']
        
        # Reinitialize model if needed
        if self.model is None:
            # Need to recreate model with correct architecture
            # This requires the original config or model architecture info
            raise ValueError(
                "Model must be initialized before loading checkpoint. "
                "Call train() first or initialize model manually."
            )
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler is not None and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
        return checkpoint

# Import torch functional
import torch.nn.functional as F

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic dataset
    batch_size = 64
    embedding_dim = 128
    
    # Create random embeddings for three modalities
    text_data = torch.randn(1000, 300)  # 1000 samples, 300-dim text embeddings
    image_data = torch.randn(1000, 2048)  # 2048-dim image features
    audio_data = torch.randn(1000, 1024)  # 1024-dim audio features
    
    dataset = AlignmentDataset(
        data_dict={
            'text': text_data,
            'image': image_data,
            'audio': audio_data
        },
        modality_pairs=[
            ('text', 'image'),
            ('text', 'audio'),
            ('image', 'audio')
        ]
    )
    
    # Create trainer
    config = AlignmentConfig(
        batch_size=32,
        num_epochs=10,
        learning_rate=1e-4,
        embedding_dim=256,
        alignment_method='contrastive'
    )
    
    trainer = AlignmentTrainer(config=config)
    
    # Train
    results = trainer.train(
        dataset=dataset,
        modality_pairs=[('text', 'image'), ('text', 'audio')]
    )
    
    print(f"Training completed in {results['metrics'].total_time:.2f}s")
    print(f"Final test loss: {results['test_metrics']['loss']:.4f}")
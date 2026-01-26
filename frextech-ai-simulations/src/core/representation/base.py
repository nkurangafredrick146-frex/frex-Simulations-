"""
Base classes for 3D representations.

This module defines the base classes and interfaces that all 3D representations
must implement, ensuring a consistent API across different representation types.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Protocol
from dataclasses import dataclass, field, asdict
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import pickle
import warnings
import math
from datetime import datetime
import logging
import sys
import io
import traceback
from contextlib import contextmanager


# ============================================================================
# ENUMS AND TYPES
# ============================================================================

class RepresentationType(Enum):
    """Enumeration of representation types."""
    NERF = "nerf"
    GAUSSIAN = "gaussian"
    MESH = "mesh"
    VOXEL = "voxel"
    POINT_CLOUD = "point_cloud"
    SDF = "sdf"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class RenderMode(Enum):
    """Enumeration of rendering modes."""
    RGB = "rgb"
    DEPTH = "depth"
    NORMAL = "normal"
    SEMANTIC = "semantic"
    FEATURES = "features"
    ALPHA = "alpha"
    ALL = "all"


class SampleStrategy(Enum):
    """Enumeration of sampling strategies."""
    UNIFORM = "uniform"
    SURFACE = "surface"
    VOLUME = "volume"
    IMPORTANCE = "importance"
    RANDOM = "random"
    GRID = "grid"


class ExportFormat(Enum):
    """Enumeration of export formats."""
    PT = "pt"  # PyTorch checkpoint
    NPZ = "npz"  # NumPy compressed
    PLY = "ply"  # Polygon file format
    OBJ = "obj"  # Wavefront OBJ
    GLTF = "gltf"  # GL Transmission Format
    GLB = "glb"  # Binary GLTF
    STL = "stl"  # Stereolithography
    FBX = "fbx"  # Autodesk FBX
    USDZ = "usdz"  # Universal Scene Description
    PNG = "png"  # Image format
    EXR = "exr"  # High dynamic range image


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class RepresentationConfig:
    """Base configuration for all representations."""
    
    # Basic settings
    name: str = "unnamed_representation"
    type: RepresentationType = RepresentationType.UNKNOWN
    version: str = "1.0.0"
    
    # Device and precision
    device: Optional[str] = None  # 'cuda', 'cpu', or None for auto
    dtype: str = "float32"  # 'float32', 'float64', 'float16', 'bfloat16'
    use_amp: bool = False  # Automatic Mixed Precision
    
    # Bounding box and scene scale
    bounds: List[List[float]] = field(default_factory=lambda: [[-1, 1], [-1, 1], [-1, 1]])
    scene_scale: float = 1.0
    auto_scale: bool = True
    
    # Optimization
    learning_rate: float = 0.01
    optimizer: str = "adam"  # 'adam', 'sgd', 'rmsprop', 'adamw', 'lbfgs'
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9
    
    # Learning rate scheduling
    lr_scheduler: str = "none"  # 'none', 'cosine', 'step', 'plateau', 'exponential'
    lr_decay: float = 0.1
    lr_decay_steps: int = 1000
    warmup_steps: int = 0
    
    # Regularization
    regularization_weight: float = 0.01
    regularization_type: str = "l2"  # 'l1', 'l2', 'elastic', 'entropy', 'tv'
    
    # Training
    batch_size: int = 1
    num_iterations: int = 1000
    checkpoint_freq: int = 100
    log_freq: int = 10
    eval_freq: int = 100
    save_best: bool = True
    
    # Export settings
    export_format: ExportFormat = ExportFormat.PT
    export_quality: str = "high"  # 'low', 'medium', 'high', 'ultra'
    export_textures: bool = True
    export_compressed: bool = True
    
    # Memory management
    use_gradient_checkpointing: bool = False
    memory_limit_gb: Optional[float] = None
    chunk_size: int = 65536  # For batched processing
    
    # Quality settings
    antialiasing: bool = True
    super_sampling: int = 1
    tonemapping: bool = False
    
    # Debugging and logging
    debug: bool = False
    verbose: bool = True
    log_level: str = "INFO"
    profile: bool = False
    
    # Metadata
    author: str = "unknown"
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert string type to enum if needed
        if isinstance(self.type, str):
            self.type = RepresentationType(self.type.lower())
        
        # Validate bounds
        if len(self.bounds) != 3:
            raise ValueError(f"Bounds must have 3 dimensions, got {len(self.bounds)}")
        
        for i, (min_val, max_val) in enumerate(self.bounds):
            if min_val >= max_val:
                raise ValueError(f"Bound {i}: min ({min_val}) must be less than max ({max_val})")
        
        # Validate optimizer
        valid_optimizers = ['adam', 'sgd', 'rmsprop', 'adamw', 'lbfgs']
        if self.optimizer.lower() not in valid_optimizers:
            raise ValueError(f"Optimizer must be one of {valid_optimizers}, got {self.optimizer}")
        
        # Validate data type
        valid_dtypes = ['float32', 'float64', 'float16', 'bfloat16']
        if self.dtype not in valid_dtypes:
            raise ValueError(f"dtype must be one of {valid_dtypes}, got {self.dtype}")
        
        # Validate learning rate scheduler
        valid_schedulers = ['none', 'cosine', 'step', 'plateau', 'exponential', 'cyclic', 'onecycle']
        if self.lr_scheduler.lower() not in valid_schedulers:
            raise ValueError(f"LR scheduler must be one of {valid_schedulers}, got {self.lr_scheduler}")
        
        # Set created timestamp if not provided
        if not self.created:
            self.created = datetime.now().isoformat()
    
    @property
    def bounds_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get bounds as PyTorch tensors."""
        min_bounds = torch.tensor([b[0] for b in self.bounds], dtype=torch.float32)
        max_bounds = torch.tensor([b[1] for b in self.bounds], dtype=torch.float32)
        return min_bounds, max_bounds
    
    @property
    def center(self) -> torch.Tensor:
        """Get center of bounds."""
        min_bounds, max_bounds = self.bounds_tensor
        return (min_bounds + max_bounds) / 2
    
    @property
    def extent(self) -> torch.Tensor:
        """Get extent (size) of bounds."""
        min_bounds, max_bounds = self.bounds_tensor
        return max_bounds - min_bounds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = asdict(self)
        data['type'] = self.type.value
        data['export_format'] = self.export_format.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RepresentationConfig:
        """Create configuration from dictionary."""
        # Convert enum values back from strings
        if 'type' in data and isinstance(data['type'], str):
            data['type'] = RepresentationType(data['type'])
        
        if 'export_format' in data and isinstance(data['export_format'], str):
            data['export_format'] = ExportFormat(data['export_format'])
        
        return cls(**data)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> RepresentationConfig:
        """Load configuration from file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def copy(self, **kwargs) -> RepresentationConfig:
        """Create a copy of the configuration with optional updates."""
        data = self.to_dict()
        data.update(kwargs)
        return self.__class__.from_dict(data)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check learning rate
        if self.learning_rate <= 0:
            issues.append("Learning rate must be positive")
        
        # Check iterations
        if self.num_iterations <= 0:
            issues.append("Number of iterations must be positive")
        
        # Check batch size
        if self.batch_size <= 0:
            issues.append("Batch size must be positive")
        
        # Check regularization weight
        if self.regularization_weight < 0:
            issues.append("Regularization weight must be non-negative")
        
        # Check memory limit
        if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
            issues.append("Memory limit must be positive")
        
        return issues
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


# ============================================================================
# METRICS AND STATISTICS
# ============================================================================

@dataclass
class RepresentationMetrics:
    """Metrics for evaluating representation quality and performance."""
    
    # Quality metrics
    psnr: float = 0.0  # Peak Signal-to-Noise Ratio
    ssim: float = 0.0  # Structural Similarity Index
    lpips: float = 0.0  # Learned Perceptual Image Patch Similarity
    mse: float = 0.0   # Mean Squared Error
    psnr_db: float = 0.0  # PSNR in decibels
    
    # Geometric metrics
    chamfer_distance: float = 0.0
    hausdorff_distance: float = 0.0
    normal_consistency: float = 0.0
    f_score: float = 0.0
    iou: float = 0.0  # Intersection over Union
    
    # Performance metrics
    render_time_ms: float = 0.0
    render_fps: float = 0.0
    training_time_s: float = 0.0
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    
    # Compression metrics
    compression_ratio: float = 1.0
    bits_per_pixel: float = 0.0
    file_size_mb: float = 0.0
    
    # Statistical metrics
    num_parameters: int = 0
    parameter_memory_mb: float = 0.0
    sparsity: float = 0.0  # Percentage of zero/near-zero parameters
    
    # Perceptual metrics
    fid: float = 0.0  # Fréchet Inception Distance
    kid: float = 0.0  # Kernel Inception Distance
    inception_score: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def update(self, **kwargs) -> None:
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom_metrics[key] = value
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        result = asdict(self)
        # Remove custom_metrics from dict and merge it
        custom = result.pop('custom_metrics', {})
        result.update(custom)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> RepresentationMetrics:
        """Create metrics from dictionary."""
        # Separate known fields from custom metrics
        known_fields = {f.name for f in fields(cls) if f.name != 'custom_metrics'}
        known_data = {}
        custom_metrics = {}
        
        for key, value in data.items():
            if key in known_fields:
                known_data[key] = value
            else:
                custom_metrics[key] = value
        
        known_data['custom_metrics'] = custom_metrics
        return cls(**known_data)
    
    def compute_summary(self) -> Dict[str, Any]:
        """Compute a summary of metrics."""
        return {
            'quality': {
                'psnr': self.psnr,
                'ssim': self.ssim,
                'lpips': self.lpips,
            },
            'geometry': {
                'chamfer': self.chamfer_distance,
                'normal_consistency': self.normal_consistency,
                'iou': self.iou,
            },
            'performance': {
                'render_fps': self.render_fps,
                'memory_mb': self.memory_usage_mb,
                'inference_ms': self.inference_time_ms,
            },
            'compression': {
                'compression_ratio': self.compression_ratio,
                'file_size_mb': self.file_size_mb,
            }
        }
    
    def __add__(self, other: RepresentationMetrics) -> RepresentationMetrics:
        """Add two metrics objects (element-wise)."""
        if not isinstance(other, RepresentationMetrics):
            raise TypeError(f"Cannot add RepresentationMetrics and {type(other)}")
        
        result = RepresentationMetrics()
        
        # Add known fields
        for field in fields(self):
            if field.name == 'custom_metrics':
                continue
            value = getattr(self, field.name) + getattr(other, field.name)
            setattr(result, field.name, value)
        
        # Merge custom metrics
        all_keys = set(self.custom_metrics.keys()) | set(other.custom_metrics.keys())
        for key in all_keys:
            val1 = self.custom_metrics.get(key, 0.0)
            val2 = other.custom_metrics.get(key, 0.0)
            result.custom_metrics[key] = val1 + val2
        
        return result
    
    def __truediv__(self, scalar: float) -> RepresentationMetrics:
        """Divide metrics by a scalar."""
        result = RepresentationMetrics()
        
        # Divide known fields
        for field in fields(self):
            if field.name == 'custom_metrics':
                continue
            value = getattr(self, field.name) / scalar
            setattr(result, field.name, value)
        
        # Divide custom metrics
        for key, value in self.custom_metrics.items():
            result.custom_metrics[key] = value / scalar
        
        return result
    
    @property
    def average(self) -> Dict[str, float]:
        """Get average values for composite metrics (for backward compatibility)."""
        return self.to_dict()


# ============================================================================
# BASE REPRESENTATION CLASS
# ============================================================================

class BaseRepresentation(ABC, nn.Module):
    """
    Abstract base class for all 3D representations.
    
    This class defines the interface that all concrete representations
    must implement, ensuring consistency across different types.
    """
    
    def __init__(
        self,
        config: Optional[RepresentationConfig] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        """
        Initialize the representation.
        
        Args:
            config: Representation configuration
            device: PyTorch device (overrides config.device if provided)
            dtype: PyTorch data type (overrides config.dtype if provided)
            **kwargs: Additional arguments (will override config values)
        """
        super().__init__()
        
        # Create default config if not provided
        if config is None:
            config = RepresentationConfig(**kwargs)
        else:
            # Update config with kwargs
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config = RepresentationConfig.from_dict(config_dict)
        
        self.config = config
        
        # Determine device
        if device is not None:
            self._device = device
        elif config.device is not None:
            self._device = torch.device(config.device)
        else:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine data type
        if dtype is not None:
            self._dtype = dtype
        else:
            self._dtype = getattr(torch, config.dtype)
        
        # Initialize AMP (Automatic Mixed Precision)
        self._scaler = torch.cuda.amp.GradScaler() if config.use_amp and self._device.type == 'cuda' else None
        
        # Initialize state
        self._is_training = False
        self._is_initialized = False
        self._iteration = 0
        self._epoch = 0
        self._loss_history = []
        self._metric_history = []
        self._checkpoint_dir = None
        self._best_metric = float('inf')
        self._best_state = None
        
        # Setup logging
        self._setup_logging()
        
        # Initialize parameters (abstract method)
        self._init_parameters()
        
        # Move to device
        self.to(self._device, self._dtype)
        
        # Mark as initialized
        self._is_initialized = True
        self.logger.info(f"Initialized {self.__class__.__name__} on {self._device}")
    
    def _setup_logging(self) -> None:
        """Setup logging for the representation."""
        # Create logger
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Only add handlers if none exist
        if not self.logger.handlers:
            # Set level
            level = getattr(logging, self.config.log_level.upper(), logging.INFO)
            self.logger.setLevel(level)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # File handler if debug mode
            if self.config.debug:
                file_handler = logging.FileHandler(f"{self.config.name}_debug.log")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
    
    @abstractmethod
    def _init_parameters(self) -> None:
        """
        Initialize representation-specific parameters.
        
        This method should create all trainable parameters and buffers.
        """
        pass
    
    @abstractmethod
    def forward(
        self,
        *args,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the representation.
        
        Returns:
            Rendered output or dictionary of outputs
        """
        pass
    
    @abstractmethod
    def render(
        self,
        cameras: Any,
        resolution: Tuple[int, int] = (512, 512),
        mode: Union[str, RenderMode] = RenderMode.RGB,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Render the representation from given camera viewpoints.
        
        Args:
            cameras: Camera parameters (intrinsics, extrinsics)
            resolution: Output resolution (height, width)
            mode: Rendering mode (RGB, depth, normal, etc.)
            **kwargs: Additional rendering parameters
            
        Returns:
            Rendered output(s)
        """
        pass
    
    @abstractmethod
    def sample_points(
        self,
        num_points: int,
        strategy: Union[str, SampleStrategy] = SampleStrategy.UNIFORM,
        return_attributes: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample points from the representation.
        
        Args:
            num_points: Number of points to sample
            strategy: Sampling strategy
            return_attributes: Whether to return point attributes (color, normal, etc.)
            **kwargs: Additional sampling parameters
            
        Returns:
            Tuple of (positions [N, 3], attributes [N, C] or None)
        """
        pass
    
    @abstractmethod
    def query(
        self,
        points: torch.Tensor,
        view_dirs: Optional[torch.Tensor] = None,
        return_gradients: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Query the representation at specific points.
        
        Args:
            points: Query points [N, 3]
            view_dirs: View directions [N, 3] (optional)
            return_gradients: Whether to compute gradients
            **kwargs: Additional query parameters
            
        Returns:
            Dictionary with queried properties (density, color, normals, etc.)
        """
        pass
    
    @abstractmethod
    def get_type(self) -> RepresentationType:
        """
        Get the type of representation.
        
        Returns:
            Representation type
        """
        pass
    
    @abstractmethod
    def get_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the axis-aligned bounding box of the representation.
        
        Returns:
            Tuple of (min_bounds [3], max_bounds [3])
        """
        pass
    
    @abstractmethod
    def compute_metrics(
        self,
        reference: Any,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> RepresentationMetrics:
        """
        Compute quality and performance metrics.
        
        Args:
            reference: Reference data (images, point cloud, etc.)
            metrics: List of metrics to compute (None for all)
            **kwargs: Additional parameters
            
        Returns:
            RepresentationMetrics object
        """
        pass
    
    # ============================================================================
    # COMMON METHODS WITH DEFAULT IMPLEMENTATIONS
    # ============================================================================
    
    def train(self, mode: bool = True) -> BaseRepresentation:
        """
        Set the representation to training mode.
        
        Args:
            mode: True for training, False for evaluation
            
        Returns:
            self
        """
        self._is_training = mode
        result = super().train(mode)
        
        if mode:
            self.logger.debug("Switched to training mode")
        else:
            self.logger.debug("Switched to evaluation mode")
        
        return result
    
    def eval(self) -> BaseRepresentation:
        """
        Set the representation to evaluation mode.
        
        Returns:
            self
        """
        return self.train(False)
    
    @property
    def is_training(self) -> bool:
        """Check if representation is in training mode."""
        return self._is_training
    
    @property
    def is_initialized(self) -> bool:
        """Check if representation is initialized."""
        return self._is_initialized
    
    @property
    def device(self) -> torch.device:
        """Get the device of the representation."""
        return self._device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the representation."""
        return self._dtype
    
    @property
    def iteration(self) -> int:
        """Get current iteration count."""
        return self._iteration
    
    @property
    def epoch(self) -> int:
        """Get current epoch count."""
        return self._epoch
    
    def step(self) -> None:
        """Increment iteration counter."""
        self._iteration += 1
    
    def next_epoch(self) -> None:
        """Increment epoch counter."""
        self._epoch += 1
    
    def reset(self) -> None:
        """Reset the representation to initial state."""
        self._iteration = 0
        self._epoch = 0
        self._loss_history = []
        self._metric_history = []
        self._best_metric = float('inf')
        self._best_state = None
        
        # Re-initialize parameters
        self._init_parameters()
        self.to(self._device, self._dtype)
        
        self.logger.info("Reset representation to initial state")
    
    def create_optimizer(
        self,
        lr: Optional[float] = None,
        optimizer_type: Optional[str] = None,
        param_groups: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        Create an optimizer for the representation parameters.
        
        Args:
            lr: Learning rate (uses config.learning_rate if None)
            optimizer_type: Optimizer type (uses config.optimizer if None)
            param_groups: List of parameter groups with different settings
            **kwargs: Additional optimizer arguments
            
        Returns:
            PyTorch optimizer
        """
        if lr is None:
            lr = self.config.learning_rate
        
        if optimizer_type is None:
            optimizer_type = self.config.optimizer
        
        # Get parameters to optimize
        if param_groups is None:
            params = self.parameters()
        else:
            # Create parameter groups
            params = []
            for group in param_groups:
                # Filter parameters by name pattern
                group_params = []
                for name, param in self.named_parameters():
                    if 'pattern' in group and group['pattern'] in name:
                        group_params.append(param)
                    elif 'params' in group:
                        group_params.extend(group['params'])
                
                if group_params:
                    group_dict = {k: v for k, v in group.items() 
                                 if k not in ['pattern', 'params']}
                    group_dict['params'] = group_params
                    params.append(group_dict)
        
        # Create optimizer
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                params,
                lr=lr,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
                **kwargs
            )
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                **kwargs
            )
        elif optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                params,
                lr=lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                **kwargs
            )
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
                **kwargs
            )
        elif optimizer_type == 'lbfgs':
            optimizer = torch.optim.LBFGS(
                params,
                lr=lr,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        self.logger.info(f"Created {optimizer_type} optimizer with lr={lr}")
        return optimizer
    
    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: Optional[str] = None,
        **kwargs
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create a learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            scheduler_type: Type of scheduler (uses config.lr_scheduler if None)
            **kwargs: Additional scheduler arguments
            
        Returns:
            Learning rate scheduler or None
        """
        if scheduler_type is None:
            scheduler_type = self.config.lr_scheduler
        
        if scheduler_type.lower() == 'none':
            return None
        
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.num_iterations,
                eta_min=lr * 0.01,  # Minimum learning rate
                **kwargs
            )
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', self.config.lr_decay_steps),
                gamma=kwargs.get('gamma', self.config.lr_decay),
                **kwargs
            )
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=kwargs.get('patience', 10),
                factor=kwargs.get('factor', self.config.lr_decay),
                verbose=self.config.verbose,
                **kwargs
            )
        elif scheduler_type == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', self.config.lr_decay),
                **kwargs
            )
        elif scheduler_type == "cyclic":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=kwargs.get('base_lr', lr * 0.1),
                max_lr=lr,
                step_size_up=kwargs.get('step_size_up', 2000),
                **kwargs
            )
        elif scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=self.config.num_iterations,
                **kwargs
            )
        else:
            self.logger.warning(f"Unknown scheduler type: {scheduler_type}")
            return None
        
        self.logger.info(f"Created {scheduler_type} learning rate scheduler")
        return scheduler
    
    def compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        loss_type: str = "mse",
        weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss between prediction and target.
        
        Args:
            prediction: Predicted values
            target: Target values
            loss_type: Type of loss
            weights: Per-element weights
            **kwargs: Additional loss parameters
            
        Returns:
            Loss tensor
        """
        loss_type = loss_type.lower()
        
        if loss_type == "mse":
            loss = F.mse_loss(prediction, target, reduction='mean')
        elif loss_type == "l1":
            loss = F.l1_loss(prediction, target, reduction='mean')
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(prediction, target, reduction='mean')
        elif loss_type == "chamfer":
            loss = self._compute_chamfer_distance(prediction, target)
        elif loss_type == "ssim":
            loss = 1.0 - self._compute_ssim(prediction, target)
        elif loss_type == "perceptual":
            loss = self._compute_perceptual_loss(prediction, target)
        elif loss_type == "binary_cross_entropy":
            loss = F.binary_cross_entropy(prediction, target, reduction='mean')
        elif loss_type == "cross_entropy":
            loss = F.cross_entropy(prediction, target, reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Apply weights if provided
        if weights is not None:
            loss = (loss * weights).mean()
        
        # Add regularization
        if self.config.regularization_weight > 0:
            reg_loss = self._compute_regularization()
            loss = loss + self.config.regularization_weight * reg_loss
        
        return loss
    
    def _compute_chamfer_distance(
        self,
        points1: torch.Tensor,
        points2: torch.Tensor,
        bidirectional: bool = True
    ) -> torch.Tensor:
        """
        Compute Chamfer distance between two point clouds.
        
        Args:
            points1: First point cloud [N, D]
            points2: Second point cloud [M, D]
            bidirectional: Whether to compute both directions
            
        Returns:
            Chamfer distance
        """
        # Compute pairwise distances
        dist_matrix = torch.cdist(points1, points2, p=2)  # [N, M]
        
        if bidirectional:
            # Chamfer distance: mean of min distances in both directions
            dist1 = dist_matrix.min(dim=1)[0].mean()
            dist2 = dist_matrix.min(dim=0)[0].mean()
            return (dist1 + dist2) / 2
        else:
            # One-directional Chamfer distance
            return dist_matrix.min(dim=1)[0].mean()
    
    def _compute_ssim(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03
    ) -> torch.Tensor:
        """
        Compute Structural Similarity Index (SSIM).
        
        Args:
            img1: First image [C, H, W] or [B, C, H, W]
            img2: Second image [C, H, W] or [B, C, H, W]
            window_size: Size of Gaussian window
            sigma: Standard deviation for Gaussian window
            k1, k2: Stability constants
            
        Returns:
            SSIM value
        """
        # Ensure batch dimension
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        
        C = img1.size(1)
        
        # Create 1D Gaussian
        coords = torch.arange(window_size, dtype=torch.float32, device=img1.device)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        # Create 2D Gaussian window
        g = g.unsqueeze(1) * g.unsqueeze(0)  # [window_size, window_size]
        window = g.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)  # [C, 1, window_size, window_size]
        
        # Compute means
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=C)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=C)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Compute variances
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=C) - mu1_mu2
        
        # SSIM constants
        L = 1.0  # Dynamic range (assuming normalized images)
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim_map.mean()
    
    def _compute_perceptual_loss(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute perceptual loss using pre-trained VGG network.
        
        Args:
            img1: First image [C, H, W] or [B, C, H, W]
            img2: Second image [C, H, W] or [B, C, H, W]
            
        Returns:
            Perceptual loss
        """
        # This is a placeholder implementation
        # In practice, you would use a pre-trained network like VGG
        # For now, use a combination of L1 and gradient loss
        
        # L1 loss
        l1_loss = F.l1_loss(img1, img2)
        
        # Gradient loss (encourage similar edges)
        def image_gradients(image):
            # Compute image gradients
            dy = image[..., 1:, :] - image[..., :-1, :]
            dx = image[..., :, 1:] - image[..., :, :-1]
            return dy, dx
        
        dy1, dx1 = image_gradients(img1)
        dy2, dx2 = image_gradients(img2)
        
        grad_loss = F.l1_loss(dy1, dy2) + F.l1_loss(dx1, dx2)
        
        return l1_loss + 0.1 * grad_loss
    
    def _compute_regularization(self) -> torch.Tensor:
        """
        Compute regularization loss.
        
        Returns:
            Regularization loss
        """
        reg_type = self.config.regularization_type.lower()
        
        if reg_type == "l2":
            # L2 regularization on all parameters
            total_reg = 0.0
            for param in self.parameters():
                if param.requires_grad:
                    total_reg += torch.norm(param, p=2)
            return total_reg
        
        elif reg_type == "l1":
            # L1 regularization on all parameters
            total_reg = 0.0
            for param in self.parameters():
                if param.requires_grad:
                    total_reg += torch.norm(param, p=1)
            return total_reg
        
        elif reg_type == "elastic":
            # Elastic net regularization (combination of L1 and L2)
            l1_reg = 0.0
            l2_reg = 0.0
            for param in self.parameters():
                if param.requires_grad:
                    l1_reg += torch.norm(param, p=1)
                    l2_reg += torch.norm(param, p=2)
            return 0.5 * l1_reg + 0.5 * l2_reg
        
        elif reg_type == "entropy":
            # Entropy regularization (for distributions)
            return self._compute_entropy_regularization()
        
        elif reg_type == "tv":
            # Total variation regularization
            return self._compute_total_variation()
        
        else:
            warnings.warn(f"Unknown regularization type: {reg_type}")
            return torch.tensor(0.0, device=self.device)
    
    def _compute_entropy_regularization(self) -> torch.Tensor:
        """Compute entropy regularization (placeholder)."""
        # This should be implemented by subclasses if needed
        return torch.tensor(0.0, device=self.device)
    
    def _compute_total_variation(self) -> torch.Tensor:
        """Compute total variation regularization (placeholder)."""
        # This should be implemented by subclasses if needed
        return torch.tensor(0.0, device=self.device)
    
    def update_loss_history(self, loss: float) -> None:
        """Update loss history with new loss value."""
        self._loss_history.append(loss)
        
        # Keep only recent history to avoid memory issues
        max_history = 10000
        if len(self._loss_history) > max_history:
            self._loss_history = self._loss_history[-max_history:]
    
    def update_metric_history(self, metrics: RepresentationMetrics) -> None:
        """Update metric history with new metrics."""
        self._metric_history.append(metrics)
        
        # Keep only recent history
        max_history = 1000
        if len(self._metric_history) > max_history:
            self._metric_history = self._metric_history[-max_history:]
        
        # Update best metric if applicable
        if self.config.save_best:
            current_metric = metrics.mse  # Use MSE as default metric
            if current_metric < self._best_metric:
                self._best_metric = current_metric
                self._best_state = self.state_dict().copy()
                self.logger.info(f"New best metric: {current_metric:.6f}")
    
    def get_loss_history(self, window: Optional[int] = None) -> List[float]:
        """
        Get loss history.
        
        Args:
            window: Number of recent losses to return (None for all)
            
        Returns:
            List of loss values
        """
        if window is None:
            return self._loss_history.copy()
        else:
            return self._loss_history[-window:]
    
    def get_metric_history(self, window: Optional[int] = None) -> List[RepresentationMetrics]:
        """
        Get metric history.
        
        Args:
            window: Number of recent metrics to return (None for all)
            
        Returns:
            List of RepresentationMetrics objects
        """
        if window is None:
            return self._metric_history.copy()
        else:
            return self._metric_history[-window:]
    
    def get_average_loss(self, window: int = 100) -> float:
        """
        Get average loss over recent iterations.
        
        Args:
            window: Number of iterations to average over
            
        Returns:
            Average loss
        """
        recent_losses = self.get_loss_history(window)
        if not recent_losses:
            return 0.0
        return sum(recent_losses) / len(recent_losses)
    
    def get_average_metrics(self, window: int = 100) -> RepresentationMetrics:
        """
        Get average metrics over recent iterations.
        
        Args:
            window: Number of iterations to average over
            
        Returns:
            Average metrics
        """
        recent_metrics = self.get_metric_history(window)
        if not recent_metrics:
            return RepresentationMetrics()
        
        # Sum all metrics
        total = recent_metrics[0]
        for metric in recent_metrics[1:]:
            total = total + metric
        
        # Divide by count
        return total / len(recent_metrics)
    
    def set_checkpoint_dir(self, directory: Union[str, Path]) -> None:
        """
        Set directory for saving checkpoints.
        
        Args:
            directory: Directory path
        """
        self._checkpoint_dir = Path(directory)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Checkpoint directory set to {self._checkpoint_dir}")
    
    def save_checkpoint(
        self,
        filepath: Optional[Union[str, Path]] = None,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        save_best: bool = False,
        **kwargs
    ) -> Path:
        """
        Save representation checkpoint.
        
        Args:
            filepath: Path to save checkpoint (auto-generated if None)
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
            optimizer: Optimizer to save
            scheduler: Scheduler to save
            save_best: Whether to save the best state (if available)
            **kwargs: Additional data to save
            
        Returns:
            Path to saved checkpoint
        """
        if filepath is None:
            if self._checkpoint_dir is None:
                self._checkpoint_dir = Path("checkpoints") / self.config.name
                self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Use best state if requested
            state_dict = self._best_state if save_best and self._best_state is not None else self.state_dict()
            suffix = "_best" if save_best else ""
            filename = f"{self.config.name}_iter{self._iteration:06d}{suffix}.pt"
            filepath = self._checkpoint_dir / filename
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'iteration': self._iteration,
            'epoch': self._epoch,
            'config': self.config.to_dict(),
            'state_dict': self.state_dict() if not save_best or self._best_state is None else self._best_state,
            'loss_history': self._loss_history,
            'metric_history': [m.to_dict() for m in self._metric_history],
            'best_metric': self._best_metric,
            'type': self.get_type().value,
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        if save_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if save_scheduler and scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if self._scaler is not None:
            checkpoint['scaler_state_dict'] = self._scaler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint to {filepath}")
        
        return filepath
    
    def load_checkpoint(
        self,
        filepath: Union[str, Path],
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        strict: bool = True,
        reset_iteration: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load representation checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            strict: Whether to strictly enforce that keys match
            reset_iteration: Whether to reset iteration count
            **kwargs: Additional loading options
            
        Returns:
            Dictionary with loaded data
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load state dict
        self.load_state_dict(checkpoint['state_dict'], strict=strict)
        
        # Load iteration and epoch
        if not reset_iteration:
            self._iteration = checkpoint.get('iteration', 0)
            self._epoch = checkpoint.get('epoch', 0)
        
        # Load loss and metric history
        self._loss_history = checkpoint.get('loss_history', [])
        metric_history = checkpoint.get('metric_history', [])
        self._metric_history = [RepresentationMetrics.from_dict(m) for m in metric_history]
        
        # Load best metric
        self._best_metric = checkpoint.get('best_metric', float('inf'))
        
        # Load optimizer state if requested
        if load_optimizer and optimizer is not None:
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("Loaded optimizer state")
            else:
                self.logger.warning("Optimizer state not found in checkpoint")
        
        # Load scheduler state if requested
        if load_scheduler and scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("Loaded scheduler state")
        
        # Load scaler state if available
        if self._scaler is not None and 'scaler_state_dict' in checkpoint:
            self._scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from {filepath} (iteration {self._iteration})")
        
        return checkpoint
    
    def export(
        self,
        filepath: Union[str, Path],
        format: Optional[Union[str, ExportFormat]] = None,
        quality: Optional[str] = None,
        overwrite: bool = True,
        **kwargs
    ) -> Path:
        """
        Export representation to file.
        
        Args:
            filepath: Path to export file
            format: Export format (uses config.export_format if None)
            quality: Export quality (uses config.export_quality if None)
            overwrite: Whether to overwrite existing file
            **kwargs: Additional export parameters
            
        Returns:
            Path to exported file
        """
        if format is None:
            format = self.config.export_format
        elif isinstance(format, str):
            format = ExportFormat(format.lower())
        
        if quality is None:
            quality = self.config.export_quality
        
        filepath = Path(filepath)
        
        # Check if file exists
        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {filepath}")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect format from extension
        if format == ExportFormat.PT and filepath.suffix.lower() != '.pt':
            # Try to infer format from extension
            ext = filepath.suffix.lower()[1:]  # Remove dot
            try:
                format = ExportFormat(ext)
            except ValueError:
                pass  # Keep PT format
        
        # Call format-specific export method
        export_method_name = f"_export_{format.value}"
        export_method = getattr(self, export_method_name, None)
        
        if export_method is not None:
            exported_path = export_method(filepath, quality=quality, **kwargs)
            self.logger.info(f"Exported to {format.value}: {exported_path}")
            return exported_path
        
        # Try default export methods
        try:
            if format in [ExportFormat.PLY, ExportFormat.OBJ, ExportFormat.GLTF, ExportFormat.GLB]:
                return self._export_mesh(filepath, format, quality, **kwargs)
            elif format == ExportFormat.NPZ:
                return self._export_npz(filepath, **kwargs)
            elif format == ExportFormat.PT:
                return self._export_pt(filepath, **kwargs)
            else:
                raise NotImplementedError(f"Export format not supported: {format}")
        except NotImplementedError:
            # Fall back to PyTorch format
            self.logger.warning(f"Format {format} not implemented, falling back to PyTorch format")
            return self._export_pt(filepath.with_suffix('.pt'), **kwargs)
    
    def _export_pt(self, filepath: Path, **kwargs) -> Path:
        """Export as PyTorch checkpoint."""
        checkpoint = {
            'type': self.get_type().value,
            'config': self.config.to_dict(),
            'state_dict': self.state_dict(),
            'iteration': self._iteration,
            'epoch': self._epoch,
            'loss_history': self._loss_history,
            'metric_history': [m.to_dict() for m in self._metric_history],
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
        }
        
        torch.save(checkpoint, filepath)
        return filepath
    
    def _export_npz(self, filepath: Path, **kwargs) -> Path:
        """Export as NPZ file."""
        data = self.to_numpy()
        
        # Add metadata
        data['_metadata'] = np.array([json.dumps({
            'type': self.get_type().value,
            'config': self.config.to_dict(),
            'iteration': self._iteration,
            'version': '1.0.0',
        })])
        
        np.savez_compressed(filepath, **data)
        return filepath
    
    def _export_mesh(
        self,
        filepath: Path,
        format: ExportFormat,
        quality: str = "high",
        **kwargs
    ) -> Path:
        """Export as mesh file (placeholder)."""
        # This should be implemented by subclasses that support mesh export
        raise NotImplementedError(f"Mesh export not implemented for {self.get_type()}")
    
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """
        Convert representation to numpy arrays.
        
        Returns:
            Dictionary of numpy arrays
        """
        data = {}
        for name, param in self.named_parameters():
            data[name] = param.detach().cpu().numpy()
        
        for name, buffer in self.named_buffers():
            data[name] = buffer.detach().cpu().numpy()
        
        return data
    
    def from_numpy(self, data: Dict[str, np.ndarray]) -> None:
        """
        Load representation from numpy arrays.
        
        Args:
            data: Dictionary of numpy arrays
        """
        # Load parameters
        for name, param in self.named_parameters():
            if name in data:
                param.data = torch.from_numpy(data[name]).to(self.device)
            else:
                self.logger.warning(f"Parameter {name} not found in numpy data")
        
        # Load buffers
        for name, buffer in self.named_buffers():
            if name in data:
                buffer.data = torch.from_numpy(data[name]).to(self.device)
    
    def clone(self, deep: bool = True) -> BaseRepresentation:
        """
        Create a copy of the representation.
        
        Args:
            deep: Whether to create a deep copy (copy parameters) or shallow (share parameters)
            
        Returns:
            Cloned representation
        """
        # Create new instance with same config
        new_rep = self.__class__(self.config.copy(), device=self.device)
        
        if deep:
            # Copy state dict
            new_rep.load_state_dict(self.state_dict())
        else:
            # Share parameters (not recommended for training)
            for (name1, param1), (name2, param2) in zip(
                self.named_parameters(), new_rep.named_parameters()
            ):
                param2.data = param1.data
        
        # Copy other state
        new_rep._iteration = self._iteration
        new_rep._epoch = self._epoch
        new_rep._loss_history = self._loss_history.copy()
        new_rep._metric_history = self._metric_history.copy()
        new_rep._best_metric = self._best_metric
        new_rep._is_training = self._is_training
        
        return new_rep
    
    def get_memory_usage(self, detailed: bool = False) -> Dict[str, float]:
        """
        Get memory usage statistics.
        
        Args:
            detailed: Whether to return detailed breakdown
            
        Returns:
            Dictionary with memory usage in MB
        """
        # Parameter memory
        param_memory = 0.0
        param_counts = {}
        
        for name, param in self.named_parameters():
            size = param.numel() * param.element_size()
            param_memory += size
            if detailed:
                param_counts[name] = {
                    'size_mb': size / (1024 * 1024),
                    'numel': param.numel(),
                    'shape': list(param.shape),
                    'dtype': str(param.dtype),
                }
        
        # Gradient memory (if training)
        grad_memory = 0.0
        if self._is_training:
            for param in self.parameters():
                if param.grad is not None:
                    grad_memory += param.grad.numel() * param.grad.element_size()
        
        # Buffers memory
        buffer_memory = 0.0
        buffer_counts = {}
        
        for name, buffer in self.buffers():
            size = buffer.numel() * buffer.element_size()
            buffer_memory += size
            if detailed:
                buffer_counts[name] = {
                    'size_mb': size / (1024 * 1024),
                    'numel': buffer.numel(),
                    'shape': list(buffer.shape),
                    'dtype': str(buffer.dtype),
                }
        
        # Convert to MB
        mb = 1024 * 1024
        
        result = {
            'parameters_mb': param_memory / mb,
            'gradients_mb': grad_memory / mb,
            'buffers_mb': buffer_memory / mb,
            'total_mb': (param_memory + grad_memory + buffer_memory) / mb,
        }
        
        if detailed:
            result['parameter_details'] = param_counts
            result['buffer_details'] = buffer_counts
        
        return result
    
    def get_complexity(self) -> Dict[str, Any]:
        """
        Get complexity metrics for the representation.
        
        Returns:
            Dictionary with complexity metrics
        """
        # Count parameters
        total_params = 0
        trainable_params = 0
        
        param_details = {}
        
        for name, param in self.named_parameters():
            numel = param.numel()
            total_params += numel
            if param.requires_grad:
                trainable_params += numel
            
            param_details[name] = {
                'numel': numel,
                'shape': list(param.shape),
                'trainable': param.requires_grad,
                'dtype': str(param.dtype),
            }
        
        # Count modules
        modules = list(self.modules())
        num_modules = len(modules)
        
        # Count layers by type
        layer_counts = {}
        for module in modules:
            module_type = module.__class__.__name__
            layer_counts[module_type] = layer_counts.get(module_type, 0) + 1
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'number_of_modules': num_modules,
            'layer_counts': layer_counts,
            'parameter_details': param_details,
            'representation_type': self.get_type().value,
            'device': str(self.device),
            'dtype': str(self.dtype),
        }
    
    def visualize(
        self,
        viewer: Optional[Any] = None,
        save_path: Optional[Union[str, Path]] = None,
        interactive: bool = True,
        **kwargs
    ) -> Any:
        """
        Visualize the representation.
        
        Args:
            viewer: Optional viewer object
            save_path: Path to save visualization
            interactive: Whether to create interactive visualization
            **kwargs: Visualization parameters
            
        Returns:
            Visualization object
        """
        # This is a placeholder implementation
        # Subclasses should implement their own visualization
        
        self.logger.warning(f"Visualization not implemented for {self.get_type()}")
        
        # Try to create a simple point cloud visualization
        try:
            # Sample points for visualization
            points, colors = self.sample_points(
                num_points=10000,
                strategy=SampleStrategy.SURFACE,
                return_attributes=True
            )
            
            if points is not None:
                # Convert to numpy
                points_np = points.cpu().numpy()
                colors_np = colors.cpu().numpy() if colors is not None else None
                
                # Create simple visualization
                if viewer is None and interactive:
                    # Try to use matplotlib
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D
                    
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    if colors_np is not None:
                        ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], 
                                  c=colors_np[:, :3], s=1, alpha=0.5)
                    else:
                        ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], 
                                  s=1, alpha=0.5)
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title(f'{self.get_type().value} Visualization')
                    
                    if save_path is not None:
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    
                    if interactive:
                        plt.show()
                    
                    return fig
                
        except Exception as e:
            self.logger.error(f"Failed to create visualization: {e}")
        
        return viewer
    
    def check_bounds(self) -> List[str]:
        """
        Check if parameters are within valid bounds.
        
        Returns:
            List of issues found
        """
        issues = []
        
        # Check each parameter
        for name, param in self.named_parameters():
            # Check for NaN
            if torch.isnan(param).any():
                issues.append(f"NaN values in parameter: {name}")
            
            # Check for Inf
            if torch.isinf(param).any():
                issues.append(f"Infinite values in parameter: {name}")
            
            # Check magnitude
            param_abs = param.abs()
            max_val = param_abs.max().item()
            if max_val > 1e6:
                issues.append(f"Very large values in parameter: {name} (max: {max_val:.2e})")
            
            # Check for zero gradients (if training)
            if self._is_training and param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm == 0:
                    issues.append(f"Zero gradient in parameter: {name}")
        
        return issues
    
    def get_config(self) -> RepresentationConfig:
        """
        Get the representation configuration.
        
        Returns:
            Configuration object
        """
        return self.config.copy()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the representation state.
        
        Returns:
            State summary dictionary
        """
        return {
            'type': self.get_type().value,
            'iteration': self._iteration,
            'epoch': self._epoch,
            'is_training': self._is_training,
            'is_initialized': self._is_initialized,
            'device': str(self.device),
            'dtype': str(self.dtype),
            'loss_history_length': len(self._loss_history),
            'metric_history_length': len(self._metric_history),
            'best_metric': self._best_metric,
            'config': self.config.to_dict(),
        }
    
    def profile(
        self,
        num_iterations: int = 100,
        warmup: int = 10,
        **kwargs
    ) -> Dict[str, float]:
        """
        Profile the representation's performance.
        
        Args:
            num_iterations: Number of iterations to profile
            warmup: Number of warmup iterations
            **kwargs: Additional profiling parameters
            
        Returns:
            Performance metrics
        """
        import time
        
        # Set to eval mode
        was_training = self.training
        self.eval()
        
        # Warmup
        for _ in range(warmup):
            _ = self.forward()
        
        # Profile forward pass
        forward_times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = self.forward()
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end = time.perf_counter()
                forward_times.append((end - start) * 1000)  # Convert to ms
        
        # Profile render if implemented
        render_times = []
        try:
            # Create dummy camera
            dummy_camera = self._create_dummy_camera()
            
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = self.render(dummy_camera, resolution=(256, 256))
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                end = time.perf_counter()
                render_times.append((end - start) * 1000)
        except Exception as e:
            self.logger.warning(f"Could not profile render: {e}")
        
        # Restore training mode
        if was_training:
            self.train()
        
        # Compute statistics
        def compute_stats(times):
            if not times:
                return {}
            times_np = np.array(times)
            return {
                'mean_ms': float(times_np.mean()),
                'std_ms': float(times_np.std()),
                'min_ms': float(times_np.min()),
                'max_ms': float(times_np.max()),
                'p50_ms': float(np.percentile(times_np, 50)),
                'p95_ms': float(np.percentile(times_np, 95)),
                'p99_ms': float(np.percentile(times_np, 99)),
            }
        
        result = {
            'forward': compute_stats(forward_times),
            'render': compute_stats(render_times) if render_times else {},
            'iterations': num_iterations,
            'warmup': warmup,
            'device': str(self.device),
        }
        
        # Add FPS
        if forward_times:
            result['forward_fps'] = 1000.0 / result['forward']['mean_ms']
        
        if render_times:
            result['render_fps'] = 1000.0 / result['render']['mean_ms']
        
        return result
    
    def _create_dummy_camera(self):
        """Create a dummy camera for profiling."""
        # This should be implemented by subclasses
        return None
    
    @contextmanager
    def gradient_checkpointing(self):
        """Context manager for gradient checkpointing."""
        if self.config.use_gradient_checkpointing:
            # This would enable gradient checkpointing for compatible modules
            # Implementation depends on the specific representation
            yield
        else:
            yield
    
    def optimize_for_inference(self) -> None:
        """Optimize the representation for inference."""
        # Set to eval mode
        self.eval()
        
        # Disable gradient computation
        for param in self.parameters():
            param.requires_grad = False
        
        # Optional: Convert to half precision if using GPU
        if self.device.type == 'cuda' and self.dtype == torch.float32:
            try:
                self.half()
                self.logger.info("Converted to half precision for inference")
            except Exception as e:
                self.logger.warning(f"Could not convert to half precision: {e}")
        
        # Optional: Apply other optimizations
        if hasattr(self, '_optimize_for_inference'):
            self._optimize_for_inference()
        
        self.logger.info("Optimized for inference")
    
    def __str__(self) -> str:
        """String representation."""
        complexity = self.get_complexity()
        return (
            f"{self.__class__.__name__}(\n"
            f"  type: {self.get_type().value}\n"
            f"  device: {self.device}\n"
            f"  dtype: {self.dtype}\n"
            f"  parameters: {complexity['total_parameters']:,}\n"
            f"  trainable: {complexity['trainable_parameters']:,}\n"
            f"  iteration: {self._iteration}\n"
            f"  epoch: {self._epoch}\n"
            f"  training: {self._is_training}\n"
            f")"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return str(self)
    
    def summary(self, detailed: bool = False) -> str:
        """
        Generate a summary of the representation.
        
        Args:
            detailed: Whether to include detailed information
            
        Returns:
            Summary string
        """
        import io
        import sys
        
        # Redirect stdout to capture model summary
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            # Print model structure
            print(str(self))
            print("\n" + "="*80)
            
            # Print parameter summary
            total_params = 0
            trainable_params = 0
            
            print("\nParameters:")
            print("-"*80)
            for name, param in self.named_parameters():
                numel = param.numel()
                total_params += numel
                if param.requires_grad:
                    trainable_params += numel
                
                if detailed:
                    print(f"{name:50} {str(list(param.shape)):20} {numel:12,} "
                          f"{'trainable' if param.requires_grad else 'frozen':12} "
                          f"{str(param.dtype):10}")
            
            print("-"*80)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Non-trainable parameters: {total_params - trainable_params:,}")
            
            # Print memory usage
            mem = self.get_memory_usage()
            print(f"\nMemory usage: {mem['total_mb']:.2f} MB")
            print(f"  Parameters: {mem['parameters_mb']:.2f} MB")
            print(f"  Buffers: {mem['buffers_mb']:.2f} MB")
            if self._is_training:
                print(f"  Gradients: {mem['gradients_mb']:.2f} MB")
            
            # Print state summary
            state = self.get_state_summary()
            print(f"\nState:")
            print(f"  Iteration: {state['iteration']}")
            print(f"  Epoch: {state['epoch']}")
            print(f"  Mode: {'Training' if state['is_training'] else 'Evaluation'}")
            print(f"  Best metric: {state['best_metric']:.6f}")
            
            # Print config summary
            if detailed:
                print(f"\nConfiguration:")
                for key, value in state['config'].items():
                    print(f"  {key}: {value}")
            
            # Get the captured output
            output = sys.stdout.getvalue()
        finally:
            # Restore stdout
            sys.stdout = old_stdout
        
        return output


# ============================================================================
# REPRESENTATION REGISTRY
# ============================================================================

class RepresentationRegistry:
    """
    Registry for representation types.
    
    This allows dynamic registration and discovery of representation types.
    """
    
    _registry: Dict[str, type] = {}
    _aliases: Dict[str, str] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        representation_class: type,
        aliases: Optional[List[str]] = None
    ) -> None:
        """
        Register a representation class.
        
        Args:
            name: Representation type name
            representation_class: Representation class
            aliases: Optional list of aliases for this type
        """
        if not issubclass(representation_class, BaseRepresentation):
            raise TypeError(
                f"Representation class must inherit from BaseRepresentation, "
                f"got {representation_class}"
            )
        
        cls._registry[name] = representation_class
        
        # Register aliases
        if aliases:
            for alias in aliases:
                cls._aliases[alias] = name
        
        logging.getLogger(__name__).info(f"Registered representation type: {name}")
    
    @classmethod
    def get(cls, name: str) -> type:
        """
        Get representation class by name.
        
        Args:
            name: Representation type name or alias
            
        Returns:
            Representation class
        """
        # Check aliases first
        actual_name = cls._aliases.get(name, name)
        
        if actual_name not in cls._registry:
            raise KeyError(f"Representation type not registered: {name}")
        
        return cls._registry[actual_name]
    
    @classmethod
    def list(cls) -> List[str]:
        """
        List all registered representation types.
        
        Returns:
            List of registered type names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[RepresentationConfig] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> BaseRepresentation:
        """
        Create a representation instance by type name.
        
        Args:
            name: Representation type name or alias
            config: Representation configuration
            device: PyTorch device
            **kwargs: Additional arguments
            
        Returns:
            Representation instance
        """
        representation_class = cls.get(name)
        return representation_class(config, device, **kwargs)
    
    @classmethod
    def get_registered_types(cls) -> Dict[str, type]:
        """
        Get all registered representation types.
        
        Returns:
            Dictionary of type names to classes
        """
        return cls._registry.copy()
    
    @classmethod
    def get_aliases(cls) -> Dict[str, str]:
        """
        Get all registered aliases.
        
        Returns:
            Dictionary of aliases to actual type names
        """
        return cls._aliases.copy()
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a representation type.
        
        Args:
            name: Representation type name
        """
        if name in cls._registry:
            del cls._registry[name]
            
            # Remove aliases pointing to this name
            aliases_to_remove = [k for k, v in cls._aliases.items() if v == name]
            for alias in aliases_to_remove:
                del cls._aliases[alias]
            
            logging.getLogger(__name__).info(f"Unregistered representation type: {name}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_representation(
    rep_type: Union[str, RepresentationType],
    config: Optional[Union[Dict[str, Any], RepresentationConfig]] = None,
    device: Optional[torch.device] = None,
    **kwargs
) -> BaseRepresentation:
    """
    Create a representation (convenience wrapper).
    
    Args:
        rep_type: Representation type
        config: Configuration dictionary or object
        device: PyTorch device
        **kwargs: Additional arguments
        
    Returns:
        Representation instance
    """
    # Convert string to RepresentationType if needed
    if isinstance(rep_type, str):
        rep_type = RepresentationType(rep_type.lower())
    
    # Convert config dict to RepresentationConfig if needed
    if isinstance(config, dict):
        config = RepresentationConfig(**{**config, **kwargs})
    elif config is None:
        config = RepresentationConfig(**kwargs)
    
    # Set type in config
    config.type = rep_type
    
    # Create representation
    return RepresentationRegistry.create(rep_type.value, config, device)


def save_representation(
    representation: BaseRepresentation,
    filepath: Union[str, Path],
    format: Optional[Union[str, ExportFormat]] = None,
    **kwargs
) -> None:
    """
    Save representation to file (convenience function).
    
    Args:
        representation: Representation to save
        filepath: Path to save file
        format: Export format
        **kwargs: Additional save parameters
    """
    filepath = Path(filepath)
    
    # Auto-detect format from extension
    if format is None:
        ext = filepath.suffix.lower()[1:]  # Remove dot
        try:
            format = ExportFormat(ext)
        except ValueError:
            format = ExportFormat.PT  # Default to PyTorch format
    
    representation.export(filepath, format=format, **kwargs)


def load_representation(
    filepath: Union[str, Path],
    device: Optional[torch.device] = None,
    **kwargs
) -> BaseRepresentation:
    """
    Load representation from file (convenience function).
    
    Args:
        filepath: Path to file
        device: PyTorch device
        **kwargs: Additional load parameters
        
    Returns:
        Loaded representation
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Check file format
    if filepath.suffix.lower() == '.pt':
        # PyTorch checkpoint
        checkpoint = torch.load(filepath, map_location=device)
        
        # Get representation type
        rep_type_str = checkpoint.get('type', 'gaussian')
        rep_type = RepresentationType(rep_type_str)
        
        # Get config
        config_dict = checkpoint.get('config', {})
        config = RepresentationConfig.from_dict(config_dict)
        
        # Create representation
        representation = create_representation(rep_type, config, device)
        
        # Load state
        representation.load_checkpoint(filepath, **kwargs)
        
        return representation
    
    else:
        # Other formats would require format-specific loading
        raise NotImplementedError(f"Loading from {filepath.suffix} format not implemented")


def validate_representation(
    representation: BaseRepresentation,
    checks: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate representation for common issues.
    
    Args:
        representation: Representation to validate
        checks: List of checks to perform
        
    Returns:
        Validation results
    """
    if checks is None:
        checks = ['nan', 'inf', 'bounds', 'memory', 'gradients', 'config']
    
    results = {
        'valid': True,
        'issues': [],
        'warnings': [],
        'stats': {},
    }
    
    # Check for NaN and Inf values
    if 'nan' in checks or 'inf' in checks:
        for name, param in representation.named_parameters():
            if torch.isnan(param).any():
                results['valid'] = False
                results['issues'].append(f"NaN values in parameter: {name}")
            
            if torch.isinf(param).any():
                results['valid'] = False
                results['issues'].append(f"Infinite values in parameter: {name}")
    
    # Check parameter bounds
    if 'bounds' in checks:
        bounds_issues = representation.check_bounds()
        if bounds_issues:
            results['issues'].extend(bounds_issues)
            results['valid'] = False
    
    # Check gradients
    if 'gradients' in checks and representation.is_training:
        for name, param in representation.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm == 0:
                    results['warnings'].append(f"Zero gradient in parameter: {name}")
                elif grad_norm > 1e6:
                    results['warnings'].append(f"Exploding gradient in parameter: {name} (norm: {grad_norm:.2e})")
    
    # Check memory usage
    if 'memory' in checks:
        mem = representation.get_memory_usage()
        results['stats']['memory'] = mem
        
        if mem['total_mb'] > 1000:  # > 1GB
            results['warnings'].append(f"Large memory usage: {mem['total_mb']:.2f} MB")
    
    # Check configuration
    if 'config' in checks:
        config_issues = representation.config.validate()
        if config_issues:
            results['issues'].extend(config_issues)
            results['valid'] = False
    
    # Get complexity stats
    complexity = representation.get_complexity()
    results['stats']['complexity'] = complexity
    
    if complexity['total_parameters'] > 1e8:  # > 100M parameters
        results['warnings'].append(f"Large model: {complexity['total_parameters']:,} parameters")
    
    return results


# ============================================================================
# BUILTIN TYPE REGISTRATION
# ============================================================================

def register_builtin_types():
    """Register built-in representation types."""
    try:
        from .nerf.nerf_model import NeRFModel
        RepresentationRegistry.register('nerf', NeRFModel, aliases=['neural_radiance_field', 'radiance_field'])
    except ImportError as e:
        logging.getLogger(__name__).warning(f"Could not register NeRFModel: {e}")
    
    try:
        from .gaussian_splatting.gaussian_model import GaussianModel
        RepresentationRegistry.register('gaussian', GaussianModel, aliases=['gaussian_splatting', '3dgs', 'splatting'])
    except ImportError as e:
        logging.getLogger(__name__).warning(f"Could not register GaussianModel: {e}")
    
    try:
        from .mesh.mesh_generator import MeshGenerator
        RepresentationRegistry.register('mesh', MeshGenerator, aliases=['triangle_mesh', 'polygon_mesh'])
    except ImportError as e:
        logging.getLogger(__name__).warning(f"Could not register MeshGenerator: {e}")
    
    try:
        from .voxel.voxel_grid import VoxelGrid
        RepresentationRegistry.register('voxel', VoxelGrid, aliases=['voxel_grid', 'volume'])
    except ImportError as e:
        logging.getLogger(__name__).warning(f"Could not register VoxelGrid: {e}")
    
    try:
        # Register hybrid representation if it exists
        from .. import HybridRepresentation
        RepresentationRegistry.register('hybrid', HybridRepresentation)
    except ImportError:
        pass


# Auto-register when module is imported
register_builtin_types()


# ============================================================================
# TYPE GUARDS AND UTILITIES
# ============================================================================

def is_representation(obj: Any) -> bool:
    """Check if an object is a representation."""
    return isinstance(obj, BaseRepresentation)


def get_representation_type(obj: Any) -> Optional[RepresentationType]:
    """Get the type of a representation object."""
    if is_representation(obj):
        return obj.get_type()
    return None


def representation_to_dict(representation: BaseRepresentation) -> Dict[str, Any]:
    """Convert representation to dictionary."""
    return {
        'type': representation.get_type().value,
        'config': representation.get_config().to_dict(),
        'state': representation.get_state_summary(),
        'complexity': representation.get_complexity(),
    }


# Export key classes and functions
__all__ = [
    # Enums
    'RepresentationType',
    'RenderMode',
    'SampleStrategy',
    'ExportFormat',
    
    # Configuration
    'RepresentationConfig',
    'RepresentationMetrics',
    
    # Base class
    'BaseRepresentation',
    
    # Registry
    'RepresentationRegistry',
    
    # Convenience functions
    'create_representation',
    'save_representation',
    'load_representation',
    'validate_representation',
    
    # Type guards
    'is_representation',
    'get_representation_type',
    'representation_to_dict',
]
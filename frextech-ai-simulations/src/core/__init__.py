"""
Core AI components for FrexTech AI Simulations.

This module contains the fundamental building blocks for world generation,
multimodal understanding, 3D representation, and interactive agents.
"""

from src.core.world_model import WorldModel
from src.core.multimodal import (
    TextEncoder,
    VisionEncoder,
    VideoEncoder,
    CrossAttentionFusion,
)
from src.core.representation import (
    NeuralRadianceField,
    GaussianSplattingModel,
    MeshGenerator,
    VoxelGrid,
)
from src.core.agents import (
    PhysicsSimulator,
    NavigationAgent,
    InteractionHandler,
)

# Version information
version = "1.0.0"
author = "FrexTech AI Research Team"
email = "research@frextech-sim.com"

# Export core classes
__all__ = [
# World Model
"WorldModel",

# Multimodal
"TextEncoder",
"VisionEncoder",
"VideoEncoder",
"CrossAttentionFusion",

# 3D Representation
"NeuralRadianceField",
"GaussianSplattingModel",
"MeshGenerator",
"VoxelGrid",

# Agents
"PhysicsSimulator",
"NavigationAgent",
"InteractionHandler",

]

# Configuration defaults
DEFAULT_CONFIG = {
"world_model": {
"latent_dim": 768,
"num_layers": 24,
"num_heads": 16,
"hidden_dim": 3072,
"dropout": 0.1,
},
"multimodal": {
"text_encoder": "clip",
"vision_encoder": "vit_large",
"fusion_method": "cross_attention",
},
"representation": {
"default_method": "gaussian_splatting",
"nerf": {
"num_samples_per_ray": 128,
"num_importance_samples": 0,
"positional_encoding_freqs": 10,
},
"gaussian": {
"max_gaussians": 1000000,
"sh_degree": 3,
},
"mesh": {
"marching_cubes_resolution": 256,
"simplify_target_faces": 50000,
},
},
"training": {
"batch_size": 32,
"learning_rate": 1e-4,
"warmup_steps": 10000,
"gradient_clip": 1.0,
},
}

def get_config(component: str = None):
    """
    Get configuration for a specific component or all components.

    Args:
        component: Optional component name (e.g., 'world_model', 'multimodal')

    Returns:
        Configuration dictionary for the specified component or all components
    """
    if component:
        return DEFAULT_CONFIG.get(component, {})
    return DEFAULT_CONFIG

def validate_config(config: dict, component: str) -> bool:
    """
    Validate configuration for a component.

    Args:
        config: Configuration dictionary to validate
        component: Component name

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    defaults = DEFAULT_CONFIG.get(component, {})

    # Check required parameters
    required_params = {
        "world_model": ["latent_dim", "num_layers", "num_heads"],
        "multimodal": ["text_encoder", "vision_encoder"],
        "representation": ["default_method"],
        "training": ["batch_size", "learning_rate"],
    }

    if component in required_params:
        for param in required_params[component]:
            if param not in config:
                raise ValueError(f"Missing required parameter '{param}' for {component}")

    # Validate parameter types and ranges
    validators = {
        "latent_dim": lambda x: isinstance(x, int) and x > 0,
        "num_layers": lambda x: isinstance(x, int) and 1 <= x <= 100,
        "num_heads": lambda x: isinstance(x, int) and 1 <= x <= 64,
        "batch_size": lambda x: isinstance(x, int) and x > 0,
        "learning_rate": lambda x: isinstance(x, (int, float)) and x > 0,
        "dropout": lambda x: isinstance(x, (int, float)) and 0 <= x < 1,
    }

    for param, value in config.items():
        if param in validators:
            if not validators[param](value):
                raise ValueError(f"Invalid value for parameter '{param}': {value}")

    return True

def list_available_models() -> dict:
    """
    List all available pre-trained models and their configurations.

    Returns:
        Dictionary mapping model names to their configurations
    """
    return {
        "world_model": {
            "v1.0": {
                "description": "Initial world model with 8B parameters",
                "parameters": 8_000_000_000,
                "latent_dim": 768,
                "num_layers": 24,
                "pretrained_on": ["LAION-5B", "Objaverse", "ScanNet"],
                "format": "gaussian",
                "download_url": "https://models.frextech-sim.com/world_model_v1.0.pt",
            },
            "v1.1": {
                "description": "Improved world model with diffusion transformer",
                "parameters": 12_000_000_000,
                "latent_dim": 1024,
                "num_layers": 32,
                "pretrained_on": ["LAION-5B", "Objaverse", "ScanNet", "CO3D"],
                "format": ["gaussian", "nerf"],
                "download_url": "https://models.frextech-sim.com/world_model_v1.1.pt",
            },
        },
        "encoders": {
            "clip_vit_large": {
                "description": "CLIP ViT-L/14",
                "vision_dim": 1024,
                "text_dim": 768,
                "parameters": 427_000_000,
                "download_url": "https://models.frextech-sim.com/clip_vit_large.pt",
            },
            "dinov2_vit_large": {
                "description": "DINOv2 ViT-L/14",
                "vision_dim": 1024,
                "parameters": 300_000_000,
                "download_url": "https://models.frextech-sim.com/dinov2_vit_large.pt",
            },
            "bert_base": {
                "description": "BERT Base Uncased",
                "text_dim": 768,
                "parameters": 110_000_000,
                "download_url": "https://models.frextech-sim.com/bert_base.pt",
            },
        },
    }


def get_model_info(model_name: str, version: str = None) -> dict:
    """
    Get detailed information about a specific model.

    Args:
        model_name: Name of the model (e.g., 'world_model', 'clip_vit_large')
        version: Optional version specifier

    Returns:
        Dictionary with model information

    Raises:
        ValueError: If model is not found
    """
    models = list_available_models()

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(models.keys())}")

    model_info = models[model_name]

    if version:
        if version not in model_info:
            raise ValueError(
                f"Version '{version}' not found for model '{model_name}'. "
                f"Available versions: {list(model_info.keys())}"
            )
        return model_info[version]

    # Return all versions if no specific version requested
    return model_info


def create_model(
    model_type: str,
    config: dict = None,
    pretrained: bool = True,
    device: str = None,
):
    """
    Create a model instance with the given configuration.

    Args:
        model_type: Type of model to create
            ('world_model', 'text_encoder', 'vision_encoder', 'nerf', 'gaussian')
        config: Model configuration dictionary
        pretrained: Whether to load pretrained weights
        device: Device to load model on ('cpu', 'cuda', 'cuda:0')

    Returns:
        Initialized model instance

    Raises:
        ValueError: If model_type is not recognized
        RuntimeError: If pretrained weights cannot be loaded
    """
    import torch
    from pathlib import Path

    # Set default device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load default config if not provided
    if config is None:
        config = DEFAULT_CONFIG.get(model_type, {})

    # Validate config
    validate_config(config, model_type)

    model_constructors = {
        "world_model": WorldModel,
        "text_encoder": TextEncoder,
        "vision_encoder": VisionEncoder,
        "nerf": NeuralRadianceField,
        "gaussian": GaussianSplattingModel,
        "mesh": MeshGenerator,
    }

    if model_type not in model_constructors:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available types: {list(model_constructors.keys())}"
        )

    # Create model instance
    model_class = model_constructors[model_type]
    model = model_class(config)

    # Load pretrained weights if requested
    if pretrained:
        model_info = get_model_info(model_type, config.get("version", "latest"))
        weights_path = download_model(model_type, model_info["download_url"])
        
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained weights: {e}")

    # Move model to device
    model.to(device)

    # Set model to evaluation mode if pretrained
    if pretrained:
        model.eval()

    return model



def download_model(model_name: str, url: str = None):
    """
    Download a pretrained model if not already cached.

    Args:
        model_name: Name of the model to download
        url: Optional direct download URL

    Returns:
        Path to the downloaded model file
    """
    import torch
    from pathlib import Path
    import requests
    from tqdm import tqdm

    # Create models directory if it doesn't exist
    models_dir = Path.home() / ".frextech" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Get model info if URL not provided
    if url is None:
        model_info = get_model_info(model_name)
        if "download_url" in model_info:
            url = model_info["download_url"]
        else:
            # Get first version's URL
            first_version = list(model_info.values())[0]
            url = first_version["download_url"]

    # Extract filename from URL
    filename = url.split("/")[-1]
    model_path = models_dir / filename

    # Check if model already exists
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        return model_path

    print(f"Downloading {model_name} from {url}...")

    # Download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(model_path, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

    print(f"Model downloaded to {model_path}")
    return model_path



def clear_model_cache(model_name: str = None):
    """
    Clear cached model files.

    Args:
        model_name: Optional specific model to clear
    """
    import shutil
    from pathlib import Path

    cache_dir = Path.home() / ".frextech" / "models"

    if not cache_dir.exists():
        return

    if model_name:
        # Clear specific model
        for file in cache_dir.glob(f"*{model_name}*"):
            file.unlink()
        print(f"Cleared cache for {model_name}")
    else:
        # Clear all cache
        shutil.rmtree(cache_dir)
        print("Cleared all model cache")



def benchmark_model(
    model,
    input_shape: tuple,
    device: str = None,
    num_runs: int = 100,
    warmup: int = 10,
) -> dict:
    """
    Benchmark model performance.

    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor (batch_size, ...)
        device: Device to run benchmark on
        num_runs: Number of inference runs
        warmup: Number of warmup runs

    Returns:
        Dictionary with benchmark results
    """
    import torch
    import time

    if device is None:
        device = next(model.parameters()).device

    # Generate random input
    batch_size = input_shape[0]
    dummy_input = torch.randn(input_shape).to(device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)

    # Benchmark inference
    torch.cuda.synchronize() if device.startswith("cuda") else None
    start_time = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)

    torch.cuda.synchronize() if device.startswith("cuda") else None
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    throughput = batch_size / avg_time

    # Memory usage
    if device.startswith("cuda"):
        memory_allocated = torch.cuda.memory_allocated(device) / 1e9  # GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1e9  # GB
    else:
        memory_allocated = memory_reserved = None

    return {
        "device": device,
        "batch_size": batch_size,
        "num_runs": num_runs,
        "total_time": total_time,
        "avg_inference_time": avg_time,
        "throughput_fps": throughput,
        "memory_allocated_gb": memory_allocated,
        "memory_reserved_gb": memory_reserved,
        "input_shape": input_shape,
    }



def export_model(
    model,
    export_format: str = "torchscript",
    output_path: str = None,
    example_input = None,
):
    """
    Export model to different formats.

    Args:
        model: Model to export
        export_format: Format to export to ('torchscript', 'onnx', 'tensorrt')
        output_path: Path to save exported model
        example_input: Example input for tracing

    Returns:
        Path to exported model
    """
    import torch
    from pathlib import Path

    if output_path is None:
        model_name = model.__class__.__name__.lower()
        output_path = f"./{model_name}.{export_format}"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    if export_format == "torchscript":
        # Trace model
        if example_input is None:
            # Create dummy input based on model's expected input
            if hasattr(model, "example_input"):
                example_input = model.example_input
            else:
                # Default input shape
                example_input = torch.randn(1, 3, 224, 224)
        
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(str(output_path))
        
    elif export_format == "onnx":
        if example_input is None:
            example_input = torch.randn(1, 3, 224, 224)
        
        torch.onnx.export(
            model,
            example_input,
            str(output_path),
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        
    elif export_format == "tensorrt":
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT is not installed. Install with: pip install tensorrt")
        
        # Convert via ONNX first
        onnx_path = output_path.with_suffix(".onnx")
        export_model(model, "onnx", onnx_path, example_input)
        
        # TODO: Implement TensorRT conversion
        raise NotImplementedError("TensorRT export not yet implemented")
        
    else:
        raise ValueError(f"Unsupported export format: {export_format}")

    print(f"Model exported to {output_path}")
    return output_path



def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    import torch
    import numpy as np
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_device_info() -> dict:
    """
    Get information about available devices.

    Returns:
        Dictionary with device information
    """
    import torch

    info = {
        "cpu": {
            "count": torch.get_num_threads(),
            "available": True,
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info[f"cuda:{i}"] = {
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
                "clock_rate_ghz": props.clock_rate / 1e6,
            }

    return info



def optimize_model(model, device: str = None, mode: str = "inference"):
    """
    Optimize model for deployment.

    Args:
        model: Model to optimize
        device: Device to optimize for
        mode: Optimization mode ('inference', 'training')

    Returns:
        Optimized model
    """
    import torch

    if device is None:
        device = next(model.parameters()).device

    model.to(device)

    if mode == "inference":
        model.eval()
        
        # Enable TF32 for Ampere GPUs
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Use torch.jit.optimize_for_inference if available
        if hasattr(torch.jit, "optimize_for_inference"):
            try:
                example_input = torch.randn(1, 3, 224, 224).to(device)
                traced = torch.jit.trace(model, example_input)
                model = torch.jit.optimize_for_inference(traced)
            except Exception as e:
                print(f"JIT optimization failed: {e}")
        
        # Enable cudnn benchmark for fixed input sizes
        torch.backends.cudnn.benchmark = True

    elif mode == "training":
        model.train()
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
        
        # Enable automatic mixed precision
        if device.startswith("cuda"):
            model = torch.cuda.amp.autocast()(model)

    return model


# Type aliases for better code completion
from typing import Union, Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn

ModelType = Union[nn.Module, WorldModel, TextEncoder, VisionEncoder]
ConfigType = Dict[str, Any]
DeviceType = Union[str, torch.device]
TensorDict = Dict[str, torch.Tensor]


class ModelRegistry:
    """
    Registry for managing model instances.
    """

    def __init__(self):
        self._models = {}
        self._configs = {}

    def register(self, name: str, model_class, config: dict = None):
        """
        Register a model class.
        
        Args:
            name: Model name
            model_class: Model class
            config: Default configuration
        """
        self._models[name] = model_class
        if config:
            self._configs[name] = config

    def create(self, name: str, **kwargs) -> ModelType:
        """
        Create a model instance.
        
        Args:
            name: Model name
            **kwargs: Model configuration
        
        Returns:
            Model instance
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not registered")
        
        model_class = self._models[name]
        config = self._configs.get(name, {}).copy()
        config.update(kwargs)
        
        return model_class(config)

    def list_models(self) -> List[str]:
        """
        List all registered models.
        
        Returns:
            List of model names
        """
        return list(self._models.keys())


# Create global registry
registry = ModelRegistry()

# Register core models
registry.register("world_model", WorldModel, DEFAULT_CONFIG["world_model"])
registry.register("text_encoder", TextEncoder, DEFAULT_CONFIG["multimodal"])
registry.register("vision_encoder", VisionEncoder, DEFAULT_CONFIG["multimodal"])
registry.register("nerf", NeuralRadianceField, DEFAULT_CONFIG["representation"]["nerf"])
registry.register("gaussian", GaussianSplattingModel, DEFAULT_CONFIG["representation"]["gaussian"])
registry.register("mesh", MeshGenerator, DEFAULT_CONFIG["representation"]["mesh"])


# Error classes

class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class ConfigurationError(ModelError):
    """Raised when model configuration is invalid."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""
    pass


class InferenceError(ModelError):
    """Raised when model inference fails."""
    pass


# Deprecation warnings
import warnings
import functools


def deprecated(message: str = ""):
    """
    Decorator to mark functions as deprecated.

    Args:
        message: Additional deprecation message
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in a future version. {message}",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Legacy compatibility
@deprecated("Use create_model() instead")
def load_model(model_type: str, **kwargs):
    """Legacy function for loading models."""
    return create_model(model_type, **kwargs)


# Test the module
if __name__ == "__main__":
    # Print module information
    print(f"FrexTech Core AI Module v{version}")
    print(f"Author: {author}")
    print(f"Email: {email}")

    # List available models
    models = list_available_models()
    print(f"\nAvailable models:")
    for category, model_list in models.items():
        print(f"  {category}:")
        for model_name in model_list.keys():
            print(f"    - {model_name}")

    # Get device info
    device_info = get_device_info()
    print(f"\nDevice information:")
    print(f"  CPU threads: {device_info['cpu']['count']}")
    print(f"  CUDA available: {device_info['cuda']['available']}")
    if device_info['cuda']['available']:
        print(f"  CUDA devices: {device_info['cuda']['device_count']}")
        for i in range(device_info['cuda']['device_count']):
            dev_info = device_info[f'cuda:{i}']
            print(f"    Device {i}: {dev_info['name']} ({dev_info['total_memory_gb']:.1f} GB)")

    # Test model registry
    print(f"\nRegistered models in registry: {registry.list_models()}")

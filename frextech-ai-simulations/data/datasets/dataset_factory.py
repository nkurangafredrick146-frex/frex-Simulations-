"""
Dataset Factory Module
Factory pattern for creating datasets by name or configuration.
"""

import os
import json
import yaml
from typing import Any, Dict, List, Optional, Union, Type
from pathlib import Path
from dataclasses import asdict

from .base_dataset import BaseDataset, DatasetConfig
from .image_dataset import ImageDataset, ImageDatasetConfig
from .video_dataset import VideoDataset, VideoDatasetConfig
from .multimodal_dataset import MultimodalDataset, MultimodalDatasetConfig
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Registry for dataset classes
_DATASET_REGISTRY = {}

def register_dataset(name: str, dataset_class: Type[BaseDataset], config_class: Type[DatasetConfig]):
    """
    Register a dataset class with the factory.
    
    Args:
        name: Name to register the dataset under
        dataset_class: Dataset class
        config_class: Configuration class for the dataset
    """
    _DATASET_REGISTRY[name] = {
        'class': dataset_class,
        'config_class': config_class
    }
    logger.info(f"Registered dataset: {name} -> {dataset_class.__name__}")

def get_dataset_class(name: str) -> Type[BaseDataset]:
    """
    Get dataset class by name.
    
    Args:
        name: Dataset name
        
    Returns:
        Dataset class
        
    Raises:
        ValueError: If dataset is not registered
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not registered. Available: {list(_DATASET_REGISTRY.keys())}")
    return _DATASET_REGISTRY[name]['class']

def get_config_class(name: str) -> Type[DatasetConfig]:
    """
    Get configuration class by dataset name.
    
    Args:
        name: Dataset name
        
    Returns:
        Configuration class
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not registered. Available: {list(_DATASET_REGISTRY.keys())}")
    return _DATASET_REGISTRY[name]['config_class']

def create_dataset(config: Union[Dict[str, Any], DatasetConfig, str], **kwargs) -> BaseDataset:
    """
    Create a dataset instance from configuration.
    
    Args:
        config: Configuration dictionary, DatasetConfig instance, or path to config file
        **kwargs: Additional arguments to pass to dataset constructor
        
    Returns:
        Dataset instance
    """
    # Load configuration if path is provided
    if isinstance(config, str):
        config_path = Path(config)
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        return create_dataset(config_dict, **kwargs)
    
    # Convert DatasetConfig to dict if needed
    if isinstance(config, DatasetConfig):
        config_dict = asdict(config)
    else:
        config_dict = config.copy()
    
    # Extract dataset name and type
    dataset_name = config_dict.get('name', 'unknown')
    dataset_type = config_dict.get('type', 'image')
    
    # Get dataset class
    if dataset_type in _DATASET_REGISTRY:
        dataset_class = _DATASET_REGISTRY[dataset_type]['class']
        config_class = _DATASET_REGISTRY[dataset_type]['config_class']
    else:
        # Try to find by name if type not found
        if dataset_name in _DATASET_REGISTRY:
            dataset_class = _DATASET_REGISTRY[dataset_name]['class']
            config_class = _DATASET_REGISTRY[dataset_name]['config_class']
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}. "
                           f"Registered types: {list(_DATASET_REGISTRY.keys())}")
    
    # Create configuration instance
    try:
        dataset_config = config_class.from_dict(config_dict)
    except Exception as e:
        logger.error(f"Failed to create config for {dataset_type}: {e}")
        # Fall back to base config
        dataset_config = DatasetConfig.from_dict(config_dict)
    
    # Create dataset instance
    try:
        dataset = dataset_class(config=dataset_config, **kwargs)
        logger.info(f"Created dataset: {dataset_name} ({dataset_type}) with {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.error(f"Failed to create dataset {dataset_name}: {e}")
        raise

def create_datasets_from_config(config_path: str, **kwargs) -> Dict[str, BaseDataset]:
    """
    Create multiple datasets from a configuration file.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional arguments for dataset creation
        
    Returns:
        Dictionary mapping dataset names to instances
    """
    config_path = Path(config_path)
    
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    datasets = {}
    
    # Handle different config structures
    if isinstance(config_dict, list):
        # List of dataset configs
        for dataset_config in config_dict:
            dataset = create_dataset(dataset_config, **kwargs)
            datasets[dataset.name] = dataset
    
    elif isinstance(config_dict, dict):
        if 'datasets' in config_dict:
            # Dict with 'datasets' key
            for name, dataset_config in config_dict['datasets'].items():
                if 'name' not in dataset_config:
                    dataset_config['name'] = name
                dataset = create_dataset(dataset_config, **kwargs)
                datasets[name] = dataset
        else:
            # Single dataset config
            dataset = create_dataset(config_dict, **kwargs)
            datasets[dataset.name] = dataset
    
    return datasets

def list_registered_datasets() -> List[str]:
    """List all registered dataset names."""
    return list(_DATASET_REGISTRY.keys())

def get_dataset_info(name: str) -> Dict[str, Any]:
    """
    Get information about a registered dataset.
    
    Args:
        name: Dataset name
        
    Returns:
        Dictionary with dataset information
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not registered")
    
    info = {
        'name': name,
        'class': _DATASET_REGISTRY[name]['class'].__name__,
        'config_class': _DATASET_REGISTRY[name]['config_class'].__name__,
        'module': _DATASET_REGISTRY[name]['class'].__module__,
        'description': _DATASET_REGISTRY[name]['class'].__doc__ or '',
    }
    
    return info

def export_dataset_schema(name: str, output_path: str) -> None:
    """
    Export JSON schema for a dataset configuration.
    
    Args:
        name: Dataset name
        output_path: Path to save schema
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' not registered")
    
    config_class = _DATASET_REGISTRY[name]['config_class']
    schema = config_class.schema()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)
    
    logger.info(f"Exported schema for {name} to {output_path}")

# Register built-in datasets
register_dataset('image', ImageDataset, ImageDatasetConfig)
register_dataset('video', VideoDataset, VideoDatasetConfig)
register_dataset('multimodal', MultimodalDataset, MultimodalDatasetConfig)

# Alias registrations
register_dataset('img', ImageDataset, ImageDatasetConfig)
register_dataset('vid', VideoDataset, VideoDatasetConfig)
register_dataset('multi', MultimodalDataset, MultimodalDatasetConfig)
register_dataset('mm', MultimodalDataset, MultimodalDatasetConfig)

class DatasetFactory:
    """
    Factory class for creating and managing datasets.
    
    Provides additional utilities for dataset management, validation, and composition.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize dataset factory.
        
        Args:
            config_dir: Directory containing dataset configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else None
        self._datasets = {}
        self._configs = {}
        
        if self.config_dir and self.config_dir.exists():
            self._load_configs_from_dir()
    
    def _load_configs_from_dir(self) -> None:
        """Load dataset configurations from configuration directory."""
        config_files = list(self.config_dir.rglob('*.yaml')) + list(self.config_dir.rglob('*.yml'))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config_dict = yaml.safe_load(f)
                
                config_name = config_file.stem
                self._configs[config_name] = config_dict
                logger.debug(f"Loaded config: {config_name} from {config_file}")
            
            except Exception as e:
                logger.error(f"Failed to load config {config_file}: {e}")
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """
        Get configuration by name.
        
        Args:
            name: Configuration name
            
        Returns:
            Configuration dictionary
        """
        if name in self._configs:
            return self._configs[name].copy()
        else:
            raise ValueError(f"Configuration '{name}' not found")
    
    def create(self, name: str, **kwargs) -> BaseDataset:
        """
        Create dataset by name.
        
        Args:
            name: Dataset or configuration name
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            Dataset instance
        """
        # Check if name is a registered dataset type
        if name in _DATASET_REGISTRY:
            # Create with default config
            config = {'name': name, 'type': name}
            return create_dataset(config, **kwargs)
        
        # Check if name is a configuration
        if name in self._configs:
            config = self._configs[name]
            if 'name' not in config:
                config['name'] = name
            return create_dataset(config, **kwargs)
        
        # Try to interpret as path
        config_path = Path(name)
        if config_path.exists():
            return create_dataset(str(config_path), **kwargs)
        
        raise ValueError(f"Unknown dataset, configuration, or path: {name}")
    
    def create_from_template(self, template_name: str, 
                           overrides: Dict[str, Any] = None,
                           **kwargs) -> BaseDataset:
        """
        Create dataset from template with overrides.
        
        Args:
            template_name: Template configuration name
            overrides: Configuration overrides
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            Dataset instance
        """
        if template_name not in self._configs:
            raise ValueError(f"Template '{template_name}' not found")
        
        config = self._configs[template_name].copy()
        
        if overrides:
            # Deep merge overrides
            self._deep_update(config, overrides)
        
        return create_dataset(config, **kwargs)
    
    def _deep_update(self, original: Dict, updates: Dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in updates.items():
            if (key in original and isinstance(original[key], dict) 
                and isinstance(value, dict)):
                self._deep_update(original[key], value)
            else:
                original[key] = value
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate dataset configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        required_fields = ['name', 'type', 'data_dir']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Check dataset type
        dataset_type = config.get('type')
        if dataset_type not in _DATASET_REGISTRY:
            errors.append(f"Unknown dataset type: {dataset_type}")
        else:
            # Validate with schema if available
            config_class = _DATASET_REGISTRY[dataset_type]['config_class']
            try:
                # Try to create config instance to validate
                config_class.from_dict(config)
            except Exception as e:
                errors.append(f"Configuration validation failed: {str(e)}")
        
        # Check data directory
        data_dir = config.get('data_dir')
        if data_dir and not Path(data_dir).exists():
            errors.append(f"Data directory does not exist: {data_dir}")
        
        return errors
    
    def save_dataset(self, dataset: BaseDataset, save_dir: str) -> None:
        """
        Save dataset configuration and metadata.
        
        Args:
            dataset: Dataset instance
            save_dir: Directory to save dataset information
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = save_dir / 'config.yaml'
        dataset.config.to_yaml(str(config_path))
        
        # Save sample list
        samples_path = save_dir / 'samples.json'
        with open(samples_path, 'w') as f:
            json.dump(dataset.samples[:100], f, indent=2, default=str)  # First 100 samples
        
        # Save statistics
        stats_path = save_dir / 'statistics.json'
        stats = dataset.compute_statistics()
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Saved dataset information to {save_dir}")
    
    def load_dataset(self, load_dir: str, **kwargs) -> BaseDataset:
        """
        Load dataset from saved configuration.
        
        Args:
            load_dir: Directory containing saved dataset information
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            Dataset instance
        """
        load_dir = Path(load_dir)
        
        # Load configuration
        config_path = load_dir / 'config.yaml'
        if not config_path.exists():
            config_path = load_dir / 'config.json'
        
        if not config_path.exists():
            raise ValueError(f"No configuration found in {load_dir}")
        
        return create_dataset(str(config_path), **kwargs)
    
    def create_composite_dataset(self, datasets_config: List[Dict[str, Any]], 
                               sampling_weights: Optional[List[float]] = None,
                               **kwargs) -> 'CompositeDataset':
        """
        Create a composite dataset from multiple datasets.
        
        Args:
            datasets_config: List of dataset configurations
            sampling_weights: Weights for sampling from each dataset
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            CompositeDataset instance
        """
        from .composite_dataset import CompositeDataset, CompositeDatasetConfig
        
        datasets = []
        for dataset_config in datasets_config:
            dataset = create_dataset(dataset_config, **kwargs)
            datasets.append(dataset)
        
        config = CompositeDatasetConfig(
            name='composite',
            datasets=datasets,
            sampling_weights=sampling_weights
        )
        
        return CompositeDataset(config)
    
    def __repr__(self) -> str:
        """String representation of factory."""
        return (f"DatasetFactory(config_dir={self.config_dir}, "
                f"configs={len(self._configs)}, "
                f"registered={len(_DATASET_REGISTRY)})")

# Default factory instance
_default_factory = DatasetFactory()

def get_factory(config_dir: Optional[str] = None) -> DatasetFactory:
    """
    Get dataset factory instance.
    
    Args:
        config_dir: Configuration directory
        
    Returns:
        DatasetFactory instance
    """
    if config_dir:
        return DatasetFactory(config_dir)
    return _default_factory

# Convenience functions
def load_dataset(name: str, **kwargs) -> BaseDataset:
    """Load dataset by name."""
    return _default_factory.create(name, **kwargs)

def validate_dataset_config(config: Dict[str, Any]) -> bool:
    """Validate dataset configuration."""
    errors = _default_factory.validate_config(config)
    if errors:
        for error in errors:
            logger.error(error)
        return False
    return True
"""
FrexTech AI Simulations - Data Module
Main data handling module for datasets, preprocessing, and embeddings.
"""

from . import datasets
from . import utils

# Version
__version__ = "1.0.0"
__author__ = "FrexTech AI Team"

# Main exports
__all__ = [
    # Datasets
    "datasets",
    
    # Utils
    "utils",
    
    # Core functions
    "load_dataset",
    "process_data",
    "create_embeddings",
    "validate_data",
    "get_data_stats",
]

# Import core functions
from .datasets import (
    BaseDataset,
    ImageDataset,
    VideoDataset,
    MultimodalDataset,
    create_dataset,
    get_factory,
)

from .utils import (
    DataValidator,
    DataPreprocessor,
    EmbeddingManager,
    DataAugmentor,
    get_data_statistics,
)

# Convenience functions
def load_dataset(config_path: str, **kwargs):
    """Load dataset from configuration file."""
    from .datasets.dataset_factory import create_dataset
    return create_dataset(config_path, **kwargs)

def process_data(input_path: str, output_path: str, processor_config: dict = None):
    """Process raw data into training-ready format."""
    from .utils.data_preprocessor import DataPreprocessor
    processor = DataPreprocessor(config=processor_config)
    return processor.process(input_path, output_path)

def create_embeddings(data_path: str, model_name: str, output_path: str, **kwargs):
    """Create embeddings for data using specified model."""
    from .utils.embedding_manager import EmbeddingManager
    manager = EmbeddingManager()
    return manager.create_embeddings(data_path, model_name, output_path, **kwargs)

def validate_data(data_path: str, schema: dict = None):
    """Validate data against schema or basic checks."""
    from .utils.data_validator import DataValidator
    validator = DataValidator(schema)
    return validator.validate(data_path)

def get_data_stats(data_path: str):
    """Get statistics about data."""
    from .utils.data_analyzer import DataAnalyzer
    analyzer = DataAnalyzer(data_path)
    return analyzer.get_statistics()
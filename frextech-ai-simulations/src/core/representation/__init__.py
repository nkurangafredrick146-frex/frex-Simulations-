"""
3D Representation module for neural scene representations.
Includes NeRF, Gaussian Splatting, Mesh, and Voxel representations.
"""

from .nerf import (
    NeRFModel,
    RaySampler,
    VolumeRenderer,
    PositionalEncoding,
    HierarchicalNeRF,
    InstantNGP
)
from .gaussian_splatting import (
    GaussianModel,
    GaussianRasterizer,
    GaussianOptimizer,
    GaussianScene,
    GaussianSplattingPipeline
)
from .mesh import (
    MeshGenerator,
    MeshRefiner,
    TextureGenerator,
    MarchingCubes,
    PoissonReconstruction
)
from .voxel import (
    VoxelGrid,
    SparseVoxelGrid,
    HashGrid,
    TriplaneRepresentation,
    FeatureGrid
)
from .utils import (
    representation_utils,
    geometry_utils,
    rendering_utils,
    conversion_utils
)
from .converter import RepresentationConverter
from .quality_assessor import RepresentationQuality

__all__ = [
    # NeRF
    'NeRFModel',
    'RaySampler',
    'VolumeRenderer',
    'PositionalEncoding',
    'HierarchicalNeRF',
    'InstantNGP',
    
    # Gaussian Splatting
    'GaussianModel',
    'GaussianRasterizer',
    'GaussianOptimizer',
    'GaussianScene',
    'GaussianSplattingPipeline',
    
    # Mesh
    'MeshGenerator',
    'MeshRefiner',
    'TextureGenerator',
    'MarchingCubes',
    'PoissonReconstruction',
    
    # Voxel
    'VoxelGrid',
    'SparseVoxelGrid',
    'HashGrid',
    'TriplaneRepresentation',
    'FeatureGrid',
    
    # Utilities
    'RepresentationConverter',
    'RepresentationQuality',
    'representation_utils',
    'geometry_utils',
    'rendering_utils',
    'conversion_utils'
]

# Version
__version__ = '1.0.0'

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Configuration
class RepresentationConfig:
    """Configuration for representation methods."""
    
    def __init__(self, **kwargs):
        # General
        self.representation_type = kwargs.get('representation_type', 'gaussian')
        self.resolution = kwargs.get('resolution', 256)
        self.feature_dim = kwargs.get('feature_dim', 32)
        self.use_cuda = kwargs.get('use_cuda', True)
        
        # NeRF specific
        self.nerf_num_layers = kwargs.get('nerf_num_layers', 8)
        self.nerf_hidden_dim = kwargs.get('nerf_hidden_dim', 256)
        self.nerf_num_samples = kwargs.get('nerf_num_samples', 64)
        self.nerf_num_samples_fine = kwargs.get('nerf_num_samples_fine', 128)
        
        # Gaussian Splatting specific
        self.gs_max_sh_degree = kwargs.get('gs_max_sh_degree', 3)
        self.gs_num_points = kwargs.get('gs_num_points', 100000)
        self.gs_opacity_init = kwargs.get('gs_opacity_init', 0.1)
        self.gs_scale_init = kwargs.get('gs_scale_init', 0.01)
        
        # Mesh specific
        self.mesh_decimation_ratio = kwargs.get('mesh_decimation_ratio', 0.5)
        self.mesh_smoothing_iterations = kwargs.get('mesh_smoothing_iterations', 3)
        self.texture_resolution = kwargs.get('texture_resolution', 1024)
        
        # Voxel specific
        self.voxel_size = kwargs.get('voxel_size', 0.01)
        self.sparse_block_size = kwargs.get('sparse_block_size', 16)
        self.hash_table_size = kwargs.get('hash_table_size', 2**19)
        
    def to_dict(self):
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def update(self, **kwargs):
        """Update configuration."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Config has no attribute: {key}")

# Factory functions
def create_representation(rep_type: str, config: dict = None):
    """
    Factory function to create a representation.
    
    Args:
        rep_type: Type of representation ('nerf', 'gaussian', 'mesh', 'voxel')
        config: Configuration dictionary
        
    Returns:
        Representation instance
    """
    if config is None:
        config = {}
    
    config['representation_type'] = rep_type
    rep_config = RepresentationConfig(**config)
    
    if rep_type == 'nerf':
        return NeRFModel(rep_config)
    elif rep_type == 'gaussian':
        return GaussianScene(rep_config)
    elif rep_type == 'mesh':
        return MeshGenerator(rep_config)
    elif rep_type == 'voxel':
        return VoxelGrid(rep_config)
    else:
        raise ValueError(
            f"Unknown representation type: {rep_type}. "
            f"Must be one of: ['nerf', 'gaussian', 'mesh', 'voxel']"
        )

def get_renderer(rep_type: str, device=None):
    """
    Get appropriate renderer for representation type.
    
    Args:
        rep_type: Representation type
        device: Torch device
        
    Returns:
        Renderer instance
    """
    if rep_type == 'nerf':
        return VolumeRenderer(device=device)
    elif rep_type == 'gaussian':
        return GaussianRasterizer(device=device)
    elif rep_type == 'mesh':
        from .mesh import MeshRenderer
        return MeshRenderer(device=device)
    elif rep_type == 'voxel':
        from .voxel import VoxelRenderer
        return VoxelRenderer(device=device)
    else:
        raise ValueError(f"Unknown representation type: {rep_type}")

# Global configuration
_global_config = RepresentationConfig()

def get_global_config():
    """Get global representation configuration."""
    return _global_config

def set_global_config(**kwargs):
    """Update global representation configuration."""
    _global_config.update(**kwargs)

# Utility functions
def estimate_memory_usage(rep_type: str, resolution: int):
    """
    Estimate memory usage for representation.
    
    Args:
        rep_type: Representation type
        resolution: Spatial resolution
        
    Returns:
        Estimated memory in MB
    """
    if rep_type == 'nerf':
        # MLP parameters
        return resolution * 0.1  # Approximate
    elif rep_type == 'gaussian':
        # 3D Gaussians: position(3) + rotation(4) + scale(3) + opacity(1) + sh_coefficients(48)
        params_per_gaussian = 59  # 3+4+3+1+48
        num_gaussians = (resolution ** 3) * 0.001  # Approximate density
        return (num_gaussians * params_per_gaussian * 4) / (1024 * 1024)  # MB
    elif rep_type == 'mesh':
        # Vertices + faces + textures
        num_vertices = (resolution ** 2) * 2
        return (num_vertices * 3 * 4) / (1024 * 1024)  # MB
    elif rep_type == 'voxel':
        # Dense voxel grid
        return (resolution ** 3 * 4) / (1024 * 1024)  # MB
    else:
        return 0

def get_supported_formats(rep_type: str):
    """Get supported export formats for representation type."""
    formats = {
        'nerf': ['.pt', '.pth', '.npz'],
        'gaussian': ['.ply', '.npz', '.pt'],
        'mesh': ['.obj', '.ply', '.glb', '.fbx'],
        'voxel': ['.npz', '.bin', '.h5']
    }
    return formats.get(rep_type, [])
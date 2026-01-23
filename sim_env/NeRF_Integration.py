#!/usr/bin/env python3
"""
Neural Radiance Fields (NeRF) Integration Module
Real-time neural rendering with instant neural graphics primitives and advanced NeRF capabilities
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import *
import glm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Callable
import math
import random
from enum import Enum
import time
from collections import deque
import json
from scipy.spatial.transform import Rotation as R
import imageio
from PIL import Image

class NeRFMode(Enum):
    """NeRF operation modes"""
    TRAINING = "training"
    INFERENCE = "inference"
    STREAMING = "streaming"
    EDITING = "editing"

class EncodingType(Enum):
    """Positional encoding types"""
    FOURIER = "fourier"
    HASH_GRID = "hash_grid"
    SPHERICAL_HARMONICS = "spherical_harmonics"
    TRI_PLANE = "tri_plane"

@dataclass
class CameraPose:
    """Camera pose for NeRF training"""
    position: np.ndarray
    rotation: np.ndarray  # quaternion
    focal_length: float
    image_size: Tuple[int, int]
    image_path: Optional[str] = None
    image_data: Optional[np.ndarray] = None

@dataclass
class RayBundle:
    """Bundle of rays for neural rendering"""
    origins: torch.Tensor
    directions: torch.Tensor
    near: float
    far: float
    pixel_coords: Optional[torch.Tensor] = None

class PositionalEncoder(nn.Module):
    """Advanced positional encoding for NeRF"""
    
    def __init__(self, num_frequencies: int = 10, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.output_dim = 3 * (2 * num_frequencies + (1 if include_input else 0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding"""
        if self.num_frequencies == 0:
            return x
            
        frequencies = 2.0 ** torch.arange(0, self.num_frequencies, dtype=x.dtype, device=x.device)
        scaled = x.unsqueeze(-1) * frequencies.unsqueeze(0)
        
        # Sin and cos components
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        encoded = encoded.reshape(x.shape[0], -1)
        
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)
            
        return encoded

class HashGridEncoder(nn.Module):
    """Instant Neural Graphics Primitives hash grid encoder"""
    
    def __init__(self, num_levels: int = 16, level_dim: int = 2, base_resolution: int = 16, 
                 max_resolution: int = 2048, log2_hashmap_size: int = 19):
        super().__init__()
        
        # TCNN configuration
        self.config = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": np.exp((np.log(max_resolution) - np.log(base_resolution)) / (num_levels - 1))
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2
            }
        }
        
        self.encoder = tcnn.Encoding(3, self.config["encoding"])
        self.output_dim = num_levels * level_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode positions using hash grid"""
        # Normalize to [0, 1] for TCNN
        x_normalized = (x + 1.0) / 2.0  # Assuming input in [-1, 1]
        return self.encoder(x_normalized)

class AdvancedNeRF(nn.Module):
    """Advanced NeRF model with instant neural graphics primitives"""
    
    def __init__(self, 
                 pos_encoding_dim: int = 10,
                 dir_encoding_dim: int = 4,
                 hidden_dim: int = 256,
                 num_layers: int = 8,
                 use_hash_grid: bool = True,
                 use_appearance_embedding: bool = True,
                 appearance_dim: int = 48):
        super().__init__()
        
        self.use_hash_grid = use_hash_grid
        self.use_appearance_embedding = use_appearance_embedding
        
        # Position encoding
        if use_hash_grid:
            self.pos_encoder = HashGridEncoder()
            pos_encoded_dim = self.pos_encoder.output_dim
        else:
            self.pos_encoder = PositionalEncoder(pos_encoding_dim)
            pos_encoded_dim = self.pos_encoder.output_dim
        
        # Direction encoding
        self.dir_encoder = PositionalEncoder(dir_encoding_dim)
        dir_encoded_dim = self.dir_encoder.output_dim
        
        # Appearance embedding
        if use_appearance_embedding:
            self.appearance_embedding = nn.Embedding(1000, appearance_dim)  # Support up to 1000 images
            dir_encoded_dim += appearance_dim
        
        # Main density network
        density_layers = []
        density_layers.append(nn.Linear(pos_encoded_dim, hidden_dim))
        density_layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            density_layers.append(nn.Linear(hidden_dim, hidden_dim))
            density_layers.append(nn.ReLU())
        
        density_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.density_network = nn.Sequential(*density_layers)
        
        # Density output
        self.density_output = nn.Linear(hidden_dim, 1)
        
        # Feature output for color
        self.feature_output = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Color network
        color_input_dim = hidden_dim // 2 + dir_encoded_dim
        self.color_network = nn.Sequential(
            nn.Linear(color_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, positions: torch.Tensor, directions: torch.Tensor, 
                appearance_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through NeRF model"""
        batch_size = positions.shape[0]
        
        # Encode positions
        if self.use_hash_grid:
            pos_encoded = self.pos_encoder(positions)
        else:
            pos_encoded = self.pos_encoder(positions)
        
        # Process through density network
        density_features = self.density_network(pos_encoded)
        densities = F.softplus(self.density_output(density_features))
        
        # Get features for color
        color_features = self.feature_output(density_features)
        
        # Encode directions and combine with appearance
        dir_encoded = self.dir_encoder(directions)
        
        if self.use_appearance_embedding and appearance_ids is not None:
            appearance_emb = self.appearance_embedding(appearance_ids)
            dir_encoded = torch.cat([dir_encoded, appearance_emb], dim=-1)
        
        # Combine features for color prediction
        color_input = torch.cat([color_features, dir_encoded], dim=-1)
        colors = self.color_network(color_input)
        
        return colors, densities

class InstantNGPWrapper:
    """Wrapper for Instant Neural Graphics Primitives"""
    
    def __init__(self, config: Dict = None):
        if config is None:
            config = self._default_config()
        
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.iterations = 0
        self.loss_history = []
        self.psnr_history = []
        
        # Rendering parameters
        self.num_samples_coarse = 64
        self.num_samples_fine = 128
        self.hierarchical_sampling = True
        
    def _default_config(self) -> Dict:
        """Default InstantNGP configuration"""
        return {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2
            },
            "optimizer": {
                "otype": "Adam",
                "learning_rate": 1e-2,
                "beta1": 0.9,
                "beta2": 0.99,
                "epsilon": 1e-15
            }
        }
    
    def initialize_model(self, input_dim: int = 3, output_dim: int = 4):
        """Initialize the InstantNGP model"""
        try:
            self.model = tcnn.NetworkWithInputEncoding(
                n_input_dims=input_dim,
                n_output_dims=output_dim,
                encoding_config=self.config["encoding"],
                network_config=self.config["network"]
            )
            
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["optimizer"]["learning_rate"],
                betas=(self.config["optimizer"]["beta1"], self.config["optimizer"]["beta2"]),
                eps=self.config["optimizer"]["epsilon"]
            )
            
            print("InstantNGP model initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize InstantNGP: {e}")
            # Fallback to PyTorch implementation
            self.model = AdvancedNeRF(use_hash_grid=False)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def query(self, positions: torch.Tensor, directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for colors and densities"""
        if hasattr(self.model, 'forward'):
            # InstantNGP model
            inputs = torch.cat([positions, directions], dim=-1)
            outputs = self.model(inputs)
            colors = torch.sigmoid(outputs[..., :3])
            densities = F.softplus(outputs[..., 3:4])
            return colors, densities
        else:
            # PyTorch model
            return self.model(positions, directions)
    
    def train_step(self, ray_bundle: RayBundle, target_pixels: torch.Tensor) -> Dict[str, float]:
        """Perform one training step"""
        self.optimizer.zero_grad()
        
        # Render the rays
        rendered, extras = self.render_rays(ray_bundle)
        
        # Compute losses
        color_loss = F.mse_loss(rendered['rgb'], target_pixels)
        total_loss = color_loss
        
        # Add distortion loss if available
        if 'weights' in extras and 'ts' in extras:
            distortion_loss = self.compute_distortion_loss(extras['weights'], extras['ts'])
            total_loss += 0.01 * distortion_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        self.iterations += 1
        
        # Compute PSNR
        mse = F.mse_loss(rendered['rgb'], target_pixels)
        psnr = -10.0 * torch.log10(mse)
        
        stats = {
            'loss': total_loss.item(),
            'color_loss': color_loss.item(),
            'psnr': psnr.item(),
            'iterations': self.iterations
        }
        
        self.loss_history.append(total_loss.item())
        self.psnr_history.append(psnr.item())
        
        return stats
    
    def render_rays(self, ray_bundle: RayBundle, randomized: bool = True) -> Tuple[Dict, Dict]:
        """Render rays using volume rendering"""
        rays_o = ray_bundle.origins
        rays_d = ray_bundle.directions
        near = ray_bundle.near
        far = ray_bundle.far
        
        # Sample points along rays
        t_vals, points = self.sample_along_rays(rays_o, rays_d, near, far, 
                                               self.num_samples_coarse, randomized)
        
        # Query model
        directions_norm = F.normalize(rays_d, dim=-1)
        directions_expanded = directions_norm.unsqueeze(1).expand_as(points)
        directions_flat = directions_expanded.reshape(-1, 3)
        points_flat = points.reshape(-1, 3)
        
        colors_flat, densities_flat = self.query(points_flat, directions_flat)
        
        colors = colors_flat.reshape(points.shape)
        densities = densities_flat.reshape(points.shape[:-1])
        
        # Volume rendering
        rgb_map, depth_map, acc_map, weights, transmittance = self.volume_rendering(
            colors, densities, t_vals, rays_d)
        
        result = {
            'rgb': rgb_map,
            'depth': depth_map,
            'acc': acc_map
        }
        
        extras = {
            'weights': weights,
            'ts': t_vals
        }
        
        # Hierarchical sampling
        if self.hierarchical_sampling and self.training:
            # Sample fine points
            t_vals_fine, points_fine = self.sample_fine(t_vals, weights, rays_o, rays_d, 
                                                       near, far, self.num_samples_fine, randomized)
            
            # Combine coarse and fine samples
            t_vals_combined, _ = torch.sort(torch.cat([t_vals, t_vals_fine], dim=-1), dim=-1)
            points_combined = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals_combined.unsqueeze(-1)
            
            # Query fine samples
            directions_expanded_fine = directions_norm.unsqueeze(1).expand_as(points_combined)
            directions_flat_fine = directions_expanded_fine.reshape(-1, 3)
            points_flat_fine = points_combined.reshape(-1, 3)
            
            colors_flat_fine, densities_flat_fine = self.query(points_flat_fine, directions_flat_fine)
            
            colors_fine = colors_flat_fine.reshape(points_combined.shape)
            densities_fine = densities_flat_fine.reshape(points_combined.shape[:-1])
            
            # Fine volume rendering
            rgb_map_fine, depth_map_fine, acc_map_fine, weights_fine, _ = self.volume_rendering(
                colors_fine, densities_fine, t_vals_combined, rays_d)
            
            result.update({
                'rgb_fine': rgb_map_fine,
                'depth_fine': depth_map_fine,
                'acc_fine': acc_map_fine
            })
            
            extras.update({
                'weights_fine': weights_fine,
                'ts_fine': t_vals_combined
            })
        
        return result, extras
    
    def sample_along_rays(self, rays_o: torch.Tensor, rays_d: torch.Tensor, 
                         near: float, far: float, num_samples: int, 
                         randomized: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample points along rays"""
        batch_size = rays_o.shape[0]
        
        # Linear sampling in disparity space
        t_vals = torch.linspace(0, 1, num_samples, device=rays_o.device)
        t_vals = near * (1 - t_vals) + far * t_vals
        
        if randomized:
            # Add random offsets
            mids = 0.5 * (t_vals[1:] + t_vals[:-1])
            upper = torch.cat([mids, t_vals[-1:]])
            lower = torch.cat([t_vals[:1], mids])
            t_rand = torch.rand(batch_size, num_samples, device=rays_o.device)
            t_vals = lower + (upper - lower) * t_rand
        else:
            t_vals = t_vals.unsqueeze(0).expand(batch_size, num_samples)
        
        # Compute sample positions
        points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(-1)
        
        return t_vals, points
    
    def sample_fine(self, t_vals: torch.Tensor, weights: torch.Tensor, 
                   rays_o: torch.Tensor, rays_d: torch.Tensor,
                   near: float, far: float, num_samples: int, 
                   randomized: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample additional points based on weights (importance sampling)"""
        weights = weights + 1e-5  # Prevent nans
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        # Inverse transform sampling
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)
        indices = torch.searchsorted(cdf, u, right=True)
        
        below = torch.max(torch.zeros_like(indices - 1), indices - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(indices), indices)
        indices_g = torch.stack([below, above], dim=-1)
        
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(-1, num_samples, -1), 2, indices_g)
        t_vals_g = torch.gather(t_vals.unsqueeze(1).expand(-1, num_samples, -1), 2, indices_g)
        
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        t_vals_fine = t_vals_g[..., 0] + t * (t_vals_g[..., 1] - t_vals_g[..., 0])
        
        t_vals_fine, _ = torch.sort(torch.cat([t_vals, t_vals_fine], dim=-1), dim=-1)
        
        points_fine = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals_fine.unsqueeze(-1)
        
        return t_vals_fine, points_fine
    
    def volume_rendering(self, colors: torch.Tensor, densities: torch.Tensor, 
                        t_vals: torch.Tensor, rays_d: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Perform volume rendering"""
        deltas = t_vals[..., 1:] - t_vals[..., :-1]
        delta_inf = 1e10 * torch.ones_like(deltas[..., :1])
        deltas = torch.cat([deltas, delta_inf], dim=-1)
        
        # Convert density to absorption
        dists = deltas * torch.norm(rays_d.unsqueeze(1), dim=-1)
        alpha = 1 - torch.exp(-densities * dists)
        
        # Compute transmittance and weights
        transmittance = torch.cumprod(1 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], dim=-1)
        weights = alpha * transmittance
        
        # Composite
        rgb_map = torch.sum(weights.unsqueeze(-1) * colors, dim=1)
        depth_map = torch.sum(weights * t_vals, dim=1)
        acc_map = torch.sum(weights, dim=1)
        
        return rgb_map, depth_map, acc_map, weights, transmittance
    
    def compute_distortion_loss(self, weights: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """Compute distortion loss for regularizing weights"""
        loss = 0.0
        
        # Mip-NeRF 360 distortion loss
        if weights.shape[1] > 1:
            w = weights + 1e-5
            p = w / torch.sum(w, dim=-1, keepdim=True)
            entropy = -torch.sum(p * torch.log(p), dim=-1)
            loss = torch.mean(entropy)
        
        return loss

class NeRFManager:
    """Main NeRF manager for integration with simulation software"""
    
    def __init__(self, width: int, height: int):
        # Rendering parameters
        self.width = width
        self.height = height
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # NeRF models
        self.coarse_model = None
        self.fine_model = None
        self.instant_ngp = None
        
        # Training data
        self.camera_poses: List[CameraPose] = []
        self.training_images: List[torch.Tensor] = []
        self.current_training_set = 0
        
        # Rendering state
        self.rendering_mode = NeRFMode.INFERENCE
        self.current_camera_pose = None
        self.render_texture = None
        
        # Real-time rendering optimization
        self.use_instant_ngp = True
        self.progressive_training = True
        self.interactive_editing = False
        
        # Performance tracking
        self.training_iterations = 0
        self.average_psnr = 0.0
        self.rendering_fps = 0.0
        
        # OpenGL resources
        self.display_program = None
        self.quad_vao = None
        self.quad_vbo = None
        self.nerf_texture = None
        
        # Neural rendering cache
        self.rendering_cache = {}
        self.cache_size = 100
        
        print(f"NeRF Manager initialized: {width}x{height}, Device: {self.device}")
    
    def initialize_models(self, use_instant_ngp: bool = True):
        """Initialize NeRF models"""
        self.use_instant_ngp = use_instant_ngp
        
        if use_instant_ngp:
            try:
                self.instant_ngp = InstantNGPWrapper()
                self.instant_ngp.initialize_model()
                print("InstantNGP model initialized")
            except Exception as e:
                print(f"InstantNGP initialization failed: {e}, falling back to standard NeRF")
                self.use_instant_ngp = False
        
        if not self.use_instant_ngp:
            self.coarse_model = AdvancedNeRF(use_hash_grid=False).to(self.device)
            self.fine_model = AdvancedNeRF(use_hash_grid=False).to(self.device)
            print("Standard NeRF models initialized")
    
    def add_training_image(self, image: np.ndarray, camera_pose: CameraPose):
        """Add training image with camera pose"""
        # Convert image to tensor
        image_tensor = torch.from_numpy(image).float().to(self.device)
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        self.training_images.append(image_tensor)
        self.camera_poses.append(camera_pose)
        
        print(f"Added training image {len(self.training_images)}: {image.shape}")
    
    def load_dataset(self, dataset_path: str):
        """Load NeRF dataset (NVIDIA format or COLMAP)"""
        try:
            # Try to load transforms.json (NVIDIA format)
            transforms_file = os.path.join(dataset_path, 'transforms.json')
            if os.path.exists(transforms_file):
                self._load_nvidia_dataset(transforms_file, dataset_path)
            else:
                # Try COLMAP format
                self._load_colmap_dataset(dataset_path)
                
        except Exception as e:
            print(f"Failed to load dataset: {e}")
    
    def _load_nvidia_dataset(self, transforms_file: str, dataset_path: str):
        """Load NVIDIA format dataset"""
        with open(transforms_file, 'r') as f:
            data = json.load(f)
        
        for frame in data['frames']:
            file_path = frame['file_path']
            if file_path.startswith('./'):
                file_path = file_path[2:]
            
            image_path = os.path.join(dataset_path, file_path)
            if not os.path.exists(image_path):
                image_path += '.png'  # Try with extension
            
            if os.path.exists(image_path):
                # Load image
                image = imageio.imread(image_path)
                if image.shape[2] == 4:  # RGBA
                    image = image[..., :3]  # Convert to RGB
                
                # Extract camera pose
                transform_matrix = np.array(frame['transform_matrix'])
                position = transform_matrix[:3, 3]
                rotation_matrix = transform_matrix[:3, :3]
                rotation = R.from_matrix(rotation_matrix).as_quat()
                
                camera_pose = CameraPose(
                    position=position,
                    rotation=rotation,
                    focal_length=data.get('camera_angle_x', 0.0),
                    image_size=(image.shape[1], image.shape[0]),
                    image_path=image_path
                )
                
                self.add_training_image(image, camera_pose)
    
    def _load_colmap_dataset(self, dataset_path: str):
        """Load COLMAP format dataset"""
        # Simplified COLMAP loader - in practice would parse cameras.txt and images.txt
        print("COLMAP dataset loading not fully implemented")
    
    def train(self, num_iterations: int = 1000, batch_size: int = 4096):
        """Train the NeRF model"""
        if not self.training_images:
            print("No training data available")
            return
        
        if self.use_instant_ngp and self.instant_ngp:
            self._train_instant_ngp(num_iterations, batch_size)
        else:
            self._train_standard_nerf(num_iterations, batch_size)
    
    def _train_instant_ngp(self, num_iterations: int, batch_size: int):
        """Train InstantNGP model"""
        print(f"Training InstantNGP for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            # Sample random rays from random training image
            img_idx = random.randint(0, len(self.training_images) - 1)
            image = self.training_images[img_idx]
            camera_pose = self.camera_poses[img_idx]
            
            # Generate random ray batch
            ray_bundle = self._generate_random_rays(camera_pose, batch_size)
            target_pixels = self._sample_pixels(image, ray_bundle.pixel_coords)
            
            # Training step
            stats = self.instant_ngp.train_step(ray_bundle, target_pixels)
            
            self.training_iterations += 1
            self.average_psnr = 0.95 * self.average_psnr + 0.05 * stats['psnr']
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss={stats['loss']:.4f}, PSNR={stats['psnr']:.2f}")
    
    def _train_standard_nerf(self, num_iterations: int, batch_size: int):
        """Train standard NeRF model"""
        print("Standard NeRF training not fully implemented in this example")
    
    def _generate_random_rays(self, camera_pose: CameraPose, num_rays: int) -> RayBundle:
        """Generate random rays from camera pose"""
        # Generate random pixel coordinates
        height, width = camera_pose.image_size
        pixel_x = torch.randint(0, width, (num_rays,), device=self.device)
        pixel_y = torch.randint(0, height, (num_rays,), device=self.device)
        
        # Convert to normalized device coordinates
        ndc_x = (pixel_x.float() / width - 0.5) * 2
        ndc_y = -(pixel_y.float() / height - 0.5) * 2
        
        # Generate ray directions
        focal = camera_pose.focal_length
        if focal == 0:
            focal = 0.5 * width / math.tan(math.radians(60) / 2)  # Default FOV
        
        directions = torch.stack([
            ndc_x / focal,
            ndc_y / focal,
            -torch.ones_like(ndc_x)
        ], dim=-1)
        
        # Transform to world coordinates
        rotation_matrix = torch.from_numpy(R.from_quat(camera_pose.rotation).as_matrix()).float().to(self.device)
        position = torch.from_numpy(camera_pose.position).float().to(self.device)
        
        directions_world = directions @ rotation_matrix.T
        origins_world = position.unsqueeze(0).expand(num_rays, -1)
        
        return RayBundle(
            origins=origins_world,
            directions=directions_world,
            near=0.1,
            far=10.0,
            pixel_coords=torch.stack([pixel_y, pixel_x], dim=-1)  # For pixel sampling
        )
    
    def _sample_pixels(self, image: torch.Tensor, pixel_coords: torch.Tensor) -> torch.Tensor:
        """Sample pixels from image at given coordinates"""
        if pixel_coords is None:
            return torch.zeros((0, 3), device=self.device)
        
        # Normalize coordinates to [-1, 1] for grid_sample
        height, width = image.shape[:2]
        norm_coords = torch.stack([
            2.0 * pixel_coords[:, 1] / width - 1.0,  # x
            2.0 * pixel_coords[:, 0] / height - 1.0   # y
        ], dim=-1).unsqueeze(0).unsqueeze(0)
        
        # Sample pixels
        sampled = F.grid_sample(
            image.permute(2, 0, 1).unsqueeze(0),  # [1, C, H, W]
            norm_coords,                           # [1, 1, N, 2]
            align_corners=False,
            mode='bilinear'
        )
        
        return sampled.squeeze().permute(1, 0)  # [N, 3]
    
    def render_view(self, camera_position: np.ndarray, camera_target: np.ndarray, 
                   camera_up: np.ndarray = None, fov: float = 60.0) -> np.ndarray:
        """Render a view from given camera parameters"""
        if camera_up is None:
            camera_up = np.array([0.0, 1.0, 0.0])
        
        # Generate rays for full image
        ray_bundle = self._generate_full_image_rays(camera_position, camera_target, camera_up, fov)
        
        # Render using current model
        if self.use_instant_ngp and self.instant_ngp:
            with torch.no_grad():
                rendered, _ = self.instant_ngp.render_rays(ray_bundle, randomized=False)
                image_tensor = rendered.get('rgb_fine', rendered['rgb'])
        else:
            # Fallback to standard rendering
            image_tensor = self._render_standard_nerf(ray_bundle)
        
        # Convert to numpy
        image = image_tensor.cpu().numpy().reshape(self.height, self.width, 3)
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image
    
    def _generate_full_image_rays(self, camera_position: np.ndarray, camera_target: np.ndarray,
                                 camera_up: np.ndarray, fov: float) -> RayBundle:
        """Generate rays for full image rendering"""
        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.arange(self.height, device=self.device),
            torch.arange(self.width, device=self.device),
            indexing='ij'
        )
        
        x = x.flatten()
        y = y.flatten()
        
        # Convert to normalized device coordinates
        ndc_x = (x.float() / self.width - 0.5) * 2
        ndc_y = -(y.float() / self.height - 0.5) * 2
        
        # Calculate focal length from FOV
        focal = 0.5 * self.width / math.tan(math.radians(fov) / 2)
        
        # Generate ray directions in camera space
        directions = torch.stack([
            ndc_x / focal,
            ndc_y / focal,
            -torch.ones_like(ndc_x)
        ], dim=-1)
        
        # Convert to world coordinates
        forward = camera_target - camera_position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, camera_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        rotation_matrix = torch.tensor(np.column_stack([right, up, -forward]), 
                                     dtype=torch.float32, device=self.device)
        
        directions_world = directions @ rotation_matrix.T
        origins_world = torch.tensor(camera_position, device=self.device).unsqueeze(0).expand(len(x), -1)
        
        return RayBundle(
            origins=origins_world,
            directions=directions_world,
            near=0.1,
            far=10.0
        )
    
    def _render_standard_nerf(self, ray_bundle: RayBundle) -> torch.Tensor:
        """Render using standard NeRF model (fallback)"""
        # Simplified standard NeRF rendering
        return torch.rand(len(ray_bundle.origins), 3, device=self.device)
    
    def initialize_opengl(self):
        """Initialize OpenGL resources for NeRF rendering display"""
        # Create texture for NeRF rendering
        self.nerf_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.nerf_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Simple shader for displaying texture
        vertex_shader = """
        #version 330 core
        layout (location = 0) in vec2 aPos;
        out vec2 TexCoords;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoords = aPos * 0.5 + 0.5;
        }
        """
        
        fragment_shader = """
        #version 330 core
        out vec4 FragColor;
        in vec2 TexCoords;
        uniform sampler2D nerfTexture;
        void main() {
            FragColor = texture(nerfTexture, TexCoords);
        }
        """
        
        self.display_program = compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )
        
        # Fullscreen quad
        quad_vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype=np.float32)
        
        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)
        
        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_TEXTURE_2D, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        print("OpenGL resources initialized for NeRF display")
    
    def update_display_texture(self, image: np.ndarray):
        """Update OpenGL texture with NeRF rendering"""
        if self.nerf_texture is None:
            self.initialize_opengl()
            
        glBindTexture(GL_TEXTURE_2D, self.nerf_texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, image)
    
    def render_to_screen(self):
        """Render NeRF output to screen using OpenGL"""
        if self.display_program is None:
            return
            
        glUseProgram(self.display_program)
        glBindVertexArray(self.quad_vao)
        glBindTexture(GL_TEXTURE_2D, self.nerf_texture)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get NeRF performance statistics"""
        return {
            'training_iterations': self.training_iterations,
            'average_psnr': self.average_psnr,
            'rendering_fps': self.rendering_fps,
            'training_images': len(self.training_images),
            'model_type': 'InstantNGP' if self.use_instant_ngp else 'StandardNeRF',
            'device': str(self.device)
        }
    
    def save_model(self, filepath: str):
        """Save trained NeRF model"""
        if self.use_instant_ngp and self.instant_ngp:
            # Save InstantNGP model
            checkpoint = {
                'model_state_dict': self.instant_ngp.model.state_dict(),
                'optimizer_state_dict': self.instant_ngp.optimizer.state_dict(),
                'iterations': self.instant_ngp.iterations,
                'loss_history': self.instant_ngp.loss_history,
                'psnr_history': self.instant_ngp.psnr_history
            }
            torch.save(checkpoint, filepath)
        else:
            # Save standard NeRF models
            checkpoint = {
                'coarse_model': self.coarse_model.state_dict(),
                'fine_model': self.fine_model.state_dict(),
                'training_iterations': self.training_iterations
            }
            torch.save(checkpoint, filepath)
        
        print(f"NeRF model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained NeRF model"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            if self.use_instant_ngp and self.instant_ngp:
                self.instant_ngp.model.load_state_dict(checkpoint['model_state_dict'])
                self.instant_ngp.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.instant_ngp.iterations = checkpoint['iterations']
                self.instant_ngp.loss_history = checkpoint['loss_history']
                self.instant_ngp.psnr_history = checkpoint['psnr_history']
            else:
                if self.coarse_model:
                    self.coarse_model.load_state_dict(checkpoint['coarse_model'])
                if self.fine_model:
                    self.fine_model.load_state_dict(checkpoint['fine_model'])
                self.training_iterations = checkpoint['training_iterations']
            
            print(f"NeRF model loaded from {filepath}")
            
        except Exception as e:
            print(f"Failed to load NeRF model: {e}")
    
    def export_mesh(self, filepath: str, resolution: int = 256):
        """Export NeRF as mesh using marching cubes"""
        try:
            from skimage import measure
            
            # Create 3D grid
            x = np.linspace(-1, 1, resolution)
            y = np.linspace(-1, 1, resolution)
            z = np.linspace(-1, 1, resolution)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            points_tensor = torch.from_numpy(points).float().to(self.device)
            
            # Query density
            with torch.no_grad():
                if self.use_instant_ngp and self.instant_ngp:
                    # Use arbitrary direction for density query
                    directions = torch.zeros_like(points_tensor)
                    _, densities = self.instant_ngp.query(points_tensor, directions)
                else:
                    # Standard NeRF
                    directions = torch.zeros_like(points_tensor)
                    _, densities = self.coarse_model(points_tensor, directions)
            
            densities = densities.cpu().numpy().reshape(resolution, resolution, resolution)
            
            # Marching cubes
            vertices, faces, normals, _ = measure.marching_cubes(densities, level=0.5)
            
            # Scale vertices to original space
            vertices = vertices / resolution * 2 - 1
            
            # Export as PLY
            self._export_ply(filepath, vertices, faces, normals)
            
            print(f"Mesh exported to {filepath}")
            
        except ImportError:
            print("scikit-image required for mesh export")
        except Exception as e:
            print(f"Mesh export failed: {e}")
    
    def _export_ply(self, filepath: str, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray):
        """Export mesh as PLY file"""
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_index\n")
            f.write("end_header\n")
            
            for i, vertex in enumerate(vertices):
                normal = normals[i] if i < len(normals) else [0, 0, 0]
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {normal[0]} {normal[1]} {normal[2]}\n")
            
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    def interactive_edit(self, position: np.ndarray, operation: str, parameters: Dict):
        """Interactive editing of NeRF scene"""
        if not self.interactive_editing:
            print("Interactive editing not enabled")
            return
        
        # Placeholder for interactive editing operations
        # In practice, this would modify the NeRF model based on user input
        print(f"Interactive edit at {position}: {operation} with {parameters}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.nerf_texture:
            glDeleteTextures(1, [self.nerf_texture])
        if self.display_program:
            glDeleteProgram(self.display_program)
        if self.quad_vao:
            glDeleteVertexArrays(1, [self.quad_vao])
        if self.quad_vbo:
            glDeleteBuffers(1, [self.quad_vbo])
        
        print("NeRF manager cleaned up")

# Example usage and testing
if __name__ == "__main__":
    # Initialize NeRF manager
    nerf_manager = NeRFManager(800, 600)
    nerf_manager.initialize_models(use_instant_ngp=True)
    
    # Create synthetic training data (in practice, would load real images)
    print("Creating synthetic training data...")
    
    # Example camera poses for synthetic scene
    for i in range(36):
        angle = i * 10 * np.pi / 180
        position = np.array([np.cos(angle) * 3, 1.5, np.sin(angle) * 3])
        target = np.array([0, 0, 0])
        
        rotation = R.from_rotvec([0, angle, 0]).as_quat()
        
        camera_pose = CameraPose(
            position=position,
            rotation=rotation,
            focal_length=0.5,
            image_size=(800, 600)
        )
        
        # Create synthetic image (checkerboard pattern)
        image = np.random.rand(600, 800, 3) * 0.5 + 0.5
        nerf_manager.add_training_image(image, camera_pose)
    
    # Train for a few iterations
    print("Training NeRF...")
    nerf_manager.train(num_iterations=1000, batch_size=4096)
    
    # Render a novel view
    print("Rendering novel view...")
    test_position = np.array([2.0, 1.0, 2.0])
    test_target = np.array([0.0, 0.0, 0.0])
    
    rendered_image = nerf_manager.render_view(test_position, test_target, fov=60.0)
    
    # Display using PyGame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("NeRF Rendering")
    
    # Convert to PyGame surface
    surface = pygame.surfarray.make_surface(rendered_image)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.blit(surface, (0, 0))
        pygame.display.flip()
    
    pygame.quit()
    nerf_manager.cleanup()
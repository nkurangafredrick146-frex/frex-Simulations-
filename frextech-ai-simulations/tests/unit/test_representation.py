"""
Unit tests for 3D representation models.
Tests NeRF, Gaussian Splatting, Mesh, and Voxel representations.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.representation.nerf.nerf_model import NeRFModel
from src.core.representation.nerf.ray_sampler import RaySampler
from src.core.representation.nerf.volume_renderer import VolumeRenderer
from src.core.representation.gaussian_splatting.gaussian_model import GaussianModel
from src.core.representation.gaussian_splatting.rasterizer import GaussianRasterizer
from src.core.representation.mesh.mesh_generator import MeshGenerator
from src.core.representation.mesh.mesh_refiner import MeshRefiner
from src.core.representation.voxel.voxel_grid import VoxelGrid


class TestNeRFModel:
    """Tests for NeRF model"""
    
    @pytest.fixture
    def nerf_model(self):
        """Create NeRFModel instance"""
        return NeRFModel(
            num_layers=8,
            hidden_dim=256,
            positional_encoding_dim=10,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_rays(self):
        """Create sample rays for testing"""
        batch_size = 4
        num_rays = 1024
        
        rays_o = torch.randn(batch_size, num_rays, 3)  # Ray origins
        rays_d = torch.randn(batch_size, num_rays, 3)  # Ray directions
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)  # Normalize directions
        
        return rays_o, rays_d
    
    @pytest.fixture
    def sample_points(self):
        """Create sample 3D points for testing"""
        batch_size = 2
        num_points = 1000
        
        points = torch.randn(batch_size, num_points, 3)  # 3D coordinates
        view_dirs = torch.randn(batch_size, num_points, 3)  # View directions
        view_dirs = torch.nn.functional.normalize(view_dirs, dim=-1)
        
        return points, view_dirs
    
    def test_nerf_model_initialization(self, nerf_model):
        """Test NeRFModel initialization"""
        assert nerf_model.num_layers == 8
        assert nerf_model.hidden_dim == 256
        assert nerf_model.positional_encoding_dim == 10
        assert nerf_model.device == "cpu"
        
        # Check model components
        assert hasattr(nerf_model, 'positional_encoding')
        assert hasattr(nerf_model, 'mlp')
        assert hasattr(nerf_model, 'density_head')
        assert hasattr(nerf_model, 'color_head')
    
    def test_positional_encoding(self, nerf_model, sample_points):
        """Test positional encoding"""
        points, _ = sample_points
        
        encoded = nerf_model.positional_encoding(points)
        
        # Verify encoding
        expected_dim = 3 * (2 * nerf_model.positional_encoding_dim)  # sin + cos for each dimension
        assert encoded.shape[-1] == expected_dim
        
        # Encoding should preserve batch and point dimensions
        assert encoded.shape[0] == points.shape[0]
        assert encoded.shape[1] == points.shape[1]
        
        # Encoded values should be bounded
        assert torch.all(encoded >= -1) and torch.all(encoded <= 1)
    
    def test_forward_pass(self, nerf_model, sample_points):
        """Test forward pass of NeRF model"""
        points, view_dirs = sample_points
        
        # Apply positional encoding
        points_encoded = nerf_model.positional_encoding(points)
        view_dirs_encoded = nerf_model.positional_encoding(view_dirs)
        
        # Forward pass
        output = nerf_model.forward(points_encoded, view_dirs_encoded)
        
        # Verify output structure
        assert "density" in output
        assert "color" in output
        assert "features" in output if nerf_model.output_features else True
        
        # Verify shapes
        batch_size, num_points, _ = points.shape
        assert output["density"].shape == (batch_size, num_points, 1)
        assert output["color"].shape == (batch_size, num_points, 3)
        
        # Density should be non-negative
        assert torch.all(output["density"] >= 0)
        
        # Colors should be in [0, 1] range (if using sigmoid)
        assert torch.all(output["color"] >= 0) and torch.all(output["color"] <= 1)
    
    def test_coarse_to_fine_sampling(self, nerf_model):
        """Test coarse-to-fine sampling strategy"""
        batch_size = 2
        num_coarse = 64
        num_fine = 128
        
        # Mock coarse samples
        coarse_samples = torch.randn(batch_size, num_coarse, 3)
        coarse_weights = torch.softmax(torch.randn(batch_size, num_coarse, 1), dim=1)
        
        # Generate fine samples
        fine_samples = nerf_model.sample_fine_points(coarse_samples, coarse_weights, num_fine)
        
        # Verify fine samples
        assert fine_samples.shape == (batch_size, num_fine, 3)
        
        # Fine samples should be near high-weight coarse samples
        # (difficult to test exactly without implementation details)
    
    def test_volume_rendering(self, nerf_model):
        """Test volume rendering from densities and colors"""
        batch_size = 2
        num_samples = 100
        
        # Create sample densities and colors
        densities = torch.rand(batch_size, num_samples, 1)  # Between 0 and 1
        colors = torch.rand(batch_size, num_samples, 3)     # RGB colors
        
        # Create sample distances along rays
        t_vals = torch.linspace(0, 1, num_samples).unsqueeze(0).unsqueeze(-1)
        t_vals = t_vals.repeat(batch_size, 1, 1)
        
        # Render
        rendered = nerf_model.render_volume(densities, colors, t_vals)
        
        # Verify rendering output
        assert "rgb" in rendered
        assert "depth" in rendered
        assert "alpha" in rendered
        assert "weights" in rendered
        
        # RGB should be in [0, 1] range
        assert torch.all(rendered["rgb"] >= 0) and torch.all(rendered["rgb"] <= 1)
        
        # Depth should be positive
        assert torch.all(rendered["depth"] >= 0)
        
        # Alpha (opacity) should be in [0, 1] range
        assert torch.all(rendered["alpha"] >= 0) and torch.all(rendered["alpha"] <= 1)
        
        # Weights should sum to alpha (or less for early termination)
        weights_sum = rendered["weights"].sum(dim=1)
        assert torch.allclose(weights_sum, rendered["alpha"], atol=1e-6)
    
    def test_hierarchical_sampling(self, nerf_model, sample_rays):
        """Test hierarchical sampling"""
        rays_o, rays_d = sample_rays
        batch_size, num_rays, _ = rays_o.shape
        
        # Hierarchical sampling
        samples = nerf_model.hierarchical_sampling(
            rays_o=rays_o,
            rays_d=rays_d,
            num_coarse=64,
            num_fine=128,
            near=0.1,
            far=10.0
        )
        
        # Verify samples structure
        assert "coarse_samples" in samples
        assert "fine_samples" in samples
        assert "coarse_weights" in samples
        assert "all_samples" in samples
        
        # Check sample counts
        assert samples["coarse_samples"].shape[1] == 64
        assert samples["fine_samples"].shape[1] == 128
        assert samples["all_samples"].shape[1] == 192  # Coarse + fine


class TestRaySampler:
    """Tests for ray sampler"""
    
    @pytest.fixture
    def ray_sampler(self):
        """Create RaySampler instance"""
        return RaySampler(
            image_height=256,
            image_width=256,
            focal_length=300.0
        )
    
    @pytest.fixture
    def sample_camera_pose(self):
        """Create sample camera pose"""
        # Identity pose (camera at origin looking along -z)
        pose = torch.eye(4, dtype=torch.float32)
        return pose.unsqueeze(0)  # Add batch dimension
    
    def test_ray_sampler_initialization(self, ray_sampler):
        """Test RaySampler initialization"""
        assert ray_sampler.image_height == 256
        assert ray_sampler.image_width == 256
        assert ray_sampler.focal_length == 300.0
        
        # Check that camera intrinsics are computed
        assert hasattr(ray_sampler, 'camera_matrix')
        assert ray_sampler.camera_matrix.shape == (3, 3)
    
    def test_generate_rays(self, ray_sampler, sample_camera_pose):
        """Test ray generation from camera pose"""
        batch_size = 2
        poses = sample_camera_pose.repeat(batch_size, 1, 1)
        
        rays = ray_sampler.generate_rays(poses)
        
        # Verify ray structure
        assert "rays_o" in rays  # Ray origins
        assert "rays_d" in rays  # Ray directions
        assert "pixel_coords" in rays  # Pixel coordinates
        
        # Verify shapes
        assert rays["rays_o"].shape == (batch_size, 256 * 256, 3)
        assert rays["rays_d"].shape == (batch_size, 256 * 256, 3)
        assert rays["pixel_coords"].shape == (256 * 256, 2)
        
        # Ray directions should be normalized
        ray_norms = torch.norm(rays["rays_d"], dim=-1)
        assert torch.allclose(ray_norms, torch.ones_like(ray_norms), atol=1e-6)
    
    def test_sample_pixels(self, ray_sampler):
        """Test pixel sampling strategies"""
        # Test random sampling
        num_samples = 1000
        random_pixels = ray_sampler.sample_pixels(num_samples, strategy="random")
        
        assert random_pixels.shape == (num_samples, 2)
        assert torch.all(random_pixels[:, 0] >= 0) and torch.all(random_pixels[:, 0] < 256)
        assert torch.all(random_pixels[:, 1] >= 0) and torch.all(random_pixels[:, 1] < 256)
        
        # Test grid sampling
        grid_pixels = ray_sampler.sample_pixels(num_samples, strategy="grid")
        
        # Grid samples should be evenly spaced
        unique_x = torch.unique(grid_pixels[:, 0])
        unique_y = torch.unique(grid_pixels[:, 1])
        assert len(unique_x) * len(unique_y) >= num_samples
    
    def test_sample_along_rays(self, ray_sampler):
        """Test sampling points along rays"""
        batch_size = 2
        num_rays = 100
        num_samples = 64
        
        # Create sample rays
        rays_o = torch.randn(batch_size, num_rays, 3)
        rays_d = torch.randn(batch_size, num_rays, 3)
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        
        # Sample points along rays
        points, t_vals = ray_sampler.sample_along_rays(
            rays_o=rays_o,
            rays_d=rays_d,
            num_samples=num_samples,
            near=0.1,
            far=10.0,
            strategy="linear"
        )
        
        # Verify shapes
        assert points.shape == (batch_size, num_rays, num_samples, 3)
        assert t_vals.shape == (batch_size, num_rays, num_samples, 1)
        
        # Points should lie along rays
        for b in range(batch_size):
            for r in range(num_rays):
                for s in range(num_samples):
                    expected_point = rays_o[b, r] + t_vals[b, r, s] * rays_d[b, r]
                    assert torch.allclose(points[b, r, s], expected_point, atol=1e-6)
    
    def test_importance_sampling(self, ray_sampler):
        """Test importance sampling based on density weights"""
        batch_size = 2
        num_rays = 50
        num_coarse = 32
        num_fine = 64
        
        # Mock coarse samples and weights
        coarse_t_vals = torch.linspace(0.1, 10.0, num_coarse).reshape(1, 1, -1, 1)
        coarse_t_vals = coarse_t_vals.repeat(batch_size, num_rays, 1, 1)
        
        coarse_weights = torch.softmax(torch.randn(batch_size, num_rays, num_coarse, 1), dim=2)
        
        # Importance sampling
        fine_t_vals = ray_sampler.importance_sampling(
            coarse_t_vals=coarse_t_vals,
            coarse_weights=coarse_weights,
            num_fine=num_fine
        )
        
        # Verify fine samples
        assert fine_t_vals.shape == (batch_size, num_rays, num_fine, 1)
        
        # Fine samples should be within coarse range
        assert torch.all(fine_t_vals >= 0.1) and torch.all(fine_t_vals <= 10.0)


class TestVolumeRenderer:
    """Tests for volume renderer"""
    
    @pytest.fixture
    def volume_renderer(self):
        """Create VolumeRenderer instance"""
        return VolumeRenderer(
            background_color=[0.0, 0.0, 0.0],  # Black background
            white_background=False
        )
    
    @pytest.fixture
    def sample_volume_data(self):
        """Create sample volume data for rendering"""
        batch_size = 2
        num_rays = 100
        num_samples = 64
        
        # Densities and colors
        densities = torch.rand(batch_size, num_rays, num_samples, 1)
        colors = torch.rand(batch_size, num_rays, num_samples, 3)
        
        # Sample distances
        t_vals = torch.linspace(0.1, 10.0, num_samples).reshape(1, 1, -1, 1)
        t_vals = t_vals.repeat(batch_size, num_rays, 1, 1)
        
        return densities, colors, t_vals
    
    def test_volume_renderer_initialization(self, volume_renderer):
        """Test VolumeRenderer initialization"""
        assert volume_renderer.background_color == [0.0, 0.0, 0.0]
        assert volume_renderer.white_background == False
        
        # Background tensor should be created
        assert hasattr(volume_renderer, 'background')
        assert volume_renderer.background.shape == (1, 1, 3)
    
    def test_compute_transmittance(self, volume_renderer, sample_volume_data):
        """Test transmittance computation"""
        densities, _, t_vals = sample_volume_data
        batch_size, num_rays, num_samples, _ = densities.shape
        
        # Compute distances between samples
        deltas = t_vals[..., 1:, :] - t_vals[..., :-1, :]
        deltas = torch.cat([deltas, torch.ones_like(deltas[..., :1, :]) * 1e10], dim=2)
        
        # Compute transmittance
        transmittance = volume_renderer.compute_transmittance(densities, deltas)
        
        # Verify transmittance
        assert transmittance.shape == (batch_size, num_rays, num_samples, 1)
        
        # Transmittance should decrease along ray (or stay same)
        for b in range(batch_size):
            for r in range(num_rays):
                for s in range(1, num_samples):
                    assert transmittance[b, r, s] <= transmittance[b, r, s-1]
        
        # Initial transmittance should be 1
        assert torch.allclose(transmittance[..., 0, :], torch.ones_like(transmittance[..., 0, :]))
    
    def test_render_volume(self, volume_renderer, sample_volume_data):
        """Test volume rendering"""
        densities, colors, t_vals = sample_volume_data
        
        rendered = volume_renderer.render(densities, colors, t_vals)
        
        # Verify rendering output
        assert "rgb" in rendered
        assert "depth" in rendered
        assert "alpha" in rendered
        assert "weights" in rendered
        
        # Shapes should match input rays
        batch_size, num_rays, num_samples, _ = densities.shape
        assert rendered["rgb"].shape == (batch_size, num_rays, 3)
        assert rendered["depth"].shape == (batch_size, num_rays, 1)
        assert rendered["alpha"].shape == (batch_size, num_rays, 1)
        assert rendered["weights"].shape == (batch_size, num_rays, num_samples, 1)
        
        # Alpha (opacity) should be in [0, 1]
        assert torch.all(rendered["alpha"] >= 0) and torch.all(rendered["alpha"] <= 1)
        
        # Weights should sum to alpha
        weights_sum = rendered["weights"].sum(dim=2)
        assert torch.allclose(weights_sum, rendered["alpha"], atol=1e-6)
    
    def test_accumulate_along_rays(self, volume_renderer):
        """Test accumulation along rays"""
        batch_size = 2
        num_rays = 50
        num_samples = 32
        
        # Create weights and values
        weights = torch.rand(batch_size, num_rays, num_samples, 1)
        weights = weights / weights.sum(dim=2, keepdim=True)  # Normalize
        
        values = torch.rand(batch_size, num_rays, num_samples, 3)
        
        # Accumulate
        accumulated = volume_renderer.accumulate_along_rays(weights, values)
        
        # Verify accumulation
        assert accumulated.shape == (batch_size, num_rays, 3)
        
        # Weighted sum should be bounded
        assert torch.all(accumulated >= 0) and torch.all(accumulated <= 1)
    
    def test_white_background_rendering(self, volume_renderer):
        """Test rendering with white background"""
        volume_renderer.white_background = True
        
        # Create simple volume data
        densities = torch.zeros(1, 1, 10, 1)  # Completely transparent
        colors = torch.rand(1, 1, 10, 3)
        t_vals = torch.linspace(0.1, 10.0, 10).reshape(1, 1, -1, 1)
        
        rendered = volume_renderer.render(densities, colors, t_vals)
        
        # With white background and transparent volume, should get white
        assert torch.allclose(rendered["rgb"], torch.ones_like(rendered["rgb"]), atol=1e-6)


class TestGaussianModel:
    """Tests for Gaussian splatting model"""
    
    @pytest.fixture
    def gaussian_model(self):
        """Create GaussianModel instance"""
        return GaussianModel(
            max_gaussians=10000,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_gaussians(self):
        """Create sample gaussians"""
        num_gaussians = 1000
        
        # Gaussian parameters
        means = torch.randn(num_gaussians, 3)  # Positions
        colors = torch.rand(num_gaussians, 3)   # RGB colors
        opacities = torch.rand(num_gaussians, 1)  # Opacities
        scales = torch.rand(num_gaussians, 3)   # Scale (size)
        rotations = torch.randn(num_gaussians, 4)  # Quaternions
        rotations = torch.nn.functional.normalize(rotations, dim=-1)
        
        return {
            "means": means,
            "colors": colors,
            "opacities": opacities,
            "scales": scales,
            "rotations": rotations
        }
    
    def test_gaussian_model_initialization(self, gaussian_model):
        """Test GaussianModel initialization"""
        assert gaussian_model.max_gaussians == 10000
        assert gaussian_model.device == "cpu"
        
        # Check that parameter tensors are initialized
        assert hasattr(gaussian_model, 'means')
        assert hasattr(gaussian_model, 'colors')
        assert hasattr(gaussian_model, 'opacities')
        assert hasattr(gaussian_model, 'scales')
        assert hasattr(gaussian_model, 'rotations')
    
    def test_gaussian_creation(self, gaussian_model, sample_gaussians):
        """Test creating gaussians from point cloud"""
        num_points = 500
        
        # Create point cloud
        points = torch.randn(num_points, 3)
        colors = torch.rand(num_points, 3)
        
        # Create gaussians from points
        gaussians = gaussian_model.create_from_points(points, colors)
        
        # Verify gaussian creation
        assert "means" in gaussians
        assert "colors" in gaussians
        assert "opacities" in gaussians
        assert "scales" in gaussians
        assert "rotations" in gaussians
        
        # Should have one gaussian per point
        assert gaussians["means"].shape[0] == num_points
        
        # Colors should be preserved
        assert torch.allclose(gaussians["colors"], colors, atol=1e-6)
    
    def test_gaussian_splatting(self, gaussian_model, sample_gaussians):
        """Test gaussian splatting to image"""
        # Create camera
        camera_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)  # Batch size 1
        camera_intrinsics = torch.tensor([[300, 0, 128], [0, 300, 128], [0, 0, 1]], dtype=torch.float32)
        camera_intrinsics = camera_intrinsics.unsqueeze(0)
        
        # Render resolution
        height, width = 256, 256
        
        # Splat gaussians
        with patch.object(gaussian_model, 'rasterizer') as mock_rasterizer:
            # Mock rasterizer output
            mock_output = {
                "image": torch.rand(1, height, width, 3),
                "depth": torch.rand(1, height, width, 1),
                "alpha": torch.rand(1, height, width, 1)
            }
            mock_rasterizer.rasterize.return_value = mock_output
            
            rendered = gaussian_model.render(
                camera_pose=camera_pose,
                camera_intrinsics=camera_intrinsics,
                height=height,
                width=width,
                gaussian_params=sample_gaussians
            )
            
            # Verify rendering output
            assert "image" in rendered
            assert "depth" in rendered
            assert "alpha" in rendered
            assert "gaussian_ids" in rendered
            
            # Verify shapes
            assert rendered["image"].shape == (1, height, width, 3)
            assert rendered["depth"].shape == (1, height, width, 1)
            assert rendered["alpha"].shape == (1, height, width, 1)
    
    def test_gaussian_optimization(self, gaussian_model):
        """Test gaussian parameter optimization"""
        num_gaussians = 100
        
        # Create target image
        target_image = torch.rand(1, 256, 256, 3)
        
        # Create initial gaussians
        initial_gaussians = {
            "means": torch.randn(num_gaussians, 3),
            "colors": torch.rand(num_gaussians, 3),
            "opacities": torch.rand(num_gaussians, 1),
            "scales": torch.rand(num_gaussians, 3),
            "rotations": torch.randn(num_gaussians, 4)
        }
        
        # Optimize
        with patch.object(gaussian_model, 'render') as mock_render:
            # Mock render to return differentiable output
            mock_render.return_value = {
                "image": torch.rand(1, 256, 256, 3),
                "depth": torch.rand(1, 256, 256, 1),
                "alpha": torch.rand(1, 256, 256, 1)
            }
            
            optimized = gaussian_model.optimize(
                initial_gaussians=initial_gaussians,
                target_image=target_image,
                num_iterations=10,
                learning_rate=0.01
            )
            
            # Verify optimization output
            assert "optimized_gaussians" in optimized
            assert "loss_history" in optimized
            assert "final_loss" in optimized
            
            # Loss should decrease (or at least be computed)
            assert optimized["final_loss"] >= 0


class TestGaussianRasterizer:
    """Tests for Gaussian rasterizer"""
    
    @pytest.fixture
    def gaussian_rasterizer(self):
        """Create GaussianRasterizer instance"""
        return GaussianRasterizer(
            tile_size=16,
            max_gaussians_per_tile=1024,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_gaussians_for_rasterization(self):
        """Create sample gaussians for rasterization"""
        num_gaussians = 500
        
        # Visible gaussians (in front of camera)
        means = torch.randn(num_gaussians, 3)
        means[:, 2] = means[:, 2].abs() + 5.0  # Ensure positive depth
        
        colors = torch.rand(num_gaussians, 3)
        opacities = torch.sigmoid(torch.randn(num_gaussians, 1))
        scales = torch.exp(torch.randn(num_gaussians, 3) * 0.5)  # Positive scales
        rotations = torch.randn(num_gaussians, 4)
        rotations = torch.nn.functional.normalize(rotations, dim=-1)
        
        return {
            "means": means,
            "colors": colors,
            "opacities": opacities,
            "scales": scales,
            "rotations": rotations
        }
    
    def test_gaussian_rasterizer_initialization(self, gaussian_rasterizer):
        """Test GaussianRasterizer initialization"""
        assert gaussian_rasterizer.tile_size == 16
        assert gaussian_rasterizer.max_gaussians_per_tile == 1024
        assert gaussian_rasterizer.device == "cpu"
    
    def test_project_gaussians(self, gaussian_rasterizer, sample_gaussians_for_rasterization):
        """Test projection of gaussians to image plane"""
        camera_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        camera_intrinsics = torch.tensor([[300, 0, 128], [0, 300, 128], [0, 0, 1]], dtype=torch.float32)
        camera_intrinsics = camera_intrinsics.unsqueeze(0)
        
        projected = gaussian_rasterizer.project_gaussians(
            gaussian_params=sample_gaussians_for_rasterization,
            camera_pose=camera_pose,
            camera_intrinsics=camera_intrinsics
        )
        
        # Verify projection output
        assert "image_coords" in projected  # 2D image coordinates
        assert "depths" in projected        # Depth values
        assert "covariances" in projected   # 2D covariance matrices
        assert "colors" in projected        # Colors (after view-dependent effects)
        assert "opacities" in projected     # Opacities
        
        # Image coordinates should be within reasonable bounds
        coords = projected["image_coords"]
        assert torch.all(coords[:, 0] >= -500) and torch.all(coords[:, 0] <= 500)  # x
        assert torch.all(coords[:, 1] >= -500) and torch.all(coords[:, 1] <= 500)  # y
        
        # Depths should be positive
        assert torch.all(projected["depths"] > 0)
    
    def test_tile_binning(self, gaussian_rasterizer):
        """Test binning gaussians into tiles"""
        height, width = 256, 256
        num_gaussians = 1000
        
        # Random image coordinates
        image_coords = torch.rand(num_gaussians, 2) * torch.tensor([width, height])
        
        # Tile binning
        tile_assignments = gaussian_rasterizer.bin_into_tiles(
            image_coords=image_coords,
            image_height=height,
            image_width=width
        )
        
        # Verify tile assignments
        assert "tile_indices" in tile_assignments
        assert "gaussian_indices" in tile_assignments
        assert "tile_bounds" in tile_assignments
        
        # Each gaussian should be assigned to a tile
        assert len(tile_assignments["gaussian_indices"]) == num_gaussians
        
        # Tile indices should be valid
        num_tiles_h = (height + gaussian_rasterizer.tile_size - 1) // gaussian_rasterizer.tile_size
        num_tiles_w = (width + gaussian_rasterizer.tile_size - 1) // gaussian_rasterizer.tile_size
        total_tiles = num_tiles_h * num_tiles_w
        
        assert torch.all(tile_assignments["tile_indices"] >= 0)
        assert torch.all(tile_assignments["tile_indices"] < total_tiles)
    
    def test_alpha_blending(self, gaussian_rasterizer):
        """Test alpha blending of gaussians"""
        num_gaussians = 100
        height, width = 32, 32
        
        # Create gaussian contributions
        colors = torch.rand(num_gaussians, 3)
        opacities = torch.rand(num_gaussians, 1)
        depths = torch.rand(num_gaussians, 1) * 10.0
        
        # Sort by depth (back to front for correct blending)
        sorted_indices = torch.argsort(depths, dim=0, descending=True).squeeze()
        
        # Alpha blending
        blended = gaussian_rasterizer.alpha_blend(
            colors=colors,
            opacities=opacities,
            depths=depths,
            image_height=height,
            image_width=width
        )
        
        # Verify blending output
        assert "image" in blended
        assert "alpha" in blended
        assert "depth" in blended
        
        # Output should be image-sized
        assert blended["image"].shape == (height, width, 3)
        assert blended["alpha"].shape == (height, width, 1)
        assert blended["depth"].shape == (height, width, 1)


class TestMeshGenerator:
    """Tests for mesh generator"""
    
    @pytest.fixture
    def mesh_generator(self):
        """Create MeshGenerator instance"""
        return MeshGenerator(
            method="marching_cubes",
            device="cpu"
        )
    
    @pytest.fixture
    def sample_sdf_grid(self):
        """Create sample signed distance field grid"""
        grid_size = 32
        
        # Create a sphere SDF
        grid = torch.zeros(1, grid_size, grid_size, grid_size)
        
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    # Distance from center
                    x = (i - grid_size/2) / (grid_size/2)
                    y = (j - grid_size/2) / (grid_size/2)
                    z = (k - grid_size/2) / (grid_size/2)
                    
                    distance = torch.sqrt(torch.tensor(x**2 + y**2 + z**2))
                    grid[0, i, j, k] = distance - 0.5  # Sphere of radius 0.5
        
        return grid
    
    def test_mesh_generator_initialization(self, mesh_generator):
        """Test MeshGenerator initialization"""
        assert mesh_generator.method == "marching_cubes"
        assert mesh_generator.device == "cpu"
    
    def test_marching_cubes(self, mesh_generator, sample_sdf_grid):
        """Test marching cubes mesh extraction"""
        mesh = mesh_generator.marching_cubes(
            sdf_grid=sample_sdf_grid,
            iso_value=0.0,
            spacing=(1.0, 1.0, 1.0)
        )
        
        # Verify mesh output
        assert "vertices" in mesh
        assert "faces" in mesh
        assert "normals" in mesh if mesh_generator.compute_normals else True
        
        # Should extract some vertices and faces
        assert len(mesh["vertices"]) > 0
        assert len(mesh["faces"]) > 0
        
        # Vertices should be 3D
        assert mesh["vertices"].shape[1] == 3
        
        # Faces should be triangles (3 vertices per face)
        assert mesh["faces"].shape[1] == 3
    
    def test_mesh_from_point_cloud(self, mesh_generator):
        """Test mesh generation from point cloud"""
        num_points = 1000
        
        # Create point cloud (sphere surface)
        points = torch.randn(num_points, 3)
        points = torch.nn.functional.normalize(points, dim=-1) * 0.5  # Radius 0.5
        
        # Generate mesh
        mesh = mesh_generator.from_point_cloud(
            points=points,
            method="poisson"  # or "ball_pivot", "alpha_shape"
        )
        
        # Verify mesh
        assert "vertices" in mesh
        assert "faces" in mesh
        
        # Should create connected mesh
        assert len(mesh["vertices"]) > 0
        assert len(mesh["faces"]) > 0
    
    def test_mesh_simplification(self, mesh_generator):
        """Test mesh simplification"""
        # Create a dense mesh (cube with subdivided faces)
        vertices = []
        faces = []
        
        # Simple cube
        cube_vertices = torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=torch.float32)
        
        cube_faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [2, 3, 7], [2, 7, 6],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 2, 6], [1, 6, 5]   # right
        ], dtype=torch.long)
        
        simplified = mesh_generator.simplify_mesh(
            vertices=cube_vertices,
            faces=cube_faces,
            target_faces=6  # Try to reduce to 6 faces (cube has 12)
        )
        
        # Verify simplification
        assert "vertices" in simplified
        assert "faces" in simplified
        
        # Should have fewer or equal faces
        assert len(simplified["faces"]) <= len(cube_faces)
        
        # Should preserve shape (approximately)
        simplified_vertices = simplified["vertices"]
        assert simplified_vertices.min() >= 0 and simplified_vertices.max() <= 1


class TestMeshRefiner:
    """Tests for mesh refiner"""
    
    @pytest.fixture
    def mesh_refiner(self):
        """Create MeshRefiner instance"""
        return MeshRefiner(
            num_iterations=10,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_mesh(self):
        """Create sample mesh for refinement"""
        # Simple cube mesh
        vertices = torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=torch.float32)
        
        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [2, 3, 7], [2, 7, 6],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 2, 6], [1, 6, 5]   # right
        ], dtype=torch.long)
        
        return {"vertices": vertices, "faces": faces}
    
    def test_mesh_refiner_initialization(self, mesh_refiner):
        """Test MeshRefiner initialization"""
        assert mesh_refiner.num_iterations == 10
        assert mesh_refiner.device == "cpu"
    
    def test_mesh_subdivision(self, mesh_refiner, sample_mesh):
        """Test mesh subdivision"""
        subdivided = mesh_refiner.subdivide_mesh(
            vertices=sample_mesh["vertices"],
            faces=sample_mesh["faces"],
            method="loop"  # Loop subdivision for triangles
        )
        
        # Verify subdivision
        assert "vertices" in subdivided
        assert "faces" in subdivided
        
        # Subdivision should increase number of faces
        assert len(subdivided["faces"]) > len(sample_mesh["faces"])
        
        # Should preserve overall shape
        subdivided_vertices = subdivided["vertices"]
        assert subdivided_vertices.min() >= 0 and subdivided_vertices.max() <= 1
    
    def test_mesh_smoothing(self, mesh_refiner, sample_mesh):
        """Test mesh smoothing"""
        # Add some noise to vertices
        noisy_vertices = sample_mesh["vertices"] + torch.randn_like(sample_mesh["vertices"]) * 0.1
        
        smoothed = mesh_refiner.smooth_mesh(
            vertices=noisy_vertices,
            faces=sample_mesh["faces"],
            method="laplacian",
            iterations=5
        )
        
        # Verify smoothing
        assert "vertices" in smoothed
        assert "faces" in smoothed
        
        # Smoothed mesh should be smoother than noisy mesh
        # (Can check by comparing vertex positions to original)
    
    def test_mesh_optimization(self, mesh_refiner, sample_mesh):
        """Test mesh optimization"""
        # Create target SDF (sphere)
        grid_size = 32
        target_sdf = torch.zeros(grid_size, grid_size, grid_size)
        
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    x = (i - grid_size/2) / (grid_size/2)
                    y = (j - grid_size/2) / (grid_size/2)
                    z = (k - grid_size/2) / (grid_size/2)
                    target_sdf[i, j, k] = torch.sqrt(torch.tensor(x**2 + y**2 + z**2)) - 0.5
        
        optimized = mesh_refiner.optimize_mesh(
            vertices=sample_mesh["vertices"],
            faces=sample_mesh["faces"],
            target_sdf=target_sdf,
            num_iterations=20
        )
        
        # Verify optimization
        assert "vertices" in optimized
        assert "faces" in optimized
        assert "loss_history" in optimized


class TestVoxelGrid:
    """Tests for voxel grid representation"""
    
    @pytest.fixture
    def voxel_grid(self):
        """Create VoxelGrid instance"""
        return VoxelGrid(
            resolution=32,
            device="cpu"
        )
    
    def test_voxel_grid_initialization(self, voxel_grid):
        """Test VoxelGrid initialization"""
        assert voxel_grid.resolution == 32
        assert voxel_grid.device == "cpu"
        
        # Grid should be initialized
        assert hasattr(voxel_grid, 'grid')
        assert voxel_grid.grid.shape == (1, 32, 32, 32)  # Batch, X, Y, Z
    
    def test_voxelization_from_points(self, voxel_grid):
        """Test voxelization from point cloud"""
        num_points = 1000
        
        # Point cloud (sphere surface)
        points = torch.randn(num_points, 3)
        points = torch.nn.functional.normalize(points, dim=-1) * 0.5  # Radius 0.5
        
        colors = torch.rand(num_points, 3)  # Point colors
        
        voxelized = voxel_grid.voxelize_points(
            points=points,
            colors=colors,
            method="trilinear"  # or "nearest"
        )
        
        # Verify voxelization
        assert "voxel_grid" in voxelized
        assert "color_grid" in voxelized if colors is not None else True
        
        # Grid should have values
        assert torch.any(voxelized["voxel_grid"] > 0)
    
    def test_voxel_raycasting(self, voxel_grid):
        """Test raycasting through voxel grid"""
        batch_size = 2
        num_rays = 100
        
        # Create rays
        rays_o = torch.zeros(batch_size, num_rays, 3)
        rays_o[:, :, 2] = -2.0  # Camera at z = -2
        
        rays_d = torch.zeros(batch_size, num_rays, 3)
        rays_d[:, :, 2] = 1.0  # Looking along +z
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        
        # Create a voxel grid with a cube
        grid = torch.zeros(1, 32, 32, 32)
        grid[0, 10:20, 10:20, 10:20] = 1.0  # Cube in center
        
        rendered = voxel_grid.raycast(
            rays_o=rays_o,
            rays_d=rays_d,
            voxel_grid=grid,
            step_size=0.05,
            max_steps=100
        )
        
        # Verify raycasting
        assert "rgb" in rendered
        assert "depth" in rendered
        assert "alpha" in rendered
        
        # Rays hitting the cube should have depth around 1.0
        # (camera at z=-2, cube at z=0.5 to 1.5 in normalized coordinates)
    
    def test_voxel_grid_operations(self, voxel_grid):
        """Test voxel grid operations (dilation, erosion, etc.)"""
        # Create binary grid with a small cube
        grid = torch.zeros(1, 32, 32, 32)
        grid[0, 14:18, 14:18, 14:18] = 1.0
        
        # Dilate
        dilated = voxel_grid.dilate(grid, iterations=2)
        
        # Erode
        eroded = voxel_grid.erode(grid, iterations=2)
        
        # Verify operations
        # Dilated grid should have more 1's
        assert torch.sum(dilated > 0) >= torch.sum(grid > 0)
        
        # Eroded grid should have fewer 1's
        assert torch.sum(eroded > 0) <= torch.sum(grid > 0)
    
    def test_voxel_to_mesh_conversion(self, voxel_grid):
        """Test conversion from voxel grid to mesh"""
        # Create voxel grid with a cube
        grid = torch.zeros(1, 32, 32, 32)
        grid[0, 10:20, 10:20, 10:20] = 1.0
        
        mesh = voxel_grid.to_mesh(
            voxel_grid=grid,
            iso_value=0.5,
            method="marching_cubes"
        )
        
        # Verify mesh conversion
        assert "vertices" in mesh
        assert "faces" in mesh
        
        # Should extract mesh for the cube
        assert len(mesh["vertices"]) > 0
        assert len(mesh["faces"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

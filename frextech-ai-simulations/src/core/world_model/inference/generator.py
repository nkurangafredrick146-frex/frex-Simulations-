"""
World Generation Inference Module
Main generator for 3D scene synthesis from text prompts and other modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json
from tqdm import tqdm
import logging
from datetime import datetime

from ..architecture.diffusion_models import DiffusionModel3D
from ..architecture.custom_layers import (
    FourierFeature3D, 
    MultiResolutionHashEncoding,
    SceneGraphAttention
)

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for world generation"""
    resolution: Tuple[int, int, int] = (64, 64, 64)
    latent_channels: int = 4  # RGB + density
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    seed: Optional[int] = None
    batch_size: int = 1
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    use_compile: bool = False
    enable_progress_bar: bool = True
    cache_intermediates: bool = False

@dataclass
class WorldGenerationRequest:
    """Input request for world generation"""
    prompt: str
    negative_prompt: Optional[str] = None
    style_reference: Optional[str] = None  # Path to style image/video
    world_dimensions: Tuple[float, float, float] = (10.0, 10.0, 10.0)  # Meters
    coordinate_system: str = "carla"  # or "unreal", "unity", "blender"
    num_samples: int = 1
    quality: str = "medium"  # "draft", "medium", "high", "ultra"
    seed: Optional[int] = None
    output_format: str = "nerf"  # "nerf", "gaussian", "mesh", "voxel"
    metadata: Optional[Dict[str, Any]] = None

class WorldGenerator:
    """Main world generation class"""
    
    def __init__(
        self,
        model: DiffusionModel3D,
        text_encoder: nn.Module,
        config: GenerationConfig,
        model_dir: Optional[Path] = None
    ):
        self.model = model
        self.text_encoder = text_encoder
        self.config = config
        self.model_dir = model_dir
        
        # Move to device and set dtype
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device, dtype=config.dtype)
        self.text_encoder = self.text_encoder.to(self.device, dtype=config.dtype)
        
        # Compile if requested
        if config.use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Compilation failed: {e}")
        
        # Set evaluation mode
        self.model.eval()
        self.text_encoder.eval()
        
        # Caches
        self.text_embeddings_cache = {}
        self.intermediate_cache = {}
        
        # Initialize sampling scheduler
        self.scheduler = self._create_scheduler()
        
        # Metrics
        self.generation_times = []
        self.successful_generations = 0
        
    def _create_scheduler(self):
        """Create diffusion scheduler for inference"""
        from .sampler import DDIMScheduler3D, DPMSolverMultistepScheduler3D
        
        if self.config.num_inference_steps <= 30:
            return DDIMScheduler3D(
                num_train_timesteps=1000,
                num_inference_steps=self.config.num_inference_steps,
                beta_schedule="linear"
            )
        else:
            return DPMSolverMultistepScheduler3D(
                num_train_timesteps=1000,
                num_inference_steps=self.config.num_inference_steps,
                beta_schedule="linear",
                algorithm_type="dpmsolver++",
                solver_order=3
            )
    
    def encode_text(self, prompt: str, negative_prompt: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Encode text prompt into embeddings"""
        cache_key = f"{prompt}_{negative_prompt}"
        if cache_key in self.text_embeddings_cache:
            return self.text_embeddings_cache[cache_key]
        
        with torch.no_grad():
            # Encode positive prompt
            pos_embeddings = self.text_encoder(prompt)
            
            # Encode negative prompt if provided
            if negative_prompt:
                neg_embeddings = self.text_encoder(negative_prompt)
            else:
                # Use unconditional embedding
                neg_embeddings = self.text_encoder("")
            
            result = {
                "prompt_embeds": pos_embeddings,
                "negative_prompt_embeds": neg_embeddings,
                "pooled_prompt_embeds": pos_embeddings.mean(dim=1, keepdim=True)
            }
            
            # Cache result
            self.text_embeddings_cache[cache_key] = {
                k: v.cpu() if v is not None else None 
                for k, v in result.items()
            }
            
            return result
    
    def _prepare_latents(
        self, 
        shape: Tuple[int, ...],
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """Prepare initial noise latents"""
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        noise = torch.randn(shape, device=self.device, dtype=self.config.dtype)
        
        # Optionally scale noise for better results
        if self.config.quality in ["high", "ultra"]:
            # Apply slight Gaussian blur for smoother initial noise
            noise = self._smooth_noise_3d(noise)
        
        return noise
    
    def _smooth_noise_3d(self, noise: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        """Apply 3D Gaussian smoothing to noise"""
        # Create 3D Gaussian kernel
        kernel_size = 5
        kernel = self._gaussian_kernel_3d(kernel_size, sigma).to(noise.device, dtype=noise.dtype)
        
        # Apply convolution to each channel separately
        smoothed = torch.zeros_like(noise)
        for c in range(noise.shape[1]):
            smoothed[:, c:c+1] = F.conv3d(
                noise[:, c:c+1], 
                kernel, 
                padding=kernel_size//2
            )
        
        return smoothed
    
    def _gaussian_kernel_3d(self, size: int, sigma: float) -> torch.Tensor:
        """Create 3D Gaussian kernel"""
        coords = torch.arange(size) - size // 2
        x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
        kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def generate_latents(
        self,
        text_embeddings: Dict[str, torch.Tensor],
        latents_shape: Tuple[int, ...],
        seed: Optional[int] = None,
        callback: Optional[callable] = None
    ) -> torch.Tensor:
        """Generate latents using diffusion model"""
        latents = self._prepare_latents(latents_shape, seed)
        
        # Prepare scheduler
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        
        # Generation loop
        progress_bar = tqdm(
            self.scheduler.timesteps, 
            desc="Generating latents",
            disable=not self.config.enable_progress_bar
        )
        
        for i, t in enumerate(progress_bar):
            # Expand latents for batch processing if needed
            latent_model_input = latents
            t_batch = t.expand(latents.shape[0])
            
            # Classifier-free guidance
            if self.config.guidance_scale > 1.0:
                # Double the batch to do unconditional + conditional
                latent_model_input = torch.cat([latent_model_input] * 2)
                t_batch = torch.cat([t_batch] * 2)
                
                # Get model output for both
                noise_pred = self.model(
                    latent_model_input,
                    t_batch,
                    text_embeddings["prompt_embeds"]
                )
                
                # Split predictions
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                
                # Apply guidance
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                # No guidance
                noise_pred = self.model(
                    latent_model_input,
                    t_batch,
                    text_embeddings["prompt_embeds"]
                )
            
            # Compute previous noisy sample
            latents = self.scheduler.step(
                noise_pred, 
                t, 
                latents,
                return_dict=False
            )[0]
            
            # Cache intermediate if requested
            if self.config.cache_intermediates and i % 5 == 0:
                self.intermediate_cache[f"step_{i}"] = latents.cpu()
            
            # Callback for progress monitoring
            if callback:
                callback(i, latents)
        
        return latents
    
    def generate_world(
        self,
        request: WorldGenerationRequest,
        return_latents: bool = False,
        return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """Generate complete 3D world from request"""
        start_time = datetime.now()
        
        try:
            # Encode text prompts
            logger.info(f"Encoding prompt: {request.prompt[:50]}...")
            text_embeddings = self.encode_text(
                request.prompt, 
                request.negative_prompt
            )
            
            # Move embeddings to device
            for k, v in text_embeddings.items():
                if v is not None:
                    text_embeddings[k] = v.to(self.device, dtype=self.config.dtype)
            
            # Determine output resolution based on quality
            resolution = self._get_resolution_for_quality(request.quality)
            
            # Generate latents
            logger.info(f"Generating latents at resolution {resolution}")
            latents_shape = (
                request.num_samples,
                self.config.latent_channels,
                *resolution
            )
            
            latents = self.generate_latents(
                text_embeddings,
                latents_shape,
                seed=request.seed
            )
            
            # Decode latents to 3D representation
            logger.info(f"Decoding to {request.output_format} format")
            world_representation = self.decode_latents(
                latents,
                output_format=request.output_format,
                world_dimensions=request.world_dimensions
            )
            
            # Calculate metrics
            generation_time = (datetime.now() - start_time).total_seconds()
            self.generation_times.append(generation_time)
            self.successful_generations += 1
            
            # Prepare result
            result = {
                "world": world_representation,
                "latents": latents if return_latents else None,
                "intermediates": self.intermediate_cache if return_intermediates else None,
                "metadata": {
                    "generation_time": generation_time,
                    "resolution": resolution,
                    "format": request.output_format,
                    "seed": request.seed,
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "dimensions": request.world_dimensions,
                    "quality": request.quality,
                    "timestamp": start_time.isoformat()
                },
                "success": True
            }
            
            logger.info(f"Generation successful in {generation_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            
            return {
                "world": None,
                "latents": None,
                "intermediates": None,
                "metadata": {
                    "error": str(e),
                    "timestamp": start_time.isoformat()
                },
                "success": False
            }
    
    def _get_resolution_for_quality(self, quality: str) -> Tuple[int, int, int]:
        """Get resolution based on quality setting"""
        base_res = self.config.resolution
        
        if quality == "draft":
            return tuple(max(1, d // 4) for d in base_res)
        elif quality == "medium":
            return tuple(max(1, d // 2) for d in base_res)
        elif quality == "high":
            return base_res
        elif quality == "ultra":
            return tuple(d * 2 for d in base_res)
        else:
            return base_res
    
    def decode_latents(
        self,
        latents: torch.Tensor,
        output_format: str = "nerf",
        world_dimensions: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    ) -> Union[Dict[str, Any], np.ndarray]:
        """Decode latents to specific 3D representation format"""
        
        if output_format == "nerf":
            return self._decode_to_nerf(latents, world_dimensions)
        elif output_format == "gaussian":
            return self._decode_to_gaussian(latents, world_dimensions)
        elif output_format == "mesh":
            return self._decode_to_mesh(latents, world_dimensions)
        elif output_format == "voxel":
            return self._decode_to_voxel(latents, world_dimensions)
        else:
            raise ValueError(f"Unknown output format: {output_format}")
    
    def _decode_to_nerf(
        self, 
        latents: torch.Tensor, 
        world_dimensions: Tuple[float, float, float]
    ) -> Dict[str, Any]:
        """Convert latents to NeRF representation"""
        from ...representation.nerf.nerf_model import NeRFModel
        
        # Split latents into density and color
        density = latents[:, 0:1]  # First channel
        color = latents[:, 1:4]    # Next 3 channels for RGB
        
        # Normalize and process
        density = torch.sigmoid(density)
        color = torch.sigmoid(color)
        
        # Create coordinate grid
        D, H, W = latents.shape[2:]
        grid = self._create_coordinate_grid(D, H, W, world_dimensions)
        
        # Create NeRF representation
        nerf_data = {
            "density_grid": density.squeeze().cpu().numpy(),
            "color_grid": color.squeeze().cpu().numpy(),
            "coordinates": grid.cpu().numpy(),
            "resolution": (D, H, W),
            "world_dimensions": world_dimensions,
            "bounds": {
                "x_min": -world_dimensions[0] / 2,
                "x_max": world_dimensions[0] / 2,
                "y_min": -world_dimensions[1] / 2,
                "y_max": world_dimensions[1] / 2,
                "z_min": -world_dimensions[2] / 2,
                "z_max": world_dimensions[2] / 2
            }
        }
        
        return nerf_data
    
    def _decode_to_gaussian(
        self,
        latents: torch.Tensor,
        world_dimensions: Tuple[float, float, float]
    ) -> Dict[str, Any]:
        """Convert latents to 3D Gaussian Splatting representation"""
        from ...representation.gaussian_splatting.gaussian_model import GaussianModel
        
        # Extract gaussian parameters from latents
        B, C, D, H, W = latents.shape
        
        # Reshape to per-voxel gaussians
        positions = self._create_coordinate_grid(D, H, W, world_dimensions)
        
        # Extract parameters (simplified - in practice would use learned mapping)
        # Here we use different channels for different gaussian parameters
        scales = torch.sigmoid(latents[:, 0:3]) * 0.1  # Scale in [0, 0.1]
        rotations = torch.tanh(latents[:, 3:7])  # Quaternion-like representation
        opacities = torch.sigmoid(latents[:, 7:8])  # Opacity
        sh_coeffs = latents[:, 8:].reshape(B, -1, 16, 3)  # Spherical harmonics
        
        # Normalize rotations to valid quaternions
        rotations = F.normalize(rotations, dim=1)
        
        gaussian_data = {
            "positions": positions.cpu().numpy(),
            "scales": scales.squeeze().cpu().numpy(),
            "rotations": rotations.squeeze().cpu().numpy(),
            "opacities": opacities.squeeze().cpu().numpy(),
            "sh_coeffs": sh_coeffs.squeeze().cpu().numpy(),
            "num_gaussians": D * H * W,
            "world_dimensions": world_dimensions
        }
        
        return gaussian_data
    
    def _decode_to_mesh(
        self,
        latents: torch.Tensor,
        world_dimensions: Tuple[float, float, float]
    ) -> Dict[str, Any]:
        """Convert latents to mesh representation using marching cubes"""
        import mcubes
        
        # Extract density field
        density = torch.sigmoid(latents[:, 0:1]).squeeze().cpu().numpy()
        
        # Extract color field
        color = torch.sigmoid(latents[:, 1:4]).squeeze().cpu().numpy()
        
        # Apply marching cubes
        threshold = 0.5
        vertices, triangles = mcubes.marching_cubes(density, threshold)
        
        # Normalize vertices to world coordinates
        D, H, W = density.shape
        vertices = vertices / np.array([D-1, H-1, W-1]) * np.array(world_dimensions)
        vertices = vertices - np.array(world_dimensions) / 2
        
        # Sample colors at vertex positions
        vertex_colors = []
        for v in vertices:
            # Convert back to grid coordinates
            grid_pos = (v + np.array(world_dimensions) / 2) / np.array(world_dimensions)
            grid_pos = grid_pos * np.array([D-1, H-1, W-1])
            grid_pos = np.clip(grid_pos.astype(int), 0, [D-1, H-1, W-1])
            
            # Sample color
            color_sample = color[:, grid_pos[0], grid_pos[1], grid_pos[2]]
            vertex_colors.append(color_sample)
        
        vertex_colors = np.array(vertex_colors)
        
        mesh_data = {
            "vertices": vertices,
            "triangles": triangles,
            "vertex_colors": vertex_colors,
            "num_vertices": len(vertices),
            "num_faces": len(triangles),
            "world_dimensions": world_dimensions,
            "density_threshold": threshold
        }
        
        return mesh_data
    
    def _decode_to_voxel(
        self,
        latents: torch.Tensor,
        world_dimensions: Tuple[float, float, float]
    ) -> np.ndarray:
        """Convert latents to voxel grid"""
        # Simple thresholding of density channel
        density = torch.sigmoid(latents[:, 0:1])
        color = torch.sigmoid(latents[:, 1:4])
        
        # Threshold to get occupancy
        occupancy = (density > 0.5).float()
        
        # Combine occupancy with color
        voxel_grid = torch.cat([occupancy, color], dim=1)
        
        return voxel_grid.squeeze().cpu().numpy()
    
    def _create_coordinate_grid(
        self,
        D: int,
        H: int,
        W: int,
        world_dimensions: Tuple[float, float, float]
    ) -> torch.Tensor:
        """Create normalized coordinate grid"""
        device = self.device
        
        # Create grid in normalized coordinates [-1, 1]
        x = torch.linspace(-1, 1, W, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        z = torch.linspace(-1, 1, D, device=device)
        
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # [D, H, W, 3]
        
        # Scale to world dimensions
        world_scale = torch.tensor(world_dimensions, device=device).view(1, 1, 1, 3) / 2
        grid = grid * world_scale
        
        return grid
    
    def batch_generate(
        self,
        requests: List[WorldGenerationRequest],
        max_batch_size: int = 4,
        parallel: bool = False
    ) -> List[Dict[str, Any]]:
        """Generate multiple worlds in batch"""
        results = []
        
        if parallel and torch.cuda.device_count() > 1:
            # Parallel generation across GPUs
            results = self._parallel_generate(requests, max_batch_size)
        else:
            # Sequential batching
            for i in range(0, len(requests), max_batch_size):
                batch_requests = requests[i:i + max_batch_size]
                
                # TODO: Implement true batch generation
                # For now, process sequentially
                for req in batch_requests:
                    result = self.generate_world(req)
                    results.append(result)
        
        return results
    
    def _parallel_generate(
        self,
        requests: List[WorldGenerationRequest],
        max_batch_size: int
    ) -> List[Dict[str, Any]]:
        """Generate worlds in parallel across multiple GPUs"""
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor
        
        num_gpus = torch.cuda.device_count()
        num_requests = len(requests)
        
        # Split requests across GPUs
        requests_per_gpu = (num_requests + num_gpus - 1) // num_gpus
        
        results = []
        
        def generate_on_gpu(gpu_id: int, gpu_requests: List[WorldGenerationRequest]):
            """Generate on specific GPU"""
            torch.cuda.set_device(gpu_id)
            
            # Create generator for this GPU
            gpu_generator = WorldGenerator(
                model=self.model,
                text_encoder=self.text_encoder,
                config=self.config
            )
            gpu_generator.device = torch.device(f"cuda:{gpu_id}")
            
            gpu_results = []
            for req in gpu_requests:
                result = gpu_generator.generate_world(req)
                gpu_results.append(result)
            
            return gpu_results
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            
            for gpu_id in range(num_gpus):
                start_idx = gpu_id * requests_per_gpu
                end_idx = min(start_idx + requests_per_gpu, num_requests)
                
                if start_idx < num_requests:
                    gpu_requests = requests[start_idx:end_idx]
                    future = executor.submit(generate_on_gpu, gpu_id, gpu_requests)
                    futures.append(future)
            
            # Collect results
            for future in futures:
                results.extend(future.result())
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        if not self.generation_times:
            return {"total_generations": 0}
        
        times = np.array(self.generation_times)
        
        return {
            "total_generations": self.successful_generations,
            "average_time": float(times.mean()),
            "min_time": float(times.min()),
            "max_time": float(times.max()),
            "std_time": float(times.std()),
            "success_rate": self.successful_generations / len(self.generation_times) 
            if self.generation_times else 0
        }
    
    def clear_cache(self):
        """Clear generation caches"""
        self.text_embeddings_cache.clear()
        self.intermediate_cache.clear()
        torch.cuda.empty_cache()
    
    def save_checkpoint(self, path: Path):
        """Save generator state"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "text_encoder_state_dict": self.text_encoder.state_dict(),
            "config": self.config,
            "statistics": self.get_statistics(),
            "version": "1.0.0"
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    @classmethod
    def from_checkpoint(
        cls, 
        checkpoint_path: Path,
        config: Optional[GenerationConfig] = None
    ) -> "WorldGenerator":
        """Load generator from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Recreate model architecture
        from ..architecture.diffusion_models import DiffusionModel3D, UNet3D
        
        # Extract model config from checkpoint
        model_config = checkpoint.get("model_config", {})
        
        # Create model
        unet = UNet3D(**model_config.get("unet_config", {}))
        model = DiffusionModel3D(**model_config.get("diffusion_config", {}))
        model.unet = unet
        
        # Create dummy text encoder (should be loaded from actual checkpoint)
        class DummyTextEncoder(nn.Module):
            def __init__(self, embed_dim=768):
                super().__init__()
                self.embed_dim = embed_dim
                self.proj = nn.Linear(512, embed_dim)
            
            def forward(self, text):
                batch_size = len(text) if isinstance(text, list) else 1
                return torch.randn(batch_size, 77, self.embed_dim)
        
        text_encoder = DummyTextEncoder()
        
        # Create generator
        config = config or checkpoint["config"]
        generator = cls(model, text_encoder, config)
        
        # Load weights
        generator.model.load_state_dict(checkpoint["model_state_dict"])
        generator.text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"])
        
        logger.info(f"Generator loaded from {checkpoint_path}")
        
        return generator

class ProgressiveGenerator(WorldGenerator):
    """Generator with progressive refinement"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refinement_stages = []
        
    def generate_progressive(
        self,
        request: WorldGenerationRequest,
        stages: List[str] = ["coarse", "medium", "fine"]
    ) -> List[Dict[str, Any]]:
        """Generate world with progressive refinement"""
        results = []
        
        for stage in stages:
            # Adjust quality for stage
            stage_request = self._create_stage_request(request, stage)
            
            # Generate at this stage
            logger.info(f"Generating at {stage} stage")
            result = self.generate_world(stage_request)
            
            if not result["success"]:
                logger.error(f"Stage {stage} generation failed")
                break
            
            results.append(result)
            
            # If not last stage, use result to initialize next stage
            if stage != stages[-1]:
                self._prepare_for_next_stage(result, stages[stages.index(stage) + 1])
        
        return results
    
    def _create_stage_request(
        self,
        base_request: WorldGenerationRequest,
        stage: str
    ) -> WorldGenerationRequest:
        """Create request for specific refinement stage"""
        # Map stage to quality
        stage_quality = {
            "coarse": "draft",
            "medium": "medium",
            "fine": "high",
            "ultra": "ultra"
        }
        
        return WorldGenerationRequest(
            prompt=base_request.prompt,
            negative_prompt=base_request.negative_prompt,
            style_reference=base_request.style_reference,
            world_dimensions=base_request.world_dimensions,
            coordinate_system=base_request.coordinate_system,
            num_samples=base_request.num_samples,
            quality=stage_quality.get(stage, "medium"),
            seed=base_request.seed,
            output_format=base_request.output_format,
            metadata=base_request.metadata
        )
    
    def _prepare_for_next_stage(self, current_result: Dict[str, Any], next_stage: str):
        """Prepare for next refinement stage"""
        # In a real implementation, this would upsample and refine
        # the current result to initialize the next stage
        pass
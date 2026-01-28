"""
Performance benchmarks for inference operations.
Measures latency, throughput, memory usage, and GPU performance.
"""

import pytest
import time
import asyncio
import json
import statistics
import tracemalloc
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import torch
from torch.cuda import memory_allocated, memory_reserved, max_memory_allocated

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.world_model.inference.generator import WorldGenerator
from src.core.world_model.inference.sampler import Sampler
from src.core.multimodal.encoders.text_encoder import TextEncoder
from src.core.multimodal.encoders.vision_encoder import VisionEncoder
from src.core.representation.nerf.nerf_model import NeRFModel
from src.core.representation.gaussian_splatting.gaussian_model import GaussianModel
from src.render.engines.webgl_engine import WebGLEngine
from src.api.utils.cache_manager import CacheManager


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    operation: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_time: float
    throughput: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float] = None
    cpu_percent: float = 0.0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "operation": self.operation,
            "iterations": self.iterations,
            "timing": {
                "total_seconds": self.total_time,
                "avg_seconds": self.avg_time,
                "min_seconds": self.min_time,
                "max_seconds": self.max_time,
                "std_seconds": self.std_time
            },
            "throughput": self.throughput,
            "memory_usage_mb": self.memory_usage_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "cpu_percent": self.cpu_percent,
            "error_rate": self.error_rate
        }


class PerformanceBenchmark:
    """Base class for performance benchmarks"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    @contextmanager
    def measure_time(self):
        """Context manager for timing operations"""
        start_time = time.perf_counter()
        start_cpu = psutil.Process().cpu_percent()
        tracemalloc.start()
        
        yield
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        end_time = time.perf_counter()
        end_cpu = psutil.Process().cpu_percent()
        
        elapsed = end_time - start_time
        memory_used = peak / 1024 / 1024  # Convert to MB
        cpu_used = max(start_cpu, end_cpu)
        
        return elapsed, memory_used, cpu_used
    
    @contextmanager
    def measure_gpu_memory(self, device_id: int = 0):
        """Context manager for GPU memory measurement"""
        if torch.cuda.is_available():
            torch.cuda.synchronize(device_id)
            start_memory = memory_allocated(device_id)
            start_reserved = memory_reserved(device_id)
            
            yield
            
            torch.cuda.synchronize(device_id)
            end_memory = memory_allocated(device_id)
            end_reserved = memory_reserved(device_id)
            peak_memory = max_memory_allocated(device_id)
            
            memory_increase = (end_memory - start_memory) / 1024 / 1024  # MB
            peak_usage = peak_memory / 1024 / 1024  # MB
            
            return memory_increase, peak_usage
        else:
            yield
            return 0.0, 0.0
    
    def run_benchmark(self, operation_func, iterations: int = 10, 
                     warmup_iterations: int = 2, **kwargs) -> BenchmarkResult:
        """Run benchmark for a specific operation"""
        # Warmup
        for _ in range(warmup_iterations):
            try:
                operation_func(**kwargs)
            except:
                pass
        
        # Clear any cached memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run benchmark iterations
        times = []
        errors = 0
        
        for i in range(iterations):
            try:
                with self.measure_time() as (elapsed, memory, cpu):
                    result = operation_func(**kwargs)
                
                times.append(elapsed)
                
                # Record memory on first iteration
                if i == 0:
                    memory_used = memory
                    cpu_used = cpu
                
            except Exception as e:
                errors += 1
                times.append(0)  # Use 0 for failed iterations
                print(f"Error in iteration {i}: {e}")
        
        # Calculate statistics
        valid_times = [t for t in times if t > 0]
        
        if valid_times:
            avg_time = statistics.mean(valid_times)
            min_time = min(valid_times)
            max_time = max(valid_times)
            std_time = statistics.stdev(valid_times) if len(valid_times) > 1 else 0
            total_time = sum(valid_times)
            throughput = len(valid_times) / total_time if total_time > 0 else 0
        else:
            avg_time = min_time = max_time = std_time = total_time = throughput = 0
        
        error_rate = errors / iterations
        
        # Get operation name
        operation_name = operation_func.__name__ if hasattr(operation_func, '__name__') else "unknown"
        
        result = BenchmarkResult(
            operation=operation_name,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_time=std_time,
            throughput=throughput,
            memory_usage_mb=memory_used,
            cpu_percent=cpu_used,
            error_rate=error_rate
        )
        
        self.results.append(result)
        return result
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to file"""
        results_dict = {
            "timestamp": time.time(),
            "system_info": self.get_system_info(),
            "benchmarks": [r.to_dict() for r in self.results]
        }
        
        output_file = self.output_dir / filename
        output_file.write_text(json.dumps(results_dict, indent=2))
        print(f"Results saved to {output_file}")
    
    def get_system_info(self) -> Dict:
        """Get system information for benchmark context"""
        import platform
        import cpuinfo
        
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu": cpuinfo.get_cpu_info()["brand_raw"],
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "memory_available_gb": psutil.virtual_memory().available / 1024**3,
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_available": True,
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "gpu_count": torch.cuda.device_count()
            })
        else:
            info["cuda_available"] = False
        
        return info
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        
        for result in self.results:
            print(f"\nOperation: {result.operation}")
            print(f"  Iterations: {result.iterations}")
            print(f"  Avg Time: {result.avg_time:.4f}s ± {result.std_time:.4f}s")
            print(f"  Min/Max: {result.min_time:.4f}s / {result.max_time:.4f}s")
            print(f"  Throughput: {result.throughput:.2f} ops/sec")
            print(f"  Memory: {result.memory_usage_mb:.2f} MB")
            if result.gpu_memory_mb:
                print(f"  GPU Memory: {result.gpu_memory_mb:.2f} MB")
            print(f"  CPU Usage: {result.cpu_percent:.1f}%")
            print(f"  Error Rate: {result.error_rate:.1%}")


class InferenceBenchmark(PerformanceBenchmark):
    """Benchmarks for inference operations"""
    
    def __init__(self, output_dir: Path = None):
        super().__init__(output_dir)
        self.setup_test_data()
    
    def setup_test_data(self):
        """Setup test data for benchmarks"""
        # Text prompts
        self.test_prompts = [
            "A beautiful mountain landscape",
            "A futuristic city at night",
            "An underwater coral reef",
            "A desert oasis with palm trees",
            "A medieval castle on a hill"
        ]
        
        # Test images
        self.test_images = [
            np.random.rand(256, 256, 3).astype(np.float32) for _ in range(5)
        ]
        
        # Test batch sizes
        self.batch_sizes = [1, 2, 4, 8, 16]
        
        # Test resolutions
        self.resolutions = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    
    def benchmark_text_encoding(self, model_name: str = "clip-text"):
        """Benchmark text encoding performance"""
        print("\n" + "="*80)
        print("BENCHMARK: TEXT ENCODING")
        print("="*80)
        
        # Initialize encoder
        encoder = TextEncoder(model_name=model_name)
        
        # Benchmark single prompt
        def encode_single(prompt):
            return encoder.encode([prompt], normalize=True)
        
        result = self.run_benchmark(
            encode_single,
            iterations=20,
            warmup_iterations=3,
            prompt=self.test_prompts[0]
        )
        
        # Benchmark batch encoding
        for batch_size in self.batch_sizes:
            if batch_size <= len(self.test_prompts):
                prompts = self.test_prompts[:batch_size]
                
                def encode_batch():
                    return encoder.encode(prompts, normalize=True)
                
                batch_result = self.run_benchmark(
                    encode_batch,
                    iterations=10,
                    warmup_iterations=2
                )
                batch_result.operation = f"text_encode_batch_{batch_size}"
    
    def benchmark_vision_encoding(self, model_name: str = "clip-vision"):
        """Benchmark vision encoding performance"""
        print("\n" + "="*80)
        print("BENCHMARK: VISION ENCODING")
        print("="*80)
        
        # Initialize encoder
        encoder = VisionEncoder(model_name=model_name)
        
        # Benchmark single image
        def encode_single(image):
            return encoder.encode([image], normalize=True)
        
        result = self.run_benchmark(
            encode_single,
            iterations=20,
            warmup_iterations=3,
            image=self.test_images[0]
        )
        
        # Benchmark batch encoding
        for batch_size in [1, 2, 4]:
            images = self.test_images[:batch_size]
            
            def encode_batch():
                return encoder.encode(images, normalize=True)
            
            batch_result = self.run_benchmark(
                encode_batch,
                iterations=10,
                warmup_iterations=2
            )
            batch_result.operation = f"vision_encode_batch_{batch_size}"
    
    def benchmark_world_generation(self, resolution: Tuple[int, int] = (256, 256)):
        """Benchmark world generation performance"""
        print("\n" + "="*80)
        print("BENCHMARK: WORLD GENERATION")
        print("="*80)
        
        # Mock generator
        generator = WorldGenerator(model_path="models/world_model/checkpoint.pt")
        
        # Mock components
        with patch.object(generator, 'text_encoder') as mock_text_encoder:
            with patch.object(generator, 'diffusion_model') as mock_diffusion:
                with patch.object(generator, 'renderer') as mock_renderer:
                    
                    # Setup mock responses
                    mock_text_encoder.encode.return_value = torch.randn(1, 512)
                    mock_diffusion.generate.return_value = {
                        'latents': torch.randn(1, 256, 32, 32, 32),
                        'timesteps': 50
                    }
                    mock_renderer.render.return_value = {
                        'video': np.random.rand(30, *resolution, 3),
                        'depth': np.random.rand(30, *resolution),
                        'normals': np.random.rand(30, *resolution, 3)
                    }
                    
                    # Benchmark generation
                    def generate_world():
                        return generator.generate(
                            prompt="test prompt",
                            num_frames=30,
                            resolution=resolution
                        )
                    
                    result = self.run_benchmark(
                        generate_world,
                        iterations=5,
                        warmup_iterations=1
                    )
    
    def benchmark_diffusion_sampling(self, sampler_type: str = "ddim"):
        """Benchmark diffusion sampling performance"""
        print("\n" + "="*80)
        print("BENCHMARK: DIFFUSION SAMPLING")
        print("="*80)
        
        # Initialize sampler
        sampler = Sampler(
            model="models/world_model/checkpoint.pt",
            sampler_type=sampler_type,
            num_steps=50
        )
        
        # Test different step counts
        for steps in [10, 25, 50, 100]:
            sampler.num_steps = steps
            
            def sample_latents():
                condition = torch.randn(1, 512)
                return sampler.sample(
                    condition=condition,
                    batch_size=1,
                    shape=(4, 32, 32)
                )
            
            result = self.run_benchmark(
                sample_latents,
                iterations=5,
                warmup_iterations=1
            )
            result.operation = f"diffusion_sampling_{steps}_steps"
    
    def benchmark_nerf_rendering(self):
        """Benchmark NeRF rendering performance"""
        print("\n" + "="*80)
        print("BENCHMARK: NeRF RENDERING")
        print("="*80)
        
        # Mock NeRF model
        nerf = NeRFModel(
            config_path="configs/model/nerf.yaml",
            pretrained=True
        )
        
        # Test different resolutions
        for res in self.resolutions:
            def render_nerf():
                # Mock input
                rays_o = torch.randn(1000, 3)
                rays_d = torch.randn(1000, 3)
                
                return nerf.render(
                    rays_o=rays_o,
                    rays_d=rays_d,
                    resolution=res
                )
            
            with patch.object(nerf, 'forward') as mock_forward:
                mock_forward.return_value = {
                    'rgb': torch.randn(1000, 3),
                    'depth': torch.randn(1000),
                    'alpha': torch.randn(1000)
                }
                
                result = self.run_benchmark(
                    render_nerf,
                    iterations=5,
                    warmup_iterations=1
                )
                result.operation = f"nerf_rendering_{res[0]}x{res[1]}"
    
    def benchmark_gaussian_rendering(self):
        """Benchmark Gaussian splatting rendering performance"""
        print("\n" + "="*80)
        print("BENCHMARK: GAUSSIAN SPLATTING RENDERING")
        print("="*80)
        
        # Mock Gaussian model
        gaussian = GaussianModel(
            config_path="configs/model/gaussian.yaml",
            pretrained=True
        )
        
        # Test different point counts
        for point_count in [1000, 5000, 10000, 50000]:
            def render_gaussian():
                # Mock input
                camera_pose = torch.eye(4)
                return gaussian.render(
                    camera_pose=camera_pose,
                    resolution=(256, 256)
                )
            
            with patch.object(gaussian, 'rasterize') as mock_rasterize:
                mock_rasterize.return_value = {
                    'image': torch.randn(256, 256, 3),
                    'depth': torch.randn(256, 256),
                    'alpha': torch.randn(256, 256)
                }
                
                # Mock gaussian count
                with patch.object(gaussian, 'gaussian_count', point_count):
                    result = self.run_benchmark(
                        render_gaussian,
                        iterations=5,
                        warmup_iterations=1
                    )
                    result.operation = f"gaussian_rendering_{point_count}_points"
    
    def benchmark_webgl_rendering(self):
        """Benchmark WebGL rendering performance"""
        print("\n" + "="*80)
        print("BENCHMARK: WEBGL RENDERING")
        print("="*80)
        
        # Mock WebGL engine
        engine = WebGLEngine(
            canvas_size=(1024, 768),
            antialias=True
        )
        
        # Test frame rendering
        def render_frame():
            # Mock scene data
            scene_data = {
                "objects": [
                    {
                        "type": "mesh",
                        "vertices": np.random.rand(1000, 3),
                        "faces": np.random.randint(0, 1000, (2000, 3))
                    }
                ],
                "camera": {
                    "position": [0, 0, 5],
                    "target": [0, 0, 0],
                    "fov": 60
                }
            }
            
            return engine.render_frame(scene_data)
        
        with patch.object(engine, '_render_to_canvas') as mock_render:
            mock_render.return_value = np.random.rand(768, 1024, 3)
            
            result = self.run_benchmark(
                render_frame,
                iterations=30,
                warmup_iterations=5
            )
    
    def benchmark_cache_performance(self):
        """Benchmark cache performance"""
        print("\n" + "="*80)
        print("BENCHMARK: CACHE PERFORMANCE")
        print("="*80)
        
        cache = CacheManager(
            cache_dir="/tmp/test_cache",
            max_size_gb=1
        )
        
        # Test cache set performance
        def cache_set():
            key = f"test_key_{np.random.randint(1000)}"
            value = {
                "embedding": np.random.randn(512),
                "timestamp": time.time()
            }
            cache.set(key, value, ttl=3600)
        
        set_result = self.run_benchmark(
            cache_set,
            iterations=100,
            warmup_iterations=10
        )
        
        # Test cache get performance
        def cache_get():
            key = "test_key_0"
            return cache.get(key)
        
        get_result = self.run_benchmark(
            cache_get,
            iterations=100,
            warmup_iterations=10
        )
        get_result.operation = "cache_get"
        
        # Test cache miss performance
        def cache_miss():
            key = f"non_existent_{np.random.randint(10000)}"
            return cache.get(key)
        
        miss_result = self.run_benchmark(
            cache_miss,
            iterations=100,
            warmup_iterations=10
        )
        miss_result.operation = "cache_miss"
    
    def benchmark_concurrent_operations(self, num_concurrent: int = 10):
        """Benchmark concurrent operations"""
        print("\n" + "="*80)
        print("BENCHMARK: CONCURRENT OPERATIONS")
        print("="*80)
        
        async def concurrent_operation(operation_id: int):
            """Simulate an async operation"""
            await asyncio.sleep(0.01)  # Simulate work
            return {"id": operation_id, "result": "success"}
        
        async def run_concurrent():
            """Run operations concurrently"""
            tasks = [
                concurrent_operation(i) for i in range(num_concurrent)
            ]
            results = await asyncio.gather(*tasks)
            return results
        
        # Run benchmark
        def run_benchmark():
            return asyncio.run(run_concurrent())
        
        result = self.run_benchmark(
            run_benchmark,
            iterations=10,
            warmup_iterations=2
        )
        result.operation = f"concurrent_operations_{num_concurrent}"
    
    def benchmark_memory_usage_growth(self, num_iterations: int = 100):
        """Benchmark memory usage growth over many iterations"""
        print("\n" + "="*80)
        print("BENCHMARK: MEMORY USAGE GROWTH")
        print("="*80)
        
        # Track memory usage
        memory_samples = []
        
        def memory_intensive_operation():
            # Allocate some memory
            data = [np.random.rand(1000, 1000) for _ in range(10)]
            # Don't return to keep reference
            return data
        
        # Run iterations and track memory
        process = psutil.Process()
        
        for i in range(num_iterations):
            # Force garbage collection
            gc.collect()
            
            # Measure memory before
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Run operation
            result = memory_intensive_operation()
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024
            
            # Calculate increase
            memory_increase = memory_after - memory_before
            memory_samples.append(memory_increase)
            
            # Clear reference
            del result
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Iteration {i + 1}/{num_iterations}: "
                     f"Memory increase: {memory_increase:.2f} MB")
        
        # Analyze memory growth
        avg_increase = statistics.mean(memory_samples)
        max_increase = max(memory_samples)
        total_increase = sum(memory_samples)
        
        print(f"\nMemory Growth Analysis:")
        print(f"  Average increase per iteration: {avg_increase:.2f} MB")
        print(f"  Maximum increase: {max_increase:.2f} MB")
        print(f"  Total increase: {total_increase:.2f} MB")
        
        # Check for memory leaks
        if avg_increase > 1.0:  # More than 1MB average increase
            print("  WARNING: Possible memory leak detected!")
        
        # Create result
        result = BenchmarkResult(
            operation="memory_usage_growth",
            iterations=num_iterations,
            total_time=0,
            avg_time=0,
            min_time=0,
            max_time=0,
            std_time=0,
            throughput=0,
            memory_usage_mb=avg_increase,
            cpu_percent=0,
            error_rate=0
        )
        
        self.results.append(result)
        return result
    
    def benchmark_scalability(self, max_batch_size: int = 16):
        """Benchmark scalability with increasing batch sizes"""
        print("\n" + "="*80)
        print("BENCHMARK: SCALABILITY")
        print("="*80)
        
        # Mock text encoder
        encoder = TextEncoder(model_name="clip-text")
        
        scalability_results = []
        
        for batch_size in [1, 2, 4, 8, 16]:
            if batch_size > max_batch_size:
                break
            
            # Create batch
            prompts = self.test_prompts[:batch_size]
            
            def encode_batch():
                return encoder.encode(prompts, normalize=True)
            
            with patch.object(encoder, 'model') as mock_model:
                mock_model.encode.return_value = torch.randn(batch_size, 512)
                
                result = self.run_benchmark(
                    encode_batch,
                    iterations=10,
                    warmup_iterations=2
                )
                result.operation = f"scalability_batch_{batch_size}"
                
                scalability_results.append({
                    "batch_size": batch_size,
                    "avg_time": result.avg_time,
                    "throughput": result.throughput,
                    "memory": result.memory_usage_mb
                })
        
        # Analyze scalability
        print("\nScalability Analysis:")
        for res in scalability_results:
            print(f"  Batch {res['batch_size']}: "
                 f"{res['avg_time']:.3f}s, "
                 f"{res['throughput']:.1f} ops/sec, "
                 f"{res['memory']:.1f} MB")
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("Running all performance benchmarks...")
        
        # System information
        print("\nSystem Information:")
        for key, value in self.get_system_info().items():
            print(f"  {key}: {value}")
        
        # Run benchmarks
        self.benchmark_text_encoding()
        self.benchmark_vision_encoding()
        self.benchmark_world_generation()
        self.benchmark_diffusion_sampling()
        self.benchmark_nerf_rendering()
        self.benchmark_gaussian_rendering()
        self.benchmark_webgl_rendering()
        self.benchmark_cache_performance()
        self.benchmark_concurrent_operations()
        self.benchmark_memory_usage_growth(num_iterations=50)
        self.benchmark_scalability()
        
        # Save and print results
        self.save_results()
        self.print_summary()


# Test classes for pytest
class TestInferencePerformance:
    """Test class for inference performance benchmarks"""
    
    def test_text_encoding_performance(self):
        """Test text encoding performance"""
        benchmark = InferenceBenchmark()
        result = benchmark.benchmark_text_encoding()
        
        # Performance requirements
        assert result.avg_time < 0.1  # Should be under 100ms
        assert result.throughput > 10  # Should handle >10 ops/sec
        assert result.error_rate == 0  # No errors allowed
    
    def test_vision_encoding_performance(self):
        """Test vision encoding performance"""
        benchmark = InferenceBenchmark()
        result = benchmark.benchmark_vision_encoding()
        
        # Performance requirements
        assert result.avg_time < 0.2  # Should be under 200ms
        assert result.throughput > 5  # Should handle >5 ops/sec
        assert result.error_rate == 0
    
    def test_world_generation_performance(self):
        """Test world generation performance"""
        benchmark = InferenceBenchmark()
        result = benchmark.benchmark_world_generation(resolution=(256, 256))
        
        # World generation is expensive, but should be reasonable
        assert result.avg_time < 30.0  # Should be under 30 seconds
        assert result.error_rate == 0
    
    def test_cache_performance(self):
        """Test cache performance"""
        benchmark = InferenceBenchmark()
        benchmark.benchmark_cache_performance()
        
        # Find cache results
        cache_results = [r for r in benchmark.results if "cache" in r.operation]
        
        for result in cache_results:
            # Cache operations should be very fast
            assert result.avg_time < 0.01  # Under 10ms
            assert result.error_rate == 0
    
    def test_memory_usage(self):
        """Test memory usage doesn't leak"""
        benchmark = InferenceBenchmark()
        result = benchmark.benchmark_memory_usage_growth(num_iterations=20)
        
        # Memory should not grow excessively
        assert result.memory_usage_mb < 5.0  # Less than 5MB average increase
    
    def test_concurrent_performance(self):
        """Test concurrent operation performance"""
        benchmark = InferenceBenchmark()
        result = benchmark.benchmark_concurrent_operations(num_concurrent=5)
        
        # Concurrent operations should complete reasonably
        assert result.avg_time < 0.5  # Under 500ms for 5 concurrent
        assert result.error_rate == 0
    
    def test_scalability(self):
        """Test scalability with batch size"""
        benchmark = InferenceBenchmark()
        benchmark.benchmark_scalability(max_batch_size=8)
        
        # Find scalability results
        scalability_results = [
            r for r in benchmark.results 
            if "scalability" in r.operation
        ]
        
        # Check that larger batches don't cause exponential slowdown
        if len(scalability_results) >= 2:
            small_batch = scalability_results[0]
            large_batch = scalability_results[-1]
            
            # Larger batch should be more efficient per item
            small_per_item = small_batch.avg_time
            large_per_item = large_batch.avg_time / (2 ** (len(scalability_results) - 1))
            
            # Large batch should not be more than 2x slower per item
            assert large_per_item < small_per_item * 2


if __name__ == "__main__":
    # Run benchmarks
    benchmark = InferenceBenchmark()
    benchmark.run_all_benchmarks()
    
    # Also run pytest tests
    pytest.main([__file__, "-v"])

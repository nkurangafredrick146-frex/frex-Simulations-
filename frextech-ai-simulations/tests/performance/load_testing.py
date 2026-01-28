"""
Load testing for the simulation system.
Tests system performance under high load and concurrent requests.
"""

import pytest
import asyncio
import aiohttp
import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue
import threading
import signal
import sys
import os

import numpy as np
from locust import HttpUser, task, between, events
from locust.runners import WorkerRunner

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.server import app
from src.api.utils.async_processor import AsyncProcessor
from src.api.middleware.rate_limiter import RateLimiter
from src.api.utils.cache_manager import CacheManager


@dataclass
class LoadTestResult:
    """Container for load test results"""
    test_name: str
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    concurrent_users: int
    error_rate: float
    system_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "test_name": self.test_name,
            "duration_seconds": self.duration,
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests
            },
            "response_times": {
                "avg": self.avg_response_time,
                "min": self.min_response_time,
                "max": self.max_response_time,
                "p95": self.p95_response_time,
                "p99": self.p99_response_time
            },
            "throughput": self.requests_per_second,
            "concurrent_users": self.concurrent_users,
            "error_rate": self.error_rate,
            "system_metrics": self.system_metrics
        }


class LoadTester:
    """Base class for load testing"""
    
    def __init__(self, base_url: str = "http://localhost:8000", 
                 output_dir: Path = None):
        self.base_url = base_url
        self.output_dir = output_dir or Path("load_test_results")
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[LoadTestResult] = []
        
        # Test data
        self.test_prompts = [
            "A beautiful landscape with mountains",
            "A futuristic city at night",
            "An underwater scene with coral",
            "A desert with pyramids",
            "A forest with ancient trees",
            "A beach sunset",
            "A snow-covered village",
            "A space station interior",
            "A medieval castle",
            "A tropical rainforest"
        ]
        
        # Statistics
        self.response_times: List[float] = []
        self.errors: List[Dict] = []
    
    async def make_request(self, session: aiohttp.ClientSession, 
                          endpoint: str, method: str = "GET",
                          data: Dict = None, headers: Dict = None) -> Tuple[float, bool]:
        """Make a single HTTP request and measure time"""
        url = f"{self.base_url}{endpoint}"
        headers = headers or {"Authorization": "Bearer test-token"}
        
        start_time = time.time()
        success = False
        
        try:
            if method == "GET":
                async with session.get(url, headers=headers) as response:
                    success = response.status == 200
            elif method == "POST":
                async with session.post(url, json=data, headers=headers) as response:
                    success = response.status in [200, 201, 202]
            elif method == "PUT":
                async with session.put(url, json=data, headers=headers) as response:
                    success = response.status in [200, 201, 202]
            elif method == "DELETE":
                async with session.delete(url, headers=headers) as response:
                    success = response.status == 200
            
            response_time = time.time() - start_time
            
            if not success:
                self.errors.append({
                    "endpoint": endpoint,
                    "method": method,
                    "timestamp": time.time(),
                    "response_time": response_time
                })
            
            return response_time, success
            
        except Exception as e:
            response_time = time.time() - start_time
            self.errors.append({
                "endpoint": endpoint,
                "method": method,
                "error": str(e),
                "timestamp": time.time(),
                "response_time": response_time
            })
            return response_time, False
    
    def calculate_statistics(self, response_times: List[float]) -> Dict:
        """Calculate response time statistics"""
        if not response_times:
            return {}
        
        sorted_times = sorted(response_times)
        n = len(sorted_times)
        
        return {
            "avg": statistics.mean(response_times),
            "min": min(response_times),
            "max": max(response_times),
            "p50": sorted_times[int(n * 0.5)] if n > 0 else 0,
            "p75": sorted_times[int(n * 0.75)] if n >= 4 else 0,
            "p90": sorted_times[int(n * 0.90)] if n >= 10 else 0,
            "p95": sorted_times[int(n * 0.95)] if n >= 20 else 0,
            "p99": sorted_times[int(n * 0.99)] if n >= 100 else 0,
            "std": statistics.stdev(response_times) if n > 1 else 0
        }
    
    def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        import psutil
        
        process = psutil.Process()
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "process_memory_mb": process.memory_info().rss / 1024 / 1024,
            "process_cpu": process.cpu_percent(),
            "disk_io": psutil.disk_io_counters()._asdict() if hasattr(psutil, 'disk_io_counters') else {},
            "network_io": psutil.net_io_counters()._asdict() if hasattr(psutil, 'net_io_counters') else {}
        }
    
    async def run_concurrent_requests(self, num_requests: int, 
                                     concurrent_tasks: int,
                                     endpoint: str, method: str = "GET",
                                     data_generator = None) -> LoadTestResult:
        """Run concurrent requests with specified concurrency"""
        print(f"\nRunning {num_requests} requests with {concurrent_tasks} concurrent tasks...")
        
        # Semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_tasks)
        
        async def limited_request(session, req_num):
            async with semaphore:
                data = None
                if data_generator:
                    data = data_generator(req_num)
                
                return await self.make_request(session, endpoint, method, data)
        
        # Create session
        connector = aiohttp.TCPConnector(limit=concurrent_tasks * 2)
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create tasks
            tasks = []
            for i in range(num_requests):
                task = asyncio.create_task(limited_request(session, i))
                tasks.append(task)
            
            # Wait for all tasks
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
        
        # Process results
        response_times = []
        successful = 0
        
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                response_time, success = result
                response_times.append(response_time)
                if success:
                    successful += 1
        
        # Calculate statistics
        stats = self.calculate_statistics(response_times)
        duration = end_time - start_time
        
        result = LoadTestResult(
            test_name=f"concurrent_{endpoint.replace('/', '_')}",
            duration=duration,
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=num_requests - successful,
            avg_response_time=stats.get("avg", 0),
            min_response_time=stats.get("min", 0),
            max_response_time=stats.get("max", 0),
            p95_response_time=stats.get("p95", 0),
            p99_response_time=stats.get("p99", 0),
            requests_per_second=num_requests / duration if duration > 0 else 0,
            concurrent_users=concurrent_tasks,
            error_rate=(num_requests - successful) / num_requests if num_requests > 0 else 0,
            system_metrics=self.get_system_metrics()
        )
        
        self.results.append(result)
        return result
    
    def save_results(self, filename: str = "load_test_results.json"):
        """Save load test results to file"""
        results_dict = {
            "timestamp": time.time(),
            "base_url": self.base_url,
            "tests": [r.to_dict() for r in self.results],
            "errors": self.errors[:100]  # Save first 100 errors
        }
        
        output_file = self.output_dir / filename
        output_file.write_text(json.dumps(results_dict, indent=2))
        print(f"Results saved to {output_file}")
    
    def print_results(self, result: LoadTestResult):
        """Print load test results"""
        print("\n" + "="*80)
        print(f"LOAD TEST RESULTS: {result.test_name}")
        print("="*80)
        print(f"Duration: {result.duration:.2f}s")
        print(f"Total Requests: {result.total_requests}")
        print(f"Successful: {result.successful_requests}")
        print(f"Failed: {result.failed_requests}")
        print(f"Error Rate: {result.error_rate:.2%}")
        print(f"\nResponse Times:")
        print(f"  Average: {result.avg_response_time:.3f}s")
        print(f"  Min: {result.min_response_time:.3f}s")
        print(f"  Max: {result.max_response_time:.3f}s")
        print(f"  P95: {result.p95_response_time:.3f}s")
        print(f"  P99: {result.p99_response_time:.3f}s")
        print(f"\nThroughput: {result.requests_per_second:.1f} req/sec")
        print(f"Concurrent Users: {result.concurrent_users}")
        print(f"\nSystem Metrics:")
        for key, value in result.system_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.1f}")
            else:
                print(f"  {key}: {value}")


class APILoadTester(LoadTester):
    """Load tester for API endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        super().__init__(base_url)
    
    def generate_generation_request(self, request_num: int) -> Dict:
        """Generate a generation request"""
        prompt = self.test_prompts[request_num % len(self.test_prompts)]
        
        return {
            "prompt": prompt,
            "parameters": {
                "resolution": "512x512",
                "style": "photorealistic",
                "duration_seconds": 5,
                "frame_rate": 30,
                "quality": "medium"
            },
            "output_format": "mp4"
        }
    
    def generate_edit_request(self, request_num: int) -> Dict:
        """Generate an edit request"""
        return {
            "world_id": f"test-world-{request_num % 100}",
            "edit_type": "style_transfer",
            "edit_prompt": "Make it look like a painting",
            "parameters": {
                "style_strength": 0.7,
                "preserve_content": True
            }
        }
    
    async def test_health_endpoint(self, num_requests: int = 1000, 
                                  concurrent_tasks: int = 100):
        """Test health endpoint under load"""
        result = await self.run_concurrent_requests(
            num_requests=num_requests,
            concurrent_tasks=concurrent_tasks,
            endpoint="/health",
            method="GET"
        )
        
        self.print_results(result)
        
        # Assertions for health endpoint
        assert result.error_rate < 0.01  # Less than 1% error rate
        assert result.avg_response_time < 0.1  # Under 100ms
        assert result.p95_response_time < 0.2  # P95 under 200ms
        
        return result
    
    async def test_generation_endpoint(self, num_requests: int = 100,
                                      concurrent_tasks: int = 10):
        """Test generation endpoint under load"""
        result = await self.run_concurrent_requests(
            num_requests=num_requests,
            concurrent_tasks=concurrent_tasks,
            endpoint="/api/v1/generate",
            method="POST",
            data_generator=self.generate_generation_request
        )
        
        self.print_results(result)
        
        # Generation is expensive, adjust expectations
        assert result.error_rate < 0.05  # Less than 5% error rate
        assert result.avg_response_time < 5.0  # Under 5 seconds for acceptance
        
        return result
    
    async def test_edit_endpoint(self, num_requests: int = 200,
                                concurrent_tasks: int = 20):
        """Test edit endpoint under load"""
        result = await self.run_concurrent_requests(
            num_requests=num_requests,
            concurrent_tasks=concurrent_tasks,
            endpoint="/api/v1/edit",
            method="POST",
            data_generator=self.generate_edit_request
        )
        
        self.print_results(result)
        
        # Edit operations should be reasonably fast
        assert result.error_rate < 0.05
        assert result.avg_response_time < 3.0  # Under 3 seconds
        
        return result
    
    async def test_status_endpoint(self, num_requests: int = 500,
                                  concurrent_tasks: int = 50):
        """Test status endpoint under load"""
        result = await self.run_concurrent_requests(
            num_requests=num_requests,
            concurrent_tasks=concurrent_tasks,
            endpoint="/api/v1/tasks/test-task-123",
            method="GET"
        )
        
        self.print_results(result)
        
        # Status checks should be very fast
        assert result.error_rate < 0.01
        assert result.avg_response_time < 0.2  # Under 200ms
        assert result.p95_response_time < 0.5  # P95 under 500ms
        
        return result
    
    async def test_mixed_workload(self, duration_seconds: int = 60,
                                 requests_per_second: int = 10):
        """Test mixed workload simulating real usage"""
        print(f"\nRunning mixed workload test for {duration_seconds} seconds...")
        
        endpoints = [
            ("/health", "GET", None),
            ("/api/v1/generate", "POST", self.generate_generation_request),
            ("/api/v1/edit", "POST", self.generate_edit_request),
            ("/api/v1/tasks/test-id", "GET", None)
        ]
        
        weights = [0.3, 0.4, 0.2, 0.1]  # Probability distribution
        
        start_time = time.time()
        request_count = 0
        response_times = []
        successes = 0
        
        # Create session
        connector = aiohttp.TCPConnector(limit=100)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            while time.time() - start_time < duration_seconds:
                tasks = []
                
                # Create requests for this second
                for _ in range(requests_per_second):
                    # Select endpoint based on weights
                    endpoint_idx = np.random.choice(len(endpoints), p=weights)
                    endpoint, method, generator = endpoints[endpoint_idx]
                    
                    data = None
                    if generator:
                        data = generator(request_count)
                    
                    task = asyncio.create_task(
                        self.make_request(session, endpoint, method, data)
                    )
                    tasks.append(task)
                    request_count += 1
                
                # Wait for batch completion
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in batch_results:
                    if isinstance(result, tuple) and len(result) == 2:
                        response_time, success = result
                        response_times.append(response_time)
                        if success:
                            successes += 1
                
                # Sleep to maintain rate
                elapsed = time.time() - start_time
                target_time = request_count / requests_per_second
                if elapsed < target_time:
                    await asyncio.sleep(target_time - elapsed)
        
        duration = time.time() - start_time
        
        # Calculate statistics
        stats = self.calculate_statistics(response_times)
        
        result = LoadTestResult(
            test_name="mixed_workload",
            duration=duration,
            total_requests=request_count,
            successful_requests=successes,
            failed_requests=request_count - successes,
            avg_response_time=stats.get("avg", 0),
            min_response_time=stats.get("min", 0),
            max_response_time=stats.get("max", 0),
            p95_response_time=stats.get("p95", 0),
            p99_response_time=stats.get("p99", 0),
            requests_per_second=request_count / duration if duration > 0 else 0,
            concurrent_users=requests_per_second,
            error_rate=(request_count - successes) / request_count if request_count > 0 else 0,
            system_metrics=self.get_system_metrics()
        )
        
        self.results.append(result)
        self.print_results(result)
        
        # Mixed workload assertions
        assert result.error_rate < 0.1  # Less than 10% error rate
        assert result.requests_per_second >= requests_per_second * 0.8  # 80% of target
        
        return result
    
    async def test_rate_limiting(self, requests_per_second: int = 20,
                                duration_seconds: int = 10):
        """Test rate limiting behavior"""
        print(f"\nTesting rate limiting at {requests_per_second} requests/second...")
        
        start_time = time.time()
        request_count = 0
        rate_limited_count = 0
        
        connector = aiohttp.TCPConnector(limit=requests_per_second)
        timeout = aiohttp.ClientTimeout(total=5)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            while time.time() - start_time < duration_seconds:
                tasks = []
                
                # Burst of requests
                for _ in range(requests_per_second):
                    task = asyncio.create_task(
                        self.make_request(session, "/health", "GET")
                    )
                    tasks.append(task)
                    request_count += 1
                
                # Wait for batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for rate limiting (status 429)
                for result in batch_results:
                    if isinstance(result, Exception):
                        if "429" in str(result):
                            rate_limited_count += 1
                
                # Small delay
                await asyncio.sleep(0.05)
        
        rate_limit_percentage = rate_limited_count / request_count if request_count > 0 else 0
        
        print(f"\nRate Limiting Test:")
        print(f"  Total Requests: {request_count}")
        print(f"  Rate Limited: {rate_limited_count}")
        print(f"  Rate Limit Percentage: {rate_limit_percentage:.1%}")
        
        # Should see some rate limiting with high request rate
        assert rate_limit_percentage > 0.1  # At least 10% should be rate limited
        
        return rate_limit_percentage
    
    async def test_long_running_requests(self, num_requests: int = 10,
                                        timeout_seconds: int = 30):
        """Test handling of long-running requests"""
        print(f"\nTesting {num_requests} long-running requests...")
        
        # Use generation endpoint which takes time
        response_times = []
        successes = 0
        
        connector = aiohttp.TCPConnector(limit=num_requests)
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            for i in range(num_requests):
                data = self.generate_generation_request(i)
                task = asyncio.create_task(
                    self.make_request(session, "/api/v1/generate", "POST", data)
                )
                tasks.append(task)
            
            # Wait with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout_seconds
                )
                
                for result in results:
                    if isinstance(result, tuple) and len(result) == 2:
                        response_time, success = result
                        response_times.append(response_time)
                        if success:
                            successes += 1
            
            except asyncio.TimeoutError:
                print(f"Timeout after {timeout_seconds} seconds")
        
        print(f"\nLong-running Requests:")
        print(f"  Completed: {successes}/{num_requests}")
        if response_times:
            print(f"  Average Time: {statistics.mean(response_times):.1f}s")
            print(f"  Max Time: {max(response_times):.1f}s")
        
        # Should complete most requests
        completion_rate = successes / num_requests
        assert completion_rate > 0.5  # At least 50% should complete
        
        return completion_rate
    
    async def run_all_tests(self):
        """Run all load tests"""
        print("Starting comprehensive load testing...")
        
        # 1. Health endpoint (high volume)
        print("\n1. Testing health endpoint...")
        await self.test_health_endpoint(num_requests=2000, concurrent_tasks=200)
        
        # 2. Status endpoint (medium volume)
        print("\n2. Testing status endpoint...")
        await self.test_status_endpoint(num_requests=1000, concurrent_tasks=100)
        
        # 3. Generation endpoint (low volume, high cost)
        print("\n3. Testing generation endpoint...")
        await self.test_generation_endpoint(num_requests=50, concurrent_tasks=5)
        
        # 4. Edit endpoint (medium volume)
        print("\n4. Testing edit endpoint...")
        await self.test_edit_endpoint(num_requests=100, concurrent_tasks=10)
        
        # 5. Mixed workload
        print("\n5. Testing mixed workload...")
        await self.test_mixed_workload(duration_seconds=30, requests_per_second=20)
        
        # 6. Rate limiting
        print("\n6. Testing rate limiting...")
        await self.test_rate_limiting(requests_per_second=50, duration_seconds=5)
        
        # 7. Long-running requests
        print("\n7. Testing long-running requests...")
        await self.test_long_running_requests(num_requests=5, timeout_seconds=60)
        
        # Save results
        self.save_results()
        
        print("\n" + "="*80)
        print("ALL LOAD TESTS COMPLETED")
        print("="*80)
        
        # Print summary
        total_requests = sum(r.total_requests for r in self.results)
        total_errors = sum(r.failed_requests for r in self.results)
        overall_error_rate = total_errors / total_requests if total_requests > 0 else 0
        
        print(f"\nSummary:")
        print(f"  Total Tests: {len(self.results)}")
        print(f"  Total Requests: {total_requests}")
        print(f"  Total Errors: {total_errors}")
        print(f"  Overall Error Rate: {overall_error_rate:.2%}")
        
        # Check overall performance
        assert overall_error_rate < 0.05  # Less than 5% overall error rate


class SystemLoadTester(LoadTester):
    """Load tester for system components (not HTTP)"""
    
    def __init__(self):
        super().__init__()
    
    def test_async_processor(self, num_tasks: int = 1000, 
                            max_workers: int = 10):
        """Test AsyncProcessor under load"""
        print(f"\nTesting AsyncProcessor with {num_tasks} tasks, {max_workers} workers...")
        
        processor = AsyncProcessor(
            max_workers=max_workers,
            queue_size=num_tasks * 2
        )
        
        # Define processing function
        def process_task(task_id: int):
            # Simulate work
            time.sleep(0.001)
            return {"task_id": task_id, "processed": True}
        
        # Submit tasks and measure
        start_time = time.time()
        
        futures = []
        for i in range(num_tasks):
            future = processor.submit(process_task, i)
            futures.append(future)
        
        # Wait for completion
        results = []
        for future in futures:
            try:
                result = future.result(timeout=10)
                results.append(result)
            except Exception as e:
                self.errors.append({
                    "component": "AsyncProcessor",
                    "error": str(e),
                    "task_id": len(results)
                })
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate statistics
        success_rate = len(results) / num_tasks
        
        print(f"AsyncProcessor Results:")
        print(f"  Tasks: {num_tasks}")
        print(f"  Completed: {len(results)}")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {num_tasks / duration:.1f} tasks/sec")
        
        assert success_rate > 0.95  # 95% success rate
        assert duration < num_tasks * 0.01  # Should be efficient
        
        return success_rate
    
    def test_cache_manager(self, num_operations: int = 10000):
        """Test CacheManager under load"""
        print(f"\nTesting CacheManager with {num_operations} operations...")
        
        cache = CacheManager(
            cache_dir="/tmp/load_test_cache",
            max_size_gb=1
        )
        
        # Test set operations
        set_times = []
        for i in range(num_operations):
            key = f"test_key_{i}"
            value = {"data": "x" * 1000, "index": i}  # 1KB value
            
            start_time = time.perf_counter()
            cache.set(key, value, ttl=60)
            set_times.append(time.perf_counter() - start_time)
        
        # Test get operations
        get_times = []
        for i in range(num_operations):
            key = f"test_key_{i}"
            
            start_time = time.perf_counter()
            value = cache.get(key)
            get_times.append(time.perf_counter() - start_time)
            
            if value is None or value["index"] != i:
                self.errors.append({
                    "component": "CacheManager",
                    "error": f"Cache miss or corruption for key {key}",
                    "expected": i,
                    "got": value["index"] if value else None
                })
        
        # Calculate statistics
        set_stats = self.calculate_statistics(set_times)
        get_stats = self.calculate_statistics(get_times)
        
        print(f"CacheManager Results:")
        print(f"  Set Operations: {num_operations}")
        print(f"    Avg Time: {set_stats.get('avg', 0):.6f}s")
        print(f"    P95 Time: {set_stats.get('p95', 0):.6f}s")
        print(f"  Get Operations: {num_operations}")
        print(f"    Avg Time: {get_stats.get('avg', 0):.6f}s")
        print(f"    P95 Time: {get_stats.get('p95', 0):.6f}s")
        
        # Performance requirements
        assert set_stats.get('avg', 0) < 0.001  # Under 1ms
        assert get_stats.get('avg', 0) < 0.0005  # Under 0.5ms
        
        return set_stats, get_stats
    
    def test_rate_limiter(self, requests_per_window: int = 100,
                         num_clients: int = 1000):
        """Test RateLimiter under load"""
        print(f"\nTesting RateLimiter with {num_clients} clients...")
        
        limiter = RateLimiter(
            requests_per_minute=requests_per_window * 60,  # Convert to per minute
            burst_size=requests_per_window * 2
        )
        
        # Simulate requests from many clients
        allowed = 0
        denied = 0
        
        start_time = time.time()
        
        for client_id in range(num_clients):
            for _ in range(requests_per_window // 10):  # Each client makes some requests
                if limiter.is_allowed(f"client_{client_id}"):
                    allowed += 1
                else:
                    denied += 1
        
        duration = time.time() - start_time
        
        denial_rate = denied / (allowed + denied) if (allowed + denied) > 0 else 0
        
        print(f"RateLimiter Results:")
        print(f"  Clients: {num_clients}")
        print(f"  Allowed: {allowed}")
        print(f"  Denied: {denied}")
        print(f"  Denial Rate: {denial_rate:.1%}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {(allowed + denied) / duration:.0f} checks/sec")
        
        # Should deny some requests with many clients
        assert denial_rate > 0.1  # At least 10% should be denied under load
        
        return denial_rate


# Locust load tests for distributed testing
class SimulationUser(HttpUser):
    """Locust user for simulating API traffic"""
    
    wait_time = between(1, 5)  # Wait between 1 and 5 seconds
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_prompts = [
            "A beautiful landscape",
            "A futuristic city",
            "An underwater scene",
            "A desert oasis",
            "A forest path"
        ]
        self.task_ids = []
    
    @task(3)  # Higher weight = more frequent
    def health_check(self):
        """Check health endpoint"""
        self.client.get("/health")
    
    @task(2)
    def generate_world(self):
        """Generate a world"""
        prompt = self.test_prompts[len(self.task_ids) % len(self.test_prompts)]
        
        data = {
            "prompt": prompt,
            "parameters": {
                "resolution": "256x256",
                "duration_seconds": 2,
                "quality": "low"
            }
        }
        
        with self.client.post("/api/v1/generate", json=data, catch_response=True) as response:
            if response.status_code == 202:  # Accepted
                task_id = response.json().get("task_id")
                if task_id:
                    self.task_ids.append(task_id)
    
    @task(2)
    def check_task_status(self):
        """Check status of a task"""
        if self.task_ids:
            task_id = self.task_ids[-1]  # Check most recent task
            self.client.get(f"/api/v1/tasks/{task_id}")
    
    @task(1)
    def edit_world(self):
        """Edit a world"""
        if self.task_ids:
            world_id = f"world-{hash(self.task_ids[-1]) % 1000}"
            
            data = {
                "world_id": world_id,
                "edit_type": "style_transfer",
                "edit_prompt": "Make it look like a painting"
            }
            
            self.client.post("/api/v1/edit", json=data)


# Test classes for pytest
class TestLoadPerformance:
    """Test class for load performance"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint_load(self):
        """Test health endpoint under load"""
        tester = APILoadTester()
        result = await tester.test_health_endpoint(num_requests=100, concurrent_tasks=10)
        
        assert result.error_rate < 0.05
        assert result.avg_response_time < 0.5
    
    @pytest.mark.asyncio
    async def test_generation_endpoint_load(self):
        """Test generation endpoint under load"""
        tester = APILoadTester()
        result = await tester.test_generation_endpoint(num_requests=20, concurrent_tasks=2)
        
        assert result.error_rate < 0.2  # Generation can fail more often under load
        assert result.avg_response_time < 10.0
    
    @pytest.mark.asyncio
    async def test_mixed_workload(self):
        """Test mixed workload"""
        tester = APILoadTester()
        result = await tester.test_mixed_workload(duration_seconds=10, requests_per_second=5)
        
        assert result.error_rate < 0.2
    
    def test_async_processor_load(self):
        """Test AsyncProcessor under load"""
        tester = SystemLoadTester()
        success_rate = tester.test_async_processor(num_tasks=100, max_workers=5)
        
        assert success_rate > 0.9
    
    def test_cache_manager_load(self):
        """Test CacheManager under load"""
        tester = SystemLoadTester()
        set_stats, get_stats = tester.test_cache_manager(num_operations=100)
        
        assert set_stats.get('avg', 0) < 0.01
        assert get_stats.get('avg', 0) < 0.005


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run load tests")
    parser.add_argument("--api-url", default="http://localhost:8000", 
                       help="Base URL for API tests")
    parser.add_argument("--test-type", choices=["api", "system", "all"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--output-dir", default="load_test_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    async def main():
        if args.test_type in ["api", "all"]:
            print("Running API load tests...")
            api_tester = APILoadTester(base_url=args.api_url)
            await api_tester.run_all_tests()
        
        if args.test_type in ["system", "all"]:
            print("\nRunning system load tests...")
            system_tester = SystemLoadTester()
            
            # Run system tests
            system_tester.test_async_processor(num_tasks=1000, max_workers=10)
            system_tester.test_cache_manager(num_operations=1000)
            system_tester.test_rate_limiter(num_clients=100)
    
    # Run async main
    asyncio.run(main())

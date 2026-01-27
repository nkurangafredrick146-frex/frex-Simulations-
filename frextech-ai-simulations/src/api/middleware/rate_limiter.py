"""
Rate limiting middleware for API endpoints
"""

import time
import asyncio
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import logging

import redis.asyncio as redis
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from ...utils.logging_config import setup_logging

logger = setup_logging("rate_limiter")


class RateLimiter:
    """Rate limiter using Redis for distributed rate limiting"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_rate_limit: int = 100,
        window_size: int = 60,  # seconds
        use_memory_fallback: bool = True
    ):
        self.redis_url = redis_url
        self.default_rate_limit = default_rate_limit
        self.window_size = window_size
        self.use_memory_fallback = use_memory_fallback
        
        self.redis_client: Optional[redis.Redis] = None
        self.memory_store: Dict[str, List[float]] = {}
        self.rate_limit_configs: Dict[str, Dict] = {}
        
        # Default rate limits per endpoint
        self.default_configs = {
            "/api/v1/generation/generate": {"limit": 10, "window": 60},
            "/api/v1/generation/batch": {"limit": 5, "window": 300},
            "/api/v1/editing/edit": {"limit": 20, "window": 60},
            "/api/v1/editing/expand": {"limit": 10, "window": 60},
            "/api/v1/export/render": {"limit": 30, "window": 60},
            "/api/v1/management/*": {"limit": 100, "window": 60},
        }
    
    async def connect(self):
        """Connect to Redis"""
        if self.redis_url:
            try:
                self.redis_client = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info(f"Connected to Redis at {self.redis_url}")
                return True
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using memory store.")
                self.redis_client = None
        
        return False
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    def _get_endpoint_config(self, path: str) -> Tuple[int, int]:
        """Get rate limit configuration for an endpoint"""
        # Check exact match
        if path in self.default_configs:
            config = self.default_configs[path]
            return config["limit"], config["window"]
        
        # Check wildcard match
        for pattern, config in self.default_configs.items():
            if pattern.endswith("*") and path.startswith(pattern[:-1]):
                return config["limit"], config["window"]
        
        # Default
        return self.default_rate_limit, self.window_size
    
    async def is_rate_limited(
        self,
        identifier: str,
        endpoint: str = None,
        increment: bool = True
    ) -> Tuple[bool, Dict]:
        """
        Check if request is rate limited
        
        Returns:
            Tuple of (is_limited, details)
        """
        if endpoint:
            limit, window = self._get_endpoint_config(endpoint)
        else:
            limit, window = self.default_rate_limit, self.window_size
        
        current_time = time.time()
        key = f"rate_limit:{identifier}:{endpoint or 'global'}"
        
        if self.redis_client:
            try:
                # Use Redis sorted set for rate limiting
                pipeline = self.redis_client.pipeline()
                
                # Remove old requests
                pipeline.zremrangebyscore(key, 0, current_time - window)
                
                # Get current count
                pipeline.zcard(key)
                
                # Add current request if incrementing
                if increment:
                    pipeline.zadd(key, {str(current_time): current_time})
                    pipeline.expire(key, window)
                
                results = await pipeline.execute()
                current_count = results[1]
                
                if increment and current_count >= limit:
                    # Calculate reset time
                    oldest = await self.redis_client.zrange(key, 0, 0, withscores=True)
                    reset_time = oldest[0][1] + window if oldest else current_time + window
                    
                    return True, {
                        "limit": limit,
                        "remaining": 0,
                        "reset": reset_time,
                        "window": window,
                        "current": current_count
                    }
                
                return False, {
                    "limit": limit,
                    "remaining": max(0, limit - current_count - (1 if increment else 0)),
                    "reset": current_time + window,
                    "window": window,
                    "current": current_count
                }
            
            except Exception as e:
                logger.error(f"Redis rate limiting error: {e}")
                if self.use_memory_fallback:
                    return await self._memory_rate_limit(
                        key, limit, window, current_time, increment
                    )
                raise
        
        # Fallback to memory store
        return await self._memory_rate_limit(
            key, limit, window, current_time, increment
        )
    
    async def _memory_rate_limit(
        self,
        key: str,
        limit: int,
        window: int,
        current_time: float,
        increment: bool
    ) -> Tuple[bool, Dict]:
        """Rate limiting using in-memory store"""
        if key not in self.memory_store:
            self.memory_store[key] = []
        
        # Remove old requests
        self.memory_store[key] = [
            ts for ts in self.memory_store[key]
            if ts > current_time - window
        ]
        
        current_count = len(self.memory_store[key])
        
        if increment:
            if current_count >= limit:
                # Calculate reset time
                reset_time = self.memory_store[key][0] + window
                
                return True, {
                    "limit": limit,
                    "remaining": 0,
                    "reset": reset_time,
                    "window": window,
                    "current": current_count
                }
            
            # Add current request
            self.memory_store[key].append(current_time)
        
        return False, {
            "limit": limit,
            "remaining": max(0, limit - current_count - (1 if increment else 0)),
            "reset": current_time + window,
            "window": window,
            "current": current_count
        }
    
    def get_identifier(self, request: Request) -> str:
        """Get identifier for rate limiting"""
        # Try to get API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            # Use hashed API key for privacy
            import hashlib
            return f"api_key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    async def get_rate_limit_info(
        self,
        identifier: str,
        endpoint: str = None
    ) -> Dict:
        """Get rate limit information without incrementing"""
        is_limited, details = await self.is_rate_limited(
            identifier, endpoint, increment=False
        )
        return {
            "is_limited": is_limited,
            **details
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting"""
    
    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for certain paths
        skip_paths = ["/health", "/docs", "/redoc", "/openapi.json"]
        if request.url.path in skip_paths:
            return await call_next(request)
        
        # Get identifier
        identifier = self.rate_limiter.get_identifier(request)
        endpoint = request.url.path
        
        # Check rate limit
        is_limited, details = await self.rate_limiter.is_rate_limited(
            identifier, endpoint
        )
        
        if is_limited:
            # Add rate limit headers
            headers = {
                "X-RateLimit-Limit": str(details["limit"]),
                "X-RateLimit-Remaining": str(details["remaining"]),
                "X-RateLimit-Reset": str(int(details["reset"])),
                "X-RateLimit-Window": str(details["window"]),
                "Retry-After": str(int(details["reset"] - time.time()))
            }
            
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": "Rate limit exceeded",
                    "details": details,
                    "timestamp": datetime.utcnow().isoformat()
                },
                headers=headers
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        response.headers["X-RateLimit-Limit"] = str(details["limit"])
        response.headers["X-RateLimit-Remaining"] = str(details["remaining"])
        response.headers["X-RateLimit-Reset"] = str(int(details["reset"]))
        response.headers["X-RateLimit-Window"] = str(details["window"])
        
        return response


# Token bucket rate limiter for more complex scenarios
class TokenBucket:
    """Token bucket rate limiter"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """Consume tokens from bucket"""
        async with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_refill = now
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait for tokens to become available"""
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        deficit = tokens - self.tokens
        return deficit / self.refill_rate


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load"""
    
    def __init__(
        self,
        base_limit: int = 100,
        min_limit: int = 10,
        max_limit: int = 1000,
        load_window: int = 60
    ):
        self.base_limit = base_limit
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.load_window = load_window
        
        self.request_times: List[float] = []
        self.error_counts: List[Tuple[float, int]] = []
        self.system_load: List[Tuple[float, float]] = []
        
    def update_load(self, load: float):
        """Update system load measurement"""
        current_time = time.time()
        self.system_load.append((current_time, load))
        
        # Remove old measurements
        cutoff = current_time - self.load_window
        self.system_load = [
            (t, l) for t, l in self.system_load
            if t > cutoff
        ]
    
    def update_error(self):
        """Record an error"""
        current_time = time.time()
        self.error_counts.append((current_time, 1))
        
        # Remove old measurements
        cutoff = current_time - self.load_window
        self.error_counts = [
            (t, c) for t, c in self.error_counts
            if t > cutoff
        ]
    
    def get_current_limit(self) -> int:
        """Calculate current rate limit based on system state"""
        current_time = time.time()
        cutoff = current_time - self.load_window
        
        # Calculate error rate
        recent_errors = sum(c for t, c in self.error_counts if t > cutoff)
        recent_requests = len([t for t in self.request_times if t > cutoff])
        
        error_rate = recent_errors / max(recent_requests, 1)
        
        # Calculate average load
        recent_loads = [l for t, l in self.system_load if t > cutoff]
        avg_load = sum(recent_loads) / max(len(recent_loads), 1)
        
        # Adjust limit based on error rate and load
        adjustment = 1.0
        
        if error_rate > 0.1:  # 10% error rate
            adjustment *= 0.5
        elif error_rate > 0.05:  # 5% error rate
            adjustment *= 0.8
        
        if avg_load > 0.8:  # 80% load
            adjustment *= 0.7
        elif avg_load > 0.6:  # 60% load
            adjustment *= 0.9
        
        new_limit = int(self.base_limit * adjustment)
        return max(self.min_limit, min(self.max_limit, new_limit))
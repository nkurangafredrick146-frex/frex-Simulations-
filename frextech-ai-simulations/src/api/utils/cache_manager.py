"""
Cache manager for API responses and intermediate results
"""

import asyncio
import json
import pickle
import zlib
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging

import redis.asyncio as redis
from redis.exceptions import RedisError

from ...utils.logging_config import setup_logging

logger = setup_logging("cache_manager")


class CacheManager:
    """
    Distributed cache manager using Redis with compression, serialization, and statistics
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 3600,  # 1 hour
        compression: bool = True,
        compression_threshold: int = 1024,  # 1KB
        namespace: str = "frextech:cache",
        max_memory: Optional[int] = None,  # Max memory in bytes
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0
    ):
        """
        Initialize cache manager
        
        Args:
            redis_url: Redis connection URL
            default_ttl: Default time-to-live in seconds
            compression: Whether to compress large values
            compression_threshold: Minimum size for compression (bytes)
            namespace: Cache namespace prefix
            max_memory: Maximum memory limit for Redis (bytes)
            reconnect_attempts: Number of reconnection attempts
            reconnect_delay: Delay between reconnection attempts
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.compression = compression
        self.compression_threshold = compression_threshold
        self.namespace = namespace
        self.max_memory = max_memory
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        self.redis_client: Optional[redis.Redis] = None
        self.is_connected = False
        self._connection_lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "compressions": 0,
            "total_size": 0,
            "start_time": datetime.utcnow().isoformat()
        }
        
        # Cache policies
        self.policies = {
            "eviction_policy": "volatile-lru",  # Redis eviction policy
            "max_keys": 100000,  # Maximum keys in cache
            "cleanup_interval": 300,  # Cleanup interval in seconds
        }
        
        logger.info(f"CacheManager initialized with namespace: {namespace}")
    
    async def connect(self) -> bool:
        """Connect to Redis"""
        async with self._connection_lock:
            if self.is_connected and self.redis_client:
                try:
                    await self.redis_client.ping()
                    return True
                except RedisError:
                    self.is_connected = False
            
            for attempt in range(self.reconnect_attempts):
                try:
                    logger.info(f"Connecting to Redis (attempt {attempt + 1}/{self.reconnect_attempts})...")
                    
                    self.redis_client = redis.from_url(
                        self.redis_url,
                        encoding="utf-8",
                        decode_responses=False,  # We handle encoding ourselves
                        socket_connect_timeout=5.0,
                        socket_timeout=5.0,
                        retry_on_timeout=True,
                        max_connections=50
                    )
                    
                    # Test connection
                    await self.redis_client.ping()
                    
                    # Configure Redis if max_memory is set
                    if self.max_memory:
                        await self._configure_redis()
                    
                    self.is_connected = True
                    logger.info(f"Connected to Redis at {self.redis_url}")
                    
                    # Start cleanup task
                    asyncio.create_task(self._periodic_cleanup())
                    
                    return True
                
                except RedisError as e:
                    logger.warning(f"Redis connection failed (attempt {attempt + 1}): {e}")
                    
                    if attempt < self.reconnect_attempts - 1:
                        await asyncio.sleep(self.reconnect_delay)
                    else:
                        logger.error(f"Failed to connect to Redis after {self.reconnect_attempts} attempts")
                        self.is_connected = False
                        return False
            
            return False
    
    async def _configure_redis(self):
        """Configure Redis settings"""
        if not self.redis_client:
            return
        
        try:
            # Set max memory and eviction policy
            await self.redis_client.config_set("maxmemory", str(self.max_memory))
            await self.redis_client.config_set("maxmemory-policy", self.policies["eviction_policy"])
            logger.info(f"Redis configured: maxmemory={self.max_memory}, policy={self.policies['eviction_policy']}")
        except RedisError as e:
            logger.warning(f"Failed to configure Redis: {e}")
    
    async def close(self):
        """Close Redis connection"""
        async with self._connection_lock:
            if self.redis_client:
                await self.redis_client.close()
                self.redis_client = None
                self.is_connected = False
                logger.info("Redis connection closed")
    
    def _make_key(self, key: str) -> str:
        """Create namespaced key"""
        return f"{self.namespace}:{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value to bytes"""
        try:
            # Try JSON serialization first (for human-readable data)
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                data = json.dumps(value, ensure_ascii=False).encode('utf-8')
                data_type = b'json'
            else:
                # Use pickle for arbitrary Python objects
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                data_type = b'pickle'
            
            # Compress if enabled and above threshold
            if self.compression and len(data) >= self.compression_threshold:
                compressed = zlib.compress(data, level=3)
                if len(compressed) < len(data):
                    data = compressed
                    data_type += b':compressed'
                    self.stats["compressions"] += 1
            
            # Add type header
            return data_type + b':' + data
        
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize bytes to value"""
        try:
            if not data:
                return None
            
            # Split type header from data
            if b':' not in data:
                # Legacy format without type header
                try:
                    return json.loads(data.decode('utf-8'))
                except:
                    return pickle.loads(data)
            
            data_type, payload = data.split(b':', 1)
            
            # Decompress if needed
            if b'compressed' in data_type:
                payload = zlib.decompress(payload)
                data_type = data_type.replace(b':compressed', b'')
            
            # Deserialize based on type
            if data_type == b'json':
                return json.loads(payload.decode('utf-8'))
            elif data_type == b'pickle':
                return pickle.loads(payload)
            else:
                # Unknown type, try both
                try:
                    return json.loads(payload.decode('utf-8'))
                except:
                    return pickle.loads(payload)
        
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache
        
        Args:
            key: Cache key
            default: Default value if key not found
        
        Returns:
            Cached value or default
        """
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return default
        
        try:
            cache_key = self._make_key(key)
            data = await self.redis_client.get(cache_key)
            
            if data is None:
                self.stats["misses"] += 1
                return default
            
            value = self._deserialize_value(data)
            self.stats["hits"] += 1
            
            # Update access time if TTL is set
            ttl = await self.redis_client.ttl(cache_key)
            if ttl > 0:
                # Refresh TTL on access (only if not near expiration)
                if ttl > 60:  # More than 1 minute remaining
                    await self.redis_client.expire(cache_key, ttl)
            
            return value
        
        except RedisError as e:
            logger.error(f"Cache get error: {e}")
            self.is_connected = False
            return default
        
        except Exception as e:
            logger.error(f"Unexpected cache error: {e}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,  # Only set if key doesn't exist
        xx: bool = False   # Only set if key exists
    ) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists
        
        Returns:
            True if set successfully, False otherwise
        """
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return False
        
        try:
            cache_key = self._make_key(key)
            data = self._serialize_value(value)
            
            ttl = ttl or self.default_ttl
            
            if nx and xx:
                raise ValueError("Cannot use both nx and xx")
            
            if nx:
                # Set only if not exists
                result = await self.redis_client.set(
                    cache_key,
                    data,
                    ex=ttl,
                    nx=True
                )
            elif xx:
                # Set only if exists
                result = await self.redis_client.set(
                    cache_key,
                    data,
                    ex=ttl,
                    xx=True
                )
            else:
                # Always set
                result = await self.redis_client.set(
                    cache_key,
                    data,
                    ex=ttl
                )
            
            if result:
                self.stats["sets"] += 1
                self.stats["total_size"] += len(data)
            
            return bool(result)
        
        except RedisError as e:
            logger.error(f"Cache set error: {e}")
            self.is_connected = False
            return False
        
        except Exception as e:
            logger.error(f"Unexpected cache error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache
        
        Args:
            key: Cache key to delete
        
        Returns:
            True if deleted, False otherwise
        """
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return False
        
        try:
            cache_key = self._make_key(key)
            result = await self.redis_client.delete(cache_key)
            
            if result > 0:
                self.stats["deletes"] += 1
            
            return result > 0
        
        except RedisError as e:
            logger.error(f"Cache delete error: {e}")
            self.is_connected = False
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return False
        
        try:
            cache_key = self._make_key(key)
            return bool(await self.redis_client.exists(cache_key))
        
        except RedisError as e:
            logger.error(f"Cache exists error: {e}")
            self.is_connected = False
            return False
    
    async def ttl(self, key: str) -> int:
        """
        Get time-to-live for key
        
        Args:
            key: Cache key
        
        Returns:
            TTL in seconds, -1 if no expiry, -2 if key doesn't exist
        """
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return -2
        
        try:
            cache_key = self._make_key(key)
            return await self.redis_client.ttl(cache_key)
        
        except RedisError as e:
            logger.error(f"Cache TTL error: {e}")
            self.is_connected = False
            return -2
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiry for key
        
        Args:
            key: Cache key
            ttl: Time-to-live in seconds
        
        Returns:
            True if expiry was set, False otherwise
        """
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return False
        
        try:
            cache_key = self._make_key(key)
            return bool(await self.redis_client.expire(cache_key, ttl))
        
        except RedisError as e:
            logger.error(f"Cache expire error: {e}")
            self.is_connected = False
            return False
    
    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment counter
        
        Args:
            key: Cache key
            amount: Increment amount
        
        Returns:
            New value or None if error
        """
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return None
        
        try:
            cache_key = self._make_key(key)
            return await self.redis_client.incrby(cache_key, amount)
        
        except RedisError as e:
            logger.error(f"Cache incr error: {e}")
            self.is_connected = False
            return None
    
    async def decr(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Decrement counter
        
        Args:
            key: Cache key
            amount: Decrement amount
        
        Returns:
            New value or None if error
        """
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return None
        
        try:
            cache_key = self._make_key(key)
            return await self.redis_client.decrby(cache_key, amount)
        
        except RedisError as e:
            logger.error(f"Cache decr error: {e}")
            self.is_connected = False
            return None
    
    async def hset(self, key: str, field: str, value: Any) -> bool:
        """
        Set hash field
        
        Args:
            key: Cache key
            field: Hash field
            value: Value to set
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return False
        
        try:
            cache_key = self._make_key(key)
            data = self._serialize_value(value)
            return bool(await self.redis_client.hset(cache_key, field, data))
        
        except RedisError as e:
            logger.error(f"Cache hset error: {e}")
            self.is_connected = False
            return False
    
    async def hget(self, key: str, field: str, default: Any = None) -> Any:
        """
        Get hash field
        
        Args:
            key: Cache key
            field: Hash field
            default: Default value if field not found
        
        Returns:
            Field value or default
        """
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return default
        
        try:
            cache_key = self._make_key(key)
            data = await self.redis_client.hget(cache_key, field)
            
            if data is None:
                return default
            
            return self._deserialize_value(data)
        
        except RedisError as e:
            logger.error(f"Cache hget error: {e}")
            self.is_connected = False
            return default
    
    async def hgetall(self, key: str) -> Dict[str, Any]:
        """
        Get all hash fields
        
        Args:
            key: Cache key
        
        Returns:
            Dictionary of all fields
        """
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return {}
        
        try:
            cache_key = self._make_key(key)
            data = await self.redis_client.hgetall(cache_key)
            
            result = {}
            for field, value in data.items():
                result[field.decode('utf-8') if isinstance(field, bytes) else field] = (
                    self._deserialize_value(value)
                )
            
            return result
        
        except RedisError as e:
            logger.error(f"Cache hgetall error: {e}")
            self.is_connected = False
            return {}
    
    async def sadd(self, key: str, *members: Any) -> int:
        """
        Add members to set
        
        Args:
            key: Cache key
            *members: Members to add
        
        Returns:
            Number of members added
        """
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return 0
        
        try:
            cache_key = self._make_key(key)
            serialized_members = [self._serialize_value(m) for m in members]
            return await self.redis_client.sadd(cache_key, *serialized_members)
        
        except RedisError as e:
            logger.error(f"Cache sadd error: {e}")
            self.is_connected = False
            return 0
    
    async def smembers(self, key: str) -> List[Any]:
        """
        Get all set members
        
        Args:
            key: Cache key
        
        Returns:
            List of set members
        """
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return []
        
        try:
            cache_key = self._make_key(key)
            data = await self.redis_client.smembers(cache_key)
            
            return [self._deserialize_value(m) for m in data]
        
        except RedisError as e:
            logger.error(f"Cache smembers error: {e}")
            self.is_connected = False
            return []
    
    async def clear_all(self) -> int:
        """
        Clear all cache keys in namespace
        
        Returns:
            Number of keys deleted
        """
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return 0
        
        try:
            pattern = f"{self.namespace}:*"
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                deleted = await self.redis_client.delete(*keys)
                self.stats["deletes"] += deleted
                return deleted
            
            return 0
        
        except RedisError as e:
            logger.error(f"Cache clear_all error: {e}")
            self.is_connected = False
            return 0
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear cache keys matching pattern
        
        Args:
            pattern: Pattern to match (supports * wildcard)
        
        Returns:
            Number of keys deleted
        """
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return 0
        
        try:
            full_pattern = f"{self.namespace}:{pattern}"
            keys = await self.redis_client.keys(full_pattern)
            
            if keys:
                deleted = await self.redis_client.delete(*keys)
                self.stats["deletes"] += deleted
                return deleted
            
            return 0
        
        except RedisError as e:
            logger.error(f"Cache clear_pattern error: {e}")
            self.is_connected = False
            return 0
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.is_connected:
            try:
                await self.connect()
            except:
                return self.stats
        
        try:
            # Get Redis info
            info = await self.redis_client.info()
            
            # Calculate hit rate
            hits = self.stats["hits"]
            misses = self.stats["misses"]
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0.0
            miss_rate = misses / total if total > 0 else 0.0
            
            stats = {
                **self.stats,
                "hit_rate": hit_rate,
                "miss_rate": miss_rate,
                "connected": self.is_connected,
                "redis": {
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "uptime_in_seconds": info.get("uptime_in_seconds", 0),
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return stats
        
        except RedisError as e:
            logger.error(f"Cache statistics error: {e}")
            self.is_connected = False
            return self.stats
    
    async def get_status(self) -> Dict[str, Any]:
        """Get cache status"""
        try:
            connected = self.is_connected
            keys = 0
            memory_used = 0
            
            if connected and self.redis_client:
                try:
                    # Get number of keys in namespace
                    pattern = f"{self.namespace}:*"
                    keys = len(await self.redis_client.keys(pattern))
                    
                    # Get memory info
                    info = await self.redis_client.info("memory")
                    memory_used = info.get("used_memory", 0)
                
                except RedisError:
                    connected = False
            
            return {
                "connected": connected,
                "keys": keys,
                "memory_used": memory_used,
                "namespace": self.namespace,
                "compression": self.compression,
                "default_ttl": self.default_ttl,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Cache status error: {e}")
            return {
                "connected": False,
                "keys": 0,
                "memory_used": 0,
                "error": str(e)
            }
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired keys"""
        while self.is_connected:
            try:
                await asyncio.sleep(self.policies["cleanup_interval"])
                await self._cleanup_expired_keys()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _cleanup_expired_keys(self):
        """Clean up expired keys"""
        if not self.is_connected or not self.redis_client:
            return
        
        try:
            # Redis automatically expires keys, but we can clean up any left
            pattern = f"{self.namespace}:*"
            keys = await self.redis_client.keys(pattern)
            
            expired_count = 0
            for key in keys:
                ttl = await self.redis_client.ttl(key)
                if ttl == -2:  # Key doesn't exist (shouldn't happen)
                    await self.redis_client.delete(key)
                    expired_count += 1
            
            if expired_count > 0:
                logger.debug(f"Cleaned up {expired_count} expired keys")
        
        except RedisError as e:
            logger.error(f"Cleanup expired keys error: {e}")
    
    async def cache_function(
        self,
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None,
        unless: Optional[Callable] = None
    ):
        """
        Decorator for caching function results
        
        Args:
            ttl: Cache TTL in seconds
            key_func: Function to generate cache key from args/kwargs
            unless: Function that returns True to skip caching
        
        Returns:
            Decorator function
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Check if caching should be skipped
                if unless and unless(*args, **kwargs):
                    return await func(*args, **kwargs)
                
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_parts = [
                        func.__module__,
                        func.__name__,
                        str(args),
                        str(sorted(kwargs.items()))
                    ]
                    key_string = ":".join(key_parts)
                    cache_key = hashlib.sha256(key_string.encode()).hexdigest()
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}: {cache_key}")
                    return cached_result
                
                # Execute function
                logger.debug(f"Cache miss for {func.__name__}: {cache_key}")
                result = await func(*args, **kwargs)
                
                # Cache result
                await self.set(cache_key, result, ttl=ttl or self.default_ttl)
                
                return result
            
            return wrapper
        
        return decorator


# Memory cache fallback for when Redis is unavailable
class MemoryCache:
    """In-memory cache fallback"""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expiry_time)
        self.access_times: Dict[str, float] = {}  # key -> last_access_time
        self.lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from memory cache"""
        async with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return default
            
            value, expiry = self.cache[key]
            
            # Check if expired
            if time.time() > expiry:
                del self.cache[key]
                del self.access_times[key]
                self.stats["misses"] += 1
                return default
            
            # Update access time
            self.access_times[key] = time.time()
            self.stats["hits"] += 1
            
            return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache"""
        async with self.lock:
            # Evict if needed
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_oldest()
            
            ttl = ttl or self.default_ttl
            expiry = time.time() + ttl
            
            self.cache[key] = (value, expiry)
            self.access_times[key] = time.time()
            self.stats["sets"] += 1
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from memory cache"""
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                self.stats["deletes"] += 1
                return True
            return False
    
    async def _evict_oldest(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        # Find least recently used key
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        
        # Remove from cache
        if oldest_key in self.cache:
            del self.cache[oldest_key]
        if oldest_key in self.access_times:
            del self.access_times[oldest_key]
        
        self.stats["evictions"] += 1
    
    async def clear(self):
        """Clear all cache"""
        async with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory cache statistics"""
        async with self.lock:
            return {
                **self.stats,
                "size": len(self.cache),
                "max_size": self.max_size,
                "default_ttl": self.default_ttl
            }


# Export
__all__ = [
    'CacheManager',
    'MemoryCache'
]
"""
Management routes for system administration and monitoring
"""

import asyncio
import psutil
import GPUtil
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import json

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..middleware.authentication import require_auth
from ..schemas.request_models import (
    WorldQuery,
    BatchRequest,
    ModelManagementRequest,
    CacheManagementRequest
)
from ..schemas.response_models import (
    WorldStatus,
    SystemStatus,
    ModelStatus,
    CacheStatus,
    APIError,
    SuccessResponse,
    StatisticsResponse
)
from ..utils.cache_manager import CacheManager
from ..utils.async_processor import AsyncProcessor
from src.core.world_model.inference.generator import WorldGenerator
from src.core.multimodal.encoders.text_encoder import TextEncoder
from src.core.multimodal.encoders.vision_encoder import VisionEncoder
from src.utils.logging_config import setup_logging
from configs.model.inference import InferenceConfig
from configs.api.server import APIConfig

logger = setup_logging("management_routes")

router = APIRouter()

# Global components
cache_manager: Optional[CacheManager] = None
async_processor: Optional[AsyncProcessor] = None
world_generator: Optional[WorldGenerator] = None
text_encoder: Optional[TextEncoder] = None
vision_encoder: Optional[VisionEncoder] = None
api_config: Optional[APIConfig] = None


class ManagementSession:
    """Manages system management operations"""
    
    def __init__(self):
        self.operations: Dict[str, Dict] = {}
        self.system_metrics: List[Dict] = []
        self.api_metrics: Dict[str, Any] = {
            "requests": {
                "total": 0,
                "by_endpoint": {},
                "by_status": {},
                "by_hour": {}
            },
            "response_times": {
                "p50": 0,
                "p90": 0,
                "p99": 0,
                "average": 0
            },
            "errors": {
                "total": 0,
                "by_type": {},
                "by_endpoint": {}
            }
        }
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        user_id: Optional[str] = None
    ):
        """Record API request metrics"""
        # Update total requests
        self.api_metrics["requests"]["total"] += 1
        
        # Update by endpoint
        endpoint_key = f"{method} {endpoint}"
        if endpoint_key not in self.api_metrics["requests"]["by_endpoint"]:
            self.api_metrics["requests"]["by_endpoint"][endpoint_key] = 0
        self.api_metrics["requests"]["by_endpoint"][endpoint_key] += 1
        
        # Update by status
        status_group = f"{status_code // 100}xx"
        if status_group not in self.api_metrics["requests"]["by_status"]:
            self.api_metrics["requests"]["by_status"][status_group] = 0
        self.api_metrics["requests"]["by_status"][status_group] += 1
        
        # Update by hour
        hour = datetime.utcnow().strftime("%Y-%m-%d %H:00")
        if hour not in self.api_metrics["requests"]["by_hour"]:
            self.api_metrics["requests"]["by_hour"][hour] = 0
        self.api_metrics["requests"]["by_hour"][hour] += 1
        
        # Update response times
        response_times = self.api_metrics.get("response_time_samples", [])
        response_times.append(response_time)
        
        # Keep only last 1000 samples
        if len(response_times) > 1000:
            response_times = response_times[-1000:]
        
        # Calculate percentiles
        if response_times:
            sorted_times = sorted(response_times)
            n = len(sorted_times)
            
            self.api_metrics["response_times"]["p50"] = sorted_times[int(n * 0.5)]
            self.api_metrics["response_times"]["p90"] = sorted_times[int(n * 0.9)]
            self.api_metrics["response_times"]["p99"] = sorted_times[int(n * 0.99)]
            self.api_metrics["response_times"]["average"] = sum(sorted_times) / n
        
        self.api_metrics["response_time_samples"] = response_times
    
    def record_error(self, error_type: str, endpoint: str, message: str):
        """Record error metrics"""
        # Update total errors
        self.api_metrics["errors"]["total"] += 1
        
        # Update by type
        if error_type not in self.api_metrics["errors"]["by_type"]:
            self.api_metrics["errors"]["by_type"][error_type] = 0
        self.api_metrics["errors"]["by_type"][error_type] += 1
        
        # Update by endpoint
        if endpoint not in self.api_metrics["errors"]["by_endpoint"]:
            self.api_metrics["errors"]["by_endpoint"][endpoint] = 0
        self.api_metrics["errors"]["by_endpoint"][endpoint] += 1
    
    def record_system_metrics(self):
        """Record system metrics"""
        try:
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu": {
                    "percent": psutil.cpu_percent(interval=1),
                    "count": psutil.cpu_count(),
                    "freq": psutil.cpu_freq().current if psutil.cpu_freq() else None
                },
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent,
                    "used": psutil.virtual_memory().used
                },
                "disk": {
                    "total": psutil.disk_usage("/").total,
                    "used": psutil.disk_usage("/").used,
                    "percent": psutil.disk_usage("/").percent,
                    "free": psutil.disk_usage("/").free
                }
            }
            
            # GPU metrics if available
            try:
                gpus = GPUtil.getGPUs()
                metrics["gpu"] = []
                for gpu in gpus:
                    metrics["gpu"].append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load * 100,
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "memory_free": gpu.memoryFree,
                        "temperature": gpu.temperature
                    })
            except Exception:
                metrics["gpu"] = None
            
            # Process metrics
            try:
                process = psutil.Process()
                metrics["process"] = {
                    "pid": process.pid,
                    "memory_rss": process.memory_info().rss,
                    "memory_percent": process.memory_percent(),
                    "cpu_percent": process.cpu_percent(),
                    "threads": process.num_threads(),
                    "open_files": len(process.open_files()),
                    "connections": len(process.connections())
                }
            except Exception:
                metrics["process"] = None
            
            self.system_metrics.append(metrics)
            
            # Keep only last 1000 metrics
            if len(self.system_metrics) > 1000:
                self.system_metrics = self.system_metrics[-1000:]
        
        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")
    
    def get_metrics_summary(self, hours: int = 24) -> Dict:
        """Get metrics summary for the last N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter recent system metrics
        recent_metrics = [
            m for m in self.system_metrics
            if datetime.fromisoformat(m["timestamp"].replace("Z", "")) > cutoff
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        summary = {
            "period_hours": hours,
            "sample_count": len(recent_metrics),
            "cpu_percent_avg": sum(m["cpu"]["percent"] for m in recent_metrics) / len(recent_metrics),
            "memory_percent_avg": sum(m["memory"]["percent"] for m in recent_metrics) / len(recent_metrics),
            "disk_percent_avg": sum(m["disk"]["percent"] for m in recent_metrics) / len(recent_metrics),
            "peak_cpu": max(m["cpu"]["percent"] for m in recent_metrics),
            "peak_memory": max(m["memory"]["percent"] for m in recent_metrics),
            "peak_disk": max(m["disk"]["percent"] for m in recent_metrics),
            "latest_timestamp": recent_metrics[-1]["timestamp"]
        }
        
        # Add GPU summary if available
        gpu_metrics = [m for m in recent_metrics if m.get("gpu")]
        if gpu_metrics and gpu_metrics[0]["gpu"]:
            summary["gpu_count"] = len(gpu_metrics[0]["gpu"])
            summary["gpu_load_avg"] = sum(
                gpu["load"] for m in gpu_metrics for gpu in m["gpu"]
            ) / (len(gpu_metrics) * summary["gpu_count"])
        
        return summary
    
    def clear_old_metrics(self, max_age_hours: int = 168):  # 1 week
        """Clear old metrics"""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        self.system_metrics = [
            m for m in self.system_metrics
            if datetime.fromisoformat(m["timestamp"].replace("Z", "")) > cutoff
        ]


# Initialize management session
management_session = ManagementSession()


@router.on_event("startup")
async def startup_event():
    """Initialize management components"""
    global cache_manager, async_processor, world_generator, text_encoder, vision_encoder, api_config
    
    logger.info("Initializing management components...")
    
    try:
        # Load configurations
        api_config = APIConfig()
        
        # Start metrics collection
        asyncio.create_task(collect_metrics_periodically())
        
        logger.info("Management components initialized successfully")
    
    except Exception as e:
        logger.error(f"Failed to initialize management components: {e}")
        raise


async def collect_metrics_periodically():
    """Periodically collect system metrics"""
    while True:
        try:
            management_session.record_system_metrics()
            # Clean up old metrics weekly
            management_session.clear_old_metrics()
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
        
        # Collect every 60 seconds
        await asyncio.sleep(60)


@router.get("/health", response_model=SystemStatus)
async def health_check(
    detailed: bool = Query(False, description="Include detailed component checks"),
    api_key=Depends(require_auth("read"))
):
    """
    Comprehensive system health check
    
    Returns overall system status and component health
    """
    
    logger.info("Health check requested")
    
    try:
        health_status = "healthy"
        components = {}
        issues = []
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        
        if cpu_percent > 90:
            health_status = "degraded"
            issues.append(f"High CPU usage: {cpu_percent}%")
        
        if memory.percent > 90:
            health_status = "degraded"
            issues.append(f"High memory usage: {memory.percent}%")
        
        if disk.percent > 90:
            health_status = "degraded"
            issues.append(f"High disk usage: {disk.percent}%")
        
        components["system"] = {
            "status": "healthy" if cpu_percent < 80 and memory.percent < 80 and disk.percent < 80 else "degraded",
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if detailed:
            # Check cache manager
            if cache_manager:
                try:
                    cache_status = await cache_manager.get_status()
                    components["cache"] = {
                        "status": "healthy",
                        "connected": cache_status["connected"],
                        "keys": cache_status["keys"],
                        "memory_used": cache_status["memory_used"]
                    }
                except Exception as e:
                    components["cache"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health_status = "degraded"
                    issues.append(f"Cache error: {e}")
            else:
                components["cache"] = {
                    "status": "disabled",
                    "connected": False
                }
            
            # Check async processor
            if async_processor:
                try:
                    queue_status = await async_processor.get_status()
                    components["async_processor"] = {
                        "status": "healthy",
                        "queue_size": queue_status["queue_size"],
                        "active_tasks": queue_status["active_tasks"],
                        "completed_tasks": queue_status["completed_tasks"]
                    }
                except Exception as e:
                    components["async_processor"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    health_status = "degraded"
                    issues.append(f"Async processor error: {e}")
            else:
                components["async_processor"] = {
                    "status": "disabled",
                    "queue_size": 0
                }
            
            # Check AI models
            model_components = {}
            
            if world_generator:
                try:
                    model_loaded = await world_generator.is_loaded()
                    model_components["world_generator"] = {
                        "status": "loaded" if model_loaded else "unloaded",
                        "loaded": model_loaded
                    }
                    if not model_loaded:
                        health_status = "degraded"
                        issues.append("World generator not loaded")
                except Exception as e:
                    model_components["world_generator"] = {
                        "status": "error",
                        "error": str(e)
                    }
                    health_status = "degraded"
                    issues.append(f"World generator error: {e}")
            
            if text_encoder:
                try:
                    model_loaded = await text_encoder.is_loaded()
                    model_components["text_encoder"] = {
                        "status": "loaded" if model_loaded else "unloaded",
                        "loaded": model_loaded
                    }
                    if not model_loaded:
                        health_status = "degraded"
                        issues.append("Text encoder not loaded")
                except Exception as e:
                    model_components["text_encoder"] = {
                        "status": "error",
                        "error": str(e)
                    }
                    health_status = "degraded"
                    issues.append(f"Text encoder error: {e}")
            
            if vision_encoder:
                try:
                    model_loaded = await vision_encoder.is_loaded()
                    model_components["vision_encoder"] = {
                        "status": "loaded" if model_loaded else "unloaded",
                        "loaded": model_loaded
                    }
                    if not model_loaded:
                        health_status = "degraded"
                        issues.append("Vision encoder not loaded")
                except Exception as e:
                    model_components["vision_encoder"] = {
                        "status": "error",
                        "error": str(e)
                    }
                    health_status = "degraded"
                    issues.append(f"Vision encoder error: {e}")
            
            components["models"] = model_components
        
        return SystemStatus(
            status=health_status,
            timestamp=datetime.utcnow(),
            uptime=get_uptime(),
            version="1.0.0",
            components=components,
            issues=issues if issues else None
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


def get_uptime() -> int:
    """Get system uptime in seconds"""
    try:
        return int(psutil.boot_time())
    except:
        return 0


@router.get("/metrics/system", response_model=Dict)
async def get_system_metrics(
    hours: int = Query(24, ge=1, le=168, description="Hours of metrics to return"),
    api_key=Depends(require_auth("admin"))
):
    """
    Get system metrics
    
    Returns detailed system metrics for monitoring
    """
    
    logger.info(f"System metrics requested for last {hours} hours")
    
    try:
        # Get metrics summary
        summary = management_session.get_metrics_summary(hours)
        
        # Get recent raw metrics (last 100)
        recent_metrics = management_session.system_metrics[-100:] if management_session.system_metrics else []
        
        return {
            "summary": summary,
            "recent_metrics": recent_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system metrics: {str(e)}"
        )


@router.get("/metrics/api", response_model=Dict)
async def get_api_metrics(
    hours: int = Query(24, ge=1, le=168, description="Hours of metrics to return"),
    api_key=Depends(require_auth("admin"))
):
    """
    Get API metrics
    
    Returns API usage and performance metrics
    """
    
    logger.info(f"API metrics requested for last {hours} hours")
    
    try:
        # Filter requests by hour
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        filtered_hours = {
            hour: count for hour, count in management_session.api_metrics["requests"]["by_hour"].items()
            if datetime.strptime(hour, "%Y-%m-%d %H:00") > cutoff
        }
        
        # Calculate request rate
        total_requests = sum(filtered_hours.values())
        request_rate = total_requests / hours if hours > 0 else 0
        
        return {
            "requests": {
                "total": management_session.api_metrics["requests"]["total"],
                "period_total": total_requests,
                "rate_per_hour": request_rate,
                "by_endpoint": management_session.api_metrics["requests"]["by_endpoint"],
                "by_status": management_session.api_metrics["requests"]["by_status"],
                "by_hour": filtered_hours
            },
            "response_times": management_session.api_metrics["response_times"],
            "errors": management_session.api_metrics["errors"],
            "timestamp": datetime.utcnow().isoformat(),
            "period_hours": hours
        }
    
    except Exception as e:
        logger.error(f"Failed to get API metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get API metrics: {str(e)}"
        )


@router.get("/models/status", response_model=List[ModelStatus])
async def get_model_status(
    api_key=Depends(require_auth("admin"))
):
    """
    Get AI model status
    
    Returns status of all loaded AI models
    """
    
    logger.info("Model status requested")
    
    try:
        models = []
        
        # World generator
        if world_generator:
            try:
                loaded = await world_generator.is_loaded()
                info = await world_generator.get_model_info()
                models.append(
                    ModelStatus(
                        name="world_generator",
                        status="loaded" if loaded else "unloaded",
                        loaded=loaded,
                        size_mb=info.get("size_mb", 0),
                        version=info.get("version", "unknown"),
                        device=info.get("device", "cpu"),
                        loaded_at=info.get("loaded_at")
                    )
                )
            except Exception as e:
                models.append(
                    ModelStatus(
                        name="world_generator",
                        status="error",
                        loaded=False,
                        error=str(e)
                    )
                )
        
        # Text encoder
        if text_encoder:
            try:
                loaded = await text_encoder.is_loaded()
                info = await text_encoder.get_model_info()
                models.append(
                    ModelStatus(
                        name="text_encoder",
                        status="loaded" if loaded else "unloaded",
                        loaded=loaded,
                        size_mb=info.get("size_mb", 0),
                        version=info.get("version", "unknown"),
                        device=info.get("device", "cpu"),
                        loaded_at=info.get("loaded_at")
                    )
                )
            except Exception as e:
                models.append(
                    ModelStatus(
                        name="text_encoder",
                        status="error",
                        loaded=False,
                        error=str(e)
                    )
                )
        
        # Vision encoder
        if vision_encoder:
            try:
                loaded = await vision_encoder.is_loaded()
                info = await vision_encoder.get_model_info()
                models.append(
                    ModelStatus(
                        name="vision_encoder",
                        status="loaded" if loaded else "unloaded",
                        loaded=loaded,
                        size_mb=info.get("size_mb", 0),
                        version=info.get("version", "unknown"),
                        device=info.get("device", "cpu"),
                        loaded_at=info.get("loaded_at")
                    )
                )
            except Exception as e:
                models.append(
                    ModelStatus(
                        name="vision_encoder",
                        status="error",
                        loaded=False,
                        error=str(e)
                    )
                )
        
        return models
    
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model status: {str(e)}"
        )


@router.post("/models/load", response_model=SuccessResponse)
async def load_models(
    request: ModelManagementRequest = Body(...),
    api_key=Depends(require_auth("admin"))
):
    """
    Load AI models
    
    Loads specified AI models into memory
    """
    
    logger.info(f"Model load requested: {request.models if request.models else 'all'}")
    
    try:
        models_to_load = request.models or ["world_generator", "text_encoder", "vision_encoder"]
        loaded_models = []
        failed_models = []
        
        for model_name in models_to_load:
            try:
                if model_name == "world_generator" and world_generator:
                    await world_generator.load_model()
                    loaded_models.append(model_name)
                elif model_name == "text_encoder" and text_encoder:
                    await text_encoder.load()
                    loaded_models.append(model_name)
                elif model_name == "vision_encoder" and vision_encoder:
                    await vision_encoder.load()
                    loaded_models.append(model_name)
                else:
                    failed_models.append(f"{model_name}: not available")
            except Exception as e:
                failed_models.append(f"{model_name}: {str(e)}")
        
        message = f"Loaded {len(loaded_models)} models"
        if failed_models:
            message += f", failed: {', '.join(failed_models)}"
        
        return SuccessResponse(
            success=len(failed_models) == 0,
            message=message,
            data={
                "loaded": loaded_models,
                "failed": failed_models
            }
        )
    
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load models: {str(e)}"
        )


@router.post("/models/unload", response_model=SuccessResponse)
async def unload_models(
    request: ModelManagementRequest = Body(...),
    api_key=Depends(require_auth("admin"))
):
    """
    Unload AI models
    
    Unloads specified AI models from memory
    """
    
    logger.info(f"Model unload requested: {request.models if request.models else 'all'}")
    
    try:
        models_to_unload = request.models or ["world_generator", "text_encoder", "vision_encoder"]
        unloaded_models = []
        failed_models = []
        
        for model_name in models_to_unload:
            try:
                if model_name == "world_generator" and world_generator:
                    await world_generator.unload_model()
                    unloaded_models.append(model_name)
                elif model_name == "text_encoder" and text_encoder:
                    await text_encoder.unload()
                    unloaded_models.append(model_name)
                elif model_name == "vision_encoder" and vision_encoder:
                    await vision_encoder.unload()
                    unloaded_models.append(model_name)
                else:
                    failed_models.append(f"{model_name}: not available")
            except Exception as e:
                failed_models.append(f"{model_name}: {str(e)}")
        
        message = f"Unloaded {len(unloaded_models)} models"
        if failed_models:
            message += f", failed: {', '.join(failed_models)}"
        
        return SuccessResponse(
            success=len(failed_models) == 0,
            message=message,
            data={
                "unloaded": unloaded_models,
                "failed": failed_models
            }
        )
    
    except Exception as e:
        logger.error(f"Failed to unload models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload models: {str(e)}"
        )


@router.get("/cache/status", response_model=CacheStatus)
async def get_cache_status(
    detailed: bool = Query(False, description="Include detailed cache statistics"),
    api_key=Depends(require_auth("admin"))
):
    """
    Get cache status
    
    Returns cache statistics and status
    """
    
    logger.info("Cache status requested")
    
    try:
        if not cache_manager:
            return CacheStatus(
                status="disabled",
                connected=False,
                keys=0,
                memory_used=0,
                message="Cache manager not initialized"
            )
        
        status = await cache_manager.get_status()
        
        if detailed:
            # Get detailed statistics
            stats = await cache_manager.get_statistics()
            status.update(stats)
        
        return CacheStatus(**status)
    
    except Exception as e:
        logger.error(f"Failed to get cache status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache status: {str(e)}"
        )


@router.post("/cache/clear", response_model=SuccessResponse)
async def clear_cache(
    request: CacheManagementRequest = Body(...),
    api_key=Depends(require_auth("admin"))
):
    """
    Clear cache
    
    Clears specified cache entries or entire cache
    """
    
    logger.info(f"Cache clear requested: pattern={request.pattern}, all={request.clear_all}")
    
    try:
        if not cache_manager:
            return SuccessResponse(
                success=False,
                message="Cache manager not initialized",
                data={}
            )
        
        if request.clear_all:
            # Clear entire cache
            cleared = await cache_manager.clear_all()
            message = f"Cleared entire cache ({cleared} keys)"
        elif request.pattern:
            # Clear by pattern
            cleared = await cache_manager.clear_pattern(request.pattern)
            message = f"Cleared cache entries matching pattern '{request.pattern}' ({cleared} keys)"
        else:
            return SuccessResponse(
                success=False,
                message="No clear operation specified",
                data={}
            )
        
        return SuccessResponse(
            success=True,
            message=message,
            data={"keys_cleared": cleared}
        )
    
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(
    period: str = Query("24h", regex="^(1h|24h|7d|30d)$", description="Time period"),
    api_key=Depends(require_auth("admin"))
):
    """
    Get system statistics
    
    Returns comprehensive system statistics for the specified period
    """
    
    logger.info(f"Statistics requested for period: {period}")
    
    try:
        # Parse period
        if period == "1h":
            hours = 1
        elif period == "24h":
            hours = 24
        elif period == "7d":
            hours = 168
        elif period == "30d":
            hours = 720
        else:
            hours = 24
        
        # Get system metrics summary
        system_summary = management_session.get_metrics_summary(hours)
        
        # Calculate API statistics
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        api_requests = {
            hour: count for hour, count in management_session.api_metrics["requests"]["by_hour"].items()
            if datetime.strptime(hour, "%Y-%m-%d %H:00") > cutoff
        }
        
        total_requests = sum(api_requests.values())
        avg_requests_per_hour = total_requests / hours if hours > 0 else 0
        
        # Get error rate
        total_errors = management_session.api_metrics["errors"]["total"]
        error_rate = (total_errors / max(management_session.api_metrics["requests"]["total"], 1)) * 100
        
        return StatisticsResponse(
            period_hours=hours,
            system=system_summary,
            api={
                "total_requests": total_requests,
                "avg_requests_per_hour": avg_requests_per_hour,
                "error_rate_percent": error_rate,
                "response_time_p50": management_session.api_metrics["response_times"]["p50"],
                "response_time_p90": management_session.api_metrics["response_times"]["p90"],
                "response_time_p99": management_session.api_metrics["response_times"]["p99"],
                "peak_hour": max(api_requests.values()) if api_requests else 0
            },
            cache={
                "keys": (await cache_manager.get_status())["keys"] if cache_manager else 0,
                "hit_rate": (await cache_manager.get_statistics())["hit_rate"] if cache_manager else 0
            },
            models={
                "loaded": sum(1 for model in await get_model_status(api_key) if model.loaded),
                "total": len(await get_model_status(api_key))
            },
            timestamp=datetime.utcnow()
        )
    
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get("/config", response_model=Dict)
async def get_configuration(
    api_key=Depends(require_auth("admin"))
):
    """
    Get current configuration
    
    Returns current system configuration (sensitive values masked)
    """
    
    logger.info("Configuration requested")
    
    try:
        if not api_config:
            return {"error": "Configuration not loaded"}
        
        # Return sanitized configuration
        config_dict = api_config.dict()
        
        # Mask sensitive values
        sensitive_keys = ["api_keys_file", "redis_url", "secret_key"]
        for key in sensitive_keys:
            if key in config_dict and config_dict[key]:
                config_dict[key] = "***MASKED***"
        
        return {
            "api": config_dict,
            "models": InferenceConfig().dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get configuration: {str(e)}"
        )


@router.post("/maintenance/cleanup", response_model=SuccessResponse)
async def run_maintenance_cleanup(
    days_old: int = Query(7, ge=1, le=365, description="Clean up data older than N days"),
    include_cache: bool = Query(True, description="Include cache cleanup"),
    include_files: bool = Query(True, description="Include temporary files cleanup"),
    api_key=Depends(require_auth("admin"))
):
    """
    Run maintenance cleanup
    
    Cleans up old data, cache, and temporary files
    """
    
    logger.info(f"Maintenance cleanup requested: {days_old} days old")
    
    try:
        cleanup_results = {}
        
        # Clean up old cache entries
        if include_cache and cache_manager:
            try:
                # Pattern for old cache entries (assuming timestamp in key)
                pattern = f"*:old:*"
                cleared = await cache_manager.clear_pattern(pattern)
                cleanup_results["cache"] = {
                    "cleared": cleared,
                    "status": "success"
                }
            except Exception as e:
                cleanup_results["cache"] = {
                    "cleared": 0,
                    "status": "failed",
                    "error": str(e)
                }
        
        # Clean up temporary files
        if include_files:
            try:
                # This would clean up temporary export files, etc.
                # In a real implementation, you would have a file cleanup system
                cleanup_results["files"] = {
                    "cleared": 0,
                    "status": "not_implemented"
                }
            except Exception as e:
                cleanup_results["files"] = {
                    "cleared": 0,
                    "status": "failed",
                    "error": str(e)
                }
        
        # Clean up old metrics
        management_session.clear_old_metrics(max_age_hours=days_old * 24)
        cleanup_results["metrics"] = {
            "cleared": "old metrics",
            "status": "success"
        }
        
        return SuccessResponse(
            success=all(result["status"] == "success" for result in cleanup_results.values()),
            message=f"Maintenance cleanup completed for data older than {days_old} days",
            data=cleanup_results
        )
    
    except Exception as e:
        logger.error(f"Maintenance cleanup failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Maintenance cleanup failed: {str(e)}"
        )


@router.post("/restart", response_model=SuccessResponse)
async def restart_service(
    component: str = Query(None, description="Component to restart (api, models, cache, all)"),
    graceful: bool = Query(True, description="Perform graceful restart"),
    api_key=Depends(require_auth("admin"))
):
    """
    Restart service components
    
    Restarts specified service components
    """
    
    logger.warning(f"Service restart requested: component={component or 'all'}, graceful={graceful}")
    
    # In a real implementation, this would trigger a restart
    # For now, we'll just log and return success
    
    return SuccessResponse(
        success=True,
        message=f"Restart scheduled for {component or 'all'} components",
        data={
            "component": component or "all",
            "graceful": graceful,
            "scheduled_at": datetime.utcnow().isoformat(),
            "note": "In production, this would trigger an actual restart"
        }
    )


@router.get("/logs", response_model=Dict)
async def get_logs(
    lines: int = Query(100, ge=1, le=10000, description="Number of lines to return"),
    level: str = Query(None, regex="^(debug|info|warning|error|critical)$", description="Filter by log level"),
    component: str = Query(None, description="Filter by component"),
    api_key=Depends(require_auth("admin"))
):
    """
    Get system logs
    
    Returns recent system logs
    """
    
    logger.info(f"Logs requested: lines={lines}, level={level}, component={component}")
    
    try:
        # In a real implementation, you would read from log files
        # For now, return mock logs
        logs = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "level": "info",
                "component": "management_routes",
                "message": f"Logs requested by {api_key.name}",
                "extra": {}
            }
        ]
        
        return {
            "logs": logs,
            "total_lines": len(logs),
            "filters": {
                "lines": lines,
                "level": level,
                "component": component
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get logs: {str(e)}"
        )


# Middleware to record API metrics
@router.middleware("http")
async def record_api_metrics(request, call_next):
    """Middleware to record API metrics"""
    start_time = datetime.utcnow()
    
    try:
        response = await call_next(request)
        end_time = datetime.utcnow()
        response_time = (end_time - start_time).total_seconds()
        
        # Record request
        management_session.record_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            response_time=response_time,
            user_id=request.state.api_key.key_id if hasattr(request.state, 'api_key') else None
        )
        
        # Record errors for 4xx and 5xx responses
        if response.status_code >= 400:
            error_type = "client_error" if response.status_code < 500 else "server_error"
            management_session.record_error(
                error_type=error_type,
                endpoint=request.url.path,
                message=f"HTTP {response.status_code}"
            )
        
        return response
    
    except Exception as e:
        end_time = datetime.utcnow()
        response_time = (end_time - start_time).total_seconds()
        
        # Record error
        management_session.record_error(
            error_type="exception",
            endpoint=request.url.path,
            message=str(e)
        )
        
        raise
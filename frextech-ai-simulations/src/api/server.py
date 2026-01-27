"""
FastAPI server for FrexTech AI Simulations API
"""

import os
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from .middleware.authentication import (
    authenticate_token,
    APIKeyMiddleware,
    setup_api_keys
)
from .middleware.rate_limiter import RateLimiter, RateLimitMiddleware
from .middleware.error_handler import APIErrorHandler
from .routes.generation import router as generation_router
from .routes.editing import router as editing_router
from .routes.export import router as export_router
from .routes.management import router as management_router
from .utils.cache_manager import CacheManager
from .utils.async_processor import AsyncProcessor
from configs.api.server import APIConfig
from src.utils.logging_config import setup_logging

# Setup logging
logger = setup_logging("api_server")

# Global instances
cache_manager: Optional[CacheManager] = None
async_processor: Optional[AsyncProcessor] = None
rate_limiter: Optional[RateLimiter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events"""
    global cache_manager, async_processor, rate_limiter
    
    # Startup
    logger.info("Starting FrexTech AI Simulations API Server")
    
    # Load configuration
    config = APIConfig()
    
    # Initialize components
    cache_manager = CacheManager(
        redis_url=config.redis_url,
        default_ttl=config.cache_ttl
    )
    
    async_processor = AsyncProcessor(
        max_workers=config.max_workers,
        queue_size=config.queue_size
    )
    
    rate_limiter = RateLimiter(
        redis_url=config.redis_url,
        default_rate_limit=config.rate_limit
    )
    
    # Setup API keys
    setup_api_keys(config.api_keys_file)
    
    logger.info(f"API Server started on {config.host}:{config.port}")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Debug mode: {config.debug}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Server")
    if async_processor:
        await async_processor.shutdown()
    if cache_manager:
        await cache_manager.close()
    logger.info("API Server shutdown complete")


def create_app(config: Optional[APIConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application"""
    if config is None:
        config = APIConfig()
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="FrexTech AI Simulations API",
        description="API for generating, editing, and managing AI-simulated worlds",
        version="1.0.0",
        docs_url="/docs" if config.enable_docs else None,
        redoc_url="/redoc" if config.enable_docs else None,
        openapi_url="/openapi.json" if config.enable_openapi else None,
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add GZip middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add custom middleware
    app.add_middleware(APIErrorHandler)
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
    app.add_middleware(APIKeyMiddleware)
    
    # Include routers
    app.include_router(
        generation_router,
        prefix="/api/v1/generation",
        tags=["Generation"],
        dependencies=[Depends(authenticate_token)] if config.require_auth else []
    )
    
    app.include_router(
        editing_router,
        prefix="/api/v1/editing",
        tags=["Editing"],
        dependencies=[Depends(authenticate_token)] if config.require_auth else []
    )
    
    app.include_router(
        export_router,
        prefix="/api/v1/export",
        tags=["Export"],
        dependencies=[Depends(authenticate_token)] if config.require_auth else []
    )
    
    app.include_router(
        management_router,
        prefix="/api/v1/management",
        tags=["Management"],
        dependencies=[Depends(authenticate_token)] if config.require_auth else []
    )
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "service": "frextech-ai-simulations"
        }
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "Welcome to FrexTech AI Simulations API",
            "version": "1.0.0",
            "documentation": "/docs" if config.enable_docs else "disabled",
            "endpoints": [
                "/api/v1/generation",
                "/api/v1/editing",
                "/api/v1/export",
                "/api/v1/management"
            ]
        }
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title="FrexTech AI Simulations API",
            version="1.0.0",
            description="API for generating, editing, and managing AI-simulated worlds",
            routes=app.routes,
        )
        
        # Add security scheme
        if config.require_auth:
            openapi_schema["components"]["securitySchemes"] = {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                }
            }
            
            # Apply security globally
            for path in openapi_schema["paths"].values():
                for method in path.values():
                    method.setdefault("security", []).append({"ApiKeyAuth": []})
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 4,
    reload: bool = False
):
    """Run the API server using uvicorn"""
    app = create_app()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_config=None,
        access_log=True
    )


# Export the app instance for Gunicorn and other WSGI servers
app = create_app()
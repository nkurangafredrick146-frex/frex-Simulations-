"""
FrexTech AI Simulations API Module
Provides interfaces for world generation, editing, and management.
"""

__version__ = "1.0.0"
__author__ = "FrexTech AI Team"
__license__ = "MIT"

from .server import create_app, run_server
from .routes.generation import router as generation_router
from .routes.editing import router as editing_router
from .routes.export import router as export_router
from .routes.management import router as management_router
from .schemas.request_models import (
    GenerationRequest,
    EditRequest,
    ExportRequest,
    WorldQuery,
    BatchRequest
)
from .schemas.response_models import (
    GenerationResponse,
    EditResponse,
    ExportResponse,
    WorldStatus,
    APIError,
    SuccessResponse
)
from .utils.async_processor import AsyncProcessor
from .utils.cache_manager import CacheManager
from .utils.file_handler import FileHandler
from .middleware.authentication import authenticate_token, require_auth
from .middleware.rate_limiter import RateLimiter
from .middleware.error_handler import APIErrorHandler

__all__ = [
    # Core functions
    'create_app',
    'run_server',
    
    # Routers
    'generation_router',
    'editing_router',
    'export_router',
    'management_router',
    
    # Request schemas
    'GenerationRequest',
    'EditRequest',
    'ExportRequest',
    'WorldQuery',
    'BatchRequest',
    
    # Response schemas
    'GenerationResponse',
    'EditResponse',
    'ExportResponse',
    'WorldStatus',
    'APIError',
    'SuccessResponse',
    
    # Utilities
    'AsyncProcessor',
    'CacheManager',
    'FileHandler',
    
    # Middleware
    'authenticate_token',
    'require_auth',
    'RateLimiter',
    'APIErrorHandler'
]
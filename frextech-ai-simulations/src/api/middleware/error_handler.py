"""
Global error handling middleware
"""

import logging
import traceback
from typing import Dict, Any
from datetime import datetime

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from ...utils.logging_config import setup_logging
from ..schemas.response_models import APIError

logger = setup_logging("error_handler")


class APIErrorHandler(BaseHTTPMiddleware):
    """Middleware for handling and logging API errors"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        
        except HTTPException as http_exc:
            # HTTP exceptions are expected, log at info level
            logger.info(
                f"HTTP error: {http_exc.status_code} - {http_exc.detail}",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": http_exc.status_code
                }
            )
            
            return JSONResponse(
                status_code=http_exc.status_code,
                content=APIError(
                    error_code=f"HTTP_{http_exc.status_code}",
                    message=str(http_exc.detail),
                    details={},
                    timestamp=datetime.utcnow().isoformat()
                ).dict()
            )
        
        except RequestValidationError as validation_exc:
            # Validation errors from Pydantic
            logger.warning(
                f"Validation error: {validation_exc}",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "errors": validation_exc.errors()
                }
            )
            
            return JSONResponse(
                status_code=422,
                content=APIError(
                    error_code="VALIDATION_ERROR",
                    message="Request validation failed",
                    details={"errors": validation_exc.errors()},
                    timestamp=datetime.utcnow().isoformat()
                ).dict()
            )
        
        except Exception as exc:
            # Unexpected exceptions
            error_id = self._generate_error_id()
            
            logger.error(
                f"Unexpected error [{error_id}]: {exc}",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "error_id": error_id,
                    "traceback": traceback.format_exc()
                }
            )
            
            # In production, don't expose internal details
            if request.app.debug:
                details = {
                    "error_id": error_id,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc().splitlines()
                }
            else:
                details = {
                    "error_id": error_id,
                    "message": "An internal server error occurred"
                }
            
            return JSONResponse(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                content=APIError(
                    error_code="INTERNAL_SERVER_ERROR",
                    message="An unexpected error occurred",
                    details=details,
                    timestamp=datetime.utcnow().isoformat()
                ).dict()
            )
    
    def _generate_error_id(self) -> str:
        """Generate a unique error ID for tracking"""
        import uuid
        import hashlib
        import time
        
        unique_str = f"{uuid.uuid4()}{time.time()}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:16]


class ErrorContext:
    """Context manager for error handling in business logic"""
    
    def __init__(self, operation: str, request_id: str = None):
        self.operation = operation
        self.request_id = request_id
        self.start_time = datetime.utcnow()
    
    def __enter__(self):
        logger.info(
            f"Starting operation: {self.operation}",
            extra={
                "request_id": self.request_id,
                "operation": self.operation,
                "start_time": self.start_time.isoformat()
            }
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            logger.info(
                f"Completed operation: {self.operation}",
                extra={
                    "request_id": self.request_id,
                    "operation": self.operation,
                    "duration": duration,
                    "status": "success"
                }
            )
        else:
            logger.error(
                f"Failed operation: {self.operation} - {exc_val}",
                extra={
                    "request_id": self.request_id,
                    "operation": self.operation,
                    "duration": duration,
                    "status": "failed",
                    "error_type": exc_type.__name__,
                    "error_message": str(exc_val),
                    "traceback": traceback.format_exception(exc_type, exc_val, exc_tb)
                }
            )
        
        # Don't suppress exceptions
        return False


def handle_async_error(task, context: Dict[str, Any]):
    """Handle errors from async tasks"""
    try:
        if task.exception():
            exc = task.exception()
            logger.error(
                f"Async task failed: {context.get('task_name', 'unknown')}",
                extra={
                    **context,
                    "error": str(exc),
                    "traceback": traceback.format_exception(
                        type(exc), exc, exc.__traceback__
                    )
                }
            )
    except (AttributeError, RuntimeError):
        pass


def create_error_response(
    error_code: str,
    message: str,
    details: Dict = None,
    status_code: int = 400
) -> JSONResponse:
    """Create a standardized error response"""
    return JSONResponse(
        status_code=status_code,
        content=APIError(
            error_code=error_code,
            message=message,
            details=details or {},
            timestamp=datetime.utcnow().isoformat()
        ).dict()
    )


# Common error codes
ERROR_CODES = {
    # Authentication errors
    "AUTH_REQUIRED": "Authentication required",
    "INVALID_API_KEY": "Invalid API key",
    "EXPIRED_API_KEY": "API key expired",
    "INSUFFICIENT_PERMISSIONS": "Insufficient permissions",
    
    # Rate limiting
    "RATE_LIMIT_EXCEEDED": "Rate limit exceeded",
    
    # Validation errors
    "INVALID_INPUT": "Invalid input parameters",
    "MISSING_REQUIRED_FIELD": "Missing required field",
    "INVALID_FORMAT": "Invalid data format",
    
    # Resource errors
    "RESOURCE_NOT_FOUND": "Resource not found",
    "RESOURCE_ALREADY_EXISTS": "Resource already exists",
    "RESOURCE_UNAVAILABLE": "Resource unavailable",
    
    # Processing errors
    "PROCESSING_FAILED": "Processing failed",
    "TIMEOUT": "Operation timeout",
    "QUEUE_FULL": "Processing queue full",
    
    # System errors
    "INTERNAL_ERROR": "Internal server error",
    "SERVICE_UNAVAILABLE": "Service unavailable",
    "DATABASE_ERROR": "Database error",
    "NETWORK_ERROR": "Network error",
    
    # World model errors
    "MODEL_NOT_LOADED": "AI model not loaded",
    "INFERENCE_FAILED": "AI inference failed",
    "OUT_OF_MEMORY": "Out of memory",
    "GPU_ERROR": "GPU processing error",
}
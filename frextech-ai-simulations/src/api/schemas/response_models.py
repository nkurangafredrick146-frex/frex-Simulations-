"""
Response schemas for API endpoints
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


# Enums for response status
class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    PROCESSING = "processing"
    QUEUED = "queued"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class ModelStatusEnum(str, Enum):
    LOADED = "loaded"
    UNLOADED = "unloaded"
    LOADING = "loading"
    ERROR = "error"
    DISABLED = "disabled"


# Base Response Models
class BaseResponse(BaseModel):
    """Base response model"""
    
    status: ResponseStatus = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SuccessResponse(BaseResponse):
    """Success response model"""
    
    success: bool = Field(True, description="Success flag")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")


class APIError(BaseModel):
    """API error model"""
    
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Generation Response Models
class GenerationResponse(BaseResponse):
    """Response for world generation"""
    
    session_id: str = Field(..., description="Generation session ID")
    task_id: Optional[str] = Field(None, description="Async task ID")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Generation progress")
    world_id: Optional[str] = Field(None, description="Generated world ID")
    world_data: Optional[Dict[str, Any]] = Field(None, description="Generated world data")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class BatchGenerationResponse(BaseResponse):
    """Response for batch generation"""
    
    batch_id: str = Field(..., description="Batch ID")
    session_ids: List[str] = Field(..., description="List of session IDs")
    total_tasks: int = Field(..., description="Total number of tasks")
    completed_tasks: int = Field(0, description="Number of completed tasks")
    failed_tasks: int = Field(0, description="Number of failed tasks")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


# Editing Response Models
class EditResponse(BaseResponse):
    """Response for world editing"""
    
    session_id: str = Field(..., description="Editing session ID")
    task_id: Optional[str] = Field(None, description="Async task ID")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Editing progress")
    new_world_id: Optional[str] = Field(None, description="New world ID after editing")
    world_data: Optional[Dict[str, Any]] = Field(None, description="Edited world data")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    parent_world_id: Optional[str] = Field(None, description="Original world ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ExpansionResponse(BaseResponse):
    """Response for world expansion"""
    
    session_id: str = Field(..., description="Expansion session ID")
    task_id: Optional[str] = Field(None, description="Async task ID")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Expansion progress")
    new_world_id: Optional[str] = Field(None, description="New world ID after expansion")
    world_data: Optional[Dict[str, Any]] = Field(None, description="Expanded world data")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    direction: Optional[str] = Field(None, description="Expansion direction")
    distance: Optional[float] = Field(None, description="Expansion distance")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class CompositionResponse(BaseResponse):
    """Response for world composition"""
    
    session_id: str = Field(..., description="Composition session ID")
    task_id: Optional[str] = Field(None, description="Async task ID")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Composition progress")
    new_world_id: Optional[str] = Field(None, description="New world ID after composition")
    world_data: Optional[Dict[str, Any]] = Field(None, description="Composed world data")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    source_world_ids: Optional[List[str]] = Field(None, description="Source world IDs")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


# Export Response Models
class ExportResponse(BaseResponse):
    """Base response for export operations"""
    
    session_id: str = Field(..., description="Export session ID")
    task_id: Optional[str] = Field(None, description="Async task ID")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Export progress")
    output_url: Optional[str] = Field(None, description="Output file URL")
    file_size: Optional[int] = Field(None, description="Output file size in bytes")
    format: Optional[str] = Field(None, description="Output format")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class RenderResponse(ExportResponse):
    """Response for render operations"""
    
    render_type: Optional[str] = Field(None, description="Type of render")
    resolution: Optional[str] = Field(None, description="Render resolution")
    duration: Optional[float] = Field(None, description="Render duration in seconds")
    frame_count: Optional[int] = Field(None, description="Number of frames")


class ConversionResponse(BaseResponse):
    """Response for conversion operations"""
    
    session_id: str = Field(..., description="Conversion session ID")
    task_id: Optional[str] = Field(None, description="Async task ID")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Conversion progress")
    new_world_id: Optional[str] = Field(None, description="New world ID after conversion")
    world_data: Optional[Dict[str, Any]] = Field(None, description="Converted world data")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    source_format: Optional[str] = Field(None, description="Source format")
    target_format: Optional[str] = Field(None, description="Target format")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class BatchExportResponse(BaseResponse):
    """Response for batch export"""
    
    batch_id: str = Field(..., description="Batch ID")
    session_ids: List[str] = Field(..., description="List of session IDs")
    total_tasks: int = Field(..., description="Total number of tasks")
    completed_tasks: int = Field(0, description="Number of completed tasks")
    failed_tasks: int = Field(0, description="Number of failed tasks")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


# Management Response Models
class WorldStatus(BaseModel):
    """World status model"""
    
    session_id: str = Field(..., description="Session ID")
    status: ResponseStatus = Field(..., description="Current status")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress percentage")
    world_id: Optional[str] = Field(None, description="World ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    error: Optional[str] = Field(None, description="Error message")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemStatus(BaseModel):
    """System status model"""
    
    status: HealthStatus = Field(..., description="System health status")
    timestamp: datetime = Field(..., description="Status timestamp")
    uptime: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="System version")
    components: Dict[str, Any] = Field(..., description="Component statuses")
    issues: Optional[List[str]] = Field(None, description="List of issues")
    recommendations: Optional[List[str]] = Field(None, description="Recommendations")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelStatus(BaseModel):
    """AI model status model"""
    
    name: str = Field(..., description="Model name")
    status: ModelStatusEnum = Field(..., description="Model status")
    loaded: bool = Field(..., description="Whether model is loaded")
    size_mb: Optional[float] = Field(None, description="Model size in MB")
    version: Optional[str] = Field(None, description="Model version")
    device: Optional[str] = Field(None, description="Device (cpu, cuda:0, etc.)")
    loaded_at: Optional[datetime] = Field(None, description="When model was loaded")
    error: Optional[str] = Field(None, description="Error message if any")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CacheStatus(BaseModel):
    """Cache status model"""
    
    status: str = Field(..., description="Cache status")
    connected: bool = Field(..., description="Whether cache is connected")
    keys: int = Field(0, description="Number of keys in cache")
    memory_used: int = Field(0, description="Memory used in bytes")
    hit_rate: Optional[float] = Field(None, description="Cache hit rate")
    miss_rate: Optional[float] = Field(None, description="Cache miss rate")
    evictions: Optional[int] = Field(None, description="Number of evictions")
    message: Optional[str] = Field(None, description="Status message")


class StatisticsResponse(BaseModel):
    """System statistics response"""
    
    period_hours: int = Field(..., description="Statistics period in hours")
    system: Dict[str, Any] = Field(..., description="System statistics")
    api: Dict[str, Any] = Field(..., description="API statistics")
    cache: Dict[str, Any] = Field(..., description="Cache statistics")
    models: Dict[str, Any] = Field(..., description="Model statistics")
    timestamp: datetime = Field(..., description="Statistics timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Progress Update Model
class ProgressUpdate(BaseModel):
    """Progress update model for streaming"""
    
    session_id: str = Field(..., description="Session ID")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Current progress")
    status: ResponseStatus = Field(..., description="Current status")
    message: str = Field(..., description="Progress message")
    timestamp: datetime = Field(..., description="Update timestamp")
    world_id: Optional[str] = Field(None, description="World ID if completed")
    error: Optional[str] = Field(None, description="Error if failed")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Region Detection Response
class RegionDetectionResponse(BaseModel):
    """Region detection response"""
    
    world_id: str = Field(..., description="World ID")
    method: str = Field(..., description="Detection method used")
    threshold: float = Field(..., description="Detection threshold used")
    regions: List[Dict[str, Any]] = Field(..., description="Detected regions")
    count: int = Field(..., description="Number of regions detected")
    timestamp: datetime = Field(..., description="Detection timestamp")


# Version History Response
class VersionHistoryResponse(BaseModel):
    """Version history response"""
    
    world_id: str = Field(..., description="World ID")
    total_versions: int = Field(..., description="Total number of versions")
    versions: List[Dict[str, Any]] = Field(..., description="List of versions")
    current_version_id: Optional[str] = Field(None, description="Current version ID")
    created_at: Optional[datetime] = Field(None, description="Original creation timestamp")
    last_modified: Optional[datetime] = Field(None, description="Last modification timestamp")


# World Information Models
class WorldInfo(BaseModel):
    """Basic world information"""
    
    world_id: str = Field(..., description="World ID")
    name: Optional[str] = Field(None, description="World name")
    format: str = Field(..., description="World format")
    size_bytes: int = Field(..., description="World size in bytes")
    created_at: datetime = Field(..., description="Creation timestamp")
    modified_at: datetime = Field(..., description="Last modification timestamp")
    version_count: int = Field(0, description="Number of versions")
    tags: List[str] = Field(default_factory=list, description="World tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="World metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WorldListResponse(BaseModel):
    """World list response"""
    
    worlds: List[WorldInfo] = Field(..., description="List of worlds")
    total: int = Field(..., description="Total number of worlds")
    page: int = Field(1, description="Current page")
    page_size: int = Field(50, description="Page size")
    total_pages: int = Field(..., description="Total number of pages")


# Capabilities Response
class GenerationCapabilities(BaseModel):
    """Generation capabilities"""
    
    text_encoder: str = Field(..., description="Text encoder model")
    vision_encoder: str = Field(..., description="Vision encoder model")
    max_prompt_length: int = Field(..., description="Maximum prompt length")
    supported_formats: List[str] = Field(..., description="Supported world formats")
    max_batch_size: int = Field(..., description="Maximum batch size")
    max_concurrent: int = Field(..., description="Maximum concurrent generations")
    default_parameters: Dict[str, Any] = Field(..., description="Default parameters")
    available_styles: List[str] = Field(..., description="Available style presets")


class EditingCapabilities(BaseModel):
    """Editing capabilities"""
    
    supported_operations: List[Dict[str, Any]] = Field(..., description="Supported operations")
    expansion: Dict[str, Any] = Field(..., description="Expansion capabilities")
    composition: Dict[str, Any] = Field(..., description="Composition capabilities")
    region_detection: Dict[str, Any] = Field(..., description="Region detection capabilities")
    limits: Dict[str, Any] = Field(..., description="Editing limits")


class ExportCapabilities(BaseModel):
    """Export capabilities"""
    
    render: Dict[str, Any] = Field(..., description="Render capabilities")
    conversion: Dict[str, Any] = Field(..., description="Conversion capabilities")
    limits: Dict[str, Any] = Field(..., description="Export limits")


class SystemCapabilities(BaseModel):
    """System capabilities"""
    
    generation: GenerationCapabilities = Field(..., description="Generation capabilities")
    editing: EditingCapabilities = Field(..., description="Editing capabilities")
    export: ExportCapabilities = Field(..., description="Export capabilities")
    system: Dict[str, Any] = Field(..., description="System capabilities")
    limits: Dict[str, Any] = Field(..., description="System limits")
    version: str = Field(..., description="System version")


# Rate Limit Information
class RateLimitInfo(BaseModel):
    """Rate limit information"""
    
    limit: int = Field(..., description="Request limit per window")
    remaining: int = Field(..., description="Remaining requests")
    reset: int = Field(..., description="Unix timestamp when limit resets")
    window: int = Field(..., description="Window size in seconds")
    current: int = Field(..., description="Current request count")
    is_limited: bool = Field(False, description="Whether rate limited")


# API Key Information
class APIKeyInfo(BaseModel):
    """API key information"""
    
    key_id: str = Field(..., description="Key ID")
    name: str = Field(..., description="Key name")
    permissions: List[str] = Field(..., description="Key permissions")
    rate_limit: int = Field(..., description="Rate limit per hour")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    last_used: Optional[datetime] = Field(None, description="Last used timestamp")
    usage_count: int = Field(0, description="Usage count")
    has_secret: bool = Field(False, description="Whether secret is available")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class APIKeyListResponse(BaseModel):
    """API key list response"""
    
    api_keys: List[APIKeyInfo] = Field(..., description="List of API keys")
    total: int = Field(..., description="Total number of keys")
    can_create: bool = Field(True, description="Whether new keys can be created")
    max_keys: int = Field(100, description="Maximum number of keys")


class APIKeyCreationResponse(BaseModel):
    """API key creation response"""
    
    key_id: str = Field(..., description="Key ID")
    secret: str = Field(..., description="API key secret (only shown once)")
    api_key_info: APIKeyInfo = Field(..., description="API key information")
    warning: str = Field("Store this secret securely. It will not be shown again.", 
                       description="Security warning")


# Validation Models
class ValidationError(BaseModel):
    """Validation error model"""
    
    field: str = Field(..., description="Field with error")
    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    value: Optional[Any] = Field(None, description="Invalid value")


class ValidationResult(BaseModel):
    """Validation result model"""
    
    valid: bool = Field(..., description="Whether validation passed")
    errors: List[ValidationError] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")


# File Information
class FileInfo(BaseModel):
    """File information model"""
    
    filename: str = Field(..., description="File name")
    size_bytes: int = Field(..., description="File size in bytes")
    content_type: Optional[str] = Field(None, description="Content type")
    checksum: Optional[str] = Field(None, description="File checksum")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    download_url: Optional[str] = Field(None, description="Download URL")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Response factory functions
def create_success_response(
    message: str = "Operation completed successfully",
    data: Optional[Dict[str, Any]] = None,
    status_code: int = 200
) -> Dict[str, Any]:
    """Create a standardized success response"""
    return {
        "status": ResponseStatus.SUCCESS,
        "message": message,
        "data": data or {},
        "timestamp": datetime.utcnow().isoformat()
    }


def create_error_response(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    status_code: int = 400
) -> Dict[str, Any]:
    """Create a standardized error response"""
    return {
        "error_code": error_code,
        "message": message,
        "details": details or {},
        "timestamp": datetime.utcnow().isoformat()
    }


def create_progress_response(
    session_id: str,
    progress: float,
    status: ResponseStatus,
    message: str,
    world_id: Optional[str] = None
) -> ProgressUpdate:
    """Create a progress update response"""
    return ProgressUpdate(
        session_id=session_id,
        progress=progress,
        status=status,
        message=message,
        timestamp=datetime.utcnow(),
        world_id=world_id
    )


# Response validation helpers
def validate_response_data(data: Dict[str, Any], schema: BaseModel) -> ValidationResult:
    """Validate response data against a schema"""
    try:
        schema.parse_obj(data)
        return ValidationResult(valid=True, errors=[], warnings=[])
    except Exception as e:
        return ValidationResult(
            valid=False,
            errors=[ValidationError(
                field="response",
                message=str(e),
                type="validation_error",
                value=data
            )]
        )


def sanitize_response_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize response data by removing sensitive information"""
    sanitized = data.copy()
    
    # Remove sensitive fields from responses
    sensitive_keys = ['api_key', 'secret', 'token', 'password', 'private_key']
    
    def recursive_sanitize(obj):
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                if key.lower() in [k.lower() for k in sensitive_keys]:
                    obj[key] = "***REDACTED***"
                elif isinstance(obj[key], (dict, list)):
                    recursive_sanitize(obj[key])
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    recursive_sanitize(item)
    
    recursive_sanitize(sanitized)
    return sanitized


# Export all models
__all__ = [
    # Enums
    'ResponseStatus',
    'HealthStatus',
    'ModelStatusEnum',
    
    # Base Response Models
    'BaseResponse',
    'SuccessResponse',
    'APIError',
    
    # Generation Response Models
    'GenerationResponse',
    'BatchGenerationResponse',
    
    # Editing Response Models
    'EditResponse',
    'ExpansionResponse',
    'CompositionResponse',
    
    # Export Response Models
    'ExportResponse',
    'RenderResponse',
    'ConversionResponse',
    'BatchExportResponse',
    
    # Management Response Models
    'WorldStatus',
    'SystemStatus',
    'ModelStatus',
    'CacheStatus',
    'StatisticsResponse',
    
    # Progress Update Model
    'ProgressUpdate',
    
    # Region Detection Response
    'RegionDetectionResponse',
    
    # Version History Response
    'VersionHistoryResponse',
    
    # World Information Models
    'WorldInfo',
    'WorldListResponse',
    
    # Capabilities Response
    'GenerationCapabilities',
    'EditingCapabilities',
    'ExportCapabilities',
    'SystemCapabilities',
    
    # Rate Limit Information
    'RateLimitInfo',
    
    # API Key Information
    'APIKeyInfo',
    'APIKeyListResponse',
    'APIKeyCreationResponse',
    
    # Validation Models
    'ValidationError',
    'ValidationResult',
    
    # File Information
    'FileInfo',
    
    # Response factory functions
    'create_success_response',
    'create_error_response',
    'create_progress_response',
    
    # Response validation helpers
    'validate_response_data',
    'sanitize_response_data'
]
"""
Request schemas for API endpoints
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, HttpUrl, confloat, conint


# Enums
class RenderType(str, Enum):
    IMAGE = "image"
    PANORAMA = "panorama"
    VIDEO = "video"
    INTERACTIVE = "interactive"
    POINT_CLOUD = "point_cloud"


class ExportType(str, Enum):
    RENDER = "render"
    CONVERT = "convert"


class QualityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class CameraType(str, Enum):
    PERSPECTIVE = "perspective"
    ORTHOGRAPHIC = "orthographic"
    PANORAMIC = "panoramic"


class EditOperationType(str, Enum):
    TEXT_PROMPT = "text_prompt"
    STYLE_TRANSFER = "style_transfer"
    OBJECT_REMOVAL = "object_removal"
    OBJECT_ADDITION = "object_addition"
    COLOR_ADJUSTMENT = "color_adjustment"
    LIGHTING_ADJUSTMENT = "lighting_adjustment"


class ExpansionDirection(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    UP = "up"
    DOWN = "down"


class WorldFormat(str, Enum):
    NERF = "nerf"
    GAUSSIAN = "gaussian"
    MESH = "mesh"
    VOXEL = "voxel"
    POINT_CLOUD = "point_cloud"
    GLTF = "gltf"
    USD = "usd"
    FBX = "fbx"


# Base Models
class BaseRequest(BaseModel):
    """Base request model with common fields"""
    
    async_mode: bool = Field(
        False,
        description="Whether to process request asynchronously"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class GenerationParameters(BaseModel):
    """Parameters for world generation"""
    
    quality: QualityLevel = Field(QualityLevel.HIGH, description="Generation quality")
    resolution: str = Field("1024x1024", description="Output resolution")
    steps: int = Field(50, ge=1, le=500, description="Number of diffusion steps")
    guidance_scale: float = Field(7.5, ge=1.0, le=20.0, description="Classifier-free guidance scale")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    batch_size: int = Field(1, ge=1, le=16, description="Batch size for generation")
    
    @validator('resolution')
    def validate_resolution(cls, v):
        """Validate resolution format"""
        if 'x' not in v:
            raise ValueError('Resolution must be in format "WIDTHxHEIGHT"')
        try:
            width, height = map(int, v.split('x'))
            if width < 64 or height < 64 or width > 8192 or height > 8192:
                raise ValueError('Resolution must be between 64x64 and 8192x8192')
        except ValueError:
            raise ValueError('Resolution must be in format "WIDTHxHEIGHT" with integers')
        return v


class CameraParameters(BaseModel):
    """Camera parameters for rendering"""
    
    type: CameraType = Field(CameraType.PERSPECTIVE, description="Camera type")
    position: List[float] = Field([0, 0, 5], min_items=3, max_items=3, description="Camera position [x, y, z]")
    target: List[float] = Field([0, 0, 0], min_items=3, max_items=3, description="Look-at target [x, y, z]")
    up: List[float] = Field([0, 1, 0], min_items=3, max_items=3, description="Up vector [x, y, z]")
    fov: float = Field(60.0, ge=1.0, le=180.0, description="Field of view in degrees")
    near: float = Field(0.1, ge=0.01, description="Near clipping plane")
    far: float = Field(1000.0, ge=1.0, description="Far clipping plane")
    
    class CameraPath(BaseModel):
        """Camera path for animation"""
        points: List[List[float]] = Field(..., description="Path points [[x, y, z], ...]")
        duration: float = Field(10.0, ge=0.1, le=300.0, description="Path duration in seconds")
        easing: str = Field("linear", description="Easing function")
    
    path: Optional[CameraPath] = Field(None, description="Camera path for animation")


class RegionSelection(BaseModel):
    """Region selection for editing operations"""
    
    type: str = Field("rectangle", description="Region type (rectangle, polygon, mask)")
    coordinates: List[List[float]] = Field(..., description="Region coordinates")
    mask: Optional[str] = Field(None, description="Base64 encoded mask image")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Selection confidence")
    
    @validator('coordinates')
    def validate_coordinates(cls, v, values):
        """Validate region coordinates"""
        if values.get('type') == 'rectangle' and len(v) != 2:
            raise ValueError('Rectangle requires exactly 2 points [min, max]')
        elif values.get('type') == 'polygon' and len(v) < 3:
            raise ValueError('Polygon requires at least 3 points')
        return v


class EditOperation(BaseModel):
    """Single edit operation"""
    
    type: EditOperationType = Field(..., description="Type of edit operation")
    prompt: Optional[str] = Field(None, description="Text prompt for the edit")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation-specific parameters")
    strength: float = Field(1.0, ge=0.0, le=2.0, description="Edit strength")
    
    @validator('prompt')
    def validate_prompt_for_operations(cls, v, values):
        """Validate prompt is provided for operations that need it"""
        if values.get('type') in [EditOperationType.TEXT_PROMPT, EditOperationType.OBJECT_ADDITION] and not v:
            raise ValueError(f"Prompt is required for {values.get('type')} operation")
        return v


class EditParameters(BaseModel):
    """Parameters for editing operations"""
    
    check_consistency: bool = Field(True, description="Check edit consistency")
    blend_strength: float = Field(0.5, ge=0.0, le=1.0, description="Blending strength for edits")
    inpaint_strength: float = Field(0.75, ge=0.0, le=1.0, description="Inpainting strength")
    preserve_structure: bool = Field(True, description="Preserve structural elements")
    
    class Config:
        extra = "allow"  # Allow additional parameters


class ExpansionParameters(BaseModel):
    """Parameters for world expansion"""
    
    seamless: bool = Field(True, description="Create seamless transition")
    blend_width: float = Field(0.1, ge=0.01, le=1.0, description="Blending width as fraction")
    preserve_style: bool = Field(True, description="Preserve original style")
    
    class Config:
        extra = "allow"


class CompositionLayout(BaseModel):
    """Layout specification for world composition"""
    
    type: str = Field("grid", description="Layout type (grid, linear, radial, custom)")
    rows: Optional[int] = Field(None, description="Number of rows for grid layout")
    cols: Optional[int] = Field(None, description="Number of columns for grid layout")
    spacing: float = Field(1.0, ge=0.0, description="Spacing between worlds")
    arrangement: Optional[List[List[float]]] = Field(None, description="Custom arrangement [[x, y, z], ...]")
    
    @validator('rows', 'cols')
    def validate_grid_layout(cls, v, values):
        """Validate grid layout parameters"""
        if values.get('type') == 'grid' and v is None:
            raise ValueError('Rows and cols are required for grid layout')
        return v


class TransitionSpecification(BaseModel):
    """Transition specification between worlds"""
    
    type: str = Field("fade", description="Transition type (fade, morph, portal, seamless)")
    duration: float = Field(1.0, ge=0.1, le=10.0, description="Transition duration in seconds")
    easing: str = Field("linear", description="Easing function")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Transition-specific parameters")


class CompositionParameters(BaseModel):
    """Parameters for world composition"""
    
    merge_geometry: bool = Field(True, description="Merge geometry between worlds")
    blend_materials: bool = Field(True, description="Blend materials at boundaries")
    optimize: bool = Field(True, description="Optimize composed world")
    
    class Config:
        extra = "allow"


class RenderParameters(BaseModel):
    """Parameters for rendering"""
    
    fps: int = Field(30, ge=1, le=120, description="Frames per second for video")
    duration: float = Field(10.0, ge=0.1, le=300.0, description="Duration in seconds for video")
    stereo: bool = Field(False, description="Render stereo panorama")
    include_controls: bool = Field(True, description="Include controls in interactive export")
    
    class Config:
        extra = "allow"


class ConversionParameters(BaseModel):
    """Parameters for format conversion"""
    
    quality: QualityLevel = Field(QualityLevel.MEDIUM, description="Conversion quality")
    optimize: bool = Field(True, description="Optimize output")
    preserve_textures: bool = Field(True, description="Preserve texture information")
    
    class Config:
        extra = "allow"


# Main Request Models
class GenerationRequest(BaseRequest):
    """Request for world generation"""
    
    prompt: str = Field(..., min_length=1, max_length=10000, description="Text description of the world")
    style_prompt: Optional[str] = Field(None, max_length=5000, description="Optional style guidance")
    reference_image: Optional[Union[HttpUrl, str]] = Field(
        None,
        description="Optional reference image URL or base64 string"
    )
    parameters: Optional[GenerationParameters] = Field(
        default_factory=GenerationParameters,
        description="Generation parameters"
    )
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Validate prompt is not empty"""
        if not v.strip():
            raise ValueError('Prompt cannot be empty or whitespace only')
        return v.strip()


class BatchRequest(BaseRequest):
    """Request for batch generation"""
    
    prompts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of text descriptions"
    )
    common_parameters: Optional[GenerationParameters] = Field(
        default_factory=GenerationParameters,
        description="Common parameters for all generations"
    )
    max_concurrent: int = Field(4, ge=1, le=16, description="Maximum concurrent generations")


class EditRequest(BaseRequest):
    """Request for editing a world"""
    
    world_id: str = Field(..., description="ID of the world to edit")
    operations: List[EditOperation] = Field(
        ...,
        min_items=1,
        max_items=20,
        description="List of edit operations to apply"
    )
    region: Optional[RegionSelection] = Field(None, description="Optional region to apply edits to")
    parameters: Optional[EditParameters] = Field(
        default_factory=EditParameters,
        description="Edit parameters"
    )


class ExpansionRequest(BaseRequest):
    """Request for expanding a world"""
    
    world_id: str = Field(..., description="ID of the world to expand")
    direction: ExpansionDirection = Field(..., description="Direction to expand")
    distance: float = Field(1.0, ge=0.1, le=100.0, description="Distance to expand")
    prompt: Optional[str] = Field(None, max_length=5000, description="Optional prompt for the new area")
    seamless: bool = Field(True, description="Whether to create seamless transition")
    parameters: Optional[ExpansionParameters] = Field(
        default_factory=ExpansionParameters,
        description="Expansion parameters"
    )


class CompositionRequest(BaseRequest):
    """Request for composing multiple worlds"""
    
    world_ids: List[str] = Field(
        ...,
        min_items=2,
        max_items=10,
        description="List of world IDs to compose"
    )
    layout: Optional[CompositionLayout] = Field(None, description="Layout specification")
    transitions: Optional[List[TransitionSpecification]] = Field(
        None,
        description="Transition specifications between worlds"
    )
    parameters: Optional[CompositionParameters] = Field(
        default_factory=CompositionParameters,
        description="Composition parameters"
    )


class RenderRequest(BaseRequest):
    """Request for rendering a world"""
    
    world_id: str = Field(..., description="ID of the world to render")
    render_type: RenderType = Field(..., description="Type of render")
    camera: Optional[CameraParameters] = Field(None, description="Camera parameters")
    resolution: str = Field("1024x1024", description="Output resolution")
    quality: QualityLevel = Field(QualityLevel.HIGH, description="Render quality")
    output_format: str = Field(..., description="Output file format")
    parameters: Optional[RenderParameters] = Field(
        default_factory=RenderParameters,
        description="Render parameters"
    )
    
    @validator('resolution')
    def validate_resolution(cls, v):
        """Validate resolution format"""
        if 'x' not in v:
            raise ValueError('Resolution must be in format "WIDTHxHEIGHT"')
        try:
            width, height = map(int, v.split('x'))
            if width < 64 or height < 64 or width > 16384 or height > 16384:
                raise ValueError('Resolution must be between 64x64 and 16384x16384')
        except ValueError:
            raise ValueError('Resolution must be in format "WIDTHxHEIGHT" with integers')
        return v
    
    @validator('output_format')
    def validate_output_format(cls, v, values):
        """Validate output format based on render type"""
        render_type = values.get('render_type')
        
        valid_formats = {
            RenderType.IMAGE: ["png", "jpg", "jpeg", "webp", "exr", "hdr"],
            RenderType.PANORAMA: ["png", "jpg", "jpeg", "webp", "exr", "hdr"],
            RenderType.VIDEO: ["mp4", "webm", "gif", "mov"],
            RenderType.INTERACTIVE: ["html", "gltf", "glb"],
            RenderType.POINT_CLOUD: ["ply", "xyz", "pcd"]
        }
        
        if render_type and v.lower() not in valid_formats.get(render_type, []):
            raise ValueError(
                f"Invalid format for {render_type}. "
                f"Must be one of: {', '.join(valid_formats[render_type])}"
            )
        
        return v.lower()


class ConversionRequest(BaseRequest):
    """Request for converting world format"""
    
    world_id: str = Field(..., description="ID of the world to convert")
    target_format: WorldFormat = Field(..., description="Target format")
    parameters: Optional[ConversionParameters] = Field(
        default_factory=ConversionParameters,
        description="Conversion parameters"
    )


class ExportRequest(BaseModel):
    """Individual export specification for batch export"""
    
    world_id: str = Field(..., description="ID of the world to export")
    export_type: ExportType = Field(..., description="Type of export")
    format: str = Field(..., description="Output format")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Export parameters")


class BatchExportRequest(BaseRequest):
    """Request for batch export"""
    
    exports: List[ExportRequest] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of export specifications"
    )
    common_parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Common parameters for all exports"
    )
    max_concurrent: int = Field(4, ge=1, le=10, description="Maximum concurrent exports")


# Management Request Models
class WorldQuery(BaseModel):
    """Query parameters for world search"""
    
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    status: Optional[str] = Field(None, description="Filter by status")
    format: Optional[str] = Field(None, description="Filter by world format")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date")
    limit: int = Field(100, ge=1, le=1000, description="Maximum results")
    offset: int = Field(0, ge=0, description="Result offset")


class ModelManagementRequest(BaseModel):
    """Request for model management"""
    
    models: Optional[List[str]] = Field(
        None,
        description="List of models to manage (empty for all)"
    )
    force: bool = Field(False, description="Force operation")


class CacheManagementRequest(BaseModel):
    """Request for cache management"""
    
    pattern: Optional[str] = Field(None, description="Pattern for cache keys to clear")
    clear_all: bool = Field(False, description="Clear entire cache")
    

# Validation utilities
def validate_world_id(world_id: str) -> bool:
    """Validate world ID format"""
    # Simple UUID validation
    import re
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(world_id))


def validate_session_id(session_id: str) -> bool:
    """Validate session ID format"""
    # Allow UUID or special values like 'cached', 'latest'
    if session_id in ['cached', 'latest']:
        return True
    return validate_world_id(session_id)


def validate_api_key_format(api_key: str) -> bool:
    """Validate API key format"""
    # API keys should be at least 32 characters
    return len(api_key) >= 32


# Helper functions for request processing
def sanitize_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize request data by removing sensitive information"""
    sanitized = data.copy()
    
    # Remove sensitive fields
    sensitive_keys = ['api_key', 'token', 'password', 'secret']
    for key in sensitive_keys:
        if key in sanitized:
            sanitized[key] = '***REDACTED***'
    
    # Truncate long strings
    for key, value in sanitized.items():
        if isinstance(value, str) and len(value) > 1000:
            sanitized[key] = value[:1000] + f"... [truncated, length: {len(value)}]"
    
    return sanitized


def validate_request_limits(request_data: Dict[str, Any]) -> List[str]:
    """Validate request against system limits"""
    errors = []
    
    # Check prompt length
    if 'prompt' in request_data and len(request_data['prompt']) > 10000:
        errors.append('Prompt exceeds maximum length of 10000 characters')
    
    # Check batch size
    if 'prompts' in request_data and len(request_data['prompts']) > 100:
        errors.append('Batch size exceeds maximum of 100 prompts')
    
    # Check resolution
    if 'resolution' in request_data:
        try:
            width, height = map(int, request_data['resolution'].split('x'))
            if width > 16384 or height > 16384:
                errors.append('Resolution exceeds maximum of 16384x16384')
        except:
            pass
    
    return errors


# Export all models
__all__ = [
    # Enums
    'RenderType',
    'ExportType',
    'QualityLevel',
    'CameraType',
    'EditOperationType',
    'ExpansionDirection',
    'WorldFormat',
    
    # Base Models
    'BaseRequest',
    'GenerationParameters',
    'CameraParameters',
    'RegionSelection',
    'EditOperation',
    'EditParameters',
    'ExpansionParameters',
    'CompositionLayout',
    'TransitionSpecification',
    'CompositionParameters',
    'RenderParameters',
    'ConversionParameters',
    
    # Main Request Models
    'GenerationRequest',
    'BatchRequest',
    'EditRequest',
    'ExpansionRequest',
    'CompositionRequest',
    'RenderRequest',
    'ConversionRequest',
    'ExportRequest',
    'BatchExportRequest',
    
    # Management Request Models
    'WorldQuery',
    'ModelManagementRequest',
    'CacheManagementRequest',
    
    # Validation functions
    'validate_world_id',
    'validate_session_id',
    'validate_api_key_format',
    'sanitize_request_data',
    'validate_request_limits'
]
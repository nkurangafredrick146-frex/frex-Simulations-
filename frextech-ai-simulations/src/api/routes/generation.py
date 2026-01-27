"""
Generation routes for world creation and synthesis
"""

import uuid
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..middleware.authentication import require_auth
from ..schemas.request_models import (
    GenerationRequest,
    BatchRequest,
    WorldQuery,
    GenerationParameters
)
from ..schemas.response_models import (
    GenerationResponse,
    BatchGenerationResponse,
    WorldStatus,
    ProgressUpdate,
    APIError,
    SuccessResponse
)
from ..utils.async_processor import AsyncProcessor, ProcessingTask
from ..utils.cache_manager import CacheManager
from ..utils.file_handler import FileHandler
from src.core.world_model.inference.generator import WorldGenerator
from src.core.multimodal.encoders.text_encoder import TextEncoder
from src.core.multimodal.encoders.vision_encoder import VisionEncoder
from src.core.multimodal.fusion.cross_attention import CrossAttentionFusion
from src.utils.logging_config import setup_logging
from configs.model.inference import InferenceConfig

logger = setup_logging("generation_routes")

router = APIRouter()

# Global components
world_generator: Optional[WorldGenerator] = None
text_encoder: Optional[TextEncoder] = None
vision_encoder: Optional[VisionEncoder] = None
fusion_model: Optional[CrossAttentionFusion] = None
cache_manager: Optional[CacheManager] = None
async_processor: Optional[AsyncProcessor] = None


class GenerationSession:
    """Manages generation sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.generated_worlds: Dict[str, Dict] = {}
    
    def create_session(self, user_id: str, request: GenerationRequest) -> str:
        """Create a new generation session"""
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "request": request.dict(),
            "status": "pending",
            "progress": 0.0,
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "result": None,
            "world_id": None
        }
        
        return session_id
    
    def update_session(
        self,
        session_id: str,
        status: str = None,
        progress: float = None,
        error: str = None,
        result: Dict = None,
        world_id: str = None
    ):
        """Update session status"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if status:
            session["status"] = status
            if status == "processing" and not session["started_at"]:
                session["started_at"] = datetime.utcnow()
            elif status in ["completed", "failed"]:
                session["completed_at"] = datetime.utcnow()
        
        if progress is not None:
            session["progress"] = max(0.0, min(1.0, progress))
        
        if error is not None:
            session["error"] = error
        
        if result is not None:
            session["result"] = result
        
        if world_id is not None:
            session["world_id"] = world_id
        
        return True
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def list_user_sessions(self, user_id: str) -> List[Dict]:
        """List all sessions for a user"""
        return [
            session for session in self.sessions.values()
            if session["user_id"] == user_id
        ]
    
    def store_world(self, world_id: str, world_data: Dict):
        """Store generated world"""
        self.generated_worlds[world_id] = {
            "world_id": world_id,
            "data": world_data,
            "created_at": datetime.utcnow(),
            "size": len(str(world_data)),
            "format": world_data.get("format", "unknown")
        }
    
    def get_world(self, world_id: str) -> Optional[Dict]:
        """Get generated world by ID"""
        return self.generated_worlds.get(world_id)


# Initialize session manager
session_manager = GenerationSession()


@router.on_event("startup")
async def startup_event():
    """Initialize generation components"""
    global world_generator, text_encoder, vision_encoder, fusion_model
    global cache_manager, async_processor
    
    logger.info("Initializing generation components...")
    
    try:
        # Load configuration
        config = InferenceConfig()
        
        # Initialize encoders
        text_encoder = TextEncoder(model_name=config.text_encoder)
        await text_encoder.load()
        
        vision_encoder = VisionEncoder(model_name=config.vision_encoder)
        await vision_encoder.load()
        
        # Initialize fusion model
        fusion_model = CrossAttentionFusion(
            text_dim=text_encoder.output_dim,
            image_dim=vision_encoder.output_dim,
            hidden_dim=config.fusion_hidden_dim
        )
        
        # Initialize world generator
        world_generator = WorldGenerator(
            text_encoder=text_encoder,
            vision_encoder=vision_encoder,
            fusion_model=fusion_model,
            config=config
        )
        await world_generator.load_model()
        
        # Get cache manager and async processor from app state
        # These will be set by the main app
        
        logger.info("Generation components initialized successfully")
    
    except Exception as e:
        logger.error(f"Failed to initialize generation components: {e}")
        raise


@router.post("/generate", response_model=GenerationResponse)
async def generate_world(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    api_key=Depends(require_auth("write"))
):
    """
    Generate a new world from text description
    
    - **prompt**: Text description of the world
    - **style_prompt**: Optional style guidance
    - **reference_image**: Optional reference image URL/base64
    - **parameters**: Generation parameters
    - **async_mode**: Whether to process asynchronously
    """
    
    logger.info(f"Generation request from {api_key.name}: {request.prompt[:50]}...")
    
    # Validate input
    if not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty"
        )
    
    if len(request.prompt) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Prompt too long (max 10000 characters)"
        )
    
    # Check cache for identical requests
    cache_key = f"generation:{hash(request.json())}"
    if cache_manager:
        cached = await cache_manager.get(cache_key)
        if cached:
            logger.info(f"Cache hit for generation: {cache_key}")
            return GenerationResponse(
                session_id="cached",
                status="completed",
                progress=1.0,
                world_id=cached["world_id"],
                world_data=cached["world_data"],
                message="Retrieved from cache",
                created_at=datetime.utcnow()
            )
    
    # Create session
    session_id = session_manager.create_session(
        user_id=api_key.key_id,
        request=request
    )
    
    if request.async_mode:
        # Queue for async processing
        if async_processor:
            task_id = await async_processor.submit_task(
                process_generation,
                session_id=session_id,
                request=request,
                api_key=api_key.key_id
            )
            
            return GenerationResponse(
                session_id=session_id,
                task_id=task_id,
                status="queued",
                progress=0.0,
                message="Generation queued for processing",
                created_at=datetime.utcnow()
            )
        else:
            # Fallback to sync processing in background
            background_tasks.add_task(
                process_generation_sync,
                session_id=session_id,
                request=request,
                api_key=api_key
            )
            
            return GenerationResponse(
                session_id=session_id,
                status="processing",
                progress=0.0,
                message="Generation started in background",
                created_at=datetime.utcnow()
            )
    else:
        # Process synchronously
        try:
            result = await process_generation_sync(
                session_id=session_id,
                request=request,
                api_key=api_key
            )
            
            return GenerationResponse(
                session_id=session_id,
                status="completed",
                progress=1.0,
                world_id=result["world_id"],
                world_data=result["world_data"],
                message="World generated successfully",
                created_at=datetime.utcnow(),
                metadata=result.get("metadata", {})
            )
        
        except Exception as e:
            session_manager.update_session(
                session_id=session_id,
                status="failed",
                error=str(e)
            )
            
            logger.error(f"Generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {str(e)}"
            )


async def process_generation(
    session_id: str,
    request: GenerationRequest,
    api_key: str
) -> Dict:
    """Process generation asynchronously"""
    logger.info(f"Processing generation session: {session_id}")
    
    try:
        session_manager.update_session(
            session_id=session_id,
            status="processing",
            progress=0.1
        )
        
        # Encode text prompt
        logger.debug("Encoding text prompt...")
        text_embeddings = await text_encoder.encode(request.prompt)
        session_manager.update_session(
            session_id=session_id,
            progress=0.3
        )
        
        # Encode style prompt if provided
        style_embeddings = None
        if request.style_prompt:
            style_embeddings = await text_encoder.encode(request.style_prompt)
        
        # Encode reference image if provided
        image_embeddings = None
        if request.reference_image:
            logger.debug("Encoding reference image...")
            image_embeddings = await vision_encoder.encode(request.reference_image)
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.5
        )
        
        # Fuse embeddings
        logger.debug("Fusing multimodal embeddings...")
        fused_embeddings = await fusion_model.fuse(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            style_embeddings=style_embeddings
        )
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.7
        )
        
        # Generate world
        logger.debug("Generating world...")
        world_data = await world_generator.generate(
            fused_embeddings=fused_embeddings,
            parameters=request.parameters.dict() if request.parameters else {}
        )
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.9
        )
        
        # Create world ID and store
        world_id = str(uuid.uuid4())
        
        # Prepare result
        result = {
            "world_id": world_id,
            "world_data": world_data,
            "prompt": request.prompt,
            "parameters": request.parameters.dict() if request.parameters else {},
            "created_at": datetime.utcnow().isoformat(),
            "format": world_data.get("format", "neural_representation"),
            "size": len(str(world_data))
        }
        
        # Store in session manager
        session_manager.store_world(world_id, world_data)
        session_manager.update_session(
            session_id=session_id,
            status="completed",
            progress=1.0,
            result=result,
            world_id=world_id
        )
        
        # Cache the result
        if cache_manager:
            await cache_manager.set(
                f"generation:{hash(request.json())}",
                result,
                ttl=3600  # 1 hour
            )
        
        logger.info(f"Generation completed: {session_id} -> {world_id}")
        
        return result
    
    except Exception as e:
        logger.error(f"Generation error in session {session_id}: {e}")
        session_manager.update_session(
            session_id=session_id,
            status="failed",
            error=str(e)
        )
        raise


async def process_generation_sync(
    session_id: str,
    request: GenerationRequest,
    api_key
) -> Dict:
    """Process generation synchronously (wrapper)"""
    return await process_generation(session_id, request, api_key.key_id)


@router.post("/batch", response_model=BatchGenerationResponse)
async def batch_generate(
    request: BatchRequest,
    api_key=Depends(require_auth("write"))
):
    """
    Generate multiple worlds in batch
    
    - **prompts**: List of text descriptions
    - **common_parameters**: Parameters applied to all generations
    - **max_concurrent**: Maximum concurrent generations
    """
    
    logger.info(f"Batch generation request for {len(request.prompts)} prompts")
    
    if len(request.prompts) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 prompts per batch"
        )
    
    if not request.prompts:
        raise HTTPException(
            status_code=400,
            detail="At least one prompt is required"
        )
    
    # Create batch session
    batch_id = str(uuid.uuid4())
    session_ids = []
    
    for prompt in request.prompts:
        gen_request = GenerationRequest(
            prompt=prompt,
            parameters=request.common_parameters,
            async_mode=True
        )
        
        session_id = session_manager.create_session(
            user_id=api_key.key_id,
            request=gen_request
        )
        session_ids.append(session_id)
        
        # Submit for processing
        if async_processor:
            await async_processor.submit_task(
                process_generation,
                session_id=session_id,
                request=gen_request,
                api_key=api_key.key_id
            )
    
    return BatchGenerationResponse(
        batch_id=batch_id,
        session_ids=session_ids,
        total_tasks=len(session_ids),
        status="queued",
        message=f"Batch generation started with {len(session_ids)} tasks"
    )


@router.get("/sessions/{session_id}", response_model=WorldStatus)
async def get_generation_status(
    session_id: str,
    api_key=Depends(require_auth("read"))
):
    """Get status of a generation session"""
    
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    # Check authorization
    if session["user_id"] != api_key.key_id and "admin" not in api_key.permissions:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access this session"
        )
    
    return WorldStatus(
        session_id=session_id,
        status=session["status"],
        progress=session["progress"],
        world_id=session.get("world_id"),
        created_at=session["created_at"],
        started_at=session.get("started_at"),
        completed_at=session.get("completed_at"),
        error=session.get("error"),
        metadata={
            "prompt": session["request"].get("prompt", "")[:100],
            "user_id": session["user_id"]
        }
    )


@router.get("/sessions", response_model=List[WorldStatus])
async def list_generation_sessions(
    status: Optional[str] = Query(None, regex="^(pending|processing|completed|failed)$"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    api_key=Depends(require_auth("read"))
):
    """List generation sessions for the authenticated user"""
    
    sessions = session_manager.list_user_sessions(api_key.key_id)
    
    # Filter by status if provided
    if status:
        sessions = [s for s in sessions if s["status"] == status]
    
    # Apply pagination
    paginated = sessions[offset:offset + limit]
    
    return [
        WorldStatus(
            session_id=s["session_id"],
            status=s["status"],
            progress=s["progress"],
            world_id=s.get("world_id"),
            created_at=s["created_at"],
            started_at=s.get("started_at"),
            completed_at=s.get("completed_at"),
            error=s.get("error"),
            metadata={
                "prompt": s["request"].get("prompt", "")[:50] + "..." if len(s["request"].get("prompt", "")) > 50 else s["request"].get("prompt", ""),
                "user_id": s["user_id"]
            }
        )
        for s in paginated
    ]


@router.get("/worlds/{world_id}")
async def get_generated_world(
    world_id: str,
    include_data: bool = Query(False, description="Include full world data"),
    api_key=Depends(require_auth("read"))
):
    """Get a generated world by ID"""
    
    world = session_manager.get_world(world_id)
    
    if not world:
        raise HTTPException(
            status_code=404,
            detail="World not found"
        )
    
    # In a real implementation, you would check ownership/permissions
    # For now, we'll allow any authenticated user
    
    if include_data:
        return {
            "world_id": world_id,
            "data": world["data"],
            "created_at": world["created_at"],
            "format": world["format"],
            "size": world["size"]
        }
    else:
        return {
            "world_id": world_id,
            "created_at": world["created_at"],
            "format": world["format"],
            "size": world["size"],
            "has_data": True
        }


@router.delete("/sessions/{session_id}", response_model=SuccessResponse)
async def cancel_generation(
    session_id: str,
    api_key=Depends(require_auth("write"))
):
    """Cancel a generation session"""
    
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    # Check authorization
    if session["user_id"] != api_key.key_id and "admin" not in api_key.permissions:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to cancel this session"
        )
    
    # Only cancel if not already completed/failed
    if session["status"] in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel session with status: {session['status']}"
        )
    
    # Update session status
    session_manager.update_session(
        session_id=session_id,
        status="cancelled",
        error="Cancelled by user"
    )
    
    # TODO: Actually cancel the processing task if possible
    
    return SuccessResponse(
        success=True,
        message="Generation session cancelled",
        data={"session_id": session_id}
    )


@router.get("/progress/{session_id}/stream")
async def stream_progress(
    session_id: str,
    api_key=Depends(require_auth("read"))
):
    """Stream progress updates for a generation session"""
    
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    # Check authorization
    if session["user_id"] != api_key.key_id and "admin" not in api_key.permissions:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access this session"
        )
    
    async def event_generator():
        """Generate Server-Sent Events for progress updates"""
        last_progress = -1.0
        
        while True:
            # Get current session state
            current_session = session_manager.get_session(session_id)
            
            if not current_session:
                yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
                break
            
            current_progress = current_session["progress"]
            status = current_session["status"]
            
            # Send update if progress changed or status changed
            if current_progress != last_progress or status != session["status"]:
                update = ProgressUpdate(
                    session_id=session_id,
                    progress=current_progress,
                    status=status,
                    message=f"Progress: {current_progress*100:.1f}%",
                    timestamp=datetime.utcnow()
                )
                
                yield f"data: {update.json()}\n\n"
                last_progress = current_progress
            
            # Check if completed
            if status in ["completed", "failed", "cancelled"]:
                # Final update
                final_update = ProgressUpdate(
                    session_id=session_id,
                    progress=current_progress,
                    status=status,
                    message=f"Generation {status}",
                    timestamp=datetime.utcnow(),
                    world_id=current_session.get("world_id"),
                    error=current_session.get("error")
                )
                yield f"data: {final_update.json()}\n\n"
                break
            
            # Wait before checking again
            await asyncio.sleep(0.5)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/capabilities")
async def get_generation_capabilities(
    api_key=Depends(require_auth("read"))
):
    """Get generation capabilities and limits"""
    
    config = InferenceConfig()
    
    return {
        "text_encoder": config.text_encoder,
        "vision_encoder": config.vision_encoder,
        "max_prompt_length": 10000,
        "supported_formats": ["neural_representation", "gaussian_splatting", "nerf", "mesh"],
        "max_batch_size": 100,
        "max_concurrent": config.max_concurrent_generations,
        "default_parameters": {
            "quality": config.default_quality,
            "resolution": config.default_resolution,
            "steps": config.default_steps,
            "guidance_scale": config.default_guidance_scale
        },
        "available_styles": [
            "realistic",
            "fantasy",
            "sci-fi",
            "cartoon",
            "painterly",
            "minimal",
            "detailed"
        ]
    }


# Helper function for JSON serialization in SSE
import json
from .schemas.response_models import ProgressUpdate
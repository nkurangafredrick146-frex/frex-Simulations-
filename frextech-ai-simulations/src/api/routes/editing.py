"""
Editing routes for modifying existing worlds
"""

import uuid
import asyncio
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import logging

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..middleware.authentication import require_auth
from ..schemas.request_models import (
    EditRequest,
    ExpansionRequest,
    CompositionRequest,
    RegionSelection,
    EditOperation
)
from ..schemas.response_models import (
    EditResponse,
    ExpansionResponse,
    CompositionResponse,
    WorldStatus,
    ProgressUpdate,
    APIError,
    SuccessResponse
)
from ..utils.async_processor import AsyncProcessor
from ..utils.cache_manager import CacheManager
from ..utils.file_handler import FileHandler
from src.interactive.editor.pano_editor import PanoramaEditor
from src.interactive.editor.region_selector import RegionSelector
from src.interactive.editor.edit_propagator import EditPropagator
from src.interactive.expansion.scene_expander import SceneExpander
from src.interactive.expansion.boundary_detector import BoundaryDetector
from src.interactive.expansion.seam_blender import SeamBlender
from src.interactive.composition.scene_composer import SceneComposer
from src.interactive.composition.transition_builder import TransitionBuilder
from src.utils.logging_config import setup_logging
from configs.model.inference import InferenceConfig

logger = setup_logging("editing_routes")

router = APIRouter()

# Global components
panorama_editor: Optional[PanoramaEditor] = None
region_selector: Optional[RegionSelector] = None
edit_propagator: Optional[EditPropagator] = None
scene_expander: Optional[SceneExpander] = None
boundary_detector: Optional[BoundaryDetector] = None
seam_blender: Optional[SeamBlender] = None
scene_composer: Optional[SceneComposer] = None
transition_builder: Optional[TransitionBuilder] = None
cache_manager: Optional[CacheManager] = None
async_processor: Optional[AsyncProcessor] = None


class EditSession:
    """Manages editing sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.edited_worlds: Dict[str, Dict] = {}
        self.world_versions: Dict[str, List[Dict]] = {}
    
    def create_session(
        self,
        user_id: str,
        world_id: str,
        operation: str,
        parameters: Dict
    ) -> str:
        """Create a new editing session"""
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "world_id": world_id,
            "operation": operation,
            "parameters": parameters,
            "status": "pending",
            "progress": 0.0,
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "result": None,
            "new_world_id": None,
            "parent_session_id": None
        }
        
        # Initialize version tracking for world
        if world_id not in self.world_versions:
            self.world_versions[world_id] = []
        
        return session_id
    
    def update_session(
        self,
        session_id: str,
        status: str = None,
        progress: float = None,
        error: str = None,
        result: Dict = None,
        new_world_id: str = None,
        parent_session_id: str = None
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
        
        if new_world_id is not None:
            session["new_world_id"] = new_world_id
            
            # Store the new world
            if result and "world_data" in result:
                self.edited_worlds[new_world_id] = {
                    "world_id": new_world_id,
                    "parent_world_id": session["world_id"],
                    "session_id": session_id,
                    "data": result["world_data"],
                    "created_at": datetime.utcnow(),
                    "operation": session["operation"],
                    "parameters": session["parameters"]
                }
                
                # Add to version history
                self.world_versions[session["world_id"]].append({
                    "version_id": new_world_id,
                    "session_id": session_id,
                    "operation": session["operation"],
                    "created_at": datetime.utcnow(),
                    "parent_version": session.get("parent_session_id")
                })
        
        if parent_session_id is not None:
            session["parent_session_id"] = parent_session_id
        
        return True
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def list_world_sessions(self, world_id: str) -> List[Dict]:
        """List all editing sessions for a world"""
        return [
            session for session in self.sessions.values()
            if session["world_id"] == world_id
        ]
    
    def list_user_sessions(self, user_id: str) -> List[Dict]:
        """List all editing sessions for a user"""
        return [
            session for session in self.sessions.values()
            if session["user_id"] == user_id
        ]
    
    def get_world_version_history(self, world_id: str) -> List[Dict]:
        """Get version history for a world"""
        return self.world_versions.get(world_id, [])
    
    def get_world_by_version(self, world_id: str, version_id: str) -> Optional[Dict]:
        """Get a specific version of a world"""
        # Check if this is the latest version
        if version_id == "latest":
            # Get the most recent version
            versions = self.world_versions.get(world_id, [])
            if versions:
                latest = versions[-1]
                return self.edited_worlds.get(latest["version_id"])
            else:
                # No versions, return original if it exists in edited_worlds
                return self.edited_worlds.get(world_id)
        
        # Return specific version
        return self.edited_worlds.get(version_id)


# Initialize session manager
session_manager = EditSession()


@router.on_event("startup")
async def startup_event():
    """Initialize editing components"""
    global panorama_editor, region_selector, edit_propagator
    global scene_expander, boundary_detector, seam_blender
    global scene_composer, transition_builder
    global cache_manager, async_processor
    
    logger.info("Initializing editing components...")
    
    try:
        # Load configuration
        config = InferenceConfig()
        
        # Initialize editing components
        panorama_editor = PanoramaEditor(config=config)
        await panorama_editor.load_model()
        
        region_selector = RegionSelector(config=config)
        
        edit_propagator = EditPropagator(config=config)
        await edit_propagator.load_model()
        
        scene_expander = SceneExpander(config=config)
        await scene_expander.load_model()
        
        boundary_detector = BoundaryDetector(config=config)
        
        seam_blender = SeamBlender(config=config)
        
        scene_composer = SceneComposer(config=config)
        await scene_composer.load_model()
        
        transition_builder = TransitionBuilder(config=config)
        
        logger.info("Editing components initialized successfully")
    
    except Exception as e:
        logger.error(f"Failed to initialize editing components: {e}")
        raise


@router.post("/edit", response_model=EditResponse)
async def edit_world(
    request: EditRequest,
    background_tasks: BackgroundTasks,
    api_key=Depends(require_auth("write"))
):
    """
    Edit an existing world
    
    - **world_id**: ID of world to edit
    - **operations**: List of edit operations
    - **region**: Optional region to apply edits to
    - **parameters**: Edit parameters
    - **async_mode**: Whether to process asynchronously
    """
    
    logger.info(f"Edit request for world {request.world_id}")
    
    # Validate input
    if not request.operations:
        raise HTTPException(
            status_code=400,
            detail="At least one edit operation is required"
        )
    
    # Check if world exists (in a real system, this would check database)
    # For now, we'll assume it exists if we have a session for it
    
    # Create edit session
    session_id = session_manager.create_session(
        user_id=api_key.key_id,
        world_id=request.world_id,
        operation="edit",
        parameters=request.dict()
    )
    
    if request.async_mode:
        # Queue for async processing
        if async_processor:
            task_id = await async_processor.submit_task(
                process_edit,
                session_id=session_id,
                request=request,
                api_key=api_key.key_id
            )
            
            return EditResponse(
                session_id=session_id,
                task_id=task_id,
                status="queued",
                progress=0.0,
                message="Edit operation queued for processing",
                created_at=datetime.utcnow()
            )
        else:
            # Fallback to sync processing in background
            background_tasks.add_task(
                process_edit_sync,
                session_id=session_id,
                request=request,
                api_key=api_key
            )
            
            return EditResponse(
                session_id=session_id,
                status="processing",
                progress=0.0,
                message="Edit operation started in background",
                created_at=datetime.utcnow()
            )
    else:
        # Process synchronously
        try:
            result = await process_edit_sync(
                session_id=session_id,
                request=request,
                api_key=api_key
            )
            
            return EditResponse(
                session_id=session_id,
                status="completed",
                progress=1.0,
                new_world_id=result["new_world_id"],
                world_data=result.get("world_data"),
                message="World edited successfully",
                created_at=datetime.utcnow(),
                metadata=result.get("metadata", {})
            )
        
        except Exception as e:
            session_manager.update_session(
                session_id=session_id,
                status="failed",
                error=str(e)
            )
            
            logger.error(f"Edit operation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Edit operation failed: {str(e)}"
            )


async def process_edit(
    session_id: str,
    request: EditRequest,
    api_key: str
) -> Dict:
    """Process edit operation asynchronously"""
    logger.info(f"Processing edit session: {session_id}")
    
    try:
        session_manager.update_session(
            session_id=session_id,
            status="processing",
            progress=0.1
        )
        
        # Get the world data (in real system, load from storage)
        # For now, we'll use a placeholder
        world_data = await load_world_data(request.world_id)
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.2
        )
        
        # Apply each edit operation
        for i, operation in enumerate(request.operations):
            logger.debug(f"Applying operation {i+1}/{len(request.operations)}: {operation.type}")
            
            # Update progress
            progress = 0.2 + (i / len(request.operations)) * 0.6
            session_manager.update_session(
                session_id=session_id,
                progress=progress
            )
            
            # Apply the operation
            world_data = await apply_edit_operation(
                world_data=world_data,
                operation=operation,
                region=request.region,
                parameters=request.parameters
            )
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.8
        )
        
        # Apply consistency checking
        if request.parameters and request.parameters.get("check_consistency", True):
            logger.debug("Checking edit consistency...")
            world_data = await panorama_editor.check_consistency(world_data)
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.9
        )
        
        # Create new world ID
        new_world_id = str(uuid.uuid4())
        
        # Prepare result
        result = {
            "new_world_id": new_world_id,
            "world_data": world_data,
            "parent_world_id": request.world_id,
            "operations": [op.dict() for op in request.operations],
            "created_at": datetime.utcnow().isoformat(),
            "format": world_data.get("format", "neural_representation"),
            "size": len(str(world_data))
        }
        
        # Store in session manager
        session_manager.update_session(
            session_id=session_id,
            status="completed",
            progress=1.0,
            result=result,
            new_world_id=new_world_id
        )
        
        logger.info(f"Edit completed: {session_id} -> {new_world_id}")
        
        return result
    
    except Exception as e:
        logger.error(f"Edit error in session {session_id}: {e}")
        session_manager.update_session(
            session_id=session_id,
            status="failed",
            error=str(e)
        )
        raise


async def process_edit_sync(
    session_id: str,
    request: EditRequest,
    api_key
) -> Dict:
    """Process edit synchronously (wrapper)"""
    return await process_edit(session_id, request, api_key.key_id)


async def apply_edit_operation(
    world_data: Dict,
    operation: EditOperation,
    region: Optional[RegionSelection],
    parameters: Dict
) -> Dict:
    """Apply a single edit operation to world data"""
    
    if operation.type == "text_prompt":
        # Edit based on text prompt
        if region:
            # Edit specific region
            return await panorama_editor.edit_region(
                world_data=world_data,
                region=region.dict(),
                prompt=operation.prompt,
                parameters=parameters
            )
        else:
            # Edit entire world
            return await panorama_editor.edit_panorama(
                world_data=world_data,
                prompt=operation.prompt,
                parameters=parameters
            )
    
    elif operation.type == "style_transfer":
        # Apply style transfer
        return await panorama_editor.transfer_style(
            world_data=world_data,
            style_prompt=operation.prompt,
            region=region.dict() if region else None,
            parameters=parameters
        )
    
    elif operation.type == "object_removal":
        # Remove object
        return await panorama_editor.remove_object(
            world_data=world_data,
            region=region.dict() if region else None,
            parameters=parameters
        )
    
    elif operation.type == "object_addition":
        # Add object
        return await panorama_editor.add_object(
            world_data=world_data,
            prompt=operation.prompt,
            region=region.dict() if region else None,
            parameters=parameters
        )
    
    elif operation.type == "color_adjustment":
        # Adjust colors
        return await panorama_editor.adjust_colors(
            world_data=world_data,
            adjustments=operation.parameters,
            region=region.dict() if region else None
        )
    
    elif operation.type == "lighting_adjustment":
        # Adjust lighting
        return await panorama_editor.adjust_lighting(
            world_data=world_data,
            adjustments=operation.parameters,
            region=region.dict() if region else None
        )
    
    else:
        raise ValueError(f"Unknown operation type: {operation.type}")


async def load_world_data(world_id: str) -> Dict:
    """Load world data from storage"""
    # In a real implementation, this would load from database or file system
    # For now, return a placeholder
    return {
        "world_id": world_id,
        "format": "neural_representation",
        "data": {"placeholder": True},
        "created_at": datetime.utcnow().isoformat()
    }


@router.post("/expand", response_model=ExpansionResponse)
async def expand_world(
    request: ExpansionRequest,
    background_tasks: BackgroundTasks,
    api_key=Depends(require_auth("write"))
):
    """
    Expand a world in a specific direction
    
    - **world_id**: ID of world to expand
    - **direction**: Direction to expand (north, south, east, west, up, down)
    - **distance**: Distance to expand
    - **prompt**: Optional prompt for the new area
    - **seamless**: Whether to create seamless transition
    - **async_mode**: Whether to process asynchronously
    """
    
    logger.info(f"Expand request for world {request.world_id} in direction {request.direction}")
    
    # Validate input
    valid_directions = ["north", "south", "east", "west", "up", "down"]
    if request.direction not in valid_directions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid direction. Must be one of: {', '.join(valid_directions)}"
        )
    
    if request.distance <= 0:
        raise HTTPException(
            status_code=400,
            detail="Distance must be positive"
        )
    
    # Create expansion session
    session_id = session_manager.create_session(
        user_id=api_key.key_id,
        world_id=request.world_id,
        operation="expand",
        parameters=request.dict()
    )
    
    if request.async_mode:
        if async_processor:
            task_id = await async_processor.submit_task(
                process_expansion,
                session_id=session_id,
                request=request,
                api_key=api_key.key_id
            )
            
            return ExpansionResponse(
                session_id=session_id,
                task_id=task_id,
                status="queued",
                progress=0.0,
                message="Expansion queued for processing",
                created_at=datetime.utcnow()
            )
        else:
            background_tasks.add_task(
                process_expansion_sync,
                session_id=session_id,
                request=request,
                api_key=api_key
            )
            
            return ExpansionResponse(
                session_id=session_id,
                status="processing",
                progress=0.0,
                message="Expansion started in background",
                created_at=datetime.utcnow()
            )
    else:
        try:
            result = await process_expansion_sync(
                session_id=session_id,
                request=request,
                api_key=api_key
            )
            
            return ExpansionResponse(
                session_id=session_id,
                status="completed",
                progress=1.0,
                new_world_id=result["new_world_id"],
                world_data=result.get("world_data"),
                message="World expanded successfully",
                created_at=datetime.utcnow(),
                metadata=result.get("metadata", {})
            )
        
        except Exception as e:
            session_manager.update_session(
                session_id=session_id,
                status="failed",
                error=str(e)
            )
            
            logger.error(f"Expansion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Expansion failed: {str(e)}"
            )


async def process_expansion(
    session_id: str,
    request: ExpansionRequest,
    api_key: str
) -> Dict:
    """Process expansion operation"""
    logger.info(f"Processing expansion session: {session_id}")
    
    try:
        session_manager.update_session(
            session_id=session_id,
            status="processing",
            progress=0.1
        )
        
        # Load world data
        world_data = await load_world_data(request.world_id)
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.3
        )
        
        # Detect boundaries
        logger.debug("Detecting boundaries...")
        boundaries = await boundary_detector.detect(world_data)
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.4
        )
        
        # Expand in specified direction
        logger.debug(f"Expanding in direction: {request.direction}")
        expanded_data = await scene_expander.expand(
            world_data=world_data,
            direction=request.direction,
            distance=request.distance,
            boundaries=boundaries,
            prompt=request.prompt,
            parameters=request.parameters
        )
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.7
        )
        
        # Blend seams if requested
        if request.seamless:
            logger.debug("Blending seams...")
            expanded_data = await seam_blender.blend(
                original_data=world_data,
                expanded_data=expanded_data,
                direction=request.direction,
                parameters=request.parameters
            )
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.9
        )
        
        # Create new world ID
        new_world_id = str(uuid.uuid4())
        
        # Prepare result
        result = {
            "new_world_id": new_world_id,
            "world_data": expanded_data,
            "parent_world_id": request.world_id,
            "direction": request.direction,
            "distance": request.distance,
            "created_at": datetime.utcnow().isoformat(),
            "format": expanded_data.get("format", "neural_representation"),
            "size": len(str(expanded_data))
        }
        
        # Store in session manager
        session_manager.update_session(
            session_id=session_id,
            status="completed",
            progress=1.0,
            result=result,
            new_world_id=new_world_id
        )
        
        logger.info(f"Expansion completed: {session_id} -> {new_world_id}")
        
        return result
    
    except Exception as e:
        logger.error(f"Expansion error in session {session_id}: {e}")
        session_manager.update_session(
            session_id=session_id,
            status="failed",
            error=str(e)
        )
        raise


async def process_expansion_sync(
    session_id: str,
    request: ExpansionRequest,
    api_key
) -> Dict:
    """Process expansion synchronously"""
    return await process_expansion(session_id, request, api_key.key_id)


@router.post("/compose", response_model=CompositionResponse)
async def compose_worlds(
    request: CompositionRequest,
    background_tasks: BackgroundTasks,
    api_key=Depends(require_auth("write"))
):
    """
    Compose multiple worlds together
    
    - **world_ids**: List of world IDs to compose
    - **layout**: Layout specification
    - **transitions**: Transition specifications
    - **parameters**: Composition parameters
    - **async_mode**: Whether to process asynchronously
    """
    
    logger.info(f"Compose request for {len(request.world_ids)} worlds")
    
    # Validate input
    if len(request.world_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 worlds are required for composition"
        )
    
    if len(request.world_ids) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 worlds per composition"
        )
    
    # Create composition session
    session_id = session_manager.create_session(
        user_id=api_key.key_id,
        world_id=request.world_ids[0],  # Use first world as parent
        operation="compose",
        parameters=request.dict()
    )
    
    if request.async_mode:
        if async_processor:
            task_id = await async_processor.submit_task(
                process_composition,
                session_id=session_id,
                request=request,
                api_key=api_key.key_id
            )
            
            return CompositionResponse(
                session_id=session_id,
                task_id=task_id,
                status="queued",
                progress=0.0,
                message="Composition queued for processing",
                created_at=datetime.utcnow()
            )
        else:
            background_tasks.add_task(
                process_composition_sync,
                session_id=session_id,
                request=request,
                api_key=api_key
            )
            
            return CompositionResponse(
                session_id=session_id,
                status="processing",
                progress=0.0,
                message="Composition started in background",
                created_at=datetime.utcnow()
            )
    else:
        try:
            result = await process_composition_sync(
                session_id=session_id,
                request=request,
                api_key=api_key
            )
            
            return CompositionResponse(
                session_id=session_id,
                status="completed",
                progress=1.0,
                new_world_id=result["new_world_id"],
                world_data=result.get("world_data"),
                message="Worlds composed successfully",
                created_at=datetime.utcnow(),
                metadata=result.get("metadata", {})
            )
        
        except Exception as e:
            session_manager.update_session(
                session_id=session_id,
                status="failed",
                error=str(e)
            )
            
            logger.error(f"Composition failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Composition failed: {str(e)}"
            )


async def process_composition(
    session_id: str,
    request: CompositionRequest,
    api_key: str
) -> Dict:
    """Process composition operation"""
    logger.info(f"Processing composition session: {session_id}")
    
    try:
        session_manager.update_session(
            session_id=session_id,
            status="processing",
            progress=0.1
        )
        
        # Load all world data
        world_data_list = []
        for i, world_id in enumerate(request.world_ids):
            logger.debug(f"Loading world {i+1}/{len(request.world_ids)}: {world_id}")
            world_data = await load_world_data(world_id)
            world_data_list.append(world_data)
            
            # Update progress
            progress = 0.1 + (i / len(request.world_ids)) * 0.3
            session_manager.update_session(
                session_id=session_id,
                progress=progress
            )
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.4
        )
        
        # Compose worlds
        logger.debug("Composing worlds...")
        composed_data = await scene_composer.compose(
            world_data_list=world_data_list,
            layout=request.layout.dict() if request.layout else None,
            parameters=request.parameters
        )
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.7
        )
        
        # Apply transitions if specified
        if request.transitions:
            logger.debug("Applying transitions...")
            composed_data = await transition_builder.build_transitions(
                world_data=composed_data,
                transitions=request.transitions,
                parameters=request.parameters
            )
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.9
        )
        
        # Create new world ID
        new_world_id = str(uuid.uuid4())
        
        # Prepare result
        result = {
            "new_world_id": new_world_id,
            "world_data": composed_data,
            "source_world_ids": request.world_ids,
            "created_at": datetime.utcnow().isoformat(),
            "format": composed_data.get("format", "neural_representation"),
            "size": len(str(composed_data))
        }
        
        # Store in session manager
        session_manager.update_session(
            session_id=session_id,
            status="completed",
            progress=1.0,
            result=result,
            new_world_id=new_world_id
        )
        
        logger.info(f"Composition completed: {session_id} -> {new_world_id}")
        
        return result
    
    except Exception as e:
        logger.error(f"Composition error in session {session_id}: {e}")
        session_manager.update_session(
            session_id=session_id,
            status="failed",
            error=str(e)
        )
        raise


async def process_composition_sync(
    session_id: str,
    request: CompositionRequest,
    api_key
) -> Dict:
    """Process composition synchronously"""
    return await process_composition(session_id, request, api_key.key_id)


@router.get("/sessions/{session_id}", response_model=WorldStatus)
async def get_edit_status(
    session_id: str,
    api_key=Depends(require_auth("read"))
):
    """Get status of an editing session"""
    
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
        world_id=session.get("new_world_id") or session["world_id"],
        created_at=session["created_at"],
        started_at=session.get("started_at"),
        completed_at=session.get("completed_at"),
        error=session.get("error"),
        metadata={
            "operation": session["operation"],
            "user_id": session["user_id"]
        }
    )


@router.get("/worlds/{world_id}/versions")
async def get_world_versions(
    world_id: str,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    api_key=Depends(require_auth("read"))
):
    """Get version history for a world"""
    
    versions = session_manager.get_world_version_history(world_id)
    
    # Apply pagination
    paginated = versions[offset:offset + limit]
    
    return {
        "world_id": world_id,
        "total_versions": len(versions),
        "versions": [
            {
                "version_id": v["version_id"],
                "session_id": v["session_id"],
                "operation": v["operation"],
                "created_at": v["created_at"],
                "parent_version": v.get("parent_version")
            }
            for v in paginated
        ]
    }


@router.get("/worlds/{world_id}/versions/{version_id}")
async def get_world_version(
    world_id: str,
    version_id: str,
    include_data: bool = Query(False, description="Include full world data"),
    api_key=Depends(require_auth("read"))
):
    """Get a specific version of a world"""
    
    world = session_manager.get_world_by_version(world_id, version_id)
    
    if not world:
        raise HTTPException(
            status_code=404,
            detail="World version not found"
        )
    
    if include_data:
        return {
            "version_id": world["world_id"],
            "parent_world_id": world.get("parent_world_id"),
            "session_id": world.get("session_id"),
            "data": world["data"],
            "created_at": world["created_at"],
            "operation": world.get("operation", "unknown"),
            "format": world.get("format", "unknown"),
            "size": world.get("size", 0)
        }
    else:
        return {
            "version_id": world["world_id"],
            "parent_world_id": world.get("parent_world_id"),
            "session_id": world.get("session_id"),
            "created_at": world["created_at"],
            "operation": world.get("operation", "unknown"),
            "format": world.get("format", "unknown"),
            "size": world.get("size", 0),
            "has_data": True
        }


@router.post("/regions/detect")
async def detect_regions(
    world_id: str,
    method: str = Query("semantic", regex="^(semantic|geometric|hybrid)$"),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    api_key=Depends(require_auth("read"))
):
    """Detect editable regions in a world"""
    
    logger.info(f"Region detection request for world {world_id}")
    
    # Load world data
    world_data = await load_world_data(world_id)
    
    # Detect regions
    regions = await region_selector.detect_regions(
        world_data=world_data,
        method=method,
        threshold=threshold
    )
    
    return {
        "world_id": world_id,
        "method": method,
        "threshold": threshold,
        "regions": regions,
        "count": len(regions)
    }


@router.post("/propagate")
async def propagate_edit(
    source_session_id: str,
    target_world_id: str,
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0),
    api_key=Depends(require_auth("write"))
):
    """Propagate edits from one world to another similar world"""
    
    logger.info(f"Edit propagation from session {source_session_id} to world {target_world_id}")
    
    # Get source session
    source_session = session_manager.get_session(source_session_id)
    if not source_session:
        raise HTTPException(
            status_code=404,
            detail="Source session not found"
        )
    
    # Check authorization
    if source_session["user_id"] != api_key.key_id and "admin" not in api_key.permissions:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access source session"
        )
    
    # Load world data
    source_world_data = await load_world_data(source_session["world_id"])
    target_world_data = await load_world_data(target_world_id)
    
    # Propagate edits
    propagated_data = await edit_propagator.propagate(
        source_world=source_world_data,
        target_world=target_world_data,
        edit_operations=source_session.get("parameters", {}).get("operations", []),
        similarity_threshold=similarity_threshold
    )
    
    # Create new session for propagated edit
    new_session_id = session_manager.create_session(
        user_id=api_key.key_id,
        world_id=target_world_id,
        operation="propagated_edit",
        parameters={
            "source_session_id": source_session_id,
            "similarity_threshold": similarity_threshold
        }
    )
    
    session_manager.update_session(
        session_id=new_session_id,
        parent_session_id=source_session_id
    )
    
    # Create new world ID
    new_world_id = str(uuid.uuid4())
    
    # Store result
    result = {
        "new_world_id": new_world_id,
        "world_data": propagated_data,
        "parent_world_id": target_world_id,
        "source_session_id": source_session_id,
        "created_at": datetime.utcnow().isoformat()
    }
    
    session_manager.update_session(
        session_id=new_session_id,
        status="completed",
        progress=1.0,
        result=result,
        new_world_id=new_world_id
    )
    
    return {
        "session_id": new_session_id,
        "new_world_id": new_world_id,
        "source_session_id": source_session_id,
        "similarity_threshold": similarity_threshold,
        "message": "Edit propagated successfully"
    }


@router.get("/capabilities")
async def get_editing_capabilities(
    api_key=Depends(require_auth("read"))
):
    """Get editing capabilities and limits"""
    
    return {
        "supported_operations": [
            {
                "type": "text_prompt",
                "description": "Edit based on text description",
                "parameters": ["prompt", "strength", "guidance_scale"]
            },
            {
                "type": "style_transfer",
                "description": "Transfer style from prompt",
                "parameters": ["style_prompt", "strength"]
            },
            {
                "type": "object_removal",
                "description": "Remove objects from region",
                "parameters": ["region", "inpaint_strength"]
            },
            {
                "type": "object_addition",
                "description": "Add objects to region",
                "parameters": ["prompt", "region", "blend_strength"]
            },
            {
                "type": "color_adjustment",
                "description": "Adjust colors",
                "parameters": ["brightness", "contrast", "saturation", "hue"]
            },
            {
                "type": "lighting_adjustment",
                "description": "Adjust lighting",
                "parameters": ["intensity", "direction", "color"]
            }
        ],
        "expansion": {
            "directions": ["north", "south", "east", "west", "up", "down"],
            "max_distance": 100.0,
            "min_distance": 0.1,
            "seamless_blending": True
        },
        "composition": {
            "max_worlds": 10,
            "supported_layouts": ["grid", "linear", "radial", "custom"],
            "transition_types": ["fade", "morph", "portal", "seamless"]
        },
        "region_detection": {
            "methods": ["semantic", "geometric", "hybrid"],
            "max_regions": 50
        },
        "limits": {
            "max_operations_per_edit": 20,
            "max_region_size": 0.5,  # Fraction of world
            "max_expansions_per_world": 100
        }
    }
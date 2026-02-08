"""
Inference API router — endpoints for managing inference sessions.

Provides REST endpoints to start/stop inference, query session status,
update YOLO-World prompts, and switch models. Uses DI via
set_inference_manager() called from app lifespan.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.core.exceptions import ConflictError, NotFoundError
from backend.inference.manager import InferenceManager
from backend.inference.models import ModelType
from backend.models.common import ErrorResponse
from backend.models.inference import (
    InferenceSessionResponse,
    InferenceStatusResponse,
    StartInferenceRequest,
    StartInferenceResponse,
    StopInferenceRequest,
    SwitchModelRequest,
    UpdatePromptsRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level DI — set from app lifespan
_inference_manager: InferenceManager | None = None


def set_inference_manager(manager: InferenceManager | None) -> None:
    """
    Inject the InferenceManager instance for the API router.

    Called from the app lifespan on startup (with instance) and
    teardown (with None).

    Args:
        manager: The InferenceManager instance, or None to reset.
    """
    global _inference_manager
    _inference_manager = manager


def _get_manager() -> InferenceManager:
    """
    Get the injected InferenceManager, raising 500 if not initialized.

    Returns:
        The active InferenceManager instance.
    """
    if _inference_manager is None:
        raise RuntimeError("InferenceManager not initialized")
    return _inference_manager


@router.post(
    "/start",
    summary="Start inference",
    description="Start inference on a feed, creating an inference output feed.",
    response_model=StartInferenceResponse,
    status_code=201,
    responses={
        404: {"model": ErrorResponse, "description": "Source feed not found"},
        409: {"model": ErrorResponse, "description": "Inference already running on feed"},
    },
)
async def start_inference(request: StartInferenceRequest) -> StartInferenceResponse:
    """
    Start an inference session on a source feed.

    Creates a derived feed with YOLO detection results. The output_feed_id
    can be subscribed to via WebSocket for real-time detection streaming.
    """
    mgr = _get_manager()

    model_type = ModelType(request.model_type)
    output_feed_id = mgr.start_inference(
        source_feed_id=request.source_feed_id,
        model_name=request.model_name,
        model_type=model_type,
        prompts=request.prompts,
        confidence_threshold=request.confidence_threshold,
    )

    return StartInferenceResponse(
        output_feed_id=output_feed_id,
        source_feed_id=request.source_feed_id,
        model_name=request.model_name,
        model_type=request.model_type,
    )


@router.post(
    "/stop",
    summary="Stop inference",
    description="Stop the active inference session.",
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def stop_inference(request: StopInferenceRequest) -> dict:
    """Stop an active inference session and clean up resources."""
    mgr = _get_manager()

    if not mgr.stop_inference(request.output_feed_id):
        raise NotFoundError("Inference session", request.output_feed_id)

    return {"status": "stopped", "output_feed_id": request.output_feed_id}


@router.get(
    "/status",
    summary="Get inference status",
    description="Get current inference state and statistics for all sessions.",
    response_model=InferenceStatusResponse,
)
async def get_inference_status() -> InferenceStatusResponse:
    """Get status of all active inference sessions."""
    mgr = _get_manager()

    sessions = mgr.get_all_sessions()
    return InferenceStatusResponse(
        sessions=[InferenceSessionResponse(**s) for s in sessions],
        count=len(sessions),
    )


@router.put(
    "/prompts",
    summary="Update prompts",
    description="Update YOLO-World detection prompts on an active session.",
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def update_prompts(request: UpdatePromptsRequest) -> dict:
    """
    Update YOLO-World text prompts on an active inference session.

    Restarts the session with new prompts. The output_feed_id will change.
    """
    mgr = _get_manager()

    if not mgr.update_prompts(request.output_feed_id, request.prompts):
        raise NotFoundError("Inference session", request.output_feed_id)

    return {"status": "updated", "output_feed_id": request.output_feed_id}


@router.put(
    "/model",
    summary="Switch model",
    description="Switch the active inference model on an existing session.",
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def switch_model(request: SwitchModelRequest) -> dict:
    """
    Switch the model on an active inference session.

    Stops the old session and starts a new one with the specified model.
    Returns the new output_feed_id.
    """
    mgr = _get_manager()

    model_type = ModelType(request.model_type)
    new_output_id = mgr.switch_model(
        request.output_feed_id,
        request.model_name,
        model_type,
        request.prompts,
    )

    if new_output_id is None:
        raise NotFoundError("Inference session", request.output_feed_id)

    return {
        "status": "switched",
        "old_output_feed_id": request.output_feed_id,
        "new_output_feed_id": new_output_id,
    }

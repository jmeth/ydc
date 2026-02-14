"""
Capture API router — endpoints for capture session management.

Orchestrates frame capture from feeds into datasets, supporting both
raw feed capture (no annotations) and inference feed capture (with
auto-generated YOLO annotations).
"""

from fastapi import APIRouter

from backend.capture.manager import CaptureManager
from backend.models.capture import (
    CaptureStatusResponse,
    ManualCaptureResponse,
    StartCaptureRequest,
    UpdateCaptureConfigRequest,
)
from backend.models.common import ErrorResponse

router = APIRouter()

_manager: CaptureManager | None = None


def set_capture_manager(manager: CaptureManager | None) -> None:
    """
    Inject the CaptureManager instance (called from app lifespan).

    Args:
        manager: CaptureManager instance, or None to clear on shutdown.
    """
    global _manager
    _manager = manager


def _get_manager() -> CaptureManager:
    """
    Get the injected CaptureManager, raising if not initialized.

    Returns:
        The current CaptureManager.

    Raises:
        RuntimeError: If set_capture_manager() hasn't been called.
    """
    if _manager is None:
        raise RuntimeError("CaptureManager not initialized")
    return _manager


@router.post(
    "/start",
    response_model=CaptureStatusResponse,
    status_code=201,
    summary="Start capture",
    description=(
        "Start a capture session that subscribes to a feed (raw or inference) "
        "and saves frames to a dataset. In inference mode, detections are "
        "converted to YOLO annotations automatically."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "Feed or dataset not found"},
        409: {"model": ErrorResponse, "description": "Capture already running"},
    },
)
async def start_capture(request: StartCaptureRequest) -> CaptureStatusResponse:
    """Start a capture session on the specified feed and dataset."""
    mgr = _get_manager()
    result = await mgr.start(
        feed_id=request.feed_id,
        dataset_name=request.dataset_name,
        split=request.split,
        capture_interval=request.capture_interval,
        negative_ratio=request.negative_ratio,
        confidence_threshold=request.confidence_threshold,
    )
    return CaptureStatusResponse(**result)


@router.post(
    "/stop",
    response_model=CaptureStatusResponse,
    summary="Stop capture",
    description="Stop the active capture session and return final statistics.",
)
async def stop_capture() -> CaptureStatusResponse:
    """Stop the active capture session."""
    mgr = _get_manager()
    result = await mgr.stop()
    return CaptureStatusResponse(**result)


@router.get(
    "/status",
    response_model=CaptureStatusResponse,
    summary="Get capture status",
    description="Get current capture state, configuration, and statistics.",
)
async def get_capture_status() -> CaptureStatusResponse:
    """Get current capture status."""
    mgr = _get_manager()
    result = mgr.get_status()
    return CaptureStatusResponse(**result)


@router.post(
    "/trigger",
    response_model=ManualCaptureResponse,
    summary="Manual capture",
    description=(
        "Trigger a manual frame capture, saving the latest frame from the "
        "feed buffer regardless of the automatic capture interval."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "No frame available"},
        409: {"model": ErrorResponse, "description": "No capture session running"},
    },
)
async def manual_capture() -> ManualCaptureResponse:
    """Trigger a manual frame capture."""
    mgr = _get_manager()
    result = await mgr.manual_trigger()
    return ManualCaptureResponse(**result)


@router.put(
    "/config",
    response_model=CaptureStatusResponse,
    summary="Update capture config",
    description="Update capture settings (interval, threshold, ratio) on the fly.",
)
async def update_capture_config(request: UpdateCaptureConfigRequest) -> CaptureStatusResponse:
    """Update capture configuration."""
    mgr = _get_manager()
    mgr.update_config(
        capture_interval=request.capture_interval,
        negative_ratio=request.negative_ratio,
        confidence_threshold=request.confidence_threshold,
    )
    result = mgr.get_status()
    return CaptureStatusResponse(**result)

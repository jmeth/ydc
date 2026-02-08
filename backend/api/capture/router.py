"""
Capture API router — stub endpoints for capture session management.

The CaptureController orchestrates between Feeds, Inference, and Dataset
subsystems. All endpoints return 501 until it is implemented.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.models.common import NotImplementedResponse

router = APIRouter()

STUB = NotImplementedResponse()
STUB_RESPONSE = {501: {"model": NotImplementedResponse}}


@router.post(
    "/start",
    summary="Start capture",
    description="Start a capture session that subscribes to an inference output feed.",
    responses=STUB_RESPONSE,
)
async def start_capture() -> JSONResponse:
    """Start capture session."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.post(
    "/stop",
    summary="Stop capture",
    description="Stop the active capture session.",
    responses=STUB_RESPONSE,
)
async def stop_capture() -> JSONResponse:
    """Stop capture session."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.get(
    "/status",
    summary="Get capture status",
    description="Get current capture state and statistics.",
    responses=STUB_RESPONSE,
)
async def get_capture_status() -> JSONResponse:
    """Get capture status."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.post(
    "/trigger",
    summary="Manual capture",
    description="Trigger a manual frame capture.",
    responses=STUB_RESPONSE,
)
async def manual_capture() -> JSONResponse:
    """Trigger manual capture."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.put(
    "/config",
    summary="Update capture config",
    description="Update capture settings (interval, threshold, etc.).",
    responses=STUB_RESPONSE,
)
async def update_capture_config() -> JSONResponse:
    """Update capture config."""
    return JSONResponse(status_code=501, content=STUB.model_dump())

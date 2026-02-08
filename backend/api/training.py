"""
Training API router — stub endpoints for model training management.

All endpoints return 501 until the Training subsystem is implemented.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.models.common import NotImplementedResponse

router = APIRouter()

STUB = NotImplementedResponse()
STUB_RESPONSE = {501: {"model": NotImplementedResponse}}


@router.post(
    "/start",
    summary="Start training",
    description="Start a training job with the specified configuration.",
    responses=STUB_RESPONSE,
)
async def start_training() -> JSONResponse:
    """Start a training job."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.post(
    "/stop",
    summary="Stop training",
    description="Stop or cancel the active training job.",
    responses=STUB_RESPONSE,
)
async def stop_training() -> JSONResponse:
    """Stop training."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.get(
    "/status",
    summary="Get training status",
    description="Get current training progress and state.",
    responses=STUB_RESPONSE,
)
async def get_training_status() -> JSONResponse:
    """Get training status."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.get(
    "/history",
    summary="Get training history",
    description="Get list of past training runs and their results.",
    responses=STUB_RESPONSE,
)
async def get_training_history() -> JSONResponse:
    """Get training history."""
    return JSONResponse(status_code=501, content=STUB.model_dump())

"""
System API router — status and configuration endpoints.

/status and /config are implemented; /resources is a stub.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.core.config import settings
from backend.models.common import NotImplementedResponse, StatusResponse

router = APIRouter()

STUB = NotImplementedResponse()
STUB_RESPONSE = {501: {"model": NotImplementedResponse}}


@router.get(
    "/status",
    summary="System status",
    description="Get overall system status including subsystem health.",
    response_model=StatusResponse,
)
async def get_system_status() -> StatusResponse:
    """Get overall system status."""
    return StatusResponse(status="ok")


@router.get(
    "/config",
    summary="Get configuration",
    description="Get current application configuration (non-sensitive fields).",
)
async def get_config() -> dict:
    """Get current application configuration."""
    return {
        "data_dir": settings.data_dir,
        "models_dir": settings.models_dir,
        "capture_interval": settings.capture_interval,
        "negative_ratio": settings.negative_ratio,
        "confidence_threshold": settings.confidence_threshold,
        "training_epochs": settings.training_epochs,
        "training_batch_size": settings.training_batch_size,
        "training_image_size": settings.training_image_size,
    }


@router.put(
    "/config",
    summary="Update configuration",
    description="Update application configuration at runtime.",
    responses=STUB_RESPONSE,
)
async def update_config() -> JSONResponse:
    """Update application configuration."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.get(
    "/resources",
    summary="Get resource usage",
    description="Get CPU, GPU, and memory usage statistics.",
    responses=STUB_RESPONSE,
)
async def get_resources() -> JSONResponse:
    """Get system resource usage."""
    return JSONResponse(status_code=501, content=STUB.model_dump())

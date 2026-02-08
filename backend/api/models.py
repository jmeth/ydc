"""
Models API router — stub endpoints for trained model management.

All endpoints return 501 until the model registry is implemented.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.models.common import NotImplementedResponse

router = APIRouter()

STUB = NotImplementedResponse()
STUB_RESPONSE = {501: {"model": NotImplementedResponse}}


@router.get(
    "",
    summary="List models",
    description="List all trained models.",
    responses=STUB_RESPONSE,
)
async def list_models() -> JSONResponse:
    """List trained models."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.get(
    "/{model_id}",
    summary="Get model details",
    description="Get details for a specific trained model.",
    responses=STUB_RESPONSE,
)
async def get_model(model_id: str) -> JSONResponse:
    """Get model details."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.delete(
    "/{model_id}",
    summary="Delete model",
    description="Delete a trained model.",
    responses=STUB_RESPONSE,
)
async def delete_model(model_id: str) -> JSONResponse:
    """Delete a model."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.put(
    "/{model_id}/activate",
    summary="Activate model",
    description="Set a model as the active inference model.",
    responses=STUB_RESPONSE,
)
async def activate_model(model_id: str) -> JSONResponse:
    """Activate a model."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.post(
    "/{model_id}/export",
    summary="Export model",
    description="Export a trained model for deployment.",
    responses=STUB_RESPONSE,
)
async def export_model(model_id: str) -> JSONResponse:
    """Export a model."""
    return JSONResponse(status_code=501, content=STUB.model_dump())

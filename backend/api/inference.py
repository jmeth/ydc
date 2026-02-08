"""
Inference API router — stub endpoints for inference management.

All endpoints return 501 until the Inference subsystem is implemented.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.models.common import NotImplementedResponse

router = APIRouter()

STUB = NotImplementedResponse()
STUB_RESPONSE = {501: {"model": NotImplementedResponse}}


@router.post(
    "/start",
    summary="Start inference",
    description="Start inference on a feed, creating an inference output feed.",
    responses=STUB_RESPONSE,
)
async def start_inference() -> JSONResponse:
    """Start inference on a feed."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.post(
    "/stop",
    summary="Stop inference",
    description="Stop the active inference session.",
    responses=STUB_RESPONSE,
)
async def stop_inference() -> JSONResponse:
    """Stop inference."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.get(
    "/status",
    summary="Get inference status",
    description="Get current inference state and statistics.",
    responses=STUB_RESPONSE,
)
async def get_inference_status() -> JSONResponse:
    """Get inference status."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.put(
    "/prompts",
    summary="Update prompts",
    description="Update YOLO-World detection prompts.",
    responses=STUB_RESPONSE,
)
async def update_prompts() -> JSONResponse:
    """Update YOLO-World prompts."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.put(
    "/model",
    summary="Switch model",
    description="Switch the active inference model.",
    responses=STUB_RESPONSE,
)
async def switch_model() -> JSONResponse:
    """Switch active model."""
    return JSONResponse(status_code=501, content=STUB.model_dump())

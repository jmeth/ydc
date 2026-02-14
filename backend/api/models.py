"""
Models API router — endpoints for managing trained YOLO models.

Provides REST endpoints to list, get, delete, activate, and export
trained models. Uses TrainingManager as DI dependency (wraps ModelStore).
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.core.exceptions import NotFoundError
from backend.models.common import ErrorResponse, NotImplementedResponse
from backend.models.training import (
    MessageResponse,
    ModelListResponse,
    ModelResponse,
)
from backend.training.manager import TrainingManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level DI — set from app lifespan
_training_manager: TrainingManager | None = None

STUB = NotImplementedResponse()


def set_training_manager(manager: TrainingManager | None) -> None:
    """
    Inject the TrainingManager instance for the models API router.

    Called from the app lifespan on startup (with instance) and
    teardown (with None).

    Args:
        manager: The TrainingManager instance, or None to reset.
    """
    global _training_manager
    _training_manager = manager


def _get_manager() -> TrainingManager:
    """
    Get the injected TrainingManager, raising 500 if not initialized.

    Returns:
        The active TrainingManager instance.
    """
    if _training_manager is None:
        raise RuntimeError("TrainingManager not initialized")
    return _training_manager


def _model_info_to_response(info) -> ModelResponse:
    """
    Convert a ModelInfo dataclass to a ModelResponse.

    Args:
        info: ModelInfo from persistence layer.

    Returns:
        ModelResponse with all fields populated.
    """
    return ModelResponse(
        name=info.name,
        base_model=info.base_model,
        dataset_name=info.dataset_name,
        created_at=info.created_at,
        epochs_completed=info.epochs_completed,
        best_map50=info.best_map50,
        is_active=info.is_active,
        path=str(info.path),
    )


@router.get(
    "",
    summary="List models",
    description="List all trained models in the registry.",
    response_model=ModelListResponse,
)
async def list_models() -> ModelListResponse:
    """List all trained models."""
    mgr = _get_manager()
    models = await mgr._model_store.list()
    responses = [_model_info_to_response(m) for m in models]
    return ModelListResponse(models=responses, count=len(responses))


@router.get(
    "/{model_id}",
    summary="Get model details",
    description="Get details for a specific trained model.",
    response_model=ModelResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
    },
)
async def get_model(model_id: str) -> ModelResponse:
    """Get a model's metadata by name."""
    mgr = _get_manager()
    info = await mgr._model_store.get(model_id)
    if info is None:
        raise NotFoundError("Model", model_id)
    return _model_info_to_response(info)


@router.delete(
    "/{model_id}",
    summary="Delete model",
    description="Delete a trained model and its files.",
    response_model=MessageResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
    },
)
async def delete_model(model_id: str) -> MessageResponse:
    """Delete a trained model."""
    mgr = _get_manager()
    deleted = await mgr._model_store.delete(model_id)
    if not deleted:
        raise NotFoundError("Model", model_id)
    logger.info("Deleted model '%s'", model_id)
    return MessageResponse(message=f"Model '{model_id}' deleted")


@router.put(
    "/{model_id}/activate",
    summary="Activate model",
    description="Set a model as the active inference model.",
    response_model=ModelResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
    },
)
async def activate_model(model_id: str) -> ModelResponse:
    """Set a model as active, clearing the flag on all others."""
    mgr = _get_manager()
    try:
        await mgr._model_store.set_active(model_id)
    except FileNotFoundError:
        raise NotFoundError("Model", model_id)

    info = await mgr._model_store.get(model_id)
    logger.info("Activated model '%s'", model_id)
    return _model_info_to_response(info)


@router.post(
    "/{model_id}/export",
    summary="Export model",
    description="Export a trained model for deployment.",
    responses={501: {"model": NotImplementedResponse}},
)
async def export_model(model_id: str) -> JSONResponse:
    """Export a model (not yet implemented)."""
    return JSONResponse(status_code=501, content=STUB.model_dump())

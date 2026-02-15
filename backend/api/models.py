"""
Models API router — endpoints for managing trained YOLO models.

Provides REST endpoints to list, get, delete, activate, export,
and download pretrained models. Uses TrainingManager as DI dependency
(wraps ModelStore).
"""

import asyncio
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from backend.core.exceptions import NotFoundError
from backend.models.common import ErrorResponse
from backend.models.training import (
    DownloadPretrainedRequest,
    MessageResponse,
    ModelListResponse,
    ModelResponse,
)
from backend.training.manager import TrainingManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level DI — set from app lifespan
_training_manager: TrainingManager | None = None


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
    "/pretrained",
    summary="Download pretrained model",
    description=(
        "Download a pretrained YOLO model from the Ultralytics hub and "
        "register it in the model store. The model_id should be a valid "
        "Ultralytics model identifier such as 'yolo11n.pt' or 'yolov8s.pt'."
    ),
    response_model=ModelResponse,
    responses={
        409: {"model": ErrorResponse, "description": "Model name already exists"},
        400: {"model": ErrorResponse, "description": "Invalid model identifier"},
    },
)
async def download_pretrained(request: DownloadPretrainedRequest) -> ModelResponse | JSONResponse:
    """
    Download a pretrained YOLO model and register it in the model store.

    Uses ultralytics.YOLO() to download the model weights, then copies
    them into managed storage via the ModelStore. The download runs in
    a thread pool executor to avoid blocking the event loop.

    Args:
        request: Contains model_id (e.g. 'yolo11n.pt') and optional name.

    Returns:
        ModelResponse for the newly registered model, or 409/400 error.
    """
    mgr = _get_manager()
    model_store = mgr._model_store

    # Derive display name from model_id if not provided
    name = request.name or request.model_id.removesuffix(".pt")

    # Check for duplicate name before downloading
    existing = await model_store.get(name)
    if existing is not None:
        return JSONResponse(
            status_code=409,
            content={"error": f"Model '{name}' already exists"},
        )

    # Download the model in a thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    try:
        weights_path = await loop.run_in_executor(
            None, _download_pretrained_model, request.model_id
        )
    except Exception as exc:
        logger.error("Failed to download pretrained model '%s': %s", request.model_id, exc)
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to download model '{request.model_id}': {exc}"},
        )

    # Register in the model store
    try:
        info = await model_store.save(
            name=name,
            weights_path=weights_path,
            base_model=request.model_id,
            dataset_name="pretrained",
            epochs_completed=0,
        )
    except ValueError as exc:
        # Race condition: another request registered the same name
        return JSONResponse(
            status_code=409,
            content={"error": str(exc)},
        )

    logger.info("Downloaded and registered pretrained model '%s' from '%s'", name, request.model_id)
    return _model_info_to_response(info)


def _download_pretrained_model(model_id: str) -> Path:
    """
    Download a pretrained YOLO model using ultralytics.

    Ultralytics caches downloaded models in its own cache directory.
    We load the model, then read ckpt_path to find the cached weights.
    The returned path remains valid because ultralytics persists its cache.

    Args:
        model_id: Ultralytics model identifier (e.g. 'yolo11n.pt').

    Returns:
        Path to the downloaded .pt weights file.

    Raises:
        Exception: If ultralytics fails to download or load the model.
    """
    from ultralytics import YOLO

    model = YOLO(model_id)

    # ckpt_path points to the actual weights file on disk
    ckpt_path = getattr(model, "ckpt_path", None)
    if ckpt_path:
        source = Path(str(ckpt_path))
        if source.exists():
            return source

    # Fallback: check if model_id was created in the cwd
    cwd_path = Path(model_id)
    if cwd_path.exists():
        return cwd_path

    raise FileNotFoundError(f"Could not locate downloaded weights for '{model_id}'")


@router.get(
    "/{model_id}/export",
    summary="Export model",
    description=(
        "Export a trained model as a downloadable zip archive containing "
        "the model weights, training config (if present), and metadata."
    ),
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"},
    },
)
async def export_model(model_id: str) -> FileResponse:
    """
    Export a model as a zip file download.

    Creates a zip archive containing best.pt, training_config.yaml
    (if present), and model_meta.json, then returns it as a download.

    Args:
        model_id: Model identifier.

    Returns:
        FileResponse with the zip archive.
    """
    mgr = _get_manager()
    model_store = mgr._model_store

    info = await model_store.get(model_id)
    if info is None:
        raise NotFoundError("Model", model_id)

    tmp_dir = Path(tempfile.mkdtemp())
    zip_path = await model_store.export_zip(model_id, tmp_dir)

    return FileResponse(
        path=str(zip_path),
        filename=f"{model_id}.zip",
        media_type="application/zip",
    )


@router.post(
    "/import",
    summary="Import model",
    description=(
        "Import a trained model from a zip archive. The zip must contain "
        "a best.pt weights file. A model_meta.json with metadata is optional."
    ),
    response_model=ModelResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid zip file"},
        409: {"model": ErrorResponse, "description": "Model name already exists"},
    },
)
async def import_model(file: UploadFile, name: str | None = None) -> ModelResponse | JSONResponse:
    """
    Import a model from an uploaded zip archive.

    Accepts a zip file, writes it to a temp location, delegates to
    ModelStore.import_zip(), then returns the registered model info.

    Args:
        file: Uploaded zip file.
        name: Optional override name for the imported model.

    Returns:
        ModelResponse for the newly imported model.
    """
    mgr = _get_manager()
    model_store = mgr._model_store

    # Write upload to a temp file
    tmp_dir = Path(tempfile.mkdtemp())
    zip_path = tmp_dir / (file.filename or "upload.zip")
    with open(zip_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        info = await model_store.import_zip(zip_path, name=name)
    except FileNotFoundError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": str(exc)},
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=409,
            content={"error": str(exc)},
        )

    logger.info("Imported model '%s' from zip", info.name)
    return _model_info_to_response(info)

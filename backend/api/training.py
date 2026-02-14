"""
Training API router — endpoints for managing YOLO model training jobs.

Provides REST endpoints to start/stop training, query job status, and
retrieve training history. Uses DI via set_training_manager() called
from app lifespan.
"""

import logging

from fastapi import APIRouter

from backend.core.config import settings
from backend.core.exceptions import ConflictError, NotFoundError
from backend.models.common import ErrorResponse
from backend.models.training import (
    MessageResponse,
    StartTrainingRequest,
    TrainingHistoryEntry,
    TrainingHistoryResponse,
    TrainingStatusResponse,
)
from backend.training.manager import TrainingConfig, TrainingManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level DI — set from app lifespan
_training_manager: TrainingManager | None = None


def set_training_manager(manager: TrainingManager | None) -> None:
    """
    Inject the TrainingManager instance for the API router.

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


def _job_to_response(job) -> TrainingStatusResponse:
    """
    Convert a TrainingJob to a TrainingStatusResponse.

    Args:
        job: TrainingJob instance.

    Returns:
        TrainingStatusResponse with all fields populated from the job.
    """
    progress = 0.0
    if job.total_epochs > 0:
        progress = round(job.current_epoch / job.total_epochs * 100, 1)

    return TrainingStatusResponse(
        job_id=job.job_id,
        status=job.status,
        current_epoch=job.current_epoch,
        total_epochs=job.total_epochs,
        progress_pct=progress,
        metrics=job.metrics,
        dataset_name=job.config.dataset_name,
        base_model=job.config.base_model,
        model_name=job.config.model_name,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
    )


@router.post(
    "/start",
    summary="Start training",
    description="Start a YOLO fine-tuning job on the specified dataset.",
    response_model=TrainingStatusResponse,
    status_code=201,
    responses={
        404: {"model": ErrorResponse, "description": "Dataset not found"},
        409: {"model": ErrorResponse, "description": "Training already running"},
        422: {"model": ErrorResponse, "description": "Validation error (e.g. model name taken)"},
    },
)
async def start_training(request: StartTrainingRequest) -> TrainingStatusResponse:
    """
    Start a training job with the given configuration.

    Uses server defaults from settings for any omitted optional fields.
    """
    mgr = _get_manager()

    config = TrainingConfig(
        dataset_name=request.dataset_name,
        base_model=request.base_model,
        epochs=request.epochs if request.epochs is not None else settings.training_epochs,
        batch_size=request.batch_size if request.batch_size is not None else settings.training_batch_size,
        image_size=request.image_size if request.image_size is not None else settings.training_image_size,
        patience=request.patience if request.patience is not None else settings.training_patience,
        freeze_layers=request.freeze_layers if request.freeze_layers is not None else settings.training_freeze_layers,
        lr0=request.lr0 if request.lr0 is not None else settings.training_lr0,
        lrf=request.lrf if request.lrf is not None else settings.training_lrf,
        model_name=request.model_name or "",
    )

    await mgr.start_training(config)

    job = mgr.get_status()
    return _job_to_response(job)


@router.post(
    "/stop",
    summary="Stop training",
    description="Cancel the currently running training job.",
    response_model=MessageResponse,
    responses={
        409: {"model": ErrorResponse, "description": "No training job running"},
    },
)
async def stop_training() -> MessageResponse:
    """Stop the active training job."""
    mgr = _get_manager()
    await mgr.stop_training()
    return MessageResponse(message="Training stop requested")


@router.get(
    "/status",
    summary="Get training status",
    description="Get the current training job status and progress.",
    response_model=TrainingStatusResponse | None,
)
async def get_training_status() -> TrainingStatusResponse | dict:
    """Get current training job status, or null if no job has been run."""
    mgr = _get_manager()
    job = mgr.get_status()

    if job is None:
        return TrainingStatusResponse(
            job_id="",
            status="idle",
        )

    return _job_to_response(job)


@router.get(
    "/history",
    summary="Get training history",
    description="Get list of past training runs and their results.",
    response_model=TrainingHistoryResponse,
)
async def get_training_history() -> TrainingHistoryResponse:
    """Get training history, newest first."""
    mgr = _get_manager()
    jobs = mgr.get_history()

    entries = []
    for job in jobs:
        entries.append(TrainingHistoryEntry(
            job_id=job.job_id,
            model_name=job.config.model_name,
            dataset_name=job.config.dataset_name,
            base_model=job.config.base_model,
            status=job.status,
            epochs_completed=job.current_epoch,
            best_map50=job.metrics.get("best_map50") or job.metrics.get("metrics/mAP50(B)"),
            started_at=job.started_at,
            completed_at=job.completed_at,
        ))

    return TrainingHistoryResponse(jobs=entries, count=len(entries))

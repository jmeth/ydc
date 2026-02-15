"""
Training manager — business logic for YOLO model fine-tuning.

Manages training job lifecycle: start, stop, progress tracking, and
model saving. Runs ultralytics training in a ThreadPoolExecutor to
avoid blocking the async event loop.

Usage:
    manager = TrainingManager(model_store, dataset_store, event_bus)
    job_id = await manager.start_training(config)
    status = manager.get_status()
"""

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.core.config import settings
from backend.core.events import (
    TRAINING_COMPLETED,
    TRAINING_ERROR,
    TRAINING_PROGRESS,
    TRAINING_STARTED,
    EventBus,
)
from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from backend.persistence.dataset_store import DatasetStore
from backend.persistence.model_store import ModelStore

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Training configuration derived from an API request plus server defaults.

    Attributes:
        dataset_name: Dataset to train on.
        base_model: Base YOLO model file (e.g. "yolo11n.pt").
        epochs: Number of training epochs.
        batch_size: Training batch size.
        image_size: Training image size in pixels.
        patience: Early stopping patience.
        freeze_layers: Number of backbone layers to freeze.
        lr0: Initial learning rate.
        lrf: Final learning rate factor.
        model_name: Name for the saved model.
        augmentation: Dict of augmentation param overrides to pass
            to ultralytics model.train(). Keys are ultralytics arg names
            (e.g. hsv_h, mosaic, fliplr). Empty dict means use defaults.
    """

    dataset_name: str
    base_model: str = "yolo11n.pt"
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    patience: int = 20
    freeze_layers: int = 10
    lr0: float = 0.001
    lrf: float = 0.01
    model_name: str = ""
    augmentation: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingJob:
    """
    Internal state for a single training job.

    Attributes:
        job_id: Unique identifier.
        config: Training parameters.
        status: Current state (training, completed, error, cancelled).
        current_epoch: Most recently completed epoch number.
        total_epochs: Target epoch count.
        metrics: Latest training metrics from ultralytics.
        started_at: Unix timestamp when training started.
        completed_at: Unix timestamp when training finished.
        error: Error message if status is 'error'.
    """

    job_id: str
    config: TrainingConfig
    status: str = "training"
    current_epoch: int = 0
    total_epochs: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None


class TrainingManager:
    """
    Manages YOLO training job lifecycle.

    Runs at most one training job at a time. Training runs in a background
    thread via ThreadPoolExecutor. Progress is reported through the EventBus.

    Args:
        model_store: Persistence store for saving trained models.
        dataset_store: Persistence store for validating datasets exist.
        event_bus: Optional EventBus for publishing training events.
    """

    def __init__(
        self,
        model_store: ModelStore,
        dataset_store: DatasetStore,
        event_bus: EventBus | None = None,
    ):
        self._model_store = model_store
        self._dataset_store = dataset_store
        self._event_bus = event_bus
        self._current_job: TrainingJob | None = None
        self._history: list[TrainingJob] = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training")
        self._cancel_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start_training(self, config: TrainingConfig) -> str:
        """
        Start a new training job.

        Validates that the dataset exists and no other job is running,
        then launches training in a background thread.

        Args:
            config: Training configuration.

        Returns:
            The new job's unique ID.

        Raises:
            NotFoundError: If the specified dataset doesn't exist.
            ConflictError: If a training job is already running.
            ValidationError: If the model name already exists in the store.
        """
        # Validate dataset exists
        ds_info = await self._dataset_store.get(config.dataset_name)
        if ds_info is None:
            raise NotFoundError("Dataset", config.dataset_name)

        # Reject if already training
        if self._current_job is not None and self._current_job.status == "training":
            raise ConflictError("A training job is already running")

        # Check model name doesn't already exist
        if config.model_name:
            existing = await self._model_store.get(config.model_name)
            if existing is not None:
                raise ValidationError(
                    f"Model '{config.model_name}' already exists",
                    detail="Choose a different model name or delete the existing model.",
                )

        # Auto-generate model name if not provided
        if not config.model_name:
            config.model_name = f"{config.dataset_name}-{config.base_model.replace('.pt', '')}-{int(time.time())}"

        # Create job
        job = TrainingJob(
            job_id=str(uuid.uuid4()),
            config=config,
            status="training",
            total_epochs=config.epochs,
            started_at=time.time(),
        )
        self._current_job = job
        self._cancel_event.clear()

        # Capture the event loop for cross-thread publishing
        self._loop = asyncio.get_running_loop()

        # Resolve data.yaml path from the dataset info
        data_yaml_path = ds_info.path / "data.yaml"

        await self._publish(TRAINING_STARTED, {
            "job_id": job.job_id,
            "dataset_name": config.dataset_name,
            "base_model": config.base_model,
            "model_name": config.model_name,
            "epochs": config.epochs,
            "message": f"Training started on dataset '{config.dataset_name}' with {config.base_model}",
        })

        # Launch training in background thread
        self._executor.submit(self._run_training, job, data_yaml_path)

        logger.info(
            "Started training job %s: dataset=%s, model=%s, epochs=%d",
            job.job_id, config.dataset_name, config.base_model, config.epochs,
        )
        return job.job_id

    async def stop_training(self) -> None:
        """
        Cancel the currently running training job.

        Sets the cancel event to signal the training thread. The thread
        will finish the current epoch and then clean up.

        Raises:
            ConflictError: If no training job is currently running.
        """
        if self._current_job is None or self._current_job.status != "training":
            raise ConflictError("No training job is currently running")

        self._cancel_event.set()
        logger.info("Cancel requested for job %s", self._current_job.job_id)

    def get_status(self) -> TrainingJob | None:
        """
        Get the current or most recently completed training job.

        Returns:
            The current TrainingJob, or None if no job has been run.
        """
        return self._current_job

    def get_history(self) -> list[TrainingJob]:
        """
        Get the history of completed/failed/cancelled training jobs.

        Returns:
            List of past TrainingJob records, newest first.
        """
        return list(reversed(self._history))

    def _run_training(self, job: TrainingJob, data_yaml_path: Path) -> None:
        """
        Execute ultralytics YOLO training in a background thread.

        Called by the ThreadPoolExecutor. Uses ultralytics callbacks for
        progress tracking and publishes events via run_coroutine_threadsafe.

        Args:
            job: The TrainingJob to execute.
            data_yaml_path: Path to the dataset's data.yaml file.
        """
        try:
            from ultralytics import YOLO

            model = YOLO(job.config.base_model)

            # Register epoch-end callback for progress updates
            def on_train_epoch_end(trainer):
                """Callback invoked after each training epoch."""
                epoch = trainer.epoch + 1  # ultralytics uses 0-indexed
                job.current_epoch = epoch

                # Extract available metrics
                metrics = {}
                if hasattr(trainer, "metrics") and trainer.metrics:
                    metrics = {k: float(v) for k, v in trainer.metrics.items()
                               if isinstance(v, (int, float))}
                if hasattr(trainer, "loss") and trainer.loss is not None:
                    try:
                        metrics["loss"] = float(trainer.loss.mean().item())
                    except (AttributeError, TypeError):
                        pass

                job.metrics = metrics

                # Publish progress event (throttle to every 5 epochs for large runs)
                if epoch % max(1, job.total_epochs // 20) == 0 or epoch == job.total_epochs:
                    self._publish_from_thread(TRAINING_PROGRESS, {
                        "job_id": job.job_id,
                        "current_epoch": epoch,
                        "total_epochs": job.total_epochs,
                        "progress_pct": round(epoch / job.total_epochs * 100, 1),
                        "metrics": metrics,
                        "message": f"Epoch {epoch}/{job.total_epochs}",
                    })

                # Check for cancellation
                if self._cancel_event.is_set():
                    raise KeyboardInterrupt("Training cancelled by user")

            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            # Build training kwargs, including any augmentation overrides
            train_kwargs = {
                "data": str(data_yaml_path),
                "epochs": job.config.epochs,
                "imgsz": job.config.image_size,
                "batch": job.config.batch_size,
                "patience": job.config.patience,
                "freeze": job.config.freeze_layers,
                "lr0": job.config.lr0,
                "lrf": job.config.lrf,
                "verbose": False,
            }
            if job.config.augmentation:
                train_kwargs.update(job.config.augmentation)

            # Run training
            results = model.train(**train_kwargs)

            # Training completed — save model
            job.status = "completed"
            job.completed_at = time.time()
            job.current_epoch = job.total_epochs

            # Extract best mAP50 from results
            best_map50 = None
            if results and hasattr(results, "results_dict"):
                best_map50 = results.results_dict.get("metrics/mAP50(B)")
            elif results and hasattr(results, "maps"):
                try:
                    best_map50 = float(results.maps[0]) if results.maps else None
                except (IndexError, TypeError):
                    pass

            # Find best weights path
            best_weights = Path(model.trainer.best) if hasattr(model, "trainer") and model.trainer else None

            if best_weights and best_weights.exists():
                # Save model to store (sync call from thread is fine — FilesystemModelStore is sync internally)
                future = asyncio.run_coroutine_threadsafe(
                    self._model_store.save(
                        name=job.config.model_name,
                        weights_path=best_weights,
                        base_model=job.config.base_model,
                        dataset_name=job.config.dataset_name,
                        epochs_completed=job.current_epoch,
                        metrics={"best_map50": best_map50} if best_map50 is not None else None,
                    ),
                    self._loop,
                )
                model_info = future.result(timeout=30)

                # Save training config YAML alongside the model for reference/reuse
                self._save_training_config(job, model_info.path.parent, best_map50)

            self._publish_from_thread(TRAINING_COMPLETED, {
                "job_id": job.job_id,
                "model_name": job.config.model_name,
                "dataset_name": job.config.dataset_name,
                "epochs_completed": job.current_epoch,
                "best_map50": best_map50,
                "message": f"Training completed: {job.config.model_name} "
                           f"({job.current_epoch} epochs, mAP50={best_map50})",
            })

            logger.info(
                "Training job %s completed: model=%s, epochs=%d, mAP50=%s",
                job.job_id, job.config.model_name, job.current_epoch, best_map50,
            )

        except KeyboardInterrupt:
            # Cancelled by user
            job.status = "cancelled"
            job.completed_at = time.time()
            logger.info("Training job %s cancelled at epoch %d", job.job_id, job.current_epoch)

        except Exception as e:
            job.status = "error"
            job.error = str(e)
            job.completed_at = time.time()

            self._publish_from_thread(TRAINING_ERROR, {
                "job_id": job.job_id,
                "error": str(e),
                "message": f"Training failed: {e}",
            })
            logger.exception("Training job %s failed: %s", job.job_id, e)

        finally:
            # Move job to history
            self._history.append(job)
            # Trim history
            max_history = settings.training_max_jobs_history
            if len(self._history) > max_history:
                self._history = self._history[-max_history:]

    def _publish_from_thread(self, event_type: str, data: dict) -> None:
        """
        Publish an EventBus event from a background thread.

        Uses asyncio.run_coroutine_threadsafe to bridge from the training
        thread to the async event loop.

        Args:
            event_type: Event type constant.
            data: Event payload.
        """
        if self._event_bus is not None and self._loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._event_bus.publish(event_type, data),
                    self._loop,
                )
                future.result(timeout=5)
            except Exception:
                logger.debug("Failed to publish %s from thread", event_type, exc_info=True)

    async def _publish(self, event_type: str, data: dict) -> None:
        """
        Publish an event via the EventBus if available (async context).

        Args:
            event_type: Event type constant.
            data: Event payload.
        """
        if self._event_bus is not None:
            await self._event_bus.publish(event_type, data)

    def _save_training_config(
        self,
        job: TrainingJob,
        model_dir: Path,
        best_map50: float | None,
    ) -> None:
        """
        Save training configuration as YAML in the model directory.

        Writes a training_config.yaml file alongside the model weights
        containing all training parameters and augmentation settings
        in a format compatible with ultralytics YOLO training args.

        Args:
            job: The completed TrainingJob with config to save.
            model_dir: Directory where the model weights are stored.
            best_map50: Best mAP@50 achieved during training, if available.
        """
        import yaml

        cfg = job.config
        config_dict: dict[str, Any] = {
            # Model & dataset
            "model": cfg.base_model,
            "data": cfg.dataset_name,
            "model_name": cfg.model_name,
            # Training hyperparameters
            "epochs": cfg.epochs,
            "batch": cfg.batch_size,
            "imgsz": cfg.image_size,
            "patience": cfg.patience,
            "freeze": cfg.freeze_layers,
            "lr0": cfg.lr0,
            "lrf": cfg.lrf,
        }

        # Add augmentation overrides (only those explicitly set)
        if cfg.augmentation:
            config_dict["augmentation"] = dict(cfg.augmentation)

        # Add results metadata
        if best_map50 is not None:
            config_dict["best_map50"] = best_map50
        config_dict["epochs_completed"] = job.current_epoch

        config_path = model_dir / "training_config.yaml"
        try:
            with open(config_path, "w") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
            logger.info("Saved training config to %s", config_path)
        except Exception:
            logger.warning("Failed to save training config to %s", config_path, exc_info=True)

    async def shutdown(self) -> None:
        """
        Shut down the training manager.

        Cancels any running job and cleans up the thread pool executor.
        """
        if self._current_job is not None and self._current_job.status == "training":
            self._cancel_event.set()
            logger.info("Cancelling training job %s on shutdown", self._current_job.job_id)

        self._executor.shutdown(wait=False)
        logger.info("TrainingManager shut down")

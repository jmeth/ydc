"""
Unit tests for the TrainingManager.

Tests cover job lifecycle (start, stop, status, history), dataset
validation, model name conflicts, cancellation, error handling,
and progress event publishing. All external dependencies
(ModelStore, DatasetStore, EventBus, ultralytics) are mocked.
"""

import asyncio
import time
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.events import (
    TRAINING_COMPLETED,
    TRAINING_ERROR,
    TRAINING_PROGRESS,
    TRAINING_STARTED,
)
from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from backend.persistence.models import DatasetInfo, ModelInfo
from backend.training.manager import TrainingConfig, TrainingJob, TrainingManager


# --- Helpers ---

def _make_model_store():
    """Create a mock ModelStore with standard success returns."""
    ms = MagicMock()
    ms.get = AsyncMock(return_value=None)
    ms.save = AsyncMock(return_value=ModelInfo(
        name="test-model", path=Path("/tmp/best.pt"),
        base_model="yolo11n.pt", dataset_name="test-ds",
        epochs_completed=10, best_map50=0.85,
    ))
    ms.list = AsyncMock(return_value=[])
    ms.delete = AsyncMock(return_value=True)
    ms.set_active = AsyncMock()
    ms.get_active = AsyncMock(return_value=None)
    return ms


def _make_dataset_store(exists: bool = True):
    """Create a mock DatasetStore with configurable dataset existence."""
    ds = MagicMock()
    if exists:
        ds.get = AsyncMock(return_value=DatasetInfo(
            name="test-ds", path=Path("/tmp/datasets/test-ds"),
            classes=["cat", "dog"],
        ))
    else:
        ds.get = AsyncMock(return_value=None)
    return ds


def _make_event_bus():
    """Create a mock EventBus."""
    eb = MagicMock()
    eb.publish = AsyncMock()
    return eb


def _make_config(**overrides):
    """Create a TrainingConfig with sensible test defaults."""
    defaults = {
        "dataset_name": "test-ds",
        "base_model": "yolo11n.pt",
        "epochs": 5,
        "batch_size": 4,
        "image_size": 320,
        "patience": 3,
        "freeze_layers": 0,
        "lr0": 0.001,
        "lrf": 0.01,
        "model_name": "test-model",
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# --- Init Tests ---

class TestTrainingManagerInit:
    """Tests for TrainingManager initialization."""

    async def test_init_with_all_deps(self):
        """Manager initializes with all dependencies provided."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=_make_event_bus(),
        )
        assert mgr.get_status() is None
        assert mgr.get_history() == []

    async def test_init_without_event_bus(self):
        """Manager initializes without event bus (optional)."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=None,
        )
        assert mgr._event_bus is None


# --- Start Training Tests ---

class TestStartTraining:
    """Tests for TrainingManager.start_training()."""

    async def test_start_success(self):
        """Successfully start a training job, returns job ID."""
        eb = _make_event_bus()
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=eb,
        )

        config = _make_config()
        job_id = await mgr.start_training(config)

        assert isinstance(job_id, str)
        assert len(job_id) > 0

        # Job should be tracked
        status = mgr.get_status()
        assert status is not None
        assert status.job_id == job_id
        assert status.status == "training"
        assert status.config.dataset_name == "test-ds"

        # TRAINING_STARTED event should be published
        eb.publish.assert_called()
        call_args = eb.publish.call_args_list[0]
        assert call_args[0][0] == TRAINING_STARTED

        await mgr.shutdown()

    async def test_start_missing_dataset(self):
        """Raise NotFoundError if dataset doesn't exist."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(exists=False),
            event_bus=_make_event_bus(),
        )

        config = _make_config()
        with pytest.raises(NotFoundError):
            await mgr.start_training(config)

        await mgr.shutdown()

    async def test_start_already_training(self):
        """Raise ConflictError if a job is already running."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=_make_event_bus(),
        )

        config = _make_config()
        await mgr.start_training(config)

        with pytest.raises(ConflictError):
            await mgr.start_training(_make_config(model_name="other-model"))

        await mgr.shutdown()

    async def test_start_model_name_conflict(self):
        """Raise ValidationError if model name already exists in store."""
        ms = _make_model_store()
        ms.get = AsyncMock(return_value=ModelInfo(
            name="existing-model", path=Path("/tmp/best.pt"),
        ))

        mgr = TrainingManager(
            model_store=ms,
            dataset_store=_make_dataset_store(),
            event_bus=_make_event_bus(),
        )

        config = _make_config(model_name="existing-model")
        with pytest.raises(ValidationError):
            await mgr.start_training(config)

        await mgr.shutdown()

    async def test_start_auto_generates_model_name(self):
        """Auto-generate model name when not provided."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=_make_event_bus(),
        )

        config = _make_config(model_name="")
        await mgr.start_training(config)

        status = mgr.get_status()
        assert status.config.model_name != ""
        assert "test-ds" in status.config.model_name
        assert "yolo11n" in status.config.model_name

        await mgr.shutdown()


# --- Stop Training Tests ---

class TestStopTraining:
    """Tests for TrainingManager.stop_training()."""

    async def test_stop_running_job(self):
        """Stop sets cancel event on a running job."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=_make_event_bus(),
        )

        await mgr.start_training(_make_config())
        await mgr.stop_training()

        assert mgr._cancel_event.is_set()

        await mgr.shutdown()

    async def test_stop_no_running_job(self):
        """Raise ConflictError if no job is running."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=_make_event_bus(),
        )

        with pytest.raises(ConflictError):
            await mgr.stop_training()

        await mgr.shutdown()


# --- Status Tests ---

class TestGetStatus:
    """Tests for TrainingManager.get_status()."""

    async def test_no_job(self):
        """Return None when no job has been started."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
        )
        assert mgr.get_status() is None
        await mgr.shutdown()

    async def test_active_job(self):
        """Return the current job when one is running."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=_make_event_bus(),
        )

        await mgr.start_training(_make_config())
        status = mgr.get_status()

        assert status is not None
        assert status.status == "training"
        assert status.total_epochs == 5
        assert status.started_at is not None

        await mgr.shutdown()


# --- History Tests ---

class TestGetHistory:
    """Tests for TrainingManager.get_history()."""

    async def test_empty_history(self):
        """Return empty list when no jobs have completed."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
        )
        assert mgr.get_history() == []
        await mgr.shutdown()

    async def test_history_after_job(self):
        """History should contain jobs after they complete."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=_make_event_bus(),
        )

        # Manually add a finished job to history
        job = TrainingJob(
            job_id="test-123",
            config=_make_config(),
            status="completed",
            current_epoch=5,
            total_epochs=5,
            completed_at=time.time(),
        )
        mgr._history.append(job)

        history = mgr.get_history()
        assert len(history) == 1
        assert history[0].job_id == "test-123"
        assert history[0].status == "completed"

        await mgr.shutdown()

    async def test_history_newest_first(self):
        """History returns newest job first."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
        )

        job1 = TrainingJob(job_id="old", config=_make_config(), status="completed")
        job2 = TrainingJob(job_id="new", config=_make_config(), status="completed")
        mgr._history.extend([job1, job2])

        history = mgr.get_history()
        assert history[0].job_id == "new"
        assert history[1].job_id == "old"

        await mgr.shutdown()


# --- Training Execution Tests ---

class TestRunTraining:
    """Tests for the _run_training background thread logic."""

    async def test_successful_training(self):
        """Training completes successfully, saves model, publishes events."""
        ms = _make_model_store()
        eb = _make_event_bus()

        mgr = TrainingManager(
            model_store=ms,
            dataset_store=_make_dataset_store(),
            event_bus=eb,
        )

        # Mock ultralytics YOLO
        with patch("ultralytics.YOLO") as mock_yolo_cls:
            mock_model = MagicMock()
            mock_yolo_cls.return_value = mock_model

            # Configure mock trainer
            mock_trainer = MagicMock()
            mock_trainer.best = "/tmp/runs/best.pt"
            mock_model.trainer = mock_trainer

            # Configure mock results
            mock_results = MagicMock()
            mock_results.results_dict = {"metrics/mAP50(B)": 0.85}
            mock_model.train.return_value = mock_results

            config = _make_config()
            job = TrainingJob(
                job_id="test-job",
                config=config,
                status="training",
                total_epochs=5,
                started_at=time.time(),
            )

            # Run in a separate thread so run_coroutine_threadsafe works
            loop = asyncio.get_running_loop()
            mgr._loop = loop

            # Patch Path.exists to return True for the best weights
            original_exists = Path.exists
            def patched_exists(self):
                if "best.pt" in str(self):
                    return True
                return original_exists(self)

            with patch.object(Path, "exists", patched_exists):
                await loop.run_in_executor(
                    None, mgr._run_training, job, Path("/tmp/datasets/test-ds/data.yaml")
                )

            # Verify model.train was called with correct params
            mock_model.train.assert_called_once()
            train_kwargs = mock_model.train.call_args[1]
            assert train_kwargs["epochs"] == 5
            assert train_kwargs["batch"] == 4
            assert train_kwargs["imgsz"] == 320

            # Job should be completed
            assert job.status == "completed"
            assert job.completed_at is not None

            # Model should be saved
            ms.save.assert_called_once()
            save_kwargs = ms.save.call_args[1]
            assert save_kwargs["name"] == "test-model"
            assert save_kwargs["base_model"] == "yolo11n.pt"
            assert save_kwargs["dataset_name"] == "test-ds"

            # Job should be in history
            assert len(mgr._history) == 1

        await mgr.shutdown()

    async def test_training_error(self):
        """Training error is captured and published."""
        eb = _make_event_bus()
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=eb,
        )

        with patch("ultralytics.YOLO") as mock_yolo_cls:
            mock_model = MagicMock()
            mock_yolo_cls.return_value = mock_model
            mock_model.train.side_effect = RuntimeError("GPU OOM")

            config = _make_config()
            job = TrainingJob(
                job_id="test-job",
                config=config,
                status="training",
                total_epochs=5,
                started_at=time.time(),
            )

            loop = asyncio.get_running_loop()
            mgr._loop = loop
            await loop.run_in_executor(
                None, mgr._run_training, job, Path("/tmp/data.yaml")
            )

            assert job.status == "error"
            assert "GPU OOM" in job.error
            assert job.completed_at is not None
            assert len(mgr._history) == 1

        await mgr.shutdown()

    async def test_training_cancellation(self):
        """Training cancellation via cancel event."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=_make_event_bus(),
        )

        with patch("ultralytics.YOLO") as mock_yolo_cls:
            mock_model = MagicMock()
            mock_yolo_cls.return_value = mock_model

            # Make train raise KeyboardInterrupt (simulating callback cancellation)
            mock_model.train.side_effect = KeyboardInterrupt("Training cancelled by user")

            config = _make_config()
            job = TrainingJob(
                job_id="test-job",
                config=config,
                status="training",
                total_epochs=5,
                started_at=time.time(),
            )

            loop = asyncio.get_running_loop()
            mgr._loop = loop
            await loop.run_in_executor(
                None, mgr._run_training, job, Path("/tmp/data.yaml")
            )

            assert job.status == "cancelled"
            assert job.completed_at is not None
            assert len(mgr._history) == 1

        await mgr.shutdown()

    async def test_epoch_callback_updates_job(self):
        """The on_train_epoch_end callback updates job progress."""
        eb = _make_event_bus()
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=eb,
        )

        with patch("ultralytics.YOLO") as mock_yolo_cls:
            mock_model = MagicMock()
            mock_yolo_cls.return_value = mock_model

            # Capture the callback registered via add_callback
            captured_callbacks = {}
            def capture_callback(event_name, fn):
                captured_callbacks[event_name] = fn
            mock_model.add_callback.side_effect = capture_callback

            # Make train call the captured callback during "training"
            mock_trainer = MagicMock()
            mock_trainer.best = "/tmp/best.pt"
            mock_trainer.metrics = {"metrics/mAP50(B)": 0.5}
            mock_trainer.loss = None
            mock_model.trainer = mock_trainer

            def fake_train(**kwargs):
                """Simulate training by invoking the epoch callback."""
                cb = captured_callbacks.get("on_train_epoch_end")
                if cb:
                    # Simulate epoch 3 (0-indexed) completing
                    mock_trainer.epoch = 2
                    cb(mock_trainer)
                    # Simulate epoch 5 (0-indexed=4) completing
                    mock_trainer.epoch = 4
                    cb(mock_trainer)
                mock_results = MagicMock()
                mock_results.results_dict = {"metrics/mAP50(B)": 0.85}
                return mock_results

            mock_model.train.side_effect = fake_train

            config = _make_config(epochs=10)
            job = TrainingJob(
                job_id="test-job",
                config=config,
                status="training",
                total_epochs=10,
                started_at=time.time(),
            )

            loop = asyncio.get_running_loop()
            mgr._loop = loop

            with patch.object(Path, "exists", return_value=True):
                await loop.run_in_executor(
                    None, mgr._run_training, job, Path("/tmp/data.yaml")
                )

            # Verify add_callback was called
            mock_model.add_callback.assert_called_once_with(
                "on_train_epoch_end", captured_callbacks["on_train_epoch_end"]
            )

            # Job epoch should have been updated by the callback
            # Last callback was epoch 4 (0-indexed) = epoch 5 (1-indexed)
            # But then training completes so current_epoch is set to total_epochs
            assert job.current_epoch == 10  # set to total_epochs on completion
            assert job.status == "completed"

        await mgr.shutdown()


# --- Shutdown Tests ---

class TestShutdown:
    """Tests for TrainingManager.shutdown()."""

    async def test_shutdown_no_active_job(self):
        """Shutdown completes cleanly with no active job."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
        )
        await mgr.shutdown()

    async def test_shutdown_cancels_active_job(self):
        """Shutdown sets cancel event for running jobs."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=_make_event_bus(),
        )

        await mgr.start_training(_make_config())
        await mgr.shutdown()

        assert mgr._cancel_event.is_set()


# --- Publish Helper Tests ---

class TestPublish:
    """Tests for async and thread-safe event publishing."""

    async def test_publish_with_event_bus(self):
        """_publish calls event_bus.publish when available."""
        eb = _make_event_bus()
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=eb,
        )

        await mgr._publish("test.event", {"key": "value"})
        eb.publish.assert_called_once_with("test.event", {"key": "value"})

        await mgr.shutdown()

    async def test_publish_without_event_bus(self):
        """_publish is a no-op when event_bus is None."""
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=None,
        )

        # Should not raise
        await mgr._publish("test.event", {"key": "value"})

        await mgr.shutdown()

    async def test_publish_from_thread(self):
        """_publish_from_thread bridges to the async event loop."""
        eb = _make_event_bus()
        mgr = TrainingManager(
            model_store=_make_model_store(),
            dataset_store=_make_dataset_store(),
            event_bus=eb,
        )

        mgr._loop = asyncio.get_running_loop()

        # Call from current thread (acts like a worker thread)
        mgr._publish_from_thread("test.event", {"key": "value"})

        # The publish should have been called
        eb.publish.assert_called_with("test.event", {"key": "value"})

        await mgr.shutdown()


# --- TrainingConfig Tests ---

class TestTrainingConfig:
    """Tests for TrainingConfig dataclass defaults."""

    def test_defaults(self):
        """Config has sensible defaults for optional fields."""
        config = TrainingConfig(dataset_name="test")
        assert config.base_model == "yolo11n.pt"
        assert config.epochs == 100
        assert config.batch_size == 16
        assert config.image_size == 640
        assert config.model_name == ""

    def test_custom_values(self):
        """Config accepts custom overrides."""
        config = TrainingConfig(
            dataset_name="test",
            base_model="yolov8s.pt",
            epochs=50,
            batch_size=8,
        )
        assert config.base_model == "yolov8s.pt"
        assert config.epochs == 50
        assert config.batch_size == 8


# --- TrainingJob Tests ---

class TestTrainingJob:
    """Tests for TrainingJob dataclass."""

    def test_default_status(self):
        """Job starts with 'training' status."""
        job = TrainingJob(
            job_id="test-123",
            config=_make_config(),
        )
        assert job.status == "training"
        assert job.current_epoch == 0
        assert job.metrics == {}
        assert job.error is None

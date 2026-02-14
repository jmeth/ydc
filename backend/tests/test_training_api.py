"""
API integration tests for training endpoints.

Tests exercise the REST API through the ASGI test client with a
mock TrainingManager to isolate API logic from training execution.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.training import set_training_manager
from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from backend.training.manager import TrainingConfig, TrainingJob, TrainingManager


# --- Fixtures ---

@pytest.fixture
def training_manager():
    """
    Fresh mock TrainingManager injected into the training API router.

    Provides a fully mocked TrainingManager with standard success
    returns. Cleans up by resetting the module-level reference.
    """
    mgr = MagicMock(spec=TrainingManager)
    mgr.start_training = AsyncMock(return_value="job-123")
    mgr.stop_training = AsyncMock()
    mgr.get_status.return_value = None
    mgr.get_history.return_value = []

    set_training_manager(mgr)
    yield mgr
    set_training_manager(None)


def _make_job(
    job_id: str = "job-123",
    status: str = "training",
    current_epoch: int = 5,
    total_epochs: int = 100,
    dataset_name: str = "test-ds",
    base_model: str = "yolo11n.pt",
    model_name: str = "test-model",
) -> TrainingJob:
    """Create a TrainingJob with sensible defaults for tests."""
    return TrainingJob(
        job_id=job_id,
        config=TrainingConfig(
            dataset_name=dataset_name,
            base_model=base_model,
            epochs=total_epochs,
            model_name=model_name,
        ),
        status=status,
        current_epoch=current_epoch,
        total_epochs=total_epochs,
        started_at=time.time(),
    )


# --- POST /api/training/start ---

class TestStartTraining:
    """Tests for POST /api/training/start."""

    async def test_start_success(self, client, training_manager):
        """Start training returns 201 with job status."""
        job = _make_job()
        training_manager.get_status.return_value = job

        resp = await client.post(
            "/api/training/start",
            json={"dataset_name": "test-ds", "model_name": "test-model"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["job_id"] == "job-123"
        assert data["status"] == "training"
        assert data["dataset_name"] == "test-ds"
        assert data["model_name"] == "test-model"

    async def test_start_with_all_params(self, client, training_manager):
        """Start with all optional params passes them through."""
        job = _make_job()
        training_manager.get_status.return_value = job

        resp = await client.post(
            "/api/training/start",
            json={
                "dataset_name": "test-ds",
                "base_model": "yolov8s.pt",
                "epochs": 50,
                "batch_size": 8,
                "image_size": 320,
                "patience": 10,
                "freeze_layers": 5,
                "lr0": 0.01,
                "lrf": 0.1,
                "model_name": "custom-model",
            },
        )
        assert resp.status_code == 201

        # Verify config was passed correctly
        call_args = training_manager.start_training.call_args[0][0]
        assert call_args.base_model == "yolov8s.pt"
        assert call_args.epochs == 50
        assert call_args.batch_size == 8
        assert call_args.image_size == 320
        assert call_args.patience == 10
        assert call_args.freeze_layers == 5

    async def test_start_missing_dataset_name(self, client, training_manager):
        """Missing required dataset_name returns 422."""
        resp = await client.post(
            "/api/training/start",
            json={},
        )
        assert resp.status_code == 422

    async def test_start_dataset_not_found(self, client, training_manager):
        """Return 404 when dataset doesn't exist."""
        training_manager.start_training = AsyncMock(
            side_effect=NotFoundError("Dataset", "nonexistent")
        )

        resp = await client.post(
            "/api/training/start",
            json={"dataset_name": "nonexistent"},
        )
        assert resp.status_code == 404

    async def test_start_already_training(self, client, training_manager):
        """Return 409 when a training job is already running."""
        training_manager.start_training = AsyncMock(
            side_effect=ConflictError("A training job is already running")
        )

        resp = await client.post(
            "/api/training/start",
            json={"dataset_name": "test-ds"},
        )
        assert resp.status_code == 409

    async def test_start_model_name_conflict(self, client, training_manager):
        """Return 422 when model name already exists."""
        training_manager.start_training = AsyncMock(
            side_effect=ValidationError("Model 'existing' already exists")
        )

        resp = await client.post(
            "/api/training/start",
            json={"dataset_name": "test-ds", "model_name": "existing"},
        )
        assert resp.status_code == 422

    async def test_start_invalid_epochs(self, client, training_manager):
        """Return 422 for invalid epoch count."""
        resp = await client.post(
            "/api/training/start",
            json={"dataset_name": "test-ds", "epochs": 0},
        )
        assert resp.status_code == 422

    async def test_start_uses_server_defaults(self, client, training_manager):
        """Omitted optional fields should use server defaults."""
        job = _make_job()
        training_manager.get_status.return_value = job

        resp = await client.post(
            "/api/training/start",
            json={"dataset_name": "test-ds"},
        )
        assert resp.status_code == 201

        call_args = training_manager.start_training.call_args[0][0]
        # Should have server defaults, not None
        assert call_args.epochs > 0
        assert call_args.batch_size > 0
        assert call_args.image_size > 0


# --- POST /api/training/stop ---

class TestStopTraining:
    """Tests for POST /api/training/stop."""

    async def test_stop_success(self, client, training_manager):
        """Stop returns 200 with confirmation message."""
        resp = await client.post("/api/training/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        training_manager.stop_training.assert_called_once()

    async def test_stop_no_running_job(self, client, training_manager):
        """Return 409 when no job is running."""
        training_manager.stop_training = AsyncMock(
            side_effect=ConflictError("No training job is currently running")
        )

        resp = await client.post("/api/training/stop")
        assert resp.status_code == 409


# --- GET /api/training/status ---

class TestGetTrainingStatus:
    """Tests for GET /api/training/status."""

    async def test_status_no_job(self, client, training_manager):
        """Return idle status when no job exists."""
        training_manager.get_status.return_value = None

        resp = await client.get("/api/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"
        assert data["job_id"] == ""

    async def test_status_active_job(self, client, training_manager):
        """Return active job status with progress."""
        job = _make_job(current_epoch=25, total_epochs=100)
        training_manager.get_status.return_value = job

        resp = await client.get("/api/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "training"
        assert data["current_epoch"] == 25
        assert data["total_epochs"] == 100
        assert data["progress_pct"] == 25.0

    async def test_status_completed_job(self, client, training_manager):
        """Return completed job status."""
        job = _make_job(status="completed", current_epoch=100, total_epochs=100)
        job.completed_at = time.time()
        job.metrics = {"best_map50": 0.85}
        training_manager.get_status.return_value = job

        resp = await client.get("/api/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["progress_pct"] == 100.0
        assert data["metrics"]["best_map50"] == 0.85

    async def test_status_error_job(self, client, training_manager):
        """Return error job status with error message."""
        job = _make_job(status="error")
        job.error = "GPU out of memory"
        training_manager.get_status.return_value = job

        resp = await client.get("/api/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert data["error"] == "GPU out of memory"


# --- GET /api/training/history ---

class TestGetTrainingHistory:
    """Tests for GET /api/training/history."""

    async def test_empty_history(self, client, training_manager):
        """Return empty history when no jobs have run."""
        resp = await client.get("/api/training/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["jobs"] == []

    async def test_history_with_entries(self, client, training_manager):
        """Return history entries for completed jobs."""
        jobs = [
            _make_job(job_id="job-1", status="completed", current_epoch=100),
            _make_job(job_id="job-2", status="error", current_epoch=50),
        ]
        jobs[0].metrics = {"best_map50": 0.85}
        jobs[1].error = "OOM"
        training_manager.get_history.return_value = jobs

        resp = await client.get("/api/training/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert data["jobs"][0]["job_id"] == "job-1"
        assert data["jobs"][0]["status"] == "completed"
        assert data["jobs"][1]["job_id"] == "job-2"
        assert data["jobs"][1]["status"] == "error"

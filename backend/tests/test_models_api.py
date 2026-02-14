"""
API integration tests for model management endpoints.

Tests exercise the REST API through the ASGI test client with a
mock TrainingManager (wrapping a mock ModelStore) to isolate API
logic from persistence operations.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.models import set_training_manager
from backend.persistence.models import ModelInfo
from backend.training.manager import TrainingManager


# --- Fixtures ---

@pytest.fixture
def models_manager():
    """
    Fresh mock TrainingManager with mock ModelStore injected into
    the models API router.

    Provides a mock _model_store with standard success returns.
    Cleans up by resetting the module-level reference.
    """
    mgr = MagicMock(spec=TrainingManager)

    # Create mock model store
    ms = MagicMock()
    ms.list = AsyncMock(return_value=[])
    ms.get = AsyncMock(return_value=None)
    ms.delete = AsyncMock(return_value=False)
    ms.set_active = AsyncMock()
    ms.get_active = AsyncMock(return_value=None)
    mgr._model_store = ms

    set_training_manager(mgr)
    yield mgr
    set_training_manager(None)


def _make_model_info(
    name: str = "test-model",
    base_model: str = "yolo11n.pt",
    dataset_name: str = "test-ds",
    epochs: int = 100,
    map50: float = 0.85,
    is_active: bool = False,
) -> ModelInfo:
    """Create a ModelInfo with sensible defaults for tests."""
    return ModelInfo(
        name=name,
        path=Path(f"/models/{name}/best.pt"),
        base_model=base_model,
        dataset_name=dataset_name,
        created_at=1700000000.0,
        epochs_completed=epochs,
        best_map50=map50,
        is_active=is_active,
    )


# --- GET /api/models ---

class TestListModels:
    """Tests for GET /api/models."""

    async def test_empty_list(self, client, models_manager):
        """Return empty list when no models exist."""
        resp = await client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["models"] == []

    async def test_list_with_models(self, client, models_manager):
        """List includes all registered models."""
        models_manager._model_store.list = AsyncMock(return_value=[
            _make_model_info("model-a"),
            _make_model_info("model-b", is_active=True),
        ])

        resp = await client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert data["models"][0]["name"] == "model-a"
        assert data["models"][0]["is_active"] is False
        assert data["models"][1]["name"] == "model-b"
        assert data["models"][1]["is_active"] is True

    async def test_list_model_fields(self, client, models_manager):
        """List response contains all expected fields."""
        models_manager._model_store.list = AsyncMock(return_value=[
            _make_model_info(),
        ])

        resp = await client.get("/api/models")
        assert resp.status_code == 200
        model = resp.json()["models"][0]
        assert "name" in model
        assert "base_model" in model
        assert "dataset_name" in model
        assert "created_at" in model
        assert "epochs_completed" in model
        assert "best_map50" in model
        assert "is_active" in model
        assert "path" in model


# --- GET /api/models/{model_id} ---

class TestGetModel:
    """Tests for GET /api/models/{model_id}."""

    async def test_get_existing_model(self, client, models_manager):
        """Return model details for existing model."""
        models_manager._model_store.get = AsyncMock(
            return_value=_make_model_info("my-model")
        )

        resp = await client.get("/api/models/my-model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "my-model"
        assert data["base_model"] == "yolo11n.pt"
        assert data["dataset_name"] == "test-ds"
        assert data["epochs_completed"] == 100
        assert data["best_map50"] == 0.85

    async def test_get_nonexistent_model(self, client, models_manager):
        """Return 404 for nonexistent model."""
        models_manager._model_store.get = AsyncMock(return_value=None)

        resp = await client.get("/api/models/nonexistent")
        assert resp.status_code == 404


# --- DELETE /api/models/{model_id} ---

class TestDeleteModel:
    """Tests for DELETE /api/models/{model_id}."""

    async def test_delete_existing_model(self, client, models_manager):
        """Delete returns confirmation message."""
        models_manager._model_store.delete = AsyncMock(return_value=True)

        resp = await client.delete("/api/models/my-model")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "my-model" in data["message"]

    async def test_delete_nonexistent_model(self, client, models_manager):
        """Return 404 for nonexistent model."""
        models_manager._model_store.delete = AsyncMock(return_value=False)

        resp = await client.delete("/api/models/nonexistent")
        assert resp.status_code == 404


# --- PUT /api/models/{model_id}/activate ---

class TestActivateModel:
    """Tests for PUT /api/models/{model_id}/activate."""

    async def test_activate_existing_model(self, client, models_manager):
        """Activate returns the updated model with is_active=True."""
        activated = _make_model_info("my-model", is_active=True)
        models_manager._model_store.get = AsyncMock(return_value=activated)

        resp = await client.put("/api/models/my-model/activate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "my-model"
        assert data["is_active"] is True

    async def test_activate_nonexistent_model(self, client, models_manager):
        """Return 404 for nonexistent model."""
        models_manager._model_store.set_active = AsyncMock(
            side_effect=FileNotFoundError("Model 'nonexistent' not found in registry")
        )

        resp = await client.put("/api/models/nonexistent/activate")
        assert resp.status_code == 404

    async def test_activate_deactivates_others(self, client, models_manager):
        """Activate calls set_active which deactivates other models."""
        activated = _make_model_info("model-b", is_active=True)
        models_manager._model_store.get = AsyncMock(return_value=activated)

        resp = await client.put("/api/models/model-b/activate")
        assert resp.status_code == 200
        models_manager._model_store.set_active.assert_called_once_with("model-b")


# --- POST /api/models/{model_id}/export ---

class TestExportModel:
    """Tests for POST /api/models/{model_id}/export."""

    async def test_export_returns_501(self, client, models_manager):
        """Export endpoint is still a stub returning 501."""
        resp = await client.post("/api/models/my-model/export")
        assert resp.status_code == 501
        data = resp.json()
        assert "error" in data
        assert "detail" in data

"""
API integration tests for the pretrained model download endpoint.

Tests exercise POST /api/models/pretrained through the ASGI test client
with mocked ultralytics and ModelStore to isolate API logic from
real model downloads and persistence.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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

    ms = MagicMock()
    ms.list = AsyncMock(return_value=[])
    ms.get = AsyncMock(return_value=None)
    ms.delete = AsyncMock(return_value=False)
    ms.set_active = AsyncMock()
    ms.get_active = AsyncMock(return_value=None)
    ms.save = AsyncMock()
    mgr._model_store = ms

    set_training_manager(mgr)
    yield mgr
    set_training_manager(None)


def _make_model_info(
    name: str = "yolo11n",
    base_model: str = "yolo11n.pt",
    dataset_name: str = "pretrained",
    epochs: int = 0,
    map50: float | None = None,
    is_active: bool = False,
) -> ModelInfo:
    """Create a ModelInfo with defaults matching pretrained downloads."""
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


# --- POST /api/models/pretrained ---

class TestDownloadPretrained:
    """Tests for POST /api/models/pretrained."""

    @patch("backend.api.models._download_pretrained_model")
    async def test_successful_download(self, mock_download, client, models_manager):
        """Download and register a pretrained model successfully."""
        mock_download.return_value = Path("/tmp/yolo11n.pt")
        saved_info = _make_model_info()
        models_manager._model_store.save = AsyncMock(return_value=saved_info)

        resp = await client.post("/api/models/pretrained", json={
            "model_id": "yolo11n.pt",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "yolo11n"
        assert data["base_model"] == "yolo11n.pt"
        assert data["dataset_name"] == "pretrained"
        assert data["epochs_completed"] == 0

    @patch("backend.api.models._download_pretrained_model")
    async def test_successful_download_with_custom_name(self, mock_download, client, models_manager):
        """Download with a custom display name."""
        mock_download.return_value = Path("/tmp/yolo11n.pt")
        saved_info = _make_model_info(name="my-custom-model")
        models_manager._model_store.save = AsyncMock(return_value=saved_info)

        resp = await client.post("/api/models/pretrained", json={
            "model_id": "yolo11n.pt",
            "name": "my-custom-model",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "my-custom-model"

    @patch("backend.api.models._download_pretrained_model")
    async def test_model_store_save_called_correctly(self, mock_download, client, models_manager):
        """Verify save() is called with correct arguments."""
        mock_download.return_value = Path("/tmp/yolo11n.pt")
        saved_info = _make_model_info()
        models_manager._model_store.save = AsyncMock(return_value=saved_info)

        await client.post("/api/models/pretrained", json={
            "model_id": "yolo11n.pt",
        })

        models_manager._model_store.save.assert_called_once_with(
            name="yolo11n",
            weights_path=Path("/tmp/yolo11n.pt"),
            base_model="yolo11n.pt",
            dataset_name="pretrained",
            epochs_completed=0,
        )

    @patch("backend.api.models._download_pretrained_model")
    async def test_duplicate_name_returns_409(self, mock_download, client, models_manager):
        """Return 409 when model name already exists."""
        existing = _make_model_info()
        models_manager._model_store.get = AsyncMock(return_value=existing)

        resp = await client.post("/api/models/pretrained", json={
            "model_id": "yolo11n.pt",
        })

        assert resp.status_code == 409
        data = resp.json()
        assert "already exists" in data["error"]
        # Download should not be called if name already exists
        mock_download.assert_not_called()

    @patch("backend.api.models._download_pretrained_model")
    async def test_download_failure_returns_400(self, mock_download, client, models_manager):
        """Return 400 when ultralytics fails to download."""
        mock_download.side_effect = RuntimeError("Network error")

        resp = await client.post("/api/models/pretrained", json={
            "model_id": "invalid-model.pt",
        })

        assert resp.status_code == 400
        data = resp.json()
        assert "Failed to download" in data["error"]

    @patch("backend.api.models._download_pretrained_model")
    async def test_race_condition_save_value_error_returns_409(
        self, mock_download, client, models_manager
    ):
        """Return 409 when model_store.save raises ValueError (race condition)."""
        mock_download.return_value = Path("/tmp/yolo11n.pt")
        models_manager._model_store.save = AsyncMock(
            side_effect=ValueError("Model 'yolo11n' already exists")
        )

        resp = await client.post("/api/models/pretrained", json={
            "model_id": "yolo11n.pt",
        })

        assert resp.status_code == 409

    @patch("backend.api.models._download_pretrained_model")
    async def test_default_name_strips_pt_extension(self, mock_download, client, models_manager):
        """Default name removes .pt extension from model_id."""
        mock_download.return_value = Path("/tmp/yolov8s.pt")
        saved_info = _make_model_info(name="yolov8s", base_model="yolov8s.pt")
        models_manager._model_store.save = AsyncMock(return_value=saved_info)

        resp = await client.post("/api/models/pretrained", json={
            "model_id": "yolov8s.pt",
        })

        assert resp.status_code == 200
        # Verify save was called with stripped name
        models_manager._model_store.save.assert_called_once()
        call_kwargs = models_manager._model_store.save.call_args
        assert call_kwargs.kwargs["name"] == "yolov8s"

    async def test_missing_model_id_returns_422(self, client, models_manager):
        """Return 422 when model_id is missing from request body."""
        resp = await client.post("/api/models/pretrained", json={})
        assert resp.status_code == 422

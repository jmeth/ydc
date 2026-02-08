"""
Unit tests for FilesystemModelStore.

All tests use tmp_path for isolated filesystem operations.
Covers save, load, list, get, delete, set_active, and get_active.
"""

import json

import pytest

from backend.persistence.filesystem.model_store import FilesystemModelStore


def _make_store(tmp_path):
    """Create a FilesystemModelStore rooted at tmp_path."""
    return FilesystemModelStore(tmp_path / "models")


def _make_weights(tmp_path, name="weights.pt"):
    """Create a fake weights file and return its path."""
    weights = tmp_path / name
    weights.write_bytes(b"fake model weights data")
    return weights


class TestModelStoreSave:
    """Tests for save()."""

    async def test_save_copies_weights(self, tmp_path):
        """save() copies the weights file into managed storage."""
        store = _make_store(tmp_path)
        weights = _make_weights(tmp_path)
        info = await store.save("my_model", weights, base_model="yolo11n")
        assert info.path.exists()
        assert info.path.name == "best.pt"

    async def test_save_records_metadata(self, tmp_path):
        """save() stores metadata in the ModelInfo."""
        store = _make_store(tmp_path)
        weights = _make_weights(tmp_path)
        info = await store.save(
            "my_model",
            weights,
            base_model="yolo11n",
            dataset_name="my_ds",
            epochs_completed=50,
            metrics={"best_map50": 0.85},
        )
        assert info.name == "my_model"
        assert info.base_model == "yolo11n"
        assert info.dataset_name == "my_ds"
        assert info.epochs_completed == 50
        assert info.best_map50 == 0.85
        assert info.is_active is False

    async def test_save_creates_registry(self, tmp_path):
        """save() creates a registry.json file."""
        store = _make_store(tmp_path)
        weights = _make_weights(tmp_path)
        await store.save("m1", weights)
        registry_path = tmp_path / "models" / "registry.json"
        assert registry_path.exists()
        with open(registry_path) as f:
            data = json.load(f)
        assert "m1" in data

    async def test_save_duplicate_raises(self, tmp_path):
        """save() raises ValueError if model name already exists."""
        store = _make_store(tmp_path)
        weights = _make_weights(tmp_path)
        await store.save("dup", weights)
        with pytest.raises(ValueError, match="already exists"):
            await store.save("dup", weights)

    async def test_save_missing_weights_raises(self, tmp_path):
        """save() raises FileNotFoundError for non-existent weights."""
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            await store.save("m1", tmp_path / "nonexistent.pt")


class TestModelStoreLoad:
    """Tests for load()."""

    async def test_load_returns_weights_path(self, tmp_path):
        """load() returns the path to the stored weights file."""
        store = _make_store(tmp_path)
        weights = _make_weights(tmp_path)
        await store.save("m1", weights)
        path = await store.load("m1")
        assert path.exists()
        assert path.name == "best.pt"

    async def test_load_missing_model_raises(self, tmp_path):
        """load() raises FileNotFoundError for unregistered model."""
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            await store.load("nonexistent")


class TestModelStoreList:
    """Tests for list()."""

    async def test_list_empty(self, tmp_path):
        """list() returns empty list when no models registered."""
        store = _make_store(tmp_path)
        assert await store.list() == []

    async def test_list_multiple_models(self, tmp_path):
        """list() returns all models in sorted order."""
        store = _make_store(tmp_path)
        await store.save("beta", _make_weights(tmp_path, "w1.pt"))
        await store.save("alpha", _make_weights(tmp_path, "w2.pt"))
        models = await store.list()
        assert len(models) == 2
        assert models[0].name == "alpha"
        assert models[1].name == "beta"


class TestModelStoreGet:
    """Tests for get()."""

    async def test_get_existing(self, tmp_path):
        """get() returns ModelInfo for existing model."""
        store = _make_store(tmp_path)
        await store.save("m1", _make_weights(tmp_path), base_model="yolo11n")
        info = await store.get("m1")
        assert info is not None
        assert info.name == "m1"
        assert info.base_model == "yolo11n"

    async def test_get_missing(self, tmp_path):
        """get() returns None for non-existent model."""
        store = _make_store(tmp_path)
        assert await store.get("nope") is None


class TestModelStoreDelete:
    """Tests for delete()."""

    async def test_delete_existing(self, tmp_path):
        """delete() removes the model directory and registry entry."""
        store = _make_store(tmp_path)
        await store.save("doomed", _make_weights(tmp_path))
        result = await store.delete("doomed")
        assert result is True
        assert not (tmp_path / "models" / "doomed").exists()
        assert await store.get("doomed") is None

    async def test_delete_missing(self, tmp_path):
        """delete() returns False for non-existent model."""
        store = _make_store(tmp_path)
        assert await store.delete("nope") is False


class TestModelStoreActive:
    """Tests for set_active() and get_active()."""

    async def test_set_active(self, tmp_path):
        """set_active() marks the specified model as active."""
        store = _make_store(tmp_path)
        await store.save("m1", _make_weights(tmp_path, "w1.pt"))
        await store.set_active("m1")
        info = await store.get("m1")
        assert info.is_active is True

    async def test_set_active_clears_previous(self, tmp_path):
        """set_active() clears the active flag on all other models."""
        store = _make_store(tmp_path)
        await store.save("m1", _make_weights(tmp_path, "w1.pt"))
        await store.save("m2", _make_weights(tmp_path, "w2.pt"))
        await store.set_active("m1")
        await store.set_active("m2")
        m1 = await store.get("m1")
        m2 = await store.get("m2")
        assert m1.is_active is False
        assert m2.is_active is True

    async def test_set_active_missing_raises(self, tmp_path):
        """set_active() raises FileNotFoundError for non-existent model."""
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            await store.set_active("nope")

    async def test_get_active_returns_active(self, tmp_path):
        """get_active() returns the currently active model."""
        store = _make_store(tmp_path)
        await store.save("m1", _make_weights(tmp_path, "w1.pt"))
        await store.set_active("m1")
        active = await store.get_active()
        assert active is not None
        assert active.name == "m1"

    async def test_get_active_returns_none_when_no_active(self, tmp_path):
        """get_active() returns None when no model is active."""
        store = _make_store(tmp_path)
        assert await store.get_active() is None

"""
Unit tests for FilesystemModelStore.

All tests use tmp_path for isolated filesystem operations.
Covers save, load, list, get, delete, set_active, get_active,
export_zip, and import_zip.
"""

import json
import zipfile

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


class TestModelStoreExportZip:
    """Tests for export_zip()."""

    async def test_export_creates_zip(self, tmp_path):
        """export_zip() creates a zip archive at the output path."""
        store = _make_store(tmp_path)
        await store.save("m1", _make_weights(tmp_path), base_model="yolo11n", dataset_name="ds1")
        output_dir = tmp_path / "export"
        zip_path = await store.export_zip("m1", output_dir)
        assert zip_path.exists()
        assert zip_path.name == "m1.zip"

    async def test_export_zip_contains_weights(self, tmp_path):
        """export_zip() includes best.pt in the archive."""
        store = _make_store(tmp_path)
        await store.save("m1", _make_weights(tmp_path))
        zip_path = await store.export_zip("m1", tmp_path / "out")
        with zipfile.ZipFile(zip_path) as zf:
            assert "best.pt" in zf.namelist()
            assert zf.read("best.pt") == b"fake model weights data"

    async def test_export_zip_contains_meta(self, tmp_path):
        """export_zip() includes model_meta.json with correct fields."""
        store = _make_store(tmp_path)
        await store.save(
            "m1", _make_weights(tmp_path),
            base_model="yolo11n", dataset_name="ds1",
            epochs_completed=50, metrics={"best_map50": 0.9},
        )
        zip_path = await store.export_zip("m1", tmp_path / "out")
        with zipfile.ZipFile(zip_path) as zf:
            meta = json.loads(zf.read("model_meta.json"))
        assert meta["name"] == "m1"
        assert meta["base_model"] == "yolo11n"
        assert meta["dataset_name"] == "ds1"
        assert meta["epochs_completed"] == 50
        assert meta["best_map50"] == 0.9
        # Should not include instance-specific fields
        assert "is_active" not in meta
        assert "path" not in meta

    async def test_export_zip_includes_training_config(self, tmp_path):
        """export_zip() includes training_config.yaml when present."""
        store = _make_store(tmp_path)
        await store.save("m1", _make_weights(tmp_path))
        # Manually add a training config to the model directory
        config_path = tmp_path / "models" / "m1" / "training_config.yaml"
        config_path.write_text("epochs: 100\nbatch: 16\n")
        zip_path = await store.export_zip("m1", tmp_path / "out")
        with zipfile.ZipFile(zip_path) as zf:
            assert "training_config.yaml" in zf.namelist()
            assert b"epochs: 100" in zf.read("training_config.yaml")

    async def test_export_zip_omits_missing_training_config(self, tmp_path):
        """export_zip() skips training_config.yaml when not present."""
        store = _make_store(tmp_path)
        await store.save("m1", _make_weights(tmp_path))
        zip_path = await store.export_zip("m1", tmp_path / "out")
        with zipfile.ZipFile(zip_path) as zf:
            assert "training_config.yaml" not in zf.namelist()

    async def test_export_missing_model_raises(self, tmp_path):
        """export_zip() raises FileNotFoundError for unregistered model."""
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            await store.export_zip("nope", tmp_path / "out")


class TestModelStoreImportZip:
    """Tests for import_zip()."""

    def _create_model_zip(self, tmp_path, zip_name="m1.zip", meta=None, include_config=False):
        """
        Helper to create a model zip archive for import tests.

        Args:
            tmp_path: Base temp directory.
            zip_name: Name for the zip file.
            meta: Optional metadata dict for model_meta.json.
            include_config: Whether to include training_config.yaml.

        Returns:
            Path to the created zip file.
        """
        zip_path = tmp_path / zip_name
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("best.pt", b"imported weights data")
            if meta is not None:
                zf.writestr("model_meta.json", json.dumps(meta))
            if include_config:
                zf.writestr("training_config.yaml", "epochs: 50\n")
        return zip_path

    async def test_import_registers_model(self, tmp_path):
        """import_zip() registers the model in the store."""
        store = _make_store(tmp_path)
        meta = {"name": "imported-m1", "base_model": "yolo11n", "dataset_name": "ds1",
                "epochs_completed": 25, "best_map50": 0.8}
        zip_path = self._create_model_zip(tmp_path, meta=meta)
        info = await store.import_zip(zip_path)
        assert info.name == "imported-m1"
        assert info.base_model == "yolo11n"
        assert info.dataset_name == "ds1"
        assert info.epochs_completed == 25
        assert info.best_map50 == 0.8
        assert info.is_active is False

    async def test_import_extracts_weights(self, tmp_path):
        """import_zip() extracts best.pt to the model directory."""
        store = _make_store(tmp_path)
        meta = {"name": "m2"}
        zip_path = self._create_model_zip(tmp_path, meta=meta)
        info = await store.import_zip(zip_path)
        assert info.path.exists()
        assert info.path.name == "best.pt"

    async def test_import_extracts_training_config(self, tmp_path):
        """import_zip() extracts training_config.yaml when present."""
        store = _make_store(tmp_path)
        meta = {"name": "m3"}
        zip_path = self._create_model_zip(tmp_path, meta=meta, include_config=True)
        await store.import_zip(zip_path)
        config_path = tmp_path / "models" / "m3" / "training_config.yaml"
        assert config_path.exists()
        assert "epochs: 50" in config_path.read_text()

    async def test_import_name_override(self, tmp_path):
        """import_zip() uses the explicit name param over model_meta.json."""
        store = _make_store(tmp_path)
        meta = {"name": "original-name"}
        zip_path = self._create_model_zip(tmp_path, meta=meta)
        info = await store.import_zip(zip_path, name="override-name")
        assert info.name == "override-name"

    async def test_import_uses_zip_filename_when_no_meta(self, tmp_path):
        """import_zip() falls back to zip filename stem when no meta."""
        store = _make_store(tmp_path)
        zip_path = self._create_model_zip(tmp_path, zip_name="fallback-model.zip")
        info = await store.import_zip(zip_path)
        assert info.name == "fallback-model"

    async def test_import_duplicate_raises(self, tmp_path):
        """import_zip() raises ValueError if model name already exists."""
        store = _make_store(tmp_path)
        await store.save("existing", _make_weights(tmp_path))
        meta = {"name": "existing"}
        zip_path = self._create_model_zip(tmp_path, meta=meta)
        with pytest.raises(ValueError, match="already exists"):
            await store.import_zip(zip_path)

    async def test_import_missing_zip_raises(self, tmp_path):
        """import_zip() raises FileNotFoundError for missing zip file."""
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError, match="Zip file not found"):
            await store.import_zip(tmp_path / "nonexistent.zip")

    async def test_import_missing_weights_raises(self, tmp_path):
        """import_zip() raises FileNotFoundError when zip lacks best.pt."""
        store = _make_store(tmp_path)
        zip_path = tmp_path / "bad.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("model_meta.json", '{"name": "bad"}')
        with pytest.raises(FileNotFoundError, match="missing required"):
            await store.import_zip(zip_path)

    async def test_import_roundtrip(self, tmp_path):
        """A model exported via export_zip() can be reimported via import_zip()."""
        store = _make_store(tmp_path)
        await store.save(
            "roundtrip", _make_weights(tmp_path),
            base_model="yolo11n", dataset_name="ds1",
            epochs_completed=100, metrics={"best_map50": 0.95},
        )
        # Export
        zip_path = await store.export_zip("roundtrip", tmp_path / "export")

        # Import into a fresh store
        store2 = FilesystemModelStore(tmp_path / "models2")
        info = await store2.import_zip(zip_path)
        assert info.name == "roundtrip"
        assert info.base_model == "yolo11n"
        assert info.dataset_name == "ds1"
        assert info.epochs_completed == 100
        assert info.best_map50 == 0.95
        assert info.path.exists()

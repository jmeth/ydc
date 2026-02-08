"""
Unit tests for FilesystemDatasetStore.

All tests use tmp_path for isolated filesystem operations.
Covers create, list, get, delete, update_classes, prompts, and zip export/import.
"""

import yaml
import pytest

from backend.persistence.filesystem.dataset_store import FilesystemDatasetStore


def _make_store(tmp_path):
    """Create a FilesystemDatasetStore rooted at tmp_path."""
    return FilesystemDatasetStore(tmp_path / "datasets")


class TestDatasetStoreCreate:
    """Tests for create()."""

    async def test_create_returns_info(self, tmp_path):
        """create() returns a DatasetInfo with correct fields."""
        store = _make_store(tmp_path)
        info = await store.create("my_ds", ["cat", "dog"])
        assert info.name == "my_ds"
        assert info.classes == ["cat", "dog"]
        assert info.num_images == {"train": 0, "val": 0, "test": 0}

    async def test_create_builds_directory_structure(self, tmp_path):
        """create() makes images/{train,val,test} and labels/{train,val,test} dirs."""
        store = _make_store(tmp_path)
        await store.create("my_ds", ["cat"])
        ds_path = tmp_path / "datasets" / "my_ds"
        for split in ("train", "val", "test"):
            assert (ds_path / "images" / split).is_dir()
            assert (ds_path / "labels" / split).is_dir()

    async def test_create_writes_data_yaml(self, tmp_path):
        """create() writes a valid data.yaml with YOLO format."""
        store = _make_store(tmp_path)
        await store.create("my_ds", ["cat", "dog"])
        data_yaml = tmp_path / "datasets" / "my_ds" / "data.yaml"
        with open(data_yaml) as f:
            data = yaml.safe_load(f)
        assert data["nc"] == 2
        assert data["names"] == {0: "cat", 1: "dog"}
        assert data["train"] == "images/train"

    async def test_create_duplicate_raises(self, tmp_path):
        """create() raises ValueError if dataset already exists."""
        store = _make_store(tmp_path)
        await store.create("dup", ["a"])
        with pytest.raises(ValueError, match="already exists"):
            await store.create("dup", ["b"])

    async def test_create_empty_classes(self, tmp_path):
        """create() allows empty class list."""
        store = _make_store(tmp_path)
        info = await store.create("empty_cls", [])
        assert info.classes == []


class TestDatasetStoreGet:
    """Tests for get()."""

    async def test_get_existing_dataset(self, tmp_path):
        """get() reads back the dataset metadata correctly."""
        store = _make_store(tmp_path)
        await store.create("ds1", ["cat", "dog", "bird"])
        info = await store.get("ds1")
        assert info is not None
        assert info.name == "ds1"
        assert info.classes == ["cat", "dog", "bird"]

    async def test_get_missing_returns_none(self, tmp_path):
        """get() returns None for non-existent dataset."""
        store = _make_store(tmp_path)
        assert await store.get("nonexistent") is None

    async def test_get_counts_images(self, tmp_path):
        """get() counts image files in each split."""
        store = _make_store(tmp_path)
        await store.create("ds1", ["a"])
        # Create some fake image files
        img_dir = tmp_path / "datasets" / "ds1" / "images" / "train"
        (img_dir / "a.jpg").write_bytes(b"fake")
        (img_dir / "b.png").write_bytes(b"fake")
        info = await store.get("ds1")
        assert info.num_images["train"] == 2
        assert info.num_images["val"] == 0


class TestDatasetStoreList:
    """Tests for list()."""

    async def test_list_empty(self, tmp_path):
        """list() returns empty list when no datasets exist."""
        store = _make_store(tmp_path)
        assert await store.list() == []

    async def test_list_multiple_datasets(self, tmp_path):
        """list() returns all datasets in sorted order."""
        store = _make_store(tmp_path)
        await store.create("beta", ["a"])
        await store.create("alpha", ["b"])
        datasets = await store.list()
        assert len(datasets) == 2
        assert datasets[0].name == "alpha"
        assert datasets[1].name == "beta"

    async def test_list_ignores_non_dataset_dirs(self, tmp_path):
        """list() skips directories without data.yaml."""
        store = _make_store(tmp_path)
        await store.create("real", ["a"])
        (tmp_path / "datasets" / "not_a_dataset").mkdir()
        datasets = await store.list()
        assert len(datasets) == 1


class TestDatasetStoreDelete:
    """Tests for delete()."""

    async def test_delete_existing(self, tmp_path):
        """delete() removes dataset directory and returns True."""
        store = _make_store(tmp_path)
        await store.create("doomed", ["x"])
        result = await store.delete("doomed")
        assert result is True
        assert not (tmp_path / "datasets" / "doomed").exists()

    async def test_delete_missing(self, tmp_path):
        """delete() returns False for non-existent dataset."""
        store = _make_store(tmp_path)
        assert await store.delete("nope") is False


class TestDatasetStoreUpdateClasses:
    """Tests for update_classes()."""

    async def test_update_classes(self, tmp_path):
        """update_classes() rewrites the names in data.yaml."""
        store = _make_store(tmp_path)
        await store.create("ds1", ["old_a", "old_b"])
        await store.update_classes("ds1", ["new_x", "new_y", "new_z"])
        info = await store.get("ds1")
        assert info.classes == ["new_x", "new_y", "new_z"]

    async def test_update_classes_missing_raises(self, tmp_path):
        """update_classes() raises FileNotFoundError for missing dataset."""
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            await store.update_classes("nope", ["a"])


class TestDatasetStorePrompts:
    """Tests for get_prompts() and save_prompts()."""

    async def test_get_prompts_empty(self, tmp_path):
        """get_prompts() returns empty dict when no prompts.yaml exists."""
        store = _make_store(tmp_path)
        await store.create("ds1", ["a"])
        result = await store.get_prompts("ds1")
        assert result == {}

    async def test_save_and_get_prompts_round_trip(self, tmp_path):
        """save_prompts() then get_prompts() round-trips correctly."""
        store = _make_store(tmp_path)
        await store.create("ds1", ["cat", "dog"])
        prompts = {0: ["a cat", "feline"], 1: ["a dog", "canine"]}
        await store.save_prompts("ds1", prompts)
        loaded = await store.get_prompts("ds1")
        assert loaded == prompts

    async def test_save_prompts_missing_raises(self, tmp_path):
        """save_prompts() raises FileNotFoundError for missing dataset."""
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            await store.save_prompts("nope", {0: ["a"]})


class TestDatasetStoreExportImport:
    """Tests for export_zip() and import_zip()."""

    async def test_export_creates_zip(self, tmp_path):
        """export_zip() creates a .zip file at the output path."""
        store = _make_store(tmp_path)
        await store.create("ds1", ["cat"])
        output = tmp_path / "exports"
        zip_path = await store.export_zip("ds1", output)
        assert zip_path.exists()
        assert zip_path.suffix == ".zip"

    async def test_export_missing_raises(self, tmp_path):
        """export_zip() raises FileNotFoundError for missing dataset."""
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            await store.export_zip("nope", tmp_path / "out")

    async def test_import_from_exported_zip(self, tmp_path):
        """import_zip() successfully imports a previously exported dataset."""
        store = _make_store(tmp_path)
        await store.create("original", ["cat", "dog"])
        # Add a fake image so the zip has content
        (tmp_path / "datasets" / "original" / "images" / "train" / "a.jpg").write_bytes(b"fake")

        output = tmp_path / "exports"
        zip_path = await store.export_zip("original", output)

        info = await store.import_zip(zip_path, name="imported")
        assert info.name == "imported"
        assert info.classes == ["cat", "dog"]
        assert info.num_images["train"] == 1

    async def test_import_duplicate_raises(self, tmp_path):
        """import_zip() raises ValueError if dataset name already exists."""
        store = _make_store(tmp_path)
        await store.create("existing", ["a"])
        output = tmp_path / "exports"
        zip_path = await store.export_zip("existing", output)
        with pytest.raises(ValueError, match="already exists"):
            await store.import_zip(zip_path, name="existing")

    async def test_import_missing_zip_raises(self, tmp_path):
        """import_zip() raises FileNotFoundError for non-existent zip."""
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            await store.import_zip(tmp_path / "fake.zip")

    async def test_import_uses_zip_stem_as_default_name(self, tmp_path):
        """import_zip() uses the zip filename stem when name is not provided."""
        store = _make_store(tmp_path)
        await store.create("mydata", ["x"])
        output = tmp_path / "exports"
        zip_path = await store.export_zip("mydata", output)
        await store.delete("mydata")
        info = await store.import_zip(zip_path)
        assert info.name == "mydata"

"""
Unit tests for FilesystemImageStore.

All tests use tmp_path for isolated filesystem operations.
Uses cv2 to create real image files for save/load round-trip tests.
"""

import numpy as np
import pytest

from backend.persistence.filesystem.image_store import FilesystemImageStore
from backend.persistence.models import Annotation


def _make_store(tmp_path):
    """Create a FilesystemImageStore rooted at tmp_path with dataset dirs."""
    base = tmp_path / "datasets"
    for split in ("train", "val", "test"):
        (base / "ds1" / "images" / split).mkdir(parents=True)
        (base / "ds1" / "labels" / split).mkdir(parents=True)
    return FilesystemImageStore(base)


def _make_image(width=100, height=80):
    """Create a simple BGR test image as a numpy array."""
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestImageStoreSave:
    """Tests for save()."""

    async def test_save_creates_file(self, tmp_path):
        """save() creates an image file in the correct split directory."""
        store = _make_store(tmp_path)
        info = await store.save("ds1", "train", "frame_001.jpg", _make_image())
        assert info.path.exists()
        assert info.filename == "frame_001.jpg"
        assert info.split == "train"

    async def test_save_returns_dimensions(self, tmp_path):
        """save() returns correct width and height from the image array."""
        store = _make_store(tmp_path)
        info = await store.save("ds1", "train", "img.jpg", _make_image(320, 240))
        assert info.width == 320
        assert info.height == 240

    async def test_save_returns_file_size(self, tmp_path):
        """save() returns a positive file size."""
        store = _make_store(tmp_path)
        info = await store.save("ds1", "train", "img.jpg", _make_image())
        assert info.size_bytes > 0

    async def test_save_creates_parent_dirs(self, tmp_path):
        """save() creates missing parent directories."""
        base = tmp_path / "datasets"
        store = FilesystemImageStore(base)
        info = await store.save("new_ds", "val", "img.jpg", _make_image())
        assert info.path.exists()


class TestImageStoreLoad:
    """Tests for load()."""

    async def test_load_round_trip(self, tmp_path):
        """load() returns an image with the same shape as saved."""
        store = _make_store(tmp_path)
        original = _make_image(200, 150)
        await store.save("ds1", "train", "img.png", original)
        loaded = await store.load("ds1", "train", "img.png")
        assert loaded.shape == original.shape

    async def test_load_missing_raises(self, tmp_path):
        """load() raises FileNotFoundError for non-existent image."""
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            await store.load("ds1", "train", "nonexistent.jpg")


class TestImageStoreDelete:
    """Tests for delete()."""

    async def test_delete_existing_file(self, tmp_path):
        """delete() removes the file and returns True."""
        store = _make_store(tmp_path)
        await store.save("ds1", "train", "img.jpg", _make_image())
        result = await store.delete("ds1", "train", "img.jpg")
        assert result is True
        assert not (tmp_path / "datasets" / "ds1" / "images" / "train" / "img.jpg").exists()

    async def test_delete_missing_file(self, tmp_path):
        """delete() returns False when file doesn't exist."""
        store = _make_store(tmp_path)
        result = await store.delete("ds1", "train", "nope.jpg")
        assert result is False


class TestImageStoreMove:
    """Tests for move()."""

    async def test_move_between_splits(self, tmp_path):
        """move() relocates an image from one split to another."""
        store = _make_store(tmp_path)
        await store.save("ds1", "train", "img.jpg", _make_image())
        await store.move("ds1", "img.jpg", "train", "val")
        assert not (tmp_path / "datasets" / "ds1" / "images" / "train" / "img.jpg").exists()
        assert (tmp_path / "datasets" / "ds1" / "images" / "val" / "img.jpg").exists()

    async def test_move_missing_raises(self, tmp_path):
        """move() raises FileNotFoundError for non-existent source."""
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            await store.move("ds1", "nope.jpg", "train", "val")


class TestImageStoreList:
    """Tests for list()."""

    async def test_list_single_split(self, tmp_path):
        """list() with a split filter returns only images in that split."""
        store = _make_store(tmp_path)
        await store.save("ds1", "train", "a.jpg", _make_image())
        await store.save("ds1", "val", "b.jpg", _make_image())
        result = await store.list("ds1", split="train")
        assert len(result) == 1
        assert result[0].filename == "a.jpg"

    async def test_list_all_splits(self, tmp_path):
        """list() without a split filter returns images across all splits."""
        store = _make_store(tmp_path)
        await store.save("ds1", "train", "a.jpg", _make_image())
        await store.save("ds1", "val", "b.jpg", _make_image())
        result = await store.list("ds1")
        assert len(result) == 2

    async def test_list_sets_zero_dimensions(self, tmp_path):
        """list() sets width=height=0 to avoid loading images."""
        store = _make_store(tmp_path)
        await store.save("ds1", "train", "a.jpg", _make_image(320, 240))
        result = await store.list("ds1", split="train")
        assert result[0].width == 0
        assert result[0].height == 0

    async def test_list_detects_labels(self, tmp_path):
        """list() sets has_labels=True when a corresponding .txt file exists."""
        store = _make_store(tmp_path)
        await store.save("ds1", "train", "a.jpg", _make_image())
        # Create a label file manually
        label = tmp_path / "datasets" / "ds1" / "labels" / "train" / "a.txt"
        label.write_text("0 0.5 0.5 0.1 0.1")
        result = await store.list("ds1", split="train")
        assert result[0].has_labels is True

    async def test_list_empty_split(self, tmp_path):
        """list() returns empty list for split with no images."""
        store = _make_store(tmp_path)
        result = await store.list("ds1", split="test")
        assert result == []


class TestImageStoreExists:
    """Tests for exists()."""

    async def test_exists_true(self, tmp_path):
        """exists() returns True when image file is present."""
        store = _make_store(tmp_path)
        await store.save("ds1", "train", "img.jpg", _make_image())
        assert await store.exists("ds1", "train", "img.jpg") is True

    async def test_exists_false(self, tmp_path):
        """exists() returns False when image file is absent."""
        store = _make_store(tmp_path)
        assert await store.exists("ds1", "train", "nope.jpg") is False

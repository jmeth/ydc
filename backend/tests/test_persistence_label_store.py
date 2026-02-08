"""
Unit tests for FilesystemLabelStore.

All tests use tmp_path for isolated filesystem operations.
Verifies YOLO-format .txt read/write, move, delete, and exists.
"""

import pytest

from backend.persistence.filesystem.label_store import FilesystemLabelStore
from backend.persistence.models import Annotation


def _make_store(tmp_path):
    """Create a FilesystemLabelStore rooted at tmp_path with dataset dirs."""
    base = tmp_path / "datasets"
    for split in ("train", "val", "test"):
        (base / "ds1" / "labels" / split).mkdir(parents=True)
    return FilesystemLabelStore(base)


def _make_annotations():
    """Create a small list of sample annotations."""
    return [
        Annotation(class_id=0, x=0.5, y=0.5, width=0.3, height=0.4),
        Annotation(class_id=1, x=0.2, y=0.8, width=0.1, height=0.2),
    ]


class TestLabelStoreSave:
    """Tests for save()."""

    async def test_save_creates_file(self, tmp_path):
        """save() creates a .txt file in the correct split directory."""
        store = _make_store(tmp_path)
        await store.save("ds1", "train", "img_001.jpg", _make_annotations())
        label_path = tmp_path / "datasets" / "ds1" / "labels" / "train" / "img_001.txt"
        assert label_path.exists()

    async def test_save_writes_yolo_format(self, tmp_path):
        """save() writes one annotation per line in YOLO format."""
        store = _make_store(tmp_path)
        await store.save("ds1", "train", "img_001.jpg", _make_annotations())
        label_path = tmp_path / "datasets" / "ds1" / "labels" / "train" / "img_001.txt"
        lines = label_path.read_text().splitlines()
        assert len(lines) == 2
        parts = lines[0].split()
        assert parts[0] == "0"  # class_id
        assert len(parts) == 5  # class_id x y w h

    async def test_save_empty_annotations(self, tmp_path):
        """save() with empty list creates an empty file."""
        store = _make_store(tmp_path)
        await store.save("ds1", "train", "img_001.jpg", [])
        label_path = tmp_path / "datasets" / "ds1" / "labels" / "train" / "img_001.txt"
        assert label_path.exists()
        assert label_path.read_text() == ""

    async def test_save_creates_parent_dirs(self, tmp_path):
        """save() creates missing parent directories."""
        base = tmp_path / "datasets"
        store = FilesystemLabelStore(base)
        await store.save("new_ds", "val", "img.jpg", _make_annotations())
        assert (base / "new_ds" / "labels" / "val" / "img.txt").exists()


class TestLabelStoreLoad:
    """Tests for load()."""

    async def test_load_returns_annotations(self, tmp_path):
        """load() parses YOLO-format file into Annotation objects."""
        store = _make_store(tmp_path)
        original = _make_annotations()
        await store.save("ds1", "train", "img_001.jpg", original)
        loaded = await store.load("ds1", "train", "img_001.jpg")
        assert len(loaded) == 2
        assert loaded[0].class_id == 0
        assert abs(loaded[0].x - 0.5) < 1e-5
        assert abs(loaded[1].width - 0.1) < 1e-5

    async def test_load_missing_file_returns_empty(self, tmp_path):
        """load() returns empty list when label file doesn't exist."""
        store = _make_store(tmp_path)
        result = await store.load("ds1", "train", "nonexistent.jpg")
        assert result == []

    async def test_load_skips_malformed_lines(self, tmp_path):
        """load() skips lines with fewer than 5 parts."""
        store = _make_store(tmp_path)
        label_path = tmp_path / "datasets" / "ds1" / "labels" / "train" / "bad.txt"
        label_path.write_text("0 0.5 0.5\n1 0.2 0.8 0.1 0.2\n")
        loaded = await store.load("ds1", "train", "bad.jpg")
        assert len(loaded) == 1
        assert loaded[0].class_id == 1


class TestLabelStoreDelete:
    """Tests for delete()."""

    async def test_delete_existing_file(self, tmp_path):
        """delete() removes the file and returns True."""
        store = _make_store(tmp_path)
        await store.save("ds1", "train", "img.jpg", _make_annotations())
        result = await store.delete("ds1", "train", "img.jpg")
        assert result is True
        assert not (tmp_path / "datasets" / "ds1" / "labels" / "train" / "img.txt").exists()

    async def test_delete_missing_file(self, tmp_path):
        """delete() returns False when file doesn't exist."""
        store = _make_store(tmp_path)
        result = await store.delete("ds1", "train", "nope.jpg")
        assert result is False


class TestLabelStoreMove:
    """Tests for move()."""

    async def test_move_between_splits(self, tmp_path):
        """move() relocates a label file from one split to another."""
        store = _make_store(tmp_path)
        await store.save("ds1", "train", "img.jpg", _make_annotations())
        await store.move("ds1", "img.jpg", "train", "val")
        assert not (tmp_path / "datasets" / "ds1" / "labels" / "train" / "img.txt").exists()
        assert (tmp_path / "datasets" / "ds1" / "labels" / "val" / "img.txt").exists()

    async def test_move_missing_file_raises(self, tmp_path):
        """move() raises FileNotFoundError for non-existent source."""
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            await store.move("ds1", "nope.jpg", "train", "val")


class TestLabelStoreExists:
    """Tests for exists()."""

    async def test_exists_true(self, tmp_path):
        """exists() returns True when label file is present."""
        store = _make_store(tmp_path)
        await store.save("ds1", "train", "img.jpg", _make_annotations())
        assert await store.exists("ds1", "train", "img.jpg") is True

    async def test_exists_false(self, tmp_path):
        """exists() returns False when label file is absent."""
        store = _make_store(tmp_path)
        assert await store.exists("ds1", "train", "nope.jpg") is False

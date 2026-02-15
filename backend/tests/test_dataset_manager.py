"""
Unit tests for DatasetManager business logic.

Tests cover dataset CRUD, image management, label management,
split changes, prompts, and validation. Uses real filesystem stores
via the persistence_stores fixture backed by tmp_path.
"""

import pytest
import numpy as np

from backend.core.events import (
    DATASET_CREATED,
    DATASET_DELETED,
    DATASET_IMAGE_ADDED,
    DATASET_IMAGE_DELETED,
    EventBus,
)
from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from backend.dataset.manager import DatasetManager
from backend.persistence.models import Annotation


@pytest.fixture
async def event_bus():
    """Fresh started EventBus for event tests."""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
def manager(persistence_stores, event_bus):
    """DatasetManager wired to real filesystem stores and EventBus."""
    return DatasetManager(
        dataset_store=persistence_stores.dataset,
        image_store=persistence_stores.image,
        label_store=persistence_stores.label,
        event_bus=event_bus,
    )


@pytest.fixture
def sample_image():
    """100x100 BGR image as numpy array."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


# --- Dataset CRUD ---


class TestCreateDataset:
    """Tests for DatasetManager.create_dataset()."""

    async def test_create_success(self, manager):
        """Create a dataset and verify returned info."""
        info = await manager.create_dataset("test-ds", ["cat", "dog"])
        assert info.name == "test-ds"
        assert info.classes == ["cat", "dog"]
        assert info.num_images == {"train": 0, "val": 0, "test": 0}

    async def test_create_invalid_name_fails(self, manager):
        """Reject names with invalid characters."""
        with pytest.raises(ValidationError, match="Invalid dataset name"):
            await manager.create_dataset("bad name!", ["cat"])

    async def test_create_name_starting_with_hyphen_fails(self, manager):
        """Reject names starting with a hyphen."""
        with pytest.raises(ValidationError, match="Invalid dataset name"):
            await manager.create_dataset("-bad", ["cat"])

    async def test_create_empty_name_fails(self, manager):
        """Reject empty name."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            await manager.create_dataset("", ["cat"])

    async def test_create_empty_classes_fails(self, manager):
        """Reject empty classes list."""
        with pytest.raises(ValidationError, match="Classes list cannot be empty"):
            await manager.create_dataset("test-ds", [])

    async def test_create_duplicate_fails(self, manager):
        """Reject duplicate dataset names."""
        await manager.create_dataset("test-ds", ["cat"])
        with pytest.raises(ConflictError):
            await manager.create_dataset("test-ds", ["dog"])

    async def test_create_publishes_event(self, manager, event_bus):
        """Verify dataset.created event is published."""
        events = []
        event_bus.subscribe(DATASET_CREATED, lambda data: events.append(data))

        await manager.create_dataset("test-ds", ["cat", "dog"])
        assert len(events) == 1
        assert events[0]["name"] == "test-ds"


class TestListDatasets:
    """Tests for DatasetManager.list_datasets()."""

    async def test_empty_list(self, manager):
        """Return empty list when no datasets exist."""
        result = await manager.list_datasets()
        assert result == []

    async def test_list_multiple(self, manager):
        """List all created datasets."""
        await manager.create_dataset("ds1", ["cat"])
        await manager.create_dataset("ds2", ["dog"])
        result = await manager.list_datasets()
        names = [d.name for d in result]
        assert "ds1" in names
        assert "ds2" in names


class TestGetDataset:
    """Tests for DatasetManager.get_dataset()."""

    async def test_get_existing(self, manager):
        """Get an existing dataset by name."""
        await manager.create_dataset("test-ds", ["cat", "dog"])
        info = await manager.get_dataset("test-ds")
        assert info.name == "test-ds"
        assert info.classes == ["cat", "dog"]

    async def test_get_not_found(self, manager):
        """Raise NotFoundError for missing dataset."""
        with pytest.raises(NotFoundError):
            await manager.get_dataset("nonexistent")


class TestDeleteDataset:
    """Tests for DatasetManager.delete_dataset()."""

    async def test_delete_success(self, manager):
        """Delete an existing dataset."""
        await manager.create_dataset("test-ds", ["cat"])
        result = await manager.delete_dataset("test-ds")
        assert result is True

        # Verify it's gone
        with pytest.raises(NotFoundError):
            await manager.get_dataset("test-ds")

    async def test_delete_not_found(self, manager):
        """Raise NotFoundError when deleting nonexistent dataset."""
        with pytest.raises(NotFoundError):
            await manager.delete_dataset("nonexistent")

    async def test_delete_publishes_event(self, manager, event_bus):
        """Verify dataset.deleted event is published."""
        events = []
        event_bus.subscribe(DATASET_DELETED, lambda data: events.append(data))

        await manager.create_dataset("test-ds", ["cat"])
        await manager.delete_dataset("test-ds")
        assert len(events) == 1
        assert events[0]["name"] == "test-ds"


class TestUpdateDataset:
    """Tests for DatasetManager.update_dataset()."""

    async def test_update_classes(self, manager):
        """Update dataset classes and verify."""
        await manager.create_dataset("test-ds", ["cat"])
        info = await manager.update_dataset("test-ds", ["cat", "dog", "bird"])
        assert info.classes == ["cat", "dog", "bird"]

    async def test_update_not_found(self, manager):
        """Raise NotFoundError for missing dataset."""
        with pytest.raises(NotFoundError):
            await manager.update_dataset("nonexistent", ["cat"])

    async def test_update_empty_classes_fails(self, manager):
        """Reject empty classes list."""
        await manager.create_dataset("test-ds", ["cat"])
        with pytest.raises(ValidationError, match="Classes list cannot be empty"):
            await manager.update_dataset("test-ds", [])

    async def test_remove_class_remaps_annotations(self, manager, sample_image):
        """Removing a middle class decrements higher class IDs in labels."""
        await manager.create_dataset("test-ds", ["cat", "dog", "bird"])
        await manager.add_image("test-ds", "train", "img.jpg", sample_image)

        # Save annotations: cat(0), dog(1), bird(2)
        await manager.save_labels("test-ds", "train", "img.jpg", [
            Annotation(class_id=0, x=0.5, y=0.5, width=0.2, height=0.2),
            Annotation(class_id=1, x=0.3, y=0.3, width=0.1, height=0.1),
            Annotation(class_id=2, x=0.7, y=0.7, width=0.3, height=0.3),
        ])

        # Remove "dog" (index 1) — bird should shift from 2 to 1
        await manager.update_dataset("test-ds", ["cat", "bird"])

        labels = await manager.get_labels("test-ds", "train", "img.jpg")
        assert len(labels) == 2
        assert labels[0].class_id == 0  # cat stays 0
        assert labels[1].class_id == 1  # bird shifted from 2 to 1

    async def test_remove_class_deletes_its_annotations(self, manager, sample_image):
        """Annotations referencing a removed class are deleted."""
        await manager.create_dataset("test-ds", ["cat", "dog", "bird"])
        await manager.add_image("test-ds", "train", "img.jpg", sample_image)

        # All annotations reference "dog" (index 1)
        await manager.save_labels("test-ds", "train", "img.jpg", [
            Annotation(class_id=1, x=0.5, y=0.5, width=0.2, height=0.2),
            Annotation(class_id=1, x=0.3, y=0.3, width=0.1, height=0.1),
        ])

        # Remove "dog" — all annotations should be deleted
        await manager.update_dataset("test-ds", ["cat", "bird"])

        labels = await manager.get_labels("test-ds", "train", "img.jpg")
        assert len(labels) == 0

    async def test_add_class_preserves_annotations(self, manager, sample_image):
        """Adding a new class doesn't change existing annotation IDs."""
        await manager.create_dataset("test-ds", ["cat", "dog"])
        await manager.add_image("test-ds", "train", "img.jpg", sample_image)

        await manager.save_labels("test-ds", "train", "img.jpg", [
            Annotation(class_id=0, x=0.5, y=0.5, width=0.2, height=0.2),
            Annotation(class_id=1, x=0.3, y=0.3, width=0.1, height=0.1),
        ])

        # Append "bird" — existing IDs unchanged
        await manager.update_dataset("test-ds", ["cat", "dog", "bird"])

        labels = await manager.get_labels("test-ds", "train", "img.jpg")
        assert len(labels) == 2
        assert labels[0].class_id == 0
        assert labels[1].class_id == 1

    async def test_remap_across_splits(self, manager, sample_image):
        """Annotation remapping applies to all splits."""
        await manager.create_dataset("test-ds", ["cat", "dog", "bird"])
        await manager.add_image("test-ds", "train", "img1.jpg", sample_image)
        await manager.add_image("test-ds", "val", "img2.jpg", sample_image)

        await manager.save_labels("test-ds", "train", "img1.jpg", [
            Annotation(class_id=2, x=0.5, y=0.5, width=0.2, height=0.2),
        ])
        await manager.save_labels("test-ds", "val", "img2.jpg", [
            Annotation(class_id=2, x=0.3, y=0.3, width=0.1, height=0.1),
        ])

        # Remove "dog" (index 1) — bird shifts from 2 to 1
        await manager.update_dataset("test-ds", ["cat", "bird"])

        train_labels = await manager.get_labels("test-ds", "train", "img1.jpg")
        val_labels = await manager.get_labels("test-ds", "val", "img2.jpg")
        assert train_labels[0].class_id == 1
        assert val_labels[0].class_id == 1

    async def test_reorder_classes_remaps_annotations(self, manager, sample_image):
        """Reordering classes correctly remaps annotation IDs."""
        await manager.create_dataset("test-ds", ["cat", "dog", "bird"])
        await manager.add_image("test-ds", "train", "img.jpg", sample_image)

        await manager.save_labels("test-ds", "train", "img.jpg", [
            Annotation(class_id=0, x=0.5, y=0.5, width=0.2, height=0.2),
            Annotation(class_id=1, x=0.3, y=0.3, width=0.1, height=0.1),
            Annotation(class_id=2, x=0.7, y=0.7, width=0.3, height=0.3),
        ])

        # Reverse order: bird=0, dog=1, cat=2
        await manager.update_dataset("test-ds", ["bird", "dog", "cat"])

        labels = await manager.get_labels("test-ds", "train", "img.jpg")
        assert len(labels) == 3
        assert labels[0].class_id == 2  # cat: 0 -> 2
        assert labels[1].class_id == 1  # dog: 1 -> 1
        assert labels[2].class_id == 0  # bird: 2 -> 0


# --- Images ---


class TestListImages:
    """Tests for DatasetManager.list_images()."""

    async def test_empty_dataset(self, manager):
        """Return empty list for dataset with no images."""
        await manager.create_dataset("test-ds", ["cat"])
        images = await manager.list_images("test-ds")
        assert images == []

    async def test_filter_by_split(self, manager, sample_image):
        """Filter images by split."""
        await manager.create_dataset("test-ds", ["cat"])
        await manager.add_image("test-ds", "train", "img1.jpg", sample_image)
        await manager.add_image("test-ds", "val", "img2.jpg", sample_image)

        train_images = await manager.list_images("test-ds", split="train")
        assert len(train_images) == 1
        assert train_images[0].filename == "img1.jpg"

    async def test_invalid_split_fails(self, manager):
        """Reject invalid split name."""
        await manager.create_dataset("test-ds", ["cat"])
        with pytest.raises(ValidationError, match="Invalid split"):
            await manager.list_images("test-ds", split="invalid")


class TestAddImage:
    """Tests for DatasetManager.add_image()."""

    async def test_add_success(self, manager, sample_image):
        """Add an image and verify metadata."""
        await manager.create_dataset("test-ds", ["cat"])
        info = await manager.add_image("test-ds", "train", "img.jpg", sample_image)
        assert info.filename == "img.jpg"
        assert info.split == "train"

    async def test_add_publishes_event(self, manager, sample_image, event_bus):
        """Verify dataset.image_added event is published."""
        events = []
        event_bus.subscribe(DATASET_IMAGE_ADDED, lambda data: events.append(data))

        await manager.create_dataset("test-ds", ["cat"])
        await manager.add_image("test-ds", "train", "img.jpg", sample_image)
        assert len(events) == 1
        assert events[0]["filename"] == "img.jpg"

    async def test_add_dataset_not_found(self, manager, sample_image):
        """Raise NotFoundError when dataset doesn't exist."""
        with pytest.raises(NotFoundError):
            await manager.add_image("nonexistent", "train", "img.jpg", sample_image)


class TestDeleteImage:
    """Tests for DatasetManager.delete_image()."""

    async def test_delete_success(self, manager, sample_image):
        """Delete an image and verify it's gone."""
        await manager.create_dataset("test-ds", ["cat"])
        await manager.add_image("test-ds", "train", "img.jpg", sample_image)
        await manager.delete_image("test-ds", "train", "img.jpg")

        images = await manager.list_images("test-ds", split="train")
        assert len(images) == 0

    async def test_delete_also_removes_labels(self, manager, sample_image):
        """Deleting an image also deletes its labels."""
        await manager.create_dataset("test-ds", ["cat"])
        await manager.add_image("test-ds", "train", "img.jpg", sample_image)
        await manager.save_labels("test-ds", "train", "img.jpg", [
            Annotation(class_id=0, x=0.5, y=0.5, width=0.3, height=0.4),
        ])

        await manager.delete_image("test-ds", "train", "img.jpg")
        labels = await manager.get_labels("test-ds", "train", "img.jpg")
        assert labels == []

    async def test_delete_not_found(self, manager):
        """Raise NotFoundError for missing image."""
        await manager.create_dataset("test-ds", ["cat"])
        with pytest.raises(NotFoundError):
            await manager.delete_image("test-ds", "train", "nonexistent.jpg")

    async def test_delete_publishes_event(self, manager, sample_image, event_bus):
        """Verify dataset.image_deleted event is published."""
        events = []
        event_bus.subscribe(DATASET_IMAGE_DELETED, lambda data: events.append(data))

        await manager.create_dataset("test-ds", ["cat"])
        await manager.add_image("test-ds", "train", "img.jpg", sample_image)
        await manager.delete_image("test-ds", "train", "img.jpg")
        assert len(events) == 1
        assert events[0]["filename"] == "img.jpg"


# --- Labels ---


class TestLabels:
    """Tests for DatasetManager.get_labels() and save_labels()."""

    async def test_get_empty_labels(self, manager):
        """Return empty list when no labels exist."""
        await manager.create_dataset("test-ds", ["cat"])
        labels = await manager.get_labels("test-ds", "train", "img.jpg")
        assert labels == []

    async def test_save_and_get_labels(self, manager):
        """Save annotations and retrieve them."""
        await manager.create_dataset("test-ds", ["cat"])
        annotations = [
            Annotation(class_id=0, x=0.5, y=0.5, width=0.3, height=0.4),
            Annotation(class_id=0, x=0.2, y=0.8, width=0.1, height=0.2),
        ]
        await manager.save_labels("test-ds", "train", "img.jpg", annotations)

        result = await manager.get_labels("test-ds", "train", "img.jpg")
        assert len(result) == 2
        assert result[0].class_id == 0
        assert result[0].x == pytest.approx(0.5)

    async def test_save_invalid_annotation_fails(self, manager):
        """Reject annotations with coordinates outside 0-1."""
        await manager.create_dataset("test-ds", ["cat"])
        bad_annotations = [
            Annotation(class_id=0, x=1.5, y=0.5, width=0.3, height=0.4),
        ]
        with pytest.raises(ValidationError, match="must be 0-1"):
            await manager.save_labels("test-ds", "train", "img.jpg", bad_annotations)


# --- Split management ---


class TestChangeSplit:
    """Tests for DatasetManager.change_split()."""

    async def test_move_image(self, manager, sample_image):
        """Move an image from train to val."""
        await manager.create_dataset("test-ds", ["cat"])
        await manager.add_image("test-ds", "train", "img.jpg", sample_image)

        await manager.change_split("test-ds", "img.jpg", "train", "val")

        train_images = await manager.list_images("test-ds", split="train")
        val_images = await manager.list_images("test-ds", split="val")
        assert len(train_images) == 0
        assert len(val_images) == 1

    async def test_move_also_moves_labels(self, manager, sample_image):
        """Moving an image also moves its labels."""
        await manager.create_dataset("test-ds", ["cat"])
        await manager.add_image("test-ds", "train", "img.jpg", sample_image)
        await manager.save_labels("test-ds", "train", "img.jpg", [
            Annotation(class_id=0, x=0.5, y=0.5, width=0.3, height=0.4),
        ])

        await manager.change_split("test-ds", "img.jpg", "train", "val")

        old_labels = await manager.get_labels("test-ds", "train", "img.jpg")
        new_labels = await manager.get_labels("test-ds", "val", "img.jpg")
        assert old_labels == []
        assert len(new_labels) == 1

    async def test_same_split_fails(self, manager, sample_image):
        """Reject moving to the same split."""
        await manager.create_dataset("test-ds", ["cat"])
        await manager.add_image("test-ds", "train", "img.jpg", sample_image)

        with pytest.raises(ValidationError, match="same"):
            await manager.change_split("test-ds", "img.jpg", "train", "train")

    async def test_image_not_found(self, manager):
        """Raise NotFoundError for missing image."""
        await manager.create_dataset("test-ds", ["cat"])
        with pytest.raises(NotFoundError):
            await manager.change_split("test-ds", "nonexistent.jpg", "train", "val")


# --- Prompts ---


class TestPrompts:
    """Tests for DatasetManager.get_prompts() and save_prompts()."""

    async def test_get_empty_prompts(self, manager):
        """Return empty dict when no prompts file exists."""
        await manager.create_dataset("test-ds", ["cat"])
        prompts = await manager.get_prompts("test-ds")
        assert prompts == {}

    async def test_save_and_get_prompts(self, manager):
        """Save prompts and retrieve them."""
        await manager.create_dataset("test-ds", ["cat", "dog"])
        prompts = {0: ["a cat", "feline"], 1: ["a dog", "canine"]}
        await manager.save_prompts("test-ds", prompts)

        result = await manager.get_prompts("test-ds")
        assert result[0] == ["a cat", "feline"]
        assert result[1] == ["a dog", "canine"]


# --- Export / Import ---


class TestExportImport:
    """Tests for DatasetManager.export_dataset() and import_dataset()."""

    async def test_export_and_import(self, manager, sample_image, tmp_path):
        """Export a dataset, then import it under a new name."""
        await manager.create_dataset("original", ["cat", "dog"])
        await manager.add_image("original", "train", "img.jpg", sample_image)

        zip_path = await manager.export_dataset("original", tmp_path / "exports")
        assert zip_path.exists()

        imported = await manager.import_dataset(zip_path, name="imported-copy")
        assert imported.name == "imported-copy"
        assert imported.classes == ["cat", "dog"]

    async def test_export_not_found(self, manager, tmp_path):
        """Raise NotFoundError for missing dataset."""
        with pytest.raises(NotFoundError):
            await manager.export_dataset("nonexistent", tmp_path)

    async def test_import_duplicate_fails(self, manager, sample_image, tmp_path):
        """Raise ConflictError when importing over existing dataset."""
        await manager.create_dataset("test-ds", ["cat"])
        zip_path = await manager.export_dataset("test-ds", tmp_path / "exports")

        with pytest.raises(ConflictError):
            await manager.import_dataset(zip_path, name="test-ds")

    async def test_import_missing_zip(self, manager, tmp_path):
        """Raise NotFoundError for nonexistent zip file."""
        with pytest.raises(NotFoundError):
            await manager.import_dataset(tmp_path / "nope.zip")


# --- DI ---


class TestDI:
    """Tests for dataset module DI functions."""

    def test_get_uninitialized_raises(self):
        """get_dataset_manager() raises RuntimeError before initialization."""
        from backend.dataset import get_dataset_manager, set_dataset_manager
        set_dataset_manager(None)
        with pytest.raises(RuntimeError, match="not initialized"):
            get_dataset_manager()

    def test_set_and_get(self, manager):
        """set/get round-trip works."""
        from backend.dataset import get_dataset_manager, set_dataset_manager
        set_dataset_manager(manager)
        assert get_dataset_manager() is manager
        set_dataset_manager(None)

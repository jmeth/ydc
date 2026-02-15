"""
Dataset manager — business logic for YOLO dataset operations.

Wraps the persistence stores (DatasetStore, ImageStore, LabelStore)
and adds validation, error handling, and event publishing. All
filesystem operations are delegated to the stores.

Usage:
    manager = DatasetManager(dataset_store, image_store, label_store, event_bus)
    info = await manager.create_dataset("my-ds", ["cat", "dog"])
"""

import logging
import re
from pathlib import Path

import numpy as np

from backend.core.events import (
    DATASET_CREATED,
    DATASET_DELETED,
    DATASET_IMAGE_ADDED,
    DATASET_IMAGE_DELETED,
    EventBus,
)
from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from backend.persistence.dataset_store import DatasetStore
from backend.persistence.image_store import ImageStore
from backend.persistence.label_store import LabelStore
from backend.persistence.models import Annotation, DatasetInfo, ImageInfo

logger = logging.getLogger(__name__)

# Regex for valid dataset names: alphanumeric, hyphens, underscores
_VALID_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

# Valid split names
_VALID_SPLITS = {"train", "val", "test"}


class DatasetManager:
    """
    Business logic layer for dataset operations.

    Delegates persistence to injected stores and publishes lifecycle
    events via the EventBus.

    Args:
        dataset_store: Store for dataset CRUD and export/import.
        image_store: Store for image file operations.
        label_store: Store for YOLO label file operations.
        event_bus: Optional EventBus for publishing dataset events.
    """

    def __init__(
        self,
        dataset_store: DatasetStore,
        image_store: ImageStore,
        label_store: LabelStore,
        event_bus: EventBus | None = None,
    ):
        self._datasets = dataset_store
        self._images = image_store
        self._labels = label_store
        self._event_bus = event_bus

    # --- Dataset CRUD ---

    async def create_dataset(self, name: str, classes: list[str]) -> DatasetInfo:
        """
        Create a new dataset with the given name and classes.

        Args:
            name: Dataset identifier (alphanumeric, hyphens, underscores).
            classes: Ordered list of class names.

        Returns:
            DatasetInfo for the newly created dataset.

        Raises:
            ValidationError: If name is invalid or classes is empty.
            ConflictError: If a dataset with this name already exists.
        """
        self._validate_name(name)
        if not classes:
            raise ValidationError("Classes list cannot be empty")

        try:
            info = await self._datasets.create(name, classes)
        except ValueError as e:
            raise ConflictError(str(e))

        await self._publish(DATASET_CREATED, {"name": name, "classes": classes})
        logger.info("Created dataset '%s' with %d classes", name, len(classes))
        return info

    async def list_datasets(self) -> list[DatasetInfo]:
        """
        List all available datasets.

        Returns:
            List of DatasetInfo for every valid dataset found.
        """
        return await self._datasets.list()

    async def get_dataset(self, name: str) -> DatasetInfo:
        """
        Get dataset info by name.

        Args:
            name: Dataset identifier.

        Returns:
            DatasetInfo for the dataset.

        Raises:
            NotFoundError: If no dataset with this name exists.
        """
        info = await self._datasets.get(name)
        if info is None:
            raise NotFoundError("Dataset", name)
        return info

    async def delete_dataset(self, name: str) -> bool:
        """
        Delete a dataset and all its contents.

        Args:
            name: Dataset identifier.

        Returns:
            True if deleted.

        Raises:
            NotFoundError: If no dataset with this name exists.
        """
        deleted = await self._datasets.delete(name)
        if not deleted:
            raise NotFoundError("Dataset", name)

        await self._publish(DATASET_DELETED, {"name": name})
        logger.info("Deleted dataset '%s'", name)
        return True

    async def update_dataset(self, name: str, classes: list[str]) -> DatasetInfo:
        """
        Update dataset classes and remap all annotation class IDs.

        When classes are removed or reordered, existing annotation files
        must be updated so class_id values match the new class list.
        Annotations referencing a removed class are deleted.

        Args:
            name: Dataset identifier.
            classes: New ordered list of class names.

        Returns:
            Updated DatasetInfo.

        Raises:
            NotFoundError: If no dataset with this name exists.
            ValidationError: If classes list is empty.
        """
        if not classes:
            raise ValidationError("Classes list cannot be empty")

        # Get current dataset info (raises NotFoundError if missing)
        info = await self.get_dataset(name)
        old_classes = info.classes

        # Build mapping from old class index to new class index.
        # If an old class name no longer exists, it maps to None (delete).
        new_index_by_name = {c: i for i, c in enumerate(classes)}
        remap: dict[int, int | None] = {}
        for old_idx, old_name in enumerate(old_classes):
            remap[old_idx] = new_index_by_name.get(old_name)

        # Only remap annotations if the mapping actually changes IDs
        needs_remap = any(
            remap.get(i) != i for i in range(len(old_classes))
        ) or len(old_classes) != len(classes)

        if needs_remap:
            await self._remap_annotations(name, remap)

        try:
            await self._datasets.update_classes(name, classes)
        except FileNotFoundError:
            raise NotFoundError("Dataset", name)

        return await self.get_dataset(name)

    async def _remap_annotations(
        self,
        name: str,
        remap: dict[int, int | None],
    ) -> None:
        """
        Remap class IDs in all annotation files for a dataset.

        Iterates every image across all splits, loads its labels,
        applies the ID mapping, and saves. Annotations whose old
        class_id maps to None (class was removed) are deleted.

        Args:
            name: Dataset identifier.
            remap: Mapping from old class_id to new class_id, or None to delete.
        """
        for split in ("train", "val", "test"):
            images = await self._images.list(name, split)
            for img in images:
                annotations = await self._labels.load(name, split, img.filename)
                if not annotations:
                    continue

                updated = []
                for ann in annotations:
                    new_id = remap.get(ann.class_id)
                    if new_id is not None:
                        updated.append(Annotation(
                            class_id=new_id,
                            x=ann.x,
                            y=ann.y,
                            width=ann.width,
                            height=ann.height,
                        ))
                    # else: class was removed, drop this annotation

                await self._labels.save(name, split, img.filename, updated)

    # --- Images ---

    async def list_images(self, name: str, split: str | None = None) -> list[ImageInfo]:
        """
        List images in a dataset, optionally filtered by split.

        Args:
            name: Dataset identifier.
            split: Optional split filter (train, val, test).

        Returns:
            List of ImageInfo for matching images.

        Raises:
            NotFoundError: If the dataset doesn't exist.
            ValidationError: If split is invalid.
        """
        await self.get_dataset(name)
        if split is not None:
            self._validate_split(split)
        return await self._images.list(name, split)

    async def add_image(
        self,
        name: str,
        split: str,
        filename: str,
        image: np.ndarray,
    ) -> ImageInfo:
        """
        Add an image to a dataset split.

        Args:
            name: Dataset identifier.
            split: Target split (train, val, test).
            filename: Image filename.
            image: Image data as numpy array (BGR, HWC format).

        Returns:
            ImageInfo for the saved image.

        Raises:
            NotFoundError: If the dataset doesn't exist.
            ValidationError: If split is invalid.
        """
        await self.get_dataset(name)
        self._validate_split(split)

        info = await self._images.save(name, split, filename, image)
        await self._publish(DATASET_IMAGE_ADDED, {
            "name": name, "split": split, "filename": filename,
        })
        logger.info("Added image '%s' to %s/%s", filename, name, split)
        return info

    async def delete_image(self, name: str, split: str, filename: str) -> None:
        """
        Delete an image and its corresponding labels from a dataset.

        Args:
            name: Dataset identifier.
            split: Source split.
            filename: Image filename.

        Raises:
            NotFoundError: If the dataset or image doesn't exist.
            ValidationError: If split is invalid.
        """
        await self.get_dataset(name)
        self._validate_split(split)

        deleted = await self._images.delete(name, split, filename)
        if not deleted:
            raise NotFoundError("Image", f"{name}/{split}/{filename}")

        # Also delete labels if they exist
        await self._labels.delete(name, split, filename)

        await self._publish(DATASET_IMAGE_DELETED, {
            "name": name, "split": split, "filename": filename,
        })
        logger.info("Deleted image '%s' from %s/%s", filename, name, split)

    # --- Labels ---

    async def get_labels(self, name: str, split: str, filename: str) -> list[Annotation]:
        """
        Get annotations for an image.

        Args:
            name: Dataset identifier.
            split: Source split.
            filename: Image filename.

        Returns:
            List of Annotation objects (empty if no label file).

        Raises:
            NotFoundError: If the dataset doesn't exist.
            ValidationError: If split is invalid.
        """
        await self.get_dataset(name)
        self._validate_split(split)
        return await self._labels.load(name, split, filename)

    async def save_labels(
        self,
        name: str,
        split: str,
        filename: str,
        annotations: list[Annotation],
    ) -> None:
        """
        Save annotations for an image.

        Validates that all coordinates are normalized (0-1) and class_id >= 0.

        Args:
            name: Dataset identifier.
            split: Source split.
            filename: Image filename.
            annotations: YOLO-format annotations to save.

        Raises:
            NotFoundError: If the dataset doesn't exist.
            ValidationError: If split is invalid or annotations are malformed.
        """
        await self.get_dataset(name)
        self._validate_split(split)
        self._validate_annotations(annotations)
        await self._labels.save(name, split, filename, annotations)
        logger.info("Saved %d annotations for %s/%s/%s", len(annotations), name, split, filename)

    # --- Split management ---

    async def change_split(
        self,
        name: str,
        filename: str,
        from_split: str,
        to_split: str,
    ) -> None:
        """
        Move an image (and its labels) between splits.

        Args:
            name: Dataset identifier.
            filename: Image filename.
            from_split: Source split.
            to_split: Destination split.

        Raises:
            NotFoundError: If the dataset or image doesn't exist.
            ValidationError: If either split is invalid or splits are the same.
        """
        await self.get_dataset(name)
        self._validate_split(from_split)
        self._validate_split(to_split)

        if from_split == to_split:
            raise ValidationError(f"Source and destination splits are the same: '{from_split}'")

        # Move image
        try:
            await self._images.move(name, filename, from_split, to_split)
        except FileNotFoundError:
            raise NotFoundError("Image", f"{name}/{from_split}/{filename}")

        # Move labels if they exist
        if await self._labels.exists(name, from_split, filename):
            await self._labels.move(name, filename, from_split, to_split)

        logger.info("Moved '%s' from %s to %s in '%s'", filename, from_split, to_split, name)

    # --- Prompts ---

    async def get_prompts(self, name: str) -> dict[int, list[str]]:
        """
        Get YOLO-World prompts for a dataset.

        Args:
            name: Dataset identifier.

        Returns:
            Mapping of class_id to prompt strings.

        Raises:
            NotFoundError: If the dataset doesn't exist.
        """
        await self.get_dataset(name)
        return await self._datasets.get_prompts(name)

    async def save_prompts(self, name: str, prompts: dict[int, list[str]]) -> None:
        """
        Save YOLO-World prompts for a dataset.

        Args:
            name: Dataset identifier.
            prompts: Mapping of class_id to prompt strings.

        Raises:
            NotFoundError: If the dataset doesn't exist.
        """
        await self.get_dataset(name)
        try:
            await self._datasets.save_prompts(name, prompts)
        except FileNotFoundError:
            raise NotFoundError("Dataset", name)
        logger.info("Saved prompts for dataset '%s'", name)

    # --- Export / Import ---

    async def export_dataset(self, name: str, output_dir: Path) -> Path:
        """
        Export a dataset as a zip file.

        Args:
            name: Dataset identifier.
            output_dir: Directory to write the zip into.

        Returns:
            Path to the created zip file.

        Raises:
            NotFoundError: If the dataset doesn't exist.
        """
        await self.get_dataset(name)
        try:
            return await self._datasets.export_zip(name, output_dir)
        except FileNotFoundError:
            raise NotFoundError("Dataset", name)

    async def import_dataset(self, zip_path: Path, name: str | None = None) -> DatasetInfo:
        """
        Import a dataset from a zip file.

        Args:
            zip_path: Path to the zip file.
            name: Optional override name (defaults to zip filename stem).

        Returns:
            DatasetInfo for the imported dataset.

        Raises:
            ConflictError: If a dataset with the resolved name already exists.
            NotFoundError: If zip_path doesn't exist.
        """
        if not zip_path.exists():
            raise NotFoundError("Zip file", str(zip_path))

        try:
            info = await self._datasets.import_zip(zip_path, name)
        except ValueError as e:
            raise ConflictError(str(e))
        except FileNotFoundError:
            raise NotFoundError("Zip file", str(zip_path))

        await self._publish(DATASET_CREATED, {"name": info.name, "classes": info.classes})
        logger.info("Imported dataset '%s' from zip", info.name)
        return info

    # --- Validation helpers ---

    @staticmethod
    def _validate_name(name: str) -> None:
        """
        Validate a dataset name.

        Args:
            name: Name to validate.

        Raises:
            ValidationError: If name is empty or contains invalid characters.
        """
        if not name:
            raise ValidationError("Dataset name cannot be empty")
        if not _VALID_NAME_RE.match(name):
            raise ValidationError(
                f"Invalid dataset name '{name}': must be alphanumeric with hyphens/underscores, "
                "starting with an alphanumeric character"
            )

    @staticmethod
    def _validate_split(split: str) -> None:
        """
        Validate a split name.

        Args:
            split: Split name to validate.

        Raises:
            ValidationError: If split is not train, val, or test.
        """
        if split not in _VALID_SPLITS:
            raise ValidationError(
                f"Invalid split '{split}': must be one of {sorted(_VALID_SPLITS)}"
            )

    @staticmethod
    def _validate_annotations(annotations: list[Annotation]) -> None:
        """
        Validate annotation coordinates are normalized (0-1) and class_id >= 0.

        Args:
            annotations: Annotations to validate.

        Raises:
            ValidationError: If any annotation has invalid values.
        """
        for i, ann in enumerate(annotations):
            if ann.class_id < 0:
                raise ValidationError(f"Annotation {i}: class_id must be >= 0, got {ann.class_id}")
            for field_name in ("x", "y", "width", "height"):
                val = getattr(ann, field_name)
                if not (0.0 <= val <= 1.0):
                    raise ValidationError(
                        f"Annotation {i}: {field_name} must be 0-1, got {val}"
                    )

    async def _publish(self, event_type: str, data: dict) -> None:
        """
        Publish an event via the EventBus if available.

        Args:
            event_type: Event type constant.
            data: Event payload.
        """
        if self._event_bus is not None:
            await self._event_bus.publish(event_type, data)

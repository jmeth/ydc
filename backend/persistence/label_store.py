"""
Abstract interface for label file operations.

Defines the contract for saving, loading, deleting, and moving
YOLO-format label files within dataset splits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.persistence.models import Annotation


class LabelStore(ABC):
    """Abstract interface for label file operations."""

    @abstractmethod
    async def save(
        self,
        dataset: str,
        split: str,
        filename: str,
        annotations: list[Annotation],
    ) -> None:
        """
        Save annotations in YOLO format for the given image.

        Args:
            dataset: Dataset identifier.
            split: Target split ("train", "val", or "test").
            filename: Image filename (label filename derived by replacing extension with .txt).
            annotations: List of YOLO-format annotations to write.
        """
        ...

    @abstractmethod
    async def load(
        self,
        dataset: str,
        split: str,
        filename: str,
    ) -> list[Annotation]:
        """
        Load annotations from a YOLO-format label file.

        Args:
            dataset: Dataset identifier.
            split: Source split.
            filename: Image filename (label filename derived automatically).

        Returns:
            List of Annotation objects. Empty list if no label file exists.
        """
        ...

    @abstractmethod
    async def delete(self, dataset: str, split: str, filename: str) -> bool:
        """
        Delete the label file for the given image.

        Args:
            dataset: Dataset identifier.
            split: Source split.
            filename: Image filename.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def move(
        self,
        dataset: str,
        filename: str,
        from_split: str,
        to_split: str,
    ) -> None:
        """
        Move a label file between splits within the same dataset.

        Args:
            dataset: Dataset identifier.
            filename: Image filename (label filename derived automatically).
            from_split: Source split.
            to_split: Destination split.

        Raises:
            FileNotFoundError: If the source label file doesn't exist.
        """
        ...

    @abstractmethod
    async def exists(self, dataset: str, split: str, filename: str) -> bool:
        """
        Check if a label file exists for the given image.

        Args:
            dataset: Dataset identifier.
            split: Target split.
            filename: Image filename.

        Returns:
            True if the label file exists.
        """
        ...

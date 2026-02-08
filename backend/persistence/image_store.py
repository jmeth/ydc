"""
Abstract interface for image file operations.

Defines the contract for saving, loading, deleting, moving, and
listing images within dataset splits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from backend.persistence.models import ImageInfo


class ImageStore(ABC):
    """Abstract interface for image file operations."""

    @abstractmethod
    async def save(
        self,
        dataset: str,
        split: str,
        filename: str,
        image: np.ndarray,
    ) -> ImageInfo:
        """
        Save an image to a dataset split.

        Args:
            dataset: Dataset identifier.
            split: Target split ("train", "val", or "test").
            filename: Image filename (e.g. "frame_001.jpg").
            image: Image data as a numpy array (BGR, HWC format).

        Returns:
            ImageInfo with metadata about the saved file.
        """
        ...

    @abstractmethod
    async def load(self, dataset: str, split: str, filename: str) -> np.ndarray:
        """
        Load an image from a dataset split.

        Args:
            dataset: Dataset identifier.
            split: Source split.
            filename: Image filename.

        Returns:
            Image data as a numpy array (BGR, HWC format).

        Raises:
            FileNotFoundError: If the image doesn't exist.
        """
        ...

    @abstractmethod
    async def delete(self, dataset: str, split: str, filename: str) -> bool:
        """
        Delete an image from a dataset split.

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
        Move an image between splits within the same dataset.

        Args:
            dataset: Dataset identifier.
            filename: Image filename.
            from_split: Source split.
            to_split: Destination split.

        Raises:
            FileNotFoundError: If the source image doesn't exist.
        """
        ...

    @abstractmethod
    async def list(
        self,
        dataset: str,
        split: str | None = None,
    ) -> list[ImageInfo]:
        """
        List images in a dataset, optionally filtered by split.

        Note: width and height are set to 0 to avoid loading every image.
        Call load() for actual dimensions.

        Args:
            dataset: Dataset identifier.
            split: Optional split filter. Lists all splits if None.

        Returns:
            List of ImageInfo for matching images.
        """
        ...

    @abstractmethod
    async def exists(self, dataset: str, split: str, filename: str) -> bool:
        """
        Check if an image exists in a dataset split.

        Args:
            dataset: Dataset identifier.
            split: Target split.
            filename: Image filename.

        Returns:
            True if the image file exists.
        """
        ...

"""
Abstract interface for dataset storage operations.

Defines the contract for creating, listing, reading, deleting, and
exporting datasets. Implementations handle the actual filesystem or
database operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from backend.persistence.models import DatasetInfo


class DatasetStore(ABC):
    """Abstract interface for dataset storage operations."""

    @abstractmethod
    async def create(self, name: str, classes: list[str]) -> DatasetInfo:
        """
        Create a new dataset with directory structure and data.yaml.

        Args:
            name: Dataset identifier (used as directory name).
            classes: Ordered list of class names.

        Returns:
            DatasetInfo for the newly created dataset.

        Raises:
            ValueError: If a dataset with this name already exists.
        """
        ...

    @abstractmethod
    async def list(self) -> list[DatasetInfo]:
        """
        List all datasets.

        Returns:
            List of DatasetInfo for every valid dataset found.
        """
        ...

    @abstractmethod
    async def get(self, name: str) -> DatasetInfo | None:
        """
        Get dataset info by name.

        Args:
            name: Dataset identifier.

        Returns:
            DatasetInfo if found, None otherwise.
        """
        ...

    @abstractmethod
    async def delete(self, name: str) -> bool:
        """
        Delete a dataset and all its contents.

        Args:
            name: Dataset identifier.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def update_classes(self, name: str, classes: list[str]) -> None:
        """
        Update dataset classes in data.yaml.

        Args:
            name: Dataset identifier.
            classes: New ordered list of class names.

        Raises:
            FileNotFoundError: If dataset doesn't exist.
        """
        ...

    @abstractmethod
    async def get_prompts(self, name: str) -> dict[int, list[str]]:
        """
        Get YOLO-World prompts from prompts.yaml.

        Args:
            name: Dataset identifier.

        Returns:
            Mapping of class_id → list of prompt strings.
        """
        ...

    @abstractmethod
    async def save_prompts(self, name: str, prompts: dict[int, list[str]]) -> None:
        """
        Save YOLO-World prompts to prompts.yaml.

        Args:
            name: Dataset identifier.
            prompts: Mapping of class_id → list of prompt strings.

        Raises:
            FileNotFoundError: If dataset doesn't exist.
        """
        ...

    @abstractmethod
    async def export_zip(self, name: str, output_path: Path) -> Path:
        """
        Export dataset as a zip file.

        Args:
            name: Dataset identifier.
            output_path: Directory to write the zip file into.

        Returns:
            Path to the created zip file.

        Raises:
            FileNotFoundError: If dataset doesn't exist.
        """
        ...

    @abstractmethod
    async def import_zip(self, zip_path: Path, name: str | None = None) -> DatasetInfo:
        """
        Import a dataset from a zip file.

        Args:
            zip_path: Path to the zip file.
            name: Optional override name (defaults to zip filename stem).

        Returns:
            DatasetInfo for the imported dataset.

        Raises:
            ValueError: If a dataset with the resolved name already exists.
            FileNotFoundError: If zip_path doesn't exist.
        """
        ...

"""
Abstract interface for trained model storage.

Defines the contract for saving, loading, listing, and managing
trained YOLO model weights and their metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from backend.persistence.models import ModelInfo


class ModelStore(ABC):
    """Abstract interface for trained model storage."""

    @abstractmethod
    async def save(
        self,
        name: str,
        weights_path: Path,
        base_model: str = "",
        dataset_name: str = "",
        epochs_completed: int = 0,
        metrics: dict | None = None,
    ) -> ModelInfo:
        """
        Save a trained model with its metadata.

        Copies the weights file into managed storage and records metadata
        in the registry.

        Args:
            name: Unique model identifier.
            weights_path: Path to the source weights file to copy.
            base_model: Name of the base model used for training.
            dataset_name: Name of the dataset used for training.
            epochs_completed: Number of training epochs completed.
            metrics: Optional dict with training metrics (e.g. {"best_map50": 0.85}).

        Returns:
            ModelInfo for the saved model.

        Raises:
            ValueError: If a model with this name already exists.
            FileNotFoundError: If weights_path doesn't exist.
        """
        ...

    @abstractmethod
    async def load(self, name: str) -> Path:
        """
        Get the path to a model's weights file.

        Args:
            name: Model identifier.

        Returns:
            Path to the weights file.

        Raises:
            FileNotFoundError: If the model doesn't exist.
        """
        ...

    @abstractmethod
    async def list(self) -> list[ModelInfo]:
        """
        List all trained models.

        Returns:
            List of ModelInfo for every registered model.
        """
        ...

    @abstractmethod
    async def get(self, name: str) -> ModelInfo | None:
        """
        Get model info by name.

        Args:
            name: Model identifier.

        Returns:
            ModelInfo if found, None otherwise.
        """
        ...

    @abstractmethod
    async def delete(self, name: str) -> bool:
        """
        Delete a model and its files.

        Args:
            name: Model identifier.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def set_active(self, name: str) -> None:
        """
        Set a model as the active model.

        Clears the active flag on all other models.

        Args:
            name: Model identifier.

        Raises:
            FileNotFoundError: If the model doesn't exist.
        """
        ...

    @abstractmethod
    async def get_active(self) -> ModelInfo | None:
        """
        Get the currently active model.

        Returns:
            ModelInfo for the active model, or None if no model is active.
        """
        ...

    @abstractmethod
    async def export_zip(self, name: str, output_dir: Path) -> Path:
        """
        Export a model as a zip archive.

        Creates a zip containing the model weights, training config (if
        present), and a model_meta.json with registry metadata.

        Args:
            name: Model identifier.
            output_dir: Directory where the zip file will be written.

        Returns:
            Path to the created zip file.

        Raises:
            FileNotFoundError: If the model doesn't exist.
        """
        ...

    @abstractmethod
    async def import_zip(self, zip_path: Path, name: str | None = None) -> ModelInfo:
        """
        Import a model from a zip archive.

        Reads model_meta.json for metadata, extracts weights and optional
        training config, and registers the model in the store.

        Args:
            zip_path: Path to the zip archive to import.
            name: Override name for the imported model. If None, uses the
                  name from model_meta.json or the zip filename stem.

        Returns:
            ModelInfo for the newly imported model.

        Raises:
            ValueError: If a model with the resolved name already exists.
            FileNotFoundError: If zip_path doesn't exist or lacks required files.
        """
        ...

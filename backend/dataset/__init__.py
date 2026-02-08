"""
Dataset subsystem — dependency injection.

Module-level DI following the set_inference_manager() pattern.
Set the DatasetManager from app lifespan and retrieve it from API routers.
"""

from __future__ import annotations

from backend.dataset.manager import DatasetManager

_dataset_manager: DatasetManager | None = None


def set_dataset_manager(manager: DatasetManager | None) -> None:
    """
    Set the global DatasetManager instance (called from app lifespan).

    Args:
        manager: DatasetManager instance, or None to clear on shutdown.
    """
    global _dataset_manager
    _dataset_manager = manager


def get_dataset_manager() -> DatasetManager:
    """
    Get the global DatasetManager instance.

    Returns:
        The currently configured DatasetManager.

    Raises:
        RuntimeError: If manager has not been initialized.
    """
    if _dataset_manager is None:
        raise RuntimeError("DatasetManager not initialized — call set_dataset_manager() first")
    return _dataset_manager

"""
Training subsystem — dependency injection.

Module-level DI following the set_dataset_manager() pattern.
Set the TrainingManager from app lifespan and retrieve it from API routers.
"""

from __future__ import annotations

from backend.training.manager import TrainingManager

__all__ = ["TrainingManager", "set_training_manager", "get_training_manager"]

_training_manager: TrainingManager | None = None


def set_training_manager(manager: TrainingManager | None) -> None:
    """
    Set the global TrainingManager instance (called from app lifespan).

    Args:
        manager: TrainingManager instance, or None to clear on shutdown.
    """
    global _training_manager
    _training_manager = manager


def get_training_manager() -> TrainingManager:
    """
    Get the global TrainingManager instance.

    Returns:
        The currently configured TrainingManager.

    Raises:
        RuntimeError: If manager has not been initialized.
    """
    if _training_manager is None:
        raise RuntimeError("TrainingManager not initialized — call set_training_manager() first")
    return _training_manager

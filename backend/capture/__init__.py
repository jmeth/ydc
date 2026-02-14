"""
Capture subsystem — dependency injection.

Module-level DI following the set_dataset_manager() pattern.
Set the CaptureManager from app lifespan and retrieve it from API routers.
"""

from __future__ import annotations

from backend.capture.manager import CaptureManager

_capture_manager: CaptureManager | None = None


def set_capture_manager(manager: CaptureManager | None) -> None:
    """
    Set the global CaptureManager instance (called from app lifespan).

    Args:
        manager: CaptureManager instance, or None to clear on shutdown.
    """
    global _capture_manager
    _capture_manager = manager


def get_capture_manager() -> CaptureManager:
    """
    Get the global CaptureManager instance.

    Returns:
        The currently configured CaptureManager.

    Raises:
        RuntimeError: If manager has not been initialized.
    """
    if _capture_manager is None:
        raise RuntimeError("CaptureManager not initialized — call set_capture_manager() first")
    return _capture_manager

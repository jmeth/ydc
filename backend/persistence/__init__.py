"""
Persistence layer — factory and dependency injection.

Provides a Stores dataclass bundling all four store implementations,
a create_stores() factory, and set_stores()/get_stores() DI accessors
following the same pattern as set_inference_manager().

Usage in app lifespan:
    stores = create_stores(data_dir=Path("datasets"), models_dir=Path("models"))
    set_stores(stores)

Usage in API routers:
    from backend.persistence import get_dataset_store
    ds = get_dataset_store()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.persistence.dataset_store import DatasetStore
from backend.persistence.image_store import ImageStore
from backend.persistence.label_store import LabelStore
from backend.persistence.model_store import ModelStore
from backend.persistence.filesystem.dataset_store import FilesystemDatasetStore
from backend.persistence.filesystem.image_store import FilesystemImageStore
from backend.persistence.filesystem.label_store import FilesystemLabelStore
from backend.persistence.filesystem.model_store import FilesystemModelStore


@dataclass
class Stores:
    """
    Container for all persistence store instances.

    Groups the four stores so they can be passed and wired as a single unit.
    """

    dataset: DatasetStore
    image: ImageStore
    label: LabelStore
    model: ModelStore


def create_stores(data_dir: Path, models_dir: Path) -> Stores:
    """
    Factory that creates filesystem-backed store instances.

    Args:
        data_dir: Root directory for dataset storage.
        models_dir: Root directory for trained model storage.

    Returns:
        Stores containing all four store implementations.
    """
    return Stores(
        dataset=FilesystemDatasetStore(data_dir),
        image=FilesystemImageStore(data_dir),
        label=FilesystemLabelStore(data_dir),
        model=FilesystemModelStore(models_dir),
    )


# --- Dependency injection (module-level singleton) ---

_stores: Stores | None = None


def set_stores(stores: Stores | None) -> None:
    """
    Set the global Stores instance (called from app lifespan).

    Args:
        stores: Stores instance, or None to clear on shutdown.
    """
    global _stores
    _stores = stores


def get_stores() -> Stores:
    """
    Get the global Stores instance.

    Returns:
        The currently configured Stores.

    Raises:
        RuntimeError: If stores have not been initialized.
    """
    if _stores is None:
        raise RuntimeError("Persistence stores not initialized — call set_stores() first")
    return _stores


def get_dataset_store() -> DatasetStore:
    """Convenience accessor for the DatasetStore."""
    return get_stores().dataset


def get_image_store() -> ImageStore:
    """Convenience accessor for the ImageStore."""
    return get_stores().image


def get_label_store() -> LabelStore:
    """Convenience accessor for the LabelStore."""
    return get_stores().label


def get_model_store() -> ModelStore:
    """Convenience accessor for the ModelStore."""
    return get_stores().model

"""
Domain models for the persistence layer.

Dataclasses used by all stores to represent datasets, images, labels,
and trained models. No external dependencies — pure data containers.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Annotation:
    """
    Single YOLO-format annotation.

    Coordinates are normalized (0-1) relative to image dimensions.
    Fields: class_id, x_center, y_center, width, height.
    """

    class_id: int
    x: float
    y: float
    width: float
    height: float


@dataclass
class DatasetInfo:
    """
    Metadata about a dataset.

    Attributes:
        name: Dataset identifier (directory name).
        path: Absolute path to the dataset root.
        classes: Ordered list of class names.
        num_images: Mapping of split name → image count.
        created_at: Unix timestamp of creation.
        modified_at: Unix timestamp of last modification.
    """

    name: str
    path: Path
    classes: list[str]
    num_images: dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0, "test": 0})
    created_at: float = 0.0
    modified_at: float = 0.0


@dataclass
class ImageInfo:
    """
    Metadata about a single image file.

    Attributes:
        filename: Image filename (e.g. "frame_001.jpg").
        split: Dataset split ("train", "val", or "test").
        path: Absolute path to the image file.
        width: Image width in pixels (0 if not loaded).
        height: Image height in pixels (0 if not loaded).
        size_bytes: File size on disk.
        has_labels: Whether a corresponding label file exists.
    """

    filename: str
    split: str
    path: Path
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    has_labels: bool = False


@dataclass
class LabelData:
    """
    Label data for a single image.

    Attributes:
        filename: Image filename this label corresponds to.
        split: Dataset split ("train", "val", or "test").
        annotations: List of YOLO-format annotations.
    """

    filename: str
    split: str
    annotations: list[Annotation] = field(default_factory=list)


@dataclass
class ModelInfo:
    """
    Metadata about a trained model.

    Attributes:
        name: Unique model identifier.
        path: Path to the model weights file.
        base_model: Name of the base model used for training.
        dataset_name: Name of the dataset used for training.
        created_at: Unix timestamp of when training completed.
        epochs_completed: Number of training epochs completed.
        best_map50: Best mAP@50 metric achieved (None if not evaluated).
        is_active: Whether this is the currently active model.
    """

    name: str
    path: Path
    base_model: str = ""
    dataset_name: str = ""
    created_at: float = 0.0
    epochs_completed: int = 0
    best_map50: float | None = None
    is_active: bool = False

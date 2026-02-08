"""
Inference domain models.

Defines the ModelType enum for supported model categories and the
LoadedModel dataclass that wraps an ultralytics YOLO model instance
with its metadata (type, name, class mapping, prompts).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelType(str, Enum):
    """Supported inference model categories."""
    YOLO_WORLD = "yolo_world"    # Zero-shot with text prompts
    FINE_TUNED = "fine_tuned"    # Custom .pt file with trained weights


@dataclass
class LoadedModel:
    """
    A loaded YOLO model ready for inference.

    Attributes:
        model_type: Whether this is a YOLO-World or fine-tuned model.
        model: The ultralytics YOLO model instance.
        model_name: Model file or identifier string (e.g. "yolo11n").
        classes: Mapping of class_id -> class_name from the model.
        prompts: Text prompts for YOLO-World models (None for fine-tuned).
    """
    model_type: ModelType
    model: Any  # ultralytics.YOLO instance
    model_name: str
    classes: dict[int, str] = field(default_factory=dict)
    prompts: list[str] | None = None

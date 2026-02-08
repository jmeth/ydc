"""
Inference subsystem — YOLO model loading, detection sessions, and management.

Public API:
    InferenceManager: Top-level coordinator for inference sessions.
    InferenceSession: Per-feed detection runner.
    ModelLoader: Load and cache ultralytics YOLO models.
    ModelType: Enum of supported model categories.
    LoadedModel: Dataclass wrapping a loaded model with metadata.
"""

from backend.inference.loader import ModelLoader
from backend.inference.manager import InferenceManager
from backend.inference.models import LoadedModel, ModelType
from backend.inference.session import InferenceSession

__all__ = [
    "InferenceManager",
    "InferenceSession",
    "ModelLoader",
    "ModelType",
    "LoadedModel",
]

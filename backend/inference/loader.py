"""
Model loader for YOLO inference models.

Handles loading both YOLO-World (zero-shot) and fine-tuned models
via the ultralytics library. Maintains a cache keyed by (model_name,
prompts) to avoid redundant loads of the same configuration.
"""

import logging
from typing import Any

from backend.inference.models import LoadedModel, ModelType

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and caches ultralytics YOLO models.

    For YOLO-World models, applies text prompts via model.set_classes().
    For fine-tuned models, reads the class mapping from model.names.
    Cached models are returned immediately on repeated load() calls
    with the same model_name and prompts.
    """

    def __init__(self):
        # Cache keyed by (model_name, tuple(prompts or []))
        self._cache: dict[tuple[str, tuple[str, ...]], LoadedModel] = {}

    def load(
        self,
        model_name: str,
        model_type: ModelType,
        prompts: list[str] | None = None,
    ) -> LoadedModel:
        """
        Load a YOLO model, returning a cached instance if available.

        Args:
            model_name: Model file path or identifier (e.g. "yolo11n").
            model_type: Whether this is a YOLO-World or fine-tuned model.
            prompts: Text prompts for YOLO-World zero-shot detection.

        Returns:
            LoadedModel wrapping the ultralytics YOLO instance.
        """
        cache_key = (model_name, tuple(prompts or []))

        if cache_key in self._cache:
            logger.debug("Model cache hit: %s", model_name)
            return self._cache[cache_key]

        model = self._load_ultralytics(model_name)
        classes: dict[int, str] = {}

        if model_type == ModelType.YOLO_WORLD and prompts:
            model.set_classes(prompts)
            classes = {i: name for i, name in enumerate(prompts)}
            logger.info("YOLO-World model loaded: %s with %d prompts", model_name, len(prompts))
        else:
            # Fine-tuned model — read class names from the model
            names = getattr(model, "names", {})
            classes = dict(names) if names else {}
            logger.info("Fine-tuned model loaded: %s with %d classes", model_name, len(classes))

        loaded = LoadedModel(
            model_type=model_type,
            model=model,
            model_name=model_name,
            classes=classes,
            prompts=prompts,
        )
        self._cache[cache_key] = loaded
        return loaded

    def _load_ultralytics(self, model_name: str) -> Any:
        """
        Import and instantiate an ultralytics YOLO model.

        Separated for testability — tests mock this method or the
        ultralytics import.

        Args:
            model_name: Model file path or identifier.

        Returns:
            An ultralytics YOLO model instance.
        """
        from ultralytics import YOLO
        return YOLO(model_name)

    def clear_cache(self) -> None:
        """Remove all cached models."""
        self._cache.clear()
        logger.debug("Model cache cleared")

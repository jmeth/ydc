"""
Unit tests for inference domain models (ModelType, LoadedModel).

Verifies enum values, dataclass defaults, and field assignments.
"""

from unittest.mock import MagicMock

from backend.inference.models import LoadedModel, ModelType


class TestModelType:
    """Tests for the ModelType enum."""

    def test_yolo_world_value(self):
        """YOLO_WORLD has the expected string value."""
        assert ModelType.YOLO_WORLD == "yolo_world"
        assert ModelType.YOLO_WORLD.value == "yolo_world"

    def test_fine_tuned_value(self):
        """FINE_TUNED has the expected string value."""
        assert ModelType.FINE_TUNED == "fine_tuned"
        assert ModelType.FINE_TUNED.value == "fine_tuned"

    def test_is_str_subclass(self):
        """ModelType values are usable as plain strings."""
        assert isinstance(ModelType.YOLO_WORLD, str)

    def test_all_members(self):
        """Exactly two model types exist."""
        assert set(ModelType) == {ModelType.YOLO_WORLD, ModelType.FINE_TUNED}


class TestLoadedModel:
    """Tests for the LoadedModel dataclass."""

    def test_defaults(self):
        """Defaults are empty classes and None prompts."""
        mock_model = MagicMock()
        loaded = LoadedModel(
            model_type=ModelType.YOLO_WORLD,
            model=mock_model,
            model_name="test-model",
        )
        assert loaded.classes == {}
        assert loaded.prompts is None
        assert loaded.model_name == "test-model"
        assert loaded.model is mock_model

    def test_with_prompts(self):
        """Prompts are stored when provided."""
        loaded = LoadedModel(
            model_type=ModelType.YOLO_WORLD,
            model=MagicMock(),
            model_name="yolov8s-worldv2",
            classes={0: "person", 1: "car"},
            prompts=["person", "car"],
        )
        assert loaded.prompts == ["person", "car"]
        assert loaded.classes == {0: "person", 1: "car"}

    def test_fine_tuned_no_prompts(self):
        """Fine-tuned models typically have no prompts."""
        loaded = LoadedModel(
            model_type=ModelType.FINE_TUNED,
            model=MagicMock(),
            model_name="custom.pt",
            classes={0: "widget", 1: "gadget"},
        )
        assert loaded.model_type == ModelType.FINE_TUNED
        assert loaded.prompts is None
        assert len(loaded.classes) == 2

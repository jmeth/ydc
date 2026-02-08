"""
Unit tests for ModelLoader.

All ultralytics imports are mocked so tests run without the real
ultralytics package installed.
"""

from unittest.mock import MagicMock, patch

from backend.inference.loader import ModelLoader
from backend.inference.models import LoadedModel, ModelType


class TestModelLoader:
    """Tests for ModelLoader load/cache behaviour."""

    def _make_mock_model(self, names=None):
        """Create a mock ultralytics YOLO model."""
        mock = MagicMock()
        mock.names = names or {}
        return mock

    @patch.object(ModelLoader, "_load_ultralytics")
    def test_load_yolo_world_sets_classes(self, mock_load):
        """YOLO-World load calls set_classes with prompts."""
        mock_model = self._make_mock_model()
        mock_load.return_value = mock_model

        loader = ModelLoader()
        result = loader.load("yolov8s-worldv2", ModelType.YOLO_WORLD, prompts=["person", "car"])

        assert isinstance(result, LoadedModel)
        assert result.model_type == ModelType.YOLO_WORLD
        assert result.model_name == "yolov8s-worldv2"
        assert result.prompts == ["person", "car"]
        assert result.classes == {0: "person", 1: "car"}
        mock_model.set_classes.assert_called_once_with(["person", "car"])

    @patch.object(ModelLoader, "_load_ultralytics")
    def test_load_fine_tuned_reads_names(self, mock_load):
        """Fine-tuned load reads class names from model.names."""
        mock_model = self._make_mock_model(names={0: "widget", 1: "gadget"})
        mock_load.return_value = mock_model

        loader = ModelLoader()
        result = loader.load("custom.pt", ModelType.FINE_TUNED)

        assert result.model_type == ModelType.FINE_TUNED
        assert result.classes == {0: "widget", 1: "gadget"}
        assert result.prompts is None
        mock_model.set_classes.assert_not_called()

    @patch.object(ModelLoader, "_load_ultralytics")
    def test_cache_hit(self, mock_load):
        """Second load with same name+prompts returns cached instance."""
        mock_load.return_value = self._make_mock_model()

        loader = ModelLoader()
        first = loader.load("model.pt", ModelType.YOLO_WORLD, prompts=["cat"])
        second = loader.load("model.pt", ModelType.YOLO_WORLD, prompts=["cat"])

        assert first is second
        assert mock_load.call_count == 1  # Only loaded once

    @patch.object(ModelLoader, "_load_ultralytics")
    def test_different_prompts_no_cache_hit(self, mock_load):
        """Different prompts for the same model name cause a new load."""
        mock_load.return_value = self._make_mock_model()

        loader = ModelLoader()
        first = loader.load("model.pt", ModelType.YOLO_WORLD, prompts=["cat"])
        mock_load.return_value = self._make_mock_model()
        second = loader.load("model.pt", ModelType.YOLO_WORLD, prompts=["dog"])

        assert first is not second
        assert mock_load.call_count == 2

    @patch.object(ModelLoader, "_load_ultralytics")
    def test_clear_cache(self, mock_load):
        """clear_cache empties the internal cache."""
        mock_load.return_value = self._make_mock_model()

        loader = ModelLoader()
        loader.load("model.pt", ModelType.FINE_TUNED)
        assert len(loader._cache) == 1

        loader.clear_cache()
        assert len(loader._cache) == 0

    @patch.object(ModelLoader, "_load_ultralytics")
    def test_yolo_world_no_prompts_reads_names(self, mock_load):
        """YOLO-World without prompts falls back to reading model.names."""
        mock_model = self._make_mock_model(names={0: "object"})
        mock_load.return_value = mock_model

        loader = ModelLoader()
        result = loader.load("yolov8s-worldv2", ModelType.YOLO_WORLD, prompts=None)

        assert result.classes == {0: "object"}
        mock_model.set_classes.assert_not_called()

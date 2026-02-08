"""
Unit tests for InferenceSession.

Uses mocked ultralytics model and FeedManager to verify that on_frame
runs detection, filters by confidence, produces InferenceFrame objects,
pushes to the derived feed, and tracks timing stats.
"""

from unittest.mock import MagicMock, call

import numpy as np

from backend.feeds.models import Detection, Frame, InferenceFrame
from backend.inference.models import LoadedModel, ModelType
from backend.inference.session import InferenceSession


def _make_mock_boxes(detections: list[tuple[int, float, list[float]]]):
    """
    Build a mock boxes object matching ultralytics Results.boxes format.

    Args:
        detections: List of (class_id, confidence, [x1, y1, x2, y2]) tuples.

    Returns:
        Mock boxes object with .cls, .conf, .xyxy, and __len__.
    """
    boxes = MagicMock()
    n = len(detections)
    boxes.__len__ = MagicMock(return_value=n)

    cls_vals = [d[0] for d in detections]
    conf_vals = [d[1] for d in detections]
    xyxy_vals = [d[2] for d in detections]

    # Make indexable arrays
    boxes.cls = cls_vals
    boxes.conf = conf_vals

    # xyxy[i].tolist() must work
    xyxy_mocks = []
    for bbox in xyxy_vals:
        m = MagicMock()
        m.tolist.return_value = bbox
        xyxy_mocks.append(m)
    boxes.xyxy = xyxy_mocks

    return boxes


def _make_model(classes: dict[int, str], detections=None):
    """Create a LoadedModel with a mock YOLO model that returns given detections."""
    mock_yolo = MagicMock()

    if detections is not None:
        result = MagicMock()
        result.boxes = _make_mock_boxes(detections)
        mock_yolo.return_value = [result]
    else:
        result = MagicMock()
        result.boxes = _make_mock_boxes([])
        mock_yolo.return_value = [result]

    return LoadedModel(
        model_type=ModelType.YOLO_WORLD,
        model=mock_yolo,
        model_name="test-model",
        classes=classes,
        prompts=list(classes.values()),
    )


def _make_frame(frame_number: int = 0) -> Frame:
    """Create a dummy Frame with a small numpy array."""
    return Frame(data=np.zeros((480, 640, 3), dtype=np.uint8), frame_number=frame_number)


class TestInferenceSession:
    """Tests for InferenceSession.on_frame and stats tracking."""

    def test_on_frame_produces_inference_frame(self):
        """on_frame pushes an InferenceFrame to the derived feed."""
        classes = {0: "person", 1: "car"}
        model = _make_model(classes, detections=[
            (0, 0.9, [10.0, 20.0, 100.0, 200.0]),
            (1, 0.7, [50.0, 60.0, 150.0, 250.0]),
        ])
        fm = MagicMock()
        fm.push_derived_frame.return_value = True

        session = InferenceSession(
            output_feed_id="out-123",
            source_feed_id="src-456",
            model=model,
            feed_manager=fm,
        )

        frame = _make_frame(frame_number=42)
        session.on_frame("src-456", frame)

        # Verify push_derived_frame was called
        assert fm.push_derived_frame.call_count == 1
        call_args = fm.push_derived_frame.call_args
        assert call_args[0][0] == "out-123"

        inf_frame = call_args[0][1]
        assert isinstance(inf_frame, InferenceFrame)
        assert inf_frame.frame is frame
        assert inf_frame.model_name == "test-model"
        assert len(inf_frame.detections) == 2
        assert inf_frame.detections[0].class_name == "person"
        assert inf_frame.detections[0].confidence == 0.9
        assert inf_frame.detections[1].class_name == "car"

    def test_confidence_filtering(self):
        """Detections below the threshold are excluded."""
        classes = {0: "person", 1: "car"}
        model = _make_model(classes, detections=[
            (0, 0.9, [10.0, 20.0, 100.0, 200.0]),  # Above threshold
            (1, 0.1, [50.0, 60.0, 150.0, 250.0]),  # Below 0.3 threshold
        ])
        fm = MagicMock()
        fm.push_derived_frame.return_value = True

        session = InferenceSession(
            output_feed_id="out",
            source_feed_id="src",
            model=model,
            feed_manager=fm,
            confidence_threshold=0.3,
        )

        session.on_frame("src", _make_frame())

        inf_frame = fm.push_derived_frame.call_args[0][1]
        assert len(inf_frame.detections) == 1
        assert inf_frame.detections[0].class_name == "person"

    def test_custom_confidence_threshold(self):
        """Higher threshold filters more aggressively."""
        classes = {0: "person"}
        model = _make_model(classes, detections=[
            (0, 0.6, [10.0, 20.0, 100.0, 200.0]),
            (0, 0.8, [50.0, 60.0, 150.0, 250.0]),
        ])
        fm = MagicMock()
        fm.push_derived_frame.return_value = True

        session = InferenceSession(
            output_feed_id="out",
            source_feed_id="src",
            model=model,
            feed_manager=fm,
            confidence_threshold=0.7,
        )

        session.on_frame("src", _make_frame())

        inf_frame = fm.push_derived_frame.call_args[0][1]
        assert len(inf_frame.detections) == 1
        assert inf_frame.detections[0].confidence == 0.8

    def test_no_detections(self):
        """Frame with no detections still pushes an InferenceFrame with empty list."""
        model = _make_model({}, detections=[])
        fm = MagicMock()
        fm.push_derived_frame.return_value = True

        session = InferenceSession(
            output_feed_id="out",
            source_feed_id="src",
            model=model,
            feed_manager=fm,
        )

        session.on_frame("src", _make_frame())

        inf_frame = fm.push_derived_frame.call_args[0][1]
        assert inf_frame.detections == []

    def test_stats_tracking(self):
        """frames_processed, avg_inference_ms, last_inference_ms update correctly."""
        model = _make_model({0: "person"}, detections=[
            (0, 0.9, [10.0, 20.0, 100.0, 200.0]),
        ])
        fm = MagicMock()
        fm.push_derived_frame.return_value = True

        session = InferenceSession(
            output_feed_id="out",
            source_feed_id="src",
            model=model,
            feed_manager=fm,
        )

        assert session.frames_processed == 0
        assert session.avg_inference_ms == 0.0
        assert session.last_inference_ms == 0.0

        session.on_frame("src", _make_frame(0))
        assert session.frames_processed == 1
        assert session.last_inference_ms > 0.0
        assert session.avg_inference_ms > 0.0

        session.on_frame("src", _make_frame(1))
        assert session.frames_processed == 2

    def test_model_called_with_frame_data(self):
        """The YOLO model is called with frame.data and verbose=False."""
        mock_yolo = MagicMock()
        result = MagicMock()
        result.boxes = _make_mock_boxes([])
        mock_yolo.return_value = [result]

        model = LoadedModel(
            model_type=ModelType.YOLO_WORLD,
            model=mock_yolo,
            model_name="test",
            classes={},
        )
        fm = MagicMock()
        fm.push_derived_frame.return_value = True

        session = InferenceSession(
            output_feed_id="out",
            source_feed_id="src",
            model=model,
            feed_manager=fm,
        )

        frame = _make_frame()
        session.on_frame("src", frame)

        mock_yolo.assert_called_once()
        call_args = mock_yolo.call_args
        np.testing.assert_array_equal(call_args[0][0], frame.data)
        assert call_args[1]["verbose"] is False

    def test_inference_error_does_not_crash(self):
        """If the model raises, on_frame logs and returns without pushing."""
        mock_yolo = MagicMock(side_effect=RuntimeError("GPU error"))
        model = LoadedModel(
            model_type=ModelType.YOLO_WORLD,
            model=mock_yolo,
            model_name="broken",
            classes={},
        )
        fm = MagicMock()

        session = InferenceSession(
            output_feed_id="out",
            source_feed_id="src",
            model=model,
            feed_manager=fm,
        )

        # Should not raise
        session.on_frame("src", _make_frame())

        fm.push_derived_frame.assert_not_called()
        assert session.frames_processed == 0

    def test_unknown_class_id_fallback(self):
        """Unknown class IDs get a 'class_N' fallback name."""
        classes = {0: "person"}  # class 1 not in mapping
        model = _make_model(classes, detections=[
            (1, 0.9, [10.0, 20.0, 100.0, 200.0]),
        ])
        fm = MagicMock()
        fm.push_derived_frame.return_value = True

        session = InferenceSession(
            output_feed_id="out",
            source_feed_id="src",
            model=model,
            feed_manager=fm,
        )

        session.on_frame("src", _make_frame())

        inf_frame = fm.push_derived_frame.call_args[0][1]
        assert inf_frame.detections[0].class_name == "class_1"

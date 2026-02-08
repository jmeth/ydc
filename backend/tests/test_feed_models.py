"""
Unit tests for feed data models.

Tests enum values, dataclass defaults, and type relationships
for the feeds subsystem models.
"""

import numpy as np

from backend.feeds.models import (
    FeedType,
    FeedStatus,
    Frame,
    Detection,
    InferenceFrame,
    FeedConfig,
    FeedInfo,
    DerivedFeed,
)


class TestFeedType:
    """Tests for FeedType enum."""

    def test_camera_value(self):
        """CAMERA enum has string value 'camera'."""
        assert FeedType.CAMERA == "camera"
        assert FeedType.CAMERA.value == "camera"

    def test_rtsp_value(self):
        """RTSP enum has string value 'rtsp'."""
        assert FeedType.RTSP == "rtsp"

    def test_file_value(self):
        """FILE enum has string value 'file'."""
        assert FeedType.FILE == "file"

    def test_str_enum_comparison(self):
        """FeedType values compare equal to plain strings."""
        assert FeedType.CAMERA == "camera"
        assert FeedType("camera") == FeedType.CAMERA

    def test_all_types(self):
        """All expected feed types exist."""
        types = {t.value for t in FeedType}
        assert types == {"camera", "rtsp", "file"}


class TestFeedStatus:
    """Tests for FeedStatus enum."""

    def test_all_statuses(self):
        """All expected statuses exist with correct values."""
        statuses = {s.value for s in FeedStatus}
        assert statuses == {"connecting", "active", "paused", "error", "disconnected"}

    def test_str_comparison(self):
        """FeedStatus values compare equal to plain strings."""
        assert FeedStatus.ACTIVE == "active"
        assert FeedStatus.PAUSED == "paused"


class TestFrame:
    """Tests for Frame dataclass."""

    def test_default_values(self):
        """Frame has auto-generated timestamp and frame_number=0."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(data=data)
        assert frame.frame_number == 0
        assert frame.timestamp > 0
        assert frame.data.shape == (480, 640, 3)

    def test_explicit_values(self):
        """Frame accepts explicit timestamp and frame_number."""
        data = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = Frame(data=data, timestamp=12345.0, frame_number=42)
        assert frame.timestamp == 12345.0
        assert frame.frame_number == 42


class TestDetection:
    """Tests for Detection dataclass."""

    def test_basic_detection(self):
        """Detection stores class, confidence, bbox, and class_id."""
        det = Detection(
            class_name="person",
            confidence=0.95,
            bbox=[10.0, 20.0, 100.0, 200.0],
            class_id=0,
        )
        assert det.class_name == "person"
        assert det.confidence == 0.95
        assert det.bbox == [10.0, 20.0, 100.0, 200.0]

    def test_default_class_id(self):
        """Detection defaults class_id to 0."""
        det = Detection(class_name="cat", confidence=0.8, bbox=[0, 0, 50, 50])
        assert det.class_id == 0


class TestInferenceFrame:
    """Tests for InferenceFrame dataclass."""

    def test_defaults(self):
        """InferenceFrame defaults to empty detections and zero inference time."""
        data = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = Frame(data=data)
        inf = InferenceFrame(frame=frame)
        assert inf.detections == []
        assert inf.model_name == ""
        assert inf.inference_time_ms == 0.0

    def test_with_detections(self):
        """InferenceFrame stores a list of Detection objects."""
        data = np.zeros((100, 100, 3), dtype=np.uint8)
        frame = Frame(data=data)
        dets = [Detection("cat", 0.9, [0, 0, 50, 50])]
        inf = InferenceFrame(frame=frame, detections=dets, model_name="yolov8n")
        assert len(inf.detections) == 1
        assert inf.model_name == "yolov8n"


class TestFeedConfig:
    """Tests for FeedConfig dataclass."""

    def test_required_fields(self):
        """FeedConfig requires feed_type and source."""
        cfg = FeedConfig(feed_type=FeedType.CAMERA, source="0")
        assert cfg.feed_type == FeedType.CAMERA
        assert cfg.source == "0"

    def test_defaults(self):
        """FeedConfig has sensible defaults for name and buffer_size."""
        cfg = FeedConfig(feed_type=FeedType.CAMERA, source="0")
        assert cfg.name == ""
        assert cfg.buffer_size == 30

    def test_custom_values(self):
        """FeedConfig accepts custom name and buffer_size."""
        cfg = FeedConfig(
            feed_type=FeedType.RTSP,
            source="rtsp://example.com/stream",
            name="Office Cam",
            buffer_size=60,
        )
        assert cfg.name == "Office Cam"
        assert cfg.buffer_size == 60


class TestFeedInfo:
    """Tests for FeedInfo dataclass."""

    def test_defaults(self):
        """FeedInfo defaults to disconnected, zero fps, no resolution."""
        cfg = FeedConfig(feed_type=FeedType.CAMERA, source="0")
        info = FeedInfo(feed_id="feed-1", config=cfg)
        assert info.status == FeedStatus.DISCONNECTED
        assert info.fps == 0.0
        assert info.resolution is None
        assert info.frame_count == 0

    def test_active_feed(self):
        """FeedInfo reflects active feed state."""
        cfg = FeedConfig(feed_type=FeedType.CAMERA, source="0")
        info = FeedInfo(
            feed_id="feed-1",
            config=cfg,
            status=FeedStatus.ACTIVE,
            fps=30.0,
            resolution=(1920, 1080),
            frame_count=150,
        )
        assert info.status == FeedStatus.ACTIVE
        assert info.resolution == (1920, 1080)


class TestDerivedFeed:
    """Tests for DerivedFeed dataclass."""

    def test_defaults(self):
        """DerivedFeed has default type 'inference' and buffer_size=30."""
        df = DerivedFeed(feed_id="inf-1", source_feed_id="feed-1")
        assert df.feed_type == "inference"
        assert df.buffer_size == 30

    def test_custom_type(self):
        """DerivedFeed accepts a custom feed_type."""
        df = DerivedFeed(feed_id="aug-1", source_feed_id="feed-1", feed_type="augmented")
        assert df.feed_type == "augmented"

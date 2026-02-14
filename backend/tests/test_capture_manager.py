"""
Unit tests for the CaptureManager.

Tests cover both raw and inference capture modes, interval timing,
positive/negative sampling, manual trigger, config updates, and
detection-to-annotation conversion. All external dependencies
(FeedManager, DatasetManager) are mocked.
"""

import asyncio

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.capture.manager import CaptureManager, detections_to_annotations
from backend.core.exceptions import ConflictError, NotFoundError
from backend.feeds.models import (
    Detection,
    FeedConfig,
    FeedInfo,
    FeedStatus,
    FeedType,
    Frame,
    InferenceFrame,
)
from backend.persistence.models import Annotation, DatasetInfo, ImageInfo
from pathlib import Path


# --- Helpers ---

def _make_feed_manager(is_derived: bool = False, feed_exists: bool = True):
    """Create a mock FeedManager with configurable behavior."""
    fm = MagicMock()
    fm.is_derived_feed.return_value = is_derived
    if feed_exists:
        fm.get_feed_info.return_value = FeedInfo(
            feed_id="test-feed",
            config=FeedConfig(feed_type=FeedType.CAMERA, source="0", name="cam0"),
            status=FeedStatus.ACTIVE,
        )
    else:
        fm.get_feed_info.return_value = None
    fm.subscribe.return_value = True
    fm.unsubscribe.return_value = True
    fm.get_frame.return_value = None
    return fm


def _make_dataset_manager():
    """Create a mock DatasetManager with standard success returns."""
    dm = MagicMock()
    dm.get_dataset = AsyncMock(return_value=DatasetInfo(
        name="test-ds", path=Path("/tmp/test-ds"), classes=["cat", "dog"],
    ))
    dm.add_image = AsyncMock(return_value=ImageInfo(
        filename="capture_test.jpg", split="train", path=Path("/tmp/test.jpg"),
        width=640, height=480,
    ))
    dm.save_labels = AsyncMock()
    return dm


def _make_event_bus():
    """Create a mock EventBus."""
    eb = MagicMock()
    eb.publish = AsyncMock()
    return eb


def _make_frame(width: int = 640, height: int = 480) -> Frame:
    """Create a test Frame with a numpy array."""
    return Frame(data=np.zeros((height, width, 3), dtype=np.uint8), frame_number=0)


def _make_inference_frame(
    detections: list[Detection] | None = None,
    width: int = 640,
    height: int = 480,
) -> InferenceFrame:
    """Create a test InferenceFrame with optional detections."""
    if detections is None:
        detections = [
            Detection(class_name="cat", confidence=0.9, bbox=[100, 100, 200, 200], class_id=0),
        ]
    frame = _make_frame(width, height)
    return InferenceFrame(frame=frame, detections=detections, model_name="yolov8n")


# --- Start / Stop tests ---

class TestStartStop:
    """Tests for CaptureManager.start() and stop()."""

    async def test_start_subscribes_to_feed(self):
        """start() subscribes to the feed via FeedManager."""
        fm = _make_feed_manager()
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        await mgr.start("test-feed", "test-ds")

        fm.subscribe.assert_called_once_with("test-feed", mgr._on_frame)

    async def test_start_inference_mode(self):
        """start() detects inference mode for derived feeds."""
        fm = _make_feed_manager(is_derived=True)
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        result = await mgr.start("derived-feed", "test-ds")

        assert result["mode"] == "inference"
        assert result["status"] == "running"

    async def test_start_raw_mode(self):
        """start() detects raw mode for non-derived feeds."""
        fm = _make_feed_manager(is_derived=False)
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        result = await mgr.start("raw-feed", "test-ds")

        assert result["mode"] == "raw"
        assert result["status"] == "running"

    async def test_start_nonexistent_feed_raises(self):
        """start() raises NotFoundError for non-existent raw feed."""
        fm = _make_feed_manager(feed_exists=False)
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        with pytest.raises(NotFoundError):
            await mgr.start("bad-feed", "test-ds")

    async def test_start_nonexistent_dataset_raises(self):
        """start() raises NotFoundError for non-existent dataset."""
        fm = _make_feed_manager()
        dm = _make_dataset_manager()
        dm.get_dataset = AsyncMock(side_effect=NotFoundError("Dataset", "no-ds"))
        mgr = CaptureManager(fm, dm, _make_event_bus())

        with pytest.raises(NotFoundError):
            await mgr.start("test-feed", "no-ds")

    async def test_start_conflict_already_running(self):
        """start() raises ConflictError if already capturing."""
        fm = _make_feed_manager()
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        await mgr.start("test-feed", "test-ds")
        with pytest.raises(ConflictError):
            await mgr.start("test-feed", "test-ds")

    async def test_stop_unsubscribes(self):
        """stop() unsubscribes from the feed."""
        fm = _make_feed_manager()
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        await mgr.start("test-feed", "test-ds")
        await mgr.stop()

        fm.unsubscribe.assert_called_once_with("test-feed", mgr._on_frame)

    async def test_stop_when_idle(self):
        """stop() returns gracefully when not running."""
        fm = _make_feed_manager()
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        result = await mgr.stop()
        assert result["status"] == "idle"

    async def test_start_publishes_event(self):
        """start() publishes CAPTURE_STARTED event."""
        eb = _make_event_bus()
        fm = _make_feed_manager()
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, eb)

        await mgr.start("test-feed", "test-ds")

        eb.publish.assert_called_once()
        call_args = eb.publish.call_args
        assert call_args[0][0] == "capture.started"

    async def test_stop_publishes_event(self):
        """stop() publishes CAPTURE_STOPPED event."""
        eb = _make_event_bus()
        fm = _make_feed_manager()
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, eb)

        await mgr.start("test-feed", "test-ds")
        eb.publish.reset_mock()
        await mgr.stop()

        eb.publish.assert_called_once()
        call_args = eb.publish.call_args
        assert call_args[0][0] == "capture.stopped"


# --- Frame processing (inference mode) ---

class TestInferenceFrameProcessing:
    """Tests for _on_frame() and _save_capture() in inference mode."""

    async def test_on_frame_positive_capture(self):
        """Inference frame with detections saves image + labels."""
        fm = _make_feed_manager(is_derived=True)
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        await mgr.start("derived-feed", "test-ds", capture_interval=0.0)

        inf_frame = _make_inference_frame()
        await mgr._save_capture(inf_frame)

        dm.add_image.assert_called_once()
        dm.save_labels.assert_called_once()

    async def test_on_frame_negative_capture(self):
        """Inference frame without detections — saves image, no labels."""
        fm = _make_feed_manager(is_derived=True)
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        await mgr.start("derived-feed", "test-ds", capture_interval=0.0)

        inf_frame = _make_inference_frame(detections=[])
        await mgr._save_capture(inf_frame)

        dm.add_image.assert_called_once()
        dm.save_labels.assert_not_called()

    async def test_on_frame_interval_skip(self):
        """Frames within interval are skipped by _on_frame()."""
        fm = _make_feed_manager(is_derived=True)
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        await mgr.start("derived-feed", "test-ds", capture_interval=999.0)

        # First call captures (interval = 0 initially)
        inf_frame = _make_inference_frame()
        mgr._on_frame("derived-feed", inf_frame)
        # Allow async task to be scheduled
        await asyncio.sleep(0.05)

        # Second call should be skipped (interval not elapsed)
        dm.add_image.reset_mock()
        mgr._on_frame("derived-feed", inf_frame)
        await asyncio.sleep(0.05)

        # add_image should not have been called again
        dm.add_image.assert_not_called()

    async def test_positive_stats_updated(self):
        """Positive capture increments positive_captures counter."""
        fm = _make_feed_manager(is_derived=True)
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        await mgr.start("derived-feed", "test-ds", capture_interval=0.0)

        inf_frame = _make_inference_frame()
        await mgr._save_capture(inf_frame)

        status = mgr.get_status()
        assert status["stats"]["total_captures"] == 1
        assert status["stats"]["positive_captures"] == 1
        assert status["stats"]["negative_captures"] == 0

    async def test_negative_stats_updated(self):
        """Negative capture increments negative_captures counter."""
        fm = _make_feed_manager(is_derived=True)
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        await mgr.start("derived-feed", "test-ds", capture_interval=0.0)

        inf_frame = _make_inference_frame(detections=[])
        await mgr._save_capture(inf_frame)

        status = mgr.get_status()
        assert status["stats"]["total_captures"] == 1
        assert status["stats"]["positive_captures"] == 0
        assert status["stats"]["negative_captures"] == 1


# --- Frame processing (raw mode) ---

class TestRawFrameProcessing:
    """Tests for _on_frame() and _save_capture() in raw mode."""

    async def test_on_raw_frame_captures_at_interval(self):
        """Raw frame saves image without labels."""
        fm = _make_feed_manager(is_derived=False)
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        await mgr.start("raw-feed", "test-ds", capture_interval=0.0)

        frame = _make_frame()
        await mgr._save_capture(frame)

        dm.add_image.assert_called_once()
        dm.save_labels.assert_not_called()

    async def test_on_raw_frame_interval_skip(self):
        """Raw frames within interval are skipped."""
        fm = _make_feed_manager(is_derived=False)
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        await mgr.start("raw-feed", "test-ds", capture_interval=999.0)

        frame = _make_frame()
        mgr._on_frame("raw-feed", frame)
        await asyncio.sleep(0.05)

        dm.add_image.reset_mock()
        mgr._on_frame("raw-feed", frame)
        await asyncio.sleep(0.05)

        dm.add_image.assert_not_called()


# --- Manual trigger ---

class TestManualTrigger:
    """Tests for manual_trigger()."""

    async def test_manual_trigger(self):
        """Manual trigger captures regardless of interval."""
        fm = _make_feed_manager(is_derived=False)
        frame = _make_frame()
        fm.get_frame.return_value = frame

        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        await mgr.start("raw-feed", "test-ds", capture_interval=999.0)
        result = await mgr.manual_trigger()

        assert "filename" in result
        assert result["split"] == "train"
        assert result["dataset_name"] == "test-ds"
        dm.add_image.assert_called_once()

    async def test_manual_trigger_inference_mode(self):
        """Manual trigger in inference mode saves labels for detections."""
        fm = _make_feed_manager(is_derived=True)
        inf_frame = _make_inference_frame()
        fm.get_frame.return_value = inf_frame

        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        await mgr.start("derived-feed", "test-ds", capture_interval=999.0)
        result = await mgr.manual_trigger()

        assert result["num_detections"] == 1
        dm.save_labels.assert_called_once()

    async def test_manual_trigger_not_running_raises(self):
        """Manual trigger when not running raises ConflictError."""
        fm = _make_feed_manager()
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        with pytest.raises(ConflictError):
            await mgr.manual_trigger()

    async def test_manual_trigger_no_frame_raises(self):
        """Manual trigger with no frame available raises NotFoundError."""
        fm = _make_feed_manager()
        fm.get_frame.return_value = None
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        await mgr.start("test-feed", "test-ds")
        with pytest.raises(NotFoundError):
            await mgr.manual_trigger()


# --- Config ---

class TestUpdateConfig:
    """Tests for update_config()."""

    async def test_update_config(self):
        """Config changes are reflected in get_status()."""
        fm = _make_feed_manager()
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        await mgr.start("test-feed", "test-ds")
        mgr.update_config(capture_interval=5.0, negative_ratio=0.5, confidence_threshold=0.8)

        status = mgr.get_status()
        assert status["config"]["capture_interval"] == 5.0
        assert status["config"]["negative_ratio"] == 0.5
        assert status["config"]["confidence_threshold"] == 0.8

    def test_update_config_partial(self):
        """Partial config update only changes provided fields."""
        fm = _make_feed_manager()
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm)

        mgr.update_config(capture_interval=10.0)
        result = mgr.update_config(negative_ratio=0.9)

        assert result["capture_interval"] == 10.0
        assert result["negative_ratio"] == 0.9


# --- Status ---

class TestGetStatus:
    """Tests for get_status()."""

    def test_get_status_idle(self):
        """Idle status when no session is running."""
        fm = _make_feed_manager()
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm)

        status = mgr.get_status()
        assert status["status"] == "idle"
        assert status["feed_id"] is None
        assert status["config"] is None
        assert status["stats"] is None

    async def test_get_status_running(self):
        """Running status includes config, stats, and mode."""
        fm = _make_feed_manager()
        dm = _make_dataset_manager()
        mgr = CaptureManager(fm, dm, _make_event_bus())

        await mgr.start("test-feed", "test-ds", split="val")

        status = mgr.get_status()
        assert status["status"] == "running"
        assert status["feed_id"] == "test-feed"
        assert status["dataset_name"] == "test-ds"
        assert status["split"] == "val"
        assert status["mode"] == "raw"
        assert status["config"] is not None
        assert status["stats"] is not None


# --- Detection to annotation conversion ---

class TestDetectionToAnnotation:
    """Tests for detections_to_annotations()."""

    def test_detection_to_annotation(self):
        """Pixel bbox [x1,y1,x2,y2] converts to normalized YOLO format."""
        dets = [Detection(class_name="cat", confidence=0.9, bbox=[100, 50, 300, 250], class_id=0)]
        result = detections_to_annotations(dets, image_width=640, image_height=480)

        assert len(result) == 1
        ann = result[0]
        assert ann.class_id == 0
        assert abs(ann.x - (200.0 / 640)) < 1e-6       # center x
        assert abs(ann.y - (150.0 / 480)) < 1e-6       # center y
        assert abs(ann.width - (200.0 / 640)) < 1e-6   # width
        assert abs(ann.height - (200.0 / 480)) < 1e-6  # height

    def test_filters_below_threshold(self):
        """Detections below confidence threshold are excluded."""
        dets = [
            Detection(class_name="cat", confidence=0.9, bbox=[0, 0, 100, 100], class_id=0),
            Detection(class_name="dog", confidence=0.1, bbox=[0, 0, 100, 100], class_id=1),
        ]
        result = detections_to_annotations(dets, 640, 480, confidence_threshold=0.5)

        assert len(result) == 1
        assert result[0].class_id == 0

    def test_empty_detections(self):
        """Empty detection list produces empty annotations."""
        result = detections_to_annotations([], 640, 480)
        assert result == []

    def test_zero_image_dimensions(self):
        """Zero image dimensions return empty annotations (no divide-by-zero)."""
        dets = [Detection(class_name="cat", confidence=0.9, bbox=[0, 0, 100, 100], class_id=0)]
        result = detections_to_annotations(dets, 0, 0)
        assert result == []

    def test_clamps_to_0_1(self):
        """Annotations are clamped to [0, 1] range."""
        # Detection that goes beyond image bounds
        dets = [Detection(
            class_name="cat", confidence=0.9,
            bbox=[-50, -50, 700, 500], class_id=0,
        )]
        result = detections_to_annotations(dets, 640, 480)

        assert len(result) == 1
        ann = result[0]
        assert 0.0 <= ann.x <= 1.0
        assert 0.0 <= ann.y <= 1.0
        assert 0.0 <= ann.width <= 1.0
        assert 0.0 <= ann.height <= 1.0

"""
Unit tests for CameraFeed and FeedFactory.

Uses mocked cv2.VideoCapture to test camera feed lifecycle
without requiring actual camera hardware.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.feeds.base import BaseFeed
from backend.feeds.camera import CameraFeed
from backend.feeds.factory import FeedFactory
from backend.feeds.models import FeedType
from backend.feeds.rtsp import RTSPFeed


def make_mock_capture(
    is_opened: bool = True,
    fps: float = 30.0,
    width: float = 640.0,
    height: float = 480.0,
    read_success: bool = True,
):
    """
    Create a mock cv2.VideoCapture with configurable behavior.

    Args:
        is_opened: Whether isOpened() returns True
        fps: Value for CAP_PROP_FPS
        width: Value for CAP_PROP_FRAME_WIDTH
        height: Value for CAP_PROP_FRAME_HEIGHT
        read_success: Whether read() returns a frame

    Returns:
        Configured MagicMock mimicking cv2.VideoCapture.
    """
    import cv2

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = is_opened

    prop_map = {
        cv2.CAP_PROP_FPS: fps,
        cv2.CAP_PROP_FRAME_WIDTH: width,
        cv2.CAP_PROP_FRAME_HEIGHT: height,
    }
    mock_cap.get.side_effect = lambda prop: prop_map.get(prop, 0.0)

    if read_success:
        frame = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
    else:
        mock_cap.read.return_value = (False, None)

    return mock_cap


class TestCameraFeedConnect:
    """Tests for CameraFeed.connect()."""

    @patch("backend.feeds.camera.cv2.VideoCapture")
    def test_connect_success(self, mock_vc_cls):
        """Successful connect opens camera and reads properties."""
        mock_vc_cls.return_value = make_mock_capture()
        feed = CameraFeed("0")

        assert feed.connect() is True
        mock_vc_cls.assert_called_once_with(0)
        assert feed.fps == 30.0
        assert feed.resolution == (640, 480)

    @patch("backend.feeds.camera.cv2.VideoCapture")
    def test_connect_failure_not_opened(self, mock_vc_cls):
        """connect returns False when camera can't be opened."""
        mock_vc_cls.return_value = make_mock_capture(is_opened=False)
        feed = CameraFeed("0")

        assert feed.connect() is False
        assert feed.resolution is None

    def test_connect_invalid_source(self):
        """connect returns False for non-integer source string."""
        feed = CameraFeed("not-a-number")
        assert feed.connect() is False

    @patch("backend.feeds.camera.cv2.VideoCapture")
    def test_connect_fallback_fps(self, mock_vc_cls):
        """connect uses 30.0 fps when camera reports 0."""
        mock_vc_cls.return_value = make_mock_capture(fps=0.0)
        feed = CameraFeed("0")
        feed.connect()
        assert feed.fps == 30.0


class TestCameraFeedRead:
    """Tests for CameraFeed.read()."""

    @patch("backend.feeds.camera.cv2.VideoCapture")
    def test_read_returns_frame(self, mock_vc_cls):
        """read returns a numpy array after successful connect."""
        mock_vc_cls.return_value = make_mock_capture()
        feed = CameraFeed("0")
        feed.connect()

        frame = feed.read()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)

    @patch("backend.feeds.camera.cv2.VideoCapture")
    def test_read_returns_none_on_failure(self, mock_vc_cls):
        """read returns None when camera fails to provide a frame."""
        mock_vc_cls.return_value = make_mock_capture(read_success=False)
        feed = CameraFeed("0")
        feed.connect()

        assert feed.read() is None

    def test_read_before_connect(self):
        """read returns None when called before connect."""
        feed = CameraFeed("0")
        assert feed.read() is None


class TestCameraFeedDisconnect:
    """Tests for CameraFeed.disconnect()."""

    @patch("backend.feeds.camera.cv2.VideoCapture")
    def test_disconnect_releases_capture(self, mock_vc_cls):
        """disconnect calls release() on the VideoCapture."""
        mock_cap = make_mock_capture()
        mock_vc_cls.return_value = mock_cap
        feed = CameraFeed("0")
        feed.connect()
        feed.disconnect()

        mock_cap.release.assert_called_once()
        assert feed.resolution is None

    def test_disconnect_before_connect(self):
        """disconnect is safe to call without prior connect."""
        feed = CameraFeed("0")
        feed.disconnect()  # Should not raise

    @patch("backend.feeds.camera.cv2.VideoCapture")
    def test_read_after_disconnect(self, mock_vc_cls):
        """read returns None after disconnect."""
        mock_vc_cls.return_value = make_mock_capture()
        feed = CameraFeed("0")
        feed.connect()
        feed.disconnect()

        assert feed.read() is None


class TestCameraFeedProperties:
    """Tests for CameraFeed property accessors."""

    @patch("backend.feeds.camera.cv2.VideoCapture")
    def test_source_property(self, mock_vc_cls):
        """source returns the original source string."""
        feed = CameraFeed("2")
        assert feed.source == "2"

    def test_initial_fps(self):
        """fps is 0.0 before connect."""
        feed = CameraFeed("0")
        assert feed.fps == 0.0

    def test_initial_resolution(self):
        """resolution is None before connect."""
        feed = CameraFeed("0")
        assert feed.resolution is None

    @patch("backend.feeds.camera.cv2.VideoCapture")
    def test_resolution_with_zero_dimensions(self, mock_vc_cls):
        """resolution stays None if camera reports zero dimensions."""
        mock_vc_cls.return_value = make_mock_capture(width=0.0, height=0.0)
        feed = CameraFeed("0")
        feed.connect()
        assert feed.resolution is None


class TestCameraFeedIsBaseFeed:
    """Tests verifying CameraFeed implements BaseFeed."""

    def test_isinstance(self):
        """CameraFeed is an instance of BaseFeed."""
        feed = CameraFeed("0")
        assert isinstance(feed, BaseFeed)


class TestFeedFactory:
    """Tests for FeedFactory.create()."""

    def test_create_camera(self):
        """create with CAMERA type returns a CameraFeed."""
        feed = FeedFactory.create(FeedType.CAMERA, "0")
        assert isinstance(feed, CameraFeed)
        assert feed.source == "0"

    def test_create_rtsp(self):
        """create with RTSP type returns an RTSPFeed."""
        feed = FeedFactory.create(FeedType.RTSP, "rtsp://example.com/stream")
        assert isinstance(feed, RTSPFeed)
        assert feed.source == "rtsp://example.com/stream"

    def test_create_file_not_supported(self):
        """create with FILE type raises ValueError."""
        with pytest.raises(ValueError, match="not yet supported"):
            FeedFactory.create(FeedType.FILE, "/path/to/video.mp4")

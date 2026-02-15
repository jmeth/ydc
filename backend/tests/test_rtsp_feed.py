"""
Unit tests for RTSPFeed.

Uses mocked cv2.VideoCapture to test RTSP feed lifecycle
without requiring an actual RTSP stream.
"""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from backend.feeds.base import BaseFeed
from backend.feeds.factory import FeedFactory
from backend.feeds.models import FeedType
from backend.feeds.rtsp import RTSPFeed


def make_mock_capture(
    is_opened: bool = True,
    fps: float = 25.0,
    width: float = 1920.0,
    height: float = 1080.0,
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


class TestRTSPFeedConnect:
    """Tests for RTSPFeed.connect()."""

    @patch("backend.feeds.rtsp.cv2.VideoCapture")
    def test_connect_success(self, mock_vc_cls):
        """Successful connect opens stream and reads properties."""
        mock_vc_cls.return_value = make_mock_capture()
        feed = RTSPFeed("rtsp://192.168.1.10:554/stream")

        assert feed.connect() is True
        mock_vc_cls.assert_called_once_with("rtsp://192.168.1.10:554/stream")
        assert feed.fps == 25.0
        assert feed.resolution == (1920, 1080)

    @patch("backend.feeds.rtsp.cv2.VideoCapture")
    def test_connect_rtsps(self, mock_vc_cls):
        """Successful connect with rtsps:// scheme."""
        mock_vc_cls.return_value = make_mock_capture()
        feed = RTSPFeed("rtsps://secure.example.com/stream")

        assert feed.connect() is True
        mock_vc_cls.assert_called_once_with("rtsps://secure.example.com/stream")

    @patch("backend.feeds.rtsp.cv2.VideoCapture")
    def test_connect_failure_not_opened(self, mock_vc_cls):
        """connect returns False when stream can't be opened."""
        mock_vc_cls.return_value = make_mock_capture(is_opened=False)
        feed = RTSPFeed("rtsp://192.168.1.10:554/stream")

        assert feed.connect() is False
        assert feed.resolution is None

    def test_connect_invalid_source_http(self):
        """connect returns False for non-RTSP URL."""
        feed = RTSPFeed("http://example.com/stream")
        assert feed.connect() is False

    def test_connect_invalid_source_empty(self):
        """connect returns False for empty source string."""
        feed = RTSPFeed("")
        assert feed.connect() is False

    def test_connect_invalid_source_plain_string(self):
        """connect returns False for a plain string without scheme."""
        feed = RTSPFeed("192.168.1.10:554/stream")
        assert feed.connect() is False

    @patch("backend.feeds.rtsp.cv2.VideoCapture")
    def test_connect_fallback_fps(self, mock_vc_cls):
        """connect uses 25.0 fps when stream reports 0."""
        mock_vc_cls.return_value = make_mock_capture(fps=0.0)
        feed = RTSPFeed("rtsp://192.168.1.10:554/stream")
        feed.connect()
        assert feed.fps == 25.0

    @patch("backend.feeds.rtsp.cv2.VideoCapture")
    def test_connect_resolution_with_zero_dimensions(self, mock_vc_cls):
        """resolution stays None if stream reports zero dimensions."""
        mock_vc_cls.return_value = make_mock_capture(width=0.0, height=0.0)
        feed = RTSPFeed("rtsp://192.168.1.10:554/stream")
        feed.connect()
        assert feed.resolution is None


class TestRTSPFeedRead:
    """Tests for RTSPFeed.read()."""

    @patch("backend.feeds.rtsp.cv2.VideoCapture")
    def test_read_returns_frame(self, mock_vc_cls):
        """read returns a numpy array after successful connect."""
        mock_vc_cls.return_value = make_mock_capture()
        feed = RTSPFeed("rtsp://192.168.1.10:554/stream")
        feed.connect()

        frame = feed.read()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (1080, 1920, 3)

    @patch("backend.feeds.rtsp.cv2.VideoCapture")
    def test_read_returns_none_on_failure(self, mock_vc_cls):
        """read returns None when stream fails to provide a frame."""
        mock_vc_cls.return_value = make_mock_capture(read_success=False)
        feed = RTSPFeed("rtsp://192.168.1.10:554/stream")
        feed.connect()

        assert feed.read() is None

    def test_read_before_connect(self):
        """read returns None when called before connect."""
        feed = RTSPFeed("rtsp://192.168.1.10:554/stream")
        assert feed.read() is None


class TestRTSPFeedDisconnect:
    """Tests for RTSPFeed.disconnect()."""

    @patch("backend.feeds.rtsp.cv2.VideoCapture")
    def test_disconnect_releases_capture(self, mock_vc_cls):
        """disconnect calls release() on the VideoCapture."""
        mock_cap = make_mock_capture()
        mock_vc_cls.return_value = mock_cap
        feed = RTSPFeed("rtsp://192.168.1.10:554/stream")
        feed.connect()
        feed.disconnect()

        mock_cap.release.assert_called_once()
        assert feed.resolution is None

    def test_disconnect_before_connect(self):
        """disconnect is safe to call without prior connect."""
        feed = RTSPFeed("rtsp://192.168.1.10:554/stream")
        feed.disconnect()  # Should not raise

    @patch("backend.feeds.rtsp.cv2.VideoCapture")
    def test_read_after_disconnect(self, mock_vc_cls):
        """read returns None after disconnect."""
        mock_vc_cls.return_value = make_mock_capture()
        feed = RTSPFeed("rtsp://192.168.1.10:554/stream")
        feed.connect()
        feed.disconnect()

        assert feed.read() is None


class TestRTSPFeedProperties:
    """Tests for RTSPFeed property accessors."""

    def test_source_property(self):
        """source returns the original source string."""
        feed = RTSPFeed("rtsp://192.168.1.10:554/stream")
        assert feed.source == "rtsp://192.168.1.10:554/stream"

    def test_initial_fps(self):
        """fps is 0.0 before connect."""
        feed = RTSPFeed("rtsp://192.168.1.10:554/stream")
        assert feed.fps == 0.0

    def test_initial_resolution(self):
        """resolution is None before connect."""
        feed = RTSPFeed("rtsp://192.168.1.10:554/stream")
        assert feed.resolution is None


class TestRTSPFeedIsBaseFeed:
    """Tests verifying RTSPFeed implements BaseFeed."""

    def test_isinstance(self):
        """RTSPFeed is an instance of BaseFeed."""
        feed = RTSPFeed("rtsp://192.168.1.10:554/stream")
        assert isinstance(feed, BaseFeed)


class TestFeedFactoryRTSP:
    """Tests for FeedFactory.create() with RTSP type."""

    def test_create_rtsp(self):
        """create with RTSP type returns an RTSPFeed."""
        feed = FeedFactory.create(FeedType.RTSP, "rtsp://192.168.1.10:554/stream")
        assert isinstance(feed, RTSPFeed)
        assert feed.source == "rtsp://192.168.1.10:554/stream"

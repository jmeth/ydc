"""
Unit tests for FeedManager.

Uses a mock FeedFactory to inject controllable BaseFeed instances,
testing feed lifecycle, capture threads, subscriptions, pause/resume,
derived feeds, and shutdown.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.feeds.manager import FeedManager
from backend.feeds.models import (
    DerivedFeed,
    FeedConfig,
    FeedStatus,
    FeedType,
    Frame,
)


def make_mock_feed(
    connect_ok: bool = True,
    fps: float = 30.0,
    resolution: tuple[int, int] | None = (640, 480),
):
    """
    Create a mock BaseFeed with configurable connect behavior and read output.

    Args:
        connect_ok: Whether connect() succeeds
        fps: Value for the fps property
        resolution: Value for the resolution property

    Returns:
        MagicMock configured to behave like a BaseFeed.
    """
    mock_feed = MagicMock()
    mock_feed.connect.return_value = connect_ok
    mock_feed.fps = fps
    mock_feed.resolution = resolution
    # Return a small test frame on each read
    frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_feed.read.return_value = frame_data
    return mock_feed


@pytest.fixture
def manager():
    """Create a FeedManager and shut it down after the test."""
    mgr = FeedManager()
    yield mgr
    mgr.shutdown()


@pytest.fixture
def camera_config():
    """Standard camera FeedConfig for testing."""
    return FeedConfig(feed_type=FeedType.CAMERA, source="0", name="Test Cam", buffer_size=10)


class TestFeedManagerAddRemove:
    """Tests for adding and removing feeds."""

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_add_feed_success(self, mock_create, manager, camera_config):
        """add_feed returns a feed_id on successful connect."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        assert feed_id is not None
        assert isinstance(feed_id, str)
        assert len(feed_id) > 0

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_add_feed_connect_failure(self, mock_create, manager, camera_config):
        """add_feed returns None when the feed fails to connect."""
        mock_create.return_value = make_mock_feed(connect_ok=False)
        feed_id = manager.add_feed(camera_config)

        assert feed_id is None

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_remove_feed_success(self, mock_create, manager, camera_config):
        """remove_feed returns True and cleans up a registered feed."""
        mock_feed = make_mock_feed()
        mock_create.return_value = mock_feed
        feed_id = manager.add_feed(camera_config)

        result = manager.remove_feed(feed_id)
        assert result is True
        mock_feed.disconnect.assert_called_once()

        # Should no longer appear in listing
        assert manager.list_feeds() == []

    def test_remove_nonexistent_feed(self, manager):
        """remove_feed returns False for an unknown feed_id."""
        assert manager.remove_feed("nonexistent") is False

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_add_multiple_feeds(self, mock_create, manager):
        """Multiple feeds can be added simultaneously."""
        mock_create.return_value = make_mock_feed()
        id1 = manager.add_feed(FeedConfig(feed_type=FeedType.CAMERA, source="0"))
        id2 = manager.add_feed(FeedConfig(feed_type=FeedType.CAMERA, source="1"))

        assert id1 != id2
        assert len(manager.list_feeds()) == 2


class TestFeedManagerCapture:
    """Tests for capture thread behavior and frame buffering."""

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_capture_populates_buffer(self, mock_create, manager, camera_config):
        """Capture thread pushes frames into the buffer."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        # Give the capture thread time to run
        time.sleep(0.2)

        frame = manager.get_frame(feed_id)
        assert frame is not None
        assert isinstance(frame, Frame)
        assert frame.data.shape == (480, 640, 3)

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_capture_increments_frame_count(self, mock_create, manager, camera_config):
        """Frame counter increases as frames are captured."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        time.sleep(0.2)

        info = manager.get_feed_info(feed_id)
        assert info is not None
        assert info.frame_count > 0

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_get_frames_multiple(self, mock_create, manager, camera_config):
        """get_frames returns multiple recent frames."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        time.sleep(0.2)

        frames = manager.get_frames(feed_id, count=3)
        assert len(frames) >= 1  # At least one frame captured
        assert all(isinstance(f, Frame) for f in frames)


class TestFeedManagerSubscriptions:
    """Tests for frame subscriber callbacks."""

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_subscribe_receives_frames(self, mock_create, manager, camera_config):
        """Subscribed callback receives (feed_id, Frame) tuples."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        received = []

        def on_frame(fid, frame):
            received.append((fid, frame))

        assert manager.subscribe(feed_id, on_frame) is True

        # Wait for capture to deliver frames
        time.sleep(0.2)
        assert len(received) > 0
        assert received[0][0] == feed_id
        assert isinstance(received[0][1], Frame)

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_unsubscribe(self, mock_create, manager, camera_config):
        """Unsubscribed callback stops receiving frames."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        received = []
        def on_frame(fid, frame):
            received.append(1)

        manager.subscribe(feed_id, on_frame)
        time.sleep(0.15)
        count_before = len(received)

        assert manager.unsubscribe(feed_id, on_frame) is True
        time.sleep(0.15)
        # Should not have received many more frames after unsubscribe
        assert len(received) - count_before <= 2

    def test_subscribe_nonexistent_feed(self, manager):
        """subscribe returns False for an unknown feed_id."""
        assert manager.subscribe("nonexistent", lambda fid, f: None) is False

    def test_unsubscribe_nonexistent(self, manager):
        """unsubscribe returns False when callback not found."""
        assert manager.unsubscribe("nonexistent", lambda fid, f: None) is False

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_subscriber_exception_doesnt_break_capture(self, mock_create, manager, camera_config):
        """A failing subscriber doesn't stop frame capture."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        def bad_callback(fid, frame):
            raise RuntimeError("subscriber error")

        good_received = []
        def good_callback(fid, frame):
            good_received.append(1)

        manager.subscribe(feed_id, bad_callback)
        manager.subscribe(feed_id, good_callback)

        time.sleep(0.2)
        # Good callback should still receive frames despite bad one failing
        assert len(good_received) > 0


class TestFeedManagerPauseResume:
    """Tests for pause and resume operations."""

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_pause(self, mock_create, manager, camera_config):
        """Pausing a feed sets status to PAUSED."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        assert manager.pause(feed_id) is True
        assert manager.get_status(feed_id) == FeedStatus.PAUSED

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_resume(self, mock_create, manager, camera_config):
        """Resuming a paused feed sets status to ACTIVE."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        manager.pause(feed_id)
        assert manager.resume(feed_id) is True
        assert manager.get_status(feed_id) == FeedStatus.ACTIVE

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_pause_stops_frame_capture(self, mock_create, manager, camera_config):
        """Paused feed stops capturing frames."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        time.sleep(0.15)
        manager.pause(feed_id)
        info_at_pause = manager.get_feed_info(feed_id)
        count_at_pause = info_at_pause.frame_count

        time.sleep(0.15)
        info_after = manager.get_feed_info(feed_id)
        # Frame count should not have grown much (may get 1-2 more due to timing)
        assert info_after.frame_count - count_at_pause <= 2

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_pause_already_paused(self, mock_create, manager, camera_config):
        """Pausing an already paused feed returns False."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        manager.pause(feed_id)
        assert manager.pause(feed_id) is False

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_resume_already_running(self, mock_create, manager, camera_config):
        """Resuming an already running feed returns False."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        assert manager.resume(feed_id) is False

    def test_pause_nonexistent(self, manager):
        """pause returns False for unknown feed."""
        assert manager.pause("nonexistent") is False

    def test_resume_nonexistent(self, manager):
        """resume returns False for unknown feed."""
        assert manager.resume("nonexistent") is False


class TestFeedManagerDerivedFeeds:
    """Tests for derived (virtual) feed management."""

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_register_derived_feed(self, mock_create, manager, camera_config):
        """register_derived_feed succeeds when source feed exists."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        derived = DerivedFeed(feed_id="inf-1", source_feed_id=feed_id)
        assert manager.register_derived_feed(derived) is True

    def test_register_derived_feed_no_source(self, manager):
        """register_derived_feed fails when source feed doesn't exist."""
        derived = DerivedFeed(feed_id="inf-1", source_feed_id="nonexistent")
        assert manager.register_derived_feed(derived) is False

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_push_and_get_derived_frame(self, mock_create, manager, camera_config):
        """Frames pushed to derived feeds are retrievable."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        derived = DerivedFeed(feed_id="inf-1", source_feed_id=feed_id)
        manager.register_derived_feed(derived)

        frame = Frame(data=np.zeros((100, 100, 3), dtype=np.uint8))
        assert manager.push_derived_frame("inf-1", frame) is True

        retrieved = manager.get_frame("inf-1")
        assert retrieved is not None
        assert retrieved.data.shape == (100, 100, 3)

    def test_push_derived_frame_no_feed(self, manager):
        """push_derived_frame returns False for unregistered derived feed."""
        frame = Frame(data=np.zeros((100, 100, 3), dtype=np.uint8))
        assert manager.push_derived_frame("nonexistent", frame) is False

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_unregister_derived_feed(self, mock_create, manager, camera_config):
        """unregister_derived_feed removes the derived feed."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        derived = DerivedFeed(feed_id="inf-1", source_feed_id=feed_id)
        manager.register_derived_feed(derived)

        assert manager.unregister_derived_feed("inf-1") is True
        assert manager.get_frame("inf-1") is None

    def test_unregister_nonexistent_derived(self, manager):
        """unregister_derived_feed returns False for unknown derived feed."""
        assert manager.unregister_derived_feed("nonexistent") is False

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_remove_feed_cleans_derived(self, mock_create, manager, camera_config):
        """Removing a source feed also removes its derived feeds."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        derived = DerivedFeed(feed_id="inf-1", source_feed_id=feed_id)
        manager.register_derived_feed(derived)

        manager.remove_feed(feed_id)
        assert manager.get_frame("inf-1") is None


class TestFeedManagerInfo:
    """Tests for feed info and listing."""

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_list_feeds_empty(self, mock_create, manager):
        """list_feeds returns empty list when no feeds exist."""
        assert manager.list_feeds() == []

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_list_feeds(self, mock_create, manager, camera_config):
        """list_feeds returns FeedInfo for each feed."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        feeds = manager.list_feeds()
        assert len(feeds) == 1
        assert feeds[0].feed_id == feed_id
        assert feeds[0].config.feed_type == FeedType.CAMERA
        assert feeds[0].status == FeedStatus.ACTIVE

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_get_feed_info(self, mock_create, manager, camera_config):
        """get_feed_info returns details for a specific feed."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        info = manager.get_feed_info(feed_id)
        assert info is not None
        assert info.feed_id == feed_id
        assert info.fps == 30.0
        assert info.resolution == (640, 480)

    def test_get_feed_info_nonexistent(self, manager):
        """get_feed_info returns None for unknown feed."""
        assert manager.get_feed_info("nonexistent") is None

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_get_status(self, mock_create, manager, camera_config):
        """get_status returns the current feed status."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        assert manager.get_status(feed_id) == FeedStatus.ACTIVE

    def test_get_status_nonexistent(self, manager):
        """get_status returns None for unknown feed."""
        assert manager.get_status("nonexistent") is None


class TestFeedManagerShutdown:
    """Tests for shutdown behavior."""

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_shutdown_stops_all_feeds(self, mock_create):
        """shutdown removes all feeds and disconnects them."""
        manager = FeedManager()
        mock_feeds = []
        for _ in range(3):
            mf = make_mock_feed()
            mock_feeds.append(mf)
        mock_create.side_effect = mock_feeds

        ids = []
        for i in range(3):
            fid = manager.add_feed(FeedConfig(feed_type=FeedType.CAMERA, source=str(i)))
            ids.append(fid)

        manager.shutdown()

        assert manager.list_feeds() == []
        for mf in mock_feeds:
            mf.disconnect.assert_called_once()

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_shutdown_empty_manager(self, mock_create):
        """shutdown on empty manager doesn't error."""
        manager = FeedManager()
        manager.shutdown()  # Should not raise


class TestFeedManagerGetBuffer:
    """Tests for get_buffer used by FeedStreamer."""

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_get_buffer(self, mock_create, manager, camera_config):
        """get_buffer returns the RingBuffer for a feed."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        buf = manager.get_buffer(feed_id)
        assert buf is not None
        assert buf.max_size == camera_config.buffer_size

    def test_get_buffer_nonexistent(self, manager):
        """get_buffer returns None for unknown feed."""
        assert manager.get_buffer("nonexistent") is None

    @patch("backend.feeds.manager.FeedFactory.create")
    def test_get_buffer_derived(self, mock_create, manager, camera_config):
        """get_buffer also works for derived feeds."""
        mock_create.return_value = make_mock_feed()
        feed_id = manager.add_feed(camera_config)

        derived = DerivedFeed(feed_id="inf-1", source_feed_id=feed_id, buffer_size=15)
        manager.register_derived_feed(derived)

        buf = manager.get_buffer("inf-1")
        assert buf is not None
        assert buf.max_size == 15

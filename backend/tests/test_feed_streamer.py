"""
Unit tests for FeedStreamer.

Tests JPEG encoding, monitor task lifecycle, and per-feed stream
task management with mocked FeedManager and ConnectionManager.
"""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from backend.feeds.streaming import FeedStreamer


@pytest.fixture
def mock_feed_manager():
    """Create a mock FeedManager."""
    manager = MagicMock()
    manager.get_buffer.return_value = None
    return manager


@pytest.fixture
def mock_connection_manager():
    """Create a mock ConnectionManager with async broadcast."""
    cm = MagicMock()
    cm.feed_subscriptions = {}
    cm.broadcast_to_feed = AsyncMock()
    return cm


@pytest.fixture
def streamer(mock_feed_manager, mock_connection_manager):
    """Create a FeedStreamer with mocked dependencies."""
    return FeedStreamer(
        feed_manager=mock_feed_manager,
        connection_manager=mock_connection_manager,
        stream_fps=30.0,
        jpeg_quality=70,
    )


class TestFeedStreamerEncode:
    """Tests for _encode_frame JPEG encoding."""

    def test_encode_valid_frame(self, streamer):
        """Valid numpy frame encodes to a non-empty base64 string."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = streamer._encode_frame(frame)
        assert result is not None
        assert len(result) > 0
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_encode_small_frame(self, streamer):
        """Small frames also encode successfully."""
        frame = np.ones((10, 10, 3), dtype=np.uint8) * 128
        result = streamer._encode_frame(frame)
        assert result is not None

    def test_encode_produces_jpeg(self, streamer):
        """Encoded output starts with JPEG magic bytes after base64 decode."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = streamer._encode_frame(frame)
        decoded = base64.b64decode(result)
        # JPEG files start with FF D8
        assert decoded[:2] == b'\xff\xd8'

    def test_encode_invalid_input(self, streamer):
        """Invalid input (not a numpy array) returns None without crashing."""
        result = streamer._encode_frame("not an array")
        assert result is None


class TestFeedStreamerLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_monitor_task(self, streamer):
        """start() creates a running monitor task."""
        await streamer.start()
        assert streamer._monitor_task is not None
        assert not streamer._monitor_task.done()
        await streamer.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_monitor(self, streamer):
        """stop() cancels the monitor task."""
        await streamer.start()
        await streamer.stop()
        assert streamer._monitor_task is None

    @pytest.mark.asyncio
    async def test_stop_cancels_stream_tasks(self, streamer, mock_feed_manager, mock_connection_manager):
        """stop() cancels all active per-feed stream tasks."""
        # Set up a subscribed feed with a buffer
        mock_buffer = MagicMock()
        mock_buffer.get_latest.return_value = None
        mock_feed_manager.get_buffer.return_value = mock_buffer
        mock_connection_manager.feed_subscriptions = {"feed-1": [MagicMock()]}

        await streamer.start()
        # Give monitor time to start stream task
        await asyncio.sleep(1.5)

        assert len(streamer._stream_tasks) > 0
        await streamer.stop()
        assert len(streamer._stream_tasks) == 0


class TestFeedStreamerSync:
    """Tests for _sync_stream_tasks subscription tracking."""

    @pytest.mark.asyncio
    async def test_starts_task_for_subscribed_feed(self, streamer, mock_feed_manager, mock_connection_manager):
        """A stream task is created when a feed has subscribers."""
        mock_buffer = MagicMock()
        mock_buffer.get_latest.return_value = None
        mock_feed_manager.get_buffer.return_value = mock_buffer
        mock_connection_manager.feed_subscriptions = {"feed-1": [MagicMock()]}

        await streamer._sync_stream_tasks()

        assert "feed-1" in streamer._stream_tasks
        # Clean up
        for task in streamer._stream_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_stops_task_when_unsubscribed(self, streamer, mock_feed_manager, mock_connection_manager):
        """Stream task is cancelled when subscribers leave."""
        mock_buffer = MagicMock()
        mock_buffer.get_latest.return_value = None
        mock_feed_manager.get_buffer.return_value = mock_buffer

        # Start with subscriber
        mock_connection_manager.feed_subscriptions = {"feed-1": [MagicMock()]}
        await streamer._sync_stream_tasks()
        assert "feed-1" in streamer._stream_tasks

        # Remove subscriber
        mock_connection_manager.feed_subscriptions = {"feed-1": []}
        await streamer._sync_stream_tasks()
        assert "feed-1" not in streamer._stream_tasks

    @pytest.mark.asyncio
    async def test_no_task_without_buffer(self, streamer, mock_feed_manager, mock_connection_manager):
        """No stream task is created if the feed has no buffer."""
        mock_feed_manager.get_buffer.return_value = None
        mock_connection_manager.feed_subscriptions = {"feed-1": [MagicMock()]}

        await streamer._sync_stream_tasks()

        assert "feed-1" not in streamer._stream_tasks

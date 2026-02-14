"""
Tests for derived feed subscription support in FeedManager.

Verifies that subscribers can receive callbacks from derived feeds
(in addition to raw feeds), and that unregister/unsubscribe properly
cleans up subscriber lists.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from backend.feeds.manager import FeedManager
from backend.feeds.models import DerivedFeed, FeedConfig, FeedType, Frame


# --- Helpers ---

def _make_frame() -> Frame:
    """Create a test Frame."""
    return Frame(data=np.zeros((480, 640, 3), dtype=np.uint8), frame_number=0)


def _add_raw_feed(fm: FeedManager) -> str:
    """Add a raw feed to the manager, returning its feed_id."""
    config = FeedConfig(feed_type=FeedType.CAMERA, source="0", name="test-cam")
    with patch("backend.feeds.factory.FeedFactory.create") as mock_create:
        mock_feed = MagicMock()
        mock_feed.connect.return_value = True
        mock_feed.fps = 30.0
        mock_feed.resolution = (640, 480)
        mock_feed.read.return_value = None  # Don't produce frames from thread
        mock_create.return_value = mock_feed
        feed_id = fm.add_feed(config)
    return feed_id


# --- Tests ---

class TestDerivedFeedSubscriptions:
    """Tests for subscribe/unsubscribe on derived feeds."""

    def test_subscribe_to_derived_feed(self):
        """Subscribers receive callbacks when frames are pushed to derived feeds."""
        fm = FeedManager()
        raw_id = _add_raw_feed(fm)

        derived = DerivedFeed(
            feed_id="derived-1", source_feed_id=raw_id, feed_type="inference",
        )
        assert fm.register_derived_feed(derived) is True

        callback = MagicMock()
        assert fm.subscribe("derived-1", callback) is True

        frame = _make_frame()
        fm.push_derived_frame("derived-1", frame)

        callback.assert_called_once_with("derived-1", frame)

        fm.remove_feed(raw_id)

    def test_unsubscribe_from_derived_feed(self):
        """After unsubscribe, callback is no longer called."""
        fm = FeedManager()
        raw_id = _add_raw_feed(fm)

        derived = DerivedFeed(
            feed_id="derived-2", source_feed_id=raw_id, feed_type="inference",
        )
        fm.register_derived_feed(derived)

        callback = MagicMock()
        fm.subscribe("derived-2", callback)
        fm.unsubscribe("derived-2", callback)

        frame = _make_frame()
        fm.push_derived_frame("derived-2", frame)

        callback.assert_not_called()

        fm.remove_feed(raw_id)

    def test_unregister_cleans_subscribers(self):
        """Unregistering a derived feed removes its subscriber list."""
        fm = FeedManager()
        raw_id = _add_raw_feed(fm)

        derived = DerivedFeed(
            feed_id="derived-3", source_feed_id=raw_id, feed_type="inference",
        )
        fm.register_derived_feed(derived)

        callback = MagicMock()
        fm.subscribe("derived-3", callback)

        fm.unregister_derived_feed("derived-3")

        # Subscriber list should be cleaned up
        assert fm.subscribe("derived-3", callback) is False

        fm.remove_feed(raw_id)

    def test_is_derived_feed(self):
        """is_derived_feed() correctly identifies derived feeds."""
        fm = FeedManager()
        raw_id = _add_raw_feed(fm)

        assert fm.is_derived_feed(raw_id) is False
        assert fm.is_derived_feed("nonexistent") is False

        derived = DerivedFeed(
            feed_id="derived-4", source_feed_id=raw_id, feed_type="inference",
        )
        fm.register_derived_feed(derived)

        assert fm.is_derived_feed("derived-4") is True
        assert fm.is_derived_feed(raw_id) is False

        fm.remove_feed(raw_id)

    def test_multiple_subscribers_on_derived(self):
        """Multiple subscribers all receive callbacks from derived feeds."""
        fm = FeedManager()
        raw_id = _add_raw_feed(fm)

        derived = DerivedFeed(
            feed_id="derived-5", source_feed_id=raw_id, feed_type="inference",
        )
        fm.register_derived_feed(derived)

        cb1 = MagicMock()
        cb2 = MagicMock()
        fm.subscribe("derived-5", cb1)
        fm.subscribe("derived-5", cb2)

        frame = _make_frame()
        fm.push_derived_frame("derived-5", frame)

        cb1.assert_called_once_with("derived-5", frame)
        cb2.assert_called_once_with("derived-5", frame)

        fm.remove_feed(raw_id)

    def test_subscriber_error_isolation(self):
        """One subscriber error doesn't prevent others from being called."""
        fm = FeedManager()
        raw_id = _add_raw_feed(fm)

        derived = DerivedFeed(
            feed_id="derived-6", source_feed_id=raw_id, feed_type="inference",
        )
        fm.register_derived_feed(derived)

        bad_cb = MagicMock(side_effect=RuntimeError("boom"))
        good_cb = MagicMock()
        fm.subscribe("derived-6", bad_cb)
        fm.subscribe("derived-6", good_cb)

        frame = _make_frame()
        fm.push_derived_frame("derived-6", frame)

        bad_cb.assert_called_once()
        good_cb.assert_called_once()

        fm.remove_feed(raw_id)

"""
Feed manager for video feed lifecycle, capture threads, and subscriptions.

FeedManager is intentionally synchronous (pure threading, no asyncio) so
it can be tested without an event loop. The async FeedStreamer polls buffers
separately for WebSocket broadcasting.
"""

import logging
import threading
import time
import uuid
from typing import Callable

import numpy as np

from backend.feeds.base import BaseFeed
from backend.feeds.buffer import RingBuffer
from backend.feeds.factory import FeedFactory
from backend.feeds.models import (
    DerivedFeed,
    FeedConfig,
    FeedInfo,
    FeedStatus,
    Frame,
)

logger = logging.getLogger(__name__)

# Type for frame subscriber callbacks: receives (feed_id, Frame)
FrameCallback = Callable[[str, Frame], None]


class FeedManager:
    """
    Manages video feed lifecycle, capture threads, and frame distribution.

    Each feed gets a background capture thread that reads frames from
    the BaseFeed implementation and pushes them into a RingBuffer.
    Subscribers receive frame callbacks from the capture thread.

    Thread safety: _lock protects all internal registries. Capture loops
    snapshot the callback list under lock, then call callbacks outside
    the lock to avoid deadlocks.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # feed_id -> BaseFeed instance
        self._feeds: dict[str, BaseFeed] = {}
        # feed_id -> FeedConfig
        self._configs: dict[str, FeedConfig] = {}
        # feed_id -> RingBuffer for raw frames
        self._buffers: dict[str, RingBuffer] = {}
        # feed_id -> capture thread
        self._threads: dict[str, threading.Thread] = {}
        # feed_id -> threading.Event (set = running, clear = paused)
        self._running_events: dict[str, threading.Event] = {}
        # feed_id -> threading.Event (set = request stop)
        self._stop_events: dict[str, threading.Event] = {}
        # feed_id -> FeedStatus
        self._statuses: dict[str, FeedStatus] = {}
        # feed_id -> frame counter
        self._frame_counts: dict[str, int] = {}
        # feed_id -> list of subscriber callbacks
        self._subscribers: dict[str, list[FrameCallback]] = {}
        # derived feed_id -> DerivedFeed config
        self._derived_feeds: dict[str, DerivedFeed] = {}
        # derived feed_id -> RingBuffer
        self._derived_buffers: dict[str, RingBuffer] = {}

    def add_feed(self, config: FeedConfig) -> str | None:
        """
        Create a feed, connect it, and start its capture thread.

        Args:
            config: Feed configuration (type, source, name, buffer_size)

        Returns:
            The new feed_id string, or None if connection failed.
        """
        feed = FeedFactory.create(config.feed_type, config.source)

        # Attempt to connect outside the lock (I/O operation)
        if not feed.connect():
            logger.warning("Failed to connect feed: type=%s source=%s", config.feed_type, config.source)
            return None

        feed_id = str(uuid.uuid4())

        with self._lock:
            self._feeds[feed_id] = feed
            self._configs[feed_id] = config
            self._buffers[feed_id] = RingBuffer(max_size=config.buffer_size)
            self._statuses[feed_id] = FeedStatus.ACTIVE
            self._frame_counts[feed_id] = 0
            self._subscribers[feed_id] = []

            # Create control events
            running_event = threading.Event()
            running_event.set()  # Start in running (not paused) state
            self._running_events[feed_id] = running_event

            stop_event = threading.Event()
            self._stop_events[feed_id] = stop_event

        # Start capture thread
        thread = threading.Thread(
            target=self._capture_loop,
            args=(feed_id,),
            name=f"feed-capture-{feed_id[:8]}",
            daemon=True,
        )
        with self._lock:
            self._threads[feed_id] = thread
        thread.start()

        logger.info("Feed added: id=%s type=%s source=%s", feed_id, config.feed_type, config.source)
        return feed_id

    def remove_feed(self, feed_id: str) -> bool:
        """
        Stop and remove a feed, cleaning up all resources.

        Args:
            feed_id: The feed to remove

        Returns:
            True if the feed existed and was removed, False if not found.
        """
        with self._lock:
            if feed_id not in self._feeds:
                return False
            stop_event = self._stop_events.get(feed_id)
            running_event = self._running_events.get(feed_id)
            thread = self._threads.get(feed_id)

        # Signal the capture thread to stop
        if stop_event:
            stop_event.set()
        # Unblock if paused so the thread can exit
        if running_event:
            running_event.set()

        # Wait for thread to finish (with timeout)
        if thread and thread.is_alive():
            thread.join(timeout=3.0)

        with self._lock:
            # Disconnect the feed
            feed = self._feeds.pop(feed_id, None)
            if feed:
                feed.disconnect()

            # Clean up all registries
            self._configs.pop(feed_id, None)
            self._buffers.pop(feed_id, None)
            self._threads.pop(feed_id, None)
            self._running_events.pop(feed_id, None)
            self._stop_events.pop(feed_id, None)
            self._statuses.pop(feed_id, None)
            self._frame_counts.pop(feed_id, None)
            self._subscribers.pop(feed_id, None)

            # Remove any derived feeds for this source
            derived_to_remove = [
                did for did, df in self._derived_feeds.items()
                if df.source_feed_id == feed_id
            ]
            for did in derived_to_remove:
                self._derived_feeds.pop(did, None)
                self._derived_buffers.pop(did, None)

        logger.info("Feed removed: %s", feed_id)
        return True

    def _capture_loop(self, feed_id: str) -> None:
        """
        Background thread loop: reads frames and distributes to buffer/subscribers.

        Respects pause (running_event) and stop (stop_event) signals.
        On read failure, sets feed status to ERROR.

        Args:
            feed_id: The feed to capture from
        """
        consecutive_failures = 0
        max_failures = 30  # ~1 second at 30fps before marking error

        while True:
            # Check stop signal
            with self._lock:
                stop_event = self._stop_events.get(feed_id)
                running_event = self._running_events.get(feed_id)
            if stop_event is None or stop_event.is_set():
                break

            # Wait if paused (blocks until running_event is set or stop)
            if running_event and not running_event.is_set():
                # Check every 100ms so we can still respond to stop
                running_event.wait(timeout=0.1)
                continue

            # Read a frame
            with self._lock:
                feed = self._feeds.get(feed_id)
                buf = self._buffers.get(feed_id)
            if feed is None or buf is None:
                break

            raw_frame = feed.read()
            if raw_frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    with self._lock:
                        self._statuses[feed_id] = FeedStatus.ERROR
                    logger.warning("Feed %s: too many read failures, marking ERROR", feed_id[:8])
                time.sleep(0.01)
                continue

            consecutive_failures = 0

            # Build Frame and push to buffer
            with self._lock:
                frame_num = self._frame_counts.get(feed_id, 0)
                self._frame_counts[feed_id] = frame_num + 1
            frame = Frame(data=raw_frame, frame_number=frame_num)
            buf.push(frame)

            # Snapshot subscribers under lock, call outside lock
            with self._lock:
                callbacks = list(self._subscribers.get(feed_id, []))

            for cb in callbacks:
                try:
                    cb(feed_id, frame)
                except Exception:
                    logger.exception("Subscriber callback error for feed %s", feed_id[:8])

            # Throttle to avoid busy-spinning (target ~capture fps)
            with self._lock:
                feed_ref = self._feeds.get(feed_id)
            target_fps = feed_ref.fps if feed_ref else 30.0
            if target_fps > 0:
                time.sleep(1.0 / target_fps)

    def subscribe(self, feed_id: str, callback: FrameCallback) -> bool:
        """
        Register a callback to receive frames from a feed.

        The callback is invoked from the capture thread with (feed_id, Frame).

        Args:
            feed_id: The feed to subscribe to
            callback: Function accepting (feed_id: str, frame: Frame)

        Returns:
            True if subscribed, False if feed not found.
        """
        with self._lock:
            if feed_id not in self._subscribers:
                return False
            self._subscribers[feed_id].append(callback)
            return True

    def unsubscribe(self, feed_id: str, callback: FrameCallback) -> bool:
        """
        Remove a callback from a feed's subscriber list.

        Args:
            feed_id: The feed to unsubscribe from
            callback: The previously registered callback

        Returns:
            True if the callback was found and removed, False otherwise.
        """
        with self._lock:
            subs = self._subscribers.get(feed_id, [])
            try:
                subs.remove(callback)
                return True
            except ValueError:
                return False

    def register_derived_feed(self, derived: DerivedFeed) -> bool:
        """
        Register a derived (virtual) feed that receives processed frames.

        Args:
            derived: DerivedFeed configuration

        Returns:
            True if registered, False if the source feed doesn't exist.
        """
        with self._lock:
            if derived.source_feed_id not in self._feeds:
                return False
            self._derived_feeds[derived.feed_id] = derived
            self._derived_buffers[derived.feed_id] = RingBuffer(max_size=derived.buffer_size)
            return True

    def push_derived_frame(self, derived_feed_id: str, frame: Frame) -> bool:
        """
        Push a processed frame into a derived feed's buffer.

        Args:
            derived_feed_id: The derived feed to push to
            frame: The processed frame

        Returns:
            True if pushed, False if the derived feed doesn't exist.
        """
        with self._lock:
            buf = self._derived_buffers.get(derived_feed_id)
        if buf is None:
            return False
        buf.push(frame)
        return True

    def unregister_derived_feed(self, derived_feed_id: str) -> bool:
        """
        Remove a derived feed and its buffer.

        Args:
            derived_feed_id: The derived feed to remove

        Returns:
            True if found and removed, False otherwise.
        """
        with self._lock:
            removed = self._derived_feeds.pop(derived_feed_id, None)
            self._derived_buffers.pop(derived_feed_id, None)
            return removed is not None

    def list_feeds(self) -> list[FeedInfo]:
        """
        Get metadata for all active feeds.

        Returns:
            List of FeedInfo objects for each registered feed.
        """
        with self._lock:
            result = []
            for feed_id, config in self._configs.items():
                feed = self._feeds.get(feed_id)
                info = FeedInfo(
                    feed_id=feed_id,
                    config=config,
                    status=self._statuses.get(feed_id, FeedStatus.DISCONNECTED),
                    fps=feed.fps if feed else 0.0,
                    resolution=feed.resolution if feed else None,
                    frame_count=self._frame_counts.get(feed_id, 0),
                )
                result.append(info)
            return result

    def get_feed_info(self, feed_id: str) -> FeedInfo | None:
        """
        Get metadata for a specific feed.

        Args:
            feed_id: The feed to look up

        Returns:
            FeedInfo if found, None otherwise.
        """
        with self._lock:
            config = self._configs.get(feed_id)
            if config is None:
                return None
            feed = self._feeds.get(feed_id)
            return FeedInfo(
                feed_id=feed_id,
                config=config,
                status=self._statuses.get(feed_id, FeedStatus.DISCONNECTED),
                fps=feed.fps if feed else 0.0,
                resolution=feed.resolution if feed else None,
                frame_count=self._frame_counts.get(feed_id, 0),
            )

    def get_status(self, feed_id: str) -> FeedStatus | None:
        """
        Get the current status of a feed.

        Args:
            feed_id: The feed to query

        Returns:
            FeedStatus if found, None otherwise.
        """
        with self._lock:
            return self._statuses.get(feed_id)

    def get_frame(self, feed_id: str) -> Frame | None:
        """
        Get the most recent frame from a feed's buffer.

        Also checks derived feed buffers.

        Args:
            feed_id: The feed (raw or derived) to get a frame from

        Returns:
            The latest Frame, or None if no frame available.
        """
        with self._lock:
            buf = self._buffers.get(feed_id) or self._derived_buffers.get(feed_id)
        if buf is None:
            return None
        return buf.get_latest()

    def get_frames(self, feed_id: str, count: int = 1) -> list[Frame]:
        """
        Get the N most recent frames from a feed's buffer.

        Args:
            feed_id: The feed to get frames from
            count: Number of frames to retrieve

        Returns:
            List of Frame objects, newest first. Empty if feed not found.
        """
        with self._lock:
            buf = self._buffers.get(feed_id) or self._derived_buffers.get(feed_id)
        if buf is None:
            return []
        return buf.get_recent(count)

    def get_buffer(self, feed_id: str) -> RingBuffer | None:
        """
        Get the RingBuffer for a feed (raw or derived).

        Used by FeedStreamer to poll for new frames.

        Args:
            feed_id: The feed whose buffer to retrieve

        Returns:
            The RingBuffer, or None if feed not found.
        """
        with self._lock:
            return self._buffers.get(feed_id) or self._derived_buffers.get(feed_id)

    def pause(self, feed_id: str) -> bool:
        """
        Pause a feed's capture thread (stops reading frames).

        Args:
            feed_id: The feed to pause

        Returns:
            True if the feed was paused, False if not found or already paused.
        """
        with self._lock:
            if feed_id not in self._feeds:
                return False
            event = self._running_events.get(feed_id)
            if event is None:
                return False
            if not event.is_set():
                return False  # Already paused
            event.clear()
            self._statuses[feed_id] = FeedStatus.PAUSED
        logger.info("Feed paused: %s", feed_id[:8])
        return True

    def resume(self, feed_id: str) -> bool:
        """
        Resume a paused feed's capture thread.

        Args:
            feed_id: The feed to resume

        Returns:
            True if the feed was resumed, False if not found or not paused.
        """
        with self._lock:
            if feed_id not in self._feeds:
                return False
            event = self._running_events.get(feed_id)
            if event is None:
                return False
            if event.is_set():
                return False  # Already running
            event.set()
            self._statuses[feed_id] = FeedStatus.ACTIVE
        logger.info("Feed resumed: %s", feed_id[:8])
        return True

    def shutdown(self) -> None:
        """Stop all feeds and clean up resources."""
        with self._lock:
            feed_ids = list(self._feeds.keys())

        for feed_id in feed_ids:
            self.remove_feed(feed_id)

        logger.info("FeedManager shutdown complete")

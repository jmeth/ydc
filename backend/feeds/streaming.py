"""
Async feed streamer for broadcasting video frames over WebSocket.

FeedStreamer runs a monitor task that watches ConnectionManager for
feed subscriptions, then starts/stops per-feed stream tasks that poll
RingBuffers and broadcast JPEG-encoded frames at a target FPS.
"""

import asyncio
import base64
import logging
import time

import cv2
import numpy as np

from backend.feeds.manager import FeedManager
from backend.websocket.manager import ConnectionManager

logger = logging.getLogger(__name__)


class FeedStreamer:
    """
    Async engine that streams feed frames to WebSocket subscribers.

    Polls FeedManager buffers at a configurable FPS, encodes frames
    as JPEG/base64, and broadcasts via ConnectionManager. A monitor
    task starts/stops per-feed stream tasks based on active subscriptions.

    Args:
        feed_manager: The FeedManager providing frame buffers
        connection_manager: WebSocket ConnectionManager for broadcasting
        stream_fps: Target frames per second for streaming (default 15.0)
        jpeg_quality: JPEG compression quality 0-100 (default 70)
    """

    def __init__(
        self,
        feed_manager: FeedManager,
        connection_manager: ConnectionManager,
        stream_fps: float = 15.0,
        jpeg_quality: int = 70,
    ):
        self._feed_manager = feed_manager
        self._connection_manager = connection_manager
        self._stream_fps = stream_fps
        self._jpeg_quality = jpeg_quality
        self._monitor_task: asyncio.Task | None = None
        # feed_id -> asyncio.Task for per-feed streaming
        self._stream_tasks: dict[str, asyncio.Task] = {}
        self._running = False

    async def start(self) -> None:
        """Start the monitor task that manages per-feed stream tasks."""
        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="feed-streamer-monitor"
        )
        logger.info("FeedStreamer started (target %.1f fps, JPEG quality %d)", self._stream_fps, self._jpeg_quality)

    async def stop(self) -> None:
        """Stop the monitor and all active stream tasks."""
        self._running = False

        # Cancel all per-feed stream tasks
        for feed_id, task in list(self._stream_tasks.items()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._stream_tasks.clear()

        # Cancel the monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("FeedStreamer stopped")

    async def _monitor_loop(self) -> None:
        """
        Periodically check which feeds have subscribers and manage stream tasks.

        Runs every 1 second. Starts tasks for newly subscribed feeds,
        stops tasks for feeds with no subscribers.
        """
        while self._running:
            try:
                await self._sync_stream_tasks()
            except Exception:
                logger.exception("FeedStreamer monitor error")

            await asyncio.sleep(1.0)

    async def _sync_stream_tasks(self) -> None:
        """Start/stop per-feed stream tasks to match current subscriptions."""
        subscribed_feeds = set(self._connection_manager.feed_subscriptions.keys())
        # Only stream feeds that actually have subscribers
        active_feeds = {
            fid for fid in subscribed_feeds
            if self._connection_manager.feed_subscriptions.get(fid)
        }
        current_tasks = set(self._stream_tasks.keys())

        # Start tasks for newly subscribed feeds
        for feed_id in active_feeds - current_tasks:
            if self._feed_manager.get_buffer(feed_id) is not None:
                task = asyncio.create_task(
                    self._stream_feed(feed_id), name=f"feed-stream-{feed_id[:8]}"
                )
                self._stream_tasks[feed_id] = task
                logger.debug("Started stream task for feed %s", feed_id[:8])

        # Stop tasks for feeds with no subscribers
        for feed_id in current_tasks - active_feeds:
            task = self._stream_tasks.pop(feed_id, None)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                logger.debug("Stopped stream task for feed %s", feed_id[:8])

    async def _stream_feed(self, feed_id: str) -> None:
        """
        Poll the feed buffer at target FPS and broadcast encoded frames.

        Runs until cancelled. Skips frames if encoding fails or no new
        frame is available.

        Args:
            feed_id: The feed to stream from
        """
        interval = 1.0 / self._stream_fps
        last_frame_num = -1

        while self._running:
            start = time.monotonic()

            buf = self._feed_manager.get_buffer(feed_id)
            if buf is None:
                break

            frame = buf.get_latest()
            if frame is not None and frame.frame_number != last_frame_num:
                last_frame_num = frame.frame_number
                encoded = self._encode_frame(frame.data)
                if encoded:
                    await self._connection_manager.broadcast_to_feed(feed_id, encoded)

            # Sleep for the remainder of the interval
            elapsed = time.monotonic() - start
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)

    def _encode_frame(self, frame_data: np.ndarray) -> str | None:
        """
        Encode a BGR numpy frame as JPEG and return base64 string.

        Args:
            frame_data: Raw frame as numpy array (H, W, C) in BGR

        Returns:
            Base64-encoded JPEG string, or None if encoding fails.
        """
        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            success, buffer = cv2.imencode(".jpg", frame_data, encode_params)
            if not success:
                return None
            return base64.b64encode(buffer).decode("ascii")
        except Exception:
            logger.exception("Frame encoding error")
            return None

"""
WebSocket connection manager singleton.

Tracks active connections, manages feed subscriptions for video
streaming, and provides broadcast helpers for both targeted (per-feed)
and global (all clients) messaging.
"""

import logging
import time
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections and feed-based subscriptions.

    Connections are tracked in a flat list. Feed subscriptions map a
    feed_id to the set of WebSocket clients watching that feed, enabling
    efficient per-feed frame broadcasting.
    """

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.feed_subscriptions: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and track a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.debug("WebSocket connected. Total: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket from all tracking structures."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        # Remove from all feed subscriptions
        for feed_id, subscribers in list(self.feed_subscriptions.items()):
            if websocket in subscribers:
                subscribers.remove(websocket)
            # Clean up empty subscription lists
            if not subscribers:
                del self.feed_subscriptions[feed_id]

        logger.debug("WebSocket disconnected. Total: %d", len(self.active_connections))

    async def subscribe_to_feed(self, websocket: WebSocket, feed_id: str) -> None:
        """
        Subscribe a client to a specific video feed.

        Args:
            websocket: The client connection
            feed_id: Feed identifier to subscribe to
        """
        if feed_id not in self.feed_subscriptions:
            self.feed_subscriptions[feed_id] = []
        if websocket not in self.feed_subscriptions[feed_id]:
            self.feed_subscriptions[feed_id].append(websocket)
            logger.debug("Client subscribed to feed '%s'", feed_id)

    async def unsubscribe_from_feed(self, websocket: WebSocket, feed_id: str | None = None) -> None:
        """
        Unsubscribe a client from a feed (or all feeds if feed_id is None).

        Args:
            websocket: The client connection
            feed_id: Specific feed to unsubscribe from, or None for all
        """
        if feed_id is not None:
            if feed_id in self.feed_subscriptions:
                if websocket in self.feed_subscriptions[feed_id]:
                    self.feed_subscriptions[feed_id].remove(websocket)
        else:
            for subscribers in self.feed_subscriptions.values():
                if websocket in subscribers:
                    subscribers.remove(websocket)

    async def broadcast_to_feed(
        self, feed_id: str, frame_data: str, detections: list[dict[str, Any]] | None = None
    ) -> None:
        """
        Broadcast a video frame to all subscribers of a feed.

        Args:
            feed_id: The feed to broadcast to
            frame_data: Base64-encoded JPEG frame
            detections: Optional list of detection dicts
        """
        subscribers = self.feed_subscriptions.get(feed_id, [])
        if not subscribers:
            return

        message = {
            "type": "frame",
            "feed_id": feed_id,
            "data": frame_data,
            "detections": detections or [],
            "timestamp": time.time(),
        }

        disconnected = []
        for websocket in subscribers:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)

        # Clean up broken connections
        for ws in disconnected:
            self.disconnect(ws)

    async def broadcast_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """
        Broadcast a system event to all connected clients.

        Args:
            event_type: Event type string (e.g. "training.progress")
            data: Optional event payload
        """
        message = {"type": event_type, **(data or {})}

        disconnected = []
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)

        for ws in disconnected:
            self.disconnect(ws)


# Module-level singleton
connection_manager = ConnectionManager()

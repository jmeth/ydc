"""
Tests for the WebSocket ConnectionManager.

Covers connect/disconnect tracking, feed subscription/unsubscription,
and broadcast behavior.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.websocket.manager import ConnectionManager


def make_mock_ws():
    """Create a mock WebSocket with accept() and send_json() methods."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_connect_tracks_connection():
    """Connected WebSockets are tracked in active_connections."""
    manager = ConnectionManager()
    ws = make_mock_ws()

    await manager.connect(ws)

    assert ws in manager.active_connections
    ws.accept.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_removes_connection():
    """Disconnected WebSockets are removed from all tracking structures."""
    manager = ConnectionManager()
    ws = make_mock_ws()

    await manager.connect(ws)
    manager.disconnect(ws)

    assert ws not in manager.active_connections


@pytest.mark.asyncio
async def test_subscribe_to_feed():
    """Clients can subscribe to a feed and appear in feed_subscriptions."""
    manager = ConnectionManager()
    ws = make_mock_ws()

    await manager.connect(ws)
    await manager.subscribe_to_feed(ws, "feed-1")

    assert ws in manager.feed_subscriptions["feed-1"]


@pytest.mark.asyncio
async def test_unsubscribe_from_feed():
    """Unsubscribing removes the client from that feed's subscriber list."""
    manager = ConnectionManager()
    ws = make_mock_ws()

    await manager.connect(ws)
    await manager.subscribe_to_feed(ws, "feed-1")
    await manager.unsubscribe_from_feed(ws, "feed-1")

    assert ws not in manager.feed_subscriptions.get("feed-1", [])


@pytest.mark.asyncio
async def test_disconnect_cleans_up_subscriptions():
    """Disconnecting removes the client from all feed subscriptions."""
    manager = ConnectionManager()
    ws = make_mock_ws()

    await manager.connect(ws)
    await manager.subscribe_to_feed(ws, "feed-1")
    await manager.subscribe_to_feed(ws, "feed-2")
    manager.disconnect(ws)

    assert ws not in manager.feed_subscriptions.get("feed-1", [])
    assert ws not in manager.feed_subscriptions.get("feed-2", [])


@pytest.mark.asyncio
async def test_broadcast_to_feed_sends_to_subscribers():
    """broadcast_to_feed sends a frame message to subscribed clients."""
    manager = ConnectionManager()
    ws1 = make_mock_ws()
    ws2 = make_mock_ws()

    await manager.connect(ws1)
    await manager.connect(ws2)
    await manager.subscribe_to_feed(ws1, "feed-1")
    # ws2 is NOT subscribed to feed-1

    await manager.broadcast_to_feed("feed-1", "base64data", [{"class": "cat"}])

    ws1.send_json.assert_called_once()
    msg = ws1.send_json.call_args[0][0]
    assert msg["type"] == "frame"
    assert msg["feed_id"] == "feed-1"
    assert msg["data"] == "base64data"
    assert msg["detections"] == [{"class": "cat"}]

    ws2.send_json.assert_not_called()


@pytest.mark.asyncio
async def test_broadcast_event_sends_to_all():
    """broadcast_event sends to all connected clients."""
    manager = ConnectionManager()
    ws1 = make_mock_ws()
    ws2 = make_mock_ws()

    await manager.connect(ws1)
    await manager.connect(ws2)

    await manager.broadcast_event("training.progress", {"epoch": 5})

    assert ws1.send_json.call_count == 1
    assert ws2.send_json.call_count == 1
    msg = ws1.send_json.call_args[0][0]
    assert msg["type"] == "training.progress"
    assert msg["epoch"] == 5


@pytest.mark.asyncio
async def test_broadcast_handles_broken_connection():
    """Broken connections are cleaned up during broadcast."""
    manager = ConnectionManager()
    good_ws = make_mock_ws()
    bad_ws = make_mock_ws()
    bad_ws.send_json.side_effect = RuntimeError("connection closed")

    await manager.connect(good_ws)
    await manager.connect(bad_ws)

    await manager.broadcast_event("test", {"data": 1})

    # Bad connection should be removed
    assert bad_ws not in manager.active_connections
    # Good connection still receives the message
    good_ws.send_json.assert_called_once()


@pytest.mark.asyncio
async def test_no_duplicate_feed_subscription():
    """Subscribing to the same feed twice doesn't duplicate the entry."""
    manager = ConnectionManager()
    ws = make_mock_ws()

    await manager.connect(ws)
    await manager.subscribe_to_feed(ws, "feed-1")
    await manager.subscribe_to_feed(ws, "feed-1")

    assert manager.feed_subscriptions["feed-1"].count(ws) == 1

"""
Tests for the EventBus pub/sub system.

Covers subscribe, publish, unsubscribe, error isolation between
subscribers, and async callback support.
"""

import pytest

from backend.core.events import EventBus


@pytest.mark.asyncio
async def test_subscribe_and_publish(event_bus):
    """Subscribers receive published events with correct data."""
    received = []

    def handler(data):
        received.append(data)

    await event_bus.start()
    event_bus.subscribe("test.event", handler)
    await event_bus.publish("test.event", {"key": "value"})

    assert len(received) == 1
    assert received[0]["key"] == "value"


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery(event_bus):
    """Unsubscribed callbacks no longer receive events."""
    received = []

    def handler(data):
        received.append(data)

    await event_bus.start()
    event_bus.subscribe("test.event", handler)
    await event_bus.publish("test.event", {"n": 1})
    event_bus.unsubscribe("test.event", handler)
    await event_bus.publish("test.event", {"n": 2})

    assert len(received) == 1
    assert received[0]["n"] == 1


@pytest.mark.asyncio
async def test_error_isolation(event_bus):
    """A failing subscriber doesn't prevent others from executing."""
    received = []

    def bad_handler(data):
        raise RuntimeError("boom")

    def good_handler(data):
        received.append(data)

    await event_bus.start()
    event_bus.subscribe("test.event", bad_handler)
    event_bus.subscribe("test.event", good_handler)
    await event_bus.publish("test.event", {"ok": True})

    assert len(received) == 1
    assert received[0]["ok"] is True


@pytest.mark.asyncio
async def test_async_callback(event_bus):
    """Async callbacks are properly awaited."""
    received = []

    async def async_handler(data):
        received.append(data)

    await event_bus.start()
    event_bus.subscribe("test.event", async_handler)
    await event_bus.publish("test.event", {"async": True})

    assert len(received) == 1
    assert received[0]["async"] is True


@pytest.mark.asyncio
async def test_publish_without_start_drops_event(event_bus):
    """Events published before start() are dropped."""
    received = []

    def handler(data):
        received.append(data)

    event_bus.subscribe("test.event", handler)
    await event_bus.publish("test.event", {"dropped": True})

    assert len(received) == 0


@pytest.mark.asyncio
async def test_multiple_subscribers(event_bus):
    """Multiple subscribers each receive the same event."""
    results_a = []
    results_b = []

    def handler_a(data):
        results_a.append(data)

    def handler_b(data):
        results_b.append(data)

    await event_bus.start()
    event_bus.subscribe("test.event", handler_a)
    event_bus.subscribe("test.event", handler_b)
    await event_bus.publish("test.event", {"multi": True})

    assert len(results_a) == 1
    assert len(results_b) == 1


@pytest.mark.asyncio
async def test_publish_no_data(event_bus):
    """Publishing with no data passes an empty dict."""
    received = []

    def handler(data):
        received.append(data)

    await event_bus.start()
    event_bus.subscribe("test.event", handler)
    await event_bus.publish("test.event")

    assert len(received) == 1
    assert received[0] == {}


@pytest.mark.asyncio
async def test_stop_clears_subscribers(event_bus):
    """Stopping the event bus clears all subscriptions."""
    received = []

    def handler(data):
        received.append(data)

    await event_bus.start()
    event_bus.subscribe("test.event", handler)
    await event_bus.stop()
    await event_bus.start()
    await event_bus.publish("test.event", {"after_stop": True})

    assert len(received) == 0

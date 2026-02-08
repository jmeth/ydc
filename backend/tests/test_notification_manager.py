"""
Tests for NotificationManager.

Covers notify storage and broadcast, mark_read/dismiss, history
filtering, clear_all, max_history eviction, and event subscription
wiring.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.core.events import (
    EventBus,
    TRAINING_COMPLETED,
    TRAINING_ERROR,
    RESOURCE_WARNING,
    RESOURCE_CRITICAL,
    FEED_ADDED,
    FEED_REMOVED,
    FEED_ERROR,
)
from backend.notifications.manager import NotificationManager
from backend.notifications.models import (
    NotificationCategory,
    NotificationLevel,
    NotificationType,
)


@pytest.fixture
def mock_connection_manager():
    """Mock ConnectionManager with async broadcast_event."""
    cm = MagicMock()
    cm.broadcast_event = AsyncMock()
    return cm


@pytest.fixture
def manager(mock_connection_manager):
    """NotificationManager wired to a mock ConnectionManager."""
    return NotificationManager(mock_connection_manager)


class TestNotify:
    """notify() creates a notification, stores it, and broadcasts."""

    @pytest.mark.asyncio
    async def test_notify_returns_notification(self, manager):
        """notify() returns the created Notification."""
        result = await manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Hello",
            message="World",
        )
        assert result.title == "Hello"
        assert result.message == "World"
        assert result.type == NotificationType.TOAST
        assert result.level == NotificationLevel.INFO

    @pytest.mark.asyncio
    async def test_notify_stores_in_history(self, manager):
        """Notification appears in get_history() after notify()."""
        await manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Test",
            message="Stored",
        )
        history = manager.get_history()
        assert len(history) == 1
        assert history[0].title == "Test"

    @pytest.mark.asyncio
    async def test_notify_broadcasts_via_websocket(self, manager, mock_connection_manager):
        """notify() calls connection_manager.broadcast_event with correct shape."""
        result = await manager.notify(
            type=NotificationType.BANNER,
            level=NotificationLevel.WARNING,
            category=NotificationCategory.TRAINING,
            title="Warning",
            message="Low memory",
            data={"usage": 0.95},
        )
        mock_connection_manager.broadcast_event.assert_called_once()
        call_args = mock_connection_manager.broadcast_event.call_args
        assert call_args[0][0] == "notification"
        payload = call_args[0][1]
        assert payload["id"] == result.id
        assert payload["notification_type"] == "banner"
        assert payload["level"] == "warning"
        assert payload["category"] == "training"
        assert payload["title"] == "Warning"
        assert payload["data"] == {"usage": 0.95}

    @pytest.mark.asyncio
    async def test_notify_with_optional_data(self, manager):
        """notify() works without data parameter."""
        result = await manager.notify(
            type=NotificationType.STATUS,
            level=NotificationLevel.SUCCESS,
            category=NotificationCategory.SYSTEM,
            title="OK",
            message="All good",
        )
        assert result.data is None


class TestMaxHistoryEviction:
    """History is capped at max_history, evicting oldest entries."""

    @pytest.mark.asyncio
    async def test_evicts_oldest_when_full(self, mock_connection_manager):
        """When max_history is exceeded, oldest notifications are dropped."""
        mgr = NotificationManager(mock_connection_manager, max_history=5)
        for i in range(7):
            await mgr.notify(
                type=NotificationType.TOAST,
                level=NotificationLevel.INFO,
                category=NotificationCategory.SYSTEM,
                title=f"N{i}",
                message=f"msg {i}",
            )
        history = mgr.get_history(limit=100)
        assert len(history) == 5
        # Newest first — titles should be N6, N5, N4, N3, N2
        titles = [n.title for n in history]
        assert titles == ["N6", "N5", "N4", "N3", "N2"]


class TestMarkRead:
    """mark_read() sets the read flag and returns True/False."""

    @pytest.mark.asyncio
    async def test_mark_read_existing(self, manager):
        """Returns True and sets read=True for an existing notification."""
        n = await manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Test",
            message="msg",
        )
        assert manager.mark_read(n.id) is True
        history = manager.get_history()
        assert history[0].read is True

    def test_mark_read_nonexistent(self, manager):
        """Returns False for a notification ID that doesn't exist."""
        assert manager.mark_read("nonexistent-id") is False


class TestDismiss:
    """dismiss() sets the dismissed flag and returns True/False."""

    @pytest.mark.asyncio
    async def test_dismiss_existing(self, manager):
        """Returns True and sets dismissed=True for an existing notification."""
        n = await manager.notify(
            type=NotificationType.BANNER,
            level=NotificationLevel.WARNING,
            category=NotificationCategory.SYSTEM,
            title="Banner",
            message="msg",
        )
        assert manager.dismiss(n.id) is True
        history = manager.get_history()
        assert history[0].dismissed is True

    def test_dismiss_nonexistent(self, manager):
        """Returns False for a notification ID that doesn't exist."""
        assert manager.dismiss("nonexistent-id") is False


class TestGetHistory:
    """get_history() supports filtering by category, unread, and limit."""

    @pytest.mark.asyncio
    async def test_filter_by_category(self, manager):
        """Only returns notifications matching the requested category."""
        await manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="System",
            message="sys",
        )
        await manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.TRAINING,
            title="Training",
            message="train",
        )
        result = manager.get_history(category=NotificationCategory.TRAINING)
        assert len(result) == 1
        assert result[0].title == "Training"

    @pytest.mark.asyncio
    async def test_filter_unread_only(self, manager):
        """Only returns unread notifications when unread_only=True."""
        n1 = await manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Read",
            message="r",
        )
        await manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Unread",
            message="u",
        )
        manager.mark_read(n1.id)
        result = manager.get_history(unread_only=True)
        assert len(result) == 1
        assert result[0].title == "Unread"

    @pytest.mark.asyncio
    async def test_limit(self, manager):
        """Returns at most `limit` notifications."""
        for i in range(10):
            await manager.notify(
                type=NotificationType.TOAST,
                level=NotificationLevel.INFO,
                category=NotificationCategory.SYSTEM,
                title=f"N{i}",
                message="m",
            )
        result = manager.get_history(limit=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_newest_first(self, manager):
        """History is returned newest-first."""
        await manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="First",
            message="m",
        )
        await manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Second",
            message="m",
        )
        result = manager.get_history()
        assert result[0].title == "Second"
        assert result[1].title == "First"


class TestClearAll:
    """clear_all() removes notifications from history."""

    @pytest.mark.asyncio
    async def test_clear_all(self, manager):
        """Clears all notifications when no category is specified."""
        await manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="A",
            message="m",
        )
        await manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.TRAINING,
            title="B",
            message="m",
        )
        manager.clear_all()
        assert len(manager.get_history()) == 0

    @pytest.mark.asyncio
    async def test_clear_by_category(self, manager):
        """Only clears notifications in the specified category."""
        await manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Keep",
            message="m",
        )
        await manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.TRAINING,
            title="Remove",
            message="m",
        )
        manager.clear_all(category=NotificationCategory.TRAINING)
        history = manager.get_history()
        assert len(history) == 1
        assert history[0].title == "Keep"


class TestEventSubscriptions:
    """setup_event_subscriptions() wires all expected event types."""

    def test_subscribes_to_all_events(self, manager):
        """All 7 event types are subscribed after setup."""
        bus = EventBus()
        manager.setup_event_subscriptions(bus)
        expected_events = [
            TRAINING_COMPLETED,
            TRAINING_ERROR,
            RESOURCE_WARNING,
            RESOURCE_CRITICAL,
            FEED_ADDED,
            FEED_REMOVED,
            FEED_ERROR,
        ]
        for event_type in expected_events:
            assert len(bus._subscribers[event_type]) == 1

    @pytest.mark.asyncio
    async def test_training_completed_creates_toast(self, manager, mock_connection_manager):
        """Publishing training.completed creates a success toast."""
        bus = EventBus()
        await bus.start()
        manager.setup_event_subscriptions(bus)
        await bus.publish(TRAINING_COMPLETED, {"message": "Done training"})
        history = manager.get_history()
        assert len(history) == 1
        assert history[0].type == NotificationType.TOAST
        assert history[0].level == NotificationLevel.SUCCESS
        assert history[0].category == NotificationCategory.TRAINING
        await bus.stop()

    @pytest.mark.asyncio
    async def test_feed_error_creates_error_toast(self, manager, mock_connection_manager):
        """Publishing feed.error creates an error toast."""
        bus = EventBus()
        await bus.start()
        manager.setup_event_subscriptions(bus)
        await bus.publish(FEED_ERROR, {"feed_id": "cam1", "message": "Connection lost"})
        history = manager.get_history()
        assert len(history) == 1
        assert history[0].type == NotificationType.TOAST
        assert history[0].level == NotificationLevel.ERROR
        assert history[0].message == "Connection lost"
        await bus.stop()

    @pytest.mark.asyncio
    async def test_resource_warning_creates_banner(self, manager, mock_connection_manager):
        """Publishing resource.warning creates a warning banner."""
        bus = EventBus()
        await bus.start()
        manager.setup_event_subscriptions(bus)
        await bus.publish(RESOURCE_WARNING, {"message": "Memory at 90%"})
        history = manager.get_history()
        assert len(history) == 1
        assert history[0].type == NotificationType.BANNER
        assert history[0].level == NotificationLevel.WARNING
        await bus.stop()

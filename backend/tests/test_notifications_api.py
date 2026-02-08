"""
Integration tests for the notifications REST API endpoints.

Tests use the ASGI test client against the real FastAPI app with
a NotificationManager injected via set_notification_manager().
"""

import pytest

from backend.notifications.models import (
    NotificationCategory,
    NotificationLevel,
    NotificationType,
)


class TestListNotifications:
    """GET /api/notifications returns notification history."""

    @pytest.mark.asyncio
    async def test_list_empty(self, client, notification_manager):
        """Empty history returns count 0."""
        response = await client.get("/api/notifications")
        assert response.status_code == 200
        data = response.json()
        assert data["notifications"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_list_after_notify(self, client, notification_manager):
        """Notifications appear in the list after creation."""
        await notification_manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Test Notification",
            message="Hello from test",
        )
        response = await client.get("/api/notifications")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["notifications"][0]["title"] == "Test Notification"

    @pytest.mark.asyncio
    async def test_list_filter_by_category(self, client, notification_manager):
        """Category filter returns only matching notifications."""
        await notification_manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="System",
            message="sys",
        )
        await notification_manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.TRAINING,
            title="Training",
            message="train",
        )
        response = await client.get("/api/notifications?category=training")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["notifications"][0]["title"] == "Training"

    @pytest.mark.asyncio
    async def test_list_invalid_category(self, client, notification_manager):
        """Invalid category returns 422."""
        response = await client.get("/api/notifications?category=bogus")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_list_unread_only(self, client, notification_manager):
        """unread_only=true excludes read notifications."""
        n = await notification_manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Read Me",
            message="m",
        )
        await notification_manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Still Unread",
            message="m",
        )
        notification_manager.mark_read(n.id)
        response = await client.get("/api/notifications?unread_only=true")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["notifications"][0]["title"] == "Still Unread"


class TestMarkRead:
    """POST /api/notifications/{id}/read marks a notification as read."""

    @pytest.mark.asyncio
    async def test_mark_read_success(self, client, notification_manager):
        """Returns 200 with updated notification when found."""
        n = await notification_manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Mark Me",
            message="m",
        )
        response = await client.post(f"/api/notifications/{n.id}/read")
        assert response.status_code == 200
        data = response.json()
        assert data["read"] is True
        assert data["id"] == n.id

    @pytest.mark.asyncio
    async def test_mark_read_not_found(self, client, notification_manager):
        """Returns 404 for a nonexistent notification ID."""
        response = await client.post("/api/notifications/nonexistent-id/read")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data


class TestDismiss:
    """POST /api/notifications/{id}/dismiss dismisses a notification."""

    @pytest.mark.asyncio
    async def test_dismiss_success(self, client, notification_manager):
        """Returns 200 with updated notification when found."""
        n = await notification_manager.notify(
            type=NotificationType.BANNER,
            level=NotificationLevel.WARNING,
            category=NotificationCategory.SYSTEM,
            title="Dismiss Me",
            message="m",
        )
        response = await client.post(f"/api/notifications/{n.id}/dismiss")
        assert response.status_code == 200
        data = response.json()
        assert data["dismissed"] is True
        assert data["id"] == n.id

    @pytest.mark.asyncio
    async def test_dismiss_not_found(self, client, notification_manager):
        """Returns 404 for a nonexistent notification ID."""
        response = await client.post("/api/notifications/nonexistent-id/dismiss")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data


class TestClearNotifications:
    """DELETE /api/notifications clears notification history."""

    @pytest.mark.asyncio
    async def test_clear_all(self, client, notification_manager):
        """Returns 204 and empties the list."""
        await notification_manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Gone",
            message="m",
        )
        response = await client.delete("/api/notifications")
        assert response.status_code == 204

        # Verify cleared
        list_response = await client.get("/api/notifications")
        assert list_response.json()["count"] == 0

    @pytest.mark.asyncio
    async def test_clear_by_category(self, client, notification_manager):
        """Clears only the specified category."""
        await notification_manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Keep",
            message="m",
        )
        await notification_manager.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.TRAINING,
            title="Remove",
            message="m",
        )
        response = await client.delete("/api/notifications?category=training")
        assert response.status_code == 204

        list_response = await client.get("/api/notifications")
        data = list_response.json()
        assert data["count"] == 1
        assert data["notifications"][0]["title"] == "Keep"

    @pytest.mark.asyncio
    async def test_clear_empty_is_204(self, client, notification_manager):
        """Clearing when already empty still returns 204."""
        response = await client.delete("/api/notifications")
        assert response.status_code == 204

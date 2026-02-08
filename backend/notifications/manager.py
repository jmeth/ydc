"""
Notification manager — central hub for creating, storing, and
broadcasting notifications to WebSocket clients.

Subscribes to EventBus events and translates them into user-facing
notifications. Maintains a capped in-memory history and exposes
read/dismiss/clear operations for the REST API.
"""

import logging
from typing import Any

from backend.core.events import (
    EventBus,
    TRAINING_COMPLETED,
    TRAINING_ERROR,
    RESOURCE_WARNING,
    RESOURCE_CRITICAL,
    FEED_ADDED,
    FEED_REMOVED,
    FEED_ERROR,
    INFERENCE_STARTED,
    INFERENCE_STOPPED,
    INFERENCE_ERROR,
    DATASET_CREATED,
    DATASET_DELETED,
    DATASET_IMAGE_ADDED,
    DATASET_IMAGE_DELETED,
)
from backend.notifications.models import (
    Notification,
    NotificationCategory,
    NotificationLevel,
    NotificationType,
)
from backend.websocket.manager import ConnectionManager

logger = logging.getLogger(__name__)


class NotificationManager:
    """
    Creates, stores, and broadcasts notifications.

    Args:
        connection_manager: WebSocket connection manager for broadcasting.
        max_history: Maximum number of notifications to keep in memory.
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        max_history: int = 100,
    ):
        self._connection_manager = connection_manager
        self._max_history = max_history
        self._notifications: list[Notification] = []

    def setup_event_subscriptions(self, event_bus: EventBus) -> None:
        """
        Wire EventBus events to notification callbacks.

        Each callback creates a notification with an appropriate type,
        level, and category for the triggering event.

        Args:
            event_bus: The event bus to subscribe to.
        """
        event_bus.subscribe(TRAINING_COMPLETED, self._on_training_completed)
        event_bus.subscribe(TRAINING_ERROR, self._on_training_error)
        event_bus.subscribe(RESOURCE_WARNING, self._on_resource_warning)
        event_bus.subscribe(RESOURCE_CRITICAL, self._on_resource_critical)
        event_bus.subscribe(FEED_ADDED, self._on_feed_added)
        event_bus.subscribe(FEED_REMOVED, self._on_feed_removed)
        event_bus.subscribe(FEED_ERROR, self._on_feed_error)
        event_bus.subscribe(INFERENCE_STARTED, self._on_inference_started)
        event_bus.subscribe(INFERENCE_STOPPED, self._on_inference_stopped)
        event_bus.subscribe(INFERENCE_ERROR, self._on_inference_error)
        event_bus.subscribe(DATASET_CREATED, self._on_dataset_created)
        event_bus.subscribe(DATASET_DELETED, self._on_dataset_deleted)
        event_bus.subscribe(DATASET_IMAGE_ADDED, self._on_dataset_image_added)
        event_bus.subscribe(DATASET_IMAGE_DELETED, self._on_dataset_image_deleted)
        logger.info("Notification event subscriptions registered")

    async def notify(
        self,
        type: NotificationType,
        level: NotificationLevel,
        category: NotificationCategory,
        title: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> Notification:
        """
        Create a notification, store it in history, and broadcast via WebSocket.

        Args:
            type: Presentation type (toast, banner, etc.).
            level: Severity level.
            category: Source subsystem.
            title: Short summary line.
            message: Descriptive text.
            data: Optional extra payload.

        Returns:
            The created Notification.
        """
        notification = Notification(
            type=type,
            level=level,
            category=category,
            title=title,
            message=message,
            data=data,
        )

        self._notifications.append(notification)

        # Trim history to max size (drop oldest)
        if len(self._notifications) > self._max_history:
            self._notifications = self._notifications[-self._max_history:]

        # Broadcast to all WebSocket clients
        await self._connection_manager.broadcast_event("notification", {
            "id": notification.id,
            "notification_type": notification.type.value,
            "level": notification.level.value,
            "category": notification.category.value,
            "title": notification.title,
            "message": notification.message,
            "timestamp": notification.timestamp.isoformat(),
            "data": notification.data,
        })

        logger.debug("Notification created: [%s] %s", notification.level.value, notification.title)
        return notification

    def mark_read(self, notification_id: str) -> bool:
        """
        Mark a notification as read.

        Args:
            notification_id: The notification ID to mark.

        Returns:
            True if found and marked, False if not found.
        """
        for n in self._notifications:
            if n.id == notification_id:
                n.read = True
                return True
        return False

    def dismiss(self, notification_id: str) -> bool:
        """
        Dismiss a notification.

        Args:
            notification_id: The notification ID to dismiss.

        Returns:
            True if found and dismissed, False if not found.
        """
        for n in self._notifications:
            if n.id == notification_id:
                n.dismissed = True
                return True
        return False

    def get_history(
        self,
        category: NotificationCategory | None = None,
        unread_only: bool = False,
        limit: int = 50,
    ) -> list[Notification]:
        """
        Retrieve notification history with optional filters.

        Args:
            category: Filter by source category (None for all).
            unread_only: If True, only return unread notifications.
            limit: Maximum number of notifications to return.

        Returns:
            List of matching notifications, newest first.
        """
        result = self._notifications
        if category is not None:
            result = [n for n in result if n.category == category]
        if unread_only:
            result = [n for n in result if not n.read]
        # Return newest first, capped at limit
        return list(reversed(result))[:limit]

    def clear_all(self, category: NotificationCategory | None = None) -> None:
        """
        Clear notification history.

        Args:
            category: If provided, only clear notifications in this category.
                      If None, clear all notifications.
        """
        if category is not None:
            self._notifications = [
                n for n in self._notifications if n.category != category
            ]
        else:
            self._notifications.clear()

    # --- Event callbacks ---

    async def _on_training_completed(self, data: dict[str, Any]) -> None:
        """Handle training.completed event."""
        await self.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.SUCCESS,
            category=NotificationCategory.TRAINING,
            title="Training Complete",
            message=data.get("message", "Model training finished successfully."),
            data=data,
        )

    async def _on_training_error(self, data: dict[str, Any]) -> None:
        """Handle training.error event."""
        await self.notify(
            type=NotificationType.ALERT,
            level=NotificationLevel.ERROR,
            category=NotificationCategory.TRAINING,
            title="Training Error",
            message=data.get("message", "An error occurred during training."),
            data=data,
        )

    async def _on_resource_warning(self, data: dict[str, Any]) -> None:
        """Handle resource.warning event."""
        await self.notify(
            type=NotificationType.BANNER,
            level=NotificationLevel.WARNING,
            category=NotificationCategory.SYSTEM,
            title="Resource Warning",
            message=data.get("message", "System resources are running low."),
            data=data,
        )

    async def _on_resource_critical(self, data: dict[str, Any]) -> None:
        """Handle resource.critical event."""
        await self.notify(
            type=NotificationType.BANNER,
            level=NotificationLevel.ERROR,
            category=NotificationCategory.SYSTEM,
            title="Resource Critical",
            message=data.get("message", "System resources are critically low."),
            data=data,
        )

    async def _on_feed_added(self, data: dict[str, Any]) -> None:
        """Handle feed.added event."""
        feed_name = data.get("name", data.get("feed_id", "Unknown"))
        await self.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SCAN,
            title="Feed Added",
            message=f"Feed '{feed_name}' has been added.",
            data=data,
        )

    async def _on_feed_removed(self, data: dict[str, Any]) -> None:
        """Handle feed.removed event."""
        feed_name = data.get("name", data.get("feed_id", "Unknown"))
        await self.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SCAN,
            title="Feed Removed",
            message=f"Feed '{feed_name}' has been removed.",
            data=data,
        )

    async def _on_feed_error(self, data: dict[str, Any]) -> None:
        """Handle feed.error event."""
        feed_name = data.get("name", data.get("feed_id", "Unknown"))
        await self.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.ERROR,
            category=NotificationCategory.SCAN,
            title="Feed Error",
            message=data.get("message", f"An error occurred with feed '{feed_name}'."),
            data=data,
        )

    async def _on_inference_started(self, data: dict[str, Any]) -> None:
        """Handle inference.started event."""
        model_name = data.get("model_name", "Unknown")
        await self.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.INFERENCE,
            title="Inference Started",
            message=f"Inference started with model '{model_name}'.",
            data=data,
        )

    async def _on_inference_stopped(self, data: dict[str, Any]) -> None:
        """Handle inference.stopped event."""
        model_name = data.get("model_name", "Unknown")
        frames = data.get("frames_processed", 0)
        await self.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.INFERENCE,
            title="Inference Stopped",
            message=f"Inference with model '{model_name}' stopped after {frames} frames.",
            data=data,
        )

    async def _on_inference_error(self, data: dict[str, Any]) -> None:
        """Handle inference.error event."""
        await self.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.ERROR,
            category=NotificationCategory.INFERENCE,
            title="Inference Error",
            message=data.get("message", "An error occurred during inference."),
            data=data,
        )

    async def _on_dataset_created(self, data: dict[str, Any]) -> None:
        """Handle dataset.created event."""
        name = data.get("name", "Unknown")
        await self.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.SUCCESS,
            category=NotificationCategory.DATASET,
            title="Dataset Created",
            message=f"Dataset '{name}' has been created.",
            data=data,
        )

    async def _on_dataset_deleted(self, data: dict[str, Any]) -> None:
        """Handle dataset.deleted event."""
        name = data.get("name", "Unknown")
        await self.notify(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.DATASET,
            title="Dataset Deleted",
            message=f"Dataset '{name}' has been deleted.",
            data=data,
        )

    async def _on_dataset_image_added(self, data: dict[str, Any]) -> None:
        """Handle dataset.image_added event."""
        name = data.get("name", "Unknown")
        filename = data.get("filename", "Unknown")
        split = data.get("split", "Unknown")
        await self.notify(
            type=NotificationType.STATUS,
            level=NotificationLevel.INFO,
            category=NotificationCategory.DATASET,
            title="Image Added",
            message=f"Image '{filename}' added to {name}/{split}.",
            data=data,
        )

    async def _on_dataset_image_deleted(self, data: dict[str, Any]) -> None:
        """Handle dataset.image_deleted event."""
        name = data.get("name", "Unknown")
        filename = data.get("filename", "Unknown")
        await self.notify(
            type=NotificationType.STATUS,
            level=NotificationLevel.INFO,
            category=NotificationCategory.DATASET,
            title="Image Deleted",
            message=f"Image '{filename}' removed from dataset '{name}'.",
            data=data,
        )

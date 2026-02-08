"""
Notifications subsystem — centralized notification management.

Provides NotificationManager for creating, storing, and broadcasting
notifications, plus domain model types.
"""

from backend.notifications.manager import NotificationManager
from backend.notifications.models import (
    Notification,
    NotificationCategory,
    NotificationLevel,
    NotificationType,
)

__all__ = [
    "NotificationManager",
    "Notification",
    "NotificationCategory",
    "NotificationLevel",
    "NotificationType",
]

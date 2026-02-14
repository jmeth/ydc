"""
Notification domain models.

Defines enums for notification type, level, and category, plus the
Notification dataclass used as the internal representation throughout
the notifications subsystem.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class NotificationType(str, Enum):
    """How the notification is presented to the user."""
    TOAST = "toast"       # Auto-dismiss after a few seconds
    BANNER = "banner"     # Persistent until manually dismissed
    ALERT = "alert"       # Same as banner for MVP
    STATUS = "status"     # Silent — no UI, for programmatic consumers


class NotificationLevel(str, Enum):
    """Severity level controlling visual styling."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class NotificationCategory(str, Enum):
    """Source subsystem that produced the notification."""
    SYSTEM = "system"
    SCAN = "scan"
    TRAINING = "training"
    DATASET = "dataset"
    INFERENCE = "inference"
    CAPTURE = "capture"


@dataclass
class Notification:
    """
    Internal notification record.

    Attributes:
        id: Unique identifier (auto-generated UUID4).
        type: Presentation type (toast, banner, alert, status).
        level: Severity level (info, success, warning, error).
        category: Source subsystem category.
        title: Short summary line.
        message: Longer descriptive text.
        timestamp: UTC creation time (auto-set).
        read: Whether the user has marked it as read.
        dismissed: Whether the user has dismissed it.
        data: Optional extra payload for programmatic consumers.
    """
    type: NotificationType
    level: NotificationLevel
    category: NotificationCategory
    title: str
    message: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    read: bool = False
    dismissed: bool = False
    data: dict[str, Any] | None = None

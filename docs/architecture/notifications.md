# Notifications Subsystem

> Part of the [YOLO Dataset Creator Architecture](../ARCHITECTURE.md)

## Responsibility

Centralized notification management for all system events. Delivers alerts to frontend via WebSocket, manages notification state, and handles external notification channels in ideal state.

## Components

```
Notifications Subsystem
├── Notification Manager
│   ├── Event Receiver (subscribes to EventBus)
│   ├── Notification Queue
│   ├── Delivery Manager
│   └── State Tracker (read/unread)
├── Channels
│   ├── WebSocket Channel (real-time to frontend)    [MVP]
│   ├── Desktop Notifications (system tray)          [Ideal]
│   ├── Sound Alerts                                 [Ideal]
│   └── Webhook Channel (external HTTP)              [Ideal]
└── Notification Types
    ├── Toast (temporary, auto-dismiss)
    ├── Banner (persistent until dismissed)
    ├── Alert (requires acknowledgment)
    └── Status Update (silent state change)
```

## Notification Model

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import time

class NotificationType(Enum):
    TOAST = "toast"           # Temporary, auto-dismiss (3-5 sec)
    BANNER = "banner"         # Persistent until dismissed
    ALERT = "alert"           # Requires user acknowledgment
    STATUS = "status"         # Silent state update

class NotificationLevel(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"

class NotificationCategory(Enum):
    SYSTEM = "system"         # Resource warnings, errors
    SCAN = "scan"             # Capture events, scan status
    TRAINING = "training"     # Progress, completion, errors
    DATASET = "dataset"       # Import/export, validation
    INFERENCE = "inference"   # Model loading, detection events

@dataclass
class Notification:
    id: str
    type: NotificationType
    level: NotificationLevel
    category: NotificationCategory
    title: str
    message: str
    timestamp: float = None
    read: bool = False
    dismissed: bool = False
    data: Optional[dict] = None      # Additional context
    action: Optional[dict] = None    # Optional action button

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
```

## Notification Manager

```python
from typing import Callable
import uuid

class NotificationManager:
    """Centralized notification handling"""

    def __init__(self, event_bus: EventBus, websocket_handler):
        self._event_bus = event_bus
        self._ws = websocket_handler
        self._notifications: list[Notification] = []
        self._channels: list[NotificationChannel] = []
        self._max_history = 100

        # Subscribe to system events
        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self):
        """Wire up event bus to notifications"""
        # Training events
        self._event_bus.subscribe(
            EventBus.TRAINING_COMPLETED,
            lambda data: self.notify(
                type=NotificationType.TOAST,
                level=NotificationLevel.SUCCESS,
                category=NotificationCategory.TRAINING,
                title="Training Complete",
                message=f"Model '{data['model_name']}' finished training",
                data=data
            )
        )

        self._event_bus.subscribe(
            EventBus.TRAINING_ERROR,
            lambda data: self.notify(
                type=NotificationType.ALERT,
                level=NotificationLevel.ERROR,
                category=NotificationCategory.TRAINING,
                title="Training Failed",
                message=data.get('error', 'Unknown error'),
                data=data
            )
        )

        # Resource events
        self._event_bus.subscribe(
            EventBus.RESOURCE_WARNING,
            lambda data: self.notify(
                type=NotificationType.BANNER,
                level=NotificationLevel.WARNING,
                category=NotificationCategory.SYSTEM,
                title="Resource Constraint",
                message=f"System resources are constrained. {data.get('action', '')}",
                data=data
            )
        )

        # Scan events
        self._event_bus.subscribe(
            EventBus.SCAN_CAPTURE,
            lambda data: self.notify(
                type=NotificationType.STATUS,
                level=NotificationLevel.INFO,
                category=NotificationCategory.SCAN,
                title="Frame Captured",
                message=f"Captured {data.get('type', '')} frame",
                data=data
            )
        )

    def notify(
        self,
        type: NotificationType,
        level: NotificationLevel,
        category: NotificationCategory,
        title: str,
        message: str,
        data: dict = None,
        action: dict = None
    ) -> Notification:
        """Create and dispatch a notification"""
        notification = Notification(
            id=str(uuid.uuid4()),
            type=type,
            level=level,
            category=category,
            title=title,
            message=message,
            data=data,
            action=action
        )

        # Store in history
        self._notifications.append(notification)
        if len(self._notifications) > self._max_history:
            self._notifications.pop(0)

        # Dispatch to all channels
        self._dispatch(notification)

        return notification

    def _dispatch(self, notification: Notification):
        """Send notification through all active channels"""
        # Always send via WebSocket to frontend
        self._ws.emit('notification', {
            'id': notification.id,
            'type': notification.type.value,
            'level': notification.level.value,
            'category': notification.category.value,
            'title': notification.title,
            'message': notification.message,
            'timestamp': notification.timestamp,
            'data': notification.data,
            'action': notification.action
        })

        # Dispatch to other channels [Ideal]
        for channel in self._channels:
            channel.send(notification)

    def dismiss(self, notification_id: str) -> bool:
        """Mark notification as dismissed"""
        for n in self._notifications:
            if n.id == notification_id:
                n.dismissed = True
                return True
        return False

    def mark_read(self, notification_id: str) -> bool:
        """Mark notification as read"""
        for n in self._notifications:
            if n.id == notification_id:
                n.read = True
                return True
        return False

    def get_history(
        self,
        category: NotificationCategory = None,
        unread_only: bool = False,
        limit: int = 50
    ) -> list[Notification]:
        """Get notification history with optional filters"""
        result = self._notifications
        if category:
            result = [n for n in result if n.category == category]
        if unread_only:
            result = [n for n in result if not n.read]
        return result[-limit:]

    def clear_all(self, category: NotificationCategory = None):
        """Clear all notifications, optionally by category"""
        if category:
            self._notifications = [
                n for n in self._notifications
                if n.category != category
            ]
        else:
            self._notifications = []
```

## WebSocket Events

```
# Server → Client
notification              New notification (all types)
notification.dismiss      Notification auto-dismissed (timeout)

# Client → Server
notification.read         Mark notification as read
notification.dismiss      Dismiss notification
notification.clear        Clear notification history
```

## MVP API Endpoints

```
GET    /api/notifications              Get notification history
POST   /api/notifications/:id/read     Mark as read
POST   /api/notifications/:id/dismiss  Dismiss notification
DELETE /api/notifications              Clear all notifications
```

## Frontend Integration

```javascript
// Notification display component
class NotificationManager {
    constructor(websocket) {
        this.notifications = [];
        this.toastContainer = document.getElementById('toast-container');
        this.bannerContainer = document.getElementById('banner-container');

        websocket.on('notification', (data) => this.handleNotification(data));
    }

    handleNotification(notification) {
        this.notifications.push(notification);

        switch (notification.type) {
            case 'toast':
                this.showToast(notification);
                break;
            case 'banner':
                this.showBanner(notification);
                break;
            case 'alert':
                this.showAlert(notification);
                break;
            case 'status':
                // Update status indicators silently
                this.updateStatus(notification);
                break;
        }
    }

    showToast(notification) {
        const toast = this.createToastElement(notification);
        this.toastContainer.appendChild(toast);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }

    showBanner(notification) {
        // Persistent banner at top of screen
        const banner = this.createBannerElement(notification);
        this.bannerContainer.appendChild(banner);
    }
}
```

## MVP vs Ideal Features

| Feature | MVP | Ideal |
|---------|-----|-------|
| Toast notifications | Yes | Yes |
| Banner notifications | Yes | Yes |
| Alert dialogs | Yes | Yes |
| WebSocket delivery | Yes | Yes |
| Notification history | No | Yes |
| Desktop notifications | No | Yes |
| Sound alerts | No | Yes |
| Webhook channel | No | Yes |
| Notification preferences | No | Yes |

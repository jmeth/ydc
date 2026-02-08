# Sending Notifications from Backend Code

This guide shows how to create and send notifications from Python backend code — either directly via the NotificationManager or indirectly by publishing EventBus events.

## Option 1: Publish an EventBus event (preferred)

The simplest approach. Publish an event and the NotificationManager's subscriptions handle the rest. Use this when your subsystem already publishes events.

```python
from backend.core.events import event_bus, FEED_ADDED, TRAINING_ERROR

# This automatically creates a toast/info/scan notification
await event_bus.publish(FEED_ADDED, {
    "feed_id": "abc-123",
    "name": "My Camera",
})

# This automatically creates an alert/error/training notification
await event_bus.publish(TRAINING_ERROR, {
    "message": "Out of memory during epoch 15",
})
```

The `message` key in the event data becomes the notification message. If omitted, a sensible default is used.

### Supported events

These events are automatically wired to notifications:

| Event constant | Import path | Notification created |
|----------------|-------------|---------------------|
| `TRAINING_COMPLETED` | `backend.core.events` | toast / success / training |
| `TRAINING_ERROR` | `backend.core.events` | alert / error / training |
| `RESOURCE_WARNING` | `backend.core.events` | banner / warning / system |
| `RESOURCE_CRITICAL` | `backend.core.events` | banner / error / system |
| `FEED_ADDED` | `backend.core.events` | toast / info / scan |
| `FEED_REMOVED` | `backend.core.events` | toast / info / scan |
| `FEED_ERROR` | `backend.core.events` | toast / error / scan |

## Option 2: Call NotificationManager.notify() directly

For notifications that don't map to an existing event, call `notify()` on the manager instance. This stores the notification in history and broadcasts it via WebSocket immediately.

```python
from backend.notifications.models import (
    NotificationCategory,
    NotificationLevel,
    NotificationType,
)

# Get the manager — typically from the API module's injected reference
from backend.api.notifications import _get_manager

manager = _get_manager()

# Send a toast
await manager.notify(
    type=NotificationType.TOAST,
    level=NotificationLevel.SUCCESS,
    category=NotificationCategory.DATASET,
    title="Export Complete",
    message="Dataset 'coco-v2' exported to /datasets/coco-v2.zip",
)

# Send a persistent banner
await manager.notify(
    type=NotificationType.BANNER,
    level=NotificationLevel.WARNING,
    category=NotificationCategory.SYSTEM,
    title="Disk Space Low",
    message="Less than 1 GB remaining on /datasets volume.",
    data={"bytes_free": 512_000_000},
)

# Send a silent status (no UI, history only)
await manager.notify(
    type=NotificationType.STATUS,
    level=NotificationLevel.INFO,
    category=NotificationCategory.INFERENCE,
    title="Model Loaded",
    message="yolov8n.pt loaded in 1.2s",
)
```

### Parameters

| Param | Type | Description |
|-------|------|-------------|
| `type` | `NotificationType` | `TOAST`, `BANNER`, `ALERT`, or `STATUS` |
| `level` | `NotificationLevel` | `INFO`, `SUCCESS`, `WARNING`, or `ERROR` |
| `category` | `NotificationCategory` | `SYSTEM`, `SCAN`, `TRAINING`, `DATASET`, or `INFERENCE` |
| `title` | `str` | Short summary line |
| `message` | `str` | Descriptive text |
| `data` | `dict` or `None` | Optional extra payload sent to WebSocket clients |

### Return value

`notify()` returns the created `Notification` dataclass with auto-generated `id` and `timestamp`:

```python
n = await manager.notify(...)
print(n.id)         # "c4a1b2d3-..."
print(n.timestamp)  # datetime(2026, 2, 8, 12, 30, tzinfo=UTC)
```

## Option 3: Add a new event subscription

To wire a new event type to automatic notifications, add a callback in `NotificationManager.setup_event_subscriptions()`:

```python
# In backend/notifications/manager.py

def setup_event_subscriptions(self, event_bus: EventBus) -> None:
    # ... existing subscriptions ...
    event_bus.subscribe(INFERENCE_STARTED, self._on_inference_started)

async def _on_inference_started(self, data: dict[str, Any]) -> None:
    """Handle inference.started event."""
    await self.notify(
        type=NotificationType.TOAST,
        level=NotificationLevel.INFO,
        category=NotificationCategory.INFERENCE,
        title="Inference Started",
        message=data.get("message", "Inference pipeline is running."),
        data=data,
    )
```

Then add a test in `backend/tests/test_notification_manager.py`:

```python
@pytest.mark.asyncio
async def test_inference_started_creates_toast(self, manager, mock_connection_manager):
    """Publishing inference.started creates an info toast."""
    bus = EventBus()
    await bus.start()
    manager.setup_event_subscriptions(bus)
    await bus.publish(INFERENCE_STARTED, {"message": "Running yolov8n"})
    history = manager.get_history()
    assert len(history) == 1
    assert history[0].type == NotificationType.TOAST
    assert history[0].level == NotificationLevel.INFO
    await bus.stop()
```

## Choosing the right type and level

| Situation | Type | Level |
|-----------|------|-------|
| Routine success (export done, feed added) | `TOAST` | `SUCCESS` or `INFO` |
| Actionable warning (low disk, slow model) | `BANNER` | `WARNING` |
| Critical failure (training crashed, GPU error) | `ALERT` | `ERROR` |
| Background status update (model loaded) | `STATUS` | `INFO` |

# Notifications API Client Guide

This guide covers how to subscribe to notifications via WebSocket and manage notification history via the REST API.

## Base URL

```
http://localhost:8000/api/notifications
```

## Receiving notifications (WebSocket)

Notifications are pushed in real time over the existing events WebSocket. Connect to:

```
ws://localhost:8000/ws/events
```

When a notification fires, the server sends a JSON message:

```json
{
  "type": "notification",
  "id": "c4a1b2d3-...",
  "notification_type": "toast",
  "level": "success",
  "category": "training",
  "title": "Training Complete",
  "message": "Model training finished successfully.",
  "timestamp": "2026-02-08T12:30:00+00:00",
  "data": {"model": "yolov8n", "epochs": 50}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"notification"` for notification messages |
| `id` | string | Unique notification ID (UUID) |
| `notification_type` | string | `"toast"`, `"banner"`, `"alert"`, or `"status"` |
| `level` | string | `"info"`, `"success"`, `"warning"`, or `"error"` |
| `category` | string | Source subsystem: `"system"`, `"scan"`, `"training"`, `"dataset"`, `"inference"` |
| `title` | string | Short summary line |
| `message` | string | Longer descriptive text |
| `timestamp` | string | ISO 8601 UTC creation time |
| `data` | object or null | Optional extra payload |

### Notification types

| Type | Behavior |
|------|----------|
| `toast` | Auto-dismiss after a few seconds |
| `banner` | Persistent until the user dismisses it |
| `alert` | Same as banner (MVP) |
| `status` | Silent — no UI, for programmatic consumers |

### JavaScript example

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/events");

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "notification") {
    console.log(`[${msg.level}] ${msg.title}: ${msg.message}`);

    if (msg.notification_type === "toast") {
      showToast(msg);
    } else if (msg.notification_type === "banner" || msg.notification_type === "alert") {
      showBanner(msg);
    }
    // "status" type — no UI, handle programmatically
  }
};
```

### Vue composable example

The frontend already handles this in `App.vue` using the notification store:

```typescript
import { watch } from 'vue'
import { useWebSocket } from '@/composables/useWebSocket'
import { useNotificationStore } from '@/stores/notifications'
import type { WsNotificationMessage } from '@/types/websocket'

const { lastMessage } = useWebSocket('/ws/events')
const notificationStore = useNotificationStore()

watch(lastMessage, (msg) => {
  if (msg && typeof msg === 'object' && 'type' in msg) {
    if ((msg as { type: string }).type === 'notification') {
      notificationStore.handleServerNotification(msg as WsNotificationMessage)
    }
  }
})
```

## REST endpoints

### List notifications

```
GET /api/notifications
```

Query parameters:

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `category` | string | (all) | Filter: `system`, `scan`, `training`, `dataset`, `inference` |
| `unread_only` | bool | `false` | Only return unread notifications |
| `limit` | int | `50` | Max results (1-100) |

**Response** `200`:
```json
{
  "notifications": [
    {
      "id": "c4a1b2d3-...",
      "type": "toast",
      "level": "success",
      "category": "training",
      "title": "Training Complete",
      "message": "Model training finished successfully.",
      "timestamp": "2026-02-08T12:30:00+00:00",
      "read": false,
      "dismissed": false,
      "data": null
    }
  ],
  "count": 1
}
```

### Mark as read

```
POST /api/notifications/{id}/read
```

**Responses**:
- `200` — Returns the updated notification with `"read": true`
- `404` — Notification not found

### Dismiss

```
POST /api/notifications/{id}/dismiss
```

**Responses**:
- `200` — Returns the updated notification with `"dismissed": true`
- `404` — Notification not found

### Clear all

```
DELETE /api/notifications
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `category` | string | (all) | Clear only this category |

**Responses**:
- `204` — Cleared (no body)

## curl examples

```bash
# List all notifications
curl http://localhost:8000/api/notifications

# List only unread training notifications
curl "http://localhost:8000/api/notifications?category=training&unread_only=true"

# Mark a notification as read
curl -X POST http://localhost:8000/api/notifications/{id}/read

# Dismiss a notification
curl -X POST http://localhost:8000/api/notifications/{id}/dismiss

# Clear all notifications
curl -X DELETE http://localhost:8000/api/notifications

# Clear only system notifications
curl -X DELETE "http://localhost:8000/api/notifications?category=system"
```

## Automatic event-driven notifications

The NotificationManager subscribes to EventBus events and creates notifications automatically. No manual action is needed for these — they fire when the underlying subsystem publishes an event.

| Event | Type | Level | Category |
|-------|------|-------|----------|
| `training.completed` | toast | success | training |
| `training.error` | alert | error | training |
| `resource.warning` | banner | warning | system |
| `resource.critical` | banner | error | system |
| `feed.added` | toast | info | scan |
| `feed.removed` | toast | info | scan |
| `feed.error` | toast | error | scan |

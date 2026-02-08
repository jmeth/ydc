# Feed API Client Guide

This guide covers how to use the Feed REST API and WebSocket stream as a client (frontend, script, or curl).

## Base URL

```
http://localhost:8000/api/feeds
```

## Endpoints

### List all feeds

```
GET /api/feeds
```

**Response** `200`:
```json
{
  "feeds": [
    {
      "feed_id": "a1b2c3d4-...",
      "feed_type": "camera",
      "source": "0",
      "name": "camera-0",
      "status": "active",
      "fps": 30.0,
      "resolution": [640, 480],
      "frame_count": 1523
    }
  ],
  "count": 1
}
```

### Create a feed

```
POST /api/feeds
Content-Type: application/json

{
  "feed_type": "camera",
  "source": "0",
  "name": "My Webcam",
  "buffer_size": 30
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `feed_type` | yes | | `"camera"`, `"rtsp"`, or `"file"` |
| `source` | yes | | Camera index (`"0"`), RTSP URL, or file path |
| `name` | no | auto-generated | Human-readable label |
| `buffer_size` | no | `30` | Ring buffer capacity (1-300) |

**Responses**:
- `201` — Feed created, returns `FeedInfo` object
- `400` — Connection failed (camera unavailable, bad URL, etc.)
- `422` — Invalid feed type or missing required fields

### Get feed details

```
GET /api/feeds/{feed_id}
```

**Responses**:
- `200` — Returns `FeedInfo` object
- `404` — Feed not found

### Delete a feed

```
DELETE /api/feeds/{feed_id}
```

Stops capture, disconnects the source, and removes the feed.

**Responses**:
- `204` — Deleted (no body)
- `404` — Feed not found

### Pause a feed

```
POST /api/feeds/{feed_id}/pause
```

Pauses frame capture. The connection stays open so resume is fast.

**Responses**:
- `200` — Returns updated `FeedInfo` with `"status": "paused"`
- `404` — Feed not found
- `409` — Feed is not in active state (already paused or errored)

### Resume a feed

```
POST /api/feeds/{feed_id}/resume
```

**Responses**:
- `200` — Returns updated `FeedInfo` with `"status": "active"`
- `404` — Feed not found
- `409` — Feed is not paused

### Get a snapshot

```
GET /api/feeds/{feed_id}/snapshot
```

Returns the latest frame as a JPEG image.

**Responses**:
- `200` — JPEG binary (`Content-Type: image/jpeg`)
- `404` — Feed not found or no frames captured yet

## WebSocket streaming

Connect to the video WebSocket to receive live frames:

```
ws://localhost:8000/ws/video
```

### Subscribe to a feed

After connecting, send a JSON message to subscribe:

```json
{"action": "subscribe", "feed_id": "a1b2c3d4-..."}
```

### Receive frames

The server sends JSON messages at the configured stream FPS (default 15):

```json
{
  "type": "frame",
  "feed_id": "a1b2c3d4-...",
  "data": "/9j/4AAQ...",
  "detections": [],
  "timestamp": 1707350400.123
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"frame"` |
| `feed_id` | string | Which feed this frame is from |
| `data` | string | Base64-encoded JPEG image |
| `detections` | array | Detection objects (empty for raw feeds) |
| `timestamp` | float | Unix timestamp of the frame |

To display the frame in an `<img>` tag:

```javascript
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "frame") {
    document.getElementById("video").src = `data:image/jpeg;base64,${msg.data}`;
  }
};
```

### Unsubscribe

```json
{"action": "unsubscribe", "feed_id": "a1b2c3d4-..."}
```

## curl examples

```bash
# List feeds
curl http://localhost:8000/api/feeds

# Create a camera feed
curl -X POST http://localhost:8000/api/feeds \
  -H "Content-Type: application/json" \
  -d '{"feed_type": "camera", "source": "0", "name": "Webcam"}'

# Get feed details
curl http://localhost:8000/api/feeds/{feed_id}

# Pause
curl -X POST http://localhost:8000/api/feeds/{feed_id}/pause

# Resume
curl -X POST http://localhost:8000/api/feeds/{feed_id}/resume

# Save a snapshot to file
curl http://localhost:8000/api/feeds/{feed_id}/snapshot -o snapshot.jpg

# Delete
curl -X DELETE http://localhost:8000/api/feeds/{feed_id}
```

## Feed statuses

| Status | Meaning |
|--------|---------|
| `connecting` | Feed is opening the source |
| `active` | Capturing frames normally |
| `paused` | Capture paused, connection held open |
| `error` | Too many consecutive read failures |
| `disconnected` | Feed removed or not yet connected |

## Configuration

These settings control streaming behavior (set via `YDC_` env vars):

| Env var | Default | Description |
|---------|---------|-------------|
| `YDC_FEED_STREAM_FPS` | `15.0` | WebSocket broadcast rate |
| `YDC_FEED_JPEG_QUALITY` | `70` | JPEG compression (0-100) |
| `YDC_FEED_DEFAULT_BUFFER_SIZE` | `30` | Default ring buffer size |

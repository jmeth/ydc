# Feeds Subsystem (Raw + Derived Feeds)

> Part of the [YOLO Dataset Creator Architecture](../ARCHITECTURE.md)

## Responsibility

Manage all video feeds - both raw input sources and derived feeds (like inference output). Abstracts different feed types behind a unified interface, handles buffering, and provides frames to subscribers.

## Key Concept: Derived Feeds

The Feeds Subsystem handles two categories of feeds:

1. **Raw Feeds**: Physical video sources (cameras, RTSP streams, video files)
2. **Derived Feeds**: Processed feeds produced by other subsystems (e.g., inference output with detections)

Both feed types share the same subscription interface, allowing any consumer (WebSocket, CaptureController, logging) to subscribe uniformly.

## Components

```
Feeds Subsystem
├── Feed Manager
│   ├── Feed Registry (active feeds - raw and derived)
│   ├── Feed Factory (create raw feeds by type)
│   ├── Derived Feed Registry (for inference output feeds)
│   └── Lifecycle Controller (start/stop/pause)
├── Raw Feed Sources
│   ├── CameraFeed (USB/CSI via OpenCV)          [MVP]
│   ├── RTSPFeed (network streams)               [Ideal]
│   ├── VideoFileFeed (mp4, etc.)                [Ideal]
│   └── ImageFolderFeed (batch processing)       [Ideal]
├── Derived Feeds
│   └── InferenceFeed (frames + detections from Inference Subsystem)
├── Frame Pipeline
│   ├── Ring Buffer (configurable depth)
│   ├── Frame Rate Normalizer                    [Ideal]
│   └── Frame Timestamper
└── Health Monitor
    ├── Connection Status
    ├── Frame Rate Tracking
    └── Auto-reconnect (for RTSP)                [Ideal]
```

## Feed Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

class FeedType(Enum):
    CAMERA = "camera"
    RTSP = "rtsp"           # Ideal
    VIDEO_FILE = "video"    # Ideal
    IMAGE_FOLDER = "images" # Ideal
    INFERENCE = "inference" # Derived feed (output from inference)

class FeedStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    PAUSED = "paused"
    ERROR = "error"
    ENDED = "ended"  # For finite sources (files)

@dataclass
class Frame:
    """Raw frame from a video source"""
    data: np.ndarray          # BGR image (OpenCV format)
    timestamp: float          # Capture time (Unix timestamp)
    sequence: int             # Frame number
    feed_id: str              # Source feed identifier
    width: int
    height: int

@dataclass
class Detection:
    """Single detection result"""
    class_id: int
    class_name: str
    confidence: float
    x_center: float           # Normalized (0-1)
    y_center: float
    width: float
    height: float

@dataclass
class InferenceFrame:
    """
    Derived frame from inference - contains both image and detections.
    This is what the Inference Subsystem produces and registers with Feeds.
    """
    data: np.ndarray          # BGR image (OpenCV format)
    timestamp: float
    sequence: int
    feed_id: str              # Inference feed ID
    source_feed_id: str       # Original raw feed ID
    width: int
    height: int
    detections: list[Detection]
    inference_time_ms: float  # How long inference took

@dataclass
class FeedInfo:
    feed_id: str
    feed_type: FeedType
    name: str
    status: FeedStatus
    fps: float
    resolution: tuple[int, int]
    frames_delivered: int
    errors: int
    source_feed_id: str | None = None  # For derived feeds

@dataclass
class FeedConfig:
    feed_type: FeedType
    source: str               # Device index, URL, or path
    name: str = ""            # Human-readable name
    buffer_size: int = 30     # Ring buffer depth (frames)
    target_fps: float = 0     # 0 = native rate
    reconnect_attempts: int = 3     # [Ideal]
    reconnect_delay: float = 2.0    # [Ideal]

class BaseFeed(ABC):
    """Abstract base class for all feed types"""

    def __init__(self, feed_id: str, config: FeedConfig):
        self.feed_id = feed_id
        self.config = config
        self._status = FeedStatus.DISCONNECTED

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to feed source"""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection and release resources"""
        ...

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Read next frame (non-blocking, returns None if not available)"""
        ...

    def get_status(self) -> FeedStatus:
        return self._status

    @property
    @abstractmethod
    def fps(self) -> float:
        """Native frame rate of source"""
        ...

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """Width, height of frames"""
        ...
```

## Feed Manager

```python
from typing import Callable, Union
from threading import Thread
from queue import Queue

FrameType = Union[Frame, InferenceFrame]  # Raw or derived frame

class FeedManager:
    """Manages multiple video feeds (raw and derived) with buffering"""

    def __init__(self):
        self._feeds: dict[str, BaseFeed] = {}
        self._derived_feeds: dict[str, DerivedFeed] = {}
        self._buffers: dict[str, RingBuffer] = {}
        self._threads: dict[str, Thread] = {}
        self._subscribers: dict[str, list[Callable]] = {}

    # === Raw Feed Management ===

    def add_feed(self, feed_id: str, config: FeedConfig) -> bool:
        """Register and start a new raw feed"""
        feed = FeedFactory.create(feed_id, config)
        if feed.connect():
            self._feeds[feed_id] = feed
            self._buffers[feed_id] = RingBuffer(config.buffer_size)
            self._start_capture_thread(feed_id)
            return True
        return False

    def remove_feed(self, feed_id: str) -> None:
        """Stop and remove a feed"""
        if feed_id in self._feeds:
            self._feeds[feed_id].disconnect()
            del self._feeds[feed_id]
            del self._buffers[feed_id]
        elif feed_id in self._derived_feeds:
            del self._derived_feeds[feed_id]
            del self._buffers[feed_id]

    # === Derived Feed Management ===

    def register_derived_feed(
        self,
        feed_id: str,
        source_feed_id: str,
        name: str = ""
    ) -> None:
        """
        Register a derived feed (e.g., inference output).
        The producer will push frames via push_derived_frame().
        """
        self._derived_feeds[feed_id] = DerivedFeed(
            feed_id=feed_id,
            source_feed_id=source_feed_id,
            name=name
        )
        self._buffers[feed_id] = RingBuffer(30)
        self._subscribers[feed_id] = []

    def push_derived_frame(self, feed_id: str, frame: InferenceFrame) -> None:
        """
        Push a frame to a derived feed (called by Inference Subsystem).
        This notifies all subscribers of the derived feed.
        """
        if feed_id not in self._derived_feeds:
            return

        buffer = self._buffers.get(feed_id)
        if buffer:
            buffer.push(frame)

        # Notify subscribers
        for callback in self._subscribers.get(feed_id, []):
            callback(frame)

    def unregister_derived_feed(self, feed_id: str) -> None:
        """Remove a derived feed"""
        if feed_id in self._derived_feeds:
            del self._derived_feeds[feed_id]
            del self._buffers[feed_id]

    # === Subscription (works for both raw and derived) ===

    def subscribe(self, feed_id: str, callback: Callable[[FrameType], None]) -> None:
        """Subscribe to receive frames as they arrive (raw or derived)"""
        if feed_id not in self._subscribers:
            self._subscribers[feed_id] = []
        self._subscribers[feed_id].append(callback)

    def unsubscribe(self, feed_id: str, callback: Callable) -> None:
        """Remove frame subscription"""
        if feed_id in self._subscribers:
            self._subscribers[feed_id].remove(callback)

    # === Query ===

    def get_frame(self, feed_id: str) -> Optional[FrameType]:
        """Get latest frame from buffer (non-blocking)"""
        if feed_id in self._buffers:
            return self._buffers[feed_id].get_latest()
        return None

    def get_frames(self, feed_id: str, count: int) -> list[FrameType]:
        """Get N most recent frames from buffer"""
        if feed_id in self._buffers:
            return self._buffers[feed_id].get_recent(count)
        return []

    def list_feeds(self) -> list[FeedInfo]:
        """List all feeds (raw and derived) with current status"""
        raw = [self._get_feed_info(fid) for fid in self._feeds]
        derived = [self._get_derived_feed_info(fid) for fid in self._derived_feeds]
        return raw + derived

    def get_status(self, feed_id: str) -> Optional[FeedStatus]:
        """Get feed connection status"""
        if feed_id in self._feeds:
            return self._feeds[feed_id].get_status()
        if feed_id in self._derived_feeds:
            return FeedStatus.CONNECTED  # Derived feeds are always "connected"
        return None

    # === Control ===

    def pause(self, feed_id: str) -> None:
        """Pause frame capture (keeps connection)"""
        ...

    def resume(self, feed_id: str) -> None:
        """Resume frame capture"""
        ...

    def _start_capture_thread(self, feed_id: str) -> None:
        """Start background thread for continuous capture (raw feeds only)"""
        def capture_loop():
            feed = self._feeds[feed_id]
            buffer = self._buffers[feed_id]
            while feed_id in self._feeds:
                frame = feed.read()
                if frame:
                    buffer.push(frame)
                    # Notify subscribers
                    for callback in self._subscribers.get(feed_id, []):
                        callback(frame)

        thread = Thread(target=capture_loop, daemon=True)
        thread.start()
        self._threads[feed_id] = thread

@dataclass
class DerivedFeed:
    """Metadata for a derived feed"""
    feed_id: str
    source_feed_id: str
    name: str
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
```

## Ring Buffer

```python
from collections import deque
from threading import Lock

class RingBuffer:
    """Thread-safe ring buffer for frames"""

    def __init__(self, max_size: int = 30):
        self._buffer = deque(maxlen=max_size)
        self._lock = Lock()

    def push(self, frame: Frame) -> None:
        with self._lock:
            self._buffer.append(frame)

    def get_latest(self) -> Optional[Frame]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_recent(self, count: int) -> list[Frame]:
        with self._lock:
            return list(self._buffer)[-count:]

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
```

## Camera Feed Implementation (MVP)

```python
import cv2

class CameraFeed(BaseFeed):
    """USB/CSI camera feed using OpenCV"""

    def __init__(self, feed_id: str, config: FeedConfig):
        super().__init__(feed_id, config)
        self._capture: Optional[cv2.VideoCapture] = None
        self._sequence = 0

    def connect(self) -> bool:
        try:
            # config.source is camera index (e.g., "0", "1")
            index = int(self.config.source)
            self._capture = cv2.VideoCapture(index)

            if not self._capture.isOpened():
                self._status = FeedStatus.ERROR
                return False

            self._status = FeedStatus.CONNECTED
            return True
        except Exception as e:
            self._status = FeedStatus.ERROR
            return False

    def disconnect(self) -> None:
        if self._capture:
            self._capture.release()
            self._capture = None
        self._status = FeedStatus.DISCONNECTED

    def read(self) -> Optional[Frame]:
        if not self._capture or self._status != FeedStatus.CONNECTED:
            return None

        ret, frame_data = self._capture.read()
        if not ret:
            return None

        self._sequence += 1
        return Frame(
            data=frame_data,
            timestamp=time.time(),
            sequence=self._sequence,
            feed_id=self.feed_id,
            width=frame_data.shape[1],
            height=frame_data.shape[0]
        )

    @property
    def fps(self) -> float:
        if self._capture:
            return self._capture.get(cv2.CAP_PROP_FPS)
        return 0.0

    @property
    def resolution(self) -> tuple[int, int]:
        if self._capture:
            w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (0, 0)
```

## Feed Factory

```python
class FeedFactory:
    """Factory for creating feed instances by type"""

    @staticmethod
    def create(feed_id: str, config: FeedConfig) -> BaseFeed:
        if config.feed_type == FeedType.CAMERA:
            return CameraFeed(feed_id, config)
        # Ideal: Add other feed types
        # elif config.feed_type == FeedType.RTSP:
        #     return RTSPFeed(feed_id, config)
        # elif config.feed_type == FeedType.VIDEO_FILE:
        #     return VideoFileFeed(feed_id, config)
        # elif config.feed_type == FeedType.IMAGE_FOLDER:
        #     return ImageFolderFeed(feed_id, config)
        else:
            raise ValueError(f"Unknown feed type: {config.feed_type}")
```

## MVP API Endpoints

```
# Feeds
GET    /api/feeds                    List all feeds
POST   /api/feeds                    Add a new feed
GET    /api/feeds/:id                Get feed status/info
DELETE /api/feeds/:id                Remove feed
POST   /api/feeds/:id/pause          Pause feed
POST   /api/feeds/:id/resume         Resume feed
GET    /api/feeds/:id/snapshot       Get current frame as JPEG
```

## MVP vs Ideal Features

| Feature | MVP | Ideal |
|---------|-----|-------|
| CameraFeed (USB/CSI) | Yes | Yes |
| RTSPFeed | No | Yes |
| VideoFileFeed | No | Yes |
| ImageFolderFeed | No | Yes |
| Ring buffer | Yes (fixed size) | Yes (configurable) |
| Frame timestamping | Yes | Yes |
| Frame rate normalization | No | Yes |
| Multi-feed parallel | No (single feed) | Yes |
| Auto-reconnect | No | Yes |
| Health monitoring | Basic (connected/error) | Full (latency, dropped frames) |
| Feed subscriptions | Yes | Yes |

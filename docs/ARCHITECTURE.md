# YOLO Dataset Creator - Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Web Browser (UI)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ HTTP/WebSocket
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API Gateway (FastAPI)                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  /api/feeds/*  /api/capture/*  /api/datasets/*  /api/training/*       │  │
│  │  /api/models/*  /api/inference/*  /api/system/*                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │  REST API   │ │ WebSocket   │ │   Static    │ │  CaptureController  │   │
│  │  Endpoints  │ │   Handler   │ │   Files     │ │  (subscribes feeds) │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────────────────────────────────────────┐   ┌───────────────────────────┐
│               Feeds Subsystem                     │   │    Training Subsystem     │
│ ┌─────────────────────────────────────────────┐   │   │  ┌───────────────────┐    │
│ │               Feed Manager                  │   │   │  │  Training Runner  │    │
│ │  ┌─────────────┐    ┌─────────────────────┐ │   │   │  │  Model Registry   │    │
│ │  │ Raw Feeds   │    │  Inference Feeds    │ │   │   │  │  Resource Monitor │    │
│ │  │ (Camera,    │───▶│  (frames + detect.) │ │   │   │  └───────────────────┘    │
│ │  │  RTSP, etc.)│    │                     │ │   │   └───────────────────────────┘
│ │  └─────────────┘    └─────────────────────┘ │   │               │
│ │         │                    │              │   │               │
│ │         ▼                    ▼              │   │               │ (reads data)
│ │  ┌─────────────────────────────────────┐    │   │               │
│ │  │   Subscribers (WebSocket, Capture)  │    │   │               │
│ │  └─────────────────────────────────────┘    │   │               │
│ └─────────────────────────────────────────────┘   │               │
└─────────────────────────────────┬─────────────────┘               │
                                  │                                 │
                 ┌────────────────┼────────────────┐                │
                 │                │                │                │
                 ▼                ▼                ▼                ▼
        ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐
        │  Inference   │  │   Dataset    │  │    Dataset Subsystem      │
        │  Subsystem   │  │  (captures)  │  │  ┌───────────────────┐    │
        │ ┌──────────┐ │  │              │  │  │   Storage Layer   │    │
        │ │YOLO-World│ │  │              │  │  │   Annotation Mgr  │    │
        │ │Fine-tuned│ │  └──────────────┘  │  │   Review Queue    │    │
        │ └──────────┘ │                    │  └───────────────────┘    │
        └──────┬───────┘                    └───────────────────────────┘
               │
               │ (inference output feed)
               │
               └───────────────────────────▶ Feeds Subsystem (registers as derived feed)

┌─────────────────────────────────────────────────────────────────────────────┐
│                           Persistence Layer                                 │
│    ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐    │
│    │  Datasets  │    │   Models   │    │   Config   │    │    Logs    │    │
│    │ (filesystem)│    │(filesystem)│    │(flat files)│    │(flat files)│    │
│    └────────────┘    └────────────┘    └────────────┘    └────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Data Flow

```
1. Raw Feed (Camera) ──▶ Feeds Subsystem ──▶ Inference Subsystem
                                                      │
2. Inference Output ───────────────────────────────────┘
   (frame + detections)
              │
              ▼
3. Inference Output Feed ──▶ Feeds Subsystem (registered as derived feed)
              │
              ├──▶ WebSocket Handler ──▶ Browser (live preview)
              │
              └──▶ CaptureController ──▶ Dataset Subsystem (captured frames)
```

### Subsystem Interactions

| From | To | Purpose |
|------|-----|---------|
| Feeds (raw) | Inference | Delivers raw frames for detection |
| Inference | Feeds (derived) | Registers inference output as subscribable feed |
| Feeds (derived) | WebSocket | Streams annotated frames to browser |
| Feeds (derived) | CaptureController | Delivers detection results for capture decisions |
| CaptureController | Dataset | Saves captured frames and auto-annotations |
| Training | Dataset | Reads images and labels for training |
| Training | Inference | Registers trained models |
| All | Persistence | Read/write data to filesystem |

---

## Interfaces

### External Interfaces

| Interface | Protocol | Purpose | MVP | Ideal |
|-----------|----------|---------|-----|-------|
| Web UI | HTTP/HTTPS | Browser-based user interface | Yes | Yes |
| WebSocket | WS | Real-time updates (video stream, training progress, stats) | Yes | Yes |
| Camera | V4L2/CSI | USB/CSI camera input | Yes | Yes |
| RTSP | RTSP/RTP | Network camera streams | No | Yes |
| Video Files | Filesystem | Pre-recorded video input | No | Yes |
| Filesystem | POSIX | Dataset, model, and config storage | Yes | Yes |

### Internal Interfaces

| Interface | Type | Purpose |
|-----------|------|---------|
| Subsystem Events | Pub/Sub (in-process) | Cross-subsystem communication |
| Feed Pipeline | Queue | Feed sources → Frame buffer → Consumers |
| Frame Subscription | Callback | Feeds → Scan/Model Mode (frame delivery) |
| Training Jobs | Queue | Async training job management |

---

## Subsystems

### 1. API Gateway (includes CaptureController)
### 2. Feeds Subsystem (raw + derived feeds)
### 3. Dataset Subsystem
### 4. Training Subsystem
### 5. Inference Subsystem (produces output feeds)
### 6. Notifications Subsystem
### 7. Authentication Subsystem [Ideal Only]
### 8. Persistence Layer
### 9. Frontend Application

**Note:** The Scan Subsystem has been eliminated. Its responsibilities are now split:
- **Detection**: Handled by Inference Subsystem (produces inference output feeds)
- **Capture logic**: Handled by CaptureController in the API Gateway layer

---

## Subsystem Details

---

## 1. API Gateway (includes CaptureController)

### Responsibility
Single entry point for all client-server communication. Routes requests to appropriate subsystems, manages WebSocket connections for real-time updates, serves static files, and hosts the CaptureController for capture logic orchestration.

### Components

```
API Gateway
├── REST Router
│   ├── /api/feeds/*       → Feeds Subsystem
│   ├── /api/capture/*     → CaptureController
│   ├── /api/inference/*   → Inference Subsystem
│   ├── /api/datasets/*    → Dataset Subsystem
│   ├── /api/training/*    → Training Subsystem
│   ├── /api/models/*      → Model management (Inference Subsystem)
│   └── /api/system/*      → System status, config
├── CaptureController
│   ├── Subscribes to inference output feeds
│   ├── Capture decision logic (interval, triggers)
│   ├── Negative frame sampling
│   └── Sends captures to Dataset Subsystem
├── WebSocket Handler
│   ├── /ws/video          → Live video stream (from inference feeds)
│   ├── /ws/events         → System events (captures, progress, alerts)
│   └── /ws/stats          → Real-time statistics
├── Static File Server
│   └── /static/*          → Frontend assets
└── Stream Proxy
    └── /stream/video      → MJPEG fallback for video
```

### MVP API Endpoints

```
# Capture (CaptureController)
POST   /api/capture/start           Start capture session (subscribes to inference feed)
POST   /api/capture/stop            Stop capture session
GET    /api/capture/status          Get capture state and stats
POST   /api/capture/trigger         Manual capture trigger
PUT    /api/capture/config          Update capture settings (interval, threshold, etc.)

# Inference
POST   /api/inference/start         Start inference on a feed (creates inference output feed)
POST   /api/inference/stop          Stop inference
GET    /api/inference/status        Get inference state
PUT    /api/inference/prompts       Update YOLO-World prompts
PUT    /api/inference/model         Switch active model

# Dataset (extends existing)
GET    /api/datasets                List datasets
POST   /api/datasets                Create dataset
POST   /api/datasets/import         Import dataset zip
GET    /api/datasets/:name          Get dataset info
PUT    /api/datasets/:name          Update dataset
DELETE /api/datasets/:name          Delete dataset
GET    /api/datasets/:name/export   Export dataset zip

# Images & Annotations (extends existing)
GET    /api/datasets/:name/images              List images with filter/pagination
POST   /api/datasets/:name/images              Upload image
DELETE /api/datasets/:name/images/:split/:file Delete image
GET    /api/datasets/:name/labels/:split/:file Get annotations
PUT    /api/datasets/:name/labels/:split/:file Save annotations
PUT    /api/datasets/:name/split/:split/:file  Change split

# Review Queue (new)
GET    /api/datasets/:name/review              Get images pending review
POST   /api/datasets/:name/review/bulk         Bulk accept/reject

# Prompts (new)
GET    /api/datasets/:name/prompts             Get class prompts
PUT    /api/datasets/:name/prompts             Save class prompts

# Training
POST   /api/training/start          Start training job
POST   /api/training/stop           Stop/cancel training
GET    /api/training/status         Get training progress
GET    /api/training/history        Get past training runs

# Models
GET    /api/models                  List trained models
GET    /api/models/:id              Get model details
DELETE /api/models/:id              Delete model
PUT    /api/models/:id/activate     Set as active model
POST   /api/models/:id/export       Export model

# System
GET    /api/system/status           Overall system status
GET    /api/system/resources        CPU, GPU, memory usage
GET    /api/system/config           Get app configuration
PUT    /api/system/config           Update app configuration
```

### WebSocket Events

```
# Server → Client
video.frame          Base64 encoded frame with detections overlay (from inference feed)
inference.stats      Detection statistics (counts, fps, classes)
capture.event        New image captured (includes metadata)
capture.stats        Capture statistics (positive/negative counts, rate)
training.progress    Training progress update
training.complete    Training finished
training.error       Training error
system.warning       Resource constraint warning
system.error         System error

# Client → Server
capture.trigger      Request manual capture
capture.config       Update capture config without restart
inference.prompts    Update YOLO-World prompts in real-time
```

### Technology
- FastAPI with native WebSocket support
- Uvicorn ASGI server (async-native)
- Pydantic for request/response validation

### FastAPI Application Structure

```python
# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api import feeds, capture, datasets, training, models, inference, system
from websocket import video_ws, events_ws
from core.events import event_bus
from core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await event_bus.start()
    yield
    # Shutdown
    await event_bus.stop()

app = FastAPI(
    title="YOLO Dataset Creator",
    version="2.0.0",
    lifespan=lifespan
)

# CORS for development (Vue dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST API routers
app.include_router(feeds.router, prefix="/api/feeds", tags=["feeds"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(capture.router, prefix="/api/capture", tags=["capture"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(system.router, prefix="/api/system", tags=["system"])

# WebSocket endpoints
app.include_router(video_ws.router, tags=["websocket"])
app.include_router(events_ws.router, tags=["websocket"])
```

### Example Router (CaptureController)

```python
# backend/api/capture.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.capture.controller import CaptureController
from models.capture import CaptureConfig, CaptureStatus, CaptureStats

router = APIRouter()

class StartCaptureRequest(BaseModel):
    inference_feed_id: str      # The inference output feed to subscribe to
    dataset_name: str
    capture_interval: float = 2.0
    negative_ratio: float = 0.2
    confidence_threshold: float = 0.3

@router.post("/start")
async def start_capture(
    request: StartCaptureRequest,
    controller: CaptureController = Depends(get_capture_controller)
) -> CaptureStatus:
    """Start capture session - subscribes to an inference output feed"""
    config = CaptureConfig(**request.model_dump())
    success = await controller.start(config)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to start capture")
    return controller.get_status()

@router.post("/stop")
async def stop_capture(
    controller: CaptureController = Depends(get_capture_controller)
) -> CaptureStatus:
    """Stop the current capture session"""
    await controller.stop()
    return controller.get_status()

@router.get("/status")
async def get_capture_status(
    controller: CaptureController = Depends(get_capture_controller)
) -> CaptureStatus:
    """Get current capture status and statistics"""
    return controller.get_status()

@router.post("/trigger")
async def manual_capture(
    controller: CaptureController = Depends(get_capture_controller)
) -> dict:
    """Trigger a manual frame capture"""
    result = await controller.manual_capture()
    return {"success": result is not None, "image": result}
```

### CaptureController Implementation

```python
# backend/api/capture/controller.py
from feeds.manager import FeedManager
from dataset.storage import DatasetManager
from models.capture import CaptureConfig, CaptureStatus

class CaptureController:
    """
    Lightweight controller that subscribes to inference output feeds
    and decides when to capture frames to the dataset.

    This is NOT a subsystem - it orchestrates between Feeds, Inference,
    and Dataset subsystems without owning any domain state.
    """

    def __init__(
        self,
        feed_manager: FeedManager,
        dataset_manager: DatasetManager
    ):
        self._feed_manager = feed_manager
        self._dataset_manager = dataset_manager
        self._config: CaptureConfig | None = None
        self._active_feed_id: str | None = None
        self._stats = CaptureStats()
        self._last_capture_time = 0.0
        self._positive_count = 0
        self._negative_count = 0

    async def start(self, config: CaptureConfig) -> bool:
        """Subscribe to inference output feed and start capture logic"""
        self._config = config

        # Subscribe to the inference output feed
        self._feed_manager.subscribe(
            config.inference_feed_id,
            self._on_inference_frame
        )
        self._active_feed_id = config.inference_feed_id
        return True

    async def stop(self) -> None:
        """Unsubscribe from feed"""
        if self._active_feed_id:
            self._feed_manager.unsubscribe(
                self._active_feed_id,
                self._on_inference_frame
            )
        self._active_feed_id = None

    def _on_inference_frame(self, frame: InferenceFrame) -> None:
        """
        Callback when new inference result arrives.
        InferenceFrame contains both the image and detections.
        """
        should_capture, capture_type = self._should_capture(frame)

        if should_capture:
            self._save_capture(frame, capture_type)
            self._last_capture_time = time.time()

    def _should_capture(self, frame: InferenceFrame) -> tuple[bool, str | None]:
        """Decide whether to capture this frame"""
        now = time.time()
        elapsed = now - self._last_capture_time

        if elapsed < self._config.capture_interval:
            return False, None

        has_detections = len(frame.detections) > 0

        if has_detections:
            return True, "positive"
        elif self._should_capture_negative():
            return True, "negative"

        return False, None

    def _should_capture_negative(self) -> bool:
        """Check if we should capture a negative frame based on ratio"""
        if self._positive_count == 0:
            return False
        current_ratio = self._negative_count / self._positive_count
        return current_ratio < self._config.negative_ratio

    def _save_capture(self, frame: InferenceFrame, capture_type: str) -> None:
        """Save frame and annotations to dataset"""
        # Convert detections to YOLO annotations
        annotations = [
            Annotation(
                class_id=d.class_id,
                x=d.x_center,
                y=d.y_center,
                width=d.width,
                height=d.height,
                confidence=d.confidence,
                auto=True
            )
            for d in frame.detections
        ]

        # Save to dataset
        self._dataset_manager.add_image(
            dataset=self._config.dataset_name,
            image=frame.image,
            annotations=annotations,
            split="train"
        )

        # Update stats
        if capture_type == "positive":
            self._positive_count += 1
        else:
            self._negative_count += 1

        self._stats.total_captures += 1
```

### WebSocket Handler

```python
# backend/websocket/video.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from websocket.manager import ConnectionManager

router = APIRouter()
manager = ConnectionManager()

@router.websocket("/ws/video")
async def video_websocket(websocket: WebSocket):
    """WebSocket endpoint for video frame streaming"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive client messages (e.g., subscribe to specific feed)
            data = await websocket.receive_json()

            if data.get("action") == "subscribe":
                feed_id = data.get("feed_id")
                await manager.subscribe_to_feed(websocket, feed_id)

            elif data.get("action") == "unsubscribe":
                await manager.unsubscribe_from_feed(websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)

# backend/websocket/manager.py
from fastapi import WebSocket
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.feed_subscriptions: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        # Remove from all feed subscriptions
        for feed_id in self.feed_subscriptions:
            if websocket in self.feed_subscriptions[feed_id]:
                self.feed_subscriptions[feed_id].remove(websocket)

    async def broadcast_frame(self, feed_id: str, frame_data: str, detections: list):
        """Broadcast video frame to all subscribers of a feed"""
        if feed_id not in self.feed_subscriptions:
            return

        message = {
            "type": "frame",
            "feed_id": feed_id,
            "data": frame_data,  # Base64 encoded JPEG
            "detections": detections,
            "timestamp": time.time()
        }

        for websocket in self.feed_subscriptions[feed_id]:
            await websocket.send_json(message)

    async def broadcast_event(self, event_type: str, data: dict):
        """Broadcast event to all connected clients"""
        message = {"type": event_type, **data}
        for websocket in self.active_connections:
            await websocket.send_json(message)
```

---

## 2. Feeds Subsystem (Raw + Derived Feeds)

### Responsibility
Manage all video feeds - both raw input sources and derived feeds (like inference output). Abstracts different feed types behind a unified interface, handles buffering, and provides frames to subscribers.

### Key Concept: Derived Feeds

The Feeds Subsystem handles two categories of feeds:

1. **Raw Feeds**: Physical video sources (cameras, RTSP streams, video files)
2. **Derived Feeds**: Processed feeds produced by other subsystems (e.g., inference output with detections)

Both feed types share the same subscription interface, allowing any consumer (WebSocket, CaptureController, logging) to subscribe uniformly.

### Components

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

### Feed Interface

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

### Feed Manager

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

### Ring Buffer

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

### Camera Feed Implementation (MVP)

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

### Feed Factory

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

### MVP API Endpoints for Feeds

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

### MVP vs Ideal Features

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

---

## 3. Dataset Subsystem

### Responsibility
Manage dataset business logic, annotations, and the review workflow. **Delegates all file operations to the Persistence Layer.**

### Components

```
Dataset Subsystem
├── Dataset Manager
│   ├── Dataset CRUD (uses Persistence Layer)
│   ├── Class/Label configuration
│   └── Statistics aggregation
├── Annotation Manager
│   ├── Annotation CRUD (uses Persistence Layer)
│   ├── Auto-annotation marker ("auto" flag)
│   ├── Format conversion (normalized ↔ pixel)
│   └── Validation (bounds checking)
├── Review Queue
│   ├── Queue Manager (pending review items)
│   ├── Bulk Operations (accept/reject)
│   └── Filter/Sort (by confidence, class, time)
└── Export/Import
    ├── Zip Export (uses Persistence Layer)
    ├── Zip Import (uses Persistence Layer)
    └── Validation (schema checking)
```

### Dependency on Persistence Layer

The Dataset Subsystem does NOT directly access the filesystem. All storage operations go through the Persistence Layer:

```python
# dataset/manager.py
from persistence import DatasetStore, ImageStore, LabelStore

class DatasetManager:
    """
    Manages dataset business logic.
    Delegates all file operations to Persistence Layer.
    """

    def __init__(
        self,
        dataset_store: DatasetStore,
        image_store: ImageStore,
        label_store: LabelStore
    ):
        self._datasets = dataset_store
        self._images = image_store
        self._labels = label_store

    async def create_dataset(self, name: str, classes: list[str]) -> Dataset:
        """Create a new dataset with the given classes"""
        # Validate name
        if not self._is_valid_name(name):
            raise ValueError(f"Invalid dataset name: {name}")

        # Delegate storage to Persistence Layer
        dataset = await self._datasets.create(name, classes)
        return dataset

    async def add_image(
        self,
        dataset_name: str,
        image: np.ndarray,
        annotations: list[Annotation],
        split: str = "train"
    ) -> ImageInfo:
        """Add an image with annotations to a dataset"""
        # Generate unique filename
        filename = f"{uuid.uuid4().hex}.jpg"

        # Delegate image storage to Persistence Layer
        image_info = await self._images.save(
            dataset_name, split, filename, image
        )

        # Delegate label storage to Persistence Layer
        if annotations:
            await self._labels.save(
                dataset_name, split, filename, annotations
            )

        return image_info

    async def get_annotations(
        self,
        dataset_name: str,
        split: str,
        filename: str
    ) -> list[Annotation]:
        """Get annotations for an image"""
        # Delegate to Persistence Layer
        raw_labels = await self._labels.load(dataset_name, split, filename)

        # Apply business logic (convert format, validate)
        return self._parse_annotations(raw_labels)

    async def update_annotations(
        self,
        dataset_name: str,
        split: str,
        filename: str,
        annotations: list[Annotation]
    ) -> None:
        """Update annotations for an image"""
        # Validate annotations
        self._validate_annotations(annotations)

        # Delegate to Persistence Layer
        await self._labels.save(dataset_name, split, filename, annotations)

    async def delete_image(
        self,
        dataset_name: str,
        split: str,
        filename: str
    ) -> None:
        """Delete an image and its annotations"""
        # Delegate to Persistence Layer
        await self._images.delete(dataset_name, split, filename)
        await self._labels.delete(dataset_name, split, filename)

    async def change_split(
        self,
        dataset_name: str,
        filename: str,
        from_split: str,
        to_split: str
    ) -> None:
        """Move an image between splits"""
        # Delegate to Persistence Layer
        await self._images.move(dataset_name, filename, from_split, to_split)
        await self._labels.move(dataset_name, filename, from_split, to_split)
```

### Dataset Structure (Extended)

```
datasets/
└── my_dataset/
    ├── data.yaml              # YOLO config (existing)
    ├── prompts.yaml           # Class prompts for YOLO-World (new)
    ├── metadata.json          # Dataset metadata (new)
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── .review/               # Review queue metadata (new)
        └── queue.json
```

### New File Formats

**prompts.yaml**
```yaml
# Maps class IDs to text prompts for YOLO-World
classes:
  0:
    name: "vehicle"
    prompts:
      - "vehicle"
      - "car"
      - "truck"
      - "military vehicle"
  1:
    name: "person"
    prompts:
      - "person"
      - "human"
      - "soldier"
```

**metadata.json**
```json
{
  "created": "2024-01-15T10:30:00Z",
  "modified": "2024-01-15T14:22:00Z",
  "scan_sessions": [
    {
      "started": "2024-01-15T10:30:00Z",
      "ended": "2024-01-15T11:45:00Z",
      "frames_captured": 150,
      "config": { ... }
    }
  ],
  "stats": {
    "total_images": 200,
    "annotated": 180,
    "pending_review": 20,
    "by_class": { "0": 450, "1": 230 }
  }
}
```

**queue.json**
```json
{
  "pending": [
    {
      "image": "images/train/20240115_103045_abc123.jpg",
      "auto_annotations": [
        {"classId": 0, "x": 0.5, "y": 0.5, "width": 0.2, "height": 0.3, "confidence": 0.45, "auto": true}
      ],
      "captured_at": "2024-01-15T10:30:45Z",
      "low_confidence": true
    }
  ]
}
```

### Annotation Model (Extended)

```python
@dataclass
class Annotation:
    class_id: int
    x: float          # center x (normalized)
    y: float          # center y (normalized)
    width: float      # width (normalized)
    height: float     # height (normalized)
    confidence: float = None  # detection confidence (if auto)
    auto: bool = False        # auto-generated flag
```

---

## 4. Training Subsystem

### Responsibility
Manage model training jobs, monitor resources, and handle background execution. **Delegates model storage to the Persistence Layer.**

### Components

```
Training Subsystem
├── Training Runner
│   ├── Job Queue (single job for MVP)
│   ├── YOLO Training Wrapper (ultralytics)
│   ├── Progress Monitor
│   └── Output Parser (loss, metrics)
├── Model Manager
│   ├── Training orchestration
│   ├── Model registration (uses ModelStore)
│   └── Active model tracking
└── Resource Monitor
    ├── GPU Memory Monitor
    ├── CPU/RAM Monitor
    └── Threshold Alerter
```

### Dependency on Persistence Layer

```python
# training/manager.py
from persistence import ModelStore, DatasetStore

class TrainingManager:
    """
    Manages training jobs and model lifecycle.
    Delegates storage to Persistence Layer.
    """

    def __init__(
        self,
        model_store: ModelStore,
        dataset_store: DatasetStore
    ):
        self._models = model_store
        self._datasets = dataset_store
        self._current_job: TrainingJob | None = None

    async def start_training(self, config: TrainingConfig) -> str:
        """Start a training job"""
        # Validate dataset exists via Persistence Layer
        dataset = await self._datasets.get(config.dataset_name)
        if not dataset:
            raise ValueError(f"Dataset not found: {config.dataset_name}")

        # Create training job
        job_id = str(uuid.uuid4())
        self._current_job = TrainingJob(job_id, config)

        # Start training in background
        asyncio.create_task(self._run_training(self._current_job))

        return job_id

    async def _run_training(self, job: TrainingJob) -> None:
        """Execute training and save model via Persistence Layer"""
        try:
            # Run YOLO training
            model = YOLO(job.config.base_model)
            results = model.train(
                data=str(await self._datasets.get_data_yaml_path(job.config.dataset_name)),
                epochs=job.config.epochs,
                imgsz=job.config.image_size,
                # ... other params
            )

            # Save model via Persistence Layer
            await self._models.save(
                name=job.config.model_name,
                weights_path=results.save_dir / "weights" / "best.pt",
                config=job.config,
                metrics=self._extract_metrics(results)
            )

            job.status = "completed"

        except Exception as e:
            job.status = "error"
            job.error = str(e)

    async def list_models(self) -> list[ModelInfo]:
        """List all trained models via Persistence Layer"""
        return await self._models.list()

    async def delete_model(self, name: str) -> bool:
        """Delete a model via Persistence Layer"""
        return await self._models.delete(name)

    async def set_active_model(self, name: str) -> None:
        """Set the active model via Persistence Layer"""
        await self._models.set_active(name)
```

### Training Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Config    │───▶│  Validate   │───▶│   Train     │───▶│   Save      │
│   Input     │    │  Dataset    │    │  (async)    │    │   Model     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            ▼
                                     ┌─────────────┐
                                     │  Progress   │
                                     │  Events     │
                                     └─────────────┘
```

### State Machine

```
┌─────────┐   start()   ┌───────────┐  complete()  ┌───────────┐
│  IDLE   │────────────▶│ TRAINING  │─────────────▶│ COMPLETED │
└─────────┘             └───────────┘              └───────────┘
     ▲                       │                          │
     │        cancel()       │                          │
     │◀──────────────────────┘                          │
     │                                                  │
     └──────────────────────────────────────────────────┘
                        (reset)
```

### Training Configuration

```python
@dataclass
class TrainingConfig:
    # Dataset
    dataset_name: str

    # Model
    base_model: str = "yolo11n.pt"  # or yolo11s, yolo11m

    # Training params
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    patience: int = 20
    freeze_layers: int = 10  # freeze backbone for small datasets

    # Learning rate
    lr0: float = 0.001
    lrf: float = 0.01

    # Output
    model_name: str = ""  # user-defined name

    # Resource limits
    max_gpu_memory_percent: float = 0.8
```

### Model Registry Structure

```
models/
└── my_dataset/
    ├── registry.json           # Model index
    └── models/
        ├── detector_v1/
        │   ├── weights/
        │   │   └── best.pt
        │   ├── config.yaml     # Training config used
        │   └── metrics.json    # Final metrics
        └── detector_v2/
            └── ...
```

**registry.json**
```json
{
  "active": "detector_v1",
  "models": {
    "detector_v1": {
      "created": "2024-01-15T14:00:00Z",
      "base_model": "yolo11n.pt",
      "epochs_completed": 87,
      "best_map50": 0.82,
      "dataset_snapshot": "200 images, 3 classes"
    }
  }
}
```

### Resource Monitor

```python
class ResourceMonitor:
    def __init__(self, gpu_threshold=0.9, ram_threshold=0.85):
        self.gpu_threshold = gpu_threshold
        self.ram_threshold = ram_threshold
        self.callbacks = []

    def check(self) -> ResourceStatus:
        gpu_usage = get_gpu_memory_usage()
        ram_usage = get_ram_usage()

        status = ResourceStatus(
            gpu_percent=gpu_usage,
            ram_percent=ram_usage,
            constrained=(gpu_usage > self.gpu_threshold or
                        ram_usage > self.ram_threshold)
        )

        if status.constrained:
            self.notify_callbacks(status)

        return status
```

---

## 5. Inference Subsystem (Produces Output Feeds)

### Responsibility
Load and run trained models or YOLO-World for detection. Subscribes to raw feeds from the Feeds Subsystem, runs detection, and **produces inference output feeds** that are registered back with Feeds. These output feeds can be consumed by WebSocket (live preview), CaptureController (frame capture), logging, etc.

### Components

```
Inference Subsystem
├── Inference Manager
│   ├── Feed Subscription (subscribes to raw feeds)
│   ├── Output Feed Producer (registers with Feeds Subsystem)
│   └── Session Lifecycle (start/stop inference sessions)
├── Model Loader
│   ├── YOLO-World Loader
│   ├── Fine-tuned Model Loader
│   └── Model Cache (keep loaded models in memory)
├── Detection Runner
│   ├── Frame Preprocessor
│   ├── Inference Executor
│   └── Result Postprocessor (NMS, filtering)
└── Model Switcher
    ├── Hot-swap Handler
    └── Memory Manager (unload old model)
```

### Key Concept: Inference Output Feeds

When inference starts on a raw feed, the Inference Subsystem:
1. Subscribes to the raw feed (e.g., camera)
2. Runs detection on each frame
3. Creates an **InferenceFrame** containing both the image and detections
4. Registers this as a derived feed with the Feeds Subsystem
5. Any subscriber (WebSocket, CaptureController) can subscribe to this derived feed

### Data Flow

```
┌─────────────────┐    subscribe    ┌─────────────────┐
│   Raw Feed      │───────────────▶│   Inference     │
│   (camera)      │                │   Subsystem     │
└─────────────────┘                └─────────────────┘
                                          │
                                          │ run detection
                                          ▼
                                   ┌─────────────────┐
                                   │ InferenceFrame  │
                                   │ (image + dets)  │
                                   └─────────────────┘
                                          │
                                          │ register as derived feed
                                          ▼
                                   ┌─────────────────┐
                                   │ Feeds Subsystem │
                                   │ (derived feed)  │
                                   └─────────────────┘
                                          │
                         ┌────────────────┼────────────────┐
                         ▼                ▼                ▼
                   ┌──────────┐    ┌───────────┐    ┌──────────┐
                   │WebSocket │    │ Capture   │    │ Logging  │
                   │(preview) │    │Controller │    │ [Ideal]  │
                   └──────────┘    └───────────┘    └──────────┘
```

### Inference Manager

```python
class InferenceManager:
    """
    Manages inference sessions. Subscribes to raw feeds,
    runs detection, and produces output feeds.
    """

    def __init__(self, feed_manager: FeedManager):
        self._feed_manager = feed_manager
        self._sessions: dict[str, InferenceSession] = {}

    async def start_inference(
        self,
        source_feed_id: str,
        model_id: str,
        prompts: dict[int, list[str]] | None = None
    ) -> str:
        """
        Start inference on a feed. Returns the output feed ID.
        """
        # Generate output feed ID
        output_feed_id = f"inference_{source_feed_id}_{uuid.uuid4().hex[:8]}"

        # Load model
        model = self._load_model(model_id, prompts)

        # Register derived feed with Feeds Subsystem
        self._feed_manager.register_derived_feed(
            feed_id=output_feed_id,
            source_feed_id=source_feed_id,
            name=f"Inference on {source_feed_id}"
        )

        # Create session and subscribe to raw feed
        session = InferenceSession(
            output_feed_id=output_feed_id,
            source_feed_id=source_feed_id,
            model=model,
            feed_manager=self._feed_manager
        )

        self._feed_manager.subscribe(
            source_feed_id,
            session.on_frame
        )

        self._sessions[output_feed_id] = session
        return output_feed_id

    async def stop_inference(self, output_feed_id: str) -> None:
        """Stop an inference session"""
        session = self._sessions.get(output_feed_id)
        if session:
            self._feed_manager.unsubscribe(
                session.source_feed_id,
                session.on_frame
            )
            self._feed_manager.unregister_derived_feed(output_feed_id)
            del self._sessions[output_feed_id]


class InferenceSession:
    """A single inference session processing frames from a source feed"""

    def __init__(
        self,
        output_feed_id: str,
        source_feed_id: str,
        model: LoadedModel,
        feed_manager: FeedManager
    ):
        self.output_feed_id = output_feed_id
        self.source_feed_id = source_feed_id
        self._model = model
        self._feed_manager = feed_manager
        self._sequence = 0

    def on_frame(self, frame: Frame) -> None:
        """Process a raw frame and push inference result to output feed"""
        # Run detection
        start_time = time.time()
        results = self._model.model(frame.data)
        inference_time = (time.time() - start_time) * 1000

        # Convert to Detection objects
        detections = self._parse_results(results)

        # Create inference frame
        self._sequence += 1
        inference_frame = InferenceFrame(
            data=frame.data,
            timestamp=frame.timestamp,
            sequence=self._sequence,
            feed_id=self.output_feed_id,
            source_feed_id=self.source_feed_id,
            width=frame.width,
            height=frame.height,
            detections=detections,
            inference_time_ms=inference_time
        )

        # Push to derived feed (notifies all subscribers)
        self._feed_manager.push_derived_frame(
            self.output_feed_id,
            inference_frame
        )
```

### Model Types

```python
class ModelType(Enum):
    YOLO_WORLD = "yolo_world"      # Zero-shot with text prompts
    FINE_TUNED = "fine_tuned"      # Custom trained model

@dataclass
class LoadedModel:
    model_type: ModelType
    model: YOLO
    classes: dict          # class_id → name
    prompts: dict = None   # For YOLO-World only
```

---

## 6. Notifications Subsystem

### Responsibility
Centralized notification management for all system events. Delivers alerts to frontend via WebSocket, manages notification state, and handles external notification channels in ideal state.

### Components

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

### Notification Model

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

### Notification Manager

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

### WebSocket Events

```
# Server → Client
notification              New notification (all types)
notification.dismiss      Notification auto-dismissed (timeout)

# Client → Server
notification.read         Mark notification as read
notification.dismiss      Dismiss notification
notification.clear        Clear notification history
```

### MVP API Endpoints

```
GET    /api/notifications              Get notification history
POST   /api/notifications/:id/read     Mark as read
POST   /api/notifications/:id/dismiss  Dismiss notification
DELETE /api/notifications              Clear all notifications
```

### Frontend Integration

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

### MVP vs Ideal Features

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

---

## 7. Authentication Subsystem [Ideal State Only]

### Responsibility
Manage user authentication, session handling, and role-based access control (RBAC). Not implemented in MVP - system operates without authentication.

### Components

```
Auth Subsystem [Ideal]
├── Auth Manager
│   ├── User Store (CRUD operations)
│   ├── Session Manager
│   ├── Token Generator (JWT)
│   └── Password Hasher (bcrypt)
├── RBAC Engine
│   ├── Role Definitions
│   ├── Permission Checks
│   └── Resource Guards
├── Middleware
│   ├── Authentication Middleware
│   ├── Authorization Middleware
│   └── Rate Limiter
└── Audit Logger
    ├── Action Logging
    └── Access Logging
```

### Data Models

```python
from enum import Enum
from dataclasses import dataclass
from typing import Set

class Role(Enum):
    ADMIN = "admin"       # Full access to everything
    OPERATOR = "operator" # Can scan, annotate, train, run inference
    VIEWER = "viewer"     # Read-only access

class Permission(Enum):
    # Scan
    SCAN_VIEW = "scan:view"
    SCAN_START = "scan:start"
    SCAN_STOP = "scan:stop"
    SCAN_CONFIGURE = "scan:configure"

    # Dataset
    DATASET_VIEW = "dataset:view"
    DATASET_CREATE = "dataset:create"
    DATASET_EDIT = "dataset:edit"
    DATASET_DELETE = "dataset:delete"
    DATASET_EXPORT = "dataset:export"
    DATASET_IMPORT = "dataset:import"

    # Training
    TRAINING_VIEW = "training:view"
    TRAINING_START = "training:start"
    TRAINING_STOP = "training:stop"

    # Models
    MODEL_VIEW = "model:view"
    MODEL_DELETE = "model:delete"
    MODEL_ACTIVATE = "model:activate"

    # Inference
    INFERENCE_VIEW = "inference:view"
    INFERENCE_START = "inference:start"
    INFERENCE_STOP = "inference:stop"

    # System
    SETTINGS_VIEW = "settings:view"
    SETTINGS_EDIT = "settings:edit"
    USERS_MANAGE = "users:manage"

# Role → Permissions mapping
ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions
    Role.OPERATOR: {
        # Scan
        Permission.SCAN_VIEW, Permission.SCAN_START,
        Permission.SCAN_STOP, Permission.SCAN_CONFIGURE,
        # Dataset
        Permission.DATASET_VIEW, Permission.DATASET_CREATE,
        Permission.DATASET_EDIT, Permission.DATASET_DELETE,
        Permission.DATASET_EXPORT, Permission.DATASET_IMPORT,
        # Training
        Permission.TRAINING_VIEW, Permission.TRAINING_START,
        Permission.TRAINING_STOP,
        # Models
        Permission.MODEL_VIEW, Permission.MODEL_DELETE,
        Permission.MODEL_ACTIVATE,
        # Inference
        Permission.INFERENCE_VIEW, Permission.INFERENCE_START,
        Permission.INFERENCE_STOP,
        # System
        Permission.SETTINGS_VIEW,
    },
    Role.VIEWER: {
        Permission.SCAN_VIEW,
        Permission.DATASET_VIEW,
        Permission.TRAINING_VIEW,
        Permission.MODEL_VIEW,
        Permission.INFERENCE_VIEW,
    }
}

@dataclass
class User:
    id: str
    username: str
    password_hash: str
    role: Role
    created_at: float
    last_login: float = None
    active: bool = True
```

### Authentication Flow

```
┌──────────┐     POST /api/auth/login      ┌──────────────┐
│  Client  │──────────────────────────────▶│ Auth Manager │
│          │  {username, password}         │              │
└──────────┘                               └──────────────┘
                                                  │
                                                  ▼ validate
                                           ┌──────────────┐
                                           │  User Store  │
                                           └──────────────┘
                                                  │
     ┌──────────────────────────────────────────────┘
     │ JWT token
     ▼
┌──────────┐     Authorization: Bearer <token>     ┌──────────────┐
│  Client  │──────────────────────────────────────▶│  Middleware  │
│          │                                       │              │
└──────────┘                                       └──────────────┘
                                                          │
                                                          ▼ check permissions
                                                   ┌──────────────┐
                                                   │ RBAC Engine  │
                                                   └──────────────┘
```

### Auth Middleware

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Dependency to extract and validate current user from JWT token"""
    token = credentials.credentials

    user = auth_manager.validate_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

    return user

def require_permission(permission: Permission):
    """Dependency factory to require a specific permission"""
    async def check_permission(user: User = Depends(get_current_user)) -> User:
        if not rbac.has_permission(user.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )
        return user
    return check_permission

# Usage example
@router.post("/api/training/start")
async def start_training(
    config: TrainingConfig,
    user: User = Depends(require_permission(Permission.TRAINING_START))
):
    # Only users with TRAINING_START permission can access
    ...
```

### API Endpoints

```
# Authentication
POST   /api/auth/login          Login, returns JWT token
POST   /api/auth/logout         Invalidate session
POST   /api/auth/refresh        Refresh token
GET    /api/auth/me             Get current user info

# User Management (Admin only)
GET    /api/users               List all users
POST   /api/users               Create user
GET    /api/users/:id           Get user details
PUT    /api/users/:id           Update user
DELETE /api/users/:id           Delete user
PUT    /api/users/:id/role      Change user role
```

### Storage

For ideal state with database:
```sql
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_login REAL,
    active INTEGER DEFAULT 1
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE audit_log (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    action TEXT NOT NULL,
    resource TEXT,
    details TEXT,
    timestamp REAL NOT NULL,
    ip_address TEXT
);
```

---

## 8. Persistence Layer

### Responsibility
Abstract all file system operations. Provides clean interfaces for data storage that other subsystems use. **No other subsystem should directly access the filesystem.**

### Components

```
Persistence Layer
├── Dataset Store
│   ├── Dataset CRUD (create, list, get, delete)
│   ├── Metadata management (data.yaml, prompts.yaml, metadata.json)
│   └── Export/Import (zip operations)
├── Image Store
│   ├── Image save/load/delete
│   ├── Image move (between splits)
│   ├── Thumbnail generation [Ideal]
│   └── Batch operations
├── Label Store
│   ├── Label save/load/delete
│   ├── Label move (between splits)
│   ├── YOLO format parsing/writing
│   └── Batch operations
├── Model Store
│   ├── Model file management
│   ├── Registry management (registry.json)
│   └── Model metadata (config, metrics)
├── Config Store
│   ├── App configuration
│   └── User preferences
└── Log Store
    ├── Training logs
    └── Capture session logs
```

### Store Interfaces

```python
# persistence/dataset_store.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetInfo:
    name: str
    path: Path
    classes: list[str]
    num_images: dict[str, int]  # split → count
    created_at: float
    modified_at: float

class DatasetStore(ABC):
    """Abstract interface for dataset storage operations"""

    @abstractmethod
    async def create(self, name: str, classes: list[str]) -> DatasetInfo:
        """Create a new dataset directory structure and data.yaml"""
        ...

    @abstractmethod
    async def list(self) -> list[DatasetInfo]:
        """List all datasets"""
        ...

    @abstractmethod
    async def get(self, name: str) -> DatasetInfo | None:
        """Get dataset info by name"""
        ...

    @abstractmethod
    async def delete(self, name: str) -> bool:
        """Delete a dataset and all its contents"""
        ...

    @abstractmethod
    async def update_classes(self, name: str, classes: list[str]) -> None:
        """Update dataset classes in data.yaml"""
        ...

    @abstractmethod
    async def get_prompts(self, name: str) -> dict[int, list[str]]:
        """Get YOLO-World prompts from prompts.yaml"""
        ...

    @abstractmethod
    async def save_prompts(self, name: str, prompts: dict[int, list[str]]) -> None:
        """Save YOLO-World prompts to prompts.yaml"""
        ...

    @abstractmethod
    async def export_zip(self, name: str, output_path: Path) -> Path:
        """Export dataset as zip file"""
        ...

    @abstractmethod
    async def import_zip(self, zip_path: Path, name: str | None = None) -> DatasetInfo:
        """Import dataset from zip file"""
        ...


# persistence/image_store.py
@dataclass
class ImageInfo:
    filename: str
    split: str
    path: Path
    width: int
    height: int
    size_bytes: int
    has_labels: bool

class ImageStore(ABC):
    """Abstract interface for image file operations"""

    @abstractmethod
    async def save(
        self,
        dataset: str,
        split: str,
        filename: str,
        image: np.ndarray
    ) -> ImageInfo:
        """Save image to dataset"""
        ...

    @abstractmethod
    async def load(self, dataset: str, split: str, filename: str) -> np.ndarray:
        """Load image from dataset"""
        ...

    @abstractmethod
    async def delete(self, dataset: str, split: str, filename: str) -> bool:
        """Delete image from dataset"""
        ...

    @abstractmethod
    async def move(
        self,
        dataset: str,
        filename: str,
        from_split: str,
        to_split: str
    ) -> None:
        """Move image between splits"""
        ...

    @abstractmethod
    async def list(
        self,
        dataset: str,
        split: str | None = None
    ) -> list[ImageInfo]:
        """List images in dataset, optionally filtered by split"""
        ...

    @abstractmethod
    async def exists(self, dataset: str, split: str, filename: str) -> bool:
        """Check if image exists"""
        ...


# persistence/label_store.py
@dataclass
class LabelData:
    filename: str
    split: str
    annotations: list[dict]  # Raw YOLO format annotations

class LabelStore(ABC):
    """Abstract interface for label file operations"""

    @abstractmethod
    async def save(
        self,
        dataset: str,
        split: str,
        filename: str,
        annotations: list[Annotation]
    ) -> None:
        """Save annotations in YOLO format"""
        ...

    @abstractmethod
    async def load(
        self,
        dataset: str,
        split: str,
        filename: str
    ) -> list[dict]:
        """Load annotations from YOLO format file"""
        ...

    @abstractmethod
    async def delete(self, dataset: str, split: str, filename: str) -> bool:
        """Delete label file"""
        ...

    @abstractmethod
    async def move(
        self,
        dataset: str,
        filename: str,
        from_split: str,
        to_split: str
    ) -> None:
        """Move label file between splits"""
        ...

    @abstractmethod
    async def exists(self, dataset: str, split: str, filename: str) -> bool:
        """Check if label file exists"""
        ...


# persistence/model_store.py
@dataclass
class ModelInfo:
    name: str
    path: Path
    base_model: str
    dataset_name: str
    created_at: float
    epochs_completed: int
    best_map50: float | None
    is_active: bool

class ModelStore(ABC):
    """Abstract interface for trained model storage"""

    @abstractmethod
    async def save(
        self,
        name: str,
        weights_path: Path,
        config: TrainingConfig,
        metrics: dict
    ) -> ModelInfo:
        """Save a trained model with its metadata"""
        ...

    @abstractmethod
    async def load(self, name: str) -> Path:
        """Get path to model weights file"""
        ...

    @abstractmethod
    async def list(self) -> list[ModelInfo]:
        """List all trained models"""
        ...

    @abstractmethod
    async def get(self, name: str) -> ModelInfo | None:
        """Get model info by name"""
        ...

    @abstractmethod
    async def delete(self, name: str) -> bool:
        """Delete a model and its files"""
        ...

    @abstractmethod
    async def set_active(self, name: str) -> None:
        """Set a model as the active model"""
        ...

    @abstractmethod
    async def get_active(self) -> ModelInfo | None:
        """Get the currently active model"""
        ...
```

### Filesystem Implementation (MVP)

```python
# persistence/filesystem/dataset_store.py
from persistence.dataset_store import DatasetStore, DatasetInfo
import yaml
import shutil

class FilesystemDatasetStore(DatasetStore):
    """Filesystem-based implementation of DatasetStore"""

    def __init__(self, base_path: Path):
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)

    async def create(self, name: str, classes: list[str]) -> DatasetInfo:
        dataset_path = self._base_path / name

        if dataset_path.exists():
            raise ValueError(f"Dataset '{name}' already exists")

        # Create directory structure
        dataset_path.mkdir()
        for split in ["train", "val", "test"]:
            (dataset_path / "images" / split).mkdir(parents=True)
            (dataset_path / "labels" / split).mkdir(parents=True)

        # Create data.yaml
        data_yaml = {
            "path": str(dataset_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(classes),
            "names": {i: name for i, name in enumerate(classes)}
        }

        with open(dataset_path / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f)

        return DatasetInfo(
            name=name,
            path=dataset_path,
            classes=classes,
            num_images={"train": 0, "val": 0, "test": 0},
            created_at=time.time(),
            modified_at=time.time()
        )

    async def delete(self, name: str) -> bool:
        dataset_path = self._base_path / name
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
            return True
        return False

    async def list(self) -> list[DatasetInfo]:
        datasets = []
        for path in self._base_path.iterdir():
            if path.is_dir() and (path / "data.yaml").exists():
                info = await self.get(path.name)
                if info:
                    datasets.append(info)
        return datasets

    # ... other methods


# persistence/filesystem/image_store.py
class FilesystemImageStore(ImageStore):
    """Filesystem-based implementation of ImageStore"""

    def __init__(self, base_path: Path):
        self._base_path = base_path

    async def save(
        self,
        dataset: str,
        split: str,
        filename: str,
        image: np.ndarray
    ) -> ImageInfo:
        path = self._base_path / dataset / "images" / split / filename

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save image
        cv2.imwrite(str(path), image)

        return ImageInfo(
            filename=filename,
            split=split,
            path=path,
            width=image.shape[1],
            height=image.shape[0],
            size_bytes=path.stat().st_size,
            has_labels=False
        )

    async def load(self, dataset: str, split: str, filename: str) -> np.ndarray:
        path = self._base_path / dataset / "images" / split / filename
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return image

    async def delete(self, dataset: str, split: str, filename: str) -> bool:
        path = self._base_path / dataset / "images" / split / filename
        if path.exists():
            path.unlink()
            return True
        return False

    async def move(
        self,
        dataset: str,
        filename: str,
        from_split: str,
        to_split: str
    ) -> None:
        src = self._base_path / dataset / "images" / from_split / filename
        dst = self._base_path / dataset / "images" / to_split / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    # ... other methods


# persistence/filesystem/label_store.py
class FilesystemLabelStore(LabelStore):
    """Filesystem-based implementation of LabelStore"""

    def __init__(self, base_path: Path):
        self._base_path = base_path

    def _label_path(self, dataset: str, split: str, image_filename: str) -> Path:
        """Get label file path for an image"""
        label_filename = Path(image_filename).stem + ".txt"
        return self._base_path / dataset / "labels" / split / label_filename

    async def save(
        self,
        dataset: str,
        split: str,
        filename: str,
        annotations: list[Annotation]
    ) -> None:
        path = self._label_path(dataset, split, filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write YOLO format: class_id x_center y_center width height
        lines = []
        for ann in annotations:
            lines.append(f"{ann.class_id} {ann.x:.6f} {ann.y:.6f} {ann.width:.6f} {ann.height:.6f}")

        with open(path, "w") as f:
            f.write("\n".join(lines))

    async def load(
        self,
        dataset: str,
        split: str,
        filename: str
    ) -> list[dict]:
        path = self._label_path(dataset, split, filename)

        if not path.exists():
            return []

        annotations = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    annotations.append({
                        "class_id": int(parts[0]),
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "width": float(parts[3]),
                        "height": float(parts[4])
                    })
        return annotations

    # ... other methods
```

### Dependency Injection

```python
# persistence/__init__.py
from persistence.filesystem.dataset_store import FilesystemDatasetStore
from persistence.filesystem.image_store import FilesystemImageStore
from persistence.filesystem.label_store import FilesystemLabelStore

def create_stores(base_path: Path) -> tuple[DatasetStore, ImageStore, LabelStore]:
    """Factory function to create persistence stores"""
    return (
        FilesystemDatasetStore(base_path / "datasets"),
        FilesystemImageStore(base_path / "datasets"),
        FilesystemLabelStore(base_path / "datasets")
    )


# main.py - Dependency injection
from persistence import create_stores
from dataset.manager import DatasetManager

# Create stores
dataset_store, image_store, label_store = create_stores(Path("./data"))

# Inject into Dataset Subsystem
dataset_manager = DatasetManager(
    dataset_store=dataset_store,
    image_store=image_store,
    label_store=label_store
)
```

### Benefits of This Design

1. **Testability**: Dataset Subsystem can be tested with mock stores
2. **Flexibility**: Can swap filesystem for database (Ideal state) without changing Dataset Subsystem
3. **Single Responsibility**: Dataset Subsystem handles business logic, Persistence Layer handles storage
4. **Consistency**: All file operations go through one place

### Directory Structure

```
yolo_dataset_creator/
├── backend/                    # FastAPI backend
│   ├── main.py                 # FastAPI app entry point
│   ├── requirements.txt
│   ├── api/                    # API routes (FastAPI routers)
│   │   ├── __init__.py
│   │   ├── feeds.py
│   │   ├── inference.py
│   │   ├── capture/            # CaptureController (in API layer)
│   │   │   ├── __init__.py
│   │   │   ├── router.py       # FastAPI router for /api/capture/*
│   │   │   └── controller.py   # CaptureController logic
│   │   ├── datasets.py
│   │   ├── training.py
│   │   ├── models.py
│   │   ├── notifications.py
│   │   └── system.py
│   ├── feeds/                  # Feeds subsystem
│   │   ├── __init__.py
│   │   ├── manager.py          # FeedManager (raw + derived feeds)
│   │   ├── base.py             # BaseFeed, Frame, FeedConfig
│   │   ├── derived.py          # DerivedFeed, InferenceFrame
│   │   ├── camera.py           # CameraFeed (MVP)
│   │   ├── rtsp.py             # RTSPFeed (Ideal)
│   │   ├── video.py            # VideoFileFeed (Ideal)
│   │   ├── images.py           # ImageFolderFeed (Ideal)
│   │   └── buffer.py           # RingBuffer
│   ├── inference/              # Inference subsystem
│   │   ├── __init__.py
│   │   ├── manager.py          # InferenceManager (produces output feeds)
│   │   ├── session.py          # InferenceSession
│   │   ├── loader.py           # Model loading
│   │   └── detector.py         # YOLO-World and fine-tuned model detection
│   ├── dataset/                # Dataset subsystem (uses Persistence Layer)
│   │   ├── __init__.py
│   │   ├── manager.py          # DatasetManager (business logic only)
│   │   ├── annotations.py      # Annotation validation/conversion
│   │   ├── review.py           # Review queue logic
│   │   └── export.py           # Export/import orchestration
│   ├── training/               # Training subsystem (uses Persistence Layer)
│   │   ├── __init__.py
│   │   ├── runner.py           # Training execution
│   │   ├── monitor.py          # Resource monitoring
│   │   └── config.py           # Training configuration
│   ├── inference/              # Inference subsystem
│   │   ├── __init__.py
│   │   ├── manager.py          # InferenceManager (produces output feeds)
│   │   ├── session.py          # InferenceSession
│   │   ├── loader.py           # Model loading
│   │   └── detector.py         # Detection execution
│   ├── persistence/            # Persistence Layer (all file operations)
│   │   ├── __init__.py         # Factory functions
│   │   ├── dataset_store.py    # DatasetStore interface
│   │   ├── image_store.py      # ImageStore interface
│   │   ├── label_store.py      # LabelStore interface
│   │   ├── model_store.py      # ModelStore interface
│   │   ├── config_store.py     # ConfigStore interface
│   │   └── filesystem/         # Filesystem implementations
│   │       ├── __init__.py
│   │       ├── dataset_store.py
│   │       ├── image_store.py
│   │       ├── label_store.py
│   │       └── model_store.py
│   ├── notifications/          # Notifications subsystem
│   │   ├── __init__.py
│   │   ├── manager.py          # NotificationManager
│   │   ├── models.py           # Notification types
│   │   └── channels.py         # WebSocket, Desktop, Webhook [Ideal]
│   ├── auth/                   # Auth subsystem [Ideal]
│   │   ├── __init__.py
│   │   ├── manager.py          # AuthManager
│   │   ├── models.py           # User, Role, Permission
│   │   ├── dependencies.py     # FastAPI auth dependencies
│   │   └── rbac.py             # Role-based access control
│   ├── websocket/              # WebSocket handlers
│   │   ├── __init__.py
│   │   ├── manager.py          # Connection manager
│   │   ├── video.py            # Video stream handler
│   │   └── events.py           # Event broadcast handler
│   ├── core/                   # Shared utilities
│   │   ├── __init__.py
│   │   ├── events.py           # Pub/sub for subsystems
│   │   ├── resources.py        # Resource monitoring
│   │   ├── config.py           # Pydantic settings
│   │   └── exceptions.py       # Custom exceptions
│   └── models/                 # Pydantic models (API schemas)
│       ├── __init__.py
│       ├── feeds.py
│       ├── scan.py
│       ├── dataset.py
│       ├── training.py
│       └── common.py
├── frontend/                   # Vue.js frontend
│   ├── src/
│   │   ├── main.ts
│   │   ├── App.vue
│   │   ├── router/
│   │   ├── stores/
│   │   ├── composables/
│   │   ├── views/
│   │   ├── components/
│   │   ├── types/
│   │   └── assets/
│   ├── index.html
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── package.json
├── data/                       # Data storage (gitignored)
│   ├── datasets/               # Dataset storage
│   ├── models/                 # Trained model storage
│   └── logs/                   # Log files
├── config/                     # App configuration
│   └── settings.yaml
├── docker/                     # Docker configuration [Ideal]
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
└── README.md
```

---

## 9. Frontend Application (Vue.js)

### Responsibility
Browser-based UI for all modes of operation, built with Vue.js 3 and Composition API.

### Technology

- **Vue.js 3** - Reactive component framework
- **Vite** - Build tool with fast HMR
- **Vue Router** - Client-side routing for modes
- **Pinia** - State management (Vue 3 native)
- **TypeScript** - Type safety throughout

### Component Structure

```
frontend/
├── src/
│   ├── main.ts                 # App entry point
│   ├── App.vue                 # Root component
│   ├── router/
│   │   └── index.ts            # Route definitions
│   ├── stores/                 # Pinia stores
│   │   ├── app.ts              # Global app state
│   │   ├── inference.ts        # Inference state (produces output feeds)
│   │   ├── capture.ts          # Capture state (subscribes to inference feeds)
│   │   ├── dataset.ts          # Dataset mode state
│   │   ├── training.ts         # Training mode state
│   │   └── notifications.ts    # Notification state
│   ├── composables/            # Reusable composition functions
│   │   ├── useApi.ts           # API client
│   │   ├── useWebSocket.ts     # WebSocket connection
│   │   └── useVideoStream.ts   # Video frame handling
│   ├── views/                  # Top-level route views
│   │   ├── ScanView.vue
│   │   ├── DatasetView.vue
│   │   ├── TrainView.vue
│   │   └── ModelView.vue
│   ├── components/
│   │   ├── common/             # Shared components
│   │   │   ├── AppHeader.vue
│   │   │   ├── AppNav.vue
│   │   │   ├── StatusBar.vue
│   │   │   ├── Modal.vue
│   │   │   └── Toast.vue
│   │   ├── inference/          # Inference components (shared by Scan/Model modes)
│   │   │   ├── VideoPlayer.vue       # Subscribes to inference output feed
│   │   │   ├── DetectionOverlay.vue  # Renders detection boxes
│   │   │   ├── PromptEditor.vue      # YOLO-World prompt configuration
│   │   │   └── InferenceStats.vue    # FPS, inference time
│   │   ├── capture/            # Capture components (Scan mode specific)
│   │   │   ├── CaptureControls.vue   # Start/stop capture, manual trigger
│   │   │   └── CaptureStats.vue      # Positive/negative counts
│   │   ├── dataset/            # Dataset mode components
│   │   │   ├── ImageGrid.vue
│   │   │   ├── ImageEditor.vue
│   │   │   ├── AnnotationCanvas.vue
│   │   │   ├── AnnotationList.vue
│   │   │   ├── BulkActions.vue
│   │   │   └── ReviewQueue.vue
│   │   ├── training/           # Training mode components
│   │   │   ├── TrainingConfig.vue
│   │   │   ├── ProgressDisplay.vue
│   │   │   └── ModelList.vue
│   │   └── inference/          # Model mode components
│   │       ├── InferencePlayer.vue
│   │       ├── ModelSelector.vue
│   │       └── DetectionStats.vue
│   ├── types/                  # TypeScript type definitions
│   │   ├── api.ts
│   │   ├── models.ts
│   │   └── websocket.ts
│   └── assets/
│       └── styles/
│           ├── main.css
│           └── variables.css
├── index.html
├── vite.config.ts
├── tsconfig.json
└── package.json
```

### Navigation

```
┌─────────────────────────────────────────────────────────────┐
│  [Scan] [Dataset] [Train] [Model]    Dataset: ▼    [⚙️]    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                     Mode-specific UI                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Status: Scanning... | Captures: 45 | Training: 23/100     │
└─────────────────────────────────────────────────────────────┘
```

### State Management (Pinia Stores)

```typescript
// stores/inference.ts
import { defineStore } from 'pinia'

interface InferenceState {
  status: 'idle' | 'running' | 'error'
  sourceFeedId: string | null      // Raw feed being processed
  outputFeedId: string | null      // Inference output feed ID
  modelId: string | null
  prompts: Record<number, string[]>
  stats: {
    fps: number
    detections: number
    inferenceTimeMs: number
  }
}

export const useInferenceStore = defineStore('inference', {
  state: (): InferenceState => ({
    status: 'idle',
    sourceFeedId: null,
    outputFeedId: null,
    modelId: null,
    prompts: {},
    stats: { fps: 0, detections: 0, inferenceTimeMs: 0 }
  }),

  actions: {
    async startInference(sourceFeedId: string, modelId: string, prompts?: Record<number, string[]>) { ... },
    async stopInference() { ... },
    async updatePrompts(prompts: Record<number, string[]>) { ... },
    updateStats(stats: Partial<InferenceState['stats']>) { ... }
  }
})

// stores/capture.ts
interface CaptureState {
  status: 'idle' | 'running' | 'paused'
  inferenceFeedId: string | null   // Inference feed we're subscribed to
  datasetName: string | null
  config: {
    captureInterval: number
    negativeRatio: number
    confidenceThreshold: number
  }
  stats: {
    totalCaptures: number
    positiveCaptures: number
    negativeCaptures: number
  }
}

export const useCaptureStore = defineStore('capture', {
  state: (): CaptureState => ({
    status: 'idle',
    inferenceFeedId: null,
    datasetName: null,
    config: {
      captureInterval: 2.0,
      negativeRatio: 0.2,
      confidenceThreshold: 0.3
    },
    stats: {
      totalCaptures: 0,
      positiveCaptures: 0,
      negativeCaptures: 0
    }
  }),

  actions: {
    async startCapture(inferenceFeedId: string, datasetName: string, config?: Partial<CaptureState['config']>) { ... },
    async stopCapture() { ... },
    async triggerManualCapture() { ... },
    updateStats(stats: Partial<CaptureState['stats']>) { ... }
  }
})

// stores/dataset.ts
export const useDatasetStore = defineStore('dataset', {
  state: () => ({
    currentDataset: null as string | null,
    images: [] as ImageInfo[],
    reviewQueue: [] as ReviewItem[],
    currentImage: null as ImageInfo | null,
    annotations: [] as Annotation[],
    filter: {
      split: 'all',
      annotated: 'all',
      search: ''
    }
  }),

  getters: {
    filteredImages: (state) => { ... },
    pendingReviewCount: (state) => state.reviewQueue.length
  },

  actions: {
    async loadDataset(name: string) { ... },
    async saveAnnotations() { ... },
    async bulkAction(imageIds: string[], action: string) { ... }
  }
})

// stores/training.ts
export const useTrainingStore = defineStore('training', {
  state: () => ({
    status: 'idle' as 'idle' | 'training' | 'completed' | 'error',
    progress: {
      epoch: 0,
      totalEpochs: 100,
      loss: 0,
      eta: null as string | null
    },
    config: null as TrainingConfig | null,
    error: null as string | null
  }),

  actions: {
    async startTraining(config: TrainingConfig) { ... },
    async stopTraining() { ... },
    updateProgress(progress: Partial<TrainingProgress>) { ... }
  }
})

// stores/notifications.ts
export const useNotificationStore = defineStore('notifications', {
  state: () => ({
    toasts: [] as Toast[],
    banners: [] as Banner[],
    history: [] as Notification[]
  }),

  actions: {
    showToast(toast: Omit<Toast, 'id'>) { ... },
    showBanner(banner: Omit<Banner, 'id'>) { ... },
    dismiss(id: string) { ... }
  }
})
```

### Composables

```typescript
// composables/useWebSocket.ts
import { ref, onMounted, onUnmounted } from 'vue'

export function useWebSocket(url: string) {
  const socket = ref<WebSocket | null>(null)
  const isConnected = ref(false)
  const lastMessage = ref<any>(null)

  const connect = () => {
    socket.value = new WebSocket(url)

    socket.value.onopen = () => {
      isConnected.value = true
    }

    socket.value.onmessage = (event) => {
      lastMessage.value = JSON.parse(event.data)
    }

    socket.value.onclose = () => {
      isConnected.value = false
      // Auto-reconnect after 2 seconds
      setTimeout(connect, 2000)
    }
  }

  const send = (data: any) => {
    if (socket.value?.readyState === WebSocket.OPEN) {
      socket.value.send(JSON.stringify(data))
    }
  }

  onMounted(connect)
  onUnmounted(() => socket.value?.close())

  return { isConnected, lastMessage, send }
}

// composables/useVideoStream.ts
export function useVideoStream(feedId: Ref<string | null>) {
  const frame = ref<string | null>(null)  // Base64 frame
  const fps = ref(0)

  // Subscribe to video frames via WebSocket
  const { lastMessage } = useWebSocket('/ws/video')

  watch(lastMessage, (msg) => {
    if (msg?.type === 'frame' && msg.feedId === feedId.value) {
      frame.value = msg.data
      fps.value = msg.fps
    }
  })

  return { frame, fps }
}

// composables/useApi.ts
export function useApi() {
  const baseUrl = '/api'

  const get = async <T>(path: string): Promise<T> => {
    const res = await fetch(`${baseUrl}${path}`)
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  const post = async <T>(path: string, data?: any): Promise<T> => {
    const res = await fetch(`${baseUrl}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: data ? JSON.stringify(data) : undefined
    })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  }

  // ... put, delete, upload methods

  return { get, post, put, delete: del, upload }
}
```

### Example Component

```vue
<!-- components/inference/VideoPlayer.vue -->
<script setup lang="ts">
import { computed } from 'vue'
import { useInferenceStore } from '@/stores/inference'
import { useVideoStream } from '@/composables/useVideoStream'
import DetectionOverlay from './DetectionOverlay.vue'

const inferenceStore = useInferenceStore()
// Subscribe to the inference output feed (which includes detections)
const { frame, fps, detections } = useVideoStream(
  computed(() => inferenceStore.outputFeedId)
)
</script>

<template>
  <div class="video-player">
    <div class="video-container">
      <img
        v-if="frame"
        :src="`data:image/jpeg;base64,${frame}`"
        alt="Video feed"
      />
      <div v-else class="no-feed">
        No video feed
      </div>
      <DetectionOverlay :detections="detections" />
    </div>
    <div class="video-stats">
      <span>{{ fps.toFixed(1) }} FPS</span>
      <span>{{ detections.length }} detections</span>
      <span>{{ inferenceStore.stats.inferenceTimeMs.toFixed(0) }}ms</span>
    </div>
  </div>
</template>

<style scoped>
.video-player {
  position: relative;
  background: #1a1a1a;
}

.video-container {
  position: relative;
  width: 100%;
  aspect-ratio: 16/9;
}

.video-container img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.video-stats {
  display: flex;
  gap: 1rem;
  padding: 0.5rem;
  font-size: 0.875rem;
  color: #888;
}
</style>
```

---

## Event System

### Cross-Subsystem Communication

```python
# events.py
class EventBus:
    """Simple pub/sub for subsystem communication"""

    # Event types
    SCAN_STARTED = "scan.started"
    SCAN_STOPPED = "scan.stopped"
    SCAN_CAPTURE = "scan.capture"
    SCAN_PAUSED = "scan.paused"

    TRAINING_STARTED = "training.started"
    TRAINING_PROGRESS = "training.progress"
    TRAINING_COMPLETED = "training.completed"
    TRAINING_ERROR = "training.error"

    RESOURCE_WARNING = "resource.warning"
    RESOURCE_CRITICAL = "resource.critical"

    def subscribe(self, event_type, callback): ...
    def publish(self, event_type, data): ...
```

### Event Flow Example: Resource Constraint

```
┌──────────────┐     RESOURCE_CRITICAL      ┌──────────────┐
│   Resource   │───────────────────────────▶│     Scan     │
│   Monitor    │                            │   Subsystem  │
└──────────────┘                            └──────────────┘
       │                                           │
       │                                           │ pause()
       │                                           ▼
       │                                    ┌──────────────┐
       │                                    │    PAUSED    │
       │                                    └──────────────┘
       │
       │           RESOURCE_WARNING          ┌──────────────┐
       └────────────────────────────────────▶│   Frontend   │
                                             │   (banner)   │
                                             └──────────────┘
```

---

## MVP Implementation Phases

### Phase 1: Project Setup & Core Infrastructure
- [ ] Project restructure (backend/, frontend/ directories)
- [ ] FastAPI app skeleton with Uvicorn
- [ ] Vue.js 3 project setup with Vite + TypeScript
- [ ] Pinia store structure
- [ ] WebSocket setup (FastAPI native)
- [ ] Event bus implementation
- [ ] Pydantic settings/configuration management
- [ ] Development proxy configuration (Vite → FastAPI)

### Phase 2: Feeds Subsystem
- [ ] Pydantic models for feeds
- [ ] BaseFeed interface and Frame dataclass
- [ ] RingBuffer implementation
- [ ] CameraFeed (USB/CSI via OpenCV)
- [ ] FeedManager with async subscriptions
- [ ] Feed API endpoints (FastAPI router)
- [ ] WebSocket video frame streaming

### Phase 3: Notifications Subsystem
- [ ] Notification Pydantic models
- [ ] NotificationManager
- [ ] WebSocket broadcast to clients
- [ ] Event bus subscriptions
- [ ] Vue Toast/Banner components
- [ ] Pinia notification store

### Phase 4: Inference Subsystem
- [ ] InferenceManager with feed subscription
- [ ] YOLO-World model loader with prompts
- [ ] Async detection runner
- [ ] Inference output feed producer (registers derived feeds)
- [ ] Fine-tuned model loader
- [ ] Model hot-swap support
- [ ] WebSocket streaming of inference frames

### Phase 5: CaptureController
- [ ] CaptureController implementation (subscribes to inference feeds)
- [ ] Capture logic (interval-based triggers)
- [ ] Negative frame capture (ratio-based sampling)
- [ ] Manual capture trigger
- [ ] Capture stats tracking and WebSocket broadcasting
- [ ] Integration with Dataset Subsystem for saving captures

### Phase 6: Dataset Subsystem Extensions
- [ ] Prompt storage (prompts.yaml)
- [ ] Metadata storage (metadata.json)
- [ ] Review queue
- [ ] Auto-annotation flag
- [ ] Bulk operations API

### Phase 7: Training Subsystem
- [ ] Async training runner (ThreadPoolExecutor)
- [ ] Progress reporting via WebSocket
- [ ] Model registry
- [ ] Resource monitor integration
- [ ] Resource-aware pausing of Scan

### Phase 8: Frontend Views & Components
- [ ] Vue Router setup (Scan, Dataset, Train, Model views)
- [ ] App layout (header, nav, status bar)
- [ ] Scan mode UI (VideoPlayer, PromptEditor, CaptureControls)
- [ ] Dataset mode UI (ImageGrid, ImageEditor, ReviewQueue)
- [ ] Train mode UI (TrainingConfig, ProgressDisplay, ModelList)
- [ ] Model mode UI (InferencePlayer, ModelSelector)

### Phase 9: Integration & Polish
- [ ] Cross-subsystem event wiring
- [ ] Resource constraint handling
- [ ] Error handling (API + frontend)
- [ ] Headless mode support
- [ ] Production build configuration

---

## Ideal State Additional Phases

### Phase 10: Extended Feed Sources
- [ ] RTSPFeed implementation
- [ ] VideoFileFeed implementation
- [ ] ImageFolderFeed implementation
- [ ] Multi-feed parallel support
- [ ] Auto-reconnect logic
- [ ] Advanced health monitoring

### Phase 11: Authentication & RBAC
- [ ] User model and storage
- [ ] Role and Permission models
- [ ] Authentication middleware
- [ ] Login/logout endpoints
- [ ] RBAC enforcement on API endpoints
- [ ] Frontend auth integration

### Phase 12: Advanced Notifications
- [ ] Notification history persistence
- [ ] Desktop notifications (system tray)
- [ ] Sound alerts
- [ ] Webhook channel
- [ ] User notification preferences

### Phase 13: Containerization & CLI
- [ ] Dockerfile
- [ ] Docker Compose setup
- [ ] CLI command structure
- [ ] CLI implementation for all features

### Phase 14: Database Backend
- [ ] Database schema design
- [ ] Migration from filesystem
- [ ] Query optimization

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Backend | Python 3.10+ | Ultralytics compatibility, ML ecosystem |
| Web Framework | FastAPI | Native async, WebSocket support, Pydantic validation |
| ASGI Server | Uvicorn | High-performance async server |
| ML Framework | Ultralytics (YOLO) | Best-in-class YOLO implementation |
| Camera | OpenCV | Universal camera support |
| Frontend | Vue.js 3 | Reactive components, composition API, excellent DX |
| Frontend Build | Vite | Fast builds, HMR, modern tooling |
| State Management | Pinia | Vue 3 native, simple and type-safe |
| Async | asyncio + ThreadPoolExecutor | Native async for I/O, threads for CPU-bound ML |
| Config | YAML | Human-readable, existing usage |
| Validation | Pydantic | Request/response validation, settings management |

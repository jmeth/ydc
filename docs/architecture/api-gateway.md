# API Gateway (includes CaptureController)

> Part of the [YOLO Dataset Creator Architecture](../ARCHITECTURE.md)

## Responsibility

Single entry point for all client-server communication. Routes requests to appropriate subsystems, manages WebSocket connections for real-time updates, serves static files, and hosts the CaptureController for capture logic orchestration.

## Components

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

## MVP API Endpoints

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

## WebSocket Events

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

## Technology

- FastAPI with native WebSocket support
- Uvicorn ASGI server (async-native)
- Pydantic for request/response validation

## FastAPI Application Structure

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

## Example Router (CaptureController)

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

## CaptureController Implementation

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

## WebSocket Handler

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

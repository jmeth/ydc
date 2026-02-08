# YOLO Dataset Creator - Architecture

## Quick start

```
#backend
python -m venv venv
source ./venv/bin/activate
uvicorn backend.main:app --reload --port 8000

#frontend
cd frontend
npm install
npm run dev
```

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

Each subsystem has detailed architecture documented in its own file:

| # | Subsystem | Description | Details |
|---|-----------|-------------|---------|
| 1 | [API Gateway](architecture/api-gateway.md) | REST/WebSocket entry point, CaptureController | Endpoints, routers, WebSocket events |
| 2 | [Feeds](architecture/feeds.md) | Raw + derived video feed management | Feed types, buffering, subscriptions |
| 3 | [Dataset](architecture/dataset.md) | Dataset business logic, annotations, review | CRUD, review queue, file formats |
| 4 | [Training](architecture/training.md) | Model training jobs, resource monitoring | Training runner, model registry |
| 5 | [Inference](architecture/inference.md) | Detection execution, output feed production | Model loading, inference sessions |
| 6 | [Notifications](architecture/notifications.md) | Centralized event notifications | Toast/banner/alert types, channels |
| 7 | [Authentication](architecture/authentication.md) | Auth & RBAC [Ideal Only] | JWT, roles, permissions |
| 8 | [Persistence](architecture/persistence.md) | All filesystem operations | Store interfaces, DI, directory layout |
| 9 | [Frontend](architecture/frontend.md) | Vue.js browser UI | Components, stores, composables |

**Note:** The Scan Subsystem has been eliminated. Its responsibilities are now split:
- **Detection**: Handled by Inference Subsystem (produces inference output feeds)
- **Capture logic**: Handled by CaptureController in the API Gateway layer

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

## Directory Structure

```
yolo_dataset_creator/
├── backend/                    # FastAPI backend
│   ├── main.py                 # FastAPI app entry point
│   ├── requirements.txt
│   ├── api/                    # API routes (FastAPI routers)
│   ├── feeds/                  # Feeds subsystem
│   ├── inference/              # Inference subsystem
│   ├── dataset/                # Dataset subsystem
│   ├── training/               # Training subsystem
│   ├── persistence/            # Persistence Layer
│   ├── notifications/          # Notifications subsystem
│   ├── auth/                   # Auth subsystem [Ideal]
│   ├── websocket/              # WebSocket handlers
│   ├── core/                   # Shared utilities
│   └── models/                 # Pydantic models (API schemas)
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

See [Persistence Layer](architecture/persistence.md) for the full backend directory tree.

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

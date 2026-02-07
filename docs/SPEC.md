# YOLO Dataset Creator - Product Specification

## Overview

Expand the YOLO Dataset Creator into a comprehensive tool for edge-deployed object detection workflows. The system supports the full lifecycle: zero-shot scanning with YOLO-World, automated dataset creation, model training, and inference deployment.

## Target Hardware

- **Primary**: NVIDIA DGX Spark (128GB RAM, GB10 GPU)
- **Secondary**: NVIDIA Jetson Orin Nano (8GB RAM)

## Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Backend** | Python 3.10+ / FastAPI | Native async, Ultralytics compatibility, Pydantic validation |
| **Frontend** | Vue.js 3 / TypeScript | Reactive components, composition API, type safety |
| **Build** | Vite | Fast dev server, HMR, optimized production builds |
| **State** | Pinia | Vue 3 native state management |
| **ML** | Ultralytics YOLO | Best-in-class object detection |
| **Video** | OpenCV | Universal camera/video support |
| **Server** | Uvicorn | High-performance ASGI server |

## User Personas

- **MVP**: Field operators needing simple workflows with smart defaults
- **Ideal**: Field operators (simple mode) + ML engineers (advanced mode)

---

## Modes of Operation

### 1. Scan Mode
Use YOLO-World with text prompts to detect objects of interest in live video. Automatically capture frames for dataset creation.

### 2. Dataset Mode
Review, edit, and manage auto-captured images and annotations. Refine auto-generated annotations before training.

### 3. Train Mode
Configure and execute model training with progress monitoring. Supports background training.

### 4. Model Mode
Run trained models for inference without dataset capture functionality.

---

## Feature Requirements

### Input & Capture (Feeds Subsystem)

| Feature | MVP | Ideal |
|---------|-----|-------|
| USB/CSI camera | Yes | Yes |
| RTSP/IP streams | No | Yes |
| Video files | No | Yes |
| Image folders | No | Yes |
| Multi-source parallel | No | Yes |
| Topology | Standalone device | Hub subscribes to thin edge capture devices |
| Edge devices | N/A | Stream only, don't run this software |

### Feed Management

| Feature | MVP | Ideal |
|---------|-----|-------|
| Unified feed interface | Yes | Yes |
| Ring buffer (frame history) | Yes (fixed size) | Yes (configurable) |
| Frame timestamping | Yes | Yes |
| Frame rate normalization | No | Yes |
| Feed subscription model | Yes (callbacks) | Yes |
| Connection health monitoring | Basic (connected/error) | Full (latency, dropped frames) |
| Auto-reconnect (RTSP) | No | Yes |
| Feed pause/resume | Yes | Yes |
| Feed API endpoints | Yes | Yes |

### Scan Mode

| Feature | MVP | Ideal |
|---------|-----|-------|
| Auto-capture trigger | Interval-based (user settable) | Change-based (new object, movement, angle) |
| Negative frame capture | Yes, configurable ratio | Yes |
| Manual capture button | Yes | Yes |
| Prompt interface | Structured list with variations per class | Same + complex mappings/filtering |
| Save prompts with dataset | Yes | Yes |
| Generate initial annotations | Yes | Yes |
| Confidence threshold | User-adjustable (global) | Per-class thresholds |
| Low-confidence handling | Flag for review | Flag for review |
| Bulk discard | Yes | Yes |
| Live video feed | Yes | Yes |
| Bounding box overlay | Yes | Yes |
| Class labels on boxes | Yes | Yes |
| Confidence scores on boxes | Yes | Yes |
| Capture indicator (flash) | Yes | Yes |
| Stats panel (counts, rate) | Yes | Yes |
| Flagged count display | No | Yes |
| Headless mode | Yes | Yes |

### Dataset Mode

| Feature | MVP | Ideal |
|---------|-----|-------|
| Review workflow | Hybrid (grid + single-image editor) | Hybrid |
| Auto annotation indicator | "Auto" badge | "Auto" badge |
| Bulk actions from grid | Yes | Yes |
| Adjust boxes (move, resize) | Yes | Yes |
| Change labels | Yes | Yes |
| Delete annotations | Yes | Yes |
| Add missed objects | Yes | Yes |
| Mark as verified | No | Yes |
| Flag for later review | No | Yes |
| Propagate corrections | No | TBD |

### Train Mode

| Feature | MVP | Ideal |
|---------|-----|-------|
| Training location | Local only | Local or remote (offload to hub) |
| Progress bar | Yes | Yes |
| Error output | Yes | Yes |
| Loss curves | No | Yes |
| Live metrics (mAP, etc.) | No | Yes |
| Sample predictions | No | Yes |
| Log output | No | Yes |
| Completion notification | Basic | Yes |
| Background training | Yes | Yes |
| Performance impact warning | Yes (banner) | Yes |
| Resource monitoring | Basic (pause scan if constrained) | Advanced thresholds |

### Model Management

| Feature | MVP | Ideal |
|---------|-----|-------|
| User-defined model names | Yes | Yes |
| Delete/cleanup models | Yes | Yes |
| Active model display | Yes | Yes |
| Retention/history | Manual cleanup | User configurable |
| Model comparison | No | Yes |
| Metadata storage | Basic | Full (params, dataset snapshot, metrics) |

### Model Mode (Inference)

| Feature | MVP | Ideal |
|---------|-----|-------|
| Display detections | Yes | Yes |
| Log to file | No | TBD |
| Database storage | No | Yes |
| Webhooks | No | Yes |
| MQTT/message bus | No | Yes |
| Screenshot on detection | No | Yes |
| Model switching (hot swap) | Yes (dropdown) | Yes |
| Simultaneous models | No | Yes (hardware dependent) |

### Concurrent Operations

| Feature | MVP | Ideal |
|---------|-----|-------|
| Scan + Annotate | Yes | Yes |
| Scan + Train | Yes (pause scan if resources constrained) | Yes |
| Multiple scans | No | Yes |

### Notifications

| Feature | MVP | Ideal |
|---------|-----|-------|
| In-app toast notifications | Yes | Yes |
| Resource warning banners | Yes | Yes |
| Training completion alerts | Yes | Yes |
| Capture event feedback | Yes (flash indicator) | Yes |
| Error notifications | Yes | Yes |
| Notification history/log | No | Yes |
| Sound alerts | No | Yes (configurable) |
| Desktop notifications | No | Yes (when headless) |
| External webhooks | No | Yes |
| Notification preferences | No | Yes |

### System Architecture

| Feature | MVP | Ideal |
|---------|-----|-------|
| Deployment | Web app (FastAPI + Vue.js, served locally) | Containerized (Docker) |
| CLI | No | Full CLI (all features) |
| Dataset storage | Filesystem | Filesystem or database-backed |
| App state | Flat files | Database |
| Model files | Filesystem | Filesystem |
| Training history | Log files | Database |
| Scan session data | Flat files | Database |

### User Experience

| Feature | MVP | Ideal |
|---------|-----|-------|
| Default mode | Simple (field operators) | Simple |
| Advanced mode | No | Yes (toggle) |
| Smart defaults | Yes | Yes |
| Dataset management | Single dataset at a time | TBD |
| Zip export/import | Yes | Yes + hub sync |
| Dataset sharing | Between devices, mission retention | Same |

### Authentication & Authorization

| Feature | MVP | Ideal |
|---------|-----|-------|
| Authentication required | No | Yes |
| User accounts | No | Yes |
| Role-based access control (RBAC) | No | Yes |
| Roles | N/A | Admin, Operator, Viewer |
| Session management | No | Yes |
| API authentication | No | Yes (token-based) |
| Audit logging | No | Yes |

#### Ideal State Roles

| Role | Scan | Annotate | Train | Model | Settings | Users |
|------|------|----------|-------|-------|----------|-------|
| **Admin** | Full | Full | Full | Full | Full | Full |
| **Operator** | Full | Full | Full | Full | View | No |
| **Viewer** | View | View | View | View | No | No |

---

## Non-Functional Requirements

### Performance
- Scan Mode must maintain real-time video preview (target 15+ FPS on Jetson Orin Nano)
- Training should not block UI
- Resource monitoring to prevent system overload

### Usability
- MVP targets field operators with minimal ML experience
- Smart defaults for all configurable options
- Clear visual feedback for all operations

### Portability
- Must run on both DGX Spark and Jetson Orin Nano
- Datasets exportable/importable via zip for device transfer

### Reliability
- Graceful handling of hardware constraints
- Auto-pause scanning when resources constrained during training
- Persistent state survives restarts

---

## Out of Scope

- GPIO/hardware triggers
- Multi-user collaboration (single operator assumed)
- Cloud deployment (local/edge only)
- Real-time model training (batch training only)

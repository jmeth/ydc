# YOLO Dataset Creator

A simple web-based tool for creating YOLO-format datasets for object detection training.

## Instructions

- Comment code. Function headers that include intent and inputs/outputs.
- All API endpoints have openapi spec definitions
- Write unit tests for new code when applicable
- Write e2e tests for new features
- If new errors are found, check to see if new unit or e2e tests can be developed to catch those errors in the future.
- Don't bypass unit tests with dummy checks, the point is to catch errors, not pass.
- If doing large feature or refactor work, commit code at regular intervals, don't push.

## Project Overview

This tool allows users to:
- View video feeds
- Capture outputs from video feeds for use in building datasets
- Run inference models against feeds
- Develop Yolo datasets for finetuning
- Finetune yolo models using datasets

## Architecture

### Tech Stack
- **Frontend**: Vue.js 3 + TypeScript + Vite + Pinia
- **Backend**: FastAPI + Pydantic + Uvicorn

### Directory Structure
```
yolo_dataset_creator/
├── server.py              # Legacy Flask backend (port 5001, do not modify)
├── static/                # Legacy Frontend (do not modify)
├── requirements.txt       # All Python deps (legacy + new)
├── pyproject.toml         # pytest config
├── backend/               # FastAPI backend
│   ├── main.py            # App entry point (uvicorn backend.main:app)
│   ├── core/              # Config, events, exceptions
│   ├── api/               # REST API routers
│   ├── websocket/         # WebSocket endpoints
│   ├── models/            # Pydantic schemas
│   ├── feeds/             # Feeds subsystem (future)
│   ├── inference/         # Inference subsystem (future)
│   ├── dataset/           # Dataset subsystem (future)
│   ├── training/          # Training subsystem (future)
│   ├── persistence/       # Persistence layer (future)
│   ├── notifications/     # Notifications subsystem (future)
│   └── tests/             # Backend unit tests
├── frontend/              # Vue.js 3 frontend
│   ├── src/
│   │   ├── views/         # Route views (Scan, Dataset, Train, Model)
│   │   ├── stores/        # Pinia stores
│   │   ├── composables/   # useApi, useWebSocket, useVideoStream
│   │   ├── components/    # Vue components
│   │   └── types/         # TypeScript type definitions
│   └── vite.config.ts     # Dev proxy: /api→:8000, /ws→ws://:8000
├── datasets/              # Dataset storage (gitignored)
├── docs/                  # Architecture and user guides
├── utils/                 # Helper scripts
└── tests/                 # E2E and integration tests
```

### Running
- **Backend**: `source venv/bin/activate && uvicorn backend.main:app --reload --port 8000`
- **Frontend**: `cd frontend && npm run dev` (Vite on port 5173)
- **Tests**: `source venv/bin/activate && python -m pytest backend/tests/ -v`
- **Legacy**: `python server.py` (port 5001, unchanged)

### Detailed Architecture

See docs/ARCHITECTURE.md for top-level design, subsystem interactions, and implementation phases.
Subsystem-specific architecture is in docs/architecture/ (api-gateway, feeds, dataset, training, inference, notifications, authentication, persistence, frontend).
See docs/SPEC.md for product specification.


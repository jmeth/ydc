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
- **Frontend**: Vue.js
- **Backend**: FastAPI

### Directory Structure
```
yolo_dataset_creator/
├── server.py              # Legacy Flask backend server
├── static/                # Legacy Frontend
├── datasets/              # Dataset storage (gitignored)
| - frontend/              # vuejs frontend
| - backend/               # fastapi backend
| - docs/                  # architecture and user guides
| - utils/                 # helper scripts
| - tests/                 # e2e and other tests that don't belong in src code
└── CLAUDE.md
```

### Detailed Architecture

See docs/ARCHITECTURE.md for top-level design, subsystem interactions, and implementation phases.
Subsystem-specific architecture is in docs/architecture/ (api-gateway, feeds, dataset, training, inference, notifications, authentication, persistence, frontend).
See docs/SPEC.md for product specification.


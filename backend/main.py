"""
FastAPI application entry point.

Creates the app with lifespan management, CORS middleware, and all
router registrations. Run with: uvicorn backend.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.core.events import event_bus
from backend.core.exceptions import (
    NotFoundError,
    ConflictError,
    NotImplementedError,
    ValidationError,
    SubsystemError,
)
from backend.feeds.manager import FeedManager
from backend.feeds.streaming import FeedStreamer
from backend.inference.manager import InferenceManager
from backend.notifications.manager import NotificationManager
from backend.websocket.manager import connection_manager
from backend.models.common import StatusResponse, ErrorResponse

# API routers
from backend.api import feeds, inference, datasets, training, models, system
from backend.api.capture import router as capture_router

# Notifications router
from backend.api import notifications

# WebSocket routers
from backend.websocket import video, events


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of shared resources."""
    await event_bus.start()

    # Initialize persistence layer
    from backend.persistence import create_stores, set_stores
    stores = create_stores(
        data_dir=Path(settings.data_dir),
        models_dir=Path(settings.models_dir),
    )
    set_stores(stores)

    # Initialize feeds subsystem
    feed_manager = FeedManager()
    feed_streamer = FeedStreamer(
        feed_manager=feed_manager,
        connection_manager=connection_manager,
        stream_fps=settings.feed_stream_fps,
        jpeg_quality=settings.feed_jpeg_quality,
    )

    # Inject FeedManager into the API router
    from backend.api.feeds import set_feed_manager
    set_feed_manager(feed_manager)

    # Initialize notifications subsystem
    notification_manager = NotificationManager(connection_manager)
    notification_manager.setup_event_subscriptions(event_bus)

    from backend.api.notifications import set_notification_manager
    set_notification_manager(notification_manager)

    # Initialize inference subsystem
    inference_manager = InferenceManager(
        feed_manager=feed_manager,
        notification_manager=notification_manager,
        event_bus=event_bus,
    )

    from backend.api.inference import set_inference_manager
    set_inference_manager(inference_manager)

    # Initialize dataset subsystem
    from backend.dataset import set_dataset_manager
    from backend.dataset.manager import DatasetManager
    from backend.persistence import get_dataset_store, get_image_store, get_label_store

    dataset_manager = DatasetManager(
        dataset_store=get_dataset_store(),
        image_store=get_image_store(),
        label_store=get_label_store(),
        event_bus=event_bus,
    )
    set_dataset_manager(dataset_manager)

    await feed_streamer.start()

    yield

    await feed_streamer.stop()
    inference_manager.stop_all()
    feed_manager.shutdown()
    set_dataset_manager(None)
    set_stores(None)
    await event_bus.stop()


app = FastAPI(
    title="YOLO Dataset Creator",
    version="2.0.0",
    description="Web-based tool for creating YOLO-format datasets for object detection training",
    lifespan=lifespan,
)

# CORS middleware for Vue dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Exception handlers ---

@app.exception_handler(NotFoundError)
async def not_found_handler(request, exc: NotFoundError):
    """Return 404 for NotFoundError."""
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=404, content={"error": exc.message, "detail": exc.detail})


@app.exception_handler(ConflictError)
async def conflict_handler(request, exc: ConflictError):
    """Return 409 for ConflictError."""
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=409, content={"error": exc.message, "detail": exc.detail})


@app.exception_handler(NotImplementedError)
async def not_implemented_handler(request, exc: NotImplementedError):
    """Return 501 for NotImplementedError."""
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=501, content={"error": exc.message, "detail": exc.detail})


@app.exception_handler(ValidationError)
async def validation_handler(request, exc: ValidationError):
    """Return 422 for ValidationError."""
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"error": exc.message, "detail": exc.detail})


@app.exception_handler(SubsystemError)
async def subsystem_handler(request, exc: SubsystemError):
    """Return 500 for SubsystemError."""
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"error": exc.message, "detail": exc.detail})


# --- REST API routers ---

app.include_router(feeds.router, prefix="/api/feeds", tags=["feeds"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(capture_router.router, prefix="/api/capture", tags=["capture"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(notifications.router, prefix="/api/notifications", tags=["notifications"])
app.include_router(system.router, prefix="/api/system", tags=["system"])

# --- WebSocket routers ---

app.include_router(video.router, tags=["websocket"])
app.include_router(events.router, tags=["websocket"])


# --- Health check ---

@app.get(
    "/api/health",
    response_model=StatusResponse,
    tags=["system"],
    summary="Health check",
    description="Returns basic service health status.",
)
async def health_check() -> StatusResponse:
    """Return service health status."""
    return StatusResponse(status="ok")

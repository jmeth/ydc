"""
Feeds API router — video feed CRUD, pause/resume, and snapshot endpoints.

Uses a FeedManager injected via set_feed_manager() during app startup
to manage feed lifecycle. EventBus events are published from the async
endpoint layer since event_bus.publish() is async.
"""

import logging

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from backend.core.events import event_bus, FEED_ADDED, FEED_REMOVED
from backend.core.exceptions import NotFoundError
from backend.feeds.manager import FeedManager
from backend.feeds.models import FeedConfig, FeedType, FeedStatus
from backend.models.feeds import CreateFeedRequest, FeedInfoResponse, FeedListResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level FeedManager reference, set during lifespan startup
_feed_manager: FeedManager | None = None


def set_feed_manager(manager: FeedManager) -> None:
    """
    Inject the FeedManager instance at startup.

    Called from the app lifespan to avoid circular imports and
    module-level singleton issues.

    Args:
        manager: The initialized FeedManager
    """
    global _feed_manager
    _feed_manager = manager


def _get_manager() -> FeedManager:
    """
    Get the injected FeedManager, raising 500 if not initialized.

    Returns:
        The active FeedManager instance.
    """
    if _feed_manager is None:
        raise RuntimeError("FeedManager not initialized")
    return _feed_manager


def _feed_info_to_response(info) -> FeedInfoResponse:
    """
    Convert an internal FeedInfo dataclass to an API response schema.

    Args:
        info: FeedInfo from FeedManager

    Returns:
        FeedInfoResponse Pydantic model.
    """
    return FeedInfoResponse(
        feed_id=info.feed_id,
        feed_type=info.config.feed_type.value,
        source=info.config.source,
        name=info.config.name,
        status=info.status.value,
        fps=info.fps,
        resolution=list(info.resolution) if info.resolution else None,
        frame_count=info.frame_count,
    )


@router.get(
    "",
    response_model=FeedListResponse,
    summary="List feeds",
    description="List all active video feeds with their current status.",
)
async def list_feeds() -> FeedListResponse:
    """Return all registered feeds."""
    manager = _get_manager()
    feeds = manager.list_feeds()
    return FeedListResponse(
        feeds=[_feed_info_to_response(f) for f in feeds],
        count=len(feeds),
    )


@router.post(
    "",
    response_model=FeedInfoResponse,
    status_code=201,
    summary="Create feed",
    description="Register and start a new video feed source.",
    responses={400: {"description": "Feed connection failed"}},
)
async def create_feed(request: CreateFeedRequest) -> JSONResponse:
    """Create a new feed and start capturing frames."""
    manager = _get_manager()

    # Validate feed type
    try:
        feed_type = FeedType(request.feed_type)
    except ValueError:
        return JSONResponse(
            status_code=422,
            content={
                "error": "Invalid feed type",
                "detail": f"Unsupported feed type '{request.feed_type}'. Valid types: {[t.value for t in FeedType]}",
            },
        )

    config = FeedConfig(
        feed_type=feed_type,
        source=request.source,
        name=request.name or f"{feed_type.value}-{request.source}",
        buffer_size=request.buffer_size,
    )

    feed_id = manager.add_feed(config)
    if feed_id is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Feed connection failed",
                "detail": f"Could not connect to {feed_type.value} source '{request.source}'",
            },
        )

    # Publish event from the async layer
    await event_bus.publish(FEED_ADDED, {"feed_id": feed_id, "feed_type": feed_type.value, "source": request.source})

    info = manager.get_feed_info(feed_id)
    response_data = _feed_info_to_response(info).model_dump()
    return JSONResponse(status_code=201, content=response_data)


@router.get(
    "/{feed_id}",
    response_model=FeedInfoResponse,
    summary="Get feed details",
    description="Get metadata for a specific video feed.",
    responses={404: {"description": "Feed not found"}},
)
async def get_feed(feed_id: str) -> FeedInfoResponse:
    """Get details for a specific feed."""
    manager = _get_manager()
    info = manager.get_feed_info(feed_id)
    if info is None:
        raise NotFoundError("Feed", feed_id)
    return _feed_info_to_response(info)


@router.delete(
    "/{feed_id}",
    status_code=204,
    summary="Delete feed",
    description="Stop and remove a video feed source.",
    responses={404: {"description": "Feed not found"}},
)
async def delete_feed(feed_id: str) -> Response:
    """Delete a feed and stop its capture thread."""
    manager = _get_manager()
    removed = manager.remove_feed(feed_id)
    if not removed:
        raise NotFoundError("Feed", feed_id)

    await event_bus.publish(FEED_REMOVED, {"feed_id": feed_id})
    return Response(status_code=204)


@router.post(
    "/{feed_id}/pause",
    response_model=FeedInfoResponse,
    summary="Pause feed",
    description="Pause frame capture on a feed (keeps connection open).",
    responses={404: {"description": "Feed not found"}, 409: {"description": "Feed not active"}},
)
async def pause_feed(feed_id: str) -> JSONResponse:
    """Pause a feed's capture thread."""
    manager = _get_manager()
    info = manager.get_feed_info(feed_id)
    if info is None:
        raise NotFoundError("Feed", feed_id)

    if not manager.pause(feed_id):
        return JSONResponse(
            status_code=409,
            content={"error": "Cannot pause feed", "detail": f"Feed '{feed_id}' is not in active state"},
        )

    info = manager.get_feed_info(feed_id)
    return JSONResponse(status_code=200, content=_feed_info_to_response(info).model_dump())


@router.post(
    "/{feed_id}/resume",
    response_model=FeedInfoResponse,
    summary="Resume feed",
    description="Resume frame capture on a paused feed.",
    responses={404: {"description": "Feed not found"}, 409: {"description": "Feed not paused"}},
)
async def resume_feed(feed_id: str) -> JSONResponse:
    """Resume a paused feed's capture thread."""
    manager = _get_manager()
    info = manager.get_feed_info(feed_id)
    if info is None:
        raise NotFoundError("Feed", feed_id)

    if not manager.resume(feed_id):
        return JSONResponse(
            status_code=409,
            content={"error": "Cannot resume feed", "detail": f"Feed '{feed_id}' is not in paused state"},
        )

    info = manager.get_feed_info(feed_id)
    return JSONResponse(status_code=200, content=_feed_info_to_response(info).model_dump())


@router.get(
    "/{feed_id}/snapshot",
    summary="Get feed snapshot",
    description="Get the latest frame from a feed as a JPEG image.",
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "JPEG snapshot"},
        404: {"description": "Feed not found or no frame available"},
    },
)
async def get_snapshot(feed_id: str) -> Response:
    """Return the latest frame as a JPEG image."""
    manager = _get_manager()
    info = manager.get_feed_info(feed_id)
    if info is None:
        raise NotFoundError("Feed", feed_id)

    frame = manager.get_frame(feed_id)
    if frame is None:
        return JSONResponse(
            status_code=404,
            content={"error": "No frame available", "detail": "Feed has not captured any frames yet"},
        )

    # Encode frame as JPEG
    success, buffer = cv2.imencode(".jpg", frame.data, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return JSONResponse(status_code=500, content={"error": "Frame encoding failed"})

    return Response(content=buffer.tobytes(), media_type="image/jpeg")

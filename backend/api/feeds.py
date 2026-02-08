"""
Feeds API router — stub endpoints for video feed management.

All endpoints return 501 until the Feeds subsystem is implemented.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.models.common import NotImplementedResponse

router = APIRouter()

STUB = NotImplementedResponse()
STUB_RESPONSE = {501: {"model": NotImplementedResponse}}


@router.get(
    "",
    summary="List feeds",
    description="List all available video feeds (raw and inference).",
    responses=STUB_RESPONSE,
)
async def list_feeds() -> JSONResponse:
    """List all available video feeds."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.get(
    "/{feed_id}",
    summary="Get feed details",
    description="Get details for a specific video feed.",
    responses=STUB_RESPONSE,
)
async def get_feed(feed_id: str) -> JSONResponse:
    """Get details for a specific feed."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.post(
    "",
    summary="Create feed",
    description="Register a new video feed source.",
    responses=STUB_RESPONSE,
)
async def create_feed() -> JSONResponse:
    """Create a new feed."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.put(
    "/{feed_id}",
    summary="Update feed",
    description="Update configuration for an existing feed.",
    responses=STUB_RESPONSE,
)
async def update_feed(feed_id: str) -> JSONResponse:
    """Update a feed."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.delete(
    "/{feed_id}",
    summary="Delete feed",
    description="Remove a video feed source.",
    responses=STUB_RESPONSE,
)
async def delete_feed(feed_id: str) -> JSONResponse:
    """Delete a feed."""
    return JSONResponse(status_code=501, content=STUB.model_dump())

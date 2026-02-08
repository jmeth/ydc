"""
Pydantic schemas for the Feeds REST API.

Defines request/response models for feed CRUD operations,
serializing internal FeedInfo dataclasses to JSON-friendly shapes.
"""

from pydantic import BaseModel, Field


class CreateFeedRequest(BaseModel):
    """Request body for creating a new video feed."""
    feed_type: str = Field(description="Feed source type: 'camera', 'rtsp', or 'file'")
    source: str = Field(description="Source identifier (camera index, URL, or file path)")
    name: str = Field(default="", description="Optional human-readable feed name")
    buffer_size: int = Field(default=30, ge=1, le=300, description="Max frames in the ring buffer")


class FeedInfoResponse(BaseModel):
    """Response body for a single feed's metadata."""
    feed_id: str = Field(description="Unique feed identifier")
    feed_type: str = Field(description="Feed source type")
    source: str = Field(description="Source identifier")
    name: str = Field(description="Human-readable feed name")
    status: str = Field(description="Current feed status")
    fps: float = Field(description="Current capture FPS")
    resolution: list[int] | None = Field(description="Frame [width, height] or null")
    frame_count: int = Field(description="Total frames captured")


class FeedListResponse(BaseModel):
    """Response body for listing all feeds."""
    feeds: list[FeedInfoResponse] = Field(description="List of active feeds")
    count: int = Field(description="Total number of feeds")

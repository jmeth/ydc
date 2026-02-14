"""
Pydantic API schemas for the capture subsystem.

Request/response models for capture session management: start/stop,
manual trigger, config updates, and status reporting. Used by the
capture API router for validation and serialization.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class StartCaptureRequest(BaseModel):
    """Request body for POST /api/capture/start."""
    feed_id: str = Field(description="Raw feed ID or inference-derived feed ID to capture from")
    dataset_name: str = Field(description="Target dataset name to save captures into")
    split: str = Field(default="train", description="Dataset split (train, val, test)")
    capture_interval: float = Field(
        default=2.0, gt=0, description="Seconds between automatic captures"
    )
    negative_ratio: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Ratio of negative (no-detection) frames to capture (inference mode only)",
    )
    confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to count a frame as positive (inference mode only)",
    )


class CaptureConfig(BaseModel):
    """Current capture configuration values."""
    capture_interval: float = Field(description="Seconds between automatic captures")
    negative_ratio: float = Field(description="Ratio of negative frames to capture")
    confidence_threshold: float = Field(description="Minimum detection confidence for positive frames")


class CaptureStats(BaseModel):
    """Statistics for the current or completed capture session."""
    total_captures: int = Field(default=0, description="Total frames captured")
    positive_captures: int = Field(default=0, description="Frames with detections (inference mode)")
    negative_captures: int = Field(default=0, description="Frames without detections (inference mode)")


class CaptureStatusResponse(BaseModel):
    """Response for GET /api/capture/status and POST /api/capture/start."""
    status: str = Field(description="Capture state: idle, running, or stopped")
    feed_id: Optional[str] = Field(default=None, description="Feed being captured from")
    dataset_name: Optional[str] = Field(default=None, description="Target dataset name")
    split: Optional[str] = Field(default=None, description="Target dataset split")
    mode: Optional[Literal["raw", "inference"]] = Field(
        default=None, description="Capture mode: raw (no annotations) or inference (with annotations)"
    )
    config: Optional[CaptureConfig] = Field(default=None, description="Current config")
    stats: Optional[CaptureStats] = Field(default=None, description="Session statistics")


class UpdateCaptureConfigRequest(BaseModel):
    """Request body for PUT /api/capture/config."""
    capture_interval: Optional[float] = Field(
        default=None, gt=0, description="New capture interval in seconds"
    )
    negative_ratio: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="New negative sampling ratio"
    )
    confidence_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="New confidence threshold"
    )


class ManualCaptureResponse(BaseModel):
    """Response for POST /api/capture/trigger."""
    filename: str = Field(description="Saved image filename")
    split: str = Field(description="Dataset split the image was saved to")
    dataset_name: str = Field(description="Dataset the image was saved to")
    num_detections: int = Field(default=0, description="Number of detections (inference mode only)")

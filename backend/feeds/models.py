"""
Feed subsystem data models.

Defines enums for feed types/statuses and dataclasses for frame data,
feed configuration, and feed metadata used throughout the feeds subsystem.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class FeedType(str, Enum):
    """Supported video feed source types."""
    CAMERA = "camera"       # USB/CSI camera via OpenCV
    RTSP = "rtsp"           # RTSP network stream (future)
    FILE = "file"           # Video file playback (future)


class FeedStatus(str, Enum):
    """Lifecycle states for a video feed."""
    CONNECTING = "connecting"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class Frame:
    """
    A single video frame captured from a feed.

    Attributes:
        data: Raw pixel data as numpy array (H, W, C in BGR format)
        timestamp: Time of capture (time.time())
        frame_number: Sequential frame counter from the feed
    """
    data: np.ndarray
    timestamp: float = field(default_factory=time.time)
    frame_number: int = 0


@dataclass
class Detection:
    """
    A single object detection result on a frame.

    Attributes:
        class_name: Detected object class label
        confidence: Detection confidence score (0.0-1.0)
        bbox: Bounding box as [x1, y1, x2, y2] in pixel coords
        class_id: Numeric class identifier
    """
    class_name: str
    confidence: float
    bbox: list[float]
    class_id: int = 0


@dataclass
class InferenceFrame:
    """
    A frame with associated inference results.

    Attributes:
        frame: The source video frame
        detections: List of detections found in the frame
        model_name: Name of the model that produced the detections
        inference_time_ms: Time taken for inference in milliseconds
    """
    frame: Frame
    detections: list[Detection] = field(default_factory=list)
    model_name: str = ""
    inference_time_ms: float = 0.0


@dataclass
class FeedConfig:
    """
    Configuration for creating a new video feed.

    Attributes:
        feed_type: Type of feed source (camera, rtsp, file)
        source: Source identifier (camera index, URL, or file path)
        name: Human-readable feed name
        buffer_size: Max frames to keep in the ring buffer
    """
    feed_type: FeedType
    source: str
    name: str = ""
    buffer_size: int = 30


@dataclass
class FeedInfo:
    """
    Runtime metadata about an active feed.

    Attributes:
        feed_id: Unique identifier for the feed
        config: The feed's creation config
        status: Current lifecycle status
        fps: Current frames per second (0.0 if not capturing)
        resolution: Frame dimensions as (width, height), or None
        frame_count: Total frames captured since creation
    """
    feed_id: str
    config: FeedConfig
    status: FeedStatus = FeedStatus.DISCONNECTED
    fps: float = 0.0
    resolution: tuple[int, int] | None = None
    frame_count: int = 0


@dataclass
class DerivedFeed:
    """
    A virtual feed that receives processed frames (e.g., with inference overlays).

    Derived feeds don't capture from a source — they receive frames
    pushed by other subsystems (like the inference engine).

    Attributes:
        feed_id: Unique identifier for the derived feed
        source_feed_id: The raw feed this derives from
        feed_type: Description of the processing applied
        buffer_size: Max frames in the ring buffer
    """
    feed_id: str
    source_feed_id: str
    feed_type: str = "inference"
    buffer_size: int = 30


# Type alias for frame data that can be either raw or inference-annotated
FrameType = Frame | InferenceFrame

"""
Capture manager — orchestrates frame capture from feeds into datasets.

Supports two modes:
- **Raw**: Subscribe to a raw feed, save frames at timed intervals or
  via manual trigger. No annotations are generated.
- **Inference**: Subscribe to an inference-derived feed, save frames with
  auto-generated YOLO annotations from detections. Supports positive/negative
  sampling based on confidence threshold and negative ratio.

The sync/async boundary is handled via asyncio.run_coroutine_threadsafe():
_on_frame() runs in the feed's capture thread and schedules async saves
on the event loop captured during start().

Usage:
    mgr = CaptureManager(feed_manager, dataset_manager, event_bus)
    await mgr.start("feed-123", "my-dataset", split="train")
    await mgr.stop()
"""

import asyncio
import logging
import random
import time
import uuid
from typing import Literal, Optional

from backend.core.events import (
    CAPTURE_FRAME_CAPTURED,
    CAPTURE_STARTED,
    CAPTURE_STOPPED,
    EventBus,
)
from backend.core.exceptions import ConflictError, NotFoundError
from backend.dataset.manager import DatasetManager
from backend.feeds.manager import FeedManager
from backend.feeds.models import Detection, Frame, InferenceFrame
from backend.persistence.models import Annotation

logger = logging.getLogger(__name__)


def detections_to_annotations(
    detections: list[Detection],
    image_width: int,
    image_height: int,
    confidence_threshold: float = 0.0,
) -> list[Annotation]:
    """
    Convert pixel-coordinate detections to normalized YOLO annotations.

    Filters by confidence_threshold and converts [x1,y1,x2,y2] pixel
    coords to normalized (x_center, y_center, width, height).

    Args:
        detections: Raw detection results with pixel-coord bboxes.
        image_width: Frame width in pixels.
        image_height: Frame height in pixels.
        confidence_threshold: Minimum confidence to include a detection.

    Returns:
        List of YOLO-format Annotation objects with normalized coords.
    """
    annotations: list[Annotation] = []
    if image_width <= 0 or image_height <= 0:
        return annotations

    for det in detections:
        if det.confidence < confidence_threshold:
            continue
        x1, y1, x2, y2 = det.bbox
        cx = ((x1 + x2) / 2.0) / image_width
        cy = ((y1 + y2) / 2.0) / image_height
        w = (x2 - x1) / image_width
        h = (y2 - y1) / image_height
        # Clamp to 0-1
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        annotations.append(Annotation(
            class_id=det.class_id, x=cx, y=cy, width=w, height=h,
        ))

    return annotations


class CaptureManager:
    """
    Orchestrates frame capture from feeds into datasets.

    Detects raw vs inference mode by checking is_derived_feed() on
    the FeedManager. Subscribes to the feed for frame-by-frame
    callbacks and saves images (+ labels in inference mode) to the
    target dataset at configured intervals.

    Args:
        feed_manager: FeedManager for subscribing to feeds.
        dataset_manager: DatasetManager for saving images and labels.
        event_bus: Optional EventBus for publishing capture events.
    """

    def __init__(
        self,
        feed_manager: FeedManager,
        dataset_manager: DatasetManager,
        event_bus: EventBus | None = None,
    ):
        self._feed_manager = feed_manager
        self._dataset_manager = dataset_manager
        self._event_bus = event_bus

        # Session state
        self._running: bool = False
        self._feed_id: str | None = None
        self._dataset_name: str | None = None
        self._split: str | None = None
        self._mode: Optional[Literal["raw", "inference"]] = None

        # Config
        self._capture_interval: float = 2.0
        self._negative_ratio: float = 0.2
        self._confidence_threshold: float = 0.3

        # Stats
        self._total_captures: int = 0
        self._positive_captures: int = 0
        self._negative_captures: int = 0

        # Timing
        self._last_capture_time: float = 0.0

        # Event loop reference for scheduling async work from sync callbacks
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(
        self,
        feed_id: str,
        dataset_name: str,
        split: str = "train",
        capture_interval: float = 2.0,
        negative_ratio: float = 0.2,
        confidence_threshold: float = 0.3,
    ) -> dict:
        """
        Start a capture session.

        Determines capture mode (raw vs inference) by checking whether
        feed_id is a derived feed. Subscribes to the feed for frame
        callbacks and stores session config.

        Args:
            feed_id: Raw feed ID or inference-derived feed ID.
            dataset_name: Target dataset to save captures into.
            split: Dataset split (train, val, test).
            capture_interval: Seconds between automatic captures.
            negative_ratio: Ratio of negative frames to capture (inference only).
            confidence_threshold: Min confidence for positive frames (inference only).

        Returns:
            Dict with session status info.

        Raises:
            ConflictError: If a capture session is already running.
            NotFoundError: If the feed or dataset doesn't exist.
        """
        if self._running:
            raise ConflictError("Capture session already running")

        # Verify feed exists (raw or derived)
        is_derived = self._feed_manager.is_derived_feed(feed_id)
        if not is_derived:
            # Check raw feed exists
            info = self._feed_manager.get_feed_info(feed_id)
            if info is None:
                raise NotFoundError("Feed", feed_id)

        # Verify dataset exists
        await self._dataset_manager.get_dataset(dataset_name)

        # Determine mode
        self._mode = "inference" if is_derived else "raw"
        self._feed_id = feed_id
        self._dataset_name = dataset_name
        self._split = split
        self._capture_interval = capture_interval
        self._negative_ratio = negative_ratio
        self._confidence_threshold = confidence_threshold

        # Reset stats
        self._total_captures = 0
        self._positive_captures = 0
        self._negative_captures = 0
        self._last_capture_time = 0.0

        # Capture event loop for async scheduling from sync callbacks
        self._loop = asyncio.get_running_loop()

        # Subscribe to feed
        subscribed = self._feed_manager.subscribe(feed_id, self._on_frame)
        if not subscribed:
            raise NotFoundError("Feed", feed_id)

        self._running = True

        await self._publish(CAPTURE_STARTED, {
            "feed_id": feed_id,
            "dataset_name": dataset_name,
            "split": split,
            "mode": self._mode,
        })

        logger.info(
            "Capture started: feed=%s dataset=%s split=%s mode=%s",
            feed_id, dataset_name, split, self._mode,
        )

        return self._build_status()

    async def stop(self) -> dict:
        """
        Stop the active capture session.

        Unsubscribes from the feed and publishes CAPTURE_STOPPED.

        Returns:
            Dict with final session stats.
        """
        if not self._running:
            return self._build_status()

        # Unsubscribe from feed
        if self._feed_id:
            self._feed_manager.unsubscribe(self._feed_id, self._on_frame)

        self._running = False

        stats = {
            "feed_id": self._feed_id,
            "dataset_name": self._dataset_name,
            "mode": self._mode,
            "total_captures": self._total_captures,
            "positive_captures": self._positive_captures,
            "negative_captures": self._negative_captures,
        }

        await self._publish(CAPTURE_STOPPED, stats)
        logger.info("Capture stopped: %d total captures", self._total_captures)

        return self._build_status()

    def _on_frame(self, feed_id: str, frame: Frame) -> None:
        """
        Sync callback from feed capture thread.

        Checks capture interval, applies sampling logic, and schedules
        async save on the event loop.

        Args:
            feed_id: The feed that produced this frame.
            frame: The raw or inference frame.
        """
        if not self._running or self._loop is None:
            return

        now = time.time()
        if now - self._last_capture_time < self._capture_interval:
            return

        # Determine whether to capture this frame
        if self._mode == "inference" and isinstance(frame, InferenceFrame):
            has_detections = self._has_positive_detections(frame)
            if not has_detections:
                # Negative sampling: capture at configured ratio
                if random.random() > self._negative_ratio:
                    return
        # Raw mode: always capture when interval is met

        self._last_capture_time = now

        # Schedule async save on the event loop
        asyncio.run_coroutine_threadsafe(
            self._save_capture(frame), self._loop
        )

    def _has_positive_detections(self, frame: InferenceFrame) -> bool:
        """
        Check whether an inference frame has detections above the threshold.

        Args:
            frame: InferenceFrame to check.

        Returns:
            True if at least one detection meets the confidence threshold.
        """
        for det in frame.detections:
            if det.confidence >= self._confidence_threshold:
                return True
        return False

    async def _save_capture(self, frame: Frame) -> None:
        """
        Save a captured frame (and optional labels) to the dataset.

        For inference mode: converts detections to YOLO annotations and
        saves labels alongside the image. For raw mode: saves image only.

        Args:
            frame: The frame to save (Frame for raw, InferenceFrame for inference).
        """
        if not self._running or self._dataset_name is None or self._split is None:
            return

        try:
            filename = f"capture_{uuid.uuid4().hex[:12]}.jpg"

            # Extract image data and detections
            if isinstance(frame, InferenceFrame):
                image_data = frame.frame.data
                detections = frame.detections
                has_detections = self._has_positive_detections(frame)
            else:
                image_data = frame.data
                detections = []
                has_detections = False

            # Save image
            image_info = await self._dataset_manager.add_image(
                self._dataset_name, self._split, filename, image_data,
            )

            # Save labels if in inference mode and there are detections
            num_detections = 0
            if self._mode == "inference" and detections:
                h, w = image_data.shape[:2]
                annotations = detections_to_annotations(
                    detections, w, h, self._confidence_threshold,
                )
                if annotations:
                    await self._dataset_manager.save_labels(
                        self._dataset_name, self._split, filename, annotations,
                    )
                num_detections = len(annotations)

            # Update stats
            self._total_captures += 1
            if self._mode == "inference":
                if has_detections:
                    self._positive_captures += 1
                else:
                    self._negative_captures += 1

            await self._publish(CAPTURE_FRAME_CAPTURED, {
                "feed_id": self._feed_id,
                "dataset_name": self._dataset_name,
                "split": self._split,
                "filename": filename,
                "mode": self._mode,
                "num_detections": num_detections,
                "total_captures": self._total_captures,
            })

        except Exception:
            logger.exception("Error saving capture")

    async def manual_trigger(self) -> dict:
        """
        Manually capture the latest frame regardless of interval timing.

        Gets the most recent frame from the feed buffer and saves it.

        Returns:
            Dict with capture details (filename, split, dataset, detections).

        Raises:
            ConflictError: If no capture session is running.
            NotFoundError: If no frame is available from the feed.
        """
        if not self._running or self._feed_id is None:
            raise ConflictError("No capture session is running")

        frame = self._feed_manager.get_frame(self._feed_id)
        if frame is None:
            raise NotFoundError("Frame", self._feed_id)

        filename = f"capture_{uuid.uuid4().hex[:12]}.jpg"

        # Extract image data and detections
        if isinstance(frame, InferenceFrame):
            image_data = frame.frame.data
            detections = frame.detections
        else:
            image_data = frame.data
            detections = []

        # Save image
        await self._dataset_manager.add_image(
            self._dataset_name, self._split, filename, image_data,
        )

        # Save labels if inference mode and detections present
        num_detections = 0
        if self._mode == "inference" and detections:
            h, w = image_data.shape[:2]
            annotations = detections_to_annotations(
                detections, w, h, self._confidence_threshold,
            )
            if annotations:
                await self._dataset_manager.save_labels(
                    self._dataset_name, self._split, filename, annotations,
                )
            num_detections = len(annotations)

        # Update stats
        self._total_captures += 1
        if self._mode == "inference":
            has_pos = any(d.confidence >= self._confidence_threshold for d in detections)
            if has_pos:
                self._positive_captures += 1
            else:
                self._negative_captures += 1

        await self._publish(CAPTURE_FRAME_CAPTURED, {
            "feed_id": self._feed_id,
            "dataset_name": self._dataset_name,
            "split": self._split,
            "filename": filename,
            "mode": self._mode,
            "num_detections": num_detections,
            "total_captures": self._total_captures,
        })

        return {
            "filename": filename,
            "split": self._split,
            "dataset_name": self._dataset_name,
            "num_detections": num_detections,
        }

    def update_config(
        self,
        capture_interval: float | None = None,
        negative_ratio: float | None = None,
        confidence_threshold: float | None = None,
    ) -> dict:
        """
        Update capture configuration on the fly.

        Only provided (non-None) fields are updated.

        Args:
            capture_interval: New seconds between captures.
            negative_ratio: New negative sampling ratio.
            confidence_threshold: New minimum confidence threshold.

        Returns:
            Dict with current config values.
        """
        if capture_interval is not None:
            self._capture_interval = capture_interval
        if negative_ratio is not None:
            self._negative_ratio = negative_ratio
        if confidence_threshold is not None:
            self._confidence_threshold = confidence_threshold

        return {
            "capture_interval": self._capture_interval,
            "negative_ratio": self._negative_ratio,
            "confidence_threshold": self._confidence_threshold,
        }

    def get_status(self) -> dict:
        """
        Get current capture session state and statistics.

        Returns:
            Dict with status, config, stats, and mode information.
        """
        return self._build_status()

    def _build_status(self) -> dict:
        """
        Build a status dict for the current session state.

        Returns:
            Dict suitable for CaptureStatusResponse serialization.
        """
        if not self._running:
            return {
                "status": "idle",
                "feed_id": None,
                "dataset_name": None,
                "split": None,
                "mode": None,
                "config": None,
                "stats": None,
            }

        return {
            "status": "running",
            "feed_id": self._feed_id,
            "dataset_name": self._dataset_name,
            "split": self._split,
            "mode": self._mode,
            "config": {
                "capture_interval": self._capture_interval,
                "negative_ratio": self._negative_ratio,
                "confidence_threshold": self._confidence_threshold,
            },
            "stats": {
                "total_captures": self._total_captures,
                "positive_captures": self._positive_captures,
                "negative_captures": self._negative_captures,
            },
        }

    async def _publish(self, event_type: str, data: dict) -> None:
        """
        Publish an event via the EventBus if available.

        Args:
            event_type: Event type constant.
            data: Event payload.
        """
        if self._event_bus is not None:
            await self._event_bus.publish(event_type, data)

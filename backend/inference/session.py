"""
Inference session that subscribes to a feed and runs YOLO detection.

InferenceSession registers as a subscriber on a source feed via
FeedManager. Each incoming frame is passed through the loaded YOLO
model, and the resulting detections are packaged into an InferenceFrame
and pushed to a derived feed buffer.
"""

import logging
import time

import numpy as np

from backend.feeds.manager import FeedManager
from backend.feeds.models import Detection, Frame, InferenceFrame
from backend.inference.models import LoadedModel

logger = logging.getLogger(__name__)


class InferenceSession:
    """
    Runs real-time detection on frames from a source feed.

    Subscribes to the source feed via FeedManager. The on_frame callback
    runs inline in the capture thread — YOLO inference is synchronous and
    fast enough per-frame for typical camera FPS.

    Args:
        output_feed_id: Derived feed ID where InferenceFrames are pushed.
        source_feed_id: The raw feed to subscribe to.
        model: A LoadedModel instance for running detection.
        feed_manager: The FeedManager providing subscribe/push APIs.
        confidence_threshold: Minimum confidence to keep a detection (0.0-1.0).
    """

    def __init__(
        self,
        output_feed_id: str,
        source_feed_id: str,
        model: LoadedModel,
        feed_manager: FeedManager,
        confidence_threshold: float = 0.3,
    ):
        self.output_feed_id = output_feed_id
        self.source_feed_id = source_feed_id
        self.model = model
        self._feed_manager = feed_manager
        self.confidence_threshold = confidence_threshold

        # Stats
        self._frames_processed: int = 0
        self._total_inference_ms: float = 0.0
        self._last_inference_ms: float = 0.0

    @property
    def frames_processed(self) -> int:
        """Total frames processed by this session."""
        return self._frames_processed

    @property
    def avg_inference_ms(self) -> float:
        """Average inference time in milliseconds across all processed frames."""
        if self._frames_processed == 0:
            return 0.0
        return self._total_inference_ms / self._frames_processed

    @property
    def last_inference_ms(self) -> float:
        """Inference time for the most recently processed frame."""
        return self._last_inference_ms

    def on_frame(self, feed_id: str, frame: Frame) -> None:
        """
        Frame callback matching the FeedManager FrameCallback signature.

        Runs detection on the frame, filters by confidence threshold,
        packages results into an InferenceFrame, and pushes to the
        derived feed buffer.

        Args:
            feed_id: Source feed identifier (from FeedManager).
            frame: The captured video frame.
        """
        start = time.perf_counter()

        try:
            results = self.model.model(frame.data, verbose=False)
            detections = self._parse_results(results)
        except Exception:
            logger.exception("Inference error on feed %s", feed_id[:8])
            return

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        inference_frame = InferenceFrame(
            frame=frame,
            detections=detections,
            model_name=self.model.model_name,
            inference_time_ms=elapsed_ms,
        )

        self._feed_manager.push_derived_frame(self.output_feed_id, inference_frame)

        # Update stats
        self._frames_processed += 1
        self._total_inference_ms += elapsed_ms
        self._last_inference_ms = elapsed_ms

    def _parse_results(self, results: list) -> list[Detection]:
        """
        Convert ultralytics Results objects to Detection dataclasses.

        Filters detections below the confidence threshold. Handles the
        standard ultralytics results format where results[0].boxes
        contains the detection tensor data.

        Args:
            results: List of ultralytics Results objects (one per image).

        Returns:
            List of Detection objects that passed the confidence threshold.
        """
        detections: list[Detection] = []

        if not results:
            return detections

        # ultralytics returns a list; we only send one image so use results[0]
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Get class names from the loaded model's class mapping
        class_names = self.model.classes

        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < self.confidence_threshold:
                continue

            class_id = int(boxes.cls[i])
            bbox = boxes.xyxy[i].tolist()
            class_name = class_names.get(class_id, f"class_{class_id}")

            detections.append(Detection(
                class_name=class_name,
                confidence=conf,
                bbox=bbox,
                class_id=class_id,
            ))

        return detections

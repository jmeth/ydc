"""
RTSP feed implementation using OpenCV VideoCapture.

Supports RTSP and RTSPS streams identified by URL.
"""

import logging

import cv2
import numpy as np

from backend.feeds.base import BaseFeed

logger = logging.getLogger(__name__)


class RTSPFeed(BaseFeed):
    """
    Video feed from an RTSP network stream via OpenCV.

    The source string must be a valid RTSP or RTSPS URL
    (e.g., "rtsp://192.168.1.10:554/stream").

    Args:
        source: RTSP stream URL (must start with rtsp:// or rtsps://)
    """

    def __init__(self, source: str):
        super().__init__(source)
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 0.0
        self._resolution: tuple[int, int] | None = None

    def connect(self) -> bool:
        """
        Open the RTSP stream and read initial properties.

        Validates the source URL starts with rtsp:// or rtsps://, then
        opens a VideoCapture. FPS falls back to 25.0 (common RTSP default)
        when the stream reports 0.

        Returns:
            True if the stream was opened successfully, False otherwise.
        """
        url = self._source.lower()
        if not (url.startswith("rtsp://") or url.startswith("rtsps://")):
            logger.error(
                "Invalid RTSP URL '%s' — must start with rtsp:// or rtsps://",
                self._source,
            )
            return False

        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            logger.error("Failed to open RTSP stream at '%s'", self._source)
            self._cap = None
            return False

        # Read native properties from the stream
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > 0 and height > 0:
            self._resolution = (width, height)

        logger.info(
            "RTSP stream '%s' opened: %s @ %.1f fps",
            self._source,
            self._resolution,
            self._fps,
        )
        return True

    def disconnect(self) -> None:
        """Release the RTSP stream."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._resolution = None
            logger.info("RTSP stream '%s' disconnected", self._source)

    def read(self) -> np.ndarray | None:
        """
        Read the next frame from the RTSP stream.

        Returns:
            BGR numpy array (H, W, C), or None if the stream is not
            opened or a frame could not be read.
        """
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    @property
    def fps(self) -> float:
        """Native FPS reported by the stream, or 25.0 as fallback."""
        return self._fps

    @property
    def resolution(self) -> tuple[int, int] | None:
        """(width, height) of the stream, or None if not connected."""
        return self._resolution

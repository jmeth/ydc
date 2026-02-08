"""
Camera feed implementation using OpenCV VideoCapture.

Supports USB and CSI cameras identified by integer device index.
"""

import logging

import cv2
import numpy as np

from backend.feeds.base import BaseFeed

logger = logging.getLogger(__name__)


class CameraFeed(BaseFeed):
    """
    Video feed from a local camera via OpenCV.

    The source string is converted to an integer device index
    (e.g., "0" for /dev/video0).

    Args:
        source: Camera device index as a string (e.g., "0", "1")
    """

    def __init__(self, source: str):
        super().__init__(source)
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 0.0
        self._resolution: tuple[int, int] | None = None

    def connect(self) -> bool:
        """
        Open the camera device and read initial properties.

        Returns:
            True if the camera was opened successfully, False otherwise.
        """
        try:
            device_index = int(self._source)
        except ValueError:
            logger.error("Invalid camera index '%s' — must be an integer", self._source)
            return False

        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            logger.error("Failed to open camera at index %d", device_index)
            self._cap = None
            return False

        # Read native properties from the camera
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > 0 and height > 0:
            self._resolution = (width, height)

        logger.info(
            "Camera %d opened: %s @ %.1f fps",
            device_index,
            self._resolution,
            self._fps,
        )
        return True

    def disconnect(self) -> None:
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._resolution = None
            logger.info("Camera %s disconnected", self._source)

    def read(self) -> np.ndarray | None:
        """
        Read the next frame from the camera.

        Returns:
            BGR numpy array (H, W, C), or None if the camera is not
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
        """Native FPS reported by the camera, or 30.0 as fallback."""
        return self._fps

    @property
    def resolution(self) -> tuple[int, int] | None:
        """(width, height) of the camera, or None if not connected."""
        return self._resolution

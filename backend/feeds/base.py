"""
Abstract base class for video feed sources.

All feed implementations (camera, RTSP, file) must implement this
interface to be usable by the FeedManager.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseFeed(ABC):
    """
    Abstract interface for a video feed source.

    Subclasses must implement connect, disconnect, read, and properties
    for fps and resolution. The FeedManager uses this interface to manage
    feed lifecycle and frame capture.

    Args:
        source: Source identifier (camera index, URL, file path, etc.)
    """

    def __init__(self, source: str):
        self._source = source

    @property
    def source(self) -> str:
        """The feed source identifier."""
        return self._source

    @abstractmethod
    def connect(self) -> bool:
        """
        Open the feed source and prepare for reading frames.

        Returns:
            True if the connection was established successfully.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Release the feed source and free resources."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """
        Read the next frame from the feed.

        Returns:
            A numpy array (H, W, C) in BGR format, or None if no frame
            is available (e.g., camera not ready, end of file).
        """

    @property
    @abstractmethod
    def fps(self) -> float:
        """
        The native frame rate of the feed source.

        Returns:
            Frames per second, or 0.0 if unknown.
        """

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int] | None:
        """
        The frame resolution of the feed source.

        Returns:
            (width, height) tuple, or None if not connected.
        """

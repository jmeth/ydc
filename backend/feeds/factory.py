"""
Feed factory for creating feed instances by type.

Centralizes feed construction so the FeedManager doesn't need to
know about specific feed implementations.
"""

import logging

from backend.feeds.base import BaseFeed
from backend.feeds.camera import CameraFeed
from backend.feeds.models import FeedType
from backend.feeds.rtsp import RTSPFeed

logger = logging.getLogger(__name__)


class FeedFactory:
    """
    Creates BaseFeed instances based on FeedType.

    Currently supports CAMERA and RTSP feeds. FILE type will be
    added in a future phase.
    """

    @staticmethod
    def create(feed_type: FeedType, source: str) -> BaseFeed:
        """
        Create a feed instance for the given type and source.

        Args:
            feed_type: The type of feed to create
            source: Source identifier (camera index, URL, etc.)

        Returns:
            A BaseFeed implementation matching the requested type.

        Raises:
            ValueError: If the feed_type is not supported.
        """
        if feed_type == FeedType.CAMERA:
            return CameraFeed(source)
        elif feed_type == FeedType.RTSP:
            return RTSPFeed(source)
        elif feed_type == FeedType.FILE:
            raise ValueError(f"Feed type '{feed_type.value}' is not yet supported")
        else:
            raise ValueError(f"Unknown feed type: '{feed_type}'")

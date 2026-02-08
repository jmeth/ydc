"""
Feeds subsystem — video feed management, capture, and streaming.

Public API:
    FeedManager  — feed lifecycle, capture threads, subscriptions
    FeedStreamer  — async WebSocket frame broadcasting
    FeedFactory   — creates BaseFeed instances by type
    BaseFeed      — abstract feed interface
    RingBuffer    — thread-safe circular frame buffer
    FeedConfig, FeedInfo, FeedType, FeedStatus — data models
"""

from backend.feeds.base import BaseFeed
from backend.feeds.buffer import RingBuffer
from backend.feeds.factory import FeedFactory
from backend.feeds.manager import FeedManager
from backend.feeds.models import (
    DerivedFeed,
    FeedConfig,
    FeedInfo,
    FeedStatus,
    FeedType,
    Frame,
)
from backend.feeds.streaming import FeedStreamer

__all__ = [
    "BaseFeed",
    "DerivedFeed",
    "FeedConfig",
    "FeedFactory",
    "FeedInfo",
    "FeedManager",
    "FeedStatus",
    "FeedStreamer",
    "FeedType",
    "Frame",
    "RingBuffer",
]

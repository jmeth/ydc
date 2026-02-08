"""
In-process event bus for cross-subsystem communication.

Supports both sync and async callbacks with error isolation so one
failing subscriber doesn't break others.

Usage:
    bus = EventBus()
    await bus.start()
    bus.subscribe("training.completed", my_handler)
    await bus.publish("training.completed", {"model": "yolov8n"})
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Event type constants
SCAN_STARTED = "scan.started"
SCAN_STOPPED = "scan.stopped"
SCAN_CAPTURE = "scan.capture"
SCAN_PAUSED = "scan.paused"

TRAINING_STARTED = "training.started"
TRAINING_PROGRESS = "training.progress"
TRAINING_COMPLETED = "training.completed"
TRAINING_ERROR = "training.error"

RESOURCE_WARNING = "resource.warning"
RESOURCE_CRITICAL = "resource.critical"

FEED_ADDED = "feed.added"
FEED_REMOVED = "feed.removed"
FEED_ERROR = "feed.error"

INFERENCE_STARTED = "inference.started"
INFERENCE_STOPPED = "inference.stopped"
INFERENCE_ERROR = "inference.error"

DATASET_CREATED = "dataset.created"
DATASET_DELETED = "dataset.deleted"
DATASET_IMAGE_ADDED = "dataset.image_added"
DATASET_IMAGE_DELETED = "dataset.image_deleted"


class EventBus:
    """
    Simple pub/sub event bus for subsystem communication.

    Callbacks are invoked in order of subscription. Async callbacks are
    awaited; sync callbacks are called directly. Errors in one callback
    are logged but don't prevent other callbacks from executing.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._running: bool = False

    async def start(self) -> None:
        """Start the event bus."""
        self._running = True
        logger.info("EventBus started")

    async def stop(self) -> None:
        """Stop the event bus and clear all subscriptions."""
        self._running = False
        self._subscribers.clear()
        logger.info("EventBus stopped")

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback for an event type.

        Args:
            event_type: Event name string (e.g. "training.completed")
            callback: Sync or async function accepting a data dict
        """
        self._subscribers[event_type].append(callback)
        logger.debug("Subscribed to '%s': %s", event_type, callback.__name__)

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Remove a callback from an event type.

        Args:
            event_type: Event name string
            callback: The previously registered callback
        """
        try:
            self._subscribers[event_type].remove(callback)
            logger.debug("Unsubscribed from '%s': %s", event_type, callback.__name__)
        except ValueError:
            logger.warning(
                "Callback %s not found in subscribers for '%s'",
                callback.__name__,
                event_type,
            )

    async def publish(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """
        Publish an event to all subscribers of that type.

        Each subscriber is called with error isolation — a failing callback
        is logged but doesn't prevent others from executing.

        Args:
            event_type: Event name string
            data: Optional event payload dict
        """
        if not self._running:
            logger.warning("EventBus not running, dropping event '%s'", event_type)
            return

        if data is None:
            data = {}

        subscribers = self._subscribers.get(event_type, [])
        logger.debug("Publishing '%s' to %d subscribers", event_type, len(subscribers))

        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception:
                logger.exception(
                    "Error in subscriber %s for event '%s'",
                    callback.__name__,
                    event_type,
                )


# Module-level singleton
event_bus = EventBus()

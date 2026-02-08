"""
Inference manager for session lifecycle, model loading, and prompt updates.

InferenceManager is the top-level coordinator for the inference subsystem.
It starts/stops InferenceSessions, manages model loading via ModelLoader,
and publishes events for cross-subsystem notifications.
"""

import asyncio
import logging
from uuid import uuid4

from backend.core.events import (
    EventBus,
    INFERENCE_STARTED,
    INFERENCE_STOPPED,
    INFERENCE_ERROR,
)
from backend.core.exceptions import ConflictError, NotFoundError
from backend.feeds.manager import FeedManager
from backend.feeds.models import DerivedFeed
from backend.inference.loader import ModelLoader
from backend.inference.models import ModelType
from backend.inference.session import InferenceSession

logger = logging.getLogger(__name__)


class InferenceManager:
    """
    Manages inference session lifecycle: start, stop, prompt update, model swap.

    Enforces a single session per source feed (MVP constraint). Publishes
    INFERENCE_STARTED/STOPPED/ERROR events to the EventBus for notification
    integration.

    Args:
        feed_manager: FeedManager for feed lookups and derived feed registration.
        notification_manager: Optional NotificationManager (unused directly;
            notifications are driven via EventBus subscriptions).
        event_bus: Optional EventBus for publishing lifecycle events.
    """

    def __init__(
        self,
        feed_manager: FeedManager,
        notification_manager=None,
        event_bus: EventBus | None = None,
    ):
        self._feed_manager = feed_manager
        self._notification_manager = notification_manager
        self._event_bus = event_bus
        self._loader = ModelLoader()

        # output_feed_id -> InferenceSession
        self._sessions: dict[str, InferenceSession] = {}
        # source_feed_id -> output_feed_id (enforce single session per source)
        self._source_to_output: dict[str, str] = {}

    def start_inference(
        self,
        source_feed_id: str,
        model_name: str,
        model_type: ModelType,
        prompts: list[str] | None = None,
        confidence_threshold: float = 0.3,
    ) -> str:
        """
        Start an inference session on a source feed.

        Creates a derived feed, loads the model, subscribes the session
        callback to the source feed, and publishes an INFERENCE_STARTED event.

        Args:
            source_feed_id: The raw feed to run detection on.
            model_name: Model file or identifier string.
            model_type: YOLO-World or fine-tuned.
            prompts: Text prompts for YOLO-World models.
            confidence_threshold: Minimum detection confidence (0.0-1.0).

        Returns:
            The output_feed_id for the new inference derived feed.

        Raises:
            NotFoundError: If the source feed doesn't exist.
            ConflictError: If inference is already running on this source feed.
        """
        # Verify source feed exists
        feed_info = self._feed_manager.get_feed_info(source_feed_id)
        if feed_info is None:
            raise NotFoundError("Feed", source_feed_id)

        # Enforce single session per source feed
        if source_feed_id in self._source_to_output:
            existing_id = self._source_to_output[source_feed_id]
            raise ConflictError(
                f"Inference already running on feed '{source_feed_id[:8]}'",
                detail=f"Active session output: {existing_id}",
            )

        # Generate output feed ID and register derived feed
        output_feed_id = f"inference_{source_feed_id[:8]}_{uuid4().hex[:8]}"

        derived = DerivedFeed(
            feed_id=output_feed_id,
            source_feed_id=source_feed_id,
            feed_type="inference",
        )
        if not self._feed_manager.register_derived_feed(derived):
            raise NotFoundError("Feed", source_feed_id)

        # Load model
        loaded_model = self._loader.load(model_name, model_type, prompts)

        # Create session and subscribe to source feed
        session = InferenceSession(
            output_feed_id=output_feed_id,
            source_feed_id=source_feed_id,
            model=loaded_model,
            feed_manager=self._feed_manager,
            confidence_threshold=confidence_threshold,
        )

        self._feed_manager.subscribe(source_feed_id, session.on_frame)

        # Track session
        self._sessions[output_feed_id] = session
        self._source_to_output[source_feed_id] = output_feed_id

        # Publish event
        self._publish_event(INFERENCE_STARTED, {
            "output_feed_id": output_feed_id,
            "source_feed_id": source_feed_id,
            "model_name": model_name,
            "model_type": model_type.value,
        })

        logger.info(
            "Inference started: %s -> %s (model=%s)",
            source_feed_id[:8], output_feed_id[:16], model_name,
        )
        return output_feed_id

    def stop_inference(self, output_feed_id: str) -> bool:
        """
        Stop an inference session and clean up its resources.

        Unsubscribes the session callback, unregisters the derived feed,
        and publishes an INFERENCE_STOPPED event.

        Args:
            output_feed_id: The inference output feed to stop.

        Returns:
            True if the session was found and stopped, False if not found.
        """
        session = self._sessions.pop(output_feed_id, None)
        if session is None:
            return False

        # Unsubscribe from source feed
        self._feed_manager.unsubscribe(session.source_feed_id, session.on_frame)

        # Unregister derived feed
        self._feed_manager.unregister_derived_feed(output_feed_id)

        # Remove source mapping
        self._source_to_output.pop(session.source_feed_id, None)

        self._publish_event(INFERENCE_STOPPED, {
            "output_feed_id": output_feed_id,
            "source_feed_id": session.source_feed_id,
            "model_name": session.model.model_name,
            "frames_processed": session.frames_processed,
        })

        logger.info("Inference stopped: %s", output_feed_id[:16])
        return True

    def stop_all(self) -> None:
        """Stop all active inference sessions. Called during app shutdown."""
        output_ids = list(self._sessions.keys())
        for output_id in output_ids:
            self.stop_inference(output_id)
        logger.info("All inference sessions stopped")

    def get_session_status(self, output_feed_id: str) -> dict | None:
        """
        Get runtime status for a specific inference session.

        Args:
            output_feed_id: The inference output feed to query.

        Returns:
            Dict with session metadata and stats, or None if not found.
        """
        session = self._sessions.get(output_feed_id)
        if session is None:
            return None

        return {
            "output_feed_id": output_feed_id,
            "source_feed_id": session.source_feed_id,
            "model_name": session.model.model_name,
            "model_type": session.model.model_type.value,
            "prompts": session.model.prompts,
            "confidence_threshold": session.confidence_threshold,
            "frames_processed": session.frames_processed,
            "avg_inference_ms": round(session.avg_inference_ms, 2),
            "last_inference_ms": round(session.last_inference_ms, 2),
            "status": "running",
        }

    def get_all_sessions(self) -> list[dict]:
        """
        Get status for all active inference sessions.

        Returns:
            List of session status dicts (same format as get_session_status).
        """
        return [
            self.get_session_status(oid)
            for oid in self._sessions
        ]

    def update_prompts(self, output_feed_id: str, prompts: list[str]) -> bool:
        """
        Update YOLO-World prompts by restarting the session with new prompts.

        Stops the existing session and starts a new one on the same source
        feed with the updated prompts. The output_feed_id changes.

        Args:
            output_feed_id: The current inference output feed.
            prompts: New text prompts for YOLO-World detection.

        Returns:
            True if the session was found and updated, False if not found.
        """
        session = self._sessions.get(output_feed_id)
        if session is None:
            return False

        source_feed_id = session.source_feed_id
        model_name = session.model.model_name
        model_type = session.model.model_type
        threshold = session.confidence_threshold

        self.stop_inference(output_feed_id)

        self.start_inference(
            source_feed_id=source_feed_id,
            model_name=model_name,
            model_type=model_type,
            prompts=prompts,
            confidence_threshold=threshold,
        )
        return True

    def switch_model(
        self,
        output_feed_id: str,
        model_name: str,
        model_type: ModelType,
        prompts: list[str] | None = None,
    ) -> str | None:
        """
        Switch the model on an existing session by stopping and restarting.

        Args:
            output_feed_id: The current inference output feed.
            model_name: New model file or identifier.
            model_type: New model type.
            prompts: New prompts (for YOLO-World models).

        Returns:
            The new output_feed_id, or None if the original session was not found.
        """
        session = self._sessions.get(output_feed_id)
        if session is None:
            return None

        source_feed_id = session.source_feed_id
        threshold = session.confidence_threshold

        self.stop_inference(output_feed_id)

        new_output_id = self.start_inference(
            source_feed_id=source_feed_id,
            model_name=model_name,
            model_type=model_type,
            prompts=prompts,
            confidence_threshold=threshold,
        )
        return new_output_id

    def _publish_event(self, event_type: str, data: dict) -> None:
        """
        Publish an event to the EventBus if available.

        Uses fire-and-forget scheduling since EventBus.publish is async
        but InferenceManager methods are synchronous.

        Args:
            event_type: Event type constant (e.g. INFERENCE_STARTED).
            data: Event payload dict.
        """
        if self._event_bus is None:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._event_bus.publish(event_type, data))
        except RuntimeError:
            # No running event loop — skip event publishing (e.g. in sync tests)
            logger.debug("No event loop for publishing %s", event_type)

"""
Notifications API router — CRUD endpoints for notification management.

Uses a NotificationManager injected via set_notification_manager() during
app startup to handle history queries, read/dismiss, and clear operations.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query, Response
from fastapi.responses import JSONResponse

from backend.models.common import ErrorResponse
from backend.models.notifications import NotificationListResponse, NotificationResponse
from backend.notifications.manager import NotificationManager
from backend.notifications.models import NotificationCategory

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level NotificationManager reference, set during lifespan startup
_notification_manager: NotificationManager | None = None


def set_notification_manager(manager: NotificationManager) -> None:
    """
    Inject the NotificationManager instance at startup.

    Called from the app lifespan to avoid circular imports and
    module-level singleton issues.

    Args:
        manager: The initialized NotificationManager.
    """
    global _notification_manager
    _notification_manager = manager


def _get_manager() -> NotificationManager:
    """
    Return the injected NotificationManager or raise if not set.

    Returns:
        The active NotificationManager instance.

    Raises:
        RuntimeError: If the manager hasn't been injected yet.
    """
    if _notification_manager is None:
        raise RuntimeError("NotificationManager not initialized")
    return _notification_manager


@router.get(
    "",
    summary="List notifications",
    description="Get notification history with optional filters.",
    response_model=NotificationListResponse,
    responses={200: {"model": NotificationListResponse}},
)
async def list_notifications(
    category: Optional[str] = Query(None, description="Filter by category (system, scan, training, dataset, inference)"),
    unread_only: bool = Query(False, description="Only return unread notifications"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of notifications to return"),
) -> NotificationListResponse:
    """
    Get notification history with optional filters.

    Args:
        category: Filter by source category name (optional).
        unread_only: If true, only return unread notifications.
        limit: Maximum number to return (1-100, default 50).

    Returns:
        NotificationListResponse with matching notifications.
    """
    mgr = _get_manager()

    # Convert category string to enum if provided
    cat_enum = None
    if category is not None:
        try:
            cat_enum = NotificationCategory(category)
        except ValueError:
            return JSONResponse(
                status_code=422,
                content={"error": "Invalid category", "detail": f"Unknown category: {category}"},
            )

    notifications = mgr.get_history(category=cat_enum, unread_only=unread_only, limit=limit)

    items = [
        NotificationResponse(
            id=n.id,
            type=n.type.value,
            level=n.level.value,
            category=n.category.value,
            title=n.title,
            message=n.message,
            timestamp=n.timestamp,
            read=n.read,
            dismissed=n.dismissed,
            data=n.data,
        )
        for n in notifications
    ]
    return NotificationListResponse(notifications=items, count=len(items))


@router.post(
    "/{notification_id}/read",
    summary="Mark as read",
    description="Mark a specific notification as read.",
    response_model=NotificationResponse,
    responses={200: {"model": NotificationResponse}, 404: {"model": ErrorResponse}},
)
async def mark_read(notification_id: str) -> NotificationResponse | JSONResponse:
    """
    Mark a notification as read by ID.

    Args:
        notification_id: The notification ID to mark as read.

    Returns:
        The updated notification, or 404 if not found.
    """
    mgr = _get_manager()
    if not mgr.mark_read(notification_id):
        return JSONResponse(
            status_code=404,
            content={"error": "Not found", "detail": f"Notification {notification_id} not found"},
        )

    # Find and return the updated notification
    for n in mgr.get_history(limit=100):
        if n.id == notification_id:
            return NotificationResponse(
                id=n.id,
                type=n.type.value,
                level=n.level.value,
                category=n.category.value,
                title=n.title,
                message=n.message,
                timestamp=n.timestamp,
                read=n.read,
                dismissed=n.dismissed,
                data=n.data,
            )


@router.post(
    "/{notification_id}/dismiss",
    summary="Dismiss notification",
    description="Dismiss a specific notification.",
    response_model=NotificationResponse,
    responses={200: {"model": NotificationResponse}, 404: {"model": ErrorResponse}},
)
async def dismiss_notification(notification_id: str) -> NotificationResponse | JSONResponse:
    """
    Dismiss a notification by ID.

    Args:
        notification_id: The notification ID to dismiss.

    Returns:
        The updated notification, or 404 if not found.
    """
    mgr = _get_manager()
    if not mgr.dismiss(notification_id):
        return JSONResponse(
            status_code=404,
            content={"error": "Not found", "detail": f"Notification {notification_id} not found"},
        )

    # Find and return the updated notification
    for n in mgr.get_history(limit=100):
        if n.id == notification_id:
            return NotificationResponse(
                id=n.id,
                type=n.type.value,
                level=n.level.value,
                category=n.category.value,
                title=n.title,
                message=n.message,
                timestamp=n.timestamp,
                read=n.read,
                dismissed=n.dismissed,
                data=n.data,
            )


@router.delete(
    "",
    summary="Clear notifications",
    description="Clear all notifications, optionally filtered by category.",
    status_code=204,
    responses={204: {"description": "Notifications cleared"}},
)
async def clear_notifications(
    category: Optional[str] = Query(None, description="Clear only this category"),
) -> Response:
    """
    Clear notification history.

    Args:
        category: If provided, only clear this category. Otherwise clear all.

    Returns:
        204 No Content on success.
    """
    mgr = _get_manager()

    cat_enum = None
    if category is not None:
        try:
            cat_enum = NotificationCategory(category)
        except ValueError:
            return JSONResponse(
                status_code=422,
                content={"error": "Invalid category", "detail": f"Unknown category: {category}"},
            )

    mgr.clear_all(category=cat_enum)
    return Response(status_code=204)

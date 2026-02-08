"""
Notifications API router — stub endpoints for notification management.

All endpoints return 501 until the Notifications subsystem is implemented.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.models.common import NotImplementedResponse

router = APIRouter()

STUB = NotImplementedResponse()
STUB_RESPONSE = {501: {"model": NotImplementedResponse}}


@router.get(
    "",
    summary="List notifications",
    description="Get notification history with optional filters.",
    responses=STUB_RESPONSE,
)
async def list_notifications() -> JSONResponse:
    """Get notification history."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.post(
    "/{notification_id}/read",
    summary="Mark as read",
    description="Mark a specific notification as read.",
    responses=STUB_RESPONSE,
)
async def mark_read(notification_id: str) -> JSONResponse:
    """Mark notification as read."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.post(
    "/{notification_id}/dismiss",
    summary="Dismiss notification",
    description="Dismiss a specific notification.",
    responses=STUB_RESPONSE,
)
async def dismiss_notification(notification_id: str) -> JSONResponse:
    """Dismiss a notification."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.delete(
    "",
    summary="Clear notifications",
    description="Clear all notifications.",
    responses=STUB_RESPONSE,
)
async def clear_notifications() -> JSONResponse:
    """Clear all notifications."""
    return JSONResponse(status_code=501, content=STUB.model_dump())

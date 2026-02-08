"""
Pydantic schemas for notification API responses.

Maps internal Notification dataclass fields to API-safe JSON shapes.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class NotificationResponse(BaseModel):
    """Single notification in API responses."""
    id: str = Field(description="Unique notification identifier")
    type: str = Field(description="Presentation type (toast, banner, alert, status)")
    level: str = Field(description="Severity level (info, success, warning, error)")
    category: str = Field(description="Source subsystem category")
    title: str = Field(description="Short summary line")
    message: str = Field(description="Descriptive text")
    timestamp: datetime = Field(description="UTC creation time")
    read: bool = Field(description="Whether the notification has been read")
    dismissed: bool = Field(description="Whether the notification has been dismissed")
    data: dict[str, Any] | None = Field(default=None, description="Optional extra payload")


class NotificationListResponse(BaseModel):
    """Response body for GET /api/notifications."""
    notifications: list[NotificationResponse] = Field(description="List of notifications")
    count: int = Field(description="Number of notifications returned")

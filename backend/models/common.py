"""
Common Pydantic response models used across all API endpoints.

These provide consistent response shapes for health checks,
errors, and stub (not-implemented) endpoints.
"""

from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    """Standard health/status response."""
    status: str = Field(description="Service status string (e.g. 'ok', 'degraded')")
    version: str = Field(default="2.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Standard error response body."""
    error: str = Field(description="Short error description")
    detail: str | None = Field(default=None, description="Extended error details")


class NotImplementedResponse(BaseModel):
    """Response for stub endpoints that are not yet implemented."""
    error: str = Field(default="Not implemented")
    detail: str = Field(default="This endpoint is a stub for future implementation")

"""
Application-specific exceptions.

Custom exceptions for consistent error handling across subsystems.
Each maps to an appropriate HTTP status code.
"""


class YDCError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str = "An error occurred", detail: str | None = None):
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class NotFoundError(YDCError):
    """Resource not found. Maps to HTTP 404."""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            message=f"{resource} '{identifier}' not found",
            detail=f"No {resource} exists with identifier '{identifier}'",
        )


class ConflictError(YDCError):
    """Resource conflict (e.g. duplicate name). Maps to HTTP 409."""

    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message=message, detail=detail)


class NotImplementedError(YDCError):
    """Feature not yet implemented. Maps to HTTP 501."""

    def __init__(self, feature: str = "This feature"):
        super().__init__(
            message=f"{feature} is not yet implemented",
            detail="This endpoint is a stub for future implementation",
        )


class ValidationError(YDCError):
    """Input validation failure. Maps to HTTP 422."""

    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message=message, detail=detail)


class SubsystemError(YDCError):
    """Internal subsystem failure. Maps to HTTP 500."""

    def __init__(self, subsystem: str, message: str, detail: str | None = None):
        super().__init__(
            message=f"{subsystem}: {message}",
            detail=detail,
        )

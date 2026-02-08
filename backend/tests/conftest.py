"""
Shared test fixtures for backend unit tests.

Provides an async HTTP client (via httpx) for testing FastAPI
endpoints, a fresh EventBus instance per test, and a
NotificationManager wired to a mock ConnectionManager.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient

from backend.core.events import EventBus
from backend.main import app


@pytest.fixture
def anyio_backend():
    """Use asyncio as the async backend for pytest-asyncio."""
    return "asyncio"


@pytest.fixture
async def client():
    """
    Async HTTP test client for the FastAPI app.

    Yields an httpx.AsyncClient wired to the ASGI app so tests
    don't need a running server.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def event_bus():
    """Fresh EventBus instance for isolated event tests."""
    return EventBus()


@pytest.fixture
def notification_manager():
    """
    Fresh NotificationManager injected into the notifications API router.

    Uses a mock ConnectionManager so tests don't need real WebSocket
    connections. Cleans up by resetting the module-level reference.
    """
    from backend.api.notifications import set_notification_manager
    from backend.notifications.manager import NotificationManager

    cm = MagicMock()
    cm.broadcast_event = AsyncMock()
    mgr = NotificationManager(cm)
    set_notification_manager(mgr)
    yield mgr
    set_notification_manager(None)

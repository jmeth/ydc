"""
Shared test fixtures for backend unit tests.

Provides an async HTTP client (via httpx) for testing FastAPI
endpoints and a fresh EventBus instance per test.
"""

import pytest
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

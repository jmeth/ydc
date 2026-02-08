"""
Shared test fixtures for backend unit tests.

Provides an async HTTP client (via httpx) for testing FastAPI
endpoints, a fresh EventBus instance per test, a
NotificationManager wired to a mock ConnectionManager, an
InferenceManager wired to a mock FeedManager, and persistence
Stores backed by a tmp_path directory.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from backend.core.events import EventBus
from backend.feeds.models import FeedConfig, FeedInfo, FeedStatus, FeedType
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


@pytest.fixture
def inference_manager():
    """
    Fresh InferenceManager injected into the inference API router.

    Uses a mock FeedManager with standard success returns. The
    ultralytics model loader is patched to avoid importing the
    real library. Cleans up by resetting the module-level reference.
    """
    from backend.api.inference import set_inference_manager
    from backend.inference.manager import InferenceManager

    fm = MagicMock()
    fm.get_feed_info.return_value = FeedInfo(
        feed_id="test-feed-1234",
        config=FeedConfig(feed_type=FeedType.CAMERA, source="0", name="cam0"),
        status=FeedStatus.ACTIVE,
    )
    fm.register_derived_feed.return_value = True
    fm.subscribe.return_value = True
    fm.unsubscribe.return_value = True
    fm.unregister_derived_feed.return_value = True
    fm.push_derived_frame.return_value = True

    with patch("backend.inference.loader.ModelLoader._load_ultralytics") as mock_load:
        mock_model = MagicMock()
        mock_model.names = {0: "person"}
        mock_load.return_value = mock_model

        mgr = InferenceManager(fm)
        set_inference_manager(mgr)
        yield mgr

    set_inference_manager(None)


@pytest.fixture
def dataset_manager(persistence_stores):
    """
    Fresh DatasetManager injected into the dataset API router.

    Wired to real filesystem stores via persistence_stores fixture
    and a mock EventBus. Cleans up by resetting the module-level reference.
    """
    from backend.dataset import set_dataset_manager
    from backend.dataset.manager import DatasetManager

    eb = MagicMock()
    eb.publish = AsyncMock()

    mgr = DatasetManager(
        dataset_store=persistence_stores.dataset,
        image_store=persistence_stores.image,
        label_store=persistence_stores.label,
        event_bus=eb,
    )
    set_dataset_manager(mgr)
    yield mgr
    set_dataset_manager(None)


@pytest.fixture
def persistence_stores(tmp_path):
    """
    Persistence Stores backed by a tmp_path directory.

    Creates all four stores via the factory and injects them into the
    module-level singleton. Cleans up by resetting to None.
    """
    from backend.persistence import create_stores, set_stores

    stores = create_stores(
        data_dir=tmp_path / "datasets",
        models_dir=tmp_path / "models",
    )
    set_stores(stores)
    yield stores
    set_stores(None)

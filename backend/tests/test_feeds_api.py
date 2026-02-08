"""
Integration tests for the /api/feeds REST API endpoints.

Injects a mock-backed FeedManager into the API router to test
endpoint behavior without real camera hardware.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from backend.api.feeds import set_feed_manager
from backend.feeds.manager import FeedManager
from backend.feeds.models import FeedConfig, FeedInfo, FeedStatus, FeedType, Frame
from backend.main import app


def _make_feed_info(
    feed_id: str = "test-feed-1",
    feed_type: FeedType = FeedType.CAMERA,
    source: str = "0",
    name: str = "Test Camera",
    status: FeedStatus = FeedStatus.ACTIVE,
    fps: float = 30.0,
    resolution: tuple[int, int] | None = (640, 480),
    frame_count: int = 100,
) -> FeedInfo:
    """Helper to create a FeedInfo with defaults for testing."""
    config = FeedConfig(feed_type=feed_type, source=source, name=name)
    return FeedInfo(
        feed_id=feed_id,
        config=config,
        status=status,
        fps=fps,
        resolution=resolution,
        frame_count=frame_count,
    )


@pytest.fixture
def mock_manager():
    """Create a mock FeedManager and inject it into the API."""
    manager = MagicMock(spec=FeedManager)
    # Default: empty feed list
    manager.list_feeds.return_value = []
    manager.get_feed_info.return_value = None
    manager.add_feed.return_value = None
    manager.remove_feed.return_value = False
    manager.pause.return_value = False
    manager.resume.return_value = False
    manager.get_frame.return_value = None

    set_feed_manager(manager)
    yield manager
    set_feed_manager(None)


@pytest.fixture
async def api_client():
    """Async HTTP test client (no lifespan needed for these tests)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestListFeeds:
    """Tests for GET /api/feeds."""

    @pytest.mark.asyncio
    async def test_list_empty(self, api_client, mock_manager):
        """Returns empty list when no feeds exist."""
        response = await api_client.get("/api/feeds")
        assert response.status_code == 200
        data = response.json()
        assert data["feeds"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_list_with_feeds(self, api_client, mock_manager):
        """Returns feed info when feeds exist."""
        mock_manager.list_feeds.return_value = [
            _make_feed_info(feed_id="feed-1"),
            _make_feed_info(feed_id="feed-2", source="1"),
        ]

        response = await api_client.get("/api/feeds")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert data["feeds"][0]["feed_id"] == "feed-1"
        assert data["feeds"][1]["feed_id"] == "feed-2"


class TestCreateFeed:
    """Tests for POST /api/feeds."""

    @pytest.mark.asyncio
    async def test_create_success(self, api_client, mock_manager):
        """201 with feed info on successful creation."""
        mock_manager.add_feed.return_value = "new-feed-id"
        mock_manager.get_feed_info.return_value = _make_feed_info(feed_id="new-feed-id")

        response = await api_client.post("/api/feeds", json={
            "feed_type": "camera",
            "source": "0",
            "name": "My Camera",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["feed_id"] == "new-feed-id"
        assert data["feed_type"] == "camera"
        assert data["status"] == "active"

    @pytest.mark.asyncio
    async def test_create_invalid_type(self, api_client, mock_manager):
        """422 for unsupported feed type."""
        response = await api_client.post("/api/feeds", json={
            "feed_type": "invalid",
            "source": "0",
        })
        assert response.status_code == 422
        data = response.json()
        assert "Invalid feed type" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_create_connection_failure(self, api_client, mock_manager):
        """400 when feed fails to connect."""
        mock_manager.add_feed.return_value = None

        response = await api_client.post("/api/feeds", json={
            "feed_type": "camera",
            "source": "99",
        })
        assert response.status_code == 400
        assert "connection failed" in response.json()["error"].lower()

    @pytest.mark.asyncio
    async def test_create_missing_fields(self, api_client, mock_manager):
        """422 for missing required fields."""
        response = await api_client.post("/api/feeds", json={})
        assert response.status_code == 422


class TestGetFeed:
    """Tests for GET /api/feeds/{feed_id}."""

    @pytest.mark.asyncio
    async def test_get_existing(self, api_client, mock_manager):
        """200 with feed info for existing feed."""
        mock_manager.get_feed_info.return_value = _make_feed_info()

        response = await api_client.get("/api/feeds/test-feed-1")
        assert response.status_code == 200
        data = response.json()
        assert data["feed_id"] == "test-feed-1"
        assert data["resolution"] == [640, 480]

    @pytest.mark.asyncio
    async def test_get_not_found(self, api_client, mock_manager):
        """404 for nonexistent feed."""
        response = await api_client.get("/api/feeds/nonexistent")
        assert response.status_code == 404


class TestDeleteFeed:
    """Tests for DELETE /api/feeds/{feed_id}."""

    @pytest.mark.asyncio
    async def test_delete_existing(self, api_client, mock_manager):
        """204 on successful deletion."""
        mock_manager.remove_feed.return_value = True

        response = await api_client.delete("/api/feeds/test-feed-1")
        assert response.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_not_found(self, api_client, mock_manager):
        """404 for nonexistent feed."""
        response = await api_client.delete("/api/feeds/nonexistent")
        assert response.status_code == 404


class TestPauseFeed:
    """Tests for POST /api/feeds/{feed_id}/pause."""

    @pytest.mark.asyncio
    async def test_pause_success(self, api_client, mock_manager):
        """200 with updated feed info on successful pause."""
        mock_manager.get_feed_info.return_value = _make_feed_info(status=FeedStatus.PAUSED)
        mock_manager.pause.return_value = True

        response = await api_client.post("/api/feeds/test-feed-1/pause")
        assert response.status_code == 200
        assert response.json()["status"] == "paused"

    @pytest.mark.asyncio
    async def test_pause_not_found(self, api_client, mock_manager):
        """404 for nonexistent feed."""
        response = await api_client.post("/api/feeds/nonexistent/pause")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_pause_already_paused(self, api_client, mock_manager):
        """409 when feed is not in active state."""
        mock_manager.get_feed_info.return_value = _make_feed_info(status=FeedStatus.PAUSED)
        mock_manager.pause.return_value = False

        response = await api_client.post("/api/feeds/test-feed-1/pause")
        assert response.status_code == 409


class TestResumeFeed:
    """Tests for POST /api/feeds/{feed_id}/resume."""

    @pytest.mark.asyncio
    async def test_resume_success(self, api_client, mock_manager):
        """200 with updated feed info on successful resume."""
        mock_manager.get_feed_info.return_value = _make_feed_info(status=FeedStatus.ACTIVE)
        mock_manager.resume.return_value = True

        response = await api_client.post("/api/feeds/test-feed-1/resume")
        assert response.status_code == 200
        assert response.json()["status"] == "active"

    @pytest.mark.asyncio
    async def test_resume_not_found(self, api_client, mock_manager):
        """404 for nonexistent feed."""
        response = await api_client.post("/api/feeds/nonexistent/resume")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_resume_already_running(self, api_client, mock_manager):
        """409 when feed is not paused."""
        mock_manager.get_feed_info.return_value = _make_feed_info(status=FeedStatus.ACTIVE)
        mock_manager.resume.return_value = False

        response = await api_client.post("/api/feeds/test-feed-1/resume")
        assert response.status_code == 409


class TestSnapshot:
    """Tests for GET /api/feeds/{feed_id}/snapshot."""

    @pytest.mark.asyncio
    async def test_snapshot_not_found(self, api_client, mock_manager):
        """404 for nonexistent feed."""
        response = await api_client.get("/api/feeds/nonexistent/snapshot")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_snapshot_no_frame(self, api_client, mock_manager):
        """404 when feed exists but has no frames."""
        mock_manager.get_feed_info.return_value = _make_feed_info()
        mock_manager.get_frame.return_value = None

        response = await api_client.get("/api/feeds/test-feed-1/snapshot")
        assert response.status_code == 404
        assert "No frame available" in response.json()["error"]

    @pytest.mark.asyncio
    async def test_snapshot_returns_jpeg(self, api_client, mock_manager):
        """200 with JPEG content when frame is available."""
        mock_manager.get_feed_info.return_value = _make_feed_info()
        frame = Frame(data=np.zeros((480, 640, 3), dtype=np.uint8))
        mock_manager.get_frame.return_value = frame

        response = await api_client.get("/api/feeds/test-feed-1/snapshot")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        # JPEG magic bytes
        assert response.content[:2] == b'\xff\xd8'


class TestFeedResponseShape:
    """Tests verifying the response shape matches the schema."""

    @pytest.mark.asyncio
    async def test_feed_info_fields(self, api_client, mock_manager):
        """Feed response contains all expected fields."""
        mock_manager.get_feed_info.return_value = _make_feed_info()

        response = await api_client.get("/api/feeds/test-feed-1")
        data = response.json()

        expected_fields = {"feed_id", "feed_type", "source", "name", "status", "fps", "resolution", "frame_count"}
        assert expected_fields.issubset(set(data.keys()))

    @pytest.mark.asyncio
    async def test_feed_null_resolution(self, api_client, mock_manager):
        """resolution is null when not available."""
        mock_manager.get_feed_info.return_value = _make_feed_info(resolution=None)

        response = await api_client.get("/api/feeds/test-feed-1")
        assert response.json()["resolution"] is None

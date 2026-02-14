"""
API integration tests for the capture endpoints.

Tests use the capture_manager fixture which provides a CaptureManager
wired to a mock FeedManager and real DatasetManager (backed by tmp_path).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from backend.feeds.models import (
    Detection,
    FeedConfig,
    FeedInfo,
    FeedStatus,
    FeedType,
    Frame,
    InferenceFrame,
)
from backend.persistence.models import DatasetInfo


# --- Helpers ---

async def _create_test_dataset(dataset_manager, name: str = "test-ds"):
    """Helper to create a dataset for capture tests."""
    return await dataset_manager.create_dataset(name, ["cat", "dog"])


def _make_raw_frame():
    """Create a test raw Frame."""
    return Frame(data=np.zeros((480, 640, 3), dtype=np.uint8), frame_number=0)


def _make_inference_frame():
    """Create a test InferenceFrame with a detection."""
    frame = Frame(data=np.zeros((480, 640, 3), dtype=np.uint8), frame_number=0)
    return InferenceFrame(
        frame=frame,
        detections=[Detection(class_name="cat", confidence=0.9, bbox=[100, 100, 200, 200], class_id=0)],
        model_name="yolov8n",
    )


# --- Tests ---

class TestStartCapture:
    """Tests for POST /api/capture/start."""

    async def test_start_capture_201(self, client, capture_manager, dataset_manager):
        """Successful capture start returns 201 with running status."""
        await _create_test_dataset(dataset_manager)

        resp = await client.post("/api/capture/start", json={
            "feed_id": "test-feed-1234",
            "dataset_name": "test-ds",
        })

        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "running"
        assert data["mode"] == "raw"
        assert data["feed_id"] == "test-feed-1234"
        assert data["dataset_name"] == "test-ds"

    async def test_start_capture_raw_201(self, client, capture_manager, dataset_manager):
        """Raw feed capture mode detected correctly."""
        await _create_test_dataset(dataset_manager)
        capture_manager._feed_manager.is_derived_feed.return_value = False

        resp = await client.post("/api/capture/start", json={
            "feed_id": "raw-feed-id",
            "dataset_name": "test-ds",
            "split": "val",
            "capture_interval": 5.0,
        })

        assert resp.status_code == 201
        data = resp.json()
        assert data["mode"] == "raw"
        assert data["split"] == "val"
        assert data["config"]["capture_interval"] == 5.0

    async def test_start_capture_inference_201(self, client, capture_manager, dataset_manager):
        """Inference feed capture mode detected correctly."""
        await _create_test_dataset(dataset_manager)
        capture_manager._feed_manager.is_derived_feed.return_value = True

        resp = await client.post("/api/capture/start", json={
            "feed_id": "inference-feed-id",
            "dataset_name": "test-ds",
        })

        assert resp.status_code == 201
        data = resp.json()
        assert data["mode"] == "inference"

    async def test_start_capture_404_feed(self, client, capture_manager, dataset_manager):
        """Non-existent feed returns 404."""
        await _create_test_dataset(dataset_manager)
        capture_manager._feed_manager.get_feed_info.return_value = None

        resp = await client.post("/api/capture/start", json={
            "feed_id": "bad-feed",
            "dataset_name": "test-ds",
        })

        assert resp.status_code == 404

    async def test_start_capture_404_dataset(self, client, capture_manager):
        """Non-existent dataset returns 404."""
        resp = await client.post("/api/capture/start", json={
            "feed_id": "test-feed-1234",
            "dataset_name": "no-such-dataset",
        })

        assert resp.status_code == 404

    async def test_start_capture_409_already_running(self, client, capture_manager, dataset_manager):
        """Starting capture when already running returns 409."""
        await _create_test_dataset(dataset_manager)

        # First start
        resp1 = await client.post("/api/capture/start", json={
            "feed_id": "test-feed-1234",
            "dataset_name": "test-ds",
        })
        assert resp1.status_code == 201

        # Second start (conflict)
        resp2 = await client.post("/api/capture/start", json={
            "feed_id": "test-feed-1234",
            "dataset_name": "test-ds",
        })
        assert resp2.status_code == 409


class TestStopCapture:
    """Tests for POST /api/capture/stop."""

    async def test_stop_capture_200(self, client, capture_manager, dataset_manager):
        """Stopping a running capture returns 200 with idle status."""
        await _create_test_dataset(dataset_manager)

        await client.post("/api/capture/start", json={
            "feed_id": "test-feed-1234",
            "dataset_name": "test-ds",
        })

        resp = await client.post("/api/capture/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"

    async def test_stop_capture_idle(self, client, capture_manager):
        """Stopping when idle returns 200 with idle status."""
        resp = await client.post("/api/capture/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"


class TestGetStatus:
    """Tests for GET /api/capture/status."""

    async def test_get_status_200(self, client, capture_manager, dataset_manager):
        """Status endpoint returns current state with mode field."""
        await _create_test_dataset(dataset_manager)

        await client.post("/api/capture/start", json={
            "feed_id": "test-feed-1234",
            "dataset_name": "test-ds",
        })

        resp = await client.get("/api/capture/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["mode"] == "raw"
        assert "config" in data
        assert "stats" in data

    async def test_get_status_idle(self, client, capture_manager):
        """Status returns idle when no session is running."""
        resp = await client.get("/api/capture/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"


class TestManualTrigger:
    """Tests for POST /api/capture/trigger."""

    async def test_manual_trigger_200(self, client, capture_manager, dataset_manager):
        """Manual trigger saves a frame and returns capture details."""
        await _create_test_dataset(dataset_manager)
        capture_manager._feed_manager.get_frame.return_value = _make_raw_frame()

        await client.post("/api/capture/start", json={
            "feed_id": "test-feed-1234",
            "dataset_name": "test-ds",
        })

        resp = await client.post("/api/capture/trigger")
        assert resp.status_code == 200
        data = resp.json()
        assert "filename" in data
        assert data["split"] == "train"
        assert data["dataset_name"] == "test-ds"

    async def test_manual_trigger_409_not_running(self, client, capture_manager):
        """Manual trigger when not running returns 409."""
        resp = await client.post("/api/capture/trigger")
        assert resp.status_code == 409


class TestUpdateConfig:
    """Tests for PUT /api/capture/config."""

    async def test_update_config_200(self, client, capture_manager, dataset_manager):
        """Config update changes settings and returns updated status."""
        await _create_test_dataset(dataset_manager)

        await client.post("/api/capture/start", json={
            "feed_id": "test-feed-1234",
            "dataset_name": "test-ds",
        })

        resp = await client.put("/api/capture/config", json={
            "capture_interval": 10.0,
            "negative_ratio": 0.5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["capture_interval"] == 10.0
        assert data["config"]["negative_ratio"] == 0.5

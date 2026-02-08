"""
REST API integration tests for the inference endpoints.

Uses the inference_manager fixture (mock FeedManager + patched ultralytics)
to test all five endpoints: start, stop, status, prompts, and model switch.
"""

import pytest


class TestStartInference:
    """Tests for POST /api/inference/start."""

    @pytest.mark.asyncio
    async def test_start_returns_201(self, client, inference_manager):
        """Successful start returns 201 with output_feed_id."""
        response = await client.post("/api/inference/start", json={
            "source_feed_id": "test-feed-1234",
            "model_name": "yolo11n",
            "model_type": "fine_tuned",
        })
        assert response.status_code == 201
        data = response.json()
        assert "output_feed_id" in data
        assert data["source_feed_id"] == "test-feed-1234"
        assert data["model_name"] == "yolo11n"
        assert data["model_type"] == "fine_tuned"
        assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_start_not_found_feed(self, client, inference_manager):
        """Start with nonexistent feed returns 404."""
        inference_manager._feed_manager.get_feed_info.return_value = None
        response = await client.post("/api/inference/start", json={
            "source_feed_id": "nonexistent",
        })
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_start_conflict_duplicate(self, client, inference_manager):
        """Starting inference twice on the same feed returns 409."""
        await client.post("/api/inference/start", json={
            "source_feed_id": "test-feed-1234",
        })
        response = await client.post("/api/inference/start", json={
            "source_feed_id": "test-feed-1234",
        })
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_start_defaults(self, client, inference_manager):
        """Start with only source_feed_id uses default model and type."""
        response = await client.post("/api/inference/start", json={
            "source_feed_id": "test-feed-1234",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["model_name"] == "yolo11n"
        assert data["model_type"] == "fine_tuned"


class TestStopInference:
    """Tests for POST /api/inference/stop."""

    @pytest.mark.asyncio
    async def test_stop_returns_200(self, client, inference_manager):
        """Stopping an active session returns 200."""
        start_resp = await client.post("/api/inference/start", json={
            "source_feed_id": "test-feed-1234",
        })
        output_id = start_resp.json()["output_feed_id"]

        response = await client.post("/api/inference/stop", json={
            "output_feed_id": output_id,
        })
        assert response.status_code == 200
        assert response.json()["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_stop_not_found(self, client, inference_manager):
        """Stopping a nonexistent session returns 404."""
        response = await client.post("/api/inference/stop", json={
            "output_feed_id": "nonexistent",
        })
        assert response.status_code == 404


class TestGetInferenceStatus:
    """Tests for GET /api/inference/status."""

    @pytest.mark.asyncio
    async def test_status_empty(self, client, inference_manager):
        """Status with no sessions returns empty list."""
        response = await client.get("/api/inference/status")
        assert response.status_code == 200
        data = response.json()
        assert data["sessions"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_status_with_session(self, client, inference_manager):
        """Status returns active session details."""
        await client.post("/api/inference/start", json={
            "source_feed_id": "test-feed-1234",
            "prompts": ["person"],
        })

        response = await client.get("/api/inference/status")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        session = data["sessions"][0]
        assert session["source_feed_id"] == "test-feed-1234"
        assert session["status"] == "running"
        assert session["frames_processed"] == 0


class TestUpdatePrompts:
    """Tests for PUT /api/inference/prompts."""

    @pytest.mark.asyncio
    async def test_update_prompts_returns_200(self, client, inference_manager):
        """Updating prompts on an active session returns 200."""
        start_resp = await client.post("/api/inference/start", json={
            "source_feed_id": "test-feed-1234",
            "prompts": ["person"],
        })
        output_id = start_resp.json()["output_feed_id"]

        response = await client.put("/api/inference/prompts", json={
            "output_feed_id": output_id,
            "prompts": ["car", "truck"],
        })
        assert response.status_code == 200
        assert response.json()["status"] == "updated"

    @pytest.mark.asyncio
    async def test_update_prompts_not_found(self, client, inference_manager):
        """Updating prompts on a nonexistent session returns 404."""
        response = await client.put("/api/inference/prompts", json={
            "output_feed_id": "nonexistent",
            "prompts": ["cat"],
        })
        assert response.status_code == 404


class TestSwitchModel:
    """Tests for PUT /api/inference/model."""

    @pytest.mark.asyncio
    async def test_switch_model_returns_200(self, client, inference_manager):
        """Switching model on an active session returns 200 with new output_feed_id."""
        start_resp = await client.post("/api/inference/start", json={
            "source_feed_id": "test-feed-1234",
        })
        output_id = start_resp.json()["output_feed_id"]

        response = await client.put("/api/inference/model", json={
            "output_feed_id": output_id,
            "model_name": "yolov8s-worldv2",
            "model_type": "yolo_world",
            "prompts": ["person", "car"],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "switched"
        assert data["old_output_feed_id"] == output_id
        assert "new_output_feed_id" in data
        assert data["new_output_feed_id"] != output_id

    @pytest.mark.asyncio
    async def test_switch_model_not_found(self, client, inference_manager):
        """Switching model on a nonexistent session returns 404."""
        response = await client.put("/api/inference/model", json={
            "output_feed_id": "nonexistent",
            "model_name": "yolov8s-worldv2",
            "model_type": "yolo_world",
        })
        assert response.status_code == 404

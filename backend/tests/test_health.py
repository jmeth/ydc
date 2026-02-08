"""
Tests for health check and system endpoints.

Verifies /api/health returns 200 with correct shape,
and /api/system/status and /api/system/config work.
"""

import pytest


@pytest.mark.asyncio
async def test_health_check_returns_ok(client):
    """Health endpoint returns 200 with status 'ok' and version."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_system_status_returns_ok(client):
    """System status endpoint returns 200 with status 'ok'."""
    response = await client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_system_config_returns_defaults(client):
    """System config endpoint returns expected configuration keys."""
    response = await client.get("/api/system/config")
    assert response.status_code == 200
    data = response.json()
    assert "data_dir" in data
    assert "capture_interval" in data
    assert "training_epochs" in data
    assert isinstance(data["capture_interval"], (int, float))

"""
Parametrized tests verifying all stub endpoints return 501.

Each stub endpoint should return a 501 status code with a JSON body
containing 'error' and 'detail' keys indicating not-yet-implemented.
"""

import pytest


# All stub endpoints: (method, path)
STUB_ENDPOINTS = [
    # Feeds
    ("GET", "/api/feeds"),
    ("GET", "/api/feeds/test-feed"),
    ("POST", "/api/feeds"),
    ("PUT", "/api/feeds/test-feed"),
    ("DELETE", "/api/feeds/test-feed"),
    # Inference
    ("POST", "/api/inference/start"),
    ("POST", "/api/inference/stop"),
    ("GET", "/api/inference/status"),
    ("PUT", "/api/inference/prompts"),
    ("PUT", "/api/inference/model"),
    # Capture
    ("POST", "/api/capture/start"),
    ("POST", "/api/capture/stop"),
    ("GET", "/api/capture/status"),
    ("POST", "/api/capture/trigger"),
    ("PUT", "/api/capture/config"),
    # Datasets
    ("GET", "/api/datasets"),
    ("POST", "/api/datasets"),
    ("POST", "/api/datasets/import"),
    ("GET", "/api/datasets/test-ds"),
    ("PUT", "/api/datasets/test-ds"),
    ("DELETE", "/api/datasets/test-ds"),
    ("GET", "/api/datasets/test-ds/export"),
    ("GET", "/api/datasets/test-ds/images"),
    ("POST", "/api/datasets/test-ds/images"),
    ("DELETE", "/api/datasets/test-ds/images/train/img.jpg"),
    ("GET", "/api/datasets/test-ds/labels/train/img.txt"),
    ("PUT", "/api/datasets/test-ds/labels/train/img.txt"),
    ("PUT", "/api/datasets/test-ds/split/train/img.jpg"),
    ("GET", "/api/datasets/test-ds/review"),
    ("POST", "/api/datasets/test-ds/review/bulk"),
    ("GET", "/api/datasets/test-ds/prompts"),
    ("PUT", "/api/datasets/test-ds/prompts"),
    # Training
    ("POST", "/api/training/start"),
    ("POST", "/api/training/stop"),
    ("GET", "/api/training/status"),
    ("GET", "/api/training/history"),
    # Models
    ("GET", "/api/models"),
    ("GET", "/api/models/test-model"),
    ("DELETE", "/api/models/test-model"),
    ("PUT", "/api/models/test-model/activate"),
    ("POST", "/api/models/test-model/export"),
    # Notifications
    ("GET", "/api/notifications"),
    ("POST", "/api/notifications/test-id/read"),
    ("POST", "/api/notifications/test-id/dismiss"),
    ("DELETE", "/api/notifications"),
    # System (resources is a stub)
    ("PUT", "/api/system/config"),
    ("GET", "/api/system/resources"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", STUB_ENDPOINTS, ids=[f"{m} {p}" for m, p in STUB_ENDPOINTS])
async def test_stub_returns_501(client, method, path):
    """Stub endpoints return 501 with error and detail fields."""
    response = await client.request(method, path)
    assert response.status_code == 501, f"{method} {path} returned {response.status_code}"
    data = response.json()
    assert "error" in data
    assert "detail" in data

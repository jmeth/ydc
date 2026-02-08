"""
Parametrized tests verifying all stub endpoints return 501.

Each stub endpoint should return a 501 status code with a JSON body
containing 'error' and 'detail' keys indicating not-yet-implemented.
"""

import pytest


# All stub endpoints: (method, path)
# Feeds endpoints removed — now implemented (see test_feeds_api.py)
# Inference endpoints removed — now implemented (see test_inference_api.py)
STUB_ENDPOINTS = [
    # Capture
    ("POST", "/api/capture/start"),
    ("POST", "/api/capture/stop"),
    ("GET", "/api/capture/status"),
    ("POST", "/api/capture/trigger"),
    ("PUT", "/api/capture/config"),
    # Datasets (most endpoints now implemented — see test_dataset_api.py)
    # Review queue stubs remain
    ("GET", "/api/datasets/test-ds/review"),
    ("POST", "/api/datasets/test-ds/review/bulk"),
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
    # Notifications endpoints removed — now implemented (see test_notifications_api.py)
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

"""
Parametrized tests verifying all stub endpoints return 501.

Each stub endpoint should return a 501 status code with a JSON body
containing 'error' and 'detail' keys indicating not-yet-implemented.
"""

import pytest


# All stub endpoints: (method, path)
# Feeds endpoints removed — now implemented (see test_feeds_api.py)
# Inference endpoints removed — now implemented (see test_inference_api.py)
# Capture endpoints removed — now implemented (see test_capture_api.py)
STUB_ENDPOINTS = [
    # Datasets (most endpoints now implemented — see test_dataset_api.py)
    # Review queue stubs remain
    ("GET", "/api/datasets/test-ds/review"),
    ("POST", "/api/datasets/test-ds/review/bulk"),
    # Training endpoints removed — now implemented (see test_training_api.py)
    # Models endpoints removed — now implemented (see test_models_api.py)
    # Export is still a stub
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

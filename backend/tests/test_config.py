"""
Tests for the Pydantic Settings configuration.

Verifies default values load correctly and that environment variable
overrides work via the YDC_ prefix.
"""

import os
import pytest

from backend.core.config import Settings


def test_default_values():
    """Settings loads with sensible defaults when no env vars are set."""
    s = Settings()
    assert s.data_dir == "datasets"
    assert s.port == 8000
    assert s.debug is False
    assert s.capture_interval == 2.0
    assert s.negative_ratio == 0.2
    assert s.confidence_threshold == 0.3
    assert s.training_epochs == 100
    assert s.training_batch_size == 16
    assert s.training_image_size == 640


def test_cors_origins_default():
    """Default CORS origins include the Vite dev server."""
    s = Settings()
    assert "http://localhost:5173" in s.cors_origins


def test_env_var_override(monkeypatch):
    """Environment variables with YDC_ prefix override defaults."""
    monkeypatch.setenv("YDC_DATA_DIR", "/custom/data")
    monkeypatch.setenv("YDC_PORT", "9000")
    monkeypatch.setenv("YDC_DEBUG", "true")
    monkeypatch.setenv("YDC_CAPTURE_INTERVAL", "5.0")

    s = Settings()
    assert s.data_dir == "/custom/data"
    assert s.port == 9000
    assert s.debug is True
    assert s.capture_interval == 5.0


def test_models_dir_default():
    """Models directory has a default value."""
    s = Settings()
    assert s.models_dir == "models"


def test_host_default():
    """Host defaults to 0.0.0.0 for container compatibility."""
    s = Settings()
    assert s.host == "0.0.0.0"

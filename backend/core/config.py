"""
Application configuration using Pydantic BaseSettings.

Loads settings from environment variables with YDC_ prefix, falling back
to defaults. Access via the module-level `settings` singleton.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Central application configuration.

    All fields can be overridden via environment variables prefixed with YDC_.
    Example: YDC_DATA_DIR=/custom/path overrides data_dir.
    """

    model_config = {"env_prefix": "YDC_"}

    # Data paths
    data_dir: str = Field(default="datasets", description="Root directory for dataset storage")
    models_dir: str = Field(default="models", description="Directory for trained model storage")
    logs_dir: str = Field(default="logs", description="Directory for log files")

    # Server
    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")

    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:5173"],
        description="Allowed CORS origins (Vite dev server by default)",
    )

    # Capture defaults
    capture_interval: float = Field(default=2.0, description="Default seconds between auto-captures")
    negative_ratio: float = Field(default=0.2, description="Default negative frame capture ratio")
    confidence_threshold: float = Field(default=0.3, description="Default detection confidence threshold")

    # Feed defaults
    feed_stream_fps: float = Field(default=15.0, description="Target FPS for WebSocket frame streaming")
    feed_jpeg_quality: int = Field(default=70, description="JPEG compression quality (0-100) for streamed frames")
    feed_default_buffer_size: int = Field(default=30, description="Default ring buffer size per feed")

    # Inference defaults
    default_model_name: str = Field(default="yolov8s-worldv2", description="Default YOLO model for inference")
    inference_max_sessions: int = Field(default=4, description="Maximum concurrent inference sessions")

    # Training defaults
    training_epochs: int = Field(default=100, description="Default training epochs")
    training_batch_size: int = Field(default=16, description="Default training batch size")
    training_image_size: int = Field(default=640, description="Default training image size")


# Module-level singleton
settings = Settings()

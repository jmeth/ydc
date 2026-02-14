"""
Pydantic API schemas for the training and models subsystems.

Request/response models for training job management, training history,
and trained model CRUD. Used by the training and models API routers
for request validation and response serialization.
"""

from pydantic import BaseModel, Field


class StartTrainingRequest(BaseModel):
    """Request body for POST /api/training/start."""
    dataset_name: str = Field(description="Name of the dataset to train on")
    base_model: str = Field(default="yolo11n.pt", description="Base YOLO model to fine-tune")
    epochs: int | None = Field(default=None, ge=1, description="Number of training epochs (uses server default if omitted)")
    batch_size: int | None = Field(default=None, ge=1, description="Training batch size (uses server default if omitted)")
    image_size: int | None = Field(default=None, ge=32, description="Training image size in pixels (uses server default if omitted)")
    patience: int | None = Field(default=None, ge=0, description="Early stopping patience (uses server default if omitted)")
    freeze_layers: int | None = Field(default=None, ge=0, description="Number of backbone layers to freeze")
    lr0: float | None = Field(default=None, gt=0, description="Initial learning rate")
    lrf: float | None = Field(default=None, gt=0, description="Final learning rate factor")
    model_name: str | None = Field(default=None, description="Name for the resulting model (auto-generated if omitted)")


class TrainingStatusResponse(BaseModel):
    """Response for GET /api/training/status and POST /api/training/start."""
    job_id: str = Field(description="Unique training job identifier")
    status: str = Field(description="Job status: idle, training, completed, error, cancelled")
    current_epoch: int = Field(default=0, description="Current training epoch (0-indexed)")
    total_epochs: int = Field(default=0, description="Total number of epochs")
    progress_pct: float = Field(default=0.0, description="Training progress as percentage (0-100)")
    metrics: dict = Field(default_factory=dict, description="Latest training metrics (loss, mAP, etc.)")
    dataset_name: str = Field(default="", description="Dataset being trained on")
    base_model: str = Field(default="", description="Base model being fine-tuned")
    model_name: str = Field(default="", description="Name of the output model")
    started_at: float | None = Field(default=None, description="Unix timestamp when training started")
    completed_at: float | None = Field(default=None, description="Unix timestamp when training finished")
    error: str | None = Field(default=None, description="Error message if status is 'error'")


class TrainingHistoryEntry(BaseModel):
    """Single entry in training history."""
    job_id: str = Field(description="Unique training job identifier")
    model_name: str = Field(description="Name of the output model")
    dataset_name: str = Field(description="Dataset that was trained on")
    base_model: str = Field(default="", description="Base model used")
    status: str = Field(description="Final status: completed, error, cancelled")
    epochs_completed: int = Field(default=0, description="Number of epochs completed")
    best_map50: float | None = Field(default=None, description="Best mAP@50 achieved")
    started_at: float | None = Field(default=None, description="Unix timestamp when training started")
    completed_at: float | None = Field(default=None, description="Unix timestamp when training finished")


class TrainingHistoryResponse(BaseModel):
    """Response for GET /api/training/history."""
    jobs: list[TrainingHistoryEntry] = Field(description="Past training jobs")
    count: int = Field(description="Total number of history entries")


class ModelResponse(BaseModel):
    """Response for a single trained model."""
    name: str = Field(description="Model identifier")
    base_model: str = Field(default="", description="Base model used for training")
    dataset_name: str = Field(default="", description="Dataset used for training")
    created_at: float = Field(default=0.0, description="Unix timestamp of creation")
    epochs_completed: int = Field(default=0, description="Number of training epochs completed")
    best_map50: float | None = Field(default=None, description="Best mAP@50 metric achieved")
    is_active: bool = Field(default=False, description="Whether this model is the active inference model")
    path: str = Field(default="", description="Path to model weights file")


class ModelListResponse(BaseModel):
    """Response for GET /api/models."""
    models: list[ModelResponse] = Field(description="All trained models")
    count: int = Field(description="Total number of models")


class DownloadPretrainedRequest(BaseModel):
    """Request body for POST /api/models/pretrained."""
    model_id: str = Field(
        description="Pretrained model identifier (e.g. 'yolo11n.pt', 'yolov8s.pt')"
    )
    name: str | None = Field(
        default=None,
        description="Display name for the model (defaults to model_id minus .pt extension)",
    )


class MessageResponse(BaseModel):
    """Generic message response for confirmations."""
    message: str = Field(description="Human-readable confirmation message")

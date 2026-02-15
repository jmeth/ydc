"""
Pydantic API schemas for the training and models subsystems.

Request/response models for training job management, training history,
and trained model CRUD. Used by the training and models API routers
for request validation and response serialization.
"""

from typing import Literal

from pydantic import BaseModel, Field


class AugmentationConfig(BaseModel):
    """
    Data augmentation parameters passed to ultralytics YOLO training.

    All fields are optional — omitted fields use ultralytics defaults.
    See https://docs.ultralytics.com/guides/yolo-data-augmentation/
    """

    # Color space augmentations
    hsv_h: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="HSV hue shift fraction (ultralytics default: 0.015)",
    )
    hsv_s: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="HSV saturation shift fraction (ultralytics default: 0.7)",
    )
    hsv_v: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="HSV value (brightness) shift fraction (ultralytics default: 0.4)",
    )

    # Geometric transformations
    degrees: float | None = Field(
        default=None, ge=0.0, le=180.0,
        description="Random rotation range in degrees (ultralytics default: 0.0)",
    )
    translate: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Random horizontal/vertical shift fraction (ultralytics default: 0.1)",
    )
    scale: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Random scaling factor range (ultralytics default: 0.5)",
    )
    shear: float | None = Field(
        default=None, ge=-180.0, le=180.0,
        description="Random shear angle in degrees (ultralytics default: 0.0)",
    )
    perspective: float | None = Field(
        default=None, ge=0.0, le=0.001,
        description="Random perspective warp fraction (ultralytics default: 0.0)",
    )
    flipud: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Vertical flip probability (ultralytics default: 0.0)",
    )
    fliplr: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Horizontal flip probability (ultralytics default: 0.5)",
    )
    bgr: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="RGB-to-BGR channel swap probability (ultralytics default: 0.0)",
    )

    # Advanced augmentations
    mosaic: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Mosaic augmentation probability — combines 4 images (ultralytics default: 1.0)",
    )
    mixup: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="MixUp augmentation probability — blends 2 images (ultralytics default: 0.0)",
    )
    copy_paste: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Copy-paste augmentation probability (ultralytics default: 0.0)",
    )
    erasing: float | None = Field(
        default=None, ge=0.0, le=0.9,
        description="Random erasing probability during classification (ultralytics default: 0.4)",
    )
    auto_augment: Literal["randaugment", "autoaugment", "augmix"] | None = Field(
        default=None,
        description="Automatic augmentation policy (ultralytics default: 'randaugment')",
    )

    def to_training_kwargs(self) -> dict:
        """
        Return a dict of only the non-None fields, suitable for
        splatting into ultralytics model.train(**kwargs).

        Returns:
            Dict mapping augmentation parameter names to their values.
        """
        return {k: v for k, v in self.model_dump().items() if v is not None}


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
    augmentation: AugmentationConfig | None = Field(
        default=None,
        description="Data augmentation overrides (omit to use ultralytics defaults)",
    )


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

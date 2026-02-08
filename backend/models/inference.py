"""
Pydantic API schemas for the inference subsystem.

Request/response models for starting, stopping, and managing
inference sessions via the REST API.
"""

from pydantic import BaseModel, Field


class StartInferenceRequest(BaseModel):
    """Request body for POST /api/inference/start."""
    source_feed_id: str = Field(description="ID of the source feed to run inference on")
    model_name: str = Field(
        default="yolov8s-worldv2",
        description="Model file or identifier (e.g. 'yolov8s-worldv2', 'custom.pt')",
    )
    model_type: str = Field(
        default="yolo_world",
        description="Model type: 'yolo_world' (zero-shot) or 'fine_tuned' (custom weights)",
    )
    prompts: list[str] | None = Field(
        default=None,
        description="Text prompts for YOLO-World zero-shot detection",
    )
    confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence (0.0-1.0)",
    )


class StartInferenceResponse(BaseModel):
    """Response body for POST /api/inference/start."""
    output_feed_id: str = Field(description="ID of the inference output derived feed")
    source_feed_id: str = Field(description="ID of the source feed")
    model_name: str = Field(description="Name of the loaded model")
    model_type: str = Field(description="Type of the loaded model")
    status: str = Field(default="running", description="Session status")


class StopInferenceRequest(BaseModel):
    """Request body for POST /api/inference/stop."""
    output_feed_id: str = Field(description="ID of the inference output feed to stop")


class InferenceSessionResponse(BaseModel):
    """Status of a single inference session."""
    output_feed_id: str = Field(description="Inference output feed ID")
    source_feed_id: str = Field(description="Source feed ID")
    model_name: str = Field(description="Active model name")
    model_type: str = Field(description="Active model type")
    prompts: list[str] | None = Field(default=None, description="YOLO-World text prompts")
    confidence_threshold: float = Field(description="Detection confidence threshold")
    frames_processed: int = Field(description="Total frames processed")
    avg_inference_ms: float = Field(description="Average inference time in ms")
    last_inference_ms: float = Field(description="Last frame inference time in ms")
    status: str = Field(description="Session status")


class InferenceStatusResponse(BaseModel):
    """Response body for GET /api/inference/status."""
    sessions: list[InferenceSessionResponse] = Field(
        default_factory=list, description="All active inference sessions",
    )
    count: int = Field(description="Number of active sessions")


class UpdatePromptsRequest(BaseModel):
    """Request body for PUT /api/inference/prompts."""
    output_feed_id: str = Field(description="Inference output feed to update")
    prompts: list[str] = Field(description="New YOLO-World text prompts")


class SwitchModelRequest(BaseModel):
    """Request body for PUT /api/inference/model."""
    output_feed_id: str = Field(description="Inference output feed to switch model on")
    model_name: str = Field(description="New model file or identifier")
    model_type: str = Field(
        default="yolo_world",
        description="New model type: 'yolo_world' or 'fine_tuned'",
    )
    prompts: list[str] | None = Field(
        default=None,
        description="Text prompts for YOLO-World models",
    )

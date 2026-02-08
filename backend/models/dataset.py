"""
Pydantic API schemas for the dataset subsystem.

Request/response models for dataset CRUD, image management, label
management, prompts, and split operations. Used by the datasets API
router for request validation and response serialization.
"""

from pydantic import BaseModel, Field


class CreateDatasetRequest(BaseModel):
    """Request body for POST /api/datasets."""
    name: str = Field(description="Dataset identifier (alphanumeric, hyphens, underscores)")
    classes: list[str] = Field(description="Ordered list of class names")


class UpdateDatasetRequest(BaseModel):
    """Request body for PUT /api/datasets/{name}."""
    classes: list[str] = Field(description="Updated ordered list of class names")


class DatasetResponse(BaseModel):
    """Response for a single dataset's metadata."""
    name: str = Field(description="Dataset identifier")
    path: str = Field(description="Absolute path to dataset root")
    classes: list[str] = Field(description="Ordered list of class names")
    num_images: dict[str, int] = Field(description="Image count per split")
    created_at: float = Field(description="Unix timestamp of creation")
    modified_at: float = Field(description="Unix timestamp of last modification")


class DatasetListResponse(BaseModel):
    """Response for listing all datasets."""
    datasets: list[DatasetResponse] = Field(description="All available datasets")
    count: int = Field(description="Total number of datasets")


class ImageResponse(BaseModel):
    """Response for a single image's metadata."""
    filename: str = Field(description="Image filename")
    split: str = Field(description="Dataset split (train, val, test)")
    width: int = Field(description="Image width in pixels")
    height: int = Field(description="Image height in pixels")
    size_bytes: int = Field(description="File size on disk in bytes")
    has_labels: bool = Field(description="Whether a label file exists")


class ImageListResponse(BaseModel):
    """Response for listing images in a dataset."""
    images: list[ImageResponse] = Field(description="Matching images")
    count: int = Field(description="Total number of matching images")


class AnnotationModel(BaseModel):
    """Single YOLO-format annotation (Pydantic mirror of persistence Annotation)."""
    class_id: int = Field(ge=0, description="Class index (zero-based)")
    x: float = Field(ge=0.0, le=1.0, description="Center x coordinate (normalized 0-1)")
    y: float = Field(ge=0.0, le=1.0, description="Center y coordinate (normalized 0-1)")
    width: float = Field(ge=0.0, le=1.0, description="Bounding box width (normalized 0-1)")
    height: float = Field(ge=0.0, le=1.0, description="Bounding box height (normalized 0-1)")


class SaveLabelsRequest(BaseModel):
    """Request body for PUT /api/datasets/{name}/labels/{split}/{file}."""
    annotations: list[AnnotationModel] = Field(description="YOLO-format annotations to save")


class LabelsResponse(BaseModel):
    """Response for getting labels for an image."""
    filename: str = Field(description="Image filename")
    split: str = Field(description="Dataset split")
    annotations: list[AnnotationModel] = Field(description="YOLO-format annotations")


class ChangeSplitRequest(BaseModel):
    """Request body for PUT /api/datasets/{name}/split/{split}/{file}."""
    to_split: str = Field(description="Destination split (train, val, or test)")


class PromptsResponse(BaseModel):
    """Response for getting dataset prompts."""
    prompts: dict[int, list[str]] = Field(description="Mapping of class_id to prompt strings")


class SavePromptsRequest(BaseModel):
    """Request body for PUT /api/datasets/{name}/prompts."""
    prompts: dict[int, list[str]] = Field(description="Mapping of class_id to prompt strings")

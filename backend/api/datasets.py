"""
Datasets API router — endpoints for dataset management.

Covers dataset CRUD, image upload/delete, label read/write, split
changes, prompts, and export/import. Review queue endpoints remain
as 501 stubs until a future phase.

Uses DI via set_dataset_manager() called from app lifespan.
"""

import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from backend.core.exceptions import NotFoundError, ValidationError
from backend.dataset import get_dataset_manager
from backend.dataset.manager import DatasetManager
from backend.models.common import ErrorResponse, NotImplementedResponse
from backend.models.dataset import (
    AnnotationModel,
    ChangeSplitRequest,
    CreateDatasetRequest,
    DatasetListResponse,
    DatasetResponse,
    ImageListResponse,
    ImageResponse,
    LabelsResponse,
    PromptsResponse,
    SaveLabelsRequest,
    SavePromptsRequest,
    UpdateDatasetRequest,
)
from backend.persistence.models import Annotation

logger = logging.getLogger(__name__)

router = APIRouter()

STUB = NotImplementedResponse()
STUB_RESPONSE = {501: {"model": NotImplementedResponse}}


def _get_manager() -> DatasetManager:
    """
    Get the injected DatasetManager, raising 500 if not initialized.

    Returns:
        The active DatasetManager instance.
    """
    return get_dataset_manager()


# --- Dataset CRUD ---


@router.get(
    "",
    summary="List datasets",
    description="List all available datasets with metadata.",
    response_model=DatasetListResponse,
)
async def list_datasets() -> DatasetListResponse:
    """List all datasets."""
    mgr = _get_manager()
    datasets = await mgr.list_datasets()
    return DatasetListResponse(
        datasets=[
            DatasetResponse(
                name=d.name,
                path=str(d.path),
                classes=d.classes,
                num_images=d.num_images,
                created_at=d.created_at,
                modified_at=d.modified_at,
            )
            for d in datasets
        ],
        count=len(datasets),
    )


@router.post(
    "",
    summary="Create dataset",
    description="Create a new dataset with a name and class list.",
    response_model=DatasetResponse,
    status_code=201,
    responses={
        409: {"model": ErrorResponse, "description": "Dataset already exists"},
        422: {"model": ErrorResponse, "description": "Invalid name or empty classes"},
    },
)
async def create_dataset(request: CreateDatasetRequest) -> DatasetResponse:
    """Create a new dataset."""
    mgr = _get_manager()
    info = await mgr.create_dataset(request.name, request.classes)
    return DatasetResponse(
        name=info.name,
        path=str(info.path),
        classes=info.classes,
        num_images=info.num_images,
        created_at=info.created_at,
        modified_at=info.modified_at,
    )


@router.post(
    "/import",
    summary="Import dataset",
    description="Import a dataset from an uploaded zip file.",
    response_model=DatasetResponse,
    status_code=201,
    responses={
        409: {"model": ErrorResponse, "description": "Dataset already exists"},
    },
)
async def import_dataset(
    file: UploadFile,
    name: str | None = Query(default=None, description="Override name (defaults to zip filename)"),
) -> DatasetResponse:
    """Import a dataset from an uploaded zip file."""
    mgr = _get_manager()

    # Write uploaded file to a temp location
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        info = await mgr.import_dataset(tmp_path, name=name)
    finally:
        tmp_path.unlink(missing_ok=True)

    return DatasetResponse(
        name=info.name,
        path=str(info.path),
        classes=info.classes,
        num_images=info.num_images,
        created_at=info.created_at,
        modified_at=info.modified_at,
    )


@router.get(
    "/{name}",
    summary="Get dataset info",
    description="Get metadata and statistics for a specific dataset.",
    response_model=DatasetResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Dataset not found"},
    },
)
async def get_dataset(name: str) -> DatasetResponse:
    """Get dataset info by name."""
    mgr = _get_manager()
    info = await mgr.get_dataset(name)
    return DatasetResponse(
        name=info.name,
        path=str(info.path),
        classes=info.classes,
        num_images=info.num_images,
        created_at=info.created_at,
        modified_at=info.modified_at,
    )


@router.put(
    "/{name}",
    summary="Update dataset",
    description="Update dataset class list.",
    response_model=DatasetResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Dataset not found"},
        422: {"model": ErrorResponse, "description": "Empty classes list"},
    },
)
async def update_dataset(name: str, request: UpdateDatasetRequest) -> DatasetResponse:
    """Update dataset classes."""
    mgr = _get_manager()
    info = await mgr.update_dataset(name, request.classes)
    return DatasetResponse(
        name=info.name,
        path=str(info.path),
        classes=info.classes,
        num_images=info.num_images,
        created_at=info.created_at,
        modified_at=info.modified_at,
    )


@router.delete(
    "/{name}",
    summary="Delete dataset",
    description="Delete a dataset and all its data.",
    responses={
        404: {"model": ErrorResponse, "description": "Dataset not found"},
    },
)
async def delete_dataset(name: str) -> dict:
    """Delete a dataset."""
    mgr = _get_manager()
    await mgr.delete_dataset(name)
    return {"status": "deleted", "name": name}


@router.get(
    "/{name}/export",
    summary="Export dataset",
    description="Export a dataset as a downloadable zip file.",
    responses={
        404: {"model": ErrorResponse, "description": "Dataset not found"},
    },
)
async def export_dataset(name: str) -> FileResponse:
    """Export dataset as a zip file download."""
    mgr = _get_manager()

    # Generate zip into a temp directory
    tmp_dir = Path(tempfile.mkdtemp())
    zip_path = await mgr.export_dataset(name, tmp_dir)

    return FileResponse(
        path=str(zip_path),
        filename=f"{name}.zip",
        media_type="application/zip",
    )


# --- Images ---


@router.get(
    "/{name}/images/{split}/{file}/data",
    summary="Get image data",
    description="Serve the actual image file for display in the frontend.",
    responses={
        404: {"model": ErrorResponse, "description": "Image not found"},
    },
)
async def get_image_data(name: str, split: str, file: str) -> FileResponse:
    """
    Serve an image file from a dataset split.

    Returns the raw image file as a FileResponse (JPEG/PNG) for use
    in <img> tags or thumbnails in the frontend.

    Args:
        name: Dataset name.
        split: Dataset split (train, val, test).
        file: Image filename.

    Returns:
        FileResponse with the image data.
    """
    mgr = _get_manager()
    # Ensure the dataset exists (raises NotFoundError otherwise)
    await mgr.get_dataset(name)

    # Resolve the image path via the image store
    from backend.persistence import get_image_store
    image_store = get_image_store()
    if not await image_store.exists(name, split, file):
        raise NotFoundError("Image", f"{name}/{split}/{file}")

    # Build file path using the image store's path convention
    image_path = image_store._image_path(name, split, file)

    # Determine media type from extension
    suffix = Path(file).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(path=str(image_path), media_type=media_type)


@router.get(
    "/{name}/images",
    summary="List images",
    description="List images in a dataset with optional split filter.",
    response_model=ImageListResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Dataset not found"},
    },
)
async def list_images(
    name: str,
    split: str | None = Query(default=None, description="Filter by split (train, val, test)"),
) -> ImageListResponse:
    """List images in a dataset."""
    mgr = _get_manager()
    images = await mgr.list_images(name, split=split)
    return ImageListResponse(
        images=[
            ImageResponse(
                filename=img.filename,
                split=img.split,
                width=img.width,
                height=img.height,
                size_bytes=img.size_bytes,
                has_labels=img.has_labels,
            )
            for img in images
        ],
        count=len(images),
    )


@router.post(
    "/{name}/images",
    summary="Upload image",
    description="Upload an image to a dataset split.",
    response_model=ImageResponse,
    status_code=201,
    responses={
        404: {"model": ErrorResponse, "description": "Dataset not found"},
        422: {"model": ErrorResponse, "description": "Invalid split or image data"},
    },
)
async def upload_image(
    name: str,
    file: UploadFile,
    split: str = Query(default="train", description="Target split (train, val, test)"),
) -> ImageResponse:
    """Upload an image to a dataset split."""
    mgr = _get_manager()

    # Read and decode image bytes
    content = await file.read()
    np_arr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValidationError("Failed to decode image data", detail="The uploaded file is not a valid image")

    filename = file.filename or "upload.jpg"
    info = await mgr.add_image(name, split, filename, image)

    return ImageResponse(
        filename=info.filename,
        split=info.split,
        width=info.width,
        height=info.height,
        size_bytes=info.size_bytes,
        has_labels=info.has_labels,
    )


@router.delete(
    "/{name}/images/{split}/{file}",
    summary="Delete image",
    description="Delete a specific image and its labels from a dataset split.",
    responses={
        404: {"model": ErrorResponse, "description": "Image not found"},
    },
)
async def delete_image(name: str, split: str, file: str) -> dict:
    """Delete an image and its labels."""
    mgr = _get_manager()
    await mgr.delete_image(name, split, file)
    return {"status": "deleted", "filename": file}


# --- Labels ---


@router.get(
    "/{name}/labels/{split}/{file}",
    summary="Get annotations",
    description="Get YOLO annotations for a specific image.",
    response_model=LabelsResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Dataset not found"},
    },
)
async def get_labels(name: str, split: str, file: str) -> LabelsResponse:
    """Get annotations for an image."""
    mgr = _get_manager()
    annotations = await mgr.get_labels(name, split, file)
    return LabelsResponse(
        filename=file,
        split=split,
        annotations=[
            AnnotationModel(
                class_id=a.class_id,
                x=a.x,
                y=a.y,
                width=a.width,
                height=a.height,
            )
            for a in annotations
        ],
    )


@router.put(
    "/{name}/labels/{split}/{file}",
    summary="Save annotations",
    description="Save YOLO annotations for a specific image.",
    responses={
        404: {"model": ErrorResponse, "description": "Dataset not found"},
        422: {"model": ErrorResponse, "description": "Invalid annotation data"},
    },
)
async def save_labels(name: str, split: str, file: str, request: SaveLabelsRequest) -> LabelsResponse:
    """Save annotations for an image."""
    mgr = _get_manager()

    annotations = [
        Annotation(
            class_id=a.class_id,
            x=a.x,
            y=a.y,
            width=a.width,
            height=a.height,
        )
        for a in request.annotations
    ]

    await mgr.save_labels(name, split, file, annotations)

    return LabelsResponse(
        filename=file,
        split=split,
        annotations=request.annotations,
    )


@router.put(
    "/{name}/split/{split}/{file}",
    summary="Change image split",
    description="Move an image to a different dataset split.",
    responses={
        404: {"model": ErrorResponse, "description": "Image not found"},
        422: {"model": ErrorResponse, "description": "Invalid split"},
    },
)
async def change_split(name: str, split: str, file: str, request: ChangeSplitRequest) -> dict:
    """Move an image between splits."""
    mgr = _get_manager()
    await mgr.change_split(name, file, split, request.to_split)
    return {"status": "moved", "filename": file, "from_split": split, "to_split": request.to_split}


# --- Review queue (deferred — stays as 501 stubs) ---


@router.get(
    "/{name}/review",
    summary="Get review queue",
    description="Get images pending review in a dataset.",
    responses=STUB_RESPONSE,
)
async def get_review_queue(name: str) -> JSONResponse:
    """Get pending review items (not yet implemented)."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.post(
    "/{name}/review/bulk",
    summary="Bulk review",
    description="Bulk accept or reject reviewed images.",
    responses=STUB_RESPONSE,
)
async def bulk_review(name: str) -> JSONResponse:
    """Bulk accept/reject reviewed images (not yet implemented)."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


# --- Prompts ---


@router.get(
    "/{name}/prompts",
    summary="Get prompts",
    description="Get YOLO-World class prompts for a dataset.",
    response_model=PromptsResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Dataset not found"},
    },
)
async def get_prompts(name: str) -> PromptsResponse:
    """Get dataset prompts."""
    mgr = _get_manager()
    prompts = await mgr.get_prompts(name)
    return PromptsResponse(prompts=prompts)


@router.put(
    "/{name}/prompts",
    summary="Save prompts",
    description="Save YOLO-World class prompts for a dataset.",
    responses={
        404: {"model": ErrorResponse, "description": "Dataset not found"},
    },
)
async def save_prompts(name: str, request: SavePromptsRequest) -> PromptsResponse:
    """Save dataset prompts."""
    mgr = _get_manager()
    await mgr.save_prompts(name, request.prompts)
    return PromptsResponse(prompts=request.prompts)

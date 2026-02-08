"""
Datasets API router — stub endpoints for dataset management.

Covers CRUD, images, labels, review queue, and prompts.
All endpoints return 501 until the Dataset subsystem is implemented.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.models.common import NotImplementedResponse

router = APIRouter()

STUB = NotImplementedResponse()
STUB_RESPONSE = {501: {"model": NotImplementedResponse}}


# --- Dataset CRUD ---

@router.get(
    "",
    summary="List datasets",
    description="List all available datasets.",
    responses=STUB_RESPONSE,
)
async def list_datasets() -> JSONResponse:
    """List all datasets."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.post(
    "",
    summary="Create dataset",
    description="Create a new dataset.",
    responses=STUB_RESPONSE,
)
async def create_dataset() -> JSONResponse:
    """Create a new dataset."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.post(
    "/import",
    summary="Import dataset",
    description="Import a dataset from a zip file.",
    responses=STUB_RESPONSE,
)
async def import_dataset() -> JSONResponse:
    """Import a dataset zip."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.get(
    "/{name}",
    summary="Get dataset info",
    description="Get metadata and statistics for a specific dataset.",
    responses=STUB_RESPONSE,
)
async def get_dataset(name: str) -> JSONResponse:
    """Get dataset info."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.put(
    "/{name}",
    summary="Update dataset",
    description="Update dataset metadata.",
    responses=STUB_RESPONSE,
)
async def update_dataset(name: str) -> JSONResponse:
    """Update dataset."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.delete(
    "/{name}",
    summary="Delete dataset",
    description="Delete a dataset and all its data.",
    responses=STUB_RESPONSE,
)
async def delete_dataset(name: str) -> JSONResponse:
    """Delete dataset."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.get(
    "/{name}/export",
    summary="Export dataset",
    description="Export a dataset as a zip file.",
    responses=STUB_RESPONSE,
)
async def export_dataset(name: str) -> JSONResponse:
    """Export dataset zip."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


# --- Images ---

@router.get(
    "/{name}/images",
    summary="List images",
    description="List images in a dataset with optional filter and pagination.",
    responses=STUB_RESPONSE,
)
async def list_images(name: str) -> JSONResponse:
    """List images in a dataset."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.post(
    "/{name}/images",
    summary="Upload image",
    description="Upload an image to a dataset.",
    responses=STUB_RESPONSE,
)
async def upload_image(name: str) -> JSONResponse:
    """Upload an image."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.delete(
    "/{name}/images/{split}/{file}",
    summary="Delete image",
    description="Delete a specific image from a dataset split.",
    responses=STUB_RESPONSE,
)
async def delete_image(name: str, split: str, file: str) -> JSONResponse:
    """Delete an image."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


# --- Labels ---

@router.get(
    "/{name}/labels/{split}/{file}",
    summary="Get annotations",
    description="Get YOLO annotations for a specific image.",
    responses=STUB_RESPONSE,
)
async def get_labels(name: str, split: str, file: str) -> JSONResponse:
    """Get annotations for an image."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.put(
    "/{name}/labels/{split}/{file}",
    summary="Save annotations",
    description="Save YOLO annotations for a specific image.",
    responses=STUB_RESPONSE,
)
async def save_labels(name: str, split: str, file: str) -> JSONResponse:
    """Save annotations for an image."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.put(
    "/{name}/split/{split}/{file}",
    summary="Change image split",
    description="Move an image to a different dataset split.",
    responses=STUB_RESPONSE,
)
async def change_split(name: str, split: str, file: str) -> JSONResponse:
    """Change image split."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


# --- Review queue ---

@router.get(
    "/{name}/review",
    summary="Get review queue",
    description="Get images pending review in a dataset.",
    responses=STUB_RESPONSE,
)
async def get_review_queue(name: str) -> JSONResponse:
    """Get pending review items."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.post(
    "/{name}/review/bulk",
    summary="Bulk review",
    description="Bulk accept or reject reviewed images.",
    responses=STUB_RESPONSE,
)
async def bulk_review(name: str) -> JSONResponse:
    """Bulk accept/reject reviewed images."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


# --- Prompts ---

@router.get(
    "/{name}/prompts",
    summary="Get prompts",
    description="Get class prompts for a dataset.",
    responses=STUB_RESPONSE,
)
async def get_prompts(name: str) -> JSONResponse:
    """Get dataset prompts."""
    return JSONResponse(status_code=501, content=STUB.model_dump())


@router.put(
    "/{name}/prompts",
    summary="Save prompts",
    description="Save class prompts for a dataset.",
    responses=STUB_RESPONSE,
)
async def save_prompts(name: str) -> JSONResponse:
    """Save dataset prompts."""
    return JSONResponse(status_code=501, content=STUB.model_dump())

"""
API integration tests for dataset endpoints.

Tests exercise the REST API through the ASGI test client with a
DatasetManager backed by real filesystem stores (via tmp_path).
"""

import io
import zipfile

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_image_bytes():
    """PNG-encoded 100x100 image bytes for multipart upload."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


# --- Dataset CRUD ---


class TestListDatasets:
    """Tests for GET /api/datasets."""

    async def test_empty_list(self, client, dataset_manager):
        """Return empty list when no datasets exist."""
        resp = await client.get("/api/datasets")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["datasets"] == []

    async def test_list_after_create(self, client, dataset_manager):
        """List includes created datasets."""
        await dataset_manager.create_dataset("ds1", ["cat"])
        resp = await client.get("/api/datasets")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1
        assert resp.json()["datasets"][0]["name"] == "ds1"


class TestCreateDataset:
    """Tests for POST /api/datasets."""

    async def test_create_success(self, client, dataset_manager):
        """Create a dataset and return 201 with info."""
        resp = await client.post(
            "/api/datasets",
            json={"name": "test-ds", "classes": ["cat", "dog"]},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "test-ds"
        assert data["classes"] == ["cat", "dog"]
        assert data["num_images"] == {"train": 0, "val": 0, "test": 0}

    async def test_create_duplicate_returns_409(self, client, dataset_manager):
        """Return 409 when creating a dataset that already exists."""
        await client.post(
            "/api/datasets",
            json={"name": "test-ds", "classes": ["cat"]},
        )
        resp = await client.post(
            "/api/datasets",
            json={"name": "test-ds", "classes": ["dog"]},
        )
        assert resp.status_code == 409

    async def test_create_invalid_name_returns_422(self, client, dataset_manager):
        """Return 422 for invalid dataset name."""
        resp = await client.post(
            "/api/datasets",
            json={"name": "bad name!", "classes": ["cat"]},
        )
        assert resp.status_code == 422

    async def test_create_empty_classes_returns_422(self, client, dataset_manager):
        """Return 422 for empty classes list."""
        resp = await client.post(
            "/api/datasets",
            json={"name": "test-ds", "classes": []},
        )
        assert resp.status_code == 422


class TestGetDataset:
    """Tests for GET /api/datasets/{name}."""

    async def test_get_existing(self, client, dataset_manager):
        """Return dataset info for existing dataset."""
        await dataset_manager.create_dataset("test-ds", ["cat", "dog"])
        resp = await client.get("/api/datasets/test-ds")
        assert resp.status_code == 200
        assert resp.json()["name"] == "test-ds"
        assert resp.json()["classes"] == ["cat", "dog"]

    async def test_get_not_found(self, client, dataset_manager):
        """Return 404 for nonexistent dataset."""
        resp = await client.get("/api/datasets/nonexistent")
        assert resp.status_code == 404


class TestUpdateDataset:
    """Tests for PUT /api/datasets/{name}."""

    async def test_update_classes(self, client, dataset_manager):
        """Update classes and return updated info."""
        await dataset_manager.create_dataset("test-ds", ["cat"])
        resp = await client.put(
            "/api/datasets/test-ds",
            json={"classes": ["cat", "dog", "bird"]},
        )
        assert resp.status_code == 200
        assert resp.json()["classes"] == ["cat", "dog", "bird"]

    async def test_update_not_found(self, client, dataset_manager):
        """Return 404 for nonexistent dataset."""
        resp = await client.put(
            "/api/datasets/nonexistent",
            json={"classes": ["cat"]},
        )
        assert resp.status_code == 404


class TestDeleteDataset:
    """Tests for DELETE /api/datasets/{name}."""

    async def test_delete_success(self, client, dataset_manager):
        """Delete dataset and return success."""
        await dataset_manager.create_dataset("test-ds", ["cat"])
        resp = await client.delete("/api/datasets/test-ds")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    async def test_delete_not_found(self, client, dataset_manager):
        """Return 404 for nonexistent dataset."""
        resp = await client.delete("/api/datasets/nonexistent")
        assert resp.status_code == 404


# --- Images ---


class TestListImages:
    """Tests for GET /api/datasets/{name}/images."""

    async def test_empty_list(self, client, dataset_manager):
        """Return empty image list for new dataset."""
        await dataset_manager.create_dataset("test-ds", ["cat"])
        resp = await client.get("/api/datasets/test-ds/images")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    async def test_filter_by_split(self, client, dataset_manager, sample_image_bytes):
        """Filter images by split query parameter."""
        await dataset_manager.create_dataset("test-ds", ["cat"])
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        await dataset_manager.add_image("test-ds", "train", "img1.jpg", img)
        await dataset_manager.add_image("test-ds", "val", "img2.jpg", img)

        resp = await client.get("/api/datasets/test-ds/images?split=train")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1


class TestUploadImage:
    """Tests for POST /api/datasets/{name}/images."""

    async def test_upload_success(self, client, dataset_manager, sample_image_bytes):
        """Upload an image and return 201."""
        await dataset_manager.create_dataset("test-ds", ["cat"])
        resp = await client.post(
            "/api/datasets/test-ds/images?split=train",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["filename"] == "test.png"
        assert data["split"] == "train"

    async def test_upload_dataset_not_found(self, client, dataset_manager, sample_image_bytes):
        """Return 404 when dataset doesn't exist."""
        resp = await client.post(
            "/api/datasets/nonexistent/images?split=train",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )
        assert resp.status_code == 404


class TestDeleteImage:
    """Tests for DELETE /api/datasets/{name}/images/{split}/{file}."""

    async def test_delete_success(self, client, dataset_manager):
        """Delete an existing image."""
        await dataset_manager.create_dataset("test-ds", ["cat"])
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        await dataset_manager.add_image("test-ds", "train", "img.jpg", img)

        resp = await client.delete("/api/datasets/test-ds/images/train/img.jpg")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    async def test_delete_not_found(self, client, dataset_manager):
        """Return 404 for nonexistent image."""
        await dataset_manager.create_dataset("test-ds", ["cat"])
        resp = await client.delete("/api/datasets/test-ds/images/train/nope.jpg")
        assert resp.status_code == 404


# --- Labels ---


class TestGetLabels:
    """Tests for GET /api/datasets/{name}/labels/{split}/{file}."""

    async def test_get_empty_labels(self, client, dataset_manager):
        """Return empty annotations for image with no labels."""
        await dataset_manager.create_dataset("test-ds", ["cat"])
        resp = await client.get("/api/datasets/test-ds/labels/train/img.jpg")
        assert resp.status_code == 200
        assert resp.json()["annotations"] == []

    async def test_get_not_found_dataset(self, client, dataset_manager):
        """Return 404 when dataset doesn't exist."""
        resp = await client.get("/api/datasets/nonexistent/labels/train/img.jpg")
        assert resp.status_code == 404


class TestSaveLabels:
    """Tests for PUT /api/datasets/{name}/labels/{split}/{file}."""

    async def test_save_and_get(self, client, dataset_manager):
        """Save annotations and retrieve them."""
        await dataset_manager.create_dataset("test-ds", ["cat"])
        annotations = [
            {"class_id": 0, "x": 0.5, "y": 0.5, "width": 0.3, "height": 0.4},
        ]
        resp = await client.put(
            "/api/datasets/test-ds/labels/train/img.jpg",
            json={"annotations": annotations},
        )
        assert resp.status_code == 200
        assert len(resp.json()["annotations"]) == 1

        # Verify retrieval
        get_resp = await client.get("/api/datasets/test-ds/labels/train/img.jpg")
        assert len(get_resp.json()["annotations"]) == 1


# --- Split ---


class TestChangeSplit:
    """Tests for PUT /api/datasets/{name}/split/{split}/{file}."""

    async def test_move_success(self, client, dataset_manager):
        """Move an image between splits."""
        await dataset_manager.create_dataset("test-ds", ["cat"])
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        await dataset_manager.add_image("test-ds", "train", "img.jpg", img)

        resp = await client.put(
            "/api/datasets/test-ds/split/train/img.jpg",
            json={"to_split": "val"},
        )
        assert resp.status_code == 200
        assert resp.json()["to_split"] == "val"

    async def test_move_not_found(self, client, dataset_manager):
        """Return 404 for nonexistent image."""
        await dataset_manager.create_dataset("test-ds", ["cat"])
        resp = await client.put(
            "/api/datasets/test-ds/split/train/nope.jpg",
            json={"to_split": "val"},
        )
        assert resp.status_code == 404


# --- Prompts ---


class TestPrompts:
    """Tests for GET/PUT /api/datasets/{name}/prompts."""

    async def test_get_empty(self, client, dataset_manager):
        """Return empty prompts for new dataset."""
        await dataset_manager.create_dataset("test-ds", ["cat"])
        resp = await client.get("/api/datasets/test-ds/prompts")
        assert resp.status_code == 200
        assert resp.json()["prompts"] == {}

    async def test_save_and_get(self, client, dataset_manager):
        """Save prompts and retrieve them."""
        await dataset_manager.create_dataset("test-ds", ["cat", "dog"])
        prompts = {"0": ["a cat"], "1": ["a dog"]}
        resp = await client.put(
            "/api/datasets/test-ds/prompts",
            json={"prompts": prompts},
        )
        assert resp.status_code == 200

        get_resp = await client.get("/api/datasets/test-ds/prompts")
        assert get_resp.json()["prompts"]["0"] == ["a cat"]


# --- Export / Import ---


class TestExport:
    """Tests for GET /api/datasets/{name}/export."""

    async def test_export_success(self, client, dataset_manager):
        """Export a dataset and receive zip bytes."""
        await dataset_manager.create_dataset("test-ds", ["cat"])
        resp = await client.get("/api/datasets/test-ds/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

    async def test_export_not_found(self, client, dataset_manager):
        """Return 404 for nonexistent dataset."""
        resp = await client.get("/api/datasets/nonexistent/export")
        assert resp.status_code == 404


class TestImport:
    """Tests for POST /api/datasets/import."""

    async def test_import_success(self, client, dataset_manager):
        """Import a dataset from a zip file."""
        # First create and export a dataset to get a valid zip
        await dataset_manager.create_dataset("source-ds", ["cat", "dog"])
        export_resp = await client.get("/api/datasets/source-ds/export")
        assert export_resp.status_code == 200

        # Import under a new name
        zip_bytes = export_resp.content
        resp = await client.post(
            "/api/datasets/import?name=imported-ds",
            files={"file": ("source-ds.zip", zip_bytes, "application/zip")},
        )
        assert resp.status_code == 201
        assert resp.json()["name"] == "imported-ds"
        assert resp.json()["classes"] == ["cat", "dog"]


# --- Review Queue (stubs) ---


class TestReviewStubs:
    """Verify review endpoints still return 501."""

    async def test_get_review_queue(self, client, dataset_manager):
        """GET review returns 501."""
        resp = await client.get("/api/datasets/test-ds/review")
        assert resp.status_code == 501

    async def test_bulk_review(self, client, dataset_manager):
        """POST bulk review returns 501."""
        resp = await client.post("/api/datasets/test-ds/review/bulk")
        assert resp.status_code == 501

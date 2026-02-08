"""
Unit tests for dataset Pydantic API models.

Validates schema constraints, defaults, and serialization for all
request/response models in backend.models.dataset.
"""

import pytest
from pydantic import ValidationError

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


class TestCreateDatasetRequest:
    """Tests for CreateDatasetRequest schema."""

    def test_valid_request(self):
        """Accept valid name and classes."""
        req = CreateDatasetRequest(name="my-dataset", classes=["cat", "dog"])
        assert req.name == "my-dataset"
        assert req.classes == ["cat", "dog"]

    def test_missing_name_fails(self):
        """Reject request without name."""
        with pytest.raises(ValidationError):
            CreateDatasetRequest(classes=["cat"])

    def test_missing_classes_fails(self):
        """Reject request without classes."""
        with pytest.raises(ValidationError):
            CreateDatasetRequest(name="test")


class TestUpdateDatasetRequest:
    """Tests for UpdateDatasetRequest schema."""

    def test_valid_request(self):
        """Accept valid classes list."""
        req = UpdateDatasetRequest(classes=["person", "car", "truck"])
        assert req.classes == ["person", "car", "truck"]


class TestAnnotationModel:
    """Tests for AnnotationModel schema constraints."""

    def test_valid_annotation(self):
        """Accept valid normalized coordinates."""
        ann = AnnotationModel(class_id=0, x=0.5, y=0.5, width=0.3, height=0.4)
        assert ann.class_id == 0
        assert ann.x == 0.5

    def test_negative_class_id_fails(self):
        """Reject negative class_id."""
        with pytest.raises(ValidationError):
            AnnotationModel(class_id=-1, x=0.5, y=0.5, width=0.3, height=0.4)

    def test_coordinate_out_of_range_fails(self):
        """Reject coordinates outside 0-1 range."""
        with pytest.raises(ValidationError):
            AnnotationModel(class_id=0, x=1.5, y=0.5, width=0.3, height=0.4)

    def test_boundary_values(self):
        """Accept exact boundary values of 0 and 1."""
        ann = AnnotationModel(class_id=0, x=0.0, y=0.0, width=1.0, height=1.0)
        assert ann.x == 0.0
        assert ann.width == 1.0


class TestDatasetResponse:
    """Tests for DatasetResponse serialization."""

    def test_serialization(self):
        """Verify all fields serialize correctly."""
        resp = DatasetResponse(
            name="test-ds",
            path="/data/test-ds",
            classes=["cat", "dog"],
            num_images={"train": 10, "val": 2, "test": 1},
            created_at=1000.0,
            modified_at=2000.0,
        )
        data = resp.model_dump()
        assert data["name"] == "test-ds"
        assert data["num_images"]["train"] == 10
        assert data["created_at"] == 1000.0


class TestDatasetListResponse:
    """Tests for DatasetListResponse."""

    def test_empty_list(self):
        """Accept empty datasets list."""
        resp = DatasetListResponse(datasets=[], count=0)
        assert resp.count == 0
        assert resp.datasets == []


class TestImageResponse:
    """Tests for ImageResponse."""

    def test_serialization(self):
        """Verify image metadata fields serialize."""
        resp = ImageResponse(
            filename="img_001.jpg",
            split="train",
            width=640,
            height=480,
            size_bytes=12345,
            has_labels=True,
        )
        assert resp.filename == "img_001.jpg"
        assert resp.has_labels is True


class TestSaveLabelsRequest:
    """Tests for SaveLabelsRequest."""

    def test_empty_annotations(self):
        """Accept empty annotations list (clear labels)."""
        req = SaveLabelsRequest(annotations=[])
        assert req.annotations == []

    def test_with_annotations(self):
        """Accept list of valid annotations."""
        req = SaveLabelsRequest(
            annotations=[
                AnnotationModel(class_id=0, x=0.5, y=0.5, width=0.3, height=0.4),
                AnnotationModel(class_id=1, x=0.2, y=0.8, width=0.1, height=0.2),
            ]
        )
        assert len(req.annotations) == 2


class TestChangeSplitRequest:
    """Tests for ChangeSplitRequest."""

    def test_valid_request(self):
        """Accept valid split target."""
        req = ChangeSplitRequest(to_split="val")
        assert req.to_split == "val"

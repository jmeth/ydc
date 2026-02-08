"""
Unit tests for persistence domain models.

Verifies dataclass construction, default values, and field types
for Annotation, DatasetInfo, ImageInfo, LabelData, and ModelInfo.
"""

from pathlib import Path

from backend.persistence.models import (
    Annotation,
    DatasetInfo,
    ImageInfo,
    LabelData,
    ModelInfo,
)


class TestAnnotation:
    """Tests for the Annotation dataclass."""

    def test_construction_with_all_fields(self):
        """Annotation stores class_id and normalized coordinates."""
        ann = Annotation(class_id=2, x=0.5, y=0.6, width=0.3, height=0.4)
        assert ann.class_id == 2
        assert ann.x == 0.5
        assert ann.y == 0.6
        assert ann.width == 0.3
        assert ann.height == 0.4

    def test_integer_class_id(self):
        """class_id should be stored as provided (integer)."""
        ann = Annotation(class_id=0, x=0.0, y=0.0, width=0.0, height=0.0)
        assert ann.class_id == 0


class TestDatasetInfo:
    """Tests for the DatasetInfo dataclass."""

    def test_construction_with_required_fields(self):
        """DatasetInfo can be created with name, path, and classes."""
        info = DatasetInfo(name="my_dataset", path=Path("/data/my_dataset"), classes=["cat", "dog"])
        assert info.name == "my_dataset"
        assert info.path == Path("/data/my_dataset")
        assert info.classes == ["cat", "dog"]

    def test_default_num_images(self):
        """num_images defaults to zero counts for train/val/test."""
        info = DatasetInfo(name="ds", path=Path("/ds"), classes=[])
        assert info.num_images == {"train": 0, "val": 0, "test": 0}

    def test_default_timestamps(self):
        """created_at and modified_at default to 0.0."""
        info = DatasetInfo(name="ds", path=Path("/ds"), classes=[])
        assert info.created_at == 0.0
        assert info.modified_at == 0.0

    def test_default_num_images_is_independent_per_instance(self):
        """Each DatasetInfo instance gets its own num_images dict."""
        a = DatasetInfo(name="a", path=Path("/a"), classes=[])
        b = DatasetInfo(name="b", path=Path("/b"), classes=[])
        a.num_images["train"] = 10
        assert b.num_images["train"] == 0


class TestImageInfo:
    """Tests for the ImageInfo dataclass."""

    def test_construction_with_required_fields(self):
        """ImageInfo can be created with filename, split, and path."""
        info = ImageInfo(filename="img.jpg", split="train", path=Path("/data/img.jpg"))
        assert info.filename == "img.jpg"
        assert info.split == "train"
        assert info.path == Path("/data/img.jpg")

    def test_defaults(self):
        """width, height, size_bytes default to 0; has_labels defaults to False."""
        info = ImageInfo(filename="img.jpg", split="val", path=Path("/x"))
        assert info.width == 0
        assert info.height == 0
        assert info.size_bytes == 0
        assert info.has_labels is False


class TestLabelData:
    """Tests for the LabelData dataclass."""

    def test_construction_with_annotations(self):
        """LabelData holds filename, split, and a list of Annotations."""
        anns = [Annotation(class_id=0, x=0.5, y=0.5, width=0.1, height=0.1)]
        ld = LabelData(filename="img.jpg", split="train", annotations=anns)
        assert ld.filename == "img.jpg"
        assert ld.split == "train"
        assert len(ld.annotations) == 1

    def test_default_empty_annotations(self):
        """annotations defaults to an empty list."""
        ld = LabelData(filename="img.jpg", split="val")
        assert ld.annotations == []


class TestModelInfo:
    """Tests for the ModelInfo dataclass."""

    def test_construction_with_required_fields(self):
        """ModelInfo can be created with just name and path."""
        info = ModelInfo(name="my_model", path=Path("/models/my_model/best.pt"))
        assert info.name == "my_model"
        assert info.path == Path("/models/my_model/best.pt")

    def test_defaults(self):
        """Optional fields have sensible defaults."""
        info = ModelInfo(name="m", path=Path("/m"))
        assert info.base_model == ""
        assert info.dataset_name == ""
        assert info.created_at == 0.0
        assert info.epochs_completed == 0
        assert info.best_map50 is None
        assert info.is_active is False

    def test_full_construction(self):
        """ModelInfo with all fields populated."""
        info = ModelInfo(
            name="yolo_finetuned",
            path=Path("/models/yolo_finetuned/best.pt"),
            base_model="yolo11n",
            dataset_name="my_dataset",
            created_at=1700000000.0,
            epochs_completed=50,
            best_map50=0.85,
            is_active=True,
        )
        assert info.best_map50 == 0.85
        assert info.is_active is True
        assert info.epochs_completed == 50

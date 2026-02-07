# Dataset Subsystem

> Part of the [YOLO Dataset Creator Architecture](../ARCHITECTURE.md)

## Responsibility

Manage dataset business logic, annotations, and the review workflow. **Delegates all file operations to the [Persistence Layer](persistence.md).**

## Components

```
Dataset Subsystem
├── Dataset Manager
│   ├── Dataset CRUD (uses Persistence Layer)
│   ├── Class/Label configuration
│   └── Statistics aggregation
├── Annotation Manager
│   ├── Annotation CRUD (uses Persistence Layer)
│   ├── Auto-annotation marker ("auto" flag)
│   ├── Format conversion (normalized ↔ pixel)
│   └── Validation (bounds checking)
├── Review Queue
│   ├── Queue Manager (pending review items)
│   ├── Bulk Operations (accept/reject)
│   └── Filter/Sort (by confidence, class, time)
└── Export/Import
    ├── Zip Export (uses Persistence Layer)
    ├── Zip Import (uses Persistence Layer)
    └── Validation (schema checking)
```

## Dependency on Persistence Layer

The Dataset Subsystem does NOT directly access the filesystem. All storage operations go through the Persistence Layer:

```python
# dataset/manager.py
from persistence import DatasetStore, ImageStore, LabelStore

class DatasetManager:
    """
    Manages dataset business logic.
    Delegates all file operations to Persistence Layer.
    """

    def __init__(
        self,
        dataset_store: DatasetStore,
        image_store: ImageStore,
        label_store: LabelStore
    ):
        self._datasets = dataset_store
        self._images = image_store
        self._labels = label_store

    async def create_dataset(self, name: str, classes: list[str]) -> Dataset:
        """Create a new dataset with the given classes"""
        # Validate name
        if not self._is_valid_name(name):
            raise ValueError(f"Invalid dataset name: {name}")

        # Delegate storage to Persistence Layer
        dataset = await self._datasets.create(name, classes)
        return dataset

    async def add_image(
        self,
        dataset_name: str,
        image: np.ndarray,
        annotations: list[Annotation],
        split: str = "train"
    ) -> ImageInfo:
        """Add an image with annotations to a dataset"""
        # Generate unique filename
        filename = f"{uuid.uuid4().hex}.jpg"

        # Delegate image storage to Persistence Layer
        image_info = await self._images.save(
            dataset_name, split, filename, image
        )

        # Delegate label storage to Persistence Layer
        if annotations:
            await self._labels.save(
                dataset_name, split, filename, annotations
            )

        return image_info

    async def get_annotations(
        self,
        dataset_name: str,
        split: str,
        filename: str
    ) -> list[Annotation]:
        """Get annotations for an image"""
        # Delegate to Persistence Layer
        raw_labels = await self._labels.load(dataset_name, split, filename)

        # Apply business logic (convert format, validate)
        return self._parse_annotations(raw_labels)

    async def update_annotations(
        self,
        dataset_name: str,
        split: str,
        filename: str,
        annotations: list[Annotation]
    ) -> None:
        """Update annotations for an image"""
        # Validate annotations
        self._validate_annotations(annotations)

        # Delegate to Persistence Layer
        await self._labels.save(dataset_name, split, filename, annotations)

    async def delete_image(
        self,
        dataset_name: str,
        split: str,
        filename: str
    ) -> None:
        """Delete an image and its annotations"""
        # Delegate to Persistence Layer
        await self._images.delete(dataset_name, split, filename)
        await self._labels.delete(dataset_name, split, filename)

    async def change_split(
        self,
        dataset_name: str,
        filename: str,
        from_split: str,
        to_split: str
    ) -> None:
        """Move an image between splits"""
        # Delegate to Persistence Layer
        await self._images.move(dataset_name, filename, from_split, to_split)
        await self._labels.move(dataset_name, filename, from_split, to_split)
```

## Dataset Structure (Extended)

```
datasets/
└── my_dataset/
    ├── data.yaml              # YOLO config (existing)
    ├── prompts.yaml           # Class prompts for YOLO-World (new)
    ├── metadata.json          # Dataset metadata (new)
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── .review/               # Review queue metadata (new)
        └── queue.json
```

## New File Formats

**prompts.yaml**
```yaml
# Maps class IDs to text prompts for YOLO-World
classes:
  0:
    name: "vehicle"
    prompts:
      - "vehicle"
      - "car"
      - "truck"
      - "military vehicle"
  1:
    name: "person"
    prompts:
      - "person"
      - "human"
      - "soldier"
```

**metadata.json**
```json
{
  "created": "2024-01-15T10:30:00Z",
  "modified": "2024-01-15T14:22:00Z",
  "scan_sessions": [
    {
      "started": "2024-01-15T10:30:00Z",
      "ended": "2024-01-15T11:45:00Z",
      "frames_captured": 150,
      "config": { ... }
    }
  ],
  "stats": {
    "total_images": 200,
    "annotated": 180,
    "pending_review": 20,
    "by_class": { "0": 450, "1": 230 }
  }
}
```

**queue.json**
```json
{
  "pending": [
    {
      "image": "images/train/20240115_103045_abc123.jpg",
      "auto_annotations": [
        {"classId": 0, "x": 0.5, "y": 0.5, "width": 0.2, "height": 0.3, "confidence": 0.45, "auto": true}
      ],
      "captured_at": "2024-01-15T10:30:45Z",
      "low_confidence": true
    }
  ]
}
```

## Annotation Model (Extended)

```python
@dataclass
class Annotation:
    class_id: int
    x: float          # center x (normalized)
    y: float          # center y (normalized)
    width: float      # width (normalized)
    height: float     # height (normalized)
    confidence: float = None  # detection confidence (if auto)
    auto: bool = False        # auto-generated flag
```

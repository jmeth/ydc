# Persistence Layer

> Part of the [YOLO Dataset Creator Architecture](../ARCHITECTURE.md)

## Responsibility

Abstract all file system operations. Provides clean interfaces for data storage that other subsystems use. **No other subsystem should directly access the filesystem.**

## Components

```
Persistence Layer
├── Dataset Store
│   ├── Dataset CRUD (create, list, get, delete)
│   ├── Metadata management (data.yaml, prompts.yaml, metadata.json)
│   └── Export/Import (zip operations)
├── Image Store
│   ├── Image save/load/delete
│   ├── Image move (between splits)
│   ├── Thumbnail generation [Ideal]
│   └── Batch operations
├── Label Store
│   ├── Label save/load/delete
│   ├── Label move (between splits)
│   ├── YOLO format parsing/writing
│   └── Batch operations
├── Model Store
│   ├── Model file management
│   ├── Registry management (registry.json)
│   └── Model metadata (config, metrics)
├── Config Store
│   ├── App configuration
│   └── User preferences
└── Log Store
    ├── Training logs
    └── Capture session logs
```

## Store Interfaces

```python
# persistence/dataset_store.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetInfo:
    name: str
    path: Path
    classes: list[str]
    num_images: dict[str, int]  # split → count
    created_at: float
    modified_at: float

class DatasetStore(ABC):
    """Abstract interface for dataset storage operations"""

    @abstractmethod
    async def create(self, name: str, classes: list[str]) -> DatasetInfo:
        """Create a new dataset directory structure and data.yaml"""
        ...

    @abstractmethod
    async def list(self) -> list[DatasetInfo]:
        """List all datasets"""
        ...

    @abstractmethod
    async def get(self, name: str) -> DatasetInfo | None:
        """Get dataset info by name"""
        ...

    @abstractmethod
    async def delete(self, name: str) -> bool:
        """Delete a dataset and all its contents"""
        ...

    @abstractmethod
    async def update_classes(self, name: str, classes: list[str]) -> None:
        """Update dataset classes in data.yaml"""
        ...

    @abstractmethod
    async def get_prompts(self, name: str) -> dict[int, list[str]]:
        """Get YOLO-World prompts from prompts.yaml"""
        ...

    @abstractmethod
    async def save_prompts(self, name: str, prompts: dict[int, list[str]]) -> None:
        """Save YOLO-World prompts to prompts.yaml"""
        ...

    @abstractmethod
    async def export_zip(self, name: str, output_path: Path) -> Path:
        """Export dataset as zip file"""
        ...

    @abstractmethod
    async def import_zip(self, zip_path: Path, name: str | None = None) -> DatasetInfo:
        """Import dataset from zip file"""
        ...


# persistence/image_store.py
@dataclass
class ImageInfo:
    filename: str
    split: str
    path: Path
    width: int
    height: int
    size_bytes: int
    has_labels: bool

class ImageStore(ABC):
    """Abstract interface for image file operations"""

    @abstractmethod
    async def save(
        self,
        dataset: str,
        split: str,
        filename: str,
        image: np.ndarray
    ) -> ImageInfo:
        """Save image to dataset"""
        ...

    @abstractmethod
    async def load(self, dataset: str, split: str, filename: str) -> np.ndarray:
        """Load image from dataset"""
        ...

    @abstractmethod
    async def delete(self, dataset: str, split: str, filename: str) -> bool:
        """Delete image from dataset"""
        ...

    @abstractmethod
    async def move(
        self,
        dataset: str,
        filename: str,
        from_split: str,
        to_split: str
    ) -> None:
        """Move image between splits"""
        ...

    @abstractmethod
    async def list(
        self,
        dataset: str,
        split: str | None = None
    ) -> list[ImageInfo]:
        """List images in dataset, optionally filtered by split"""
        ...

    @abstractmethod
    async def exists(self, dataset: str, split: str, filename: str) -> bool:
        """Check if image exists"""
        ...


# persistence/label_store.py
@dataclass
class LabelData:
    filename: str
    split: str
    annotations: list[dict]  # Raw YOLO format annotations

class LabelStore(ABC):
    """Abstract interface for label file operations"""

    @abstractmethod
    async def save(
        self,
        dataset: str,
        split: str,
        filename: str,
        annotations: list[Annotation]
    ) -> None:
        """Save annotations in YOLO format"""
        ...

    @abstractmethod
    async def load(
        self,
        dataset: str,
        split: str,
        filename: str
    ) -> list[dict]:
        """Load annotations from YOLO format file"""
        ...

    @abstractmethod
    async def delete(self, dataset: str, split: str, filename: str) -> bool:
        """Delete label file"""
        ...

    @abstractmethod
    async def move(
        self,
        dataset: str,
        filename: str,
        from_split: str,
        to_split: str
    ) -> None:
        """Move label file between splits"""
        ...

    @abstractmethod
    async def exists(self, dataset: str, split: str, filename: str) -> bool:
        """Check if label file exists"""
        ...


# persistence/model_store.py
@dataclass
class ModelInfo:
    name: str
    path: Path
    base_model: str
    dataset_name: str
    created_at: float
    epochs_completed: int
    best_map50: float | None
    is_active: bool

class ModelStore(ABC):
    """Abstract interface for trained model storage"""

    @abstractmethod
    async def save(
        self,
        name: str,
        weights_path: Path,
        config: TrainingConfig,
        metrics: dict
    ) -> ModelInfo:
        """Save a trained model with its metadata"""
        ...

    @abstractmethod
    async def load(self, name: str) -> Path:
        """Get path to model weights file"""
        ...

    @abstractmethod
    async def list(self) -> list[ModelInfo]:
        """List all trained models"""
        ...

    @abstractmethod
    async def get(self, name: str) -> ModelInfo | None:
        """Get model info by name"""
        ...

    @abstractmethod
    async def delete(self, name: str) -> bool:
        """Delete a model and its files"""
        ...

    @abstractmethod
    async def set_active(self, name: str) -> None:
        """Set a model as the active model"""
        ...

    @abstractmethod
    async def get_active(self) -> ModelInfo | None:
        """Get the currently active model"""
        ...
```

## Filesystem Implementation (MVP)

```python
# persistence/filesystem/dataset_store.py
from persistence.dataset_store import DatasetStore, DatasetInfo
import yaml
import shutil

class FilesystemDatasetStore(DatasetStore):
    """Filesystem-based implementation of DatasetStore"""

    def __init__(self, base_path: Path):
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)

    async def create(self, name: str, classes: list[str]) -> DatasetInfo:
        dataset_path = self._base_path / name

        if dataset_path.exists():
            raise ValueError(f"Dataset '{name}' already exists")

        # Create directory structure
        dataset_path.mkdir()
        for split in ["train", "val", "test"]:
            (dataset_path / "images" / split).mkdir(parents=True)
            (dataset_path / "labels" / split).mkdir(parents=True)

        # Create data.yaml
        data_yaml = {
            "path": str(dataset_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(classes),
            "names": {i: name for i, name in enumerate(classes)}
        }

        with open(dataset_path / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f)

        return DatasetInfo(
            name=name,
            path=dataset_path,
            classes=classes,
            num_images={"train": 0, "val": 0, "test": 0},
            created_at=time.time(),
            modified_at=time.time()
        )

    async def delete(self, name: str) -> bool:
        dataset_path = self._base_path / name
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
            return True
        return False

    async def list(self) -> list[DatasetInfo]:
        datasets = []
        for path in self._base_path.iterdir():
            if path.is_dir() and (path / "data.yaml").exists():
                info = await self.get(path.name)
                if info:
                    datasets.append(info)
        return datasets

    # ... other methods


# persistence/filesystem/image_store.py
class FilesystemImageStore(ImageStore):
    """Filesystem-based implementation of ImageStore"""

    def __init__(self, base_path: Path):
        self._base_path = base_path

    async def save(
        self,
        dataset: str,
        split: str,
        filename: str,
        image: np.ndarray
    ) -> ImageInfo:
        path = self._base_path / dataset / "images" / split / filename

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save image
        cv2.imwrite(str(path), image)

        return ImageInfo(
            filename=filename,
            split=split,
            path=path,
            width=image.shape[1],
            height=image.shape[0],
            size_bytes=path.stat().st_size,
            has_labels=False
        )

    async def load(self, dataset: str, split: str, filename: str) -> np.ndarray:
        path = self._base_path / dataset / "images" / split / filename
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return image

    async def delete(self, dataset: str, split: str, filename: str) -> bool:
        path = self._base_path / dataset / "images" / split / filename
        if path.exists():
            path.unlink()
            return True
        return False

    async def move(
        self,
        dataset: str,
        filename: str,
        from_split: str,
        to_split: str
    ) -> None:
        src = self._base_path / dataset / "images" / from_split / filename
        dst = self._base_path / dataset / "images" / to_split / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    # ... other methods


# persistence/filesystem/label_store.py
class FilesystemLabelStore(LabelStore):
    """Filesystem-based implementation of LabelStore"""

    def __init__(self, base_path: Path):
        self._base_path = base_path

    def _label_path(self, dataset: str, split: str, image_filename: str) -> Path:
        """Get label file path for an image"""
        label_filename = Path(image_filename).stem + ".txt"
        return self._base_path / dataset / "labels" / split / label_filename

    async def save(
        self,
        dataset: str,
        split: str,
        filename: str,
        annotations: list[Annotation]
    ) -> None:
        path = self._label_path(dataset, split, filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write YOLO format: class_id x_center y_center width height
        lines = []
        for ann in annotations:
            lines.append(f"{ann.class_id} {ann.x:.6f} {ann.y:.6f} {ann.width:.6f} {ann.height:.6f}")

        with open(path, "w") as f:
            f.write("\n".join(lines))

    async def load(
        self,
        dataset: str,
        split: str,
        filename: str
    ) -> list[dict]:
        path = self._label_path(dataset, split, filename)

        if not path.exists():
            return []

        annotations = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    annotations.append({
                        "class_id": int(parts[0]),
                        "x": float(parts[1]),
                        "y": float(parts[2]),
                        "width": float(parts[3]),
                        "height": float(parts[4])
                    })
        return annotations

    # ... other methods
```

## Dependency Injection

```python
# persistence/__init__.py
from persistence.filesystem.dataset_store import FilesystemDatasetStore
from persistence.filesystem.image_store import FilesystemImageStore
from persistence.filesystem.label_store import FilesystemLabelStore

def create_stores(base_path: Path) -> tuple[DatasetStore, ImageStore, LabelStore]:
    """Factory function to create persistence stores"""
    return (
        FilesystemDatasetStore(base_path / "datasets"),
        FilesystemImageStore(base_path / "datasets"),
        FilesystemLabelStore(base_path / "datasets")
    )


# main.py - Dependency injection
from persistence import create_stores
from dataset.manager import DatasetManager

# Create stores
dataset_store, image_store, label_store = create_stores(Path("./data"))

# Inject into Dataset Subsystem
dataset_manager = DatasetManager(
    dataset_store=dataset_store,
    image_store=image_store,
    label_store=label_store
)
```

## Benefits of This Design

1. **Testability**: Dataset Subsystem can be tested with mock stores
2. **Flexibility**: Can swap filesystem for database (Ideal state) without changing Dataset Subsystem
3. **Single Responsibility**: Dataset Subsystem handles business logic, Persistence Layer handles storage
4. **Consistency**: All file operations go through one place

## Backend Directory Structure

```
backend/
├── main.py                 # FastAPI app entry point
├── requirements.txt
├── api/                    # API routes (FastAPI routers)
│   ├── __init__.py
│   ├── feeds.py
│   ├── inference.py
│   ├── capture/            # CaptureController (in API layer)
│   │   ├── __init__.py
│   │   ├── router.py       # FastAPI router for /api/capture/*
│   │   └── controller.py   # CaptureController logic
│   ├── datasets.py
│   ├── training.py
│   ├── models.py
│   ├── notifications.py
│   └── system.py
├── feeds/                  # Feeds subsystem
│   ├── __init__.py
│   ├── manager.py          # FeedManager (raw + derived feeds)
│   ├── base.py             # BaseFeed, Frame, FeedConfig
│   ├── derived.py          # DerivedFeed, InferenceFrame
│   ├── camera.py           # CameraFeed (MVP)
│   ├── rtsp.py             # RTSPFeed (Ideal)
│   ├── video.py            # VideoFileFeed (Ideal)
│   ├── images.py           # ImageFolderFeed (Ideal)
│   └── buffer.py           # RingBuffer
├── inference/              # Inference subsystem
│   ├── __init__.py
│   ├── manager.py          # InferenceManager (produces output feeds)
│   ├── session.py          # InferenceSession
│   ├── loader.py           # Model loading
│   └── detector.py         # YOLO-World and fine-tuned model detection
├── dataset/                # Dataset subsystem (uses Persistence Layer)
│   ├── __init__.py
│   ├── manager.py          # DatasetManager (business logic only)
│   ├── annotations.py      # Annotation validation/conversion
│   ├── review.py           # Review queue logic
│   └── export.py           # Export/import orchestration
├── training/               # Training subsystem (uses Persistence Layer)
│   ├── __init__.py
│   ├── runner.py           # Training execution
│   ├── monitor.py          # Resource monitoring
│   └── config.py           # Training configuration
├── persistence/            # Persistence Layer (all file operations)
│   ├── __init__.py         # Factory functions
│   ├── dataset_store.py    # DatasetStore interface
│   ├── image_store.py      # ImageStore interface
│   ├── label_store.py      # LabelStore interface
│   ├── model_store.py      # ModelStore interface
│   ├── config_store.py     # ConfigStore interface
│   └── filesystem/         # Filesystem implementations
│       ├── __init__.py
│       ├── dataset_store.py
│       ├── image_store.py
│       ├── label_store.py
│       └── model_store.py
├── notifications/          # Notifications subsystem
│   ├── __init__.py
│   ├── manager.py          # NotificationManager
│   ├── models.py           # Notification types
│   └── channels.py         # WebSocket, Desktop, Webhook [Ideal]
├── auth/                   # Auth subsystem [Ideal]
│   ├── __init__.py
│   ├── manager.py          # AuthManager
│   ├── models.py           # User, Role, Permission
│   ├── dependencies.py     # FastAPI auth dependencies
│   └── rbac.py             # Role-based access control
├── websocket/              # WebSocket handlers
│   ├── __init__.py
│   ├── manager.py          # Connection manager
│   ├── video.py            # Video stream handler
│   └── events.py           # Event broadcast handler
├── core/                   # Shared utilities
│   ├── __init__.py
│   ├── events.py           # Pub/sub for subsystems
│   ├── resources.py        # Resource monitoring
│   ├── config.py           # Pydantic settings
│   └── exceptions.py       # Custom exceptions
└── models/                 # Pydantic models (API schemas)
    ├── __init__.py
    ├── feeds.py
    ├── scan.py
    ├── dataset.py
    ├── training.py
    └── common.py
```

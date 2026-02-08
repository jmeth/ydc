# Using the Persistence Layer

This guide covers how to use the persistence layer from other subsystems (dataset manager, capture controller, training runner, API routers, etc.).

## Overview

The persistence layer provides four stores accessed via dependency injection. You never touch the filesystem directly — always go through a store. This keeps file operations centralized, testable, and swappable.

| Store | What it manages | Accessor |
|-------|----------------|----------|
| `DatasetStore` | Dataset CRUD, classes, prompts, zip export/import | `get_dataset_store()` |
| `ImageStore` | Image save/load/delete/move within dataset splits | `get_image_store()` |
| `LabelStore` | YOLO-format annotation files | `get_label_store()` |
| `ModelStore` | Trained model weights and metadata | `get_model_store()` |

## Getting a store

Import the convenience accessor from `backend.persistence`:

```python
from backend.persistence import get_dataset_store, get_image_store

async def my_handler():
    ds = get_dataset_store()
    datasets = await ds.list()
```

The stores are initialized during app startup (`backend/main.py` lifespan) and available for the lifetime of the app. Calling a getter before startup raises `RuntimeError`.

### In an API router

```python
from fastapi import APIRouter
from backend.persistence import get_dataset_store

router = APIRouter()

@router.get("/datasets")
async def list_datasets():
    ds = get_dataset_store()
    datasets = await ds.list()
    return {"datasets": [d.name for d in datasets]}
```

### In a subsystem class (constructor injection)

For classes that need stores throughout their lifetime, accept them as constructor args:

```python
from backend.persistence.dataset_store import DatasetStore
from backend.persistence.image_store import ImageStore
from backend.persistence.label_store import LabelStore


class DatasetManager:
    """Business logic for dataset operations."""

    def __init__(
        self,
        dataset_store: DatasetStore,
        image_store: ImageStore,
        label_store: LabelStore,
    ) -> None:
        self._datasets = dataset_store
        self._images = image_store
        self._labels = label_store
```

Wire it up in the lifespan:

```python
# In backend/main.py lifespan, after set_stores():
from backend.persistence import get_dataset_store, get_image_store, get_label_store

dataset_manager = DatasetManager(
    dataset_store=get_dataset_store(),
    image_store=get_image_store(),
    label_store=get_label_store(),
)
```

## Common operations

### Create a dataset

```python
ds = get_dataset_store()
info = await ds.create("my_dataset", classes=["cat", "dog", "bird"])
# info.name == "my_dataset"
# info.num_images == {"train": 0, "val": 0, "test": 0}
```

### Save a captured image with annotations

```python
import numpy as np
from backend.persistence import get_image_store, get_label_store
from backend.persistence.models import Annotation

img_store = get_image_store()
lbl_store = get_label_store()

# frame is an np.ndarray (BGR, HWC) from a feed
image_info = await img_store.save("my_dataset", "train", "frame_001.jpg", frame)

# Save detected annotations
annotations = [
    Annotation(class_id=0, x=0.45, y=0.50, width=0.30, height=0.60),
    Annotation(class_id=1, x=0.80, y=0.20, width=0.10, height=0.15),
]
await lbl_store.save("my_dataset", "train", "frame_001.jpg", annotations)
```

### List images and check for labels

```python
img_store = get_image_store()

# List all images across splits
all_images = await img_store.list("my_dataset")

# List only training images
train_images = await img_store.list("my_dataset", split="train")

for img in train_images:
    print(f"{img.filename}: {img.size_bytes} bytes, labeled={img.has_labels}")
```

Note: `list()` sets `width=height=0` to avoid loading every file. Call `load()` for actual dimensions:

```python
image_data = await img_store.load("my_dataset", "train", "frame_001.jpg")
h, w = image_data.shape[:2]
```

### Move images between splits

```python
img_store = get_image_store()
lbl_store = get_label_store()

# Move image and its labels from train to val
await img_store.move("my_dataset", "frame_001.jpg", "train", "val")
await lbl_store.move("my_dataset", "frame_001.jpg", "train", "val")
```

### Load and inspect annotations

```python
lbl_store = get_label_store()

annotations = await lbl_store.load("my_dataset", "train", "frame_001.jpg")
for ann in annotations:
    print(f"class={ann.class_id} center=({ann.x}, {ann.y}) size=({ann.width}x{ann.height})")
```

### Update dataset classes

```python
ds = get_dataset_store()
await ds.update_classes("my_dataset", ["person", "car", "truck"])
```

### Manage YOLO-World prompts

```python
ds = get_dataset_store()

# Save prompts (class_id → list of text prompts)
await ds.save_prompts("my_dataset", {
    0: ["a person", "a human"],
    1: ["a car", "an automobile"],
})

# Load prompts
prompts = await ds.get_prompts("my_dataset")
# {0: ["a person", "a human"], 1: ["a car", "an automobile"]}
```

### Export and import datasets

```python
from pathlib import Path

ds = get_dataset_store()

# Export to zip
zip_path = await ds.export_zip("my_dataset", output_path=Path("/tmp/exports"))
# zip_path == Path("/tmp/exports/my_dataset.zip")

# Import from zip
imported = await ds.import_zip(zip_path, name="imported_copy")
```

### Save a trained model

```python
from pathlib import Path
from backend.persistence import get_model_store

mdl = get_model_store()

# After training completes, save the best weights
info = await mdl.save(
    name="yolo_cats_v1",
    weights_path=Path("runs/detect/train/weights/best.pt"),
    base_model="yolo11n",
    dataset_name="my_dataset",
    epochs_completed=50,
    metrics={"best_map50": 0.87},
)

# Set as the active model
await mdl.set_active("yolo_cats_v1")
```

### Load and list models

```python
mdl = get_model_store()

# List all trained models
models = await mdl.list()
for m in models:
    print(f"{m.name}: mAP50={m.best_map50}, active={m.is_active}")

# Get the active model's weights path
active = await mdl.get_active()
if active:
    weights_path = await mdl.load(active.name)
    # Pass weights_path to ultralytics for inference
```

## Testing with stores

### Using the conftest fixture

The shared `persistence_stores` fixture in `backend/tests/conftest.py` provides stores backed by `tmp_path`:

```python
async def test_my_feature(persistence_stores):
    ds = persistence_stores.dataset
    info = await ds.create("test_ds", ["a", "b"])
    assert info.name == "test_ds"
```

### Mocking stores in unit tests

For unit-testing subsystem classes, mock the store interfaces:

```python
from unittest.mock import AsyncMock, MagicMock
from backend.persistence.models import DatasetInfo
from pathlib import Path


def test_dataset_manager_lists_datasets():
    mock_ds = AsyncMock()
    mock_ds.list.return_value = [
        DatasetInfo(name="ds1", path=Path("/ds1"), classes=["cat"]),
    ]

    manager = DatasetManager(
        dataset_store=mock_ds,
        image_store=AsyncMock(),
        label_store=AsyncMock(),
    )

    # Test your business logic...
    mock_ds.list.assert_called_once()
```

### Integration tests with real stores

For integration tests that need real filesystem behavior, use `create_stores()` with `tmp_path`:

```python
from backend.persistence import create_stores

async def test_full_capture_workflow(tmp_path):
    stores = create_stores(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
    )

    # Create dataset
    info = await stores.dataset.create("test", ["person"])

    # Save image + labels
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    await stores.image.save("test", "train", "frame.jpg", frame)
    await stores.label.save("test", "train", "frame.jpg", [
        Annotation(class_id=0, x=0.5, y=0.5, width=0.3, height=0.4),
    ])

    # Verify
    images = await stores.image.list("test", split="train")
    assert len(images) == 1
    assert images[0].has_labels is True
```

## Domain models quick reference

```python
from backend.persistence.models import (
    Annotation,     # class_id, x, y, width, height
    DatasetInfo,    # name, path, classes, num_images, created_at, modified_at
    ImageInfo,      # filename, split, path, width, height, size_bytes, has_labels
    LabelData,      # filename, split, annotations
    ModelInfo,      # name, path, base_model, dataset_name, created_at,
                    # epochs_completed, best_map50, is_active
)
```

## Error handling

Stores raise standard Python exceptions:

| Exception | When |
|-----------|------|
| `ValueError` | Creating a dataset/model with a name that already exists |
| `FileNotFoundError` | Loading/moving/updating an item that doesn't exist |
| `RuntimeError` | Calling `get_stores()` before app startup |

Handle these in your API routers using the existing exception handlers in `backend/main.py`, or catch them in your business logic:

```python
from backend.persistence import get_dataset_store

async def create_dataset(name: str, classes: list[str]):
    ds = get_dataset_store()
    try:
        return await ds.create(name, classes)
    except ValueError:
        raise ConflictError(f"Dataset '{name}' already exists")
```

## Rules

1. **Never access the filesystem directly** — always use a store
2. **Always `await` store methods** — they are all `async def`
3. **Use the ABC type in signatures** — accept `DatasetStore`, not `FilesystemDatasetStore`
4. **Use accessors in routers, injection in classes** — `get_dataset_store()` for simple access, constructor args for subsystem classes
5. **Move images and labels together** — when moving between splits, call both `image_store.move()` and `label_store.move()`

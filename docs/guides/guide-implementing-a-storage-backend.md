# Implementing a New Storage Backend

This guide walks through adding an alternative storage backend (e.g., SQLite, S3) for the persistence layer. The MVP ships with filesystem-based stores; you can swap in a different backend by implementing the same abstract interfaces.

## Overview

The persistence layer has four abstract store interfaces (`DatasetStore`, `ImageStore`, `LabelStore`, `ModelStore`) in `backend/persistence/`. The filesystem implementations live in `backend/persistence/filesystem/`. To add a new backend you create a parallel package with classes that subclass the same ABCs, then update the factory.

## Architecture

```
backend/persistence/
├── __init__.py              # Factory + DI (create_stores, set_stores, get_stores)
├── models.py                # Shared dataclasses (Annotation, DatasetInfo, etc.)
├── dataset_store.py         # DatasetStore ABC
├── image_store.py           # ImageStore ABC
├── label_store.py           # LabelStore ABC
├── model_store.py           # ModelStore ABC
├── filesystem/              # MVP filesystem backend
│   ├── dataset_store.py
│   ├── image_store.py
│   ├── label_store.py
│   └── model_store.py
└── yourbackend/             # <-- your new backend
    ├── __init__.py
    ├── dataset_store.py
    ├── image_store.py
    ├── label_store.py
    └── model_store.py
```

## Steps

### 1. Create the backend package

```bash
mkdir -p backend/persistence/sqlite
touch backend/persistence/sqlite/__init__.py
```

### 2. Implement LabelStore (start simple)

The `LabelStore` is the simplest — no dependencies on cv2 or yaml. Start here to validate your approach.

Create `backend/persistence/sqlite/label_store.py`:

```python
"""SQLite-based implementation of LabelStore."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.persistence.label_store import LabelStore
from backend.persistence.models import Annotation


class SqliteLabelStore(LabelStore):
    """
    SQLite-backed label storage.

    Stores annotations in a `labels` table instead of .txt files.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create the labels table if it doesn't exist."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS labels (
                    dataset TEXT, split TEXT, filename TEXT,
                    class_id INTEGER, x REAL, y REAL, width REAL, height REAL
                )
            """)

    async def save(
        self,
        dataset: str,
        split: str,
        filename: str,
        annotations: list[Annotation],
    ) -> None:
        """Write annotations to the database."""
        label_name = Path(filename).stem + ".txt"
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "DELETE FROM labels WHERE dataset=? AND split=? AND filename=?",
                (dataset, split, label_name),
            )
            conn.executemany(
                "INSERT INTO labels VALUES (?,?,?,?,?,?,?,?)",
                [(dataset, split, label_name, a.class_id, a.x, a.y, a.width, a.height)
                 for a in annotations],
            )

    async def load(
        self,
        dataset: str,
        split: str,
        filename: str,
    ) -> list[Annotation]:
        """Read annotations from the database."""
        label_name = Path(filename).stem + ".txt"
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT class_id, x, y, width, height FROM labels "
                "WHERE dataset=? AND split=? AND filename=?",
                (dataset, split, label_name),
            ).fetchall()
        return [Annotation(*row) for row in rows]

    async def delete(self, dataset: str, split: str, filename: str) -> bool:
        """Delete annotations for an image."""
        label_name = Path(filename).stem + ".txt"
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM labels WHERE dataset=? AND split=? AND filename=?",
                (dataset, split, label_name),
            )
        return cursor.rowcount > 0

    async def move(
        self,
        dataset: str,
        filename: str,
        from_split: str,
        to_split: str,
    ) -> None:
        """Move annotations between splits."""
        label_name = Path(filename).stem + ".txt"
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "UPDATE labels SET split=? WHERE dataset=? AND split=? AND filename=?",
                (to_split, dataset, from_split, label_name),
            )
        if cursor.rowcount == 0:
            raise FileNotFoundError(f"No labels for {filename} in {from_split}")

    async def exists(self, dataset: str, split: str, filename: str) -> bool:
        """Check if annotations exist."""
        label_name = Path(filename).stem + ".txt"
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM labels WHERE dataset=? AND split=? AND filename=? LIMIT 1",
                (dataset, split, label_name),
            ).fetchone()
        return row is not None
```

#### Contract requirements

Every store ABC method is `async def`. All four stores share these rules:

| Rule | Detail |
|------|--------|
| All methods are async | Use `async def` even if the backend is synchronous |
| Return the correct types | See the ABC docstrings for exact return types |
| Use `from __future__ import annotations` | Required because the `list()` method name shadows the builtin |
| Raise the documented exceptions | `ValueError` for duplicates, `FileNotFoundError` for missing items |
| Accept the same parameters | Don't change signatures — other subsystems depend on them |

### 3. Implement the remaining stores

Work through each ABC in order of complexity:

| Store | ABC file | Key notes |
|-------|----------|-----------|
| `LabelStore` | `label_store.py` | Simplest — text data only |
| `ImageStore` | `image_store.py` | Must handle `np.ndarray` images (BGR, HWC format) |
| `ModelStore` | `model_store.py` | Must copy/manage weights files, track active model |
| `DatasetStore` | `dataset_store.py` | Most complex — CRUD, classes, prompts, zip import/export |

For each store, refer to the ABC file for the full method list and docstrings.

### 4. Update the factory

In `backend/persistence/__init__.py`, add a factory function or modify `create_stores()`:

```python
from backend.persistence.sqlite.dataset_store import SqliteDatasetStore
from backend.persistence.sqlite.image_store import SqliteImageStore
from backend.persistence.sqlite.label_store import SqliteLabelStore
from backend.persistence.sqlite.model_store import SqliteModelStore


def create_sqlite_stores(db_path: Path, models_dir: Path) -> Stores:
    """Factory that creates SQLite-backed store instances."""
    return Stores(
        dataset=SqliteDatasetStore(db_path),
        image=SqliteImageStore(db_path),
        label=SqliteLabelStore(db_path),
        model=SqliteModelStore(db_path, models_dir),
    )
```

Then update `backend/main.py` to call your factory instead of `create_stores()`:

```python
# In lifespan:
from backend.persistence import create_sqlite_stores, set_stores

stores = create_sqlite_stores(
    db_path=Path(settings.data_dir) / "ydc.db",
    models_dir=Path(settings.models_dir),
)
set_stores(stores)
```

The rest of the app doesn't change — all consumers use the ABC interfaces via `get_dataset_store()`, `get_image_store()`, etc.

### 5. Write tests

Create `backend/tests/test_persistence_sqlite_*.py` files. The existing filesystem tests are a good template — the assertions should be identical since the ABC contract is the same. Key patterns:

```python
import pytest
from backend.persistence.sqlite.label_store import SqliteLabelStore
from backend.persistence.models import Annotation


def _make_store(tmp_path):
    """Create a SqliteLabelStore backed by a temp database."""
    return SqliteLabelStore(tmp_path / "test.db")


class TestSqliteLabelStoreSave:
    async def test_save_and_load_round_trip(self, tmp_path):
        store = _make_store(tmp_path)
        anns = [Annotation(class_id=0, x=0.5, y=0.5, width=0.3, height=0.4)]
        await store.save("ds1", "train", "img.jpg", anns)
        loaded = await store.load("ds1", "train", "img.jpg")
        assert len(loaded) == 1
        assert loaded[0].class_id == 0
```

Use `tmp_path` for all database files so tests are isolated.

### 6. Verify

```bash
# Run your new tests
python -m pytest backend/tests/test_persistence_sqlite_*.py -v

# Run the full suite to check nothing broke
python -m pytest backend/tests/ -v
```

## Domain models reference

All stores use the dataclasses from `backend/persistence/models.py`:

| Dataclass | Used by | Fields |
|-----------|---------|--------|
| `Annotation` | LabelStore | `class_id`, `x`, `y`, `width`, `height` |
| `DatasetInfo` | DatasetStore | `name`, `path`, `classes`, `num_images`, `created_at`, `modified_at` |
| `ImageInfo` | ImageStore | `filename`, `split`, `path`, `width`, `height`, `size_bytes`, `has_labels` |
| `LabelData` | (convenience) | `filename`, `split`, `annotations` |
| `ModelInfo` | ModelStore | `name`, `path`, `base_model`, `dataset_name`, `created_at`, `epochs_completed`, `best_map50`, `is_active` |

## Gotchas

- **`list` shadows builtin**: Every ABC file that has a `list()` method needs `from __future__ import annotations` at the top. Your implementations need it too.
- **ImageStore.list() skips dimensions**: Return `width=0, height=0` in `list()` to avoid loading every image. Consumers call `load()` when they need actual dimensions.
- **ModelStore.save() signature**: Takes individual fields (`base_model`, `dataset_name`, `epochs_completed`, `metrics`) — there's no `TrainingConfig` object yet.
- **Async wrapping**: All methods must be `async def`. For synchronous backends this is fine — just use sync calls inside async methods. For truly async backends (e.g., async database drivers), use `await` natively.

## File checklist

| Action | File |
|--------|------|
| Create package | `backend/persistence/yourbackend/__init__.py` |
| Implement LabelStore | `backend/persistence/yourbackend/label_store.py` |
| Implement ImageStore | `backend/persistence/yourbackend/image_store.py` |
| Implement DatasetStore | `backend/persistence/yourbackend/dataset_store.py` |
| Implement ModelStore | `backend/persistence/yourbackend/model_store.py` |
| Add factory function | `backend/persistence/__init__.py` |
| Update lifespan | `backend/main.py` |
| Write tests | `backend/tests/test_persistence_yourbackend_*.py` |

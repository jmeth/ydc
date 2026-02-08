"""
Filesystem-based implementation of DatasetStore.

Manages dataset directory structures with YOLO conventions:
  {base_path}/{name}/
    ├── data.yaml         # YOLO dataset config
    ├── prompts.yaml      # YOLO-World prompts (optional)
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/

Uses PyYAML for data.yaml/prompts.yaml and zipfile for export/import.
"""

from __future__ import annotations

import shutil
import time
import zipfile
from pathlib import Path

import yaml

from backend.persistence.dataset_store import DatasetStore
from backend.persistence.models import DatasetInfo

SPLITS = ("train", "val", "test")

# Image extensions to count when scanning for num_images
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class FilesystemDatasetStore(DatasetStore):
    """
    Filesystem-based dataset storage.

    Creates and manages YOLO-format dataset directories.

    Args:
        base_path: Root directory where all datasets are stored.
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)

    async def create(self, name: str, classes: list[str]) -> DatasetInfo:
        """Create a new dataset with directory structure and data.yaml."""
        dataset_path = self._base_path / name

        if dataset_path.exists():
            raise ValueError(f"Dataset '{name}' already exists")

        # Create directory structure
        dataset_path.mkdir()
        for split in SPLITS:
            (dataset_path / "images" / split).mkdir(parents=True)
            (dataset_path / "labels" / split).mkdir(parents=True)

        # Create data.yaml following YOLO convention
        now = time.time()
        data_yaml = {
            "path": str(dataset_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(classes),
            "names": {i: c for i, c in enumerate(classes)},
        }

        with open(dataset_path / "data.yaml", "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        return DatasetInfo(
            name=name,
            path=dataset_path,
            classes=classes,
            num_images={"train": 0, "val": 0, "test": 0},
            created_at=now,
            modified_at=now,
        )

    async def list(self) -> list[DatasetInfo]:
        """List all datasets found under the base path."""
        datasets: list[DatasetInfo] = []
        if not self._base_path.exists():
            return datasets

        for path in sorted(self._base_path.iterdir()):
            if path.is_dir() and (path / "data.yaml").exists():
                info = await self.get(path.name)
                if info:
                    datasets.append(info)
        return datasets

    async def get(self, name: str) -> DatasetInfo | None:
        """Get dataset info by parsing its data.yaml and counting images."""
        dataset_path = self._base_path / name
        data_yaml_path = dataset_path / "data.yaml"

        if not data_yaml_path.exists():
            return None

        with open(data_yaml_path) as f:
            data = yaml.safe_load(f)

        # Extract class names from the names dict
        names_dict = data.get("names", {})
        classes = [names_dict[k] for k in sorted(names_dict.keys())]

        # Count images per split
        num_images: dict[str, int] = {}
        for split in SPLITS:
            images_dir = dataset_path / "images" / split
            if images_dir.exists():
                count = sum(
                    1 for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )
                num_images[split] = count
            else:
                num_images[split] = 0

        # Use directory mtime as modified_at, ctime as created_at
        stat = dataset_path.stat()
        return DatasetInfo(
            name=name,
            path=dataset_path,
            classes=classes,
            num_images=num_images,
            created_at=stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_ctime,
            modified_at=stat.st_mtime,
        )

    async def delete(self, name: str) -> bool:
        """Delete a dataset directory and all its contents."""
        dataset_path = self._base_path / name
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
            return True
        return False

    async def update_classes(self, name: str, classes: list[str]) -> None:
        """Update the class names in data.yaml."""
        dataset_path = self._base_path / name
        data_yaml_path = dataset_path / "data.yaml"

        if not data_yaml_path.exists():
            raise FileNotFoundError(f"Dataset '{name}' not found")

        with open(data_yaml_path) as f:
            data = yaml.safe_load(f)

        data["nc"] = len(classes)
        data["names"] = {i: c for i, c in enumerate(classes)}

        with open(data_yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    async def get_prompts(self, name: str) -> dict[int, list[str]]:
        """Read YOLO-World prompts from prompts.yaml."""
        prompts_path = self._base_path / name / "prompts.yaml"

        if not prompts_path.exists():
            return {}

        with open(prompts_path) as f:
            data = yaml.safe_load(f)

        if not data:
            return {}

        # Ensure keys are ints and values are lists of strings
        return {int(k): v for k, v in data.items()}

    async def save_prompts(self, name: str, prompts: dict[int, list[str]]) -> None:
        """Write YOLO-World prompts to prompts.yaml."""
        dataset_path = self._base_path / name

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset '{name}' not found")

        with open(dataset_path / "prompts.yaml", "w") as f:
            yaml.dump(dict(prompts), f, default_flow_style=False)

    async def export_zip(self, name: str, output_path: Path) -> Path:
        """Export a dataset as a zip file containing all files."""
        dataset_path = self._base_path / name

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset '{name}' not found")

        output_path.mkdir(parents=True, exist_ok=True)
        zip_path = output_path / f"{name}.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(dataset_path.rglob("*")):
                if file_path.is_file():
                    arcname = file_path.relative_to(dataset_path)
                    zf.write(file_path, arcname)

        return zip_path

    async def import_zip(self, zip_path: Path, name: str | None = None) -> DatasetInfo:
        """Import a dataset from a zip file."""
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        resolved_name = name or zip_path.stem
        dataset_path = self._base_path / resolved_name

        if dataset_path.exists():
            raise ValueError(f"Dataset '{resolved_name}' already exists")

        dataset_path.mkdir(parents=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dataset_path)

        # Update the path field in data.yaml to point to the new location
        data_yaml_path = dataset_path / "data.yaml"
        if data_yaml_path.exists():
            with open(data_yaml_path) as f:
                data = yaml.safe_load(f)
            data["path"] = str(dataset_path.absolute())
            with open(data_yaml_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)

        info = await self.get(resolved_name)
        if info is None:
            raise ValueError(f"Imported zip does not contain a valid dataset (missing data.yaml)")
        return info

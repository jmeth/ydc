"""
Filesystem-based implementation of ImageStore.

Saves and loads images using cv2.imwrite/imread. Images are stored
under `{base_path}/{dataset}/images/{split}/`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from backend.persistence.image_store import ImageStore
from backend.persistence.models import ImageInfo

# Supported image extensions for listing
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class FilesystemImageStore(ImageStore):
    """
    Filesystem-based image storage using OpenCV.

    Args:
        base_path: Root directory containing dataset directories.
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def _image_path(self, dataset: str, split: str, filename: str) -> Path:
        """Get the full path for an image file."""
        return self._base_path / dataset / "images" / split / filename

    async def save(
        self,
        dataset: str,
        split: str,
        filename: str,
        image: np.ndarray,
    ) -> ImageInfo:
        """Save an image to disk using cv2.imwrite."""
        path = self._image_path(dataset, split, filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(path), image)

        h, w = image.shape[:2]
        return ImageInfo(
            filename=filename,
            split=split,
            path=path,
            width=w,
            height=h,
            size_bytes=path.stat().st_size,
            has_labels=False,
        )

    async def load(self, dataset: str, split: str, filename: str) -> np.ndarray:
        """Load an image from disk using cv2.imread."""
        path = self._image_path(dataset, split, filename)
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return image

    async def delete(self, dataset: str, split: str, filename: str) -> bool:
        """Delete an image file. Returns True if file existed."""
        path = self._image_path(dataset, split, filename)
        if path.exists():
            path.unlink()
            return True
        return False

    async def move(
        self,
        dataset: str,
        filename: str,
        from_split: str,
        to_split: str,
    ) -> None:
        """Move an image between splits within the same dataset."""
        src = self._image_path(dataset, from_split, filename)
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {src}")

        dst = self._image_path(dataset, to_split, filename)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    async def list(
        self,
        dataset: str,
        split: str | None = None,
    ) -> list[ImageInfo]:
        """
        List images in a dataset, optionally filtered by split.

        Sets width=height=0 to avoid loading every image file.
        Checks for a corresponding label file to set has_labels.
        """
        splits = [split] if split else ["train", "val", "test"]
        results: list[ImageInfo] = []

        for s in splits:
            images_dir = self._base_path / dataset / "images" / s
            if not images_dir.exists():
                continue

            labels_dir = self._base_path / dataset / "labels" / s

            for path in sorted(images_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    label_path = labels_dir / (path.stem + ".txt")
                    results.append(
                        ImageInfo(
                            filename=path.name,
                            split=s,
                            path=path,
                            width=0,
                            height=0,
                            size_bytes=path.stat().st_size,
                            has_labels=label_path.exists(),
                        )
                    )

        return results

    async def exists(self, dataset: str, split: str, filename: str) -> bool:
        """Check if an image file exists."""
        return self._image_path(dataset, split, filename).exists()

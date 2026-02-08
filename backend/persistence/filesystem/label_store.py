"""
Filesystem-based implementation of LabelStore.

Reads and writes YOLO-format annotation files (.txt) with one
annotation per line: `class_id x_center y_center width height`.
Label files live under `{base_path}/{dataset}/labels/{split}/`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from backend.persistence.label_store import LabelStore
from backend.persistence.models import Annotation


class FilesystemLabelStore(LabelStore):
    """
    Filesystem-based label storage using YOLO .txt format.

    Args:
        base_path: Root directory containing dataset directories.
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def _label_path(self, dataset: str, split: str, image_filename: str) -> Path:
        """
        Derive the label file path from an image filename.

        Replaces the image extension with .txt and places it under
        the labels/{split}/ directory.
        """
        label_filename = Path(image_filename).stem + ".txt"
        return self._base_path / dataset / "labels" / split / label_filename

    async def save(
        self,
        dataset: str,
        split: str,
        filename: str,
        annotations: list[Annotation],
    ) -> None:
        """Write annotations to a YOLO-format .txt file."""
        path = self._label_path(dataset, split, filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for ann in annotations:
            lines.append(
                f"{ann.class_id} {ann.x:.6f} {ann.y:.6f} "
                f"{ann.width:.6f} {ann.height:.6f}"
            )

        path.write_text("\n".join(lines))

    async def load(
        self,
        dataset: str,
        split: str,
        filename: str,
    ) -> list[Annotation]:
        """Read annotations from a YOLO-format .txt file."""
        path = self._label_path(dataset, split, filename)

        if not path.exists():
            return []

        annotations: list[Annotation] = []
        for line in path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                annotations.append(
                    Annotation(
                        class_id=int(parts[0]),
                        x=float(parts[1]),
                        y=float(parts[2]),
                        width=float(parts[3]),
                        height=float(parts[4]),
                    )
                )

        return annotations

    async def delete(self, dataset: str, split: str, filename: str) -> bool:
        """Delete a label file. Returns True if file existed."""
        path = self._label_path(dataset, split, filename)
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
        """Move a label file between splits."""
        src = self._label_path(dataset, from_split, filename)
        if not src.exists():
            raise FileNotFoundError(f"Label file not found: {src}")

        dst = self._label_path(dataset, to_split, filename)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    async def exists(self, dataset: str, split: str, filename: str) -> bool:
        """Check if a label file exists for the given image."""
        return self._label_path(dataset, split, filename).exists()

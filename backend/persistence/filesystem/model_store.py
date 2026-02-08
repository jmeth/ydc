"""
Filesystem-based implementation of ModelStore.

Stores trained model weights and metadata under:
  {base_path}/{name}/best.pt

A registry.json file at {base_path}/registry.json tracks metadata
for all models (base_model, dataset, epochs, metrics, active flag).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from backend.persistence.model_store import ModelStore
from backend.persistence.models import ModelInfo


class FilesystemModelStore(ModelStore):
    """
    Filesystem-based model storage with a JSON registry.

    Args:
        base_path: Root directory where all model directories are stored.
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._base_path / "registry.json"

    def _load_registry(self) -> dict[str, dict]:
        """Load the model registry from disk."""
        if not self._registry_path.exists():
            return {}
        with open(self._registry_path) as f:
            return json.load(f)

    def _save_registry(self, registry: dict[str, dict]) -> None:
        """Write the model registry to disk."""
        with open(self._registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    def _to_model_info(self, name: str, entry: dict) -> ModelInfo:
        """Convert a registry entry dict into a ModelInfo dataclass."""
        return ModelInfo(
            name=name,
            path=Path(entry["path"]),
            base_model=entry.get("base_model", ""),
            dataset_name=entry.get("dataset_name", ""),
            created_at=entry.get("created_at", 0.0),
            epochs_completed=entry.get("epochs_completed", 0),
            best_map50=entry.get("best_map50"),
            is_active=entry.get("is_active", False),
        )

    async def save(
        self,
        name: str,
        weights_path: Path,
        base_model: str = "",
        dataset_name: str = "",
        epochs_completed: int = 0,
        metrics: dict | None = None,
    ) -> ModelInfo:
        """Copy weights into managed storage and register metadata."""
        registry = self._load_registry()

        if name in registry:
            raise ValueError(f"Model '{name}' already exists")

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        # Create model directory and copy weights
        model_dir = self._base_path / name
        model_dir.mkdir(parents=True, exist_ok=True)
        dest = model_dir / "best.pt"
        shutil.copy2(str(weights_path), str(dest))

        now = time.time()
        best_map50 = (metrics or {}).get("best_map50")

        entry = {
            "path": str(dest),
            "base_model": base_model,
            "dataset_name": dataset_name,
            "created_at": now,
            "epochs_completed": epochs_completed,
            "best_map50": best_map50,
            "is_active": False,
        }

        registry[name] = entry
        self._save_registry(registry)

        return self._to_model_info(name, entry)

    async def load(self, name: str) -> Path:
        """Get the path to a model's weights file."""
        registry = self._load_registry()

        if name not in registry:
            raise FileNotFoundError(f"Model '{name}' not found in registry")

        weights_path = Path(registry[name]["path"])
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file missing: {weights_path}")

        return weights_path

    async def list(self) -> list[ModelInfo]:
        """List all registered models."""
        registry = self._load_registry()
        return [self._to_model_info(name, entry) for name, entry in sorted(registry.items())]

    async def get(self, name: str) -> ModelInfo | None:
        """Get model info by name."""
        registry = self._load_registry()
        entry = registry.get(name)
        if entry is None:
            return None
        return self._to_model_info(name, entry)

    async def delete(self, name: str) -> bool:
        """Delete a model's files and registry entry."""
        registry = self._load_registry()

        if name not in registry:
            return False

        # Remove model directory
        model_dir = self._base_path / name
        if model_dir.exists():
            shutil.rmtree(model_dir)

        del registry[name]
        self._save_registry(registry)
        return True

    async def set_active(self, name: str) -> None:
        """Set a model as active, clearing the flag on all others."""
        registry = self._load_registry()

        if name not in registry:
            raise FileNotFoundError(f"Model '{name}' not found in registry")

        for entry in registry.values():
            entry["is_active"] = False
        registry[name]["is_active"] = True
        self._save_registry(registry)

    async def get_active(self) -> ModelInfo | None:
        """Get the currently active model, or None."""
        registry = self._load_registry()
        for model_name, entry in registry.items():
            if entry.get("is_active"):
                return self._to_model_info(model_name, entry)
        return None

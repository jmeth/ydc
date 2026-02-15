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
import zipfile
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

    async def export_zip(self, name: str, output_dir: Path) -> Path:
        """
        Export a model as a zip archive.

        Creates a zip at output_dir/{name}.zip containing best.pt,
        training_config.yaml (if present), and model_meta.json with
        registry metadata (excluding is_active and absolute path).

        Args:
            name: Model identifier to export.
            output_dir: Directory where the zip file will be written.

        Returns:
            Path to the created zip file.

        Raises:
            FileNotFoundError: If the model doesn't exist in the registry
                or the weights file is missing.
        """
        registry = self._load_registry()
        if name not in registry:
            raise FileNotFoundError(f"Model '{name}' not found in registry")

        entry = registry[name]
        model_dir = self._base_path / name
        weights_path = model_dir / "best.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file missing: {weights_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output_dir / f"{name}.zip"

        # Build portable metadata (no absolute paths or instance state)
        meta = {
            "name": name,
            "base_model": entry.get("base_model", ""),
            "dataset_name": entry.get("dataset_name", ""),
            "created_at": entry.get("created_at", 0.0),
            "epochs_completed": entry.get("epochs_completed", 0),
            "best_map50": entry.get("best_map50"),
        }

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(weights_path, "best.pt")

            config_path = model_dir / "training_config.yaml"
            if config_path.exists():
                zf.write(config_path, "training_config.yaml")

            zf.writestr("model_meta.json", json.dumps(meta, indent=2))

        return zip_path

    async def import_zip(self, zip_path: Path, name: str | None = None) -> ModelInfo:
        """
        Import a model from a zip archive.

        Reads model_meta.json for metadata, extracts best.pt and optional
        training_config.yaml, and registers the model in the store.

        Name resolution order: explicit name param > model_meta.json name
        > zip filename stem.

        Args:
            zip_path: Path to the zip archive.
            name: Override name for the imported model.

        Returns:
            ModelInfo for the newly imported model.

        Raises:
            FileNotFoundError: If zip_path doesn't exist or lacks best.pt.
            ValueError: If a model with the resolved name already exists.
        """
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zip_names = zf.namelist()

            if "best.pt" not in zip_names:
                raise FileNotFoundError("Zip archive missing required 'best.pt'")

            # Read metadata if present
            meta: dict = {}
            if "model_meta.json" in zip_names:
                meta = json.loads(zf.read("model_meta.json"))

            # Resolve model name
            resolved_name = name or meta.get("name") or zip_path.stem
            resolved_name = str(resolved_name)

            registry = self._load_registry()
            if resolved_name in registry:
                raise ValueError(f"Model '{resolved_name}' already exists")

            # Extract files into model directory
            model_dir = self._base_path / resolved_name
            model_dir.mkdir(parents=True, exist_ok=True)

            weights_dest = model_dir / "best.pt"
            with zf.open("best.pt") as src, open(weights_dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

            if "training_config.yaml" in zip_names:
                config_dest = model_dir / "training_config.yaml"
                with zf.open("training_config.yaml") as src, open(config_dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        # Register in the registry using metadata from the archive
        now = time.time()
        entry = {
            "path": str(weights_dest),
            "base_model": meta.get("base_model", ""),
            "dataset_name": meta.get("dataset_name", ""),
            "created_at": meta.get("created_at", now),
            "epochs_completed": meta.get("epochs_completed", 0),
            "best_map50": meta.get("best_map50"),
            "is_active": False,
        }

        registry[resolved_name] = entry
        self._save_registry(registry)

        return self._to_model_info(resolved_name, entry)

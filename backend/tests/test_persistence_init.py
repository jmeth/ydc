"""
Unit tests for the persistence factory and dependency injection.

Covers create_stores(), set_stores()/get_stores(), and the
per-store convenience accessors.
"""

import pytest

from backend.persistence import (
    Stores,
    create_stores,
    set_stores,
    get_stores,
    get_dataset_store,
    get_image_store,
    get_label_store,
    get_model_store,
)
from backend.persistence.dataset_store import DatasetStore
from backend.persistence.image_store import ImageStore
from backend.persistence.label_store import LabelStore
from backend.persistence.model_store import ModelStore
from backend.persistence.filesystem.dataset_store import FilesystemDatasetStore
from backend.persistence.filesystem.image_store import FilesystemImageStore
from backend.persistence.filesystem.label_store import FilesystemLabelStore
from backend.persistence.filesystem.model_store import FilesystemModelStore


class TestCreateStores:
    """Tests for the create_stores() factory."""

    def test_returns_stores_dataclass(self, tmp_path):
        """create_stores() returns a Stores instance."""
        stores = create_stores(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
        assert isinstance(stores, Stores)

    def test_creates_correct_implementations(self, tmp_path):
        """create_stores() uses filesystem implementations."""
        stores = create_stores(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
        assert isinstance(stores.dataset, FilesystemDatasetStore)
        assert isinstance(stores.image, FilesystemImageStore)
        assert isinstance(stores.label, FilesystemLabelStore)
        assert isinstance(stores.model, FilesystemModelStore)

    def test_stores_satisfy_abc_contracts(self, tmp_path):
        """All store instances are subtypes of their ABC interfaces."""
        stores = create_stores(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
        assert isinstance(stores.dataset, DatasetStore)
        assert isinstance(stores.image, ImageStore)
        assert isinstance(stores.label, LabelStore)
        assert isinstance(stores.model, ModelStore)


class TestSetGetStores:
    """Tests for the set_stores()/get_stores() DI pattern."""

    def test_get_stores_before_set_raises(self):
        """get_stores() raises RuntimeError when not initialized."""
        set_stores(None)
        with pytest.raises(RuntimeError, match="not initialized"):
            get_stores()

    def test_set_then_get_returns_same_instance(self, tmp_path):
        """set_stores() then get_stores() returns the same object."""
        stores = create_stores(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
        set_stores(stores)
        try:
            assert get_stores() is stores
        finally:
            set_stores(None)

    def test_set_none_clears_stores(self, tmp_path):
        """set_stores(None) resets the module-level singleton."""
        stores = create_stores(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
        set_stores(stores)
        set_stores(None)
        with pytest.raises(RuntimeError):
            get_stores()


class TestConvenienceAccessors:
    """Tests for get_dataset_store(), get_image_store(), etc."""

    @pytest.fixture(autouse=True)
    def _setup_stores(self, tmp_path):
        """Set up and tear down stores for each test."""
        stores = create_stores(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
        set_stores(stores)
        yield stores
        set_stores(None)

    def test_get_dataset_store(self, _setup_stores):
        """get_dataset_store() returns the dataset store."""
        ds = get_dataset_store()
        assert isinstance(ds, DatasetStore)
        assert ds is _setup_stores.dataset

    def test_get_image_store(self, _setup_stores):
        """get_image_store() returns the image store."""
        img = get_image_store()
        assert isinstance(img, ImageStore)
        assert img is _setup_stores.image

    def test_get_label_store(self, _setup_stores):
        """get_label_store() returns the label store."""
        lbl = get_label_store()
        assert isinstance(lbl, LabelStore)
        assert lbl is _setup_stores.label

    def test_get_model_store(self, _setup_stores):
        """get_model_store() returns the model store."""
        mdl = get_model_store()
        assert isinstance(mdl, ModelStore)
        assert mdl is _setup_stores.model

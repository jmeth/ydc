"""
Unit tests for InferenceManager lifecycle: start, stop, status,
prompt update, model switch, and stop_all.

FeedManager and ultralytics are fully mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.core.exceptions import ConflictError, NotFoundError
from backend.feeds.models import FeedConfig, FeedInfo, FeedStatus, FeedType
from backend.inference.manager import InferenceManager
from backend.inference.models import ModelType


def _make_feed_manager():
    """Create a mock FeedManager with standard behaviour."""
    fm = MagicMock()
    fm.get_feed_info.return_value = FeedInfo(
        feed_id="src-feed-1234",
        config=FeedConfig(feed_type=FeedType.CAMERA, source="0", name="cam0"),
        status=FeedStatus.ACTIVE,
    )
    fm.register_derived_feed.return_value = True
    fm.subscribe.return_value = True
    fm.unsubscribe.return_value = True
    fm.unregister_derived_feed.return_value = True
    fm.push_derived_frame.return_value = True
    return fm


def _make_mock_model():
    """Create a mock ultralytics YOLO model."""
    mock = MagicMock()
    mock.names = {0: "person"}
    return mock


@patch("backend.inference.loader.ModelLoader._load_ultralytics")
class TestInferenceManagerStart:
    """Tests for start_inference."""

    def test_start_returns_output_feed_id(self, mock_load):
        """start_inference returns a derived feed ID string."""
        mock_load.return_value = _make_mock_model()
        fm = _make_feed_manager()
        mgr = InferenceManager(fm)

        output_id = mgr.start_inference("src-feed-1234", "yolov8s-worldv2", ModelType.YOLO_WORLD, prompts=["person"])

        assert output_id.startswith("inference_src-feed")
        assert len(output_id) > 10

    def test_start_registers_derived_feed(self, mock_load):
        """start_inference registers a derived feed with FeedManager."""
        mock_load.return_value = _make_mock_model()
        fm = _make_feed_manager()
        mgr = InferenceManager(fm)

        mgr.start_inference("src-feed-1234", "model.pt", ModelType.FINE_TUNED)

        fm.register_derived_feed.assert_called_once()
        derived = fm.register_derived_feed.call_args[0][0]
        assert derived.source_feed_id == "src-feed-1234"
        assert derived.feed_type == "inference"

    def test_start_subscribes_to_source(self, mock_load):
        """start_inference subscribes the session callback to the source feed."""
        mock_load.return_value = _make_mock_model()
        fm = _make_feed_manager()
        mgr = InferenceManager(fm)

        mgr.start_inference("src-feed-1234", "model.pt", ModelType.FINE_TUNED)

        fm.subscribe.assert_called_once()
        assert fm.subscribe.call_args[0][0] == "src-feed-1234"

    def test_start_source_not_found(self, mock_load):
        """start_inference raises NotFoundError if source feed doesn't exist."""
        fm = _make_feed_manager()
        fm.get_feed_info.return_value = None
        mgr = InferenceManager(fm)

        with pytest.raises(NotFoundError):
            mgr.start_inference("nonexistent", "model.pt", ModelType.FINE_TUNED)

    def test_start_conflict_duplicate_source(self, mock_load):
        """start_inference raises ConflictError for duplicate source feed."""
        mock_load.return_value = _make_mock_model()
        fm = _make_feed_manager()
        mgr = InferenceManager(fm)

        mgr.start_inference("src-feed-1234", "model.pt", ModelType.FINE_TUNED)

        with pytest.raises(ConflictError):
            mgr.start_inference("src-feed-1234", "model.pt", ModelType.FINE_TUNED)


@patch("backend.inference.loader.ModelLoader._load_ultralytics")
class TestInferenceManagerStop:
    """Tests for stop_inference and stop_all."""

    def test_stop_returns_true(self, mock_load):
        """stop_inference returns True for active session."""
        mock_load.return_value = _make_mock_model()
        fm = _make_feed_manager()
        mgr = InferenceManager(fm)

        output_id = mgr.start_inference("src-feed-1234", "model.pt", ModelType.FINE_TUNED)
        assert mgr.stop_inference(output_id) is True

    def test_stop_returns_false_for_unknown(self, mock_load):
        """stop_inference returns False if session doesn't exist."""
        fm = _make_feed_manager()
        mgr = InferenceManager(fm)

        assert mgr.stop_inference("nonexistent") is False

    def test_stop_unsubscribes_and_unregisters(self, mock_load):
        """stop_inference cleans up subscription and derived feed."""
        mock_load.return_value = _make_mock_model()
        fm = _make_feed_manager()
        mgr = InferenceManager(fm)

        output_id = mgr.start_inference("src-feed-1234", "model.pt", ModelType.FINE_TUNED)
        mgr.stop_inference(output_id)

        fm.unsubscribe.assert_called_once()
        fm.unregister_derived_feed.assert_called_once_with(output_id)

    def test_stop_allows_restart_on_same_source(self, mock_load):
        """After stopping, a new session can start on the same source feed."""
        mock_load.return_value = _make_mock_model()
        fm = _make_feed_manager()
        mgr = InferenceManager(fm)

        output_id = mgr.start_inference("src-feed-1234", "model.pt", ModelType.FINE_TUNED)
        mgr.stop_inference(output_id)

        new_id = mgr.start_inference("src-feed-1234", "model.pt", ModelType.FINE_TUNED)
        assert new_id != output_id

    def test_stop_all(self, mock_load):
        """stop_all stops every active session."""
        mock_load.return_value = _make_mock_model()
        fm = _make_feed_manager()
        # Allow two different source feeds
        fm.get_feed_info.side_effect = lambda fid: FeedInfo(
            feed_id=fid,
            config=FeedConfig(feed_type=FeedType.CAMERA, source="0"),
            status=FeedStatus.ACTIVE,
        )
        mgr = InferenceManager(fm)

        mgr.start_inference("feed-a", "model.pt", ModelType.FINE_TUNED)
        mgr.start_inference("feed-b", "model.pt", ModelType.FINE_TUNED)
        assert len(mgr.get_all_sessions()) == 2

        mgr.stop_all()
        assert len(mgr.get_all_sessions()) == 0


@patch("backend.inference.loader.ModelLoader._load_ultralytics")
class TestInferenceManagerStatus:
    """Tests for get_session_status and get_all_sessions."""

    def test_get_session_status(self, mock_load):
        """get_session_status returns expected fields."""
        mock_load.return_value = _make_mock_model()
        fm = _make_feed_manager()
        mgr = InferenceManager(fm)

        output_id = mgr.start_inference(
            "src-feed-1234", "yolov8s-worldv2", ModelType.YOLO_WORLD,
            prompts=["person", "car"],
        )

        status = mgr.get_session_status(output_id)
        assert status is not None
        assert status["output_feed_id"] == output_id
        assert status["source_feed_id"] == "src-feed-1234"
        assert status["model_name"] == "yolov8s-worldv2"
        assert status["model_type"] == "yolo_world"
        assert status["prompts"] == ["person", "car"]
        assert status["status"] == "running"
        assert status["frames_processed"] == 0

    def test_get_session_status_not_found(self, mock_load):
        """get_session_status returns None for unknown output_feed_id."""
        fm = _make_feed_manager()
        mgr = InferenceManager(fm)

        assert mgr.get_session_status("nonexistent") is None

    def test_get_all_sessions(self, mock_load):
        """get_all_sessions returns list of all active session statuses."""
        mock_load.return_value = _make_mock_model()
        fm = _make_feed_manager()
        fm.get_feed_info.side_effect = lambda fid: FeedInfo(
            feed_id=fid,
            config=FeedConfig(feed_type=FeedType.CAMERA, source="0"),
            status=FeedStatus.ACTIVE,
        )
        mgr = InferenceManager(fm)

        mgr.start_inference("feed-a", "model.pt", ModelType.FINE_TUNED)
        mgr.start_inference("feed-b", "model.pt", ModelType.FINE_TUNED)

        sessions = mgr.get_all_sessions()
        assert len(sessions) == 2
        assert all(s["status"] == "running" for s in sessions)


@patch("backend.inference.loader.ModelLoader._load_ultralytics")
class TestInferenceManagerPrompts:
    """Tests for update_prompts."""

    def test_update_prompts_restarts_session(self, mock_load):
        """update_prompts stops old session and starts new one."""
        mock_load.return_value = _make_mock_model()
        fm = _make_feed_manager()
        mgr = InferenceManager(fm)

        output_id = mgr.start_inference(
            "src-feed-1234", "yolov8s-worldv2", ModelType.YOLO_WORLD,
            prompts=["person"],
        )

        result = mgr.update_prompts(output_id, ["car", "truck"])
        assert result is True

        # Old session should be gone, new one active
        assert mgr.get_session_status(output_id) is None
        sessions = mgr.get_all_sessions()
        assert len(sessions) == 1

    def test_update_prompts_not_found(self, mock_load):
        """update_prompts returns False for unknown session."""
        fm = _make_feed_manager()
        mgr = InferenceManager(fm)

        assert mgr.update_prompts("nonexistent", ["cat"]) is False


@patch("backend.inference.loader.ModelLoader._load_ultralytics")
class TestInferenceManagerSwitch:
    """Tests for switch_model."""

    def test_switch_model_returns_new_id(self, mock_load):
        """switch_model stops old session and returns new output_feed_id."""
        mock_load.return_value = _make_mock_model()
        fm = _make_feed_manager()
        mgr = InferenceManager(fm)

        old_id = mgr.start_inference("src-feed-1234", "model-a.pt", ModelType.FINE_TUNED)
        new_id = mgr.switch_model(old_id, "model-b.pt", ModelType.FINE_TUNED)

        assert new_id is not None
        assert new_id != old_id
        assert mgr.get_session_status(old_id) is None
        assert mgr.get_session_status(new_id) is not None

    def test_switch_model_not_found(self, mock_load):
        """switch_model returns None for unknown session."""
        fm = _make_feed_manager()
        mgr = InferenceManager(fm)

        assert mgr.switch_model("nonexistent", "model.pt", ModelType.FINE_TUNED) is None

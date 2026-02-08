"""
Tests for notification domain models.

Verifies enum values, dataclass defaults, and auto-generated fields
on the Notification dataclass.
"""

from datetime import datetime, timezone

from backend.notifications.models import (
    Notification,
    NotificationCategory,
    NotificationLevel,
    NotificationType,
)


class TestNotificationType:
    """NotificationType enum has the expected members and string values."""

    def test_toast_value(self):
        assert NotificationType.TOAST.value == "toast"

    def test_banner_value(self):
        assert NotificationType.BANNER.value == "banner"

    def test_alert_value(self):
        assert NotificationType.ALERT.value == "alert"

    def test_status_value(self):
        assert NotificationType.STATUS.value == "status"

    def test_is_str_subclass(self):
        """Enum values can be compared to plain strings."""
        assert NotificationType.TOAST == "toast"


class TestNotificationLevel:
    """NotificationLevel enum has the expected members."""

    def test_info_value(self):
        assert NotificationLevel.INFO.value == "info"

    def test_success_value(self):
        assert NotificationLevel.SUCCESS.value == "success"

    def test_warning_value(self):
        assert NotificationLevel.WARNING.value == "warning"

    def test_error_value(self):
        assert NotificationLevel.ERROR.value == "error"


class TestNotificationCategory:
    """NotificationCategory enum has the expected members."""

    def test_system_value(self):
        assert NotificationCategory.SYSTEM.value == "system"

    def test_scan_value(self):
        assert NotificationCategory.SCAN.value == "scan"

    def test_training_value(self):
        assert NotificationCategory.TRAINING.value == "training"

    def test_dataset_value(self):
        assert NotificationCategory.DATASET.value == "dataset"

    def test_inference_value(self):
        assert NotificationCategory.INFERENCE.value == "inference"


class TestNotification:
    """Notification dataclass auto-generates id, timestamp, and defaults."""

    def test_auto_generated_id(self):
        """Each notification gets a unique UUID id."""
        n1 = Notification(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Test",
            message="test message",
        )
        n2 = Notification(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Test",
            message="test message",
        )
        assert n1.id != n2.id
        assert len(n1.id) == 36  # UUID4 string length

    def test_auto_timestamp(self):
        """Timestamp is set to approximately now in UTC."""
        before = datetime.now(timezone.utc)
        n = Notification(
            type=NotificationType.BANNER,
            level=NotificationLevel.WARNING,
            category=NotificationCategory.TRAINING,
            title="Test",
            message="test",
        )
        after = datetime.now(timezone.utc)
        assert before <= n.timestamp <= after

    def test_default_read_false(self):
        n = Notification(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Test",
            message="test",
        )
        assert n.read is False

    def test_default_dismissed_false(self):
        n = Notification(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Test",
            message="test",
        )
        assert n.dismissed is False

    def test_default_data_none(self):
        n = Notification(
            type=NotificationType.TOAST,
            level=NotificationLevel.INFO,
            category=NotificationCategory.SYSTEM,
            title="Test",
            message="test",
        )
        assert n.data is None

    def test_custom_data(self):
        """Extra data dict is stored correctly."""
        n = Notification(
            type=NotificationType.STATUS,
            level=NotificationLevel.SUCCESS,
            category=NotificationCategory.INFERENCE,
            title="Done",
            message="finished",
            data={"model": "yolov8n", "epochs": 50},
        )
        assert n.data == {"model": "yolov8n", "epochs": 50}

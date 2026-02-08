# Implementing a New Feed Type

This guide walks through adding a new video feed source (e.g., RTSP stream, video file) to the feeds subsystem.

## Overview

Every feed source implements the `BaseFeed` abstract class, gets a `FeedType` enum value, and is wired into `FeedFactory`. The FeedManager handles threading and buffering automatically — you only need to write the connect/read/disconnect logic.

## Steps

### 1. Add a FeedType enum value

In `backend/feeds/models.py`, the type should already exist or you can add one:

```python
class FeedType(str, Enum):
    CAMERA = "camera"
    RTSP = "rtsp"       # <-- your new type
    FILE = "file"
```

### 2. Create the feed class

Create a new file in `backend/feeds/`, e.g. `backend/feeds/rtsp.py`. Subclass `BaseFeed` and implement all five abstract members:

```python
"""RTSP network stream feed."""

import logging
import cv2
import numpy as np
from backend.feeds.base import BaseFeed

logger = logging.getLogger(__name__)


class RtspFeed(BaseFeed):
    """
    Video feed from an RTSP network stream via OpenCV.

    Args:
        source: RTSP URL (e.g., "rtsp://192.168.1.10:554/stream")
    """

    def __init__(self, source: str):
        super().__init__(source)
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 0.0
        self._resolution: tuple[int, int] | None = None

    def connect(self) -> bool:
        """Open the RTSP stream. Returns True on success."""
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            logger.error("Failed to open RTSP stream: %s", self._source)
            self._cap = None
            return False

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            self._resolution = (w, h)

        logger.info("RTSP stream opened: %s @ %.1f fps", self._source, self._fps)
        return True

    def disconnect(self) -> None:
        """Release the RTSP stream."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._resolution = None

    def read(self) -> np.ndarray | None:
        """Read the next frame. Returns BGR numpy array or None."""
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> tuple[int, int] | None:
        return self._resolution
```

#### Contract requirements

| Method | Must do | Returns |
|--------|---------|---------|
| `connect()` | Open the source, read fps/resolution | `True` on success, `False` on failure |
| `disconnect()` | Release all resources, safe to call if not connected | Nothing |
| `read()` | Return the next frame, non-blocking if possible | `np.ndarray` (H,W,C BGR) or `None` |
| `fps` | Report native frame rate | `float` (0.0 if unknown) |
| `resolution` | Report frame dimensions | `(width, height)` or `None` |

### 3. Register in FeedFactory

In `backend/feeds/factory.py`, import your class and add it to the `create` method:

```python
from backend.feeds.rtsp import RtspFeed

class FeedFactory:
    @staticmethod
    def create(feed_type: FeedType, source: str) -> BaseFeed:
        if feed_type == FeedType.CAMERA:
            return CameraFeed(source)
        elif feed_type == FeedType.RTSP:
            return RtspFeed(source)      # <-- add this
        elif feed_type == FeedType.FILE:
            raise ValueError(f"Feed type '{feed_type.value}' is not yet supported")
        else:
            raise ValueError(f"Unknown feed type: '{feed_type}'")
```

That's it for production code. The API, FeedManager, FeedStreamer, and WebSocket broadcasting all work automatically through the `BaseFeed` interface.

### 4. Write tests

Create `backend/tests/test_rtsp_feed.py`. Mock `cv2.VideoCapture` the same way `test_camera_feed.py` does:

```python
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from backend.feeds.rtsp import RtspFeed

def make_mock_capture(is_opened=True, fps=25.0, width=1920.0, height=1080.0):
    import cv2
    mock = MagicMock()
    mock.isOpened.return_value = is_opened
    mock.get.side_effect = lambda p: {
        cv2.CAP_PROP_FPS: fps,
        cv2.CAP_PROP_FRAME_WIDTH: width,
        cv2.CAP_PROP_FRAME_HEIGHT: height,
    }.get(p, 0.0)
    mock.read.return_value = (True, np.zeros((int(height), int(width), 3), dtype=np.uint8))
    return mock

class TestRtspFeedConnect:
    @patch("backend.feeds.rtsp.cv2.VideoCapture")
    def test_connect_success(self, mock_vc):
        mock_vc.return_value = make_mock_capture()
        feed = RtspFeed("rtsp://example.com/stream")
        assert feed.connect() is True
        assert feed.fps == 25.0
        assert feed.resolution == (1920, 1080)

    @patch("backend.feeds.rtsp.cv2.VideoCapture")
    def test_connect_failure(self, mock_vc):
        mock_vc.return_value = make_mock_capture(is_opened=False)
        feed = RtspFeed("rtsp://bad-url")
        assert feed.connect() is False
```

Also add a factory test:

```python
def test_factory_creates_rtsp():
    feed = FeedFactory.create(FeedType.RTSP, "rtsp://example.com/stream")
    assert isinstance(feed, RtspFeed)
```

### 5. Verify

```bash
# Run your new tests
python -m pytest backend/tests/test_rtsp_feed.py -v

# Run the full suite to check nothing broke
python -m pytest backend/tests/ -v

# Manual test (if you have an RTSP source)
# POST /api/feeds {"feed_type": "rtsp", "source": "rtsp://192.168.1.10:554/stream"}
```

## File checklist

| Action | File |
|--------|------|
| Add/verify enum | `backend/feeds/models.py` |
| Create class | `backend/feeds/yourtype.py` |
| Register in factory | `backend/feeds/factory.py` |
| Write tests | `backend/tests/test_yourtype_feed.py` |

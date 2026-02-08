"""
Thread-safe ring buffer for video frame storage.

Provides a fixed-size circular buffer backed by collections.deque,
protected by a threading Lock for safe use from capture threads
and async consumers.
"""

import threading
from collections import deque
from typing import TypeVar

T = TypeVar("T")


class RingBuffer:
    """
    Thread-safe fixed-size ring buffer.

    When full, new items evict the oldest. All operations are
    protected by a threading.Lock for safe cross-thread access.

    Args:
        max_size: Maximum number of items to store
    """

    def __init__(self, max_size: int = 30):
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._max_size = max_size

    @property
    def max_size(self) -> int:
        """Maximum capacity of the buffer."""
        return self._max_size

    def push(self, item) -> None:
        """
        Add an item to the buffer, evicting the oldest if full.

        Args:
            item: The item to store
        """
        with self._lock:
            self._buffer.append(item)

    def get_latest(self):
        """
        Get the most recently pushed item.

        Returns:
            The newest item, or None if the buffer is empty.
        """
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1]

    def get_recent(self, count: int = 1) -> list:
        """
        Get the N most recently pushed items, newest first.

        Args:
            count: Number of items to retrieve

        Returns:
            List of items ordered newest-to-oldest, up to `count` items.
        """
        with self._lock:
            if not self._buffer:
                return []
            # Slice from the right end, then reverse for newest-first
            n = min(count, len(self._buffer))
            items = list(self._buffer)[-n:]
            items.reverse()
            return items

    def clear(self) -> None:
        """Remove all items from the buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        """Return the current number of items in the buffer."""
        with self._lock:
            return len(self._buffer)

    def __bool__(self) -> bool:
        """Return True if the buffer contains any items."""
        with self._lock:
            return len(self._buffer) > 0

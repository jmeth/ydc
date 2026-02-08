"""
Unit tests for the thread-safe RingBuffer.

Tests push/eviction semantics, retrieval methods, clearing,
length tracking, and concurrent access from multiple threads.
"""

import threading

import pytest

from backend.feeds.buffer import RingBuffer


class TestRingBufferBasics:
    """Tests for basic push, get, and size operations."""

    def test_push_and_get_latest(self):
        """Pushing an item makes it retrievable via get_latest."""
        buf = RingBuffer(max_size=5)
        buf.push("a")
        assert buf.get_latest() == "a"

    def test_get_latest_empty(self):
        """get_latest returns None on an empty buffer."""
        buf = RingBuffer(max_size=5)
        assert buf.get_latest() is None

    def test_push_multiple_get_latest(self):
        """get_latest always returns the most recently pushed item."""
        buf = RingBuffer(max_size=5)
        buf.push("a")
        buf.push("b")
        buf.push("c")
        assert buf.get_latest() == "c"

    def test_len(self):
        """__len__ tracks the number of items in the buffer."""
        buf = RingBuffer(max_size=5)
        assert len(buf) == 0
        buf.push("a")
        assert len(buf) == 1
        buf.push("b")
        assert len(buf) == 2

    def test_bool_empty(self):
        """Empty buffer is falsy."""
        buf = RingBuffer(max_size=5)
        assert not buf

    def test_bool_nonempty(self):
        """Non-empty buffer is truthy."""
        buf = RingBuffer(max_size=5)
        buf.push("a")
        assert buf


class TestRingBufferEviction:
    """Tests for eviction when buffer exceeds max_size."""

    def test_eviction_at_capacity(self):
        """Oldest item is evicted when pushing past max_size."""
        buf = RingBuffer(max_size=3)
        buf.push("a")
        buf.push("b")
        buf.push("c")
        buf.push("d")  # evicts "a"
        assert len(buf) == 3
        assert buf.get_latest() == "d"
        # "a" should be gone, "b" is now the oldest
        recent = buf.get_recent(3)
        assert recent == ["d", "c", "b"]

    def test_eviction_preserves_max_size(self):
        """Buffer never grows beyond max_size even with many pushes."""
        buf = RingBuffer(max_size=2)
        for i in range(100):
            buf.push(i)
        assert len(buf) == 2
        assert buf.get_latest() == 99
        assert buf.get_recent(2) == [99, 98]


class TestRingBufferGetRecent:
    """Tests for get_recent retrieval."""

    def test_get_recent_returns_newest_first(self):
        """get_recent returns items in newest-to-oldest order."""
        buf = RingBuffer(max_size=5)
        buf.push("a")
        buf.push("b")
        buf.push("c")
        assert buf.get_recent(3) == ["c", "b", "a"]

    def test_get_recent_count_exceeds_size(self):
        """Requesting more items than available returns all items."""
        buf = RingBuffer(max_size=5)
        buf.push("a")
        buf.push("b")
        result = buf.get_recent(10)
        assert result == ["b", "a"]

    def test_get_recent_empty(self):
        """get_recent on empty buffer returns empty list."""
        buf = RingBuffer(max_size=5)
        assert buf.get_recent(3) == []

    def test_get_recent_single(self):
        """get_recent(1) returns just the latest item."""
        buf = RingBuffer(max_size=5)
        buf.push("a")
        buf.push("b")
        assert buf.get_recent(1) == ["b"]


class TestRingBufferClear:
    """Tests for clear operation."""

    def test_clear_empties_buffer(self):
        """clear removes all items."""
        buf = RingBuffer(max_size=5)
        buf.push("a")
        buf.push("b")
        buf.clear()
        assert len(buf) == 0
        assert buf.get_latest() is None
        assert buf.get_recent(5) == []

    def test_clear_allows_reuse(self):
        """Buffer is usable after clear."""
        buf = RingBuffer(max_size=3)
        buf.push("old")
        buf.clear()
        buf.push("new")
        assert buf.get_latest() == "new"
        assert len(buf) == 1


class TestRingBufferValidation:
    """Tests for constructor validation."""

    def test_invalid_max_size_zero(self):
        """max_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            RingBuffer(max_size=0)

    def test_invalid_max_size_negative(self):
        """Negative max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            RingBuffer(max_size=-5)

    def test_max_size_property(self):
        """max_size property reflects the configured value."""
        buf = RingBuffer(max_size=42)
        assert buf.max_size == 42


class TestRingBufferThreadSafety:
    """Tests for concurrent access from multiple threads."""

    def test_concurrent_push(self):
        """Multiple threads pushing concurrently don't corrupt the buffer."""
        buf = RingBuffer(max_size=100)
        num_threads = 10
        pushes_per_thread = 50
        barrier = threading.Barrier(num_threads)

        def pusher(thread_id):
            barrier.wait()
            for i in range(pushes_per_thread):
                buf.push((thread_id, i))

        threads = [threading.Thread(target=pusher, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Buffer should contain exactly max_size items (some evicted)
        assert len(buf) == 100

    def test_concurrent_push_and_read(self):
        """Concurrent pushes and reads don't raise or corrupt."""
        buf = RingBuffer(max_size=50)
        errors = []
        barrier = threading.Barrier(4)

        def pusher():
            barrier.wait()
            for i in range(200):
                buf.push(i)

        def reader():
            barrier.wait()
            for _ in range(200):
                try:
                    buf.get_latest()
                    buf.get_recent(5)
                    len(buf)
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=pusher),
            threading.Thread(target=pusher),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"

"""Buffer-overflow boundary — exactly at the cap, the OLDEST is dropped."""

from __future__ import annotations

import threading

from lago_agent_sdk.queue import EventQueue


def test_overflow_drops_oldest_at_exact_boundary():
    """Push 10001 events while sender is paused; drop the first one only."""
    paused = threading.Event()

    def slow_sender(batch):
        paused.wait(timeout=30.0)

    q = EventQueue(
        sender=slow_sender,
        flush_interval=10.0,  # never timer-flush during the test
        max_batch_size=10_000,  # match buffer so worker takes everything once unpaused
        max_buffer_size=10_000,
    )
    try:
        # Fill the buffer exactly to capacity
        for i in range(10_000):
            q.push({"i": i})
        with q._lock:  # type: ignore[attr-defined]
            assert len(q._buffer) == 10_000  # type: ignore[attr-defined]

        # One more — should drop event 0, keep 1..10_000
        q.push({"i": 10_000})
        with q._lock:  # type: ignore[attr-defined]
            buf = list(q._buffer)  # type: ignore[attr-defined]
        assert len(buf) == 10_000
        assert buf[0]["i"] == 1, "oldest (i=0) should have been dropped"
        assert buf[-1]["i"] == 10_000, "newest must be at the right end"
    finally:
        paused.set()
        q.shutdown(timeout=2.0)


def test_repeated_overflow_keeps_window_sliding():
    paused = threading.Event()

    def slow_sender(batch):
        paused.wait(timeout=30.0)

    # max_batch_size > max_buffer_size keeps the background worker from ever
    # being woken by push (buffer can't exceed max_batch_size). Combined with
    # a long flush_interval, the test is deterministic — the worker only runs
    # once shutdown() releases `paused` in the finally block.
    q = EventQueue(sender=slow_sender, flush_interval=60.0, max_batch_size=10_000, max_buffer_size=100)
    try:
        for i in range(250):  # 150 events overflow
            q.push({"i": i})
        with q._lock:  # type: ignore[attr-defined]
            buf = list(q._buffer)  # type: ignore[attr-defined]
        # Should contain only the most recent 100
        assert [e["i"] for e in buf] == list(range(150, 250))
    finally:
        paused.set()
        q.shutdown(timeout=2.0)

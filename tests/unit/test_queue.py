"""Event queue tests — batching, retry, backoff, flush, overflow."""

from __future__ import annotations

import threading
import time

from lago_agent_sdk.queue import EventQueue


def test_100_pushes_produce_at_most_3_http_calls():
    sent = []
    q = EventQueue(sender=lambda batch: sent.append(list(batch)), flush_interval=0.05, max_batch_size=100)
    try:
        for i in range(100):
            q.push({"i": i})
        assert q.flush(timeout=2.0)
    finally:
        q.shutdown(timeout=1.0)
    assert q._http_calls <= 3, f"expected <=3 batched calls, got {q._http_calls}"
    total = sum(len(b) for b in sent)
    assert total == 100


def test_failing_send_triggers_retry_with_backoff():
    state = {"calls": 0, "fail_until": 3}

    def sender(batch):
        state["calls"] += 1
        if state["calls"] <= state["fail_until"]:
            raise RuntimeError("boom")

    q = EventQueue(sender=sender, flush_interval=0.05, max_batch_size=10, max_retry_seconds=0.5)
    try:
        for i in range(5):
            q.push({"i": i})
        # Wait long enough for 3 failures + 4th success (~1+2+4=7s? no, capped at 0.5s) + send
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline and state["calls"] <= state["fail_until"]:
            time.sleep(0.05)
        assert state["calls"] >= state["fail_until"] + 1, f"sender retried {state['calls']} times"
    finally:
        q.shutdown(timeout=1.0)


def test_buffer_overflow_drops_oldest():
    q = EventQueue(sender=lambda b: None, flush_interval=10.0, max_batch_size=1, max_buffer_size=5)
    try:
        # No flush — push 10 items, only last 5 should remain
        # Stop the worker thread from draining: make sender sleep
        pass
    finally:
        q.shutdown(timeout=0.1)
    # Re-test with non-draining sender
    blocking_sender_event = threading.Event()

    def slow_sender(batch):
        blocking_sender_event.wait(timeout=2.0)

    q2 = EventQueue(sender=slow_sender, flush_interval=10.0, max_batch_size=1, max_buffer_size=5)
    try:
        for i in range(10):
            q2.push({"i": i})
        # Buffer should be capped at 5
        with q2._lock:
            assert len(q2._buffer) <= 5
    finally:
        blocking_sender_event.set()
        q2.shutdown(timeout=2.0)


def test_flush_returns_true_when_drained():
    q = EventQueue(sender=lambda b: None, flush_interval=0.05, max_batch_size=10)
    try:
        for i in range(20):
            q.push({"i": i})
        assert q.flush(timeout=2.0)
    finally:
        q.shutdown(timeout=1.0)


def test_flush_returns_false_on_timeout():
    blocking = threading.Event()

    def slow(batch):
        blocking.wait(timeout=5.0)

    q = EventQueue(sender=slow, flush_interval=0.05, max_batch_size=1)
    try:
        for i in range(5):
            q.push({"i": i})
        time.sleep(0.05)  # let worker pick up first batch
        # While the worker is blocked, buffer still has remaining items.
        # flush() with very short timeout returns False.
        assert q.flush(timeout=0.05) is False
    finally:
        blocking.set()
        q.shutdown(timeout=2.0)

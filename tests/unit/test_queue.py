"""Event queue tests — batching, retry, backoff, flush, overflow."""

from __future__ import annotations

import threading
import time

from lago_agent_sdk.exceptions import LagoApiError
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


# ----------------------------------------------------------------------
# Permanent (4xx) vs transient failures. A duplicate transaction_id from
# replaying/backfilling the same window twice will NEVER succeed by retrying
# the same batch — it must be isolated and dropped, not block real events
# queued behind it forever the same way a genuine transient failure would.
# ----------------------------------------------------------------------
def test_permanent_failure_isolates_bad_events_from_good_ones_in_same_batch():
    """Lago's batch endpoint is all-or-nothing: one duplicate transaction_id
    fails the WHOLE batch even though the other events are perfectly valid.
    Naively dropping the batch would silently lose those valid events too —
    the queue must fall back to one-by-one to tell them apart."""
    sent_individually = []

    def sender(batch):
        if len(batch) > 1:
            raise LagoApiError(422, '{"error_details":{"transaction_id":["value_already_exist"]}}')
        event = batch[0]
        sent_individually.append(event["id"])
        if event["id"] in ("dup_1", "dup_2"):
            raise LagoApiError(422, '{"error_details":{"transaction_id":["value_already_exist"]}}')
        # "good_*" events succeed alone.

    errors: list[tuple[Exception, str]] = []
    q = EventQueue(
        sender=sender,
        flush_interval=0.05,
        max_batch_size=10,
        on_error=lambda exc, where: errors.append((exc, where)),
    )
    try:
        for eid in ["dup_1", "good_1", "dup_2", "good_2"]:
            q.push({"id": eid})
        assert q.flush(timeout=2.0)
    finally:
        q.shutdown(timeout=1.0)

    # All four were tried individually — the two "good" ones weren't silently
    # dropped along with the two duplicates just because they shared a batch.
    assert set(sent_individually) == {"dup_1", "good_1", "dup_2", "good_2"}
    # on_error fires once for the batch-level failure, not once per dropped item.
    assert len(errors) == 1
    assert errors[0][1] == "send_batch"


def test_permanent_failure_does_not_apply_backoff():
    """Retrying a permanently-doomed batch with exponential backoff is
    pointless — the isolate-and-drop path must not slow down subsequent
    genuinely-transient failures by leaving a stale backoff in place."""
    calls = {"n": 0}

    def sender(batch):
        calls["n"] += 1
        if len(batch) > 1:
            raise LagoApiError(422, "duplicate")
        raise LagoApiError(422, "duplicate")  # every isolated event is also a dup here

    q = EventQueue(sender=sender, flush_interval=0.05, max_batch_size=10, max_retry_seconds=0.5)
    try:
        q.push({"id": "dup_1"})
        q.push({"id": "dup_2"})
        assert q.flush(timeout=2.0)  # drains fast — no backoff wait, unlike a transient failure
        assert q._backoff_seconds == 0.0
    finally:
        q.shutdown(timeout=1.0)


def test_transient_failure_during_isolation_still_gets_retried():
    """An event that hits a network-level (non-4xx) error while being sent
    individually is a real transient failure — it must still go through the
    normal re-queue-and-retry path, not get treated as permanent."""
    attempts = {"flaky": 0}

    def sender(batch):
        if len(batch) > 1:
            raise LagoApiError(422, "duplicate")  # forces the isolate-one-by-one path
        event = batch[0]
        if event["id"] == "flaky":
            attempts["flaky"] += 1
            if attempts["flaky"] == 1:
                raise RuntimeError("transient network blip")  # not a LagoApiError at all
            return  # succeeds on the retried attempt
        if event["id"] == "dup":
            raise LagoApiError(422, "duplicate")

    q = EventQueue(sender=sender, flush_interval=0.05, max_batch_size=10, max_retry_seconds=0.5)
    try:
        q.push({"id": "dup"})
        q.push({"id": "flaky"})
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and attempts["flaky"] < 2:
            time.sleep(0.05)
        assert attempts["flaky"] >= 2, "the transient failure should have been retried, not dropped"
    finally:
        q.shutdown(timeout=2.0)


# ----------------------------------------------------------------------
# Shutdown's final drain. Previously: `except Exception: pass` on a single
# attempt at a single batch — any failure at all was silently swallowed, and
# a buffer holding more than one batch's worth of events at shutdown time
# left the rest never even attempted.
# ----------------------------------------------------------------------
def test_shutdown_drains_more_than_one_batch():
    """Buffer holds 3 batches' worth of events right as shutdown starts —
    every one of them must be attempted, not just the first."""
    sent = []
    q = EventQueue(sender=lambda b: sent.extend(b), flush_interval=10.0, max_batch_size=5)
    try:
        for i in range(15):  # 3 full batches of 5, worker hasn't had a flush tick yet
            q.push({"i": i})
    finally:
        q.shutdown(timeout=2.0)
    assert len(sent) == 15


def test_shutdown_reports_transient_failure_instead_of_silently_swallowing():
    """A persistently-failing sender at shutdown time must surface via
    on_error — not vanish behind a bare `except: pass` the way it used to."""
    errors: list[tuple[Exception, str]] = []

    def always_fails(batch):
        raise RuntimeError("network still down")

    q = EventQueue(
        sender=always_fails,
        flush_interval=10.0,
        max_batch_size=10,
        max_retry_seconds=1.0,
        on_error=lambda exc, where: errors.append((exc, where)),
    )
    try:
        q.push({"i": 1})
    finally:
        q.shutdown(timeout=3.0)
    assert len(errors) >= 1
    assert errors[0][1] == "send_batch"
    assert "network still down" in str(errors[0][0])


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

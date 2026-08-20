"""Event queue tests — batching, retry, backoff, flush, overflow."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import threading
import time

import pytest

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


_REENTRANT_OVERFLOW_PROGRAM = """
import sys, threading
sys.path.insert(0, {src!r})
from lago_agent_sdk.queue import EventQueue

calls = {{"n": 0}}
def on_error(exc, where):
    calls["n"] += 1
    if calls["n"] > 200:          # runaway guard so this exits rather than spinning
        raise SystemExit(3)
    q.push({{"diagnostic": True}})   # re-enters push() from inside the hook

q = EventQueue(sender=lambda b: None, flush_interval=10.0, max_batch_size=1000,
               max_buffer_size=1, on_error=on_error)
for i in range(3):
    q.push({{"i": i}})
q.shutdown(timeout=1.0)
print("OK", calls["n"])
"""


def _run_reentrant_overflow(timeout: float = 20.0) -> subprocess.CompletedProcess:
    """Run the re-entrant-overflow scenario in a SUBPROCESS.

    It has to be a subprocess: once the deadlock happens, the wedged producer holds
    `_lock` forever, and `EventQueue.__init__`'s `atexit` shutdown then blocks on that
    same lock at interpreter exit. The process is poisoned, so an in-process test
    would hang the whole session instead of reporting a failure. Out-of-process, a
    hang is just a timeout we can assert on.
    """
    src = str(pathlib.Path(__file__).resolve().parents[2] / "src")
    return subprocess.run(
        [sys.executable, "-c", _REENTRANT_OVERFLOW_PROGRAM.format(src=src)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_overflow_report_does_not_deadlock_a_reentrant_callback():
    """The report must run with the lock RELEASED, and must not re-enter unboundedly.

    Two failure modes, one scenario. `_lock` is a plain Lock, not an RLock, so
    reporting inside it meant a customer `on_error` that touched the SDK at all —
    emitting a diagnostic, forcing a flush — blocked forever on a lock its own thread
    already held; overflow happens under sustained load, which is exactly when such a
    hook fires. Moving the report out of the lock then exposed the other half: the
    buffer is full again by the time the hook runs, so a hook that pushes overflows
    again and re-enters without bound.
    """
    try:
        proc = _run_reentrant_overflow()
    except subprocess.TimeoutExpired:
        raise AssertionError(
            "re-entrant on_error during overflow hung the process — the report is holding the lock"
        ) from None
    assert proc.returncode == 0, f"exit {proc.returncode}: " + (
        "runaway re-entrant reporting" if proc.returncode == 3 else proc.stderr[-400:]
    )
    assert proc.stdout.startswith("OK"), proc.stdout
    reports = int(proc.stdout.split()[1])
    assert reports <= 10, f"expected a bounded number of overflow reports, got {reports}"


def test_overflow_is_reported_through_on_error():
    """An overflow drops BILLABLE events. It was logger.warning only, so a customer
    watching on_error never learned revenue had been lost; the JS port already
    reported it."""
    errors: list = []
    q = EventQueue(
        sender=lambda b: None,
        flush_interval=10.0,  # keep the worker idle so the buffer really fills
        max_batch_size=1000,
        max_buffer_size=2,
        on_error=lambda exc, where: errors.append((str(exc), where)),
    )
    try:
        for i in range(5):
            q.push({"id": i})
        assert errors, "overflow must reach on_error"
        assert any(w == "overflow" for _, w in errors)
        assert any("overflow" in m for m, _ in errors)
    finally:
        q.shutdown(timeout=1.0)


# ----------------------------------------------------------------------
# The throttling 4xxs. 429 and 408 sit inside the 400-499 range but mean "try
# again, later" — classifying them as permanent dropped billable events and
# aimed `max_batch_size` extra requests at a server that had just asked us to
# slow down.
# ----------------------------------------------------------------------
# 402 is deliberately NOT here — see `test_402_is_held_not_dropped`. What makes a
# 413/415 batch fail is a property of the batch itself; a 402 is a property of the
# account and resolves out-of-band, so splitting it only drops every event faster.
@pytest.mark.parametrize("status", [413, 415])
def test_batch_only_4xx_is_split_not_head_of_line_blocked(status: int):
    """For these the SAME batch can never succeed, but its events can individually.

    Treating them as transient re-prepended the identical batch at the head of the FIFO
    and backed off to 60s forever, blocking everything behind it. Routing them to
    `_send_individually` splits the batch and delivers what is deliverable — which is
    what the isolation path was built for, and it was unreachable for exactly the batch
    that most needed it.
    """
    sent_individually: list = []

    def sender(batch):
        if len(batch) > 1:
            raise LagoApiError(status, "batch too large / unacceptable as-is")
        sent_individually.append(batch[0]["id"])  # each event succeeds alone

    q = EventQueue(sender=sender, flush_interval=0.05, max_batch_size=10, max_retry_seconds=0.5)
    try:
        for i in range(4):
            q.push({"id": i})
        assert q.flush(timeout=3.0), "queue should drain, not head-of-line block"
        assert sorted(sent_individually) == [0, 1, 2, 3], (
            f"every event should have been delivered individually, got {sent_individually}"
        )
        assert q._backoff_seconds == 0.0, "splitting must not leave a stale backoff"
    finally:
        q.shutdown(timeout=1.0)


def test_402_is_held_not_dropped():
    """A 402 must survive to be re-sent — it resolves out-of-band.

    Regression: 402 used to be permanent, which routed the batch to
    `_send_individually`, where every isolated send 402ed too and was dropped for good.
    Measured against a server returning 402: 5 events in, 6 HTTP calls out, 0
    recoverable — a lapsed Lago account silently discarded every billable event for the
    whole outage. "Payment required" is a property of the ACCOUNT: it stops being true
    the moment someone pays.
    """
    attempts = {"n": 0}
    delivered: list = []
    per_request_sizes: list = []

    def sender(batch):
        attempts["n"] += 1
        per_request_sizes.append(len(batch))
        # Account is lapsed for the first two attempts, then someone pays.
        if attempts["n"] <= 2:
            raise LagoApiError(402, '{"error":"payment required"}')
        delivered.extend(batch)

    q = EventQueue(sender=sender, flush_interval=0.05, max_batch_size=10, max_retry_seconds=0.5)
    try:
        for i in "abcde":
            q.push({"id": i})
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not delivered:
            time.sleep(0.05)
        # Nothing lost: all five arrive once the account is current again.
        assert [e["id"] for e in delivered] == list("abcde")
        # Never fanned out — every request carried the whole batch, so no event was
        # ever isolated and dropped.
        assert all(n == 5 for n in per_request_sizes), per_request_sizes
    finally:
        q.shutdown(timeout=2.0)


@pytest.mark.parametrize("status", [429, 408, 402])
def test_throttling_4xx_is_retried_not_dropped(status: int):
    """A rate-limited or timed-out batch must reach Lago eventually. Dropping
    it loses revenue, and isolating it one-by-one multiplies the load on a
    server that is already shedding it."""
    attempts = {"n": 0}
    delivered: list = []

    def sender(batch):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise LagoApiError(status, '{"error":"too many requests"}')
        delivered.extend(batch)  # succeeds once the throttle lifts

    q = EventQueue(sender=sender, flush_interval=0.05, max_batch_size=10, max_retry_seconds=0.5)
    try:
        q.push({"id": "a"})
        q.push({"id": "b"})
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not delivered:
            time.sleep(0.05)
        assert [e["id"] for e in delivered] == ["a", "b"], "throttled events must still be delivered"
        # Delivered as one batch, i.e. never fanned out into per-event requests.
        assert attempts["n"] == 2
    finally:
        q.shutdown(timeout=2.0)


@pytest.mark.parametrize("status", [429, 408, 402])
def test_throttling_4xx_applies_backoff(status: int):
    """The inverse of test_permanent_failure_does_not_apply_backoff: a
    throttling failure is transient, so it MUST leave a backoff in place —
    that pause is the whole point of respecting a rate limit."""

    def sender(batch):
        raise LagoApiError(status, "slow down")

    q = EventQueue(sender=sender, flush_interval=0.05, max_batch_size=10, max_retry_seconds=0.5)
    try:
        q.push({"id": "a"})
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and q._backoff_seconds == 0.0:
            time.sleep(0.05)
        assert q._backoff_seconds > 0.0, "a throttling 4xx must back off, not isolate-and-drop"
    finally:
        q.shutdown(timeout=1.0)


def test_unrecognized_4xx_is_treated_as_transient():
    """Only the enumerated validation statuses are permanent. An unfamiliar 4xx
    errs toward retrying: a needless delay costs latency, a wrong drop costs
    revenue."""
    attempts = {"n": 0}
    delivered: list = []

    def sender(batch):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise LagoApiError(418, "i am a teapot")
        delivered.extend(batch)

    q = EventQueue(sender=sender, flush_interval=0.05, max_batch_size=10, max_retry_seconds=0.5)
    try:
        q.push({"id": "a"})
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not delivered:
            time.sleep(0.05)
        assert [e["id"] for e in delivered] == ["a"]
    finally:
        q.shutdown(timeout=2.0)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_validation_4xx_still_isolates_and_drops(status: int):
    """The statuses that genuinely cannot succeed on a re-send keep the
    isolate-one-by-one behaviour, so a single bad transaction_id still doesn't
    take the rest of its batch down with it."""
    sent_individually: list = []

    def sender(batch):
        if len(batch) > 1:
            raise LagoApiError(status, "batch rejected")
        sent_individually.append(batch[0]["id"])
        if batch[0]["id"].startswith("bad"):
            raise LagoApiError(status, "this one really is invalid")

    q = EventQueue(sender=sender, flush_interval=0.05, max_batch_size=10, max_retry_seconds=0.5)
    try:
        q.push({"id": "bad_1"})
        q.push({"id": "good_1"})
        assert q.flush(timeout=2.0)
        assert set(sent_individually) == {"bad_1", "good_1"}
        assert q._backoff_seconds == 0.0
    finally:
        q.shutdown(timeout=1.0)


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


def test_isolated_retries_keep_their_fifo_order() -> None:
    """`_replay_failed` PREPENDS, so calling it once per event inside the isolation
    loop reversed the survivors: a 413 batch of a,b,c,d,e whose b,c,d fail
    transiently while isolated came back as d,c,b. FIFO is the queue's contract —
    it is what makes oldest-dropped-first overflow and Lago's own event ordering
    mean anything — so a recovery path must not silently invert it."""

    def sender(batch):
        if batch[0]["id"] in ("b", "c", "d"):
            raise LagoApiError(503, "transient while isolated")

    q = EventQueue(sender=sender, flush_interval=60.0, max_batch_size=10, max_buffer_size=100)
    try:
        q._send_individually([{"id": i} for i in "abcde"], LagoApiError(413, "too large"))
        with q._lock:
            assert [e["id"] for e in q._buffer] == ["b", "c", "d"]
    finally:
        q.shutdown(timeout=1.0)

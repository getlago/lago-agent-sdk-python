"""Async batched event queue.

Thread-safe, in-memory. Background thread flushes every `flush_interval`
seconds or immediately when buffer reaches `max_batch_size`. On a TRANSIENT
send failure (network error, 5xx), re-prepends the batch and applies
exponential backoff (1s, 2s, 4s, 8s, capped at 60s). Resets on next success.

A PERMANENT failure (a Lago *validation* 4xx — e.g. a duplicate
`transaction_id` from replaying/backfilling the same window twice) is
different: retrying it will never succeed, so it is logged and dropped instead
of re-queued. Without this distinction, one permanently-doomed batch sits at
the front of the FIFO buffer and blocks every event queued behind it —
including brand new, perfectly valid ones — for the full backoff ceiling, over
and over, since a batch that can never succeed is retried exactly like one
that might.

Note that "permanent" is a specific list of statuses, NOT the whole 4xx range —
see `_PERMANENT_STATUSES`.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from .exceptions import LagoApiError

# Statuses where re-sending the SAME batch can never succeed: the request itself
# is the problem (malformed body, bad credentials, a transaction_id Lago has
# already accepted — which Lago reports as 422 `value_already_exist`, verified live,
# NOT as 409; 409 is in the set only as defence against an intermediary that uses it).
# Deliberately an explicit list rather than the 400-499 range, because two 4xx statuses
# mean "try again, later": 429 (rate limited) and 408 (request timeout). Treating those
# as permanent dropped billable events AND fanned one throttled batch out into up to
# `max_batch_size` extra requests aimed at the server that had just asked us to slow
# down.
#
# 413/415 are in the set for the OPPOSITE reason to 429: re-sending the same batch
# provably cannot succeed, because what makes it fail is a property OF THE BATCH (its
# size, its media type) and that is constant across retries. Treating them as transient
# re-prepended the identical batch at the head of the FIFO and backed off to 60s
# forever, blocking every event behind it until the buffer overflowed. Being "permanent"
# here routes them to `_send_individually`, which SPLITS the batch and delivers what is
# deliverable — so a 413 on a 100-event batch becomes 100 single-event sends rather than
# a stalled queue.
#
# Neither is reachable from Lago itself: its API surface is 400/401/403/404/405/422/429,
# and an oversized batch comes back 422 `too_many_events` (verified live against a real
# instance) which already routes to the split path. They are kept for an intermediary in
# front of Lago — nginx's `client_max_body_size` genuinely does answer 413 without the
# request ever reaching Rails.
#
# 402 was in this set and is NOT, deliberately. It fails the same test: "payment
# required" is a property of the ACCOUNT, not of the batch, so it stops being true the
# moment someone pays — the same shape as 429, recoverable by an out-of-band change
# rather than by sending something different. Classified permanent it was actively
# destructive: the batch 402s, routes to `_send_individually`, every isolated send 402s
# too, and each one is logged and dropped, so a lapsed account silently discarded every
# billable event for the whole outage with one `on_error` for the lot. Measured against a
# server returning 402: 5 events in, 6 HTTP calls out, 0 recoverable. As transient they
# are held and retried instead — a lapsed account head-of-line-blocks at the 60s ceiling
# until `max_buffer_size` overflows, which is bounded, oldest-first and reported, and
# fully recoverable if the account is fixed inside the buffer window.
#
# 405/410 stay transient: they usually indicate a misrouted or retired endpoint, which a
# deploy can fix.
_PERMANENT_STATUSES = frozenset({400, 401, 403, 404, 409, 413, 415, 422})


def _is_permanent_failure(exc: Exception) -> bool:
    """True when re-sending this exact batch can never succeed.

    A validation 4xx (bad request, duplicate transaction_id, revoked key) will
    fail identically forever, so it is isolated and dropped. Everything else —
    5xx, a network-level exception (timeout, connection error, no LagoApiError at
    all), and the throttling 4xxs 429/408 — might succeed later and stays
    retryable. An unrecognized 4xx is treated as transient: waiting on an event
    that would have been dropped costs a delay, dropping one that would have been
    accepted costs revenue.
    """
    return isinstance(exc, LagoApiError) and exc.status in _PERMANENT_STATUSES


logger = logging.getLogger("lago_agent_sdk.queue")


class EventQueue:
    def __init__(
        self,
        sender: Callable[[list[dict[str, Any]]], None],
        flush_interval: float = 1.0,
        max_batch_size: int = 100,
        max_buffer_size: int = 10_000,
        max_retry_seconds: float = 60.0,
        on_error: Callable[[Exception, str], None] | None = None,
        pricing: Any | None = None,
    ) -> None:
        self._sender = sender
        self._flush_interval = flush_interval
        self._max_batch_size = max_batch_size
        self._max_buffer_size = max_buffer_size
        self._max_retry_seconds = max_retry_seconds
        self._on_error = on_error
        # Optional PricingProvider — its (blocking) HTTP refresh runs on this
        # background thread so the customer's call is never blocked on pricing.
        self._pricing = pricing

        self._buffer: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._backoff_seconds = 0.0
        self._http_calls = 0  # for tests
        # Per-thread "already reporting an overflow" flag — see push().
        self._reporting = threading.local()

        self._thread = threading.Thread(target=self._run, name="lago-queue", daemon=True)
        self._thread.start()
        atexit.register(self._atexit_shutdown)

        # After fork, the daemon thread is gone in the child. Recreate it
        # along with fresh sync primitives — the buffer's contents are copied
        # over (which is fine: child re-emits its own events) but the lock
        # state from the parent is unsafe to reuse.
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_in_child)

    def _after_in_child(self) -> None:
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._buffer = deque()  # don't replay parent's events from the child
        self._backoff_seconds = 0.0
        self._http_calls = 0
        self._reporting = threading.local()
        # Note: the PricingProvider self-heals on fork via a PID check inside
        # lookup()/maybe_refresh(); we deliberately do NOT call into it from this
        # fork handler (touching it here changes thread timing enough to trip
        # macOS's objc fork-safety abort).
        self._thread = threading.Thread(target=self._run, name="lago-queue", daemon=True)
        self._thread.start()

    def wake(self) -> None:
        """Nudge the background thread to run its tick (drain + pricing
        `maybe_refresh()`) right now instead of waiting up to
        `flush_interval` seconds for its next scheduled tick. Just sets an
        in-memory flag — never blocks, never does I/O on the caller's
        thread."""
        self._wake.set()

    def push(self, event: dict[str, Any]) -> None:
        with self._lock:
            overflowed = len(self._buffer) >= self._max_buffer_size
            if overflowed:
                self._buffer.popleft()
            self._buffer.append(event)
            should_wake = len(self._buffer) >= self._max_batch_size
        # Both of these run with the lock RELEASED, the same shape `should_wake`
        # already used. Reporting inside the lock deadlocked the caller: `_lock` is a
        # plain Lock, not an RLock, so a customer `on_error` that touched the SDK at
        # all — emitting a diagnostic, forcing a flush — blocked forever on a lock its
        # own thread already held. Overflow happens under sustained load, which is
        # exactly when such a hook fires, and the failure is worse than the drop it
        # reports: an unnoticed dropped event costs one event, a hung producer thread
        # costs the application. The surrounding try/except cannot help, because a
        # deadlock is not an exception.
        #
        # Keeping the callback out of the lock also stops a full buffer from running
        # the hook plus a log write synchronously on the customer's LLM-call thread
        # while holding the lock every producer and the drain thread need.
        if overflowed and not getattr(self._reporting, "active", False):
            # Re-entrancy guard, per thread. Moving the report out of the lock fixed
            # the deadlock but exposed the other half: the buffer is full again by the
            # time the hook runs, so a hook that calls `push()` overflows again and
            # re-enters without bound. Suppressing the nested report breaks the cycle
            # while still letting the hook's own event be buffered. Per-thread so one
            # producer's hook can never silence another producer's report.
            self._reporting.active = True
            try:
                logger.warning("lago queue overflow at %d events; dropping oldest", self._max_buffer_size)
                # Also through on_error: an overflow drops BILLABLE events, and a
                # customer watching only the error hook — the documented channel for
                # every other billing gap — never learned revenue had been lost.
                self._report_error(
                    RuntimeError(
                        f"queue overflow at {self._max_buffer_size} events; dropped the oldest event"
                    ),
                    "overflow",
                )
            finally:
                self._reporting.active = False
        if should_wake:
            self._wake.set()

    def flush(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                empty = not self._buffer
            if empty:
                return True
            self._wake.set()
            time.sleep(0.01)
        return False

    def shutdown(self, timeout: float = 5.0) -> None:
        self.flush(timeout=timeout)
        self._stopping.set()
        self._wake.set()
        self._thread.join(timeout=timeout)

    def _atexit_shutdown(self) -> None:
        try:
            self.shutdown(timeout=2.0)
        except Exception:
            pass

    def _take_batch(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self._buffer:
                return []
            n = min(self._max_batch_size, len(self._buffer))
            batch = [self._buffer.popleft() for _ in range(n)]
        return batch

    def _replay_failed(self, batch: list[dict[str, Any]]) -> None:
        with self._lock:
            self._buffer.extendleft(reversed(batch))

    def _report_error(self, exc: Exception, where: str = "send_batch") -> None:
        """Best-effort `on_error` callback — a customer's own callback must
        never be allowed to break the queue's send/retry loop."""
        if self._on_error:
            try:
                self._on_error(exc, where)
            except Exception:  # noqa: BLE001
                pass

    def _send_individually(self, batch: list[dict[str, Any]], batch_exc: Exception) -> None:
        """Recovery path for a batch that failed with a permanent (4xx) error.

        Each event is sent alone: one that individually 4xxs (e.g. its own
        transaction_id really is a duplicate) is logged and dropped for good;
        one that succeeds alone is done; one that hits a TRANSIENT error while
        isolated is re-queued for the normal backoff-and-retry path, same as
        any other event. Reports once via on_error for the batch as a whole
        (the original exception) so a caller isn't flooded with N callbacks
        for what's really one root cause.
        """
        self._report_error(batch_exc)
        # Collected and re-queued ONCE at the end, not per event. `_replay_failed`
        # prepends, so calling it inside the loop reversed the survivors' relative
        # order: a 413 batch of a,b,c,d,e whose b,c,d fail transiently while isolated
        # came back as d,c,b. FIFO is the queue's contract — it is what makes the
        # oldest-dropped-first overflow policy and Lago's own event ordering
        # meaningful — so a recovery path must not silently invert it.
        retry: list[dict[str, Any]] = []
        for event in batch:
            try:
                self._http_calls += 1
                self._sender([event])
            except Exception as exc:  # noqa: BLE001
                if _is_permanent_failure(exc):
                    logger.warning(
                        "lago dropping event (permanent failure, will not retry): transaction_id=%s: %s",
                        event.get("transaction_id"),
                        exc,
                    )
                else:
                    logger.warning("lago send failed for isolated event, will retry: %s", exc)
                    retry.append(event)
        if retry:
            self._replay_failed(retry)

    def _run(self) -> None:
        while not self._stopping.is_set():
            self._wake.wait(timeout=self._flush_interval)
            self._wake.clear()

            # Refresh pricing tables on this background thread (off the hot path).
            if self._pricing is not None:
                try:
                    self._pricing.maybe_refresh()
                except Exception:  # noqa: BLE001 — pricing must never break the queue
                    pass

            while True:
                batch = self._take_batch()
                if not batch:
                    break
                if self._backoff_seconds:
                    if self._stopping.wait(timeout=self._backoff_seconds):
                        self._replay_failed(batch)
                        return
                try:
                    self._http_calls += 1
                    self._sender(batch)
                    self._backoff_seconds = 0.0
                except Exception as exc:  # noqa: BLE001
                    if _is_permanent_failure(exc):
                        # Lago's batch endpoint is all-or-nothing: a single bad
                        # transaction_id fails the WHOLE batch, even if the rest
                        # are perfectly valid — re-queuing the batch as-is would
                        # retry (and re-fail) forever, but dropping it outright
                        # would silently lose those valid events too. Isolate by
                        # falling back to one-by-one for this batch only; only
                        # the events that individually 4xx get dropped.
                        self._send_individually(batch, exc)
                        self._backoff_seconds = 0.0
                        continue
                    self._replay_failed(batch)
                    self._report_error(exc)
                    logger.warning("lago send_batch failed: %s", exc)
                    self._backoff_seconds = (
                        1.0
                        if self._backoff_seconds == 0
                        else min(self._backoff_seconds * 2, self._max_retry_seconds)
                    )
                    break
        # Drain on exit — keep sending until the buffer is truly empty, not
        # just one batch's worth (a buffer holding more than max_batch_size
        # events at shutdown previously left the rest never even attempted).
        # No more retries are possible once this thread exits, so unlike the
        # main loop, a transient failure here is ALSO final: it must be
        # logged, never silently swallowed the way a bare `except: pass`
        # previously did — that's what actually lost events, not the network
        # blip itself, which by itself is recoverable if it's just reported.
        # `_send_individually` re-queues transient sub-failures for retry —
        # appropriate for the main loop, which lives on, but during this exit
        # drain that could spin forever against a persistently-down network.
        # Bound the whole drain by wall-clock time; whatever's still in the
        # buffer once the budget is spent is logged as lost, not retried
        # forever in an exiting daemon thread.
        drain_deadline = time.monotonic() + min(self._max_retry_seconds, 10.0)
        while time.monotonic() < drain_deadline:
            batch = self._take_batch()
            if not batch:
                break
            try:
                self._sender(batch)
            except Exception as exc:  # noqa: BLE001
                if _is_permanent_failure(exc):
                    self._send_individually(batch, exc)
                else:
                    self._report_error(exc)
                    logger.warning(
                        "lago: %d event(s) LOST on shutdown — final drain failed with no more "
                        "retries possible: %s",
                        len(batch),
                        exc,
                    )
        with self._lock:
            stranded = len(self._buffer)
        if stranded:
            logger.warning("lago: %d event(s) LOST on shutdown — drain time budget exhausted", stranded)

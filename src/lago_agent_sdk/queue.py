"""Async batched event queue.

Thread-safe, in-memory. Background thread flushes every `flush_interval`
seconds or immediately when buffer reaches `max_batch_size`. On a TRANSIENT
send failure (network error, 5xx), re-prepends the batch and applies
exponential backoff (1s, 2s, 4s, 8s, capped at 60s). Resets on next success.

A PERMANENT failure (Lago 4xx — e.g. a duplicate `transaction_id` from
replaying/backfilling the same window twice) is different: retrying it will
never succeed, so it is logged and dropped instead of re-queued. Without this
distinction, one permanently-doomed batch sits at the front of the FIFO buffer
and blocks every event queued behind it — including brand new, perfectly
valid ones — for the full backoff ceiling, over and over, since a batch that
can never succeed is retried exactly like one that might.
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


def _is_permanent_failure(exc: Exception) -> bool:
    """A Lago 4xx (bad request, validation error, duplicate transaction_id,
    ...) will never succeed by retrying the exact same batch. A 5xx or a
    network-level exception (timeout, connection error, no LagoApiError at
    all) might — those stay retryable."""
    return isinstance(exc, LagoApiError) and 400 <= exc.status < 500


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
            if len(self._buffer) >= self._max_buffer_size:
                self._buffer.popleft()
                logger.warning("lago queue overflow at %d events; dropping oldest", self._max_buffer_size)
            self._buffer.append(event)
            should_wake = len(self._buffer) >= self._max_batch_size
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
                    self._replay_failed([event])

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

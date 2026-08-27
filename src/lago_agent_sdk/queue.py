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

# Statuses where re-sending the SAME batch can never succeed, because the BATCH is what
# is wrong. Deliberately an explicit list, not the 400-499 range.
#
# The test is: **is what makes this fail a property of the batch?** If a DIFFERENT
# PAYLOAD is what it takes to succeed, the batch is doomed and belongs here, where
# `_send_individually` splits it and delivers whatever is deliverable. If an OUT-OF-BAND
# change fixes it — someone rotates a key back, pays an invoice, corrects a URL, fixes a
# proxy — the events are still perfectly billable and must be HELD, because dropping
# them is unrecoverable while holding them is bounded (`max_buffer_size`, oldest-first,
# reported through `on_error`).
#
# Every line below was measured by driving this queue over a real socket at a server
# returning that status, counting events actually delivered (`probes/t11_status_matrix`):
#
#   400  malformed body — a different payload is the only fix. PERMANENT.
#   413  too large — the size IS the batch. Isolating it is a real recovery, not a
#        formality: against an nginx-style server answering 413 over a byte limit and
#        200 under it, the split path delivered 5 of 5. Held instead, it delivered 0 and
#        stalled at the backoff ceiling forever. Not reachable from Lago itself — an
#        oversized batch there answers 422 `too_many_events` (probed live, 20k events /
#        3.5 MiB) — so this exists for `client_max_body_size` in front of Lago.
#   409  a conflicting id. Lago answers 422 for a replayed `transaction_id`, not 409
#        (probed live); 409 stays as defence against an intermediary that uses it.
#   422  Lago's real answer for a duplicate id, an oversized batch and a bad
#        content-type. PERMANENT — but note it reaches `_send_individually`, which is
#        what lets the valid events in a batch survive one bad transaction_id.
#
# Everything else is transient, including these, which used to be here and lost money:
#
#   401/403  a rotated or revoked key. Measured with a server that healed after 3s —
#            i.e. the key put back — classified permanent this destroyed all 5 events
#            inside the first second, and none of them ever reached Lago. Held, all 5
#            were delivered when it healed.
#   402      payment required — a property of the ACCOUNT; it stops being true the
#            moment someone pays. Measured against a 402 server: 5 events in, 6 HTTP
#            calls out, 0 recoverable, one `on_error` for the lot.
#   404      the endpoint, not the events: Lago answers 404 `resource_not_found` for a
#            wrong PATH (probed live), which is a mistyped `api_url` — fixed out-of-band
#            like a rotated key, and the same class as the 405/410 that were already
#            transient here for exactly that reason. It was destroying every event.
#   415      a wrong media type comes from a proxy, and splitting cannot help: this
#            client always sends `application/json`, so every isolated send fails the
#            same way — measured, all 5 dropped. Held, they survive the proxy being
#            fixed. (Lago itself answers 422 to a bad content-type, probed live.)
#   429/408  throttling — fanning a batch into N isolated sends aims more traffic at a
#            server that just asked us to slow down.
#
# An unrecognized 4xx is transient too: waiting on an event that would have been dropped
# costs a delay, dropping one that would have been accepted costs revenue.
_PERMANENT_STATUSES = frozenset({400, 409, 413, 422})


def _is_permanent_failure(exc: Exception) -> bool:
    """True when re-sending this exact batch can never succeed.

    Only a malformed or unacceptable BATCH qualifies — see `_PERMANENT_STATUSES` for
    the test and for what each status cost when it was on the wrong side of it.
    Everything else (5xx, a network-level exception with no LagoApiError at all, a
    credential or account or endpoint 4xx, an unrecognized 4xx) might succeed later
    and stays retryable.
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
        # `_take_batch` pops events out before the send, so mid-POST they are in
        # neither the buffer nor Lago. `flush()` has to wait on this too.
        self._in_flight = 0
        # Per-thread "already reporting an overflow" flag — see push().
        self._reporting = threading.local()

        self._thread = threading.Thread(target=self._run, name="lago-queue", daemon=True)
        self._thread.start()
        atexit.register(self._atexit_shutdown)

        # After fork the daemon thread is gone in the child. Recreate it with fresh
        # sync primitives; the parent's buffer is dropped rather than inherited (see
        # `_after_in_child`) so the two can never both deliver the same events.
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_in_child)

    def _after_in_child(self) -> None:
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._buffer = deque()  # don't replay parent's events from the child
        self._backoff_seconds = 0.0
        self._http_calls = 0
        self._in_flight = 0
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
                settled = not self._buffer and self._in_flight == 0
            if settled:
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
            self._in_flight += n
        return batch

    def _settle(self, n: int) -> None:
        """Delivered, dropped for good, or back on the buffer — no longer in flight."""
        if n <= 0:
            return
        with self._lock:
            self._in_flight = max(0, self._in_flight - n)

    def _replay_failed(self, batch: list[dict[str, Any]]) -> None:
        with self._lock:
            self._buffer.extendleft(reversed(batch))
            # Same lock acquisition as the re-queue, or flush() sees neither.
            self._in_flight = max(0, self._in_flight - len(batch))

    def _report_error(self, exc: Exception, where: str = "send_batch") -> None:
        """Best-effort `on_error` callback — a customer's own callback must
        never be allowed to break the queue's send/retry loop."""
        if self._on_error:
            try:
                self._on_error(exc, where)
            except Exception:  # noqa: BLE001
                pass

    def _send_individually(
        self,
        batch: list[dict[str, Any]],
        batch_exc: Exception,
        requeue_transient: bool = True,
    ) -> int:
        """Recovery path for a batch that failed with a permanent (4xx) error.

        Each event is sent alone: one that individually 4xxs (e.g. its own
        transaction_id really is a duplicate) is logged and dropped for good;
        one that succeeds alone is done; one that hits a TRANSIENT error while
        isolated is re-queued for the normal backoff-and-retry path, same as
        any other event. Reports once via on_error for the batch as a whole
        (the original exception) so a caller isn't flooded with N callbacks
        for what's really one root cause.

        Returns the number of events re-queued, which the caller needs in order to
        decide whether it may keep draining immediately or must back off first —
        see `_drain_buffer`. `requeue_transient=False` is for the exit drain, where
        there is no later retry to re-queue TO: an event that fails there is lost
        and must be reported as such rather than put back on a buffer nobody will
        read again.
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
                elif requeue_transient:
                    logger.warning("lago send failed for isolated event, will retry: %s", exc)
                    retry.append(event)
                else:
                    self._report_error(exc)
                    logger.warning(
                        "lago: event LOST on shutdown — no retry left: transaction_id=%s: %s",
                        event.get("transaction_id"),
                        exc,
                    )
        if retry:
            self._replay_failed(retry)  # settles those
        self._settle(len(batch) - len(retry))
        return len(retry)

    def _next_backoff(self) -> float:
        """1s -> 2s -> 4s -> ... -> `max_retry_seconds`."""
        if self._backoff_seconds == 0:
            return 1.0
        return min(self._backoff_seconds * 2, self._max_retry_seconds)

    def _drain_buffer(self) -> None:
        """Send everything buffered, one batch at a time, until the buffer is empty or
        a failure hands the batch to the retry backoff.

        Returns rather than looping on a failure: the caller waits out
        `flush_interval` and comes back.
        """
        while True:
            batch = self._take_batch()
            if not batch:
                return
            # Re-checked every iteration, not only around the backoff wait. Always
            # `return` after re-queuing, never a path that abandons the batch: the exit
            # drain is what reports whatever it cannot send, so the events have to be
            # back on the buffer before this leaves.
            if self._stopping.is_set():
                self._replay_failed(batch)
                return
            if self._backoff_seconds:
                if self._stopping.wait(timeout=self._backoff_seconds):
                    self._replay_failed(batch)
                    return
            try:
                self._http_calls += 1
                self._sender(batch)
                self._backoff_seconds = 0.0
                self._settle(len(batch))
            except Exception as exc:  # noqa: BLE001
                if _is_permanent_failure(exc):
                    # Lago's batch endpoint is all-or-nothing: a single bad
                    # transaction_id fails the WHOLE batch, even if the rest
                    # are perfectly valid — re-queuing the batch as-is would
                    # retry (and re-fail) forever, but dropping it outright
                    # would silently lose those valid events too. Isolate by
                    # falling back to one-by-one for this batch only; only
                    # the events that individually 4xx get dropped.
                    requeued = self._send_individually(batch, exc)
                    if requeued == 0:
                        # Batch fully resolved — the buffer shrank, so keep draining.
                        self._backoff_seconds = 0.0
                        continue
                    # Some isolated sends failed transiently and went back on the
                    # buffer. Continuing here would re-take them with no delay and
                    # re-fail at the speed of the failure. Measured on this exact pair
                    # (422 on the batch, 429 on every isolated send): 280,388 HTTP
                    # requests in 1.2s, aimed at the server that had just asked us to
                    # slow down. They must go through the normal backoff path.
                    self._backoff_seconds = self._next_backoff()
                    return
                self._replay_failed(batch)
                self._report_error(exc)
                logger.warning("lago send_batch failed: %s", exc)
                self._backoff_seconds = self._next_backoff()
                return

    def _run(self) -> None:
        while not self._stopping.is_set():
            self._wake.wait(timeout=self._flush_interval)
            self._wake.clear()

            # Drain BEFORE refreshing pricing, not after. `maybe_refresh()` does HTTP —
            # up to a 10s timeout per source — and refreshing first put that latency in
            # front of every queued billable event, on every tick: measured, a 600ms
            # refresh delayed the first delivery to 629ms. Nothing in the drain depends
            # on it — an event's price was already resolved at emit() time, so a fresh
            # table only ever matters to the NEXT call.
            self._drain_buffer()

            # Refresh pricing tables on this background thread (off the hot path).
            # Skipped once shutting down: a fetch here can take the full HTTP timeout,
            # and it would spend the caller's shutdown budget on a table nothing will
            # ever read.
            if self._pricing is not None and not self._stopping.is_set():
                try:
                    self._pricing.maybe_refresh()
                except Exception:  # noqa: BLE001 — pricing must never break the queue
                    pass

            # Anything pushed while the refresh was in flight would otherwise wait out
            # a whole flush interval on top of it.
            if not self._stopping.is_set():
                self._drain_buffer()

        # Drain on exit — keep sending until the buffer is truly empty, not
        # just one batch's worth (a buffer holding more than max_batch_size
        # events at shutdown previously left the rest never even attempted).
        # No more retries are possible once this thread exits, so unlike the
        # main loop, a transient failure here is ALSO final: it must be
        # logged, never silently swallowed the way a bare `except: pass`
        # previously did — that's what actually lost events, not the network
        # blip itself, which by itself is recoverable if it's just reported.
        # `_send_individually` re-queues transient sub-failures for retry —
        # appropriate for the main loop, which lives on, but here it would spin
        # against a persistently-down network, so this drain passes
        # `requeue_transient=False`. The drain is additionally bounded by wall-clock
        # time; whatever's still in the buffer once the budget is spent is logged as
        # lost, not retried forever in an exiting daemon thread.
        drain_deadline = time.monotonic() + min(self._max_retry_seconds, 10.0)
        while time.monotonic() < drain_deadline:
            batch = self._take_batch()
            if not batch:
                break
            try:
                self._sender(batch)
                self._settle(len(batch))
            except Exception as exc:  # noqa: BLE001
                if _is_permanent_failure(exc):
                    # `requeue_transient=False`: re-queuing here would put the event
                    # back on a buffer this loop immediately re-takes, with no later
                    # retry to reach — a hot loop for the whole drain budget. Report
                    # it as lost instead.
                    self._send_individually(batch, exc, requeue_transient=False)
                else:
                    self._report_error(exc)
                    logger.warning(
                        "lago: %d event(s) LOST on shutdown — final drain failed with no more "
                        "retries possible: %s",
                        len(batch),
                        exc,
                    )
                    self._settle(len(batch))
        with self._lock:
            stranded = len(self._buffer)
        if stranded:
            logger.warning("lago: %d event(s) LOST on shutdown — drain time budget exhausted", stranded)

"""Async batched event queue.

Thread-safe, in-memory. Background thread flushes every `flush_interval`
seconds or immediately when buffer reaches `max_batch_size`. On send
failure, re-prepends the batch and applies exponential backoff
(1s, 2s, 4s, 8s, capped at 60s). Resets on next success.
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
                    self._replay_failed(batch)
                    if self._on_error:
                        try:
                            self._on_error(exc, "send_batch")
                        except Exception:  # noqa: BLE001
                            pass
                    logger.warning("lago send_batch failed: %s", exc)
                    self._backoff_seconds = (
                        1.0
                        if self._backoff_seconds == 0
                        else min(self._backoff_seconds * 2, self._max_retry_seconds)
                    )
                    break
        # drain on exit
        batch = self._take_batch()
        if batch:
            try:
                self._sender(batch)
            except Exception:  # noqa: BLE001
                pass

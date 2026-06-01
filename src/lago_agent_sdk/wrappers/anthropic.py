"""anthropic SDK wrapper.

Wraps the public methods of `Anthropic.messages` (and `AsyncAnthropic.messages`)
in place — instrumentation never breaks the customer's call.

Methods wrapped:
  - .create(...)                   — non-streaming and stream=True both supported
  - .stream(...)                   — sync context-manager helper
  - AsyncMessages.create(...)      — async non-streaming and stream=True
  - AsyncMessages.stream(...)      — async context-manager helper

Per-call override: pop `extra_lago={"subscription": ..., "dimensions": ...}` from kwargs
before forwarding so Anthropic's strict validation doesn't reject it.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ..adapters import extract_anthropic_native

logger = logging.getLogger("lago_agent_sdk.wrappers.anthropic")

_INSTRUMENTED_ATTR = "_lago_instrumented"
_LAGO_KWARG = "extra_lago"


def _pop_lago_kwarg(kwargs: dict[str, Any]) -> dict[str, Any]:
    return kwargs.pop(_LAGO_KWARG, {}) or {}


def _is_message_like(obj: Any) -> bool:
    """Anthropic Message objects expose `.usage` and `.content`; streams don't.

    Safe against properties that raise — falls through to False so the customer's
    call is never broken by attribute-access surprises in their custom objects.
    """
    try:
        if isinstance(obj, dict):
            return "usage" in obj
        # hasattr propagates non-AttributeError exceptions on Py3; guard explicitly.
        return hasattr(obj, "usage")
    except Exception:  # noqa: BLE001
        return False


def wrap_anthropic_client(
    sdk: Any,
    client: Any,
    dimensions: dict[str, Any] | None = None,
    subscription: str | None = None,
) -> Any:
    """In-place wrap of an `anthropic.Anthropic` or `anthropic.AsyncAnthropic` client. Idempotent."""
    if getattr(client, _INSTRUMENTED_ATTR, False):
        logger.info("lago: anthropic client already wrapped — skipping")
        return client

    base_dims = dict(dimensions or {})
    base_sub = subscription

    messages = getattr(client, "messages", None)
    if messages is None:
        logger.warning("lago: anthropic client has no .messages — skipping wrap")
        return client

    original_create = getattr(messages, "create", None)
    original_stream = getattr(messages, "stream", None)
    is_async = type(client).__name__.startswith("Async")

    def _resolve_opts(lago_opts: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        sub = lago_opts.get("subscription") or base_sub
        dims = {**base_dims, **(lago_opts.get("dimensions") or {})}
        return sub, dims

    def _emit_from(payload: Any, model_id: str, sub: str | None, dims: dict[str, Any]) -> None:
        try:
            usage = extract_anthropic_native(payload, model_id=model_id)
            sdk.emit(usage, subscription=sub, dimensions=dims)
        except Exception as exc:  # noqa: BLE001
            logger.warning("lago: anthropic emit failed: %s", exc)

    # ------------------------------------------------------------------
    # Sync messages.create — auto-detects streaming via response shape
    # ------------------------------------------------------------------
    def _create(*args: Any, **kwargs: Any) -> Any:
        assert original_create is not None
        lago_opts = _pop_lago_kwarg(kwargs)
        model_id = kwargs.get("model", "")
        sub, dims = _resolve_opts(lago_opts)
        response = original_create(*args, **kwargs)

        if _is_message_like(response):
            _emit_from(response, model_id, sub, dims)
            return response

        # Streaming — wrap the iterator to capture the final usage on close.
        def _wrap_stream(src: Iterator[Any]) -> Iterator[Any]:
            last_usage: dict[str, Any] | None = None
            try:
                for event in src:
                    payload = event.model_dump() if hasattr(event, "model_dump") else event
                    if isinstance(payload, dict):
                        usage = payload.get("usage")
                        if isinstance(usage, dict):
                            last_usage = {"usage": usage}
                    yield event
            finally:
                if last_usage is not None:
                    _emit_from(last_usage, model_id, sub, dims)

        return _wrap_stream(response)

    # ------------------------------------------------------------------
    # Async messages.create — same as sync, awaited
    # ------------------------------------------------------------------
    async def _create_async(*args: Any, **kwargs: Any) -> Any:
        assert original_create is not None
        lago_opts = _pop_lago_kwarg(kwargs)
        model_id = kwargs.get("model", "")
        sub, dims = _resolve_opts(lago_opts)
        response = await original_create(*args, **kwargs)

        if _is_message_like(response):
            _emit_from(response, model_id, sub, dims)
            return response

        async def _wrap_async_stream(src: AsyncIterator[Any]) -> AsyncIterator[Any]:
            last_usage: dict[str, Any] | None = None
            try:
                async for event in src:
                    payload = event.model_dump() if hasattr(event, "model_dump") else event
                    if isinstance(payload, dict):
                        usage = payload.get("usage")
                        if isinstance(usage, dict):
                            last_usage = {"usage": usage}
                    yield event
            finally:
                if last_usage is not None:
                    _emit_from(last_usage, model_id, sub, dims)

        return _wrap_async_stream(response)

    # ------------------------------------------------------------------
    # messages.stream context manager (sync + async)
    #
    # Anthropic returns a MessageStreamManager (sync) / AsyncMessageStreamManager
    # (async). Both have .__enter__/.__exit__ and the inner stream object
    # exposes .get_final_message() after the with-block closes.
    # ------------------------------------------------------------------
    def _wrap_stream_manager(*args: Any, **kwargs: Any) -> Any:
        assert original_stream is not None
        lago_opts = _pop_lago_kwarg(kwargs)
        model_id = kwargs.get("model", "")
        sub, dims = _resolve_opts(lago_opts)
        inner = original_stream(*args, **kwargs)
        return _LagoStreamManager(inner, sdk, model_id, sub, dims, is_async=is_async)

    if original_create is not None:
        messages.create = _create_async if is_async else _create
    if original_stream is not None:
        messages.stream = _wrap_stream_manager

    setattr(client, _INSTRUMENTED_ATTR, True)
    return client


class _LagoStreamManager:
    """Proxies Anthropic's MessageStreamManager and emits on close.

    Works for both sync (`with`) and async (`async with`) variants by detecting
    which __exit__ kind is being called.
    """

    def __init__(
        self,
        inner: Any,
        sdk: Any,
        model_id: str,
        sub: str | None,
        dims: dict[str, Any],
        *,
        is_async: bool,
    ) -> None:
        self._inner = inner
        self._sdk = sdk
        self._model_id = model_id
        self._sub = sub
        self._dims = dims
        self._stream: Any = None
        self._is_async = is_async

    # ----- sync -----
    def __enter__(self) -> Any:
        self._stream = self._inner.__enter__()
        return self._stream

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            result = self._inner.__exit__(exc_type, exc, tb)
        finally:
            self._emit_final()
        return result

    # ----- async -----
    async def __aenter__(self) -> Any:
        self._stream = await self._inner.__aenter__()
        return self._stream

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        try:
            result = await self._inner.__aexit__(exc_type, exc, tb)
        finally:
            await self._emit_final_async()
        return result

    def _emit_final(self) -> None:
        """Sync path — `get_final_message()` returns the final message directly."""
        try:
            if not self._stream or not hasattr(self._stream, "get_final_message"):
                return
            final = self._stream.get_final_message()
            if final is not None:
                from ..adapters import extract_anthropic_native

                usage = extract_anthropic_native(final, model_id=self._model_id)
                self._sdk.emit(usage, subscription=self._sub, dimensions=self._dims)
        except Exception as exc:  # noqa: BLE001
            logger.warning("lago: anthropic stream-manager emit failed: %s", exc)

    async def _emit_final_async(self) -> None:
        """Async path — `AsyncMessageStream.get_final_message()` is a coroutine.

        Calling it without `await` returns an un-awaited coroutine object that
        the adapter sees as `{}` → zero usage emitted, plus a RuntimeWarning.
        Must await.
        """
        try:
            if not self._stream or not hasattr(self._stream, "get_final_message"):
                return
            final = await self._stream.get_final_message()
            if final is not None:
                from ..adapters import extract_anthropic_native

                usage = extract_anthropic_native(final, model_id=self._model_id)
                self._sdk.emit(usage, subscription=self._sub, dimensions=self._dims)
        except Exception as exc:  # noqa: BLE001
            logger.warning("lago: anthropic async stream-manager emit failed: %s", exc)

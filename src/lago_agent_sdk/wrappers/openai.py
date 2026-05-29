"""openai SDK wrapper.

Wraps the public methods of `OpenAI` (and `AsyncOpenAI`) clients in place —
instrumentation never breaks the customer's call.

Methods wrapped:
  - .chat.completions.create(...)  — non-streaming and stream=True both supported
  - .responses.create(...)         — Responses API, sync + streaming
  - AsyncOpenAI variants of both   — async non-streaming and stream=True

Streaming behavior:
  When `stream=True` is passed without `stream_options={"include_usage": True}`
  (Chat Completions) we automatically inject it so the final chunk carries the
  usage payload we need to bill. Without that flag, OpenAI's stream emits no
  usage data and the customer gets silent under-billing.

Per-call override: pop `extra_lago={"subscription": ..., "dimensions": ...}` from
kwargs before forwarding so OpenAI's strict validation doesn't reject it.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ..adapters import extract_openai_native

logger = logging.getLogger("lago_agent_sdk.wrappers.openai")

_INSTRUMENTED_ATTR = "_lago_instrumented"
_LAGO_KWARG = "extra_lago"


def _pop_lago_kwarg(kwargs: dict[str, Any]) -> dict[str, Any]:
    return kwargs.pop(_LAGO_KWARG, {}) or {}


def _ensure_stream_options_include_usage(kwargs: dict[str, Any]) -> None:
    """If stream=True without include_usage, inject it. No-op otherwise.

    Only meaningful for Chat Completions; the Responses API exposes usage on its
    own final event already.
    """
    if not kwargs.get("stream"):
        return
    so = kwargs.get("stream_options")
    if isinstance(so, dict):
        # Respect customer's explicit choice if they set it
        if "include_usage" in so:
            return
        kwargs["stream_options"] = {**so, "include_usage": True}
    else:
        kwargs["stream_options"] = {"include_usage": True}


def _is_response_like(obj: Any) -> bool:
    """Real responses expose `.usage`; Stream iterators don't.

    Safe against properties that raise — falls through to False so the customer's
    call is never broken.
    """
    try:
        if isinstance(obj, dict):
            return "usage" in obj
        return hasattr(obj, "usage")
    except Exception:  # noqa: BLE001
        return False


def wrap_openai_client(
    sdk: Any,
    client: Any,
    dimensions: dict[str, Any] | None = None,
    subscription: str | None = None,
) -> Any:
    """In-place wrap of an `openai.OpenAI` or `openai.AsyncOpenAI` client. Idempotent."""
    if getattr(client, _INSTRUMENTED_ATTR, False):
        logger.info("lago: openai client already wrapped — skipping")
        return client

    base_dims = dict(dimensions or {})
    base_sub = subscription
    is_async = type(client).__name__.startswith("Async")

    def _resolve_opts(lago_opts: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        sub = lago_opts.get("subscription") or base_sub
        dims = {**base_dims, **(lago_opts.get("dimensions") or {})}
        return sub, dims

    def _emit_from(payload: Any, model_id: str, sub: str | None, dims: dict[str, Any]) -> None:
        try:
            usage = extract_openai_native(payload, model_id=model_id)
            sdk.emit(usage, subscription=sub, dimensions=dims)
        except Exception as exc:  # noqa: BLE001
            logger.warning("lago: openai emit failed: %s", exc)

    def _make_sync_create(original: Any) -> Any:
        def _create(*args: Any, **kwargs: Any) -> Any:
            lago_opts = _pop_lago_kwarg(kwargs)
            _ensure_stream_options_include_usage(kwargs)
            model_id = kwargs.get("model", "")
            sub, dims = _resolve_opts(lago_opts)
            response = original(*args, **kwargs)

            if _is_response_like(response):
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
                            if isinstance(usage, dict) and usage:
                                last_usage = {"usage": usage}
                        yield event
                finally:
                    if last_usage is not None:
                        _emit_from(last_usage, model_id, sub, dims)

            return _wrap_stream(response)

        return _create

    def _make_async_create(original: Any) -> Any:
        async def _create_async(*args: Any, **kwargs: Any) -> Any:
            lago_opts = _pop_lago_kwarg(kwargs)
            _ensure_stream_options_include_usage(kwargs)
            model_id = kwargs.get("model", "")
            sub, dims = _resolve_opts(lago_opts)
            response = await original(*args, **kwargs)

            if _is_response_like(response):
                _emit_from(response, model_id, sub, dims)
                return response

            async def _wrap_async_stream(src: AsyncIterator[Any]) -> AsyncIterator[Any]:
                last_usage: dict[str, Any] | None = None
                try:
                    async for event in src:
                        payload = event.model_dump() if hasattr(event, "model_dump") else event
                        if isinstance(payload, dict):
                            usage = payload.get("usage")
                            if isinstance(usage, dict) and usage:
                                last_usage = {"usage": usage}
                        yield event
                finally:
                    if last_usage is not None:
                        _emit_from(last_usage, model_id, sub, dims)

            return _wrap_async_stream(response)

        return _create_async

    # ------------------------------------------------------------------
    # chat.completions.create
    # ------------------------------------------------------------------
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None) if chat is not None else None
    if completions is not None:
        original_chat_create = getattr(completions, "create", None)
        if original_chat_create is not None:
            completions.create = (
                _make_async_create(original_chat_create) if is_async else _make_sync_create(original_chat_create)
            )

    # ------------------------------------------------------------------
    # responses.create
    # ------------------------------------------------------------------
    responses_namespace = getattr(client, "responses", None)
    if responses_namespace is not None:
        original_responses_create = getattr(responses_namespace, "create", None)
        if original_responses_create is not None:
            responses_namespace.create = (
                _make_async_create(original_responses_create)
                if is_async
                else _make_sync_create(original_responses_create)
            )

    setattr(client, _INSTRUMENTED_ATTR, True)
    return client

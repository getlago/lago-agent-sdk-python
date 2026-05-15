"""mistralai SDK wrapper.

Wraps `Mistral.chat.complete` and `.stream` in place — instrumentation never
breaks the customer's call. Streaming captures usage from the final chunk.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

from ..adapters import extract_mistral_native

logger = logging.getLogger("lago_agent_sdk.wrappers.mistral")

_INSTRUMENTED_ATTR = "_lago_instrumented"
_LAGO_KWARG = "extra_lago"


def _pop_lago_kwarg(kwargs: dict[str, Any]) -> dict[str, Any]:
    return kwargs.pop(_LAGO_KWARG, {}) or {}


def _to_dict(obj: Any) -> dict[str, Any]:
    """Best-effort pydantic-or-dict to dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return cast(dict[str, Any], obj.model_dump())
        except Exception:  # noqa: BLE001
            pass
    return {}


def wrap_mistral_client(
    sdk: Any,
    client: Any,
    dimensions: dict[str, Any] | None = None,
    subscription: str | None = None,
) -> Any:
    """In-place wrap of a `mistralai.client.Mistral` client. Idempotent."""
    if getattr(client, _INSTRUMENTED_ATTR, False):
        logger.info("lago: mistral client already wrapped — skipping")
        return client

    base_dims = dict(dimensions or {})
    base_sub = subscription

    chat = getattr(client, "chat", None)
    if chat is None:
        logger.warning("lago: mistral client has no .chat — skipping wrap")
        return client

    original_complete = getattr(chat, "complete", None)
    original_stream = getattr(chat, "stream", None)
    original_complete_async = getattr(chat, "complete_async", None)
    original_stream_async = getattr(chat, "stream_async", None)

    def _resolve_opts(lago_opts: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        sub = lago_opts.get("subscription") or base_sub
        dims = {**base_dims, **(lago_opts.get("dimensions") or {})}
        return sub, dims

    # ------------------------------------------------------------------
    # chat.complete — non-streaming
    # ------------------------------------------------------------------
    def _complete(*args: Any, **kwargs: Any) -> Any:
        assert original_complete is not None  # guaranteed by outer if-guard
        lago_opts = _pop_lago_kwarg(kwargs)
        model_id = kwargs.get("model", "")
        response = original_complete(*args, **kwargs)
        try:
            usage = extract_mistral_native(_to_dict(response), model_id=model_id)
            sub, dims = _resolve_opts(lago_opts)
            sdk.emit(usage, subscription=sub, dimensions=dims)
        except Exception as exc:  # noqa: BLE001 — never break the call
            logger.warning("lago: mistral.chat.complete instrumentation failed: %s", exc)
        return response

    # ------------------------------------------------------------------
    # chat.stream — capture usage from final chunk's data.usage
    # ------------------------------------------------------------------
    def _stream(*args: Any, **kwargs: Any) -> Any:
        assert original_stream is not None  # guaranteed by outer if-guard
        lago_opts = _pop_lago_kwarg(kwargs)
        model_id = kwargs.get("model", "")
        original_iter = original_stream(*args, **kwargs)

        def _wrap_iter() -> Iterator[Any]:
            last_usage: dict[str, Any] | None = None
            try:
                for event in original_iter:
                    payload = _to_dict(event)
                    inner = payload.get("data") or {}
                    if isinstance(inner, dict) and inner.get("usage"):
                        last_usage = {"usage": inner["usage"], "model": inner.get("model", model_id)}
                    yield event
            finally:
                if last_usage is not None:
                    try:
                        usage = extract_mistral_native(last_usage, model_id=model_id)
                        sub, dims = _resolve_opts(lago_opts)
                        sdk.emit(usage, subscription=sub, dimensions=dims)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("lago: mistral.chat.stream instrumentation failed: %s", exc)

        return _wrap_iter()

    # ------------------------------------------------------------------
    # async variants — same shape, awaited
    # ------------------------------------------------------------------
    async def _complete_async(*args: Any, **kwargs: Any) -> Any:
        assert original_complete_async is not None  # guaranteed by outer if-guard
        lago_opts = _pop_lago_kwarg(kwargs)
        model_id = kwargs.get("model", "")
        response = await original_complete_async(*args, **kwargs)
        try:
            usage = extract_mistral_native(_to_dict(response), model_id=model_id)
            sub, dims = _resolve_opts(lago_opts)
            sdk.emit(usage, subscription=sub, dimensions=dims)
        except Exception as exc:  # noqa: BLE001
            logger.warning("lago: mistral.chat.complete_async instrumentation failed: %s", exc)
        return response

    def _stream_async(*args: Any, **kwargs: Any) -> Any:
        assert original_stream_async is not None  # guaranteed by outer if-guard
        lago_opts = _pop_lago_kwarg(kwargs)
        model_id = kwargs.get("model", "")

        async def _agen() -> AsyncIterator[Any]:
            assert original_stream_async is not None
            ait = original_stream_async(*args, **kwargs)
            last_usage: dict[str, Any] | None = None
            try:
                async for event in ait:
                    payload = _to_dict(event)
                    inner = payload.get("data") or {}
                    if isinstance(inner, dict) and inner.get("usage"):
                        last_usage = {"usage": inner["usage"], "model": inner.get("model", model_id)}
                    yield event
            finally:
                if last_usage is not None:
                    try:
                        usage = extract_mistral_native(last_usage, model_id=model_id)
                        sub, dims = _resolve_opts(lago_opts)
                        sdk.emit(usage, subscription=sub, dimensions=dims)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("lago: mistral.chat.stream_async instrumentation failed: %s", exc)

        return _agen()

    if original_complete is not None:
        chat.complete = _complete
    if original_stream is not None:
        chat.stream = _stream
    if original_complete_async is not None:
        chat.complete_async = _complete_async
    if original_stream_async is not None:
        chat.stream_async = _stream_async

    setattr(client, _INSTRUMENTED_ATTR, True)
    return client

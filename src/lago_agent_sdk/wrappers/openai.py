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

Gateway cache-hit detection (non-streaming only):
  Non-streaming calls go through `.with_raw_response.create(...)` instead of
  `.create(...)` so we can see response headers before parsing the body. If a
  gateway in front of the provider (e.g. Cloudflare AI Gateway) marks the
  response `cf-aig-cache-status: HIT`, the provider served it from cache at zero
  cost to the customer — we skip billing it. `.parse()` on the raw response
  returns the exact same object `.create()` would, so nothing downstream changes.
  This is a no-op when there's no gateway in the path: the header is simply
  absent. Streaming calls are NOT covered — OpenAI recommends
  `.with_streaming_response` for that, which behaves differently and hasn't been
  verified end-to-end, so streaming keeps using the plain `.create()` path.

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


def _is_cache_hit(raw_response: Any) -> bool:
    """True if a gateway in front of the provider served this from cache.

    A cache hit (Cloudflare AI Gateway: `cf-aig-cache-status: HIT`) costs the
    provider — and the customer — nothing. Billing it would overcharge for a
    call that never actually happened. Safe no-op with no gateway in the path:
    `.headers.get(...)` simply returns None.
    """
    try:
        return bool(raw_response.headers.get("cf-aig-cache-status") == "HIT")
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

    def _resolve_opts(lago_opts: dict[str, Any]) -> dict[str, Any]:
        return {
            "subscription": lago_opts.get("subscription") or base_sub,
            "dimensions": {**base_dims, **(lago_opts.get("dimensions") or {})},
            "mode": lago_opts.get("mode"),
            "markup": lago_opts.get("markup"),
        }

    def _emit_from(payload: Any, model_id: str, opts: dict[str, Any]) -> None:
        try:
            usage = extract_openai_native(payload, model_id=model_id)
            sdk.emit(usage, **opts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("lago: openai emit failed: %s", exc)

    def _extract_stream_usage(payload: Any) -> dict[str, Any] | None:
        """Pull usage out of a stream event, handling both API shapes.

        Chat Completions: usage sits at the top of the final chunk
        (`{"usage": {...}}`).
        Responses API:    usage sits under `event.response.usage` on the
        terminal `response.completed` event (`{"type": "response.completed",
        "response": {"usage": {...}}}`).
        """
        if not isinstance(payload, dict):
            return None
        usage = payload.get("usage")
        if isinstance(usage, dict) and usage:
            return {"usage": usage}
        # Responses API stream events nest usage under `.response.usage`
        response = payload.get("response")
        if isinstance(response, dict):
            nested = response.get("usage")
            if isinstance(nested, dict) and nested:
                return {"usage": nested}
        return None

    def _make_sync_create(original: Any, raw_create: Any | None, is_responses_api: bool = False) -> Any:
        def _create(*args: Any, **kwargs: Any) -> Any:
            lago_opts = _pop_lago_kwarg(kwargs)
            # `stream_options.include_usage` is a Chat-Completions-only knob.
            # The Responses API rejects it; injecting would break the call.
            if not is_responses_api:
                _ensure_stream_options_include_usage(kwargs)
            model_id = kwargs.get("model", "")
            opts = _resolve_opts(lago_opts)

            if not kwargs.get("stream") and raw_create is not None:
                # Non-streaming with `.with_raw_response` available: see gateway
                # headers before parsing — `.parse()` returns the identical object
                # `.create()` would have, so the customer sees no difference.
                raw = raw_create(*args, **kwargs)
                response = raw.parse()
                if _is_response_like(response) and not _is_cache_hit(raw):
                    _emit_from(response, model_id, opts)
                return response

            # Streaming, or `.with_raw_response` unavailable (older/custom client)
            # — plain `.create()`. No cache-hit detection possible on this path.
            response = original(*args, **kwargs)

            if _is_response_like(response):
                _emit_from(response, model_id, opts)
                return response

            def _wrap_stream(src: Iterator[Any]) -> Iterator[Any]:
                last_usage: dict[str, Any] | None = None
                try:
                    for event in src:
                        payload = event.model_dump() if hasattr(event, "model_dump") else event
                        extracted = _extract_stream_usage(payload)
                        if extracted is not None:
                            last_usage = extracted
                        yield event
                finally:
                    if last_usage is not None:
                        _emit_from(last_usage, model_id, opts)

            return _wrap_stream(response)

        return _create

    def _make_async_create(original: Any, raw_create: Any | None, is_responses_api: bool = False) -> Any:
        async def _create_async(*args: Any, **kwargs: Any) -> Any:
            lago_opts = _pop_lago_kwarg(kwargs)
            if not is_responses_api:
                _ensure_stream_options_include_usage(kwargs)
            model_id = kwargs.get("model", "")
            opts = _resolve_opts(lago_opts)

            if not kwargs.get("stream") and raw_create is not None:
                raw = await raw_create(*args, **kwargs)
                response = raw.parse()
                if _is_response_like(response) and not _is_cache_hit(raw):
                    _emit_from(response, model_id, opts)
                return response

            response = await original(*args, **kwargs)

            if _is_response_like(response):
                _emit_from(response, model_id, opts)
                return response

            async def _wrap_async_stream(src: AsyncIterator[Any]) -> AsyncIterator[Any]:
                last_usage: dict[str, Any] | None = None
                try:
                    async for event in src:
                        payload = event.model_dump() if hasattr(event, "model_dump") else event
                        extracted = _extract_stream_usage(payload)
                        if extracted is not None:
                            last_usage = extracted
                        yield event
                finally:
                    if last_usage is not None:
                        _emit_from(last_usage, model_id, opts)

            return _wrap_async_stream(response)

        return _create_async

    # ------------------------------------------------------------------
    # chat.completions.create  (is_responses_api=False)
    # ------------------------------------------------------------------
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None) if chat is not None else None
    if completions is not None:
        original_chat_create = getattr(completions, "create", None)
        raw_chat_create = getattr(getattr(completions, "with_raw_response", None), "create", None)
        if original_chat_create is not None:
            completions.create = (
                _make_async_create(original_chat_create, raw_chat_create, is_responses_api=False)
                if is_async
                else _make_sync_create(original_chat_create, raw_chat_create, is_responses_api=False)
            )

    # ------------------------------------------------------------------
    # responses.create  (is_responses_api=True — skips stream_options injection)
    # ------------------------------------------------------------------
    responses_namespace = getattr(client, "responses", None)
    if responses_namespace is not None:
        original_responses_create = getattr(responses_namespace, "create", None)
        raw_responses_create = getattr(getattr(responses_namespace, "with_raw_response", None), "create", None)
        if original_responses_create is not None:
            responses_namespace.create = (
                _make_async_create(original_responses_create, raw_responses_create, is_responses_api=True)
                if is_async
                else _make_sync_create(original_responses_create, raw_responses_create, is_responses_api=True)
            )

    setattr(client, _INSTRUMENTED_ATTR, True)
    return client

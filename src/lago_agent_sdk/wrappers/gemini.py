"""google-genai SDK wrapper.

Wraps the public methods of `genai.Client.models` (sync) and `genai.Client.aio.models`
(async) in place — instrumentation never breaks the customer's call.

Methods wrapped:
  - models.generate_content(...)         — sync, returns GenerateContentResponse
  - models.generate_content_stream(...)  — sync, returns iterator of chunks (last has usage)
  - aio.models.generate_content(...)     — async, awaited
  - aio.models.generate_content_stream(...) — async, yields chunks

Per-call override: pop `extra_lago={"subscription": ..., "dimensions": ...}` from
kwargs before forwarding so the SDK's strict validation doesn't reject it.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ..adapters import extract_gemini_native

logger = logging.getLogger("lago_agent_sdk.wrappers.gemini")

_INSTRUMENTED_ATTR = "_lago_instrumented"
_LAGO_KWARG = "extra_lago"


def _pop_lago_kwarg(kwargs: dict[str, Any]) -> dict[str, Any]:
    return kwargs.pop(_LAGO_KWARG, {}) or {}


def wrap_gemini_client(
    sdk: Any,
    client: Any,
    dimensions: dict[str, Any] | None = None,
    subscription: str | None = None,
) -> Any:
    """In-place wrap of a `google.genai.Client`. Idempotent."""
    if getattr(client, _INSTRUMENTED_ATTR, False):
        logger.info("lago: gemini client already wrapped — skipping")
        return client

    base_dims = dict(dimensions or {})
    base_sub = subscription

    def _resolve_opts(lago_opts: dict[str, Any]) -> dict[str, Any]:
        return {
            "subscription": lago_opts.get("subscription") or base_sub,
            "dimensions": {**base_dims, **(lago_opts.get("dimensions") or {})},
            "mode": lago_opts.get("mode"),
            "markup": lago_opts.get("markup"),
        }

    def _emit_from(payload: Any, model_id: str, opts: dict[str, Any]) -> None:
        try:
            usage = extract_gemini_native(payload, model_id=model_id)
            sdk.emit(usage, **opts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("lago: gemini emit failed: %s", exc)

    def _make_sync_generate(original: Any) -> Any:
        def _generate(*args: Any, **kwargs: Any) -> Any:
            lago_opts = _pop_lago_kwarg(kwargs)
            model_id = kwargs.get("model") or (args[0] if args else "")
            opts = _resolve_opts(lago_opts)
            response = original(*args, **kwargs)
            _emit_from(response, str(model_id), opts)
            return response

        return _generate

    def _make_async_generate(original: Any) -> Any:
        async def _generate_async(*args: Any, **kwargs: Any) -> Any:
            lago_opts = _pop_lago_kwarg(kwargs)
            model_id = kwargs.get("model") or (args[0] if args else "")
            opts = _resolve_opts(lago_opts)
            response = await original(*args, **kwargs)
            _emit_from(response, str(model_id), opts)
            return response

        return _generate_async

    def _make_sync_stream(original: Any) -> Any:
        def _stream(*args: Any, **kwargs: Any) -> Iterator[Any]:
            lago_opts = _pop_lago_kwarg(kwargs)
            model_id = kwargs.get("model") or (args[0] if args else "")
            opts = _resolve_opts(lago_opts)
            src = original(*args, **kwargs)

            def _iter() -> Iterator[Any]:
                last_with_usage: Any = None
                try:
                    for chunk in src:
                        payload = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                        if isinstance(payload, dict) and payload.get("usage_metadata"):
                            last_with_usage = {"usage_metadata": payload["usage_metadata"]}
                        yield chunk
                finally:
                    if last_with_usage is not None:
                        _emit_from(last_with_usage, str(model_id), opts)

            return _iter()

        return _stream

    def _make_async_stream(original: Any) -> Any:
        async def _stream_async(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            lago_opts = _pop_lago_kwarg(kwargs)
            model_id = kwargs.get("model") or (args[0] if args else "")
            opts = _resolve_opts(lago_opts)
            src = await original(*args, **kwargs)

            async def _aiter() -> AsyncIterator[Any]:
                last_with_usage: Any = None
                try:
                    async for chunk in src:
                        payload = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                        if isinstance(payload, dict) and payload.get("usage_metadata"):
                            last_with_usage = {"usage_metadata": payload["usage_metadata"]}
                        yield chunk
                finally:
                    if last_with_usage is not None:
                        _emit_from(last_with_usage, str(model_id), opts)

            return _aiter()

        return _stream_async

    # ------------------------------------------------------------------
    # client.models.* (sync)
    # ------------------------------------------------------------------
    models = getattr(client, "models", None)
    if models is not None:
        original_generate = getattr(models, "generate_content", None)
        if original_generate is not None:
            models.generate_content = _make_sync_generate(original_generate)

        original_stream = getattr(models, "generate_content_stream", None)
        if original_stream is not None:
            models.generate_content_stream = _make_sync_stream(original_stream)

    # ------------------------------------------------------------------
    # client.aio.models.* (async)
    # ------------------------------------------------------------------
    aio = getattr(client, "aio", None)
    if aio is not None:
        aio_models = getattr(aio, "models", None)
        if aio_models is not None:
            original_aio_generate = getattr(aio_models, "generate_content", None)
            if original_aio_generate is not None:
                aio_models.generate_content = _make_async_generate(original_aio_generate)

            original_aio_stream = getattr(aio_models, "generate_content_stream", None)
            if original_aio_stream is not None:
                aio_models.generate_content_stream = _make_async_stream(original_aio_stream)

    setattr(client, _INSTRUMENTED_ATTR, True)
    return client

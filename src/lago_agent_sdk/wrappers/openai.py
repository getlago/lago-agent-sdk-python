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
import urllib.parse
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ..adapters import extract_openai_native
from ..adapters.openai_native import RAMP_ROUTER_PROVIDER

# The one import a wrapper takes from gateway code, and it is load-bearing: the REST-view
# dedup only works if this wrapper and `gateway/snowflake.py` compute the IDENTICAL
# transaction id, so both call one helper instead of keeping two copies of a string
# format that would drift without an error. The module is pure (canonical-only imports,
# no I/O), so this pulls nothing heavy into the wrap() path.
from ..gateway.adapters.snowflake_cortex import SNOWFLAKE_EVENT_ID_PREFIX, snowflake_event_id

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


def _snowflake_request_id(header_owner: Any) -> str:
    """The Snowflake-side id of a Cortex call, read off the raw response.

    `x-snowflake-request-id` IS the `REQUEST_ID` the call lands under in
    `CORTEX_REST_API_USAGE_HISTORY` (measured byte-identical, 2026-08-26), which is what
    lets the live path and a REST-view backfill produce one transaction id — see
    `snowflake_event_id`. The response BODY is not a route in: Cortex returns `"id": ""`.

    `header_owner` is whatever carries `.headers` on the path at hand: the raw response
    on the `.with_raw_response` path, the stream's own `.response` (httpx) when
    streaming — headers arrive before the body, so reading them consumes nothing. Same
    defensiveness as `_is_cache_hit`: an owner without headers, or a response without
    the header, yields "" and the event keeps its UUID — never breaks the customer's
    call.
    """
    try:
        headers = getattr(header_owner, "headers", None)
        if headers is None:
            return ""
        return str(headers.get("x-snowflake-request-id") or "")
    except Exception:  # noqa: BLE001 — a broken fake's property must not break billing
        return ""


# Provider overrides implied by the client's `base_url`, in match order — the first
# substring found wins. A TABLE rather than a chain of `if`s: this is the second
# surface to need the mechanism and a third (Ramp) is already coming, and each extra
# `if` is one more place to get the try/except and the ordering wrong. Order matters
# only where one entry's substring is a prefix of another's; keep the more specific
# entry first.
#
# Every entry matches a PATH, never a host, and that is the whole design: both of
# these vendors serve plenty of non-inference APIs from the very same host.
_PROVIDER_BY_BASE_URL_PATH: tuple[tuple[str, str], ...] = (
    # A Databricks-HOSTED foundation model answers on the unified mlflow surface. It
    # has to be told apart from an OpenAI-BYOK call, which uses the SAME
    # `openai.OpenAI` class against `/ai-gateway/openai/v1` — and the response gives
    # no clue: a hosted call echoes a served-entity name ("meta-llama-4-maverick-040225")
    # with no distinguishing marker, so `_infer_provider`'s model-string rule cannot
    # see it. Matching `/ai-gateway/mlflow/`, NOT `/ai-gateway/`, is the point: the
    # openai and anthropic BYOK surfaces live under that same prefix and must keep
    # their real vendor provider so they price against OpenRouter.
    ("/ai-gateway/mlflow/", "databricks"),
    # Snowflake Cortex is OpenAI-WIRE-compatible, so customers reach it with the
    # `openai.OpenAI` client pointed at
    # `https://<account>.snowflakecomputing.com/api/v2/cortex/…`. The response is an
    # ordinary OpenAI chat completion naming `claude-sonnet-4-5` or `openai-gpt-5`, so
    # without this row `_infer_provider` reads the model string and answers "openai" —
    # and every event for the call goes out labelled as OpenAI usage. Measured against
    # the live OpenRouter catalog on 2026-08-25, none of the ids this surface actually
    # serves match a price key, so today the mislabelling also costs a permanent
    # price-miss report on every single request. Both halves are fixed by stamping the
    # provider that really served the call: "snowflake" is absent from `_VENDOR_MAP`
    # by design, so it cannot match a price at all, and `TOKEN_BILLED_PROVIDERS`
    # carries it to token events with no error. That also forecloses the accident the
    # docstring below describes — Snowflake renaming a model to a bare `gpt-4.1` would
    # otherwise let it match OpenAI's own rate while Snowflake bills in credits.
    #
    # `/api/v2/cortex/` and not the `snowflakecomputing.com` host: the host also serves
    # `/api/v2/statements` (the SQL API this SDK's own gateway reader drives) and every
    # other Snowflake API, none of which is model inference.
    ("/api/v2/cortex/", "snowflake"),
)


# Ramp Router cannot be a row in the path table above: it serves every provider it
# fronts through one dedicated host with no distinguishing path, so the HOST is the
# signal — and it must be the PARSED host, never a substring test. A substring row
# ("api.router.com") also matches `https://evil.example.com/api.router.com/v1`, which
# would stamp an unrelated endpoint's traffic as Router-served. The `.router.com`
# suffix arm covers a regional or staging host without widening to arbitrary domains —
# `evilrouter.com` does not end in `.router.com`. The path table keeps first say: its
# rows are more specific, and no Snowflake or Databricks URL lives under router.com.
_RAMP_ROUTER_HOST = "api.router.com"
_RAMP_ROUTER_DOMAIN = ".router.com"


def _provider_hint_for(client: Any) -> str:
    """Return a provider override implied by the client's base_url, or "".

    Every provider named in the table above is deliberately ABSENT from pricing's
    `_VENDOR_MAP`, so a hinted call CANNOT hit a price table. `emit()` then emits
    token counts via TOKEN_BILLED_PROVIDERS with no error reported — that is the
    complete answer for these models, not a fallback.

    Deliberate, and the reason a hint exists at all: Databricks bills hosted models
    in DBUs against a rate card published only as HTML and present in no system
    table, and Snowflake bills Cortex in credits at an edition/region/contract rate
    that is machine-readable nowhere — while OpenRouter DOES list bare
    `openai/gpt-oss-20b` and `meta-llama/llama-4-maverick` at 0.2-0.4x of Databricks'
    real rate. Left as "openai", a rename of the served entity to an 8-digit date
    suffix would let `_strip_version` strip it into a match and silently under-bill
    2.5-5x. Stamping the real provider turns that accident into a guaranteed honest
    miss.
    """
    try:
        base_url = str(getattr(client, "base_url", "") or "")
    except Exception:  # noqa: BLE001 — some client variants don't expose it
        return ""
    for path, provider in _PROVIDER_BY_BASE_URL_PATH:
        if path in base_url:
            return provider
    try:
        host = urllib.parse.urlsplit(base_url).hostname or ""
    except ValueError:
        # A relative or malformed base_url is not a gateway. Never throw out of wrap().
        return ""
    if host == _RAMP_ROUTER_HOST or host.endswith(_RAMP_ROUTER_DOMAIN):
        return RAMP_ROUTER_PROVIDER
    return ""


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
    # Resolved once, here, and reached only through `_emit_from` below — every emit
    # path in this wrapper closes over it, so no call site can forget to pass it. That
    # matters because "" is a legitimate value (it is what every non-Databricks client
    # resolves to), so an omission would be indistinguishable at runtime from a real
    # answer: the call would bill as provider="openai" for a Databricks-HOSTED model,
    # dropping it out of TOKEN_BILLED_PROVIDERS. The JS port's stream wrapper is a
    # module-level generator rather than a closure, so it keeps the same guarantee by
    # making its `providerHint` parameter required.
    provider_hint = _provider_hint_for(client)

    def _resolve_opts(lago_opts: dict[str, Any]) -> dict[str, Any]:
        return {
            "subscription": lago_opts.get("subscription") or base_sub,
            "dimensions": {**base_dims, **(lago_opts.get("dimensions") or {})},
            "mode": lago_opts.get("mode"),
            "markup": lago_opts.get("markup"),
        }

    def _with_rest_event_id(opts: dict[str, Any], header_owner: Any) -> dict[str, Any]:
        """Attach the idempotency key that makes a REST-view backfill of this same call
        a duplicate Lago rejects, in the EXACT shape the reader builds.

        Gated on the provider hint rather than on the header existing: real OpenAI never
        sends the header today, but the gate makes that a property of this code instead
        of the provider's behaviour — a proxy injecting the header at an OpenAI base_url
        must not change that call's transaction_id shape. The key must name the
        subscription `emit()` will bill, so it is resolved NOW, at call time, and the
        resolved value is passed through — leaving it to `emit()` would let a context
        change between the call and a stream's exhaustion put one subscription in the
        key and bill another. No header (or no headers at all) → opts unchanged →
        `emit()` keeps its per-event UUID fallback; an extra guard here would be
        redundant with that, and a constant fallback would collide every call in the
        window onto one id.
        """
        if provider_hint != "snowflake":
            return opts
        request_id = _snowflake_request_id(header_owner)
        if not request_id:
            return opts
        sub = sdk._resolve_subscription(opts.get("subscription"))
        return {
            **opts,
            "subscription": sub,
            "event_id": snowflake_event_id(SNOWFLAKE_EVENT_ID_PREFIX, "rest", sub, request_id),
        }

    def _emit_from(payload: Any, model_id: str, opts: dict[str, Any]) -> None:
        try:
            usage = extract_openai_native(payload, model_id=model_id, provider_hint=provider_hint)
            sdk.emit(usage, **opts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("lago: openai emit failed: %s", exc)
            sdk._report_error(exc, "emit")

    def _extract_stream_usage(payload: Any) -> dict[str, Any] | None:
        """Pull usage out of a stream event, handling both API shapes.

        Chat Completions: usage sits at the top of the final chunk
        (`{"usage": {...}}`).
        Responses API:    usage sits under `event.response.usage` on the
        terminal `response.completed` event (`{"type": "response.completed",
        "response": {"usage": {...}}}`).

        Carries the chunk's own `model` through alongside the usage. Rebuilding a
        usage-ONLY payload made `resolve_model` fall back to the requested alias
        on every streaming call, which is precisely the attribution bug the
        non-streaming path was fixed for: a streamed `gpt-5-chat-latest` stayed
        `gpt-5-chat-latest` instead of resolving to the dated snapshot OpenRouter
        lists, so price mode missed and silently degraded to token events. It
        matters most on a gateway, where the resolved name is what decides which
        price table the call is even looked up in.
        """
        if not isinstance(payload, dict):
            return None
        usage = payload.get("usage")
        if isinstance(usage, dict) and usage:
            return {"usage": usage, "model": payload.get("model")}
        # Responses API stream events nest usage under `.response.usage` — and the
        # resolved model under `.response.model`, not at the event's top level.
        response = payload.get("response")
        if isinstance(response, dict):
            nested = response.get("usage")
            if isinstance(nested, dict) and nested:
                return {"usage": nested, "model": response.get("model")}
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
                    _emit_from(response, model_id, _with_rest_event_id(opts, raw))
                return response

            # Streaming, or `.with_raw_response` unavailable (older/custom client)
            # — plain `.create()`. No cache-hit detection possible on this path.
            response = original(*args, **kwargs)

            if _is_response_like(response):
                _emit_from(response, model_id, opts)
                return response

            # Streaming reaches the dedup header too: `openai.Stream` carries the
            # httpx response as `.response`, whose headers arrived before the body —
            # reading them consumes nothing (verified live on a streamed Cortex call).
            # Computed HERE, at call time, not in the generator's finally.
            stream_opts = _with_rest_event_id(opts, getattr(response, "response", None))

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
                        _emit_from(last_usage, model_id, stream_opts)

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
                    _emit_from(response, model_id, _with_rest_event_id(opts, raw))
                return response

            response = await original(*args, **kwargs)

            if _is_response_like(response):
                _emit_from(response, model_id, opts)
                return response

            # Same call-time header read as the sync stream path; see there.
            stream_opts = _with_rest_event_id(opts, getattr(response, "response", None))

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
                        _emit_from(last_usage, model_id, stream_opts)

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
        raw_responses_create = getattr(
            getattr(responses_namespace, "with_raw_response", None), "create", None
        )
        if original_responses_create is not None:
            responses_namespace.create = (
                _make_async_create(original_responses_create, raw_responses_create, is_responses_api=True)
                if is_async
                else _make_sync_create(original_responses_create, raw_responses_create, is_responses_api=True)
            )

    setattr(client, _INSTRUMENTED_ATTR, True)
    return client

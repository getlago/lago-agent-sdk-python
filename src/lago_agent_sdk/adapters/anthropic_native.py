"""Anthropic native adapter — verified against real fixtures.

Field mapping:
  usage.input_tokens                                 → input
  usage.output_tokens                                → output
  usage.cache_read_input_tokens                      → cache_read
  usage.cache_creation_input_tokens                  → cache_write
  usage.cache_creation.ephemeral_5m_input_tokens     → cache_write_5m
  usage.cache_creation.ephemeral_1h_input_tokens     → cache_write_1h
  count of content[].type == "tool_use"              → tool_calls

Not exposed by Anthropic (folded into output_tokens):
  reasoning_tokens — even with extended thinking enabled

Unknown usage fields (service_tier, inference_geo, server_tool_use, …) land in extras.

Ramp Router's `/v1/messages` surface answers in this exact shape for EVERY vendor it
fronts, so the same extractor serves it — with a `provider_hint` from the wrapper, since
nothing in the body says Router was in the path (see RAMP_ROUTER_MESSAGES_API).
"""

from __future__ import annotations

from typing import Any, cast

from ..canonical import CanonicalUsage
from .openai_native import RAMP_ROUTER_PROVIDER

#: `api` stamped on a Router call that arrived through `/v1/messages`.
#:
#: Distinct from the Responses surface's stamp ("ramp_router", which sits in
#: OPENAI_SHAPED_APIS) because the two surfaces report the SAME vendor's numbers under
#: DIFFERENT conventions. Measured 2026-09-04 and 2026-09-07 against a live account:
#: `/v1/messages` keeps Anthropic's additive shape for every vendor — haiku reports
#: `input_tokens: 16` beside `cache_read_input_tokens: 20113`; an xAI model reports
#: `input_tokens: 65` beside `cache_read_input_tokens: 128` and `thinking_tokens: 200`
#: INSIDE `output_tokens: 201` — while `/v1/responses` folds the cached block inside
#: `input_tokens`. Token semantics key on the surface, so this stamp must stay OUT of
#: OPENAI_SHAPED_APIS: the provider-keyed sets do not name "ramp_router", which leaves
#: the all-additive default, the measured answer here. Putting the Responses stamp on
#: this surface would subtract a cached block that was never inside `input`.
#:
#: The write count Router's Responses surface cannot report for Anthropic models IS
#: reported here (`cache_creation_input_tokens`, with the 5m/1h split), and reconciled
#: exactly against the dashboard — the reason this surface is worth detecting at all.
RAMP_ROUTER_MESSAGES_API = "ramp_router_messages"

_KNOWN_USAGE_FIELDS = {
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation",
}


def _safe_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _safe_int(v: Any) -> int:
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def _to_dict(obj: Any) -> dict[str, Any]:
    """Best-effort pydantic-or-dict to dict (Anthropic SDK returns pydantic Message objects)."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return cast(dict[str, Any], obj.model_dump())
        except Exception:  # noqa: BLE001
            pass
    return {}


def _resolve_model(response_model: Any, requested_model: str) -> str:
    """Prefer the model the response reports over the one requested.

    Anthropic can resolve a short alias to a more specific name — e.g.
    "claude-sonnet-4-5" → "claude-sonnet-4-5-20250929" — with no gateway or
    fallback involved at all. Pricing and attribution must key off what actually
    answered. Falls back to the requested model only when the response is silent
    about its own model (e.g. a synthetic streaming usage blob).
    """
    if isinstance(response_model, str) and response_model:
        return response_model
    return requested_model or ""


def extract_anthropic_native(response: Any, model_id: str = "", provider_hint: str = "") -> CanonicalUsage:
    """Translate an Anthropic native response (Message or dict) → CanonicalUsage.

    Accepts the SDK's pydantic Message object, a dict (e.g. captured fixture),
    or a synthetic `{"usage": {...}}` blob produced by the streaming wrapper.

    `provider_hint` is the wrapper's word that the client was pointed at a gateway; only
    the wrapper can know, because the body never says. Today the one value it takes is
    RAMP_ROUTER_PROVIDER, which stamps the call as Router traffic on the Messages
    surface. The served tier needs no special handling: Router puts `service_tier`
    INSIDE `usage` on this surface (buffered, and on both `message_start` and
    `message_delta` when streamed — measured), so the drift sweep below already lands it
    in `extras["service_tier"]`, where the price-mode tier gate reads it.
    """
    resp = _to_dict(response) if not isinstance(response, dict) else response
    provider, api = "anthropic", "native"
    if provider_hint == RAMP_ROUTER_PROVIDER:
        provider, api = RAMP_ROUTER_PROVIDER, RAMP_ROUTER_MESSAGES_API

    usage = _safe_dict(resp.get("usage"))
    cache_creation = _safe_dict(usage.get("cache_creation"))

    content = resp.get("content")
    tool_calls = (
        sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_use")
        if isinstance(content, list)
        else 0
    )

    extras: dict[str, Any] = {}
    for k, v in usage.items():
        if k not in _KNOWN_USAGE_FIELDS:
            extras[k] = v

    return CanonicalUsage(
        input=_safe_int(usage.get("input_tokens")),
        output=_safe_int(usage.get("output_tokens")),
        cache_read=_safe_int(usage.get("cache_read_input_tokens")),
        cache_write=_safe_int(usage.get("cache_creation_input_tokens")),
        cache_write_5m=_safe_int(cache_creation.get("ephemeral_5m_input_tokens")),
        cache_write_1h=_safe_int(cache_creation.get("ephemeral_1h_input_tokens")),
        tool_calls=tool_calls,
        model=_resolve_model(resp.get("model"), model_id),
        provider=provider,
        api=api,
        extras=extras,
    )

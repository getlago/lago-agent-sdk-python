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
"""

from __future__ import annotations

from typing import Any, cast

from ..canonical import CanonicalUsage

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


def extract_anthropic_native(response: Any, model_id: str = "") -> CanonicalUsage:
    """Translate an Anthropic native response (Message or dict) → CanonicalUsage.

    Accepts the SDK's pydantic Message object, a dict (e.g. captured fixture),
    or a synthetic `{"usage": {...}}` blob produced by the streaming wrapper.
    """
    resp = _to_dict(response) if not isinstance(response, dict) else response

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
        model=model_id or (resp.get("model") if isinstance(resp.get("model"), str) else "") or "",
        provider="anthropic",
        api="native",
        extras=extras,
    )

"""Mistral native adapter — verified against real fixtures (see docs/mistral-native-findings.md).

Verified mappings:
  - cache_read field is `usage.prompt_tokens_details.cached_tokens`
    (NOT `usage.prompt_cache_hit_tokens` — that field does not exist)
  - Reasoning, cache_write, image_input, audio_input not exposed by Mistral.
"""
from __future__ import annotations

from typing import Any

from ..canonical import CanonicalUsage

_KNOWN_USAGE_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_tokens_details",
}


def _safe_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _safe_int(v: Any) -> int:
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def extract_mistral_native(response: Any, model_id: str = "") -> CanonicalUsage:
    """Translate a Mistral chat completion response → CanonicalUsage.

    Accepts either a dict (from `.model_dump()`) or any object with attribute access.
    Streaming chunks should be reduced to the final chunk before calling this.
    """
    if not isinstance(response, dict):
        # support pydantic models from mistralai SDK
        response = getattr(response, "model_dump", lambda: {})() or {}

    usage = _safe_dict(response.get("usage"))
    prompt_details = _safe_dict(usage.get("prompt_tokens_details"))
    choices = response.get("choices") or []
    message = _safe_dict(choices[0].get("message")) if choices else {}
    tool_calls = message.get("tool_calls") or []

    extras: dict[str, Any] = {}
    for k, v in usage.items():
        if k not in _KNOWN_USAGE_FIELDS:
            extras[k] = v

    return CanonicalUsage(
        input=_safe_int(usage.get("prompt_tokens")),
        output=_safe_int(usage.get("completion_tokens")),
        cache_read=_safe_int(prompt_details.get("cached_tokens")),
        tool_calls=len(tool_calls) if isinstance(tool_calls, list) else 0,
        model=model_id or response.get("model", "") or "",
        provider="mistral",
        api="native",
        extras=extras,
    )

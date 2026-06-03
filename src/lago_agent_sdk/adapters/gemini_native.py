"""Gemini native adapter — verified against real fixtures.

Wraps the modern `google-genai` SDK (`from google import genai`). Both
`client.models.generate_content` (sync + async) and
`client.models.generate_content_stream` (sync + async) put usage in
`response.usage_metadata` (the final chunk for streaming).

Field mapping (`usage_metadata.*`):
  prompt_token_count                                      → input
  candidates_token_count                                  → output
  cached_content_token_count                              → cache_read
  thoughts_token_count                                    → reasoning
                                                            (Gemini 2.5; ADDITIVE
                                                            to candidates, not a subset)
  prompt_tokens_details[modality=AUDIO].token_count       → audio_input
  prompt_tokens_details[modality=IMAGE].token_count       → image_input
  candidates_tokens_details[modality=AUDIO].token_count   → audio_output

Tool calls: count of candidates[0].content.parts[] entries that have a
non-null `function_call` field.

Semantic note vs OpenAI:
  Gemini's `thoughts_token_count` is ADDITIVE to `candidates_token_count`
  (total billable output for Google = candidates + thoughts).
  OpenAI's `reasoning_tokens` is a SUBSET of `completion_tokens`.
  When a customer bills on both `llm_output_tokens` and `llm_reasoning_tokens`
  as separate Lago metrics, the Gemini-side sum reflects the full Google bill;
  the OpenAI-side `llm_output_tokens` already includes reasoning.

Unknown top-level usage fields land in `extras` (drift detection).
"""

from __future__ import annotations

from typing import Any, cast

from ..canonical import CanonicalUsage

_KNOWN_USAGE_FIELDS = {
    "prompt_token_count",
    "candidates_token_count",
    "cached_content_token_count",
    "thoughts_token_count",
    "tool_use_prompt_token_count",
    "total_token_count",
    "prompt_tokens_details",
    "candidates_tokens_details",
    "cache_tokens_details",
    "tool_use_prompt_tokens_details",
    "traffic_type",
}


def _safe_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _safe_int(v: Any) -> int:
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def _to_dict(obj: Any) -> dict[str, Any]:
    """Best-effort pydantic-or-dict → dict (google-genai returns pydantic objects)."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return cast(dict[str, Any], obj.model_dump())
        except Exception:  # noqa: BLE001
            pass
    return {}


def _modality_token_count(details: list[dict[str, Any]] | Any, modality: str) -> int:
    """Sum token_count from a list of {modality, token_count} entries matching the given modality."""
    if not isinstance(details, list):
        return 0
    total = 0
    for entry in details:
        if isinstance(entry, dict) and entry.get("modality") == modality:
            total += _safe_int(entry.get("token_count"))
    return total


def _count_tool_calls(resp: dict[str, Any]) -> int:
    """Count parts in candidates[0].content.parts[] that have a function_call."""
    candidates = resp.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return 0
    first = candidates[0]
    if not isinstance(first, dict):
        return 0
    content = _safe_dict(first.get("content"))
    parts = content.get("parts")
    if not isinstance(parts, list):
        return 0
    return sum(1 for p in parts if isinstance(p, dict) and p.get("function_call") is not None)


def extract_gemini_native(response: Any, model_id: str = "") -> CanonicalUsage:
    """Translate a google-genai response (GenerateContentResponse or dict) → CanonicalUsage.

    Accepts the SDK's pydantic objects, dicts (e.g. captured fixtures), or a
    synthetic `{"usage_metadata": {...}}` blob produced by the streaming wrapper.
    """
    resp = _to_dict(response) if not isinstance(response, dict) else response
    usage = _safe_dict(resp.get("usage_metadata"))

    prompt_details = usage.get("prompt_tokens_details")
    candidates_details = usage.get("candidates_tokens_details")

    extras: dict[str, Any] = {}
    for k, v in usage.items():
        if k not in _KNOWN_USAGE_FIELDS:
            extras[k] = v

    return CanonicalUsage(
        input=_safe_int(usage.get("prompt_token_count")),
        output=_safe_int(usage.get("candidates_token_count")),
        cache_read=_safe_int(usage.get("cached_content_token_count")),
        reasoning=_safe_int(usage.get("thoughts_token_count")),
        audio_input=_modality_token_count(prompt_details, "AUDIO"),
        audio_output=_modality_token_count(candidates_details, "AUDIO"),
        image_input=_modality_token_count(prompt_details, "IMAGE"),
        tool_calls=_count_tool_calls(resp),
        model=model_id
        or (resp.get("model_version") if isinstance(resp.get("model_version"), str) else "")
        or "",
        provider="gemini",
        api="native",
        extras=extras,
    )

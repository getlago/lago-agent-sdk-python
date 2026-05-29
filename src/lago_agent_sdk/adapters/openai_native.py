"""OpenAI native adapter — verified against real fixtures.

Handles both Chat Completions API (`client.chat.completions.create`) and the
Responses API (`client.responses.create`). They share a similar concept but
use different field names — we detect which by looking at the usage shape.

CHAT COMPLETIONS field mapping (`usage.*`):
  prompt_tokens                                    → input
  completion_tokens                                → output
  prompt_tokens_details.cached_tokens              → cache_read
  prompt_tokens_details.audio_tokens               → audio_input
  completion_tokens_details.reasoning_tokens       → reasoning   (o-series models)
  completion_tokens_details.audio_tokens           → audio_output (GPT-4o-audio output)
  count of choices[0].message.tool_calls           → tool_calls

RESPONSES API field mapping (`usage.*`):
  input_tokens                                     → input
  output_tokens                                    → output
  input_tokens_details.cached_tokens               → cache_read
  output_tokens_details.reasoning_tokens           → reasoning
  count of output[].type == "function_call"        → tool_calls

Not exposed by either API:
  cache_write, cache_write_5m, cache_write_1h — OpenAI auto-caches without
  surfacing creation counts.

Known gaps (intentional, documented):
  - completion_tokens_details.accepted_prediction_tokens — Predicted Outputs
    feature: subset of completion_tokens (the ones that matched the prediction).
    Skipped to avoid double-counting against completion_tokens.
  - completion_tokens_details.rejected_prediction_tokens — Predicted Outputs:
    extra cost beyond completion_tokens (prediction tokens the model rejected).
    Skipped for v1 — customers using Predicted Outputs can read this from
    `extras["completion_tokens_details"]` (if drift-detection captures it) or
    via the openai response object directly.
"""

from __future__ import annotations

from typing import Any, cast

from ..canonical import CanonicalUsage

# Top-level usage fields we recognize across BOTH chat completions and responses APIs.
_KNOWN_USAGE_FIELDS = {
    # chat completions
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_tokens_details",
    "completion_tokens_details",
    # responses API
    "input_tokens",
    "output_tokens",
    "input_tokens_details",
    "output_tokens_details",
}


def _safe_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _safe_int(v: Any) -> int:
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def _to_dict(obj: Any) -> dict[str, Any]:
    """Best-effort pydantic-or-dict to dict (OpenAI SDK returns pydantic objects)."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return cast(dict[str, Any], obj.model_dump())
        except Exception:  # noqa: BLE001
            pass
    return {}


def _count_chat_tool_calls(resp: dict[str, Any]) -> int:
    """choices[0].message.tool_calls is a list of called functions in Chat Completions."""
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return 0
    first = choices[0]
    if not isinstance(first, dict):
        return 0
    message = _safe_dict(first.get("message"))
    tcs = message.get("tool_calls")
    return len(tcs) if isinstance(tcs, list) else 0


def _count_responses_tool_calls(resp: dict[str, Any]) -> int:
    """In the Responses API, tool invocations are items in `output` with type == "function_call"."""
    output = resp.get("output")
    if not isinstance(output, list):
        return 0
    return sum(1 for item in output if isinstance(item, dict) and item.get("type") == "function_call")


def extract_openai_native(response: Any, model_id: str = "") -> CanonicalUsage:
    """Translate an OpenAI response (chat completion or responses API) → CanonicalUsage.

    Accepts the SDK's pydantic objects, dicts (e.g. captured fixtures), or the
    synthetic `{"usage": {...}}` blob produced by the streaming wrapper.
    """
    resp = _to_dict(response) if not isinstance(response, dict) else response
    usage = _safe_dict(resp.get("usage"))

    # Detect which API shape we have. Chat Completions uses prompt_tokens;
    # Responses API uses input_tokens. They never both appear.
    is_responses_api = "input_tokens" in usage and "prompt_tokens" not in usage

    if is_responses_api:
        input_tokens = _safe_int(usage.get("input_tokens"))
        output_tokens = _safe_int(usage.get("output_tokens"))
        input_details = _safe_dict(usage.get("input_tokens_details"))
        output_details = _safe_dict(usage.get("output_tokens_details"))
        cache_read = _safe_int(input_details.get("cached_tokens"))
        reasoning = _safe_int(output_details.get("reasoning_tokens"))
        audio_input = _safe_int(input_details.get("audio_tokens"))
        audio_output = 0  # not exposed by Responses API today
        tool_calls = _count_responses_tool_calls(resp)
        api = "responses"
    else:
        input_tokens = _safe_int(usage.get("prompt_tokens"))
        output_tokens = _safe_int(usage.get("completion_tokens"))
        prompt_details = _safe_dict(usage.get("prompt_tokens_details"))
        completion_details = _safe_dict(usage.get("completion_tokens_details"))
        cache_read = _safe_int(prompt_details.get("cached_tokens"))
        reasoning = _safe_int(completion_details.get("reasoning_tokens"))
        audio_input = _safe_int(prompt_details.get("audio_tokens"))
        audio_output = _safe_int(completion_details.get("audio_tokens"))
        tool_calls = _count_chat_tool_calls(resp)
        api = "chat_completions"

    extras: dict[str, Any] = {}
    for k, v in usage.items():
        if k not in _KNOWN_USAGE_FIELDS:
            extras[k] = v

    return CanonicalUsage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        reasoning=reasoning,
        audio_input=audio_input,
        audio_output=audio_output,
        tool_calls=tool_calls,
        model=model_id or (resp.get("model") if isinstance(resp.get("model"), str) else "") or "",
        provider="openai",
        api=api,
        extras=extras,
    )

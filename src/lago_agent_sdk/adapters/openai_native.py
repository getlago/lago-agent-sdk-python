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
from ._common import resolve_model

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

# Nested keys inside the *_tokens_details sub-objects that we actually MAP onto a
# CanonicalUsage field. Anything nested that isn't listed here is drift and gets
# surfaced in `extras` under a dotted key.
#
# Sweeping only top-level keys was a real hole: `prompt_tokens_details` is itself
# a KNOWN top-level key, so nothing inside it was ever inspected. A live
# gpt-5.6-sol response carries `prompt_tokens_details.cache_write_tokens: 3022`
# and those 3022 tokens vanished with no error — a silent violation of the drift
# contract test_drift.py exists to pin, which passed only because it never looked
# one level down.
#
# NOTE the billing subtlety: cache_write_tokens must NOT be mapped to
# CanonicalUsage.cache_write. For OpenAI it sits INSIDE prompt_tokens (measured:
# prompt_tokens=3025 with cache_write_tokens=3022) and bills at the plain input
# rate — Databricks charged exactly what billing all 3025 as input produces. But
# OpenRouter does publish a separate cache_write rate for the model, so mapping it
# would charge those tokens twice: $0.0341 against a true $0.0152, a 2.24x
# over-bill. Anthropic is the opposite — its cache_creation_input_tokens sits
# OUTSIDE input_tokens, which is why mapping is correct there and wrong here.
# Surfacing in extras keeps the field visible without touching the money.
_MAPPED_DETAIL_FIELDS = {
    "prompt_tokens_details": {"cached_tokens", "audio_tokens"},
    "input_tokens_details": {"cached_tokens", "audio_tokens"},
    "completion_tokens_details": {"reasoning_tokens", "audio_tokens"},
    # NOTE `output_tokens_details` deliberately omits `audio_tokens`: the Responses
    # branch hardcodes `audio_output = 0` because the API does not expose it today, so
    # listing it here would exclude a real, unmapped count from `extras` — 500 audio
    # tokens vanishing with no error, which is the exact hole this table closes. Add it
    # back only together with a Responses branch that reads it.
    "output_tokens_details": {"reasoning_tokens"},
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


def _infer_provider(resolved_model: str) -> str:
    """The SDK shape only ever tells you "this looks like an OpenAI response" —
    it can't tell you who actually served it. Going through a gateway's
    OpenAI-compatible endpoint (e.g. Cloudflare's `.../compat`), the resolved
    model string is the only real signal: "@cf/..." is Cloudflare Workers AI's
    own naming convention, never a real OpenAI model. This isn't cosmetic —
    `provider` is what price-mode keys pricing off of, and Workers AI has a
    genuinely different price table (Cloudflare's own catalog) than real
    OpenAI models (OpenRouter); stamping "openai" on a Workers AI call would
    have made it permanently unpriceable, quietly, at the extraction layer."""
    if resolved_model.startswith("@cf/"):
        return "workers-ai"
    return "openai"


def extract_openai_native(response: Any, model_id: str = "", provider_hint: str = "") -> CanonicalUsage:
    """Translate an OpenAI response (chat completion or responses API) → CanonicalUsage.

    Accepts the SDK's pydantic objects, dicts (e.g. captured fixtures), or the
    synthetic `{"usage": {...}}` blob produced by the streaming wrapper.

    `provider_hint` overrides the model-string inference below. Only the wrapper
    can supply it, because the only reliable signal for some gateways is the
    client's `base_url` — which the response never carries. Databricks is the
    case that forced it: a Databricks-HOSTED model answers on
    `/ai-gateway/mlflow/v1` but echoes a served-entity name
    ("meta-llama-4-maverick-040225") with no marker of its own, so no rule based
    on the model string can identify it. See `wrappers/openai.py`.
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

    # Drift sweep one level down, into the *_tokens_details sub-objects. Without
    # this, an unrecognized nested field is silently dropped (see
    # _MAPPED_DETAIL_FIELDS) because its container is a known top-level key.
    for container, mapped in _MAPPED_DETAIL_FIELDS.items():
        for k, v in _safe_dict(usage.get(container)).items():
            if k not in mapped:
                extras[f"{container}.{k}"] = v

    # Consistency guard: for genuine OpenAI, total_tokens always equals
    # prompt + completion (reasoning is a SUBSET of completion, never additive).
    # Verified across every captured real OpenAI-shaped response — zero deltas.
    # So a POSITIVE delta means tokens exist that neither named bucket accounts
    # for, which only happens behind an OpenAI-COMPATIBLE proxy that under-reports.
    #
    # Measured on Gemini through Google's own OpenAI-compat layer:
    # prompt_tokens=57, completion_tokens=47, total_tokens=1253 — 1149 real
    # thinking tokens reported nowhere, and no completion_tokens_details to
    # recover them from. Billing prompt+completion drops 92% of the call, at the
    # output rate. Folding the remainder into `output` is the honest read: the
    # provider's own total proves those tokens were generated.
    #
    # Deliberately NOT assigned to `reasoning`: compute_cost zeroes reasoning for
    # providers in _OUTPUT_INCLUDES_REASONING, so for real OpenAI that would set the
    # field and immediately discard it, recovering nothing.
    #
    # `reasoning` is subtracted from the accounted total, and that subtraction is
    # load-bearing rather than cosmetic. This adapter no longer only ever emits
    # provider="openai" — it also emits "workers-ai" (Cloudflare `/compat`) and
    # "databricks" (via provider_hint), and for those compute_cost bills reasoning
    # ADDITIVELY. A payload reporting both `reasoning_tokens` and an inflated
    # `total_tokens` would then be charged for them twice: once inside the grown
    # `output` and again as a separate reasoning line. Subtracting first means a
    # provider that already broke reasoning out gets no second bill, while the case
    # this guard exists for — a thinking model behind a proxy that reports NO
    # breakdown at all (measured: prompt 57, completion 47, total 1253) — still
    # recovers its 1,149 tokens, because reasoning is 0 there.
    #
    # A no-op for real OpenAI either way: total always equals prompt + completion.
    declared_total = _safe_int(usage.get("total_tokens"))
    if declared_total:
        unaccounted = declared_total - (input_tokens + output_tokens + reasoning)
        if unaccounted > 0:
            output_tokens += unaccounted
            extras["unaccounted_output_tokens"] = unaccounted

    resolved_model = resolve_model(resp.get("model"), model_id)
    return CanonicalUsage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        reasoning=reasoning,
        audio_input=audio_input,
        audio_output=audio_output,
        tool_calls=tool_calls,
        model=resolved_model,
        provider=provider_hint or _infer_provider(resolved_model),
        api=api,
        extras=extras,
    )

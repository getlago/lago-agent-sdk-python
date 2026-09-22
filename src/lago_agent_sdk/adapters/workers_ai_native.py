"""Workers AI `/ai/run` adapter — maps a run response to CanonicalUsage.

Verified against real captures (fixtures/workers_ai/, 2026-09-21) of the model-in-body
route `POST /accounts/{id}/ai/run {"model": ..., "input": {...}}`. That route matters
because it is the only one that reaches every Workers AI model: partner models such as
`typesafe/jev` have no `@cf/` prefix and the path-style `/ai/run/{model}` answers "No
route for that URI" for them, as does the gateway's `/compat` endpoint, which requires
a `messages` array the model rejects.

Two usage vocabularies come back from the one endpoint:

  chat models    result.usage.prompt_tokens / completion_tokens / total_tokens
                 result.usage.prompt_tokens_details.cached_tokens
                 result.usage.neurons                               (01-04, 06, 07)
  typesafe/jev   result.result.usage.input_tokens / output_tokens   (05: one level
                 deeper — a partner model's answer is wrapped as
                 `result: {state, result: {model, answers, usage}, gatewayMetadata}`;
                 08 is the same object straight from TypeSafe's API, unwrapped)

The REQUESTED model id is the one carried, not the served name — the opposite of the
native adapters' rule, for a measured reason. Cloudflare's price catalog is keyed by the
id you request, and the served name drifts from it in ways the catalog's version-strip
fallback does not cover: `@cf/mistralai/mistral-small-3.1-24b-instruct` answers as
`...-24b-v2` (04) and that name MISSES the catalog while the requested id prices;
`typesafe/jev` answers as `jev-1.13.0` (05, 08). The served name is kept in
`extras["served_model"]` when it differs, so nothing is lost — only the billing key stays
the one Cloudflare itself bills by.

`neurons` is Cloudflare's own billing unit, not a token count: it lands in `extras`
and is never a metric. `gatewayMetadata` (`keySource: "BYOK"` on 05) says whose key paid
for the call and lands in `extras` too — under BYOK Cloudflare charged nothing and the
partner bills the customer directly, which matters to anyone reconciling against the
Cloudflare dashboard. A reasoning model (02, deepseek-r1-distill) bundles its thinking
into `completion_tokens` with no separate field, so `reasoning` stays 0 — the same
shape Magistral has on Mistral. `cached_tokens` is a subset of `prompt_tokens`, which
is why "workers-ai" sits in `INPUT_INCLUDES_CACHE_READ`.

A failure body (402 no credits / 403 not on plan / 400 no such model — seen live, not
kept as fixtures) carries `result: {}` and `errors: [...]`. The adapter yields an all-zero
usage for it rather than raising — it is a pure function and cannot know the HTTP status;
the client decides what a failure means.
"""

from __future__ import annotations

from typing import Any

from ..canonical import CanonicalUsage

# Every `usage` key this adapter maps or deliberately ignores. Anything else is drift
# and is swept into `extras["usage"]` — never silently dropped, never miscounted.
_KNOWN_USAGE_KEYS = frozenset(
    {
        # chat models
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        # typesafe/jev
        "input_tokens",
        "output_tokens",
        # Cloudflare's billing unit — kept in extras, not a metric
        "neurons",
    }
)
_KNOWN_DETAIL_KEYS = frozenset({"cached_tokens"})


def _safe_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _safe_int(v: Any) -> int:
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def extract_workers_ai_native(response: Any, model_id: str = "") -> CanonicalUsage:
    """Translate a Workers AI `/ai/run` response body → CanonicalUsage.

    Accepts the full envelope (`{"result": {...}, "success": true, ...}`) or the bare
    `result` object. `model_id` is the model the caller requested and is the id carried
    (see the module docstring); the served name, when it differs, lands in
    `extras["served_model"]`.
    """
    payload = _safe_dict(response)
    result = _safe_dict(payload.get("result")) or payload
    gateway_meta = _safe_dict(result.get("gatewayMetadata"))
    inner = _safe_dict(result.get("result"))
    if inner and ("usage" in inner or "answers" in inner):
        # Partner-model envelope (05): the model's own object sits one level down.
        result = inner
    # The envelope may also carry `usage` at the top level; check both so a shape change
    # moves nothing to zero.
    usage = _safe_dict(result.get("usage")) or _safe_dict(payload.get("usage"))
    details = _safe_dict(usage.get("prompt_tokens_details"))

    extras: dict[str, Any] = {}
    if gateway_meta:
        extras["gateway_metadata"] = gateway_meta
    served = result.get("model")
    if isinstance(served, str) and served and served != model_id:
        extras["served_model"] = served
    if "neurons" in usage:
        extras["neurons"] = usage["neurons"]
    drift = {k: v for k, v in usage.items() if k not in _KNOWN_USAGE_KEYS}
    detail_drift = {k: v for k, v in details.items() if k not in _KNOWN_DETAIL_KEYS}
    if detail_drift:
        drift["prompt_tokens_details"] = detail_drift
    if drift:
        extras["usage"] = drift

    return CanonicalUsage(
        input=_safe_int(usage.get("prompt_tokens")) or _safe_int(usage.get("input_tokens")),
        output=_safe_int(usage.get("completion_tokens")) or _safe_int(usage.get("output_tokens")),
        cache_read=_safe_int(details.get("cached_tokens")),
        model=model_id or (served if isinstance(served, str) else ""),
        provider="workers-ai",
        api="workers_ai_run",
        extras=extras,
    )

"""Cloudflare AI Gateway log adapter — maps a Logs API entry to CanonicalUsage.

Verified against a real captured log entry (live account, real Anthropic call
routed through a real gateway, real Lago rollup confirmed exact).

Field mapping (`GET .../ai-gateway/gateways/{id}/logs` and the single-entry
`GET .../logs/{log_id}`):
  tokens_in                                   → input
  tokens_out                                  → output
  usage_metadata.input_cached_tokens          → cache_read
  usage_metadata.input_cache_creation_tokens  → cache_write
  usage_metadata.reasoningTokens/reasoning_tokens → reasoning
  model, provider                             → passed straight through

`usage_metadata`'s exact key casing is NOT normalized by Cloudflare — it passes
through whatever convention the underlying provider's own usage object used
(Anthropic/OpenAI: snake_case `input_cached_tokens`; a real captured Gemini
entry: camelCase `reasoningTokens`). Both cases are checked for every field
we map; this is observed behavior across two providers, not a documented
guarantee, so a third provider could use a convention we haven't seen yet.

Unlike the provider-native adapters (`adapters/openai_native.py`,
`adapters/anthropic_native.py`), there is no request-side model kwarg to prefer
or fall back on here — a Cloudflare log entry always reports the model that
actually served the request. This adapter is immune, by construction, to the
alias-vs-resolved-model bug fixed in those two.

Billing *policy* is deliberately not decided here — this module only extracts.
`cached`, `step`, and the log's own `id` land in `extras` because the caller
(the poller) needs them: `cached` to decide whether to skip billing a request
Cloudflare served for free, `id` as the idempotency key against replays.
`resolve_subscription()` is separate from extraction because attribution can be
absent, and dropping vs. warning on that is also a caller policy decision.
"""

from __future__ import annotations

from typing import Any

from ...canonical import CanonicalUsage


def _safe_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _safe_int(v: Any) -> int:
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def _safe_str(v: Any) -> str:
    return v if isinstance(v, str) else ""


# Cloudflare AI Gateway logs its OWN provider vocabulary, which is not the name
# the pricing tables and token-semantics tables key off — and not always its own
# URL slug either (the logs say "workers-ai" where the endpoint path says
# "workersai"). Passed through verbatim, "google-ai-studio" matched no vendor in
# pricing's _VENDOR_MAP, so every Gemini call backfilled through the gateway
# missed on price; worse, it also missed _INPUT_INCLUDES_CACHE_READ, so Gemini's
# cache_read — a SUBSET of its input count, not additive — was billed twice.
#
# Only providers this SDK can actually price need an entry. Anything else passes
# through unchanged: an unrecognized provider is one we have no table for, and a
# clean miss falls back to token events, which is strictly better than inventing
# a mapping. AWS Bedrock is deliberately absent for that reason — Bedrock prices
# are keyed off `api.startswith("bedrock")`, and this connector always sets
# api="cloudflare_gateway", so mapping its provider name would route it to
# OpenRouter under a vendor that cannot match. A miss there is honest.
_PROVIDER_ALIASES = {
    "google-ai-studio": "gemini",
    "google-vertex-ai": "gemini",
    "vertex": "gemini",
    "azure-openai": "openai",
    "azureopenai": "openai",
    "workersai": "workers-ai",
}


def _normalize_provider(v: Any) -> str:
    """Map Cloudflare's provider name onto the SDK's own provider vocabulary."""
    p = _safe_str(v).lower()
    return _PROVIDER_ALIASES.get(p, p)


def extract_cloudflare_log(entry: dict[str, Any]) -> CanonicalUsage:
    """Translate one Cloudflare AI Gateway log entry → CanonicalUsage.

    Accepts a single log entry dict as returned by the Logs API (either the
    list endpoint or the single-entry endpoint — same shape). Missing/malformed
    fields degrade to zero/empty rather than raising, matching the defensive
    style of the other adapters — a poller processing a batch of log entries
    must not have one malformed entry take down the whole run.
    """
    usage_meta = _safe_dict(entry.get("usage_metadata"))

    return CanonicalUsage(
        input=_safe_int(entry.get("tokens_in")),
        output=_safe_int(entry.get("tokens_out")),
        cache_read=_safe_int(usage_meta.get("input_cached_tokens")),
        cache_write=_safe_int(usage_meta.get("input_cache_creation_tokens")),
        reasoning=_safe_int(usage_meta.get("reasoningTokens") or usage_meta.get("reasoning_tokens")),
        model=_safe_str(entry.get("model")),
        provider=_normalize_provider(entry.get("provider")),
        api="cloudflare_gateway",
        extras={
            "cached": entry.get("cached"),
            "step": entry.get("step"),
            "log_id": entry.get("id"),
        },
    )


def resolve_subscription(entry: dict[str, Any]) -> str | None:
    """Pull the Lago subscription id from the customer's `cf-aig-metadata` header.

    Returns None if the customer never set `lago_subscription` — the caller
    decides what to do with an unattributed entry (drop it, log a warning, ...);
    this function only reports whether attribution is present.
    """
    metadata = _safe_dict(entry.get("metadata"))
    value = metadata.get("lago_subscription")
    return value if isinstance(value, str) and value else None

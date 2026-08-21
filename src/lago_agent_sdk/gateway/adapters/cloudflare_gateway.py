"""Cloudflare AI Gateway log adapter — maps a Logs API entry to CanonicalUsage.

Verified against a real captured log entry (live account, real Anthropic call
routed through a real gateway, real Lago rollup confirmed exact).

Field mapping (`GET .../ai-gateway/gateways/{id}/logs` and the single-entry
`GET .../logs/{log_id}`):
  tokens_in                                   → input
  tokens_out                                  → output
  usage_metadata.input_cached_tokens          → cache_read
  usage_metadata.input_cache_creation_tokens  → cache_write
  usage_metadata.reasoningTokens              → reasoning
  model, provider                             → passed straight through

Cloudflare reports its OWN counter vocabulary here, not the provider's. Across all
14 captured fixtures — Anthropic, Workers AI, Mistral and Gemini, via every ingress
method — the keys that appear are `input_tokens`, `output_tokens`, `total_tokens`,
`input_cached_tokens`, `input_cache_creation_tokens`, `neurons`, `input_text_tokens`
and `reasoningTokens`. Not one provider-native key shows up: no Anthropic
`cache_read_input_tokens`, no Gemini `thoughtsTokenCount` or
`cachedContentTokenCount`.

That list is a snapshot and has already been overtaken once: a live Logs API pull
also returned `units`, which appears in none of the fixtures. Treat the enumeration
as illustrative, not exhaustive — `_MAPPED_USAGE_KEYS` plus the drift sweep into
`extras["usage_metadata"]` is what actually keeps an unrecognized counter from being
lost, and it needs no re-audit to stay correct.

That vocabulary is *mostly* snake_case, with `reasoningTokens` as a camelCase
outlier — Cloudflare's own inconsistency, not a provider key leaking through
(Gemini's native spelling for the same quantity is `thoughtsTokenCount`, which
appears nowhere). The extra spellings checked below are therefore unobserved
insurance against a convention we have not seen, not handling for a known case.

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


# Every `usage_metadata` spelling this adapter accounts for: the ones `_first_int`
# consults below, plus the three that are redundant with the top-level `tokens_in` /
# `tokens_out` the adapter reads directly. Anything NOT in here is swept into
# `extras["usage_metadata"]` rather than dropped — see the drift note on `extras`.
#
# Keep this in sync with the `_first_int` calls. It is the mechanism that makes the
# module docstring's key enumeration self-maintaining instead of a hand-audited
# snapshot: a spelling nobody has seen shows up in `extras` on its own.
_MAPPED_USAGE_KEYS = frozenset(
    {
        # cache_read
        "input_cached_tokens",
        "inputCachedTokens",
        "cachedContentTokenCount",
        "cache_read_input_tokens",
        # cache_write
        "input_cache_creation_tokens",
        "inputCacheCreationTokens",
        "cache_creation_input_tokens",
        # reasoning
        "reasoningTokens",
        "reasoning_tokens",
        "thoughtsTokenCount",
        # Read from the top level instead, so not drift when they appear here.
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }
)


def _unmapped_usage(usage_meta: dict[str, Any]) -> dict[str, Any]:
    """Any `usage_metadata` key this adapter does not account for, wrapped for `extras`.

    Returns an empty dict when everything was recognized, so the key is absent rather
    than present-and-empty in the overwhelmingly common case.
    """
    unmapped = {k: v for k, v in usage_meta.items() if k not in _MAPPED_USAGE_KEYS}
    return {"usage_metadata": unmapped} if unmapped else {}


def _first_int(meta: dict[str, Any], *names: str) -> int:
    """First of `names` present in `meta` with a usable value, as an int.

    Cloudflare's counter names are its own and mostly snake_case, but not
    reliably so — `reasoningTokens` is camelCase in the real Gemini entry, right
    next to snake_case `input_tokens` in the same object. Since the vocabulary is
    internally inconsistent, the spelling it will use for a provider we have no
    capture for is genuinely unknown.

    Checking every plausible spelling costs nothing and the downside is lopsided
    — though it is lopsided in OPPOSITE DIRECTIONS depending on the provider, so
    neither "over-bill" nor "under-bill" describes it alone:

    - For a SUBTRACTIVE provider (`gemini`, `openai`, `workers-ai` — in
      `_INPUT_INCLUDES_CACHE_READ`), `compute_cost` subtracts `cache_read` out of
      `input`. A missed cache key leaves those tokens billed at the full prompt
      rate instead of the cache rate: an OVER-bill.
    - For an ADDITIVE provider (`anthropic`), `cache_read` is billed as its own
      line on top of `input`. A missed key means those tokens are not billed at
      all: an UNDER-bill, which is the direction this SDK treats as worse.

    Uses `or`-style fallthrough (not "first key present"), so a provider that sends
    both its own name and the gateway's with one of them zeroed still resolves to
    the real count.
    """
    for name in names:
        v = _safe_int(meta.get(name))
        if v:
            return v
    return 0


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
        # Cloudflare's own key first — that is the only spelling ever observed
        # (`input_cached_tokens` in 8 of the 14 captured fixtures). Everything after
        # it is unobserved insurance: its camelCase form, then the two big providers'
        # native names, in case Cloudflare ever forwards a provider's usage object
        # rather than rewriting it into its own vocabulary. Kept because
        # `_first_int` fallthrough is free and a missed cache key mis-bills in one
        # direction or the other for EVERY provider (see `_first_int`) — but this is
        # belt-and-braces, not handling for a case we have seen.
        cache_read=_first_int(
            usage_meta,
            "input_cached_tokens",
            "inputCachedTokens",
            "cachedContentTokenCount",  # Gemini native
            "cache_read_input_tokens",  # Anthropic native
        ),
        cache_write=_first_int(
            usage_meta,
            "input_cache_creation_tokens",
            "inputCacheCreationTokens",
            "cache_creation_input_tokens",  # Anthropic native
        ),
        reasoning=_first_int(
            usage_meta,
            "reasoningTokens",  # Cloudflare's own camelCase outlier — the observed one
            "reasoning_tokens",
            "thoughtsTokenCount",  # Gemini native
        ),
        model=_safe_str(entry.get("model")),
        provider=_normalize_provider(entry.get("provider")),
        api="cloudflare_gateway",
        extras={
            "cached": entry.get("cached"),
            "step": entry.get("step"),
            "log_id": entry.get("id"),
            # Drift sweep — the same contract `adapters/openai_native.py` enforces, and
            # for the same reason: a counter this adapter does not map must not vanish
            # without an error or an on_error. `extras` used to be exactly the three
            # keys above, so `usage_metadata` got no sweep at all, and that was not
            # hypothetical — a live Logs API pull found `neurons` (Cloudflare's Workers
            # AI billing unit) and `units` (a cost quantity) being dropped on every row,
            # and `units` appears in NO captured fixture, so the hand-maintained
            # enumeration had already drifted past what this file claimed to know. A
            # money-relevant counter going missing this way surfaces first as a
            # reconciliation gap, not as a failure.
            #
            # NESTED, not merged flat into `extras`: the poller reads `extras["cached"]`
            # to decide whether to skip billing a request Cloudflare served for free, so
            # a future `usage_metadata` key called `cached` or `step` must not be able to
            # shadow it. Omitted entirely when there is no drift, to keep the common case
            # identical to what callers already see.
            **_unmapped_usage(usage_meta),
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

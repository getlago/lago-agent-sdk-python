"""Cloudflare AI Gateway log adapter — verified against a real captured log entry."""

from __future__ import annotations

import json
import pathlib
from decimal import Decimal

import pytest

from lago_agent_sdk import CanonicalUsage
from lago_agent_sdk.gateway.adapters import extract_cloudflare_log, resolve_subscription
from lago_agent_sdk.gateway.adapters.cloudflare_gateway import _MAPPED_USAGE_KEYS
from lago_agent_sdk.pricing import compute_cost, lookup_openrouter, parse_openrouter

FIX = pathlib.Path(__file__).parent / "fixtures" / "cloudflare_gateway"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


# --------------------------------------------------------------------------
# Real fixtures
# --------------------------------------------------------------------------
def test_real_anthropic_call() -> None:
    """The exact log entry captured against a live Cloudflare account + real
    Anthropic call. These numbers were independently confirmed to roll up
    correctly in a real Lago instance (16.0 / 7.0 units billed, exact match)."""
    entry = _load("01_real_anthropic_call.json")
    u = extract_cloudflare_log(entry)
    assert u.input == 16
    assert u.output == 7
    assert u.cache_read == 0
    assert u.cache_write == 0
    assert u.model == "claude-sonnet-4-5-20250929"
    assert u.provider == "anthropic"
    assert u.api == "cloudflare_gateway"
    assert u.extras["cached"] is False
    assert u.extras["step"] == 0
    assert u.extras["log_id"] == "01KZ3Y993DV0Z5CAQCA4CJ3GRD"
    assert resolve_subscription(entry) == "cf_gateway_test_sub"


def test_real_wholesale_credits_failure_has_zero_usage() -> None:
    """A 402 (Unified Billing out of credits) never reaches the provider —
    tokens_in/out are 0 and usage_metadata is null. Must not raise, must not
    fabricate nonzero usage."""
    entry = _load("02_real_wholesale_credits_failure.json")
    u = extract_cloudflare_log(entry)
    assert u.input == 0
    assert u.output == 0
    assert not u.nonzero_numeric()
    assert resolve_subscription(entry) is None  # metadata is null on this entry


def test_real_workers_ai_provider_and_model_pass_through() -> None:
    """A different provider entirely — confirms the mapping isn't Anthropic/OpenAI-
    specific; provider/model pass through verbatim regardless of which one it is."""
    entry = _load("03_real_workers_ai_failed.json")
    u = extract_cloudflare_log(entry)
    assert u.provider == "workers-ai"
    assert u.model == "@cf/moonshotai/kimi-k2.7-code"
    assert u.input == 0
    assert u.output == 0


# --------------------------------------------------------------------------
# Real fixtures — three separate ingress methods into the same gateway.
# extract_cloudflare_log() never sees how the call was made (curl, the real
# OpenAI SDK, or a Workers AI binding) — only Cloudflare's own normalized log
# entry. These four fixtures prove that holds across every ingress method.
# --------------------------------------------------------------------------
def test_real_rest_api_bare_model_success() -> None:
    """REST API (`POST /accounts/{account}/ai/run`), a bare Workers AI model
    string with no provider prefix. Real call, real success, no BYOK needed —
    Workers AI is billed directly by Cloudflare."""
    entry = _load("09_real_rest_bare_model_success.json")
    u = extract_cloudflare_log(entry)
    assert u.input == 38
    assert u.output == 8
    assert u.provider == "workers-ai"
    assert u.model == "@cf/meta/llama-3.2-3b-instruct"
    assert resolve_subscription(entry) is None  # no metadata sent on this call


def test_real_rest_api_anthropic_402_is_provider_agnostic() -> None:
    """Same funding failure as 02_real_wholesale_credits_failure.json, but for
    Anthropic instead of OpenAI — confirms the "no BYOK/wholesale credits"
    failure mode isn't specific to one provider, and still extracts as zero
    usage regardless of which provider was requested."""
    entry = _load("07_real_rest_anthropic_402.json")
    u = extract_cloudflare_log(entry)
    assert u.input == 0
    assert u.output == 0
    assert u.provider == "anthropic"
    assert u.model == "anthropic/claude-opus-4.8"
    assert resolve_subscription(entry) is None


def test_real_unified_compat_success() -> None:
    """Unified API (`.../compat/chat/completions`), called with the real `openai`
    Python client pointed at Cloudflare's compat endpoint, routed to a Workers AI
    model. `path` and `user_agent` on the raw log entry ("OpenAI/Python 2.38.0")
    confirm this came from a real SDK call, not a raw curl — proves the log
    schema is identical regardless of which client library made the request."""
    entry = _load("08_real_unified_compat_success.json")
    assert entry["path"] == "chat/completions"
    u = extract_cloudflare_log(entry)
    assert u.input == 43
    assert u.output == 41
    assert u.provider == "workers-ai"
    assert u.model == "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


def test_real_llama_guard_moderation_model_unusual_token_shape() -> None:
    """A moderation/classifier model (llama-guard), not a chat model — input is
    dominated by the full conversation-plus-policy being classified (203 tokens)
    against a tiny 3-token verdict output. Confirms extraction doesn't assume a
    "normal" chat-shaped input/output ratio; captured from a real sweep across
    22 distinct Workers AI models with zero extraction failures."""
    entry = _load("10_real_llama_guard_moderation.json")
    u = extract_cloudflare_log(entry)
    assert u.input == 203
    assert u.output == 3
    assert u.provider == "workers-ai"
    assert u.model == "@cf/meta/llama-guard-3-8b"


def test_real_paid_plan_required_403_is_distinct_from_funding_402() -> None:
    """A different real failure mode: 403 "requires a Workers Paid plan", not the
    402 "insufficient balance" case covered elsewhere. Different status code,
    same shape otherwise — still extracts as zero usage, no attribution."""
    entry = _load("11_real_paid_plan_required_403.json")
    assert entry["status_code"] == 403
    u = extract_cloudflare_log(entry)
    assert u.input == 0
    assert u.output == 0
    assert not u.nonzero_numeric()
    assert resolve_subscription(entry) is None


def test_real_mistral_via_dedicated_endpoint() -> None:
    """Real `mistralai` SDK client, wrapped via `wrap_mistral_client`, pointed at
    Cloudflare's dedicated `.../mistral` passthrough (not the Unified/compat
    endpoint) — proves Path A generalizes to a fourth native SDK, using a real
    customer-supplied Mistral key rather than Cloudflare-side BYOK/credits."""
    entry = _load("15_real_mistral_via_dedicated_endpoint.json")
    u = extract_cloudflare_log(entry)
    assert u.input == 23
    assert u.output == 30
    assert u.provider == "mistral"
    assert u.model == "mistral-small-latest"


def test_real_gemini_reasoning_tokens_mapped_from_camelcase_field() -> None:
    """Real `google-genai` SDK client, wrapped via `wrap_gemini_client`, through
    Cloudflare's dedicated `.../google-ai-studio` passthrough.

    This is the fixture that caught a real gap: Cloudflare's log for this call
    has `usage_metadata.reasoningTokens: 852` (camelCase, unlike Anthropic's
    snake_case `input_cached_tokens`) — `extract_cloudflare_log()` didn't map it
    until this fixture surfaced it. `tokens_out` itself is only 21 (just the
    visible completion); the 852 reasoning tokens exist ONLY in usage_metadata."""
    entry = _load("16_real_gemini_via_dedicated_endpoint.json")
    assert entry["usage_metadata"]["reasoningTokens"] == 852
    u = extract_cloudflare_log(entry)
    assert u.input == 9
    assert u.output == 21
    assert u.reasoning == 852
    # Cloudflare logs this as "google-ai-studio"; the SDK's own vocabulary calls
    # it "gemini", which is what the price and token-semantics tables key off.
    assert entry["provider"] == "google-ai-studio"
    assert u.provider == "gemini"
    assert u.model == "gemini-2.5-flash"


def test_real_native_binding_with_metadata_resolves_subscription() -> None:
    """Native/binding method (`env.AI.run(model, input, {gateway: {id, metadata}})`),
    only reachable from inside a deployed Cloudflare Worker — `user_agent` on the
    raw entry is literally "cloudflare-worker". The binding's `gateway.metadata`
    option maps to the same `metadata` field as the `cf-aig-metadata` header used
    by the other two methods, so attribution resolves identically."""
    entry = _load("06_real_native_binding_with_metadata.json")
    assert entry["user_agent"] == "cloudflare-worker"
    u = extract_cloudflare_log(entry)
    assert u.input == 19
    assert u.output == 35
    assert u.provider == "workers-ai"
    assert u.model == "@cf/meta/llama-3.2-1b-instruct"
    assert resolve_subscription(entry) == "cf_gateway_test_sub"


# --------------------------------------------------------------------------
# Two DIFFERENT "cache" concepts, both verified live — don't conflate them:
#   1. Gateway-level response cache (`cached` boolean) — the entire call was
#      served from Cloudflare's own cache, costing the provider (and customer)
#      nothing at all.
#   2. Provider-level PROMPT cache (`usage_metadata.input_cache_creation_tokens`
#      / `input_cached_tokens`) — a real, separately-priced Anthropic feature
#      (`cache_control` on a content block) for reusing part of a long prompt
#      across calls that still fully execute.
# --------------------------------------------------------------------------
def test_real_gateway_cache_hit_has_zero_tokens() -> None:
    """Captured by sending the exact same request twice with cf-aig-cache-ttl
    set; the second call came back in 8ms (vs 296ms) with `cached: true`.

    Correction from an earlier assumption: a gateway cache HIT does NOT report
    the token counts the call "would have" cost — Cloudflare's own log already
    reports tokens_in/tokens_out as 0. Billing policy doesn't need to branch on
    `cached` at all; a real cache hit already extracts as zero usage."""
    entry = _load("14_real_gateway_cache_hit.json")
    u = extract_cloudflare_log(entry)
    assert u.input == 0
    assert u.output == 0
    assert u.extras["cached"] is True


def test_real_cache_write_then_read_from_anthropic_prompt_cache() -> None:
    """Two real, back-to-back Anthropic calls through the gateway with the same
    long (>1024 token) `cache_control: {"type": "ephemeral"}` system block.

    Call 1 (cache miss, writes the cache): Anthropic's own response reported
    cache_creation_input_tokens=3429, cache_read_input_tokens=0 — Cloudflare's
    log matches those exact numbers under different field names.
    Call 2 (cache hit, reads it back): the numbers flip — Anthropic reported
    cache_creation_input_tokens=0, cache_read_input_tokens=3429 — again an
    exact match in the gateway log. Unlike the gateway-level cache above, this
    call still executes and still bills the non-cached tokens normally."""
    write_entry = _load("13_real_cache_write.json")
    read_entry = _load("12_real_cache_read.json")

    w = extract_cloudflare_log(write_entry)
    assert w.input == 9
    assert w.output == 5
    assert w.cache_write == 3429
    assert w.cache_read == 0

    r = extract_cloudflare_log(read_entry)
    assert r.input == 10
    assert r.output == 4
    assert r.cache_write == 0
    assert r.cache_read == 3429


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------
def test_resolve_subscription_missing_metadata_key_returns_none() -> None:
    entry = {"metadata": {"some_other_key": "x"}}
    assert resolve_subscription(entry) is None


def test_resolve_subscription_empty_string_returns_none() -> None:
    """An empty string is falsy attribution, not a real subscription id."""
    entry = {"metadata": {"lago_subscription": ""}}
    assert resolve_subscription(entry) is None


def test_resolve_subscription_non_dict_metadata_returns_none() -> None:
    assert resolve_subscription({"metadata": "not-a-dict"}) is None
    assert resolve_subscription({"metadata": None}) is None
    assert resolve_subscription({}) is None


# --------------------------------------------------------------------------
# Robustness — a poller processes entries in a batch; one malformed entry
# must not take down the whole run.
# --------------------------------------------------------------------------
def test_survives_missing_fields() -> None:
    u = extract_cloudflare_log({})
    assert u.input == 0
    assert u.output == 0
    assert u.model == ""
    assert u.provider == ""
    assert not u.nonzero_numeric()


def test_survives_non_dict_usage_metadata() -> None:
    u = extract_cloudflare_log({"tokens_in": 5, "tokens_out": 3, "usage_metadata": "bogus"})
    assert u.input == 5
    assert u.output == 3
    assert u.cache_read == 0
    assert u.cache_write == 0


def test_survives_non_string_model_and_provider() -> None:
    u = extract_cloudflare_log({"model": 123, "provider": None, "tokens_in": 1, "tokens_out": 1})
    assert u.model == ""
    assert u.provider == ""


def test_survives_negative_and_non_numeric_tokens() -> None:
    assert extract_cloudflare_log({"tokens_in": -5}).input == 0
    assert extract_cloudflare_log({"tokens_out": "bogus"}).output == 0


# --------------------------------------------------------------------------
# Provider vocabulary — Cloudflare's names are not the SDK's names
# --------------------------------------------------------------------------
def test_provider_aliases_map_onto_sdk_vocabulary() -> None:
    """Cloudflare's log vocabulary differs from the names the pricing tables and
    the token-semantics sets key off. Verified live: `lookup_openrouter` with
    provider="google-ai-studio" missed against the real 400-model OpenRouter
    table and hit as "gemini"."""
    for raw, expected in [
        ("google-ai-studio", "gemini"),
        ("google-vertex-ai", "gemini"),
        ("vertex", "gemini"),
        ("azure-openai", "openai"),
        ("azureopenai", "openai"),
        ("workersai", "workers-ai"),
    ]:
        u = extract_cloudflare_log({"provider": raw, "tokens_in": 1})
        assert u.provider == expected, f"{raw} -> {u.provider}, expected {expected}"


def test_provider_passthrough_for_names_we_already_agree_on() -> None:
    for raw in ("anthropic", "openai", "mistral", "workers-ai"):
        assert extract_cloudflare_log({"provider": raw, "tokens_in": 1}).provider == raw


def test_unknown_provider_passes_through_untouched() -> None:
    """An unrecognized provider is one we have no price table for; a clean miss
    falls back to token events, which beats inventing a mapping."""
    assert extract_cloudflare_log({"provider": "perplexity", "tokens_in": 1}).provider == "perplexity"
    # AWS Bedrock is deliberately NOT aliased — its prices key off the `api`
    # field, which this connector always sets to "cloudflare_gateway".
    assert extract_cloudflare_log({"provider": "bedrock", "tokens_in": 1}).provider == "bedrock"


def test_normalized_gemini_provider_prices_and_bills_cache_correctly() -> None:
    """The two downstream consequences of the alias, end to end: the Gemini
    price is now findable, and cache_read is treated as a SUBSET of input
    (Gemini's semantics) instead of being billed on top of it."""
    entry = _load("16_real_gemini_via_dedicated_endpoint.json")
    u = extract_cloudflare_log(entry)
    table = parse_openrouter(
        {
            "data": [
                {
                    "id": "google/gemini-2.5-flash",
                    "pricing": {
                        "prompt": "0.0000003",
                        "completion": "0.0000025",
                        "input_cache_read": "0.000000075",
                    },
                }
            ]
        }
    )
    price = lookup_openrouter(table, u.provider, u.model)
    assert price is not None, "gemini price must resolve after normalization"

    cached = CanonicalUsage(model=u.model, provider=u.provider, api=u.api, input=1000, cache_read=800)
    b = compute_cost(cached, price, Decimal("1"))
    assert b.fields["input"]["tokens"] == "200"  # 1000 - 800, not 1000
    assert b.fields["cache_read"]["tokens"] == "800"


def test_lookup_openrouter_strips_a_redundant_vendor_prefix() -> None:
    """Real fixture 07 reports model="anthropic/claude-opus-4.8" alongside
    provider="anthropic"; unstripped that built "anthropic/anthropic/..." and
    never matched."""
    table = parse_openrouter(
        {"data": [{"id": "anthropic/claude-opus-4.8", "pricing": {"prompt": "0.000005"}}]}
    )
    assert lookup_openrouter(table, "anthropic", "anthropic/claude-opus-4.8") is not None
    assert lookup_openrouter(table, "anthropic", "claude-opus-4.8") is not None
    # Still vendor-gated: a model claiming a different vendor must not match.
    assert lookup_openrouter(table, "openai", "anthropic/claude-opus-4.8") is None


# ----------------------------------------------------------------------
# Cache-key casing. The gateway forwards some provider keys unnormalized — the
# real Gemini fixture carries camelCase `reasoningTokens` — and a missed cache
# key does not merely lose a field: `gemini` is in _INPUT_INCLUDES_CACHE_READ,
# so compute_cost needs `cache_read` populated to SUBTRACT the cached portion
# out of `input`. A silent 0 bills those tokens at the full prompt rate.
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "key",
    ["input_cached_tokens", "inputCachedTokens", "cachedContentTokenCount"],
)
def test_cache_read_is_read_under_every_plausible_spelling(key: str) -> None:
    u = extract_cloudflare_log(
        {"tokens_in": 100, "tokens_out": 10, "provider": "google-ai-studio", "usage_metadata": {key: 90}}
    )
    assert u.cache_read == 90, f"{key} must resolve"


@pytest.mark.parametrize(
    "key",
    ["input_cache_creation_tokens", "inputCacheCreationTokens", "cache_creation_input_tokens"],
)
def test_cache_write_is_read_under_every_plausible_spelling(key: str) -> None:
    u = extract_cloudflare_log(
        {"tokens_in": 100, "tokens_out": 10, "provider": "anthropic", "usage_metadata": {key: 40}}
    )
    assert u.cache_write == 40, f"{key} must resolve"


def test_a_zeroed_alias_falls_through_to_the_real_count() -> None:
    """Fallthrough is on a falsy value, not just a missing key. A provider sending
    both its own name and the gateway's, with one zeroed, must resolve to the real
    count — this is where JS's `??` diverged from Python's `or`."""
    u = extract_cloudflare_log(
        {
            "tokens_in": 100,
            "tokens_out": 10,
            "provider": "google-ai-studio",
            "usage_metadata": {"input_cached_tokens": 0, "cachedContentTokenCount": 77},
        }
    )
    assert u.cache_read == 77


def test_cache_read_still_zero_when_genuinely_absent() -> None:
    u = extract_cloudflare_log(
        {"tokens_in": 100, "tokens_out": 10, "provider": "anthropic", "usage_metadata": {}}
    )
    assert u.cache_read == 0
    assert u.cache_write == 0


def test_provider_native_cache_and_reasoning_spellings_are_accepted() -> None:
    """SYNTHETIC entries — no provider-native key appears in ANY of the 14 captured
    fixtures (they carry only Cloudflare's own vocabulary). These pin the unobserved
    insurance spellings so the fallthrough list cannot be trimmed by accident.

    The direction of the harm differs by provider, which is why both matter:
    Anthropic's cache_read is ADDITIVE, so a missed key means those tokens are never
    billed (under-bill); Gemini's is SUBTRACTIVE, so a missed key bills them at the
    full prompt rate (over-bill).
    """
    anthropic_native = extract_cloudflare_log(
        {
            "tokens_in": 100,
            "tokens_out": 10,
            "provider": "anthropic",
            "usage_metadata": {"cache_read_input_tokens": 4242},
        }
    )
    assert anthropic_native.cache_read == 4242

    gemini_native = extract_cloudflare_log(
        {
            "tokens_in": 100,
            "tokens_out": 10,
            "provider": "google-ai-studio",
            "usage_metadata": {"thoughtsTokenCount": 852},
        }
    )
    assert gemini_native.reasoning == 852

    # Cloudflare's own spelling still wins when both are present
    both = extract_cloudflare_log(
        {
            "tokens_in": 100,
            "tokens_out": 10,
            "provider": "anthropic",
            "usage_metadata": {"input_cached_tokens": 11, "cache_read_input_tokens": 4242},
        }
    )
    assert both.cache_read == 11


# ----------------------------------------------------------------------
# Drift contract for `usage_metadata`.
#
# `extras` used to be a fixed three-key dict, so any counter this adapter did not map
# was silently dropped. That was not hypothetical: a live Logs API pull found `neurons`
# and `units` vanishing on every row, and `units` appears in no captured fixture — the
# hand-maintained enumeration in the module docstring had already drifted past reality.
# Same contract `test_drift.py` pins for the native adapters.
# ----------------------------------------------------------------------
def test_drift_keeps_an_unmapped_counter_instead_of_dropping_it() -> None:
    """Exactly the shape seen live (entry 01M0FEZ2Y7QMQR1HT11GVT2HCE)."""
    u = extract_cloudflare_log(
        {
            "id": "log_1",
            "cached": False,
            "step": 0,
            "tokens_in": 37,
            "tokens_out": 2,
            "provider": "workers-ai",
            "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "usage_metadata": {
                "input_tokens": 37,
                "output_tokens": 2,
                "total_tokens": 39,
                "input_cached_tokens": 0,
                "neurons": 1.396314412355423,
                "units": 0.00001535945853590965,
            },
        }
    )
    # Cloudflare's Workers AI billing unit, and a cost quantity — both money-relevant.
    assert u.extras["usage_metadata"] == {
        "neurons": 1.396314412355423,
        "units": 0.00001535945853590965,
    }


def test_drift_sweeps_a_counter_nobody_has_ever_seen() -> None:
    u = extract_cloudflare_log(
        {
            "tokens_in": 10,
            "tokens_out": 1,
            "provider": "anthropic",
            "usage_metadata": {"input_tokens": 10, "audio_input_tokens": 512},
        }
    )
    assert u.extras["usage_metadata"] == {"audio_input_tokens": 512}


def test_drift_never_shadows_the_pollers_own_billing_inputs() -> None:
    """`extras["cached"]` decides whether to skip billing a request Cloudflare served
    for free. A usage_metadata key of the same name must not be able to overwrite it —
    which is why the sweep is nested rather than merged flat into extras."""
    u = extract_cloudflare_log(
        {
            "id": "log_2",
            "cached": True,
            "step": 3,
            "tokens_in": 5,
            "tokens_out": 1,
            "provider": "anthropic",
            "usage_metadata": {"cached": False, "step": 99, "log_id": "spoofed"},
        }
    )
    assert u.extras["cached"] is True
    assert u.extras["step"] == 3
    assert u.extras["log_id"] == "log_2"
    assert u.extras["usage_metadata"] == {"cached": False, "step": 99, "log_id": "spoofed"}


def test_drift_omits_the_key_entirely_when_there_is_none() -> None:
    """The common case must look exactly as it did before the sweep existed."""
    u = extract_cloudflare_log(
        {
            "id": "log_3",
            "cached": False,
            "step": 0,
            "tokens_in": 9,
            "tokens_out": 21,
            "provider": "anthropic",
            "usage_metadata": {
                "input_tokens": 9,
                "output_tokens": 21,
                "total_tokens": 30,
                "input_cached_tokens": 4,
            },
        }
    )
    assert u.extras == {"cached": False, "step": 0, "log_id": "log_3"}
    assert "usage_metadata" not in u.extras


@pytest.mark.parametrize(
    "key",
    [
        "input_cached_tokens",
        "inputCachedTokens",
        "cachedContentTokenCount",
        "cache_read_input_tokens",
        "input_cache_creation_tokens",
        "inputCacheCreationTokens",
        "cache_creation_input_tokens",
        "reasoningTokens",
        "reasoning_tokens",
        "thoughtsTokenCount",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ],
)
def test_drift_every_mapped_spelling_stays_out_of_the_sweep(key: str) -> None:
    """A key that IS consumed must not also show up as drift — that would read as an
    unhandled counter in reconciliation and invite double-counting."""
    u = extract_cloudflare_log(
        {
            "tokens_in": 100,
            "tokens_out": 10,
            "provider": "anthropic",
            "usage_metadata": {key: 7},
        }
    )
    assert "usage_metadata" not in u.extras, f"{key} should be mapped, not swept"


def test_drift_no_captured_fixture_loses_a_counter() -> None:
    """The sweep against real data, not constructed entries.

    Before the sweep existed this dropped `neurons` in 4 of the 14 captured entries and
    `input_text_tokens` in 1 — measured, not hypothesised. Iterating the fixtures rather
    than asserting a fixed key list is what makes this test survive a recapture: a new
    counter Cloudflare starts sending is caught by the next `capture` run, with no test
    edit and no re-audit of the module docstring.
    """
    fixtures = sorted(FIX.glob("*.json"))
    # Absent fixtures read as "not covered", never as a pass — same rule as the sweeps.
    assert fixtures, "no captured Cloudflare fixtures found"
    for path in fixtures:
        entry = _load(path.name)
        meta = entry.get("usage_metadata")
        if not isinstance(meta, dict):
            continue
        u = extract_cloudflare_log(entry)
        swept = u.extras.get("usage_metadata", {})
        for key in meta:
            assert key in _MAPPED_USAGE_KEYS or key in swept, (
                f"{path.name}: {key!r} is neither mapped nor swept into extras"
            )

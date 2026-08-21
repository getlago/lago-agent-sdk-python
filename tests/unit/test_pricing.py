"""Pricing tests — matching, money math, provider cache, and SDK price mode."""

from __future__ import annotations

import json
import pathlib
import re
import uuid
from decimal import Decimal
from typing import Any

import pytest

from lago_agent_sdk import CanonicalUsage, LagoConfig, LagoSDK, ModelPrice
from lago_agent_sdk.adapters.openai_native import extract_openai_native
from lago_agent_sdk.canonical import WORKERS_AI_COMPAT_PREFIX
from lago_agent_sdk.pricing import (
    HttpPricingFetcher,
    PricingProvider,
    _parse_price,
    _pick_mistral_canonical,
    _strip_version,
    apply_markup,
    bedrock_model_key,
    coerce_markup,
    compute_cost,
    compute_precomputed_cost,
    deoverlapped_token_total,
    lookup_bedrock,
    lookup_cloudflare_workers_ai,
    lookup_openrouter,
    parse_bedrock_offer,
    parse_bedrock_region,
    parse_cloudflare_workers_ai,
    parse_mistral_aliases,
    parse_openrouter,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "pricing"


# ----------------------------------------------------------------------
# Stub fetcher (no network) — mirrors the queue's injectable sender pattern
# ----------------------------------------------------------------------
class StubFetcher:
    def __init__(
        self,
        openrouter: dict | None = None,
        bedrock: dict | None = None,
        cloudflare_workers_ai: dict[str, ModelPrice] | None = None,
        mistral_aliases: dict[str, str] | None = None,
    ) -> None:
        self._openrouter = openrouter or {"exact": {}, "norm": {}}
        self._bedrock = bedrock or {}
        self._cloudflare_workers_ai = cloudflare_workers_ai or {}
        self._mistral_aliases = mistral_aliases or {}
        self.openrouter_calls = 0
        self.bedrock_calls: list[str] = []
        self.cloudflare_workers_ai_calls = 0
        self.mistral_aliases_calls = 0
        self.last_mistral_api_key: str | None = None

    def fetch_openrouter(self) -> dict[str, Any]:
        self.openrouter_calls += 1
        return self._openrouter

    def fetch_bedrock(self, region: str) -> dict[str, ModelPrice]:
        self.bedrock_calls.append(region)
        return self._bedrock.get(region, {})

    def fetch_cloudflare_workers_ai(self) -> dict[str, ModelPrice]:
        self.cloudflare_workers_ai_calls += 1
        return self._cloudflare_workers_ai

    def fetch_mistral_aliases(self, api_key: str | None = None) -> dict[str, str]:
        self.mistral_aliases_calls += 1
        self.last_mistral_api_key = api_key
        return self._mistral_aliases


_OPENROUTER_RAW = {
    "data": [
        {
            "id": "anthropic/claude-opus-4.8",
            "pricing": {
                "prompt": "0.000005",
                "completion": "0.000025",
                "input_cache_read": "0.0000005",
                "input_cache_write": "0.00000625",
                "internal_reasoning": "0.000025",
            },
        },
        {
            "id": "openai/gpt-4o",
            "pricing": {
                "prompt": "0.0000025",
                "completion": "0.00001",
                "input_cache_read": "0.00000125",
                "internal_reasoning": "0.00001",
            },
        },
        {"id": "mistralai/mistral-large", "pricing": {"prompt": "0.000002", "completion": "0.000006"}},
        # Real case: OpenRouter lists the dated snapshot, never the "-latest"
        # alias a customer actually requests.
        {
            "id": "mistralai/mistral-small-2603",
            "pricing": {
                "prompt": "0.00000015",
                "completion": "0.0000006",
                "input_cache_read": "0.000000015",
            },
        },
        {
            "id": "google/gemini-2.5-flash",
            "pricing": {
                "prompt": "0.0000003",
                "completion": "0.0000025",
                "input_cache_read": "0.000000075",
                "internal_reasoning": "0.0000025",
            },
        },
    ]
}

# Real data, captured live from /accounts/{id}/ai/models/search — this exact
# shape (including the non-token unit types and the no-price model) is what's
# actually in the catalog, not a synthetic guess at its structure.
_CLOUDFLARE_MODELS_RAW = [
    {
        "name": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "properties": [
            {"property_id": "context_window", "value": "24000"},
            {
                "property_id": "price",
                "value": [
                    {"unit": "per M input tokens", "price": 0.293, "currency": "USD"},
                    {"unit": "per M output tokens", "price": 2.253, "currency": "USD"},
                ],
            },
        ],
    },
    {
        "name": "@cf/moonshotai/kimi-k2.7-code",
        "properties": [
            {
                "property_id": "price",
                "value": [
                    {"unit": "per M input tokens", "price": 0.95, "currency": "USD"},
                    {"unit": "per M output tokens", "price": 4, "currency": "USD"},
                    {"unit": "per M cached input tokens", "price": 0.19, "currency": "USD"},
                ],
            },
        ],
    },
    {
        # Real non-token-priced model — must be skipped entirely, not stored
        # with a bogus/zero token price.
        "name": "@cf/pipecat-ai/smart-turn-v2",
        "properties": [
            {
                "property_id": "price",
                "value": [{"unit": "per audio minute", "price": 0.000338, "currency": "USD"}],
            },
        ],
    },
    {
        # Real case: some models have no `price` property at all.
        "name": "@cf/some/unpriced-model",
        "properties": [{"property_id": "context_window", "value": "8192"}],
    },
]

# Real data, captured live from Mistral's own /v1/models — "mistral-small-2603"
# is the dated snapshot that actually answers; "mistral-small-latest" (what a
# customer requests) is one of several aliases pointing at it.
_MISTRAL_MODELS_RAW = {
    "data": [
        {
            "id": "mistral-small-2603",
            "aliases": ["mistral-small-latest", "mistral-vibe-cli-fast", "magistral-small-latest"],
        },
        {"id": "mistral-large-2411", "aliases": ["mistral-large-latest"]},
        {"id": "codestral-2508", "aliases": []},
    ]
}

# Real data, captured live — the messy shape that actually broke this feature
# in production. Mistral's real /v1/models does NOT have one clean canonical
# entry with pure aliases: "mistral-small-2603", "mistral-small-latest", AND
# "magistral-small-latest" each appear as their OWN top-level `id`, each
# listing the other two as `aliases`. A naive "map each alias -> this
# entry's id" parser resolves "mistral-small-latest" to whichever of these
# three entries happens to be processed last — here, "magistral-small-latest"
# (index 13, after "mistral-small-latest" at index 11) — instead of the real
# dated snapshot OpenRouter lists.
_MISTRAL_MODELS_RAW_MUTUAL_ALIASING = {
    "data": [
        {
            "id": "mistral-small-2603",
            "aliases": ["mistral-small-latest", "mistral-vibe-cli-fast", "magistral-small-latest"],
        },
        {
            "id": "mistral-small-latest",
            "aliases": ["mistral-small-2603", "mistral-vibe-cli-fast", "magistral-small-latest"],
        },
        {"id": "mistral-vibe-cli-fast", "aliases": ["mistral-small-2603"]},
        {
            "id": "magistral-small-latest",
            "aliases": ["mistral-small-2603", "mistral-small-latest", "mistral-vibe-cli-fast"],
        },
        {"id": "voxtral-small-2507", "aliases": ["voxtral-small-latest"]},
        {"id": "voxtral-small-latest", "aliases": ["voxtral-small-2507"]},
    ]
}


# ----------------------------------------------------------------------
# OpenRouter parsing + matching
# ----------------------------------------------------------------------
def test_openrouter_exact_and_normalized_match() -> None:
    table = parse_openrouter(_OPENROUTER_RAW)
    # normalized: our "claude-opus-4-8" matches OpenRouter "claude-opus-4.8"
    mp = lookup_openrouter(table, "anthropic", "claude-opus-4-8")
    assert mp is not None
    assert mp.input == Decimal("0.000005")
    assert mp.output == Decimal("0.000025")
    assert mp.cache_read == Decimal("0.0000005")
    assert mp.reasoning == Decimal("0.000025")
    assert mp.source == "openrouter"


def test_openrouter_vendor_map_mistral_and_gemini() -> None:
    table = parse_openrouter(_OPENROUTER_RAW)
    # provider "mistral" -> vendor "mistralai"
    assert lookup_openrouter(table, "mistral", "mistral-large") is not None
    # provider "gemini" -> vendor "google"
    assert lookup_openrouter(table, "gemini", "gemini-2.5-flash") is not None


def test_openrouter_date_version_stripped_match() -> None:
    table = parse_openrouter(
        {"data": [{"id": "anthropic/claude-haiku-4.5", "pricing": {"prompt": "0.000001"}}]}
    )
    # our id carries a date suffix; matcher strips it
    mp = lookup_openrouter(table, "anthropic", "claude-haiku-4-5-20251001")
    assert mp is not None
    assert mp.input == Decimal("0.000001")


@pytest.mark.parametrize("alias_first", [True, False])
def test_openrouter_moving_alias_never_overwrites_a_real_listing(alias_first: bool) -> None:
    """A "~" alias and a real listing can collide on the same key. Which one wins
    must not depend on catalog order — with plain assignment it did, and the moving
    alias's rate (0.009) could replace the real listing's (0.001) purely by
    arriving later in the response."""
    real = {"id": "google/gemini-flash-latest", "pricing": {"prompt": "0.001"}}
    alias = {"id": "~google/gemini-flash-latest", "pricing": {"prompt": "0.009"}}
    data = [alias, real] if alias_first else [real, alias]
    table = parse_openrouter({"data": data})
    mp = lookup_openrouter(table, "gemini", "gemini-flash-latest")
    assert mp is not None
    assert mp.input == Decimal("0.001"), (
        f"real listing must win regardless of order (alias_first={alias_first})"
    )
    # the "~"-spelled id still resolves to its own entry
    assert table["exact"]["~google/gemini-flash-latest"].input == Decimal("0.009")


def test_three_digit_revision_is_stripped_for_openrouter_only() -> None:
    """The "-002" arm exists for Gemini's revision, which only OpenRouter omits.

    It must NOT reach the AWS/Bedrock key builder: there a shortened key does not
    merely miss, it collapses two models onto one key whose per-direction prices
    are assigned in place, so one silently overwrites the other's rate.
    """
    table = parse_openrouter({"data": [{"id": "google/gemini-2.5-flash", "pricing": {"prompt": "0.001"}}]})
    assert lookup_openrouter(table, "gemini", "gemini-2.5-flash-002") is not None

    # shared helper keeps a 3-digit tail, so distinct Bedrock ids stay distinct
    assert bedrock_model_key("amazon.titan-text-001") == "titantext001"
    assert bedrock_model_key("amazon.titan-text-002") == "titantext002"
    assert bedrock_model_key("amazon.titan-text-001") != bedrock_model_key("amazon.titan-text-002")
    # real dated/versioned ids are unaffected
    assert bedrock_model_key("anthropic.claude-haiku-4-5-20251001-v1:0") == "claudehaiku45"
    assert bedrock_model_key("eu.anthropic.claude-sonnet-4-6") == "claudesonnet46"


def test_unparseable_markup_keeps_the_cost_instead_of_zeroing_it() -> None:
    """Defence in depth, and cross-port parity — `coerce_markup` means neither
    branch is reachable through `emit()` (see the coerce test further down).

    The two bad inputs are not interchangeable: a bad COST leaves nothing to bill,
    but a bad MARKUP only loses the multiplier, and returning 0 for it would
    discard a good cost. JS already fell back to 1.0 here; Python returned "0", so
    the ports would have billed differently had anything reached it.
    """
    assert apply_markup("0.0042", "1.5") == "0.0063"
    for bad in ("abc", "", "1,5", "None"):
        assert apply_markup("0.0042", bad) == "0.0042", f"markup={bad!r} must not zero the cost"
    # an unparseable COST is different: there is nothing to bill
    assert apply_markup("abc", "1.5") == "0"


def test_openrouter_miss_returns_none() -> None:
    table = parse_openrouter(_OPENROUTER_RAW)
    assert lookup_openrouter(table, "anthropic", "totally-made-up-model") is None
    # vendor-gated: right model name, wrong vendor -> miss
    assert lookup_openrouter(table, "openai", "claude-opus-4-8") is None


# ----------------------------------------------------------------------
# Cloudflare Workers AI parsing + matching
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# OpenRouter's "~" moving-alias marker. Measured live: 11 ids across 6 vendors,
# every one a "-latest" moniker with real token pricing, and every one
# unpriceable before this — the vendor parsed as "~anthropic"/"~openai"/"~google",
# which match nothing in _VENDOR_MAP.
# ----------------------------------------------------------------------
_TILDE_RAW = {
    "data": [
        {
            "id": "~anthropic/claude-sonnet-latest",
            "pricing": {"prompt": "0.000002", "completion": "0.00001"},
        },
        {"id": "~openai/gpt-latest", "pricing": {"prompt": "0.0000025", "completion": "0.000015"}},
        {
            "id": "~google/gemini-flash-latest",
            "pricing": {"prompt": "0.000000375", "completion": "0.000001875"},
        },
    ]
}


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("anthropic", "claude-sonnet-latest"),
        ("openai", "gpt-latest"),
        ("gemini", "gemini-flash-latest"),
    ],
)
def test_moving_alias_ids_are_priceable(provider: str, model: str) -> None:
    """A "-latest" alias a customer plausibly requests must resolve. Billing
    nothing at all is the outcome in an llm_cost-only setup."""
    t = parse_openrouter(_TILDE_RAW)
    assert lookup_openrouter(t, provider, model) is not None


def test_moving_alias_still_indexed_under_its_verbatim_id() -> None:
    """Stripping the marker must ADD a key, not replace one — the raw id stays
    resolvable so nothing that already worked breaks."""
    t = parse_openrouter(_TILDE_RAW)
    assert "~openai/gpt-latest" in t["exact"]
    assert "openai/gpt-latest" in t["exact"]


def test_three_digit_revision_suffix_strips_to_a_hit() -> None:
    """Gemini's `model_version` can report a "-002" revision where OpenRouter
    lists only the bare name. Verified against the live catalog that no real id's
    model part ends in exactly three digits, so this arm is safe."""
    t = parse_openrouter(_TILDE_RAW)
    assert lookup_openrouter(t, "gemini", "gemini-flash-latest-002") is not None


def test_cloudflare_parses_real_price_shape() -> None:
    table = parse_cloudflare_workers_ai(_CLOUDFLARE_MODELS_RAW)
    mp = lookup_cloudflare_workers_ai(table, "@cf/meta/llama-3.3-70b-instruct-fp8-fast")
    assert mp is not None
    assert mp.source == "cloudflare_workers_ai"
    # $0.293/M input -> $0.000000293/token; $2.253/M output -> $0.000002253/token
    assert mp.input == Decimal("0.000000293")
    assert mp.output == Decimal("0.000002253")
    assert mp.cache_read is None  # this model has no cached-input price


def test_cloudflare_maps_cached_input_tokens_to_cache_read() -> None:
    table = parse_cloudflare_workers_ai(_CLOUDFLARE_MODELS_RAW)
    mp = lookup_cloudflare_workers_ai(table, "@cf/moonshotai/kimi-k2.7-code")
    assert mp is not None
    assert mp.input == Decimal("0.00000095")
    assert mp.output == Decimal("0.000004")
    assert mp.cache_read == Decimal("0.00000019")


def test_cloudflare_skips_non_token_priced_model() -> None:
    """A real model priced only in "per audio minute" — not a canonical priced
    field — must be absent from the table entirely, not stored with a bogus
    zero/None price that could be mistaken for "free"."""
    table = parse_cloudflare_workers_ai(_CLOUDFLARE_MODELS_RAW)
    assert "@cf/pipecat-ai/smart-turn-v2" not in table


def test_cloudflare_skips_model_with_no_price_property() -> None:
    table = parse_cloudflare_workers_ai(_CLOUDFLARE_MODELS_RAW)
    assert "@cf/some/unpriced-model" not in table


def test_cloudflare_lookup_miss_returns_none() -> None:
    table = parse_cloudflare_workers_ai(_CLOUDFLARE_MODELS_RAW)
    assert lookup_cloudflare_workers_ai(table, "@cf/totally/made-up-model") is None


def test_cloudflare_lookup_version_suffix_fallback() -> None:
    """Real drift we've observed: a live response naming a model with a
    trailing "-v2" the catalog itself doesn't have listed separately."""
    table = parse_cloudflare_workers_ai(_CLOUDFLARE_MODELS_RAW)
    mp = lookup_cloudflare_workers_ai(table, "@cf/meta/llama-3.3-70b-instruct-fp8-fast-v2")
    assert mp is not None
    assert mp.input == Decimal("0.000000293")


@pytest.mark.parametrize(
    "requested",
    [
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        # The routing prefix and the version-suffix drift, together.
        "workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast-v2",
    ],
)
def test_cloudflare_lookup_accepts_the_compat_routing_prefix(requested: str) -> None:
    """Cloudflare's catalog lists bare "@cf/..." names, but reaching a model
    through the gateway's OpenAI-compatible `/compat` endpoint requires the
    "workers-ai/" prefix — the form the README prescribes and the only form a
    streaming call reports. Both must price to the same rate."""
    table = parse_cloudflare_workers_ai(_CLOUDFLARE_MODELS_RAW)
    mp = lookup_cloudflare_workers_ai(table, requested)
    assert mp is not None, f"{requested} should have priced"
    assert mp.input == Decimal("0.000000293")


def test_cloudflare_lookup_miss_is_still_a_miss_with_the_prefix() -> None:
    """The prefix strip must not turn an unknown model into a false hit."""
    table = parse_cloudflare_workers_ai(_CLOUDFLARE_MODELS_RAW)
    assert lookup_cloudflare_workers_ai(table, "workers-ai/@cf/nope/not-a-model") is None


@pytest.mark.parametrize(
    "requested",
    [
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    ],
)
def test_workers_ai_provider_inferred_from_both_spellings(requested: str) -> None:
    """A streaming Workers AI call carries no response model, so the requested
    string — which the docs give in prefixed form — is all `_infer_provider`
    has. Stamping "openai" there priced it against OpenRouter, missed, and
    silently degraded to token events."""
    u = extract_openai_native({"usage": {"prompt_tokens": 10, "completion_tokens": 5}}, model_id=requested)
    assert u.provider == "workers-ai"
    # The model keeps the spelling the customer used — the strip happens at lookup,
    # so reporting stays faithful to the request.
    assert u.model == requested


@pytest.mark.parametrize(
    ("usage", "expected", "why"),
    [
        # Ancor's cited case: a real captured Gemini row. `input + output` dropped
        # 852 additive reasoning tokens and published unit="30" for 882 consumed.
        (
            CanonicalUsage(input=9, output=21, reasoning=852, provider="gemini", api="x", model="m"),
            882,
            "gemini reasoning is additive",
        ),
        # Cache-inclusive provider: cache_read sits INSIDE input, so counting both
        # would double it.
        (
            CanonicalUsage(input=10000, output=100, cache_read=9000, provider="openai", api="x", model="m"),
            10100,
            "openai cache_read is a subset of input",
        ),
        # Additive provider: cache_read/cache_write are real extra consumption, and
        # the old basis under-reported this by 9.6x.
        (
            CanonicalUsage(
                input=1000,
                output=100,
                cache_read=9000,
                cache_write=500,
                provider="anthropic",
                api="x",
                model="m",
            ),
            10600,
            "anthropic cache is additive",
        ),
        # reasoning ⊆ output for openai — must not be added on top.
        (
            CanonicalUsage(input=10, output=100, reasoning=80, provider="openai", api="x", model="m"),
            110,
            "openai reasoning is a subset of output",
        ),
        # tool_calls is a CALL COUNT, not tokens, so it must never land in a token total.
        (
            CanonicalUsage(input=10, output=20, tool_calls=3, provider="openai", api="x", model="m"),
            30,
            "tool_calls excluded",
        ),
        # --- gateway SURFACES that re-shape every vendor (_OPENAI_SHAPED_APIS) ---
        # Real shape from system.ai_gateway.usage: input CONTAINS cache_read even for
        # Anthropic, whose own API reports it additively. Keying on the vendor billed
        # 48,798 tokens against 31,091 consumed on a real backfill (1.570x). The honest
        # total is the table's own total_tokens, i.e. input + output.
        (
            CanonicalUsage(
                input=1822,
                output=4,
                cache_read=1812,
                provider="anthropic",
                api="databricks_gateway",
                model="m",
            ),
            1826,
            "databricks_gateway folds cache_read into input for every vendor",
        ),
        # The write half of the same shape — the overlap no provider-keyed set covers,
        # because no vendor's native API reports cache_write inside input.
        (
            CanonicalUsage(
                input=1825,
                output=4,
                cache_write=1812,
                provider="anthropic",
                api="databricks_gateway",
                model="m",
            ),
            1829,
            "databricks_gateway folds cache_write into input too",
        ),
        # Hosted Databricks models bill as TOKEN COUNTS (TOKEN_BILLED_PROVIDERS), so
        # this path IS the bill. Latent today (0 of 96 hosted rows carry cache) and a
        # direct 1.991x over-bill the day one does.
        (
            CanonicalUsage(
                input=1825,
                output=4,
                cache_read=1812,
                provider="databricks",
                api="databricks_gateway",
                model="m",
            ),
            1829,
            "hosted databricks rows carry the surface's shape, not a vendor's",
        ),
        # A vendor the surface set must NOT reach: reasoning is inside output here even
        # though gemini reports thoughts additively on its own API.
        (
            CanonicalUsage(
                input=500,
                output=200,
                reasoning=50,
                provider="gemini",
                api="databricks_gateway",
                model="m",
            ),
            700,
            "databricks_gateway folds reasoning into output for every vendor",
        ),
        # Cloudflare is deliberately NOT in _OPENAI_SHAPED_APIS: measured on real logs,
        # an anthropic entry reads input=10, output=4, total=14 with cache OUTSIDE that
        # total. Adding it to the set would UNDER-bill by the cached portion.
        (
            CanonicalUsage(
                input=10,
                output=4,
                cache_read=3429,
                provider="anthropic",
                api="cloudflare_gateway",
                model="m",
            ),
            3443,
            "cloudflare preserves each vendor's native shape",
        ),
    ],
)
def test_deoverlapped_token_total(usage: CanonicalUsage, expected: int, why: str) -> None:
    assert deoverlapped_token_total(usage) == expected, why


@pytest.mark.parametrize("provider", ["openai", "workers-ai"])
def test_openai_shaped_providers_treat_reasoning_as_a_subset(provider: str) -> None:
    """workers-ai is reached ONLY through Cloudflare's OpenAI-compatible endpoint, so
    reasoning is a subset of output there exactly as it is for real OpenAI. Omitting it
    from _OUTPUT_INCLUDES_REASONING counted the subset twice — 1900 against 1100."""
    u = CanonicalUsage(input=100, output=1000, reasoning=800, model="m", provider=provider, api="chat")
    assert deoverlapped_token_total(u) == 1100


def test_precomputed_unit_matches_the_split_path_basis() -> None:
    """The two cost branches must report the same quantity for one call — that was
    the actual complaint: `unit` on the single-event path used a different basis
    from `parts["tokens"]` on the split path."""
    received: list = []
    provider = _warm_provider()
    sdk, got = _price_sdk(provider)
    u = CanonicalUsage(
        input=1000, output=100, cache_read=900, model="claude-opus-4-8", provider="anthropic", api="native"
    )
    # Split path (real per-field breakdown).
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    split = [e for batch in got for e in batch]
    split_total = sum(int(e["properties"]["unit"]) for e in split)

    # Single-event path (precomputed cost).
    sdk2, got2 = _price_sdk(_warm_provider())
    sdk2.emit(u, usd_cost=0.05)
    assert sdk2.flush(timeout=2.0)
    sdk2.shutdown(timeout=1.0)
    single = [e for batch in got2 for e in batch]
    assert len(single) == 1
    assert int(single[0]["properties"]["unit"]) == split_total, (
        f"single-event unit {single[0]['properties']['unit']} != split total {split_total}"
    )
    _ = received


def test_cloudflare_entry_with_null_properties_does_not_unprice_everything() -> None:
    """`.get("properties", [])` only defaults when the key is ABSENT — an explicit
    JSON null returns None and `for p in None` raised TypeError out of this
    function into maybe_refresh's handler, leaving the whole table None. One
    malformed entry would unprice EVERY Workers AI model, not just its own."""
    raw = [
        {"name": "@cf/broken/model", "properties": None},
        {
            "name": "@cf/good/model",
            "properties": [
                {
                    "property_id": "price",
                    "value": [{"unit": "per M input tokens", "price": 1.0, "currency": "USD"}],
                }
            ],
        },
    ]
    table = parse_cloudflare_workers_ai(raw)
    assert "@cf/good/model" in table, "a sibling entry must survive a malformed one"
    assert "@cf/broken/model" not in table


def _cf_pages(*counts: int, total_count: int | None = None) -> list[dict]:
    """Fake paged responses: `counts[i]` models on page i+1."""
    pages = []
    for n in counts:
        info: dict = {"page": len(pages) + 1, "per_page": 50, "count": n}
        if total_count is not None:
            info["total_count"] = total_count
        pages.append(
            {
                "result": [
                    {
                        "name": f"@cf/m/p{len(pages) + 1}-{i}",
                        "properties": [
                            {
                                "property_id": "price",
                                "value": [{"unit": "per M input tokens", "price": 1.0, "currency": "USD"}],
                            }
                        ],
                    }
                    for i in range(n)
                ],
                "result_info": info,
            }
        )
    return pages


def _run_cf_fetch(pages: list[dict]) -> tuple[int, list[int]]:
    """Drive fetch_cloudflare_workers_ai against faked pages; return (models, pages hit)."""
    import requests as _rq

    seen: list[int] = []
    orig = _rq.get

    class _Resp:
        def __init__(self, body):
            self._b = body

        def raise_for_status(self):
            pass

        def json(self):
            return self._b

    def fake(url, **kw):
        page = int((kw.get("params") or {}).get("page", 1))
        seen.append(page)
        return _Resp(pages[page - 1] if page - 1 < len(pages) else {"result": [], "result_info": {}})

    _rq.get = fake
    try:
        f = HttpPricingFetcher(cloudflare_account_id="acct", cloudflare_api_token="tok")
        table = f.fetch_cloudflare_workers_ai()
    finally:
        _rq.get = orig
    return len(table), seen


def test_cloudflare_pagination_walks_until_a_short_page() -> None:
    """Matches the real endpoint, which serves 50 then 14 then 0."""
    n, seen = _run_cf_fetch(_cf_pages(50, 14, total_count=291))
    assert seen == [1, 2], f"should stop after the short page, hit {seen}"
    assert n == 64


def test_cloudflare_pagination_survives_a_missing_total_count() -> None:
    """The bug: `total_count` defaulting to len(models) made an absent count break
    after page one, silently keeping 50 of the 64 available."""
    n, seen = _run_cf_fetch(_cf_pages(50, 14, total_count=None))
    assert seen == [1, 2], f"a missing total_count must not stop paging, hit {seen}"
    assert n == 64


def test_cloudflare_pagination_ignores_a_wrong_total_count() -> None:
    """Measured live: the endpoint reports total_count=291 while serving 64, so a
    `len(models) >= total` test can never be the terminator."""
    n, _ = _run_cf_fetch(_cf_pages(50, 14, total_count=291))
    assert n == 64


def test_cloudflare_pagination_is_bounded() -> None:
    """This runs on the queue's flush tick ahead of the drain, so an endpoint that
    always returns a full page must not stall event delivery."""
    n, seen = _run_cf_fetch(_cf_pages(*([50] * 60), total_count=100000))
    assert len(seen) <= 40, f"loop must be bounded, hit {len(seen)} pages"


def test_cloudflare_fetcher_returns_empty_without_credentials() -> None:
    """No account id / token set — Workers AI pricing is simply unavailable,
    not an error; the fetch never even makes a request."""
    fetcher = HttpPricingFetcher()
    assert fetcher.fetch_cloudflare_workers_ai() == {}


# ----------------------------------------------------------------------
# Mistral alias resolution
# ----------------------------------------------------------------------
def test_mistral_parses_real_alias_shape() -> None:
    aliases = parse_mistral_aliases(_MISTRAL_MODELS_RAW)
    assert aliases["mistral-small-latest"] == "mistral-small-2603"
    assert aliases["mistral-vibe-cli-fast"] == "mistral-small-2603"
    assert aliases["magistral-small-latest"] == "mistral-small-2603"
    assert aliases["mistral-large-latest"] == "mistral-large-2411"


def test_mistral_model_with_no_aliases_contributes_nothing() -> None:
    aliases = parse_mistral_aliases(_MISTRAL_MODELS_RAW)
    assert "codestral-2508" not in aliases  # it's an id, never requested as an alias


def test_mistral_alias_resolves_to_a_real_openrouter_listing() -> None:
    """The whole point: the resolved id isn't a dead end — OpenRouter lists it."""
    aliases = parse_mistral_aliases(_MISTRAL_MODELS_RAW)
    table = parse_openrouter(_OPENROUTER_RAW)
    resolved = aliases["mistral-small-latest"]
    mp = lookup_openrouter(table, "mistral", resolved)
    assert mp is not None
    assert mp.input == Decimal("0.00000015")
    assert mp.output == Decimal("0.0000006")
    assert mp.cache_read == Decimal("0.000000015")


def test_mistral_fetcher_returns_empty_without_credentials() -> None:
    """No API key set — alias resolution is simply skipped, not an error; the
    fetch never even makes a request."""
    fetcher = HttpPricingFetcher()
    assert fetcher.fetch_mistral_aliases() == {}


def test_mistral_fetcher_accepts_a_key_passed_at_call_time() -> None:
    """The key learned from a wrapped client (see PricingProvider.learn_mistral_api_key)
    is passed per-call, not baked into the fetcher at construction — no
    explicit config key is required for this path to work."""
    fetcher = HttpPricingFetcher()
    calls = []

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": []}

    def _fake_get(url, headers=None, timeout=None):
        calls.append(headers)
        return _FakeResp()

    import requests as _requests

    orig = _requests.get
    _requests.get = _fake_get
    try:
        fetcher.fetch_mistral_aliases(api_key="learned-key-123")
    finally:
        _requests.get = orig
    assert calls == [{"Authorization": "Bearer learned-key-123"}]


def test_mistral_fetcher_explicit_config_key_wins_over_learned_key() -> None:
    """A key deliberately set via LagoConfig.mistral_api_key must not be
    silently shadowed by one auto-detected from a wrapped client."""
    fetcher = HttpPricingFetcher(mistral_api_key="configured-key")
    calls = []

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": []}

    def _fake_get(url, headers=None, timeout=None):
        calls.append(headers)
        return _FakeResp()

    import requests as _requests

    orig = _requests.get
    _requests.get = _fake_get
    try:
        fetcher.fetch_mistral_aliases(api_key="learned-key-123")
    finally:
        _requests.get = orig
    assert calls == [{"Authorization": "Bearer configured-key"}]


def test_mistral_mutual_aliasing_resolves_to_the_dated_snapshot_not_another_alias() -> None:
    """Real bug, found live: naively mapping "each alias -> this entry's id"
    is order-dependent when Mistral lists a "-latest" moniker as its OWN
    top-level `id` too (it does, for every alias in this real shape) — it
    resolved "mistral-small-latest" to "magistral-small-latest" (whichever
    entry got processed last), not "mistral-small-2603". OpenRouter lists
    the dated snapshot, never the sibling alias, so that resolution was a
    dead end in production. Every one of the 4 mutually-aliasing names must
    land on the single dated snapshot, regardless of which entry mentions
    which or what order they're processed in."""
    aliases = parse_mistral_aliases(_MISTRAL_MODELS_RAW_MUTUAL_ALIASING)
    assert aliases["mistral-small-latest"] == "mistral-small-2603"
    assert aliases["mistral-vibe-cli-fast"] == "mistral-small-2603"
    assert aliases["magistral-small-latest"] == "mistral-small-2603"
    # The canonical name itself is never a key — nothing should "resolve" it
    # to something else.
    assert "mistral-small-2603" not in aliases


def test_mistral_mutual_aliasing_reversed_input_order_gives_same_result() -> None:
    """The result must not depend on which entry the source API happens to
    list first — that's exactly the bug this replaced (last-write-wins)."""
    reversed_data = {"data": list(reversed(_MISTRAL_MODELS_RAW_MUTUAL_ALIASING["data"]))}
    aliases = parse_mistral_aliases(reversed_data)
    assert aliases["mistral-small-latest"] == "mistral-small-2603"
    assert aliases["magistral-small-latest"] == "mistral-small-2603"


def test_mistral_two_way_aliasing_still_resolves() -> None:
    """The simplest mutual case — just id A and id B each listing the
    other — must also converge on one canonical (the dated one), not stay
    as a symmetric pair or resolve backwards."""
    aliases = parse_mistral_aliases(_MISTRAL_MODELS_RAW_MUTUAL_ALIASING)
    assert aliases["voxtral-small-latest"] == "voxtral-small-2507"
    assert "voxtral-small-2507" not in aliases


def test_mistral_family_resolves_to_the_NEWEST_dated_snapshot() -> None:
    """Regression: the tie-break used to resolve on the date ASCENDING.

    Every dated id in one family is the same length, so `(len(n), n)` fell
    through to the alphabetical term — which for `-2402` / `-2407` / `-2411` is
    the date, oldest first. The whole family collapsed onto `mistral-large-2402`
    and got priced at a two-year-old rate.
    """
    family = [
        "mistral-large-2402",
        "mistral-large-2407",
        "mistral-large-2411",
        "mistral-large-latest",
    ]
    data = {"data": [{"id": n, "aliases": [x for x in family if x != n]} for n in family]}
    aliases = parse_mistral_aliases(data)
    assert aliases["mistral-large-latest"] == "mistral-large-2411"


def test_mistral_explicit_dated_snapshot_is_never_remapped() -> None:
    """An exact snapshot request is already the id OpenRouter lists, so it must
    pass through untouched. Remapping it onto the group's canonical priced it at
    a sibling's rate — a mispricing, not a miss."""
    family = ["mistral-large-2402", "mistral-large-2411", "mistral-large-latest"]
    data = {"data": [{"id": n, "aliases": [x for x in family if x != n]} for n in family]}
    aliases = parse_mistral_aliases(data)
    assert "mistral-large-2402" not in aliases
    assert "mistral-large-2411" not in aliases
    assert aliases["mistral-large-latest"] == "mistral-large-2411"


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        # Mistral's own 4-digit YYMM convention.
        (["m-2402", "m-2411", "m-latest"], "m-2411"),
        # Mixed widths: "20250929" sorts BELOW "2411" as a raw string, so the
        # normalization to one scale is what makes this come out right.
        (["m-2411", "m-20250929", "m-latest"], "m-20250929"),
        # No dated candidate at all — deterministic shortest-then-code-point.
        (["mm-latest", "m-latest"], "m-latest"),
    ],
)
def test_mistral_canonical_picks_newest_across_suffix_shapes(names: list[str], expected: str) -> None:
    assert _pick_mistral_canonical(names) == expected


def test_mistral_canonical_orders_by_code_point_not_locale() -> None:
    """Cross-repo parity: the JS port must not use `localeCompare`, which is
    ICU/locale-dependent. Both repos must pick the same canonical for a group
    whose members differ only by case/separator — and the pick has to be the one
    that still normalizes onto a name OpenRouter lists."""
    assert _pick_mistral_canonical(["mistral-small-2603", "Mistral-Small-2603"]) == "Mistral-Small-2603"


# ----------------------------------------------------------------------
# Bedrock region + key + offer parsing
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "model,expected",
    [
        ("eu.anthropic.claude-sonnet-4-6", "eu-west-1"),
        ("us.anthropic.claude-sonnet-4-6", "us-east-1"),
        ("apac.anthropic.claude-sonnet-4-6", "ap-southeast-1"),
        ("anthropic.claude-haiku-4-5-20251001-v1:0", "us-east-1"),  # no prefix -> default
    ],
)
def test_bedrock_region_detection(model: str, expected: str) -> None:
    assert parse_bedrock_region(model, "us-east-1") == expected


@pytest.mark.parametrize(
    "model,expected_key",
    [
        ("eu.anthropic.claude-sonnet-4-6", "claudesonnet46"),
        ("anthropic.claude-haiku-4-5-20251001-v1:0", "claudehaiku45"),
        ("mistral.mixtral-8x7b-instruct-v0:1", "mixtral8x7binstruct"),
    ],
)
def test_bedrock_model_key(model: str, expected_key: str) -> None:
    assert bedrock_model_key(model) == expected_key


def _aws_product(model: str, inference_type: str, usd: str, unit: str = "1K tokens") -> tuple[dict, dict]:
    """Build one (product, term) pair matching the real AWS Bedrock offer schema."""
    sku = f"{model}:{inference_type}".replace(" ", "")
    product = {
        sku: {
            "productFamily": "...",
            "attributes": {
                "model": model,
                "usagetype": f"USE1-{model.replace(' ', '')}-{inference_type.replace(' ', '-')}",
                "inferenceType": inference_type,
                "feature": "On-demand Inference",
                "provider": "Anthropic",
            },
        }
    }
    term = {sku: {"off": {"priceDimensions": {"d": {"pricePerUnit": {"USD": usd}, "unit": unit}}}}}
    return product, term


def test_bedrock_offer_parse_and_lookup() -> None:
    # Real AWS schema: inferenceType distinguishes direction; unit is "1K tokens".
    p_in, t_in = _aws_product("Claude Sonnet 4.6", "Input tokens", "0.003")  # $3/M
    p_out, t_out = _aws_product("Claude Sonnet 4.6", "Output tokens", "0.015")  # $15/M
    offer = {"products": {**p_in, **p_out}, "terms": {"OnDemand": {**t_in, **t_out}}}
    table = parse_bedrock_offer(offer, "us-east-1")
    mp = lookup_bedrock(table, "us.anthropic.claude-sonnet-4-6")
    assert mp is not None
    assert mp.input == Decimal("0.000003")  # 0.003 per 1K -> 3e-6 per token
    assert mp.output == Decimal("0.000015")
    assert mp.source == "aws_bedrock"


def test_bedrock_offer_rejects_tier_variants() -> None:
    # Standard on-demand tier must win over priority/flex/batch variants.
    p_std, t_std = _aws_product("Claude Sonnet 4.6", "Input tokens", "0.003")
    p_pri, t_pri = _aws_product("Claude Sonnet 4.6", "Input tokens priority", "0.006")
    p_flex, t_flex = _aws_product("Claude Sonnet 4.6", "Input tokens flex", "0.0015")
    offer = {
        "products": {**p_std, **p_pri, **p_flex},
        "terms": {"OnDemand": {**t_std, **t_pri, **t_flex}},
    }
    table = parse_bedrock_offer(offer, "us-east-1")
    mp = lookup_bedrock(table, "anthropic.claude-sonnet-4-6")
    assert mp is not None
    assert mp.input == Decimal("0.000003")  # the standard tier, not priority/flex


def test_bedrock_usagetype_fallback_when_no_inference_type() -> None:
    # Resilience: if AWS ever drops inferenceType, fall back to usagetype scan.
    offer = {
        "products": {
            "S": {"attributes": {"model": "Mixtral 8x7B Instruct", "usagetype": "USE1-Input-Tokens"}}
        },
        "terms": {
            "OnDemand": {
                "S": {
                    "o": {"priceDimensions": {"d": {"pricePerUnit": {"USD": "0.0005"}, "unit": "1K tokens"}}}
                }
            }
        },
    }
    table = parse_bedrock_offer(offer, "us-east-1")
    mp = lookup_bedrock(table, "mistral.mixtral-8x7b-instruct-v0:1")
    assert mp is not None
    assert mp.input == Decimal("0.0000005")  # 0.0005 per 1K -> 5e-7 per token


# ----------------------------------------------------------------------
# compute_cost + golden money parity
# ----------------------------------------------------------------------
def test_compute_cost_excludes_unpriced_fields() -> None:
    price = ModelPrice(source="openrouter", input=Decimal("0.000003"), output=Decimal("0.000015"))
    # tool_calls is not a priced field; image_input has no unit price
    usage = CanonicalUsage(
        input=1000, output=500, tool_calls=3, image_input=50, model="m", provider="p", api="native"
    )
    b = compute_cost(usage, price, Decimal("1"))
    assert set(b.fields) == {"input", "output"}
    assert b.base == "0.0105"
    assert b.total == "0.0105"


def test_compute_cost_only_unpriced_fields_yields_zero() -> None:
    # model priced but the call's only count is an unpriced field
    price = ModelPrice(source="openrouter", input=Decimal("0.000003"))
    usage = CanonicalUsage(tool_calls=5, model="m", provider="p", api="native")
    b = compute_cost(usage, price, Decimal("1"))
    assert b.total == "0"
    assert b.fields == {}


def test_compute_precomputed_cost_matches_gateway_reported_amount() -> None:
    """Cloudflare AI Gateway reports its own real cost per call (e.g. the
    `cost` field on a log entry, in USD) — this must bill that exact amount,
    not something recomputed from a per-token table."""
    b = compute_precomputed_cost(0.00010472, Decimal("1"))
    assert b.total == "0.00010472"
    assert b.total_cents == "0.010472"
    assert b.base == "0.00010472"
    assert b.source == "precomputed"
    assert b.fields == {}  # no per-field breakdown — Cloudflare gives one lump sum


def test_compute_precomputed_cost_applies_markup() -> None:
    b = compute_precomputed_cost(0.0001, Decimal("2"))
    assert b.base == "0.0001"
    assert b.total == "0.0002"
    assert b.total_cents == "0.02"


def test_compute_precomputed_cost_negative_floors_to_zero() -> None:
    b = compute_precomputed_cost(-5, Decimal("1"))
    assert b.total == "0"
    assert b.base == "0"


def test_money_golden_cases() -> None:
    cases = json.loads((FIXTURES / "money_golden.json").read_text())["cases"]
    for c in cases:
        prices = {k: Decimal(v) for k, v in c["prices"].items()}
        price = ModelPrice(source="openrouter", **prices)
        # `provider` is optional and defaults to a name in no _INCLUDES_ set, so
        # the pre-existing cases keep their original semantics; cases that pin
        # per-provider token semantics set it explicitly. `api` defaults to
        # "native" for the same reason — only the cases pinning a gateway
        # surface's own token shape (_OPENAI_SHAPED_APIS) set it.
        usage = CanonicalUsage(
            model="m", provider=c.get("provider", "p"), api=c.get("api", "native"), **c["counts"]
        )
        b = compute_cost(usage, price, Decimal(c["markup"]))
        assert b.base == c["base"], f"{c['name']}: base {b.base} != {c['base']}"
        assert b.total == c["total"], f"{c['name']}: total {b.total} != {c['total']}"
        assert b.total_cents == c["total_cents"], f"{c['name']}: cents {b.total_cents} != {c['total_cents']}"


def test_money_golden_precomputed_cases() -> None:
    """The gateway path: a lump sum the caller already knows.

    Several of these are verbatim `cost` values from real Cloudflare AI Gateway
    log entries. JS renders any number below 1e-6 in exponential notation, so
    these are the cases where the two repos silently disagreed on real money.
    """
    cases = json.loads((FIXTURES / "money_golden.json").read_text())["precomputed_cases"]
    for c in cases:
        b = compute_precomputed_cost(c["usd_cost"], Decimal(c["markup"]))
        assert b.base == c["base"], f"{c['name']}: base {b.base} != {c['base']}"
        assert b.total == c["total"], f"{c['name']}: total {b.total} != {c['total']}"
        assert b.total_cents == c["total_cents"], f"{c['name']}: cents {b.total_cents} != {c['total_cents']}"


def test_workers_ai_cache_read_is_subtracted_from_input() -> None:
    """Regression: Workers AI is reached only through Cloudflare's OpenAI-COMPATIBLE
    endpoint, so its `prompt_tokens` already includes the cached tokens. Counts and
    rates here are real — a live cached call reported prompt=23233/cached=23168, and
    @cf/moonshotai/kimi-k2.6 lists input $0.95/M with cached input $0.16/M. Billing
    all 23233 at the input rate charged the cached portion twice (+583%).
    """
    price = ModelPrice(
        source="cloudflare_workers_ai",
        input=Decimal("0.00000095"),
        cache_read=Decimal("0.00000016"),
    )
    usage = CanonicalUsage(
        model="@cf/moonshotai/kimi-k2.6",
        provider="workers-ai",
        api="chat.completions",
        input=23233,
        cache_read=23168,
    )
    b = compute_cost(usage, price, Decimal("1"))
    # only the 65 uncached tokens are billed at the input rate
    assert b.fields["input"]["tokens"] == "65"
    assert b.fields["cache_read"]["tokens"] == "23168"
    assert b.total == "0.00376863"


def test_anthropic_cache_read_stays_additive() -> None:
    """The other side of the same rule: Anthropic reports input EXCLUSIVE of cache,
    so nothing may be subtracted. Same counts/rates as the workers-ai case above."""
    price = ModelPrice(
        source="openrouter",
        input=Decimal("0.00000095"),
        cache_read=Decimal("0.00000016"),
    )
    usage = CanonicalUsage(
        model="claude-x", provider="anthropic", api="native", input=23233, cache_read=23168
    )
    b = compute_cost(usage, price, Decimal("1"))
    assert b.fields["input"]["tokens"] == "23233"
    assert b.total == "0.02577823"


def test_parse_price_accepts_exponential_notation() -> None:
    """Real gateway costs below 1e-6 arrive in exponential form. Python has always
    handled these; the golden fixture pins JS to the same values."""
    assert _parse_price(9.807224944233895e-07) == Decimal("0.000000980722")
    assert _parse_price("8.91e-7") == Decimal("0.000000891")
    assert _parse_price("9.78e-07") == Decimal("0.000000978")
    # below the 12dp floor -> zero, not None (a real but unbillably small amount)
    assert _parse_price(1e-13) == Decimal(0)


def test_parse_price_returns_none_instead_of_raising_on_huge_values() -> None:
    """Regression: `.quantize()` sat outside the try, so any value >= 1e16 raised
    InvalidOperation straight out of this function — past every caller that relies
    on the documented None, and out of compute_precomputed_cost into emit()'s
    catch-all, where the event was dropped as an unknown error rather than taking
    the normal 'no price' path."""
    assert _parse_price("1e15") == Decimal("1000000000000000")
    assert _parse_price("1e16") is None
    assert _parse_price("1e30") is None
    assert _parse_price("1e999999999") is None
    # and the tiny end stays a real zero, not a None
    assert _parse_price("1e-999999999") == Decimal(0)


def test_coerce_markup() -> None:
    assert coerce_markup(1.2) == (Decimal("1.2"), True)
    assert coerce_markup("2") == (Decimal("2"), True)
    assert coerce_markup(0) == (Decimal("1"), False)
    assert coerce_markup(-1) == (Decimal("1"), False)
    assert coerce_markup("nonsense") == (Decimal("1"), False)


# ----------------------------------------------------------------------
# PricingProvider — cache + refresh + non-blocking lookup
# ----------------------------------------------------------------------
def test_provider_cold_lookup_flags_stale_then_refresh_warms() -> None:
    fetcher = StubFetcher(openrouter=parse_openrouter(_OPENROUTER_RAW))
    p = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    # cold: no table yet -> None, and source flagged for refresh
    assert p.lookup("anthropic", "claude-opus-4-8", "native") is None
    assert fetcher.openrouter_calls == 0
    # background worker would call this; we call it directly
    p.maybe_refresh()
    assert fetcher.openrouter_calls == 1
    # now warm
    mp = p.lookup("anthropic", "claude-opus-4-8", "native")
    assert mp is not None and mp.input == Decimal("0.000005")


def test_provider_token_mode_does_no_fetch() -> None:
    fetcher = StubFetcher(openrouter=parse_openrouter(_OPENROUTER_RAW))
    p = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    # No lookups performed -> nothing flagged stale -> refresh is a no-op.
    p.maybe_refresh()
    assert fetcher.openrouter_calls == 0


def test_provider_cloudflare_workers_ai_cold_then_warm() -> None:
    cf_table = parse_cloudflare_workers_ai(_CLOUDFLARE_MODELS_RAW)
    fetcher = StubFetcher(cloudflare_workers_ai=cf_table)
    p = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    # cold: no table yet -> None, and flags it for refresh
    assert p.lookup("workers-ai", "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "cloudflare_gateway") is None
    assert fetcher.cloudflare_workers_ai_calls == 0
    p.maybe_refresh()
    assert fetcher.cloudflare_workers_ai_calls == 1
    mp = p.lookup("workers-ai", "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "cloudflare_gateway")
    assert mp is not None and mp.input == Decimal("0.000000293")


def test_provider_cloudflare_workers_ai_only_fetched_for_workers_ai_provider() -> None:
    """A lookup for a totally different provider must not flag the Cloudflare
    source stale — each source only ever fetches for the traffic that needs it."""
    fetcher = StubFetcher(openrouter=parse_openrouter(_OPENROUTER_RAW))
    p = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    p.lookup("anthropic", "claude-opus-4-8", "native")
    p.maybe_refresh()
    assert fetcher.cloudflare_workers_ai_calls == 0


def test_provider_mistral_alias_cold_miss_then_warm_resolves() -> None:
    """Cold: the alias table hasn't been fetched yet, so the raw alias string
    is looked up against OpenRouter directly and misses safely — never worse
    than before this resolution step existed. Warm: it resolves and hits."""
    fetcher = StubFetcher(
        openrouter=parse_openrouter(_OPENROUTER_RAW),
        mistral_aliases=parse_mistral_aliases(_MISTRAL_MODELS_RAW),
    )
    p = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    # cold: openrouter table is ALSO cold here, so this exercises both misses
    # at once — the important thing is it's a clean None, not an exception.
    assert p.lookup("mistral", "mistral-small-latest", "native") is None
    p.maybe_refresh()
    assert fetcher.mistral_aliases_calls == 1
    assert fetcher.openrouter_calls == 1
    mp = p.lookup("mistral", "mistral-small-latest", "native")
    assert mp is not None
    assert mp.input == Decimal("0.00000015")
    assert mp.output == Decimal("0.0000006")


def test_learn_mistral_api_key_is_used_on_next_fetch() -> None:
    """No LagoConfig.mistral_api_key was ever configured — the key is
    learned instead (e.g. from a wrapped client) and still reaches the
    fetcher on the next refresh."""
    fetcher = StubFetcher(mistral_aliases=parse_mistral_aliases(_MISTRAL_MODELS_RAW))
    p = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    p.learn_mistral_api_key("learned-from-client-key")
    p.prime(providers=["mistral"])
    p.maybe_refresh()
    assert fetcher.mistral_aliases_calls == 1
    assert fetcher.last_mistral_api_key == "learned-from-client-key"


def test_learn_mistral_api_key_does_not_overwrite_an_already_learned_key() -> None:
    """First-learned key wins — a second call (e.g. wrap() invoked again for
    a second client) doesn't clobber it."""
    fetcher = StubFetcher(mistral_aliases=parse_mistral_aliases(_MISTRAL_MODELS_RAW))
    p = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    p.learn_mistral_api_key("first-key")
    p.learn_mistral_api_key("second-key")
    p.prime(providers=["mistral"])
    p.maybe_refresh()
    assert fetcher.last_mistral_api_key == "first-key"


def test_learn_mistral_api_key_ignores_empty_string() -> None:
    fetcher = StubFetcher(mistral_aliases=parse_mistral_aliases(_MISTRAL_MODELS_RAW))
    p = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    p.learn_mistral_api_key("")
    p.prime(providers=["mistral"])
    p.maybe_refresh()
    assert fetcher.last_mistral_api_key is None


def test_provider_mistral_lookup_without_credentials_falls_back_to_raw_model() -> None:
    """No Mistral API key configured -> fetch_mistral_aliases returns {} ->
    the alias string is looked up as-is against OpenRouter, same behavior as
    before this feature existed (a safe miss for an alias, a hit for a
    non-aliased model like "mistral-large" that's already the exact id)."""
    fetcher = StubFetcher(openrouter=parse_openrouter(_OPENROUTER_RAW))  # mistral_aliases defaults to {}
    p = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    p.lookup("mistral", "mistral-large", "native")
    p.maybe_refresh()
    mp = p.lookup("mistral", "mistral-large", "native")
    assert mp is not None and mp.input == Decimal("0.000002")


def test_provider_mistral_alias_only_fetched_for_mistral_provider() -> None:
    """A lookup for a totally different provider must not flag the Mistral
    alias source stale — each source only ever fetches for the traffic that
    needs it."""
    fetcher = StubFetcher(openrouter=parse_openrouter(_OPENROUTER_RAW))
    p = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    p.lookup("anthropic", "claude-opus-4-8", "native")
    p.maybe_refresh()
    assert fetcher.mistral_aliases_calls == 0
    assert fetcher.openrouter_calls == 1


def test_prime_only_eagerly_warms_openrouter_not_cloudflare_or_mistral() -> None:
    """prime() (called automatically when pricing_mode="price" is the global
    default, and by warm_pricing()) must not force-fetch Cloudflare/Mistral —
    both are credential-gated and provider-specific, and most price-mode
    customers never call either. Eagerly hitting their APIs at construction
    time regardless of actual usage would be pure waste. Only a real lookup
    for that specific provider should ever trigger their fetch."""
    fetcher = StubFetcher(
        openrouter=parse_openrouter(_OPENROUTER_RAW),
        cloudflare_workers_ai=parse_cloudflare_workers_ai(_CLOUDFLARE_MODELS_RAW),
        mistral_aliases=parse_mistral_aliases(_MISTRAL_MODELS_RAW),
    )
    p = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    p.prime()
    p.maybe_refresh()
    assert fetcher.openrouter_calls == 1
    assert fetcher.cloudflare_workers_ai_calls == 0
    assert fetcher.mistral_aliases_calls == 0
    # Confirms it's not just "hasn't fetched yet" — a real lookup for either
    # provider afterward still works, fetching lazily on its own trigger.
    assert p.lookup("workers-ai", "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "cloudflare_gateway") is None
    p.maybe_refresh()
    assert fetcher.cloudflare_workers_ai_calls == 1
    mp = p.lookup("workers-ai", "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "cloudflare_gateway")
    assert mp is not None


def test_prime_with_providers_eagerly_warms_the_named_ones_too() -> None:
    """Opt-in escape hatch: a caller who already knows they're about to call
    Mistral and/or Workers AI this session can say so up front and skip the
    one-time lazy cold-start cost for THAT provider's first call too —
    without going back to unconditionally warming both for every customer."""
    fetcher = StubFetcher(
        openrouter=parse_openrouter(_OPENROUTER_RAW),
        cloudflare_workers_ai=parse_cloudflare_workers_ai(_CLOUDFLARE_MODELS_RAW),
        mistral_aliases=parse_mistral_aliases(_MISTRAL_MODELS_RAW),
    )
    p = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    p.prime(providers=["mistral", "workers-ai"])
    p.maybe_refresh()
    assert fetcher.openrouter_calls == 1
    assert fetcher.cloudflare_workers_ai_calls == 1
    assert fetcher.mistral_aliases_calls == 1
    # Both now resolve correctly on their very first real lookup — no cold miss.
    assert p.lookup("mistral", "mistral-small-latest", "native") is not None
    assert (
        p.lookup("workers-ai", "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "cloudflare_gateway") is not None
    )


def test_prime_with_unknown_provider_name_is_ignored_not_an_error() -> None:
    """This is a hint, not a contract — a typo'd or unrecognized provider
    name is silently ignored rather than raising."""
    fetcher = StubFetcher(openrouter=parse_openrouter(_OPENROUTER_RAW))
    p = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    p.prime(providers=["totally-made-up-provider"])
    p.maybe_refresh()
    assert fetcher.openrouter_calls == 1
    assert fetcher.cloudflare_workers_ai_calls == 0
    assert fetcher.mistral_aliases_calls == 0


def test_warm_pricing_with_providers_threads_through_from_sdk() -> None:
    """Same opt-in escape hatch, exercised through LagoSDK.warm_pricing()
    rather than the PricingProvider directly."""
    fetcher = StubFetcher(
        openrouter=parse_openrouter(_OPENROUTER_RAW),
        mistral_aliases=parse_mistral_aliases(_MISTRAL_MODELS_RAW),
    )
    provider = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    cfg = LagoConfig(
        api_key="dummy",
        default_subscription_id="sub_default",
        pricing_mode="price",
        pricing_provider=provider,
    )
    sdk = LagoSDK(api_key="dummy", config=cfg)
    sdk.warm_pricing(providers=["mistral"])
    assert fetcher.mistral_aliases_calls == 1
    assert provider.lookup("mistral", "mistral-small-latest", "native") is not None


def test_provider_bedrock_region_routing() -> None:
    bedrock_table = parse_bedrock_offer(
        {
            "products": {
                "S": {
                    "attributes": {
                        "model": "Claude Sonnet 4.6",
                        "usagetype": "Input-Tokens",
                        "unit": "tokens",
                    }
                }
            },
            "terms": {
                "OnDemand": {
                    "S": {
                        "o": {
                            "priceDimensions": {"d": {"pricePerUnit": {"USD": "0.000003"}, "unit": "tokens"}}
                        }
                    }
                }
            },
        },
        "eu-west-1",
    )
    fetcher = StubFetcher(bedrock={"eu-west-1": bedrock_table})
    p = PricingProvider(fetcher=fetcher, ttl_seconds=3600, default_region="us-east-1")
    assert p.lookup("anthropic", "eu.anthropic.claude-sonnet-4-6", "bedrock_converse") is None
    p.maybe_refresh()
    assert fetcher.bedrock_calls == ["eu-west-1"]
    mp = p.lookup("anthropic", "eu.anthropic.claude-sonnet-4-6", "bedrock_converse")
    assert mp is not None and mp.input == Decimal("0.000003")


# ----------------------------------------------------------------------
# SDK price mode integration
# ----------------------------------------------------------------------
def _warm_provider() -> PricingProvider:
    p = PricingProvider(fetcher=StubFetcher(openrouter=parse_openrouter(_OPENROUTER_RAW)), ttl_seconds=3600)
    p.prime()
    p.maybe_refresh()  # warm synchronously for deterministic tests
    return p


def _warm_cloudflare_provider() -> PricingProvider:
    cf_table = parse_cloudflare_workers_ai(_CLOUDFLARE_MODELS_RAW)
    p = PricingProvider(fetcher=StubFetcher(cloudflare_workers_ai=cf_table), ttl_seconds=3600)
    # prime() no longer eagerly warms Cloudflare (it's credential-gated and
    # provider-specific — see prime()'s docstring) — a real first lookup for
    # this provider is what flags it stale, same as production.
    p.lookup("workers-ai", "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "cloudflare_gateway")
    p.maybe_refresh()
    return p


def _price_sdk(
    provider: PricingProvider, default_sub: str = "sub_default", on_error=None, markup: float = 1.0
):
    received: list = []
    cfg = LagoConfig(
        api_key="dummy",
        default_subscription_id=default_sub,
        pricing_mode="price",
        markup=markup,
        pricing_provider=provider,
        on_error=on_error,
    )
    sdk = LagoSDK(api_key="dummy", config=cfg)
    sdk._queue._sender = lambda b: received.append(list(b))  # type: ignore[attr-defined]
    return sdk, received


def _by_token_type(received: list) -> dict[str, dict]:
    flat = [e for batch in received for e in batch]
    assert all(e["code"] == "llm_cost" for e in flat)
    return {e["properties"]["token_type"]: e for e in flat}


def test_warm_pricing_closes_the_cold_start_race() -> None:
    """Without warm_pricing(), a call made immediately after construction hits
    a cold table and emit() falls back to token events (or, with no token
    metric configured, loses the event entirely). warm_pricing() blocks until
    the table is fetched, so the very first call in price mode prices
    correctly instead of racing the background thread's first tick."""
    fetcher = StubFetcher(openrouter=parse_openrouter(_OPENROUTER_RAW))
    provider = PricingProvider(fetcher=fetcher, ttl_seconds=3600)
    cfg = LagoConfig(
        api_key="dummy",
        default_subscription_id="sub_default",
        pricing_mode="price",
        pricing_provider=provider,
    )
    sdk = LagoSDK(api_key="dummy", config=cfg)
    received: list = []
    sdk._queue._sender = lambda b: received.append(list(b))  # type: ignore[attr-defined]
    try:
        assert provider.lookup("anthropic", "claude-opus-4-8", "native") is None  # genuinely cold

        sdk.warm_pricing()

        assert provider.lookup("anthropic", "claude-opus-4-8", "native") is not None  # now warm
        u = CanonicalUsage(
            input=1000, output=500, model="claude-opus-4-8", provider="anthropic", api="native"
        )
        sdk.emit(u)
        assert sdk.flush(timeout=2.0)
    finally:
        sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert all(e["code"] == "llm_cost" for e in flat)  # priced, not a token-event fallback


def test_bad_markup_is_coerced_to_one_reported_and_still_bills_the_cost() -> None:
    """`markup` is customer input (`extra_lago={"markup": ...}`), so a comma decimal
    like "1,5" genuinely arrives. `coerce_markup` is the guard that catches it; this
    pins the end-to-end consequence, which no test covered: the cost is still billed
    at 1.0 rather than zeroed, and the lost markup reaches on_error."""
    seen: list[tuple[Exception, str]] = []
    sdk, received = _price_sdk(_warm_provider(), on_error=lambda e, c: seen.append((e, c)))
    try:
        u = CanonicalUsage(
            input=1000, output=500, model="claude-opus-4-8", provider="anthropic", api="native"
        )
        # annotated `float | None`, but it arrives from untyped customer input
        sdk.emit(u, markup="1,5")  # type: ignore[arg-type]
        assert sdk.flush(timeout=2.0)
    finally:
        sdk.shutdown(timeout=1.0)

    events = _by_token_type(received)
    assert events, "a bad markup must not drop the cost events"
    for token_type, ev in events.items():
        props = ev["properties"]
        assert props["markup"] == "1", f"{token_type}: bad markup should coerce to 1.0"
        assert props["value"] == props["base_cost"], (
            f"{token_type}: should bill the un-marked-up cost, not {props['value']!r}"
        )
        assert Decimal(props["value"]) > 0, f"{token_type}: a bad markup must not zero the bill"

    contexts = [c for _, c in seen]
    assert "pricing" in contexts, f"the invalid markup must reach on_error; got {contexts}"
    assert any("markup" in str(e) and "1,5" in str(e) for e, _ in seen)


def test_price_mode_emits_one_event_per_token_type() -> None:
    """A real per-field breakdown (OpenRouter has both input/output prices for
    this model) splits into one llm_cost event per token_type, so Lago's
    `grouped_by: ["model", "token_type"]` charge can break it down by both —
    not one summed event that hides the split."""
    sdk, received = _price_sdk(_warm_provider())
    u = CanonicalUsage(input=1000, output=500, model="claude-opus-4-8", provider="anthropic", api="native")
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_type = _by_token_type(received)
    assert set(by_type) == {"input", "output"}

    inp = by_type["input"]
    assert inp["properties"]["unit"] == "1000"
    assert inp["properties"]["value"] == "0.005"  # 1000 * 0.000005
    assert inp["properties"]["unit_price"] == "0.000005"
    assert inp["properties"]["model"] == "claude-opus-4-8"
    assert inp["properties"]["price_source"] == "openrouter"
    # Lago dynamic charge cents = 0.005 USD * 100 = 0.5
    assert inp["precise_total_amount_cents"] == "0.5"

    out = by_type["output"]
    assert out["properties"]["unit"] == "500"
    assert out["properties"]["value"] == "0.0125"  # 500 * 0.000025
    assert out["precise_total_amount_cents"] == "1.25"

    # Same call's split transaction ids don't collide with each other.
    assert inp["transaction_id"] != out["transaction_id"]


def test_price_mode_workers_ai_uses_cloudflare_catalog_not_openrouter() -> None:
    """Real captured shape: 38 input / 2 output tokens through
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast" — same call this catalog price
    was verified against live (predicted $0.00001564 vs Cloudflare's own
    real-charged $0.00001552; the ~0.8% gap is the catalog's own displayed
    rate rounding to 3dp, not our computation)."""
    sdk, received = _price_sdk(_warm_cloudflare_provider())
    u = CanonicalUsage(
        input=38,
        output=2,
        model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        provider="workers-ai",
        api="cloudflare_gateway",
    )
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_type = _by_token_type(received)
    assert by_type["input"]["properties"]["price_source"] == "cloudflare_workers_ai"
    # 38 * 0.000000293 + 2 * 0.000002253 = 0.000011134 + 0.000004506 = 0.00001564
    assert by_type["input"]["properties"]["value"] == "0.000011134"
    assert by_type["output"]["properties"]["value"] == "0.000004506"


def test_price_mode_markup_scales_each_token_type_event() -> None:
    sdk, received = _price_sdk(_warm_provider(), markup=2.0)
    u = CanonicalUsage(input=1000, output=500, model="claude-opus-4-8", provider="anthropic", api="native")
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_type = _by_token_type(received)
    assert by_type["input"]["properties"]["base_cost"] == "0.005"
    assert by_type["input"]["properties"]["value"] == "0.01"  # 0.005 * 2
    assert by_type["input"]["properties"]["markup"] == "2"
    assert by_type["output"]["properties"]["value"] == "0.025"  # 0.0125 * 2


def test_per_call_markup_overrides_global() -> None:
    sdk, received = _price_sdk(_warm_provider(), markup=1.0)
    u = CanonicalUsage(input=1000, output=500, model="claude-opus-4-8", provider="anthropic", api="native")
    sdk.emit(u, markup=3.0)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_type = _by_token_type(received)
    assert by_type["input"]["properties"]["value"] == "0.015"  # 0.005 * 3
    assert by_type["output"]["properties"]["value"] == "0.0375"  # 0.0125 * 3


# ----------------------------------------------------------------------
# Subset semantics: some providers report `input` INCLUSIVE of cache_read and
# `output` INCLUSIVE of reasoning. Pricing the parent at full count AND the
# subset separately would double-bill — these tests lock the de-overlap.
# ----------------------------------------------------------------------
def test_price_mode_openai_cache_read_subset_not_double_billed() -> None:
    sdk, received = _price_sdk(_warm_provider())
    # OpenAI: input (prompt_tokens)=1000 ALREADY includes cache_read=800.
    u = CanonicalUsage(
        input=1000, output=500, cache_read=800, model="gpt-4o", provider="openai", api="native"
    )
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_type = _by_token_type(received)
    assert set(by_type) == {"input", "cache_read", "output"}
    # input billed for only the non-cached portion (1000 - 800); cache billed at cache rate
    assert by_type["input"]["properties"]["unit"] == "200"
    assert by_type["cache_read"]["properties"]["unit"] == "800"
    # 200*0.0000025=0.0005, 800*0.00000125=0.001, 500*0.00001=0.005 (the bug would bill input at full 1000)
    assert by_type["input"]["properties"]["value"] == "0.0005"
    assert by_type["cache_read"]["properties"]["value"] == "0.001"
    assert by_type["output"]["properties"]["value"] == "0.005"


def test_price_mode_gemini_cache_subset_and_reasoning_additive() -> None:
    sdk, received = _price_sdk(_warm_provider())
    # Gemini: input=1000 INCLUDES cache_read=300; reasoning(thoughts)=100 is ADDITIVE.
    u = CanonicalUsage(
        input=1000,
        output=400,
        cache_read=300,
        reasoning=100,
        model="gemini-2.5-flash",
        provider="gemini",
        api="native",
    )
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_type = _by_token_type(received)
    assert set(by_type) == {"input", "cache_read", "output", "reasoning"}
    assert by_type["input"]["properties"]["unit"] == "700"  # 1000 - 300 cached
    assert by_type["cache_read"]["properties"]["unit"] == "300"
    assert by_type["output"]["properties"]["unit"] == "400"
    assert by_type["reasoning"]["properties"]["unit"] == "100"  # billed separately (additive for Gemini)
    # 700*3e-7=0.00021, 300*7.5e-8=0.0000225, 400*2.5e-6=0.001, 100*2.5e-6=0.00025
    assert by_type["input"]["properties"]["value"] == "0.00021"
    assert by_type["cache_read"]["properties"]["value"] == "0.0000225"
    assert by_type["output"]["properties"]["value"] == "0.001"
    assert by_type["reasoning"]["properties"]["value"] == "0.00025"


def test_price_mode_openai_reasoning_in_output_not_double_billed() -> None:
    sdk, received = _price_sdk(_warm_provider())
    # OpenAI o-series: output (completion_tokens)=500 ALREADY includes reasoning=200.
    u = CanonicalUsage(input=100, output=500, reasoning=200, model="gpt-4o", provider="openai", api="native")
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_type = _by_token_type(received)
    # reasoning folded into output — no separate reasoning event, output billed in full
    assert set(by_type) == {"input", "output"}
    assert by_type["output"]["properties"]["unit"] == "500"
    # 100*0.0000025=0.00025, 500*0.00001=0.005 (bug would add a separate 200*1e-5=0.002 reasoning event)
    assert by_type["input"]["properties"]["value"] == "0.00025"
    assert by_type["output"]["properties"]["value"] == "0.005"


def test_price_mode_anthropic_cache_is_additive() -> None:
    sdk, received = _price_sdk(_warm_provider())
    # Anthropic: input EXCLUDES cache; cache_read/cache_write are additive (no subtraction).
    u = CanonicalUsage(
        input=1000,
        output=500,
        cache_read=400,
        cache_write=200,
        model="claude-opus-4-8",
        provider="anthropic",
        api="native",
    )
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_type = _by_token_type(received)
    assert set(by_type) == {"input", "output", "cache_read", "cache_write"}
    assert by_type["input"]["properties"]["unit"] == "1000"  # unchanged — additive provider
    assert by_type["cache_read"]["properties"]["unit"] == "400"
    assert by_type["cache_write"]["properties"]["unit"] == "200"
    # 1000*5e-6=0.005, 500*25e-6=0.0125, 400*5e-7=0.0002, 200*6.25e-6=0.00125
    assert by_type["input"]["properties"]["value"] == "0.005"
    assert by_type["output"]["properties"]["value"] == "0.0125"
    assert by_type["cache_read"]["properties"]["value"] == "0.0002"
    assert by_type["cache_write"]["properties"]["value"] == "0.00125"


# ----------------------------------------------------------------------
# usd_cost — the gateway-connector entrypoint: skip our own price lookup
# entirely and bill the caller's already-known real cost.
# ----------------------------------------------------------------------
def test_usd_cost_skips_pricing_lookup_entirely() -> None:
    """A COLD, never-warmed provider — if this passed, `emit` would have had
    to fall back to token events (no price available). It doesn't: usd_cost
    bypasses `_pricing.lookup` altogether, so a cold provider is irrelevant."""
    cold_provider = PricingProvider(fetcher=StubFetcher(openrouter={}), ttl_seconds=3600)
    sdk, received = _price_sdk(cold_provider)
    u = CanonicalUsage(
        input=38, output=41, model="@cf/meta/llama-3.3-70b", provider="workers-ai", api="cloudflare_gateway"
    )
    sdk.emit(u, usd_cost=0.00010472)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert len(flat) == 1
    ev = flat[0]
    assert ev["code"] == "llm_cost"
    assert ev["precise_total_amount_cents"] == "0.010472"
    props = ev["properties"]
    assert props["price_source"] == "precomputed"
    assert props["value"] == "0.00010472"
    # No per-field breakdown available — unit falls back to raw input+output.
    assert props["unit"] == "79"
    assert "input_tokens" not in props


def test_usd_cost_applies_markup_same_as_looked_up_price() -> None:
    cold_provider = PricingProvider(fetcher=StubFetcher(openrouter={}), ttl_seconds=3600)
    sdk, received = _price_sdk(cold_provider, markup=1.5)
    u = CanonicalUsage(input=10, output=5, model="m", provider="workers-ai", api="cloudflare_gateway")
    sdk.emit(u, usd_cost=0.0001)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    ev = [e for batch in received for e in batch][0]
    assert ev["properties"]["base_cost"] == "0.0001"
    assert ev["properties"]["value"] == "0.00015"


def test_usd_cost_ignored_in_token_mode() -> None:
    """usd_cost is a price-mode-only override — in the default token mode it
    must not do anything; the call still emits ordinary token events."""
    cold_provider = PricingProvider(fetcher=StubFetcher(openrouter={}), ttl_seconds=3600)
    received: list = []
    cfg = LagoConfig(api_key="dummy", default_subscription_id="sub_default", pricing_provider=cold_provider)
    sdk = LagoSDK(api_key="dummy", config=cfg)
    sdk._queue._sender = lambda b: received.append(list(b))  # type: ignore[attr-defined]
    u = CanonicalUsage(input=10, output=5, model="m", provider="workers-ai", api="cloudflare_gateway")
    sdk.emit(u, usd_cost=0.0001)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    codes = {e["code"] for e in flat}
    assert codes == {"llm_input_tokens", "llm_output_tokens"}


def test_event_id_used_as_transaction_id_in_price_mode() -> None:
    """The connector's idempotency key: pass the source log entry's own id so
    re-running a backfill over the same window doesn't double-bill."""
    cold_provider = PricingProvider(fetcher=StubFetcher(openrouter={}), ttl_seconds=3600)
    sdk, received = _price_sdk(cold_provider)
    u = CanonicalUsage(input=10, output=5, model="m", provider="workers-ai", api="cloudflare_gateway")
    sdk.emit(u, usd_cost=0.0001, event_id="backfill_01ABC")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    ev = [e for batch in received for e in batch][0]
    assert ev["transaction_id"] == "backfill_01ABC"


def test_event_id_suffixed_per_field_in_token_mode() -> None:
    """Token mode can push several events from one call (input, output, ...);
    reusing the same event_id verbatim for all of them would collide, so each
    field gets its own suffix off the same base id — in the `_tok_` namespace,
    which keeps it distinct from the cost path's suffix for the same field."""
    received: list = []
    cfg = LagoConfig(api_key="dummy", default_subscription_id="sub_default")
    sdk = LagoSDK(api_key="dummy", config=cfg)
    sdk._queue._sender = lambda b: received.append(list(b))  # type: ignore[attr-defined]
    u = CanonicalUsage(input=10, output=5, model="m", provider="workers-ai", api="cloudflare_gateway")
    sdk.emit(u, event_id="backfill_01ABC")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    ids = {e["transaction_id"] for e in flat}
    assert ids == {"backfill_01ABC_tok_input", "backfill_01ABC_tok_output"}


def test_token_fallback_and_cost_ids_never_collide_for_one_event_id() -> None:
    """The bug this namespacing exists for.

    A price miss falls back to token events; the SAME window re-run once the
    table is warm takes the cost path. Under one shared namespace both emitted
    `{event_id}_input`, so Lago rejected the second as a duplicate — and since
    `/events/batch` is all-or-nothing, that rejection failed every other event
    in the batch too. The dollar amounts for that window were never billed,
    only the raw token counts, and nothing surfaced it.
    """
    u = CanonicalUsage(input=10, output=5, model="claude-opus-4-8", provider="anthropic", api="native")

    # Run 1: cold table -> price miss -> token fallback, same event_id.
    cold = PricingProvider(fetcher=StubFetcher(openrouter={}), ttl_seconds=3600)
    sdk_cold, got_cold = _price_sdk(cold)
    sdk_cold.emit(u, event_id="backfill_01ABC")
    assert sdk_cold.flush(timeout=2.0)
    sdk_cold.shutdown(timeout=1.0)
    cold_ids = {e["transaction_id"] for batch in got_cold for e in batch}

    # Run 2: warm table -> real per-field cost events, same event_id.
    sdk_warm, got_warm = _price_sdk(_warm_provider())
    sdk_warm.emit(u, event_id="backfill_01ABC")
    assert sdk_warm.flush(timeout=2.0)
    sdk_warm.shutdown(timeout=1.0)
    warm_ids = {e["transaction_id"] for batch in got_warm for e in batch}

    assert cold_ids, "cold run should have emitted token events"
    assert warm_ids, "warm run should have emitted cost events"
    assert not (cold_ids & warm_ids), (
        f"token-fallback and cost ids must not collide; overlap={cold_ids & warm_ids}"
    )
    assert all("_tok_" in i for i in cold_ids)
    assert all("_cost_" in i for i in warm_ids)


def test_no_event_id_still_falls_back_to_random_uuid() -> None:
    """A live, one-shot call has no natural id to reuse — must still work
    exactly as before this option existed."""
    cold_provider = PricingProvider(fetcher=StubFetcher(openrouter={}), ttl_seconds=3600)
    sdk, received = _price_sdk(cold_provider)
    u = CanonicalUsage(input=10, output=5, model="m", provider="workers-ai", api="cloudflare_gateway")
    sdk.emit(u, usd_cost=0.0001)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    ev = [e for batch in received for e in batch][0]
    uuid.UUID(ev["transaction_id"])  # raises if not a valid UUID


def test_price_unavailable_falls_back_to_token_events_and_reports() -> None:
    errors: list = []
    # warm provider but ask for an unknown model -> price None -> fallback
    sdk, received = _price_sdk(
        _warm_provider(), on_error=lambda exc, where: errors.append((type(exc).__name__, where))
    )
    u = CanonicalUsage(input=10, output=20, model="unknown-model-xyz", provider="anthropic", api="native")
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    codes = {e["code"] for e in flat}
    assert codes == {"llm_input_tokens", "llm_output_tokens"}  # token fallback
    assert any(name == "PricingUnavailableError" and where == "pricing" for name, where in errors)


def test_token_billed_provider_emits_tokens_without_reporting_an_error() -> None:
    """A Databricks-hosted model has no per-token rate anywhere — not a cold table, not
    an unmatched name, none exists. So token counts are the complete answer, and calling
    that a failure on every request trains the reader to ignore on_error entirely."""
    errors: list = []
    sdk, received = _price_sdk(
        _warm_provider(), on_error=lambda exc, where: errors.append((type(exc).__name__, where))
    )
    u = CanonicalUsage(
        input=11,
        output=4,
        model="meta-llama-4-maverick-040225",
        provider="databricks",
        api="chat_completions",
    )
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert {e["code"] for e in flat} == {"llm_input_tokens", "llm_output_tokens"}
    assert [e["properties"]["value"] for e in flat if e["code"] == "llm_input_tokens"] == ["11"]
    assert errors == []


def test_a_real_price_miss_still_reports() -> None:
    """The narrow exception above must not become a blanket silence: an unmatched model
    on a provider that DOES publish rates is a genuine miss the customer can act on."""
    errors: list = []
    sdk, received = _price_sdk(
        _warm_provider(), on_error=lambda exc, where: errors.append((type(exc).__name__, where))
    )
    sdk.emit(CanonicalUsage(input=5, model="no-such-model", provider="anthropic", api="native"))
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert any(n == "PricingUnavailableError" and w == "pricing" for n, w in errors)


def test_token_billed_note_is_logged_once_per_model(caplog) -> None:
    """It is a standing fact about the provider, not an event about this call."""
    import logging as _logging

    sdk, _ = _price_sdk(_warm_provider())
    with caplog.at_level(_logging.INFO, logger="lago_agent_sdk"):
        for _ in range(3):
            sdk.emit(CanonicalUsage(input=1, model="llama-4-maverick", provider="databricks", api="x"))
        sdk.emit(CanonicalUsage(input=1, model="gpt-oss-20b", provider="databricks", api="x"))
    sdk.shutdown(timeout=1.0)
    notes = [r for r in caplog.records if "in its own units" in r.getMessage()]
    assert len(notes) == 2  # one per distinct model, not one per call
    assert any("llama-4-maverick" in n.getMessage() for n in notes)


def test_byok_through_the_same_gateway_still_prices() -> None:
    """TOKEN_BILLED_PROVIDERS keys on provider, so it covers Databricks-HOSTED models
    only — BYOK traffic through the same gateway is stamped with the real vendor and
    must keep pricing normally."""
    sdk, received = _price_sdk(_warm_provider())
    sdk.emit(
        CanonicalUsage(input=100, output=50, model="claude-opus-4.8", provider="anthropic", api="native")
    )
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert {e["code"] for e in flat} == {"llm_cost"}


def test_per_call_price_mode_overrides_global_tokens() -> None:
    # global mode is tokens (default); per-call asks for price
    provider = _warm_provider()
    received: list = []
    cfg = LagoConfig(api_key="dummy", default_subscription_id="sub_default", pricing_provider=provider)
    sdk = LagoSDK(api_key="dummy", config=cfg)
    sdk._queue._sender = lambda b: received.append(list(b))  # type: ignore[attr-defined]
    u = CanonicalUsage(input=1000, output=500, model="claude-opus-4-8", provider="anthropic", api="native")
    sdk.emit(u, mode="price")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert len(flat) == 2 and all(e["code"] == "llm_cost" for e in flat)  # one per token_type: input, output


def test_default_mode_is_tokens_unchanged() -> None:
    provider = _warm_provider()
    received: list = []
    cfg = LagoConfig(api_key="dummy", default_subscription_id="sub_default", pricing_provider=provider)
    sdk = LagoSDK(api_key="dummy", config=cfg)
    sdk._queue._sender = lambda b: received.append(list(b))  # type: ignore[attr-defined]
    u = CanonicalUsage(input=1000, output=500, model="claude-opus-4-8", provider="anthropic", api="native")
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert {e["code"] for e in flat} == {"llm_input_tokens", "llm_output_tokens"}


# ----------------------------------------------------------------------
# Date-suffix shapes — both vendors' conventions must strip
# ----------------------------------------------------------------------

# OpenRouter lists BARE ids for the current OpenAI lineup; the API returns dated
# ones. `resolve_model` prefers the response's own name, so the dated form is what
# reaches lookup.
_BARE_OPENAI_TABLE = parse_openrouter(
    {
        "data": [
            {"id": f"openai/{m}", "pricing": {"prompt": "0.000001", "completion": "0.000002"}}
            for m in ("gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-mini", "o3", "o4-mini")
        ]
    }
)


@pytest.mark.parametrize(
    "dated",
    [
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini-2025-04-14",
        "gpt-5-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "o3-2025-04-16",
        "o4-mini-2025-04-16",
    ],
)
def test_openai_hyphenated_date_suffix_strips_to_a_hit(dated: str) -> None:
    """OpenAI stamps HYPHENATED dates ("gpt-5-2025-08-07"), Anthropic COMPACT ones
    ("claude-sonnet-4-5-20250929"). Handling only the compact shape silently broke
    price mode for every current OpenAI model — all six of these missed and fell
    back to token events. Verified against the live 400-model OpenRouter table
    before and after.
    """
    assert lookup_openrouter(_BARE_OPENAI_TABLE, "openai", dated) is not None


@pytest.mark.parametrize(
    "dated,bare",
    [
        ("claude-sonnet-4-5-20250929", "anthropic/claude-sonnet-4.5"),
        ("claude-haiku-4-5-20251001", "anthropic/claude-haiku-4.5"),
        ("claude-opus-4-5-20251101", "anthropic/claude-opus-4.5"),
    ],
)
def test_anthropic_compact_date_suffix_still_strips(dated: str, bare: str) -> None:
    """Regression guard: widening the pattern must not break the compact form."""
    table = parse_openrouter({"data": [{"id": bare, "pricing": {"prompt": "0.000003"}}]})
    assert lookup_openrouter(table, "anthropic", dated) is not None


def test_non_date_suffix_is_not_stripped() -> None:
    """`gpt-5.6-sol` resolves with a `-sol` suffix that is neither a date nor a
    version tag. It must be left intact — OpenRouter lists it verbatim as
    "openai/gpt-5.6-sol", so stripping would turn a hit into a miss."""
    assert _strip_version("gpt-5.6-sol") == "gpt-5.6-sol"
    table = parse_openrouter({"data": [{"id": "openai/gpt-5.6-sol", "pricing": {"prompt": "0.000005"}}]})
    assert lookup_openrouter(table, "openai", "gpt-5.6-sol") is not None


def test_workers_ai_model_names_are_never_date_stripped() -> None:
    """Workers AI ids carry dotted versions and fp8 suffixes, not dates. The
    widened pattern must leave them untouched or the Cloudflare catalog lookup
    breaks."""
    for m in (
        "@cf/meta/llama-3.2-1b-instruct",
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "@cf/moonshotai/kimi-k2.7-code",
    ):
        assert _strip_version(m) == m


def test_workers_ai_compat_prefix_is_defined_exactly_once() -> None:
    """Two unrelated layers must agree on this string: `adapters/openai_native`
    decides the PROVIDER from it, `pricing` strips it before a catalog lookup. They
    must never import each other, so it lives in `canonical`. A drift between two
    copies is a silently unpriced call, not a crash — which is why this is asserted
    rather than left to review."""
    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "lago_agent_sdk"
    definitions = [
        f"{path.relative_to(src)}:{i}"
        for path in src.rglob("*.py")
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.match(r"\s*_?WORKERS_AI_COMPAT_PREFIX\s*=", line)
    ]
    assert definitions == ["canonical.py:19"] or len(definitions) == 1, (
        f"expected one definition, found {definitions}"
    )
    assert WORKERS_AI_COMPAT_PREFIX == "workers-ai/"

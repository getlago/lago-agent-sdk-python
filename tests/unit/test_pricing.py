"""Pricing tests — matching, money math, provider cache, and SDK price mode."""

from __future__ import annotations

import json
import pathlib
import uuid
from decimal import Decimal
from typing import Any

import pytest

from lago_agent_sdk import CanonicalUsage, LagoConfig, LagoSDK, ModelPrice
from lago_agent_sdk.pricing import (
    HttpPricingFetcher,
    PricingProvider,
    bedrock_model_key,
    coerce_markup,
    compute_cost,
    compute_precomputed_cost,
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


def test_openrouter_miss_returns_none() -> None:
    table = parse_openrouter(_OPENROUTER_RAW)
    assert lookup_openrouter(table, "anthropic", "totally-made-up-model") is None
    # vendor-gated: right model name, wrong vendor -> miss
    assert lookup_openrouter(table, "openai", "claude-opus-4-8") is None


# ----------------------------------------------------------------------
# Cloudflare Workers AI parsing + matching
# ----------------------------------------------------------------------
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
        usage = CanonicalUsage(model="m", provider="p", api="native", **c["counts"])
        b = compute_cost(usage, price, Decimal(c["markup"]))
        assert b.base == c["base"], f"{c['name']}: base {b.base} != {c['base']}"
        assert b.total == c["total"], f"{c['name']}: total {b.total} != {c['total']}"
        assert b.total_cents == c["total_cents"], f"{c['name']}: cents {b.total_cents} != {c['total_cents']}"


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
    field gets its own suffix off the same base id."""
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
    assert ids == {"backfill_01ABC_input", "backfill_01ABC_output"}


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

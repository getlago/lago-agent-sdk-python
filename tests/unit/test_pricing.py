"""Pricing tests — matching, money math, provider cache, and SDK price mode."""

from __future__ import annotations

import json
import pathlib
from decimal import Decimal
from typing import Any

import pytest

from lago_agent_sdk import CanonicalUsage, LagoConfig, LagoSDK, ModelPrice
from lago_agent_sdk.pricing import (
    PricingProvider,
    bedrock_model_key,
    coerce_markup,
    compute_cost,
    lookup_bedrock,
    lookup_openrouter,
    parse_bedrock_offer,
    parse_bedrock_region,
    parse_openrouter,
)

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "pricing"


# ----------------------------------------------------------------------
# Stub fetcher (no network) — mirrors the queue's injectable sender pattern
# ----------------------------------------------------------------------
class StubFetcher:
    def __init__(self, openrouter: dict | None = None, bedrock: dict | None = None) -> None:
        self._openrouter = openrouter or {"exact": {}, "norm": {}}
        self._bedrock = bedrock or {}
        self.openrouter_calls = 0
        self.bedrock_calls: list[str] = []

    def fetch_openrouter(self) -> dict[str, Any]:
        self.openrouter_calls += 1
        return self._openrouter

    def fetch_bedrock(self, region: str) -> dict[str, ModelPrice]:
        self.bedrock_calls.append(region)
        return self._bedrock.get(region, {})


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


def test_price_mode_emits_single_cost_event() -> None:
    sdk, received = _price_sdk(_warm_provider())
    u = CanonicalUsage(input=1000, output=500, model="claude-opus-4-8", provider="anthropic", api="native")
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert len(flat) == 1
    ev = flat[0]
    assert ev["code"] == "llm_cost"
    # Lago dynamic charge: top-level cents amount = 0.0175 USD * 100 = 1.75
    assert ev["precise_total_amount_cents"] == "1.75"
    props = ev["properties"]
    # `unit` = total tokens (1000 + 500) — the sum-aggregation quantity
    assert props["unit"] == "1500"
    # 1000*0.000005 + 500*0.000025 = 0.005 + 0.0125 = 0.0175
    assert props["value"] == "0.0175"
    assert props["base_cost"] == "0.0175"
    assert props["price_source"] == "openrouter"
    assert props["input_tokens"] == "1000"
    assert props["input_unit_price"] == "0.000005"
    assert props["output_cost"] == "0.0125"


def test_price_mode_markup_scales_value() -> None:
    sdk, received = _price_sdk(_warm_provider(), markup=2.0)
    u = CanonicalUsage(input=1000, output=500, model="claude-opus-4-8", provider="anthropic", api="native")
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    ev = [e for batch in received for e in batch][0]
    assert ev["properties"]["base_cost"] == "0.0175"
    assert ev["properties"]["value"] == "0.035"  # 0.0175 * 2
    assert ev["properties"]["markup"] == "2"


def test_per_call_markup_overrides_global() -> None:
    sdk, received = _price_sdk(_warm_provider(), markup=1.0)
    u = CanonicalUsage(input=1000, output=500, model="claude-opus-4-8", provider="anthropic", api="native")
    sdk.emit(u, markup=3.0)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    ev = [e for batch in received for e in batch][0]
    assert ev["properties"]["value"] == "0.0525"  # 0.0175 * 3


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
    props = [e for batch in received for e in batch][0]["properties"]
    # input billed for only the non-cached portion (1000 - 800); cache billed at cache rate
    assert props["input_tokens"] == "200"
    assert props["cache_read_tokens"] == "800"
    # 200*0.0000025 + 800*0.00000125 + 500*0.00001 = 0.0005 + 0.001 + 0.005 = 0.0065
    # (the bug would bill input at full 1000 -> 0.0085)
    assert props["value"] == "0.0065"
    # unit = billed tokens 200 + 800 + 500 = 1500 = prompt(1000) + completion(500)
    assert props["unit"] == "1500"


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
    props = [e for batch in received for e in batch][0]["properties"]
    assert props["input_tokens"] == "700"  # 1000 - 300 cached
    assert props["cache_read_tokens"] == "300"
    assert props["output_tokens"] == "400"
    assert props["reasoning_tokens"] == "100"  # billed separately (additive for Gemini)
    # 700*3e-7 + 300*7.5e-8 + 400*2.5e-6 + 100*2.5e-6 = 0.00021+0.0000225+0.001+0.00025 = 0.0014825
    assert props["value"] == "0.0014825"
    # unit = 700+300+400+100 = 1500 = prompt(1000)+candidates(400)+thoughts(100)
    assert props["unit"] == "1500"


def test_price_mode_openai_reasoning_in_output_not_double_billed() -> None:
    sdk, received = _price_sdk(_warm_provider())
    # OpenAI o-series: output (completion_tokens)=500 ALREADY includes reasoning=200.
    u = CanonicalUsage(input=100, output=500, reasoning=200, model="gpt-4o", provider="openai", api="native")
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    props = [e for batch in received for e in batch][0]["properties"]
    # reasoning folded into output — no separate reasoning line, output billed in full
    assert "reasoning_tokens" not in props
    assert props["output_tokens"] == "500"
    # 100*0.0000025 + 500*0.00001 = 0.00025 + 0.005 = 0.00525 (bug would add 200*1e-5=0.002)
    assert props["value"] == "0.00525"
    assert props["unit"] == "600"  # 100 + 500; reasoning not double-counted


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
    props = [e for batch in received for e in batch][0]["properties"]
    assert props["input_tokens"] == "1000"  # unchanged — additive provider
    assert props["cache_read_tokens"] == "400"
    assert props["cache_write_tokens"] == "200"
    # 1000*5e-6 + 500*25e-6 + 400*5e-7 + 200*6.25e-6 = 0.005+0.0125+0.0002+0.00125 = 0.01895
    assert props["value"] == "0.01895"
    assert props["unit"] == "2100"  # 1000+500+400+200, all additive


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
    assert len(flat) == 1 and flat[0]["code"] == "llm_cost"


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

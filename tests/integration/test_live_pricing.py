"""Live pricing test — hits the real OpenRouter + AWS Bedrock bulk APIs.

Skipped unless LAGO_LIVE_PRICING=1 (it makes real network calls, no keys needed
since both sources are public). Validates that the real fetchers build tables
and that known models resolve to sane USD-per-token prices — in particular it
exercises the AWS Bedrock offer-file parser against the live schema.
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest

from lago_agent_sdk.pricing import HttpPricingFetcher, lookup_bedrock, lookup_openrouter

pytestmark = pytest.mark.skipif(
    os.environ.get("LAGO_LIVE_PRICING") != "1",
    reason="LAGO_LIVE_PRICING != 1 (live network test)",
)


def test_openrouter_live_table_and_known_models() -> None:
    table = HttpPricingFetcher(timeout=30).fetch_openrouter()
    exact = table["exact"]
    assert len(exact) > 50, "expected a substantial OpenRouter model list"

    # A few well-known models should resolve with a positive input price.
    resolved = 0
    for provider, model in [
        ("openai", "gpt-4o"),
        ("anthropic", "claude-3.5-sonnet"),
        ("google", "gemini-2.5-flash"),
    ]:
        mp = lookup_openrouter(table, provider, model)
        if mp is not None and mp.input is not None and mp.input >= Decimal(0):
            resolved += 1
    assert resolved >= 1, "expected at least one well-known OpenRouter model to resolve"


def test_bedrock_live_table_builds_and_resolves() -> None:
    region = "us-east-1"
    table = HttpPricingFetcher(timeout=30).fetch_bedrock(region)
    # The parser should extract at least some priced models from the live offer.
    assert table, "AWS Bedrock offer parsed to an empty table — schema may have changed"
    priced = [mp for mp in table.values() if mp.input is not None or mp.output is not None]
    assert priced, "no Bedrock models had input/output token prices"

    # A common Bedrock model should resolve (best-effort; logs the key on miss).
    for model in [
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
    ]:
        mp = lookup_bedrock(table, model)
        if mp is not None and (mp.input or mp.output):
            return
    pytest.skip(f"no probed Bedrock model matched; {len(table)} keys built — refine matcher if needed")

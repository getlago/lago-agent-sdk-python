"""Bedrock Converse adapter tests — one per shape family, against real fixtures."""
from __future__ import annotations

import json
import pathlib

from lago_agent_sdk.adapters import extract_bedrock_converse

FIX = pathlib.Path(__file__).parent / "fixtures" / "bedrock" / "converse"

# One representative model per shape family.
_FAMILY_FIXTURES = {
    "standard":         "eu.amazon.nova-lite-v1_0.json",          # 33 models — non-Anthropic
    "cache_read_only":  "eu.anthropic.claude-opus-4-7.json",      # Opus 4.7
    "full_cache":       "eu.anthropic.claude-sonnet-4-6.json",    # Sonnet 4.5/4.6, Haiku 4.5, Opus 4.5/4.6
}


def _load(family: str) -> tuple[str, dict]:
    data = json.loads((FIX / _FAMILY_FIXTURES[family]).read_text())
    return data["_model_id"], data["_response"]


def test_standard_family_nova_lite():
    model_id, resp = _load("standard")
    u = extract_bedrock_converse(resp, model_id=model_id)
    assert u.input == 5
    assert u.output == 17
    assert u.cache_read == 0
    assert u.cache_write == 0
    assert u.api == "bedrock_converse"
    assert u.provider == "amazon"
    # serverToolUsage = {} should NOT add tool_calls or extras
    assert u.tool_calls == 0
    assert "serverToolUsage" not in u.extras


def test_cache_read_only_family_opus_4_7():
    model_id, resp = _load("cache_read_only")
    u = extract_bedrock_converse(resp, model_id=model_id)
    assert u.input == 21
    assert u.output == 37
    # No `cacheWriteInputTokens` field for Opus 4.7 — must not blow up.
    assert u.cache_read == 0  # not exercised in fixture, but adapter should accept the field
    assert u.cache_write == 0
    assert u.provider == "anthropic"


def test_full_cache_family_sonnet_4_6():
    model_id, resp = _load("full_cache")
    u = extract_bedrock_converse(resp, model_id=model_id)
    assert u.input == 12
    assert u.output == 36
    assert u.cache_read == 0
    assert u.cache_write == 0
    assert u.provider == "anthropic"


def test_cachereadinputtokencount_alias_is_ignored():
    """Newer alias `cacheReadInputTokenCount` is a duplicate of `*Tokens` — must not double count."""
    resp = {"usage": {"inputTokens": 10, "outputTokens": 20, "cacheReadInputTokens": 100, "cacheReadInputTokenCount": 999}}
    u = extract_bedrock_converse(resp, model_id="eu.anthropic.claude-opus-4-7")
    assert u.cache_read == 100
    # Alias should not pollute extras (it's a known field)
    assert "cacheReadInputTokenCount" not in u.extras


def test_servertoolusage_nonempty_flattens_to_tool_calls():
    resp = {"usage": {"inputTokens": 1, "outputTokens": 2, "serverToolUsage": {"webSearchRequests": 3}}}
    u = extract_bedrock_converse(resp, model_id="eu.amazon.nova-pro-v1:0")
    assert u.tool_calls == 3
    assert u.extras.get("serverToolUsage") == {"webSearchRequests": 3}


def test_unknown_field_lands_in_extras():
    resp = {"usage": {"inputTokens": 1, "outputTokens": 2, "noveltyField": "drift"}}
    u = extract_bedrock_converse(resp, model_id="eu.something.new")
    assert u.extras.get("noveltyField") == "drift"


def test_no_usage_field_returns_zeros():
    u = extract_bedrock_converse({}, model_id="x")
    assert u.input == 0
    assert u.output == 0
    assert not u.nonzero_numeric()

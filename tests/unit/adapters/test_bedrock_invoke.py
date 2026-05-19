"""Bedrock InvokeModel adapter tests — one per shape family, against real fixtures."""

from __future__ import annotations

import json
import pathlib

import pytest

from lago_agent_sdk.adapters import extract_bedrock_invoke, pick_invoke_adapter

FIX = pathlib.Path(__file__).parent / "fixtures" / "bedrock" / "invoke"

# One representative model per InvokeModel shape family.
_FAMILY_FIXTURES = {
    "openai_compat_basic": "openai.gpt-oss-20b-1_0.json",
    "openai_compat_with_details": "openai.gpt-oss-safeguard-20b.json",
    "anthropic": "eu.anthropic.claude-sonnet-4-6.json",
    "opus_4_7": "eu.anthropic.claude-opus-4-7.json",
    "nova": "eu.amazon.nova-lite-v1_0.json",
    "pixtral": "eu.mistral.pixtral-large-2502-v1_0.json",
    "mistral_legacy": "mistral.mistral-large-2402-v1_0.json",
}


def _load(family: str) -> tuple[str, dict]:
    data = json.loads((FIX / _FAMILY_FIXTURES[family]).read_text())
    return data["_model_id"], data["_response"]


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("eu.anthropic.claude-sonnet-4-6", "anthropic"),
        ("eu.anthropic.claude-opus-4-7", "opus_4_7"),
        ("eu.amazon.nova-lite-v1:0", "nova"),
        ("eu.mistral.pixtral-large-2502-v1:0", "pixtral"),
        ("mistral.mistral-large-2402-v1:0", "mistral_legacy"),
        ("mistral.mistral-7b-instruct-v0:2", "mistral_legacy"),
        ("mistral.mixtral-8x7b-instruct-v0:1", "mistral_legacy"),
        ("eu.mistral.ministral-3b-2410-v1:0", "openai_compat_basic"),
        ("eu.mistral.magistral-small-2509-v1:0", "openai_compat_basic"),
        ("openai.gpt-oss-safeguard-20b-1:0", "openai_compat_with_details"),
        ("openai.gpt-oss-safeguard-120b-1:0", "openai_compat_with_details"),
        ("eu.minimax.minimax-m2-v1:0", "openai_compat_with_details"),
        ("openai.gpt-oss-20b-1:0", "openai_compat_basic"),
        ("eu.qwen.qwen3-235b-a22b-instruct-2507-v1:0", "openai_compat_basic"),
    ],
)
def test_dispatch(model_id: str, expected: str) -> None:
    assert pick_invoke_adapter(model_id) == expected


# --------------------------------------------------------------------------
# Per-family fixtures
# --------------------------------------------------------------------------
def test_openai_compat_basic_gpt_oss_20b():
    model_id, resp = _load("openai_compat_basic")
    u = extract_bedrock_invoke(resp, model_id=model_id)
    assert u.input == 72
    assert u.output == 40
    assert u.api == "bedrock_invoke"
    assert u.provider == "openai"


def test_openai_compat_with_details_gpt_oss_safeguard():
    model_id, resp = _load("openai_compat_with_details")
    u = extract_bedrock_invoke(resp, model_id=model_id)
    assert u.input == 72
    assert u.output == 40
    assert u.reasoning == 0  # fixture has no reasoning_tokens
    # prompt_tokens_details lands in extras (drift)
    assert "prompt_tokens_details" in u.extras


def test_openai_compat_with_details_extracts_reasoning_when_present():
    resp = {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 12},
        }
    }
    u = extract_bedrock_invoke(resp, model_id="openai.gpt-oss-safeguard-20b-1:0")
    assert u.reasoning == 12


def test_anthropic_sonnet_4_6():
    model_id, resp = _load("anthropic")
    u = extract_bedrock_invoke(resp, model_id=model_id)
    assert u.input == 12
    assert u.output == 36
    assert u.cache_write == 0
    assert u.cache_read == 0
    assert u.cache_write_5m == 0
    assert u.cache_write_1h == 0
    assert u.provider == "anthropic"


def test_anthropic_extracts_cache_creation_ephemerals():
    resp = {
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_creation_input_tokens": 50,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 30,
                "ephemeral_1h_input_tokens": 20,
            },
        }
    }
    u = extract_bedrock_invoke(resp, model_id="eu.anthropic.claude-sonnet-4-6")
    assert u.cache_write == 50
    assert u.cache_write_5m == 30
    assert u.cache_write_1h == 20


def test_anthropic_tool_calls_from_content_blocks():
    resp = {
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "content": [
            {"type": "text", "text": "..."},
            {"type": "tool_use", "id": "t1"},
            {"type": "tool_use", "id": "t2"},
        ],
    }
    u = extract_bedrock_invoke(resp, model_id="eu.anthropic.claude-sonnet-4-6")
    assert u.tool_calls == 2


def test_opus_4_7_lands_service_tier_in_extras():
    model_id, resp = _load("opus_4_7")
    u = extract_bedrock_invoke(resp, model_id=model_id)
    assert u.input == 21
    assert u.output == 36
    assert u.extras.get("service_tier") == "standard"


def test_nova_lite():
    model_id, resp = _load("nova")
    u = extract_bedrock_invoke(resp, model_id=model_id)
    assert u.input == 5
    assert u.output == 18
    assert u.cache_read == 0
    assert u.cache_write == 0
    assert u.provider == "amazon"


def test_pixtral_large_request_count_in_extras():
    model_id, resp = _load("pixtral")
    u = extract_bedrock_invoke(resp, model_id=model_id)
    assert u.input == 10
    assert u.output == 21
    # request_count is null in fixture — still stored
    assert "request_count" in u.extras


def test_mistral_legacy_returns_no_usage_extras():
    model_id, resp = _load("mistral_legacy")
    u = extract_bedrock_invoke(resp, model_id=model_id)
    assert u.extras.get("_no_usage") is True
    # No nonzero numeric fields — emit() will skip
    assert not u.nonzero_numeric()
    assert u.provider == "mistral"

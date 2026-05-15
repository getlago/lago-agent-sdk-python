"""Drift detection — unknown fields land in extras, not in numeric counts."""

from __future__ import annotations

from lago_agent_sdk.adapters import (
    extract_bedrock_converse,
    extract_bedrock_invoke,
)


def test_converse_unknown_top_level_usage_field_goes_to_extras():
    resp = {"usage": {"inputTokens": 10, "outputTokens": 20, "futureCacheReadAtL1Tokens": 99}}
    u = extract_bedrock_converse(resp, model_id="eu.something.future")
    assert u.input == 10
    assert u.output == 20
    assert u.extras.get("futureCacheReadAtL1Tokens") == 99


def test_converse_known_aliases_do_not_pollute_extras():
    resp = {
        "usage": {
            "inputTokens": 10,
            "outputTokens": 20,
            "cacheReadInputTokens": 5,
            "cacheReadInputTokenCount": 5,  # alias, ignored
            "cacheWriteInputTokenCount": 0,  # alias, ignored
            "totalTokens": 30,
            "serverToolUsage": {},
        }
    }
    u = extract_bedrock_converse(resp, model_id="eu.anthropic.claude-sonnet-4-6")
    assert u.cache_read == 5
    assert "cacheReadInputTokenCount" not in u.extras
    assert "cacheWriteInputTokenCount" not in u.extras
    assert "totalTokens" not in u.extras


def test_invoke_anthropic_unknown_top_usage_field_goes_to_extras():
    resp = {
        "usage": {"input_tokens": 13, "output_tokens": 39, "newSpecialField": "spectacular"},
        "content": [],
    }
    u = extract_bedrock_invoke(resp, model_id="eu.anthropic.claude-sonnet-4-6")
    assert u.extras.get("newSpecialField") == "spectacular"


def test_invoke_opus_4_7_service_tier_in_extras():
    resp = {"usage": {"input_tokens": 5, "output_tokens": 7, "service_tier": "priority"}, "content": []}
    u = extract_bedrock_invoke(resp, model_id="eu.anthropic.claude-opus-4-7")
    assert u.extras.get("service_tier") == "priority"


def test_invoke_openai_compat_prompt_tokens_details_lands_in_extras():
    """Spec maps only completion_tokens_details.reasoning_tokens — anything in
    prompt_tokens_details is real drift signal we want to surface."""
    resp = {
        "usage": {
            "prompt_tokens": 73,
            "completion_tokens": 80,
            "prompt_tokens_details": {"cached_tokens": 48},
        }
    }
    u = extract_bedrock_invoke(resp, model_id="openai.gpt-oss-safeguard-20b-1:0")
    assert "prompt_tokens_details" in u.extras
    assert u.extras["prompt_tokens_details"] == {"cached_tokens": 48}

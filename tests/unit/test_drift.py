"""Drift detection — unknown fields land in extras, not in numeric counts."""

from __future__ import annotations

from lago_agent_sdk.adapters import (
    extract_bedrock_converse,
    extract_bedrock_invoke,
    extract_openai_native,
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


# ----------------------------------------------------------------------
# Native OpenAI adapter — drift must be caught ONE LEVEL DOWN too
# ----------------------------------------------------------------------


def test_openai_native_nested_detail_drift_reaches_extras():
    """The drift contract has to hold inside the *_tokens_details sub-objects,
    not just at the top level.

    This is the hole a live `gpt-5.6-sol` response found: it reports
    `prompt_tokens_details.cache_write_tokens: 3022`, and because
    `prompt_tokens_details` is itself a KNOWN top-level key, the old sweep never
    looked inside it. 3022 real tokens were discarded with no error and no
    on_error — the exact failure this module exists to prevent. Every drift test
    passed, because none of them looked one level down.
    """
    resp = {
        "usage": {
            "prompt_tokens": 3025,
            "completion_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 3022},
            "completion_tokens_details": {"reasoning_tokens": 0, "future_nested_xyz": 42},
        }
    }
    u = extract_openai_native(resp)
    assert u.extras["prompt_tokens_details.cache_write_tokens"] == 3022
    assert u.extras["completion_tokens_details.future_nested_xyz"] == 42


def test_openai_native_mapped_nested_fields_do_not_pollute_extras():
    """The mirror of the above: a nested key we DO map must not also appear in
    extras, or every event carries a duplicate of a value already billed."""
    resp = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 40, "audio_tokens": 5},
            "completion_tokens_details": {"reasoning_tokens": 20, "audio_tokens": 3},
        }
    }
    u = extract_openai_native(resp)
    assert u.cache_read == 40 and u.reasoning == 20
    assert u.audio_input == 5 and u.audio_output == 3
    for k in u.extras:
        assert not k.endswith((".cached_tokens", ".reasoning_tokens", ".audio_tokens")), k


def test_openai_native_responses_api_nested_drift_reaches_extras():
    """Same guarantee on the Responses-API shape, whose detail containers are
    named differently (`input_tokens_details` / `output_tokens_details`)."""
    resp = {
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "input_tokens_details": {"cached_tokens": 2, "novel_input_detail": "x"},
            "output_tokens_details": {"reasoning_tokens": 1, "novel_output_detail": "y"},
        }
    }
    u = extract_openai_native(resp)
    assert u.api == "responses"
    assert u.extras["input_tokens_details.novel_input_detail"] == "x"
    assert u.extras["output_tokens_details.novel_output_detail"] == "y"


def test_anthropic_service_tier_and_inference_geo_reach_extras():
    """Two fields that appeared on live Anthropic responses through the Databricks
    gateway and are in no fixture predating it: `service_tier` ("standard") and
    `inference_geo` ("global" for sonnet-4-6, "not_available" for the others).

    Neither is a token count, so both must land in extras — never be miscounted as
    a metric, and never silently dropped."""
    from lago_agent_sdk.adapters import extract_anthropic_native

    resp = {
        "model": "claude-sonnet-4-6",
        "usage": {
            "input_tokens": 8,
            "output_tokens": 4,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "service_tier": "standard",
            "inference_geo": "global",
        },
    }
    u = extract_anthropic_native(resp)
    assert u.input == 8 and u.output == 4
    assert u.extras["service_tier"] == "standard"
    assert u.extras["inference_geo"] == "global"
    # and they must not have leaked into any numeric field
    assert u.nonzero_numeric() == {"input": 8, "output": 4}


def test_responses_audio_tokens_reach_extras_because_nothing_maps_them() -> None:
    """`output_tokens_details.audio_tokens` was listed as a MAPPED nested key, so it was
    excluded from extras — while the Responses branch hardcodes `audio_output = 0`
    because the API doesn't expose it. Both true at once means the count is neither
    billed nor surfaced: 500 real tokens gone with no error, which is the precise hole
    this module exists to close."""
    resp = {
        "usage": {
            "input_tokens": 10,
            "output_tokens": 500,
            "output_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 500},
        }
    }
    u = extract_openai_native(resp)
    assert u.api == "responses"
    assert u.audio_output == 0, "Responses API does not expose it, so it must not be invented"
    assert u.extras["output_tokens_details.audio_tokens"] == 500


def test_unaccounted_total_does_not_double_bill_additive_reasoning() -> None:
    """The `total_tokens` guard folds an unexplained delta into `output`. For a provider
    whose reasoning is ADDITIVE (this adapter now stamps `databricks` and `workers-ai`,
    not only `openai`), a payload reporting BOTH `reasoning_tokens` and an inflated total
    would be charged for them twice — inside the grown output and again as a reasoning
    line. Subtracting reasoning from the accounted total prevents that."""
    resp = {
        "usage": {
            "prompt_tokens": 57,
            "completion_tokens": 47,
            "total_tokens": 1253,
            "completion_tokens_details": {"reasoning_tokens": 1149},
        }
    }
    u = extract_openai_native(resp, provider_hint="databricks")
    assert u.reasoning == 1149
    assert u.output == 47, "reasoning already accounts for the delta; output must not grow"
    assert "unaccounted_output_tokens" not in u.extras


def test_unaccounted_total_still_recovers_tokens_nobody_broke_out() -> None:
    """The case the guard was written for is unchanged: a thinking model behind a proxy
    that reports no breakdown at all. Measured live — prompt 57, completion 47, total
    1253, and no `completion_tokens_details` to recover the 1,149 from."""
    resp = {"usage": {"prompt_tokens": 57, "completion_tokens": 47, "total_tokens": 1253}}
    u = extract_openai_native(resp)
    assert u.output == 47 + 1149
    assert u.extras["unaccounted_output_tokens"] == 1149

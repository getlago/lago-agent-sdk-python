"""OpenAI native adapter — verified against real fixtures."""

from __future__ import annotations

import json
import pathlib

from lago_agent_sdk.adapters import extract_openai_native

FIX = pathlib.Path(__file__).parent / "fixtures" / "openai_native"


def _load(name: str) -> tuple[str, dict]:
    data = json.loads((FIX / name).read_text())
    return data["_model_id"], data["_response"]


# --------------------------------------------------------------------------
# Chat Completions fixtures
# --------------------------------------------------------------------------
def test_plain_chat() -> None:
    model_id, resp = _load("01_plain_chat.json")
    u = extract_openai_native(resp, model_id=model_id)
    assert u.input == 13
    assert u.output == 23
    assert u.cache_read == 0
    assert u.reasoning == 0
    assert u.tool_calls == 0
    assert u.audio_input == 0
    assert u.audio_output == 0
    assert u.api == "chat_completions"
    assert u.provider == "openai"


def test_tool_use_chat_counts_tool_calls() -> None:
    model_id, resp = _load("02_tool_use_chat.json")
    u = extract_openai_native(resp, model_id=model_id)
    assert u.input == 60
    assert u.output == 5
    assert u.tool_calls == 1
    assert u.api == "chat_completions"


def test_cache_call1_no_cache_yet() -> None:
    """First call with a long prompt — OpenAI hasn't cached it yet."""
    model_id, resp = _load("03_cache_call1_chat.json")
    u = extract_openai_native(resp, model_id=model_id)
    assert u.input == 3819
    assert u.output == 20
    assert u.cache_read == 0


def test_cache_call2_auto_cached() -> None:
    """Second call with the same long prompt — OpenAI auto-caches, exposes cached_tokens."""
    model_id, resp = _load("04_cache_call2_chat.json")
    u = extract_openai_native(resp, model_id=model_id)
    assert u.input == 3819
    assert u.output == 20
    assert u.cache_read == 3712  # most of the system prompt cached
    # OpenAI doesn't expose cache_write / cache_write_5m / cache_write_1h
    assert u.cache_write == 0
    assert u.cache_write_5m == 0


def test_streaming_chat_final_chunk_carries_usage() -> None:
    """When stream_options.include_usage=True, the final chunk carries the usage payload."""
    model_id, resp = _load("05_streaming_chat.json")
    chunks = resp["chunks"]
    # Find the chunk with usage (it's the last one)
    final_with_usage = next((c for c in reversed(chunks) if c.get("usage")), None)
    assert final_with_usage is not None
    u = extract_openai_native(final_with_usage, model_id=model_id)
    assert u.input == 13
    assert u.output == 29
    assert u.api == "chat_completions"


def test_reasoning_chat_exposes_reasoning_tokens() -> None:
    """o-series models populate completion_tokens_details.reasoning_tokens — first provider to do so."""
    model_id, resp = _load("06_reasoning_chat.json")
    u = extract_openai_native(resp, model_id=model_id)
    assert u.input == 33
    assert u.output == 1579
    assert u.reasoning == 832  # actual measured value — not folded away
    assert u.tool_calls == 0


def test_multi_turn_chat() -> None:
    model_id, resp = _load("07_multi_turn_chat.json")
    u = extract_openai_native(resp, model_id=model_id)
    assert u.input == 34
    assert u.output == 8


# --------------------------------------------------------------------------
# Responses API fixtures
# --------------------------------------------------------------------------
def test_plain_responses() -> None:
    model_id, resp = _load("08_plain_responses.json")
    u = extract_openai_native(resp, model_id=model_id)
    assert u.input == 13
    assert u.output == 19
    assert u.api == "responses"
    assert u.provider == "openai"


def test_tool_use_responses_counts_function_calls() -> None:
    """Responses API encodes tool calls as items in `output[]` with type 'function_call'."""
    model_id, resp = _load("09_tool_use_responses.json")
    u = extract_openai_native(resp, model_id=model_id)
    assert u.input == 53
    assert u.output == 6
    assert u.tool_calls == 1
    assert u.api == "responses"


def test_reasoning_responses() -> None:
    model_id, resp = _load("10_reasoning_responses.json")
    u = extract_openai_native(resp, model_id=model_id)
    assert u.input == 33
    assert u.output == 981
    assert u.reasoning == 320
    assert u.api == "responses"


# --------------------------------------------------------------------------
# API detection
# --------------------------------------------------------------------------
def test_chat_completions_shape_detected() -> None:
    """`prompt_tokens` in usage → Chat Completions."""
    u = extract_openai_native(
        {"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        model_id="gpt-4o",
    )
    assert u.api == "chat_completions"


def test_responses_api_shape_detected() -> None:
    """`input_tokens` (without prompt_tokens) → Responses API."""
    u = extract_openai_native(
        {"usage": {"input_tokens": 1, "output_tokens": 1}},
        model_id="gpt-4o",
    )
    assert u.api == "responses"


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------
def test_handles_pydantic_via_model_dump() -> None:
    class FakePydantic:
        def model_dump(self) -> dict:
            return {
                "model": "gpt-4o-mini",
                "choices": [{"message": {"tool_calls": [{"id": "t1"}, {"id": "t2"}]}}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                    "completion_tokens_details": {
                        "reasoning_tokens": 3,
                        "audio_tokens": 0,
                    },
                },
            }

    u = extract_openai_native(FakePydantic(), model_id="gpt-4o-mini")
    assert u.input == 5
    assert u.output == 7
    assert u.reasoning == 3
    assert u.tool_calls == 2
    assert u.api == "chat_completions"


def test_no_usage_returns_zeros() -> None:
    u = extract_openai_native({}, model_id="gpt-4o-mini")
    assert u.input == 0
    assert u.output == 0
    assert not u.nonzero_numeric()


def test_survives_non_dict_usage() -> None:
    assert extract_openai_native({"usage": True}, model_id="x").input == 0
    assert extract_openai_native({"usage": "bogus"}, model_id="x").output == 0
    assert extract_openai_native(None, model_id="x").input == 0


def test_unknown_top_usage_field_lands_in_extras() -> None:
    """If OpenAI adds a new top-level field, drift detection picks it up."""
    resp = {
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "future_field_xyz": "novel",
        }
    }
    u = extract_openai_native(resp, model_id="gpt-4o")
    assert u.extras.get("future_field_xyz") == "novel"


def test_audio_input_mapped_from_prompt_details() -> None:
    """Chat Completions audio input lives at usage.prompt_tokens_details.audio_tokens."""
    resp = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {"audio_tokens": 42, "cached_tokens": 0},
            "completion_tokens_details": {"audio_tokens": 0, "reasoning_tokens": 0},
        }
    }
    u = extract_openai_native(resp, model_id="gpt-4o-audio")
    assert u.audio_input == 42
    assert u.audio_output == 0


def test_audio_output_mapped_from_completion_details() -> None:
    """GPT-4o-audio output audio lives at usage.completion_tokens_details.audio_tokens."""
    resp = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
            "completion_tokens_details": {"audio_tokens": 33, "reasoning_tokens": 0},
        }
    }
    u = extract_openai_native(resp, model_id="gpt-4o-audio")
    assert u.audio_input == 0
    assert u.audio_output == 33

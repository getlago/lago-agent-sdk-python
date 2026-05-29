"""Gemini native adapter — verified against real fixtures captured via google-genai."""

from __future__ import annotations

import json
import pathlib

from lago_agent_sdk.adapters import extract_gemini_native

FIX = pathlib.Path(__file__).parent / "fixtures" / "gemini_native"


def _load(name: str) -> tuple[str, dict]:
    data = json.loads((FIX / name).read_text())
    return data["_model_id"], data["_response"]


# --------------------------------------------------------------------------
# Real fixtures
# --------------------------------------------------------------------------
def test_plain_flash() -> None:
    """Plain call to gemini-2.5-flash: input/output/reasoning all populated."""
    model_id, resp = _load("01_plain_flash.json")
    u = extract_gemini_native(resp, model_id=model_id)
    assert u.input == 7
    assert u.output == 23
    # Gemini 2.5 emits thoughts even without explicit thinking config
    assert u.reasoning == 442
    assert u.tool_calls == 0
    assert u.cache_read == 0
    assert u.api == "native"
    assert u.provider == "gemini"


def test_tool_use_counts_function_calls() -> None:
    """A function_call in candidates[0].content.parts[] increments tool_calls."""
    model_id, resp = _load("02_tool_use.json")
    u = extract_gemini_native(resp, model_id=model_id)
    assert u.input == 49
    assert u.output == 15
    assert u.tool_calls == 1


def test_streaming_final_chunk_carries_usage() -> None:
    """The streaming wrapper grabs usage from the last chunk that has it."""
    model_id, resp = _load("03_streaming.json")
    chunks = resp["chunks"]
    final = next((c for c in reversed(chunks) if c.get("usage_metadata")), None)
    assert final is not None
    u = extract_gemini_native(final, model_id=model_id)
    assert u.input == 14
    assert u.output == 9
    assert u.reasoning == 29


def test_thinking_mode_populates_reasoning() -> None:
    """Gemini 2.5 with explicit thinking_config emits a large thoughts_token_count."""
    model_id, resp = _load("04_thinking.json")
    u = extract_gemini_native(resp, model_id=model_id)
    assert u.input == 27
    assert u.output == 1003
    assert u.reasoning == 1546
    # Math check: candidates + thoughts + prompt = total (additive, not subset)
    assert u.input + u.output + u.reasoning == 2576  # matches usage_metadata.total_token_count


def test_multi_turn() -> None:
    model_id, resp = _load("05_multi_turn.json")
    u = extract_gemini_native(resp, model_id=model_id)
    assert u.input == 22
    assert u.output == 25


# --------------------------------------------------------------------------
# Synthetic — edge cases the fixtures didn't cover (no real audio/image test traffic)
# --------------------------------------------------------------------------
def test_audio_input_from_modality_details() -> None:
    """Multimodal AUDIO input lives in usage_metadata.prompt_tokens_details[modality=AUDIO]."""
    resp = {
        "usage_metadata": {
            "prompt_token_count": 1000,
            "candidates_token_count": 50,
            "prompt_tokens_details": [
                {"modality": "TEXT", "token_count": 200},
                {"modality": "AUDIO", "token_count": 800},
            ],
        }
    }
    u = extract_gemini_native(resp, model_id="gemini-2.5-flash")
    assert u.input == 1000
    assert u.audio_input == 800
    assert u.image_input == 0


def test_image_input_from_modality_details() -> None:
    resp = {
        "usage_metadata": {
            "prompt_token_count": 500,
            "candidates_token_count": 50,
            "prompt_tokens_details": [
                {"modality": "TEXT", "token_count": 300},
                {"modality": "IMAGE", "token_count": 200},
            ],
        }
    }
    u = extract_gemini_native(resp, model_id="gemini-2.5-flash")
    assert u.image_input == 200


def test_audio_output_from_modality_details() -> None:
    """Audio output (e.g. TTS-capable model) lives in candidates_tokens_details[modality=AUDIO]."""
    resp = {
        "usage_metadata": {
            "prompt_token_count": 50,
            "candidates_token_count": 1500,
            "candidates_tokens_details": [
                {"modality": "AUDIO", "token_count": 1500},
            ],
        }
    }
    u = extract_gemini_native(resp, model_id="gemini-2.5-flash-audio")
    assert u.audio_output == 1500


def test_cached_content_token_count() -> None:
    """When CachedContent API has been primed, cached_content_token_count fires."""
    resp = {
        "usage_metadata": {
            "prompt_token_count": 5000,
            "candidates_token_count": 30,
            "cached_content_token_count": 4800,
        }
    }
    u = extract_gemini_native(resp, model_id="gemini-2.5-flash")
    assert u.cache_read == 4800


def test_multiple_function_calls_counted() -> None:
    resp = {
        "usage_metadata": {"prompt_token_count": 10, "candidates_token_count": 20},
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "..."},
                        {"function_call": {"name": "fn1"}},
                        {"function_call": {"name": "fn2"}},
                        {"function_call": {"name": "fn3"}},
                    ]
                }
            }
        ],
    }
    u = extract_gemini_native(resp, model_id="gemini-2.5-flash")
    assert u.tool_calls == 3


def test_handles_pydantic_via_model_dump() -> None:
    class FakePydantic:
        def model_dump(self) -> dict:
            return {
                "model_version": "gemini-2.5-flash",
                "candidates": [
                    {"content": {"parts": [{"function_call": {"name": "x"}}]}}
                ],
                "usage_metadata": {
                    "prompt_token_count": 10,
                    "candidates_token_count": 20,
                    "thoughts_token_count": 5,
                },
            }

    u = extract_gemini_native(FakePydantic(), model_id="gemini-2.5-flash")
    assert u.input == 10
    assert u.output == 20
    assert u.reasoning == 5
    assert u.tool_calls == 1
    assert u.api == "native"


def test_no_usage_metadata_returns_zeros() -> None:
    u = extract_gemini_native({}, model_id="gemini-2.5-flash")
    assert u.input == 0
    assert u.output == 0
    assert not u.nonzero_numeric()


def test_survives_non_dict_usage_metadata() -> None:
    assert extract_gemini_native({"usage_metadata": True}, model_id="x").input == 0
    assert extract_gemini_native({"usage_metadata": "bogus"}, model_id="x").output == 0
    assert extract_gemini_native(None, model_id="x").input == 0


def test_unknown_usage_field_lands_in_extras() -> None:
    """If Google adds a new top-level usage field, drift detection picks it up."""
    resp = {
        "usage_metadata": {
            "prompt_token_count": 10,
            "candidates_token_count": 20,
            "future_field_xyz": "novel",
        }
    }
    u = extract_gemini_native(resp, model_id="gemini-2.5-flash")
    assert u.extras.get("future_field_xyz") == "novel"


def test_traffic_type_lands_in_known_fields_not_extras() -> None:
    """traffic_type is a known metadata field; it shouldn't leak into extras."""
    resp = {
        "usage_metadata": {
            "prompt_token_count": 10,
            "candidates_token_count": 20,
            "traffic_type": "PAID",
        }
    }
    u = extract_gemini_native(resp, model_id="gemini-2.5-flash")
    assert "traffic_type" not in u.extras

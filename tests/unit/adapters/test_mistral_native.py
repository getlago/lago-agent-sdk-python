"""Mistral native adapter — verified against real fixtures."""

from __future__ import annotations

import json
import pathlib

from lago_agent_sdk.adapters import extract_mistral_native

FIX = pathlib.Path(__file__).parent / "fixtures" / "mistral_native"


def _load(name: str) -> tuple[str, dict]:
    data = json.loads((FIX / name).read_text())
    return data["_model_id"], data["_response"]


# --------------------------------------------------------------------------
# Real fixtures
# --------------------------------------------------------------------------
def test_plain_small():
    model_id, resp = _load("01_plain_small.json")
    u = extract_mistral_native(resp, model_id=model_id)
    assert u.input == 22
    assert u.output == 19
    assert u.cache_read == 0
    assert u.tool_calls == 0
    assert u.provider == "mistral"
    assert u.api == "native"
    assert u.model == "mistral-small-latest"


def test_plain_large():
    model_id, resp = _load("02_plain_large.json")
    u = extract_mistral_native(resp, model_id=model_id)
    assert u.input == 10
    assert u.output == 19
    assert u.cache_read == 0


def test_tool_use_counts_tool_calls():
    model_id, resp = _load("03_tool_use.json")
    u = extract_mistral_native(resp, model_id=model_id)
    assert u.tool_calls == 1
    assert u.input == 83
    assert u.output == 23


def test_magistral_reasoning_is_bundled_into_completion():
    """Magistral DOES reason but bundles tokens into completion_tokens — no reasoning_tokens field."""
    model_id, resp = _load("04_reasoning_magistral.json")
    u = extract_mistral_native(resp, model_id=model_id)
    assert u.input == 54
    assert u.output == 600
    # The reasoning is invisible at the usage layer — confirm we don't double-count
    assert u.reasoning == 0


def test_multi_turn_accumulates_input_tokens():
    model_id, resp = _load("06_multi_turn.json")
    u = extract_mistral_native(resp, model_id=model_id)
    assert u.input == 37
    assert u.output == 18


def test_cache_attempt_returns_zero_when_not_engaged():
    """Even with identical 1210-token prompts back-to-back, Mistral's cache didn't engage in the fixture."""
    _, resp1 = _load("07_cache_attempt_call1.json")
    _, resp2 = _load("07_cache_attempt_call2.json")
    u1 = extract_mistral_native(resp1, model_id="mistral-large-latest")
    u2 = extract_mistral_native(resp2, model_id="mistral-large-latest")
    assert u1.cache_read == 0
    assert u2.cache_read == 0


# --------------------------------------------------------------------------
# Synthetic — verify cached_tokens path works when populated
# --------------------------------------------------------------------------
def test_cache_read_extracted_when_present():
    resp = {
        "model": "mistral-large-latest",
        "choices": [{"message": {"content": "hi", "tool_calls": None}}],
        "usage": {
            "prompt_tokens": 1500,
            "completion_tokens": 5,
            "total_tokens": 1505,
            "prompt_tokens_details": {"cached_tokens": 1200},
        },
    }
    u = extract_mistral_native(resp, model_id="mistral-large-latest")
    assert u.input == 1500
    assert u.cache_read == 1200


def test_three_tool_calls_counted():
    resp = {
        "model": "x",
        "choices": [{"message": {"tool_calls": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    u = extract_mistral_native(resp, model_id="x")
    assert u.tool_calls == 3


def test_unknown_usage_field_lands_in_extras():
    resp = {
        "model": "x",
        "choices": [{"message": {}}],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "novel_field": 99,
        },
    }
    u = extract_mistral_native(resp, model_id="x")
    assert u.extras.get("novel_field") == 99


def test_no_usage_returns_zeros():
    u = extract_mistral_native({}, model_id="m")
    assert u.input == 0
    assert u.output == 0
    assert u.tool_calls == 0
    assert not u.nonzero_numeric()


def test_handles_pydantic_response_via_model_dump():
    """Real mistralai SDK returns pydantic objects — adapter should accept them."""

    class FakePydantic:
        def model_dump(self):
            return {
                "model": "mistral-small-latest",
                "choices": [{"message": {"content": "hi", "tool_calls": None}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }

    u = extract_mistral_native(FakePydantic(), model_id="mistral-small-latest")
    assert u.input == 5
    assert u.output == 3

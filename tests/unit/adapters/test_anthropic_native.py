"""Anthropic native adapter — verified against real fixtures."""

from __future__ import annotations

import json
import pathlib

from lago_agent_sdk.adapters import extract_anthropic_native

FIX = pathlib.Path(__file__).parent / "fixtures" / "anthropic_native"


def _load(name: str) -> tuple[str, dict]:
    data = json.loads((FIX / name).read_text())
    return data["_model_id"], data["_response"]


# --------------------------------------------------------------------------
# Real fixtures
# --------------------------------------------------------------------------
def test_plain_haiku() -> None:
    model_id, resp = _load("01_plain_haiku.json")
    u = extract_anthropic_native(resp, model_id=model_id)
    assert u.input == 13
    assert u.output == 35
    assert u.cache_read == 0
    assert u.cache_write == 0
    assert u.tool_calls == 0
    assert u.api == "native"
    assert u.provider == "anthropic"
    assert u.model == "claude-haiku-4-5-20251001"


def test_plain_sonnet() -> None:
    model_id, resp = _load("02_plain_sonnet.json")
    u = extract_anthropic_native(resp, model_id=model_id)
    assert u.input == 13
    assert u.output == 39


def test_tool_use_counts_tool_calls() -> None:
    model_id, resp = _load("03_tool_use.json")
    u = extract_anthropic_native(resp, model_id=model_id)
    assert u.input == 658
    assert u.output == 38
    assert u.tool_calls == 1


def test_cache_create_5m() -> None:
    model_id, resp = _load("04_cache_create_5m.json")
    u = extract_anthropic_native(resp, model_id=model_id)
    assert u.cache_write == 2803
    assert u.cache_write_5m == 2803
    assert u.cache_write_1h == 0
    assert u.cache_read == 0


def test_cache_read_after_create() -> None:
    model_id, resp = _load("05_cache_read.json")
    u = extract_anthropic_native(resp, model_id=model_id)
    assert u.cache_read == 2803
    assert u.cache_write == 0
    assert u.cache_write_5m == 0


def test_cache_create_1h() -> None:
    model_id, resp = _load("06_cache_create_1h.json")
    u = extract_anthropic_native(resp, model_id=model_id)
    assert u.cache_write == 2808
    assert u.cache_write_1h == 2808
    assert u.cache_write_5m == 0


def test_extended_thinking_bundles_into_output_tokens() -> None:
    """Anthropic's extended thinking does NOT expose reasoning_tokens — they're folded into output_tokens."""
    model_id, resp = _load("07_extended_thinking.json")
    u = extract_anthropic_native(resp, model_id=model_id)
    assert u.input == 66
    assert u.output == 862  # all 862 includes thinking + final answer
    assert u.reasoning == 0  # confirmed: Anthropic doesn't separate it
    # content has both 'thinking' and 'text' blocks — neither counts as a tool call
    assert u.tool_calls == 0


def test_multi_turn() -> None:
    model_id, resp = _load("09_multi_turn.json")
    u = extract_anthropic_native(resp, model_id=model_id)
    assert u.input == 34
    assert u.output == 14


def test_unknown_top_usage_field_lands_in_extras() -> None:
    """service_tier, inference_geo, server_tool_use are new fields → drift detection."""
    model_id, resp = _load("01_plain_haiku.json")
    u = extract_anthropic_native(resp, model_id=model_id)
    assert "service_tier" in u.extras
    assert "inference_geo" in u.extras
    assert "server_tool_use" in u.extras


# --------------------------------------------------------------------------
# Synthetic
# --------------------------------------------------------------------------
def test_handles_pydantic_via_model_dump() -> None:
    class FakePydantic:
        def model_dump(self) -> dict:
            return {
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "hi"}],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 7,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 0,
                        "ephemeral_1h_input_tokens": 0,
                    },
                },
            }

    u = extract_anthropic_native(FakePydantic(), model_id="claude-sonnet-4-6")
    assert u.input == 5
    assert u.output == 7
    assert u.api == "native"


def test_multiple_tool_use_blocks_counted() -> None:
    resp = {
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "content": [
            {"type": "text", "text": "..."},
            {"type": "tool_use", "id": "t1"},
            {"type": "tool_use", "id": "t2"},
            {"type": "tool_use", "id": "t3"},
        ],
    }
    u = extract_anthropic_native(resp, model_id="claude-sonnet-4-6")
    assert u.tool_calls == 3


def test_no_usage_returns_zeros() -> None:
    u = extract_anthropic_native({}, model_id="claude-sonnet-4-6")
    assert u.input == 0
    assert u.output == 0
    assert not u.nonzero_numeric()


def test_survives_non_dict_usage() -> None:
    assert extract_anthropic_native({"usage": True}, model_id="x").input == 0
    assert extract_anthropic_native({"usage": "bogus"}, model_id="x").output == 0
    assert extract_anthropic_native(None, model_id="x").input == 0

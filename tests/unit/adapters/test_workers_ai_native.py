"""Workers AI `/ai/run` adapter — verified against real captured fixtures."""

from __future__ import annotations

import json
import pathlib

import pytest

from lago_agent_sdk.adapters import extract_workers_ai_native

FIX = pathlib.Path(__file__).parent / "fixtures" / "workers_ai"


def _all() -> list[pathlib.Path]:
    return sorted(FIX.glob("*.json")) if FIX.exists() else []


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


# --------------------------------------------------------------------------
# Catalog models — OpenAI-shaped usage plus Cloudflare's `neurons`
# --------------------------------------------------------------------------
def test_chat_direct_maps_openai_shaped_usage():
    d = _load("01_chat_direct.json")
    u = extract_workers_ai_native(d["_response"], model_id=d["_model_id"])
    assert u.input == 41
    assert u.output == 31
    assert u.cache_read == 0
    assert u.reasoning == 0
    assert u.provider == "workers-ai"
    assert u.api == "workers_ai_run"


def test_requested_id_is_the_billing_key_and_served_name_is_kept():
    """`...-3b-instruct` answers as `...-3b-instruct-v2`. The requested id is what the price
    catalog is keyed by, so it is the one carried; the served name is not lost."""
    d = _load("01_chat_direct.json")
    u = extract_workers_ai_native(d["_response"], model_id=d["_model_id"])
    assert u.model == "@cf/meta/llama-3.2-3b-instruct"
    assert u.extras["served_model"] == "@cf/meta/llama-3.2-3b-instruct-v2"


def test_served_name_that_would_miss_the_catalog_does_not_become_the_model():
    """The case that decided the rule: Mistral small answers as `...-24b-v2` — `-instruct`
    dropped, `-v2` added — and that name is NOT in Cloudflare's catalog (measured
    2026-09-21) while the requested id is."""
    d = _load("04_mistral_small_direct.json")
    u = extract_workers_ai_native(d["_response"], model_id=d["_model_id"])
    assert u.model == "@cf/mistralai/mistral-small-3.1-24b-instruct"
    assert u.extras["served_model"] == "@cf/mistralai/mistral-small-3.1-24b-v2"
    assert (u.input, u.output) == (10, 19)


def test_served_name_equal_to_requested_adds_nothing_to_extras():
    d = _load("03_gpt_oss_direct.json")
    u = extract_workers_ai_native(d["_response"], model_id=d["_model_id"])
    assert u.model == "@cf/openai/gpt-oss-120b"
    assert "served_model" not in u.extras
    assert (u.input, u.output) == (73, 40)


def test_neurons_land_in_extras_not_in_a_metric():
    d = _load("01_chat_direct.json")
    u = extract_workers_ai_native(d["_response"], model_id=d["_model_id"])
    assert u.extras["neurons"] == pytest.approx(1.1343257427215576)
    assert "usage" not in u.extras  # every other key was recognised — no drift reported


def test_reasoning_model_bundles_thinking_into_completion():
    """deepseek-r1-distill reasons but reports no separate field — do not invent one."""
    d = _load("02_reasoning_direct.json")
    u = extract_workers_ai_native(d["_response"], model_id=d["_model_id"])
    assert u.input == 13
    assert u.output == 60
    assert u.reasoning == 0


def test_silent_response_keeps_the_requested_model():
    d = _load("02_reasoning_direct.json")
    assert d["_response"]["result"].get("model") is None
    u = extract_workers_ai_native(d["_response"], model_id=d["_model_id"])
    assert u.model == "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"
    assert "served_model" not in u.extras


def test_gateway_hit_and_miss_bodies_are_identical():
    """The gateway replays the cached body byte-for-byte, usage included. Nothing in the
    body distinguishes a HIT — only the header does, which is why the client, not the
    adapter, decides to skip billing."""
    miss = _load("06_chat_gateway_miss.json")
    hit = _load("07_chat_gateway_hit.json")
    assert miss["_headers"]["cf-aig-cache-status"] == "MISS"
    assert hit["_headers"]["cf-aig-cache-status"] == "HIT"
    um = extract_workers_ai_native(miss["_response"], model_id=miss["_model_id"])
    uh = extract_workers_ai_native(hit["_response"], model_id=hit["_model_id"])
    assert (um.input, um.output) == (uh.input, uh.output) == (41, 39)


# --------------------------------------------------------------------------
# Partner model — typesafe/jev via Cloudflare (05, BYOK) and straight from TypeSafe (08)
# --------------------------------------------------------------------------
def test_jev_via_cloudflare_is_wrapped_one_level_deeper():
    """`result: {state, result: {model, answers, usage}, gatewayMetadata}` — the partner's
    own object sits under `result.result`. Captured through the unified `/ai/run` path with
    `cf-aig-gateway-id` naming the gateway whose BYOK holds the TypeSafe key."""
    d = _load("05_jev_byok_gateway.json")
    r = d["_response"]["result"]
    assert r["state"] == "Completed" and "usage" in r["result"]
    u = extract_workers_ai_native(d["_response"], model_id=d["_model_id"])
    assert u.input == 446
    assert u.output == 73
    assert u.cache_read == 0 and u.reasoning == 0
    assert u.provider == "workers-ai" and u.api == "workers_ai_run"


def test_jev_keeps_its_catalog_id_and_reports_the_served_version():
    for name in ("05_jev_byok_gateway.json", "08_jev_typesafe_direct.json"):
        d = _load(name)
        u = extract_workers_ai_native(d["_response"], model_id=d["_model_id"])
        assert u.model == "typesafe/jev"
        assert u.extras["served_model"] == "jev-1.13.0"


def test_jev_byok_key_source_lands_in_extras():
    """Under BYOK Cloudflare charged nothing — TypeSafe bills the customer directly. Whoever
    reconciles against the Cloudflare dashboard needs to know which."""
    d = _load("05_jev_byok_gateway.json")
    u = extract_workers_ai_native(d["_response"], model_id=d["_model_id"])
    assert u.extras["gateway_metadata"] == {"keySource": "BYOK"}
    assert "usage" not in u.extras and "neurons" not in u.extras


def test_jev_from_typesafe_directly_is_the_same_object_unwrapped():
    d = _load("08_jev_typesafe_direct.json")
    assert "typesafe.ai" in d["_source"]
    u = extract_workers_ai_native(d["_response"], model_id=d["_model_id"])
    assert (u.input, u.output) == (446, 73)
    assert set(u.extras) == {"served_model"}


# --------------------------------------------------------------------------
# Every capture, and the failure envelope
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not _all(), reason="Workers AI fixtures not captured (run fixtures/capture_workers_ai.py)"
)
@pytest.mark.parametrize("path", _all(), ids=lambda p: p.stem)
def test_every_capture_is_a_success_that_bills(path: pathlib.Path):
    d = json.loads(path.read_text())
    assert d["_status"] == 200, f"{path.stem}: only successful responses are kept as fixtures"
    u = extract_workers_ai_native(d["_response"], model_id=d["_model_id"])
    assert u.provider == "workers-ai"
    assert u.input > 0 and u.output > 0, f"{path.stem}: adapter broken or capture stale"


def test_failure_envelope_yields_zero_usage_without_raising():
    """The shape Cloudflare returns on 402/403/400 (seen live: `result: {}` + `errors`). The
    client raises before billing; the adapter must still be safe to call on it."""
    body = {"errors": [{"message": "Insufficient balance", "code": 2021}], "success": False, "result": {}}
    u = extract_workers_ai_native(body, model_id="typesafe/jev")
    assert not u.nonzero_numeric()
    assert u.model == "typesafe/jev"

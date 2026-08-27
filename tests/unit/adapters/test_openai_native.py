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
# Model attribution — bill on what answered, not what was requested
# --------------------------------------------------------------------------
def test_model_resolves_to_response_value_not_request_alias() -> None:
    """OpenAI resolves a short alias to a dated snapshot in the response.

    Every non-streaming fixture in this suite shows this exact mismatch — e.g.
    `model_id="gpt-4o-mini"` was requested, but the response reports
    "gpt-4o-mini-2024-07-18". Pricing/attribution must key off what actually
    answered, or every alias-based call gets billed under the wrong model.
    """
    model_id, resp = _load("01_plain_chat.json")
    assert model_id == "gpt-4o-mini"  # sanity: the alias that was requested
    u = extract_openai_native(resp, model_id=model_id)
    assert u.model == "gpt-4o-mini-2024-07-18"  # the resolved model that actually answered


def test_model_falls_back_to_request_when_response_is_silent() -> None:
    """The synthetic usage blob the streaming wrapper builds carries no top-level
    `model` — fall back to the requested model rather than emitting an empty string."""
    u = extract_openai_native({"usage": {"prompt_tokens": 1, "completion_tokens": 1}}, model_id="gpt-4o-mini")
    assert u.model == "gpt-4o-mini"


def test_snowflake_cortex_keeps_the_customers_model_spelling() -> None:
    """A Cortex fine-tune answers as `database.schema.model`. CanonicalUsage.model
    keeps it verbatim — normalising here would report a model the customer cannot
    find in their own Snowflake account. (The hint is what a wrapped client whose
    base_url matches the Cortex path supplies — see _provider_hint_for.)"""
    u = extract_openai_native(
        {
            "model": "mydb.myschema.my_tuned_model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        },
        model_id="mydb.myschema.my_tuned_model",
        provider_hint="snowflake",
    )
    assert u.model == "mydb.myschema.my_tuned_model"
    assert u.provider == "snowflake"


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


def test_workers_ai_model_via_openai_sdk_infers_correct_provider() -> None:
    """Real shape: the openai SDK pointed at Cloudflare's `.../compat` endpoint,
    routed to a Workers AI model. The SDK shape looks identical to a real
    OpenAI response — "provider" can only be told apart by the resolved model
    string itself. Getting this wrong made Workers AI calls permanently
    unpriceable in price mode (stamped "openai", which has no Workers AI
    entries in its price table) — this is what fixed it."""
    resp = {
        "model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "usage": {"prompt_tokens": 38, "completion_tokens": 2},
    }
    u = extract_openai_native(resp, model_id="workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast")
    assert u.provider == "workers-ai"
    assert u.model == "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


def test_real_openai_model_still_gets_openai_provider() -> None:
    """The inference rule must not become over-eager — a genuine OpenAI model
    (no "@cf/" prefix) still gets "openai", unchanged."""
    resp = {"model": "gpt-4o-mini-2024-07-18", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    u = extract_openai_native(resp, model_id="gpt-4o-mini")
    assert u.provider == "openai"


# ----------------------------------------------------------------------
# Nested drift sweep + total_tokens consistency guard
# ----------------------------------------------------------------------


def test_cache_write_tokens_surfaces_in_extras_and_is_not_mapped() -> None:
    """Real captured `gpt-5.6-sol` shape: `prompt_tokens_details.cache_write_tokens`.

    Two assertions, and the second is the important one. The field must be
    SURFACED (it used to vanish entirely: `extras` swept only top-level keys and
    `prompt_tokens_details` is itself a known top-level key, so nothing nested
    was ever inspected). But it must NOT be mapped to `cache_write` — for OpenAI
    these tokens sit INSIDE `prompt_tokens` and bill at the plain input rate,
    while OpenRouter publishes a separate cache_write rate, so mapping them would
    charge the same 3022 tokens twice ($0.0341 against a true $0.0152, 2.24x).
    Anthropic is the opposite case, which is why mapping is right there.
    """
    resp = {
        "model": "gpt-5.6-sol",
        "usage": {
            "prompt_tokens": 3025,
            "completion_tokens": 4,
            "total_tokens": 3029,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 3022, "audio_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0},
        },
    }
    u = extract_openai_native(resp)
    assert u.extras["prompt_tokens_details.cache_write_tokens"] == 3022
    assert u.cache_write == 0, "cache_write_tokens must not be billed as cache_write for OpenAI"
    assert u.input == 3025


def test_predicted_output_details_surface_in_extras() -> None:
    """The module docstring promised customers could read the Predicted Outputs
    counts from extras. They never arrived, for the same nested-sweep reason.
    Now they do."""
    resp = {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "accepted_prediction_tokens": 7,
                "rejected_prediction_tokens": 3,
            },
        }
    }
    u = extract_openai_native(resp)
    assert u.extras["completion_tokens_details.accepted_prediction_tokens"] == 7
    assert u.extras["completion_tokens_details.rejected_prediction_tokens"] == 3


def test_total_tokens_guard_recovers_unaccounted_output() -> None:
    """Measured against Gemini through Google's own OpenAI-compatible layer:
    prompt=57, completion=47, total=1253. The 1149 thinking tokens are reported
    in NEITHER named bucket and there is no completion_tokens_details to recover
    them from — only `total_tokens` proves they exist. Billing prompt+completion
    drops 92% of the call, at the output rate.

    The remainder folds into `output`, deliberately NOT into `reasoning`:
    compute_cost zeroes reasoning whenever provider is in
    _OUTPUT_INCLUDES_REASONING, and an OpenAI-shaped payload is stamped
    provider="openai" by definition, so that would recover nothing."""
    resp = {
        "model": "gemini-2.5-flash",
        "usage": {"prompt_tokens": 57, "completion_tokens": 47, "total_tokens": 1253},
    }
    u = extract_openai_native(resp)
    assert u.input == 57
    assert u.output == 1196, "47 reported + 1149 unaccounted"
    assert u.extras["unaccounted_output_tokens"] == 1149


def test_total_tokens_guard_does_not_fold_an_additive_cache_write() -> None:
    """The payload shape raised in review on PY #14: a proxy reporting cache-creation
    tokens outside `prompt_tokens` but inside `total_tokens`. It was answered
    "unreachable on the three surfaces we have", which was true at the time —
    Snowflake Cortex then shipped the same class of payload with `cached_tokens`.
    The write is accounted for from the raw payload because `cache_write_tokens`
    is deliberately never mapped to CanonicalUsage.cache_write and so has no
    other route into the accounting — but only under a provider whose convention
    IS additive: for OpenAI itself the write sits inside prompt_tokens (see
    test_total_tokens_guard_survives_openai_cache_write_beside_a_remainder)."""
    u = extract_openai_native(
        {
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 4,
                "total_tokens": 1829,
                "prompt_tokens_details": {"cache_write_tokens": 1812},
            }
        },
        provider_hint="snowflake",
    )
    assert u.output == 4, "was 1816"
    assert "unaccounted_output_tokens" not in u.extras
    assert u.extras["prompt_tokens_details.cache_write_tokens"] == 1812


def test_total_tokens_guard_handles_an_additive_cache_write_on_the_responses_shape() -> None:
    """Same convention, other API branch: the Responses shape spells the container
    `input_tokens_details`, so a chat-only `prompt_tokens_details` lookup would
    leave this exact payload folding 1,812 cached-write tokens into `output` —
    the two API shapes must not disagree about one provider's convention."""
    u = extract_openai_native(
        {
            "usage": {
                "input_tokens": 13,
                "output_tokens": 4,
                "total_tokens": 1829,
                "input_tokens_details": {"cache_write_tokens": 1812},
            }
        },
        provider_hint="snowflake",
    )
    assert u.api == "responses"
    assert u.output == 4, "was 1816"
    assert "unaccounted_output_tokens" not in u.extras
    assert u.extras["input_tokens_details.cache_write_tokens"] == 1812


def test_total_tokens_guard_still_recovers_a_remainder_beside_a_cache_count() -> None:
    """The two corrections must not cancel each other: an additive cache block AND
    hidden thinking tokens in the same payload. 20 + 5 + 100 = 125 accounted,
    total 200, so 75 are real unreported output and must still fold."""
    u = extract_openai_native(
        {
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 200,
                "prompt_tokens_details": {"cached_tokens": 100},
            }
        },
        provider_hint="snowflake",
    )
    assert u.output == 80
    assert u.extras["unaccounted_output_tokens"] == 75


def test_total_tokens_guard_keeps_a_subtractive_fold_beside_a_cache_count() -> None:
    """The case that rules out subtracting the cache unconditionally, raised in
    review on #23: a SUBSET-convention surface reporting a cached block AND a
    genuine remainder. Gemini through Google's own OpenAI-compat layer reports
    `cached_tokens` inside `prompt_tokens` (that is why "gemini"/"openai" are in
    INPUT_INCLUDES_CACHE_READ) while thinking tokens appear only in the total —
    so the 1,000 cached tokens are ALREADY accounted for by prompt_tokens, and
    also adding them to the accounted sum would shrink the fold to 149: 1,000
    generated tokens unbilled, silently, with no on_error. The full 1,149 must
    fold."""
    u = extract_openai_native(
        {
            "model": "gemini-2.5-flash",
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 47,
                "total_tokens": 2396,
                "prompt_tokens_details": {"cached_tokens": 1000},
            },
        }
    )
    assert u.provider == "openai", "no hint: OpenAI-compat traffic is stamped openai"
    assert u.cache_read == 1000
    assert u.output == 1196, "47 reported + 1149 unaccounted — NOT 196"
    assert u.extras["unaccounted_output_tokens"] == 1149


def test_total_tokens_guard_survives_openai_cache_write_beside_a_remainder() -> None:
    """The documented-real OpenAI shape (see the NOTE on _MAPPED_DETAIL_FIELDS:
    prompt_tokens=3025 measured WITH cache_write_tokens=3022 inside it) behind a
    proxy that under-reports 1,171 tokens. OpenAI's write sits inside
    prompt_tokens, so it must NOT join the accounted sum — subtracting it
    unconditionally would swallow the delta and disarm the guard on the one
    payload shape this file documents as measured."""
    u = extract_openai_native(
        {
            "usage": {
                "prompt_tokens": 3025,
                "completion_tokens": 4,
                "total_tokens": 4200,
                "prompt_tokens_details": {"cache_write_tokens": 3022},
            }
        }
    )
    assert u.output == 1175, "4 reported + 1171 unaccounted — NOT 4"
    assert u.extras["unaccounted_output_tokens"] == 1171


def test_total_tokens_guard_is_a_noop_for_genuine_openai() -> None:
    """For real OpenAI total_tokens == prompt + completion always holds, because
    reasoning is a SUBSET of completion rather than additive. Verified across
    every captured real response — zero deltas. The guard must therefore never
    fire here, including for a reasoning model that spent its whole budget
    thinking."""
    for usage in (
        {
            "prompt_tokens": 31,
            "completion_tokens": 220,
            "total_tokens": 251,
            "completion_tokens_details": {"reasoning_tokens": 220},
        },
        {
            "prompt_tokens": 3026,
            "completion_tokens": 2,
            "total_tokens": 3028,
            "prompt_tokens_details": {"cached_tokens": 2816},
        },
        {"prompt_tokens": 16, "total_tokens": 16},  # embeddings: no completion_tokens at all
    ):
        u = extract_openai_native({"usage": usage})
        assert u.output == (usage.get("completion_tokens") or 0)
        assert "unaccounted_output_tokens" not in u.extras


def test_total_tokens_guard_ignores_a_negative_delta() -> None:
    """A total SMALLER than the parts is nonsense, not drift — never subtract."""
    u = extract_openai_native({"usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 10}})
    assert u.output == 50
    assert "unaccounted_output_tokens" not in u.extras


# --------------------------------------------------------------------------
# Snowflake Cortex — an OpenAI-wire endpoint with ADDITIVE cache
#
# Cortex answers on `/api/v2/cortex/v1/chat/completions` with OpenAI's exact
# payload shape, so this adapter serves it — but it does NOT follow OpenAI's
# token convention. Captured live 2026-08-25 by capture_snowflake_cortex.py;
# never hand-edit these numbers, recapture instead.
# --------------------------------------------------------------------------
def test_snowflake_cortex_plain_call() -> None:
    """No cache: total reconciles to prompt + completion, guard never fires."""
    model_id, resp = _load("11_snowflake_cortex_plain_chat.json")
    u = extract_openai_native(resp, model_id=model_id, provider_hint="snowflake")
    assert u.input == 21
    assert u.output == 4
    assert u.cache_read == 0
    assert u.provider == "snowflake"
    assert u.api == "chat_completions"
    assert "unaccounted_output_tokens" not in u.extras


def test_snowflake_cortex_cached_tokens_are_additive() -> None:
    """THE regression. 7 + 4805 + 6 = 4818, so under the old accounting (input +
    output + reasoning only) the 4,805 cached tokens looked unaccounted and were
    folded into `output`: 4,811 reported for a call that generated 6, while the
    same tokens also shipped as cache_read — 2.0x on the call, 800x on the output
    line. Revert the cache subtraction in openai_native.py and this fails on
    `output`.

    This exact hazard was raised in review on PY #14 (2026-08-17) and answered
    "measured 0 on all three surfaces we have" — true then. Cortex is the surface
    that did not exist yet."""
    model_id, resp = _load("12_snowflake_cortex_cache_chat.json")
    usage = resp["usage"]
    # Read the cached count off the fixture rather than pinning a literal: the
    # assertion is the additive IDENTITY, so a recapture with a different cached
    # count must keep passing instead of nudging someone toward the hand-edit
    # the header above forbids.
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    assert cached > 0, "recapture produced no cached block — see capture_snowflake_cortex.py"
    assert usage["prompt_tokens"] + cached + usage["completion_tokens"] == usage["total_tokens"]

    u = extract_openai_native(resp, model_id=model_id, provider_hint="snowflake")
    assert u.input == usage["prompt_tokens"]
    assert u.output == usage["completion_tokens"], "NOT completion + cached"
    assert u.cache_read == cached
    # cache_write stays unmapped BY DESIGN (u.cache_write is 0 on every path, so
    # asserting it proves nothing) — the load-bearing check is that the raw key
    # is still visible in extras rather than silently consumed by the guard.
    assert u.extras["prompt_tokens_details.cache_write_tokens"] == 0
    assert u.reasoning == 0
    assert "unaccounted_output_tokens" not in u.extras


def test_snowflake_cortex_without_a_hint_is_stamped_openai_and_folds() -> None:
    """Pins what REAL traffic does until the wrapper carries a Cortex base_url
    rule (PR #26 adds `/api/v2/cortex/` → "snowflake" to _provider_hint_for):
    the response body has no marker of its own, so an unhinted Cortex payload is
    stamped "openai", whose SUBSET convention folds the additive cached block
    into `output` again. This is deliberate — the payload cannot carry the
    convention, so identification is the fix, not looser arithmetic. If this
    test starts failing because the fold stopped, the guard has been loosened
    for every genuine OpenAI-compat proxy; if it fails on `provider`, the hint
    now reaches this adapter by default and the test should assert the fixed
    behaviour instead."""
    model_id, resp = _load("12_snowflake_cortex_cache_chat.json")
    usage = resp["usage"]
    cached = usage["prompt_tokens_details"]["cached_tokens"]

    u = extract_openai_native(resp, model_id=model_id)
    assert u.provider == "openai"
    assert u.output == usage["completion_tokens"] + cached
    assert u.extras["unaccounted_output_tokens"] == cached

"""Property-based fuzzing — adapters never crash and never produce negatives."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from lago_agent_sdk.adapters import (
    extract_bedrock_converse,
    extract_bedrock_invoke,
    pick_invoke_adapter,
)
from lago_agent_sdk.canonical import CanonicalUsage

_garbage = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(-1_000_000, 1_000_000),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=20),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(min_size=1, max_size=10), children, max_size=5),
    ),
    max_leaves=20,
)

_some_model_ids = st.sampled_from(
    [
        "eu.anthropic.claude-sonnet-4-6",
        "eu.anthropic.claude-opus-4-7",
        "eu.amazon.nova-lite-v1:0",
        "openai.gpt-oss-20b-1:0",
        "openai.gpt-oss-safeguard-20b-1:0",
        "eu.mistral.pixtral-large-2502-v1:0",
        "mistral.mistral-large-2402-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
        "eu.minimax.minimax-m2-v1:0",
        "eu.qwen.qwen3-235b-a22b-instruct-2507-v1:0",
        "",  # also try an empty model id
    ]
)


def _assert_canonical_invariants(u: CanonicalUsage) -> None:
    assert isinstance(u, CanonicalUsage)
    for f in CanonicalUsage.NUMERIC_FIELDS:
        v = getattr(u, f)
        assert isinstance(v, int)
        assert v >= 0, f"{f} went negative: {v}"
    assert isinstance(u.extras, dict)


@given(garbage=_garbage, model_id=_some_model_ids)
@settings(max_examples=300, deadline=None)
def test_converse_adapter_survives_random_input(garbage, model_id):
    if not isinstance(garbage, dict):
        garbage = {"usage": garbage}
    u = extract_bedrock_converse(garbage, model_id=model_id)
    _assert_canonical_invariants(u)


@given(garbage=_garbage, model_id=_some_model_ids)
@settings(max_examples=300, deadline=None)
def test_invoke_adapter_survives_random_input(garbage, model_id):
    if not isinstance(garbage, dict):
        garbage = {"usage": garbage}
    u = extract_bedrock_invoke(garbage, model_id=model_id)
    _assert_canonical_invariants(u)


@given(model_id=st.text(max_size=80))
@settings(max_examples=300, deadline=None)
def test_pick_invoke_adapter_returns_known_family_for_any_string(model_id):
    family = pick_invoke_adapter(model_id)
    assert family in {
        "openai_compat_basic",
        "openai_compat_with_details",
        "anthropic",
        "opus_4_7",
        "nova",
        "pixtral",
        "mistral_legacy",
    }

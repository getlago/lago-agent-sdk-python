"""token_semantics — the one table three billing paths answer from.

These tests pin the DECISIONS, not the mechanism: each provider's entry (or
deliberate absence) traces to a measurement recorded in token_semantics.py, and
a provider the SDK can stamp without a recorded decision is a red test here —
absence must always be a choice somebody made.
"""

from __future__ import annotations

from lago_agent_sdk.token_semantics import (
    INPUT_INCLUDES_CACHE_READ,
    INPUT_INCLUDES_CACHE_WRITE,
    KNOWN_PROVIDERS,
    OPENAI_SHAPED_APIS,
    OUTPUT_INCLUDES_REASONING,
    token_semantics,
)

# Every provider string the SDK's own code can stamp on a CanonicalUsage today.
# Kept explicit rather than scraped from the source: when an adapter or wrapper
# grows a new stamp, add it BOTH here and (with its measured decision) to
# KNOWN_PROVIDERS — this list failing to cover a stamp is exactly the silent
# default the roster exists to prevent. Gateway backfills additionally pass
# vendor names through from their logs verbatim; those arrive with a surface
# `api` and are decided by OPENAI_SHAPED_APIS (or the vendor's own entry), not
# by this list.
_STAMPABLE = {
    # adapters/openai_native.py: _infer_provider
    "openai",
    "workers-ai",
    # wrappers/openai.py: _provider_hint_for (base_url table)
    "databricks",
    "snowflake",
    # adapters/anthropic_native.py, gemini_native.py, mistral_native.py
    "anthropic",
    "gemini",
    "mistral",
    # adapters/bedrock_converse.py / bedrock_invoke.py: _provider_from_model
    "amazon",
    "meta",
    "cohere",
    "qwen",
    "google",
    "minimax",
    "nvidia",
    "zai",
    "bedrock",
}


def test_every_stampable_provider_has_a_recorded_decision() -> None:
    missing = _STAMPABLE - KNOWN_PROVIDERS
    assert not missing, f"providers stamped by the SDK with no semantics decision: {sorted(missing)}"


def test_the_subset_sets_only_name_known_providers() -> None:
    """A set entry for a name nothing can stamp is dead weight at best and a
    typo silently reverting a measured decision at worst."""
    for s in (INPUT_INCLUDES_CACHE_READ, INPUT_INCLUDES_CACHE_WRITE, OUTPUT_INCLUDES_REASONING):
        assert s <= KNOWN_PROVIDERS


def test_openai_convention_is_subset_on_all_three_dimensions() -> None:
    assert token_semantics("openai", "chat_completions") == (True, True, True)


def test_snowflake_convention_is_additive_on_all_three_dimensions() -> None:
    """Measured live 2026-08-25: prompt 7 / cached 4805 / completion 6 /
    total 4818 — Anthropic's additive convention on OpenAI's wire. This single
    row is what the total_tokens guard, compute_cost and
    deoverlapped_token_total all read; if it ever flips, all three flip
    together or 4,805 tokens bill twice."""
    assert token_semantics("snowflake", "chat_completions") == (False, False, False)


def test_anthropic_convention_is_additive() -> None:
    assert token_semantics("anthropic", "messages") == (False, False, False)


def test_gemini_cache_is_subset_but_reasoning_is_additive() -> None:
    """cachedContentTokenCount ⊆ promptTokenCount, thoughtsTokenCount additive —
    Google documents totalTokenCount = prompt + thoughts + candidates."""
    assert token_semantics("gemini", "generate_content") == (True, True, False)


def test_an_openai_shaped_surface_overrides_the_vendor() -> None:
    """A provider="anthropic" ROW from Databricks' system table is in the
    gateway's re-reported shape — everything subset — while the same vendor
    name from its own API is additive. The surface wins."""
    assert token_semantics("anthropic", "databricks_gateway") == (True, True, True)
    assert "databricks_gateway" in OPENAI_SHAPED_APIS


def test_an_unknown_provider_defaults_to_additive() -> None:
    """No overlap is removed for a name nobody measured: the conservative
    direction — it can over-count a subset into a token total, but it can never
    silently drop generated tokens or zero a real cache line."""
    assert token_semantics("some-new-gateway", "chat_completions") == (False, False, False)
    assert token_semantics("", "") == (False, False, False)

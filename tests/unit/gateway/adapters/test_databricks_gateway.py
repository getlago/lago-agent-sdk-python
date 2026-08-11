"""Databricks AI Gateway usage adapter — verified against real captured table rows.

Fixtures were read from a live workspace's `system.ai_gateway.usage` over the SQL
Statement Execution API, one file per scenario, exactly as the adapter receives them.
"""

from __future__ import annotations

import json
import pathlib

from lago_agent_sdk.gateway.adapters import extract_databricks_log, resolve_databricks_subscription

FIX = pathlib.Path(__file__).parent / "fixtures" / "databricks_gateway"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text())


# --------------------------------------------------------------------------
# Real fixtures — the two destination types
# --------------------------------------------------------------------------
def test_real_hosted_chat_row() -> None:
    """A Databricks-hosted (pay-per-token) foundation model via the mlflow surface."""
    u = extract_databricks_log(_load("hosted_chat.json"))
    assert u.input == 11
    assert u.output == 4
    assert u.model == "llama-4-maverick"
    assert u.provider == "databricks"
    assert u.api == "databricks_gateway"


def test_real_hosted_embeddings_row() -> None:
    """Embeddings report input only — `output_tokens` is NULL, not 0, and must not
    become a phantom output event."""
    u = extract_databricks_log(_load("hosted_embeddings.json"))
    assert u.input == 13
    assert u.output == 0
    assert u.provider == "databricks"
    assert u.extras["api_type"] == "mlflow/v1/embeddings"


def test_real_byok_anthropic_row() -> None:
    """BYOK: the model comes from `destination_model` and the provider from the
    leading segment of `api_type`."""
    u = extract_databricks_log(_load("byok_anthropic_cache_read.json"))
    assert u.model == "claude-sonnet-4-5"
    assert u.provider == "anthropic"
    assert u.extras["destination_type"] == "EXTERNAL_FOUNDATION_MODEL"


def test_real_byok_openai_reasoning_row() -> None:
    """`token_details.output_reasoning_tokens` IS broken out in the table, even
    though the mlflow response body reports no reasoning at all — the live and
    backfill paths genuinely disagree on this field."""
    u = extract_databricks_log(_load("byok_openai_reasoning.json"))
    assert u.provider == "openai"
    assert u.reasoning == 220
    assert u.output == 220


# --------------------------------------------------------------------------
# The two naming quirks that a docs-only reading gets wrong
# --------------------------------------------------------------------------
def test_hosted_model_comes_from_destination_name_not_destination_model() -> None:
    """For hosted rows `destination_model` is unstable — the same
    `destination_name` was observed reporting both `llama-4-maverick` and the
    display label `Llama 4 Maverick`. `destination_name` is the stable id, so it
    wins, with the `system.ai.` prefix stripped."""
    row = {
        "destination_type": "PAY_PER_TOKEN_FOUNDATION_MODEL",
        "destination_name": "system.ai.gpt-oss-20b",
        "destination_model": "GPT OSS 20B",  # display label, spaces and capitals
        "api_type": "mlflow/v1/chat/completions",
        "input_tokens": "102",
        "output_tokens": "4",
    }
    u = extract_databricks_log(row)
    assert u.model == "gpt-oss-20b"
    assert u.provider == "databricks"


def test_hosted_destination_name_sheds_its_endpoint_prefix() -> None:
    """Most hosted entities are named `system.ai.databricks-<model>`, not
    `system.ai.<model>` — measured on a live workspace, 38 of 48 distinct hosted
    `destination_name`s carry that inner `databricks-`. It is a serving-endpoint
    artefact, not part of the model id: leaving it in emits
    `databricks-qwen35-122b-a10b`, which both reads as a vendor prefix and splits
    one model into two rows in Lago against the live path's own name.

    Real captured row, not hand-written."""
    u = extract_databricks_log(_load("hosted_chat_endpoint_prefixed_name.json"))
    assert u.model == "qwen35-122b-a10b"
    assert u.provider == "databricks"
    assert u.input == 37 and u.output == 200
    # The raw name stays visible for reconciliation against Databricks' own console.
    assert u.extras["destination_name"] == "system.ai.databricks-qwen35-122b-a10b"


def test_hosted_prefix_stripping_does_not_rename_a_genuinely_databricks_model() -> None:
    """Databricks publishes models whose own names start with `databricks-`
    (`databricks-dbrx-instruct`, `databricks-dolly-v2`), so an unconditional strip
    would rename them. `destination_model` is the tie-breaker: it agrees with the
    shed form when the prefix is an endpoint artefact, and with the full name when
    the model is really called that."""
    artefact = {
        "destination_type": "PAY_PER_TOKEN_FOUNDATION_MODEL",
        "destination_name": "system.ai.databricks-claude-sonnet-4-5",
        "destination_model": "claude-sonnet-4-5",
        "api_type": "mlflow/v1/chat/completions",
        "input_tokens": "5",
        "output_tokens": "5",
    }
    assert extract_databricks_log(artefact).model == "claude-sonnet-4-5"

    real_name = {**artefact, "destination_name": "system.ai.databricks-dbrx-instruct"}
    real_name["destination_model"] = "databricks-dbrx-instruct"
    assert extract_databricks_log(real_name).model == "databricks-dbrx-instruct"

    # Disagreement (the unstable display-label case) keeps the raw name rather than
    # guessing — an ugly id beats a wrong one.
    ambiguous = {**artefact, "destination_model": "Claude Sonnet 4.5"}
    assert extract_databricks_log(ambiguous).model == "databricks-claude-sonnet-4-5"


def test_byok_never_uses_destination_name_as_the_model() -> None:
    """For BYOK rows `destination_name` is the PROVIDER SERVICE — a Unity Catalog
    credential name, not a model. Falling back to it would bill
    `workspace.default.anthropickey` as the model on every BYOK row."""
    row = {
        "destination_type": "EXTERNAL_FOUNDATION_MODEL",
        "destination_name": "workspace.default.anthropickey",
        "destination_model": "claude-opus-4-5",
        "api_type": "anthropic/v1/messages",
        "input_tokens": "16",
        "output_tokens": "47",
    }
    u = extract_databricks_log(row)
    assert u.model == "claude-opus-4-5"
    assert "workspace.default" not in u.model
    assert u.extras["destination_name"] == "workspace.default.anthropickey"


def test_provider_is_derived_from_api_type_leading_segment() -> None:
    """`api_type` is the full ingress path, and its leading segment already IS this
    SDK's provider vocabulary — so no alias table is needed."""
    for api_type, expected in (
        ("anthropic/v1/messages", "anthropic"),
        ("openai/v1/chat/completions", "openai"),
        ("gemini/v1/generateContent", "gemini"),
        ("unmanaged", "unmanaged"),
    ):
        u = extract_databricks_log({"destination_type": "EXTERNAL_FOUNDATION_MODEL", "api_type": api_type})
        assert u.provider == expected


def test_hosted_provider_cannot_match_a_vendor_price_table() -> None:
    """`provider="databricks"` is deliberate: it matches no vendor in pricing's
    _VENDOR_MAP, so the lookup CANNOT hit and emit() falls back to token events.
    OpenRouter does list bare `openai/gpt-oss-20b` at ~0.4x of Databricks' own DBU
    rate, so an accidental match would under-bill 2.5-5x."""
    from lago_agent_sdk.pricing import lookup_openrouter, parse_openrouter

    table = parse_openrouter({"data": [{"id": "openai/gpt-oss-20b", "pricing": {"prompt": "0.00000003"}}]})
    u = extract_databricks_log(_load("hosted_chat.json"))
    assert lookup_openrouter(table, u.provider, u.model) is None


# --------------------------------------------------------------------------
# STRUCT / MAP columns arrive as JSON strings over the REST API
# --------------------------------------------------------------------------
def test_token_details_parses_from_a_json_string() -> None:
    """The SQL drivers hand back real dicts, but the Statement Execution API
    serializes STRUCT columns as JSON strings. Both must work, or the adapter
    silently reads zeros from a string it never parsed."""
    as_string = {
        "destination_type": "EXTERNAL_FOUNDATION_MODEL",
        "api_type": "anthropic/v1/messages",
        "destination_model": "claude-sonnet-4-5",
        "input_tokens": "1825",
        "output_tokens": "4",
        "token_details": '{"cache_read_input_tokens":"1812","cache_creation_input_tokens":null}',
    }
    as_dict = {**as_string, "token_details": {"cache_read_input_tokens": 1812}}
    for row in (as_string, as_dict):
        u = extract_databricks_log(row)
        assert u.cache_read == 1812
        assert u.cache_write == 0


def test_input_tokens_includes_cache_so_the_difference_is_recoverable() -> None:
    """Measured, and the inverse of every provider's own response body: this table's
    `input_tokens` INCLUDES cache_read and cache_write. The fixture pair below came
    from calls whose response bodies reported `input_tokens: 9`.

    The adapter extracts faithfully rather than subtracting — billing takes
    Databricks' own metered USD, which never touches these counts. This test pins
    that the arithmetic stays recoverable: only one of read/write is ever non-zero,
    so input - read - write is the true non-cached input.
    """
    for name in ("byok_anthropic_cache_read.json", "byok_anthropic_cache_write.json"):
        u = extract_databricks_log(_load(name))
        assert not (u.cache_read and u.cache_write), "only one direction per row"
        assert u.input - u.cache_read - u.cache_write == 9


def test_request_tags_parses_from_a_json_string_too() -> None:
    for tags in ('{"lago_subscription":"sub_acme","team":"x"}', {"lago_subscription": "sub_acme"}):
        assert resolve_databricks_subscription({"request_tags": tags}) == "sub_acme"


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------
def test_real_row_resolves_its_subscription() -> None:
    assert resolve_databricks_subscription(_load("byok_openai_cache_read.json")) == "sub_openai"


def test_untagged_row_has_no_subscription() -> None:
    """Untagged calls do produce rows, with `request_tags` empty. Attribution is
    absent, and what to do about that is the caller's decision."""
    # `hosted_chat.json` IS the untagged capture — its `request_tags` is `{}`. A separate
    # `untagged.json` existed and was byte-identical, so it is gone rather than kept as a
    # second name for the same bytes.
    assert resolve_databricks_subscription(_load("hosted_chat.json")) is None


def test_missing_or_malformed_request_tags_resolve_to_none() -> None:
    for tags in (None, "{}", {}, "not json", [], 7, {"lago_subscription": ""}):
        assert resolve_databricks_subscription({"request_tags": tags}) is None
    assert resolve_databricks_subscription({}) is None


# --------------------------------------------------------------------------
# Failure rows must bill nothing
# --------------------------------------------------------------------------
def test_failed_rows_extract_to_zero_so_nothing_is_billed() -> None:
    """Failed calls are recorded with NULL token counts. They must extract to
    all-zero, leaving `nonzero_numeric()` empty so the caller emits nothing — the
    same way a Cloudflare cache hit extracts to zero."""
    for name in ("failed_null_tokens.json", "gemini_broken.json", "unmanaged_path.json"):
        u = extract_databricks_log(_load(name))
        assert u.nonzero_numeric() == {}


# --------------------------------------------------------------------------
# Robustness — one malformed row must not take down a batch
# --------------------------------------------------------------------------
def test_empty_row_is_all_zero() -> None:
    u = extract_databricks_log({})
    assert u.nonzero_numeric() == {}
    assert u.model == ""
    assert u.provider == ""
    assert u.api == "databricks_gateway"


def test_negative_and_non_numeric_counts_clamp_to_zero() -> None:
    u = extract_databricks_log({"input_tokens": -5, "output_tokens": "bogus", "total_tokens": "9"})
    assert u.input == 0
    assert u.output == 0


def test_non_string_model_and_destination_fields_do_not_crash() -> None:
    u = extract_databricks_log(
        {"destination_type": 7, "destination_name": [], "destination_model": {}, "api_type": None}
    )
    assert u.model == ""
    assert u.provider == ""


def test_total_tokens_is_not_mapped() -> None:
    """It is derived from input+output; mapping it would double-count. Same reason
    the Cloudflare adapter skips `usage_metadata.total_tokens`."""
    u = extract_databricks_log({"input_tokens": "10", "output_tokens": "5", "total_tokens": "15"})
    assert u.nonzero_numeric() == {"input": 10, "output": 5}


# --------------------------------------------------------------------------
# Sweep — every captured fixture must extract cleanly
# --------------------------------------------------------------------------
def test_all_captured_fixtures_extract() -> None:
    """Iterate the whole fixture directory, mirroring `test_all_models_sweep`.

    Without this, a capture that no named test mentions asserts nothing — 12 of the
    files here were in exactly that state, shipped and inert. A sweep also means the
    next capture is covered the moment it lands, rather than when someone remembers to
    write a test for it. Skips cleanly if the directory is absent, so a missing capture
    reads as "not covered" rather than as a pass.
    """
    fixtures = sorted(FIX.glob("*.json"))
    if not fixtures:
        import pytest

        pytest.skip("no databricks_gateway fixtures captured")

    for path in fixtures:
        row = json.loads(path.read_text())
        u = extract_databricks_log(row)
        assert u.api == "databricks_gateway", path.name
        # Every numeric field is a count: never negative, never a float.
        for field_name in u.NUMERIC_FIELDS:
            value = getattr(u, field_name)
            assert isinstance(value, int) and value >= 0, f"{path.name}:{field_name}={value!r}"
        # A row with tokens must name a model; a row without is a failure/rejected row.
        if u.nonzero_numeric():
            assert u.model, f"{path.name} has tokens but no model"
            assert u.provider, f"{path.name} has tokens but no provider"
        # The subscription resolver must never raise on a real row, whatever its tags.
        resolve_databricks_subscription(row)


def test_no_two_fixtures_are_byte_identical() -> None:
    """A duplicate file is a second name for the same evidence, and it lies about
    coverage: three pairs existed here, one of which ("plain" Anthropic BYOK) was
    actually the cache-write capture, so the scenario it claimed to hold had never
    been captured at all."""
    import hashlib

    seen: dict[str, str] = {}
    for path in sorted(FIX.glob("*.json")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest not in seen, f"{path.name} is byte-identical to {seen[digest]}"
        seen[digest] = path.name

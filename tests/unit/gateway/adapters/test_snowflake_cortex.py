"""Snowflake Cortex REST adapter — verified against real captured view rows.

Fixtures were read from a live account's `CORTEX_REST_API_USAGE_HISTORY` over the SQL
API, one file per scenario, exactly as the adapter receives them. Hand-made rows appear
only where the live surface cannot produce the shape (a failed row, malformed JSON, a
fine-tuned model) and say so.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from lago_agent_sdk.gateway.adapters import (
    extract_snowflake_rest_log,
    resolve_snowflake_subscription,
)
from lago_agent_sdk.pricing import deoverlapped_token_total

FIX = pathlib.Path(__file__).parent / "fixtures" / "snowflake_cortex"

REST_FIXTURES = sorted(p.name for p in FIX.glob("rest_*.json"))


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIX / name).read_text())


# --------------------------------------------------------------------------
# Real fixtures
# --------------------------------------------------------------------------
def test_real_plain_row() -> None:
    u = extract_snowflake_rest_log(_load("rest_plain.json"))
    assert u.input == 9
    assert u.output == 12
    assert u.cache_read == 0
    assert u.cache_write == 0
    assert u.model == "claude-sonnet-4-5"
    assert u.provider == "snowflake"
    assert u.api == "snowflake_cortex_rest"


def test_real_cache_read_row_is_additive() -> None:
    """`input` EXCLUDES the cached block on this view — the whole billing question.

    8 fresh input tokens, 4,684 served from cache, 6 generated, and `TOKENS: 4698`
    is their sum. Reading that total as the input count, or assuming OpenAI's
    convention where the cached block sits inside `input`, bills 2.0x.
    """
    u = extract_snowflake_rest_log(_load("rest_cache_read.json"))
    assert u.input == 8
    assert u.cache_read == 4684
    assert u.cache_write == 0
    assert u.output == 6


def test_real_cache_write_row() -> None:
    """The write half of a matched pair: the same prompt, sent twice, reported
    identically by the endpoint (`cached_tokens: 4684, cache_write_tokens: 0` both
    times) and split by the view into a write row and a read row."""
    u = extract_snowflake_rest_log(_load("rest_cache_write.json"))
    assert u.cache_write == 4684
    assert u.cache_read == 0
    assert u.input == 8
    assert u.output == 6


def test_real_thinking_row_reports_no_reasoning() -> None:
    """Extended thinking is INSIDE `output` here and has no key of its own.

    The same call reported `thinking_tokens: 127` of 262 output tokens on
    Snowflake's Anthropic wire; the view reports only `{input: 60, output: 262}`.
    A `reasoning` count invented from that would be billed twice.
    """
    u = extract_snowflake_rest_log(_load("rest_thinking.json"))
    assert u.output == 262
    assert u.reasoning == 0
    assert u.input == 60


def test_every_real_row_reconciles_against_snowflakes_own_total() -> None:
    """`TOKENS` == the SDK's own de-overlapped total, on every captured row.

    This is the reconciliation INT-231 exists to prove, asserted here on the two
    inputs it depends on: that the adapter maps the granular keys, and that nothing
    downstream treats this surface as OpenAI-shaped. Adding
    `snowflake_cortex_rest` to `_OPENAI_SHAPED_APIS` — or `snowflake` to
    `_INPUT_INCLUDES_CACHE_READ` — zeroes `cache_read` here and drops a cached row
    from 4698 to 14, which this fails on.
    """
    assert REST_FIXTURES, "no rest_*.json fixtures — capture is missing, not passing"
    for name in REST_FIXTURES:
        row = _load(name)
        usage = extract_snowflake_rest_log(row)
        assert deoverlapped_token_total(usage) == int(row["TOKENS"]), name


def test_tokens_column_is_never_mapped_to_a_token_field() -> None:
    """`TOKENS` is the additive total; mapping it double-bills the cached block.

    It reaches `extras` so a reconciliation can read it, and no numeric field.
    Mirrors the Databricks adapter's refusal to map that table's `total_tokens`.
    """
    row = _load("rest_cache_read.json")
    u = extract_snowflake_rest_log(row)
    assert u.extras["tokens"] == row["TOKENS"]
    assert u.input + u.output + u.cache_read + u.cache_write == int(row["TOKENS"])
    assert u.input != int(row["TOKENS"])


def test_non_token_columns_reach_extras() -> None:
    u = extract_snowflake_rest_log(_load("rest_cache_write.json"))
    assert u.extras["request_id"] == "7d1649a5-1460-4786-acac-5dd74666d9c7"
    assert u.extras["inference_region"] == ""
    assert u.extras["user_id"] == "1"
    assert u.extras["query_tag"] is None
    assert u.extras["start_time"] == "1787680800.000000000 1440"
    assert u.extras["end_time"] == "1787684400.000000000 1440"


# --------------------------------------------------------------------------
# Shapes the live surface cannot produce — hand-made rows, labelled as such
# --------------------------------------------------------------------------
def test_failed_row_extracts_to_all_zero() -> None:
    """A failed call produces NO row on this view (driven live: a 403 and a 400
    alongside a success; only the success appeared). This row is therefore
    hypothetical — it guards the shape a mid-generation failure would take."""
    u = extract_snowflake_rest_log(
        {
            "REQUEST_ID": "00000000-0000-0000-0000-000000000000",
            "MODEL_NAME": "claude-sonnet-4-5",
            "TOKENS": None,
            "TOKENS_GRANULAR": None,
            "USER_ID": "1",
        }
    )
    assert u.nonzero_numeric() == {}
    assert u.model == "claude-sonnet-4-5"
    # No `TOKENS` to contradict the zeros, so this is a genuinely empty row rather
    # than usage we failed to split — the marker must stay off.
    assert "tokens_granular_missing" not in u.extras


def test_positive_tokens_with_no_granular_is_marked_not_silently_dropped() -> None:
    """Real usage the adapter cannot split extracts to all-zero, so nothing is
    emitted — a 100% under-bill that must not look like a correctly-ignored failure."""
    u = extract_snowflake_rest_log({"TOKENS": "4818", "TOKENS_GRANULAR": None})
    assert u.nonzero_numeric() == {}
    assert u.extras["tokens_granular_missing"] is True


def test_granular_holding_only_unknown_keys_is_marked_too() -> None:
    """The drift sweep alone would let this pass quietly: every count is preserved in
    `extras`, but nothing is billable and `TOKENS` says the call consumed 500."""
    u = extract_snowflake_rest_log({"TOKENS": "500", "TOKENS_GRANULAR": '{"audio_input": 500}'})
    assert u.nonzero_numeric() == {}
    assert u.extras["tokens_granular.audio_input"] == 500
    assert u.extras["tokens_granular_missing"] is True


def test_malformed_granular_json_degrades_without_throwing() -> None:
    u = extract_snowflake_rest_log(
        {"MODEL_NAME": "llama3.1-70b", "TOKENS": "20", "TOKENS_GRANULAR": "{not json"}
    )
    assert u.input == 0
    assert u.output == 0
    assert u.model == "llama3.1-70b"
    assert u.extras["tokens"] == "20"


def test_granular_accepted_as_an_object_not_only_a_json_string() -> None:
    """The SQL API serializes OBJECT columns as TEXT; a typed connector hands back a
    real dict. Reading zeros out of one of them is a silent 100% under-bill."""
    as_string = extract_snowflake_rest_log(_load("rest_cache_read.json"))
    row = _load("rest_cache_read.json")
    row["TOKENS_GRANULAR"] = json.loads(row["TOKENS_GRANULAR"])
    as_object = extract_snowflake_rest_log(row)
    assert as_object.nonzero_numeric() == as_string.nonzero_numeric()


def test_unmapped_granular_key_reaches_extras_and_is_not_counted() -> None:
    """A new key here is a token count nobody has classified. It must not be
    miscounted as one of the four mapped fields, and it must not vanish — this view
    grew two cache keys and a whole column inside one day."""
    u = extract_snowflake_rest_log(
        {
            "TOKENS": "1300",
            "TOKENS_GRANULAR": (
                '{"input": 1000, "output": 200, "cache_read_input": 100, "image_input_tokens": 42}'
            ),
        }
    )
    assert u.input == 1000
    assert u.output == 200
    assert u.cache_read == 100
    assert u.extras["tokens_granular.image_input_tokens"] == 42
    assert u.image_input == 0
    assert "tokens_granular.input" not in u.extras


def test_fine_tuned_model_keeps_the_customers_spelling() -> None:
    """Fully-qualified `database.schema.model`. Normalizing for a price lookup is
    pricing's job; what Lago is told the model was must be what the customer typed.
    Hand-made — only four bare models are live on the capture account."""
    u = extract_snowflake_rest_log(
        {"MODEL_NAME": "MY_DB.MY_SCHEMA.my_tuned_llama", "TOKENS_GRANULAR": '{"input": 5}'}
    )
    assert u.model == "MY_DB.MY_SCHEMA.my_tuned_llama"


def test_lowercase_column_keys_are_accepted() -> None:
    """Unquoted identifiers arrive UPPERCASE, but a quoted lowercase alias — or a
    caller who normalized keys — must not extract zeros from a row that has values."""
    u = extract_snowflake_rest_log(
        {"model_name": "claude-opus-4-5", "tokens": "16", "tokens_granular": '{"input": 8, "output": 8}'}
    )
    assert u.input == 8
    assert u.output == 8
    assert u.model == "claude-opus-4-5"


def test_never_throws_on_a_malformed_row() -> None:
    """A backfill runs a window of rows through this; one bad row must not take the
    run down."""
    for row in ({}, {"TOKENS_GRANULAR": 7}, {"TOKENS_GRANULAR": []}, {"MODEL_NAME": 42}):
        u = extract_snowflake_rest_log(row)  # type: ignore[arg-type]
        assert u.nonzero_numeric() == {}
        assert u.api == "snowflake_cortex_rest"


# --------------------------------------------------------------------------
# Subscription resolution
# --------------------------------------------------------------------------
def test_subscription_from_query_tag() -> None:
    assert resolve_snowflake_subscription({"QUERY_TAG": '{"lago_subscription": "sub_123"}'}) == "sub_123"


def test_query_tag_without_the_key_resolves_nothing_from_that_source() -> None:
    """Snowflake writes its own QUERY_TAGs — a captured row carries
    `{"app": "cortex_code_sandbox", ...}`. Treating an arbitrary tag as a
    subscription id bills somebody's tooling label to a customer."""
    row = {"QUERY_TAG": '{"app": "cortex_code_sandbox"}'}
    assert resolve_snowflake_subscription(row, order=("query_tag",)) is None


def test_non_json_query_tag_is_ignored() -> None:
    assert resolve_snowflake_subscription({"QUERY_TAG": "nightly-etl"}, order=("query_tag",)) is None


def test_role_names_then_user_id() -> None:
    row = {"ROLE_NAMES": '["TENANT_ACME", "PUBLIC"]', "USER_ID": "1"}
    assert resolve_snowflake_subscription(row) == "TENANT_ACME"
    assert resolve_snowflake_subscription(row, order=("user_id",)) == "1"


def test_user_id_is_the_same_id_whichever_way_the_row_was_read() -> None:
    """Numeric column: "1" over the SQL API, 1 from a typed connector. Two spellings
    of one row must not bill to two different subscriptions."""
    assert resolve_snowflake_subscription({"USER_ID": "1"}, order=("user_id",)) == "1"
    assert resolve_snowflake_subscription({"USER_ID": 1}, order=("user_id",)) == "1"


def test_order_is_honoured_and_first_hit_wins() -> None:
    row = {
        "QUERY_TAG": '{"lago_subscription": "sub_tag"}',
        "ROLE_NAMES": '["TENANT_ACME"]',
        "USER_ID": "1",
    }
    assert resolve_snowflake_subscription(row) == "sub_tag"
    assert resolve_snowflake_subscription(row, order=("role_names", "query_tag")) == "TENANT_ACME"
    assert resolve_snowflake_subscription(row, order=()) is None


def test_a_real_rest_row_resolves_to_the_snowflake_user_only() -> None:
    """The honest state of this view: no QUERY_TAG value has ever been observed on it
    and it has no ROLE_NAMES at all, so the default order reaches `USER_ID` — a
    Snowflake identity, not a Lago subscription. A caller without that mapping should
    pass `order=("query_tag",)` and let the row go unattributed."""
    row = _load("rest_plain.json")
    assert resolve_snowflake_subscription(row) == "1"
    assert resolve_snowflake_subscription(row, order=("query_tag",)) is None


def test_role_names_accepted_as_a_list_not_only_a_json_string() -> None:
    """ARRAY columns arrive as TEXT over the SQL API and as a real list from a typed
    connector — the same row must attribute the same way through either."""
    assert resolve_snowflake_subscription({"ROLE_NAMES": ["TENANT_ACME"]}) == "TENANT_ACME"


def test_malformed_role_names_resolves_nothing_rather_than_throwing() -> None:
    row = {"ROLE_NAMES": "[not json", "USER_ID": "1"}
    assert resolve_snowflake_subscription(row, order=("role_names",)) is None
    assert resolve_snowflake_subscription(row) == "1"


def test_non_numeric_tokens_column_does_not_throw() -> None:
    """`TOKENS` is only ever read to decide whether zeros mean "no usage" or "usage we
    could not split", so a value that is not a number must degrade, not raise."""
    u = extract_snowflake_rest_log({"TOKENS": "n/a", "TOKENS_GRANULAR": None})
    assert u.nonzero_numeric() == {}
    assert "tokens_granular_missing" not in u.extras

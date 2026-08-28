"""Snowflake Cortex adapters — verified against real captured view rows.

Fixtures were read from a live account's `CORTEX_REST_API_USAGE_HISTORY` and
`CORTEX_AI_FUNCTIONS_USAGE_HISTORY` over the SQL API, one file per scenario, exactly as
the adapters receive them. Hand-made rows appear only where the live surface cannot
produce the shape (a failed row, malformed JSON, a fine-tuned model) and say so.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from lago_agent_sdk.gateway.adapters import (
    extract_snowflake_functions_log,
    extract_snowflake_rest_log,
    resolve_snowflake_subscription,
)
from lago_agent_sdk.pricing import deoverlapped_token_total

FIX = pathlib.Path(__file__).parent / "fixtures" / "snowflake_cortex"

REST_FIXTURES = sorted(p.name for p in FIX.glob("rest_*.json"))
FUNCTIONS_FIXTURES = sorted(p.name for p in FIX.glob("functions_*.json"))


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


def test_an_untagged_row_resolves_to_nothing_by_default() -> None:
    """The correction INT-230 measured live: every row either view produces carries a
    populated ROLE_NAMES/USER_ID, so a default including them never returns None —
    `default_subscription` was dead code and every untagged row billed to a Snowflake
    role name, which Lago ACCEPTED for the nonexistent subscription with a 200 (30
    events, zero errors on any hook). Reverting the default re-breaks this test."""
    row = {"ROLE_NAMES": '["TENANT_ACME", "PUBLIC"]', "USER_ID": "1"}
    assert resolve_snowflake_subscription(row) is None


def test_role_names_and_user_id_resolve_only_when_opted_into() -> None:
    row = {"ROLE_NAMES": '["TENANT_ACME", "PUBLIC"]', "USER_ID": "1"}
    assert resolve_snowflake_subscription(row, order=("role_names", "user_id")) == "TENANT_ACME"
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


def test_a_real_rest_row_is_unattributed_by_default_the_snowflake_user_on_opt_in() -> None:
    """The honest state of this view: no QUERY_TAG value has ever been observed on it
    and it has no ROLE_NAMES at all — `USER_ID` is a Snowflake identity, not a Lago
    subscription, so by default a real REST row goes unattributed and falls to the
    backfill's default."""
    row = _load("rest_plain.json")
    assert resolve_snowflake_subscription(row) is None
    assert resolve_snowflake_subscription(row, order=("user_id",)) == "1"


def test_role_names_accepted_as_a_list_not_only_a_json_string() -> None:
    """ARRAY columns arrive as TEXT over the SQL API and as a real list from a typed
    connector — the same row must attribute the same way through either."""
    assert resolve_snowflake_subscription({"ROLE_NAMES": ["TENANT_ACME"]}, order=("role_names",)) == (
        "TENANT_ACME"
    )


def test_malformed_role_names_resolves_nothing_rather_than_throwing() -> None:
    row = {"ROLE_NAMES": "[not json", "USER_ID": "1"}
    assert resolve_snowflake_subscription(row, order=("role_names",)) is None
    assert resolve_snowflake_subscription(row, order=("role_names", "user_id")) == "1"


def test_non_numeric_tokens_column_does_not_throw() -> None:
    """`TOKENS` is only ever read to decide whether zeros mean "no usage" or "usage we
    could not split", so a value that is not a number must degrade, not raise."""
    u = extract_snowflake_rest_log({"TOKENS": "n/a", "TOKENS_GRANULAR": None})
    assert u.nonzero_numeric() == {}
    assert "tokens_granular_missing" not in u.extras


# --------------------------------------------------------------------------
# CORTEX_AI_FUNCTIONS_USAGE_HISTORY — real fixtures
# --------------------------------------------------------------------------
def test_real_ai_complete_row_splits_input_and_output() -> None:
    """The one function type that reports a split: `{input, output}`, no total."""
    u = extract_snowflake_functions_log(_load("functions_ai_complete.json"))
    assert u.input == 13
    assert u.output == 5
    assert u.model == "claude-sonnet-4-5"
    assert u.provider == "snowflake"
    assert u.api == "snowflake_cortex_functions"
    assert u.extras["function_name"] == "AI_COMPLETE"
    # No total on this row, so nothing to record and no invented split to declare.
    assert "metrics_total" not in u.extras
    assert "metrics_total_only" not in u.extras


def test_total_only_row_bills_its_tokens_instead_of_zero() -> None:
    """The whole reason `total` is a mapped key.

    `AI_CLASSIFY` reports `{total: 195}` and nothing else — measured, and true of five
    of the six function types. Leaving `total` to the drift sweep extracts all-zero
    here, so `nonzero_numeric()` is empty, the caller emits nothing, and every
    task-specific AI SQL function bills ZERO with no error anywhere. This test fails if
    `total` is ever demoted to `extras`.
    """
    u = extract_snowflake_functions_log(_load("functions_total_only_no_model.json"))
    assert u.nonzero_numeric() == {"input": 195}
    assert u.extras["metrics_total"] == 195
    # The count is Snowflake's; the split is ours, and it says so.
    assert u.extras["metrics_total_only"] is True
    assert "metrics_unmapped" not in u.extras


def test_total_only_row_with_a_model_bills_the_same_way() -> None:
    """`AI_EMBED` reports `{total: 3}` AND a model, so the empty `MODEL_NAME` and the
    total-only shape vary independently — an adapter keyed off one to detect the other
    would mis-handle both of these rows."""
    u = extract_snowflake_functions_log(_load("functions_total_only_with_model.json"))
    assert u.nonzero_numeric() == {"input": 3}
    assert u.model == "snowflake-arctic-embed-m"
    assert u.extras["metrics_total_only"] is True


def test_empty_model_name_is_reported_not_a_crash() -> None:
    """The four task functions take no model argument, so `MODEL_NAME` is "" on a
    perfectly good row."""
    u = extract_snowflake_functions_log(_load("functions_total_only_no_model.json"))
    assert u.model == ""
    assert u.extras["function_name"] == "AI_CLASSIFY"


def test_every_captured_functions_row_bills_something() -> None:
    """No captured row may extract to all-zero. This is the 100% under-bill guard: five
    of the six function types report `{total}` alone, so an adapter that maps only
    `input`/`output` passes every other test in this file and bills nothing for them."""
    assert FUNCTIONS_FIXTURES, "no functions_*.json fixtures — capture is missing, not passing"
    for name in FUNCTIONS_FIXTURES:
        u = extract_snowflake_functions_log(_load(name))
        assert u.nonzero_numeric(), name
        assert u.provider == "snowflake", name
        assert u.api == "snowflake_cortex_functions", name


def test_billed_tokens_reconcile_against_the_views_own_metrics() -> None:
    """What Lago is told equals what Snowflake metered, on every captured row.

    Sums the token values out of the raw `METRICS` array and compares against the SDK's
    own de-overlapped total. Nothing on this view overlaps — no cache key, no reasoning
    key, 0 of 42 rows — so the two are equal by construction, and this fails if a future
    change makes the adapter double-count (mapping `total` alongside a split) or drop a
    metric.
    """
    for name in FUNCTIONS_FIXTURES:
        row = _load(name)
        metered = sum(
            entry["value"] for entry in json.loads(row["METRICS"]) if entry["key"]["unit"] == "tokens"
        )
        assert deoverlapped_token_total(extract_snowflake_functions_log(row)) == metered, name


def test_credits_are_recorded_and_never_billed() -> None:
    """`CREDITS` is what a customer sees in Snowflake's own cost view, so it is what they
    reconcile against — and it is not a billing input. There is no price mode on this
    path: no credit rate, no dollar figure, no cost event."""
    row = _load("functions_ai_complete.json")
    u = extract_snowflake_functions_log(row)
    assert u.extras["credits"] == "0.000068400"
    assert set(u.nonzero_numeric()) == {"input", "output"}
    assert not [k for k in u.extras if "cost" in k or "usd" in k]


def test_row_identity_and_grouping_keys_are_read() -> None:
    """`QUERY_ID` is the row id the caller's idempotency key is built from — rows are
    per query, and a key derived from the hour bucket instead collapses the twelve
    identical calls that share one bucket into a single event. It stays in `extras`
    rather than the dimensions purely on cardinality grounds; `FUNCTION_NAME` and
    `MODEL_NAME` are the dimensions worth grouping by."""
    u = extract_snowflake_functions_log(_load("functions_query_tag.json"))
    assert u.extras["query_id"] == "01c67fd2-0302-c6c5-001e-6063000320e6"
    assert u.extras["function_name"] == "AI_COMPLETE"
    assert u.model == "llama3.1-8b"
    assert u.extras["warehouse_id"] == "21"
    assert u.extras["is_completed"] == "true"
    # `timestamp_ltz`, so a bare epoch — not REST's "epoch nanos offset" triple. Handed
    # over unparsed; the caller stamps the event.
    assert u.extras["start_time"] == "1787162400"
    assert u.extras["end_time"] == "1787166000"


def test_a_real_functions_row_is_unattributed_by_default_resolves_through_role_names_on_opt_in() -> None:
    """Unlike the REST view, this one carries `ROLE_NAMES` — and a Snowflake-written
    `QUERY_TAG` (`{"app": "cortex_code_sandbox"}`) that must not be read as an id. By
    default neither is: a real untagged row resolves to nothing, and the role only on
    opt-in."""
    row = _load("functions_large_prompt.json")
    assert resolve_snowflake_subscription(row) is None
    assert resolve_snowflake_subscription(row, order=("role_names",)) == "LAGO_CORTEX_ROLE"
    tagged = _load("functions_query_tag.json")
    assert resolve_snowflake_subscription(tagged) is None
    assert resolve_snowflake_subscription(tagged, order=("query_tag", "role_names")) == "ACCOUNTADMIN"


def test_metrics_accepted_as_a_list_not_only_a_json_string() -> None:
    """The SQL API serializes ARRAY columns as TEXT; a typed connector hands back a real
    list. Reading zeros out of one of them is a silent 100% under-bill."""
    row = _load("functions_ai_complete.json")
    as_string = extract_snowflake_functions_log(row)
    row["METRICS"] = json.loads(row["METRICS"])
    assert extract_snowflake_functions_log(row).nonzero_numeric() == as_string.nonzero_numeric()


# --------------------------------------------------------------------------
# Shapes the live surface cannot produce — hand-made rows, labelled as such
# --------------------------------------------------------------------------
def test_a_split_row_that_also_reports_a_total_bills_the_split_only() -> None:
    """Unobserved — no captured row reports both (0 of 42). Guarded because it is the
    one shape where mapping `total` double-bills: 12 real tokens billed as 24."""
    u = extract_snowflake_functions_log(
        {
            "FUNCTION_NAME": "AI_COMPLETE",
            "METRICS": (
                '[{"key": {"metric": "input", "unit": "tokens"}, "value": 10},'
                ' {"key": {"metric": "output", "unit": "tokens"}, "value": 2},'
                ' {"key": {"metric": "total", "unit": "tokens"}, "value": 12}]'
            ),
        }
    )
    assert u.nonzero_numeric() == {"input": 10, "output": 2}
    assert u.extras["metrics_total"] == 12
    assert "metrics_total_only" not in u.extras


def test_failed_or_empty_row_extracts_to_all_zero_and_is_not_marked() -> None:
    """A failed call produces NO row on this view either (driven live: a 403 and a 400
    alongside a success; only the success appeared). This row is hypothetical, and with
    no metric and no credits there is nothing to say — the markers must stay off, or
    every ignorable row looks like lost revenue."""
    u = extract_snowflake_functions_log(
        {
            "FUNCTION_NAME": "AI_COMPLETE",
            "MODEL_NAME": "claude-sonnet-4-5",
            "METRICS": None,
            "CREDITS": None,
            "IS_COMPLETED": "true",
        }
    )
    assert u.nonzero_numeric() == {}
    assert u.model == "claude-sonnet-4-5"
    assert "metrics_unmapped" not in u.extras
    assert "metrics_total_only" not in u.extras


def test_malformed_metrics_json_degrades_without_throwing_and_is_marked() -> None:
    """Malformed JSON bills zero — but `CREDITS` proves Snowflake charged for the row,
    so this is lost revenue rather than an ignorable failure and it says so."""
    u = extract_snowflake_functions_log(
        {"FUNCTION_NAME": "AI_SUMMARIZE", "METRICS": "[not json", "CREDITS": "0.000271050"}
    )
    assert u.nonzero_numeric() == {}
    assert u.extras["function_name"] == "AI_SUMMARIZE"
    assert u.extras["metrics_unmapped"] is True


def test_unmapped_metric_reaches_extras_and_is_not_counted() -> None:
    """A metric name nobody has classified must not be miscounted as one of the three
    mapped ones, and must not vanish. Cortex adds functions continually, and the sibling
    REST view grew two token keys and a whole column inside one day."""
    u = extract_snowflake_functions_log(
        {
            "METRICS": (
                '[{"key": {"metric": "input", "unit": "tokens"}, "value": 100},'
                ' {"key": {"metric": "guardrail", "unit": "tokens"}, "value": 42}]'
            ),
        }
    )
    assert u.nonzero_numeric() == {"input": 100}
    assert u.extras["metrics.guardrail"] == 42
    assert "metrics.input" not in u.extras


def test_a_metric_measured_in_something_other_than_tokens_is_not_billed_as_tokens() -> None:
    """`METRICS` is a metric NAME plus a UNIT and only the pair means anything — Cortex
    meters `AI_PARSE_DOCUMENT` per page. Billing 12 pages as 12 tokens is wrong in a way
    no later test can see, so a foreign unit goes to the sweep — and because that leaves
    the row billing zero, it is marked."""
    u = extract_snowflake_functions_log(
        {
            "FUNCTION_NAME": "AI_PARSE_DOCUMENT",
            "METRICS": '[{"key": {"metric": "input", "unit": "pages"}, "value": 12}]',
            "CREDITS": "0.010000000",
        }
    )
    assert u.nonzero_numeric() == {}
    assert u.extras["metrics.input.pages"] == 12
    assert u.extras["metrics_unmapped"] is True


def test_a_metric_repeated_in_the_array_is_summed() -> None:
    """`METRICS` is a list, not an object, so it can carry a metric twice. Last-wins
    would drop the first value with nothing to show for it."""
    u = extract_snowflake_functions_log(
        {
            "METRICS": (
                '[{"key": {"metric": "input", "unit": "tokens"}, "value": 30},'
                ' {"key": {"metric": "input", "unit": "tokens"}, "value": 12}]'
            ),
        }
    )
    assert u.input == 42


def test_an_incomplete_row_is_handed_over_rather_than_judged() -> None:
    """Hypothetical row, but no longer a hypothetical CASE. Measured 2026-08-26: an
    in-flight query writes no row at all (19 polls over the 937s one ran, nothing), and
    the row lands 141s after the query ends already `true` and never moves — so FALSE is
    not reachable the way this test once guessed. It IS reachable across an hour
    boundary, where the flag means "completed in THIS aggregation window" and one query
    writes a row per bucket. The adapter extracts what is there and passes the flag to
    the caller, who owns the window and idempotency rules."""
    u = extract_snowflake_functions_log(
        {
            "FUNCTION_NAME": "AI_COMPLETE",
            "IS_COMPLETED": "false",
            "METRICS": '[{"key": {"metric": "total", "unit": "tokens"}, "value": 7}]',
            "CREDITS": "0.000000900",
        }
    )
    assert u.extras["is_completed"] == "false"
    assert u.nonzero_numeric() == {"input": 7}


def test_lowercase_functions_column_keys_are_accepted() -> None:
    u = extract_snowflake_functions_log(
        {
            "function_name": "AI_SENTIMENT",
            "model_name": "",
            "metrics": '[{"key": {"metric": "total", "unit": "tokens"}, "value": 21}]',
        }
    )
    assert u.input == 21
    assert u.extras["function_name"] == "AI_SENTIMENT"


def test_functions_extractor_never_throws_on_a_malformed_row() -> None:
    """A window of rows runs through this; one bad row must not take the run down."""
    rows: list[Any] = [
        {},
        {"METRICS": 7},
        {"METRICS": "{}"},
        {"METRICS": '[{"key": "input", "value": 5}]'},
        {"METRICS": '[["input", 5]]'},
        {"METRICS": '[{"key": {"metric": "input"}, "value": "n/a"}]'},
        {"MODEL_NAME": 42, "CREDITS": "n/a"},
    ]
    for row in rows:
        u = extract_snowflake_functions_log(row)
        assert u.nonzero_numeric() == {}
        assert u.api == "snowflake_cortex_functions"


def test_an_array_entry_of_the_wrong_shape_is_kept_by_position() -> None:
    """Two unusable entries must not collide in `extras`, and neither may disappear."""
    u = extract_snowflake_functions_log({"METRICS": '["input", "output"]'})
    assert u.extras["metrics.0"] == "input"
    assert u.extras["metrics.1"] == "output"

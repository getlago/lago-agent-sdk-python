"""Databricks usage reader — the I/O half, exercised without touching a warehouse.

`DatabricksSource.query` is faked here; the SQL it would run is asserted, and the
COLUMNAR response shape is reproduced exactly as the Statement Execution API returns
it (`manifest.schema.columns` plus a positional `data_array`, one chunk inline).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from lago_agent_sdk import CanonicalUsage, LagoSDK
from lago_agent_sdk.gateway.databricks import (
    DatabricksSource,
    DatabricksUsageRow,
    _floor_hour,
    _timestamp_sql,
    _window_bounds,
)

# --------------------------------------------------------------------------
# Fake rows, in the exact shapes the two tables return
# --------------------------------------------------------------------------
_HOSTED = {
    "invocation_id": "inv-hosted-1",
    "request_id": "req-hosted-1",
    "event_time": "2026-08-07 14:22:03.123",
    "destination_type": "PAY_PER_TOKEN_FOUNDATION_MODEL",
    "destination_name": "system.ai.llama-4-maverick",
    "destination_model": "llama-4-maverick",
    "api_type": "mlflow/v1/chat/completions",
    "endpoint_name": "system.ai.llama-4-maverick",
    "input_tokens": "11",
    "output_tokens": "4",
    "request_tags": '{"lago_subscription":"sub_hosted"}',
}

_BYOK_USAGE = {
    "invocation_id": "inv-byok-1",
    "request_id": "req-byok-1",
    "event_time": "2026-08-07 14:22:59.900",
    "destination_type": "EXTERNAL_FOUNDATION_MODEL",
    "destination_name": "workspace.default.anthropickey",
    "destination_model": "claude-sonnet-4-5",
    "api_type": "anthropic/v1/messages",
    "endpoint_name": "workspace.default.anthropickey",
    "status_code": "200",
    "input_tokens": "1825",
    "output_tokens": "47",
    "token_details": '{"cache_read_input_tokens":1812}',
    "request_tags": '{"lago_subscription":"sub_byok"}',
}

_BYOK_SPEND = {
    "record_id": "rec-1",
    "bucket": "2026-08-07 14:00:00",
    "provider": "anthropic",
    "model": "claude-sonnet-4-5",
    "request_tags": '{"lago_subscription":"sub_byok"}',
    "usage_quantity": "0.0011187",
}

_FAILED = {
    "invocation_id": "inv-failed",
    "event_time": "2026-08-07 14:30:00",
    "destination_type": "PAY_PER_TOKEN_FOUNDATION_MODEL",
    "destination_name": "system.ai.gpt-oss-20b",
    "api_type": "mlflow/v1/chat/completions",
    "input_tokens": None,
    "output_tokens": None,
    "status_code": "403",
}

# The BYOK half of the same thing. Shaped from the live table: a rejected external call
# is logged with NULL tokens and — because the gateway never got far enough to resolve
# one — an EMPTY `destination_model`.
_FAILED_BYOK = {
    "invocation_id": "inv-failed-byok",
    "event_time": "2026-08-07 14:31:00",
    "destination_type": "EXTERNAL_FOUNDATION_MODEL",
    "destination_name": "workspace.default.anthropickey",
    "destination_model": None,
    "api_type": "anthropic/v1/messages",
    "input_tokens": None,
    "output_tokens": None,
    "status_code": "403",
}


def _source(spend: list[dict], usage: list[dict]) -> DatabricksSource:
    """A source whose `query` answers from canned rows, keyed on which table."""
    src = DatabricksSource(host="https://x", token="t", warehouse_id="w")
    seen: list[str] = []

    def fake_query(sql: str) -> list[dict[str, Any]]:
        seen.append(sql)
        return spend if "external_model_spend" in sql else usage

    src.query = fake_query  # type: ignore[method-assign]
    src.queries = seen  # type: ignore[attr-defined]
    return src


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------
_NOW = datetime(2026, 8, 21, 13, 34, 12, tzinfo=timezone.utc)


def test_interval_strings_resolve_to_instants_not_sql() -> None:
    """Resolved in Python, so both statements can share one bound. A `current_timestamp()`
    expression is re-evaluated per statement and the two reads then cover different
    windows — measured 5.1s apart, and a hosted row in the gap is billed by neither."""
    assert _window_bounds("1 day", now=_NOW) == (
        datetime(2026, 8, 20, 13, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 21, 13, 0, tzinfo=timezone.utc),
    )
    assert _window_bounds("36 hours", now=_NOW)[0] == datetime(2026, 8, 20, 1, 0, tzinfo=timezone.utc)
    assert _window_bounds("2 weeks", now=_NOW)[0] == datetime(2026, 8, 7, 13, 0, tzinfo=timezone.utc)


def test_datetime_window_renders_as_a_literal() -> None:
    assert _timestamp_sql(datetime(2026, 8, 7, 14, 0, 0)) == "TIMESTAMP '2026-08-07 14:00:00+00:00'"


def test_the_window_is_floored_to_the_hour_at_both_ends() -> None:
    """`external_model_spend` is an hourly aggregate whose `usage_start_time` is always
    the hour start (65 of 65 live rows), so a mid-hour bound drops the hour CONTAINING
    it while the usage table still yields that hour's rows. Live: `since` of 13:30 read
    11 of 65 spend rows, dropping $0.1256 of $0.1723, and still read 35 BYOK usage rows
    from inside the dropped hour."""
    lower, upper = _window_bounds(datetime(2026, 8, 7, 13, 30, 45, tzinfo=timezone.utc), now=_NOW)
    assert lower == datetime(2026, 8, 7, 13, 0, tzinfo=timezone.utc)
    assert upper == datetime(2026, 8, 21, 13, 0, tzinfo=timezone.utc)


def test_the_still_aggregating_hour_is_excluded() -> None:
    """A spend row cannot be complete before its hour closes — the 08:00–09:00 row
    appeared ~7 min AFTER 09:00. Billing the open hour bills a fraction of it under
    that hour's `record_id`, and Lago then rejects the corrected re-run as a duplicate
    `transaction_id`, so the remainder is never billed."""
    assert _window_bounds("1 day", now=_NOW)[1] == _floor_hour(_NOW) < _NOW


def test_timestamp_literals_name_their_zone() -> None:
    """A bare literal is parsed in the warehouse's `spark.sql.session.timeZone`, so on a
    non-UTC workspace the same literal names a different instant and the window slides
    by the offset. Verified live that the suffixed form is accepted."""
    assert _timestamp_sql(_NOW).endswith("+00:00'")


@pytest.mark.parametrize(
    "bad",
    [
        "1 day; DROP TABLE system.ai_gateway.usage",
        "1 day OR 1=1",
        "yesterday",
        "-1 day",
        "",
    ],
)
def test_unrecognized_window_is_refused_not_interpolated(bad: str) -> None:
    """Anything but a bare count-plus-unit is refused outright. The string no longer
    reaches SQL — the bound is resolved to an instant first — so this is no longer an
    injection guard; it is what stops a window being quietly read as something other
    than what the caller wrote."""
    with pytest.raises(ValueError, match="not understood"):
        _window_bounds(bad)


def test_read_usage_scopes_both_queries_to_one_shared_window() -> None:
    """The point of resolving the bounds in Python: both statements must carry the SAME
    two literals. Two `current_timestamp()` expressions drift, and the read that runs
    second covers the narrower window — a hosted row in the gap is lost, since hosted
    is billed from `usage` alone."""
    src = _source([], [])
    before = _floor_hour(datetime.now(timezone.utc))
    list(src.read_usage("3 days"))
    after = _floor_hour(datetime.now(timezone.utc))
    assert len(src.queries) == 2  # type: ignore[attr-defined]
    spend, usage = src.queries  # type: ignore[attr-defined]
    assert "current_timestamp()" not in spend and "current_timestamp()" not in usage
    ceilings = {_timestamp_sql(before), _timestamp_sql(after)}
    for sql, column in ((spend, "usage_start_time"), (usage, "event_time")):
        lower = _timestamp_sql(before - timedelta(days=3))
        assert (
            f"{column} >= {lower}" in sql or f"{column} >= {_timestamp_sql(after - timedelta(days=3))}" in sql
        )
        assert any(f"{column} < {c}" in sql for c in ceilings)


def test_a_sub_hour_interval_collapses_to_an_empty_window() -> None:
    """Both bounds floor to the same hour, so there is nothing left to read. Asserted
    against a pinned clock: driving this through the real one makes the test pass or
    fail on the current MINUTE — "30 minutes" spans two hours at 13:34 and one at
    13:04, which is a coin flip in CI, not a property of the code."""
    lower, upper = _window_bounds("30 minutes", now=_NOW)
    assert lower == upper == datetime(2026, 8, 21, 13, 0, tzinfo=timezone.utc)


def test_a_window_entirely_inside_the_open_hour_reads_nothing_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Excluding the open hour means such a window can resolve to nothing. Zero rows
    then says nothing about whether there was traffic, so it must not pass silently —
    and it must not spend warehouse time either.

    `since` is half an hour AHEAD of the real clock so the guard fires whatever minute
    this runs at; the collapse itself is pinned in the test above."""
    src = _source([], [])
    inside_the_open_hour = datetime.now(timezone.utc) + timedelta(minutes=30)
    with caplog.at_level(logging.WARNING):
        assert list(src.read_usage(inside_the_open_hour)) == []
    assert src.queries == []  # type: ignore[attr-defined]
    assert "widen the window" in caplog.text


# --------------------------------------------------------------------------
# The BYOK / hosted split — the double-billing guard
# --------------------------------------------------------------------------
def test_byok_bills_once_from_spend_and_hosted_once_from_usage() -> None:
    """A BYOK call appears in BOTH tables. It must yield exactly one row, carrying
    Databricks' own metered USD; the token row it also has must not become a second
    billable row."""
    rows = list(_source([_BYOK_SPEND], [_HOSTED, _BYOK_USAGE]).read_usage("1 day"))
    assert len(rows) == 2

    byok = [r for r in rows if r.is_byok]
    hosted = [r for r in rows if not r.is_byok]
    assert len(byok) == 1 and len(hosted) == 1
    assert byok[0].usd_cost == pytest.approx(0.0011187)
    assert byok[0].usage.model == "claude-sonnet-4-5"
    assert hosted[0].usage.model == "llama-4-maverick"
    assert hosted[0].usd_cost is None


def test_byok_row_carries_the_token_counts_joined_from_the_usage_table() -> None:
    """The dollar figure is authoritative, but the event should still report real
    tokens — they are joined on (hour, provider, model, tags), the spend table's own
    aggregation key."""
    (byok,) = [r for r in _source([_BYOK_SPEND], [_BYOK_USAGE]).read_usage("1 day") if r.is_byok]
    assert byok.usage.input == 1825
    assert byok.usage.output == 47
    assert byok.usage.cache_read == 1812


def test_several_calls_in_one_spend_bucket_have_their_tokens_summed() -> None:
    """The spend table aggregates per (hour, model, provider, tags), so N calls in the
    same hour collapse to ONE dollar row while `ai_gateway.usage` still holds N token
    rows. Reporting only the first would understate the tokens behind a cost the
    customer can see — so they sum."""
    second = {**_BYOK_USAGE, "invocation_id": "inv-byok-2", "input_tokens": "100", "output_tokens": "3"}
    (byok,) = list(_source([_BYOK_SPEND], [_BYOK_USAGE, second]).read_usage("1 day"))
    assert byok.usage.input == 1925
    assert byok.usage.output == 50
    assert byok.usage.cache_read == 3624
    # Still ONE event: the dollar figure already covers both calls.
    assert byok.usd_cost == pytest.approx(0.0011187)


def test_tokens_only_merge_within_the_same_hour() -> None:
    """The bucket is part of the join key, so a call in the next hour belongs to a
    different spend row and must not inflate this one."""
    next_hour = {**_BYOK_USAGE, "invocation_id": "inv-byok-3", "event_time": "2026-08-07 15:04:00"}
    (byok,) = list(_source([_BYOK_SPEND], [_BYOK_USAGE, next_hour]).read_usage("1 day"))
    assert byok.usage.input == 1825


def test_unparseable_request_tags_do_not_crash_the_join() -> None:
    """A tag column that isn't JSON still has to produce a stable key rather than
    raising — one malformed row must not take down the batch."""
    rows = list(_source([{**_BYOK_SPEND, "request_tags": "not json"}], [_BYOK_USAGE]).read_usage("1 day"))
    assert len(rows) == 1
    assert rows[0].usd_cost == pytest.approx(0.0011187)


def test_byok_spend_with_no_matching_usage_still_bills_its_dollars() -> None:
    """A join miss (a row aggregated across an hour boundary, say) must not drop
    revenue — the cost is what Databricks charged either way, just with no tokens."""
    (byok,) = list(_source([_BYOK_SPEND], []).read_usage("1 day"))
    assert byok.usd_cost == pytest.approx(0.0011187)
    assert byok.usage.model == "claude-sonnet-4-5"
    assert byok.usage.provider == "anthropic"
    assert byok.usage.nonzero_numeric() == {}


def test_zero_dollar_spend_rows_are_skipped() -> None:
    assert list(_source([{**_BYOK_SPEND, "usage_quantity": "0"}], []).read_usage("1 day")) == []


def test_failed_calls_yield_nothing() -> None:
    """403/404s are recorded with NULL token counts. Emitting them would bill an
    empty event for a call that never reached a provider."""
    assert list(_source([], [_FAILED]).read_usage("1 day")) == []


def test_failed_byok_calls_do_not_enter_the_join_index(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rejected external call bought nothing, so Databricks meters no dollars for it
    and its key can never match a spend row. Indexing it manufactures a bucket the
    unbilled warning then tells the operator to re-run the window for — advice that can
    never bill it, because there is nothing to bill."""
    with caplog.at_level(logging.WARNING):
        assert list(_source([], [_FAILED_BYOK]).read_usage("1 day")) == []
    assert "NOT billed" not in caplog.text


def test_the_unbilled_warning_counts_only_buckets_with_real_tokens(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning exists to name genuine spend-table lag. Live over 2026-08-06 it
    reported 29 buckets of which 28 were failed calls, and the single example row it
    showed the operator was one of them — so the one real lagging bucket was the thing
    least likely to be read."""
    lagging = {**_BYOK_USAGE, "invocation_id": "inv-lagging", "destination_model": "claude-opus-4-1"}
    with caplog.at_level(logging.WARNING):
        list(_source([], [_FAILED_BYOK, lagging]).read_usage("1 day"))
    assert "1 BYOK token bucket(s)" in caplog.text
    assert "model=claude-opus-4-1" in caplog.text


def test_hosted_rows_keep_the_databricks_provider() -> None:
    """Which is what makes the price lookup miss deliberately rather than matching
    some other vendor's rate for a DBU-billed model."""
    (hosted,) = list(_source([], [_HOSTED]).read_usage("1 day"))
    assert hosted.usage.provider == "databricks"
    assert hosted.usage.api == "databricks_gateway"


# --------------------------------------------------------------------------
# Chunked results — the silent-truncation guard
# --------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


def test_query_zips_columns_and_follows_every_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only chunk 0 arrives inline. A reader that stops there works on a small window
    and silently bills a fraction of a large one — so all `total_chunk_count` chunks
    are fetched and the columnar rows zipped back into dicts."""
    import requests

    first = {
        "statement_id": "stmt-1",
        "status": {"state": "SUCCEEDED"},
        "manifest": {
            "schema": {"columns": [{"name": "invocation_id"}, {"name": "input_tokens"}]},
            "total_chunk_count": 3,
        },
        "result": {"data_array": [["a", "1"]]},
    }
    chunks = {1: {"data_array": [["b", "2"]]}, 2: {"data_array": [["c", "3"]]}}
    fetched: list[str] = []

    def fake_post(url: str, **_kw: Any) -> _FakeResponse:
        return _FakeResponse(first)

    def fake_get(url: str, **_kw: Any) -> _FakeResponse:
        fetched.append(url)
        return _FakeResponse(chunks[int(url.rsplit("/", 1)[-1])])

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    rows = DatabricksSource(host="https://x/", token="t", warehouse_id="w").query("SELECT 1")
    assert rows == [
        {"invocation_id": "a", "input_tokens": "1"},
        {"invocation_id": "b", "input_tokens": "2"},
        {"invocation_id": "c", "input_tokens": "3"},
    ]
    assert [u.rsplit("/", 1)[-1] for u in fetched] == ["1", "2"]
    assert all(u.startswith("https://x/api/2.0/sql/statements/stmt-1/result/chunks/") for u in fetched)


def test_query_raises_when_a_chunk_fetch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed chunk fetch must NOT be swallowed into a short row set.

    The error body carries no `data_array`, so `or []` would append nothing, the loop
    would move to the next index, and `query()` would return a partial window reporting
    success. Measured against a live warehouse: a 403 on chunk 1 of 2 returned 6,750 of
    9,000 rows with no exception — 25% of the window billed as if it were all of it.
    """
    import requests

    first = {
        "statement_id": "stmt-1",
        "status": {"state": "SUCCEEDED"},
        "manifest": {
            "schema": {"columns": [{"name": "invocation_id"}, {"name": "input_tokens"}]},
            "total_chunk_count": 2,
        },
        "result": {"data_array": [["a", "1"]]},
    }
    # exactly what the API returns for an expired statement / revoked token mid-read
    denied = {"error_code": "PERMISSION_DENIED", "message": "does not have required scopes: sql"}

    monkeypatch.setattr(requests, "post", lambda url, **_kw: _FakeResponse(first))
    monkeypatch.setattr(requests, "get", lambda url, **_kw: _FakeResponse(denied, status_code=403))

    src = DatabricksSource(host="https://x/", token="t", warehouse_id="w")
    with pytest.raises(RuntimeError) as excinfo:
        src.query("SELECT 1")
    message = str(excinfo.value)
    assert "result chunk 1 of 2" in message
    # the operator must see the API's own cause, not just a status line
    assert "does not have required scopes: sql" in message


def test_query_raises_when_the_row_count_misses_the_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chunk that returns HTTP 200 with fewer rows than promised is still truncation.
    No per-request status check can catch that, so the assembled count is compared with
    `manifest.total_row_count` before any of it is billed."""
    import requests

    first = {
        "statement_id": "stmt-1",
        "status": {"state": "SUCCEEDED"},
        "manifest": {
            "schema": {"columns": [{"name": "invocation_id"}]},
            "total_chunk_count": 2,
            "total_row_count": 3,
        },
        "result": {"data_array": [["a"]]},
    }
    monkeypatch.setattr(requests, "post", lambda url, **_kw: _FakeResponse(first))
    # HTTP 200, but one row short of the promised three
    monkeypatch.setattr(requests, "get", lambda url, **_kw: _FakeResponse({"data_array": [["b"]]}))

    src = DatabricksSource(host="https://x/", token="t", warehouse_id="w")
    with pytest.raises(RuntimeError, match=r"returned 2 row\(s\) but the manifest promised 3"):
        src.query("SELECT 1")


def test_query_raises_when_rows_arrive_with_no_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows with no column names zip to `{}` each, which every layer downstream degrades
    cleanly and wrongly on — all-zero usage and a confident `{"cost": 0, "tokens": 0}`
    for a window that had real traffic. Not observed on this API; guards the decode."""
    import requests

    first = {
        "statement_id": "stmt-1",
        "status": {"state": "SUCCEEDED"},
        "manifest": {"total_chunk_count": 1},
        "result": {"data_array": [["a"], ["b"]]},
    }
    monkeypatch.setattr(requests, "post", lambda url, **_kw: _FakeResponse(first))

    src = DatabricksSource(host="https://x/", token="t", warehouse_id="w")
    with pytest.raises(RuntimeError, match="no `manifest.schema.columns`"):
        src.query("SELECT 1")


def test_query_error_carries_the_api_error_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-OK submission used to surface as `Databricks statement None: {...}` — the
    state was absent, so the poll loop's own guard raised with a misleading prefix. The
    cause was in the body all along; name it."""
    import requests

    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **_kw: _FakeResponse(
            {"error_code": "NOT_FOUND", "message": "The warehouse w was not found."},
            status_code=404,
        ),
    )
    src = DatabricksSource(host="https://x/", token="t", warehouse_id="w")
    with pytest.raises(RuntimeError) as excinfo:
        src.query("SELECT 1")
    message = str(excinfo.value)
    assert "statement submission failed: HTTP 404" in message
    assert "The warehouse w was not found." in message
    assert "statement None" not in message


def test_query_raises_on_a_failed_statement(monkeypatch: pytest.MonkeyPatch) -> None:
    """A FAILED statement returns 200 with the failure in the body. Reading rows from
    it would report an empty window as "no usage" and bill nothing."""
    import requests

    monkeypatch.setattr(
        requests,
        "post",
        lambda *_a, **_kw: _FakeResponse({"status": {"state": "FAILED", "error": {"message": "boom"}}}),
    )
    with pytest.raises(RuntimeError, match="FAILED"):
        DatabricksSource(host="https://x", token="t", warehouse_id="w").query("SELECT 1")


# --------------------------------------------------------------------------
# Idempotency keys
# --------------------------------------------------------------------------
def test_event_ids_are_unique_per_row_and_scoped_by_subscription() -> None:
    rows = list(_source([_BYOK_SPEND], [_HOSTED, _BYOK_USAGE]).read_usage("1 day"))
    ids = [r.event_id for r in rows]
    assert len(set(ids)) == len(ids)
    assert "sub_byok" in [i for i in ids if "spend" in i][0]
    assert "sub_hosted" in [i for i in ids if "usage" in i][0]


def test_event_id_prefix_namespaces_the_whole_read() -> None:
    rows = list(_source([_BYOK_SPEND], [_HOSTED]).read_usage("1 day", event_id_prefix="tenant7"))
    assert all(r.event_id.startswith("tenant7_") for r in rows)


def test_event_id_for_rescopes_without_changing_the_row_key() -> None:
    """`transaction_id` is unique account-wide, so the same source row billed to two
    subscriptions needs two ids — and the id must follow the subscription actually
    billed, which for an untagged row is the caller's default, not the row's tag."""
    row = DatabricksUsageRow(
        usage=None,  # type: ignore[arg-type]
        subscription=None,
        row_id="rec-9",
        kind="spend",
        usd_cost=1.0,
    )
    assert row.event_id == "dbx_spend_none_rec-9"
    assert row.event_id_for("sub_a") == "dbx_spend_sub_a_rec-9"
    assert row.event_id_for("sub_b") == "dbx_spend_sub_b_rec-9"
    assert row.event_id_for("sub_a") != row.event_id_for("sub_b")


# --------------------------------------------------------------------------
# The one-liner
# --------------------------------------------------------------------------
class _Recorder:
    """Collects delivered events, so assertions read the real emitted shape."""

    def __init__(self) -> None:
        self.batches: list[list[dict]] = []

    @property
    def events(self) -> list[dict]:
        return [e for b in self.batches for e in b]


def _sdk() -> tuple[LagoSDK, _Recorder]:
    rec = _Recorder()
    sdk = LagoSDK(api_key="dummy")
    sdk._queue._sender = lambda b: rec.batches.append(list(b))  # type: ignore[attr-defined]
    return sdk, rec


def _drain(sdk: LagoSDK) -> None:
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)


def test_backfill_counts_cost_tokens_and_skips() -> None:
    sdk, q = _sdk()
    src = _source([_BYOK_SPEND], [_HOSTED, _BYOK_USAGE, {**_HOSTED, "request_tags": "{}"}])
    counts = sdk.backfill_databricks(src, "1 day")
    _drain(sdk)
    # The untagged row has no subscription and no default to fall back on.
    assert counts == {"cost": 1, "tokens": 1, "skipped": 1}
    assert {e["external_subscription_id"] for e in q.events} == {"sub_byok", "sub_hosted"}


def test_backfill_falls_back_to_the_default_subscription() -> None:
    sdk, q = _sdk()
    src = _source([], [{**_HOSTED, "request_tags": "{}"}])
    assert sdk.backfill_databricks(src, "1 day", default_subscription="sub_fb")["skipped"] == 0
    _drain(sdk)
    assert {e["external_subscription_id"] for e in q.events} == {"sub_fb"}
    # ...and the id follows the subscription billed, not the row's absent tag.
    assert all("sub_fb" in e["transaction_id"] for e in q.events)


def test_backfill_unified_ignores_per_row_tags() -> None:
    """One gateway serving one customer: everything lands on one subscription even
    though the rows carry their own tags."""
    sdk, q = _sdk()
    src = _source([_BYOK_SPEND], [_HOSTED, _BYOK_USAGE])
    sdk.backfill_databricks(src, "1 day", default_subscription="sub_one", unified=True)
    _drain(sdk)
    assert {e["external_subscription_id"] for e in q.events} == {"sub_one"}
    assert all("sub_one" in e["transaction_id"] for e in q.events)


def test_backfill_bills_byok_as_cost_and_hosted_as_tokens() -> None:
    sdk, q = _sdk()
    src = _source([_BYOK_SPEND], [_HOSTED, _BYOK_USAGE])
    sdk.backfill_databricks(src, "1 day")
    _drain(sdk)

    cost = [e for e in q.events if e["code"] == "llm_cost"]
    tokens = [e for e in q.events if e["code"] != "llm_cost"]
    assert len(cost) == 1
    # Databricks' own $0.0011187 -> 0.11187 cents, passed through, not recomputed.
    assert cost[0]["precise_total_amount_cents"].startswith("0.11187")
    assert cost[0]["properties"]["price_source"] == "precomputed"
    # Hosted has no dollar figure anywhere in Databricks' tables, so: token events.
    assert {e["code"] for e in tokens} == {"llm_input_tokens", "llm_output_tokens"}
    assert all("precise_total_amount_cents" not in e for e in tokens)


def test_backfill_is_idempotent_across_a_re_run() -> None:
    """Re-reading the same window must produce byte-identical transaction ids, so
    Lago rejects the duplicates instead of double-billing."""
    ids = []
    for _ in range(2):
        sdk, q = _sdk()
        sdk.backfill_databricks(_source([_BYOK_SPEND], [_HOSTED, _BYOK_USAGE]), "1 day")
        _drain(sdk)
        ids.append([e["transaction_id"] for e in q.events])
    assert ids[0] == ids[1]


def test_backfill_survives_one_malformed_row() -> None:
    """Instrumentation never breaks the caller: a row that extracts to nothing usable
    is skipped, and the rows around it still bill."""
    sdk, q = _sdk()
    src = _source([_BYOK_SPEND], [{"nonsense": True}, _HOSTED, _BYOK_USAGE])
    counts = sdk.backfill_databricks(src, "1 day", default_subscription="sub_fb")
    _drain(sdk)
    assert counts["cost"] == 1 and counts["tokens"] == 1
    assert len(q.events) >= 3


# --------------------------------------------------------------------------
# Reconciliation dimensions — the whole point of the connector being checkable
# --------------------------------------------------------------------------
def test_hosted_events_carry_the_endpoint_the_gateway_page_groups_by() -> None:
    """Our `model` is normalized (`llama-4-maverick`) where the AI Gateway usage page
    shows `system.ai.llama-4-maverick`. Without the endpoint on the event, grouping
    Lago one way and Databricks the other fails on naming alone."""
    sdk, q = _sdk()
    sdk.backfill_databricks(_source([], [_HOSTED]), "1 day")
    _drain(sdk)
    assert q.events
    for e in q.events:
        assert e["properties"]["endpoint_name"] == "system.ai.llama-4-maverick"


def test_byok_events_carry_the_hour_bucket_not_a_sampled_endpoint() -> None:
    """A spend row covers an hour of requests, so its authoritative key is the hour —
    `external_model_spend`'s own aggregation key. A per-request field here would be one
    sampled value presented as a property of the whole bucket."""
    sdk, q = _sdk()
    sdk.backfill_databricks(_source([_BYOK_SPEND], [_BYOK_USAGE]), "1 day")
    _drain(sdk)
    (event,) = q.events
    assert event["properties"]["bucket"] == "2026-08-07 14:00:00"
    assert "endpoint_name" not in event["properties"]


def test_caller_dimensions_are_added_and_win_on_a_collision() -> None:
    sdk, q = _sdk()
    sdk.backfill_databricks(
        _source([], [_HOSTED]),
        "1 day",
        dimensions={"team": "platform", "endpoint_name": "mine"},
    )
    _drain(sdk)
    for e in q.events:
        assert e["properties"]["team"] == "platform"
        # An explicit dimension is the caller's decision, so it overrides the auto key
        # rather than being silently discarded.
        assert e["properties"]["endpoint_name"] == "mine"


def test_a_row_with_no_endpoint_adds_no_empty_dimension() -> None:
    """An empty string would create a phantom Lago group rather than saying nothing."""
    sdk, q = _sdk()
    sdk.backfill_databricks(_source([], [{**_HOSTED, "endpoint_name": None}]), "1 day")
    _drain(sdk)
    for e in q.events:
        assert "endpoint_name" not in e["properties"]


def test_merged_bucket_drops_per_request_extras_but_keeps_the_endpoint() -> None:
    """`invocation_id` and `status_code` describe one request. Carrying them on an
    hourly aggregate states one sampled request's value as if it covered the hour —
    and once dimensions are emitted from extras, that becomes a live mis-statement."""
    second = {**_BYOK_USAGE, "invocation_id": "inv-byok-2", "input_tokens": "100"}
    (byok,) = list(_source([_BYOK_SPEND], [_BYOK_USAGE, second]).read_usage("1 day"))
    extras = byok.usage.extras
    assert extras["endpoint_name"] == _BYOK_USAGE["endpoint_name"]
    assert extras["api_type"] == "anthropic/v1/messages"
    for per_request in ("invocation_id", "request_id", "status_code"):
        assert per_request not in extras


def test_a_single_request_bucket_is_described_the_same_way() -> None:
    """Otherwise `status_code` survives on quiet hours and vanishes on busy ones —
    the same bucket shape reporting different fields depending on traffic."""
    (byok,) = list(_source([_BYOK_SPEND], [_BYOK_USAGE]).read_usage("1 day"))
    assert "invocation_id" not in byok.usage.extras
    assert byok.usage.extras["endpoint_name"] == _BYOK_USAGE["endpoint_name"]


def test_from_env_names_every_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_WAREHOUSE_ID"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError) as exc:
        DatabricksSource.from_env()
    assert "DATABRICKS_HOST" in str(exc.value)
    assert "DATABRICKS_WAREHOUSE_ID" in str(exc.value)


def test_from_env_trims_a_trailing_slash_off_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Or every URL doubles its separator — Databricks 404s on `//api/2.0/...`."""
    monkeypatch.setenv("DATABRICKS_HOST", "https://dbc-x.cloud.databricks.com/")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-x")
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "wh-1")
    assert DatabricksSource.from_env().host == "https://dbc-x.cloud.databricks.com"


def test_json_string_columns_survive_the_round_trip() -> None:
    """STRUCT/MAP columns arrive as JSON strings over the Statement Execution API.
    The reader joins on the tag map, so it has to parse the same way the adapter
    does or every BYOK row misses its token counts."""
    src = _source(
        [{**_BYOK_SPEND, "request_tags": json.dumps({"lago_subscription": "sub_byok"})}],
        [_BYOK_USAGE],
    )
    (byok,) = list(src.read_usage("1 day"))
    assert byok.usage.input == 1825
    assert byok.subscription == "sub_byok"


# --------------------------------------------------------------------------
# Post-review hardening — each of these pins a bug found by code review
# --------------------------------------------------------------------------
def test_rows_with_no_usable_id_do_not_collide() -> None:
    """`_safe_str(a or b)` returned "" for a row whose ids were NULL, and also for one
    whose id a driver handed back as a non-str (the `or` picks it, `_safe_str` rejects the
    type, `request_id` is never tried). Every such row then shared one `transaction_id`,
    so Lago billed the first and rejected the rest as duplicates — silently."""
    import uuid as _uuid

    a = {**_HOSTED, "invocation_id": None, "request_id": None, "input_tokens": "7"}
    b = {**_HOSTED, "invocation_id": None, "request_id": None, "input_tokens": "9"}
    rows = list(_source([], [a, b]).read_usage("1 day"))
    assert len(rows) == 2
    assert rows[0].row_id and rows[1].row_id
    assert rows[0].event_id != rows[1].event_id

    # A non-str id must be used, not skipped into the fallback.
    ident = _uuid.uuid4()
    (row,) = list(_source([], [{**_HOSTED, "invocation_id": ident}]).read_usage("1 day"))
    assert row.row_id == str(ident)


def test_the_id_fallback_is_deterministic_so_re_runs_stay_idempotent() -> None:
    """A random UUID would bill an id-less row again on every run."""
    row = {**_HOSTED, "invocation_id": None, "request_id": None}
    first = list(_source([], [row]).read_usage("1 day"))[0].event_id
    second = list(_source([], [row]).read_usage("1 day"))[0].event_id
    assert first == second


def test_byok_tokens_with_no_spend_row_are_reported_not_lost(caplog) -> None:
    """`external_model_spend` lags `ai_gateway.usage`, so the newest hour has token rows
    whose dollar row does not exist yet. The spend loop skips them (no dollars) and the
    hosted loop skips them (not databricks), so they were billed by neither and counted
    by nothing. Losing them quietly is the failure "never silently under-bill" forbids."""
    import logging as _logging

    with caplog.at_level(_logging.WARNING, logger="lago_agent_sdk.gateway.databricks"):
        rows = list(_source([], [_BYOK_USAGE]).read_usage("1 day"))
    assert rows == []
    assert any("no external_model_spend row yet" in r.getMessage() for r in caplog.records)


def test_an_aware_datetime_window_is_converted_to_utc() -> None:
    """`strftime` ignores tzinfo, so a Europe/Paris caller rendered local wall time
    against Databricks' UTC columns — a window two hours in the future that reads
    nothing and reports success. Also the JS port converts, so this kept the two repos
    reading different windows from the same input."""
    paris = timezone(timedelta(hours=2))
    aware = _window_bounds(datetime(2026, 8, 11, 14, 0, 0, tzinfo=paris), now=_NOW)[0]
    assert _timestamp_sql(aware) == "TIMESTAMP '2026-08-11 12:00:00+00:00'"
    # Naive is taken as UTC, matching the JS port's Date handling.
    naive = _window_bounds(datetime(2026, 8, 11, 14, 0, 0), now=_NOW)[0]
    assert _timestamp_sql(naive) == "TIMESTAMP '2026-08-11 14:00:00+00:00'"


def test_datetime_timestamp_columns_still_bucket_and_reconcile() -> None:
    """`databricks-sql-connector` returns TIMESTAMPs as `datetime`, not str. `_safe_str`
    mapped those to "", collapsing every hour into one join bucket and dropping the
    `bucket` reconcile dimension."""
    stamp = datetime(2026, 8, 7, 14, 22, 3)
    spend = {**_BYOK_SPEND, "bucket": datetime(2026, 8, 7, 14, 0, 0)}
    (byok,) = list(_source([spend], [{**_BYOK_USAGE, "event_time": stamp}]).read_usage("1 day"))
    assert byok.usage.input == 1825, "hour key must survive a datetime"
    # ISO-8601, matching the string form the REST API returns and the JS port's output.
    assert byok.reconcile_dimensions["bucket"] == "2026-08-07T14:00:00"


def test_a_malformed_usage_quantity_skips_its_row_instead_of_aborting() -> None:
    """`float("NULL")` raised out of the generator, through `backfill_databricks`, and
    into the caller — half a window emitted with no record of where it stopped, against
    a docstring promising one bad row cannot take down the batch."""
    rows = list(_source([{**_BYOK_SPEND, "usage_quantity": "NULL"}], [_BYOK_USAGE]).read_usage("1 day"))
    assert rows == []


def test_query_polls_a_statement_that_is_still_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """A statement still executing when `wait_timeout` elapses returns HTTP 200 with
    `state: PENDING` — not an error. Raising on it broke the exact usage this class
    recommends: one wide window per run, which on a cold warehouse exceeds the 50s
    ceiling Databricks allows."""
    import requests

    pending = {"statement_id": "s1", "status": {"state": "PENDING"}}
    running = {"statement_id": "s1", "status": {"state": "RUNNING"}}
    done = {
        "statement_id": "s1",
        "status": {"state": "SUCCEEDED"},
        "manifest": {"schema": {"columns": [{"name": "a"}]}, "total_chunk_count": 1},
        "result": {"data_array": [["1"]]},
    }
    replies = iter([running, done])
    monkeypatch.setattr(requests, "post", lambda *_a, **_k: _FakeResponse(pending))
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: _FakeResponse(next(replies)))
    monkeypatch.setattr("time.sleep", lambda _s: None)
    src = DatabricksSource(host="https://x", token="t", warehouse_id="w")
    assert src.query("SELECT 1") == [{"a": "1"}]


def test_query_gives_up_on_a_statement_that_never_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    pending = {"statement_id": "s1", "status": {"state": "PENDING"}}
    monkeypatch.setattr(requests, "post", lambda *_a, **_k: _FakeResponse(pending))
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: _FakeResponse(pending))
    monkeypatch.setattr("time.sleep", lambda _s: None)
    src = DatabricksSource(host="https://x", token="t", warehouse_id="w", timeout=0.0)
    with pytest.raises(RuntimeError, match="still PENDING"):
        src.query("SELECT 1")


def test_backfill_accepts_already_read_rows_without_querying_again() -> None:
    """The demo read the window to print a summary and then handed the SOURCE to
    `backfill_databricks`, re-running both warehouse queries — doubling the cost of the
    expensive half, and letting the printed summary disagree with what was billed."""
    src = _source([_BYOK_SPEND], [_HOSTED, _BYOK_USAGE])
    rows = list(src.read_usage("1 day"))
    queries_after_read = len(src.queries)  # type: ignore[attr-defined]

    sdk, q = _sdk()
    counts = sdk.backfill_databricks(rows, default_subscription="sub_x")
    _drain(sdk)
    assert counts == {"cost": 1, "tokens": 1, "skipped": 0}
    assert len(src.queries) == queries_after_read, "must not re-read"  # type: ignore[attr-defined]
    assert len(q.events) >= 3


# --------------------------------------------------------------------------
# Event time — a backfill runs long after the usage it bills
# --------------------------------------------------------------------------
def _utc(*args: int) -> int:
    return int(datetime(*args, tzinfo=timezone.utc).timestamp())  # type: ignore[arg-type]


def test_occurred_at_reads_each_kinds_own_time_column() -> None:
    """A usage row's own instant; a spend row's hour START — the only instant certain
    to sit inside the hour that row aggregates."""
    hosted, byok = list(_source([_BYOK_SPEND], [_HOSTED]).read_usage("1 day"))
    assert hosted.kind == "spend" and byok.kind == "usage"
    assert hosted.occurred_at == _utc(2026, 8, 7, 14, 0, 0)
    assert byok.occurred_at == _utc(2026, 8, 7, 14, 22, 3)


def test_occurred_at_reads_a_datetime_column_the_same_way() -> None:
    """`databricks-sql-connector` returns TIMESTAMPs as `datetime`, the REST API as
    ISO-8601 strings ending in "Z". Both are supported access paths, so both must
    resolve to the same instant."""
    rows = list(
        _source(
            [{**_BYOK_SPEND, "bucket": datetime(2026, 8, 7, 14, 0, 0)}],
            [{**_HOSTED, "event_time": "2026-08-07T14:22:03.123Z"}],
        ).read_usage("1 day")
    )
    assert {r.occurred_at for r in rows} == {_utc(2026, 8, 7, 14, 0, 0), _utc(2026, 8, 7, 14, 22, 3)}


def test_a_row_with_no_readable_time_has_no_occurred_at() -> None:
    """None leaves `emit()` stamping `now`, which is wrong but billed — better than
    losing the row over a bad column."""
    for bad in (None, "", "not a timestamp"):
        row = DatabricksUsageRow(
            usage=CanonicalUsage(model="m", provider="databricks", api="databricks_gateway"),
            subscription="sub",
            row_id="r",
            kind="usage",
            raw={"event_time": bad},
        )
        assert row.occurred_at is None


def test_backfill_events_carry_the_source_rows_time_not_the_run_time() -> None:
    """Live-proven before the fix: 128 events off one window spanning 2026-08-06 to
    2026-08-11 all carried the run's own clock, billing historical usage into the
    current period."""
    sdk, q = _sdk()
    sdk.backfill_databricks(_source([_BYOK_SPEND], [_HOSTED, _BYOK_USAGE]), "1 day")
    _drain(sdk)

    cost = [e for e in q.events if e["code"] == "llm_cost"]
    tokens = [e for e in q.events if e["code"] != "llm_cost"]
    assert cost and tokens
    # The spend row's hour, and the hosted request's own second.
    assert {e["timestamp"] for e in cost} == {_utc(2026, 8, 7, 14, 0, 0)}
    assert {e["timestamp"] for e in tokens} == {_utc(2026, 8, 7, 14, 22, 3)}
    assert all(e["timestamp"] < int(time.time()) - 86400 for e in q.events), "not the run time"

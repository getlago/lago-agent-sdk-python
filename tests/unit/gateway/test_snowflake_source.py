"""Snowflake Cortex usage reader — the I/O half, exercised without touching a warehouse.

Two levels. `SnowflakeSource.query` is faked for the reading tests, so the SQL it would
run is asserted; and `requests` itself is faked for the transport tests, reproducing the
SQL API's envelope exactly as it comes back — `resultSetMetaData.rowType` plus a
POSITIONAL `data` list, partition 0 inline, `partitionInfo` counting it.

The row shapes are the captured fixtures, not invented ones, except where a case has never
been observed on a live account (a multi-bucket query, a partial row). Those are hand-made
and labelled as such — the whole point of the deferral they exercise is that nobody has
seen the shape.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any

import pytest

from lago_agent_sdk import LagoSDK
from lago_agent_sdk.config import LagoConfig
from lago_agent_sdk.gateway.snowflake import (
    FUNCTIONS_COLUMNS,
    REST_COLUMNS,
    SnowflakeSource,
    SnowflakeUsageRow,
    _floor_hour,
    _timestamp_sql,
    _window_bounds,
)

_FIXTURES = pathlib.Path(__file__).parent / "adapters" / "fixtures" / "snowflake_cortex"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / f"{name}.json").read_text())


# --------------------------------------------------------------------------
# Rows. Captured ones first.
# --------------------------------------------------------------------------
_AI_COMPLETE = _fixture("functions_ai_complete")
_AI_EMBED = _fixture("functions_total_only_with_model")
_REST_PLAIN = _fixture("rest_plain")
_REST_CACHED = _fixture("rest_cache_read")
_ENVELOPE = _fixture("api_partition_envelope")

# A tagged functions row, so attribution has something to resolve. The tag shape is the one
# `resolve_snowflake_subscription` documents and the same key Cloudflare and Databricks read
# from their own metadata.
_TAGGED = {
    **_AI_COMPLETE,
    "QUERY_ID": "01c67fe9-tagged",
    "QUERY_TAG": '{"lago_subscription": "sub_tagged"}',
}

# HAND-MADE, and deliberately: an hour-bucketed query spanning three buckets writes three
# rows under ONE QUERY_ID. Snowflake documents it; no captured row shows it, because every
# query this account has ever run finished inside one bucket (48 of 48). The numbers are the
# INT-224 probe's real totals split three ways, which is what makes the ambiguity concrete —
# incremental they sum to the truth, cumulative they do not.
_SPANNING = [
    {
        **_AI_COMPLETE,
        "QUERY_ID": "01c67fe9-spanning",
        "START_TIME": start,
        "END_TIME": str(int(start) + 3600),
        "IS_COMPLETED": "true" if i == 2 else "false",
        "METRICS": json.dumps(
            [
                {"key": {"metric": "input", "unit": "tokens"}, "value": 900 + i * 100},
                {"key": {"metric": "output", "unit": "tokens"}, "value": 2000 + i * 100},
            ]
        ),
    }
    for i, start in enumerate(["1787162400", "1787166000", "1787169600"])
]

# A partial row alone in the window — the other half of the deferral rule. Its QUERY_ID does
# not collide with anything here, so only the flag defers it.
_INCOMPLETE = {**_AI_COMPLETE, "QUERY_ID": "01c67fe9-incomplete", "IS_COMPLETED": "false"}

# A failed call produces NO row on either view (measured with a same-batch control), so a
# zero-token row is a shape nobody has seen. Written anyway: it must bill nothing rather
# than emit an empty event.
_ZERO_USAGE = {**_AI_COMPLETE, "QUERY_ID": "01c67fe9-zero", "METRICS": "[]", "CREDITS": "0"}


def _source(functions: list[dict] | None = None, rest: list[dict] | None = None) -> SnowflakeSource:
    """A source whose `query` answers from canned rows, keyed on which view the SQL names."""
    src = SnowflakeSource("ORG-ACCT", "tok", warehouse="COMPUTE_WH")
    seen: list[str] = []

    def fake_query(sql: str) -> list[dict[str, Any]]:
        seen.append(sql)
        if "CORTEX_REST_API_USAGE_HISTORY" in sql:
            return list(rest or [])
        return list(functions or [])

    src.query = fake_query  # type: ignore[method-assign]
    src.queries = seen  # type: ignore[attr-defined]
    return src


# --------------------------------------------------------------------------
# Rule 4 — the window
# --------------------------------------------------------------------------
_NOW = datetime(2026, 8, 26, 13, 34, 12, tzinfo=timezone.utc)


def test_window_resolves_interval_strings_to_instants_not_sql() -> None:
    """Resolved in Python so both views can share one bound. A `DATEADD(hour, -N,
    CURRENT_TIMESTAMP())` in the SQL is re-evaluated per statement, and a row landing in
    the drift between the two reads is read by neither."""
    lower, upper = _window_bounds("2 hours", now=_NOW)
    assert lower == datetime(2026, 8, 26, 11, tzinfo=timezone.utc)
    assert upper == datetime(2026, 8, 26, 13, tzinfo=timezone.utc)


def test_window_rejects_an_interval_it_does_not_understand() -> None:
    """Under-reading is the one direction that loses money, so a window that cannot be
    parsed must not silently become a narrower one."""
    with pytest.raises(ValueError, match="not understood"):
        _window_bounds("last tuesday", now=_NOW)
    with pytest.raises(ValueError, match="not understood"):
        _window_bounds("2 fortnights", now=_NOW)


def test_window_floors_both_bounds_to_the_hour() -> None:
    """Both views are hour-bucketed and START_TIME is always the hour START, so a mid-hour
    lower bound drops every row of the hour containing it — including calls made after the
    bound itself."""
    lower, upper = _window_bounds(datetime(2026, 8, 26, 9, 47, 31, tzinfo=timezone.utc), now=_NOW)
    assert lower == datetime(2026, 8, 26, 9, tzinfo=timezone.utc)
    assert upper == datetime(2026, 8, 26, 13, tzinfo=timezone.utc)


def test_window_excludes_the_still_aggregating_hour() -> None:
    """13:34 reads up to 13:00, never past it. The functions view lands a row ~141s after
    its query ends, so the open hour is incomplete by construction; billing it early burns
    the row's QUERY_ID-derived transaction_id and the correction is then rejected as a
    duplicate."""
    assert _window_bounds("1 day", now=_NOW)[1] == datetime(2026, 8, 26, 13, tzinfo=timezone.utc)


def test_window_reads_a_naive_datetime_as_utc() -> None:
    """A naive bound formatted as local wall time would move the window by the machine's
    offset, so a Europe/Paris caller reads two hours into the future and bills nothing."""
    lower, _ = _window_bounds(datetime(2026, 8, 26, 9, 30), now=_NOW)
    assert lower == datetime(2026, 8, 26, 9, tzinfo=timezone.utc)


def test_timestamp_literals_name_the_zone() -> None:
    """Rule 6. A bare literal is parsed in the session's TIMEZONE, and Snowflake accounts
    default to America/Los_Angeles — so the whole window would slide 7-8 hours."""
    assert _timestamp_sql(datetime(2026, 8, 26, 13, tzinfo=timezone.utc)) == "'2026-08-26 13:00:00+00:00'"


def test_floor_hour_moves_the_instant() -> None:
    assert _floor_hour(datetime(2026, 8, 26, 13, 59, 59, tzinfo=timezone.utc)) == datetime(
        2026, 8, 26, 13, tzinfo=timezone.utc
    )


def test_both_views_are_scoped_to_one_shared_window() -> None:
    """Rule 4's core: ONE literal pair, both statements. Two separately-resolved windows is
    the drift bug again."""
    import re

    src = _source([_AI_COMPLETE], [_REST_PLAIN])
    list(src.read_usage("3 hours", views=("functions", "rest")))
    queries = src.queries  # type: ignore[attr-defined]
    assert len(queries) == 2
    literals = [re.findall(r"'[^']+\+00:00'", q) for q in queries]
    assert len(literals[0]) == 2
    assert literals[0] == literals[1]


def test_a_window_inside_the_open_hour_reads_nothing_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Zero rows here says nothing about whether there was traffic, so it must not read as
    success.

    The clock is not pinned the way the JS port's is — `read_usage` calls `_window_bounds`
    with no `now`, so a sub-hour interval only collapses when it is evaluated past the half
    hour. Asserted through `_window_bounds` directly instead, plus the reader's behaviour on
    a window it has already resolved as empty.
    """
    lower, upper = _window_bounds("30 minutes", now=_NOW)
    assert lower == upper

    src = _source([_AI_COMPLETE])
    with caplog.at_level("WARNING"):
        # An explicit datetime inside the open hour is the same empty window, and does not
        # depend on when the test runs.
        assert list(src.read_usage(datetime.now(timezone.utc))) == []
    assert src.queries == []  # type: ignore[attr-defined]
    assert "still-aggregating hour" in caplog.text


# --------------------------------------------------------------------------
# Rule 5 — the projection
# --------------------------------------------------------------------------
def _projection_of(sql: str) -> list[str]:
    import re

    m = re.search(r"SELECT\s+(.*?)\s+FROM", sql, re.S | re.I)
    return [s.strip() for s in (m.group(1) if m else "").split(",")]


def test_the_projection_never_uses_select_star() -> None:
    """`*` risks the inline-response size cap, which FAILS a statement rather than
    paginating past it — and these views gain columns without notice (this account watched
    the REST view go from 8 to 9 in eight hours), so `*` is a width nobody controls."""
    src = _source([_AI_COMPLETE], [_REST_PLAIN])
    list(src.read_usage("3 hours", views=("functions", "rest")))
    for sql in src.queries:  # type: ignore[attr-defined]
        assert "SELECT *" not in sql


def test_the_projection_names_exactly_its_columns() -> None:
    src = _source([_AI_COMPLETE], [_REST_PLAIN])
    list(src.read_usage("3 hours", views=("functions", "rest")))
    queries = src.queries  # type: ignore[attr-defined]
    assert _projection_of(queries[0]) == list(FUNCTIONS_COLUMNS)
    assert _projection_of(queries[1]) == list(REST_COLUMNS)


def test_the_projection_covers_every_column_the_extraction_reads() -> None:
    """THE coupling test. A column dropped from the projection reaches the adapter as
    ABSENT, where every field degrades to zero rather than raising — an under-billed event
    with no error anywhere. Asserted against the captured rows' own keys, so a column the
    fixtures prove exists cannot be quietly dropped from the read."""
    for key in _AI_COMPLETE:
        assert key in FUNCTIONS_COLUMNS
    for key in _REST_PLAIN:
        assert key in REST_COLUMNS
    assert "IS_COMPLETED" in FUNCTIONS_COLUMNS
    assert "METRICS" in FUNCTIONS_COLUMNS
    assert "TOKENS_GRANULAR" in REST_COLUMNS


# --------------------------------------------------------------------------
# Rules 1, 2, 3 — the transport
# --------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


def _envelope(rows: list[list[Any]], num_rows: int | None = None, partitions: int = 1) -> dict:
    """One SQL API response, in the exact envelope shape the API returns."""
    return {
        "resultSetMetaData": {
            "numRows": str(num_rows if num_rows is not None else len(rows)),
            "rowType": [{"name": "QUERY_ID"}, {"name": "METRICS"}],
            "partitionInfo": [{"rowCount": len(rows)} for _ in range(partitions)],
            "partitionContentEncoding": "gzip",
        },
        "data": rows,
        "statementHandle": "01c6-handle",
    }


def _transport(**kwargs: Any) -> SnowflakeSource:
    kwargs.setdefault("warehouse", "COMPUTE_WH")
    return SnowflakeSource("ORG-ACCT", "tok", **kwargs)


def test_query_zips_row_type_against_the_positional_data_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """The API returns rows POSITIONALLY, not as objects. Getting this wrong yields `{}` per
    row, which every layer downstream degrades cleanly and wrongly on."""
    import requests

    monkeypatch.setattr(requests, "post", lambda url, **_kw: _FakeResponse(_envelope([["q1", "[]"]])))
    assert _transport().query("SELECT 1") == [{"QUERY_ID": "q1", "METRICS": "[]"}]


def test_the_captured_envelope_counts_partition_zero() -> None:
    """Verified against the captured 60,000-row envelope: 8 partitionInfo entries whose
    rowCounts sum to numRows, the FIRST being the rows already inline. Starting the fetch
    loop at 1 is what makes that true — starting at 0 re-reads the inline rows and doubles
    them."""
    assert _ENVELOPE["partitionCount"] == 8
    assert _ENVELOPE["partitionInfo"][0]["rowCount"] == _ENVELOPE["inlineRowCount"]
    assert sum(p["rowCount"] for p in _ENVELOPE["partitionInfo"]) == _ENVELOPE["numRows"]


def test_query_follows_every_partition_but_never_partition_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    seen: list[int] = []

    def fake_get(url: str, **kw: Any) -> _FakeResponse:
        seen.append(int(kw["params"]["partition"]))
        return _FakeResponse({"data": [["q2", "[]"]]})

    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **_kw: _FakeResponse(_envelope([["q1", "[]"]], num_rows=3, partitions=3)),
    )
    monkeypatch.setattr(requests, "get", fake_get)
    rows = _transport().query("SELECT 1")
    assert [r["QUERY_ID"] for r in rows] == ["q1", "q2", "q2"]
    assert seen == [1, 2]


def test_query_raises_when_a_partition_fetch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """RULE 1, and the single worst outcome this reader can produce. A failed partition
    returns a JSON error body with no `data`, so a tolerant `or []` appends nothing, the
    loop continues, and the read reports success over a fraction of the window. On the
    captured envelope that is 472 of 60,000 rows."""
    import requests

    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **_kw: _FakeResponse(_envelope([["q1", "[]"]], num_rows=2, partitions=2)),
    )
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, **_kw: _FakeResponse(
            {"code": "390403", "message": "insufficient privileges"}, status_code=403
        ),
    )
    with pytest.raises(RuntimeError, match=r"partition 1 of 2 failed: HTTP 403"):
        _transport().query("SELECT 1")


def test_query_error_carries_snowflakes_own_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """`003001` alone has four distinct causes on this account, so the message is the only
    thing that tells an operator which one they hit."""
    import requests

    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **_kw: _FakeResponse(
            {"code": "390318", "message": "Authentication token has expired"}, status_code=401
        ),
    )
    with pytest.raises(RuntimeError, match="Authentication token has expired"):
        _transport().query("SELECT 1")


def test_query_raises_when_the_row_count_misses_num_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """RULE 2, end-to-end and independent of cause: catches a partition that answers HTTP
    200 with fewer rows than promised, which no per-request status check can see."""
    import requests

    monkeypatch.setattr(
        requests, "post", lambda url, **_kw: _FakeResponse(_envelope([["q1", "[]"]], num_rows=9))
    )
    with pytest.raises(RuntimeError, match=r"returned 1 row\(s\) but the statement promised 9"):
        _transport().query("SELECT 1")


def test_query_raises_when_rows_arrive_with_no_row_type(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **_kw: _FakeResponse(
            {
                "resultSetMetaData": {
                    "numRows": "1",
                    "rowType": [],
                    "partitionInfo": [{"rowCount": 1}],
                },
                "data": [["q1", "[]"]],
                "statementHandle": "h",
            }
        ),
    )
    with pytest.raises(RuntimeError, match="no `resultSetMetaData.rowType`"):
        _transport().query("SELECT 1")


def test_query_polls_a_202_to_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """RULE 3. A 202 is the EXPECTED answer for the one wide window per run this class tells
    operators to read; treating it as fatal breaks exactly that case."""
    import requests

    polls = {"n": 0}

    def fake_get(url: str, **_kw: Any) -> _FakeResponse:
        polls["n"] += 1
        if polls["n"] < 2:
            return _FakeResponse({"statementHandle": "h", "message": "running"}, status_code=202)
        return _FakeResponse(_envelope([["q1", "[]"]]))

    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **_kw: _FakeResponse(
            {"statementHandle": "h", "statementStatusUrl": "/api/v2/statements/h"}, status_code=202
        ),
    )
    monkeypatch.setattr(requests, "get", fake_get)
    assert _transport(poll_interval=0.0).query("SELECT 1") == [{"QUERY_ID": "q1", "METRICS": "[]"}]
    assert polls["n"] == 2


def test_query_bounds_a_202_that_never_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both bounds, because either alone leaves a hole: a statement answering every poll
    instantly would spin timeout/interval times on the cap alone, and a slow poll would
    outlive the deadline without it.

    Also the port's answer to the JS AC about a leftover timer: Python has no event loop to
    keep alive, so the property that matters is that the loop is finite. Asserted as an
    exact poll count.
    """
    import requests

    polls = {"n": 0}

    def fake_get(url: str, **_kw: Any) -> _FakeResponse:
        polls["n"] += 1
        return _FakeResponse({"statementHandle": "h", "message": "running"}, status_code=202)

    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **_kw: _FakeResponse(
            {"statementHandle": "h", "statementStatusUrl": "/api/v2/statements/h"}, status_code=202
        ),
    )
    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(RuntimeError, match=r"still running after .* 3 poll\(s\)"):
        _transport(poll_interval=0.0, max_polls=3).query("SELECT 1")
    assert polls["n"] == 3


def test_query_refuses_a_202_with_no_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    monkeypatch.setattr(
        requests, "post", lambda url, **_kw: _FakeResponse({"message": "accepted"}, status_code=202)
    )
    with pytest.raises(RuntimeError, match="202 with no statementHandle"):
        _transport().query("SELECT 1")


def test_query_sends_the_pat_token_type_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without this header a PAT is read as an OAuth token and rejected as invalid — and
    the error says nothing about the header, so it reads like a bad secret."""
    import requests

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kw: Any) -> _FakeResponse:
        captured.update(kw)
        return _FakeResponse(_envelope([]))

    monkeypatch.setattr(requests, "post", fake_post)
    _transport().query("SELECT 1")
    assert captured["headers"]["X-Snowflake-Authorization-Token-Type"] == "PROGRAMMATIC_ACCESS_TOKEN"
    assert captured["headers"]["Authorization"] == "Bearer tok"


def test_query_passes_the_warehouse_and_role_as_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Context values are case-SENSITIVE here, unlike an unquoted SQL identifier. And
    reading ACCOUNT_USAGE needs a running warehouse at all — the live wrap() path does not,
    which is why this is the setup step people miss."""
    import requests

    captured: dict[str, Any] = {}

    def fake_post(url: str, **kw: Any) -> _FakeResponse:
        captured.update(kw)
        return _FakeResponse(_envelope([]))

    monkeypatch.setattr(requests, "post", fake_post)
    _transport(role="LAGO_CORTEX_ROLE").query("SELECT 1")
    assert captured["json"]["warehouse"] == "COMPUTE_WH"
    assert captured["json"]["role"] == "LAGO_CORTEX_ROLE"
    assert captured["json"]["timeout"] == 120


# --------------------------------------------------------------------------
# The functions view, and the QUERY_ID that is not unique
# --------------------------------------------------------------------------
def test_a_single_bucket_row_bills_keyed_on_its_query_id() -> None:
    rows = list(_source([_AI_COMPLETE]).read_usage("3 hours"))
    assert len(rows) == 1
    assert rows[0].kind == "functions"
    assert rows[0].row_id == "01c67fe9-0302-ce36-001e-606300034516"
    assert rows[0].usage.input == 13
    assert rows[0].usage.output == 5


def test_a_total_only_row_bills_which_is_five_of_six_function_types() -> None:
    """The adapter maps `total` onto `input` and marks it; without that, five of six
    function types extract to all-zero and emit nothing — a silent 100% under-bill."""
    rows = list(_source([_AI_EMBED]).read_usage("3 hours"))
    assert rows[0].usage.input == 3
    assert rows[0].usage.extras["metrics_total_only"] is True


def test_every_row_of_a_colliding_query_id_is_deferred(caplog: pytest.LogCaptureFixture) -> None:
    """The collision itself: `{prefix}_{kind}_{sub}_{QUERY_ID}` cannot distinguish these
    three rows, so Lago accepts one and rejects two as duplicates — taking the whole batch
    down the all-or-nothing split path. And whether the per-window METRICS is incremental or
    cumulative is unmeasured, so neither summing them nor taking the last is defensible: on
    this row set summing bills 3300 input and taking the final row bills 1100, and only one
    of those is right."""
    src = _source(_SPANNING)
    with caplog.at_level("WARNING"):
        assert list(src.read_usage("1 day")) == []
    assert len(src.deferred_rows) == 3
    assert src.deferred_rows[0]["reason"] == "multi_bucket"
    assert src.deferred_rows[0]["buckets"] == ["1787162400", "1787166000", "1787169600"]
    assert "were NOT billed" in caplog.text


def test_a_row_flagged_incomplete_is_deferred_even_alone(caplog: pytest.LogCaptureFixture) -> None:
    src = _source([_INCOMPLETE])
    with caplog.at_level("WARNING"):
        assert list(src.read_usage("1 day")) == []
    assert src.deferred_rows[0]["reason"] == "incomplete"


def test_an_absent_or_unrecognized_is_completed_does_not_defer() -> None:
    """Only an EXPLICIT false defers. A view that stopped populating the column, or one that
    spelled the value differently, must not silently defer an entire window — that turns a
    guard into a 100% under-bill."""
    no_flag = {**_AI_COMPLETE, "QUERY_ID": "q-noflag", "IS_COMPLETED": None}
    odd = {**_AI_COMPLETE, "QUERY_ID": "q-odd", "IS_COMPLETED": "COMPLETED"}
    src = _source([no_flag, odd])
    assert len(list(src.read_usage("1 day"))) == 2
    assert src.deferred_rows == []


def test_is_completed_is_read_as_a_real_boolean_too(caplog: pytest.LogCaptureFixture) -> None:
    """A typed connector yields a real `False` where the SQL API yields the string."""
    src = _source([{**_AI_COMPLETE, "QUERY_ID": "q-bool", "IS_COMPLETED": False}])
    with caplog.at_level("WARNING"):
        assert list(src.read_usage("1 day")) == []
    assert src.deferred_rows[0]["reason"] == "incomplete"


def test_healthy_rows_still_bill_in_a_window_holding_a_deferred_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One bad row degrades, the rest still bills. A deferral that took the window down with
    it would be a far worse bug than the one it guards."""
    src = _source([_AI_COMPLETE, *_SPANNING, _AI_EMBED])
    with caplog.at_level("WARNING"):
        rows = list(src.read_usage("1 day"))
    assert sorted(r.row_id for r in rows) == sorted(
        ["01c67fe9-0302-ce36-001e-606300034516", "01c6a022-0102-dd95-001e-6063000d07fa"]
    )
    assert len(src.deferred_rows) == 3


def test_a_row_that_bills_nothing_is_skipped() -> None:
    assert list(_source([_ZERO_USAGE]).read_usage("1 day")) == []


def test_a_row_with_unparseable_metrics_degrades_alone() -> None:
    """Rule 7: a parse failure degrades to empty rather than raising, so one bad row cannot
    take down a window. Nothing billable comes out, so nothing is emitted."""
    src = _source([{**_AI_COMPLETE, "QUERY_ID": "q-bad", "METRICS": "{not json"}, _AI_EMBED])
    rows = list(src.read_usage("1 day"))
    assert [r.row_id for r in rows] == ["01c6a022-0102-dd95-001e-6063000d07fa"]


def test_a_second_read_clears_the_first_reads_deferred_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A later read of a healthy window must not leave an earlier read's gap standing as if
    it were current."""
    src = _source(_SPANNING)
    with caplog.at_level("WARNING"):
        list(src.read_usage("1 day"))
    assert len(src.deferred_rows) == 3
    src.query = lambda sql: [_AI_COMPLETE]  # type: ignore[method-assign]
    list(src.read_usage("1 day"))
    assert src.deferred_rows == []


# --------------------------------------------------------------------------
# The REST view, and the double-bill it can cause
# --------------------------------------------------------------------------
def test_the_rest_view_is_not_read_by_default() -> None:
    """THE default that keeps a live-path customer from billing every REST call twice. The
    live path's transaction_id is a random UUID and this reader's derives from REQUEST_ID,
    so Lago cannot reject the duplicate."""
    src = _source([_AI_COMPLETE], [_REST_PLAIN])
    rows = list(src.read_usage("3 hours"))
    assert all(r.kind == "functions" for r in rows)
    queries = src.queries  # type: ignore[attr-defined]
    assert len(queries) == 1
    assert "CORTEX_AI_FUNCTIONS_USAGE_HISTORY" in queries[0]


def test_the_rest_view_is_read_when_asked_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    src = _source([], [_REST_PLAIN])
    with caplog.at_level("WARNING"):
        rows = list(src.read_usage("3 hours", views=("rest",)))
    assert len(rows) == 1
    assert rows[0].kind == "rest"
    assert rows[0].row_id == "8e1249f1-a9af-463e-8bb8-ed409269c61c"
    assert "live path ALREADY billed them" in caplog.text


def test_the_rest_views_additive_cache_block_stays_out_of_input(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The token-shape guard, restated at the reader level. `TOKENS` INCLUDES the cached
    block, so mapping it onto input would bill 8758 for 7 real input tokens and re-bill the
    cache a second time as cache_read — 2.0x on the call."""
    with caplog.at_level("WARNING"):
        rows = list(_source([], [_REST_CACHED]).read_usage("3 hours", views=("rest",)))
    usage = rows[0].usage
    assert usage.cache_read > 0
    assert usage.input < usage.cache_read
    # The reconciliation tripwire: the billed fields must sum to Snowflake's own TOKENS.
    assert sum(usage.nonzero_numeric().values()) == int(_REST_CACHED["TOKENS"])


# --------------------------------------------------------------------------
# Event ids
# --------------------------------------------------------------------------
def test_event_ids_are_unique_per_row_and_scoped_by_subscription() -> None:
    """`transaction_id` is unique ORG-wide, so an unscoped id permanently blocks that row
    from ever reaching a second subscription."""
    rows = list(_source([_AI_COMPLETE, _AI_EMBED]).read_usage("1 day"))
    assert len({r.event_id_for("sub_a") for r in rows}) == 2
    assert rows[0].event_id_for("sub_a") != rows[0].event_id_for("sub_b")
    assert rows[0].event_id_for("sub_a") == "sfc_functions_sub_a_01c67fe9-0302-ce36-001e-606300034516"


def test_an_unattributed_row_says_none_rather_than_collapsing() -> None:
    usage = list(_source([_AI_COMPLETE]).read_usage("1 day"))[0].usage
    row = SnowflakeUsageRow(usage=usage, subscription=None, row_id="r1", kind="functions")
    assert row.event_id == "sfc_functions_none_r1"


def test_the_prefix_namespaces_the_whole_read() -> None:
    rows = list(_source([_AI_COMPLETE]).read_usage("1 day", event_id_prefix="sfc2"))
    assert rows[0].event_id_for("sub_a").startswith("sfc2_functions_")


def test_a_row_with_no_id_falls_back_to_a_deterministic_hash() -> None:
    """A row with a NULL id yields "", and `event_id_for` would still produce a well-formed
    key — so EVERY such row in the window would share one transaction_id, Lago would accept
    the first and reject the rest, and those calls would never bill. The hash stays
    deterministic so a re-run is still idempotent, which a UUID would break."""
    a = {**_AI_COMPLETE, "QUERY_ID": None}
    b = {**_AI_COMPLETE, "QUERY_ID": None, "CREDITS": "0.000099999"}
    rows = list(_source([a, b]).read_usage("1 day"))
    assert len(rows) == 2
    assert rows[0].row_id.startswith("sha")
    assert len(rows[0].row_id) == 35
    assert rows[0].row_id != rows[1].row_id
    again = list(_source([a]).read_usage("1 day"))
    assert again[0].row_id == rows[0].row_id


# --------------------------------------------------------------------------
# Stamping and dimensions
# --------------------------------------------------------------------------
def test_occurred_at_reads_the_functions_views_bare_ltz_epoch() -> None:
    rows = list(_source([_AI_COMPLETE]).read_usage("1 day"))
    assert rows[0].occurred_at == 1787162400


def test_occurred_at_reads_the_rest_views_tz_epoch_ignoring_the_display_offset(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ "1787162400.000000000 1440" — the epoch is the true instant and 1440 is offset
    minutes + 1440, i.e. UTC. Adding it would slide every REST event by the account's
    timezone."""
    with caplog.at_level("WARNING"):
        rows = list(_source([], [_REST_PLAIN]).read_usage("1 day", views=("rest",)))
    assert rows[0].occurred_at == 1787162400


def test_occurred_at_is_the_start_of_the_hour() -> None:
    rows = list(_source([_AI_COMPLETE]).read_usage("1 day"))
    stamped = datetime.fromtimestamp(rows[0].occurred_at or 0, tz=timezone.utc)
    # Exactly on the hour, which is what makes the stamp safe: an hourly bucket covers
    # [start, start + 1h), so the start is the only instant certain to sit inside the row's
    # own coverage.
    assert stamped == datetime(2026, 8, 19, 18, tzinfo=timezone.utc)


def test_occurred_at_degrades_to_none_on_an_unreadable_stamp() -> None:
    """emit() then stamps `now` — worse than the real time, better than dropping the row."""
    rows = list(_source([{**_AI_COMPLETE, "START_TIME": "not a time"}]).read_usage("1 day"))
    assert rows[0].occurred_at is None


def test_occurred_at_truncates_toward_zero_like_math_trunc() -> None:
    """A pre-epoch stamp must round the same way in both ports, or the two repos bill the
    same row a second apart."""
    rows = list(_source([{**_AI_COMPLETE, "START_TIME": "-1.5"}]).read_usage("1 day"))
    assert rows[0].occurred_at == -1


def test_reconcile_dimensions_carry_the_functions_views_grouping_key() -> None:
    rows = list(_source([_AI_COMPLETE]).read_usage("1 day"))
    assert rows[0].reconcile_dimensions == {
        "function_name": "AI_COMPLETE",
        "model_name": "claude-sonnet-4-5",
    }


def test_reconcile_dimensions_omit_an_empty_model() -> None:
    """AI_SUMMARIZE/TRANSLATE/SENTIMENT/CLASSIFY take no model argument, so empty is a fact
    about the row. A blank dimension value is a group nobody can read."""
    rows = list(_source([{**_AI_COMPLETE, "QUERY_ID": "q-nomodel", "MODEL_NAME": ""}]).read_usage("1 day"))
    assert rows[0].reconcile_dimensions == {"function_name": "AI_COMPLETE"}


def test_reconcile_dimensions_carry_the_rest_views_region_not_its_request_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A per-request id is a list, not a grouping — one Lago group per request is not a
    comparison anyone can read."""
    with caplog.at_level("WARNING"):
        rows = list(_source([], [_REST_PLAIN]).read_usage("1 day", views=("rest",)))
    assert rows[0].reconcile_dimensions == {"inference_region": "aws_global"}


# --------------------------------------------------------------------------
# backfill_snowflake
# --------------------------------------------------------------------------
class _Recorder:
    """Collects delivered events, so assertions read the real emitted shape."""

    def __init__(self) -> None:
        self.batches: list[list[dict]] = []

    @property
    def events(self) -> list[dict]:
        return [e for b in self.batches for e in b]


def _sdk(errors: list[tuple[str, str]] | None = None) -> tuple[LagoSDK, _Recorder]:
    rec = _Recorder()
    cfg = (
        LagoConfig(api_key="dummy", on_error=lambda exc, where: errors.append((where, str(exc))))
        if errors is not None
        else None
    )
    sdk = LagoSDK(api_key="dummy", config=cfg)
    sdk._queue._sender = lambda b: rec.batches.append(list(b))  # type: ignore[attr-defined]
    return sdk, rec


def _drain(sdk: LagoSDK) -> None:
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)


def test_backfill_counts_tokens_and_skips_and_never_a_cost() -> None:
    """There is no price mode on this path: Snowflake meters Cortex in CREDITS against a
    rate card that exists in no view, so there is no per-request dollar figure to pass
    through and `provider = "snowflake"` is in TOKEN_BILLED_PROVIDERS."""
    sdk, _ = _sdk()
    counts = sdk.backfill_snowflake(
        _source([_TAGGED, _AI_EMBED]), "1 day", default_subscription="sub_default"
    )
    _drain(sdk)
    assert counts == {"tokens": 2, "skipped": 0}


def test_backfill_emits_no_cost_event() -> None:
    sdk, q = _sdk()
    sdk.backfill_snowflake(_source([_TAGGED]), "1 day")
    _drain(sdk)
    assert q.events
    assert not any("cost" in str(e.get("code")) for e in q.events)


def test_backfill_routes_a_tagged_row_to_its_own_subscription() -> None:
    sdk, q = _sdk()
    sdk.backfill_snowflake(
        _source([_TAGGED]),
        "1 day",
        default_subscription="sub_default",
        subscription_order=("query_tag",),
    )
    _drain(sdk)
    assert {e["external_subscription_id"] for e in q.events} == {"sub_tagged"}


def test_backfill_falls_back_to_the_default_subscription() -> None:
    sdk, q = _sdk()
    sdk.backfill_snowflake(
        _source([_AI_COMPLETE]),
        "1 day",
        default_subscription="sub_default",
        subscription_order=("query_tag",),
    )
    _drain(sdk)
    assert {e["external_subscription_id"] for e in q.events} == {"sub_default"}


def test_backfill_unified_ignores_each_rows_own_attribution() -> None:
    sdk, q = _sdk()
    sdk.backfill_snowflake(_source([_TAGGED]), "1 day", default_subscription="sub_one", unified=True)
    _drain(sdk)
    assert {e["external_subscription_id"] for e in q.events} == {"sub_one"}


def test_backfill_stamps_each_event_with_the_rows_own_hour() -> None:
    """A backfill that stamps `now` bills last week's usage into this week's period, and
    nothing in Lago can tell afterwards."""
    sdk, q = _sdk()
    sdk.backfill_snowflake(_source([_TAGGED]), "1 day", default_subscription="s")
    _drain(sdk)
    assert all(e["timestamp"] == 1787162400 for e in q.events)


def test_backfill_merges_row_dimensions_under_the_callers() -> None:
    sdk, q = _sdk()
    sdk.backfill_snowflake(
        _source([_TAGGED]),
        "1 day",
        default_subscription="s",
        dimensions={"model_name": "caller wins", "tenant": "acme"},
    )
    _drain(sdk)
    props = q.events[0]["properties"]
    assert props["function_name"] == "AI_COMPLETE"
    assert props["model_name"] == "caller wins"
    assert props["tenant"] == "acme"


def test_backfill_is_idempotent_across_a_re_run() -> None:
    """Every id derives from the source row, so Lago rejects the duplicates rather than
    double-billing. Asserted as identical transaction_ids, which is what Lago keys on."""
    sdk, q = _sdk()
    sdk.backfill_snowflake(_source([_TAGGED]), "1 day", default_subscription="s")
    _drain(sdk)
    first = sorted(e["transaction_id"] for e in q.events)

    sdk2, q2 = _sdk()
    sdk2.backfill_snowflake(_source([_TAGGED]), "1 day", default_subscription="s")
    _drain(sdk2)
    assert sorted(e["transaction_id"] for e in q2.events) == first


def test_backfill_does_not_read_the_rest_view_unless_named() -> None:
    sdk, _ = _sdk()
    src = _source([_AI_COMPLETE], [_REST_PLAIN])
    sdk.backfill_snowflake(src, "1 day", default_subscription="s")
    _drain(sdk)
    assert len(src.queries) == 1  # type: ignore[attr-defined]


def test_backfill_forwards_views_when_the_caller_names_it(caplog: pytest.LogCaptureFixture) -> None:
    sdk, _ = _sdk()
    src = _source([], [_REST_PLAIN])
    with caplog.at_level("WARNING"):
        counts = sdk.backfill_snowflake(src, "1 day", default_subscription="s", views=("rest",))
    _drain(sdk)
    assert counts["tokens"] == 1
    assert "CORTEX_REST_API_USAGE_HISTORY" in src.queries[0]  # type: ignore[attr-defined]


def test_backfill_accepts_an_already_read_iterable() -> None:
    """Reading twice is not just slow: a SQL warehouse is a real cost centre, and rows
    landing between the two reads make the summary you printed disagree with what was
    billed."""
    sdk, _ = _sdk()
    rows = list(_source([_TAGGED]).read_usage("1 day"))
    assert sdk.backfill_snowflake(rows, "1 day", default_subscription="s") == {
        "tokens": 1,
        "skipped": 0,
    }
    _drain(sdk)


def test_an_unattributed_row_is_reported_through_on_error() -> None:
    errors: list[tuple[str, str]] = []
    sdk, _ = _sdk(errors)
    counts = sdk.backfill_snowflake(_source([_AI_COMPLETE]), "1 day", subscription_order=("query_tag",))
    _drain(sdk)
    assert counts == {"tokens": 0, "skipped": 1}
    assert len(errors) == 1
    assert errors[0][0] == "backfill"
    assert "no resolvable subscription" in errors[0][1]


def test_a_deferred_row_is_reported_through_on_error_and_counted_as_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A deferral that only logged would be a billing gap no automated caller could see —
    `tokens` dropping is not something a script can read as a gap."""
    errors: list[tuple[str, str]] = []
    sdk, _ = _sdk(errors)
    with caplog.at_level("WARNING"):
        counts = sdk.backfill_snowflake(_source(_SPANNING), "1 day", default_subscription="s")
    _drain(sdk)
    assert counts == {"tokens": 0, "skipped": 3}
    assert "unbilled" in " ".join(e[1] for e in errors)
    assert "01c67fe9-spanning" in " ".join(e[1] for e in errors)


def test_the_two_skip_causes_are_reported_separately(caplog: pytest.LogCaptureFixture) -> None:
    """They are fixed differently: an unattributed row needs a tag or a default, while a
    deferred row is billable revenue awaiting a measurement."""
    errors: list[tuple[str, str]] = []
    sdk, _ = _sdk(errors)
    with caplog.at_level("WARNING"):
        counts = sdk.backfill_snowflake(
            _source([_AI_COMPLETE, *_SPANNING]), "1 day", subscription_order=("query_tag",)
        )
    _drain(sdk)
    assert counts == {"tokens": 0, "skipped": 4}
    assert len(errors) == 2
    assert "no resolvable subscription" in errors[0][1]
    assert "unbilled" in errors[1][1]


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------
def test_from_env_names_every_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SNOWFLAKE_ACCOUNT", raising=False)
    monkeypatch.delenv("SNOWFLAKE_HOST", raising=False)
    monkeypatch.delenv("SNOWFLAKE_PAT", raising=False)
    with pytest.raises(ValueError, match="SNOWFLAKE_ACCOUNT.*SNOWFLAKE_PAT"):
        SnowflakeSource.from_env()


def test_the_account_identifier_and_a_full_host_both_work() -> None:
    """Snowflake's docs and error messages use the account form; appending
    `.snowflakecomputing.com` twice is a 404 that reads like a bad account."""
    assert SnowflakeSource("MYORG-ACCOUNT123", "t").host == "MYORG-ACCOUNT123.snowflakecomputing.com"
    assert SnowflakeSource("acct.snowflakecomputing.com", "t").host == "acct.snowflakecomputing.com"
    assert SnowflakeSource("https://acct.snowflakecomputing.com/", "t").host == "acct.snowflakecomputing.com"

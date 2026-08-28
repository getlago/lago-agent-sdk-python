"""Snowflake Cortex usage reader — the I/O half of the connector.

`gateway/adapters/snowflake_cortex.py` stays a pure function with no I/O. This module
is its sibling: it does the reading, and it lives in the SDK rather than in the example
notebook because every rule it encodes is a way to lose money silently. A customer who
hand-rolls this read gets working code on the first try and a wrong invoice on the
hundredth.

The eight, each one measured or documented rather than guessed:

  1. **A failed partition raises.** Partition 0 arrives inline with the statement; the
     rest come from `GET /api/v2/statements/{handle}?partition=N`. An error body carries
     no `data`, so a tolerant `or []` appends zero rows, the loop continues, and the read
     returns a PARTIAL window reporting success. Captured live, the envelope for a
     60,000-row read put 472 rows inline across 8 partitions — a tolerant reader would
     have billed 0.8% of that window and said nothing.
  2. **Assert the assembled row count against `numRows`.** An end-to-end check catches a
     short read no per-request status can reveal: a partition returning HTTP 200 with
     fewer rows than its `partitionInfo` promised, or a manifest/partition disagreement.
  3. **HTTP 202 is not an error.** A statement over ~45s returns a `QueryStatus` envelope
     with no `data`; poll `statementStatusUrl`, bounded by BOTH a deadline and an
     iteration cap so a stuck statement cannot hang a poller forever.
  4. **Whole closed hours only, both bounds floored, ONE literal pair for both views.**
     See `_window_bounds`.
  5. **Name your columns.** `SELECT *` risks the API's inline-response size cap, which
     fails a statement rather than paginating past it, and a column omitted from the
     projection reaches the adapter as ABSENT — where every field degrades to zero rather
     than raising. An under-billed event, silently. `REST_COLUMNS` / `FUNCTIONS_COLUMNS`
     are pinned against what the extractors actually read.
  6. **`+00:00` on every timestamp literal.** A bare literal is parsed in the session's
     timezone and the whole window slides by that offset.
  7. **Every value arrives as a string, and `ARRAY`/`OBJECT` columns arrive as JSON
     text.** `METRICS`, `TOKENS_GRANULAR` and `ROLE_NAMES` all need parsing, and the
     adapter degrades a parse failure to empty rather than raising so one bad row cannot
     take down a window.
  8. **A SQL warehouse is a real cost centre** — measured elsewhere in this tree at
     roughly 1,500x the model-serving usage it was reporting on. Read ONE wide window per
     run; never poll in a tight loop.

Deliberately NOT here: scheduler, cursor store, credential store, retry-forever loop. You
pass a window and get what is there; this does not remember where it got to. That is the
poller, and it stays a separate concern.

TWO VIEWS, TWO TRAFFIC STREAMS — and this is the one thing to get right before running a
backfill at all:

  * `CORTEX_AI_FUNCTIONS_USAGE_HISTORY` reports `AI_COMPLETE` and friends, which run
    inside the warehouse as SQL. There is no client to wrap, so the live `wrap()` path can
    NEVER see this traffic. It exists only in this view, and a backfill is the only way to
    bill it.
  * `CORTEX_REST_API_USAGE_HISTORY` reports the `/api/v2/cortex/v1` calls that the live
    `wrap()` path already bills. Backfilling a window the live path covered re-reports
    every one of those calls.

Which is why `read_usage` reads the functions view by DEFAULT and the REST view only when
asked (`views=("rest",)`). The Databricks reader's rule — "pick one path per traffic
stream" — cannot be followed here, because a customer using both surfaces has to backfill
one view and must not backfill the other. The default is the safe half of that.

The re-report is caught by Lago itself, because both paths now derive ONE key. Measured
rather than assumed, 2026-08-26: one Cortex call whose response header read
`x-snowflake-request-id: 21763baf-8a80-4e42-969e-1383dd5968d6` landed in the REST view
~2 minutes later as exactly that `REQUEST_ID`. (The response BODY is not the way in:
Cortex returns `"id": ""`.) The OpenAI wrapper therefore adopts that header as its
`event_id`, this reader derives the same key from `REQUEST_ID` via `snowflake_event_id`,
and Lago rejects the backfill's copy as a duplicate `transaction_id`. The dedup holds
ONLY while both sides compute the identical string — same `event_id_prefix` ("sfc") and
same resolved subscription; see `backfill_snowflake` for the two ways a caller can break
that. The opt-in default stays worth having anyway: a live-path customer never reads
this view at all, and headerless calls still fall back to a UUID.

Uses `requests`, already a core dependency, so this adds nothing to the install.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..canonical import CanonicalUsage
from .adapters.snowflake_cortex import (
    SNOWFLAKE_EVENT_ID_PREFIX,
    _column,
    extract_snowflake_functions_log,
    extract_snowflake_rest_log,
    resolve_snowflake_subscription,
    snowflake_event_id,
)

logger = logging.getLogger("lago_agent_sdk.gateway.snowflake")

_STATEMENTS_PATH = "/api/v2/statements"

_REST_VIEW = "SNOWFLAKE.ACCOUNT_USAGE.CORTEX_REST_API_USAGE_HISTORY"
_FUNCTIONS_VIEW = "SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AI_FUNCTIONS_USAGE_HISTORY"

# An interval string never reaches SQL — `_window_bounds` resolves it to an instant and
# `_timestamp_sql` renders the literal — but it is still validated strictly rather than
# loosely parsed. Only a bare count plus a unit is accepted: a window quietly read as
# something other than what the caller wrote under-reads, and under-reading is the one
# direction that loses money. Same grammar as the Databricks reader, deliberately.
_INTERVAL_RE = re.compile(r"^\s*(\d{1,5})\s+(second|minute|hour|day|week)s?\s*$", re.I)

# Every `CORTEX_REST_API_USAGE_HISTORY` column `extract_snowflake_rest_log` reads.
#
# Keep in sync with that function's `_EXTRA_COLUMNS` plus `TOKENS_GRANULAR`: a column
# dropped from here reaches the adapter as absent, which it degrades to zero on rather than
# raising — an under-billed event with no error anywhere. `the projection covers every
# column the extraction reads` pins that coupling in both repos.
#
# `SELECT *` is not merely untidy on these views. The SQL API's inline response has a size
# cap that FAILS a statement rather than paginating past it, and Snowflake extends these
# views without notice — this one gained a whole `QUERY_TAG` column between two captures
# eight hours apart — so the width of `*` is not a number anyone controls.
REST_COLUMNS = (
    "REQUEST_ID",  # the per-request idempotency key; see `SnowflakeUsageRow.event_id_for`
    "MODEL_NAME",
    "TOKENS",  # the ADDITIVE total. Never mapped to a token field — see the adapter
    "TOKENS_GRANULAR",  # input / output / cache_read_input / cache_write_input
    "INFERENCE_REGION",  # this view's own grouping key, emitted as a dimension
    "USER_ID",
    "QUERY_TAG",
    "START_TIME",  # the hour bucket, and what every event is stamped with
    "END_TIME",
)

# Every `CORTEX_AI_FUNCTIONS_USAGE_HISTORY` column `extract_snowflake_functions_log`
# reads. Same coupling and the same hazard as `REST_COLUMNS`. `IS_COMPLETED` is projected
# even though the adapter only reports it: this reader ACTS on it, see `read_usage`.
FUNCTIONS_COLUMNS = (
    "QUERY_ID",  # the idempotency key, and NOT unique — see `read_usage`
    "FUNCTION_NAME",  # half of this view's grouping key
    "MODEL_NAME",  # the other half; empty on the task functions, which take no model
    "METRICS",  # an ARRAY of {key: {metric, unit}, value}
    "CREDITS",
    "IS_COMPLETED",  # "was the query completed in THIS aggregation window"
    "QUERY_TAG",  # the only customer-injectable attribution key on either view
    "ROLE_NAMES",
    "USER_ID",
    "WAREHOUSE_ID",
    "START_TIME",
    "END_TIME",
)


def _raise_for_api_error(resp: Any, what: str) -> None:
    """Raise with Snowflake's own error text when a SQL API call is not OK.

    Deliberately NOT `raise_for_status()`: Snowflake puts the useful part in the BODY
    (`{"code": "390318", "message": "Authentication token has expired"}`) and `requests`
    shows only the status line. Code `003001` alone has four distinct causes on one
    account, so the message is the only thing that tells an operator which one they hit.

    Reads `status_code` rather than `.ok` so a duck-typed response — which is what the
    tests and any caller-supplied transport hand over — works the same way. Truncated
    because these bodies can carry a multi-KB `data` array.
    """
    status = getattr(resp, "status_code", 200)
    if 200 <= int(status) < 300:
        return
    try:
        detail = json.dumps(resp.json())
    except ValueError:
        detail = getattr(resp, "text", "") or "<no body>"
    raise RuntimeError(f"Snowflake {what} failed: HTTP {status}: {detail[:500]}")


@dataclass
class SnowflakeUsageRow:
    """One billable row, already shaped for `emit()`.

    No `usd_cost`, and there is no field to add one to. Snowflake meters Cortex in CREDITS
    against a rate card that exists in no view, so there is no per-request dollar figure to
    pass through — every row on this path bills as token counts. `provider = "snowflake"`
    is in `TOKEN_BILLED_PROVIDERS`, so `emit()` routes it to token events even for a
    customer running `pricing_mode="price"` globally, with no price-miss report. That is
    what the set is for; forcing `mode="tokens"` per call would diverge from it.
    """

    usage: CanonicalUsage
    subscription: str | None
    row_id: str
    kind: str
    prefix: str = SNOWFLAKE_EVENT_ID_PREFIX
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        """Idempotency key for billing this row to the subscription its own tags name."""
        return self.event_id_for(self.subscription)

    @property
    def reconcile_dimensions(self) -> dict[str, str]:
        """The Snowflake-side grouping key for this row, to be emitted as dimensions.

        This is what makes the connector checkable: the customer runs a `GROUP BY` on the
        view, groups Lago the same way, and reads the two side by side. Each kind gets the
        key its OWN view aggregates by, and only keys that are true of the whole row:

          * functions — `FUNCTION_NAME` + `MODEL_NAME`. `MODEL_NAME` is empty on
            `AI_SUMMARIZE`/`AI_TRANSLATE`/`AI_SENTIMENT`/`AI_CLASSIFY`, which take no model
            argument, so it is emitted only when populated rather than as an empty string.
          * rest — `INFERENCE_REGION`, the only non-identifying column this view groups by.

        `QUERY_ID` and `REQUEST_ID` are deliberately excluded even though both views carry
        one: a per-request id is not a grouping, it is a list, and one Lago group per
        request is not a comparison anyone can read. They reach `extras` instead.
        """
        dims: dict[str, str] = {}
        if self.kind == "functions":
            fn = _str(self.usage.extras.get("function_name"))
            if fn:
                dims["function_name"] = fn
            # Empty is a fact about the row, not a failure to read it — see the adapter.
            model = _str(self.usage.extras.get("model_name"))
            if model:
                dims["model_name"] = model
            return dims
        region = _str(self.usage.extras.get("inference_region"))
        if region:
            dims["inference_region"] = region
        return dims

    @property
    def occurred_at(self) -> int | None:
        """When this row's usage happened, as unix seconds for `emit()`.

        The whole point of a backfill is that it runs long after the usage it bills, so the
        run's own clock is never the right answer: a window reaching back a week must bill
        into the periods those calls fell in, not into the period the script happens to run
        in.

        `START_TIME` on both views, which is the START of the hour the row is bucketed
        into. Both views are hour-bucketed — twelve `AI_COMPLETE` calls in one query
        produced twelve rows sharing one `START_TIME`/`END_TIME` pair — so the start is the
        only instant certain to sit inside the row's own coverage. The hour's END would push
        a bucket closing exactly on a period boundary into the following period.

        None when the column is absent or unreadable, which leaves `emit()` to stamp `now`.
        Worse than the real time, better than dropping the event.
        """
        value = _column(self.raw, "START_TIME")
        return _rest_epoch(value) if self.kind == "rest" else _functions_epoch(value)

    def event_id_for(self, subscription: str | None) -> str:
        """The same key, scoped to whichever subscription is actually billed.

        Scoping is not cosmetic: Lago's `transaction_id` is unique org-wide, so an id built
        from the source row alone silently blocks that row from ever reaching a second
        subscription. And the subscription billed is not always the one on the row — an
        unattributed row falls back to the caller's default — so the key is built from the
        RESOLVED value, not from `self.subscription`.
        """
        return snowflake_event_id(self.prefix, self.kind, subscription, self.row_id)


def _as_utc(moment: datetime) -> datetime:
    """A naive datetime is taken as UTC; an aware one is CONVERTED, never reformatted.

    Snowflake stores `START_TIME` as an absolute instant, so formatting an aware datetime
    as-is would emit local wall time and a Europe/Paris caller would read a window two
    hours in the future, bill nothing, and report success. Same rule as `_snowflake_epoch`
    below and as `sdk.py`'s event timestamps.
    """
    return (
        moment.astimezone(timezone.utc) if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)
    )


def _floor_hour(moment: datetime) -> datetime:
    """The start of the hour containing `moment`. Moves an instant, and it decides what
    gets read at all."""
    return moment.replace(minute=0, second=0, microsecond=0)


def _timestamp_sql(moment: datetime) -> str:
    """Render an instant as a zone-explicit SQL timestamp literal.

    The `+00:00` is not decoration. A bare `'2026-08-26 13:00:00'` is parsed in the
    session's `TIMEZONE` parameter, so on an account set to anything but UTC the identical
    literal names a different instant and the whole window slides by that offset. Snowflake
    accounts default to `America/Los_Angeles`, not UTC, so this is the common case rather
    than the exotic one.

    Rendered as a bare string literal for `TO_TIMESTAMP_TZ`, which parses the offset. The
    comparison against a `timestamp_ltz` column then happens on the instant, not the wall
    clock.
    """
    return f"'{moment.strftime('%Y-%m-%d %H:%M:%S')}+00:00'"


def _window_bounds(since: str | datetime, *, now: datetime | None = None) -> tuple[datetime, datetime]:
    """Resolve the read window to ONE pair of instants, both floored to the hour.

    Rejects anything unrecognized. Four money bugs live in leaving any part of this to SQL:

      * **Two statements, two windows.** `DATEADD(hour, -N, CURRENT_TIMESTAMP())` is a SQL
        *string*, so it is re-evaluated per statement — measured at 5.1s of drift between
        two reads on the equivalent Databricks tables. A row landing in the gap is read by
        neither statement and is simply lost. Resolving the bounds HERE and rendering one
        literal pair for both views is what makes them the same window.
      * **The boundary hour.** Both views are hour-bucketed and `START_TIME` is always the
        hour START. Compared against a mid-hour bound, the hour CONTAINING that bound fails
        the predicate while its usage really happened — so a `since` of 13:30 drops every
        row of the 13:00 hour, including calls made at 13:45.
      * **The open hour.** A bucket cannot be complete before its hour closes, and the
        functions view lands a row ~141s after its query ends (measured). Billing the open
        hour early burns that row's `REQUEST_ID`/`QUERY_ID`-derived `transaction_id`, and
        the corrected re-run is then rejected by Lago as a duplicate — so the remainder
        never bills at all. Hence the upper bound, and hence both views get it.
      * **View latency.** These views lag ~3-5 minutes (measured, and far less than
        `ACCOUNT_USAGE` is usually assumed to). The newest closed hour is therefore
        readable almost immediately, but a window ending exactly now still catches the hour
        that has not finished being written.

    Flooring the lower bound can read rows slightly older than the caller asked for. That is
    deliberate: every `transaction_id` derives from the source row, so a row already billed
    is rejected as a duplicate and one not billed yet SHOULD be. Over-read deliberately;
    under-reading is the only direction that loses money.
    """
    moment = _as_utc(now) if now is not None else datetime.now(timezone.utc)
    if isinstance(since, datetime):
        lower = _as_utc(since)
    else:
        m = _INTERVAL_RE.match(str(since))
        if not m:
            raise ValueError(
                f"since={since!r} not understood — pass a datetime, or a string like "
                "'7 days' / '24 hours' / '30 minutes'"
            )
        lower = moment - timedelta(**{f"{m.group(2).lower()}s": int(m.group(1))})
    return _floor_hour(lower), _floor_hour(moment)


class SnowflakeSource:
    """Reads Snowflake Cortex usage over the SQL Statement Execution API.

    Needs a PAT plus a running warehouse — the live `wrap()` path needs neither. Four
    non-obvious things gate the setup, all four hit on a real account: model access moved
    to RBAC (`GRANT APPLICATION ROLE SNOWFLAKE."CORTEX-MODEL-ROLE-ALL"`, without which the
    role can call zero models); a PAT's `ROLE_RESTRICTION` is a case-sensitive string
    literal; a warehouse shipped with `AUTO_RESUME = FALSE` fails every statement with
    "warehouse is suspended", which reads like a privilege error; and a PAT cannot
    authenticate at all without an active network policy.

    A SQL warehouse is a real cost centre: measured on the equivalent Databricks setup,
    warehouse queries cost roughly 1,500x the model-serving usage they were reporting on.
    Read one wide window per run; never poll in a tight loop.
    """

    def __init__(
        self,
        account: str,
        token: str,
        *,
        timeout: float = 180.0,
        statement_timeout: int = 120,
        warehouse: str | None = None,
        role: str | None = None,
        database: str | None = None,
        schema: str | None = None,
        poll_interval: float = 2.0,
        max_polls: int = 90,
    ) -> None:
        # Accepts either the account identifier (`MYORG-ACCOUNT123`) or a full host. The
        # account form is what Snowflake's own docs and error messages use, and getting
        # `.snowflakecomputing.com` appended twice is a 404 that reads like a bad account.
        bare = re.sub(r"^https?://", "", account).rstrip("/")
        self.host = bare if "." in bare else f"{bare}.snowflakecomputing.com"
        self.token = token
        self.timeout = timeout
        self.statement_timeout = statement_timeout
        self.warehouse = warehouse
        self.role = role
        self.database = database
        self.schema = schema
        self.poll_interval = poll_interval
        # Bounded independently of `timeout` per rule 3: a deadline alone leaves a statement
        # that answers instantly but never leaves RUNNING spinning at the poll interval for
        # the whole timeout, and a cap alone lets a slow poll outlive it.
        self.max_polls = max_polls
        # Rows the most recent `read_usage` refused to bill, in the shape its warning names
        # them. A log line is not something a caller can act on: `backfill_snowflake` turns
        # this into part of its `skipped` count and an `on_error` report, and a caller
        # reading the window itself can re-run exactly these queries once the ambiguity is
        # settled. See `read_usage` for why they go unbilled.
        self.deferred_rows: list[dict[str, Any]] = []

    @classmethod
    def from_env(cls, **kwargs: Any) -> SnowflakeSource:
        """Build from `SNOWFLAKE_ACCOUNT` (or `SNOWFLAKE_HOST`) and `SNOWFLAKE_PAT`."""
        import os

        account = os.environ.get("SNOWFLAKE_ACCOUNT") or os.environ.get("SNOWFLAKE_HOST")
        token = os.environ.get("SNOWFLAKE_PAT")
        missing = []
        if not account:
            missing.append("SNOWFLAKE_ACCOUNT (or SNOWFLAKE_HOST)")
        if not token:
            missing.append("SNOWFLAKE_PAT")
        if missing:
            raise ValueError(f"missing environment variable(s): {', '.join(missing)}")
        kwargs.setdefault("warehouse", os.environ.get("SNOWFLAKE_WAREHOUSE"))
        kwargs.setdefault("role", os.environ.get("SNOWFLAKE_ROLE"))
        return cls(str(account), str(token), **kwargs)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            # Without this a PAT is read as an OAuth token and rejected as invalid — the
            # error says nothing about the header, so it reads like a bad secret.
            "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # SQL
    # ------------------------------------------------------------------
    def query(self, sql: str) -> list[dict[str, Any]]:
        """Run one statement and return every row as a dict keyed by column name.

        Handles the three things a naive reader gets wrong, all three of them rules above:
        the response is POSITIONAL (`data` is a list of lists, zipped against
        `resultSetMetaData.rowType`), only partition 0 arrives inline, and a statement that
        outruns its `timeout` answers 202 rather than failing.
        """
        import requests

        body: dict[str, Any] = {"statement": sql, "timeout": self.statement_timeout}
        # Context values are CASE-SENSITIVE here, unlike an unquoted identifier in SQL, so
        # they are passed exactly as the caller wrote them rather than upper-cased.
        if self.warehouse:
            body["warehouse"] = self.warehouse
        if self.role:
            body["role"] = self.role
        if self.database:
            body["database"] = self.database
        if self.schema:
            body["schema"] = self.schema

        resp = requests.post(
            f"https://{self.host}{_STATEMENTS_PATH}",
            headers=self._headers,
            json=body,
            timeout=self.timeout,
        )
        _raise_for_api_error(resp, "statement submission")
        # Rule 3: a 202 carries a handle and no rows. It is the EXPECTED answer for the one
        # wide window per run this class tells operators to read.
        envelope = self._await_statement(resp.status_code, resp.json())

        meta = envelope.get("resultSetMetaData") or {}
        columns = [str(c.get("name") or "") for c in (meta.get("rowType") or [])]
        rows: list[list[Any]] = list(envelope.get("data") or [])

        handle = envelope.get("statementHandle")
        # `partitionInfo` INCLUDES partition 0, which arrived inline — verified against a
        # captured 60,000-row envelope: 8 entries whose rowCounts sum to numRows, the first
        # being the 472 rows already in `data`. Starting at 1 is what makes that true.
        partitions = meta.get("partitionInfo") or []
        for index in range(1, len(partitions)):
            part_resp = requests.get(
                f"https://{self.host}{_STATEMENTS_PATH}/{handle}",
                headers=self._headers,
                params={"partition": index},
                timeout=self.timeout,
            )
            # THE check this whole method exists for — rule 1. A failed partition fetch
            # returns a JSON error body with no `data`, so `or []` would append zero rows,
            # the loop would continue, and `query()` would return a PARTIAL window
            # reporting success. On the captured envelope that is 472 of 60,000 rows billed
            # with no error at all, which is the single worst outcome this reader can
            # produce.
            _raise_for_api_error(part_resp, f"result partition {index} of {len(partitions)}")
            rows.extend(part_resp.json().get("data") or [])

        # Rule 2: end-to-end, independent of cause. Catches a partition that answers HTTP
        # 200 with fewer rows than promised, and a manifest/partition disagreement — neither
        # of which any per-request status check can see. `numRows` arrives as a string like
        # every other value, hence the int().
        promised = meta.get("numRows")
        if promised is not None and int(promised) != len(rows):
            raise RuntimeError(
                f"Snowflake returned {len(rows)} row(s) but the statement promised "
                f"{int(promised)} across {len(partitions)} partition(s) — refusing to bill "
                f"a partial window (statementHandle={handle})"
            )
        # Rows with no column names decode to `{}` each, which every layer downstream
        # degrades cleanly and WRONGLY on: all-zero usage, and a confident `{"tokens": 0}`
        # for a window that had real traffic. Not observed — every SELECT returns a full
        # `rowType` — so this guards the decode, not a known bug. Keeps `strict=False` on
        # the zip: a length mismatch is an API-contract violation the row-count check above
        # already catches, and strict=True would newly reject reads that work today.
        if rows and not columns:
            raise RuntimeError(
                "Snowflake returned rows with no `resultSetMetaData.rowType` — cannot "
                f"decode {len(rows)} row(s) (statementHandle={handle})"
            )

        return [dict(zip(columns, row, strict=False)) for row in rows]

    def _await_statement(self, status: int, envelope: dict[str, Any]) -> dict[str, Any]:
        """Poll a 202'd statement until it carries results.

        Rule 3. A statement still executing when its `timeout` elapses returns **HTTP 202**
        with a `statementHandle` and `statementStatusUrl` and no `data` — not an error.
        Treating that as fatal breaks exactly the case this class tells operators to use.

        Bounded by BOTH a deadline and an iteration cap, because either alone leaves a hole:
        a stuck statement that answers every poll instantly would spin
        `timeout / poll_interval` times on the cap alone, and a slow poll would outlive the
        deadline without the cap.
        """
        import requests

        if status != 202:
            return envelope
        handle = envelope.get("statementHandle")
        if not handle:
            raise RuntimeError(
                "Snowflake answered 202 with no statementHandle, so the statement cannot "
                f"be polled: {json.dumps(envelope)[:500]}"
            )
        deadline = time.monotonic() + self.timeout
        # `statementStatusUrl` is a path on the same host, and it carries the query params
        # Snowflake wants echoed back. Falling back to the handle keeps this working if a
        # response ever omits it.
        status_path = _str(envelope.get("statementStatusUrl")) or f"{_STATEMENTS_PATH}/{handle}"
        for _ in range(self.max_polls):
            if time.monotonic() >= deadline:
                break
            # A plain sleep, so nothing is left armed once this returns — a backfill script
            # returns and the process exits.
            time.sleep(self.poll_interval)
            resp = requests.get(
                f"https://{self.host}{status_path}",
                headers=self._headers,
                timeout=self.timeout,
            )
            # A non-OK poll already failed loudly below (no `data`, no `rowType`); checking
            # here only changes WHAT the operator reads — "Authentication token has
            # expired" instead of a decode error three frames away.
            _raise_for_api_error(resp, "statement poll")
            body = resp.json()
            # Still 202 means still running. Anything else that came back OK carries the
            # result set, whose presence is the terminal signal — `resultSetMetaData` is
            # absent on the QueryStatus envelope and present on the result envelope.
            if resp.status_code != 202 and body.get("resultSetMetaData"):
                return dict(body)
        raise RuntimeError(
            f"Snowflake statement {handle} was still running after {self.timeout}s / "
            f"{self.max_polls} poll(s) — raise `timeout` or narrow the window"
        )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def read_usage(
        self,
        since: str | datetime = "1 day",
        *,
        event_id_prefix: str = SNOWFLAKE_EVENT_ID_PREFIX,
        views: Sequence[str] = ("functions",),
        subscription_order: Sequence[str] | None = None,
    ) -> Iterator[SnowflakeUsageRow]:
        """Yield every billable row in the window, shaped for `emit()`.

        Reads the FUNCTIONS view by default. The REST view reports the calls the live
        `wrap()` path already bills, so reading both and emitting both double-bills every
        REST call — see the module docs. Pass `views=("rest",)` only for an account with no
        live path, or `("functions", "rest")` knowing exactly why.

        Reads **whole closed hours only**, one literal pair across both views;
        `_window_bounds` documents why each of those properties is load-bearing. The
        practical consequence is that the newest hour of traffic arrives on the NEXT run, so
        pass a window comfortably wider than your run interval — this reader keeps no
        cursor.

        Rows whose usage extracts to all-zero are skipped, so nothing emits an empty event.
        A failed Cortex call produces NO ROW on either view (measured with a same-batch
        control: a 403 and a 400 driven alongside a success, and only the success ever
        appeared), so that path guards a shape nobody has seen rather than the common case.

        THE FUNCTIONS VIEW'S `QUERY_ID` IS NOT UNIQUE, and this is where that is handled.
        `IS_COMPLETED` means "was the query completed in THIS aggregation window", and the
        view is hour-bucketed: Snowflake documents a query running 5:30->8:30 writing FOUR
        rows, one per bucket, all sharing one `QUERY_ID`. Two things follow, and only the
        first is a nuisance:

          * The idempotency key `{prefix}_{kind}_{sub}_{QUERY_ID}` COLLIDES across those
            rows. Lago accepts the first and rejects rows 2..N as duplicate
            `transaction_id`s, which takes the whole batch down the all-or-nothing split
            path.
          * Whether each window's `METRICS` is INCREMENTAL or CUMULATIVE is unmeasured —
            the probe was built and not run. The two answers disagree by a factor of the
            query's hour count: on a 3-hour query reporting 3800 input tokens in total,
            summing four incremental rows bills 3800 and summing four cumulative ones bills
            9500, while billing only the final row bills 3800 if cumulative and 900 if
            incremental.

        So a `QUERY_ID` this window yields more than one row for is DEFERRED, not guessed
        at, and so is any row explicitly flagged `IS_COMPLETED = false`. Both are counted in
        `deferred_rows`, reported by `backfill_snowflake` through `on_error`, and billable
        once the shape is measured. Guessing wrong over-bills by 2.5x or under-bills by
        76%, and neither is recoverable once invoiced; the deferral costs a rare long
        query's tokens until then. Every row ever captured on this account is a
        single-bucket, sub-second query — 48 of 48 — so this fires on a shape that has never
        been observed.

        The residual gap, stated rather than hidden: a window containing ONLY the trailing
        `IS_COMPLETED = true` row of a multi-bucket query is indistinguishable from a normal
        single-bucket row, and bills as one. Pass windows wider than your longest query,
        which rule 4 already asks for.
        """
        # Rewritten per read rather than appended to, so a later read of a healthy window
        # cannot leave an earlier read's gap standing as if it were current. Cleared ahead
        # of the empty-window return below, so that path clears it too. Complete once the
        # caller has drained the iterator.
        self.deferred_rows = []

        lower, upper = _window_bounds(since)
        if lower >= upper:
            # Not an error, but it must not read as success either: the caller asked for a
            # window lying entirely inside the hour this reader excludes, so zero rows here
            # says nothing about whether there was traffic.
            logger.warning(
                "lago: since=%r resolves to [%s, %s), which is empty — the window falls "
                "inside the current, still-aggregating hour that this reader excludes. "
                "Nothing was read; widen the window past the hour boundary.",
                since,
                lower.isoformat(),
                upper.isoformat(),
            )
            return
        # One literal pair, both views — see `_window_bounds`. Resolving the bounds here
        # rather than in SQL is what makes the two reads the same window.
        window, ceiling = _timestamp_sql(lower), _timestamp_sql(upper)

        if "functions" in views:
            yield from self._read_functions(window, ceiling, event_id_prefix, subscription_order)
        if "rest" in views:
            yield from self._read_rest(window, ceiling, event_id_prefix, subscription_order)

    def _read_functions(
        self,
        window: str,
        ceiling: str,
        prefix: str,
        subscription_order: Sequence[str] | None,
    ) -> Iterator[SnowflakeUsageRow]:
        # No ORDER BY: nothing downstream reads row order — every event's `transaction_id`
        # derives from the row's own id and the deferred report is sorted — so it would only
        # buy the warehouse a sort over the widest read this module makes.
        rows = self.query(f"""
            SELECT {", ".join(FUNCTIONS_COLUMNS)}
            FROM {_FUNCTIONS_VIEW}
            WHERE START_TIME >= TO_TIMESTAMP_TZ({window}) AND START_TIME < TO_TIMESTAMP_TZ({ceiling})
        """)

        # Group by the raw QUERY_ID to find the collisions described in `read_usage`. Rows
        # carrying no QUERY_ID are deliberately NOT grouped: they would all key under "" and
        # read as one enormous collision, when in fact they are unrelated rows that each get
        # a content hash for an id and cannot be detected as continuations either way.
        by_query_id: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            query_id = _str(_column(row, "QUERY_ID"))
            if query_id:
                by_query_id.setdefault(query_id, []).append(row)

        for row in rows:
            query_id = _str(_column(row, "QUERY_ID"))
            group = by_query_id.get(query_id, [row]) if query_id else [row]
            # More than one row under one QUERY_ID is the collision itself, whatever the
            # buckets say — the key cannot distinguish them, so none of them is billable.
            multi_bucket = len(group) > 1
            # Only an EXPLICIT false defers. Absent or unparseable must not, or a view that
            # stops populating the column would silently defer an entire window.
            incomplete = _is_explicitly_false(_column(row, "IS_COMPLETED"))
            if multi_bucket or incomplete:
                self.deferred_rows.append(
                    {
                        "query_id": query_id or "<no QUERY_ID>",
                        "buckets": sorted({_str(_column(r, "START_TIME")) for r in group}),
                        "reason": "multi_bucket" if multi_bucket else "incomplete",
                    }
                )
                continue

            usage = extract_snowflake_functions_log(row)
            # A row that bills nothing emits nothing — `nonzero_numeric()` would be empty
            # and `emit()` would produce no events. Skipping here keeps it out of the row
            # stream so a caller counting rows counts billable ones. The adapter has already
            # marked the case where zero is suspicious (`metrics_unmapped`), so nothing is
            # lost silently.
            if not usage.nonzero_numeric():
                continue
            yield SnowflakeUsageRow(
                usage=usage,
                subscription=(
                    resolve_snowflake_subscription(row, tuple(subscription_order))
                    if subscription_order is not None
                    else resolve_snowflake_subscription(row)
                ),
                # One row per query, so QUERY_ID is the natural key — see `event_id_for` for
                # why it is still subscription-scoped, and `read_usage` for why a colliding
                # one never reaches this line.
                row_id=_row_id(row, "QUERY_ID"),
                kind="functions",
                prefix=prefix,
                raw=row,
            )

        if self.deferred_rows:
            first = self.deferred_rows[0]
            logger.warning(
                "lago: %d Snowflake functions row(s) in this window were NOT billed "
                "(e.g. QUERY_ID=%s, reason=%s, buckets=%s). An hour-bucketed query writes "
                "one row per bucket under one QUERY_ID, so the idempotency key collides and "
                "whether each row's METRICS is incremental or cumulative is unmeasured — "
                "billing either way would over- or under-charge by the query's hour count. "
                "Re-run once that is settled.",
                len(self.deferred_rows),
                first["query_id"],
                first["reason"],
                ",".join(first["buckets"]),
            )

    def _read_rest(
        self,
        window: str,
        ceiling: str,
        prefix: str,
        subscription_order: Sequence[str] | None,
    ) -> Iterator[SnowflakeUsageRow]:
        rows = self.query(f"""
            SELECT {", ".join(REST_COLUMNS)}
            FROM {_REST_VIEW}
            WHERE START_TIME >= TO_TIMESTAMP_TZ({window}) AND START_TIME < TO_TIMESTAMP_TZ({ceiling})
        """)

        # Said on every REST read, not just in the docs: the caller who reaches this line
        # has already opted in, and the one who did it by accident is the one who needs
        # telling.
        logger.warning(
            "lago: reading %d row(s) from %s. If these calls were made through a wrapped "
            "client, the live path ALREADY billed them and this read re-reports them. "
            "Lago rejects the copies as duplicate transaction_ids ONLY when this backfill "
            "uses the default event_id_prefix and resolves the same subscription the live "
            "path billed. Only backfill this view for traffic wrap() never saw.",
            len(rows),
            _REST_VIEW,
        )

        for row in rows:
            usage = extract_snowflake_rest_log(row)
            if not usage.nonzero_numeric():
                continue
            yield SnowflakeUsageRow(
                usage=usage,
                subscription=(
                    resolve_snowflake_subscription(row, tuple(subscription_order))
                    if subscription_order is not None
                    else resolve_snowflake_subscription(row)
                ),
                # One row per request, so REQUEST_ID is the natural key — and unlike the
                # functions view's QUERY_ID it really is unique: this view does not window.
                row_id=_row_id(row, "REQUEST_ID"),
                kind="rest",
                prefix=prefix,
                raw=row,
            )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _str(value: Any) -> str:
    """A string value, or "" for anything else. Never coerces a number."""
    return value if isinstance(value, str) else ""


def _rest_epoch(value: Any) -> int | None:
    """A `timestamp_tz` column as unix seconds — the REST view's `START_TIME`/`END_TIME`.

    The SQL API renders `timestamp_tz` as `"<epoch seconds>.<nanos> <offset minutes+1440>"`,
    measured: `"1787162400.000000000 1440"`. The epoch part is the true instant and the
    trailing field is display offset only (1440 == UTC), so the offset is deliberately
    ignored rather than added — adding it would slide every REST event by the account's
    timezone.

    Separate from `_functions_epoch` even though the arithmetic currently coincides: these
    are two different Snowflake types on two different views, and a later "fix" to one must
    not quietly redefine the other. Each has its own test with its own captured wire form.
    """
    return _snowflake_epoch(value)


def _functions_epoch(value: Any) -> int | None:
    """A `timestamp_ltz` column as unix seconds — the functions view's `START_TIME`.

    Renders as a BARE epoch, measured: `"1787162400"`. No offset field, and none is wanted:
    the value is already absolute.
    """
    return _snowflake_epoch(value)


def _snowflake_epoch(value: Any) -> int | None:
    """Both wire forms, plus the shapes a typed connector yields, as unix seconds.

    Unreadable returns None rather than raising: a bad timestamp column must not cost the
    caller the whole row — `emit()` falls back to stamping `now`.

    `int()` truncates toward zero, which is what the JS port's `Math.trunc` does, so a
    pre-epoch stamp rounds the same way in both repos.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(_as_utc(value).timestamp())
    if isinstance(value, bool):
        # `bool` is an `int` subclass, and a timestamp column that arrives as True must not
        # silently become the epoch second 1.
        return None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (ValueError, OverflowError):
            return None
    text = str(value).strip()
    if not text:
        return None
    # The epoch is everything before the offset field. Splitting first means a malformed
    # offset cannot leak into the number.
    try:
        return int(float(text.split()[0]))
    except (ValueError, OverflowError, IndexError):
        return None


def _is_explicitly_false(value: Any) -> bool:
    """True only for an explicit boolean False or the string "false".

    Deliberately NOT falsy-testing. `IS_COMPLETED` arrives as the string `"false"` over the
    SQL API and as a real `False` from a typed connector, and both must defer — but NULL, an
    absent column and an unrecognized value must NOT, or a view that stops populating the
    column would silently defer every row in every window.
    """
    if value is False:
        return True
    return isinstance(value, str) and value.strip().lower() == "false"


def _row_id(row: dict[str, Any], *columns: str) -> str:
    """First usable id among `columns`, falling back to a hash of the whole row.

    Two ways the obvious `_str(a or b)` goes wrong, both silent and both losing money. A row
    with NULL ids yields ""; so does a row whose id a driver hands back as a non-string,
    because `or` selects it and `_str` rejects the type without trying the next column.
    Either way `event_id_for` still produces a well-formed key (`sfc_functions_sub_x_`), so
    EVERY such row in the window shares one `transaction_id` — Lago accepts the first and
    rejects the rest as duplicates, and those calls never bill at all.

    The content hash keeps the key deterministic, so re-running the same window stays
    idempotent, which a random UUID would break.
    """
    for name in columns:
        value = _column(row, name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    canonical = json.dumps(row, sort_keys=True, default=str)
    return f"sha{hashlib.sha256(canonical.encode()).hexdigest()[:32]}"

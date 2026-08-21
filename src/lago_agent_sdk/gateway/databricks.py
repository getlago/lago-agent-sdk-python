"""Databricks AI Gateway usage reader — the I/O half of the connector.

`gateway/adapters/databricks_gateway.py` stays a pure function with no I/O. This
module is its sibling: it does the reading, and it exists because reading usage
out of Databricks is genuinely hard in a way Cloudflare's is not.

Cloudflare is one paginated GET, about twelve lines. Databricks needs a SQL
warehouse, the Statement Execution API, columnar-to-dict zipping, chunked result
fetching, and TWO different tables whose rows must not be billed twice. Hand-rolled
that is ~100 lines in which several money-losing mistakes are easy:

  * **Silent truncation.** The Statement Execution API returns only chunk 0 inline;
    `manifest.total_chunk_count` can be higher and the rest need separate fetches.
    A naive reader works on a small window and quietly bills a fraction of a large
    one, with no error.
  * **Double billing.** A BYOK call appears in BOTH `ai_gateway.usage` (tokens) and
    `ai_gateway.external_model_spend` (USD). Bill both and you charge twice.
  * **Unscoped idempotency keys.** `transaction_id` is unique account-wide, so an
    unscoped row id silently blocks that row from ever reaching a second
    subscription.

Deliberately NOT here: scheduler, cursor store, credential store. You pass an
explicit window and this returns what it finds; it does not remember where it got
to. That is the poller, and it stays a separate concern — as the Cloudflare
connector's changelog already states.

Uses `requests`, already a core dependency, so this adds nothing to the install.
`databricks-sql-connector` would also work and is the better choice for
interactive analysis, but it is a heavy extra to require for a batch read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..canonical import CanonicalUsage
from .adapters.databricks_gateway import (
    _safe_str,
    extract_databricks_log,
    resolve_databricks_subscription,
)

logger = logging.getLogger("lago_agent_sdk.gateway.databricks")

_STATEMENTS_PATH = "/api/2.0/sql/statements"

# An interval string no longer reaches SQL at all — `_window_bounds` resolves it to an
# instant and `_timestamp_sql` renders the literal — but it stays validated strictly
# rather than loosely parsed. Only a bare count plus a unit is accepted: a window quietly
# read as something other than what the caller wrote under-reads, and under-reading is
# the one direction that loses money.
_INTERVAL_RE = re.compile(r"^\s*(\d{1,5})\s+(second|minute|hour|day|week)s?\s*$", re.I)


def _raise_for_api_error(resp: Any, what: str) -> None:
    """Raise with the API's own error text when a Statement Execution call is not OK.

    Deliberately NOT `raise_for_status()`: Databricks puts the useful part in the BODY
    (`{"error_code": "PERMISSION_DENIED", "message": "... does not have required scopes:
    sql"}`) and `requests` shows only the status line. The `403 does not have required
    scopes: sql` that this class's docstring warns operators about is the error most
    likely to hit a first-time setup, so it is the one that must read clearly.

    Truncated because these bodies can carry a multi-KB `details` array.
    """
    status = getattr(resp, "status_code", 200)
    if 200 <= int(status) < 300:
        return
    try:
        detail = json.dumps(resp.json())
    except ValueError:
        detail = getattr(resp, "text", "") or "<no body>"
    raise RuntimeError(f"Databricks {what} failed: HTTP {status}: {detail[:500]}")


@dataclass
class DatabricksUsageRow:
    """One billable row, already shaped for `emit()`.

    `usd_cost` is set only for BYOK rows, where Databricks meters the provider cost
    itself in `external_model_spend`. Hosted rows leave it None: Databricks bills
    those in DBUs against a rate card that exists in no system table, so there is no
    per-request dollar figure to pass through and they bill as token counts.
    """

    usage: CanonicalUsage
    subscription: str | None
    row_id: str
    kind: str
    usd_cost: float | None = None
    prefix: str = "dbx"
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_byok(self) -> bool:
        return self.usd_cost is not None

    @property
    def event_id(self) -> str:
        """Idempotency key for billing this row to the subscription its tags name."""
        return self.event_id_for(self.subscription)

    @property
    def reconcile_dimensions(self) -> dict[str, str]:
        """The Databricks-side grouping key for this row, to be emitted as dimensions.

        This is what makes the connector checkable: the customer opens the Databricks
        page, groups Lago the same way, and reads the two side by side. Without it the
        comparison fails on naming alone — our `model` is normalized
        (`qwen35-122b-a10b`) where the gateway page shows `system.ai.qwen35-122b-a10b`
        or even a display label (`GPT OSS 20B`).

        Each kind gets the key that its OWN Databricks surface aggregates by, and only
        keys that are true of the whole row:

          * hosted — `endpoint_name`, how the AI Gateway usage page groups.
          * BYOK   — `bucket`, the hour, which is `external_model_spend`'s own
            aggregation key. Deliberately NOT `endpoint_name` here: a spend row covers
            an hour of requests, so any per-request field would be one sampled value
            dressed up as a property of the bucket.

        `invocation_id` / `request_id` / `status_code` are excluded for the same reason
        plus cardinality — one Lago group per request is not a comparison, it's a list.
        """
        if self.kind == "spend":
            bucket = _stamp(self.raw.get("bucket"))
            return {"bucket": bucket} if bucket else {}
        endpoint = _safe_str(self.usage.extras.get("endpoint_name"))
        return {"endpoint_name": endpoint} if endpoint else {}

    @property
    def occurred_at(self) -> int | None:
        """When this row's usage actually happened, as unix seconds for `emit()`.

        The whole point of a backfill is that it runs long after the usage it bills,
        so the run's own clock is never the right answer: a window reaching back a
        week must bill into the periods those calls fell in, not into the period the
        script happens to run in.

        Each kind reports the time its OWN surface is keyed by:

          * usage — `event_time`, the request's own instant.
          * spend — `bucket`, the START of the hour it aggregates. An hourly total
            covers [bucket, bucket + 1h), so the start is the only instant certain to
            sit inside the row's own coverage; the hour's end would push a bucket
            closing exactly on a period boundary into the following period.

        None when the column is absent or unreadable, which leaves `emit()` to stamp
        `now` — the pre-existing behaviour, and better than dropping the event.
        """
        return _epoch(self.raw.get("bucket") if self.kind == "spend" else self.raw.get("event_time"))

    def event_id_for(self, subscription: str | None) -> str:
        """The same key, scoped to whichever subscription is actually billed.

        Scoping is not cosmetic: Lago's `transaction_id` is unique account-wide, so an
        id built from the source row alone silently blocks that row from ever reaching
        a second subscription. And the subscription billed is not always the one on the
        row — an untagged row falls back to the caller's default — so the key has to be
        built from the resolved value, not from `self.subscription`.
        """
        return f"{self.prefix}_{self.kind}_{subscription or 'none'}_{self.row_id}"


def _as_utc(moment: datetime) -> datetime:
    """A naive datetime is taken as UTC; an aware one is CONVERTED, never reformatted.

    Databricks stores `event_time`/`usage_start_time` in UTC, so formatting an aware
    datetime as-is would emit local wall time and a Europe/Paris caller would read a
    window two hours in the future, bill nothing, and report success. Same rule as
    `_epoch` below and as `sdk.py`'s event timestamps, and the same rule the JS port's
    `Date` arithmetic follows.
    """
    return (
        moment.astimezone(timezone.utc) if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)
    )


def _floor_hour(moment: datetime) -> datetime:
    """The start of the hour containing `moment`.

    Distinct from `_truncate_hour`, which trims a timestamp STRING to build a join key.
    This one moves an instant, and it decides what gets read at all.
    """
    return moment.replace(minute=0, second=0, microsecond=0)


def _timestamp_sql(moment: datetime) -> str:
    """Render an instant as a zone-explicit SQL TIMESTAMP literal.

    The `+00:00` is not decoration. A bare `TIMESTAMP '2026-08-07 13:00:00'` is parsed
    in the warehouse's own `spark.sql.session.timeZone`, so on a workspace set to
    anything but UTC the identical literal names a different instant and the whole
    window slides by that offset. Verified live: the suffixed form is accepted and
    resolves to the same epoch as the bare form under this warehouse's `Etc/UTC`.
    """
    return f"TIMESTAMP '{moment.strftime('%Y-%m-%d %H:%M:%S')}+00:00'"


def _window_bounds(since: str | datetime, *, now: datetime | None = None) -> tuple[datetime, datetime]:
    """Resolve the read window to ONE pair of instants, both floored to the hour.

    Rejects anything unrecognized. Three money bugs live in leaving any part of this
    to SQL, all three measured on real gateway tables:

      * **Two statements, two windows.** `current_timestamp() - INTERVAL 1 DAY` is a
        SQL *string*, so it is re-evaluated per statement — 5.1s of drift measured
        between the spend read and the usage read. Spend runs first, so the usage
        window is the narrower one, and a hosted row in the gap is read by neither
        statement. Hosted is billed from `usage` alone, so that row is simply lost.
      * **The boundary hour.** `external_model_spend` is an hourly aggregate whose
        `usage_start_time` is always the hour START (65 of 65 rows). Compared against
        a mid-hour bound, the hour CONTAINING that bound fails the predicate while its
        usage rows pass: live, a `since` of 13:30 read 11 of 65 spend rows and dropped
        $0.1256 of $0.1723, while still reading 35 BYOK usage rows from inside the
        hour it dropped. Flooring is what makes the two tables agree on one window.
      * **The open hour.** A spend row cannot be complete before its hour closes (the
        08:00–09:00 row appeared ~7 min AFTER 09:00). Billing it early bills a
        fraction of the hour under that hour's `record_id`, and the corrected re-run
        is then rejected by Lago as a duplicate `transaction_id` — so the remainder is
        never billed at all. Hence the upper bound, and hence both tables get it: a
        window whose halves cover different hours is the first bug again.

    Flooring the lower bound can read rows slightly older than the caller asked for.
    That is deliberate: every `transaction_id` is derived from the source row, so a row
    already billed is rejected as a duplicate and one not billed yet SHOULD be.
    Under-reading is the only direction that loses money.
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


class DatabricksSource:
    """Reads Databricks AI Gateway usage over the SQL Statement Execution API.

    Needs a PAT carrying the **`sql`** scope plus a SQL warehouse — the live
    `wrap()` path needs neither. Without them every warehouse route returns
    `403 "does not have required scopes: sql"`.

    A SQL warehouse is a real cost centre: measured on a test workspace, warehouse
    queries cost roughly 1,500x the model-serving usage they were reporting on. Read
    one wide window per run; never poll in a tight loop.
    """

    def __init__(
        self,
        host: str,
        token: str,
        warehouse_id: str,
        *,
        timeout: float = 180.0,
        wait_timeout: str = "50s",
    ) -> None:
        self.host = host.rstrip("/")
        self.token = token
        self.warehouse_id = warehouse_id
        self.timeout = timeout
        # Databricks rejects anything outside 0s or 5-50s.
        self.wait_timeout = wait_timeout
        # Buckets the most recent `read_usage` could not bill, in the shape its
        # warning names them. A log line is not something a caller can act on:
        # `backfill_databricks` turns this into a count in its return value and an
        # `on_error` report, and a caller reading the window itself can re-run
        # exactly these hours. See `read_usage` for why they go unbilled.
        self.deferred_buckets: list[dict[str, str]] = []

    @classmethod
    def from_env(cls, **kwargs: Any) -> DatabricksSource:
        """Build from `DATABRICKS_HOST` / `DATABRICKS_TOKEN` / `DATABRICKS_WAREHOUSE_ID`."""
        import os

        missing = [
            k
            for k in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_WAREHOUSE_ID")
            if not os.environ.get(k)
        ]
        if missing:
            raise ValueError(f"missing environment variable(s): {', '.join(missing)}")
        return cls(
            host=os.environ["DATABRICKS_HOST"],
            token=os.environ["DATABRICKS_TOKEN"],
            warehouse_id=os.environ["DATABRICKS_WAREHOUSE_ID"],
            **kwargs,
        )

    # ------------------------------------------------------------------
    # SQL
    # ------------------------------------------------------------------
    def query(self, sql: str) -> list[dict[str, Any]]:
        """Run one statement and return every row as a dict.

        Handles the three things a naive reader gets wrong: the response is COLUMNAR
        (`manifest.schema.columns` plus a positional `data_array`); only chunk 0
        arrives inline — the rest must be fetched, or a wide window truncates
        silently; and a statement still running when `wait_timeout` elapses comes back
        as HTTP 200 with `state: PENDING`, which has to be polled rather than treated
        as a failure.
        """
        import requests

        headers = {"Authorization": f"Bearer {self.token}"}
        resp = requests.post(
            f"{self.host}{_STATEMENTS_PATH}",
            headers=headers,
            json={
                "statement": sql,
                "warehouse_id": self.warehouse_id,
                "wait_timeout": self.wait_timeout,
            },
            timeout=self.timeout,
        )
        _raise_for_api_error(resp, "statement submission")
        body = resp.json()
        body = self._await_statement(body, headers)

        manifest = body.get("manifest") or {}
        columns = [c["name"] for c in (manifest.get("schema") or {}).get("columns", [])]
        result = body.get("result") or {}
        arrays: list[list[Any]] = list(result.get("data_array") or [])

        total_chunks = int(manifest.get("total_chunk_count") or 1)
        statement_id = body.get("statement_id")
        for index in range(1, total_chunks):
            chunk_resp = requests.get(
                f"{self.host}{_STATEMENTS_PATH}/{statement_id}/result/chunks/{index}",
                headers=headers,
                timeout=self.timeout,
            )
            # THE check this whole method exists for. A failed chunk fetch returns a JSON
            # error body with no `data_array`, so `or []` would append zero rows, the loop
            # would continue, and `query()` would return a PARTIAL result reporting
            # success — measured live: a 403/404/503 on chunk 1 of 2 silently dropped 25%
            # of the window. Billing a fraction of a window with no error is the single
            # worst outcome this reader can produce, so it must raise.
            _raise_for_api_error(chunk_resp, f"result chunk {index} of {total_chunks}")
            arrays.extend((chunk_resp.json()).get("data_array") or [])
        if total_chunks > 1:
            logger.info("lago: databricks result spanned %d chunks (%d rows)", total_chunks, len(arrays))

        # End-to-end truncation check, independent of cause: catches a short read that no
        # per-request status could reveal (a chunk that returns HTTP 200 with fewer rows
        # than promised, a manifest/chunk disagreement). `total_row_count` is absent on
        # some statement kinds, so only assert when Databricks actually stated a count.
        promised = manifest.get("total_row_count")
        if promised is not None and int(promised) != len(arrays):
            raise RuntimeError(
                f"Databricks returned {len(arrays)} row(s) but the manifest promised "
                f"{int(promised)} across {total_chunks} chunk(s) — refusing to bill a "
                f"partial window (statement_id={statement_id})"
            )
        # A row set with no column names decodes to `{}` per row, which every layer
        # downstream degrades cleanly and wrongly on: all-zero usage, and a confident
        # `{"cost": 0, "tokens": 0}` for a window that had real traffic. Not observed on
        # this API (every SELECT returns a full schema, zero-row reads included) — this
        # guards the decode, not a known bug. Deliberately keeps `strict=False` on the
        # zip below: a length mismatch is an API-contract violation the row-count check
        # above already catches, and strict=True would newly reject reads that work today.
        if arrays and not columns:
            raise RuntimeError(
                "Databricks returned rows with no `manifest.schema.columns` — cannot "
                f"decode {len(arrays)} row(s) (statement_id={statement_id})"
            )

        return [dict(zip(columns, row, strict=False)) for row in arrays]

    def _await_statement(self, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        """Poll a statement to a terminal state, returning the body that carries results.

        A statement still executing when the request's `wait_timeout` elapses returns
        **HTTP 200** with `state: PENDING`/`RUNNING` and a `statement_id` — not an error.
        Treating that as fatal breaks exactly the case this class tells operators to use:
        one wide window per run, which on a cold warehouse routinely takes longer than
        the 50s ceiling Databricks allows for `wait_timeout`.
        """
        import time

        import requests

        deadline = time.monotonic() + self.timeout
        while True:
            state = (body.get("status") or {}).get("state")
            if state == "SUCCEEDED":
                return body
            if state not in ("PENDING", "RUNNING"):
                raise RuntimeError(f"Databricks statement {state}: {(body.get('status') or body)}")
            statement_id = body.get("statement_id")
            if not statement_id or time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Databricks statement still {state} after {self.timeout}s "
                    f"(statement_id={statement_id}); raise `timeout` or narrow the window"
                )
            time.sleep(2.0)
            poll = requests.get(
                f"{self.host}{_STATEMENTS_PATH}/{statement_id}",
                headers=headers,
                timeout=self.timeout,
            )
            # This path already failed loudly without a status check — a non-OK body has
            # no `status.state`, so the loop's own `state not in (PENDING, RUNNING)` branch
            # raised. Checking here only changes WHAT the operator reads: the real cause
            # ("Invalid access token", "statement expired") instead of
            # `Databricks statement None: {...}`.
            _raise_for_api_error(poll, "statement poll")
            body = poll.json()

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def read_usage(
        self, since: str | datetime = "1 day", *, event_id_prefix: str = "dbx"
    ) -> Iterator[DatabricksUsageRow]:
        """Yield every billable row in the window, shaped for `emit()`.

        BYOK and hosted are read from DIFFERENT tables and must not overlap, or a
        call gets billed twice:

          * BYOK  — `external_model_spend`, which carries Databricks' own metered
            USD *and* your `request_tags`, so cost arrives already attributed per
            subscription. Token counts are joined on from `ai_gateway.usage` for
            reporting; they are not used to compute the price.
          * hosted — `ai_gateway.usage` only, billed as token counts.

        Rows whose usage is entirely zero (failed calls are recorded with NULL token
        counts) are skipped, so nothing emits an empty event.

        Reads **whole closed hours only**: the window is floored to the hour at both
        ends and the current, still-aggregating hour is excluded, because a spend row
        for it cannot be complete yet. `_window_bounds` documents why each of those
        three properties is load-bearing. The practical consequence for a caller is
        that the newest hour of traffic arrives on the NEXT run, so pass a window
        comfortably wider than your run interval — this reader keeps no cursor.
        """
        # Rewritten per read rather than appended to, so a later read of a healthy
        # window cannot leave an earlier read's gap standing as if it were current.
        # Cleared here, ahead of the empty-window return below, so that path clears
        # it too. Complete once the caller has drained the iterator.
        self.deferred_buckets = []
        lower, upper = _window_bounds(since)
        if lower >= upper:
            # Not an error, but it must not read as success either: the caller asked
            # for a window that lies entirely inside the hour this reader excludes, so
            # zero rows here says nothing about whether there was traffic.
            logger.warning(
                "lago: since=%r resolves to [%s, %s), which is empty — the window falls "
                "inside the current, still-aggregating hour that this reader excludes. "
                "Nothing was read; widen the window past the hour boundary.",
                since,
                lower.isoformat(),
                upper.isoformat(),
            )
            return
        # One pair of literals, both statements — see `_window_bounds`. Resolving the
        # bounds here rather than in SQL is what makes the two reads the same window.
        window, ceiling = _timestamp_sql(lower), _timestamp_sql(upper)

        spend = self.query(f"""
            SELECT record_id,
                   date_trunc('HOUR', usage_start_time) AS bucket,
                   usage_metadata.provider              AS provider,
                   usage_metadata.model                 AS model,
                   to_json(custom_tags.request_tags)    AS request_tags,
                   usage_quantity
            FROM system.ai_gateway.external_model_spend
            WHERE usage_start_time >= {window} AND usage_start_time < {ceiling}
        """)

        usage = self.query(f"""
            SELECT * FROM system.ai_gateway.usage
            WHERE event_time >= {window} AND event_time < {ceiling}
            ORDER BY event_time
        """)

        # Extract once per row and reuse: this loop and the hosted loop below both need
        # the CanonicalUsage, and extraction parses several JSON-string columns.
        extracted = [(row, extract_databricks_log(row)) for row in usage]

        # Index token counts by the spend table's own grouping key, so a BYOK event
        # can carry real counts alongside Databricks' dollar figure.
        tokens: dict[tuple[Any, ...], CanonicalUsage] = {}
        for row, u in extracted:
            if u.provider == "databricks":
                continue
            if not u.nonzero_numeric():
                # A failed call carries NULL tokens and bought nothing, so Databricks
                # meters no dollars for it — its key can never match a spend row. Indexed,
                # it becomes a bucket the warning below tells the operator to re-run the
                # window for, which can never bill it: measured live over 2026-08-06,
                # 28 of the 29 reported buckets were these phantoms (all 83 zero-usage
                # rows were 4xx/5xx), including the one example row the warning names.
                # The hosted loop applies the same guard for the same reason.
                continue
            key = (
                _bucket_of(row.get("event_time")),
                u.provider,
                str(row.get("destination_model") or ""),
                _canonical_tags(row.get("request_tags")),
            )
            prior = tokens.get(key)
            tokens[key] = _merge_usage(prior, u) if prior else _as_bucket(u)
        billed_keys: set[tuple[Any, ...]] = set()

        for row in spend:
            usd = _safe_float(row.get("usage_quantity"))
            if not usd:
                continue
            key = (
                _truncate_hour(_stamp(row.get("bucket"))),
                str(row.get("provider") or ""),
                str(row.get("model") or ""),
                _canonical_tags(row.get("request_tags")),
            )
            billed_keys.add(key)
            joined = tokens.get(key)
            usage_obj = joined or CanonicalUsage(
                model=str(row.get("model") or ""),
                provider=str(row.get("provider") or ""),
                api="databricks_gateway",
            )
            sub = resolve_databricks_subscription({"request_tags": row.get("request_tags")})
            yield DatabricksUsageRow(
                usage=usage_obj,
                subscription=sub,
                # record_id is unique per aggregated spend row — a natural
                # idempotency key. See `event_id_for` for why it is still scoped.
                row_id=_row_id(row, "record_id"),
                kind="spend",
                usd_cost=usd,
                prefix=event_id_prefix,
                raw=row,
            )

        # A BYOK bucket with no spend row is billed by NEITHER loop, so say so rather
        # than lose it. `external_model_spend` is an hourly aggregate that lags
        # `ai_gateway.usage`, so the window's most recent hour routinely has token rows
        # whose dollar row does not exist yet; a $0 metered row does the same. Re-running
        # the window once Databricks has aggregated picks them up — but only if the
        # operator knows to, which is what this warning is for.
        unbilled = sorted(set(tokens) - billed_keys)
        # Same facts as the warning, in a shape a caller can act on rather than grep
        # for — `backfill_databricks` reports the count through `on_error`.
        self.deferred_buckets = [
            {"hour": k[0], "provider": k[1], "model": k[2], "request_tags": k[3]} for k in unbilled
        ]
        if unbilled:
            logger.warning(
                "lago: %d BYOK token bucket(s) in this window have no external_model_spend "
                "row yet and were NOT billed (e.g. hour=%s provider=%s model=%s). The spend "
                "table lags; re-run this window later to bill them.",
                len(unbilled),
                unbilled[0][0],
                unbilled[0][1],
                unbilled[0][2],
            )

        for row, u in extracted:
            if u.provider != "databricks":
                continue  # BYOK already billed from spend above — never twice
            if not u.nonzero_numeric():
                continue  # failed calls carry NULL tokens
            yield DatabricksUsageRow(
                usage=u,
                subscription=resolve_databricks_subscription(row),
                # One request with a fallback yields several invocations, so
                # invocation_id is the per-row key; request_id is the fallback for a
                # row that somehow carries no invocation.
                row_id=_row_id(row, "invocation_id", "request_id"),
                kind="usage",
                usd_cost=None,
                prefix=event_id_prefix,
                raw=row,
            )


def _row_id(row: dict[str, Any], *columns: str) -> str:
    """First usable id among `columns`, falling back to a hash of the whole row.

    Two ways the obvious `_safe_str(a or b)` goes wrong, both silent and both losing
    money. A row with NULL ids yields ""; so does a row whose id a driver hands back as
    a UUID or int object rather than a str, because `or` selects it and `_safe_str`
    rejects the type without ever trying the next column. Either way `event_id_for`
    still produces a well-formed key (`dbx_usage_sub_x_`), so EVERY such row in the
    window shares one `transaction_id` — Lago accepts the first and rejects the rest as
    duplicates, and those calls are never billed at all.

    The content hash keeps the key deterministic, so re-running the same window is still
    idempotent, which a random UUID would break.
    """
    for column in columns:
        value = row.get(column)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    digest = hashlib.sha256(
        json.dumps(row, sort_keys=True, default=str).encode("utf-8", "replace")
    ).hexdigest()
    return f"sha{digest[:32]}"


def _safe_float(v: Any) -> float:
    """Coerce a decimal(38,18) column to float. Returns 0.0 on anything unparseable.

    `float()` raises on a non-numeric string, and this runs inside a generator whose
    docstring promises one malformed row cannot take down the batch — an exception here
    would abort the window mid-emit with no record of where it stopped. 0.0 means the
    row is skipped like any other zero-dollar row. Mirrors the JS port, where
    `Number()` yields NaN and the same `if (!usd)` skips it.
    """
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _truncate_hour(value: str) -> str:
    """Normalize a timestamp string to its hour, for joining across the two tables."""
    return value[:13] if len(value) >= 13 else value


def _stamp(value: Any) -> str:
    """Stringify a timestamp column, whatever the access path produced.

    The Statement Execution API returns TIMESTAMPs as strings, but
    `databricks-sql-connector` returns real `datetime` objects — a documented, supported
    input path. `_safe_str` would map those to "", collapsing every hour of the window
    into one join bucket and dropping the `bucket` reconcile dimension entirely.
    """
    if value is None:
        return ""
    # ISO-8601 for a real datetime, matching what the REST API returns as a string and
    # what the JS port's `toISOString()` produces — so the hour prefix `_truncate_hour`
    # keys on is the same across both access paths and both repos.
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _epoch(value: Any) -> int | None:
    """A timestamp column as unix seconds, from either access path.

    The Statement Execution API returns TIMESTAMPs as ISO-8601 strings ending in "Z";
    `databricks-sql-connector` returns real `datetime` objects. Both are supported
    input paths and both have to yield the same instant.

    The trailing "Z" is rewritten by hand because Python 3.10 — still supported here —
    rejects it in `fromisoformat`, and every string this table returns carries one. A
    stamp with no offset at all is read as UTC, which is both the warehouse's own
    session timezone and what the JS port does with the same string.

    Unparseable returns None rather than raising: a bad timestamp column must not cost
    the caller the whole row.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return int(moment.timestamp())
    text = str(value).strip()
    if not text:
        return None
    if text[-1] in "Zz":
        text = f"{text[:-1]}+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    try:
        return int(moment.timestamp())
    # A year outside the C time range; platform dependent, hence caught here too.
    except (OverflowError, OSError):
        return None


def _bucket_of(value: Any) -> str:
    return _truncate_hour(_stamp(value))


def _canonical_tags(value: Any) -> str:
    """Stable string form of a request_tags map, for use as a join key."""
    import json

    if isinstance(value, str):
        try:
            value = json.loads(value or "{}")
        except ValueError:
            return str(value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return "{}"


# Extras that describe the endpoint a spend bucket's requests went to, rather than any
# one of those requests. Everything else the adapter captures — `invocation_id`,
# `request_id`, `status_code` — is per-request, and carrying it on an hourly aggregate
# states one sampled request's value as if it described the whole hour.
_BUCKET_INVARIANT_EXTRAS = (
    "endpoint_name",
    "endpoint_id",
    "destination_type",
    "destination_name",
    "api_type",
)


def _as_bucket(u: CanonicalUsage) -> CanonicalUsage:
    """One usage row restated as a spend-bucket representative.

    Applied to the FIRST row of a bucket as well as to merges, so a bucket holding one
    request is described the same way as a bucket holding ten — otherwise `status_code`
    would survive on single-request hours and vanish on busy ones.
    """
    out = CanonicalUsage(
        model=u.model,
        provider=u.provider,
        api=u.api,
        extras={k: v for k, v in u.extras.items() if k in _BUCKET_INVARIANT_EXTRAS},
    )
    for name in CanonicalUsage.NUMERIC_FIELDS:
        setattr(out, name, getattr(u, name))
    return out


def _merge_usage(a: CanonicalUsage, b: CanonicalUsage) -> CanonicalUsage:
    """Sum the numeric fields of two rows in the same spend bucket."""
    merged = _as_bucket(a)
    for name in CanonicalUsage.NUMERIC_FIELDS:
        setattr(merged, name, getattr(a, name) + getattr(b, name))
    return merged

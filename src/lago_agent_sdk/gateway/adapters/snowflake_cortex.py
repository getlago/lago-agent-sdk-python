"""Snowflake Cortex usage adapter — maps an ACCOUNT_USAGE row to CanonicalUsage.

Verified against real rows read from a live Snowflake account over the SQL API
(`POST /api/v2/statements`). Two views report Cortex usage and this module
serves both:

  CORTEX_REST_API_USAGE_HISTORY      one row per request -> extract_snowflake_rest_log
  CORTEX_AI_FUNCTIONS_USAGE_HISTORY  one row per query   -> extract_snowflake_functions_log

They are two extractors rather than one because almost nothing about them lines
up: tokens arrive in `TOKENS_GRANULAR` (an OBJECT, four keys, cache ADDITIVE)
against `METRICS` (an ARRAY of metric/unit/value records, no cache key at all),
the functions view alone carries `FUNCTION_NAME`, `CREDITS`, `IS_COMPLETED`,
`QUERY_ID` and `ROLE_NAMES`, and the two views' timestamps are different
Snowflake types. One function over both would branch on every one of those.

A row reaches this function as a plain dict of `{COLUMN_NAME: value}`, whichever
way the caller read it: the SQL API returns columnar `data` the caller zips
against `resultSetMetaData.rowType`, and `snowflake-connector-python`'s
`DictCursor` yields the same shape. Snowflake upper-cases unquoted identifiers,
so the keys arrive UPPERCASE unless the caller quoted a lowercase alias — both
are accepted, see `_column`.

Field mapping (`CORTEX_REST_API_USAGE_HISTORY`), every line of it live-verified
against the fixture named beside it:

  TOKENS_GRANULAR.input              -> input          rest_plain.json
  TOKENS_GRANULAR.output             -> output         rest_plain.json
  TOKENS_GRANULAR.cache_read_input   -> cache_read     rest_cache_read.json
  TOKENS_GRANULAR.cache_write_input  -> cache_write    rest_cache_write.json
  TOKENS_GRANULAR.<anything else>    -> extras["tokens_granular.<key>"]
  MODEL_NAME                         -> model          (customer's spelling, unnormalized)
  TOKENS                             -> extras["tokens"]   NEVER a token field, see below
  REQUEST_ID / INFERENCE_REGION / USER_ID / QUERY_TAG / START_TIME / END_TIME
                                     -> extras (lower-cased keys)
  provider                           -> constant "snowflake"
  api                                -> constant "snowflake_cortex_rest"

ASSUMED, not verified: nothing above. The one thing this module claims without a
captured row is that a NULL `TOKENS_GRANULAR` can occur at all — see
`_MISSING_GRANULAR_KEY`.

THE TOKEN SHAPE, which is the part that decides whether this over-bills.
Measured across every row the live view holds (24 of 24): `TOKENS` equals the
sum of EVERY value in `TOKENS_GRANULAR`, cached block included. So on this
surface:

  * cache is ADDITIVE — `input` EXCLUDES `cache_read_input`/`cache_write_input`,
    which is Anthropic's convention and the inverse of OpenAI's. A cached row
    reads `{"cache_read_input": 8745, "input": 7, "output": 6}` against
    `TOKENS: 8758`.
  * `TOKENS = input + output` therefore holds ONLY where no cache key is
    present. Mapping `TOKENS` onto `input` bills 8758 for 7 real input tokens
    and re-bills the cached block a second time as `cache_read` — 2.0x on the
    call. It stays in `extras`, exactly as the Databricks adapter refuses to map
    that table's `total_tokens`.
  * `api = "snowflake_cortex_rest"` must NOT be added to pricing's
    `_OPENAI_SHAPED_APIS`. That set says "cache_read already sits inside input,
    do not subtract it again"; here it does not, and claiming otherwise
    under-bills every cached call by the whole cached block.

Reasoning has NO key on this view, verified rather than assumed: the call that
reported `thinking_tokens: 127` of 262 output tokens on Snowflake's Anthropic
wire lands here as `{"input": 60, "output": 262}` (rest_thinking.json). Thinking
is inside `output`, so there is nothing to map and nothing to subtract.

A FAILED call produces NO ROW at all — a 403 (unknown model) and a 400
(rejected parameter) were driven alongside a successful call, and only the
success reached the view. Both are pre-inference failures, the only kind the
capture account can produce, so a row that dies mid-generation is unobserved
rather than impossible. That is why the all-zero path below is written and
tested from a hand-made row: it protects against a shape nobody has seen.

ATTRIBUTION. This view cannot identify a tenant on its own — `USER_ID` is a
numeric Snowflake user, not a Lago subscription. REST traffic is meant to bill
through the live `wrap()` path (INT-228); the view exists so a customer can
reconcile what Lago billed against what Snowflake recorded (INT-231). See
`resolve_snowflake_subscription` for what little attribution is available.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ...canonical import CanonicalUsage

# `TOKENS_GRANULAR` keys this adapter MAPS onto a CanonicalUsage field. Anything else
# nested there is drift and is surfaced in `extras` under a dotted key.
#
# The two cache spellings are the reason this set is written down rather than inferred:
# they are `cache_read_input` / `cache_write_input`, which appear in no Snowflake
# documentation and match no spelling used anywhere else in this tree (`cache_read`,
# `cached_tokens`, `cache_creation_input_tokens`). An adapter written from the docs maps
# `{input, output}` alone and sends 8,745 tokens to `extras` while billing 13.
#
# Dotted per unknown key, NOT the object swept whole under `extras["tokens_granular"]`:
# four keys here are mapped, so publishing the container whole would re-publish counts
# already billed as token events. Same convention as `openai_native`'s detail sweep.
_MAPPED_GRANULAR_KEYS = frozenset(
    {
        "input",
        "output",
        "cache_read_input",
        "cache_write_input",
    }
)

# Columns lifted into `extras` verbatim, lower-cased. `TOKENS` is here on purpose and
# is never mapped to a numeric field (see the module docstring); the timestamps are
# here because the caller — not this pure function — decides how to stamp an event,
# and they need parsing this module deliberately does not do: REST is `timestamp_tz`,
# arriving as "1787677200.000000000 1440" (epoch seconds, nanos, offset-minutes+1440),
# while the functions view is `timestamp_ltz` and arrives as a bare "1787677200". One
# parser does not serve both, and neither is a per-request time — both views bucket to
# the hour, so two calls a minute apart carry identical START_TIME.
_EXTRA_COLUMNS = (
    "REQUEST_ID",
    "MODEL_NAME",
    "TOKENS",
    "INFERENCE_REGION",
    "USER_ID",
    "QUERY_TAG",
    "START_TIME",
    "END_TIME",
)

# Set when a row carries a positive `TOKENS` but no usable `TOKENS_GRANULAR`, i.e. real
# usage this adapter cannot split into billable fields. It extracts to all-zero, so the
# caller emits nothing — and without this marker that 100% under-bill is indistinguishable
# from a correctly-ignored failed row, which is precisely the shape of silent loss this
# codebase keeps paying for. Unobserved on the live view (0 of 24 rows), so the marker is
# a guard rather than a workaround.
#
# It is a MARKER here and a MAPPING on the functions view, where a `total`-only row is
# billed as `input` under `_METRICS_TOTAL_ONLY_KEY`. The difference is not taste:
# `METRICS.total` is a plain token count, while `TOKENS` INCLUDES the cached block, so
# billing it as `input` on a cached row re-bills that block a second time — 2.0x on the
# call, which is worse than the zero this marker reports.
_MISSING_GRANULAR_KEY = "tokens_granular_missing"

# Attribution sources, in the order `resolve_snowflake_subscription` tries them by
# default. Named strings rather than positional arguments so a caller's override reads
# as an intention (`order=("query_tag",)`) instead of a permutation.
_DEFAULT_SUBSCRIPTION_ORDER = ("query_tag", "role_names", "user_id")

# The key a customer puts inside QUERY_TAG. Identical to Cloudflare's
# `cf-aig-metadata.lago_subscription` and Databricks' `request_tags.lago_subscription`,
# so one instruction covers all three gateways.
_SUBSCRIPTION_TAG_KEY = "lago_subscription"


def _column(row: dict[str, Any], name: str) -> Any:
    """Read a column by name, tolerating the case Snowflake actually returned.

    Unquoted identifiers are folded to UPPERCASE, which is what every documented
    read path yields — but `SELECT tokens AS "tokens"` preserves a quoted lowercase
    alias, and a caller who normalized keys themselves would otherwise silently
    extract zeros from a row that has every value. Exact match wins so a row
    carrying both spellings is never ambiguous.
    """
    if name in row:
        return row[name]
    lowered = name.lower()
    return row.get(lowered)


def _safe_dict(v: Any) -> dict[str, Any]:
    """Coerce an OBJECT column to a dict, accepting either shape it arrives in.

    `TOKENS_GRANULAR` is an OBJECT. The SQL API serializes structured columns as
    TEXT — measured, it arrives as '{\\n  "input": 9,\\n  "output": 12\\n}' — while a
    connector with native types hands back a real dict. Accepting both means the
    adapter works whichever read path the caller chose, instead of reading zeros out
    of a string it never parsed. A parse failure degrades to empty; a malformed row
    must not take down a backfill mid-window.
    """
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip().startswith("{"):
        try:
            parsed = json.loads(v)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _safe_list(v: Any) -> list[Any]:
    """Coerce an ARRAY column (`ROLE_NAMES`) to a list, same two shapes as above."""
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            parsed = json.loads(v)
        except ValueError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _safe_int(v: Any) -> int:
    """Coerce to a non-negative int.

    Numeric columns arrive as STRINGS over the SQL API ("8758", and `USER_ID` is a
    numeric column too), and as None on any row that reports no usage. Both must land
    on 0 rather than raise.
    """
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def _safe_str(v: Any) -> str:
    return v if isinstance(v, str) else ""


def extract_snowflake_rest_log(row: dict[str, Any]) -> CanonicalUsage:
    """Translate one `CORTEX_REST_API_USAGE_HISTORY` row -> CanonicalUsage.

    Pure: no HTTP, no SDK state, no import from `adapters/`. Missing or malformed
    fields degrade to zero/empty rather than raising, so one bad row in a backfill
    window cannot take down the run.
    """
    granular = _safe_dict(_column(row, "TOKENS_GRANULAR"))

    extras: dict[str, Any] = {name.lower(): _column(row, name) for name in _EXTRA_COLUMNS}

    # Drift sweep. A key that appears here later is a token count nobody has classified:
    # it must not be silently miscounted as one of the four above, and it must not vanish.
    # Not hypothetical on this surface — `cache_read_input` and `cache_write_input` were
    # themselves absent from the two rows captured first, and the view grew a whole COLUMN
    # (`QUERY_TAG`, 8 columns to 9) between two captures eight hours apart. A hand-written
    # key list here is stale within a day of being written.
    for key, value in granular.items():
        if key not in _MAPPED_GRANULAR_KEYS:
            extras[f"tokens_granular.{key}"] = value

    usage = CanonicalUsage(
        input=_safe_int(granular.get("input")),
        output=_safe_int(granular.get("output")),
        cache_read=_safe_int(granular.get("cache_read_input")),
        cache_write=_safe_int(granular.get("cache_write_input")),
        # `MODEL_NAME` keeps the customer's spelling, fine-tunes included
        # (`database.schema.model`). Normalizing for a price lookup is the pricing
        # layer's job and must not reach what Lago is told the model was.
        model=_safe_str(_column(row, "MODEL_NAME")),
        provider="snowflake",
        api="snowflake_cortex_rest",
        extras=extras,
    )

    # See _MISSING_GRANULAR_KEY. Deliberately keyed off "nothing mapped" rather than
    # "granular is empty": a row whose granular object holds ONLY unrecognized keys is
    # the same 100% loss, and the drift sweep alone would let it pass quietly.
    if not usage.nonzero_numeric() and _safe_int(_column(row, "TOKENS")) > 0:
        extras[_MISSING_GRANULAR_KEY] = True

    return usage


# ---------------------------------------------------------------------------
# CORTEX_AI_FUNCTIONS_USAGE_HISTORY — the AI SQL functions view
# ---------------------------------------------------------------------------

# `METRICS` metrics this adapter MAPS onto a CanonicalUsage field, keyed by an entry's
# `key.metric`. Anything else in the array is drift and reaches `extras` under a dotted
# key.
#
# `total` is in this set, and putting it there is the billing decision this view forces.
# Measured across all 42 captured rows: `AI_COMPLETE` reports `{input, output}`, and
# every other function — `AI_SUMMARIZE`, `AI_TRANSLATE`, `AI_SENTIMENT`, `AI_CLASSIFY`,
# `AI_EMBED` — reports `{total}` ALONE, never both. `CanonicalUsage` has no `total`
# field and does not get one (11 numeric fields, every pricing and emit path keyed off
# them), so an adapter that maps `input`/`output` and leaves `total` to the drift sweep
# extracts all-zero for five of the six function types and bills NOTHING for them: a
# 100% under-bill on every task-specific AI SQL function, and silent, because `extras`
# is never sent to Lago.
_MAPPED_METRICS = frozenset({"input", "output", "total"})

# `key.unit` values this adapter is willing to read as a token count. Every captured row
# says "tokens"; the empty string covers an entry that omits the unit.
#
# The guard exists because a `METRICS` entry is a metric NAME plus a UNIT, and only the
# pair means anything: Cortex meters some functions in units that are not tokens at all
# (AI_PARSE_DOCUMENT is documented per page), so `{"metric": "input", "unit": "pages"}`
# is a shape this array can express. Billing 12 pages as 12 tokens is the kind of wrong
# no later test can see, so a foreign unit goes to the drift sweep under
# `extras["metrics.<metric>.<unit>"]` — and if it was the row's only figure it trips
# `_METRICS_UNMAPPED_KEY`, so the unbilled row is loud instead of a zero.
_TOKEN_UNITS = frozenset({"tokens", ""})

# Columns lifted into `extras` verbatim, lower-cased. Three are here for a reason rather
# than for completeness:
#
#   CREDITS       what a customer sees in Snowflake's own cost view, so it is the figure
#                 they reconcile a Lago invoice against. Read, NEVER billed: there is no
#                 price mode on the Snowflake path and no credit rate anywhere in this
#                 SDK. It is evidence, not a billing input.
#   QUERY_ID      the per-row id the caller builds the idempotency key from. Rows are
#                 per query, so it is a genuine property of its row — it stays in
#                 `extras` and out of the dimensions on CARDINALITY grounds: one
#                 dimension value per query is a group-by nobody can use.
#   IS_COMPLETED  handed over rather than acted on — see the docstring. Measured: an
#                 in-flight query writes no row at all, so a FALSE row is reachable only
#                 where a query spans two hour buckets — exactly where `QUERY_ID` also
#                 stops being unique. Both are the caller's rule, not a pure extractor's.
_FUNCTIONS_EXTRA_COLUMNS = (
    "FUNCTION_NAME",
    "MODEL_NAME",
    "CREDITS",
    "IS_COMPLETED",
    "QUERY_ID",
    "QUERY_TAG",
    "ROLE_NAMES",
    "USER_ID",
    "WAREHOUSE_ID",
    "START_TIME",
    "END_TIME",
)

# `total`'s value, recorded whenever the array carries one — exactly as `extras["tokens"]`
# records the REST view's additive total, and for the same reason: it is what a
# reconciliation compares against, and on a row that ALSO reports `input`/`output` it
# must never be added to a numeric field or the row bills twice.
_METRICS_TOTAL_KEY = "metrics_total"

# Set when `total` was the row's ONLY token figure and was therefore billed as `input`.
#
# The COUNT is exact — `total` is every token the call consumed, and it is the only
# figure Snowflake reports for these functions. The SPLIT is the guess: `input` is
# chosen because it is right by construction for `AI_EMBED` (nothing is generated) and
# close for the classifiers (`AI_SENTIMENT`/`AI_CLASSIFY` return a label), while
# `AI_SUMMARIZE`/`AI_TRANSLATE` genuinely generate and are the case this understates.
# `output` was rejected as the default because it errs the other way on every function
# and over-bills a customer, which is not a recoverable direction.
#
# So the marker is the honest part: the row is billed at its true token count under one
# field, and a customer reconciling against Snowflake's own figures can see which rows
# carry a split this SDK invented. If the view ever reports both, the mapping below
# prefers the real split with no code change.
_METRICS_TOTAL_ONLY_KEY = "metrics_total_only"

# The functions-view twin of `_MISSING_GRANULAR_KEY`: real usage that extracted to
# all-zero, so the caller emits nothing. Set when a metric could not be mapped (unknown
# name, or a unit that is not tokens) or when `CREDITS` says Snowflake charged the
# account for a row this adapter found nothing billable in. Without it, that 100% loss
# is indistinguishable from a correctly-ignored failed row — which is the exact shape of
# silent under-bill this connector keeps being audited for.
_METRICS_UNMAPPED_KEY = "metrics_unmapped"


def _safe_float(v: Any) -> float:
    """Coerce to a non-negative float.

    `CREDITS` arrives as "0.000068400" over the SQL API and as a `Decimal` from a typed
    connector. It is read ONLY as evidence that a row consumed something (see
    `_METRICS_UNMAPPED_KEY`) and never as a billing input — nothing on the Snowflake path
    turns credits into money.
    """
    try:
        return max(0.0, float(v or 0))
    except (TypeError, ValueError):
        return 0.0


def _read_metrics(v: Any) -> tuple[dict[str, int], dict[str, Any]]:
    """Fold a `METRICS` ARRAY into mapped token counts plus the drift leftovers.

    The column is an ARRAY of `{"key": {"metric": ..., "unit": ...}, "value": N}` — not
    the flat `{input, output}` object the connector brief assumed. Measured, it arrives
    over the SQL API as '[\\n  {\\n    "key": {\\n      "metric": "input",\\n
    "unit": "tokens"\\n    },\\n    "value": 13\\n  }, ...]' and as a real list from a
    typed connector; both are accepted, same as `_safe_dict` for the REST view.

    Returns `({metric: tokens}, {dotted_extras_key: raw_value})`. Nothing is dropped: an
    entry this adapter cannot bill is a token count nobody has classified yet, and this
    view is months old on an actively-extended surface — the REST one grew two token keys
    and a whole column inside a single day.
    """
    counts: dict[str, int] = {}
    drift: dict[str, Any] = {}
    for index, entry in enumerate(_safe_list(v)):
        if not isinstance(entry, dict):
            # Not the documented entry shape at all. Kept under its position in the
            # array so two of them cannot collide and neither disappears.
            drift[f"metrics.{index}"] = entry
            continue
        key = _safe_dict(entry.get("key"))
        metric = _safe_str(key.get("metric")).strip().lower()
        unit = _safe_str(key.get("unit")).strip().lower()
        value = entry.get("value")
        if metric in _MAPPED_METRICS and unit in _TOKEN_UNITS:
            # Summed, not last-wins: one metric appearing twice is a shape this array can
            # express (it is a list, not an object), and keeping only the last value would
            # under-bill by the first with nothing to show for it.
            counts[metric] = counts.get(metric, 0) + _safe_int(value)
            continue
        # The unit joins the key only when it is NOT a token unit, so an unrecognized
        # metric reads as `metrics.<metric>` while a known metric in a foreign unit reads
        # as `metrics.input.pages` and cannot overwrite the token-counted one.
        suffix = "" if unit in _TOKEN_UNITS else f".{unit}"
        drift[f"metrics.{metric or index}{suffix}"] = value
    return counts, drift


def extract_snowflake_functions_log(row: dict[str, Any]) -> CanonicalUsage:
    """Translate one `CORTEX_AI_FUNCTIONS_USAGE_HISTORY` row -> CanonicalUsage.

    Pure: no HTTP, no SDK state, no import from `adapters/`. Missing or malformed fields
    degrade to zero/empty rather than raising, so one bad row cannot take down a window.

    Field mapping, each line labelled LIVE (a captured row proves it) or ASSUMED:

      METRICS[metric=input]   -> input     functions_ai_complete.json        LIVE
      METRICS[metric=output]  -> output    functions_ai_complete.json        LIVE
      METRICS[metric=total]   -> input     functions_total_only_*.json       LIVE figure,
                                           ASSUMED split, marked — see
                                           `_METRICS_TOTAL_ONLY_KEY`
      METRICS[anything else]  -> extras["metrics.<metric>[.<unit>]"]         ASSUMED
                                           (no captured row carries one; 0 of 42)
      MODEL_NAME              -> model     empty on the task functions,      LIVE
                                           populated for AI_EMBED
      CREDITS, IS_COMPLETED, QUERY_ID, QUERY_TAG, ROLE_NAMES, USER_ID,
      WAREHOUSE_ID, START_TIME, END_TIME
                              -> extras (lower-cased keys)                   LIVE
      provider                -> constant "snowflake"
      api                     -> constant "snowflake_cortex_functions"

    ONE ROW PER QUERY, not per hour — measured, and it decides the caller's idempotency
    key. Twelve identical `AI_COMPLETE` calls produced twelve rows sharing one
    `START_TIME`/`END_TIME` pair: only the TIMESTAMPS are hour-bucketed. So the key is
    built from `QUERY_ID`; a bucket-derived key collapses twelve billable calls into one
    event and bills a twelfth of the traffic.

    TIMESTAMPS ARE NOT PARSED HERE, and they are not the REST view's shape: these columns
    are `timestamp_ltz` and arrive as a bare "1787162400" where REST's `timestamp_tz`
    arrive as "1787162400.000000000 1440". One parser does not serve both, and an
    offset-less epoch has to be read as UTC in both ports or the same row bills hours
    apart in each. The caller stamps the event; this function only hands the value over.

    NO CACHE AND NO REASONING METRIC EXISTS ON THIS VIEW — 0 of 42 rows, across all six
    function types. So `"snowflake_cortex_functions"` must NOT be added to pricing's
    `_OPENAI_SHAPED_APIS`: that set exists to stop a cached block being subtracted twice,
    and here there is nothing to subtract. Adding it can only remove counts that were
    billed correctly.

    A FAILED CALL PRODUCES NO ROW, same as the REST view: driven live, a 403 and a 400
    alongside a success, and only the success ever appeared. The all-zero path below is
    therefore written and tested from hand-made rows — it guards a shape nobody has seen.

    `IS_COMPLETED` IS REPORTED, NOT ACTED ON — and the reason is measured, not guessed.
    Driven live 2026-08-26: one query ran 200 `AI_COMPLETE` calls then held itself open
    for 900s, with the whole view polled every 45s. Across 19 polls covering all 937s it
    ran, NO ROW EXISTED. The row appeared 141s AFTER the query completed, already `true`,
    with final METRICS and CREDITS, and did not move over the next six minutes. Snowflake
    documents "running queries are updated every 2 minutes (best effort), SLA 5 minutes";
    that did not happen with three times the SLA to observe it. So usage is invisible
    until its query ends, then lands complete — the RESTATEMENT this flag was feared to
    signal did not reproduce, and rows already billed are not rewritten underneath us.

    Do NOT read that as "FALSE is unreachable". The flag means "was the query completed IN
    THIS AGGREGATION WINDOW", and the view is hour-bucketed: Snowflake documents a query
    running 5:30->8:30 writing FOUR rows, one per bucket. The run above stayed inside one
    bucket, so the multi-window case is untested — and there the flag is the smaller half
    of the problem, because `QUERY_ID` stops being unique and an idempotency key built
    from it collides. That is a caller's rule, not a pure extractor's, which is why the
    flag is handed over rather than acted on here.
    """
    counts, drift = _read_metrics(_column(row, "METRICS"))

    extras: dict[str, Any] = {name.lower(): _column(row, name) for name in _FUNCTIONS_EXTRA_COLUMNS}
    extras.update(drift)

    input_tokens = counts.get("input", 0)
    output_tokens = counts.get("output", 0)
    total = counts.get("total", 0)
    if "total" in counts:
        extras[_METRICS_TOTAL_KEY] = total

    # The `{total}`-only row: five of the six function types, and the whole reason `total`
    # is a mapped key. Conditioned on there being no split to prefer, so a row reporting
    # both bills its real `input`/`output` and keeps `total` as evidence only.
    if total > 0 and input_tokens == 0 and output_tokens == 0:
        input_tokens = total
        extras[_METRICS_TOTAL_ONLY_KEY] = True

    usage = CanonicalUsage(
        input=input_tokens,
        output=output_tokens,
        # Empty on `AI_SUMMARIZE`/`AI_TRANSLATE`/`AI_SENTIMENT`/`AI_CLASSIFY` — those
        # functions take no model argument — and populated for `AI_EMBED` and
        # `AI_COMPLETE`. Empty is a fact about the row, not a failure to read it, and the
        # two vary independently of the token shape.
        model=_safe_str(_column(row, "MODEL_NAME")),
        provider="snowflake",
        api="snowflake_cortex_functions",
        extras=extras,
    )

    # Nothing billable came out, yet something says the call really ran: a metric this
    # adapter could not map, or CREDITS Snowflake charged the account for. Both are real
    # usage billing zero, and unmarked they are indistinguishable from the failed row
    # above, which SHOULD bill zero. `CREDITS` is the better witness of the two here —
    # this view has no `TOKENS` column to fall back on, and credits are non-zero on every
    # captured row.
    if not usage.nonzero_numeric() and (
        any(_safe_int(value) > 0 for value in drift.values()) or _safe_float(_column(row, "CREDITS")) > 0
    ):
        extras[_METRICS_UNMAPPED_KEY] = True

    return usage


def resolve_snowflake_subscription(
    row: dict[str, Any],
    order: Sequence[str] | None = None,
) -> str | None:
    """Pull the Lago subscription id from a Cortex usage row.

    Returns None when nothing in `order` yields one; the caller decides what an
    unattributed row means (drop it, route it to a default, warn). This function only
    reports whether attribution is present.

    `order` names the sources to try, first hit wins. Default
    `("query_tag", "role_names", "user_id")`:

      query_tag   `QUERY_TAG` parsed as JSON, then its "lago_subscription" key. This is
                  the ONLY customer-injectable attribution key on either view —
                  `ALTER SESSION SET QUERY_TAG = '{"lago_subscription": "sub_123"}'` —
                  and it is the same key Cloudflare and Databricks read from their own
                  metadata. A non-JSON tag is ignored rather than used whole: Snowflake
                  populates QUERY_TAG itself on some surfaces (a captured row carries
                  `{"app": "cortex_code_sandbox", ...}`), so treating an arbitrary tag as
                  a subscription id bills somebody's tooling label to a customer.
      role_names  first entry of the `ROLE_NAMES` array — the functions view's proxy for
                  a tenant on accounts where one role means one customer.
      user_id     `USER_ID`, stringified.

    **This view offers only `USER_ID`, and a Snowflake user is not a Lago
    subscription.** `ROLE_NAMES` does not exist on the REST view at all, and `QUERY_TAG`
    — present since the column appeared, NULL on all 24 captured rows — has never been
    observed carrying a value here. So on REST traffic the default order resolves to
    `USER_ID` in practice, i.e. a numeric Snowflake identity ("1") that matches a Lago
    subscription only if the customer maintains that mapping themselves. Callers who do
    not should pass `order=("query_tag",)` and let the row go unattributed, which is
    recoverable; billing the wrong subscription is not. Do NOT invent a mapping table
    here — REST usage is meant to bill through the live `wrap()` path, where the
    subscription is known at call time.
    """
    for source in order if order is not None else _DEFAULT_SUBSCRIPTION_ORDER:
        if source == "query_tag":
            tag = _safe_dict(_column(row, "QUERY_TAG"))
            value = tag.get(_SUBSCRIPTION_TAG_KEY)
            if isinstance(value, str) and value:
                return value
        elif source == "role_names":
            roles = _safe_list(_column(row, "ROLE_NAMES"))
            if roles and isinstance(roles[0], str) and roles[0]:
                return roles[0]
        elif source == "user_id":
            user_id = _column(row, "USER_ID")
            # Numeric column, so it arrives as "1" over the SQL API and as 1 from a
            # typed connector. Both have to produce the same subscription id, or the
            # same row bills to two different places depending on how it was read.
            if isinstance(user_id, (str, int)) and not isinstance(user_id, bool):
                text = str(user_id).strip()
                if text:
                    return text
    return None

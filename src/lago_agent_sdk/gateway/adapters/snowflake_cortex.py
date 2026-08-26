"""Snowflake Cortex usage adapter — maps an ACCOUNT_USAGE row to CanonicalUsage.

Verified against real rows read from a live Snowflake account over the SQL API
(`POST /api/v2/statements`). Two views report Cortex usage and this module
serves both; only the REST one is implemented here.

  CORTEX_REST_API_USAGE_HISTORY   one row per request  -> extract_snowflake_rest_log
  CORTEX_AI_FUNCTIONS_USAGE_HISTORY  one row per query -> (INT-225)

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
# a guard rather than a workaround: what to DO about such a row (bill `TOKENS` as input,
# report it, drop it) is the same open question as the functions view's `total`-only
# rows, and is decided in INT-225, not here.
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

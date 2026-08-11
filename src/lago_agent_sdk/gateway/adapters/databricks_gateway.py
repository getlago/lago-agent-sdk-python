"""Databricks AI Gateway usage adapter — maps a `system.ai_gateway.usage` row to CanonicalUsage.

Verified against real rows read from a live workspace over the SQL Statement
Execution API (226 rows, all 36 columns; the public docs undercount at ~28 and
omit `service_*`, `mcp_metadata`, `routing_information`, `invocation_metadata`).

Unlike Cloudflare, Databricks exposes no REST logs API — usage lands in a Unity
Catalog Delta table queried over SQL. The row reaches this function as a plain
dict: `databricks-sql-connector` yields `Row` objects with `.asDict()`, the
Node driver yields column-keyed objects natively, and the raw Statement
Execution API returns columnar `data_array` the caller zips. All three end up
here as `{column_name: value}`.

Field mapping (`system.ai_gateway.usage`):
  input_tokens                                  → input
  output_tokens                                 → output
  token_details.cache_read_input_tokens         → cache_read
  token_details.cache_creation_input_tokens     → cache_write
  token_details.output_reasoning_tokens         → reasoning
  destination_type + destination_name/_model    → model, provider  (see below)
  api                                           → hardcoded "databricks_gateway"
  extras                                         → routing/identity columns

`total_tokens` is deliberately NOT mapped: it is derived from the others and
mapping it would double-count. Same reason the Cloudflare adapter skips
`usage_metadata.total_tokens`.

TWO MEASURED QUIRKS drive the shapes below. Both were wrong in an earlier draft
of this connector that reasoned from the docs alone.

1. `destination_name` means DIFFERENT things per destination type. For a hosted
   model it is the model (`system.ai.llama-4-maverick`); for BYOK it is the
   PROVIDER SERVICE (`workspace.default.anthropickey`) — a credential name, not
   a model. So a single "model, falling back to name" rule yields a credential
   as the model for every BYOK row.

2. `destination_model` is unstable for hosted models. The same
   `destination_name` was observed reporting both `llama-4-maverick` and
   `Llama 4 Maverick` — a human display label with spaces and capitals — and
   likewise `gpt-oss-20b` / `GPT OSS 20B`. It is clean and stable for BYOK
   (`claude-sonnet-4-5`, `gpt-4o`), so it is authoritative there and unusable
   for hosted.

BILLING HAZARD, documented because it is the inverse of every other adapter
here: this table's `input_tokens` INCLUDES both cache_read and cache_write,
where the providers' own response bodies EXCLUDE them. Measured per row —
`input=1825, cache_read=1812` for a call whose response body reported
`input_tokens: 13`. Only one of cache_read/cache_write is ever non-zero per
row, so `input - cache_read - cache_write` recovers the true non-cached input
exactly. This adapter extracts the row FAITHFULLY and does not subtract: the
intended billing path takes Databricks' own metered USD from
`system.ai_gateway.external_model_spend` via `emit(usd_cost=...)`, which never
touches token counts. Computing cost from these tokens instead would over-bill
3.04x with no subtraction, or 1.40x subtracting only cache_read.

If a computed fallback is ever added, the correction needs BOTH keys, not one.
`api == "databricks_gateway"` alone distinguishes a table row from a live call
(a `provider="anthropic"` row from this table needs correcting; a live
`provider="anthropic"` call must not) — but it is not sufficient, because
`compute_cost` ALREADY subtracts cache_read for providers in
_INPUT_INCLUDES_CACHE_READ. So an openai/gemini row must pass through untouched
while an anthropic row must be pre-subtracted. Measured by getting it wrong:
correcting an openai row double-subtracts and billed $0.00354 against a true
$0.004065, a 13% UNDER-bill.

Failed calls (403/404, and every Gemini call while that connection is broken)
are recorded with NULL token counts. They extract to all-zero, so
`nonzero_numeric()` is empty and the caller emits nothing — the same way a
Cloudflare cache hit extracts to zero.
"""

from __future__ import annotations

import json
from typing import Any

from ...canonical import CanonicalUsage

# Databricks' own name for a first-party pay-per-token foundation model. Any other
# destination type (observed: "EXTERNAL_FOUNDATION_MODEL", or NULL on rows rejected
# before routing) is BYOK — the customer's own vendor credential behind a Unity
# Catalog connection.
_HOSTED_DESTINATION_TYPE = "PAY_PER_TOKEN_FOUNDATION_MODEL"

# Unity Catalog prefix on every hosted model's `destination_name`.
_HOSTED_NAME_PREFIX = "system.ai."

# A second, INNER prefix that most hosted entities also carry:
# `system.ai.databricks-claude-sonnet-4-5`, `system.ai.databricks-qwen35-122b-a10b`.
# Measured on a live workspace: 38 of 48 distinct hosted `destination_name`s have it
# and 10 do not (`system.ai.gpt-oss-20b`, `system.ai.llama-4-maverick`, ...). It is a
# serving-endpoint naming artefact, not part of the model id — leaving it in emits
# `databricks-qwen35-122b-a10b` as the model, which both reads as a vendor prefix and
# splits one model into two rows in Lago against the live path's own name.
#
# It is NOT safe to strip unconditionally: Databricks also publishes models whose own
# names begin the same way (`databricks-dbrx-instruct`, `databricks-dolly-v2`), and no
# amount of string inspection tells the two apart. `destination_model` does — it was
# the clean name on all 38 prefixed rows — so the prefix comes off only when the two
# columns agree that it is an artefact. Disagreement keeps the raw name: a model
# emitted under a slightly ugly id is recoverable, a silently renamed one is not.
_HOSTED_ENDPOINT_PREFIX = "databricks-"


def _safe_dict(v: Any) -> dict[str, Any]:
    """Coerce a STRUCT/MAP column to a dict, accepting either shape it arrives in.

    The SQL drivers hand back real dicts (pyarrow-backed), but the raw Statement
    Execution API serializes STRUCT and MAP columns as JSON STRINGS — measured:
    `token_details` arrives as '{"cache_read_input_tokens":1812}'. Tolerating both
    means the adapter works whichever access path the caller chose, rather than
    silently reading zeros from a string it never parsed.
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


def _safe_int(v: Any) -> int:
    """Coerce to a non-negative int. Token columns arrive as STRINGS over the REST
    API ("1825") and as NULL on failed calls; both must land on 0 rather than raise."""
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def _safe_str(v: Any) -> str:
    return v if isinstance(v, str) else ""


def _model_and_provider(row: dict[str, Any]) -> tuple[str, str]:
    """Resolve (model, provider) — type-dependent, for the reasons in the module docstring."""
    destination_type = _safe_str(row.get("destination_type"))
    destination_name = _safe_str(row.get("destination_name"))

    if destination_type == _HOSTED_DESTINATION_TYPE:
        # `destination_name` is the stable id here; `destination_model` flips
        # between a slug and a display label for the very same model — measured,
        # `system.ai.gpt-oss-20b` reports both "gpt-oss-20b" and "GPT OSS 20B".
        model = destination_name
        if model.startswith(_HOSTED_NAME_PREFIX):
            model = model[len(_HOSTED_NAME_PREFIX) :]
        if model.startswith(_HOSTED_ENDPOINT_PREFIX):
            shed = model[len(_HOSTED_ENDPOINT_PREFIX) :]
            if shed == _safe_str(row.get("destination_model")):
                model = shed
        # Deliberately "databricks", which matches no vendor in pricing's
        # _VENDOR_MAP — so a price lookup CANNOT hit and emit() falls back to
        # token events (see TOKEN_BILLED_PROVIDERS — no error, since no rate
        # exists to miss), rather than silently
        # pricing a DBU-billed model at some other vendor's rate. OpenRouter does
        # list bare `openai/gpt-oss-20b` etc. at 0.2-0.4x of Databricks' own rate,
        # so an accidental match here would under-bill 2.5-5x.
        return model, "databricks"

    # BYOK: `destination_model` is the clean requested alias, and `api_type` names
    # the native surface the call went through — "anthropic/v1/messages",
    # "openai/v1/chat/completions", "gemini/v1/generateContent". Its leading
    # segment already IS this SDK's provider vocabulary, so no alias table is
    # needed. "unmanaged" (an unrecognized path) yields "unmanaged", which no
    # vendor matches — an honest miss, and those rows carry no usage anyway.
    provider = _safe_str(row.get("api_type")).split("/")[0]
    return _safe_str(row.get("destination_model")), provider


def extract_databricks_log(row: dict[str, Any]) -> CanonicalUsage:
    """Translate one `system.ai_gateway.usage` row → CanonicalUsage.

    Missing/malformed fields degrade to zero/empty rather than raising, matching
    the defensive style of the other adapters — a backfill processing a batch of
    rows must not have one malformed row take down the whole run.
    """
    details = _safe_dict(row.get("token_details"))
    model, provider = _model_and_provider(row)

    return CanonicalUsage(
        input=_safe_int(row.get("input_tokens")),
        output=_safe_int(row.get("output_tokens")),
        cache_read=_safe_int(details.get("cache_read_input_tokens")),
        cache_write=_safe_int(details.get("cache_creation_input_tokens")),
        reasoning=_safe_int(details.get("output_reasoning_tokens")),
        model=model,
        provider=provider,
        api="databricks_gateway",
        extras={
            # `invocation_id` is per individual inference call while `request_id`
            # is per request — one request with a fallback produces several
            # invocations, the same distinction Cloudflare's `step` marks. Keep
            # both; `invocation_id` is the row's natural idempotency key.
            "request_id": row.get("request_id"),
            "invocation_id": row.get("invocation_id"),
            # A THIRD naming variant: `endpoint_name` is the requested form
            # (`databricks-llama-4-maverick`, `system.ai.gemma-3-12b`) where
            # `destination_name` is the resolved entity (`system.ai.gemma-3-12b-it`).
            # Kept for reconciliation; never price off it.
            "endpoint_name": row.get("endpoint_name"),
            "endpoint_id": row.get("endpoint_id"),
            "destination_type": row.get("destination_type"),
            "destination_name": row.get("destination_name"),
            "api_type": row.get("api_type"),
            "status_code": row.get("status_code"),
        },
    )


def resolve_databricks_subscription(row: dict[str, Any]) -> str | None:
    """Pull the Lago subscription id from the caller's `request_tags`.

    Customers set these with the `Databricks-Ai-Gateway-Request-Tags` header (a
    JSON object of string→string), the direct analogue of Cloudflare's
    `cf-aig-metadata`. Note they are also a first-class AGGREGATION DIMENSION on
    `system.ai_gateway.external_model_spend`, so tagging `lago_subscription`
    yields cost already attributed per subscription — no token-share
    apportioning needed for BYOK.

    Returns None if the caller never set `lago_subscription` — untagged calls do
    produce rows, with `request_tags` empty. The caller decides what to do with
    an unattributed row (drop it, route to a default, warn); this function only
    reports whether attribution is present.
    """
    tags = _safe_dict(row.get("request_tags"))
    value = tags.get("lago_subscription")
    return value if isinstance(value, str) and value else None

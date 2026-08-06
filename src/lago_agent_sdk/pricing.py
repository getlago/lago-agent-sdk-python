"""Pricing — optional dollar-cost computation for price mode.

Fetches live, public, no-auth per-token unit prices and computes the cost of a
call as ``Σ(unit_price × token_count) × markup``.

Sources:
  - OpenRouter (``https://openrouter.ai/api/v1/models``) for native providers
    (anthropic / openai / mistral / gemini). Prices are USD per token.
  - AWS Bedrock Price List **Bulk** API (public, no credentials) for Bedrock.
  - Cloudflare's own model catalog (``/accounts/{id}/ai/models/search``) for
    ``workers-ai`` — the actual rate the gateway bills at, not a third party's
    price for hosting the same open-weight model elsewhere (verified live:
    Cloudflare's real charged cost for one call matched this catalog's rate
    exactly; OpenRouter's listing for the same underlying model came out ~3.5x
    lower — a genuinely different price, not just a naming mismatch). Needs
    an account id + API token (Cloudflare's catalog isn't public/no-auth the
    way OpenRouter/AWS are); without both set, this source is simply empty.
  - Mistral's own ``/v1/models`` for *alias resolution*, not pricing directly.
    Mistral has no per-token price table of its own (confirmed: their pricing
    page lists one FAQ example, not a structured/JSON price list) — it genuinely
    has no analogue to Cloudflare's catalog. But a customer request commonly
    uses a moving alias (``mistral-small-latest``) and Mistral's response never
    resolves it (unlike Anthropic/OpenAI, which report the dated snapshot that
    answered) — so the OpenRouter lookup below misses even though OpenRouter
    *does* list the resolved id (e.g. ``mistralai/mistral-small-2603``) with
    real pricing. ``/v1/models`` exposes the resolution directly via each
    model's ``aliases`` array; needs the customer's own Mistral API key.

Design constraints (mirror the queue's non-blocking guarantee):
  - ``lookup()`` is pure in-memory and O(1); it NEVER does network I/O, so the
    customer's LLM call is never blocked on pricing.
  - All HTTP happens in ``maybe_refresh()``, which the EventQueue's background
    worker calls on its flush tick. Tables are swapped atomically under a lock.
  - A cold/missing table returns ``None`` from ``lookup`` → the caller falls back
    to emitting token events (see sdk.emit), so we never silently under-bill.

Money is computed with ``decimal.Decimal`` and floored to 12 decimal places
(ROUND_DOWN) so results are deterministic and match the JS implementation
byte-for-byte.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Any, Protocol

from .canonical import CanonicalUsage

logger = logging.getLogger("lago_agent_sdk.pricing")

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
AWS_PRICING_HOST = "https://pricing.us-east-1.amazonaws.com"
AWS_BEDROCK_REGION_INDEX = f"{AWS_PRICING_HOST}/offers/v1.0/aws/AmazonBedrock/current/region_index.json"
CLOUDFLARE_MODELS_URL_TEMPLATE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search"
MISTRAL_MODELS_URL = "https://api.mistral.ai/v1/models"

# Canonical usage fields we know how to price.
PRICED_FIELDS = ("input", "output", "cache_read", "cache_write", "reasoning")

# Providers whose reported `input` token count ALREADY includes the cached
# (`cache_read`) tokens — i.e. cache_read is a subset of input, not additive.
# For these, the cached portion must be billed at the cache-read rate, not the
# full prompt rate, so compute_cost moves it out of `input`. Anthropic reports
# input EXCLUSIVE of cache (cache_read/cache_write are additive), so it's absent.
_INPUT_INCLUDES_CACHE_READ = frozenset({"openai", "gemini"})

# Providers whose reported `output` token count ALREADY includes the reasoning
# tokens (reasoning is a subset of output). For these, reasoning is billed as
# part of output and must NOT be billed again separately. (Gemini's `thoughts`
# are additive to output, so it's absent here.)
_OUTPUT_INCLUDES_REASONING = frozenset({"openai"})

# Canonical field -> OpenRouter pricing key.
_OPENROUTER_FIELD_MAP = {
    "input": "prompt",
    "output": "completion",
    "cache_read": "input_cache_read",
    "cache_write": "input_cache_write",
    "reasoning": "internal_reasoning",
}

# Our provider name -> OpenRouter vendor prefix.
_VENDOR_MAP = {
    "anthropic": "anthropic",
    "openai": "openai",
    "mistral": "mistralai",
    "gemini": "google",
    "google": "google",
}

# Cloudflare's catalog price unit -> canonical field. Real, surveyed units also
# include "per 1k characters", "per step", "per 512 by 512 tile", "per audio
# minute (websocket)", "per audio minute", "per inference request" — none of
# those are token-based, so they're deliberately absent: a model priced only in
# those units yields a ModelPrice with no input/output/cache_read at all, which
# `compute_cost` already treats as "unpriced field, skip it" — the same safe
# behavior as any other model with no usable price.
_CLOUDFLARE_UNIT_FIELD_MAP = {
    "per M input tokens": "input",
    "per M output tokens": "output",
    "per M cached input tokens": "cache_read",
}

# Bedrock cross-region inference prefix -> a representative AWS region.
_BEDROCK_REGION_PREFIX = {
    "us": "us-east-1",
    "eu": "eu-west-1",
    "apac": "ap-southeast-1",
}

# Vendor words that may lead an AWS Bedrock product's model name.
_BEDROCK_VENDOR_WORDS = {
    "anthropic",
    "mistral",
    "mistralai",
    "ai21",
    "cohere",
    "meta",
    "amazon",
    "stability",
    "stabilityai",
    "google",
}

_SCALE = 12
_Q = Decimal(1).scaleb(-_SCALE)  # Decimal("1E-12")
_VERSION_DATE_SUFFIX = re.compile(r"-(?:\d{8}|v\d+)$")


# ----------------------------------------------------------------------
# Money helpers (kept in lock-step with the JS implementation)
# ----------------------------------------------------------------------
def _parse_price(value: Any) -> Decimal | None:
    """Parse a price into a Decimal floored to 12 dp. None on invalid/negative."""
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if d.is_nan() or d.is_infinite() or d < 0:
        return None
    return d.quantize(_Q, rounding=ROUND_DOWN)


def money_str_to_cents(usd: str) -> str:
    """A money string (already floored to 12dp) → the same amount in cents,
    same floor-and-format conventions as everywhere else."""
    return _fmt_money(Decimal(usd) * 100)


def apply_markup(usd: str, markup: str) -> str:
    """`compute_cost`'s per-field `cost` values are PRE-markup — only the
    summed `total` has markup applied. Splitting a breakdown into one event
    per field (per token_type) needs markup applied to each field individually,
    with the same floor-to-12dp convention as everywhere else, or a markup
    != 1.0 would silently vanish from every per-field/token_type event."""
    return _fmt_money((Decimal(usd) * Decimal(markup)).quantize(_Q, rounding=ROUND_DOWN))


def _fmt_money(d: Decimal) -> str:
    """Floor to 12 dp, render as a plain decimal string, trim trailing zeros."""
    q = d.quantize(_Q, rounding=ROUND_DOWN)
    s = format(q, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _norm(s: str) -> str:
    """Lowercase + unify '.'/'-' so 'claude-opus-4.8' == 'claude-opus-4-8'."""
    return s.lower().replace(".", "-")


def _alnum(s: str) -> str:
    """Lowercase, keep only [a-z0-9] — for cross-format (AWS) matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _strip_version(model: str) -> str:
    """Drop a trailing -YYYYMMDD date or -vN version tag."""
    return _VERSION_DATE_SUFFIX.sub("", model)


# ----------------------------------------------------------------------
# Price tables
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelPrice:
    """Per-token USD prices for one model. None = no price for that field."""

    source: str
    input: Decimal | None = None
    output: Decimal | None = None
    cache_read: Decimal | None = None
    cache_write: Decimal | None = None
    reasoning: Decimal | None = None

    def get(self, field_name: str) -> Decimal | None:
        return getattr(self, field_name, None)


@dataclass
class CostBreakdown:
    """Result of compute_cost — all amounts are money strings ready for an event."""

    total: str  # after-markup total in USD (billable value)
    total_cents: str  # same total in CENTS — Lago dynamic charge `precise_total_amount_cents`
    base: str  # pre-markup
    markup: str
    source: str
    fields: dict[str, dict[str, str]]  # field -> {tokens, unit_price, cost}


def compute_cost(usage: CanonicalUsage, price: ModelPrice, markup: Decimal) -> CostBreakdown:
    """Compute ``Σ(unit_price × count) × markup`` for the priced fields present.

    Fields without a unit price are excluded from the sum (recorded nowhere); a
    call whose only counts are unpriced yields total "0" so it stays accounted
    for.
    """
    provider = (usage.provider or "").lower()
    counts = {f: (getattr(usage, f, 0) or 0) for f in PRICED_FIELDS}
    # Remove double-counting where a provider's `input`/`output` already include
    # a separately-listed subset (see the _INCLUDES_ sets above):
    #   • reasoning ⊆ output  → bill it as output only (drop the separate line).
    #   • cache_read ⊆ input  → bill the cached portion at the cache-read rate,
    #     so subtract it from input (only when a cache_read price exists; with no
    #     cache price the cached tokens stay in input at the prompt rate).
    if provider in _OUTPUT_INCLUDES_REASONING:
        counts["reasoning"] = 0
    if provider in _INPUT_INCLUDES_CACHE_READ and price.get("cache_read") is not None:
        counts["input"] = max(0, counts["input"] - counts["cache_read"])

    base = Decimal(0)
    fields: dict[str, dict[str, str]] = {}
    for f in PRICED_FIELDS:
        count = counts[f]
        if not count:
            continue
        unit = price.get(f)
        if unit is None:
            continue
        cost = unit * count
        base += cost
        fields[f] = {
            "tokens": str(count),
            "unit_price": _fmt_money(unit),
            "cost": _fmt_money(cost),
        }
    return _finalize_breakdown(base, markup, price.source, fields)


def _finalize_breakdown(
    base: Decimal, markup: Decimal, source: str, fields: dict[str, dict[str, str]]
) -> CostBreakdown:
    """Shared tail for `compute_cost`/`compute_precomputed_cost`: floor the
    USD total to 12 dp FIRST, then derive cents from it, so cents ==
    billed-USD × 100 exactly (matches the JS integer-division implementation)."""
    total = (base * markup).quantize(_Q, rounding=ROUND_DOWN)
    return CostBreakdown(
        total=_fmt_money(total),
        total_cents=_fmt_money(total * 100),
        base=_fmt_money(base),
        markup=_fmt_money(markup),
        source=source,
        fields=fields,
    )


def compute_precomputed_cost(usd_cost: Any, markup: Decimal) -> CostBreakdown:
    """Build a CostBreakdown from a cost the CALLER already knows.

    For a gateway that reports its own real, metered price per call (e.g.
    Cloudflare AI Gateway's `cost` field), computing our own per-token estimate
    via the OpenRouter/Bedrock tables would be redundant AND less accurate than
    the number the gateway already gives us. This skips `compute_cost` entirely
    — there's one lump sum, not a per-field breakdown, so `fields` is empty and
    the invalid/negative case floors to 0 the same way `_parse_price` always has,
    rather than raising or silently mis-billing.
    """
    base = _parse_price(usd_cost) or Decimal(0)
    return _finalize_breakdown(base, markup, "precomputed", {})


def coerce_markup(markup: Any) -> tuple[Decimal, bool]:
    """Return (markup_decimal, ok). Falls back to 1.0 when invalid/non-positive."""
    d = _parse_price(markup)
    if d is None or d <= 0:
        return Decimal(1), False
    return d, True


# ----------------------------------------------------------------------
# OpenRouter parsing + matching
# ----------------------------------------------------------------------
def parse_openrouter(data: Any) -> dict[str, Any]:
    """Parse the /models response into {'exact': {...}, 'norm': {...}} tables."""
    exact: dict[str, ModelPrice] = {}
    norm: dict[tuple[str, str], ModelPrice] = {}
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return {"exact": exact, "norm": norm}
    for m in models:
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        pricing = m.get("pricing")
        if not isinstance(mid, str) or not isinstance(pricing, dict):
            continue
        mp = ModelPrice(
            source="openrouter",
            input=_parse_price(pricing.get(_OPENROUTER_FIELD_MAP["input"])),
            output=_parse_price(pricing.get(_OPENROUTER_FIELD_MAP["output"])),
            cache_read=_parse_price(pricing.get(_OPENROUTER_FIELD_MAP["cache_read"])),
            cache_write=_parse_price(pricing.get(_OPENROUTER_FIELD_MAP["cache_write"])),
            reasoning=_parse_price(pricing.get(_OPENROUTER_FIELD_MAP["reasoning"])),
        )
        exact[mid] = mp
        if "/" in mid:
            vendor, _, suffix = mid.partition("/")
            norm[(vendor.lower(), _norm(suffix))] = mp
    return {"exact": exact, "norm": norm}


# A real dated Mistral snapshot ends in a short numeric tag (e.g. "-2603",
# "-2411", "-2508") — never a "-latest"-style moniker. Used to pick the one
# genuine canonical name out of a family that mutually lists each other (see
# parse_mistral_aliases).
_MISTRAL_DATED_ID = re.compile(r"-\d{4,8}$")


def _pick_mistral_canonical(names: list[str]) -> str:
    """Prefer a dated snapshot id (what OpenRouter actually lists models
    under) over a "-latest"-style moniker. Falls back to shortest-then-
    alphabetical so the choice is always deterministic even with no dated
    candidate in the group."""
    dated = [n for n in names if _MISTRAL_DATED_ID.search(n)]
    pool = dated or names
    return sorted(pool, key=lambda n: (len(n), n))[0]


def parse_mistral_aliases(data: Any) -> dict[str, str]:
    """Parse Mistral's `/v1/models` response into {alias: canonical_id}.

    Naively mapping "each name in this entry's `aliases` -> this entry's
    `id`" is wrong: Mistral's real response lists EVERY name in a family as
    its own top-level entry, each one's `aliases` pointing at the others —
    e.g. `id="mistral-small-2603"`, `id="mistral-small-latest"`, AND
    `id="magistral-small-latest"` each appear separately, each listing the
    other two as `aliases`. A directional last-write-wins map is then
    order-dependent and can resolve an alias to ANOTHER alias instead of the
    real dated snapshot (confirmed live: this resolved
    "mistral-small-latest" -> "magistral-small-latest", which OpenRouter
    doesn't list, instead of -> "mistral-small-2603", which it does).

    Union-find instead: treat a model's id + its aliases as one connected
    group regardless of which entry mentions which, then pick a single
    canonical name per group (see `_pick_mistral_canonical`) and map every
    other member of the group to it.
    """
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return {}

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    names: set[str] = set()
    for m in models:
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        if not isinstance(mid, str) or not mid:
            continue
        parent.setdefault(mid, mid)
        names.add(mid)
        for alias in m.get("aliases") or []:
            if isinstance(alias, str) and alias:
                parent.setdefault(alias, alias)
                names.add(alias)
                union(mid, alias)

    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)

    result: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue  # no aliasing at all — nothing to resolve
        canonical = _pick_mistral_canonical(members)
        for name in members:
            if name != canonical:
                result[name] = canonical
    return result


def lookup_openrouter(table: dict[str, Any], provider: str, model: str) -> ModelPrice | None:
    """Match (provider, model) to an OpenRouter price. Conservative: vendor-gated."""
    vendor = _VENDOR_MAP.get((provider or "").lower(), (provider or "").lower())
    exact: dict[str, ModelPrice] = table.get("exact", {})
    norm: dict[tuple[str, str], ModelPrice] = table.get("norm", {})
    # 1. exact id
    hit = exact.get(f"{vendor}/{model}")
    if hit is not None:
        return hit
    # 2. normalized suffix (. <-> -)
    hit = norm.get((vendor, _norm(model)))
    if hit is not None:
        return hit
    # 3. date/version-stripped, normalized
    hit = norm.get((vendor, _norm(_strip_version(model))))
    if hit is not None:
        return hit
    return None


# ----------------------------------------------------------------------
# Cloudflare Workers AI parsing + matching
#
# Unlike OpenRouter/Bedrock, this is the ACTUAL rate the gateway bills at — not
# a third party's price for hosting the same open-weight model elsewhere, which
# can (and does) differ meaningfully. Model strings (e.g.
# "@cf/meta/llama-3.3-70b-instruct-fp8-fast") are already exact and
# self-contained; no vendor-prefix mapping is needed the way OpenRouter needs
# one to disambiguate "anthropic" -> "anthropic" vs "mistral" -> "mistralai".
# ----------------------------------------------------------------------
def parse_cloudflare_workers_ai(models: Any) -> dict[str, ModelPrice]:
    """Parse `/ai/models/search` results into {model_name: ModelPrice}.

    A model with no `price` property at all, or whose price entries are all
    non-token units (per-image, per-audio-minute, ...), is simply absent from
    the table — `lookup` then returns None, same as any other priced-nowhere
    model, and the caller safely falls back to token events.
    """
    table: dict[str, ModelPrice] = {}
    if not isinstance(models, list):
        return table
    for m in models:
        if not isinstance(m, dict):
            continue
        name = m.get("name")
        if not isinstance(name, str) or not name:
            continue
        price_prop = next(
            (p for p in m.get("properties", []) if isinstance(p, dict) and p.get("property_id") == "price"),
            None,
        )
        if not isinstance(price_prop, dict):
            continue
        entries = price_prop.get("value")
        if not isinstance(entries, list):
            continue
        fields: dict[str, Decimal] = {}
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("currency") != "USD":
                continue
            field = _CLOUDFLARE_UNIT_FIELD_MAP.get(str(entry.get("unit", "")))
            if field is None:
                continue
            per_million = _parse_price(entry.get("price"))
            if per_million is None:
                continue
            fields[field] = (per_million / Decimal(1_000_000)).quantize(_Q, rounding=ROUND_DOWN)
        if fields:
            table[name] = ModelPrice(source="cloudflare_workers_ai", **fields)
    return table


def lookup_cloudflare_workers_ai(table: dict[str, ModelPrice], model: str) -> ModelPrice | None:
    """Exact match first; a version-suffix fallback covers the same drift we've
    seen in practice — e.g. a live response naming a model
    "...instruct-v2" when the catalog itself only lists "...instruct"."""
    hit = table.get(model)
    if hit is not None:
        return hit
    return table.get(_strip_version(model))


# ----------------------------------------------------------------------
# Bedrock parsing + matching
#
# The AWS Price List offer schema is large and its attribute keys vary by
# product; this parser is deliberately defensive and is validated end-to-end by
# the env-gated live test. A miss returns None → safe token fallback.
# ----------------------------------------------------------------------
def parse_bedrock_region(model: str, default_region: str) -> str:
    head = model.split(".", 1)[0].lower() if "." in model else ""
    return _BEDROCK_REGION_PREFIX.get(head, default_region)


def bedrock_model_key(model: str) -> str:
    """Reduce a Bedrock model id to the alnum key used to index AWS prices.

    e.g. 'eu.anthropic.claude-sonnet-4-6' -> 'claudesonnet46';
         'anthropic.claude-haiku-4-5-20251001-v1:0' -> 'claudehaiku45';
         'mistral.mixtral-8x7b-instruct-v0:1' -> 'mixtral8x7binstruct'.
    """
    parts = model.split(".")
    if parts and parts[0].lower() in _BEDROCK_REGION_PREFIX:
        parts = parts[1:]
    if len(parts) > 1:
        model_part = ".".join(parts[1:])  # drop vendor
    else:
        model_part = parts[0] if parts else ""
    model_part = re.sub(r":\d+$", "", model_part)  # ':0'
    model_part = re.sub(r"-v\d+$", "", model_part)  # '-v1'
    model_part = _strip_version(model_part)
    return _alnum(model_part)


def _aws_model_keys(name: str) -> list[str]:
    """Candidate alnum keys for an AWS model name (with/without vendor prefix)."""
    base = _strip_version(_norm(name))
    keys = {_alnum(base)}
    words = name.split()
    if words and words[0].lower() in _BEDROCK_VENDOR_WORDS:
        keys.add(_alnum(_strip_version(_norm(" ".join(words[1:])))))
    return [k for k in keys if k]


def _usd_per_token(term: Any) -> Decimal | None:
    """Extract a USD-per-token price from a terms.OnDemand[sku] entry."""
    if not isinstance(term, dict):
        return None
    for offer in term.values():
        dims = offer.get("priceDimensions") if isinstance(offer, dict) else None
        if not isinstance(dims, dict):
            continue
        for dim in dims.values():
            if not isinstance(dim, dict):
                continue
            ppu = dim.get("pricePerUnit")
            usd = ppu.get("USD") if isinstance(ppu, dict) else None
            price = _parse_price(usd)
            if price is None:
                continue
            unit = str(dim.get("unit", "")).lower()
            # AWS sometimes prices per 1K tokens.
            if "1k" in unit or "1000" in unit or "thousand" in unit:
                price = (price / Decimal(1000)).quantize(_Q, rounding=ROUND_DOWN)
            return price
    return None


def parse_bedrock_offer(offer: Any, region: str) -> dict[str, ModelPrice]:
    """Build {alnum_model_key: ModelPrice(input/output)} from an AWS offer file."""
    if not isinstance(offer, dict):
        return {}
    products = offer.get("products")
    terms = offer.get("terms")
    on_demand = terms.get("OnDemand") if isinstance(terms, dict) else None
    if not isinstance(products, dict) or not isinstance(on_demand, dict):
        return {}

    table: dict[str, dict[str, Decimal]] = {}
    for sku, product in products.items():
        if not isinstance(product, dict):
            continue
        attrs = product.get("attributes")
        if not isinstance(attrs, dict):
            continue
        name = attrs.get("model") or attrs.get("titleModelId") or attrs.get("modelName")
        if not isinstance(name, str) or not name:
            continue
        direction = _bedrock_direction(attrs)
        if direction is None:
            continue
        price = _usd_per_token(on_demand.get(sku))
        if price is None:
            continue
        for key in _aws_model_keys(name):
            table.setdefault(key, {})[direction] = price

    return {
        key: ModelPrice(source="aws_bedrock", input=v.get("input"), output=v.get("output"))
        for key, v in table.items()
    }


def _bedrock_direction(attrs: dict[str, Any]) -> str | None:
    """Classify a Bedrock product as standard on-demand 'input'/'output' tokens.

    Prefers the explicit ``inferenceType`` ("Input tokens" / "Output tokens").
    Rejects tiered variants ("... priority/flex/batch") so we capture the
    standard on-demand price, not a discounted/surge tier. Falls back to a
    usagetype scan only when inferenceType is absent.
    """
    it = str(attrs.get("inferenceType", "")).strip().lower()
    if it == "input tokens":
        return "input"
    if it == "output tokens":
        return "output"
    if it:
        # Present but a tier variant (priority/flex/batch) or non-token → skip.
        return None
    # inferenceType absent: fall back to usagetype, excluding batch/non-token.
    blob = " ".join(str(attrs.get(k, "")) for k in ("usagetype", "operation", "feature")).lower()
    if "batch" in blob or "token" not in blob:
        return None
    if "input" in blob:
        return "input"
    if "output" in blob:
        return "output"
    return None


def lookup_bedrock(region_table: dict[str, ModelPrice], model: str) -> ModelPrice | None:
    return region_table.get(bedrock_model_key(model))


# ----------------------------------------------------------------------
# Fetcher (real HTTP; injectable for tests)
# ----------------------------------------------------------------------
class PricingFetcher(Protocol):
    def fetch_openrouter(self) -> dict[str, Any]: ...
    def fetch_bedrock(self, region: str) -> dict[str, ModelPrice]: ...
    def fetch_cloudflare_workers_ai(self) -> dict[str, ModelPrice]: ...
    def fetch_mistral_aliases(self, api_key: str | None = None) -> dict[str, str]: ...


class HttpPricingFetcher:
    """Default fetcher using ``requests`` (already a core dependency).

    ``cloudflare_account_id``/``cloudflare_api_token``: unlike OpenRouter/AWS,
    Cloudflare's model catalog is account-scoped and needs auth — there's no
    public, no-credentials equivalent. Without both set,
    ``fetch_cloudflare_workers_ai`` returns an empty table rather than raising,
    so Workers AI pricing is simply unavailable (safe token-event fallback)
    instead of breaking price mode for every other provider.

    ``mistral_api_key``: same story — Mistral's ``/v1/models`` needs the
    customer's own key. Without it, ``fetch_mistral_aliases`` returns an
    empty map, so alias resolution is simply skipped and lookups fall back to
    whatever the request already spelled out (safe miss, not a break).
    """

    def __init__(
        self,
        timeout: float = 10.0,
        cloudflare_account_id: str | None = None,
        cloudflare_api_token: str | None = None,
        mistral_api_key: str | None = None,
    ) -> None:
        self._timeout = timeout
        self._cf_account_id = cloudflare_account_id
        self._cf_api_token = cloudflare_api_token
        self._mistral_api_key = mistral_api_key

    def fetch_openrouter(self) -> dict[str, Any]:
        import requests

        resp = requests.get(OPENROUTER_URL, timeout=self._timeout)
        resp.raise_for_status()
        return parse_openrouter(resp.json())

    def fetch_bedrock(self, region: str) -> dict[str, ModelPrice]:
        import requests

        idx = requests.get(AWS_BEDROCK_REGION_INDEX, timeout=self._timeout)
        idx.raise_for_status()
        regions = idx.json().get("regions", {})
        entry = regions.get(region)
        if not isinstance(entry, dict) or not entry.get("currentVersionUrl"):
            return {}
        offer = requests.get(AWS_PRICING_HOST + entry["currentVersionUrl"], timeout=self._timeout)
        offer.raise_for_status()
        return parse_bedrock_offer(offer.json(), region)

    def fetch_cloudflare_workers_ai(self) -> dict[str, ModelPrice]:
        import requests

        if not self._cf_account_id or not self._cf_api_token:
            return {}
        url = CLOUDFLARE_MODELS_URL_TEMPLATE.format(account_id=self._cf_account_id)
        headers = {"Authorization": f"Bearer {self._cf_api_token}"}
        models: list[Any] = []
        page = 1
        while True:
            resp = requests.get(
                url, headers=headers, params={"per_page": 50, "page": page}, timeout=self._timeout
            )
            resp.raise_for_status()
            body = resp.json()
            batch = body.get("result") or []
            models.extend(batch)
            total = body.get("result_info", {}).get("total_count", len(models))
            if len(batch) < 50 or len(models) >= total:
                break
            page += 1
        return parse_cloudflare_workers_ai(models)

    def fetch_mistral_aliases(self, api_key: str | None = None) -> dict[str, str]:
        import requests

        # An explicitly configured key (LagoConfig.mistral_api_key) always
        # wins over one learned from a wrapped client — a deliberate config
        # value shouldn't be silently shadowed by an auto-detected one.
        key = self._mistral_api_key or api_key
        if not key:
            return {}
        headers = {"Authorization": f"Bearer {key}"}
        resp = requests.get(MISTRAL_MODELS_URL, headers=headers, timeout=self._timeout)
        resp.raise_for_status()
        return parse_mistral_aliases(resp.json())


# ----------------------------------------------------------------------
# PricingProvider — cache + background refresh + non-blocking lookup
# ----------------------------------------------------------------------
class PricingProvider:
    def __init__(
        self,
        fetcher: PricingFetcher | None = None,
        ttl_seconds: float = 3600.0,
        default_region: str = "us-east-1",
        on_error: Callable[[Exception, str], None] | None = None,
        cloudflare_account_id: str | None = None,
        cloudflare_api_token: str | None = None,
        mistral_api_key: str | None = None,
    ) -> None:
        self._fetcher: PricingFetcher = fetcher or HttpPricingFetcher(
            cloudflare_account_id=cloudflare_account_id,
            cloudflare_api_token=cloudflare_api_token,
            mistral_api_key=mistral_api_key,
        )
        self._ttl = ttl_seconds
        self._default_region = default_region
        self._on_error = on_error
        self._lock = threading.Lock()
        self._pid = os.getpid()
        self._openrouter: dict[str, Any] | None = None
        self._openrouter_fetched = 0.0
        # Not stale by default: token-mode SDKs never trigger a pricing fetch.
        # A price-mode lookup flags the relevant source stale on first use.
        self._openrouter_stale = False
        self._bedrock: dict[str, dict[str, ModelPrice]] = {}
        self._bedrock_fetched: dict[str, float] = {}
        self._bedrock_stale: set[str] = set()
        self._cloudflare_workers_ai: dict[str, ModelPrice] | None = None
        self._cloudflare_fetched = 0.0
        self._cloudflare_stale = False
        self._mistral_aliases: dict[str, str] | None = None
        self._mistral_fetched = 0.0
        self._mistral_stale = False
        # Learned from a wrapped Mistral client (see LagoSDK._auto_prime_pricing_for),
        # not configured — the customer's own client already carries this key
        # for making real calls, so alias resolution can reuse it without
        # ever requiring a separate LagoConfig.mistral_api_key.
        self._mistral_api_key_override: str | None = None
        self._refreshing: set[str] = set()

    def _heal_fork(self) -> None:
        """Self-heal after a fork: a lock copied from the parent may be held by a
        thread that doesn't exist in the child. Detect a PID change and replace
        the lock + mark tables stale so the child's queue thread refetches. Cheap
        PID read on the hot path; avoids os.register_at_fork (whose extra
        fork-time work trips macOS's objc fork-safety abort)."""
        if os.getpid() != self._pid:
            self._lock = threading.Lock()
            self._pid = os.getpid()
            self._openrouter_stale = self._openrouter is not None or self._openrouter_stale
            self._bedrock_stale = set(self._bedrock.keys())
            self._cloudflare_stale = self._cloudflare_workers_ai is not None or self._cloudflare_stale
            self._mistral_stale = self._mistral_aliases is not None or self._mistral_stale
            self._refreshing = set()

    def prime(self, providers: Iterable[str] = ()) -> None:
        """Flag OpenRouter for an eager background warm (used when price mode
        is the global default) to shrink the cold-start window.

        Deliberately does NOT also eagerly warm Cloudflare Workers AI or
        Mistral alias resolution by default — both are credential-gated and
        provider-specific; most price-mode customers never touch Workers AI
        or Mistral at all, and eagerly hitting either's API at construction
        time regardless of actual usage is real, unnecessary work (an extra
        network round-trip per SDK instance, every TTL cycle, for a provider
        that may never be called). Instead they stay purely reactive: the
        first real `lookup()` for that provider flags it stale (see below),
        `maybe_refresh()` fetches it on the queue's very next tick, and every
        call after that — even the one a second later — hits the cache, with
        zero further network calls until the TTL expires. Only that first
        call for a given provider can race a cold cache; every provider that
        session never calls costs nothing.

        Pass `providers=["mistral"]` and/or `["workers-ai"]` when you already
        know, in advance, which of these two you're about to call this
        session — this eagerly warms exactly that source too, so even ITS
        first call prices correctly instead of paying the one-time lazy
        cold-start cost. Unknown provider names are silently ignored (no
        source is warmed) rather than raising, since this is a hint, not a
        contract."""
        with self._lock:
            self._openrouter_stale = True
            for p in providers:
                key = (p or "").lower()
                if key == "workers-ai":
                    self._cloudflare_stale = True
                elif key == "mistral":
                    self._mistral_stale = True

    def learn_mistral_api_key(self, api_key: str) -> None:
        """Adopt a Mistral API key discovered from a wrapped client, so
        alias resolution can run without ever requiring the customer to
        also declare it in `LagoConfig` — their Mistral client already
        carries the exact credential needed. Pure in-memory, no I/O. A key
        explicitly set via `LagoConfig.mistral_api_key` always wins over one
        learned this way (see `HttpPricingFetcher.fetch_mistral_aliases`);
        this only fills the gap when no explicit key was configured."""
        if not api_key:
            return
        with self._lock:
            if not self._mistral_api_key_override:
                self._mistral_api_key_override = api_key

    # ---- non-blocking lookup (customer thread) ----
    def lookup(self, provider: str, model: str, api: str) -> ModelPrice | None:
        try:
            self._heal_fork()
            if (api or "").startswith("bedrock"):
                region = parse_bedrock_region(model, self._default_region)
                with self._lock:
                    table = self._bedrock.get(region)
                    fresh = (
                        table is not None
                        and (time.time() - self._bedrock_fetched.get(region, 0.0)) < self._ttl
                    )
                    if not fresh:
                        self._bedrock_stale.add(region)
                return lookup_bedrock(table, model) if table is not None else None
            if (provider or "").lower() == "workers-ai":
                with self._lock:
                    table_cf = self._cloudflare_workers_ai
                    fresh_cf = table_cf is not None and (time.time() - self._cloudflare_fetched) < self._ttl
                    if not fresh_cf:
                        self._cloudflare_stale = True
                return lookup_cloudflare_workers_ai(table_cf, model) if table_cf is not None else None
            resolved_model = model
            is_mistral = (provider or "").lower() == "mistral"
            with self._lock:
                if is_mistral:
                    aliases = self._mistral_aliases
                    fresh_m = aliases is not None and (time.time() - self._mistral_fetched) < self._ttl
                    if not fresh_m:
                        self._mistral_stale = True
                    # Cold/miss: resolved_model stays the alias as-requested,
                    # and the OpenRouter lookup below misses safely, same as
                    # before this resolution step existed — never worse than
                    # the old behavior, only better once the table is warm.
                    if aliases:
                        resolved_model = aliases.get(model, model)
                table_or = self._openrouter
                fresh = table_or is not None and (time.time() - self._openrouter_fetched) < self._ttl
                if not fresh:
                    self._openrouter_stale = True
            return lookup_openrouter(table_or, provider, resolved_model) if table_or is not None else None
        except Exception:  # noqa: BLE001 — lookup must never raise
            return None

    # ---- background refresh (queue worker thread) ----
    def maybe_refresh(self) -> None:
        self._heal_fork()
        # Lock-free fast path: when nothing is stale (the common case, and always
        # in token mode), do no work at all — not even acquire the lock. This
        # keeps the queue's background tick essentially free and avoids extra
        # cross-thread lock churn. The reads are racy but harmless: a missed flag
        # just defers a refresh by one tick.
        if (
            not self._openrouter_stale
            and not self._bedrock_stale
            and not self._cloudflare_stale
            and not self._mistral_stale
        ):
            return
        with self._lock:
            do_openrouter = self._openrouter_stale and "openrouter" not in self._refreshing
            if do_openrouter:
                self._refreshing.add("openrouter")
            do_cloudflare = self._cloudflare_stale and "cloudflare_workers_ai" not in self._refreshing
            if do_cloudflare:
                self._refreshing.add("cloudflare_workers_ai")
            do_mistral = self._mistral_stale and "mistral_aliases" not in self._refreshing
            if do_mistral:
                self._refreshing.add("mistral_aliases")
            regions = [r for r in self._bedrock_stale if f"bedrock:{r}" not in self._refreshing]
            for r in regions:
                self._refreshing.add(f"bedrock:{r}")

        if do_openrouter:
            try:
                table = self._fetcher.fetch_openrouter()
                with self._lock:
                    self._openrouter = table
                    self._openrouter_fetched = time.time()
                    self._openrouter_stale = False
            except Exception as exc:  # noqa: BLE001
                self._report(exc, "pricing.fetch_openrouter")
            finally:
                with self._lock:
                    self._refreshing.discard("openrouter")

        if do_cloudflare:
            try:
                table_cf = self._fetcher.fetch_cloudflare_workers_ai()
                with self._lock:
                    self._cloudflare_workers_ai = table_cf
                    self._cloudflare_fetched = time.time()
                    self._cloudflare_stale = False
            except Exception as exc:  # noqa: BLE001
                self._report(exc, "pricing.fetch_cloudflare_workers_ai")
            finally:
                with self._lock:
                    self._refreshing.discard("cloudflare_workers_ai")

        if do_mistral:
            try:
                with self._lock:
                    learned_key = self._mistral_api_key_override
                aliases = self._fetcher.fetch_mistral_aliases(learned_key)
                with self._lock:
                    self._mistral_aliases = aliases
                    self._mistral_fetched = time.time()
                    self._mistral_stale = False
            except Exception as exc:  # noqa: BLE001
                self._report(exc, "pricing.fetch_mistral_aliases")
            finally:
                with self._lock:
                    self._refreshing.discard("mistral_aliases")

        for r in regions:
            try:
                table = self._fetcher.fetch_bedrock(r)
                with self._lock:
                    self._bedrock[r] = table
                    self._bedrock_fetched[r] = time.time()
                    self._bedrock_stale.discard(r)
            except Exception as exc:  # noqa: BLE001
                self._report(exc, "pricing.fetch_bedrock")
            finally:
                with self._lock:
                    self._refreshing.discard(f"bedrock:{r}")

    def _report(self, exc: Exception, where: str) -> None:
        if self._on_error:
            try:
                self._on_error(exc, where)
            except Exception:  # noqa: BLE001
                pass
        logger.warning("lago %s failed: %s", where, exc)

"""Pricing — optional dollar-cost computation for price mode.

Fetches live, public, no-auth per-token unit prices and computes the cost of a
call as ``Σ(unit_price × token_count) × markup``.

Sources:
  - OpenRouter (``https://openrouter.ai/api/v1/models``) for native providers
    (anthropic / openai / mistral / gemini). Prices are USD per token.
  - AWS Bedrock Price List **Bulk** API (public, no credentials) for Bedrock.

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
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from typing import Any, Protocol

from .canonical import CanonicalUsage

logger = logging.getLogger("lago_agent_sdk.pricing")

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
AWS_PRICING_HOST = "https://pricing.us-east-1.amazonaws.com"
AWS_BEDROCK_REGION_INDEX = f"{AWS_PRICING_HOST}/offers/v1.0/aws/AmazonBedrock/current/region_index.json"

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
    # Floor the USD total to 12 dp FIRST, then derive cents from it, so cents ==
    # billed-USD × 100 exactly (matches the JS integer-division implementation).
    total = (base * markup).quantize(_Q, rounding=ROUND_DOWN)
    return CostBreakdown(
        total=_fmt_money(total),
        total_cents=_fmt_money(total * 100),
        base=_fmt_money(base),
        markup=_fmt_money(markup),
        source=price.source,
        fields=fields,
    )


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


class HttpPricingFetcher:
    """Default fetcher using ``requests`` (already a core dependency)."""

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout

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
    ) -> None:
        self._fetcher: PricingFetcher = fetcher or HttpPricingFetcher()
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
            self._refreshing = set()

    def prime(self) -> None:
        """Flag the OpenRouter table for an eager background warm (used when
        price mode is the global default) to shrink the cold-start window."""
        with self._lock:
            self._openrouter_stale = True

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
            with self._lock:
                table_or = self._openrouter
                fresh = table_or is not None and (time.time() - self._openrouter_fetched) < self._ttl
                if not fresh:
                    self._openrouter_stale = True
            return lookup_openrouter(table_or, provider, model) if table_or is not None else None
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
        if not self._openrouter_stale and not self._bedrock_stale:
            return
        with self._lock:
            do_openrouter = self._openrouter_stale and "openrouter" not in self._refreshing
            if do_openrouter:
                self._refreshing.add("openrouter")
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

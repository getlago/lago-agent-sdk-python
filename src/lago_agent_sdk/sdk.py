"""LagoSDK — primary entrypoint."""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from collections.abc import Iterable
from typing import Any

from .canonical import CanonicalUsage
from .config import LagoConfig
from .detector import detect_client_kind
from .exceptions import PricingUnavailableError, UnknownClientError
from .lago_client import LagoClient
from .pricing import (
    CostBreakdown,
    PricingProvider,
    apply_markup,
    coerce_markup,
    compute_cost,
    compute_precomputed_cost,
    money_str_to_cents,
)
from .queue import EventQueue

logger = logging.getLogger("lago_agent_sdk")

_subscription_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "lago_subscription", default=None
)


class LagoSDK:
    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.getlago.com/api/v1",
        default_subscription_id: str | None = None,
        config: LagoConfig | None = None,
    ) -> None:
        self.config = config or LagoConfig(
            api_key=api_key,
            api_url=api_url,
            default_subscription_id=default_subscription_id,
        )
        # explicit args win over `config`
        self.config.api_key = api_key or self.config.api_key
        if api_url:
            self.config.api_url = api_url
        if default_subscription_id is not None:
            self.config.default_subscription_id = default_subscription_id

        self._lago_client = LagoClient(
            api_key=self.config.api_key,
            api_url=self.config.api_url,
            timeout=self.config.request_timeout_seconds,
            verify_ssl=self.config.verify_ssl,
        )
        # Pricing provider (price mode). Default does no network until a
        # price-mode lookup flags a source stale; refreshes run on the queue
        # thread, never on the customer's call.
        self._pricing: PricingProvider = self.config.pricing_provider or PricingProvider(
            ttl_seconds=self.config.pricing_ttl_seconds,
            default_region=self.config.bedrock_default_region,
            on_error=self.config.on_error,
            cloudflare_account_id=self.config.cloudflare_account_id,
            cloudflare_api_token=self.config.cloudflare_api_token,
            mistral_api_key=self.config.mistral_api_key,
        )
        if self.config.pricing_mode == "price":
            self._pricing.prime()  # eager warm when price mode is the global default
        self._queue = EventQueue(
            sender=self._lago_client.send_batch,
            flush_interval=self.config.flush_interval_seconds,
            max_batch_size=self.config.max_batch_size,
            max_buffer_size=self.config.max_buffer_size,
            max_retry_seconds=self.config.max_retry_seconds,
            on_error=self.config.on_error,
            pricing=self._pricing,
        )

    # ------------------------------------------------------------------
    # Subscription resolution: per-call > contextvar > default
    # ------------------------------------------------------------------
    def set_subscription(self, subscription_id: str) -> contextvars.Token[str | None]:
        return _subscription_var.set(subscription_id)

    def reset_subscription(self, token: contextvars.Token[str | None]) -> None:
        _subscription_var.reset(token)

    def _resolve_subscription(self, override: str | None) -> str | None:
        return override or _subscription_var.get() or self.config.default_subscription_id

    def _auto_prime_pricing_for(self, kind: str, client: Any) -> None:
        """Best-effort, automatic, non-blocking warm-up for the two
        credential-gated pricing sources — triggered by `wrap()` itself,
        which the customer already calls, so there's no separate function to
        remember. `wrap()` almost always happens some real time before the
        customer's first actual completion call (building the prompt,
        setting up messages, etc.), so kicking the fetch off here — instead
        of waiting for that first completion call to flag it stale — gives
        it a real head start: often enough to be warm before that first call
        even lands, not just for every call after it.

        Only runs when `pricing_mode == "price"` is the global default —
        otherwise there's nothing to warm for. `prime()`/`wake()` are both
        pure in-memory (no I/O on this thread); the actual HTTP fetch still
        happens on the queue's background thread, never here.
        """
        if self.config.pricing_mode != "price":
            return
        provider: str | None = None
        if kind == "mistral":
            # The client being wrapped already carries the exact Mistral API
            # key needed to call Mistral's own /v1/models for alias
            # resolution — no separate LagoConfig.mistral_api_key required.
            key = self._extract_mistral_api_key(client)
            if key:
                self._pricing.learn_mistral_api_key(key)
            provider = "mistral"
        elif kind == "openai":
            # A generic OpenAI-shaped client can point at real OpenAI OR, via
            # Cloudflare's `.../compat` endpoint, at Workers AI — the client
            # kind alone can't tell them apart. `base_url` is the one signal
            # that can, without waiting for a response to resolve a model
            # string. Defensive: some client variants may not expose it.
            try:
                base_url = str(getattr(client, "base_url", "") or "")
            except Exception:  # noqa: BLE001
                base_url = ""
            if "gateway.ai.cloudflare.com" in base_url:
                provider = "workers-ai"
        if provider:
            self._pricing.prime([provider])
            self._queue.wake()

    @staticmethod
    def _extract_mistral_api_key(client: Any) -> str | None:
        """The mistralai SDK stores the constructor's `api_key=...` at
        `client.sdk_configuration.security.api_key` (verified against a real
        client instance). Defensive: an SDK version change to this internal
        path degrades to "no key learned" (falls back to
        `LagoConfig.mistral_api_key` if set, else the existing lazy-miss
        behavior) rather than raising."""
        try:
            key = client.sdk_configuration.security.api_key
        except Exception:  # noqa: BLE001
            return None
        return key if isinstance(key, str) and key else None

    # ------------------------------------------------------------------
    # Wrap()
    # ------------------------------------------------------------------
    def wrap(
        self, client: Any, dimensions: dict[str, Any] | None = None, subscription: str | None = None
    ) -> Any:
        kind = detect_client_kind(client)
        self._auto_prime_pricing_for(kind, client)
        if kind == "bedrock":
            from .wrappers.boto3_bedrock import wrap_boto3_bedrock_client

            return wrap_boto3_bedrock_client(self, client, dimensions=dimensions, subscription=subscription)
        if kind == "mistral":
            from .wrappers.mistral import wrap_mistral_client

            return wrap_mistral_client(self, client, dimensions=dimensions, subscription=subscription)
        if kind == "anthropic":
            from .wrappers.anthropic import wrap_anthropic_client

            return wrap_anthropic_client(self, client, dimensions=dimensions, subscription=subscription)
        if kind == "openai":
            from .wrappers.openai import wrap_openai_client

            return wrap_openai_client(self, client, dimensions=dimensions, subscription=subscription)
        if kind == "gemini":
            from .wrappers.gemini import wrap_gemini_client

            return wrap_gemini_client(self, client, dimensions=dimensions, subscription=subscription)
        if kind == "gemini_legacy":
            raise UnknownClientError(
                "The legacy google-generativeai SDK "
                "(`import google.generativeai; genai.GenerativeModel(...)`) is not "
                "supported — its surface differs from the unified SDK and cannot be "
                "instrumented. Migrate to google-genai: `pip install google-genai`, "
                "then `from google import genai; client = genai.Client(...)` and wrap "
                "the Client. See https://ai.google.dev/gemini-api/docs/migrate."
            )
        if kind == "unknown":
            raise UnknownClientError(
                f"Unknown client passed to wrap(): {type(client).__module__}.{type(client).__name__}. "
                "Supported: boto3 bedrock-runtime, mistralai.client.Mistral, "
                "anthropic.Anthropic / AsyncAnthropic, openai.OpenAI / AsyncOpenAI, "
                "google.genai.Client."
            )
        raise UnknownClientError(
            f"Client kind '{kind}' is not yet supported. "
            "Implemented: 'bedrock', 'mistral', 'anthropic', 'openai', 'gemini'."
        )

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------
    def emit(
        self,
        usage: CanonicalUsage,
        subscription: str | None = None,
        dimensions: dict[str, Any] | None = None,
        mode: str | None = None,
        markup: float | None = None,
        usd_cost: float | None = None,
        event_id: str | None = None,
    ) -> None:
        """Emit usage to Lago.

        In ``tokens`` mode (default), pushes one event per nonzero token field.
        In ``price`` mode, pushes a single dollar-cost event; if no price is
        available it falls back to token events and reports via on_error.
        Precedence for mode/markup: per-call arg > config default.

        ``usd_cost``: skip this SDK's own OpenRouter/Bedrock price lookup and
        bill this exact amount instead. For a gateway that reports its own
        real, metered cost per call (e.g. Cloudflare AI Gateway's `cost`
        field on a log entry), that number is more accurate than anything we'd
        compute ourselves — this is the connector's one-call entrypoint rather
        than hand-building a `precise_total_amount_cents` event. Only consulted
        when the effective mode is "price"; ignored in token mode.

        ``event_id``: use this as Lago's idempotency key (`transaction_id`)
        instead of a random UUID — pass the source log entry's own id when
        replaying/backfilling from a gateway's logs, so re-running against the
        same window never double-bills. A live, one-shot call has no natural
        id to reuse and should leave this as None.

        Both multi-event paths suffix per field so they don't collide with each
        other, and they use DIFFERENT namespaces so they can't collide across
        modes either:

          * token events      ``f"{event_id}_tok_{field_name}"``
          * split cost events ``f"{event_id}_cost_{field_name}"``
          * single cost event ``event_id`` (one event, nothing to disambiguate)

        The namespaces are load-bearing. Both paths are reachable for the SAME
        `event_id`: a price lookup that misses falls back to token events, and
        the same window re-run once the table is warm takes the cost path. Under
        one shared namespace the second run re-sent `{event_id}_input` under a
        different metric code, Lago rejected it as a duplicate — and because
        `/events/batch` is all-or-nothing, that rejection failed every other
        event in the batch too. Net effect: the dollar amounts for that window
        were never billed, only the raw token counts, and nothing surfaced it.
        """
        try:
            sub = self._resolve_subscription(subscription)
            if not sub:
                logger.error(
                    "lago: dropping events for model=%s — no resolvable subscription",
                    usage.model,
                )
                return

            effective_mode = mode or self.config.pricing_mode
            if effective_mode != "price":
                self._emit_token_events(usage, sub, dimensions, event_id)
                return

            markup_value, ok = coerce_markup(markup if markup is not None else self.config.markup)
            if not ok:
                self._report_error(
                    ValueError(
                        f"invalid markup {markup if markup is not None else self.config.markup!r}; using 1.0"
                    ),
                    "pricing",
                )

            if usd_cost is not None:
                breakdown = compute_precomputed_cost(usd_cost, markup_value)
            else:
                price = self._pricing.lookup(usage.provider, usage.model, usage.api)
                if price is None:
                    # Don't silently under-bill: fall back to token events + report.
                    self._report_error(
                        PricingUnavailableError(usage.provider, usage.model, usage.api), "pricing"
                    )
                    self._emit_token_events(usage, sub, dimensions, event_id)
                    return
                breakdown = compute_cost(usage, price, markup_value)

            self._push_cost_event(usage, breakdown, sub, dimensions, event_id)
        except Exception as exc:  # noqa: BLE001 — never raise from emit
            self._report_error(exc, "emit")

    def _emit_token_events(
        self, usage: CanonicalUsage, sub: str, dimensions: dict[str, Any] | None, event_id: str | None = None
    ) -> None:
        nonzero = usage.nonzero_numeric()
        if not nonzero:
            # Mistral legacy / empty — nothing to bill
            return
        now = int(time.time())
        for field_name, value in nonzero.items():
            code = self.config.metric_codes.get(field_name)
            if not code:
                continue
            event = {
                # `_tok_` namespace: the cost path suffixes with the same field
                # vocabulary, and both are reachable for one `event_id` (price
                # miss -> token fallback, then the cost path once the table is
                # warm). See emit()'s docstring for what a shared namespace cost.
                "transaction_id": f"{event_id}_tok_{field_name}" if event_id else str(uuid.uuid4()),
                "external_subscription_id": sub,
                "code": code,
                "timestamp": now,
                "properties": {
                    "value": str(value),
                    "model": usage.model,
                    "provider": usage.provider,
                    "api": usage.api,
                    **(dimensions or {}),
                },
            }
            self._queue.push(event)

    def _push_cost_event(
        self,
        usage: CanonicalUsage,
        breakdown: CostBreakdown,
        sub: str,
        dimensions: dict[str, Any] | None,
        event_id: str | None = None,
    ) -> None:
        """Push one llm_cost event — or several, one per token_type, when a
        real per-field breakdown exists.

        `breakdown.fields` only exists when we priced via our own per-token
        table (`compute_cost`): the live wrap() path, where an OpenRouter/
        Bedrock unit price is available for input/output/cache/reasoning
        separately. There, billing is split one event per field, each tagged
        `token_type`, so Lago's `grouped_by: ["model", "token_type"]` charge
        can break llm_cost down by both dimensions.

        A precomputed breakdown (`usd_cost` — e.g. Cloudflare AI Gateway's own
        already-metered `cost` per call) has no such split: the gateway gives
        one lump sum, not "$X of this was input tokens" — inventing a
        proportional split would substitute our own guess for the real number
        we specifically avoided guessing at. That path bills a single event,
        grouped by model only; no `token_type` at all rather than a fabricated
        one.
        """
        now = int(time.time())
        base_properties: dict[str, Any] = {
            "model": usage.model,
            "provider": usage.provider,
            "api": usage.api,
            "price_source": breakdown.source,
            "markup": breakdown.markup,
            **(dimensions or {}),
        }

        if not breakdown.fields:
            properties = {
                **base_properties,
                "unit": str(usage.input + usage.output),
                "value": breakdown.total,
                "base_cost": breakdown.base,
            }
            self._queue.push(
                {
                    # Unsuffixed: this branch pushes exactly ONE event, so there is
                    # nothing to disambiguate. It cannot collide with the namespaced
                    # multi-event ids below or in _emit_token_events.
                    "transaction_id": event_id or str(uuid.uuid4()),
                    "external_subscription_id": sub,
                    "code": self.config.cost_metric_code,
                    "timestamp": now,
                    "precise_total_amount_cents": breakdown.total_cents,
                    "properties": properties,
                }
            )
            return

        for field_name, parts in breakdown.fields.items():
            # parts["cost"] is PRE-markup (compute_cost only applies markup to
            # the summed total) — apply it here or a markup != 1.0 silently
            # vanishes from every split event.
            billed_cost = apply_markup(parts["cost"], breakdown.markup)
            properties = {
                **base_properties,
                "token_type": field_name,
                "unit": parts["tokens"],
                "value": billed_cost,
                "base_cost": parts["cost"],
                "unit_price": parts["unit_price"],
            }
            self._queue.push(
                {
                    # `_cost_` namespace — see the `_tok_` note in _emit_token_events.
                    "transaction_id": f"{event_id}_cost_{field_name}" if event_id else str(uuid.uuid4()),
                    "external_subscription_id": sub,
                    "code": self.config.cost_metric_code,
                    "timestamp": now,
                    "precise_total_amount_cents": money_str_to_cents(billed_cost),
                    "properties": properties,
                }
            )

    def _report_error(self, exc: Exception, where: str) -> None:
        if self.config.on_error:
            try:
                self.config.on_error(exc, where)
            except Exception:  # noqa: BLE001
                pass
        logger.warning("lago %s failed: %s", where, exc)

    def warm_pricing(self, providers: Iterable[str] = ()) -> None:
        """Block until the given price table(s) are fetched, instead of
        waiting for the queue's background thread to pick them up on its
        next tick (up to `flush_interval` seconds later, by default ~1s).

        A call made immediately after construction — the common shape in a
        script, notebook, or one-shot job, as opposed to a long-running server
        where the first real call naturally lands well after that first tick
        — races a still-cold cache. `emit()` never silently under-bills, so a
        miss falls back to token events; but with no token-metric charge
        configured at all (a single `llm_cost`-only billing setup), there is
        nowhere left to fall back to and the event is lost. Call this once,
        right after constructing the SDK with `pricing_mode="price"`, to close
        that window deterministically for OpenRouter — the table nearly every
        native provider prices against — which is always warmed regardless
        of `providers`.

        Cloudflare Workers AI and Mistral alias resolution are NOT warmed by
        default: both are credential-gated and provider-specific, and
        eagerly hitting either's API at construction time regardless of
        whether that provider is ever actually called would be pure waste
        for the common case. Left alone, they stay reactive — the first real
        call to that provider triggers the fetch, and every call after that
        (even the one a moment later) is cached — so only a session's first
        Workers AI or Mistral call can race a cold cache.

        If you already know you're about to call one or both this session,
        say so and skip that one-time cost too: `providers=["mistral"]`
        and/or `["workers-ai"]`. A no-op for any source that isn't stale
        (e.g. the SDK isn't in price mode, was already warmed, or the
        provider name wasn't recognized)."""
        self._pricing.prime(providers)
        self._pricing.maybe_refresh()

    def flush(self, timeout: float = 5.0) -> bool:
        return self._queue.flush(timeout=timeout)

    def shutdown(self, timeout: float = 5.0) -> None:
        self._queue.shutdown(timeout=timeout)

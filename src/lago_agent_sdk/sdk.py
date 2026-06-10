"""LagoSDK — primary entrypoint."""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from typing import Any

from .canonical import CanonicalUsage
from .config import LagoConfig
from .detector import detect_client_kind
from .exceptions import PricingUnavailableError, UnknownClientError
from .lago_client import LagoClient
from .pricing import PricingProvider, coerce_markup, compute_cost
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
        )
        # Pricing provider (price mode). Default does no network until a
        # price-mode lookup flags a source stale; refreshes run on the queue
        # thread, never on the customer's call.
        self._pricing: PricingProvider = self.config.pricing_provider or PricingProvider(
            ttl_seconds=self.config.pricing_ttl_seconds,
            default_region=self.config.bedrock_default_region,
            on_error=self.config.on_error,
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

    # ------------------------------------------------------------------
    # Wrap()
    # ------------------------------------------------------------------
    def wrap(
        self, client: Any, dimensions: dict[str, Any] | None = None, subscription: str | None = None
    ) -> Any:
        kind = detect_client_kind(client)
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
    ) -> None:
        """Emit usage to Lago.

        In ``tokens`` mode (default), pushes one event per nonzero token field.
        In ``price`` mode, pushes a single dollar-cost event; if no price is
        available it falls back to token events and reports via on_error.
        Precedence for mode/markup: per-call arg > config default.
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
                self._emit_token_events(usage, sub, dimensions)
                return

            price = self._pricing.lookup(usage.provider, usage.model, usage.api)
            if price is None:
                # Don't silently under-bill: fall back to token events + report.
                self._report_error(PricingUnavailableError(usage.provider, usage.model, usage.api), "pricing")
                self._emit_token_events(usage, sub, dimensions)
                return

            markup_value, ok = coerce_markup(markup if markup is not None else self.config.markup)
            if not ok:
                self._report_error(
                    ValueError(
                        f"invalid markup {markup if markup is not None else self.config.markup!r}; using 1.0"
                    ),
                    "pricing",
                )
            self._emit_cost_event(usage, price, markup_value, sub, dimensions)
        except Exception as exc:  # noqa: BLE001 — never raise from emit
            self._report_error(exc, "emit")

    def _emit_token_events(self, usage: CanonicalUsage, sub: str, dimensions: dict[str, Any] | None) -> None:
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
                "transaction_id": str(uuid.uuid4()),
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

    def _emit_cost_event(
        self,
        usage: CanonicalUsage,
        price: Any,
        markup: Any,
        sub: str,
        dimensions: dict[str, Any] | None,
    ) -> None:
        breakdown = compute_cost(usage, price, markup)
        # `unit` = total tokens for the call — the quantity the sum-aggregation
        # billable metric sums (the dynamic charge's fee comes from
        # precise_total_amount_cents; unit is the displayed usage quantity).
        # Sum the *billed* per-field counts from the breakdown, which compute_cost
        # has already de-overlapped (e.g. cache_read carved out of input), so
        # subset fields aren't double-counted in the displayed total.
        unit = sum(int(parts["tokens"]) for parts in breakdown.fields.values())
        properties: dict[str, Any] = {
            "unit": str(unit),
            "value": breakdown.total,
            "base_cost": breakdown.base,
            "markup": breakdown.markup,
            "model": usage.model,
            "provider": usage.provider,
            "api": usage.api,
            "price_source": breakdown.source,
        }
        for field_name, parts in breakdown.fields.items():
            properties[f"{field_name}_tokens"] = parts["tokens"]
            properties[f"{field_name}_unit_price"] = parts["unit_price"]
            properties[f"{field_name}_cost"] = parts["cost"]
        properties.update(dimensions or {})
        self._queue.push(
            {
                "transaction_id": str(uuid.uuid4()),
                "external_subscription_id": sub,
                "code": self.config.cost_metric_code,
                "timestamp": int(time.time()),
                # Top-level amount (in cents) for Lago's dynamic charge model —
                # the charge sums these into a single fee.
                "precise_total_amount_cents": breakdown.total_cents,
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

    def flush(self, timeout: float = 5.0) -> bool:
        return self._queue.flush(timeout=timeout)

    def shutdown(self, timeout: float = 5.0) -> None:
        self._queue.shutdown(timeout=timeout)

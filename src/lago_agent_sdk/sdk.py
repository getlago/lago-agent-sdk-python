"""LagoSDK — primary entrypoint."""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from .canonical import CanonicalUsage
from .config import LagoConfig
from .detector import detect_client_kind
from .exceptions import PricingUnavailableError, UnknownClientError
from .gateway.adapters.snowflake_cortex import SNOWFLAKE_EVENT_ID_PREFIX
from .lago_client import LagoClient
from .pricing import (
    TOKEN_BILLED_PROVIDERS,
    CostBreakdown,
    PricingProvider,
    apply_markup,
    coerce_markup,
    compute_cost,
    compute_precomputed_cost,
    deoverlapped_token_total,
    money_str_to_cents,
)
from .queue import EventQueue

logger = logging.getLogger("lago_agent_sdk")

_subscription_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "lago_subscription", default=None
)


def _to_epoch_seconds(value: int | float | datetime) -> int:
    """A caller-supplied event time as the unix seconds Lago's `timestamp` wants."""
    if isinstance(value, datetime):
        # A naive datetime is taken as UTC — the same rule `_as_utc` documents
        # for the window bound, so a caller who reads a window and bills it cannot
        # have the two disagree by their machine's UTC offset.
        moment = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return int(moment.timestamp())
    # `not bool`: it is an `int` subclass, so `True` would otherwise bill at epoch 1.
    # The JS port's `typeof value === "number"` rejects it, and the two must agree.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    raise TypeError(
        f"timestamp={value!r} not understood — pass a datetime or epoch seconds "
        "(an ISO-8601 string is deliberately not accepted; see emit())"
    )


class LagoSDK:
    def __init__(
        self,
        api_key: str,
        api_url: str | None = None,
        default_subscription_id: str | None = None,
        config: LagoConfig | None = None,
        verify_ssl: bool | None = None,
    ) -> None:
        """Explicit args win over anything set on ``config``; ``config`` supplies
        every field they don't mention.

        ``api_url`` defaults to None, NOT to the production URL. That distinction
        is load-bearing: with a truthy default, ``if api_url:`` always fired and
        overwrote ``config.api_url``, so
        ``LagoSDK(api_key=k, config=LagoConfig(api_url="http://localhost:3000/api/v1"))``
        silently sent every event to PRODUCTION Lago. That is the shortest path to
        the bug, too — a custom ``api_url`` and ``verify_ssl=False`` go together in
        exactly the local-dev-Lago setup ``verify_ssl`` exists to serve.

        ``verify_ssl`` is accepted directly so that setup needs no ``LagoConfig``
        at all: a local instance behind a self-signed cert (Traefik's default) is
        reachable with ``LagoSDK(api_key=..., api_url=..., verify_ssl=False)``.
        """
        if isinstance(api_key, LagoConfig):
            # `LagoSDK(cfg)` is the natural-looking call and it is silently, totally
            # wrong: the config becomes the BEARER TOKEN while ``config`` stays None, so
            # a fresh default ``LagoConfig`` is built and every field the caller set is
            # discarded. Measured live: a config naming a local Lago produced an SDK
            # posting to PRODUCTION ``api.getlago.com`` with an unusable key — every
            # event 401, ``flush()`` still returning True, and the caller's own
            # ``on_error`` never invoked, because that hook was one of the discarded
            # fields. The only trace was a WARNING per event. Whether the queue then
            # drops those events or holds them is beside the point: no key the caller
            # passed is ever used, so nothing downstream can recover. It has to fail at
            # construction.
            raise TypeError(
                "LagoSDK's first positional parameter is `api_key`, not `config`. "
                "Passing a LagoConfig here makes it the bearer token and leaves the "
                "rest of your config unused, so every event is sent to the default "
                "api_url with a key that 401s. Use LagoSDK(config.api_key, config=config)."
            )
        self.config = config or LagoConfig(api_key=api_key)
        # explicit args win over `config` — guarded on "was it actually passed?"
        # rather than on truthiness, so a config value survives when it wasn't.
        self.config.api_key = api_key or self.config.api_key
        # `api_url` is the one exception: an EMPTY string must not win either. The bug
        # this guard was written for was a *truthy default* overwriting config, so
        # accepting "" swapped one silent misroute for a worse one —
        # `api_url=os.environ.get("LAGO_API_URL", "")` with the var unset used to keep
        # the production URL and instead wrote "". Downstream that is unrecoverable:
        # `requests` raises MissingSchema, which is not a LagoApiError, so the queue
        # classifies it transient, re-prepends the batch and retries at the 60s ceiling
        # forever. All billing stops, nothing is dropped or escalated, and the only
        # symptom is a growing buffer.
        #
        # Falling back is right, but it must not be SILENT — see the report below.
        if api_url:
            self.config.api_url = api_url
        if default_subscription_id is not None:
            self.config.default_subscription_id = default_subscription_id
        if verify_ssl is not None:
            self.config.verify_ssl = verify_ssl

        # A caller who passed `api_url` explicitly MEANT to point somewhere specific.
        # Discarding a falsy one is the safe choice for delivery, but doing it silently
        # is the one outcome that must not happen here: `LagoConfig`'s default is
        # PRODUCTION, so `api_url=os.environ.get("LAGO_API_URL", "")` with the var unset
        # now resolves to production Lago and every event is accepted. For a CI job or a
        # developer holding a real production key that writes live billing data, and
        # ingested events cannot be un-ingested. `on_error` is opt-in, so this reports
        # through the same log-plus-callback floor as every other drop path rather than
        # trusting a callback to exist.
        if api_url is not None and not api_url:
            self._report_error(
                ValueError(
                    f"api_url was explicitly set to an empty value; falling back to "
                    f"{self.config.api_url}. Set LAGO_API_URL (or pass config.api_url) "
                    f"if you did not intend to send events there."
                ),
                "config.api_url",
            )

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
        # (provider, model) pairs already noted as token-billed, so the explanation is
        # logged once rather than on every call. See `_note_token_billed`.
        self._token_billed_noted: set[tuple[str, str]] = set()
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
        timestamp: int | float | datetime | None = None,
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

        ``timestamp``: bill the events at this instant instead of at now — pass
        the source row's own time when replaying/backfilling from a gateway's
        logs, or a window reaching back a week bills every one of its calls into
        the period the script happens to run in. Accepts a ``datetime`` (a naive
        one is read as UTC) or epoch seconds. Deliberately NOT an ISO-8601
        string: Python 3.10 is still supported here and its ``fromisoformat``
        rejects the trailing "Z" that gateway APIs emit, while the JS port's
        ``new Date()`` accepts it — so a string would parse in one repo and fail
        in the other. Connectors parse their own source column instead, where the
        shapes that column really returns are known and tested (see
        ``DatabricksUsageRow.occurred_at``). A live call has no source time and
        should leave this as None.

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
            # Resolved ONCE, ahead of every branch: a price-lookup miss falls through
            # to the token path, so one usage row can reach two of the push paths
            # below. Two separate `time.time()` reads there let a call that straddles
            # a billing-period boundary land half in each period.
            at = self._event_time(timestamp)
            sub = self._resolve_subscription(subscription)
            if not sub:
                # `_report_error` is the single channel: it invokes on_error AND
                # logs. An extra logger.error here emitted the same drop twice under
                # two different levels, so a customer grepping logs counted one lost
                # call as two — and the JS port logged nothing at all, so the two
                # repos reported 2 lines vs 0 for the same event.
                self._report_error(
                    ValueError(
                        f"no resolvable subscription for model={usage.model!r}; events dropped. "
                        f"Pass subscription=..., use with_subscription(), or set "
                        f"LagoConfig.default_subscription_id."
                    ),
                    "emit",
                )
                return

            effective_mode = mode or self.config.pricing_mode
            if effective_mode != "price":
                if usd_cost is not None:
                    # A caller who went to the trouble of supplying a real metered
                    # cost gets told it was dropped, rather than discovering later
                    # that a whole backfill billed token counts only. Reported per
                    # occurrence, deliberately not deduped: the number of discarded
                    # costs is exactly what a caller reconciling on `on_error`
                    # needs, and the documented backfill pattern passes an explicit
                    # `mode="price"`, so reaching this at volume means a real
                    # misconfiguration rather than normal operation.
                    self._report_error(
                        ValueError(
                            f"usd_cost={usd_cost!r} ignored: effective pricing mode is "
                            f"{effective_mode!r}, not 'price' — emitting token counts "
                            f"instead. Pass mode='price' per call, or set "
                            f"LagoConfig.pricing_mode='price'."
                        ),
                        "pricing",
                    )
                self._emit_token_events(usage, sub, dimensions, event_id, at)
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
            elif usage.provider in TOKEN_BILLED_PROVIDERS:
                # NOT a failure, so deliberately not routed through on_error: this
                # provider publishes no per-token rate at all, so token counts are the
                # complete answer rather than a fallback. Said once per model instead
                # of once per call. See TOKEN_BILLED_PROVIDERS for the reasoning.
                self._note_token_billed(usage)
                self._emit_token_events(usage, sub, dimensions, event_id, at)
                return
            else:
                price = self._pricing.lookup(usage.provider, usage.model, usage.api)
                if price is None:
                    # Don't silently under-bill: fall back to token events + report.
                    self._report_error(
                        PricingUnavailableError(usage.provider, usage.model, usage.api), "pricing"
                    )
                    self._emit_token_events(usage, sub, dimensions, event_id, at)
                    return
                breakdown = compute_cost(usage, price, markup_value)

            self._push_cost_event(usage, breakdown, sub, dimensions, event_id, at)
        except Exception as exc:  # noqa: BLE001 — never raise from emit
            self._report_error(exc, "emit")

    def _event_time(self, timestamp: int | float | datetime | None) -> int:
        """The instant to stamp this call's events with — the caller's, or now.

        A value we cannot read is reported and falls back to `now` rather than
        dropping the call. Stamping the wrong period is a reconciliation problem the
        operator can see and fix; losing the event is revenue that never appears at
        all. Same trade-off as a missed price lookup.
        """
        if timestamp is not None:
            try:
                return _to_epoch_seconds(timestamp)
            # OverflowError/OSError: `datetime.timestamp()` raises them, platform
            # dependently, for a year outside the C time range.
            except (TypeError, ValueError, OverflowError, OSError) as exc:
                self._report_error(exc, "timestamp")
        return int(time.time())

    def _note_token_billed(self, usage: CanonicalUsage) -> None:
        """Say it once per model, at info level.

        It is a standing fact about the provider, not an event about this call, so
        repeating it per request would bury the log in something the reader can neither
        fix nor act on.
        """
        key = (usage.provider, usage.model)
        if key in self._token_billed_noted:
            return
        self._token_billed_noted.add(key)
        logger.info(
            "lago: %s bills %r in its own units, not per token — emitting token counts "
            "for it instead of a dollar cost",
            usage.provider,
            usage.model,
        )

    def _emit_token_events(
        self,
        usage: CanonicalUsage,
        sub: str,
        dimensions: dict[str, Any] | None,
        event_id: str | None = None,
        at: int | None = None,
    ) -> None:
        nonzero = usage.nonzero_numeric()
        # A negative count is silently unbillable — Lago would otherwise sum it into
        # a negative quantity. It was the only drop path in the SDK that never
        # reached on_error, so a caller who built a CanonicalUsage with a bad delta
        # saw nothing at all. Reported before the empty-check, because an event whose
        # only fields were negative leaves `nonzero` empty and would return below
        # without a word.
        negatives = usage.negative_numeric()
        if negatives:
            self._report_error(
                ValueError(f"dropped negative token counts for model={usage.model!r}: {negatives}"),
                "negative_tokens",
            )
        if not nonzero:
            # Mistral legacy / empty — nothing to bill
            return
        # `emit` already resolved the instant; the fallback covers nothing today and
        # is kept only so this stays callable on its own without stamping the epoch.
        now = at if at is not None else int(time.time())
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
        at: int | None = None,
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
        # See the note in `_emit_token_events` — `emit` is the one authority on this.
        now = at if at is not None else int(time.time())
        # Caller dimensions are spread LAST in each `properties` below, not here —
        # they must win over every SDK-computed key, exactly as they already do in
        # `_emit_token_events`. Spreading them into `base_properties` put them
        # *before* `unit`/`value`/`base_cost`/`unit_price`, so those four silently
        # overwrote a caller's same-named dimension on this path while honouring it
        # on the token path — one customer config, two different outcomes.
        base_properties: dict[str, Any] = {
            "model": usage.model,
            "provider": usage.provider,
            "api": usage.api,
            "price_source": breakdown.source,
            "markup": breakdown.markup,
        }

        if not breakdown.fields:
            properties = {
                **base_properties,
                # Same basis as the split path below (which reports the
                # de-overlapped per-field `parts["tokens"]`), so the two branches
                # can't report different quantities for one call. `input + output`
                # dropped `reasoning` and `cache_write` entirely — on a real
                # captured Gemini row with 9 in / 21 out / 852 reasoning it
                # published unit="30" for a call that consumed 882 — and counted a
                # cache-inclusive provider's cached tokens at full weight.
                "unit": str(deoverlapped_token_total(usage)),
                "value": breakdown.total,
                "base_cost": breakdown.base,
                **(dimensions or {}),
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
                **(dimensions or {}),
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

    def backfill_databricks(
        self,
        source: Any,
        since: Any = "1 day",
        *,
        default_subscription: str | None = None,
        unified: bool = False,
        dimensions: dict[str, Any] | None = None,
        event_id_prefix: str = "dbx",
    ) -> dict[str, int]:
        """Read a window of Databricks AI Gateway usage and bill all of it.

        The one-call entrypoint: give it a window, it does the rest. Returns counts
        of what it handed to ``emit()``, e.g.
        ``{"cost": 56, "tokens": 45, "skipped": 0, "deferred": 0}``.

        The last two are billing GAPS, and both are also reported through
        ``config.on_error`` (``where="backfill"``) — the hook every other gap in this
        SDK uses — so a caller does not have to inspect the return value to notice
        one. They fail differently: ``skipped`` rows had no resolvable subscription
        and stay lost until they are tagged or a default is set, while ``deferred``
        buckets are billable revenue that the NEXT run of the same window collects
        once Databricks has aggregated their spend row. A run with both at 0 is the
        only one that billed the whole window.

        ``source`` is normally a :class:`DatabricksSource`, and ``since`` the window.
        It also accepts an already-read iterable of ``DatabricksUsageRow`` — pass one
        when you have inspected the rows first, so the window is read ONCE. Reading
        twice is not just slow: a SQL warehouse costs roughly 1,500x the model-serving
        usage it reports on, and rows landing between the two reads make the summary
        you printed disagree with what was billed.

        Billing follows the rule the connector establishes rather than re-deriving
        it: a BYOK row carries Databricks' own metered USD and bills as a dollar
        cost; a Databricks-hosted row has no per-request dollar figure anywhere in
        Databricks' system tables and bills as token counts.

        ``unified=True`` bills everything to ``default_subscription``, ignoring
        per-call ``request_tags`` — right when one gateway serves one customer.
        Left False, each row goes to the subscription its own tags name, falling
        back to ``default_subscription`` only when a row is untagged.

        Every event also carries the Databricks-side grouping key for its row —
        ``endpoint_name`` for hosted, ``bucket`` for BYOK — so grouping Lago the
        same way the Databricks page groups puts the two side by side. See
        ``DatabricksUsageRow.reconcile_dimensions``. Anything in ``dimensions``
        is added on top and wins on a key collision.

        Idempotent: every event id is derived from the source row's own id and
        scoped by subscription, so re-running the same window has Lago reject the
        duplicates rather than double-bill. Does not flush — call ``flush()`` when
        you want to block on delivery.
        """
        counts = {"cost": 0, "tokens": 0, "skipped": 0, "deferred": 0}
        reader = source if hasattr(source, "read_usage") else None
        rows = reader.read_usage(since, event_id_prefix=event_id_prefix) if reader else source
        for row in rows:
            sub = default_subscription if unified else (row.subscription or default_subscription)
            if not sub:
                # No attribution and no fallback — emit() would drop it anyway, but
                # counting it here makes the gap visible instead of silent.
                counts["skipped"] += 1
                continue
            # Row's own reconciliation key first, so an explicit caller dimension of
            # the same name wins rather than being silently overwritten.
            dims = {**row.reconcile_dimensions, **(dimensions or {})}
            if row.usd_cost is not None:
                self.emit(
                    row.usage,
                    subscription=sub,
                    dimensions=dims,
                    mode="price",
                    usd_cost=row.usd_cost,
                    # Keyed off the subscription actually billed, not the row's own
                    # tag — an untagged row billed to the default must not carry an
                    # id that blocks it from a different default on a later run.
                    event_id=row.event_id_for(sub),
                    # The row's own time, not the run's — see `occurred_at`. A
                    # backfill that stamps `now` bills last week's usage into this
                    # week's period, and nothing in Lago can tell afterwards.
                    timestamp=row.occurred_at,
                )
                counts["cost"] += 1
            else:
                self.emit(
                    row.usage,
                    subscription=sub,
                    dimensions=dims,
                    mode="tokens",
                    event_id=row.event_id_for(sub),
                    timestamp=row.occurred_at,
                )
                counts["tokens"] += 1

        # Both gaps below were counted but never reported: measured live over a
        # window with one hour's spend rows withheld — the shape of real spend-table
        # lag — this returned `{'cost': 12, 'tokens': 54, 'skipped': 0}` while 54
        # BYOK buckets went unbilled and `on_error` fired zero times. `cost` alone
        # dropping from 66 to 12 is not something an automated caller can read as a
        # gap, so route both through the hook that already means "billing gap".
        if counts["skipped"]:
            self._report_error(
                ValueError(
                    f"{counts['skipped']} Databricks row(s) had no resolvable subscription "
                    f"and were NOT billed. Pass default_subscription=..., set "
                    f"LagoConfig.default_subscription_id, or tag the calls."
                ),
                "backfill",
            )
        # Only the reader knows about a bucket it never yielded, so a caller who
        # passed an already-read iterable gets 0 here — they hold the source and can
        # read `deferred_buckets` off it directly. `getattr` because `source` is
        # duck-typed: a caller's own reader need not carry the attribute.
        deferred = list(getattr(reader, "deferred_buckets", ())) if reader is not None else []
        counts["deferred"] = len(deferred)
        if deferred:
            first = deferred[0]
            # `read_usage` logs this too. That is deliberate, not a stutter: a caller who
            # reads the window itself never reaches this line, and one who ran the backfill
            # needs it on the channel they reconcile against. Worded from the RUN's side so
            # the two read as one gap seen from two layers rather than as two gaps.
            self._report_error(
                ValueError(
                    f"this run left {len(deferred)} Databricks BYOK bucket(s) unbilled: no "
                    f"external_model_spend row yet (e.g. hour={first['hour']} "
                    f"provider={first['provider']} model={first['model']}). The spend table "
                    f"lags; re-run this window later to bill them."
                ),
                "backfill",
            )
        return counts

    def backfill_snowflake(
        self,
        source: Any,
        since: Any = "1 day",
        *,
        default_subscription: str | None = None,
        unified: bool = False,
        dimensions: dict[str, Any] | None = None,
        event_id_prefix: str = SNOWFLAKE_EVENT_ID_PREFIX,
        views: Any = None,
        subscription_order: Any = None,
    ) -> dict[str, int]:
        """Read a window of Snowflake Cortex usage and bill all of it.

        The one-call entrypoint: give it a window, it does the rest. Returns
        ``{"tokens": ..., "skipped": ...}`` — and those are the only two counts there
        can be. There is no ``cost``, because Snowflake meters Cortex in CREDITS
        against a rate card that exists in no view, so every row on this path bills as
        token counts; ``provider = "snowflake"`` is in ``TOKEN_BILLED_PROVIDERS``,
        which is what routes a customer running ``pricing_mode="price"`` globally to
        token events here with no price-miss report. There is no ``deferred`` separate
        from ``skipped`` either — a deferred row is simply one this run did not bill.

        ``skipped`` is a billing GAP and is also reported through ``config.on_error``
        (``where="backfill"``) — the hook every other gap in this SDK uses — so a
        caller does not have to inspect the return value to notice one. It has two
        causes, reported separately because they are fixed differently: a row with no
        resolvable subscription stays lost until it is tagged or a default is set,
        while a DEFERRED row is billable revenue held back because an hour-bucketed
        query's ``QUERY_ID`` is not unique and whether its per-window ``METRICS`` is
        incremental or cumulative is unmeasured. See ``SnowflakeSource.read_usage``. A
        run with ``skipped == 0`` is the only one that billed the whole window.

        **Reads the functions view only, unless you ask for more.** The REST view
        reports the calls a wrapped client already billed live, so backfilling it over
        the same window bills each of those calls twice — the two ``transaction_id``s
        are unrelated and Lago cannot reject the duplicate. Pass ``views=("rest",)``
        only for traffic ``wrap()`` never saw. The functions view is the opposite
        case: ``AI_COMPLETE`` and friends run as SQL inside the warehouse, there is no
        client to patch, and a backfill is the ONLY way to bill them.

        ``unified=True`` bills everything to ``default_subscription``, ignoring each
        row's own attribution — right when one account serves one customer. Left
        False, each row goes to the subscription its ``QUERY_TAG`` names, falling back
        to ``default_subscription``. By default that tag is the ONLY source consulted:
        every live row carries a populated ``ROLE_NAMES``/``USER_ID``, so an order
        including them never falls through to the default — it bills untagged rows to
        a Snowflake role or user id instead, which Lago accepts for the nonexistent
        subscription without an error anywhere (measured, 2026-08-28). Opt into
        ``subscription_order=("query_tag", "role_names")`` (or ``"user_id"``) only
        when your account really maps that identity to a Lago subscription.

        Every event also carries the Snowflake-side grouping key for its row —
        ``FUNCTION_NAME`` + ``MODEL_NAME`` for functions rows, ``INFERENCE_REGION``
        for REST — so grouping Lago the same way you ``GROUP BY`` the view puts the
        two side by side. Anything in ``dimensions`` is merged on top and wins on a
        key collision.

        Idempotent: every event id derives from the source row's own id and is scoped
        by subscription, so re-running the same window has Lago reject the duplicates
        rather than double-bill. The same key protects against the live path: the
        OpenAI wrapper stamps a REST call's events with the id this backfill would
        derive from that call's ``REQUEST_ID``, so a ``views=("rest",)`` read over a
        live-billed window is rejected by Lago instead of billing twice. That
        protection holds ONLY when this backfill runs with the default
        ``event_id_prefix`` AND resolves the same subscription the live path billed —
        a custom prefix, or a different ``default_subscription`` than the wrapper's
        resolution, silently makes the two keys unrelated again and nothing reports
        it. Nor does it cover a cache-creation call's cached block: the wire reports
        creations as reads, so the live path billed ``cache_read`` while this backfill
        bills the row's ``cache_write`` under a different id — input and output dedup,
        that one component double-counts. Know also what a fully-duplicate window costs: ``/events/batch`` rejects a
        batch containing any duplicate wholesale, so the queue re-sends it one event
        at a time — N rows become N individual POSTs. Correct billing, through the
        mechanism built for emergencies; acceptable because reading this view at all
        is opt-in. Does not flush — call ``flush()`` when you want to block on
        delivery.

        ``source`` is normally a :class:`SnowflakeSource`, and ``since`` the window. It
        also accepts an already-read iterable of ``SnowflakeUsageRow`` — pass one when
        you have inspected the rows first, so the window is read ONCE. Reading twice is
        not just slow: a SQL warehouse is a real cost centre, and rows landing between
        the two reads make the summary you printed disagree with what was billed.
        """
        counts = {"tokens": 0, "skipped": 0}
        reader = source if hasattr(source, "read_usage") else None
        if reader is not None:
            kwargs: dict[str, Any] = {"event_id_prefix": event_id_prefix}
            # Only forwarded when set, so the reader's own safe default — functions
            # only — is what applies when a caller says nothing.
            if views is not None:
                kwargs["views"] = views
            if subscription_order is not None:
                kwargs["subscription_order"] = subscription_order
            rows = reader.read_usage(since, **kwargs)
        else:
            rows = source
        unattributed = 0
        for row in rows:
            sub = default_subscription if unified else (row.subscription or default_subscription)
            if not sub:
                # No attribution and no fallback — emit() would drop it anyway, but
                # counting it here makes the gap visible instead of silent.
                unattributed += 1
                continue
            # Row's own reconciliation key first, so an explicit caller dimension of
            # the same name wins rather than being silently overwritten.
            dims = {**row.reconcile_dimensions, **(dimensions or {})}
            # Never mode="tokens" per call: `TOKEN_BILLED_PROVIDERS` is checked inside
            # the price-mode branch, so forcing the mode here would diverge from what a
            # global pricing_mode="price" customer gets and suppress the
            # discarded-usd_cost report.
            self.emit(
                row.usage,
                subscription=sub,
                dimensions=dims,
                # Keyed off the subscription actually billed, not the row's own tag —
                # an unattributed row billed to the default must not carry an id that
                # blocks it from a different default on a later run.
                event_id=row.event_id_for(sub),
                # The row's own time, not the run's — the start of its hour bucket. A
                # backfill that stamps `now` bills last week's usage into this week's
                # period, and nothing in Lago can tell afterwards.
                timestamp=row.occurred_at,
            )
            counts["tokens"] += 1

        # Only the reader knows about a row it never yielded, so a caller who passed an
        # already-read iterable gets 0 here — they hold the source and can read
        # `deferred_rows` off it directly. `getattr` because `source` is duck-typed: a
        # caller's own reader need not carry the attribute.
        deferred = list(getattr(reader, "deferred_rows", ())) if reader is not None else []
        # Both causes are the same kind of outcome — revenue this run did not bill — so
        # they share one count, per the connector's {tokens, skipped} contract. They are
        # reported separately below because the operator fixes them differently.
        counts["skipped"] = unattributed + len(deferred)

        if unattributed:
            self._report_error(
                ValueError(
                    f"{unattributed} Snowflake row(s) had no resolvable subscription and "
                    f"were NOT billed. Pass default_subscription=..., set "
                    f"LagoConfig.default_subscription_id, or tag the queries with "
                    f"""ALTER SESSION SET QUERY_TAG = '{{"lago_subscription": "..."}}'."""
                ),
                "backfill",
            )
        if deferred:
            first = deferred[0]
            # `read_usage` logs this too. Deliberate, not a stutter: a caller who reads
            # the window themselves never reaches this line, and one who ran the backfill
            # needs it on the channel they reconcile against. Worded from the RUN's side
            # so the two read as one gap seen from two layers rather than as two gaps.
            self._report_error(
                ValueError(
                    f"this run left {len(deferred)} Snowflake functions row(s) unbilled "
                    f"(e.g. QUERY_ID={first.get('query_id')} reason={first.get('reason')}). "
                    f"An hour-bucketed query writes one row per bucket under one QUERY_ID, "
                    f"so the idempotency key collides and whether each row's METRICS is "
                    f"incremental or cumulative is unmeasured — billing either way would "
                    f"over- or under-charge by the query's hour count."
                ),
                "backfill",
            )
        return counts

    def flush(self, timeout: float = 5.0) -> bool:
        return self._queue.flush(timeout=timeout)

    def shutdown(self, timeout: float = 5.0) -> None:
        # Drain first, then release the socket — the queue's exit drain still needs it.
        self._queue.shutdown(timeout=timeout)
        self._lago_client.close()

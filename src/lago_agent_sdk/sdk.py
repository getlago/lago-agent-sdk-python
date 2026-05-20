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
from .exceptions import UnknownClientError
from .lago_client import LagoClient
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
        self._queue = EventQueue(
            sender=self._lago_client.send_batch,
            flush_interval=self.config.flush_interval_seconds,
            max_batch_size=self.config.max_batch_size,
            max_buffer_size=self.config.max_buffer_size,
            max_retry_seconds=self.config.max_retry_seconds,
            on_error=self.config.on_error,
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
        if kind == "unknown":
            raise UnknownClientError(
                f"Unknown client passed to wrap(): {type(client).__module__}.{type(client).__name__}. "
                "Supported: boto3 bedrock-runtime, mistralai.client.Mistral, anthropic.Anthropic / AsyncAnthropic."
            )
        raise UnknownClientError(
            f"Client kind '{kind}' is not yet supported. Implemented: 'bedrock', 'mistral', 'anthropic'."
        )

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------
    def emit(
        self,
        usage: CanonicalUsage,
        subscription: str | None = None,
        dimensions: dict[str, Any] | None = None,
    ) -> None:
        try:
            sub = self._resolve_subscription(subscription)
            if not sub:
                logger.error(
                    "lago: dropping events for model=%s — no resolvable subscription",
                    usage.model,
                )
                return

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
        except Exception as exc:  # noqa: BLE001 — never raise from emit
            if self.config.on_error:
                try:
                    self.config.on_error(exc, "emit")
                except Exception:  # noqa: BLE001
                    pass
            logger.warning("lago emit failed: %s", exc)

    def flush(self, timeout: float = 5.0) -> bool:
        return self._queue.flush(timeout=timeout)

    def shutdown(self, timeout: float = 5.0) -> None:
        self._queue.shutdown(timeout=timeout)

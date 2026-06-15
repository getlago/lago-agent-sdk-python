"""LagoConfig — runtime configuration for the SDK."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

DEFAULT_METRIC_CODES: dict[str, str] = {
    "input": "llm_input_tokens",
    "output": "llm_output_tokens",
    "cache_read": "llm_cached_input_tokens",
    "cache_write": "llm_cache_creation_tokens",
    "cache_write_5m": "llm_cache_write_5m_tokens",
    "cache_write_1h": "llm_cache_write_1h_tokens",
    "reasoning": "llm_reasoning_tokens",
    "tool_calls": "llm_tool_calls",
    "image_input": "llm_image_input_tokens",
    "audio_input": "llm_audio_input_tokens",
    "audio_output": "llm_audio_output_tokens",
}

# Metric code for the single per-call dollar-cost event emitted in price mode.
DEFAULT_COST_METRIC_CODE = "llm_cost"

# Pricing mode: emit raw token counts (default, backward-compatible) or a single
# computed dollar-cost event per call.
PricingMode = Literal["tokens", "price"]


def _mask_api_key(api_key: str) -> str:
    """Render an api key safe for logs/repr: keeps a 4-char tail for debuggability."""
    if not api_key:
        return "<unset>"
    if len(api_key) <= 8:
        return "***"
    return f"***{api_key[-4:]}"


@dataclass
class LagoConfig:
    api_key: str = field(default="", repr=False)
    api_url: str = "https://api.getlago.com/api/v1"
    default_subscription_id: str | None = None
    metric_codes: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_METRIC_CODES))
    flush_interval_seconds: float = 1.0
    max_batch_size: int = 100
    max_buffer_size: int = 10_000
    request_timeout_seconds: float = 10.0
    max_retry_seconds: float = 60.0
    on_error: Callable[[Exception, str], None] | None = None

    # --- pricing (price mode) ---
    # Global default mode. "tokens" preserves the existing behavior exactly.
    pricing_mode: PricingMode = "tokens"
    # Multiplier applied to the computed cost (1.0 = no markup, 1.2 = +20%).
    markup: float = 1.0
    # Metric code for the single dollar-cost event emitted in price mode.
    cost_metric_code: str = DEFAULT_COST_METRIC_CODE
    # How long a fetched pricing table stays fresh before a background refresh.
    pricing_ttl_seconds: float = 3600.0
    # Region used for Bedrock pricing when the model id carries no region prefix.
    bedrock_default_region: str = "us-east-1"
    # Optional injected PricingProvider (or a stub) — primarily for tests/overrides.
    # Typed Any to avoid a config→pricing import cycle.
    pricing_provider: Any | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"LagoConfig(api_key={_mask_api_key(self.api_key)!r}, "
            f"api_url={self.api_url!r}, "
            f"default_subscription_id={self.default_subscription_id!r}, "
            f"flush_interval_seconds={self.flush_interval_seconds}, "
            f"max_batch_size={self.max_batch_size}, "
            f"max_buffer_size={self.max_buffer_size}, "
            f"request_timeout_seconds={self.request_timeout_seconds}, "
            f"max_retry_seconds={self.max_retry_seconds}, "
            f"pricing_mode={self.pricing_mode!r}, "
            f"markup={self.markup}, "
            f"cost_metric_code={self.cost_metric_code!r}, "
            f"pricing_ttl_seconds={self.pricing_ttl_seconds}, "
            f"bedrock_default_region={self.bedrock_default_region!r})"
        )

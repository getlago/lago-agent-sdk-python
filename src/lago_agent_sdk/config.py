"""LagoConfig — runtime configuration for the SDK."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

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

    def __repr__(self) -> str:
        return (
            f"LagoConfig(api_key={_mask_api_key(self.api_key)!r}, "
            f"api_url={self.api_url!r}, "
            f"default_subscription_id={self.default_subscription_id!r}, "
            f"flush_interval_seconds={self.flush_interval_seconds}, "
            f"max_batch_size={self.max_batch_size}, "
            f"max_buffer_size={self.max_buffer_size}, "
            f"request_timeout_seconds={self.request_timeout_seconds}, "
            f"max_retry_seconds={self.max_retry_seconds})"
        )

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
}


@dataclass
class LagoConfig:
    api_key: str = ""
    api_url: str = "https://api.getlago.com/api/v1"
    default_subscription_id: str | None = None
    metric_codes: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_METRIC_CODES))
    flush_interval_seconds: float = 1.0
    max_batch_size: int = 100
    max_buffer_size: int = 10_000
    request_timeout_seconds: float = 10.0
    max_retry_seconds: float = 60.0
    on_error: Callable[[Exception, str], None] | None = None

"""Bedrock Converse adapter — single function, 3 shape families.

Verified against 39 models in eu-west-1.

Families:
- standard         : just inputTokens / outputTokens (33 models — non-Anthropic).
- cache-read-only  : adds cacheReadInputTokens (Claude Opus 4.7).
- full-cache       : adds cacheReadInputTokens + cacheWriteInputTokens
                     (Claude Sonnet 4.5/4.6, Haiku 4.5, Opus 4.5/4.6).
"""
from __future__ import annotations

from typing import Any

from ..canonical import CanonicalUsage

# Fields recognised at the top level of `usage`. Anything else lands in extras.
_KNOWN_USAGE_FIELDS = {
    "inputTokens",
    "outputTokens",
    "totalTokens",
    "cacheReadInputTokens",
    "cacheWriteInputTokens",
    "cacheReadInputTokenCount",  # newer alias — duplicate, ignored
    "cacheWriteInputTokenCount",  # newer alias — duplicate, ignored
    "serverToolUsage",
}


def _provider_from_model(model_id: str) -> str:
    mid = (model_id or "").lower()
    if "anthropic" in mid:
        return "anthropic"
    if "amazon" in mid or "nova" in mid or "titan" in mid:
        return "amazon"
    if "meta" in mid or "llama" in mid:
        return "meta"
    if "mistral" in mid or "mixtral" in mid or "pixtral" in mid:
        return "mistral"
    if "cohere" in mid:
        return "cohere"
    if "openai" in mid or "gpt-oss" in mid:
        return "openai"
    if "qwen" in mid:
        return "qwen"
    if "gemma" in mid:
        return "google"
    if "minimax" in mid:
        return "minimax"
    if "nvidia" in mid or "nemotron" in mid:
        return "nvidia"
    if "zai" in mid or "glm" in mid:
        return "zai"
    return "bedrock"


def extract_bedrock_converse(response: dict[str, Any], model_id: str = "") -> CanonicalUsage:
    if not isinstance(response, dict):
        response = {}
    usage = response.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    extras: dict[str, Any] = {}

    def _safe_int(v: Any) -> int:
        try:
            return max(0, int(v or 0))
        except (TypeError, ValueError):
            return 0

    out = CanonicalUsage(
        input=_safe_int(usage.get("inputTokens")),
        output=_safe_int(usage.get("outputTokens")),
        cache_read=_safe_int(usage.get("cacheReadInputTokens")),
        cache_write=_safe_int(usage.get("cacheWriteInputTokens")),
        model=model_id or "",
        provider=_provider_from_model(model_id),
        api="bedrock_converse",
    )

    # serverToolUsage: flatten into tool_calls if non-empty.
    server_tool_usage = usage.get("serverToolUsage")
    if isinstance(server_tool_usage, dict) and server_tool_usage:
        out.tool_calls = sum(_safe_int(v) for v in server_tool_usage.values())
        extras["serverToolUsage"] = server_tool_usage

    # Drift detection: anything we don't recognize lands in extras.
    for k, v in usage.items():
        if k not in _KNOWN_USAGE_FIELDS:
            extras[k] = v

    if extras:
        out.extras = extras
    return out

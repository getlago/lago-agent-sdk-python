"""Bedrock InvokeModel adapters — 7 shape families.

Dispatch by substring match on `modelId`. Verified against 39 models in eu-west-1.

  4.6.1 openai_compat_basic        — Gemma, Qwen, gpt-oss-120b/20b, Voxtral, MiniMax M2.5,
                                     Magistral, Devstral, Ministral, NVIDIA Nemotron Nano, GLM
  4.6.2 openai_compat_with_details — gpt-oss Safeguard 120B/20B, MiniMax M2, MiniMax M2.1
  4.6.3 anthropic                  — Claude Sonnet 4.5/4.6, Haiku 4.5, Opus 4.5/4.6
  4.6.4 anthropic_opus_4_7         — Claude Opus 4.7 (extra `service_tier` → extras)
  4.6.5 nova                       — Amazon Nova Pro/Lite/Micro/2-Lite
  4.6.6 pixtral                    — Mistral Pixtral Large
  4.6.7 mistral_legacy             — Mistral 7B / Mixtral 8x7B / Mistral Large 24.02
                                     (no usage; emit WARN, return _no_usage extras)
"""
from __future__ import annotations

import logging
from typing import Any

from ..canonical import CanonicalUsage

logger = logging.getLogger("lago_agent_sdk.adapters.bedrock_invoke")


def _safe_usage(resp: Any) -> dict[str, Any]:
    """Return resp['usage'] only if it's a dict; otherwise an empty dict."""
    if not isinstance(resp, dict):
        return {}
    u = resp.get("usage")
    return u if isinstance(u, dict) else {}


def _safe_int(v: Any) -> int:
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------
# Dispatch (per spec bottom block — verbatim)
# --------------------------------------------------------------------------
def pick_invoke_adapter(model_id: str) -> str:
    mid = (model_id or "").lower()
    if "anthropic" in mid:
        return "opus_4_7" if "opus-4-7" in mid else "anthropic"
    if "nova" in mid:
        return "nova"
    if "pixtral" in mid:
        return "pixtral"
    if "mistral" in mid or "mixtral" in mid:
        legacy = ["mistral-7b", "mixtral-8x7b", "mistral-large-2402"]
        if any(x in mid for x in legacy):
            return "mistral_legacy"
        return "openai_compat_basic"
    if any(x in mid for x in ["gpt-oss-safeguard", "minimax-m2"]):
        return "openai_compat_with_details"
    return "openai_compat_basic"  # default for everything else


# --------------------------------------------------------------------------
# Provider mapping (mirrors converse adapter)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Family extractors
# --------------------------------------------------------------------------
def _extract_openai_compat_basic(resp: dict[str, Any], model_id: str) -> CanonicalUsage:
    usage = _safe_usage(resp)
    extras: dict[str, Any] = {}
    known = {"prompt_tokens", "completion_tokens", "total_tokens"}
    for k, v in usage.items():
        if k not in known:
            extras[k] = v
    return CanonicalUsage(
        input=_safe_int(usage.get("prompt_tokens")),
        output=_safe_int(usage.get("completion_tokens")),
        model=model_id,
        provider=_provider_from_model(model_id),
        api="bedrock_invoke",
        extras=extras,
    )


def _extract_openai_compat_with_details(resp: dict[str, Any], model_id: str) -> CanonicalUsage:
    usage = _safe_usage(resp)
    extras: dict[str, Any] = {}
    # Note: completion_tokens_details is partially mapped (reasoning_tokens), so we
    # mark it as known. prompt_tokens_details is unmapped — let it land in extras
    # so drift detection surfaces it for follow-up.
    known = {"prompt_tokens", "completion_tokens", "total_tokens", "completion_tokens_details"}
    for k, v in usage.items():
        if k not in known:
            extras[k] = v

    details = usage.get("completion_tokens_details") or {}
    reasoning = _safe_int(details.get("reasoning_tokens")) if isinstance(details, dict) else 0

    return CanonicalUsage(
        input=_safe_int(usage.get("prompt_tokens")),
        output=_safe_int(usage.get("completion_tokens")),
        reasoning=reasoning,
        model=model_id,
        provider=_provider_from_model(model_id),
        api="bedrock_invoke",
        extras=extras,
    )


def _extract_anthropic(resp: dict[str, Any], model_id: str) -> CanonicalUsage:
    usage = _safe_usage(resp)
    extras: dict[str, Any] = {}
    known_top = {
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "cache_creation",
    }

    cache_creation = usage.get("cache_creation") or {}
    cache_write_5m = 0
    cache_write_1h = 0
    if isinstance(cache_creation, dict):
        cache_write_5m = _safe_int(cache_creation.get("ephemeral_5m_input_tokens"))
        cache_write_1h = _safe_int(cache_creation.get("ephemeral_1h_input_tokens"))

    # Tool calls — count `type == "tool_use"` content blocks
    content = resp.get("content") if isinstance(resp, dict) else None
    tool_calls = (
        sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_use") if isinstance(content, list) else 0
    )

    for k, v in usage.items():
        if k not in known_top:
            extras[k] = v

    return CanonicalUsage(
        input=_safe_int(usage.get("input_tokens")),
        output=_safe_int(usage.get("output_tokens")),
        cache_read=_safe_int(usage.get("cache_read_input_tokens")),
        cache_write=_safe_int(usage.get("cache_creation_input_tokens")),
        cache_write_5m=cache_write_5m,
        cache_write_1h=cache_write_1h,
        tool_calls=tool_calls,
        model=model_id,
        provider="anthropic",
        api="bedrock_invoke",
        extras=extras,
    )


def _extract_anthropic_opus_4_7(resp: dict[str, Any], model_id: str) -> CanonicalUsage:
    out = _extract_anthropic(resp, model_id)
    usage = _safe_usage(resp)
    if "service_tier" in usage:
        out.extras["service_tier"] = usage["service_tier"]
    return out


def _extract_nova(resp: dict[str, Any], model_id: str) -> CanonicalUsage:
    usage = _safe_usage(resp)
    known = {
        "inputTokens",
        "outputTokens",
        "totalTokens",
        "cacheReadInputTokenCount",
        "cacheWriteInputTokenCount",
    }
    extras: dict[str, Any] = {k: v for k, v in usage.items() if k not in known}

    return CanonicalUsage(
        input=_safe_int(usage.get("inputTokens")),
        output=_safe_int(usage.get("outputTokens")),
        cache_read=_safe_int(usage.get("cacheReadInputTokenCount")),
        cache_write=_safe_int(usage.get("cacheWriteInputTokenCount")),
        model=model_id,
        provider="amazon",
        api="bedrock_invoke",
        extras=extras,
    )


def _extract_pixtral(resp: dict[str, Any], model_id: str) -> CanonicalUsage:
    out = _extract_openai_compat_basic(resp, model_id)
    usage = _safe_usage(resp)
    if "request_count" in usage:
        out.extras["request_count"] = usage["request_count"]
    return out


def _extract_mistral_legacy(resp: dict[str, Any], model_id: str) -> CanonicalUsage:
    logger.warning(
        "Bedrock InvokeModel returned no usage for legacy Mistral model %s — cannot bill via InvokeModel; switch to Converse",
        model_id,
    )
    return CanonicalUsage(
        model=model_id,
        provider="mistral",
        api="bedrock_invoke",
        extras={"_no_usage": True},
    )


_DISPATCH = {
    "openai_compat_basic": _extract_openai_compat_basic,
    "openai_compat_with_details": _extract_openai_compat_with_details,
    "anthropic": _extract_anthropic,
    "opus_4_7": _extract_anthropic_opus_4_7,
    "nova": _extract_nova,
    "pixtral": _extract_pixtral,
    "mistral_legacy": _extract_mistral_legacy,
}


def extract_bedrock_invoke(response: dict[str, Any], model_id: str) -> CanonicalUsage:
    family = pick_invoke_adapter(model_id)
    return _DISPATCH[family](response or {}, model_id)

"""Detect which client kind was passed to wrap()."""

from __future__ import annotations

from typing import Any


def detect_client_kind(client: Any) -> str:
    """Return 'bedrock' | 'unknown' for Phase 1.

    Native client kinds (openai/anthropic/mistral/google) are reserved
    for Phase 2 — we still detect them here so error messages are useful.
    """
    cls_name = type(client).__name__.lower()
    module = getattr(type(client), "__module__", "") or ""

    # boto3 botocore client for bedrock-runtime
    if "botocore" in module and (
        cls_name.startswith("bedrockruntime") or "bedrock-runtime" in str(getattr(client, "_endpoint", ""))
    ):
        return "bedrock"

    # Service-name fallback: boto3 clients expose .meta.service_model.service_name
    try:
        svc = client.meta.service_model.service_name
        if svc == "bedrock-runtime":
            return "bedrock"
    except Exception:  # noqa: BLE001
        pass

    if "anthropic" in module:
        return "anthropic"
    if "openai" in module:
        return "openai"
    if "mistralai" in module:
        return "mistral"
    # Older mistralai versions or aliased imports
    if cls_name == "mistral" and "mistral" in module:
        return "mistral"
    # New unified google-genai SDK only (`google.genai.Client`). The legacy
    # google-generativeai SDK (`google.generativeai.GenerativeModel`) has a
    # different surface — no `.models` / `.aio` — that the gemini wrapper cannot
    # instrument, so it would silently wrap nothing. Flag it separately so wrap()
    # can reject it with a clear migration message instead.
    #
    # "genai" is not a substring of "generativeai", so these never overlap.
    if "google" in module and "genai" in module:
        return "gemini"
    if "google" in module and "generativeai" in module:
        return "gemini_legacy"

    return "unknown"

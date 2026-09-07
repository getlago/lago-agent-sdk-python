"""Ramp Router host detection, shared by every wrapper a customer can point at Router.

Router serves every provider it fronts through one dedicated host on two surfaces —
`/v1/responses` (reached with an OpenAI client) and `/v1/messages` (reached with an
Anthropic client). Neither surface leaves a mark in the response body: an
Anthropic-served answer on the first is byte-indistinguishable from real OpenAI, and any
vendor's answer on the second is rendered in Anthropic's schema. So the client's base URL
is the ONLY signal, and both wrappers must read it the same way — a copy in each would
drift, and the Anthropic wrapper going without one billed Router traffic as native
Anthropic (wrong price table, no Router key learned).

It must be the PARSED host, never a substring test. A substring row ("api.router.com")
also matches `https://evil.example.com/api.router.com/v1`, which would stamp an unrelated
endpoint's traffic as Router-served. The `.router.com` suffix arm covers a regional or
staging host without widening to arbitrary domains — `evilrouter.com` does not end in
`.router.com`.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

RAMP_ROUTER_HOST = "api.router.com"
RAMP_ROUTER_DOMAIN = ".router.com"


def is_ramp_router_base_url(base_url: Any) -> bool:
    """True when `base_url` names Router's host. A relative, malformed or non-string value
    is not a gateway — and never raises, because this runs inside `wrap()`."""
    try:
        host = urllib.parse.urlsplit(str(base_url or "")).hostname or ""
    except ValueError:
        return False
    return host == RAMP_ROUTER_HOST or host.endswith(RAMP_ROUTER_DOMAIN)


def client_points_at_ramp_router(client: Any) -> bool:
    """Read the client's `base_url` defensively and test it. Both the openai and
    anthropic SDKs expose the constructor's URL as `.base_url` (an httpx.URL); some
    client variants may not, and a property that raises must not break `wrap()`."""
    try:
        base_url = getattr(client, "base_url", "")
    except Exception:  # noqa: BLE001 — a custom client's property may raise
        return False
    return is_ramp_router_base_url(base_url)

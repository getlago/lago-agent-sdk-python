"""Gateway connector code — a second front door into the same billing kernel.

Everything under `lago_agent_sdk.gateway` maps a third-party AI gateway's own
usage-reporting surface (Cloudflare's Logs API, Vercel's Reporting API, ...)
into the SDK's existing `CanonicalUsage` shape. It is consumed by a standalone
poller service, not by `wrap()` — there is no client to monkey-patch here.

This is intentionally a separate namespace from `lago_agent_sdk.adapters`
(which extracts usage from a provider-native response inside a wrapped call).
The two never import from each other; both target `CanonicalUsage`.
"""

from __future__ import annotations

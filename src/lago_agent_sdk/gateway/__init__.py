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

# NO RAMP ROUTER ADAPTER, deliberately, and this is the note saying so rather than an
# omission to rediscover later.
#
# Router is the first gateway this SDK supports that exposes no programmatic usage
# surface at all. Checked against every page of its documentation: the only routes are
# `GET /v1/models`, `POST /v1/responses`, `POST /v1/messages` and
# `POST /v1/messages/count_tokens`. Usage lives in the dashboard's Logs view, and
# `guides/monitor` describes what that view DISPLAYS — model, provider, status, tokens,
# cost, latency, ids, API key, token breakdown, metadata, service tier, fallback
# candidates — without offering any way to fetch it.
#
# An "analytics API" is referenced exactly once in the whole corpus, in the limits table
# on `api/errors-and-limits` ("the analytics API accepts at most 93 days"), with no path,
# no auth and no record shape. That is not enough to build against: an adapter written
# over guessed field names would have tests proving only that it matches the guess, and
# the fixtures behind it would not be captures of anything.
#
# So Router's live `wrap()` path is the whole integration for now. When the analytics
# API is published, the adapter goes here as `extract_ramp_router_log` /
# `resolve_ramp_router_subscription` alongside the Cloudflare pair, keyed for idempotent
# replay off whatever per-record id it exposes. Tracked as LAGO-1853, which also carries
# the questions to ask Router.

# Cloudflare AI Gateway

Point any of the supported clients at your gateway instead of the provider directly — `wrap()` detects it and bills correctly, with two behaviors on top of the plain provider case:

```python
from anthropic import Anthropic
from lago_agent_sdk import LagoSDK

sdk = LagoSDK(api_key="...", default_subscription_id="sub_acme")
client = sdk.wrap(Anthropic(
    api_key="...",
    base_url=f"https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/anthropic",
    default_headers={"cf-aig-authorization": f"Bearer {gateway_auth}"},
))
client.messages.create(model="claude-sonnet-4-6", max_tokens=200, messages=[{"role": "user", "content": "Hello"}])
sdk.flush()
```

- **Gateway cache hits aren't billed.** If the gateway serves a response from its own cache (`cf-aig-cache-status: HIT`), the provider was never called, so the SDK skips emitting for that response.
- **Workers AI gets priced automatically.** Wrap an OpenAI-shaped client against the gateway's `/compat` endpoint (`model="workers-ai/@cf/..."`) with `pricing_mode="price"`, and the SDK fetches Cloudflare's own published Workers AI rates in the background — no separate price table to maintain.

## Backfill from the Logs API

For usage that already happened, backfill straight from the gateway's own Logs API instead of replaying calls — `lago_agent_sdk.gateway.adapters` extracts a log entry into `CanonicalUsage` and bills Cloudflare's own metered `cost` for it, so there's no separate price lookup and re-running over the same window never double-bills:

```python
from lago_agent_sdk.gateway.adapters import extract_cloudflare_log, resolve_subscription

for entry in fetch_gateway_logs():  # GET .../ai-gateway/gateways/{id}/logs
    usage = extract_cloudflare_log(entry)
    sub = resolve_subscription(entry) or "sub_default"  # from the call's cf-aig-metadata, if set
    sdk.emit(usage, subscription=sub, mode="price", usd_cost=entry.get("cost") or 0, event_id=f"cf_{entry['id']}")
sdk.flush()
```

This page is the complete picture; runnable notebooks are kept out of the repo (see `.gitignore`) because their saved cells and outputs carry account identifiers and live subscription ids.

**Gateway-routed calls are billed at the gateway's metered cost.** Cloudflare reports its own `cost` per log entry and the backfill passes that straight through, so Lago reconciles against the dashboard you actually look at. One measured consequence to be aware of: that field excludes additive *reasoning* tokens, so a thinking-heavy Gemini call bills about 4% of what Google charges (verified live at 22.8x on one call, 39.6x on another — the ratio tracks each prompt's thinking-to-output ratio). Cloudflare is exact on input, output, cache-read and cache-write.

**If you hand-roll a poller, don't use `urllib`.** `gateway.ai.cloudflare.com` returns `403` with body `error code: 1010` to `Python-urllib` — its bot-signature check. Any other User-Agent passes, and `requests` (which this SDK uses) is fine. The failure looks like an auth error because the body is otherwise empty.

## Workers AI models with no client to wrap

Cloudflare-hosted models can be reached through the OpenAI-compatible `/compat` endpoint above, but only when they are chat-shaped. Partner models are not: `typesafe/jev` takes `{state, questions}` and refuses a `messages` array, and the official `cloudflare` package cannot address any Workers AI model at all (it percent-encodes the slash in the model name). For these the SDK ships its own one-method client:

```python
ai = sdk.workers_ai(
    account_id,
    cf_api_token,                       # a Cloudflare API token with Workers AI access
    gateway_id=gateway_id,              # optional — see what it buys below
    gateway_auth=gateway_auth,          # the gateway's cf-aig-authorization token, if authentication is on
    subscription="sub_acme",            # default for every call; extra_lago={"subscription": ...} per call
)

# a partner model: the partner's key must be stored under the gateway's Provider Keys (BYOK)
out = ai.run(
    "typesafe/jev",
    {
        "state": "I was charged twice and need the duplicate refunded before Friday.",
        "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "Payments, refunds", "technical": "Bugs, outages"}},
        },
    },
    extra_lago={"dimensions": {"ticket": "T-4821"}},
)
out["result"]["result"]["answers"]["department"]["choice"]   # "billing"

# a catalog model: same client, same billing
ai.run("@cf/meta/llama-3.2-3b-instruct", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50})
sdk.flush()
```

`run()` returns Cloudflare's full response envelope unchanged and raises `WorkersAIError` on an error status (a 402 for a partner model whose key is not stored, a 403 for a model not on your Workers plan), before anything is billed.

**What the gateway buys.** With `gateway_id` set, `@cf/...` calls go through the gateway host: a cache hit (`cf-aig-cache-status: HIT`) is not billed, and every event carries a `cf_log_id` dimension that matches the entry's `id` in the Logs API. Partner models go through the unified `api.cloudflare.com/.../ai/run` path with the gateway named in a header, which is the only route where the gateway's stored partner key is consulted; that path returns no cache header, so the client asks the gateway to skip its cache for those calls rather than risk billing a cached replay. Pass `extra_headers={"cf-aig-skip-cache": "false"}` to opt back in.

**Billing.** Token events from the response's own `usage`: `prompt_tokens`/`completion_tokens` for catalog models, `input_tokens`/`output_tokens` for partner models. Catalog models price from Cloudflare's published Workers AI rates in price mode as usual. Partner models are not in that catalog; their rates come from AI Gateway's own cost table (`GET .../ai-gateway/costs`), fetched per model in the background the first time one is seen. Left alone, the very first call to a partner model in a process bills tokens and reports the miss via `on_error`, and every call after it bills dollars; name the ids up front with `sdk.warm_pricing(["workers-ai"], workers_ai_models=["typesafe/jev"])` and even the first call prices. A fetched rate keeps serving past its TTL while it refreshes, so an expiry never bills a call as tokens. Jev lists input at $0.042 per million with free output, the same rate the gateway stamps as `cost` on its log entries. Under BYOK the gateway's Logs API still reports a `cost` for the partner call at Cloudflare's list price although Cloudflare charged nothing — the entry's `byok` field (surfaced in `extras["byok"]` by `extract_cloudflare_log`) is what tells a backfill not to bill it. The id billed is the one you requested, because that is what Cloudflare's price catalog is keyed by; the name the model reports (`jev-1.13.0`, `...-24b-v2`) is kept in `extras["served_model"]`.

Streaming is not supported by this client; use the `/compat` endpoint through a wrapped OpenAI client for streamed chat.

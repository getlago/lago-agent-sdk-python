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

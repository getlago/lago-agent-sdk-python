# Databricks AI Gateway

Unlike [Cloudflare](cloudflare.md), Databricks has **no unified endpoint** — each provider is reachable only through its own native surface, and two of them use the same `openai.OpenAI` class. Which `base_url` you point at decides how the call is priced.

**Databricks-hosted foundation models** (`system.ai.*`) — billed by Databricks in DBUs:

```python
from openai import OpenAI
from lago_agent_sdk import LagoSDK

sdk = LagoSDK(api_key="...", default_subscription_id="sub_acme")
client = sdk.wrap(OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"{DATABRICKS_HOST}/ai-gateway/mlflow/v1",
    default_headers={"Databricks-Ai-Gateway-Request-Tags": json.dumps({"lago_subscription": "sub_acme"})},
))
client.chat.completions.create(model="system.ai.llama-4-maverick", messages=[{"role": "user", "content": "Hi"}])
```

**Your own vendor key (BYOK)** — Anthropic via its native passthrough, note `api_key="unused"` because the real credential goes in `Authorization`, and the Unity Catalog connection holding your Anthropic key is named in `Databricks-Model-Provider-Service`:

```python
from anthropic import Anthropic
client = sdk.wrap(Anthropic(
    api_key="unused",
    base_url=f"{DATABRICKS_HOST}/ai-gateway/anthropic",
    default_headers={
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Databricks-Model-Provider-Service": "workspace.default.anthropickey",
    },
))
```

OpenAI BYOK is the same `OpenAI` class as the hosted example, against `/ai-gateway/openai/v1` with its own `Databricks-Model-Provider-Service`.

## What gets billed

| Path | Live `wrap()` | Backfill |
|---|---|---|
| BYOK (OpenAI / Anthropic) | **dollar cost**, priced from the vendor's published rates | dollar cost from Databricks' own `external_model_spend` |
| Hosted (`system.ai.*`) | **token counts** | **token counts** |

BYOK prices live because you pay the vendor directly, so the vendor's rate *is* your cost — verified against Databricks' own metered spend on 38 of 38 real buckets, exactly. Hosted models bill in DBUs against a rate card published only as HTML and present in no system table, so there is no rate to look up: those calls emit token counts instead of a dollar cost. That is the complete answer for them, not a degraded one, so it is **not** reported as an error — `TOKEN_BILLED_PROVIDERS` lists the providers this applies to, and the SDK notes it once per model at info level rather than warning on every call. A genuine price miss — a cold table, an unmatched model name — still reports through `on_error` as before.

**Hosted dollars exist, and are deliberately not billed from.** `system.billing.usage` × `list_prices` (or `account_prices` for your contract rate) does yield exact USD per hour and endpoint. It is not used because it comes from a *different Databricks screen* than the gateway view: it carries no `request_tags`, so per-subscription splits would be ours rather than Databricks', and it lags the gateway by roughly a day — measured at ~19h on a live workspace. Every number this connector sends is one you can find on a Databricks **gateway** page, which is the property that makes it checkable.

**Grouping matches the Databricks page.** Each backfilled event carries the grouping key of the surface it came from — `endpoint_name` for hosted, `bucket` (the hour) for BYOK. Group Lago by `endpoint_name` and you get the AI Gateway → Usage table row for row. Pass `dimensions={...}` to add your own keys; yours win on a name collision.

**Don't run the live path and the backfill over the same hosted traffic.** Both emit token events, with different `transaction_id`s, so Lago accepts both and the counts double. Pick one per traffic stream: `wrap()` for real time, the backfill for completeness.

`Databricks-Ai-Gateway-Request-Tags` is what makes attribution work. It lands in `request_tags` on `system.ai_gateway.usage` **and** is a first-class aggregation dimension on `external_model_spend`, so tagging `lago_subscription` means BYOK cost arrives already split per subscription — no apportioning needed.

## Backfill — give it a window, it does the rest

```python
from lago_agent_sdk.gateway.databricks import DatabricksSource

source = DatabricksSource.from_env()   # DATABRICKS_HOST / _TOKEN / _WAREHOUSE_ID
print(sdk.backfill_databricks(source, "7 days", default_subscription="sub_default"))
sdk.flush()
# {'cost': 60, 'tokens': 47, 'skipped': 0}
```

Pass a `datetime` instead of `"7 days"` for an exact lower bound, and `unified=True` to bill the whole window to `default_subscription` regardless of per-call tags.

The window reads **whole closed hours only**: it is floored to the hour at both ends and the current, still-aggregating hour is excluded, because `external_model_spend` is an hourly aggregate whose row for an hour does not exist until that hour closes. So the newest hour of traffic arrives on the next run — pass a window comfortably wider than your run interval, since this reader keeps no cursor.

Unlike Cloudflare's single paginated GET, this one is worth having in the SDK — hand-rolling it is ~100 lines with three money-losing traps in them. The Statement Execution API returns only **chunk 0** inline, so a wide window silently truncates and bills a fraction of it with no error. A BYOK call appears in **both** `ai_gateway.usage` and `external_model_spend`, so billing both charges twice. And `transaction_id` is unique account-wide, so an unscoped row id blocks that row from ever reaching a second subscription.

To inspect a window before billing it, or to route rows yourself, read them directly — each row is already shaped for `emit()`:

```python
for row in source.read_usage("7 days"):
    print(row.usage.model, row.subscription, row.usd_cost)  # usd_cost is None for hosted
```

Reading the system tables needs a PAT with the **`sql`** scope plus a SQL warehouse — the live calls above need neither. The pure `extract_databricks_log(row)` / `resolve_databricks_subscription(row)` functions stay available from `lago_agent_sdk.gateway.adapters` if you already have rows from `databricks-sql-connector` or your own warehouse job.

**One cost note:** a SQL warehouse is a real cost centre. Measured on a test workspace, the warehouse queries cost roughly 1,500× the model-serving usage they were reporting on. Run the backfill as one query over a wide window, never as a tight polling loop.

## Gotchas worth knowing

- **`gpt-oss` models inflate input by ~100 tokens** from a server-injected preamble — a 2-character prompt bills 102. Not an SDK error.
- **`claude-opus-4-5` does not cache through this gateway**: reproducibly `cache_read`/`cache_write` of 0 with the full prompt billed as input, on a request shape where `claude-sonnet-4-5` caches fine. An opus customer silently gets no cache discount.
- **Hosted models report three different name strings.** `system.ai.llama-4-maverick` and `databricks-llama-4-maverick` both work as requests, and the response echoes a third (`meta-llama-4-maverick-040225`). Pricing keys off the resolved name, so reconciling by requested id will not line up.
- **Embeddings** work on `/ai-gateway/mlflow/v1/embeddings` and report input only — no `completion_tokens` at all.

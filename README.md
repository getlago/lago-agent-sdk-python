# lago-agent-sdk

Instrument LLM clients and emit usage events to [Lago](https://www.getlago.com) for billing.

```text
                  ┌──────────────┐
your code ──────► │ wrapped client│ ──► provider (Bedrock / Mistral / …)
                  └──────┬───────┘
                         │ (extract usage)
                         ▼
                  ┌──────────────┐
                  │  Lago events │ ──► api.getlago.com
                  └──────────────┘
```

## What it does

- Wraps your existing LLM client in place — no API surface change for your application code.
- Extracts usage from each response into a normalized shape (`CanonicalUsage`).
- Buffers events in memory, flushes them in batches to Lago's `/events/batch` endpoint.
- Survives provider/Lago outages with exponential backoff and a bounded buffer.
- p99 wrap-overhead under 5 ms — your call is never blocked on Lago.

## Install

```bash
pip install lago-agent-sdk
```

For Bedrock support: `pip install 'lago-agent-sdk[bedrock]'` (adds `boto3`).
For Mistral support: `pip install 'lago-agent-sdk[mistral]'` (adds `mistralai`).
For Anthropic native support: `pip install 'lago-agent-sdk[anthropic]'` (adds `anthropic`).
For OpenAI native support: `pip install 'lago-agent-sdk[openai]'` (adds `openai`).
For Gemini native support: `pip install 'lago-agent-sdk[gemini]'` (adds `google-genai`).

## Quickstart — Bedrock

```python
import boto3
from lago_agent_sdk import LagoSDK

sdk = LagoSDK(
    api_key="<YOUR_LAGO_API_KEY>",
    api_url="https://api.getlago.com/api/v1/",
    default_subscription_id="sub_acme",
)
client = sdk.wrap(boto3.client("bedrock-runtime", region_name="eu-west-1"))

resp = client.converse(
    modelId="eu.amazon.nova-lite-v1:0",
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
)
sdk.flush()
```

The wrapped client behaves identically to the original — same arguments, same return shape, same exceptions. The SDK adds an in-memory queue that batches events to Lago in the background.

## Quickstart — Anthropic

```python
from anthropic import Anthropic
from lago_agent_sdk import LagoSDK

sdk = LagoSDK(api_key="...", default_subscription_id="sub_acme")
client = sdk.wrap(Anthropic(api_key="..."))

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=200,
    messages=[{"role": "user", "content": "Hello"}],
)
sdk.flush()
```

Works with `Anthropic` and `AsyncAnthropic`. Both `messages.create(..., stream=True)` and the `messages.stream(...)` context manager are instrumented — usage is captured from the final `message_delta` event in either case.

## Quickstart — Mistral

```python
from mistralai.client import Mistral
from lago_agent_sdk import LagoSDK

sdk = LagoSDK(api_key="...", default_subscription_id="sub_acme")
client = sdk.wrap(Mistral(api_key="..."))

resp = client.chat.complete(
    model="mistral-small-latest",
    messages=[{"role": "user", "content": "Hello"}],
)
sdk.flush()
```

## Quickstart — OpenAI

```python
from openai import OpenAI
from lago_agent_sdk import LagoSDK

sdk = LagoSDK(api_key="...", default_subscription_id="sub_acme")
client = sdk.wrap(OpenAI(api_key="..."))

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    max_completion_tokens=200,
)
sdk.flush()
```

Works with `OpenAI` and `AsyncOpenAI`. Covers both **Chat Completions** (`client.chat.completions.create`) and the newer **Responses API** (`client.responses.create`), sync + streaming. For streaming, the wrapper auto-injects `stream_options={"include_usage": True}` so the final chunk carries usage data — without it OpenAI emits no usage on streamed responses.

**Reasoning tokens** (`llm_reasoning_tokens`) populate automatically when you call an o-series model (`o4-mini`, `o1`, etc.) — OpenAI is the first provider to expose this metric separately.

## Quickstart — Gemini

```python
from google import genai
from lago_agent_sdk import LagoSDK

sdk = LagoSDK(api_key="...", default_subscription_id="sub_acme")
client = sdk.wrap(genai.Client(api_key="..."))

resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello",
)
sdk.flush()
```

Wraps the modern `google-genai` SDK (`from google import genai`). Covers `client.models.generate_content` + `generate_content_stream`, sync + async (via `client.aio.models`).

**Reasoning tokens** populate automatically on Gemini 2.5 — the model reasons internally by default and surfaces `thoughts_token_count` (see the note on reasoning semantics below).

## Cloudflare AI Gateway

Point any of the clients above at your gateway instead of the provider directly — `wrap()` detects it and bills correctly, with two behaviors on top of the plain provider case:

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

For usage that already happened, backfill straight from the gateway's own Logs API instead of replaying calls — `lago_agent_sdk.gateway.adapters` extracts a log entry into `CanonicalUsage` and bills Cloudflare's own metered `cost` for it, so there's no separate price lookup and re-running over the same window never double-bills:

```python
from lago_agent_sdk.gateway.adapters import extract_cloudflare_log, resolve_subscription

for entry in fetch_gateway_logs():  # GET .../ai-gateway/gateways/{id}/logs
    usage = extract_cloudflare_log(entry)
    sub = resolve_subscription(entry) or "sub_default"  # from the call's cf-aig-metadata, if set
    sdk.emit(usage, subscription=sub, mode="price", usd_cost=entry.get("cost") or 0, event_id=f"cf_{entry['id']}")
sdk.flush()
```

The sections above are the complete picture; runnable notebooks are kept out of the repo (see `.gitignore`) because their saved cells and outputs carry account identifiers and live subscription ids.

**Gateway-routed calls are billed at the gateway's metered cost.** Cloudflare reports its own `cost` per log entry and the backfill passes that straight through, so Lago reconciles against the dashboard you actually look at. One measured consequence to be aware of: that field excludes additive *reasoning* tokens, so a thinking-heavy Gemini call bills about 4% of what Google charges (verified live at 22.8x on one call, 39.6x on another — the ratio tracks each prompt's thinking-to-output ratio). Cloudflare is exact on input, output, cache-read and cache-write.

**If you hand-roll a poller, don't use `urllib`.** `gateway.ai.cloudflare.com` returns `403` with body `error code: 1010` to `Python-urllib` — its bot-signature check. Any other User-Agent passes, and `requests` (which this SDK uses) is fine. The failure looks like an auth error because the body is otherwise empty.

## Databricks AI Gateway

Unlike Cloudflare, Databricks has **no unified endpoint** — each provider is reachable only through its own native surface, and two of them use the same `openai.OpenAI` class. Which `base_url` you point at decides how the call is priced.

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

### What gets billed

| Path | Live `wrap()` | Backfill |
|---|---|---|
| BYOK (OpenAI / Anthropic) | **dollar cost**, priced from the vendor's published rates | dollar cost from Databricks' own `external_model_spend` |
| Hosted (`system.ai.*`) | **token counts** | **token counts** |

BYOK prices live because you pay the vendor directly, so the vendor's rate *is* your cost — verified against Databricks' own metered spend on 38 of 38 real buckets, exactly. Hosted models bill in DBUs against a rate card published only as HTML and present in no system table, so there is no rate to look up: those calls emit token counts instead of a dollar cost. That is the complete answer for them, not a degraded one, so it is **not** reported as an error — `TOKEN_BILLED_PROVIDERS` lists the providers this applies to, and the SDK notes it once per model at info level rather than warning on every call. A genuine price miss — a cold table, an unmatched model name — still reports through `on_error` as before.

**Hosted dollars exist, and are deliberately not billed from.** `system.billing.usage` × `list_prices` (or `account_prices` for your contract rate) does yield exact USD per hour and endpoint. It is not used because it comes from a *different Databricks screen* than the gateway view: it carries no `request_tags`, so per-subscription splits would be ours rather than Databricks', and it lags the gateway by roughly a day — measured at ~19h on a live workspace. Every number this connector sends is one you can find on a Databricks **gateway** page, which is the property that makes it checkable.

**Grouping matches the Databricks page.** Each backfilled event carries the grouping key of the surface it came from — `endpoint_name` for hosted, `bucket` (the hour) for BYOK. Group Lago by `endpoint_name` and you get the AI Gateway → Usage table row for row. Pass `dimensions={...}` to add your own keys; yours win on a name collision.

**Don't run the live path and the backfill over the same hosted traffic.** Both emit token events, with different `transaction_id`s, so Lago accepts both and the counts double. Pick one per traffic stream: `wrap()` for real time, the backfill for completeness.

`Databricks-Ai-Gateway-Request-Tags` is what makes attribution work. It lands in `request_tags` on `system.ai_gateway.usage` **and** is a first-class aggregation dimension on `external_model_spend`, so tagging `lago_subscription` means BYOK cost arrives already split per subscription — no apportioning needed.

### Backfill — give it a window, it does the rest

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

### Gotchas worth knowing

- **`gpt-oss` models inflate input by ~100 tokens** from a server-injected preamble — a 2-character prompt bills 102. Not an SDK error.
- **`claude-opus-4-5` does not cache through this gateway**: reproducibly `cache_read`/`cache_write` of 0 with the full prompt billed as input, on a request shape where `claude-sonnet-4-5` caches fine. An opus customer silently gets no cache discount.
- **Hosted models report three different name strings.** `system.ai.llama-4-maverick` and `databricks-llama-4-maverick` both work as requests, and the response echoes a third (`meta-llama-4-maverick-040225`). Pricing keys off the resolved name, so reconciling by requested id will not line up.
- **Embeddings** work on `/ai-gateway/mlflow/v1/embeddings` and report input only — no `completion_tokens` at all.

## Snowflake Cortex

Snowflake serves Cortex two ways, and they need two different halves of this SDK. That split is the thing to understand before anything else here.

| Surface | How you call it | How Lago sees it |
|---|---|---|
| **Cortex REST** — `/api/v2/cortex/v1` | an OpenAI-compatible client you hand to `wrap()` | live, per call |
| **AI SQL functions** — `AI_COMPLETE`, `AI_EMBED`, … | SQL, inside the warehouse | **backfill only** — there is no client to wrap |

**Everything on this path bills as token counts.** Snowflake meters Cortex in credits against a rate card that lives in no view, so there is no per-request dollar figure to pass through and no price mode here. `provider` is `"snowflake"`, which is listed in `TOKEN_BILLED_PROVIDERS`, so a customer running `pricing_mode="price"` globally still gets token events for Snowflake rows — with no price-miss error, because a structural absence of a rate card is not a lookup failure.

### Live — the REST surface

```python
import os
from openai import OpenAI

client = sdk.wrap(
    OpenAI(
        base_url=f"https://{os.environ['SNOWFLAKE_ACCOUNT']}.snowflakecomputing.com/api/v2/cortex/v1",
        api_key=os.environ["SNOWFLAKE_PAT"],
    )
)
```

The `base_url` is what identifies these calls as Snowflake rather than OpenAI — an OpenAI-shaped endpoint says nothing about whose tokens they are. Use `max_completion_tokens`; Cortex rejects `max_tokens` outright.

**Cortex's `cached_tokens` is additive, the opposite of OpenAI's convention.** A cached call reports `prompt_tokens: 7`, `cached_tokens: 8745`, `completion_tokens: 6`, `total_tokens: 8758` — the cached block is *not* inside `prompt_tokens`. Caching also only happens when you send an explicit `cache_control: {"type": "ephemeral"}` content part; the same prompt twice without one reports zero cached both times.

**The wire cannot tell a cache creation from a read; the view can.** A creation call reports the same `cached_tokens` with `cache_write_tokens: 0`, so the live `wrap()` path bills a creation as a cache read (`llm_cached_input_tokens`). The REST view records the same call as `cache_write_input`, so a backfilled row bills `llm_cache_creation_tokens` instead. Verified live on a matched pair (INT-230): identical wire usage, one `cache_write_input` row and three `cache_read_input` rows. If you price creation and read differently, know that live-path traffic reports everything at the read metric. It also bounds the REST-view dedup: a backfilled creation row emits its cached block under a *different* transaction id than the live path did (`_tok_cache_write` vs `_tok_cache_read`), so that one component bills on both metrics if you backfill a live-billed window — the call's input and output stay deduplicated.

### Backfill — the SQL functions surface

```python
from lago_agent_sdk.gateway.snowflake import SnowflakeSource

source = SnowflakeSource.from_env()  # SNOWFLAKE_ACCOUNT / _PAT, plus a warehouse
print(sdk.backfill_snowflake(source, "7 days", default_subscription="sub_default"))
sdk.flush()
# {'tokens': 47, 'skipped': 0}
```

Two counts, and there cannot be more: `tokens` is what got billed, `skipped` is what did not. Both causes of a skip are also reported through `on_error` with `where="backfill"`, so an automated caller notices a gap without inspecting the return value.

**It reads the functions view only.** The REST view reports the calls `wrap()` already billed above. Both sides derive one idempotency key from the call's `REQUEST_ID` — the wrapper reads it off the `x-snowflake-request-id` response header — so Lago rejects a backfill's copies as duplicate `transaction_id`s instead of billing them twice. That protection holds only when the backfill uses the default `event_id_prefix` and resolves the same subscription the live path billed, and it does not cover a cache-creation call's cached block (see the cache note above) or calls billed without the header. So the rule stands: pass `views=("rest",)` only for REST traffic `wrap()` never saw:

```python
sdk.backfill_snowflake(
    source,
    "7 days",
    default_subscription="sub_default",
    views=("rest",),  # ONLY if no wrapped client is billing this traffic
)
```

The window reads **whole closed hours only** — floored at both ends, with the current hour excluded, because a bucket is not complete until its hour closes and billing it early burns that row's idempotency key so the correction is rejected as a duplicate. The newest hour therefore arrives on the next run: pass a window comfortably wider than your run interval, since this reader keeps no cursor. One more boundary Lago itself draws: events stamped before the subscription started are **accepted and silently never billed** — a window reaching back past the subscription's start reports its rows as billed while nothing lands in usage, so start backfills at the subscription's start date.

**Attribution comes from `QUERY_TAG`**, the only customer-injectable key on either view, and the same `lago_subscription` key Cloudflare and Databricks read from their own metadata:

```sql
ALTER SESSION SET QUERY_TAG = '{"lago_subscription": "sub_123"}';
SELECT AI_COMPLETE('claude-sonnet-4-5', 'summarize this');
```

By default the tag is the **only** attribution source: an untagged row falls to `default_subscription`, and to a skip (counted, reported) when there is none. `role_names` and `user_id` are opt-in — `subscription_order=("query_tag", "role_names")` — for accounts that really map one Snowflake role or user to one customer. They are not in the default because every live row carries both, so they would swallow untagged rows and bill them to a Snowflake identity instead of your default: that is a wrong subscription, and unlike a skip it is not recoverable.

**Grouping matches the view.** Each event carries the grouping key of the surface it came from — `function_name` + `model_name` for functions rows, `inference_region` for REST — so a `GROUP BY` on the view and the same grouping in Lago line up. `dimensions={...}` adds your own; yours win on a collision.

### A long-running query is deferred, not guessed at

`IS_COMPLETED` means "did the query finish *in this aggregation window*", and these views are hour-bucketed — Snowflake documents a query running 5:30→8:30 writing **four rows, one per hour, all sharing one `QUERY_ID`**. Two things follow. The key `{prefix}_{kind}_{sub}_{QUERY_ID}` collides across those rows, and whether each row's `METRICS` is incremental or cumulative is unmeasured. On a 3-hour query using 3,800 input tokens, summing four incremental rows bills 3,800 and summing four cumulative ones bills 9,500; billing only the last row gives 3,800 if cumulative and 900 if incremental.

So a `QUERY_ID` that yields more than one row in a window is **not billed**. It is counted in `skipped`, reported through `on_error`, listed on `source.deferred_rows`, and billable once the shape is settled. Guessing over-bills by 2.5× or under-bills by 76%, neither recoverable once invoiced. Every query ever observed on a real account finished inside one bucket, so this fires on a shape nobody has seen.

To inspect a window before billing it, read the rows directly — `read_usage()` is a generator and each row is already shaped for `emit()`:

```python
for row in source.read_usage("7 days"):
    print(row.kind, row.usage.model, row.subscription, row.occurred_at)
```

### Setting up the account

Reading the views needs a PAT plus a **running warehouse**; the live calls above need neither. Four things block a first-time setup and none of them says so clearly:

- **Model access moved to RBAC.** `CORTEX_MODELS_ALLOWLIST` is deprecated and accepts only `'NONE'`; you need `GRANT APPLICATION ROLE SNOWFLAKE."CORTEX-MODEL-ROLE-ALL" TO ROLE …`, without which the role can call zero models.
- **A PAT's `ROLE_RESTRICTION` is a quoted string literal**, so it is case-sensitive — `'LAGO_CORTEX_ROLE'`, not the lowercase spelling that works everywhere else.
- **A warehouse with `AUTO_RESUME = FALSE`** fails every statement with "warehouse is suspended", which reads like a privilege error.
- **A PAT cannot authenticate without an active network policy.** Prefer reusing a broad one: recovering from an IP lockout needs Snowflake Support, with no self-service path back.

Error code `003001` has four distinct causes — account entitlement, unknown model, model not granted to the role, and a bare fine-tuned model name — so it is not diagnostic on its own.

**One cost note:** a SQL warehouse is a real cost centre. Measured on the equivalent Databricks setup, warehouse queries cost roughly 1,500× the model-serving usage they reported on. Run the backfill as one query over a wide window, never as a tight polling loop.

The pure `extract_snowflake_functions_log(row)` / `extract_snowflake_rest_log(row)` / `resolve_snowflake_subscription(row)` functions stay available from `lago_agent_sdk.gateway.adapters` if you already have rows from your own warehouse job.

## Ramp Router

[Ramp Router](https://router.com) is an OpenAI-Responses-compatible gateway in front of OpenAI, Anthropic, Google Vertex, Fireworks and xAI. Point an OpenAI client at it and `wrap()` detects it from the `base_url` — no other code change:

```python
from openai import OpenAI
from lago_agent_sdk import LagoSDK

sdk = LagoSDK(api_key=os.environ["LAGO_API_KEY"])
client = sdk.wrap(
    OpenAI(
        api_key=os.environ["RAMP_ROUTER_API_KEY"],
        base_url="https://api.router.com/v1",
    ),
    subscription="sub_acme",
)

# A model id from GET /v1/models — Router's ids are account-specific.
client.responses.create(model=os.environ["RAMP_ROUTER_MODEL"], input="Summarize this invoice.")
sdk.flush()
```

- **The model that answered is the one billed.** Router diverges from what you asked for in two ways: a `models` fallback list sends no `model` field at all, and Switchyard routing can serve a different model than the one requested. The SDK bills the model the response reports, and records the served provider and any pinned service tier in `extras`.
- **`provider:provider-model[:service-tier]` candidates are parsed.** `openai:gpt-5.4-mini:flex` bills as model `gpt-5.4-mini` with `service_tier: "flex"`, so a Router-served model rolls up in Lago against the same name a direct call to it reports.
- **Streaming bills once**, from the terminal usage event. `models` fallback, buffered and streamed calls all work unmodified — note the typed Python client rejects the non-standard `models` kwarg, so a fallback list goes through `extra_body={"models": [...]}` (verified live; the served model bills either way).

Attribution works the same way as anywhere else in this SDK — `subscription` at wrap time or `extra_lago={"subscription": ...}` per call. Separately, it is worth putting the same id in Router's own `metadata` field, which Router stores with its usage record and shows in the request detail:

```python
client.responses.create(
    model=os.environ["RAMP_ROUTER_MODEL"],
    input="Summarize this invoice.",
    # Router stores this with its usage record. The SDK does not send it for you.
    metadata={"lago_subscription": "sub_acme"},
)
```

That costs nothing today and is what a backfill would key off later.

### Price mode falls back to token events

In `pricing_mode="price"`, a Router call currently emits **token events rather than `llm_cost`**. That is deliberate, not a gap, and it is worth understanding before turning price mode on for Router traffic:

- **A BYOK-served request costs $0 through Router.** Router's own words: "When a request is served with your provider key, your provider bills you directly and Ramp Router does not charge for that usage." Nothing in the response says which key served it, so pricing at list rate would bill a customer the full amount for usage Router never charged for.
- **A non-default service tier does not bill at the published base rate.** Router's catalog says outright that "service tiers, long contexts, caching, and other features may use different rates", and that a Fast tier's "pricing may differ from the base rates shown here". Billing flex at the standard rate over-bills.
- **The token overlap semantics are Router's own, and they are measured.** OpenAI counts cached tokens inside `input_tokens` and reasoning inside `output_tokens`; Anthropic counts both additively. Router normalizes the _numbers_ to OpenAI's convention, not just the schema — verified live on an Anthropic-served model, where a warm `cache_control` call reported the cached block inside an unchanged `input_tokens`. But that convention is Router's, not the served vendor's, which is exactly why the served vendor must not be stamped as the provider: it would de-overlap with the wrong rules whenever its native convention differs.

So Router is treated as a provider of its own, matching no vendor in the price tables. Token mode — the default — is exact and unaffected: it emits the counts Router reported, the same per-field `llm_*` events a direct provider call produces. Price mode takes a clean miss and falls back to those same token events, with no error on your call path. Bill Router traffic in token mode and price it with a Lago plan.

### Measured behaviours and limitations

- **There is no backfill path**, because Router exposes no programmatic usage surface. Its only routes are `GET /v1/models`, `POST /v1/responses`, `POST /v1/messages` and `POST /v1/messages/count_tokens`; usage lives in the dashboard's Logs view. An "analytics API" is mentioned once in Router's limits table with no path, auth or record shape. Unlike the Cloudflare connector above, there is no Logs API loop to show you.
- **Nothing is skipped as a gateway cache hit**, because Router has no response cache: "Self-service Router response caching, which would reuse an entire previous response without calling a model provider, is a separate optimization and is not currently configurable." Provider _prompt_ caching does pass through, and those cache-read and cache-write tokens are billed like any other.
- **Router's Anthropic Messages surface is not instrumented yet.** `POST /v1/messages` exists and routes to the same providers, so pointing a wrapped `Anthropic` client at `https://api.router.com/v1` will _work_ as an LLM client but bill nothing. Use the Responses surface until that lands.
- **A proxy in front of Router is not detected.** Detection matches the `api.router.com` host (and `*.router.com`). Reaching Router through your own hostname bills as plain OpenAI, with the wrong provider and an unparsed model id.
- **`api.router.com` sits behind bot management.** A rejected client can get an HTML challenge page rather than Router's documented JSON error envelope. The SDK degrades to zero usage rather than throwing, either way.
- **Failures never bill.** Every documented status — including 402 `insufficient_credits`, 429, and 502 `all_candidates_failed` — emits nothing, as does a response reporting zero usage.

## Multi-tenant — pick a subscription per call

Three ways to set the `external_subscription_id`, in priority order:

```python
# 1. Per-call override (highest precedence)
client.converse(..., extra_lago={"subscription": "sub_acme", "dimensions": {"feature": "summarize"}})

# 2. Context-bound (use in middleware to set once per request)
sdk.set_subscription("sub_acme")
# all calls in this thread/asyncio task → sub_acme

# 3. Default at init (fallback)
sdk = LagoSDK(api_key="...", default_subscription_id="sub_default")
```

Backed by `contextvars` for safe propagation across `asyncio` tasks.

## Supported providers

| Provider | Access | Status |
|---|---|---|
| AWS Bedrock | `Converse` (sync + stream) | ✓ |
| AWS Bedrock | `InvokeModel` (sync + stream), 7 model families | ✓ |
| Anthropic | native SDK (`messages.create` + `messages.stream`, sync + async) | ✓ |
| Mistral | native SDK (`chat.complete` + `chat.stream`) | ✓ |
| OpenAI | native SDK (`chat.completions.create` + `responses.create`, sync + async + stream) | ✓ |
| Google Gemini | native SDK (`google-genai`: `models.generate_content` + `generate_content_stream`, sync + async) | ✓ |

## Token dimensions captured

`CanonicalUsage` carries 11 numeric fields. Which ones populate depends on the provider:

| Field | Lago metric code | Bedrock | Anthropic | Mistral | OpenAI | Gemini |
|---|---|---|---|---|---|---|
| input | `llm_input_tokens` | ✓ | ✓ | ✓ | ✓ | ✓ |
| output | `llm_output_tokens` | ✓ | ✓ | ✓ | ✓ | ✓ |
| cache_read | `llm_cached_input_tokens` | ✓ (Anthropic) | ✓ | ✓ (when cache hits) | ✓ (auto-cache) | ✓ (CachedContent API) |
| cache_write | `llm_cache_creation_tokens` | ✓ (Anthropic) | ✓ | ✗ | ✗ | ✗ |
| cache_write_5m / 1h | `llm_cache_write_5m/1h_tokens` | ✓ (Anthropic InvokeModel) | ✓ | ✗ | ✗ | ✗ |
| reasoning | `llm_reasoning_tokens` | ✗ (folded into output) | ✗ (folded into output, even with extended thinking) | ✗ (folded into output) | **✓ (o-series, subset)** | **✓ (Gemini 2.5, additive)** |
| tool_calls | `llm_tool_calls` | ✓ | ✓ | ✓ | ✓ | ✓ |
| audio_input | `llm_audio_input_tokens` | ✗ | ✗ | ✗ | ✓ (GPT-4o-audio) | ✓ (multimodal AUDIO) |
| audio_output | `llm_audio_output_tokens` | ✗ | ✗ | ✗ | ✓ (GPT-4o-audio) | ✓ (multimodal AUDIO) |
| image_input | `llm_image_input_tokens` | ✗ | ✗ | ✗ | ✗ | ✓ (multimodal IMAGE) |

**Reasoning:** OpenAI's `reasoning_tokens` is a *subset* of `output` (already counted in `completion_tokens`). Gemini's `thoughts_token_count` is *additive* to `output` (`candidates + thoughts = total billable output`).

**Cache/audio/image on OpenAI and Gemini are subsets of `input`, not additive.** Both providers count cached/audio/image tokens *within* their input total, so summing `llm_input_tokens + llm_cached_input_tokens` (or `+ audio/image`) double-counts. Bill on `llm_input_tokens` alone; use the breakdown fields only for cost attribution (e.g. a discounted cache rate).

## Pricing mode — send dollar cost instead of tokens

By default the SDK emits **token counts** (`pricing_mode="tokens"`). Set `pricing_mode="price"` to instead emit the **dollar cost** of each call: `Σ(unit_price_per_token × tokens) × markup`.

```python
from lago_agent_sdk import LagoSDK, LagoConfig

sdk = LagoSDK(api_key="...", config=LagoConfig(
    api_key="...",
    default_subscription_id="sub_123",
    pricing_mode="price",     # "tokens" (default) | "price"
    markup=1.2,               # optional cost multiplier (1.2 = +20%)
))
client = sdk.wrap(anthropic_client)
# ... use the client normally ...
```

Price mode emits one `llm_cost` event per priced field (input, output, cache, ...), each carrying `precise_total_amount_cents` for Lago's **dynamic charge model** plus a `token_type` property so a single billable metric can be grouped by both `model` and `token_type`. Prices come from public sources (OpenRouter for native providers, the AWS Bedrock price list for Bedrock), fetched and cached in the background — your LLM call is never blocked on pricing. If a price isn't available yet, the SDK falls back to token-count events and reports via `on_error` rather than under-billing.

Per-call override via `extra_lago`:

```python
client.messages.create(model="claude-...", messages=[...],
                        extra_lago={"mode": "price", "markup": 1.5})
```

## Error policy

The SDK never breaks your LLM call. If anything in instrumentation fails (adapter bug, Lago down, network error, no subscription resolved), it's swallowed, logged, and your call returns normally. Wire your own observability via `LagoConfig.on_error`:

```python
from lago_agent_sdk import LagoConfig, LagoSDK

def on_error(exc: Exception, where: str) -> None:
    sentry.capture_exception(exc, tags={"sdk_phase": where})

sdk = LagoSDK(
    api_key="...",
    config=LagoConfig(api_key="...", on_error=on_error),
)
```

## Setting up Lago

The SDK ships with default metric codes (`llm_input_tokens`, `llm_output_tokens`, etc.). You need to register matching billable metrics in your Lago tenant before events count toward charges. See [Lago docs — Billable Metrics](https://docs.getlago.com/api-reference/billable-metrics/create).

## Development

```bash
git clone https://github.com/getlago/lago-agent-sdk-python
cd lago-agent-sdk-python
python -m venv venv && source venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Security

Found a vulnerability? See [SECURITY.md](SECURITY.md).

## License

[MIT LICENSE](LICENSE).

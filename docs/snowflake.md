# Snowflake Cortex

Snowflake serves Cortex two ways, and they need two different halves of this SDK. That split is the thing to understand before anything else here.

| Surface | How you call it | How Lago sees it |
|---|---|---|
| **Cortex REST** — `/api/v2/cortex/v1` | an OpenAI-compatible client you hand to `wrap()` | live, per call |
| **AI SQL functions** — `AI_COMPLETE`, `AI_EMBED`, … | SQL, inside the warehouse | **backfill only** — there is no client to wrap |

**Everything on this path bills as token counts.** Snowflake meters Cortex in credits against a rate card that lives in no view, so there is no per-request dollar figure to pass through and no price mode here. `provider` is `"snowflake"`, which is listed in `TOKEN_BILLED_PROVIDERS`, so a customer running `pricing_mode="price"` globally still gets token events for Snowflake rows — with no price-miss error, because a structural absence of a rate card is not a lookup failure.

## Live — the REST surface

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

## Backfill — the SQL functions surface

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

## A long-running query is deferred, not guessed at

`IS_COMPLETED` means "did the query finish *in this aggregation window*", and these views are hour-bucketed — Snowflake documents a query running 5:30→8:30 writing **four rows, one per hour, all sharing one `QUERY_ID`**. Two things follow. The key `{prefix}_{kind}_{sub}_{QUERY_ID}` collides across those rows, and whether each row's `METRICS` is incremental or cumulative is unmeasured. On a 3-hour query using 3,800 input tokens, summing four incremental rows bills 3,800 and summing four cumulative ones bills 9,500; billing only the last row gives 3,800 if cumulative and 900 if incremental.

So a `QUERY_ID` that yields more than one row in a window is **not billed**. It is counted in `skipped`, reported through `on_error`, listed on `source.deferred_rows`, and billable once the shape is settled. Guessing over-bills by 2.5× or under-bills by 76%, neither recoverable once invoiced. Every query ever observed on a real account finished inside one bucket, so this fires on a shape nobody has seen.

To inspect a window before billing it, read the rows directly — `read_usage()` is a generator and each row is already shaped for `emit()`:

```python
for row in source.read_usage("7 days"):
    print(row.kind, row.usage.model, row.subscription, row.occurred_at)
```

## Setting up the account

Reading the views needs a PAT plus a **running warehouse**; the live calls above need neither. Four things block a first-time setup and none of them says so clearly:

- **Model access moved to RBAC.** `CORTEX_MODELS_ALLOWLIST` is deprecated and accepts only `'NONE'`; you need `GRANT APPLICATION ROLE SNOWFLAKE."CORTEX-MODEL-ROLE-ALL" TO ROLE …`, without which the role can call zero models.
- **A PAT's `ROLE_RESTRICTION` is a quoted string literal**, so it is case-sensitive — `'LAGO_CORTEX_ROLE'`, not the lowercase spelling that works everywhere else.
- **A warehouse with `AUTO_RESUME = FALSE`** fails every statement with "warehouse is suspended", which reads like a privilege error.
- **A PAT cannot authenticate without an active network policy.** Prefer reusing a broad one: recovering from an IP lockout needs Snowflake Support, with no self-service path back.

Error code `003001` has four distinct causes — account entitlement, unknown model, model not granted to the role, and a bare fine-tuned model name — so it is not diagnostic on its own.

**One cost note:** a SQL warehouse is a real cost centre. Measured on the equivalent Databricks setup, warehouse queries cost roughly 1,500× the model-serving usage they reported on. Run the backfill as one query over a wide window, never as a tight polling loop.

The pure `extract_snowflake_functions_log(row)` / `extract_snowflake_rest_log(row)` / `resolve_snowflake_subscription(row)` functions stay available from `lago_agent_sdk.gateway.adapters` if you already have rows from your own warehouse job.

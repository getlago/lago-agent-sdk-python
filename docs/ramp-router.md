# Ramp Router

[Ramp Router](https://router.com) is an OpenAI-Responses-compatible gateway in front of OpenAI, Anthropic, Google Vertex, Fireworks and xAI. Point an OpenAI client at it and `wrap()` detects it from the `base_url` — no other code change. An Anthropic client pointed at Router is detected the same way; see [Two surfaces](#two-surfaces-responses-and-messages) for when to use which:

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

- **The model that answered is the one billed.** Router diverges from what you asked for in two ways: a `models` fallback list sends no `model` field at all, and Switchyard routing can serve a different model than the one requested. The SDK bills the model the response reports.
- **Router answers with a resolved vendor snapshot, so nothing needs stripping.** Ask for `openai:gpt-5.4-nano` and the response says `gpt-5.4-nano-2026-03-17`; that bare name is what bills, so a Router-served model rolls up in Lago against the same row a direct call to it reports. (A `provider:model[:tier]` candidate is still split if one ever reaches the adapter unresolved, but no captured response carries that shape.)
- **The served service tier is recorded** in `usage.extras["service_tier"]`, read from the response's own `service_tier` field — `flex`, `default`, and so on. Note `extras` is diagnostic: it is **not** sent to Lago, so the tier is visible to an `on_error`/debug hook but does not reach your events or split a charge. In price mode a non-default tier is a reported miss rather than a multiplied rate — see below.
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

## Price mode bills Router's own catalog

In `pricing_mode="price"`, a Router call is priced from **Router's own `GET /v1/models` catalog** — the rate Router bills, not OpenRouter's listing for the same model. Measured against a live account's dashboard export on 2026-09-04 and 2026-09-07: every default-tier call whose token counts the response fully reports reconciled at exactly the catalog rate, across all five served vendors (OpenAI, Anthropic, xAI, Fireworks, Baseten), including cache reads (xAI, 194 in / 192 cached: `2 × input + 192 × cache_read + output`, to the last digit) and OpenAI-served cache writes (`gpt-5.6-luna`, 4,490 written tokens at the `cache_write_input` rate). One model is the exception, noted below.

- **No extra configuration.** The catalog is account-specific and needs your Router key, and `wrap()` learns it from the client you pass in. Set `LagoConfig.ramp_router_api_key` only to price Router usage without calling `wrap()`; an explicit value always wins over a learned one.
- **The served model is what is looked up.** Router answers with the vendor's own snapshot (`gpt-5.4-nano-2026-03-17`) or path (`accounts/fireworks/models/…`), and the lookup resolves each of those back to its catalog entry — by exact name, alias, or version-strip. Cost events carry `price_source: "ramp_router"`.
- **Only the default tier is priced.** A call Router served at `flex` (measured 0.5x the catalog rate) or `priority` (measured 2x) emits token events plus an `on_error` report that names the tier. The SDK applies no tier multiplier: the factors are Router's policy, published nowhere machine-readable. Pin the default tier in the request if every call must price (Anthropic models reject a pinned tier and serve the default unpinned). A response that reports **no** tier bills at the base rate: in a 237-call sweep Router omitted `service_tier` only on responses that stopped with zero output, and billed all of them at standard.
- **A BYOK-served request is billed at the catalog rate.** Router charges $0 for it and your vendor bills you directly; the response is byte-identical either way, so nothing in the SDK can tell. The catalog rate equals what the vendor charges, so the amount is right even though the payee differs. Router also falls back to its shared key when yours fails, and bills normally then.
- **Eight OpenAI models bill off Router's own catalog. The SDK bills the catalog anyway; correct it with `markup` if you want to match Router's dashboard.** Reconciled row by row against the dashboard export of a 237-call sweep on 2026-09-07, these models billed a constant multiple of their published rate on every field (input, output, cache read, cache write), at both served tiers, with no exceptions; every other model on every vendor billed exactly the catalog rate.

  | model | Router billed, as a multiple of its own catalog rate | markup that matched the dashboard on 2026-09-07 |
  |---|---|---|
  | `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.5`, `gpt-5.5-pro`, `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-6-astra` | 1.1x | `1.1` |
  | `gpt-5.6-sol` | 0.55x: a **50% off** promotion shown on Router's model page ($2.00 / $10.00 against the catalog's $4 / $20), times the same 1.1 | `0.55` while the promotion lasts |

  The SDK does not apply these factors, on purpose. Sol's is a time-limited promotion and the 1.1 looks like a catalog that lags Router's newest models; either can change on a day of Router's choosing, and a factor baked into the SDK would become the thing that is wrong, for every customer, until a release ships. A markup is yours: apply it per call with `extra_lago={"markup": 1.1}` on those models (or globally with `LagoConfig.markup` if your traffic is all one model), reconcile against Router's dashboard from time to time, and drop it when the catalog catches up. Router's documentation names no fee. Treat the table as a measurement with a date, not a rate card.
- **A catalog entry served through a different backend is a miss, not a price.** Ten Fireworks-owned entries also carry a Baseten alias (`deepseek-ai/…`, `zai-org/…`, `moonshotai/…`, `nvidia/…`, `openai/gpt-oss-120b`). When Router serves one through Baseten it bills Baseten's rate, which the catalog does not publish (measured 1.11x to 2.4x away from the Fireworks rate), and the response names the model by that alias. The SDK does not index those aliases, so such a call emits token events plus an `on_error` report rather than a wrong price. The cost is that Baseten-served calls whose rate happened to match also miss.
- **Anthropic-served cache writes bill at the input rate.** `/v1/responses` reports no write count for Anthropic models — a cold `cache_control` write is byte-identical to an uncached call of the same size — so the written prefix is priced as `input` rather than at the `cache_write_input_5m`/`_1h` rate. Measured shortfall on such a call: up to 20% with the 5-minute TTL and up to 50% with the 1-hour TTL, in proportion to how much of the prompt was the cached prefix. Reads are exact. Router's `/v1/messages` surface does report the write count, and an Anthropic client pointed at Router bills it exactly — see [Two surfaces](#two-surfaces-responses-and-messages).

Any other miss — the table still cold on the very first call, a model the catalog does not list, no key learned — falls back to token events and reports through `on_error`, like every other provider. Token mode, the default, is unaffected: it emits the counts Router reported.

## Two surfaces: Responses and Messages

Router exposes every model in your catalog on two API surfaces. The SDK detects Router on both from the client's base URL, prices both from the same catalog with the same key, and stamps them differently because they report the same vendor under different conventions.

| | `POST /v1/responses` | `POST /v1/messages` |
|---|---|---|
| Client | `OpenAI(base_url="https://api.router.com/v1")` | `Anthropic(base_url="https://api.router.com")` |
| Schema | OpenAI Responses, for every vendor | Anthropic Messages, for every vendor |
| Token convention (measured) | cached tokens **inside** `input_tokens` | cached tokens **beside** `input_tokens` — Anthropic's additive convention, whichever vendor served |
| Anthropic cache **write** reported | no | yes: `cache_creation_input_tokens`, with the 5-minute / 1-hour split |
| Price mode | catalog rate, default tier only | same, and each cache write billed at `cache_write_input_5m` or `_1h` |
| `service_tier` | top level of the response | inside `usage`, buffered and streamed |
| Events stamped | `provider=ramp_router api=ramp_router` | `provider=ramp_router api=ramp_router_messages` |

```python
from anthropic import Anthropic

client = sdk.wrap(
    Anthropic(api_key=os.environ["RAMP_ROUTER_API_KEY"], base_url="https://api.router.com"),
    subscription="sub_acme",
)
client.messages.create(
    model=os.environ["RAMP_ROUTER_MODEL"],
    max_tokens=256,
    system=[{"type": "text", "text": LONG_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
    messages=[{"role": "user", "content": "Summarize this invoice."}],
)
```

**Which to use.** If your Anthropic-model traffic uses prompt caching, reach Router with an Anthropic client. It is the only surface on which a cold cache write is priced exactly — reconciled to the cent against Router's dashboard on 2026-09-04: 16 input + 20,113 written tokens at the 5-minute rate + 5 output on `claude-haiku-4-5` = $0.02518225, and the warm repeat $0.0020523. On the Responses surface the same write is invisible and bills at the input rate (see above). Everything else bills identically on both surfaces: reads, plain calls, OpenAI-served cache writes, and the default-tier rule.

Both surfaces were measured with the same models on the same account. `/v1/messages` accepts non-Anthropic models too (an OpenAI and an xAI model were captured), and renders their usage in Anthropic's shape with thinking tokens inside `output_tokens`; the adapter treats them exactly like a native Anthropic response. The two `api` stamps exist so the token-overlap rules can never be applied to the wrong surface: the Responses stamp subtracts cached tokens from `input`, the Messages stamp does not, and the SDK pins that distinction in tests.

## Measured behaviours and limitations

- **There is no backfill path**, because Router exposes no programmatic usage surface. Its only routes are `GET /v1/models`, `POST /v1/responses`, `POST /v1/messages` and `POST /v1/messages/count_tokens`; usage lives in the dashboard's Logs view. An "analytics API" is mentioned once in Router's limits table with no path, auth or record shape. Unlike the [Cloudflare connector](cloudflare.md), there is no Logs API loop to show you.
- **Nothing is skipped as a gateway cache hit**, because Router has no response cache: "Self-service Router response caching, which would reuse an entire previous response without calling a model provider, is a separate optimization and is not currently configurable." Provider _prompt_ caching does pass through, and those cache-read and cache-write tokens are billed like any other.
- **Two deprecated OpenAI models never price.** Router serves `gpt-3.5-turbo` and `gpt-4` as `gpt-3.5-turbo-0125` and `gpt-4-0613`, a 4-digit month-day snapshot suffix the SDK's version strip does not remove, so they miss the catalog and emit token events. A blind 4-digit strip would be wrong: `deepseek-v4-pro-0813` is its own catalog entry, and stripping it lands on a different model's rate.
- **A proxy in front of Router is not detected.** Detection matches the `api.router.com` host (and `*.router.com`). Reaching Router through your own hostname bills as plain OpenAI, with the wrong provider and an unparsed model id.
- **`api.router.com` sits behind bot management.** A rejected client can get an HTML challenge page rather than Router's documented JSON error envelope. The SDK degrades to zero usage rather than throwing, either way.
- **Failures never bill.** Every documented status — including 402 `insufficient_credits`, 429, and 502 `all_candidates_failed` — emits nothing, as does a response reporting zero usage.

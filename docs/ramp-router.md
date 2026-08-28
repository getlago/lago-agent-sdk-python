# Ramp Router

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

## Price mode falls back to token events

In `pricing_mode="price"`, a Router call currently emits **token events rather than `llm_cost`**. That is deliberate, not a gap, and it is worth understanding before turning price mode on for Router traffic:

- **A BYOK-served request costs $0 through Router.** Router's own words: "When a request is served with your provider key, your provider bills you directly and Ramp Router does not charge for that usage." Nothing in the response says which key served it, so pricing at list rate would bill a customer the full amount for usage Router never charged for.
- **A non-default service tier does not bill at the published base rate.** Router's catalog says outright that "service tiers, long contexts, caching, and other features may use different rates", and that a Fast tier's "pricing may differ from the base rates shown here". Billing flex at the standard rate over-bills.
- **The token overlap semantics are Router's own, and they are measured.** OpenAI counts cached tokens inside `input_tokens` and reasoning inside `output_tokens`; Anthropic counts both additively. Router normalizes the _numbers_ to OpenAI's convention, not just the schema — verified live on an Anthropic-served model, where a warm `cache_control` call reported the cached block inside an unchanged `input_tokens`. But that convention is Router's, not the served vendor's, which is exactly why the served vendor must not be stamped as the provider: it would de-overlap with the wrong rules whenever its native convention differs.

So Router is treated as a provider of its own, matching no vendor in the price tables. Token mode — the default — is exact and unaffected: it emits the counts Router reported, the same per-field `llm_*` events a direct provider call produces. Price mode takes a clean miss and falls back to those same token events, with no error on your call path. Bill Router traffic in token mode and price it with a Lago plan.

## Measured behaviours and limitations

- **There is no backfill path**, because Router exposes no programmatic usage surface. Its only routes are `GET /v1/models`, `POST /v1/responses`, `POST /v1/messages` and `POST /v1/messages/count_tokens`; usage lives in the dashboard's Logs view. An "analytics API" is mentioned once in Router's limits table with no path, auth or record shape. Unlike the [Cloudflare connector](cloudflare.md), there is no Logs API loop to show you.
- **Nothing is skipped as a gateway cache hit**, because Router has no response cache: "Self-service Router response caching, which would reuse an entire previous response without calling a model provider, is a separate optimization and is not currently configurable." Provider _prompt_ caching does pass through, and those cache-read and cache-write tokens are billed like any other.
- **Router's Anthropic Messages surface is not instrumented yet.** `POST /v1/messages` exists and routes to the same providers, so pointing a wrapped `Anthropic` client at `https://api.router.com/v1` will _work_ as an LLM client but bill nothing. Use the Responses surface until that lands.
- **A proxy in front of Router is not detected.** Detection matches the `api.router.com` host (and `*.router.com`). Reaching Router through your own hostname bills as plain OpenAI, with the wrong provider and an unparsed model id.
- **`api.router.com` sits behind bot management.** A rejected client can get an HTML challenge page rather than Router's documented JSON error envelope. The SDK degrades to zero usage rather than throwing, either way.
- **Failures never bill.** Every documented status — including 402 `insufficient_credits`, 429, and 502 `all_candidates_failed` — emits nothing, as does a response reporting zero usage.

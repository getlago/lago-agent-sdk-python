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

See [`examples/cloudflare_gateway_demo.ipynb`](examples/cloudflare_gateway_demo.ipynb) for a runnable end-to-end version of both.

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

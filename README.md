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

**Reasoning tokens** populate automatically on Gemini 2.5 — the model reasons internally by default and surfaces `thoughts_token_count`. Note the semantic difference vs OpenAI:
- **OpenAI:** `reasoning_tokens` is a *subset* of `completion_tokens` (already counted in output)
- **Gemini:** `thoughts_token_count` is *additive* to `candidates_token_count` (total Google bill = output + reasoning)

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
| LiteLLM | callback bridge | Phase 4 |

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
| image_input | `llm_image_input_tokens` | ✗ | ✗ | ✗ | ✗ (Phase 3) | ✓ (multimodal IMAGE) |

**Semantic note on `reasoning`:**
- **OpenAI's `reasoning_tokens` is a SUBSET of `output`** — already counted in `completion_tokens`.
- **Gemini's `thoughts_token_count` is ADDITIVE to `output`** — `candidates + thoughts = total billable output`.

**Semantic note on input breakdowns (avoid double-counting):**
For both OpenAI and Gemini, `cache_read`, `audio_input`, and `image_input` are **subsets of `input`**, not additive to it — they are a breakdown of tokens already counted in `llm_input_tokens`. For example, OpenAI reports `cached_tokens` under `prompt_tokens_details` *within* `prompt_tokens`, and Gemini's docs state `prompt_token_count` "includes the number of tokens in the cached content". A billable metric that sums `llm_input_tokens + llm_cached_input_tokens` (or `+ llm_audio_input_tokens`, `+ llm_image_input_tokens`) will **double-count**. Bill on `llm_input_tokens` as the total; use the breakdown fields only for cost attribution or discounted-rate tiers (e.g. cached input billed at a lower rate), subtracting them from `input` rather than adding.

OpenAI's Predicted Outputs tokens (`accepted_prediction_tokens`, `rejected_prediction_tokens`) are not surfaced — see the OpenAI adapter docstring for details on this intentional gap.

## Error policy

The SDK never breaks your LLM call. If anything in instrumentation fails (adapter bug, Lago down, network error), the SDK swallows it, logs a warning, and your call returns normally.

## Subscription resolution returns nothing → drop with `ERROR` log

Configurable via `LagoConfig.on_error` callback to integrate with Sentry, Datadog, etc.:

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

Run live integration tests (requires real credentials):

```bash
AWS_BEARER_TOKEN_BEDROCK="..." \
MISTRAL_API_KEY="..." \
LAGO_API_URL="https://api.getlago.com/api/v1/" \
LAGO_API_KEY="..." \
LAGO_EXTERNAL_SUBSCRIPTION_ID="sub_..." \
pytest tests/integration
```

## Security

Found a vulnerability? See [SECURITY.md](SECURITY.md).

## License

[MIT LICENSE](LICENSE).

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
| Mistral | native SDK (`chat.complete` + `chat.stream`) | ✓ |
| OpenAI | native SDK | Phase 2 |
| Anthropic | native SDK | Phase 2 |
| Google Gemini | native SDK | Phase 2 |
| LiteLLM | callback bridge | Phase 4 |

## Token dimensions captured

`CanonicalUsage` carries 10 numeric fields. Which ones populate depends on the provider:

| Field | Lago metric code | Bedrock | Mistral native |
|---|---|---|---|
| input | `llm_input_tokens` | ✓ | ✓ |
| output | `llm_output_tokens` | ✓ | ✓ |
| cache_read | `llm_cached_input_tokens` | ✓ (Anthropic) | ✓ (when cache hits) |
| cache_write | `llm_cache_creation_tokens` | ✓ (Anthropic) | ✗ |
| cache_write_5m / 1h | `llm_cache_write_5m/1h_tokens` | ✓ (Anthropic InvokeModel) | ✗ |
| reasoning | `llm_reasoning_tokens` | ✗ (folded into output) | ✗ (folded into output) |
| tool_calls | `llm_tool_calls` | ✓ | ✓ |
| image_input / audio_input | `llm_image/audio_input_tokens` | ✗ | ✗ |

Reasoning, image, and audio fields will populate when Phase 2 native OpenAI ships.

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

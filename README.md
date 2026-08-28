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

Every provider is covered sync + async + streaming. The full coverage matrix, the token fields each provider populates, and the per-provider quirks that affect billing (cache/reasoning overlap semantics, OpenAI's `stream_options` auto-inject, Gemini's additive `thoughts_token_count`) live in [docs/providers.md](docs/providers.md).

## Gateways

`wrap()` also detects a client pointed at a gateway and bills what the gateway actually did — each guide covers backfill, attribution and the measured billing caveats for that gateway.

### Cloudflare AI Gateway

Point any supported client at your gateway URL; cache hits (`cf-aig-cache-status: HIT`) are not billed, and Workers AI models get priced automatically from Cloudflare's published rates.

```python
client = sdk.wrap(Anthropic(
    api_key="...",
    base_url=f"https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/anthropic",
    default_headers={"cf-aig-authorization": f"Bearer {gateway_auth}"},
))
```

Full guide, including backfill from the gateway's Logs API: [docs/cloudflare.md](docs/cloudflare.md).

### Databricks AI Gateway

BYOK calls bill dollar cost at the vendor's rates; Databricks-hosted models (`system.ai.*`) bill token counts, and `sdk.backfill_databricks(source, "7 days")` fills in what `wrap()` didn't see.

```python
client = sdk.wrap(OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"{DATABRICKS_HOST}/ai-gateway/mlflow/v1",
    default_headers={"Databricks-Ai-Gateway-Request-Tags": json.dumps({"lago_subscription": "sub_acme"})},
))
```

Full guide, including BYOK setup, backfill and gotchas: [docs/databricks.md](docs/databricks.md).

### Snowflake Cortex

The Cortex REST surface wraps like any OpenAI-compatible client; the AI SQL functions (`AI_COMPLETE`, …) have no client to wrap and are backfilled from Snowflake's usage views. Everything bills as token counts.

```python
client = sdk.wrap(OpenAI(
    base_url=f"https://{os.environ['SNOWFLAKE_ACCOUNT']}.snowflakecomputing.com/api/v2/cortex/v1",
    api_key=os.environ["SNOWFLAKE_PAT"],
))

sdk.backfill_snowflake(SnowflakeSource.from_env(), "7 days", default_subscription="sub_default")
```

Full guide, including cache semantics, dedup, attribution via `QUERY_TAG` and account setup: [docs/snowflake.md](docs/snowflake.md).

### Ramp Router

An OpenAI-Responses-compatible gateway in front of OpenAI, Anthropic, Google Vertex, Fireworks and xAI. The model that answered is the one billed — Router can serve a different model than the one requested.

```python
client = sdk.wrap(
    OpenAI(api_key=os.environ["RAMP_ROUTER_API_KEY"], base_url="https://api.router.com/v1"),
    subscription="sub_acme",
)
client.responses.create(model=os.environ["RAMP_ROUTER_MODEL"], input="Summarize this invoice.")
```

Full guide, including why price mode falls back to token events for Router traffic: [docs/ramp-router.md](docs/ramp-router.md).

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
```

Price mode emits one `llm_cost` event per priced field (input, output, cache, ...), each carrying `precise_total_amount_cents` for Lago's **dynamic charge model** plus a `token_type` property so a single billable metric can be grouped by both `model` and `token_type`. Prices come from public sources (OpenRouter for native providers, the AWS Bedrock price list for Bedrock), fetched and cached in the background — your LLM call is never blocked on pricing. If a price isn't available yet, the SDK falls back to token-count events and reports via `on_error` rather than under-billing. Per-call override: `extra_lago={"mode": "price", "markup": 1.5}`.

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

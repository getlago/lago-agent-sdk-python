# Changelog

All notable changes to this project will be documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org).

## [Unreleased]

### Fixed
- **OpenAI/Anthropic adapters mis-tagged usage with the requested model instead of the model that actually answered.** `extract_openai_native`/`extract_anthropic_native` preferred the request's `model` kwarg over the response's own `model` field. Harmless calling a provider directly with a fully-qualified model id, but wrong the moment a provider resolves a short alias to a dated snapshot — confirmed live with no gateway involved at all: requesting `claude-sonnet-4-5` answered as `claude-sonnet-4-5-20250929`. Nearly every captured OpenAI fixture in this suite shows the same pattern (`gpt-4o-mini` → `gpt-4o-mini-2024-07-18`). Both adapters now prefer the response's own `model`, falling back to the request only when the response is silent about it (e.g. a synthetic streaming usage blob). Pricing and per-model attribution now key off what actually served the request.

### Added
- **Gateway cache-hit detection for OpenAI/Anthropic wrappers.** Non-streaming `.create(...)` calls now go through `.with_raw_response.create(...)` so the wrapper can see response headers before parsing the body. If a gateway in front of the provider (e.g. Cloudflare AI Gateway) marks the response `cf-aig-cache-status: HIT`, the provider served it from cache at zero cost to the customer, and the wrapper skips billing it. `.parse()` on the raw response returns the identical object `.create()` would have, so this is invisible to the customer and a no-op with no gateway in the path. Streaming calls are not covered yet — gateways typically recommend `.with_streaming_response` for that, which behaves differently and hasn't been verified end-to-end; streaming keeps using the plain `.create()` path. Falls back to the pre-existing behavior if `.with_raw_response` isn't available on the client (older SDK versions).
- **`lago_agent_sdk.gateway.adapters.cloudflare_gateway`** — `extract_cloudflare_log()` maps a Cloudflare AI Gateway Logs API entry (`tokens_in`/`tokens_out`/`usage_metadata`/`model`/`provider`) to `CanonicalUsage`, and `resolve_subscription()` reads Lago attribution from the customer's `cf-aig-metadata` header. Lives in a new `lago_agent_sdk.gateway` namespace, separate from the provider-native `adapters/` used by `wrap()` — this is the extraction half of a standalone log-polling connector (not part of `wrap()`), verified against a real captured log entry whose token counts were independently confirmed to roll up correctly in a real Lago instance. The poller itself (scheduler, cursor store, credential store) is not part of this SDK and isn't built yet.
- **Verified `extract_cloudflare_log()` against all three of Cloudflare's ingress methods, live**, not just the provider-native `/{provider}` routes covered above: the REST API (`POST /accounts/{account}/ai/run`), the Unified/OpenAI-compat endpoint (`.../compat/chat/completions`, called with the real `openai` SDK), and the Native/binding method (`env.AI.run(model, input, {gateway: {id, metadata}})`, only reachable from inside a deployed Cloudflare Worker). Same extraction function, zero code changes, correct results and correct attribution (`resolve_subscription()`) across all three — confirms the log schema is normalized regardless of how the call reached the gateway. Also swept 26 real Workers AI models through the REST API in one pass (22 succeeded, 4 failed for real account/licensing reasons — Workers Paid plan required, or a model needing explicit license acceptance — none a compatibility gap); extraction had zero failures across the full spread, including an unusual moderation-model shape (`llama-guard-3-8b`: 203 input / 3 output tokens). New fixtures 06–11 in `tests/unit/gateway/adapters/fixtures/cloudflare_gateway/` capture this.
- **`wrap_gemini_client`/`wrap_mistral_client` verified through Cloudflare's dedicated per-provider passthrough endpoints, live, with real customer API keys** (`.../google-ai-studio` and `.../mistral`) — real calls, real Lago billing, same pattern already proven for Anthropic: the customer's own key is forwarded directly, no Cloudflare-side BYOK/wholesale credits needed. Both required an explicit `cf-aig-authorization` header the SDK doesn't add on its own (`http_options.headers` for `google-genai`, `http_headers=` per-call for `mistralai`).
- **Fixed a real gap this surfaced: `extract_cloudflare_log()` never mapped reasoning tokens.** The real Gemini call's log entry has `usage_metadata.reasoningTokens: 852` (camelCase) sitting right next to `tokens_out: 21` (the visible completion only) — Cloudflare doesn't normalize `usage_metadata`'s key casing across providers; it passes through whatever convention each provider's own usage object used (Anthropic: snake_case, Gemini: camelCase). Now checks both cases for the reasoning field.
- **Replaced two hand-built synthetic cache fixtures with real captured ones — and corrected a wrong assumption in the process.** The old synthetic gateway-cache-hit fixture assumed a `cached: true` entry still reports the token counts the call "would have" cost, leaving billing policy to decide whether to skip it. A real captured cache hit (same request sent twice with `cf-aig-cache-ttl` set; the second came back in 8ms vs 296ms) proves that's wrong: Cloudflare's own log already reports `tokens_in`/`tokens_out` as 0 on a real hit — no caller-side branching on `cached` is needed. Separately, real back-to-back Anthropic calls through the gateway with a >1024-token `cache_control: {"type": "ephemeral"}` block confirm `usage_metadata.input_cache_creation_tokens`/`input_cached_tokens` exactly match Anthropic's own `cache_creation_input_tokens`/`cache_read_input_tokens` (3429 tokens, both directions) — this mapping was previously untested against real data.

## [0.2.0] - 2026-06-15

### Added
- **Price mode — emit computed dollar cost instead of token counts.** New `pricing_mode` config (`"tokens"` default | `"price"`), plus `markup`, `cost_metric_code` (default `llm_cost`), `pricing_ttl_seconds`, and `bedrock_default_region`. In price mode the SDK emits one `llm_cost` event per call carrying a top-level `precise_total_amount_cents` (cost in cents, after markup) for Lago's **dynamic charge model**, with a full per-field breakdown in `properties` (value in USD, base, markup, source, per-field tokens/unit_price/cost). Live unit prices come from public, no-auth sources: OpenRouter (`/api/v1/models`) for native anthropic/openai/mistral/gemini, and the AWS Bedrock Price List **Bulk** API for Bedrock. Prices are fetched + cached on the background queue thread (never blocking the customer's call); a missing price falls back to token events and calls `on_error` (never silently under-bills). Mode and markup are overridable per-call via `extra_lago={"mode": "price", "markup": 1.5}`. Money is computed with `Decimal` floored to 12 dp, identical to the JS implementation (cross-repo golden fixture). New `pricing.py` module + `PricingProvider`; default `pricing_mode="tokens"` keeps existing behavior unchanged.

### Fixed
- **Anthropic `messages.create(stream=True)` under-billed input tokens.** The stream wrapper read only top-level `usage`, which on a basic stream appears only on `message_delta` as `{output_tokens: N}` — the authoritative `input_tokens` / `cache_*` counts arrive nested under `message.usage` on the `message_start` event and were ignored, so input billed 0. The wrapper now merges usage from `message_start` (input/cache) and `message_delta` (cumulative output). Sync + async paths; regression tests use the realistic wire shape (delta carries no input echo).
- **Legacy `google-generativeai` SDK silently emitted no events.** The detector matched both the new `google-genai` and the deprecated `google-generativeai` SDKs, but the wrapper only instruments the unified `Client.models` / `.aio` surface — a legacy `GenerativeModel` routed through and wrapped nothing. `wrap()` now rejects legacy clients with a clear pointer to migrate to `google-genai`.

### Security
- Hardened the publish workflow: least-privilege `permissions: contents: read` default (only `publish` gets `id-token: write`, only `release` gets `contents: write`), and every third-party action pinned to a full commit SHA so a re-pointed tag can't inject code into the OIDC-token-minting job.
- Added `if: startsWith(github.ref, 'refs/tags/v')` to the `publish` job as defense-in-depth — it refuses to run on a non-tag ref even if the environment's protected-tag rule is misconfigured.
- Added `.github/dependabot.yml` (github-actions ecosystem) so the SHA pins stay fresh — Dependabot bumps the SHA and version comment together rather than letting actions silently age.
- RELEASING.md now documents `pypi` environment protection (required reviewers + protected-tag restriction) as a **required** setup step, not optional, since trusted publishing is only as strong as that environment's rules.

### Documentation
- README: clarified that `cache_read`, `audio_input`, and `image_input` are **subsets** of `input` for OpenAI and Gemini (not additive) — summing them with `llm_input_tokens` double-counts.

### Added
- Native `google-genai` SDK support covering `client.models.generate_content` + `generate_content_stream`, sync + async (`client.aio.models`).
- `extract_gemini_native` adapter maps `usage_metadata`: `prompt_token_count → input`, `candidates_token_count → output`, `cached_content_token_count → cache_read`, `thoughts_token_count → reasoning`, `prompt_tokens_details[modality=AUDIO/IMAGE] → audio_input/image_input`, `candidates_tokens_details[modality=AUDIO] → audio_output`, count of `candidates[0].content.parts[].function_call → tool_calls`.
- **Gemini 2.5 surfaces reasoning tokens by default** (`thoughts_token_count`) — fires `llm_reasoning_tokens` automatically. Note the semantic difference vs OpenAI: Gemini's reasoning is ADDITIVE to output (`candidates + thoughts = total billable output`); OpenAI's reasoning is a SUBSET of `completion_tokens`. Documented in adapter docstring + README.
- `gemini` optional dependency group: `pip install 'lago-agent-sdk[gemini]'`.
- 21 new unit tests (15 adapter + 6 wrapper) and 4 live integration tests (gated on `GEMINI_API_KEY`). Total: 304 unit tests.
- 5 captured response fixtures from the real Gemini API (plain, tool use, streaming, thinking, multi-turn).
- Detector now returns `gemini` (was `google`) for `google-genai` clients.

### Added (OpenAI — earlier in this branch)
- Native `openai` SDK support covering both APIs: `chat.completions.create` and `responses.create`, each with sync + streaming. Same coverage on `AsyncOpenAI`.
- `extract_openai_native` adapter handles both API shapes with auto-detection:
  - Chat Completions: `prompt_tokens`, `completion_tokens`, `prompt_tokens_details.{cached_tokens, audio_tokens}`, `completion_tokens_details.{reasoning_tokens, audio_tokens}`, count of `choices[0].message.tool_calls`.
  - Responses API: `input_tokens`, `output_tokens`, `input_tokens_details.cached_tokens`, `output_tokens_details.reasoning_tokens`, count of `output[].type == "function_call"`.
- **First provider to populate `llm_reasoning_tokens`** — OpenAI o-series models (`o4-mini`, `o1`, etc.) surface reasoning token counts separately.
- Auto-injection of `stream_options={"include_usage": True}` when the customer sets `stream=True` without it, so streamed Chat Completions emit usage on the final chunk.
- `audio_output` field added to `CanonicalUsage` (maps to `llm_audio_output_tokens`), populated by GPT-4o-audio responses.
- `openai` optional dependency group: `pip install 'lago-agent-sdk[openai]'`.
- 27 new unit tests (18 adapter + 9 wrapper) and 5 live integration tests (gated on `OPENAI_API_KEY`). Total: 283 unit tests.
- 10 captured response fixtures from the real OpenAI API (plain chat, tool use, auto-caching, streaming with usage, o-series reasoning, multi-turn, Responses API plain + tool use + reasoning).

### Previously in unreleased (Anthropic)
- Native `anthropic` SDK support. Wraps `Anthropic.messages.create` (including `stream=True`) and `Anthropic.messages.stream(...)` context manager. Same coverage on `AsyncAnthropic` (sync + async variants).
- `extract_anthropic_native` adapter with the full Anthropic field map: `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `cache_creation.ephemeral_5m_input_tokens`, `cache_creation.ephemeral_1h_input_tokens`, `content[].type == "tool_use"`.
- `anthropic` optional dependency group: `pip install 'lago-agent-sdk[anthropic]'`.
- 19 unit tests (adapter + wrapper) and 3 live integration tests (gated on `ANTHROPIC_API_KEY`).
- 9 captured response fixtures from the real Anthropic API (plain, tool use, 5m + 1h prompt caching, extended thinking, streaming, multi-turn).


## [0.1.0] — initial release

### Added
- `LagoSDK` core with batched async event queue, exponential backoff, bounded buffer, async-local subscription resolution.
- `boto3` Bedrock wrapper covering `Converse`, `ConverseStream`, `InvokeModel`, `InvokeModelWithResponseStream`.
- 7 InvokeModel family adapters (`anthropic`, `opus_4_7`, `nova`, `pixtral`, `mistral_legacy`, `openai_compat_basic`, `openai_compat_with_details`) with substring-match dispatch.
- `mistralai` native wrapper covering `chat.complete`, `chat.stream`, async variants.
- Three subscription-resolution tiers: per-call `extra_lago`, context-bound `set_subscription`, init-time default.
- 245 tests: 237 unit + 8 integration; verified against 159 fixtures captured from real provider responses.
- p99 wrap-overhead ≤ 5 ms benchmark.

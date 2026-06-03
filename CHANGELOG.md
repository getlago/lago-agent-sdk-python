# Changelog

All notable changes to this project will be documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org).

## [Unreleased]

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

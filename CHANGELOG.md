# Changelog

All notable changes to this project will be documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org).

## [Unreleased]

### Added
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

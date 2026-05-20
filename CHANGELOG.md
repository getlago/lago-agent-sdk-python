# Changelog

All notable changes to this project will be documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org).

## [Unreleased]

### Added
- Native `anthropic` SDK support. Wraps `Anthropic.messages.create` (including `stream=True`) and `Anthropic.messages.stream(...)` context manager. Same coverage on `AsyncAnthropic` (sync + async variants).
- `extract_anthropic_native` adapter with the full Anthropic field map: `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `cache_creation.ephemeral_5m_input_tokens`, `cache_creation.ephemeral_1h_input_tokens`, `content[].type == "tool_use"`.
- `anthropic` optional dependency group: `pip install 'lago-agent-sdk[anthropic]'`.
- 19 new unit tests (adapter + wrapper) and 3 live integration tests (gated on `ANTHROPIC_API_KEY`). Total: 256 unit tests, ≥80% coverage maintained.
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

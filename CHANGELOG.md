# Changelog

All notable changes to this project will be documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org).

## [Unreleased]

## [0.1.0] — initial release

### Added
- `LagoSDK` core with batched async event queue, exponential backoff, bounded buffer, async-local subscription resolution.
- `boto3` Bedrock wrapper covering `Converse`, `ConverseStream`, `InvokeModel`, `InvokeModelWithResponseStream`.
- 7 InvokeModel family adapters (`anthropic`, `opus_4_7`, `nova`, `pixtral`, `mistral_legacy`, `openai_compat_basic`, `openai_compat_with_details`) with substring-match dispatch.
- `mistralai` native wrapper covering `chat.complete`, `chat.stream`, async variants.
- Three subscription-resolution tiers: per-call `extra_lago`, context-bound `set_subscription`, init-time default.
- 245 tests: 237 unit + 8 integration; verified against 159 fixtures captured from real provider responses.
- p99 wrap-overhead ≤ 5 ms benchmark.

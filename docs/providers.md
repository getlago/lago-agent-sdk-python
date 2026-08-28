# Providers — coverage and token semantics

The [README](../README.md) shows the minimal quickstart per provider. This page holds the full coverage matrix, the token fields each provider populates, and the per-provider quirks that affect billing.

## Supported providers

| Provider | Access | Status |
|---|---|---|
| AWS Bedrock | `Converse` (sync + stream) | ✓ |
| AWS Bedrock | `InvokeModel` (sync + stream), 7 model families | ✓ |
| Anthropic | native SDK (`messages.create` + `messages.stream`, sync + async) | ✓ |
| Mistral | native SDK (`chat.complete` + `chat.stream`) | ✓ |
| OpenAI | native SDK (`chat.completions.create` + `responses.create`, sync + async + stream) | ✓ |
| Google Gemini | native SDK (`google-genai`: `models.generate_content` + `generate_content_stream`, sync + async) | ✓ |

## Per-provider notes

**Anthropic** — works with `Anthropic` and `AsyncAnthropic`. Both `messages.create(..., stream=True)` and the `messages.stream(...)` context manager are instrumented — usage is captured from the final `message_delta` event in either case.

**OpenAI** — works with `OpenAI` and `AsyncOpenAI`. Covers both **Chat Completions** (`client.chat.completions.create`) and the newer **Responses API** (`client.responses.create`), sync + streaming. For streaming, the wrapper auto-injects `stream_options={"include_usage": True}` so the final chunk carries usage data — without it OpenAI emits no usage on streamed responses. **Reasoning tokens** (`llm_reasoning_tokens`) populate automatically when you call an o-series model (`o4-mini`, `o1`, etc.) — OpenAI is the first provider to expose this metric separately.

**Gemini** — wraps the modern `google-genai` SDK (`from google import genai`). Covers `client.models.generate_content` + `generate_content_stream`, sync + async (via `client.aio.models`). **Reasoning tokens** populate automatically on Gemini 2.5 — the model reasons internally by default and surfaces `thoughts_token_count` (see the note on reasoning semantics below).

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

"""Token-count conventions per provider — the single source of truth.

Providers disagree about whether a subset count (cached tokens, reasoning
tokens) is reported INSIDE its parent count or beside it. OpenAI reports
`cached_tokens` inside `prompt_tokens`; Anthropic reports `cache_read_input_tokens`
outside `input_tokens`. The wire shape says nothing about which convention is in
play — Snowflake Cortex answers on a byte-for-byte OpenAI wire with Anthropic's
additive convention (measured live 2026-08-25: prompt 7, cached 4805, completion 6,
total 4818) — so the convention can only be KEYED, never inferred from a payload.
Two payload-only inference gates were tried and both were shown unsound on real
shapes (see PR #23 review); do not reintroduce one.

Three places have to answer the same question, and this module exists so they
cannot answer it differently:

  - `adapters/openai_native.extract_openai_native` — deciding which reported
    counts sit inside `total_tokens` when reconciling it,
  - `pricing.compute_cost` — deciding which subsets to move out of their parent
    before pricing each field,
  - `pricing.deoverlapped_token_total` — deciding which subsets to drop from the
    single-event token total.

The two failure directions are not symmetric and both have happened here:
treat a subtractive surface as additive and real tokens are silently never
billed; treat an additive surface as subtractive and the same tokens bill twice
(measured at 1.570x, 2.0x, and 6.15x on three different providers). Entries are
added on MEASUREMENT of a real payload, never on the wire shape or vendor
documentation alone.
"""

from __future__ import annotations

# Providers whose reported `input` token count ALREADY includes the cached
# (`cache_read`) tokens — i.e. cache_read is a subset of input, not additive.
# For these, the cached portion must be billed at the cache-read rate, not the
# full prompt rate, so compute_cost moves it out of `input`. Anthropic reports
# input EXCLUSIVE of cache (cache_read/cache_write are additive), so it's absent.
#
# "workers-ai" belongs here because it is only ever reached through Cloudflare's
# OpenAI-COMPATIBLE endpoint (`.../compat`), so its usage payload is the OpenAI
# shape: `prompt_tokens` includes `prompt_tokens_details.cached_tokens`. It is a
# distinct provider only because it prices against Cloudflare's own catalog
# (see _infer_provider in adapters/openai_native.py) — the token semantics are
# still OpenAI's. Omitting it billed the cached tokens twice: once at the full
# input rate because they were never subtracted, and again at the cache-read
# rate, which Cloudflare's catalog does publish for some models.
#
# "mistral" belongs here for the same reason: the API is OpenAI-shaped and reports
# `prompt_tokens_details.cached_tokens` as a SUBSET of `prompt_tokens`. Mistral's own
# documented example is unambiguous — prompt_tokens=1013, cached_tokens=1008, and
# total_tokens=1043 = prompt + completion, which only reconciles if the cached tokens
# sit inside the prompt count. Omitting it double-billed the cached portion by 6.15x
# on that payload. 13 of 18 Mistral models on OpenRouter publish a cache-read rate,
# so the wrong path was reachable for most of them, including Mistral traffic routed
# through a Cloudflare gateway (the gateway adapter leaves provider="mistral" as-is).
#
# "snowflake" is deliberately ABSENT even though Cortex answers on an OpenAI-WIRE
# endpoint: measured live 2026-08-25, `prompt_tokens: 7`, `cached_tokens: 4805`,
# `completion_tokens: 6`, `total_tokens: 4818` — the cached block sits OUTSIDE
# `prompt_tokens` and INSIDE the total, Anthropic's additive convention on OpenAI's
# wire. Measured for the Claude family, which is the only family on that surface
# that caches at all today: llama accepts `cache_control` and ignores it
# (cached_tokens 0 on a matched pair, total = prompt + completion, measured
# 2026-08-27), and the OpenAI family needs cross-region inference the capture
# account cannot enable — re-verify the day an OpenAI-family model becomes
# reachable, since Cortex documents its caching behaviour per model family.
INPUT_INCLUDES_CACHE_READ = frozenset({"openai", "gemini", "workers-ai", "mistral"})

# Providers whose reported `input` token count ALREADY includes the cache-WRITE
# tokens. OpenAI is measured: a live gpt-5.6-sol response carries
# prompt_tokens=3025 with prompt_tokens_details.cache_write_tokens=3022, and
# Databricks' metered spend for that call matched billing all 3025 at the plain
# input rate — the write sits inside the prompt count (OpenAI's own docs now say
# the same: weighted input = ordinary + cached + cache-write portions). That is
# also why the field is deliberately NOT mapped to CanonicalUsage.cache_write —
# billing it separately over-charged 2.24x (see adapters/openai_native.py).
#
# The other three subset-cache providers are carried here on the SHAPE argument
# that earned "workers-ai" its cache_read entry: their surfaces re-report usage
# in the OpenAI shape, where every prompt_tokens_details member is a subset of
# prompt_tokens. None of the three emits the key today (Gemini and Mistral have
# no cache_write concept on these wires), so for them membership decides only
# what the total_tokens guard does if the key ever appears — and for a
# subset-convention surface the guard must stay live (a genuine remainder still
# folds), which membership preserves. Snowflake is absent: cache_write_tokens
# exists on its wire (a key OpenAI never sends) and was 0 in every capture
# including cache-creation calls — creation reports under cached_tokens there.
INPUT_INCLUDES_CACHE_WRITE = frozenset({"openai", "gemini", "workers-ai", "mistral"})

# Providers whose reported `output` token count ALREADY includes the reasoning
# tokens (reasoning is a subset of output). For these, reasoning is billed as
# part of output and must NOT be billed again separately. (Gemini's `thoughts`
# are additive to output, so it's absent here.)
#
# "workers-ai" belongs here for the same reason it is in INPUT_INCLUDES_CACHE_READ
# above: it is only ever reached through Cloudflare's OpenAI-COMPATIBLE endpoint, so
# its usage payload is the OpenAI shape — and in that shape
# `completion_tokens_details.reasoning_tokens` is a SUBSET of `completion_tokens`,
# exactly as it is for real OpenAI. `extract_openai_native` fills `reasoning` from that
# key with no provider gate, so omitting it counted the subset twice: measured, a
# 100/1000/reasoning-800 call reported unit=1900 against 1100 consumed. `compute_cost`
# would double-BILL the same tokens and does not today only because
# _CLOUDFLARE_UNIT_FIELD_MAP happens to carry no reasoning unit — an accident, not a
# guard, and Cloudflare hosts reasoning models (deepseek-r1, qwen, glm).
#
# "snowflake" absence is a measured no-op rather than an open question:
# `reasoning_tokens` is always 0 on Cortex's OpenAI-compat wire (reasoning_effort
# is accepted and ignored; extended thinking exists only on Cortex's Anthropic
# wire, which this adapter never serves). Re-measure if that wire ever starts
# reporting it.
OUTPUT_INCLUDES_REASONING = frozenset({"openai", "workers-ai"})

# Gateway SURFACES that re-report every vendor's usage in the OpenAI shape: `input`
# already contains cache_read AND cache_write, and `output` already contains
# reasoning, no matter which vendor actually served the call.
#
# This keys on `CanonicalUsage.api` rather than the provider because on a gateway
# it is the SURFACE that decides the shape, and a surface row reuses the live
# vendor names. A `provider="anthropic"` row read from Databricks' system table
# needs the correction; a `provider="anthropic"` response from Anthropic's own API
# must NOT get it. The vendor name cannot tell those two apart, so it is the wrong
# key — unlike "workers-ai" above, which names a vendor reachable through exactly
# one surface and so works as a provider entry.
#
# Measured on `system.ai_gateway.usage`, 246 rows across 6 vendors: `total_tokens
# == input + output` for EVERY vendor group, with cache_read and cache_write inside
# input and reasoning inside output. Anthropic's own API reports the exact opposite
# (cache_read=3962 against input=9, additive), which is why keying on the vendor
# over-billed a real backfill 1.570x — 48,798 tokens reported against 31,091
# consumed, the excess being exactly cache_read + cache_write.
#
# Cloudflare AI Gateway is deliberately ABSENT: its logs preserve each vendor's
# native shape instead of normalising them. A real Anthropic entry there reads
# input=10, output=4, total=14 with input_cached_tokens=3429 sitting OUTSIDE that
# total — additive, exactly like the native API — so the provider-keyed sets are
# already right for it and adding it here would UNDER-bill the cached portion.
#
# Neither Snowflake Cortex surface qualifies, established from real rows in INT-224
# and left out on purpose. The REST view is additive: `TOKENS` equals the sum of every
# `TOKENS_GRANULAR` value on 24 of 24 captured rows, so `input` EXCLUDES the cached
# block (`rest_cache_read.json`, and the wire agrees — see INPUT_INCLUDES_CACHE_READ
# above). It also spells its cache keys `cache_read_input` / `cache_write_input`,
# which no other surface in this tree uses. The functions view reports no cache keys
# at all. Adding either would drop a cached row from 4,698 tokens to 14; INT-221's
# reconciliation test asserts the sum against Snowflake's own TOKENS column and fails
# if someone does.
OPENAI_SHAPED_APIS = frozenset(
    {
        "databricks_gateway",
        # Ramp Router normalizes the NUMBERS to OpenAI's convention, not just the schema —
        # measured 2026-08-28 against a live account, on an ANTHROPIC-served model, which
        # is the case that would diverge if anything did: a warm cache_control call
        # reported input_tokens unchanged (18825) with cached_tokens 18810 INSIDE it,
        # total = input + output exactly (06b_real_cache_control_warm.json); reasoning
        # came back inside output (128 of 169, total = input + output,
        # 07_real_reasoning.json). cache_write is inferred from the same normalization
        # rather than measured — every observed write reported cache_write_tokens: 0
        # while the warm read proved the cache existed, and the arithmetic keeps the
        # write inside input — re-verify the day a nonzero write appears in a capture.
        "ramp_router",
    }
)

# Every provider name the SDK's own code can stamp on a CanonicalUsage, so that
# absence from the sets above is always a recorded DECISION and never a default
# nobody made. The convention for a name not in any set is "everything additive" —
# correct for Anthropic-style reporters and for gateway vendors whose logs
# preserve the native shape, and the conservative direction for an unknown (it
# can over-count a subset into the total but never silently drop generated
# tokens). test_token_semantics.py pins this list against the stamps in the
# adapters and wrappers; when adding a provider, add it here IN THE SAME CHANGE
# as its (measured) set entries, per the recipe in CONTRIBUTING.md.
#
# The bedrock_* names are the vendor spellings `_provider_from_model` can emit
# for Bedrock model ids. Bedrock reports cache counts ADDITIVELY for every
# vendor it hosts (its `inputTokens` excludes `cacheRead/WriteInputTokens`), and
# today only its Anthropic and Nova families cache at all — both stamped with
# names absent from the subset sets, so the additive default is the measured
# answer. If Bedrock ever enables caching for a vendor whose name IS in a subset
# set ("openai" via gpt-oss, "mistral"), the bedrock adapters must start
# stamping a surface-distinct api the way databricks_gateway does — the vendor
# name alone would answer wrongly there.
KNOWN_PROVIDERS = frozenset(
    {
        # adapters/, by inference or wrapper hint
        "openai",
        "workers-ai",
        "anthropic",
        "gemini",
        "mistral",
        "databricks",
        "snowflake",
        # Ramp Router, from the wrapper's host hint. Its convention lives in
        # OPENAI_SHAPED_APIS (the adapter stamps api="ramp_router", and the surface
        # wins): measured OpenAI-normalized on cache_read and reasoning, see the entry
        # there.
        "ramp_router",
        # adapters/bedrock_*, from _provider_from_model
        "amazon",
        "meta",
        "cohere",
        "qwen",
        "google",
        "minimax",
        "nvidia",
        "zai",
        "bedrock",
    }
)


def token_semantics(provider: str, api: str) -> tuple[bool, bool, bool]:
    """Which of a record's subsets are ALREADY inside their parent count.

    Returns ``(input_includes_cache_read, input_includes_cache_write,
    output_includes_reasoning)``, the three overlaps the billing paths have to
    remove and the total_tokens guard has to leave alone. The SURFACE wins over
    the vendor: a gateway that re-reports usage in its own shape has already
    decided the convention, so `api` is checked first and the provider-keyed
    sets only answer for a native call.
    """
    p = (provider or "").lower()
    shaped = (api or "").lower() in OPENAI_SHAPED_APIS
    return (
        shaped or p in INPUT_INCLUDES_CACHE_READ,
        shaped or p in INPUT_INCLUDES_CACHE_WRITE,
        shaped or p in OUTPUT_INCLUDES_REASONING,
    )

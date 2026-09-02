"""OpenAI native adapter — verified against real fixtures.

Handles both Chat Completions API (`client.chat.completions.create`) and the
Responses API (`client.responses.create`). They share a similar concept but
use different field names — we detect which by looking at the usage shape.

CHAT COMPLETIONS field mapping (`usage.*`):
  prompt_tokens                                    → input
  completion_tokens                                → output
  prompt_tokens_details.cached_tokens              → cache_read
  prompt_tokens_details.audio_tokens               → audio_input
  completion_tokens_details.reasoning_tokens       → reasoning   (o-series models)
  completion_tokens_details.audio_tokens           → audio_output (GPT-4o-audio output)
  count of choices[0].message.tool_calls           → tool_calls

RESPONSES API field mapping (`usage.*`):
  input_tokens                                     → input
  output_tokens                                    → output
  input_tokens_details.cached_tokens               → cache_read
  output_tokens_details.reasoning_tokens           → reasoning
  count of output[].type == "function_call"        → tool_calls

Not exposed by either API:
  cache_write, cache_write_5m, cache_write_1h — OpenAI auto-caches without
  surfacing creation counts.

Known gaps (intentional, documented):
  - completion_tokens_details.accepted_prediction_tokens — Predicted Outputs
    feature: subset of completion_tokens (the ones that matched the prediction).
    Skipped to avoid double-counting against completion_tokens.
  - completion_tokens_details.rejected_prediction_tokens — Predicted Outputs:
    extra cost beyond completion_tokens (prediction tokens the model rejected).
    Skipped for v1 — customers using Predicted Outputs can read this from
    `extras["completion_tokens_details"]` (if drift-detection captures it) or
    via the openai response object directly.
"""

from __future__ import annotations

import re
from typing import Any, cast

from ..canonical import WORKERS_AI_COMPAT_PREFIX, CanonicalUsage
from ..token_semantics import token_semantics
from ._common import resolve_model

# Cloudflare Workers AI names every model "@cf/<vendor>/<model>". Reaching one
# through the gateway's OpenAI-compatible `/compat` endpoint additionally requires
# the "workers-ai/" routing prefix, so the same model arrives under two spellings
# depending on which surface the customer used. `pricing.lookup_cloudflare_workers_ai`
# strips the routing prefix before matching, because Cloudflare's own catalog lists
# only the bare form.
_WORKERS_AI_MODEL_PREFIX = "@cf/"

# Top-level usage fields we recognize across BOTH chat completions and responses APIs.
#: Provider stamped on any call that reached a model through Ramp Router.
#:
#: Router is an OpenAI-Responses-compatible gateway in front of OpenAI, Anthropic, Google
#: Vertex, Fireworks and xAI. It is treated as a provider in its own right here, rather
#: than resolved to the vendor that actually served the call, because Router's model ids
#: are ACCOUNT-SPECIFIC and opaque — its docs are explicit: "Valid model IDs are
#: account-specific. They come from `GET /v1/models`. Never invent one or reuse a
#: provider's public model name." So unless the caller named an explicit
#: `provider:provider-model` candidate, nothing in the response says who served it, and
#: the model-string rule `_infer_provider` uses cannot see it.
#:
#: "ramp_router" is in `TOKEN_BILLED_PROVIDERS` and deliberately absent from
#: `_VENDOR_MAP`. Two distinct things would otherwise go wrong at once:
#:
#:   * A price lookup under a guessed vendor can be flatly wrong. Router bills at list
#:     price on its shared key but $0 for a BYOK-served request, and a non-default
#:     service tier bills at a rate its own catalog says "may differ" from the base one.
#:   * The overlap semantics belong to ROUTER, not to the served vendor. Measured
#:     2026-08-28 on an Anthropic-served model — the case that would diverge if anything
#:     did: Router normalizes the NUMBERS to OpenAI's convention, not just the schema
#:     (cached block INSIDE input, reasoning inside output; fixtures
#:     06b_real_cache_control_warm.json / 07_real_reasoning.json). So stamping the served
#:     vendor would de-overlap with the WRONG convention whenever that vendor's native
#:     one differs — "ramp_router" carries its own OPENAI_SHAPED_APIS entry instead.
#:
#: Token mode is unaffected and exact either way: it emits the counts Router reported.
#: Price mode routes to those same token events via `TOKEN_BILLED_PROVIDERS`, with no
#: per-call price-miss report — a structural, permanent miss must not cry wolf on the
#: error hook (the same decision Databricks and Snowflake got). The catalog DOES publish
#: per-model rates (`router.pricing`, 01_real_models_catalog.json), so a Router price
#: mode is buildable — but the response still cannot say whether a BYOK key served the
#: call ($0) or which tier rate applied, and every observed catalog entry carries an
#: EMPTY input rate, so token counts stay the honest default.
RAMP_ROUTER_PROVIDER = "ramp_router"

# Router's documented service tiers, appearing as the third segment of a candidate id
# (`openai:gpt-5.4-mini:flex`). OpenAI sells `auto`/`default`/`flex`/`priority`,
# Fireworks `default`/`priority`.
#
# Matched against this set rather than read as "whatever follows the last colon": a model
# segment may contain a colon of its own, and mistaking one for a tier would silently
# rename the model and split it into a second row in Lago. An unrecognized trailing
# segment therefore stays part of the model, which is recoverable; a renamed model is
# not.
_ROUTER_SERVICE_TIERS = frozenset({"auto", "default", "flex", "priority"})

# A provider segment is a short lowercase token. Anything else is part of the model.
_ROUTER_PROVIDER_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")


def _parse_router_model(model_id: str) -> tuple[str, str, str]:
    """Split a Router model id into its (provider, model, service-tier) parts.

    Router names a model two ways, and only one of them is parseable. A plain `model` is
    an account-specific id that reveals nothing; a `models` candidate is
    `provider:provider-model[:service-tier]`. Both arrive in the same response field, so
    this decides which one it is looking at rather than assuming.

    Split on the FIRST colon, never on all of them: Fireworks candidates carry a path as
    their model segment (`fireworks:accounts/fireworks/models/kimi-k2p7-code`), so a
    naive split loses everything after the second separator.

    The model comes back BARE, provider prefix and tier stripped, so a model served
    through Router rolls up in Lago against the same name a direct call to it reports.
    Leaving the prefix on splits one model across two rows for no billing benefit.
    """
    first_colon = model_id.find(":")
    if first_colon <= 0:
        return "", model_id, ""

    head = model_id[:first_colon].lower()
    rest = model_id[first_colon + 1 :]
    # A head that is not a plausible provider token — a path, or something long — means
    # this is an opaque id that merely happens to contain a colon, not a candidate.
    if not rest or not _ROUTER_PROVIDER_SEGMENT.match(head):
        return "", model_id, ""

    tier = ""
    last_colon = rest.rfind(":")
    if last_colon > 0:
        trailing = rest[last_colon + 1 :].lower()
        if trailing in _ROUTER_SERVICE_TIERS:
            tier = trailing
            rest = rest[:last_colon]
    return head, rest, tier


_KNOWN_USAGE_FIELDS = {
    # chat completions
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_tokens_details",
    "completion_tokens_details",
    # responses API
    "input_tokens",
    "output_tokens",
    "input_tokens_details",
    "output_tokens_details",
}

# Nested keys inside the *_tokens_details sub-objects that we actually MAP onto a
# CanonicalUsage field. Anything nested that isn't listed here is drift and gets
# surfaced in `extras` under a dotted key.
#
# Sweeping only top-level keys was a real hole: `prompt_tokens_details` is itself
# a KNOWN top-level key, so nothing inside it was ever inspected. A live
# gpt-5.6-sol response carries `prompt_tokens_details.cache_write_tokens: 3022`
# and those 3022 tokens vanished with no error — a silent violation of the drift
# contract test_drift.py exists to pin, which passed only because it never looked
# one level down.
#
# NOTE the billing subtlety: cache_write_tokens must NOT be mapped to
# CanonicalUsage.cache_write. For OpenAI it sits INSIDE prompt_tokens (measured:
# prompt_tokens=3025 with cache_write_tokens=3022) and bills at the plain input
# rate — Databricks charged exactly what billing all 3025 as input produces. But
# OpenRouter does publish a separate cache_write rate for the model, so mapping it
# would charge those tokens twice: $0.0341 against a true $0.0152, a 2.24x
# over-bill. Anthropic is the opposite — its cache_creation_input_tokens sits
# OUTSIDE input_tokens, which is why mapping is correct there and wrong here.
# Surfacing in extras keeps the field visible without touching the money.
_MAPPED_DETAIL_FIELDS = {
    "prompt_tokens_details": {"cached_tokens", "audio_tokens"},
    "input_tokens_details": {"cached_tokens", "audio_tokens"},
    "completion_tokens_details": {"reasoning_tokens", "audio_tokens"},
    # NOTE `output_tokens_details` deliberately omits `audio_tokens`: the Responses
    # branch hardcodes `audio_output = 0` because the API does not expose it today, so
    # listing it here would exclude a real, unmapped count from `extras` — 500 audio
    # tokens vanishing with no error, which is the exact hole this table closes. Add it
    # back only together with a Responses branch that reads it.
    "output_tokens_details": {"reasoning_tokens"},
}


def _safe_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _safe_int(v: Any) -> int:
    try:
        return max(0, int(v or 0))
    except (TypeError, ValueError):
        return 0


def _to_dict(obj: Any) -> dict[str, Any]:
    """Best-effort pydantic-or-dict to dict (OpenAI SDK returns pydantic objects)."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return cast(dict[str, Any], obj.model_dump())
        except Exception:  # noqa: BLE001
            pass
    return {}


def _count_chat_tool_calls(resp: dict[str, Any]) -> int:
    """choices[0].message.tool_calls is a list of called functions in Chat Completions."""
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return 0
    first = choices[0]
    if not isinstance(first, dict):
        return 0
    message = _safe_dict(first.get("message"))
    tcs = message.get("tool_calls")
    return len(tcs) if isinstance(tcs, list) else 0


def _count_responses_tool_calls(resp: dict[str, Any]) -> int:
    """In the Responses API, tool invocations are items in `output` with type == "function_call"."""
    output = resp.get("output")
    if not isinstance(output, list):
        return 0
    return sum(1 for item in output if isinstance(item, dict) and item.get("type") == "function_call")


def _infer_provider(resolved_model: str) -> str:
    """The SDK shape only ever tells you "this looks like an OpenAI response" —
    it can't tell you who actually served it. Going through a gateway's
    OpenAI-compatible endpoint (e.g. Cloudflare's `.../compat`), the resolved
    model string is the only real signal: "@cf/..." is Cloudflare Workers AI's
    own naming convention, never a real OpenAI model. This isn't cosmetic —
    `provider` is what price-mode keys pricing off of, and Workers AI has a
    genuinely different price table (Cloudflare's own catalog) than real
    OpenAI models (OpenRouter); stamping "openai" on a Workers AI call would
    have made it permanently unpriceable, quietly, at the extraction layer.

    BOTH spellings have to match. Cloudflare's `/compat` endpoint takes the
    provider-prefixed form — `workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast`
    — which is what the README and the demo notebook prescribe, and what a
    streaming call always reports (the synthetic usage payload carries no model,
    so `resolve_model` falls back to the requested string verbatim). Matching
    only the bare `@cf/` left every documented Workers AI call stamped "openai",
    priced against OpenRouter, missed, and silently degraded to token events.
    """
    if resolved_model.startswith(_WORKERS_AI_MODEL_PREFIX) or resolved_model.startswith(
        f"{WORKERS_AI_COMPAT_PREFIX}{_WORKERS_AI_MODEL_PREFIX}"
    ):
        return "workers-ai"
    return "openai"


def extract_openai_native(response: Any, model_id: str = "", provider_hint: str = "") -> CanonicalUsage:
    """Translate an OpenAI response (chat completion or responses API) → CanonicalUsage.

    Accepts the SDK's pydantic objects, dicts (e.g. captured fixtures), or the
    synthetic `{"usage": {...}}` blob produced by the streaming wrapper.

    `provider_hint` overrides the model-string inference below. Only the wrapper
    can supply it, because the only reliable signal for some gateways is the
    client's `base_url` — which the response never carries. Databricks is the
    case that forced it: a Databricks-HOSTED model answers on
    `/ai-gateway/mlflow/v1` but echoes a served-entity name
    ("meta-llama-4-maverick-040225") with no marker of its own, so no rule based
    on the model string can identify it. See `wrappers/openai.py`.
    """
    resp = _to_dict(response) if not isinstance(response, dict) else response
    usage = _safe_dict(resp.get("usage"))

    # Detect which API shape we have. Chat Completions uses prompt_tokens;
    # Responses API uses input_tokens. They never both appear.
    is_responses_api = "input_tokens" in usage and "prompt_tokens" not in usage

    # `cache_write` here is the RAW reported count, kept out of CanonicalUsage
    # (see _MAPPED_DETAIL_FIELDS) and read per-branch from that branch's own
    # details container — the Responses API spells it input_tokens_details, so a
    # single prompt_tokens_details lookup would leave the Responses branch
    # answering the total_tokens reconciliation below differently from the chat
    # branch for the same convention.
    if is_responses_api:
        input_tokens = _safe_int(usage.get("input_tokens"))
        output_tokens = _safe_int(usage.get("output_tokens"))
        input_details = _safe_dict(usage.get("input_tokens_details"))
        output_details = _safe_dict(usage.get("output_tokens_details"))
        cache_read = _safe_int(input_details.get("cached_tokens"))
        cache_write = _safe_int(input_details.get("cache_write_tokens"))
        reasoning = _safe_int(output_details.get("reasoning_tokens"))
        audio_input = _safe_int(input_details.get("audio_tokens"))
        audio_output = 0  # not exposed by Responses API today
        tool_calls = _count_responses_tool_calls(resp)
        api = "responses"
    else:
        input_tokens = _safe_int(usage.get("prompt_tokens"))
        output_tokens = _safe_int(usage.get("completion_tokens"))
        prompt_details = _safe_dict(usage.get("prompt_tokens_details"))
        completion_details = _safe_dict(usage.get("completion_tokens_details"))
        cache_read = _safe_int(prompt_details.get("cached_tokens"))
        cache_write = _safe_int(prompt_details.get("cache_write_tokens"))
        reasoning = _safe_int(completion_details.get("reasoning_tokens"))
        audio_input = _safe_int(prompt_details.get("audio_tokens"))
        audio_output = _safe_int(completion_details.get("audio_tokens"))
        tool_calls = _count_chat_tool_calls(resp)
        api = "chat_completions"

    extras: dict[str, Any] = {}
    for k, v in usage.items():
        if k not in _KNOWN_USAGE_FIELDS:
            extras[k] = v

    # Drift sweep one level down, into the *_tokens_details sub-objects. Without
    # this, an unrecognized nested field is silently dropped (see
    # _MAPPED_DETAIL_FIELDS) because its container is a known top-level key.
    for container, mapped in _MAPPED_DETAIL_FIELDS.items():
        for k, v in _safe_dict(usage.get(container)).items():
            if k not in mapped:
                extras[f"{container}.{k}"] = v

    resolved_model = resolve_model(resp.get("model"), model_id)
    provider = provider_hint or _infer_provider(resolved_model)

    # MUST stay ABOVE the total_tokens guard below. The guard asks
    # `token_semantics(provider, api)` the same convention question `compute_cost` and
    # `deoverlapped_token_total` ask, and Router is the one surface here that REASSIGNS
    # `api` mid-function. Read before the reassignment, the guard sees
    # ("ramp_router", "responses") — all-additive — while the money paths see the
    # stamped api="ramp_router" and de-overlap as subset. That divergence is exactly
    # what token_semantics.py exists to make impossible, and it under-folds a genuine
    # remainder by cache_read + reasoning, or suppresses the fold entirely when the
    # over-count exceeds the declared total — silently, with no on_error.
    #
    # `resolve_model` prefers the response's own model over the requested one, which is
    # what makes a Router fallback bill correctly with no extra work: a `models` request
    # sends no `model` at all, and Switchyard routing can serve a different model than
    # the one asked for, so the response is the only place the SERVED model appears.
    model = resolved_model
    if provider_hint == RAMP_ROUTER_PROVIDER:
        router_provider, parsed_model, tier = _parse_router_model(resolved_model)
        if router_provider:
            model = parsed_model
            # Recorded, not promoted to `provider` — see RAMP_ROUTER_PROVIDER for why
            # the served vendor cannot drive the de-overlap convention.
            extras["router_provider"] = router_provider
        # The tier is billing-relevant on its own: Router's catalog says tiers "may use
        # different rates" than the base ones it publishes, so a pinned non-default tier
        # is the difference between a correct price and an over-bill at the standard
        # rate.
        if tier:
            extras["service_tier"] = tier
        # Which of Router's two OpenAI-shaped surfaces answered. Router documents only
        # `/v1/responses` (`/v1/chat/completions` 404s), so a `chat_completions` value
        # here is drift worth seeing rather than a case to handle.
        extras["router_surface"] = api
        api = RAMP_ROUTER_PROVIDER

    # Consistency guard: for genuine OpenAI, total_tokens always equals
    # prompt + completion (reasoning is a SUBSET of completion, never additive).
    # Verified across every fixture under openai_native/ — zero deltas. So a
    # POSITIVE delta means tokens exist that neither named bucket accounts for,
    # which only happens behind an OpenAI-COMPATIBLE proxy that under-reports.
    #
    # Measured on Gemini through Google's own OpenAI-compat layer:
    # prompt_tokens=57, completion_tokens=47, total_tokens=1253 — 1149 real
    # thinking tokens reported nowhere, and no completion_tokens_details to
    # recover them from. Billing prompt+completion drops 92% of the call, at the
    # output rate. Folding the remainder into `output` is the honest read: the
    # provider's own total proves those tokens were generated.
    #
    # Deliberately NOT assigned to `reasoning`: compute_cost zeroes reasoning for
    # providers in OUTPUT_INCLUDES_REASONING, so for real OpenAI that would set the
    # field and immediately discard it, recovering nothing.
    #
    # WHAT COUNTS AS ACCOUNTED is a per-provider fact, not a payload fact. The
    # wire is one shape, but the convention behind it splits: OpenAI puts the
    # cache and reasoning counts INSIDE prompt/completion, while Snowflake
    # Cortex answers on the same wire with Anthropic's ADDITIVE convention —
    # measured 2026-08-25, prompt_tokens=7, cached_tokens=4805,
    # completion_tokens=6, total_tokens=4818: the cached block sits OUTSIDE
    # prompt_tokens and INSIDE total_tokens. Accounting for input+output only
    # made those 4,805 cached tokens look unaccounted, so they were folded into
    # `output` — 4,811 reported for a call that generated 6, while the same
    # tokens also shipped as cache_read. 2.0x on the call, 800x on the output
    # line. See 12_snowflake_cortex_cache_chat.json.
    #
    # So the accounted sum adds each subset field exactly when the provider
    # reports it OUTSIDE its parent count, read from the same token_semantics
    # table compute_cost and deoverlapped_token_total bill from — the guard and
    # the money paths cannot answer the convention question differently. It
    # cannot be decided from the payload instead: `accounted <= total` admits a
    # small subtractive cache (folds too little), `cache_read > input` rejects a
    # small additive one (folds tokens never generated) — both were tried
    # against real shapes and both leak. And subtracting unconditionally
    # disarms the guard where it is load-bearing: on a SUBSET surface the cache
    # is already inside `input`, so also adding it to the accounted sum eats a
    # genuine remainder — Gemini-compat's own cached+thinking payload would
    # under-fold by exactly the cached count, silently, with no on_error.
    #
    # `cache_write` is the raw prompt/input_tokens_details count because it is
    # deliberately NOT mapped to CanonicalUsage.cache_write (for OpenAI it sits
    # inside prompt_tokens and billing it separately over-charges 2.24x — see
    # _MAPPED_DETAIL_FIELDS), yet an additive cache WRITE would inflate `output`
    # exactly the way the read did. `cache_write_tokens` is the only spelling
    # accounted for — an OpenAI-compat proxy re-reporting Anthropic's
    # `cache_creation_input_tokens` (or `cache_creation.*`) inside a details
    # block would still fold. Known limit: those spellings land in `extras` via
    # the drift sweep, which is the signal to add them HERE, deliberately —
    # deriving the accounting from the sweep itself would assume every unmapped
    # count is additive, the same payload-only guess ruled out above.
    #
    # The residual: an UNRECOGNIZED additive proxy arrives as provider="openai"
    # and folds its cached block, exactly as Cortex did before its base_url rule
    # existed. The payload carries no convention, so identification (a
    # _provider_hint_for entry) is the fix — not loosening this arithmetic.
    #
    # A no-op for real OpenAI either way: total always equals prompt + completion.
    declared_total = _safe_int(usage.get("total_tokens"))
    if declared_total:
        inc_cache_read, inc_cache_write, inc_reasoning = token_semantics(provider, api)
        accounted = input_tokens + output_tokens
        if not inc_reasoning:
            accounted += reasoning
        if not inc_cache_read:
            accounted += cache_read
        if not inc_cache_write:
            accounted += cache_write
        unaccounted = declared_total - accounted
        if unaccounted > 0:
            output_tokens += unaccounted
            extras["unaccounted_output_tokens"] = unaccounted

    return CanonicalUsage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        reasoning=reasoning,
        audio_input=audio_input,
        audio_output=audio_output,
        tool_calls=tool_calls,
        model=model,
        provider=provider,
        api=api,
        extras=extras,
    )

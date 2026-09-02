"""Ramp Router live path — fake client, no live API."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from lago_agent_sdk import LagoSDK
from lago_agent_sdk.adapters.openai_native import RAMP_ROUTER_PROVIDER, extract_openai_native
from lago_agent_sdk.pricing import (
    TOKEN_BILLED_PROVIDERS,
    PricingProvider,
    parse_openrouter,
)
from lago_agent_sdk.token_semantics import KNOWN_PROVIDERS, token_semantics
from lago_agent_sdk.wrappers.openai import _provider_hint_for

ROUTER_BASE_URL = "https://api.router.com/v1"


def router_response(model: str, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    """A Router response, in the shape its docs specify: "Every request and response
    uses the OpenAI Responses schema, whichever provider serves it."

    Hand-built rather than captured, and deliberately so for now: these tests pin the
    SDK's own decisions — detection, candidate parsing, which field becomes the model —
    none of which depend on Router's exact numbers. The assertions that need real
    numbers live with the captured fixtures.
    """
    return {
        "id": "resp_test",
        "object": "response",
        "model": model,
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "pong"}]}],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 3,
            "total_tokens": 14,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
            **(usage or {}),
        },
    }


class _FakeStreamChunk:
    """Mimics a Responses-API stream event."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class FakeRouterResponses:
    def __init__(self, reply: Any) -> None:
        self._reply = reply
        self.create_calls = 0
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.create_calls += 1
        assert "extra_lago" not in kwargs  # wrapper must strip lago opts
        self.last_kwargs = dict(kwargs)
        return self._reply(kwargs)


class FakeRouterClient:
    def __init__(self, base_url: str, reply: Any) -> None:
        self.base_url = base_url
        self.responses = FakeRouterResponses(reply)


# The detector keys on the module; Router is reached with an OpenAI client.
FakeRouterClient.__module__ = "openai.fake"


def _new_sdk(default_sub: str = "sub_test", **config: Any) -> tuple[LagoSDK, list[dict]]:
    received: list[dict] = []

    def sender(batch: list[dict]) -> None:
        received.extend(batch)

    from lago_agent_sdk import LagoConfig

    cfg = LagoConfig(**config) if config else None
    sdk = LagoSDK(api_key="dummy", default_subscription_id=default_sub, config=cfg)
    sdk._queue._sender = sender  # type: ignore[attr-defined]
    return sdk, received


def _by_code(received: list[dict]) -> dict[str, float]:
    """code -> numeric value, the reduction every wrapper test in this repo uses."""
    return {e["code"]: float(e["properties"]["value"]) for e in received}


# ----------------------------------------------------------------------
# Detection. `base_url` is the ONLY signal: Router's model ids are
# account-specific and opaque, and an Anthropic-served response arrives in
# OpenAI's schema, so nothing in the response body distinguishes the two.
# ----------------------------------------------------------------------
class _Base:
    def __init__(self, base_url: Any) -> None:
        self.base_url = base_url


@pytest.mark.parametrize(
    "base_url,expected",
    [
        ("https://api.router.com/v1", RAMP_ROUTER_PROVIDER),
        ("https://api.router.com/v1/", RAMP_ROUTER_PROVIDER),
        ("https://API.Router.COM/v1", RAMP_ROUTER_PROVIDER),
        # A regional or staging host under the same domain still bills as Router.
        ("https://api-eu.router.com/v1", RAMP_ROUTER_PROVIDER),
        # Direct providers and other gateways must be untouched.
        ("https://api.openai.com/v1", ""),
        ("https://gateway.ai.cloudflare.com/v1/acct/gw/compat", ""),
    ],
)
def test_detection_base_url_is_the_only_signal(base_url: str, expected: str) -> None:
    assert _provider_hint_for(_Base(base_url)) == expected


def test_a_lookalike_host_that_merely_contains_the_router_path_is_not_router() -> None:
    """The reason detection parses the host instead of a substring test: a substring
    stamps this unrelated endpoint's traffic as Router-served."""
    assert _provider_hint_for(_Base("https://evil.example.com/api.router.com/v1")) == ""
    assert _provider_hint_for(_Base("https://evilrouter.com/v1")) == ""


def test_a_missing_malformed_or_exotic_base_url_never_throws_out_of_wrap() -> None:
    class _NoUrl:
        pass

    class _Explodes:
        @property
        def base_url(self) -> str:
            raise RuntimeError("client blew up")

    assert _provider_hint_for(_NoUrl()) == ""
    assert _provider_hint_for(None) == ""
    assert _provider_hint_for(_Base("/v1")) == ""
    assert _provider_hint_for(_Base(42)) == ""
    assert _provider_hint_for(_Explodes()) == ""


# ----------------------------------------------------------------------
# Candidate parsing. Router names a model two ways and both arrive in the
# same response field: an opaque account-specific id, or an explicit
# `provider:provider-model[:service-tier]` candidate.
# ----------------------------------------------------------------------
def _extract(model: str) -> Any:
    return extract_openai_native(router_response(model), model_id="", provider_hint=RAMP_ROUTER_PROVIDER)


def test_stamps_api_and_provider_as_ramp_router_keeping_the_surface_in_extras() -> None:
    u = _extract("gpt-5.4-nano")
    assert u.api == RAMP_ROUTER_PROVIDER
    # The provider is NOT the vendor that served the call: Router's overlap semantics
    # are its OWN (measured OpenAI-shaped) — see RAMP_ROUTER_PROVIDER.
    assert u.provider == RAMP_ROUTER_PROVIDER
    assert u.extras["router_surface"] == "responses"


def test_leaves_an_opaque_account_specific_id_exactly_as_reported() -> None:
    """ "Valid model IDs are account-specific... Never invent one or reuse a provider's
    public model name." So there is nothing to parse and nothing to strip."""
    u = _extract("my-org-fast-tier-7")
    assert u.model == "my-org-fast-tier-7"
    assert "router_provider" not in u.extras
    assert "service_tier" not in u.extras


def test_splits_an_explicit_candidate_into_a_bare_model_plus_the_provider() -> None:
    u = _extract("openai:gpt-5.4-mini")
    # Bare, so a Router-served model rolls up in Lago against the same name a direct
    # call to it reports rather than splitting into a second row.
    assert u.model == "gpt-5.4-mini"
    assert u.extras["router_provider"] == "openai"


def test_keeps_a_fireworks_models_whole_path_which_contains_slashes() -> None:
    """The reason the split is on the FIRST colon only. A naive split on every colon
    would keep "accounts" and lose the rest of the id."""
    u = _extract("fireworks:accounts/fireworks/models/kimi-k2p7-code")
    assert u.model == "accounts/fireworks/models/kimi-k2p7-code"
    assert u.extras["router_provider"] == "fireworks"


def test_pulls_a_pinned_service_tier_out_into_extras() -> None:
    """Billing-relevant on its own: Router's catalog says tiers "may use different
    rates" than the base ones it publishes, so pricing must be able to see this."""
    u = _extract("openai:gpt-5.4-mini:flex")
    assert u.model == "gpt-5.4-mini"
    assert u.extras["router_provider"] == "openai"
    assert u.extras["service_tier"] == "flex"


@pytest.mark.parametrize("tier", ["auto", "default", "flex", "priority"])
def test_recognizes_the_documented_tiers(tier: str) -> None:
    u = _extract(f"openai:gpt-5.4-mini:{tier}")
    assert u.extras["service_tier"] == tier
    assert u.model == "gpt-5.4-mini"


def test_treats_an_unrecognized_trailing_segment_as_part_of_the_model_not_a_tier() -> None:
    """A wrongly-stripped segment silently renames the model and splits it into a second
    row in Lago. Keeping it is recoverable; renaming is not."""
    u = _extract("openai:gpt-5.4-mini:turbo")
    assert u.model == "gpt-5.4-mini:turbo"
    assert "service_tier" not in u.extras


def test_does_not_read_a_path_shaped_prefix_as_a_provider() -> None:
    u = _extract("accounts/fireworks/models/foo:bar")
    assert u.model == "accounts/fireworks/models/foo:bar"
    assert "router_provider" not in u.extras


def test_bills_the_served_model_not_the_requested_one() -> None:
    """Two ways requested and served diverge on Router: a `models` fallback list sends
    no `model` field at all, and Switchyard routing can serve a different model than the
    one asked for. The response is the only place the served model appears."""
    u = extract_openai_native(
        router_response("anthropic:claude-haiku-4-5"),
        model_id="openai:gpt-5.4-mini",
        provider_hint=RAMP_ROUTER_PROVIDER,
    )
    assert u.model == "claude-haiku-4-5"
    assert u.extras["router_provider"] == "anthropic"


def test_leaves_a_non_router_clients_provider_inference_alone() -> None:
    u = extract_openai_native(router_response("gpt-4o-mini-2024-07-18"), model_id="")
    assert u.provider == "openai"
    assert u.api == "responses"
    assert "router_surface" not in u.extras


# ----------------------------------------------------------------------
# Token mode is the default and must be exact: the counts Router reported,
# no field invented, none derived.
# ----------------------------------------------------------------------
def test_a_router_pointed_client_bills_with_no_code_change_but_base_url() -> None:
    sdk, received = _new_sdk()
    client = sdk.wrap(FakeRouterClient(ROUTER_BASE_URL, lambda kw: router_response("openai:gpt-5.4-mini")))
    client.responses.create(model="gpt-5.4-mini", input="ping")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    by_code = _by_code(received)
    assert by_code["llm_input_tokens"] == 11
    assert by_code["llm_output_tokens"] == 3
    assert len(received) == 2  # input + output only — total_tokens is derived
    assert all(e["properties"]["model"] == "gpt-5.4-mini" for e in received)


def test_emits_the_same_fields_a_direct_provider_call_would() -> None:
    sdk, received = _new_sdk()
    client = sdk.wrap(
        FakeRouterClient(
            ROUTER_BASE_URL,
            lambda kw: router_response(
                "anthropic:claude-haiku-4-5",
                {
                    "input_tokens": 1200,
                    "output_tokens": 40,
                    "total_tokens": 1240,
                    "input_tokens_details": {"cached_tokens": 900},
                    "output_tokens_details": {"reasoning_tokens": 25},
                },
            ),
        )
    )
    client.responses.create(model="x", input="ping")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    by_code = _by_code(received)
    # Faithful extraction. Whether cache_read sits inside input is a PRICING question,
    # not an extraction one — token mode reports what Router reported either way.
    assert by_code["llm_input_tokens"] == 1200
    assert by_code["llm_output_tokens"] == 40
    assert by_code["llm_cached_input_tokens"] == 900
    assert by_code["llm_reasoning_tokens"] == 25
    # Exactly four events. `total_tokens` is derived from the others, so mapping it
    # would double-count — a fifth event here would mean it had been.
    assert len(received) == 4


def test_a_streamed_call_bills_exactly_once_from_the_terminal_event() -> None:
    def reply(kwargs: dict[str, Any]) -> Any:
        if kwargs.get("stream") is not True:
            return router_response("openai:gpt-5.4-mini")
        # Router returns "OpenAI Responses server-sent events", which nest both usage
        # and the resolved model under `.response`.
        events = [
            _FakeStreamChunk(
                {"type": "response.created", "response": {"id": "resp_1", "model": "openai:gpt-5.4-mini"}}
            ),
            _FakeStreamChunk({"type": "response.output_text.delta", "delta": "po"}),
            _FakeStreamChunk(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "openai:gpt-5.4-mini",
                        "usage": {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
                    },
                }
            ),
        ]
        return iter(events)

    sdk, received = _new_sdk()
    client = sdk.wrap(FakeRouterClient(ROUTER_BASE_URL, reply))
    list(client.responses.create(model="gpt-5.4-mini", input="ping", stream=True))
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    assert len(received) == 2  # one input + one output, not two of each
    assert _by_code(received)["llm_input_tokens"] == 11
    # The stream carries the served candidate too, parsed the same way.
    assert all(e["properties"]["model"] == "gpt-5.4-mini" for e in received)


def test_a_models_fallback_request_bills_the_candidate_that_answered() -> None:
    sdk, received = _new_sdk()
    client = sdk.wrap(
        FakeRouterClient(
            ROUTER_BASE_URL,
            # Second candidate served it. Billing the requested list would bill the
            # wrong model, and the request carried no `model` field to fall back on.
            lambda kw: router_response("fireworks:accounts/fireworks/models/kimi-k2p7-code"),
        )
    )
    client.responses.create(
        models=["openai:gpt-5.4-mini", "fireworks:accounts/fireworks/models/kimi-k2p7-code"],
        input="ping",
    )
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    assert len(received) == 2
    assert received[0]["properties"]["model"] == "accounts/fireworks/models/kimi-k2p7-code"


# ----------------------------------------------------------------------
# A failure must never bill, and a malformed payload must never throw on the
# customer's call path.
# ----------------------------------------------------------------------
_ROUTER_ERRORS = [
    (400, "invalid_request"),
    (401, "invalid_api_key"),
    (401, "api_key_deactivated"),
    (402, "insufficient_credits"),
    (403, "provider_unavailable"),
    (404, "model_not_found"),
    (413, "request_too_large"),
    (429, "rate_limit_exceeded"),
    (500, "internal_error"),
    (501, "not_implemented_error"),
    (502, "provider_request_failed"),
    (502, "all_candidates_failed"),
    (503, "service_unavailable"),
    (504, "provider_request_failed"),
]


@pytest.mark.parametrize("status,code", _ROUTER_ERRORS)
def test_router_errors_emit_nothing(status: int, code: str) -> None:
    """Every status Router's errors-and-limits page documents, with its code."""

    def reply(kwargs: dict[str, Any]) -> Any:
        err = RuntimeError(f"router {status}")
        err.status = status  # type: ignore[attr-defined]
        raise err

    sdk, received = _new_sdk()
    client = sdk.wrap(FakeRouterClient(ROUTER_BASE_URL, reply))
    with pytest.raises(RuntimeError):
        client.responses.create(model="x", input="ping", _code=code)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert received == []


def test_a_zero_usage_response_emits_nothing_rather_than_a_zero_valued_event() -> None:
    sdk, received = _new_sdk()
    client = sdk.wrap(
        FakeRouterClient(
            ROUTER_BASE_URL,
            lambda kw: router_response(
                "openai:gpt-5.4-mini", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            ),
        )
    )
    client.responses.create(model="x", input="ping")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert received == []


@pytest.mark.parametrize(
    "payload",
    [
        # api.router.com sits behind Cloudflare bot management, so a non-2xx can
        # genuinely be an HTML challenge page rather than Router's documented JSON.
        "<!DOCTYPE html><title>Attention Required! | Cloudflare</title>",
        None,
        7,
        {"id": "resp_1", "model": "openai:gpt-5.4-mini"},
        {"id": "resp_1", "model": "openai:gpt-5.4-mini", "usage": None},
        {"model": "openai:gpt-5.4-mini", "usage": {"input_tokens": "nope"}},
        {"model": "openai:gpt-5.4-mini", "usage": {"input_tokens": -5}},
        {"model": None, "usage": {"input_tokens": 4, "output_tokens": 1}},
    ],
)
def test_degrades_to_zero_rather_than_throwing_on_malformed_payloads(payload: Any) -> None:
    u = extract_openai_native(payload, model_id="", provider_hint=RAMP_ROUTER_PROVIDER)
    assert u.api == RAMP_ROUTER_PROVIDER
    assert u.input >= 0


# ----------------------------------------------------------------------
# Price mode. Every Router call currently takes a clean pricing MISS and falls
# back to token events, because no vendor can be assigned to it safely yet.
#
# A real price table is loaded for these tests, and the same model is billed
# both directly and through Router. Without that contrast the tests would pass
# on an empty table, proving nothing: everything misses when nothing is priced.
# ----------------------------------------------------------------------
PRICED_MODEL = "gpt-5.4-mini"
# Built through the real parser from a real-shaped OpenRouter payload, not from a
# hand-written key: `norm()` rewrites "." to "-", and a test whose table silently fails
# to load proves nothing about a miss.
#
# $0.75/M input and $4.50/M output are Router's own published base rates for this model.
_OPENROUTER_TABLE = parse_openrouter(
    {
        "data": [
            {"id": f"openai/{PRICED_MODEL}", "pricing": {"prompt": "0.00000075", "completion": "0.0000045"}}
        ]
    }
)


class _StubFetcher:
    def fetch_openrouter(self) -> dict[str, Any]:
        return _OPENROUTER_TABLE

    def fetch_bedrock(self, region: str) -> dict[str, Any]:
        return {}

    def fetch_cloudflare_workers_ai(self) -> dict[str, Any]:
        return {}

    def fetch_mistral_aliases(self, api_key: str | None = None) -> dict[str, str]:
        return {}


def _priced_sdk() -> tuple[LagoSDK, list[dict], PricingProvider]:
    provider = PricingProvider(fetcher=_StubFetcher(), ttl_seconds=3600.0)
    sdk, received = _new_sdk(pricing_mode="price", pricing_provider=provider)
    # The table has to be warm before the call, or the miss under test is just a cold
    # cache. `maybe_refresh` is the queue worker's own warm-up, called synchronously.
    provider.prime(["openrouter"])
    provider.maybe_refresh()
    return sdk, received, provider


def test_the_same_model_does_price_when_called_directly_the_table_is_real() -> None:
    """The control. If this fails, every "misses" assertion below is vacuous."""
    sdk, received, provider = _priced_sdk()
    assert provider.lookup("openai", PRICED_MODEL, "responses") is not None
    client = sdk.wrap(FakeRouterClient("https://api.openai.com/v1", lambda kw: router_response(PRICED_MODEL)))
    client.responses.create(model=PRICED_MODEL, input="ping")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    codes = [e["code"] for e in received]
    assert "llm_cost" in codes
    assert "llm_input_tokens" not in codes


def test_the_identical_model_through_router_misses_and_falls_back_to_token_events() -> None:
    """Same table, same model, same usage — only the base URL differs. The miss is
    caused by the Router provider vocabulary, which is the decision under test: Router
    bills $0 for a BYOK-served request and a non-default tier at a rate its catalog says
    "may differ", so a list-price lookup can be flatly wrong."""
    sdk, received, _ = _priced_sdk()
    client = sdk.wrap(FakeRouterClient(ROUTER_BASE_URL, lambda kw: router_response(f"openai:{PRICED_MODEL}")))
    client.responses.create(model=PRICED_MODEL, input="ping")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    by_code = _by_code(received)
    assert "llm_cost" not in by_code
    # Not a silent drop. The usage is billed, exactly, as tokens.
    assert by_code["llm_input_tokens"] == 11
    assert by_code["llm_output_tokens"] == 3


def test_a_flex_tier_call_is_never_billed_at_the_base_rate() -> None:
    """supported-models: "Service tiers, long contexts, caching, and other features may
    use different rates." Billing flex at the standard rate over-bills."""
    sdk, received, _ = _priced_sdk()
    client = sdk.wrap(
        FakeRouterClient(ROUTER_BASE_URL, lambda kw: router_response(f"openai:{PRICED_MODEL}:flex"))
    )
    client.responses.create(model="x", input="ping")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert "llm_cost" not in [e["code"] for e in received]


def test_a_pricing_miss_never_reaches_the_caller_as_an_exception() -> None:
    sdk, _, _ = _priced_sdk()
    client = sdk.wrap(FakeRouterClient(ROUTER_BASE_URL, lambda kw: router_response(f"openai:{PRICED_MODEL}")))
    assert client.responses.create(model="x", input="ping") is not None
    sdk.shutdown(timeout=1.0)


# ----------------------------------------------------------------------
# The hot path. Billing is enqueue-only, so concurrency must not lose or
# duplicate an event, and detection must not add per-call work.
# ----------------------------------------------------------------------
def test_200_concurrent_calls_bill_exactly_200_input_events() -> None:
    sdk, received = _new_sdk()
    client = sdk.wrap(FakeRouterClient(ROUTER_BASE_URL, lambda kw: router_response("openai:gpt-5.4-mini")))

    def call() -> None:
        client.responses.create(model="x", input="ping")

    threads = [threading.Thread(target=call) for _ in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sdk.flush(timeout=5.0)
    sdk.shutdown(timeout=2.0)

    inputs = [e for e in received if e["code"] == "llm_input_tokens"]
    assert len(inputs) == 200
    assert len({e["transaction_id"] for e in received}) == len(received)


# ----------------------------------------------------------------------
# The two recorded decisions behind "ramp_router", pinned so neither can be
# reverted silently. The generic roster tests cannot see them: the hint comes
# from the wrapper's HOST arm, not from _PROVIDER_BY_BASE_URL_PATH, so nothing
# else in the suite fails if either set entry disappears.
# ----------------------------------------------------------------------
def test_ramp_router_is_token_billed_a_price_mode_call_emits_token_events_with_no_error_report() -> None:
    """Router is structurally unpriceable today (BYOK requests bill $0, tiers have
    unpublished rates, every observed catalog input rate is empty), so a price miss is
    permanent — and a permanent miss must not cry wolf on the error hook per call. Same
    decision as Databricks and Snowflake."""
    assert RAMP_ROUTER_PROVIDER in TOKEN_BILLED_PROVIDERS

    errors: list[Any] = []
    provider = PricingProvider(fetcher=_StubFetcher(), ttl_seconds=3600.0)
    sdk, received = _new_sdk(
        pricing_mode="price", pricing_provider=provider, on_error=lambda exc, where: errors.append(exc)
    )
    provider.prime(["openrouter"])
    provider.maybe_refresh()
    client = sdk.wrap(FakeRouterClient(ROUTER_BASE_URL, lambda kw: router_response("openai:gpt-5.4-mini")))
    client.responses.create(model="x", input="ping")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert sorted(e["code"] for e in received) == ["llm_input_tokens", "llm_output_tokens"]
    assert errors == []


def test_ramp_routers_token_convention_is_a_recorded_measurement_openai_shaped_on_every_axis() -> None:
    """Measured live 2026-08-28, on an Anthropic-served model — the case that would
    diverge if anything did: a warm cache_control call reported the cached block INSIDE
    input_tokens (06b_real_cache_control_warm.json), and reasoning came back inside
    output (07_real_reasoning.json). Router normalizes the NUMBERS to OpenAI's
    convention, not just the schema. The entry lives in OPENAI_SHAPED_APIS because the
    adapter stamps api="ramp_router" and the surface wins over the vendor."""
    assert RAMP_ROUTER_PROVIDER in KNOWN_PROVIDERS
    assert token_semantics(RAMP_ROUTER_PROVIDER, RAMP_ROUTER_PROVIDER) == (True, True, True)


# ----------------------------------------------------------------------
# Ordering of the api stamp against the total_tokens guard. Router is the
# only surface in this tree that REASSIGNS `api` mid-extract, so the stamp
# has to land before the guard reads it.
# ----------------------------------------------------------------------
def _misreporting_router_response(total: int) -> dict[str, Any]:
    """A Router payload whose declared total does NOT equal input + output, with both
    subsets non-zero. No captured fixture has this shape — all ten report
    total == input + output, streamed included — so this is the only cover the guard's
    Router branch has."""
    return router_response(
        "gpt-5.4-nano",
        {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": total,
            "input_tokens_details": {"cached_tokens": 80},
            "output_tokens_details": {"reasoning_tokens": 30},
        },
    )


def test_the_totals_guard_reads_the_stamped_router_api_not_the_pre_stamp_surface() -> None:
    """The guard, compute_cost and deoverlapped_token_total must answer the overlap
    question identically — the whole reason token_semantics.py exists. Read before the
    stamp, the guard sees ("ramp_router", "responses"), which is in no subset set, and so
    adds cache_read + reasoning to an accounted sum that already contains them.
    """
    u = extract_openai_native(_misreporting_router_response(1000), provider_hint=RAMP_ROUTER_PROVIDER)
    # 1000 - (100 + 50). The cached block sits INSIDE input and reasoning INSIDE output,
    # so neither is accounted twice; folding 740 would lose exactly cache_read + reasoning.
    assert u.extras["unaccounted_output_tokens"] == 850
    assert u.output == 50 + 850
    # Read from before the stamp — moving the block above the guard must not cost this.
    assert u.extras["router_surface"] == "responses"


def test_a_router_remainder_smaller_than_its_subsets_still_folds_rather_than_vanishing() -> None:
    """The suppression case, and the one that loses money silently rather than merely
    under-counting: with the wrong semantics the accounted sum (260) EXCEEDS the declared
    total, `unaccounted` goes negative, the guard never fires, and 50 generated tokens are
    dropped with no extras key and no on_error report."""
    u = extract_openai_native(_misreporting_router_response(200), provider_hint=RAMP_ROUTER_PROVIDER)
    assert u.extras["unaccounted_output_tokens"] == 50
    assert u.output == 100

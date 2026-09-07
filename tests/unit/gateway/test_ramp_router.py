"""Ramp Router live path — fake client, no live API."""

from __future__ import annotations

import json
import pathlib
import threading
from decimal import Decimal
from typing import Any

import pytest

from lago_agent_sdk import LagoSDK
from lago_agent_sdk.adapters.anthropic_native import RAMP_ROUTER_MESSAGES_API, extract_anthropic_native
from lago_agent_sdk.adapters.openai_native import RAMP_ROUTER_PROVIDER, extract_openai_native
from lago_agent_sdk.exceptions import PricingUnavailableError
from lago_agent_sdk.pricing import (
    TOKEN_BILLED_PROVIDERS,
    PricingProvider,
    lookup_ramp_router,
    parse_openrouter,
    parse_ramp_router,
)
from lago_agent_sdk.token_semantics import KNOWN_PROVIDERS, OPENAI_SHAPED_APIS, token_semantics
from lago_agent_sdk.wrappers.anthropic import _merge_stream_usage
from lago_agent_sdk.wrappers.openai import _provider_hint_for
from lago_agent_sdk.wrappers.ramp_router import client_points_at_ramp_router, is_ramp_router_base_url

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
# Price mode. A Router call prices against Router's OWN catalog — the rate the
# gateway bills, reconciled exact against a live account's dashboard export —
# never against OpenRouter's listing for the "same" model.
#
# The Router table is built through the real parser from the REAL captured
# catalog, and an OpenRouter table listing the same model at a DIFFERENT rate
# is loaded beside it. Without that contrast a test could pass by pricing from
# the wrong table, and a table that silently failed to load would make every
# assertion below vacuous — so the control test prices the same model directly.
# ----------------------------------------------------------------------
_CATALOG_FIXTURE = (
    pathlib.Path(__file__).parents[1]
    / "adapters"
    / "fixtures"
    / "ramp_router"
    / "01_real_models_catalog.json"
)
_ROUTER_TABLE = parse_ramp_router(json.loads(_CATALOG_FIXTURE.read_text())["_body"])
PRICED_MODEL = "gpt-5.4-nano"  # Router's catalog: $0.20/M input, $1.25/M output, $0.02/M cached
SERVED_MODEL = f"{PRICED_MODEL}-2026-03-17"  # what Router actually answers with (fixture 02)
# OpenRouter deliberately lists it at a rate that is NOT Router's, so a cost event priced
# from the wrong table shows up in the numbers, not only in `price_source`.
_OPENROUTER_TABLE = parse_openrouter(
    {"data": [{"id": f"openai/{PRICED_MODEL}", "pricing": {"prompt": "0.000001", "completion": "0.000001"}}]}
)


class _StubFetcher:
    def __init__(self, router_table: dict[str, Any] | None = None) -> None:
        self._router = _ROUTER_TABLE if router_table is None else router_table
        self.ramp_router_keys: list[str | None] = []

    def fetch_openrouter(self) -> dict[str, Any]:
        return _OPENROUTER_TABLE

    def fetch_bedrock(self, region: str) -> dict[str, Any]:
        return {}

    def fetch_cloudflare_workers_ai(self) -> dict[str, Any]:
        return {}

    def fetch_mistral_aliases(self, api_key: str | None = None) -> dict[str, str]:
        return {}

    def fetch_ramp_router(self, api_key: str | None = None) -> dict[str, Any]:
        self.ramp_router_keys.append(api_key)
        return self._router


def _priced_sdk(
    router_table: dict[str, Any] | None = None, on_error: Any = None
) -> tuple[LagoSDK, list[dict], PricingProvider]:
    provider = PricingProvider(fetcher=_StubFetcher(router_table), ttl_seconds=3600.0)
    config: dict[str, Any] = {"pricing_mode": "price", "pricing_provider": provider}
    if on_error is not None:
        config["on_error"] = on_error
    sdk, received = _new_sdk(**config)
    # Both tables have to be warm before the call, or a miss under test is just a cold
    # cache. `maybe_refresh` is the queue worker's own warm-up, called synchronously.
    provider.prime(["ramp_router"])
    provider.maybe_refresh()
    return sdk, received, provider


def _tiered(model: str, tier: str | None = "default", usage: dict[str, Any] | None = None) -> dict[str, Any]:
    """A Router response carrying its top-level `service_tier`, as every captured one does."""
    body = router_response(model, usage)
    if tier is not None:
        body["service_tier"] = tier
    return body


def _cost_by_type(received: list[dict]) -> dict[str, dict]:
    return {e["properties"]["token_type"]: e for e in received if e["code"] == "llm_cost"}


def test_the_same_model_priced_directly_comes_from_openrouter_the_control() -> None:
    """If this fails, every Router assertion below proves nothing about which table won."""
    sdk, received, provider = _priced_sdk()
    assert provider.lookup("openai", PRICED_MODEL, "responses") is not None
    client = sdk.wrap(FakeRouterClient("https://api.openai.com/v1", lambda kw: router_response(PRICED_MODEL)))
    client.responses.create(model=PRICED_MODEL, input="ping")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    costs = _cost_by_type(received)
    assert costs and all(e["properties"]["price_source"] == "openrouter" for e in costs.values())
    assert costs["input"]["properties"]["unit_price"] == "0.000001"
    assert "llm_input_tokens" not in [e["code"] for e in received]


def test_the_identical_model_through_router_prices_from_routers_own_catalog() -> None:
    """Same usage, same model family — only the base URL differs — and the money comes
    from Router's table: $0.20/M input, not OpenRouter's $1/M."""
    sdk, received, _ = _priced_sdk()
    client = sdk.wrap(FakeRouterClient(ROUTER_BASE_URL, lambda kw: _tiered(SERVED_MODEL)))
    client.responses.create(model=PRICED_MODEL, input="ping")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    costs = _cost_by_type(received)
    assert set(costs) == {"input", "output"}
    for e in costs.values():
        assert e["properties"]["price_source"] == "ramp_router"
        assert e["properties"]["provider"] == RAMP_ROUTER_PROVIDER
        # Billed under the served snapshot, the same row a direct call to it reports.
        assert e["properties"]["model"] == SERVED_MODEL
    assert costs["input"]["properties"]["unit_price"] == "0.0000002"
    assert costs["input"]["properties"]["value"] == "0.0000022"  # 11 tokens
    assert costs["output"]["properties"]["value"] == "0.00000375"  # 3 tokens x $1.25/M
    assert "llm_input_tokens" not in [e["code"] for e in received]


@pytest.mark.parametrize("tier", ["flex", "priority", "turbo"])
def test_a_non_default_tier_is_a_named_miss_never_a_multiplied_rate(tier: str) -> None:
    """flex measured 0.5x, priority 2.0x, and a tier Router adds later is unknown. None
    of them bill at the catalog rate, and the SDK applies no factor of its own: token
    events, plus an on_error that says WHICH tier, since the same model priced fine a
    moment ago. Decided 2026-09-07."""
    errors: list[tuple[Exception, str]] = []
    sdk, received, _ = _priced_sdk(on_error=lambda exc, where: errors.append((exc, where)))
    client = sdk.wrap(FakeRouterClient(ROUTER_BASE_URL, lambda kw: _tiered(SERVED_MODEL, tier)))
    client.responses.create(model="x", input="ping")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    by_code = _by_code(received)
    assert "llm_cost" not in by_code
    # Not a silent drop. The usage is billed, exactly, as tokens.
    assert by_code["llm_input_tokens"] == 11
    assert by_code["llm_output_tokens"] == 3
    misses = [(exc, where) for exc, where in errors if isinstance(exc, PricingUnavailableError)]
    assert len(misses) == 1
    exc, where = misses[0]
    assert where == "pricing"
    assert exc.detail is not None and tier in exc.detail
    assert tier in str(exc)


def test_a_router_response_reporting_no_tier_bills_at_the_base_rate() -> None:
    """Sweep 2026-09-07: Router omitted `service_tier` on six `incomplete` zero-output
    responses (both surfaces) and billed every one at standard, while flex and priority
    were always reported explicitly. Absence means standard; only a reported non-base tier
    is a miss."""
    errors: list[Exception] = []
    sdk, received, _ = _priced_sdk(on_error=lambda exc, where: errors.append(exc))
    client = sdk.wrap(FakeRouterClient(ROUTER_BASE_URL, lambda kw: _tiered(SERVED_MODEL, tier=None)))
    client.responses.create(model="x", input="ping")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert "llm_cost" in [e["code"] for e in received]
    assert not any(isinstance(e, PricingUnavailableError) for e in errors)


_LUNA_COLD_WRITE = {
    "input_tokens": 4493,
    "output_tokens": 5,
    "total_tokens": 4498,
    "input_tokens_details": {"cache_write_tokens": 4490, "cached_tokens": 0},
}
_LUNA_WARM_READ = {
    "input_tokens": 4493,
    "output_tokens": 5,
    "total_tokens": 4498,
    "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 4490},
}


def test_an_openai_served_cache_write_bills_at_the_catalog_write_rate() -> None:
    """Reconciled against the dashboard on 2026-09-07 (gpt-5.6-luna, default tier): at the
    published rates 3 x $0.20/M + 4490 x $0.25/M + 5 x $1.20/M = $0.0011291; Router charged
    exactly 1.1x that, a documented per-model mismatch the SDK does not correct. What this
    pins is the write arithmetic: the count sits INSIDE input_tokens, so it is moved out
    before pricing — never billed at the input rate AND the write rate."""
    sdk, received, _ = _priced_sdk()
    client = sdk.wrap(
        FakeRouterClient(ROUTER_BASE_URL, lambda kw: _tiered("gpt-5.6-luna", usage=_LUNA_COLD_WRITE))
    )
    client.responses.create(model="x", input="ping")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    costs = _cost_by_type(received)
    assert set(costs) == {"input", "cache_write", "output"}
    assert costs["input"]["properties"]["unit"] == "3"
    assert costs["cache_write"]["properties"]["unit"] == "4490"
    assert costs["cache_write"]["properties"]["unit_price"] == "0.00000025"
    assert sum(Decimal(e["properties"]["value"]) for e in costs.values()) == Decimal("0.0011291")


def test_the_warm_repeat_bills_the_cached_block_at_the_cache_read_rate() -> None:
    """Same prompt a second later: 4490 cached at $0.02/M: $0.0000964 at the published rates
    (Router charged 1.1x that — the documented luna mismatch)."""
    sdk, received, _ = _priced_sdk()
    client = sdk.wrap(
        FakeRouterClient(ROUTER_BASE_URL, lambda kw: _tiered("gpt-5.6-luna", usage=_LUNA_WARM_READ))
    )
    client.responses.create(model="x", input="ping")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    costs = _cost_by_type(received)
    assert set(costs) == {"input", "cache_read", "output"}
    assert costs["cache_read"]["properties"]["unit"] == "4490"
    assert sum(Decimal(e["properties"]["value"]) for e in costs.values()) == Decimal("0.0000964")


def test_the_cache_write_count_is_mapped_for_router_and_stays_in_extras_for_openai() -> None:
    """Same wire shape, two measured billing conventions: Router bills the write at its
    catalog rate (mapped, not drift); OpenAI-native was metered at the plain input rate
    (unmapped, surfaced in extras — see _MAPPED_DETAIL_FIELDS)."""
    body = _tiered("gpt-5.6-luna", usage=_LUNA_COLD_WRITE)
    via_router = extract_openai_native(body, provider_hint=RAMP_ROUTER_PROVIDER)
    assert via_router.cache_write == 4490
    assert "input_tokens_details.cache_write_tokens" not in via_router.extras
    direct = extract_openai_native(body)
    assert direct.cache_write == 0
    assert direct.extras["input_tokens_details.cache_write_tokens"] == 4490


def test_a_streamed_router_call_carries_the_served_tier_and_prices() -> None:
    """The terminal `response.completed` event carries `service_tier` (fixture 04). The
    stream wrapper used to forward usage and model only, so every streamed Router call
    reached price mode tier-less — and a missing tier is a miss."""
    events = [
        _FakeStreamChunk(
            {"type": "response.created", "response": {"model": SERVED_MODEL, "service_tier": "default"}}
        ),
        _FakeStreamChunk({"type": "response.output_text.delta", "delta": "po"}),
        _FakeStreamChunk({"type": "response.completed", "response": _tiered(SERVED_MODEL)}),
    ]
    sdk, received, _ = _priced_sdk()
    client = sdk.wrap(FakeRouterClient(ROUTER_BASE_URL, lambda kw: iter(events)))
    for _ in client.responses.create(model="x", input="ping", stream=True):
        pass
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    costs = _cost_by_type(received)
    assert set(costs) == {"input", "output"}
    assert costs["input"]["properties"]["price_source"] == "ramp_router"


def test_a_pricing_miss_never_reaches_the_caller_as_an_exception() -> None:
    sdk, _, _ = _priced_sdk()
    client = sdk.wrap(FakeRouterClient(ROUTER_BASE_URL, lambda kw: _tiered(SERVED_MODEL, "flex")))
    assert client.responses.create(model="x", input="ping") is not None
    sdk.shutdown(timeout=1.0)


def test_a_cold_or_empty_router_table_is_a_reported_miss_not_a_silent_token_fallback() -> None:
    """Router used to sit in TOKEN_BILLED_PROVIDERS, which swallowed the miss on purpose
    because nothing could fix it. Now a miss is actionable — no Router key learned, table
    still cold, catalog missing the model — so it must reach on_error like any other."""
    assert RAMP_ROUTER_PROVIDER not in TOKEN_BILLED_PROVIDERS

    errors: list[Exception] = []
    sdk, received, _ = _priced_sdk(router_table={}, on_error=lambda exc, where: errors.append(exc))
    client = sdk.wrap(FakeRouterClient(ROUTER_BASE_URL, lambda kw: _tiered(SERVED_MODEL)))
    client.responses.create(model="x", input="ping")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert sorted(e["code"] for e in received) == ["llm_input_tokens", "llm_output_tokens"]
    misses = [e for e in errors if isinstance(e, PricingUnavailableError)]
    assert len(misses) == 1 and misses[0].detail is None


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
# The recorded token-convention decision behind "ramp_router", pinned so it
# cannot be reverted silently. The generic roster tests cannot see it: the hint
# comes from the wrapper's HOST arm, not from _PROVIDER_BY_BASE_URL_PATH.
# ----------------------------------------------------------------------


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


# ----------------------------------------------------------------------
# The captured responses, run through the adapter. The tests above pin the
# SDK's decisions against a hand-built shape; these pin them against what
# Router actually sent. Skips cleanly when the captures are absent, so a
# missing capture reads as "not covered" rather than as a pass.
# ----------------------------------------------------------------------
_CAPTURES = pathlib.Path(__file__).parents[1] / "adapters" / "fixtures" / "ramp_router"


def _captured_bodies(surface: str = "responses") -> list[tuple[str, dict[str, Any]]]:
    """Every captured 200 that carries usage, buffered or streamed, on ONE surface.

    Router has two: `/v1/responses` (OpenAI-shaped, fixtures 01-10) and `/v1/messages`
    (Anthropic-shaped, fixtures 11-15, named `_messages_`). They report `service_tier` in
    different places and go through different adapters, so a test must say which it means.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(_CAPTURES.glob("*.json")):
        if ("_messages_" in path.name) != (surface == "messages"):
            continue
        blob = json.loads(path.read_text())
        body = blob.get("_body")
        if not (isinstance(body, dict) and body.get("usage")):
            # The streamed capture keeps its payload under `.response` per event; the
            # terminal one is the only one carrying usage, and is what the wrapper bills.
            body = None
            for event in blob.get("_events") or []:
                candidate = (event or {}).get("response")
                if isinstance(candidate, dict) and candidate.get("usage"):
                    body = candidate
        if isinstance(body, dict) and body.get("usage"):
            out.append((path.name, body))
    return out


@pytest.mark.skipif(not _captured_bodies(), reason="Router fixtures not captured")
@pytest.mark.parametrize("name,body", _captured_bodies(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_captured_response_reports_its_served_tier(name: str, body: dict[str, Any]) -> None:
    """The regression this file previously had no way to catch. The tier was read only
    from a `provider:model:tier` candidate suffix, a shape Router resolves away before
    answering, so `service_tier` was dropped on 100% of live traffic while the hand-built
    tests stayed green."""
    u = extract_openai_native(body, provider_hint=RAMP_ROUTER_PROVIDER)
    assert u.extras["service_tier"] == body["service_tier"]


@pytest.mark.skipif(not _captured_bodies(), reason="Router fixtures not captured")
@pytest.mark.parametrize("name,body", _captured_bodies(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_captured_response_bills_the_bare_served_snapshot(name: str, body: dict[str, Any]) -> None:
    """Router answers with a resolved vendor snapshot, never a compound candidate — the
    reason the suffix parse is a fallback rather than the live path. Billing the model
    verbatim is what rolls a Router-served call up against the same Lago row a direct
    call to that model reports."""
    u = extract_openai_native(body, provider_hint=RAMP_ROUTER_PROVIDER)
    assert u.model == body["model"]
    assert ":" not in u.model
    assert u.provider == RAMP_ROUTER_PROVIDER


@pytest.mark.skipif(not _captured_bodies(), reason="Router fixtures not captured")
@pytest.mark.parametrize("name,body", _captured_bodies(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_captured_served_model_resolves_in_the_real_catalog(name: str, body: dict[str, Any]) -> None:
    """The served name is what price mode looks up, and it is never the catalog id: a
    dated snapshot for OpenAI and Anthropic, the vendor's own path for Fireworks. Every
    response Router has actually sent must land on exactly one catalog entry."""
    u = extract_openai_native(body, provider_hint=RAMP_ROUTER_PROVIDER)
    assert lookup_ramp_router(_ROUTER_TABLE, u.model) is not None


# ----------------------------------------------------------------------
# Router's SECOND surface: `POST /v1/messages`, reached with an Anthropic client.
# Same host, same catalog, same key — but Anthropic's schema and Anthropic's
# ADDITIVE convention for every vendor, and the one place an Anthropic-served
# cache WRITE is reported. Detection is the shared host helper; the wrapper
# threads a provider hint into the Anthropic adapter, which stamps a distinct
# `api` so the token semantics cannot be confused with the Responses surface.
# ----------------------------------------------------------------------
def messages_response(model: str, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    """A Router `/v1/messages` response, in the shape fixture 11 actually carries:
    Anthropic's schema, `service_tier` INSIDE usage."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "pong"}],
        "usage": {
            "input_tokens": 16,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "service_tier": "standard",
            **(usage or {}),
        },
    }


class FakeRouterAnthropicMessages:
    def __init__(self, reply: Any) -> None:
        self._reply = reply

    def create(self, **kwargs: Any) -> Any:
        assert "extra_lago" not in kwargs
        return self._reply(kwargs)


class FakeRouterAnthropicClient:
    def __init__(self, base_url: str, reply: Any, api_key: str = "sk-router-from-client") -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.messages = FakeRouterAnthropicMessages(reply)


# The detector keys on the module; Router's Messages surface is reached with an Anthropic client.
FakeRouterAnthropicClient.__module__ = "anthropic.fake"

ANTHROPIC_ROUTER_BASE_URL = "https://api.router.com"
HAIKU_SERVED = "claude-haiku-4-5-20251001"  # what Router answers with (fixture 11)


@pytest.mark.parametrize(
    "base_url,expected",
    [
        ("https://api.router.com", True),
        ("https://api.router.com/v1", True),
        ("https://api-eu.router.com", True),
        ("https://api.anthropic.com", False),
        ("https://evil.example.com/api.router.com", False),
        ("https://evilrouter.com", False),
        ("/v1", False),
        (None, False),
        (42, False),
    ],
)
def test_the_shared_host_helper_is_the_one_answer_both_wrappers_read(base_url: Any, expected: bool) -> None:
    assert is_ramp_router_base_url(base_url) is expected
    assert client_points_at_ramp_router(_Base(base_url)) is expected


def test_the_openai_wrapper_and_the_shared_helper_cannot_disagree() -> None:
    for url in ("https://api.router.com/v1", "https://api-eu.router.com/v1", "https://api.openai.com/v1"):
        assert (_provider_hint_for(_Base(url)) == RAMP_ROUTER_PROVIDER) is is_ramp_router_base_url(url)


def test_an_anthropic_client_pointed_at_router_bills_as_router_on_the_messages_surface() -> None:
    sdk, received = _new_sdk()
    client = sdk.wrap(
        FakeRouterAnthropicClient(ANTHROPIC_ROUTER_BASE_URL, lambda kw: messages_response(HAIKU_SERVED))
    )
    client.messages.create(
        model="claude-haiku-4-5", max_tokens=16, messages=[{"role": "user", "content": "ping"}]
    )
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    assert _by_code(received) == {"llm_input_tokens": 16, "llm_output_tokens": 5}
    for e in received:
        assert e["properties"]["provider"] == RAMP_ROUTER_PROVIDER
        assert e["properties"]["api"] == RAMP_ROUTER_MESSAGES_API
        assert e["properties"]["model"] == HAIKU_SERVED


def test_an_anthropic_client_pointed_at_anthropic_is_untouched() -> None:
    sdk, received = _new_sdk()
    client = sdk.wrap(
        FakeRouterAnthropicClient("https://api.anthropic.com", lambda kw: messages_response(HAIKU_SERVED))
    )
    client.messages.create(model="claude-haiku-4-5", max_tokens=16, messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert {(e["properties"]["provider"], e["properties"]["api"]) for e in received} == {
        ("anthropic", "native")
    }


def test_the_messages_surface_keeps_anthropics_additive_convention_for_every_vendor() -> None:
    """Measured: haiku `input_tokens: 16` beside `cache_read_input_tokens: 20113` (2026-09-04,
    reconciled exactly); an xAI model `input_tokens: 65` beside `cache_read_input_tokens:
    128` with thinking inside output (2026-09-07). The Responses stamp is in
    OPENAI_SHAPED_APIS; this one must never be."""
    assert RAMP_ROUTER_MESSAGES_API not in OPENAI_SHAPED_APIS
    assert token_semantics(RAMP_ROUTER_PROVIDER, RAMP_ROUTER_MESSAGES_API) == (False, False, False)
    assert token_semantics(RAMP_ROUTER_PROVIDER, RAMP_ROUTER_PROVIDER) == (True, True, True)


def test_the_adapter_stamps_router_only_when_the_wrapper_says_so() -> None:
    body = messages_response(HAIKU_SERVED)
    assert (extract_anthropic_native(body).provider, extract_anthropic_native(body).api) == (
        "anthropic",
        "native",
    )
    hinted = extract_anthropic_native(body, provider_hint=RAMP_ROUTER_PROVIDER)
    assert (hinted.provider, hinted.api) == (RAMP_ROUTER_PROVIDER, RAMP_ROUTER_MESSAGES_API)
    # The tier rides inside usage on this surface and lands in extras with no special code.
    assert hinted.extras["service_tier"] == "standard"


# Fixture 12 / 13, verbatim: a 7,481-token cache_control prefix on claude-haiku-4-5.
_MESSAGES_COLD_WRITE = {
    "input_tokens": 15,
    "output_tokens": 5,
    "cache_creation_input_tokens": 7481,
    "cache_creation": {"ephemeral_5m_input_tokens": 7481, "ephemeral_1h_input_tokens": 0},
}
_MESSAGES_WARM_READ = {"input_tokens": 15, "output_tokens": 6, "cache_read_input_tokens": 7481}


def test_an_anthropic_cache_write_on_the_messages_surface_bills_at_the_ttl_write_rate() -> None:
    """The gap the Responses surface cannot close. Router publishes cache_write_input_5m
    ($1.25/M on haiku) and the Messages surface reports the written count with its TTL, so
    the write bills at its own rate: 15 x $1/M + 7481 x $1.25/M + 5 x $5/M = $0.00939125.
    The lump `cache_write` line is consumed entirely by the split — never billed twice."""
    sdk, received, _ = _priced_sdk()
    client = sdk.wrap(
        FakeRouterAnthropicClient(
            ANTHROPIC_ROUTER_BASE_URL, lambda kw: messages_response(HAIKU_SERVED, _MESSAGES_COLD_WRITE)
        )
    )
    client.messages.create(model="x", max_tokens=16, messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    costs = _cost_by_type(received)
    assert set(costs) == {"input", "cache_write_5m", "output"}
    assert costs["input"]["properties"]["unit"] == "15"  # additive: NOT reduced by the write
    assert costs["cache_write_5m"]["properties"]["unit"] == "7481"
    assert costs["cache_write_5m"]["properties"]["unit_price"] == "0.00000125"
    assert all(e["properties"]["api"] == RAMP_ROUTER_MESSAGES_API for e in costs.values())
    assert sum(Decimal(e["properties"]["value"]) for e in costs.values()) == Decimal("0.00939125")


def test_the_warm_repeat_on_the_messages_surface_bills_the_read_beside_input() -> None:
    """Additive: 15 input tokens stay 15; the 7,481 cached bill at $0.10/M. $0.0007931."""
    sdk, received, _ = _priced_sdk()
    client = sdk.wrap(
        FakeRouterAnthropicClient(
            ANTHROPIC_ROUTER_BASE_URL, lambda kw: messages_response(HAIKU_SERVED, _MESSAGES_WARM_READ)
        )
    )
    client.messages.create(model="x", max_tokens=16, messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    costs = _cost_by_type(received)
    assert set(costs) == {"input", "cache_read", "output"}
    assert costs["input"]["properties"]["unit"] == "15"
    assert costs["cache_read"]["properties"]["unit"] == "7481"
    assert sum(Decimal(e["properties"]["value"]) for e in costs.values()) == Decimal("0.0007931")


@pytest.mark.parametrize("tier,priced", [("standard", True), ("default", True), ("priority", False)])
def test_the_tier_gate_reads_the_messages_surfaces_in_usage_tier(tier: str, priced: bool) -> None:
    errors: list[Exception] = []
    sdk, received, _ = _priced_sdk(on_error=lambda exc, where: errors.append(exc))
    client = sdk.wrap(
        FakeRouterAnthropicClient(
            ANTHROPIC_ROUTER_BASE_URL, lambda kw: messages_response(HAIKU_SERVED, {"service_tier": tier})
        )
    )
    client.messages.create(model="x", max_tokens=16, messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert ("llm_cost" in [e["code"] for e in received]) is priced
    assert any(isinstance(e, PricingUnavailableError) and tier in str(e) for e in errors) is not priced


def test_a_streamed_messages_call_carries_the_tier_through_the_merge_and_prices() -> None:
    """Fixture 14: `service_tier` sits inside `message_start.message.usage` AND
    `message_delta.usage`; the wrapper's merge keeps it, so the adapter's drift sweep
    lands it in extras and the tier gate passes."""
    start = {
        "type": "message_start",
        "message": {
            "model": HAIKU_SERVED,
            "usage": {
                "input_tokens": 16,
                "output_tokens": 4,
                "cache_read_input_tokens": 0,
                "service_tier": "standard",
            },
        },
    }
    delta = {"type": "message_delta", "usage": {"output_tokens": 5, "service_tier": "standard"}}
    events = [
        _FakeStreamChunk(start),
        _FakeStreamChunk({"type": "content_block_delta"}),
        _FakeStreamChunk(delta),
    ]
    sdk, received, _ = _priced_sdk()
    client = sdk.wrap(FakeRouterAnthropicClient(ANTHROPIC_ROUTER_BASE_URL, lambda kw: iter(events)))
    for _ in client.messages.create(model="x", max_tokens=16, messages=[], stream=True):
        pass
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)

    costs = _cost_by_type(received)
    assert set(costs) == {"input", "output"}
    assert costs["input"]["properties"]["unit"] == "16"
    assert costs["output"]["properties"]["unit"] == "5"
    assert costs["input"]["properties"]["api"] == RAMP_ROUTER_MESSAGES_API


@pytest.mark.skipif(not _captured_bodies("messages"), reason="Router /v1/messages fixtures not captured")
@pytest.mark.parametrize(
    "name,body", _captured_bodies("messages"), ids=lambda v: v if isinstance(v, str) else ""
)
def test_every_captured_messages_response_stamps_router_and_resolves_in_the_catalog(
    name: str, body: dict[str, Any]
) -> None:
    u = extract_anthropic_native(body, provider_hint=RAMP_ROUTER_PROVIDER)
    assert (u.provider, u.api) == (RAMP_ROUTER_PROVIDER, RAMP_ROUTER_MESSAGES_API)
    assert u.model == body["model"]
    # The tier is INSIDE usage on this surface — the OpenAI-served model included.
    assert u.extras["service_tier"] == body["usage"]["service_tier"]
    assert lookup_ramp_router(_ROUTER_TABLE, u.model) is not None
    # Additive: the lump write equals the TTL split (Anthropic's contract, every capture).
    assert u.cache_write == u.cache_write_5m + u.cache_write_1h


@pytest.mark.skipif(
    not (_CAPTURES / "14_real_messages_streamed.json").exists(), reason="fixture not captured"
)
def test_the_captured_messages_stream_merges_to_a_priced_tier() -> None:
    blob = json.loads((_CAPTURES / "14_real_messages_streamed.json").read_text())
    accumulated: dict[str, Any] = {}
    model: str | None = None
    for event in blob["_events"]:
        model = _merge_stream_usage(accumulated, event) or model
    u = extract_anthropic_native({"usage": accumulated, "model": model}, provider_hint=RAMP_ROUTER_PROVIDER)
    assert u.model == HAIKU_SERVED
    assert u.input == 16 and u.output == 5
    assert u.extras["service_tier"] == "standard"
    assert lookup_ramp_router(_ROUTER_TABLE, u.model) is not None

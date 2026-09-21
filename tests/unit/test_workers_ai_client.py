"""WorkersAI client tests — mocked HTTP, no live API."""

from __future__ import annotations

import json
from typing import Any

import pytest
import responses

from lago_agent_sdk import LagoSDK, WorkersAIError

ACCT, GW = "acct_test", "gw_test"
LLAMA = "@cf/meta/llama-3.2-3b-instruct"
# `@cf/...` ids take the gateway host's path route (cache + log headers, model logged);
# partner ids take the unified `/ai/run` path with `cf-aig-gateway-id`, the only route
# where the gateway's BYOK key is consulted. See the workers_ai module docstring.
DIRECT_RUN = f"https://api.cloudflare.com/client/v4/accounts/{ACCT}/ai/run"
DIRECT_LLAMA = f"{DIRECT_RUN}/{LLAMA}"
GATEWAY_BASE = f"https://gateway.ai.cloudflare.com/v1/{ACCT}/{GW}/workers-ai"
GATEWAY_LLAMA = f"{GATEWAY_BASE}/{LLAMA}"

CHAT_BODY = {
    "result": {
        "response": "Hello there!",
        "model": "@cf/meta/llama-3.2-3b-instruct-v2",
        "usage": {"prompt_tokens": 41, "completion_tokens": 34, "total_tokens": 75, "neurons": 1.22},
    },
    "success": True,
    "errors": [],
    "messages": [],
}
JEV_INPUT = {"state": "charged twice", "questions": {"is_urgent": {"type": "noul", "instructions": "?"}}}
JEV_BODY = {  # the partner-model envelope: the model's object one level down (fixture 03)
    "result": {
        "state": "Completed",
        "result": {
            "model": "jev-1.13.0",
            "answers": {"is_urgent": {"type": "noul", "noul": 0.97}},
            "usage": {"input_tokens": 446, "output_tokens": 73},
        },
        "gatewayMetadata": {"keySource": "BYOK"},
    },
    "success": True,
    "errors": [],
    "messages": [],
}
JEV_402 = {
    "errors": [{"message": "Insufficient balance; add money to your gateway or use BYOK", "code": 2021}],
    "success": False,
    "result": {},
    "messages": [],
}


def _new_sdk(default_sub: str | None = "sub_test") -> tuple[LagoSDK, list[dict], list[str]]:
    received: list[dict] = []
    errors: list[str] = []

    def sender(batch: list[dict]) -> None:
        received.extend(batch)

    sdk = LagoSDK(api_key="dummy", default_subscription_id=default_sub)
    sdk._queue._sender = sender  # type: ignore[attr-defined]
    sdk.config.on_error = lambda exc, where: errors.append(f"{where}: {exc}")
    return sdk, received, errors


def _by_code(received: list[dict]) -> dict[str, int]:
    return {e["code"]: int(e["properties"]["value"]) for e in received}


# --------------------------------------------------------------------------
# Catalog models (`@cf/...`) — gateway host path route
# --------------------------------------------------------------------------
@responses.activate
def test_run_bills_tokens_with_model_provider_api_and_log_id() -> None:
    sdk, received, errors = _new_sdk()
    responses.post(
        GATEWAY_LLAMA, json=CHAT_BODY, headers={"cf-aig-cache-status": "MISS", "cf-aig-log-id": "01LOG"}
    )
    ai = sdk.workers_ai(ACCT, "tok", gateway_id=GW, gateway_auth="gwtok")
    out = ai.run(LLAMA, {"messages": [{"role": "user", "content": "hi"}]})
    assert out["result"]["response"] == "Hello there!"  # envelope returned unchanged
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert errors == []
    assert _by_code(received) == {"llm_input_tokens": 41, "llm_output_tokens": 34}
    props = received[0]["properties"]
    assert props["model"] == LLAMA  # requested id = catalog key; served "-v2" name stays in extras
    assert props["provider"] == "workers-ai"
    assert props["api"] == "workers_ai_run"
    assert props["cf_log_id"] == "01LOG"
    assert received[0]["external_subscription_id"] == "sub_test"


@responses.activate
def test_catalog_model_takes_the_gateway_host_with_gateway_auth() -> None:
    sdk, _, _ = _new_sdk("sub_acme")
    responses.post(GATEWAY_LLAMA, json=CHAT_BODY)
    ai = sdk.workers_ai(ACCT, "tok", gateway_id=GW, gateway_auth="gwtok")
    assert ai.url_for(LLAMA) == GATEWAY_LLAMA
    ai.run(LLAMA, {"messages": []}, extra_headers={"cf-aig-cache-ttl": "300"})
    sdk.shutdown(timeout=1.0)
    req = responses.calls[0].request
    assert json.loads(req.body) == {"messages": []}  # path route: the body IS the input
    assert req.headers["Authorization"] == "Bearer tok"
    assert req.headers["cf-aig-authorization"] == "Bearer gwtok"
    assert req.headers["cf-aig-cache-ttl"] == "300"
    assert "cf-aig-gateway-id" not in req.headers and "cf-aig-skip-cache" not in req.headers
    # The resolved subscription rides along, so the Logs API backfill attributes the same way.
    assert json.loads(req.headers["cf-aig-metadata"]) == {"lago_subscription": "sub_acme"}


@responses.activate
def test_direct_route_when_no_gateway_sets_no_gateway_headers() -> None:
    sdk, received, _ = _new_sdk()
    responses.post(DIRECT_LLAMA, json=CHAT_BODY)
    ai = sdk.workers_ai(ACCT, "tok")
    assert ai.url_for(LLAMA) == DIRECT_LLAMA
    assert ai.url_for("typesafe/jev") == DIRECT_RUN
    ai.run(LLAMA, {"messages": []})
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    req = responses.calls[0].request
    assert json.loads(req.body) == {"messages": []}
    for h in ("cf-aig-authorization", "cf-aig-gateway-id", "cf-aig-skip-cache", "cf-aig-metadata"):
        assert h not in req.headers  # nothing stores or reads these without a gateway
    assert _by_code(received) == {"llm_input_tokens": 41, "llm_output_tokens": 34}
    assert "cf_log_id" not in received[0]["properties"]


@responses.activate
def test_gateway_cache_hit_is_not_billed() -> None:
    """The gateway replays the identical body, usage included (fixtures 06/07). Only the
    header says the model never ran — so only the header can stop the bill."""
    sdk, received, errors = _new_sdk()
    responses.post(
        GATEWAY_LLAMA, json=CHAT_BODY, headers={"cf-aig-cache-status": "HIT", "cf-aig-log-id": "01HIT"}
    )
    ai = sdk.workers_ai(ACCT, "tok", gateway_id=GW, gateway_auth="gwtok")
    out = ai.run(LLAMA, {"messages": []})
    assert out["result"]["usage"]["prompt_tokens"] == 41  # the caller still gets the body
    sdk.shutdown(timeout=1.0)
    assert received == []
    assert errors == []


# --------------------------------------------------------------------------
# Partner models (`typesafe/jev`) — unified path through the BYOK gateway
# --------------------------------------------------------------------------
@responses.activate
def test_partner_model_takes_the_unified_path_through_the_byok_gateway() -> None:
    """Model in the body, `cf-aig-gateway-id` names the gateway whose BYOK holds the partner
    key, cache skipped because this path returns no cache header, and the nested
    `result.result.usage` bills."""
    sdk, received, errors = _new_sdk("sub_acme")
    responses.post(DIRECT_RUN, json=JEV_BODY)
    ai = sdk.workers_ai(ACCT, "tok", gateway_id=GW, gateway_auth="gwtok")
    assert ai.url_for("typesafe/jev") == DIRECT_RUN
    out = ai.run("typesafe/jev", JEV_INPUT)
    assert out["result"]["result"]["answers"]["is_urgent"]["noul"] == 0.97
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    req = responses.calls[0].request
    assert json.loads(req.body) == {"model": "typesafe/jev", "input": JEV_INPUT}
    assert req.headers["Authorization"] == "Bearer tok"
    assert req.headers["cf-aig-gateway-id"] == GW
    assert req.headers["cf-aig-skip-cache"] == "true"
    assert "cf-aig-authorization" not in req.headers  # gateway auth belongs to the gateway host only
    assert json.loads(req.headers["cf-aig-metadata"]) == {"lago_subscription": "sub_acme"}
    assert errors == []
    assert _by_code(received) == {"llm_input_tokens": 446, "llm_output_tokens": 73}
    assert received[0]["properties"]["model"] == "typesafe/jev"
    assert received[0]["properties"]["provider"] == "workers-ai"
    assert "cf_log_id" not in received[0]["properties"]  # the unified path returns no log id


@responses.activate
def test_extra_headers_win_over_the_client_defaults() -> None:
    sdk, _, _ = _new_sdk()
    responses.post(DIRECT_RUN, json=JEV_BODY)
    ai = sdk.workers_ai(ACCT, "tok", gateway_id=GW)
    ai.run("typesafe/jev", JEV_INPUT, extra_headers={"cf-aig-skip-cache": "false"})
    sdk.shutdown(timeout=1.0)
    assert responses.calls[0].request.headers["cf-aig-skip-cache"] == "false"


@responses.activate
def test_402_raises_workers_ai_error_and_bills_nothing() -> None:
    sdk, received, errors = _new_sdk()
    responses.post(DIRECT_RUN, json=JEV_402, status=402)
    ai = sdk.workers_ai(ACCT, "tok", gateway_id=GW, gateway_auth="gwtok")
    with pytest.raises(WorkersAIError) as ei:
        ai.run("typesafe/jev", JEV_INPUT)
    assert ei.value.status_code == 402
    assert ei.value.errors[0]["code"] == 2021
    assert "Insufficient balance" in str(ei.value)
    sdk.shutdown(timeout=1.0)
    assert received == []
    assert errors == []  # the customer's error, not an instrumentation failure


@responses.activate
def test_success_false_with_200_is_still_an_error() -> None:
    sdk, received, _ = _new_sdk()
    responses.post(DIRECT_RUN, json={**JEV_402}, status=200)
    ai = sdk.workers_ai(ACCT, "tok")
    with pytest.raises(WorkersAIError):
        ai.run("typesafe/jev", JEV_INPUT)
    sdk.shutdown(timeout=1.0)
    assert received == []


# --------------------------------------------------------------------------
# Options, attribution, failure isolation
# --------------------------------------------------------------------------
@responses.activate
def test_per_call_extra_lago_overrides_subscription_and_adds_dimensions() -> None:
    sdk, received, _ = _new_sdk("sub_default")
    responses.post(GATEWAY_LLAMA, json=CHAT_BODY)
    ai = sdk.workers_ai(ACCT, "tok", gateway_id=GW, gateway_auth="gwtok", dimensions={"team": "billing"})
    ai.run(
        LLAMA,
        {"messages": []},
        extra_lago={"subscription": "sub_override", "dimensions": {"ticket": "T-1"}},
    )
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert all(e["external_subscription_id"] == "sub_override" for e in received)
    assert received[0]["properties"]["team"] == "billing"
    assert received[0]["properties"]["ticket"] == "T-1"
    req = responses.calls[0].request
    assert json.loads(req.headers["cf-aig-metadata"]) == {"lago_subscription": "sub_override"}


@responses.activate
def test_no_resolvable_subscription_drops_with_on_error() -> None:
    sdk, received, errors = _new_sdk(default_sub=None)
    responses.post(DIRECT_LLAMA, json=CHAT_BODY)
    ai = sdk.workers_ai(ACCT, "tok")
    ai.run(LLAMA, {"messages": []})
    sdk.shutdown(timeout=1.0)
    assert received == []
    assert any("no resolvable subscription" in e for e in errors)


def test_stream_is_refused_before_any_request() -> None:
    sdk, _, _ = _new_sdk()
    ai = sdk.workers_ai(ACCT, "tok")
    with pytest.raises(ValueError, match="stream=True"):
        ai.run(LLAMA, {"messages": [], "stream": True})
    sdk.shutdown(timeout=1.0)


@responses.activate
def test_instrumentation_failure_does_not_break_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    import lago_agent_sdk.workers_ai as mod

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("adapter bug")

    monkeypatch.setattr(mod, "extract_workers_ai_native", boom)
    sdk, received, errors = _new_sdk()
    responses.post(DIRECT_LLAMA, json=CHAT_BODY)
    ai = sdk.workers_ai(ACCT, "tok")
    out = ai.run(LLAMA, {"messages": []})
    assert out["result"]["response"] == "Hello there!"
    sdk.shutdown(timeout=1.0)
    assert received == []
    assert errors and "adapter bug" in errors[0]


@responses.activate
def test_non_json_error_body_still_raises_cleanly() -> None:
    sdk, received, _ = _new_sdk()
    responses.post(DIRECT_LLAMA, body="<html>bad gateway</html>", status=502)
    ai = sdk.workers_ai(ACCT, "tok")
    with pytest.raises(WorkersAIError) as ei:
        ai.run(LLAMA, {"messages": []})
    assert ei.value.status_code == 502
    assert ei.value.errors == []
    sdk.shutdown(timeout=1.0)
    assert received == []

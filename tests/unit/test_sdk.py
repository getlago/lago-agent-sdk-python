"""LagoSDK tests — emit, subscription resolution, error policy."""

from __future__ import annotations

import pytest

from lago_agent_sdk import CanonicalUsage, LagoSDK
from lago_agent_sdk.exceptions import UnknownClientError


def _new_sdk(default_sub: str | None = None, sender=None) -> tuple[LagoSDK, list]:
    received: list = []
    sender = sender or (lambda b: received.append(list(b)))
    sdk = LagoSDK(api_key="dummy", default_subscription_id=default_sub)
    sdk._queue._sender = sender  # type: ignore[attr-defined]
    return sdk, received


def test_emit_only_nonzero_fields():
    sdk, received = _new_sdk(default_sub="sub_default")
    u = CanonicalUsage(input=10, output=20, cache_read=0, model="m", provider="p", api="bedrock_invoke")
    sdk.emit(u)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    codes = {e["code"] for e in flat}
    assert codes == {"llm_input_tokens", "llm_output_tokens"}
    for e in flat:
        assert e["external_subscription_id"] == "sub_default"


def test_per_call_subscription_overrides_contextvar_and_default():
    sdk, received = _new_sdk(default_sub="sub_default")
    tok = sdk.set_subscription("sub_ctx")
    try:
        u = CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke")
        sdk.emit(u, subscription="sub_call")
        assert sdk.flush(timeout=2.0)
        sdk.shutdown(timeout=1.0)
    finally:
        sdk.reset_subscription(tok)
    flat = [e for batch in received for e in batch]
    assert all(e["external_subscription_id"] == "sub_call" for e in flat)


def test_contextvar_overrides_default():
    sdk, received = _new_sdk(default_sub="sub_default")
    tok = sdk.set_subscription("sub_ctx")
    try:
        u = CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke")
        sdk.emit(u)
        assert sdk.flush(timeout=2.0)
        sdk.shutdown(timeout=1.0)
    finally:
        sdk.reset_subscription(tok)
    flat = [e for batch in received for e in batch]
    assert all(e["external_subscription_id"] == "sub_ctx" for e in flat)


def test_no_resolvable_subscription_drops_events():
    sdk, received = _new_sdk(default_sub=None)
    u = CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke")
    sdk.emit(u)
    assert sdk.flush(timeout=1.0)
    sdk.shutdown(timeout=1.0)
    assert not received  # nothing emitted


def test_emit_never_raises_on_inner_failure():
    """emit() must swallow internal errors — instrumentation never breaks the call."""
    sdk, _ = _new_sdk(default_sub="sub")
    # Force the queue to be broken
    sdk._queue.push = lambda e: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[assignment]
    u = CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke")
    sdk.emit(u)  # must not raise
    sdk.shutdown(timeout=1.0)


def test_wrap_unknown_client_raises_at_wrap_time():
    sdk, _ = _new_sdk()
    with pytest.raises(UnknownClientError):
        sdk.wrap(object())
    sdk.shutdown(timeout=1.0)


def test_dimensions_merge_into_event_properties():
    sdk, received = _new_sdk(default_sub="sub")
    u = CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke")
    sdk.emit(u, dimensions={"project": "demo", "tenant": "acme"})
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert flat[0]["properties"]["project"] == "demo"
    assert flat[0]["properties"]["tenant"] == "acme"


def test_caller_dimensions_win_on_a_collision_on_both_emitters():
    """One rule across both paths: a caller dimension overrides every
    SDK-computed property of the same name.

    The cost path used to spread dimensions into `base_properties`, i.e. BEFORE
    `unit`/`value`/`base_cost`/`unit_price`, so those four silently overwrote a
    same-named caller dimension there while the token path honoured it. Same
    customer config, two different outcomes depending on the mode.
    """
    dims = {"unit": "seat", "value": "CUSTOM", "model": "my-label", "team": "platform"}
    u = CanonicalUsage(input=100, output=50, model="claude-sonnet-4-5", provider="anthropic", api="native")

    # Token path.
    sdk_tok, got_tok = _new_sdk(default_sub="sub")
    sdk_tok.emit(u, dimensions=dims, mode="tokens")
    assert sdk_tok.flush(timeout=2.0)
    sdk_tok.shutdown(timeout=1.0)
    tok = [e for batch in got_tok for e in batch]

    # Cost path (precomputed, so no price table needed).
    sdk_cost, got_cost = _new_sdk(default_sub="sub")
    sdk_cost.emit(u, dimensions=dims, mode="price", usd_cost=0.01)
    assert sdk_cost.flush(timeout=2.0)
    sdk_cost.shutdown(timeout=1.0)
    cost = [e for batch in got_cost for e in batch]

    assert tok and cost
    for label, events in (("token", tok), ("cost", cost)):
        for e in events:
            p = e["properties"]
            assert p["unit"] == "seat", f"{label}: caller `unit` must win"
            assert p["value"] == "CUSTOM", f"{label}: caller `value` must win"
            assert p["model"] == "my-label", f"{label}: caller `model` must win"
            assert p["team"] == "platform"

    # The accepted consequence of that rule, pinned deliberately: a dimension
    # named `value` overrides the reported quantity. It is NOT able to touch the
    # charged amount on a cost event, because `precise_total_amount_cents` is a
    # sibling of `properties`, not a member of it.
    assert cost[0]["precise_total_amount_cents"] == "1"

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

"""Wrapper tests — return value preserved, body re-readable, idempotent, never raises."""
from __future__ import annotations

import io
import json

from lago_agent_sdk import LagoSDK


class FakeServiceModel:
    service_name = "bedrock-runtime"


class FakeMeta:
    service_model = FakeServiceModel()


class FakeBedrockClient:
    """Mimics the surface area of `boto3.client('bedrock-runtime')`."""

    def __init__(self, converse_response: dict | None = None, invoke_response_body: bytes | None = None):
        self.meta = FakeMeta()
        self._converse_response = converse_response or {
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30, "serverToolUsage": {}},
            "output": {},
        }
        self._invoke_body_bytes = invoke_response_body or json.dumps(
            {"usage": {"input_tokens": 5, "output_tokens": 7}, "content": [{"type": "text", "text": "hi"}]}
        ).encode("utf-8")
        self.converse_calls = 0
        self.invoke_calls = 0

    def converse(self, **kwargs):
        self.converse_calls += 1
        # Verify lago-only kwarg is stripped before reaching the client
        assert "extra_lago" not in kwargs
        return dict(self._converse_response)

    def converse_stream(self, **kwargs):
        self.converse_calls += 1
        assert "extra_lago" not in kwargs
        events = [
            {"contentBlockDelta": {"delta": {"text": "hi"}, "contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 11, "outputTokens": 22, "totalTokens": 33}}},
        ]
        return {"stream": iter(events)}

    def invoke_model(self, **kwargs):
        self.invoke_calls += 1
        assert "extra_lago" not in kwargs
        body = io.BytesIO(self._invoke_body_bytes)
        body_obj = type("FakeBody", (), {"read": body.read, "_io": body})()
        return {"body": body_obj, "contentType": "application/json"}

    def invoke_model_with_response_stream(self, **kwargs):
        self.invoke_calls += 1
        assert "extra_lago" not in kwargs
        chunks = [
            # delta chunks the customer iterates through
            {"chunk": {"bytes": json.dumps({"type": "content_block_delta", "delta": {"text": "hi"}}).encode()}},
            {"chunk": {"bytes": json.dumps({"type": "content_block_stop"}).encode()}},
            # final chunk carries Anthropic-style usage
            {"chunk": {"bytes": json.dumps({"type": "message_delta", "usage": {"input_tokens": 9, "output_tokens": 14}}).encode()}},
        ]
        return {"body": iter(chunks), "contentType": "application/json"}


def _make_sdk(default_sub: str = "sub_test") -> tuple[LagoSDK, list]:
    received: list = []
    sdk = LagoSDK(api_key="dummy", default_subscription_id=default_sub)
    sdk._queue._sender = lambda b: received.append(list(b))  # type: ignore[attr-defined]
    return sdk, received


def test_wrap_preserves_converse_return_value():
    sdk, received = _make_sdk()
    fake = FakeBedrockClient()
    client = sdk.wrap(fake)
    resp = client.converse(modelId="eu.amazon.nova-lite-v1:0", messages=[])
    assert resp["usage"]["inputTokens"] == 10
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    codes = {e["code"] for e in flat}
    assert codes == {"llm_input_tokens", "llm_output_tokens"}


def test_wrap_invoke_model_body_remains_readable():
    sdk, _ = _make_sdk()
    fake = FakeBedrockClient()
    client = sdk.wrap(fake)
    resp = client.invoke_model(modelId="eu.anthropic.claude-sonnet-4-6", body=b"{}")
    raw = resp["body"].read()
    parsed = json.loads(raw.decode("utf-8"))
    assert parsed["usage"]["input_tokens"] == 5
    assert parsed["usage"]["output_tokens"] == 7
    sdk.shutdown(timeout=1.0)


def test_wrap_double_wrap_is_idempotent():
    sdk, _ = _make_sdk()
    fake = FakeBedrockClient()
    once = sdk.wrap(fake)
    twice = sdk.wrap(once)
    assert once is twice
    # Original method is wrapped once, not twice
    resp = twice.converse(modelId="eu.amazon.nova-lite-v1:0", messages=[])
    assert resp["usage"]["inputTokens"] == 10
    assert fake.converse_calls == 1
    sdk.shutdown(timeout=1.0)


def test_wrap_double_wrap_emits_exactly_once_per_call():
    """Beyond identity check — the second wrap MUST NOT install a second instrumentation layer."""
    sdk, received = _make_sdk()
    fake = FakeBedrockClient()
    sdk.wrap(fake)
    sdk.wrap(fake)  # second wrap must be a no-op
    sdk.wrap(fake)  # and a third
    fake.converse(modelId="eu.amazon.nova-lite-v1:0", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    # 1 call → 2 events (input + output). Triple-wrap would produce 6.
    assert len(flat) == 2, f"double-wrap produced {len(flat)} events instead of 2"
    assert fake.converse_calls == 1, f"underlying converse hit {fake.converse_calls}× instead of 1"


def test_wrap_strips_extra_lago_kwarg_and_uses_per_call_sub():
    sdk, received = _make_sdk(default_sub="sub_default")
    fake = FakeBedrockClient()
    client = sdk.wrap(fake)
    client.converse(
        modelId="eu.amazon.nova-lite-v1:0",
        messages=[],
        extra_lago={"subscription": "sub_per_call", "dimensions": {"feature": "X"}},
    )
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert all(e["external_subscription_id"] == "sub_per_call" for e in flat)
    assert flat[0]["properties"]["feature"] == "X"


def test_wrap_instrumentation_failure_does_not_break_call():
    sdk, _ = _make_sdk()
    fake = FakeBedrockClient(
        converse_response={"usage": "not-a-dict-on-purpose"}  # adapter will receive bad shape
    )
    client = sdk.wrap(fake)
    # If the adapter throws on this bad shape, the wrapper must still return resp.
    resp = client.converse(modelId="eu.something", messages=[])
    assert resp["usage"] == "not-a-dict-on-purpose"
    sdk.shutdown(timeout=1.0)


def test_wrap_converse_stream_captures_usage_from_metadata_event():
    sdk, received = _make_sdk()
    fake = FakeBedrockClient()
    client = sdk.wrap(fake)
    resp = client.converse_stream(modelId="eu.amazon.nova-lite-v1:0", messages=[])
    # Drain the stream — wrapper extracts usage on completion
    events = list(resp["stream"])
    assert len(events) == 3
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    codes = {e["code"] for e in flat}
    assert codes == {"llm_input_tokens", "llm_output_tokens"}


def test_wrap_invoke_model_stream_captures_usage_from_final_chunk():
    sdk, received = _make_sdk()
    fake = FakeBedrockClient()
    client = sdk.wrap(fake)
    resp = client.invoke_model_with_response_stream(
        modelId="eu.anthropic.claude-sonnet-4-6", body=b"{}"
    )
    # Customer drains the body iterator — wrapper extracts usage on completion.
    chunks = list(resp["body"])
    assert len(chunks) == 3
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in flat}
    assert by_code["llm_input_tokens"] == 9
    assert by_code["llm_output_tokens"] == 14


def test_wrap_emit_called_once_per_call():
    sdk, received = _make_sdk()
    fake = FakeBedrockClient()
    client = sdk.wrap(fake)
    client.converse(modelId="eu.amazon.nova-lite-v1:0", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    # 2 events for input/output — not 4 (no double-emit)
    assert len(flat) == 2

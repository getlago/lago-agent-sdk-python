"""Mistral wrapper tests — fake client, no live API."""

from __future__ import annotations

from lago_agent_sdk import LagoSDK


# ----------------------------------------------------------------------
# Fake mistral SDK that mimics the surface area of mistralai.client.Mistral
# ----------------------------------------------------------------------
class FakePydanticResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self) -> dict:
        return self._payload


class FakeStreamChunk:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self) -> dict:
        return self._payload


class FakeChat:
    def __init__(self):
        self.complete_calls = 0
        self.stream_calls = 0

    def complete(self, **kwargs):
        self.complete_calls += 1
        assert "extra_lago" not in kwargs
        return FakePydanticResponse(
            {
                "model": kwargs.get("model", "mistral-small-latest"),
                "choices": [{"message": {"content": "hi", "tool_calls": None}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
            }
        )

    def stream(self, **kwargs):
        self.stream_calls += 1
        assert "extra_lago" not in kwargs
        chunks = [
            FakeStreamChunk({"data": {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]}}),
            FakeStreamChunk(
                {
                    "data": {
                        "choices": [{"delta": {"content": "."}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
                    }
                }
            ),
        ]
        return iter(chunks)


class FakeMistral:
    """Mimics `from mistralai.client import Mistral; Mistral(api_key=...)`."""

    __module__ = "mistralai.client.sdk"

    def __init__(self):
        self.chat = FakeChat()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _make_sdk(default_sub: str = "sub_test") -> tuple[LagoSDK, list]:
    received: list = []
    sdk = LagoSDK(api_key="dummy", default_subscription_id=default_sub)
    sdk._queue._sender = lambda b: received.append(list(b))  # type: ignore[attr-defined]
    return sdk, received


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_wrap_chat_complete_emits_input_and_output():
    sdk, received = _make_sdk()
    fake = FakeMistral()
    client = sdk.wrap(fake)
    resp = client.chat.complete(model="mistral-small-latest", messages=[])
    assert resp.model_dump()["usage"]["prompt_tokens"] == 12
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in flat}
    assert by_code["llm_input_tokens"] == 12
    assert by_code["llm_output_tokens"] == 7


def test_wrap_strips_extra_lago_kwarg_and_uses_per_call_sub():
    sdk, received = _make_sdk("sub_default")
    fake = FakeMistral()
    client = sdk.wrap(fake)
    client.chat.complete(
        model="mistral-small-latest",
        messages=[],
        extra_lago={"subscription": "sub_per_call", "dimensions": {"feature": "X"}},
    )
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert all(e["external_subscription_id"] == "sub_per_call" for e in flat)
    assert flat[0]["properties"]["feature"] == "X"


def test_wrap_double_wrap_is_idempotent():
    sdk, received = _make_sdk()
    fake = FakeMistral()
    sdk.wrap(fake)
    sdk.wrap(fake)
    sdk.wrap(fake)
    fake.chat.complete(model="mistral-small-latest", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    # 2 events from 1 call (no triple-wrap = no 6 events)
    assert len(flat) == 2
    assert fake.chat.complete_calls == 1


def test_wrap_chat_stream_captures_usage_from_final_chunk():
    sdk, received = _make_sdk()
    fake = FakeMistral()
    client = sdk.wrap(fake)
    chunks = list(client.chat.stream(model="mistral-small-latest", messages=[]))
    assert len(chunks) == 2
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in flat}
    assert by_code["llm_input_tokens"] == 9
    assert by_code["llm_output_tokens"] == 4


def test_wrap_instrumentation_failure_does_not_break_call():
    """Adapter failure must not propagate to the customer's call."""
    sdk, _ = _make_sdk()

    class BadResp:
        def model_dump(self):
            raise RuntimeError("boom")

    class BadChat:
        def complete(self, **_kw):
            return BadResp()

    class BadFake:
        __module__ = "mistralai.client.sdk"

        def __init__(self):
            self.chat = BadChat()

    fake = BadFake()
    client = sdk.wrap(fake)
    # Must not raise even though our adapter will crash on this response
    resp = client.chat.complete(model="x", messages=[])
    assert resp is not None
    sdk.shutdown(timeout=1.0)

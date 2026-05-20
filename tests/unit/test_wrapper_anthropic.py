"""Anthropic wrapper tests — fake client, no live API."""

from __future__ import annotations

from typing import Any

from lago_agent_sdk import LagoSDK


class FakeMessage:
    """Mimics Anthropic's Message pydantic object."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        # expose .usage and .content as attribute access for _is_message_like check
        self.usage = payload.get("usage")
        self.content = payload.get("content", [])

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class FakeStreamEvent:
    """Mimics one of Anthropic's MessageStreamEvent objects (MessageDelta/Start/etc.)."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class FakeMessages:
    def __init__(self) -> None:
        self.create_calls = 0
        self.stream_calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.create_calls += 1
        assert "extra_lago" not in kwargs
        if kwargs.get("stream") is True:
            events = [
                FakeStreamEvent({"type": "message_start", "message": {"usage": {"input_tokens": 12}}}),
                FakeStreamEvent(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"input_tokens": 12, "output_tokens": 22},
                    }
                ),
                FakeStreamEvent({"type": "message_stop"}),
            ]
            return iter(events)
        return FakeMessage(
            {
                "model": kwargs.get("model", "claude-sonnet-4-6"),
                "content": [{"type": "text", "text": "hi"}],
                "usage": {
                    "input_tokens": 8,
                    "output_tokens": 16,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 0,
                        "ephemeral_1h_input_tokens": 0,
                    },
                },
            }
        )

    def stream(self, **kwargs: Any) -> Any:
        self.stream_calls += 1
        assert "extra_lago" not in kwargs
        outer = self

        class _FakeStreamManager:
            def __enter__(self_inner) -> Any:
                outer._final = FakeMessage(
                    {
                        "model": kwargs.get("model", "claude-sonnet-4-6"),
                        "content": [{"type": "text", "text": "hi"}],
                        "usage": {
                            "input_tokens": 5,
                            "output_tokens": 11,
                        },
                    }
                )
                return _FakeStreamHandle(outer._final)

            def __exit__(self_inner, exc_type, exc, tb) -> Any:  # noqa: D401
                return False

        return _FakeStreamManager()


class _FakeStreamHandle:
    def __init__(self, final: FakeMessage) -> None:
        self._final = final
        self.text_stream = iter(["hi"])

    def get_final_message(self) -> FakeMessage:
        return self._final


class FakeAnthropic:
    """Mimics `from anthropic import Anthropic; Anthropic(api_key=...)`."""

    def __init__(self) -> None:
        self.messages = FakeMessages()


# Module path needs to contain 'anthropic' so detector.py routes to anthropic wrapper.
FakeAnthropic.__module__ = "anthropic.fake"


def _new_sdk(default_sub: str = "sub_test") -> tuple[LagoSDK, list[dict]]:
    received: list[dict] = []

    def sender(batch: list[dict]) -> None:
        received.extend(batch)

    sdk = LagoSDK(api_key="dummy", default_subscription_id=default_sub)
    sdk._queue._sender = sender  # type: ignore[attr-defined]
    return sdk, received


def test_wrap_messages_create_emits_input_and_output() -> None:
    sdk, received = _new_sdk()
    fake = FakeAnthropic()
    client = sdk.wrap(fake)
    resp = client.messages.create(model="claude-sonnet-4-6", messages=[])
    assert resp.usage["input_tokens"] == 8
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code["llm_input_tokens"] == 8
    assert by_code["llm_output_tokens"] == 16


def test_wrap_strips_extra_lago_and_uses_per_call_sub() -> None:
    sdk, received = _new_sdk("sub_default")
    fake = FakeAnthropic()
    client = sdk.wrap(fake)
    client.messages.create(
        model="claude-sonnet-4-6",
        messages=[],
        extra_lago={"subscription": "sub_per_call", "dimensions": {"feature": "X"}},
    )
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert all(e["external_subscription_id"] == "sub_per_call" for e in received)
    assert received[0]["properties"]["feature"] == "X"


def test_wrap_double_wrap_is_idempotent() -> None:
    sdk, received = _new_sdk()
    fake = FakeAnthropic()
    sdk.wrap(fake)
    sdk.wrap(fake)
    sdk.wrap(fake)
    fake.messages.create(model="claude-sonnet-4-6", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert len(received) == 2  # input + output, not 6
    assert fake.messages.create_calls == 1


def test_wrap_create_with_stream_captures_usage_from_message_delta() -> None:
    sdk, received = _new_sdk()
    fake = FakeAnthropic()
    client = sdk.wrap(fake)
    events = list(client.messages.create(model="claude-sonnet-4-6", messages=[], stream=True))
    assert len(events) == 3
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code["llm_input_tokens"] == 12
    assert by_code["llm_output_tokens"] == 22


def test_wrap_messages_stream_context_manager_emits_on_close() -> None:
    sdk, received = _new_sdk()
    fake = FakeAnthropic()
    client = sdk.wrap(fake)
    with client.messages.stream(model="claude-sonnet-4-6", messages=[]) as stream:
        list(stream.text_stream)
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code["llm_input_tokens"] == 5
    assert by_code["llm_output_tokens"] == 11


def test_instrumentation_failure_does_not_break_call() -> None:
    sdk, _ = _new_sdk()

    class BadMessage:
        @property
        def usage(self):
            raise RuntimeError("boom")

        @property
        def content(self):
            return []

        def model_dump(self):
            raise RuntimeError("boom")

    class BadMessages:
        def create(self, **_kw):
            return BadMessage()

    class BadAnthropic:
        def __init__(self):
            self.messages = BadMessages()

    BadAnthropic.__module__ = "anthropic.fake"

    client = sdk.wrap(BadAnthropic())
    # Adapter will crash inside, but wrap must still return resp.
    resp = client.messages.create(model="x", messages=[])
    assert resp is not None
    sdk.shutdown(timeout=1.0)

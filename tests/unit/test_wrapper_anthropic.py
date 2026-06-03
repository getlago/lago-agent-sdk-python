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
            # Mirrors the real wire shape: message_start carries the authoritative
            # input/cache counts (output only primed to 1) nested under
            # message.usage; message_delta carries ONLY the cumulative output at
            # the top level — it does NOT echo input_tokens. A wrapper that reads
            # only top-level usage therefore bills input_tokens=0.
            events = [
                FakeStreamEvent(
                    {
                        "type": "message_start",
                        "message": {
                            "usage": {
                                "input_tokens": 12,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 0,
                                "output_tokens": 1,
                            }
                        },
                    }
                ),
                FakeStreamEvent(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 22},
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


def test_wrap_create_with_stream_merges_message_start_and_delta() -> None:
    """Regression: input/cache come from message_start, output from message_delta.

    message_delta does not echo input_tokens (only cumulative output), so the
    stream wrapper must merge message_start's nested message.usage with the
    delta. Reading only top-level usage would bill input_tokens=0.
    """
    sdk, received = _new_sdk()
    fake = FakeAnthropic()
    client = sdk.wrap(fake)
    events = list(client.messages.create(model="claude-sonnet-4-6", messages=[], stream=True))
    assert len(events) == 3
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code["llm_input_tokens"] == 12, (
        "input_tokens lost — wrapper ignored message_start's nested message.usage"
    )
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


# ==========================================================================
# ASYNC PATH — AsyncAnthropic + AsyncMessageStream context manager.
# Exercises the async wrapper paths: _create_async, _wrap_async_stream,
# and _LagoStreamManager.__aenter__ / __aexit__ where the final-message
# coroutine must be awaited.
# ==========================================================================
import pytest  # noqa: E402  — late import keeps the file's main top-of-file clean


class FakeAsyncMessages:
    def __init__(self) -> None:
        self.create_calls = 0
        self.stream_calls = 0
        self.final_message_awaited = False  # tracks whether the async path was actually awaited

    async def create(self, **kwargs: Any) -> Any:
        self.create_calls += 1
        assert "extra_lago" not in kwargs

        if kwargs.get("stream") is True:

            async def _aiter():
                # Realistic wire shape: input/cache only on message_start;
                # message_delta carries cumulative output, no input echo.
                yield FakeStreamEvent(
                    {
                        "type": "message_start",
                        "message": {
                            "usage": {
                                "input_tokens": 12,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 0,
                                "output_tokens": 1,
                            }
                        },
                    },
                )
                yield FakeStreamEvent(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 22},
                    }
                )
                yield FakeStreamEvent({"type": "message_stop"})

            return _aiter()

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

        final_message = FakeMessage(
            {
                "model": kwargs.get("model", "claude-sonnet-4-6"),
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 5, "output_tokens": 11},
            }
        )

        class _FakeAsyncStreamManager:
            async def __aenter__(self_inner) -> Any:
                return _FakeAsyncStreamHandle(final_message, outer)

            async def __aexit__(self_inner, exc_type, exc, tb) -> Any:  # noqa: D401
                return False

        return _FakeAsyncStreamManager()


class _FakeAsyncStreamHandle:
    def __init__(self, final: Any, parent: FakeAsyncMessages) -> None:
        self._final = final
        self._parent = parent

    async def get_final_message(self) -> Any:
        """async by design — mirrors the real AsyncMessageStream.get_final_message()."""
        self._parent.final_message_awaited = True
        return self._final


class FakeAsyncAnthropic:
    """Mimics `from anthropic import AsyncAnthropic; AsyncAnthropic(api_key=...)`."""

    def __init__(self) -> None:
        self.messages = FakeAsyncMessages()


# Wrapper detects async via type(client).__name__.startswith("Async"), so we
# override __name__ to "AsyncAnthropic" to mimic the real `AsyncAnthropic` class.
FakeAsyncAnthropic.__module__ = "anthropic.fake"
FakeAsyncAnthropic.__name__ = "AsyncAnthropic"


@pytest.mark.asyncio
async def test_async_wrap_messages_create_emits() -> None:
    sdk, received = _new_sdk()
    fake = FakeAsyncAnthropic()
    client = sdk.wrap(fake)
    resp = await client.messages.create(model="claude-sonnet-4-6", messages=[])
    assert resp.usage["input_tokens"] == 8
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code["llm_input_tokens"] == 8
    assert by_code["llm_output_tokens"] == 16


@pytest.mark.asyncio
async def test_async_wrap_messages_create_stream_captures_usage() -> None:
    """Async iteration of `messages.create(stream=True)` — wraps an async generator."""
    sdk, received = _new_sdk()
    fake = FakeAsyncAnthropic()
    client = sdk.wrap(fake)
    stream = await client.messages.create(model="claude-sonnet-4-6", messages=[], stream=True)
    events = [e async for e in stream]
    assert len(events) == 3
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code["llm_input_tokens"] == 12
    assert by_code["llm_output_tokens"] == 22


@pytest.mark.asyncio
async def test_async_wrap_messages_stream_context_manager_emits() -> None:
    """Regression test: async messages.stream(...) context manager must emit usage.

    `async with client.messages.stream(...)` exits and the wrapper's _emit_final
    is called from __aexit__. On AsyncMessageStream, `get_final_message()` is
    a coroutine — calling it without `await` yields a coroutine object that the
    adapter sees as `{}`, so nothing gets billed. The async exit path must await.
    """
    sdk, received = _new_sdk()
    fake = FakeAsyncAnthropic()
    client = sdk.wrap(fake)
    async with client.messages.stream(model="claude-sonnet-4-6", messages=[]) as stream:
        # Customer would iterate via stream.text_stream; not required for this test.
        pass
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code.get("llm_input_tokens") == 5, (
        "Async messages.stream context-manager did not emit usage. "
        "Likely _emit_final calls get_final_message() synchronously, but on "
        "AsyncMessageStream it is a coroutine — needs `await` on the __aexit__ path."
    )
    assert by_code.get("llm_output_tokens") == 11
    assert fake.messages.final_message_awaited, (
        "get_final_message() was never awaited — its coroutine was discarded."
    )

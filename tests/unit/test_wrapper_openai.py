"""OpenAI wrapper tests — fake client, no live API."""

from __future__ import annotations

from typing import Any

from lago_agent_sdk import LagoSDK


class FakeChatCompletion:
    """Mimics openai's ChatCompletion pydantic object."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        # expose .usage so the wrapper's _is_response_like check passes
        self.usage = payload.get("usage")

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class FakeResponsesResponse:
    """Mimics openai's Response object (Responses API)."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.usage = payload.get("usage")

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class FakeStreamChunk:
    """Mimics a ChatCompletionChunk."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class FakeCompletions:
    def __init__(self) -> None:
        self.create_calls = 0
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.create_calls += 1
        # extra_lago must be stripped by the wrapper before reaching here
        assert "extra_lago" not in kwargs
        self.last_kwargs = dict(kwargs)

        if kwargs.get("stream") is True:
            # Stream yields several chunks; the LAST one carries usage
            # (because the wrapper auto-injects stream_options.include_usage).
            chunks = [
                FakeStreamChunk(
                    {"choices": [{"delta": {"content": "hi"}}], "usage": None},
                ),
                FakeStreamChunk(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 22,
                            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                            "completion_tokens_details": {
                                "reasoning_tokens": 0,
                                "audio_tokens": 0,
                            },
                        },
                    }
                ),
            ]
            return iter(chunks)

        # Non-streaming: return a ChatCompletion-like object with .usage
        return FakeChatCompletion(
            {
                "model": kwargs.get("model", "gpt-4o-mini"),
                "choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": None}}],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 16,
                    "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                    "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0},
                },
            }
        )


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeResponsesNamespace:
    def __init__(self) -> None:
        self.create_calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.create_calls += 1
        assert "extra_lago" not in kwargs
        return FakeResponsesResponse(
            {
                "model": kwargs.get("model", "gpt-4o-mini"),
                "output": [{"type": "function_call", "name": "get_weather"}],
                "usage": {
                    "input_tokens": 53,
                    "output_tokens": 6,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )


class FakeOpenAI:
    """Mimics `from openai import OpenAI; OpenAI(api_key=...)`."""

    def __init__(self) -> None:
        self.chat = FakeChat()
        self.responses = FakeResponsesNamespace()


# Module path needs to contain 'openai' so detector routes to openai wrapper.
FakeOpenAI.__module__ = "openai.fake"


def _new_sdk(default_sub: str = "sub_test") -> tuple[LagoSDK, list[dict]]:
    received: list[dict] = []

    def sender(batch: list[dict]) -> None:
        received.extend(batch)

    sdk = LagoSDK(api_key="dummy", default_subscription_id=default_sub)
    sdk._queue._sender = sender  # type: ignore[attr-defined]
    return sdk, received


# --------------------------------------------------------------------------
# Chat Completions
# --------------------------------------------------------------------------
def test_wrap_chat_completions_create_emits_input_and_output() -> None:
    sdk, received = _new_sdk()
    fake = FakeOpenAI()
    client = sdk.wrap(fake)
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[])
    assert resp.usage["prompt_tokens"] == 8
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code["llm_input_tokens"] == 8
    assert by_code["llm_output_tokens"] == 16


def test_wrap_strips_extra_lago_and_uses_per_call_sub() -> None:
    sdk, received = _new_sdk("sub_default")
    fake = FakeOpenAI()
    client = sdk.wrap(fake)
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[],
        extra_lago={"subscription": "sub_per_call", "dimensions": {"feature": "X"}},
    )
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert all(e["external_subscription_id"] == "sub_per_call" for e in received)
    assert received[0]["properties"]["feature"] == "X"


def test_wrap_double_wrap_is_idempotent() -> None:
    sdk, received = _new_sdk()
    fake = FakeOpenAI()
    sdk.wrap(fake)
    sdk.wrap(fake)
    sdk.wrap(fake)
    fake.chat.completions.create(model="gpt-4o-mini", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert len(received) == 2  # input + output, not 6
    assert fake.chat.completions.create_calls == 1


def test_wrap_create_with_stream_captures_usage_from_final_chunk() -> None:
    sdk, received = _new_sdk()
    fake = FakeOpenAI()
    client = sdk.wrap(fake)
    chunks = list(client.chat.completions.create(model="gpt-4o-mini", messages=[], stream=True))
    assert len(chunks) == 2  # first chunk + usage chunk
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code["llm_input_tokens"] == 12
    assert by_code["llm_output_tokens"] == 22


def test_wrap_auto_injects_stream_options_include_usage() -> None:
    """Customer passes stream=True without stream_options — wrapper injects include_usage:True."""
    sdk, _ = _new_sdk()
    fake = FakeOpenAI()
    client = sdk.wrap(fake)
    list(client.chat.completions.create(model="gpt-4o-mini", messages=[], stream=True))
    sdk.shutdown(timeout=1.0)
    seen = fake.chat.completions.last_kwargs or {}
    assert seen.get("stream_options") == {"include_usage": True}


def test_wrap_respects_customer_explicit_include_usage_false() -> None:
    """If customer set include_usage=False explicitly, we don't override."""
    sdk, _ = _new_sdk()
    fake = FakeOpenAI()
    client = sdk.wrap(fake)
    list(
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[],
            stream=True,
            stream_options={"include_usage": False},
        )
    )
    sdk.shutdown(timeout=1.0)
    seen = fake.chat.completions.last_kwargs or {}
    assert seen.get("stream_options") == {"include_usage": False}


def test_wrap_preserves_existing_stream_options_keys() -> None:
    """Existing stream_options keys are kept; include_usage is added alongside."""
    sdk, _ = _new_sdk()
    fake = FakeOpenAI()
    client = sdk.wrap(fake)
    list(
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[],
            stream=True,
            stream_options={"some_other_option": "value"},
        )
    )
    sdk.shutdown(timeout=1.0)
    seen = fake.chat.completions.last_kwargs or {}
    assert seen.get("stream_options") == {"some_other_option": "value", "include_usage": True}


# --------------------------------------------------------------------------
# Responses API
# --------------------------------------------------------------------------
def test_wrap_responses_create_emits_input_output_and_tool_calls() -> None:
    sdk, received = _new_sdk()
    fake = FakeOpenAI()
    client = sdk.wrap(fake)
    resp = client.responses.create(model="gpt-4o-mini", input="hi")
    assert resp.usage["input_tokens"] == 53
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code["llm_input_tokens"] == 53
    assert by_code["llm_output_tokens"] == 6
    assert by_code["llm_tool_calls"] == 1


# --------------------------------------------------------------------------
# Failure isolation
# --------------------------------------------------------------------------
def test_instrumentation_failure_does_not_break_call() -> None:
    sdk, _ = _new_sdk()

    class BadResp:
        @property
        def usage(self):
            raise RuntimeError("boom")

        def model_dump(self):
            raise RuntimeError("boom")

    class BadCompletions:
        def create(self, **_kw):
            return BadResp()

    class BadChat:
        def __init__(self):
            self.completions = BadCompletions()

    class BadOpenAI:
        def __init__(self):
            self.chat = BadChat()
            self.responses = None  # responses namespace deliberately omitted

    BadOpenAI.__module__ = "openai.fake"

    client = sdk.wrap(BadOpenAI())
    # Adapter will crash inside, but wrap must still return resp.
    resp = client.chat.completions.create(model="x", messages=[])
    assert resp is not None
    sdk.shutdown(timeout=1.0)


# ==========================================================================
# ASYNC PATH — AsyncOpenAI variants (mirror of the sync tests above).
# These cover the async wrapper code paths: _create_async, _wrap_async_stream,
# and the Responses-API streaming injection guard.
# ==========================================================================
import pytest  # noqa: E402  — late import keeps the file's main top-of-file clean


class FakeAsyncCompletions:
    def __init__(self) -> None:
        self.create_calls = 0
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.create_calls += 1
        assert "extra_lago" not in kwargs
        self.last_kwargs = dict(kwargs)

        if kwargs.get("stream") is True:

            async def _aiter():
                yield FakeStreamChunk({"choices": [{"delta": {"content": "hi"}}], "usage": None})
                yield FakeStreamChunk(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 22,
                            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                            "completion_tokens_details": {
                                "reasoning_tokens": 0,
                                "audio_tokens": 0,
                            },
                        },
                    }
                )

            return _aiter()

        return FakeChatCompletion(
            {
                "model": kwargs.get("model", "gpt-4o-mini"),
                "choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": None}}],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 16,
                    "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                    "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0},
                },
            }
        )


class FakeAsyncChat:
    def __init__(self) -> None:
        self.completions = FakeAsyncCompletions()


class FakeAsyncResponsesNamespace:
    def __init__(self) -> None:
        self.create_calls = 0
        self.last_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.create_calls += 1
        assert "extra_lago" not in kwargs
        self.last_kwargs = dict(kwargs)

        if kwargs.get("stream") is True:

            async def _aiter():
                yield FakeStreamChunk(
                    {
                        "type": "response.completed",
                        "response": {"usage": {"input_tokens": 53, "output_tokens": 6}},
                    }
                )

            return _aiter()

        return FakeResponsesResponse(
            {
                "model": kwargs.get("model", "gpt-4o-mini"),
                "output": [{"type": "function_call", "name": "get_weather"}],
                "usage": {
                    "input_tokens": 53,
                    "output_tokens": 6,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )


class FakeAsyncOpenAI:
    """Mimics `from openai import AsyncOpenAI; AsyncOpenAI(api_key=...)`."""

    def __init__(self) -> None:
        self.chat = FakeAsyncChat()
        self.responses = FakeAsyncResponsesNamespace()


# Wrapper detects async via type(client).__name__.startswith("Async"), so we
# override __name__ to "AsyncOpenAI" to mimic the real `AsyncOpenAI` class.
FakeAsyncOpenAI.__module__ = "openai.fake"
FakeAsyncOpenAI.__name__ = "AsyncOpenAI"


@pytest.mark.asyncio
async def test_async_wrap_chat_completions_emits() -> None:
    sdk, received = _new_sdk()
    fake = FakeAsyncOpenAI()
    client = sdk.wrap(fake)
    resp = await client.chat.completions.create(model="gpt-4o-mini", messages=[])
    assert resp.usage["prompt_tokens"] == 8
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code["llm_input_tokens"] == 8
    assert by_code["llm_output_tokens"] == 16


@pytest.mark.asyncio
async def test_async_wrap_chat_completions_stream_captures_usage() -> None:
    sdk, received = _new_sdk()
    fake = FakeAsyncOpenAI()
    client = sdk.wrap(fake)
    stream = await client.chat.completions.create(model="gpt-4o-mini", messages=[], stream=True)
    chunks = [c async for c in stream]
    assert len(chunks) == 2
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code["llm_input_tokens"] == 12
    assert by_code["llm_output_tokens"] == 22


@pytest.mark.asyncio
async def test_async_wrap_responses_create_emits() -> None:
    sdk, received = _new_sdk()
    fake = FakeAsyncOpenAI()
    client = sdk.wrap(fake)
    resp = await client.responses.create(model="gpt-4o-mini", input="hi")
    assert resp.usage["input_tokens"] == 53
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code["llm_input_tokens"] == 53
    assert by_code["llm_output_tokens"] == 6
    assert by_code["llm_tool_calls"] == 1


@pytest.mark.asyncio
async def test_async_responses_create_with_stream_does_NOT_inject_stream_options() -> None:
    """Regression test: Responses API + stream=True must not get stream_options.

    The Responses API does not accept the `stream_options` parameter — passing it
    would raise TypeError or HTTP 400. The wrapper must inject `stream_options.
    include_usage=True` ONLY on the chat-completions path.
    """
    sdk, _ = _new_sdk()
    fake = FakeAsyncOpenAI()
    client = sdk.wrap(fake)
    # Stream from the Responses API — the wrapper should NOT inject stream_options.
    stream = await client.responses.create(model="gpt-4o-mini", input="hi", stream=True)
    async for _ in stream:
        pass
    sdk.shutdown(timeout=1.0)
    seen_kwargs = fake.responses.last_kwargs or {}
    assert "stream_options" not in seen_kwargs, (
        "Responses API received `stream_options` — would cause TypeError / 400. "
        "The wrapper should only inject this on Chat Completions, not Responses."
    )


@pytest.mark.asyncio
async def test_async_chat_completions_stream_DOES_inject_stream_options() -> None:
    """Contrast with the test above: on chat.completions the injection IS correct."""
    sdk, _ = _new_sdk()
    fake = FakeAsyncOpenAI()
    client = sdk.wrap(fake)
    stream = await client.chat.completions.create(model="gpt-4o-mini", messages=[], stream=True)
    async for _ in stream:
        pass
    sdk.shutdown(timeout=1.0)
    seen_kwargs = fake.chat.completions.last_kwargs or {}
    assert seen_kwargs.get("stream_options") == {"include_usage": True}


@pytest.mark.asyncio
async def test_async_responses_create_stream_extracts_usage_from_completed_event() -> None:
    """Regression test: Responses API stream events nest usage under `event.response.usage`.

    The terminal `response.completed` event carries the final usage on
    `event.response.usage`, NOT at the event's top level. The stream-wrapper's
    usage extraction must look at the nested field for the Responses API.
    """
    sdk, received = _new_sdk()
    fake = FakeAsyncOpenAI()
    client = sdk.wrap(fake)
    stream = await client.responses.create(model="gpt-4o-mini", input="hi", stream=True)
    async for _ in stream:
        pass
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code.get("llm_input_tokens") == 53, (
        "Responses API stream did not emit usage. Likely the streaming wrapper "
        "looks only at event.usage (top-level), but Responses uses event.response.usage."
    )
    assert by_code.get("llm_output_tokens") == 6

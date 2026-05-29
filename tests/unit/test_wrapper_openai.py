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

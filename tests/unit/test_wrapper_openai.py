"""OpenAI wrapper tests — fake client, no live API."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
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


# What OpenAI resolves the requested "gpt-4o-mini" alias to. Streaming chunks
# report it on every frame; the wrapper must carry it through to the event, or
# pricing looks up an alias OpenRouter doesn't list.
_RESOLVED_STREAM_MODEL = "gpt-4o-mini-2024-07-18"


class FakeStreamChunk:
    """Mimics a ChatCompletionChunk."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return self._payload


class FakeRawResponse:
    """Mimics the return value of `.with_raw_response.create(...)`: `.headers` + `.parse()`."""

    def __init__(self, parsed: Any, headers: dict[str, str] | None = None) -> None:
        self._parsed = parsed
        self.headers = headers or {}

    def parse(self) -> Any:
        return self._parsed


class _RawResponseProxy:
    """Mimics `.with_raw_response` — delegates to the owner's `.create()`, wraps the
    result with whatever headers the test configured on `owner.raw_response_headers`.

    Captures the owner's `.create` bound method at construction time (i.e. before
    `sdk.wrap()` can monkey-patch it) — looking it up dynamically via
    `self._owner.create` at call time would resolve to the *wrapped* method once
    `sdk.wrap()` reassigns it, causing infinite recursion.
    """

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._original_create = owner.create

    def create(self, **kwargs: Any) -> FakeRawResponse:
        parsed = self._original_create(**kwargs)
        return FakeRawResponse(parsed, self._owner.raw_response_headers)


class _AsyncRawResponseProxy:
    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._original_create = owner.create

    async def create(self, **kwargs: Any) -> FakeRawResponse:
        parsed = await self._original_create(**kwargs)
        return FakeRawResponse(parsed, self._owner.raw_response_headers)


class FakeCompletions:
    def __init__(self) -> None:
        self.create_calls = 0
        self.last_kwargs: dict[str, Any] | None = None
        self.raw_response_headers: dict[str, str] = {}
        self.with_raw_response = _RawResponseProxy(self)

    def create(self, **kwargs: Any) -> Any:
        self.create_calls += 1
        # extra_lago must be stripped by the wrapper before reaching here
        assert "extra_lago" not in kwargs
        self.last_kwargs = dict(kwargs)

        if kwargs.get("stream") is True:
            # Stream yields several chunks; the LAST one carries usage
            # (because the wrapper auto-injects stream_options.include_usage).
            # Every real chunk carries the RESOLVED model — a short alias like
            # "gpt-4o-mini" comes back as a dated snapshot. Pricing keys off it.
            chunks = [
                FakeStreamChunk(
                    {
                        "choices": [{"delta": {"content": "hi"}}],
                        "usage": None,
                        "model": _RESOLVED_STREAM_MODEL,
                    },
                ),
                FakeStreamChunk(
                    {
                        "choices": [],
                        "model": _RESOLVED_STREAM_MODEL,
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
        self.raw_response_headers: dict[str, str] = {}
        self.with_raw_response = _RawResponseProxy(self)

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


def test_stream_attributes_the_resolved_model_not_the_requested_alias() -> None:
    """The model-attribution fix has to reach the streaming path too.

    The wrapper rebuilds a synthetic usage payload from the chunks, and dropping
    the chunk's own `model` made `resolve_model` fall back to the requested alias
    — so a streamed call was attributed (and priced) as "gpt-4o-mini" while the
    identical non-streaming call correctly resolved to the dated snapshot. In
    price mode that means the OpenRouter lookup misses and silently degrades to
    token events.
    """
    sdk, received = _new_sdk()
    client = sdk.wrap(FakeOpenAI())
    list(client.chat.completions.create(model="gpt-4o-mini", messages=[], stream=True))
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    models = {e["properties"]["model"] for e in received}
    assert models == {_RESOLVED_STREAM_MODEL}, f"expected the resolved snapshot, got {models}"


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
# Gateway cache-hit detection (non-streaming only)
# --------------------------------------------------------------------------
def test_wrap_cache_miss_still_bills_normally() -> None:
    """No gateway, or a MISS: bills exactly as before — .with_raw_response is the
    new code path, but must be behaviorally invisible with no cache header set."""
    sdk, received = _new_sdk()
    fake = FakeOpenAI()
    client = sdk.wrap(fake)
    client.chat.completions.create(model="gpt-4o-mini", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code["llm_input_tokens"] == 8
    assert by_code["llm_output_tokens"] == 16


def test_wrap_cache_hit_skips_billing_chat_completions() -> None:
    """A gateway-served cache HIT cost the customer nothing — bill nothing for it."""
    sdk, received = _new_sdk()
    fake = FakeOpenAI()
    fake.chat.completions.raw_response_headers = {"cf-aig-cache-status": "HIT"}
    client = sdk.wrap(fake)
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[])
    assert resp.usage["prompt_tokens"] == 8  # customer still gets the real response
    sdk.shutdown(timeout=1.0)
    assert received == []


def test_wrap_cache_hit_skips_billing_responses_api() -> None:
    sdk, received = _new_sdk()
    fake = FakeOpenAI()
    fake.responses.raw_response_headers = {"cf-aig-cache-status": "HIT"}
    client = sdk.wrap(fake)
    resp = client.responses.create(model="gpt-4o-mini", input="hi")
    assert resp.usage["input_tokens"] == 53
    sdk.shutdown(timeout=1.0)
    assert received == []


def test_wrap_cache_status_other_than_hit_still_bills() -> None:
    """Only an exact "HIT" suppresses billing — "MISS", "EXPIRED", or anything else bills."""
    sdk, received = _new_sdk()
    fake = FakeOpenAI()
    fake.chat.completions.raw_response_headers = {"cf-aig-cache-status": "MISS"}
    client = sdk.wrap(fake)
    client.chat.completions.create(model="gpt-4o-mini", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert len(received) == 2


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
        self.raw_response_headers: dict[str, str] = {}
        self.with_raw_response = _AsyncRawResponseProxy(self)

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
        self.raw_response_headers: dict[str, str] = {}
        self.with_raw_response = _AsyncRawResponseProxy(self)

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
async def test_async_wrap_cache_hit_skips_billing() -> None:
    sdk, received = _new_sdk()
    fake = FakeAsyncOpenAI()
    fake.chat.completions.raw_response_headers = {"cf-aig-cache-status": "HIT"}
    client = sdk.wrap(fake)
    resp = await client.chat.completions.create(model="gpt-4o-mini", messages=[])
    assert resp.usage["prompt_tokens"] == 8
    sdk.shutdown(timeout=1.0)
    assert received == []


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


# ----------------------------------------------------------------------
# Databricks: base_url decides the provider, and streaming quirks
# ----------------------------------------------------------------------
from lago_agent_sdk.wrappers.openai import _provider_hint_for  # noqa: E402

_DBX = "https://dbc-0223ef70-2638.cloud.databricks.com"


class _FakeClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url


@pytest.mark.parametrize(
    "base_url,expected",
    [
        # Hosted foundation models — DBU-billed, must NOT reach a vendor price table.
        (f"{_DBX}/ai-gateway/mlflow/v1", "databricks"),
        (f"{_DBX}/ai-gateway/mlflow/v1/", "databricks"),
        # BYOK surfaces keep their real vendor so they price against OpenRouter.
        (f"{_DBX}/ai-gateway/openai/v1", ""),
        (f"{_DBX}/ai-gateway/anthropic", ""),
        # Unrelated clients are untouched.
        ("https://api.openai.com/v1", ""),
        ("https://gateway.ai.cloudflare.com/v1/acct/gw/compat", ""),
        ("", ""),
    ],
)
def test_provider_hint_keys_on_the_mlflow_path_only(base_url: str, expected: str) -> None:
    """Two of Databricks' four surfaces use the SAME openai.OpenAI class, and the
    response body cannot tell them apart — a hosted call echoes a served-entity
    name with no marker. base_url is the only signal.

    Matching `/ai-gateway/mlflow/` and not `/ai-gateway/` is load-bearing: the
    openai/anthropic BYOK surfaces share that prefix and must keep their vendor
    provider, or they would stop being priceable."""
    assert _provider_hint_for(_FakeClient(base_url)) == expected


def test_provider_hint_survives_a_client_without_base_url() -> None:
    """Some client variants don't expose it; instrumentation must never break the
    customer's call over that."""

    class NoBaseUrl:
        pass

    class Raises:
        @property
        def base_url(self) -> str:
            raise RuntimeError("boom")

    assert _provider_hint_for(NoBaseUrl()) == ""
    assert _provider_hint_for(Raises()) == ""


def test_databricks_hosted_call_is_stamped_databricks_end_to_end() -> None:
    """Through the real wrapper: a hosted model must come out as
    provider="databricks" so the price lookup cannot hit. OpenRouter lists bare
    `openai/gpt-oss-20b` at ~0.4x of Databricks' own DBU rate, so being stamped
    "openai" would silently under-bill 2.5-5x the moment a served-entity rename
    let _strip_version match it."""
    sdk, received = _new_sdk()
    fake = FakeOpenAI()
    fake.base_url = f"{_DBX}/ai-gateway/mlflow/v1"
    client = sdk.wrap(fake)
    client.chat.completions.create(model="system.ai.llama-4-maverick", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert received, "nothing emitted"
    assert all(e["properties"]["provider"] == "databricks" for e in received)


def test_databricks_byok_call_keeps_its_vendor_provider() -> None:
    """The mirror: the same client class against the OpenAI BYOK surface must stay
    "openai", because that path IS priceable and was verified exact against
    Databricks' own metered spend on 38 of 38 buckets."""
    sdk, received = _new_sdk()
    fake = FakeOpenAI()
    fake.base_url = f"{_DBX}/ai-gateway/openai/v1"
    client = sdk.wrap(fake)
    client.chat.completions.create(model="gpt-4o", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert all(e["properties"]["provider"] == "openai" for e in received)


class _DbxStreamCompletions:
    """Minimal fake reproducing Databricks' streaming convention, which differs from
    OpenAI's in two measured ways: usage is on EVERY frame and is CUMULATIVE, and
    there is no final usage-only frame — the last frame is an ordinary delta."""

    def __init__(self, cumulative: list[int]) -> None:
        self._cumulative = cumulative
        self.with_raw_response = None  # force the plain .create() path

    def create(self, **kwargs: Any) -> Any:
        assert kwargs.get("stream") is True
        return iter(
            [
                FakeStreamChunk(
                    {
                        "model": "meta-llama-4-maverick-040225",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "a"},
                                "finish_reason": "stop" if n == self._cumulative[-1] else None,
                            }
                        ],
                        "usage": {"prompt_tokens": 14, "completion_tokens": n, "total_tokens": 14 + n},
                    }
                )
                for n in self._cumulative
            ]
        )


class _DbxStreamClient:
    def __init__(self, cumulative: list[int]) -> None:
        self.chat = type("C", (), {"completions": _DbxStreamCompletions(cumulative)})()
        self.base_url = f"{_DBX}/ai-gateway/mlflow/v1"


_DbxStreamClient.__module__ = "openai.fake"


def test_databricks_streaming_cumulative_usage_takes_the_last_frame() -> None:
    """last-usage-wins lands on the correct total by construction. This pins it,
    because a "sum the frames" implementation would bill 1+7+15=23 instead of 15,
    and a "first frame wins" one would bill 1."""
    sdk, received = _new_sdk()
    client = sdk.wrap(_DbxStreamClient([1, 7, 15]))
    list(client.chat.completions.create(model="system.ai.llama-4-maverick", messages=[], stream=True))
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code["llm_input_tokens"] == 14, "cumulative input must not be summed"
    assert by_code["llm_output_tokens"] == 15, "final cumulative value, not 1+7+15"
    assert all(e["properties"]["provider"] == "databricks" for e in received)


def test_databricks_abandoned_stream_bills_the_partial_total() -> None:
    """A behavioral divergence worth pinning rather than discovering later.

    Against real OpenAI, abandoning a stream yields no usage at all — it only
    arrives on a final usage-only chunk — so nothing is billed. Databricks puts a
    cumulative usage on every frame, so the `finally`-block emit bills whatever had
    been generated when the consumer walked away. Arguably better (it bills real
    work), but NOT what the OpenAI path does."""
    sdk, received = _new_sdk()
    client = sdk.wrap(_DbxStreamClient([1, 7, 15]))
    stream = client.chat.completions.create(model="system.ai.llama-4-maverick", messages=[], stream=True)
    for i, _ in enumerate(stream):
        if i == 1:  # abandon after the second frame
            break
    stream.close()  # trigger the generator's finally-block emit deterministically
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code.get("llm_output_tokens") == 7, "partial cumulative count at abandonment"


# ----------------------------------------------------------------------
# Snowflake Cortex: an OpenAI-WIRE endpoint that is not OpenAI
#
# Cortex answers chat completions at
# `https://<account>.snowflakecomputing.com/api/v2/cortex/v1/chat/completions`, so a
# customer reaches it with the ordinary `openai.OpenAI` client and a base_url. The
# response body is an ordinary chat completion — nothing in it names Snowflake — so
# base_url is again the only signal.
# ----------------------------------------------------------------------
_SNOW = "https://example-account.snowflakecomputing.com"
_SNOW_CORTEX = f"{_SNOW}/api/v2/cortex/v1"


@pytest.mark.parametrize(
    "base_url,expected",
    [
        # The OpenAI-compatible wire, i.e. what a wrapped client is actually pointed at.
        (_SNOW_CORTEX, "snowflake"),
        (f"{_SNOW_CORTEX}/", "snowflake"),
        # The Anthropic wire and the native inference endpoint live under the same
        # path, and both are model inference billed in credits.
        (f"{_SNOW}/api/v2/cortex/v1/messages", "snowflake"),
        (f"{_SNOW}/api/v2/cortex/inference:complete", "snowflake"),
        # NOT the host: the SQL API on the same host is what this SDK's own gateway
        # reader drives, and a warehouse query is not model inference.
        (f"{_SNOW}/api/v2/statements", ""),
        (_SNOW, ""),
        (f"{_SNOW}/", ""),
        # No segment after `cortex` is not a reachable OpenAI base_url — the client
        # would POST `/api/v2/cortex/chat/completions`, which Cortex does not serve.
        # Requiring the trailing slash is what keeps `/api/v2/cortexsomething` out.
        (f"{_SNOW}/api/v2/cortex", ""),
    ],
)
def test_provider_hint_keys_on_the_cortex_path_only(base_url: str, expected: str) -> None:
    assert _provider_hint_for(_FakeClient(base_url)) == expected


def test_the_hint_table_is_ordered_and_first_match_wins() -> None:
    """The shape, not the rows: a base_url matching two entries resolves to the first,
    so entries stay ordered most-specific-first. Pinned because the next row (Ramp) is
    added by someone reading the table, not this test."""
    from lago_agent_sdk.wrappers.openai import _PROVIDER_BY_BASE_URL_PATH

    assert _provider_hint_for(_FakeClient(f"{_DBX}/ai-gateway/mlflow/v1/api/v2/cortex/v1")) == "databricks"
    # Every provider a hint can produce must be one `emit()` bills as token counts,
    # or the hint silently turns a priceable call into a permanent price miss.
    from lago_agent_sdk.pricing import TOKEN_BILLED_PROVIDERS

    assert {p for _, p in _PROVIDER_BY_BASE_URL_PATH} <= TOKEN_BILLED_PROVIDERS


class _CortexCompletions:
    """A Cortex chat-completions endpoint. Streaming and non-streaming report the SAME
    usage on purpose — the QA scenario is that the two paths agree — and the numbers
    are an UNCACHED call so they read the same whether or not the additive-cache fix
    for `total_tokens` is on the branch. The cached shape is pinned by that fix's own
    adapter fixtures (11/12_snowflake_cortex_*.json), not here."""

    #: What Cortex echoes back. Note it is Anthropic's model name on an OpenAI wire.
    MODEL = "claude-sonnet-4-5"
    USAGE = {
        "prompt_tokens": 42,
        "completion_tokens": 7,
        "total_tokens": 49,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None
        self.raw_response_headers: dict[str, str] = {}
        self.with_raw_response = _RawResponseProxy(self)

    def create(self, **kwargs: Any) -> Any:
        self.last_kwargs = dict(kwargs)
        model = kwargs.get("model") or self.MODEL
        if kwargs.get("stream"):
            return iter(
                [
                    FakeStreamChunk({"model": model, "choices": [{"delta": {"content": "hi"}}]}),
                    FakeStreamChunk({"model": model, "choices": [], "usage": dict(self.USAGE)}),
                ]
            )
        return FakeChatCompletion(
            {"model": model, "choices": [{"message": {"content": "hi"}}], "usage": dict(self.USAGE)}
        )


class _CortexClient:
    def __init__(self, base_url: str = _SNOW_CORTEX) -> None:
        self.completions = _CortexCompletions()
        self.chat = type("C", (), {"completions": self.completions})()
        self.base_url = base_url


_CortexClient.__module__ = "openai.fake"


def test_cortex_call_is_stamped_snowflake_end_to_end() -> None:
    """The stamp asserted on a WRAPPED call, not on `_provider_hint_for` in isolation:
    the stream-hint bug survived a green suite precisely because one repo only pinned
    the helper. Without the hint every one of these events says provider="openai" for
    usage Snowflake billed in credits."""
    sdk, received = _new_sdk()
    client = sdk.wrap(_CortexClient())
    client.chat.completions.create(model="claude-sonnet-4-5", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert received, "nothing emitted"
    assert all(e["properties"]["provider"] == "snowflake" for e in received)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code == {"llm_input_tokens": 42, "llm_output_tokens": 7}


def test_cortex_streaming_matches_the_non_streaming_call() -> None:
    """Same usage, same stamp, and the model comes from the response rather than the
    requested string — the streaming path takes the hint through a different code
    route (a closure over a synthetic payload), so it needs its own assertion."""
    sdk, received = _new_sdk()
    client = sdk.wrap(_CortexClient())
    list(client.chat.completions.create(model="claude-sonnet-4-5", messages=[], stream=True))
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code == {"llm_input_tokens": 42, "llm_output_tokens": 7}
    assert all(e["properties"]["provider"] == "snowflake" for e in received)
    assert all(e["properties"]["model"] == "claude-sonnet-4-5" for e in received)


def test_a_snowflake_host_that_is_not_cortex_is_not_stamped() -> None:
    """The mirror of the BYOK case: same host, same client class, not model inference.
    A host-only match would stamp "snowflake" on it and make it unpriceable forever."""
    sdk, received = _new_sdk()
    client = sdk.wrap(_CortexClient(base_url=f"{_SNOW}/api/v2/statements"))
    client.chat.completions.create(model="gpt-4o", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert all(e["properties"]["provider"] == "openai" for e in received)


def test_a_plain_openai_client_is_behaviourally_unchanged() -> None:
    """The regression the table refactor could break: no row matches api.openai.com,
    so an ordinary client must emit exactly what it emitted before."""
    sdk, received = _new_sdk()
    fake = FakeOpenAI()
    fake.base_url = "https://api.openai.com/v1"
    client = sdk.wrap(fake)
    client.chat.completions.create(model="gpt-4o-mini", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in received}
    assert by_code == {"llm_input_tokens": 8, "llm_output_tokens": 16}
    assert all(e["properties"]["provider"] == "openai" for e in received)


def test_cortex_params_are_not_mutated_across_calls() -> None:
    """`stream_options` is the one nested object the wrapper writes to. Mutating the
    caller's copy would leak `include_usage` into a later non-streaming call and, worse,
    make a params dict reused across two clients carry the first one's settings."""
    sdk, received = _new_sdk()
    client = sdk.wrap(_CortexClient())
    params: dict[str, Any] = {"model": "claude-sonnet-4-5", "messages": [], "stream_options": {}}
    list(client.chat.completions.create(**params, stream=True))
    list(client.chat.completions.create(**params, stream=True))
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert params["stream_options"] == {}, "the caller's nested dict was written to"
    assert all(e["properties"]["provider"] == "snowflake" for e in received)
    assert len([e for e in received if e["code"] == "llm_input_tokens"]) == 2


def test_a_fully_qualified_cortex_model_keeps_the_customers_spelling() -> None:
    """A fine-tuned Cortex model is named `database.schema.model`. `CanonicalUsage.model`
    reports it verbatim — normalizing it here would misreport what the customer ran, and
    there is nothing to normalize it FOR: "snowflake" reaches no price table at all."""
    sdk, received = _new_sdk()
    client = sdk.wrap(_CortexClient())
    client.chat.completions.create(model="LAGO_DB.CORTEX.my_tuned_mistral7b", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert all(e["properties"]["model"] == "LAGO_DB.CORTEX.my_tuned_mistral7b" for e in received)
    assert all(e["properties"]["provider"] == "snowflake" for e in received)


# ----------------------------------------------------------------------
# Snowflake Cortex: REST-view dedup key (INT-246)
#
# The live path stamps a Cortex call's events with the id a `views=("rest",)`
# backfill derives from that call's REQUEST_ID, so Lago rejects the backfill's
# copy as a duplicate transaction_id instead of billing the call twice. The
# response header `x-snowflake-request-id` IS the view's REQUEST_ID, measured
# byte-identical live 2026-08-26.
# ----------------------------------------------------------------------
_REST_FIXTURE_PATH = (
    Path(__file__).parent / "gateway" / "adapters" / "fixtures" / "snowflake_cortex" / "rest_plain.json"
)
_REST_FIXTURE: dict[str, Any] = json.loads(_REST_FIXTURE_PATH.read_text())
_REST_REQUEST_ID = str(_REST_FIXTURE["REQUEST_ID"])
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class _FakeCortexStream:
    """Mimics `openai.Stream`: an iterator that also carries the httpx response."""

    def __init__(self, chunks: list[Any], headers: dict[str, str]) -> None:
        self._it = iter(chunks)
        self.response = type("R", (), {"headers": headers})()

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        return next(self._it)


def _headered_cortex_client(headers: dict[str, str], base_url: str = _SNOW_CORTEX) -> _CortexClient:
    """A Cortex client whose raw responses and streams carry the given headers.

    Non-streaming usage matches `rest_plain.json`'s FIELD SET (input + output, no
    cache), because the dedup assertion compares transaction ids and those embed the
    canonical field names.
    """
    client = _CortexClient(base_url=base_url)
    client.completions.raw_response_headers = headers
    original_create = client.completions.create

    def create(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            model = kwargs.get("model") or _CortexCompletions.MODEL
            return _FakeCortexStream(
                [
                    FakeStreamChunk({"model": model, "choices": [{"delta": {"content": "hi"}}]}),
                    FakeStreamChunk({"model": model, "choices": [], "usage": dict(_CortexCompletions.USAGE)}),
                ],
                headers,
            )
        return original_create(**kwargs)

    client.completions.create = create  # type: ignore[method-assign]
    return client


def test_a_live_call_and_a_backfill_of_the_same_row_produce_the_same_transaction_ids() -> None:
    """End to end on both sides, not two helpers pinned in isolation: the wrapper path
    goes wrap() -> header -> emit, the backfill path goes view row -> read_usage ->
    backfill_snowflake -> emit, and the assertion is on the events' transaction_ids."""
    from lago_agent_sdk.gateway.snowflake import SnowflakeSource

    sdk, received = _new_sdk()
    client = sdk.wrap(_headered_cortex_client({"x-snowflake-request-id": _REST_REQUEST_ID}))
    client.chat.completions.create(model="claude-sonnet-4-5", messages=[])
    assert sdk.flush(timeout=2.0)
    live_ids = sorted(e["transaction_id"] for e in received)
    assert len(live_ids) == 2  # input + output
    for tx in live_ids:
        assert tx.startswith(f"sfc_rest_sub_test_{_REST_REQUEST_ID}_tok_")

    src = SnowflakeSource("ORG-ACCT", "tok", warehouse="COMPUTE_WH")
    src.query = lambda sql: [_REST_FIXTURE]  # type: ignore[method-assign]
    received.clear()
    sdk.backfill_snowflake(src, "3 hours", views=("rest",), default_subscription="sub_test")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    backfill_ids = sorted(e["transaction_id"] for e in received)
    # The whole ticket in one line: Lago sees the SAME ids and rejects the copies.
    assert backfill_ids == live_ids


def test_a_streamed_call_reaches_the_header_and_carries_the_same_key() -> None:
    """The header is reachable on the stream path: `openai.Stream` exposes the httpx
    response whose headers arrived before the body. Verified live on a streamed
    Cortex call."""
    sdk, received = _new_sdk()
    client = sdk.wrap(_headered_cortex_client({"x-snowflake-request-id": _REST_REQUEST_ID}))
    list(client.chat.completions.create(model="claude-sonnet-4-5", messages=[], stream=True))
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert len(received) == 2
    for e in received:
        assert e["transaction_id"].startswith(f"sfc_rest_sub_test_{_REST_REQUEST_ID}_tok_")


def test_a_real_openai_call_keeps_its_uuid_even_if_a_proxy_injects_the_header() -> None:
    """The gate, not the header, decides. Remove the provider-hint check and this
    fails: the id would come out sfc_rest_-prefixed for a call Snowflake never
    served."""
    sdk, received = _new_sdk()
    client = sdk.wrap(
        _headered_cortex_client(
            {"x-snowflake-request-id": _REST_REQUEST_ID}, base_url="https://api.openai.com/v1"
        )
    )
    client.chat.completions.create(model="gpt-4o-mini", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert len(received) == 2
    for e in received:
        assert _UUID_RE.match(e["transaction_id"])


def test_a_cortex_response_without_the_header_falls_back_to_the_uuid() -> None:
    sdk, received = _new_sdk()
    client = sdk.wrap(_headered_cortex_client({}))
    client.chat.completions.create(model="claude-sonnet-4-5", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert len(received) == 2
    for e in received:
        assert _UUID_RE.match(e["transaction_id"])


def test_a_client_without_with_raw_response_bills_normally_on_a_uuid_no_throw() -> None:
    """The plain-`create()` path has no headers to read — the exact simplified-client
    shape the defensiveness exists for."""
    sdk, received = _new_sdk()
    client = _CortexClient()
    client.completions.with_raw_response = None  # force the plain .create() path
    client = sdk.wrap(client)
    client.chat.completions.create(model="claude-sonnet-4-5", messages=[])
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    assert len(received) == 2
    for e in received:
        assert _UUID_RE.match(e["transaction_id"])

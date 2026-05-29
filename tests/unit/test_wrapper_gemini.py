"""Gemini wrapper tests — fake client, no live API."""

from __future__ import annotations

from typing import Any

from lago_agent_sdk import LagoSDK


# ----------------------------------------------------------------------
# Fake google-genai client mimicking genai.Client.models surface area
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


class FakeModels:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.stream_calls = 0

    def generate_content(self, **kwargs: Any) -> Any:
        self.generate_calls += 1
        assert "extra_lago" not in kwargs
        return FakePydanticResponse(
            {
                "model_version": kwargs.get("model", "gemini-2.5-flash"),
                "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finish_reason": "STOP"}],
                "usage_metadata": {
                    "prompt_token_count": 7,
                    "candidates_token_count": 23,
                    "thoughts_token_count": 0,
                    "total_token_count": 30,
                },
            }
        )

    def generate_content_stream(self, **kwargs: Any) -> Any:
        self.stream_calls += 1
        assert "extra_lago" not in kwargs
        chunks = [
            FakeStreamChunk(
                {
                    "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
                    "usage_metadata": None,  # intermediate chunks don't carry usage
                }
            ),
            FakeStreamChunk(
                {
                    "candidates": [{"content": {"parts": [{"text": "."}]}, "finish_reason": "STOP"}],
                    "usage_metadata": {
                        "prompt_token_count": 9,
                        "candidates_token_count": 4,
                        "thoughts_token_count": 0,
                        "total_token_count": 13,
                    },
                }
            ),
        ]
        return iter(chunks)


class FakeGeminiClient:
    """Mimics `from google import genai; genai.Client(api_key=...)`."""

    __module__ = "google.genai.client"

    def __init__(self) -> None:
        self.models = FakeModels()
        # No .aio in this fake — tests cover the sync path only


# ----------------------------------------------------------------------
# Helpers (same pattern as Bedrock/Mistral wrapper tests)
# ----------------------------------------------------------------------
def _make_sdk(default_sub: str = "sub_test") -> tuple[LagoSDK, list]:
    received: list = []
    sdk = LagoSDK(api_key="dummy", default_subscription_id=default_sub)
    sdk._queue._sender = lambda b: received.append(list(b))  # type: ignore[attr-defined]
    return sdk, received


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_wrap_generate_content_emits_input_and_output() -> None:
    sdk, received = _make_sdk()
    fake = FakeGeminiClient()
    client = sdk.wrap(fake)
    resp = client.models.generate_content(model="gemini-2.5-flash", contents="hi")
    assert resp.model_dump()["usage_metadata"]["prompt_token_count"] == 7
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in flat}
    assert by_code["llm_input_tokens"] == 7
    assert by_code["llm_output_tokens"] == 23


def test_wrap_strips_extra_lago_kwarg_and_uses_per_call_sub() -> None:
    sdk, received = _make_sdk("sub_default")
    fake = FakeGeminiClient()
    client = sdk.wrap(fake)
    client.models.generate_content(
        model="gemini-2.5-flash",
        contents="hi",
        extra_lago={"subscription": "sub_per_call", "dimensions": {"feature": "X"}},
    )
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    assert all(e["external_subscription_id"] == "sub_per_call" for e in flat)
    assert flat[0]["properties"]["feature"] == "X"


def test_wrap_double_wrap_is_idempotent() -> None:
    sdk, received = _make_sdk()
    fake = FakeGeminiClient()
    sdk.wrap(fake)
    sdk.wrap(fake)
    sdk.wrap(fake)
    fake.models.generate_content(model="gemini-2.5-flash", contents="hi")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    # 2 events from 1 call (no triple-wrap = no 6 events)
    assert len(flat) == 2
    assert fake.models.generate_calls == 1


def test_wrap_generate_content_stream_captures_usage_from_final_chunk() -> None:
    sdk, received = _make_sdk()
    fake = FakeGeminiClient()
    client = sdk.wrap(fake)
    chunks = list(client.models.generate_content_stream(model="gemini-2.5-flash", contents="hi"))
    assert len(chunks) == 2
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in flat}
    assert by_code["llm_input_tokens"] == 9
    assert by_code["llm_output_tokens"] == 4


def test_wrap_thinking_emits_reasoning_separately() -> None:
    """Gemini 2.5 emits thoughts_token_count → llm_reasoning_tokens event."""
    sdk, received = _make_sdk()

    class ThinkingModels:
        def generate_content(self, **kwargs):
            return FakePydanticResponse(
                {
                    "usage_metadata": {
                        "prompt_token_count": 10,
                        "candidates_token_count": 50,
                        "thoughts_token_count": 200,
                    }
                }
            )

    class ThinkingClient:
        __module__ = "google.genai.client"

        def __init__(self):
            self.models = ThinkingModels()

    client = sdk.wrap(ThinkingClient())
    client.models.generate_content(model="gemini-2.5-flash", contents="hi")
    assert sdk.flush(timeout=2.0)
    sdk.shutdown(timeout=1.0)
    flat = [e for batch in received for e in batch]
    by_code = {e["code"]: int(float(e["properties"]["value"])) for e in flat}
    assert by_code["llm_input_tokens"] == 10
    assert by_code["llm_output_tokens"] == 50
    assert by_code["llm_reasoning_tokens"] == 200


def test_wrap_instrumentation_failure_does_not_break_call() -> None:
    """Adapter failure must not propagate to the customer's call."""
    sdk, _ = _make_sdk()

    class BadResp:
        def model_dump(self):
            raise RuntimeError("boom")

    class BadModels:
        def generate_content(self, **_kw):
            return BadResp()

    class BadClient:
        __module__ = "google.genai.client"

        def __init__(self):
            self.models = BadModels()

    client = sdk.wrap(BadClient())
    # Must not raise even though our adapter will crash on this response
    resp = client.models.generate_content(model="x", contents="hi")
    assert resp is not None
    sdk.shutdown(timeout=1.0)

"""End-to-end OpenAI integration test — live API + mocked Lago.

Skipped unless OPENAI_API_KEY is set.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lago_agent_sdk import LagoSDK

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


class _MockLago(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        self.server.received.append(json.loads(body))  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *_args, **_kwargs):
        return


def _spawn_lago():
    s = HTTPServer(("127.0.0.1", 0), _MockLago)
    s.received = []  # type: ignore[attr-defined]
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}"


def _collect_events(server) -> list[dict]:
    return [e for p in server.received for e in p["events"]]


def _codes(events) -> set[str]:
    return {e["code"] for e in events}


# --------------------------------------------------------------------------
# Chat Completions
# --------------------------------------------------------------------------
def test_live_openai_chat_completions_create_emits_to_lago() -> None:
    from openai import OpenAI

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hi"}],
            max_completion_tokens=20,
        )
        assert sdk.flush(timeout=10.0)
        sdk.shutdown(timeout=2.0)
        events = _collect_events(server)
        codes = _codes(events)
        assert "llm_input_tokens" in codes
        assert "llm_output_tokens" in codes
        for e in events:
            assert e["properties"]["api"] == "chat_completions"
            assert e["properties"]["provider"] == "openai"
    finally:
        server.shutdown()


def test_live_openai_chat_completions_streaming_emits_from_final_chunk() -> None:
    from openai import OpenAI

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
        # Note: stream_options.include_usage is auto-injected by the wrapper
        for _ in client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hi"}],
            max_completion_tokens=20,
            stream=True,
        ):
            pass
        assert sdk.flush(timeout=10.0)
        sdk.shutdown(timeout=2.0)
        events = _collect_events(server)
        codes = _codes(events)
        assert "llm_input_tokens" in codes
        assert "llm_output_tokens" in codes
    finally:
        server.shutdown()


def test_live_openai_chat_completions_tool_use_emits_tool_calls() -> None:
    from openai import OpenAI

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            max_completion_tokens=200,
        )
        assert sdk.flush(timeout=10.0)
        sdk.shutdown(timeout=2.0)
        events = _collect_events(server)
        assert "llm_tool_calls" in _codes(events)
    finally:
        server.shutdown()


def test_live_openai_reasoning_model_emits_reasoning_tokens() -> None:
    """o-series models populate completion_tokens_details.reasoning_tokens.
    First provider to actually expose this metric."""
    from openai import OpenAI

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
        client.chat.completions.create(
            model="o4-mini",
            messages=[{"role": "user", "content": "What is 17 * 23? Just the number."}],
            max_completion_tokens=2000,
        )
        assert sdk.flush(timeout=30.0)
        sdk.shutdown(timeout=2.0)
        events = _collect_events(server)
        codes = _codes(events)
        assert "llm_input_tokens" in codes
        assert "llm_output_tokens" in codes
        assert "llm_reasoning_tokens" in codes  # ← the key win for OpenAI
    finally:
        server.shutdown()


# --------------------------------------------------------------------------
# Responses API
# --------------------------------------------------------------------------
def test_live_openai_responses_create_emits_to_lago() -> None:
    from openai import OpenAI

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
        client.responses.create(
            model="gpt-4o-mini",
            input="Say hi",
            max_output_tokens=20,
        )
        assert sdk.flush(timeout=10.0)
        sdk.shutdown(timeout=2.0)
        events = _collect_events(server)
        codes = _codes(events)
        assert "llm_input_tokens" in codes
        assert "llm_output_tokens" in codes
        for e in events:
            assert e["properties"]["api"] == "responses"
            assert e["properties"]["provider"] == "openai"
    finally:
        server.shutdown()


def test_live_openai_responses_create_with_stream_emits_to_lago() -> None:
    """Live regression test for two bugs in the Responses API streaming path:

    1. The wrapper must NOT inject `stream_options.include_usage` — Responses
       rejects that param and the call would fail with HTTP 400.
    2. The wrapper must extract usage from `event.response.usage` on the
       terminal `response.completed` event (not from a top-level `event.usage`).
    """
    from openai import OpenAI

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
        stream = client.responses.create(
            model="gpt-4o-mini",
            input="Say hi",
            max_output_tokens=20,
            stream=True,
        )
        # Drain — also verifies the customer's call wasn't broken by injection.
        for _ in stream:
            pass
        assert sdk.flush(timeout=10.0)
        sdk.shutdown(timeout=2.0)
        events = _collect_events(server)
        codes = _codes(events)
        assert "llm_input_tokens" in codes
        assert "llm_output_tokens" in codes
        for e in events:
            assert e["properties"]["api"] == "responses"
    finally:
        server.shutdown()

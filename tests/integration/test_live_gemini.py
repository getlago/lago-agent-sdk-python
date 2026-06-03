"""End-to-end Gemini integration test — live API + mocked Lago.

Skipped unless GEMINI_API_KEY is set.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lago_agent_sdk import LagoSDK

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
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


def test_live_gemini_generate_content_emits_to_lago() -> None:
    from google import genai

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(genai.Client(api_key=os.environ["GEMINI_API_KEY"]))
        client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say hi",
        )
        assert sdk.flush(timeout=10.0)
        sdk.shutdown(timeout=2.0)
        events = _collect_events(server)
        codes = _codes(events)
        assert "llm_input_tokens" in codes
        assert "llm_output_tokens" in codes
        for e in events:
            assert e["properties"]["api"] == "native"
            assert e["properties"]["provider"] == "gemini"
    finally:
        server.shutdown()


def test_live_gemini_streaming_captures_usage_from_final_chunk() -> None:
    from google import genai

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(genai.Client(api_key=os.environ["GEMINI_API_KEY"]))
        for _ in client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents="Count from 1 to 3.",
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


def test_live_gemini_thinking_emits_reasoning() -> None:
    """Gemini 2.5 emits thoughts_token_count → llm_reasoning_tokens event."""
    from google import genai

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(genai.Client(api_key=os.environ["GEMINI_API_KEY"]))
        client.models.generate_content(
            model="gemini-2.5-flash",
            contents="What is 17 * 23? Show your reasoning step by step.",
        )
        assert sdk.flush(timeout=15.0)
        sdk.shutdown(timeout=2.0)
        events = _collect_events(server)
        codes = _codes(events)
        assert "llm_input_tokens" in codes
        assert "llm_output_tokens" in codes
        # Gemini 2.5 reasons even without explicit thinking_config
        assert "llm_reasoning_tokens" in codes
    finally:
        server.shutdown()


def test_live_gemini_tool_use_emits_tool_calls() -> None:
    from google import genai
    from google.genai import types as genai_types

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(genai.Client(api_key=os.environ["GEMINI_API_KEY"]))
        weather_fn = genai_types.FunctionDeclaration(
            name="get_weather",
            description="Get the current weather for a city.",
            parameters=genai_types.Schema(
                type="OBJECT",
                properties={"city": genai_types.Schema(type="STRING")},
                required=["city"],
            ),
        )
        client.models.generate_content(
            model="gemini-2.5-flash",
            contents="What's the weather in Tokyo?",
            config=genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(function_declarations=[weather_fn])],
                tool_config=genai_types.ToolConfig(
                    function_calling_config=genai_types.FunctionCallingConfig(mode="ANY"),
                ),
            ),
        )
        assert sdk.flush(timeout=10.0)
        sdk.shutdown(timeout=2.0)
        events = _collect_events(server)
        assert "llm_tool_calls" in _codes(events)
    finally:
        server.shutdown()

"""End-to-end Anthropic integration test — live API + mocked Lago.

Skipped unless ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lago_agent_sdk import LagoSDK

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
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


def test_live_anthropic_messages_create_emits_to_lago() -> None:
    from anthropic import Anthropic

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            messages=[{"role": "user", "content": "Say hi"}],
        )
        assert sdk.flush(timeout=10.0)
        sdk.shutdown(timeout=2.0)
        events = [e for p in server.received for e in p["events"]]  # type: ignore[attr-defined]
        codes = {e["code"] for e in events}
        assert "llm_input_tokens" in codes
        assert "llm_output_tokens" in codes
        for e in events:
            assert e["properties"]["api"] == "native"
            assert e["properties"]["provider"] == "anthropic"
    finally:
        server.shutdown()


def test_live_anthropic_streaming_emits_from_final_delta() -> None:
    from anthropic import Anthropic

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
        for _ in client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            messages=[{"role": "user", "content": "Say hi"}],
            stream=True,
        ):
            pass
        assert sdk.flush(timeout=10.0)
        sdk.shutdown(timeout=2.0)
        events = [e for p in server.received for e in p["events"]]  # type: ignore[attr-defined]
        codes = {e["code"] for e in events}
        assert "llm_input_tokens" in codes
        assert "llm_output_tokens" in codes
    finally:
        server.shutdown()


def test_live_anthropic_messages_stream_context_manager() -> None:
    from anthropic import Anthropic

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            messages=[{"role": "user", "content": "Say hi"}],
        ) as stream:
            for _ in stream.text_stream:
                pass
        assert sdk.flush(timeout=10.0)
        sdk.shutdown(timeout=2.0)
        events = [e for p in server.received for e in p["events"]]  # type: ignore[attr-defined]
        codes = {e["code"] for e in events}
        assert "llm_input_tokens" in codes
        assert "llm_output_tokens" in codes
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_live_async_anthropic_messages_stream_context_manager_emits() -> None:
    """Live regression test for the async messages.stream(...) context manager.

    Bug: __aexit__ called the sync _emit_final, which invoked
    get_final_message() without await. On AsyncMessageStream that method is
    a coroutine, so the un-awaited object fell through to the adapter as {}
    → zero usage emitted, plus a "coroutine was never awaited" RuntimeWarning.
    """
    from anthropic import AsyncAnthropic

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
        async with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            messages=[{"role": "user", "content": "Say hi"}],
        ) as stream:
            async for _ in stream.text_stream:
                pass
        assert sdk.flush(timeout=10.0)
        sdk.shutdown(timeout=2.0)
        events = [e for p in server.received for e in p["events"]]  # type: ignore[attr-defined]
        codes = {e["code"] for e in events}
        assert "llm_input_tokens" in codes
        assert "llm_output_tokens" in codes
    finally:
        server.shutdown()

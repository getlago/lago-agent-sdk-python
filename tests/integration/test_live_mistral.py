"""End-to-end Mistral integration test — live API + mocked Lago.

Skipped unless MISTRAL_API_KEY is set.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lago_agent_sdk import LagoSDK

pytestmark = pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set",
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


def test_live_mistral_chat_complete_emits_to_lago():
    from mistralai.client import Mistral

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(Mistral(api_key=os.environ["MISTRAL_API_KEY"]))
        client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=20,
        )
        assert sdk.flush(timeout=10.0)
        sdk.shutdown(timeout=2.0)
        events = [e for p in server.received for e in p["events"]]  # type: ignore[attr-defined]
        codes = {e["code"] for e in events}
        assert "llm_input_tokens" in codes
        assert "llm_output_tokens" in codes
        for e in events:
            assert e["properties"]["api"] == "native"
            assert e["properties"]["provider"] == "mistral"
    finally:
        server.shutdown()


def test_live_mistral_chat_stream_emits_to_lago():
    from mistralai.client import Mistral

    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = sdk.wrap(Mistral(api_key=os.environ["MISTRAL_API_KEY"]))
        for _ in client.chat.stream(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=20,
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

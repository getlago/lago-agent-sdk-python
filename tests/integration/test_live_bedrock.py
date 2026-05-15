"""End-to-end integration test — live Bedrock REST + mocked Lago endpoint.

Skipped unless `AWS_BEARER_TOKEN_BEDROCK` is set. Mocks Lago so no real
events are sent. Verifies that wrapping the bearer-token REST flow
produces correctly-shaped events at the Lago HTTP boundary.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests

from lago_agent_sdk import LagoSDK
from lago_agent_sdk.adapters import extract_bedrock_converse

REGION = "eu-west-1"
PROMPT = "One sentence about dolphins."

pytestmark = pytest.mark.skipif(
    not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"),
    reason="AWS_BEARER_TOKEN_BEDROCK not set — skipping live Bedrock integration",
)


class _MockLagoHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        self.server.received_payloads.append(json.loads(body))  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *_args, **_kwargs):  # silence
        return


def _start_mock_lago() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), _MockLagoHandler)
    server.received_payloads = []  # type: ignore[attr-defined]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, f"http://127.0.0.1:{server.server_port}"


def _bearer_call_converse(api_key: str, model_id: str) -> dict:
    url = f"https://bedrock-runtime.{REGION}.amazonaws.com/model/{model_id}/converse"
    body = {
        "messages": [{"role": "user", "content": [{"text": PROMPT}]}],
        "inferenceConfig": {"maxTokens": 50},
    }
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def test_live_converse_to_mocked_lago():
    api_key = os.environ["AWS_BEARER_TOKEN_BEDROCK"]
    server, base_url = _start_mock_lago()
    try:
        sdk = LagoSDK(api_key="lago_dummy", api_url=base_url, default_subscription_id="sub_int")
        model_id = "eu.amazon.nova-lite-v1:0"
        # Use the bearer-token REST surface (works without IAM creds in env)
        resp = _bearer_call_converse(api_key, model_id)
        usage = extract_bedrock_converse(resp, model_id=model_id)
        sdk.emit(usage)
        assert sdk.flush(timeout=5.0)
        sdk.shutdown(timeout=2.0)

        assert len(server.received_payloads) >= 1  # type: ignore[attr-defined]
        events = [e for p in server.received_payloads for e in p["events"]]  # type: ignore[attr-defined]
        codes = {e["code"] for e in events}
        assert "llm_input_tokens" in codes
        assert "llm_output_tokens" in codes
        for e in events:
            assert e["external_subscription_id"] == "sub_int"
            assert e["properties"]["api"] == "bedrock_converse"
            assert e["properties"]["provider"] == "amazon"
    finally:
        server.shutdown()

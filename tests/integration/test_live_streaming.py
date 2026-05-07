"""Live streaming end-to-end against real Bedrock + mock Lago.

Skipped unless AWS_BEARER_TOKEN_BEDROCK is set. Drives real
`converse_stream` and `invoke_model_with_response_stream` via the bearer
REST surface, reshaped into the same flow our wrapper drains.
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import boto3
import pytest

from lago_agent_sdk import LagoSDK

REGION = "eu-west-1"
PROMPT = "One sentence about dolphins."

pytestmark = pytest.mark.skipif(
    not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"),
    reason="AWS_BEARER_TOKEN_BEDROCK not set",
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

    def log_message(self, *_args, **_kwargs):  # silence
        return


def _spawn_lago():
    s = HTTPServer(("127.0.0.1", 0), _MockLago)
    s.received = []  # type: ignore[attr-defined]
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}"


def _fresh_client(sdk: LagoSDK):
    return sdk.wrap(boto3.client("bedrock-runtime", region_name=REGION))


def test_live_converse_stream_emits_events():
    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = _fresh_client(sdk)
        resp = client.converse_stream(
            modelId="eu.amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": PROMPT}]}],
            inferenceConfig={"maxTokens": 30},
        )
        # Drain — wrapper extracts usage from the metadata event
        for _event in resp["stream"]:
            pass
        assert sdk.flush(timeout=10.0)
        sdk.shutdown(timeout=2.0)
        events = [e for p in server.received for e in p["events"]]  # type: ignore[attr-defined]
        codes = {e["code"] for e in events}
        assert "llm_input_tokens" in codes and "llm_output_tokens" in codes
        for e in events:
            assert e["properties"]["api"] == "bedrock_converse"
    finally:
        server.shutdown()


def test_live_invoke_model_stream_emits_events():
    server, url = _spawn_lago()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_int")
        client = _fresh_client(sdk)
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 40,
            "messages": [{"role": "user", "content": PROMPT}],
        })
        resp = client.invoke_model_with_response_stream(
            modelId="eu.anthropic.claude-sonnet-4-6", body=body
        )
        for _event in resp["body"]:
            pass
        assert sdk.flush(timeout=10.0)
        sdk.shutdown(timeout=2.0)
        events = [e for p in server.received for e in p["events"]]  # type: ignore[attr-defined]
        codes = {e["code"] for e in events}
        assert "llm_input_tokens" in codes and "llm_output_tokens" in codes
        for e in events:
            assert e["properties"]["api"] == "bedrock_invoke"
    finally:
        server.shutdown()

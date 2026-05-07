"""Outage replay — Lago fails for N seconds; events buffer and arrive in order on recovery."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from lago_agent_sdk import CanonicalUsage, LagoSDK


class _ToggleableLago(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        if self.server.failing:  # type: ignore[attr-defined]
            self.send_response(503)
            self.end_headers()
            return
        self.server.received.append(json.loads(body))  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *_args, **_kwargs):
        return


def _spawn():
    s = HTTPServer(("127.0.0.1", 0), _ToggleableLago)
    s.received = []  # type: ignore[attr-defined]
    s.failing = False  # type: ignore[attr-defined]
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}"


def test_outage_replay_preserves_order_and_count():
    server, url = _spawn()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_test")
        # Cap backoff low so the test doesn't take a minute
        sdk._queue._max_retry_seconds = 1.0  # type: ignore[attr-defined]

        # 1. Lago is down — push 200 events
        server.failing = True  # type: ignore[attr-defined]
        for i in range(200):
            sdk.emit(
                CanonicalUsage(input=1, model=f"m{i:03d}", provider="p", api="bedrock_invoke"),
            )

        # Give the queue worker a few attempts during the outage
        time.sleep(2.0)

        # 2. Lago comes back
        server.failing = False  # type: ignore[attr-defined]
        assert sdk.flush(timeout=15.0), "queue did not drain after recovery"
        sdk.shutdown(timeout=2.0)
    finally:
        server.shutdown()

    flat = [e for p in server.received for e in p["events"]]  # type: ignore[attr-defined]
    assert len(flat) == 200, f"expected 200 events, got {len(flat)}"

    # Order preserved — model field is m000, m001, ..., m199
    models = [e["properties"]["model"] for e in flat]
    assert models == [f"m{i:03d}" for i in range(200)]


def test_long_outage_at_buffer_cap_drops_oldest_then_drains():
    """Outage long enough to overflow the (small) buffer — oldest dropped, rest drain."""
    server, url = _spawn()
    try:
        sdk = LagoSDK(api_key="x", api_url=url, default_subscription_id="sub_test")
        # Tiny buffer + tiny backoff so the test runs quickly
        sdk._queue._max_buffer_size = 30  # type: ignore[attr-defined]
        sdk._queue._max_retry_seconds = 0.5  # type: ignore[attr-defined]

        server.failing = True  # type: ignore[attr-defined]
        # Push 50 — buffer caps at 30, so 20 oldest get dropped (model='m00'..'m19')
        for i in range(50):
            sdk.emit(CanonicalUsage(input=1, model=f"m{i:02d}", provider="p", api="bedrock_invoke"))
        time.sleep(0.5)

        server.failing = False  # type: ignore[attr-defined]
        assert sdk.flush(timeout=15.0)
        sdk.shutdown(timeout=2.0)
    finally:
        server.shutdown()

    flat = [e for p in server.received for e in p["events"]]  # type: ignore[attr-defined]
    # Expect exactly 30 events, the most recent ones (m20..m49)
    assert len(flat) == 30
    models = sorted({e["properties"]["model"] for e in flat})
    assert models == [f"m{i:02d}" for i in range(20, 50)]

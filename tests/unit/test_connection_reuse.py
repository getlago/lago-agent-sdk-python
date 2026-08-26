"""LagoClient keeps one connection alive across batches.

These count accepts on a real local server rather than asserting that some particular
function was called: a test bound to the call path stops meaning anything the moment
the call path changes, which is how the `verify_ssl` tests came to be issuing live
requests to api.getlago.com.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

import pytest

from lago_agent_sdk import LagoSDK
from lago_agent_sdk.canonical import CanonicalUsage
from lago_agent_sdk.config import LagoConfig
from lago_agent_sdk.lago_client import LagoClient
from lago_agent_sdk.queue import EventQueue


class CountingServer:
    """Minimal HTTP/1.1 server that keeps connections open and counts accepts."""

    def __init__(self, reject_batches: bool = False) -> None:
        # `reject_batches` models Lago: one duplicate transaction_id rolls the whole
        # batch back with a 422, while each event re-sent alone succeeds.
        self.reject_batches = reject_batches
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self.port: int = self._sock.getsockname()[1]
        self.connections = 0
        self.live = 0
        self.requests = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    @property
    def api_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/api/v1"

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        self.live += 1
        buf = b""
        try:
            while not self._stop.is_set():
                while b"\r\n\r\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
                head, _, rest = buf.partition(b"\r\n\r\n")
                length = 0
                for line in head.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        length = int(line.split(b":")[1])
                while len(rest) < length:
                    rest += conn.recv(65536)
                body, buf = rest[:length], rest[length:]
                self.requests += 1
                events = json.loads(body or b'{"events": []}').get("events", [])
                status = (
                    b"422 Unprocessable Entity" if (self.reject_batches and len(events) > 1) else b"200 OK"
                )
                conn.sendall(
                    b"HTTP/1.1 " + status + b"\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\n{}"
                )
        except OSError:
            pass
        finally:
            self.live -= 1
            conn.close()

    def close(self) -> None:
        self._stop.set()
        self._sock.close()


@pytest.fixture
def server() -> Any:
    srv = CountingServer()
    try:
        yield srv
    finally:
        srv.close()


def test_many_batches_share_one_connection(server: CountingServer) -> None:
    """N batches must cost ONE handshake, not N. Each extra connection is ~2 RTT —
    about 276ms against api.getlago.com."""
    client = LagoClient(api_key="k", api_url=server.api_url)
    for _ in range(8):
        client.send_batch([{"transaction_id": "t"}])

    assert server.requests == 8, "server should have seen every batch"
    assert server.connections == 1, (
        f"{server.connections} connections for 8 batches — the client is reopening the "
        "connection per batch and paying a handshake each time"
    )


def test_isolation_after_a_422_reuses_the_same_connection() -> None:
    """Where reuse matters most. One duplicate transaction_id 422s the whole batch, and
    `_send_individually` re-sends all 100 events alone to rescue the valid ones. The
    normal path amortises a handshake over 100 events; this one paid it PER EVENT —
    101 requests costing 101 connections, ~27s of handshake at 137ms RTT."""
    srv = CountingServer(reject_batches=True)
    try:
        client = LagoClient(api_key="k", api_url=srv.api_url)
        q = EventQueue(
            sender=client.send_batch, flush_interval=0.02, max_batch_size=100, max_buffer_size=1000
        )
        try:
            for i in range(100):
                q.push({"transaction_id": f"t{i}", "code": "c"})
            # Wait on the thing under test, not on flush() — this is about how many
            # connections 101 sends cost, and coupling it to flush()'s correctness
            # makes it fail for an unrelated reason.
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and srv.requests < 101:
                time.sleep(0.02)
        finally:
            q._stopping.set()

        assert srv.requests == 101, f"expected 1 batch + 100 isolated sends, got {srv.requests}"
        assert srv.connections == 1, (
            f"the isolation path opened {srv.connections} connections — a handshake per "
            "rescued event is 100x the per-event overhead of the normal path"
        )
    finally:
        srv.close()


def test_shutdown_releases_the_connection(server: CountingServer) -> None:
    """Holding a Session means there is now a socket to leak. `requests.post` closed
    its own per call, so before this nothing outlived `shutdown()` — the SDK has to
    release it explicitly, after the queue's exit drain has finished with it."""
    sdk = LagoSDK(
        api_key="k",
        config=LagoConfig(default_subscription_id="s", api_url=server.api_url),
    )
    sdk.emit(CanonicalUsage(input=10, output=5, model="m", provider="anthropic", api="native"))
    assert sdk.flush(timeout=5) is True
    assert server.live == 1

    sdk.shutdown(timeout=2)

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and server.live:
        time.sleep(0.02)
    assert server.live == 0, "shutdown() left a socket open"

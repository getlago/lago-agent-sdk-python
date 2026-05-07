"""os.fork() — child process must be able to emit and flush.

Without help, a daemon thread does not survive fork on POSIX. The SDK uses
`os.register_at_fork` to recreate the queue thread in the child.
"""
from __future__ import annotations

import json
import os
import sys
import threading

import pytest

from lago_agent_sdk import CanonicalUsage, LagoSDK


@pytest.mark.skipif(sys.platform == "win32", reason="fork unsupported on Windows")
def test_emit_works_after_fork(tmp_path):
    """Parent constructs the SDK; child forks, emits, flushes, writes received events to a file.
    Parent reads the file and asserts at least one event landed."""

    output_path = tmp_path / "child_received.json"

    received_in_parent: list = []
    parent_lock = threading.Lock()

    def parent_sender(batch):
        with parent_lock:
            received_in_parent.extend(batch)

    sdk = LagoSDK(api_key="dummy", default_subscription_id="sub_fork")
    sdk._queue._sender = parent_sender  # type: ignore[attr-defined]

    pid = os.fork()
    if pid == 0:
        # --- child ---
        try:
            child_received: list = []
            child_lock = threading.Lock()

            def child_sender(batch):
                with child_lock:
                    child_received.extend(batch)

            # Re-attach a sender for the child's freshly-created thread.
            sdk._queue._sender = child_sender  # type: ignore[attr-defined]

            sdk.emit(CanonicalUsage(input=42, output=11, model="m", provider="p", api="bedrock_invoke"))
            ok = sdk.flush(timeout=5.0)
            output_path.write_text(json.dumps({"flushed": ok, "events": child_received}))
            os._exit(0)
        except Exception as exc:  # noqa: BLE001
            output_path.write_text(json.dumps({"error": str(exc)}))
            os._exit(2)

    # --- parent ---
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, f"child exited with {status}"

    data = json.loads(output_path.read_text())
    assert data.get("flushed") is True, f"child failed to flush: {data}"
    events = data.get("events", [])
    assert len(events) == 2, f"expected 2 events in child (input+output), got {len(events)}"
    codes = {e["code"] for e in events}
    assert codes == {"llm_input_tokens", "llm_output_tokens"}

    sdk.shutdown(timeout=2.0)

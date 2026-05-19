"""Live Lago reconciliation — emit N events, poll current_usage, verify exact match.

Skipped unless LAGO_API_URL, LAGO_API_KEY, and LAGO_EXTERNAL_SUBSCRIPTION_ID
are set. Requires `truststore` if Lago is on a self-signed dev cert.
"""

from __future__ import annotations

import os
import time

import pytest
import requests

from lago_agent_sdk import CanonicalUsage, LagoSDK

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001
    pass

API_URL = (os.environ.get("LAGO_API_URL") or "").rstrip("/")
API_KEY = os.environ.get("LAGO_API_KEY") or ""
SUB_ID = os.environ.get("LAGO_EXTERNAL_SUBSCRIPTION_ID") or ""
CUST_ID = os.environ.get("LAGO_EXTERNAL_CUSTOMER_ID") or "cust_demo"

pytestmark = pytest.mark.skipif(
    not (API_URL and API_KEY and SUB_ID),
    reason="LAGO_API_URL / LAGO_API_KEY / LAGO_EXTERNAL_SUBSCRIPTION_ID not set",
)


def _read_usage() -> dict[str, float]:
    r = requests.get(
        f"{API_URL}/customers/{CUST_ID}/current_usage",
        params={"external_subscription_id": SUB_ID},
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=15,
    )
    r.raise_for_status()
    out: dict[str, float] = {}
    for c in r.json().get("customer_usage", {}).get("charges_usage", []) or []:
        code = c.get("billable_metric", {}).get("code", "")
        out[code] = float(c.get("units", 0) or 0)
    return out


def test_emit_then_reconcile_with_live_lago():
    """Send 5 known-shape events; assert input/output totals incremented correctly."""
    sdk = LagoSDK(api_key=API_KEY, api_url=API_URL, default_subscription_id=SUB_ID)

    before = _read_usage()
    in_before = before.get("llm_input_tokens", 0.0)
    out_before = before.get("llm_output_tokens", 0.0)

    # Emit 5 events with stable values for arithmetic
    for _ in range(5):
        sdk.emit(
            CanonicalUsage(
                input=100,
                output=200,
                model="claude-sonnet-4-6",
                provider="anthropic",
                api="bedrock_invoke",
            )
        )

    assert sdk.flush(timeout=10.0)
    sdk.shutdown(timeout=3.0)

    # Lago is async — poll for up to 30s
    deadline = time.time() + 30
    after = before
    while time.time() < deadline:
        after = _read_usage()
        in_delta = after.get("llm_input_tokens", 0.0) - in_before
        out_delta = after.get("llm_output_tokens", 0.0) - out_before
        if in_delta >= 500 and out_delta >= 1000:
            break
        time.sleep(1.0)

    in_delta = after.get("llm_input_tokens", 0.0) - in_before
    out_delta = after.get("llm_output_tokens", 0.0) - out_before
    assert in_delta == 500, f"input delta {in_delta} != 500 — events lost or duplicated"
    assert out_delta == 1000, f"output delta {out_delta} != 1000 — events lost or duplicated"

"""Live Lago reconciliation — emit N events, poll current_usage, verify exact match.

This is the ONLY test that proves Lago *accepts* what the SDK emits. Every other
integration test points at an in-process mock, so a wrong metric code, a missing
`dynamic` charge model, or a rejected `precise_total_amount_cents` would pass
there and only surface in production.

Skipped unless LAGO_API_URL, LAGO_API_KEY, and LAGO_EXTERNAL_SUBSCRIPTION_ID are
set. For a local dev Lago behind a self-signed cert (Traefik's default), set
LAGO_VERIFY_SSL=false — the SDK has `LagoConfig.verify_ssl` for exactly that, and
this test honours the same switch on its own reads. `truststore` is an
alternative if the cert is in the OS trust store.
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
# Mirrors LagoConfig.verify_ssl: a local dev instance on a self-signed cert is a
# real, common setup, and without this BOTH halves of this test fail on SSL — the
# SDK's POST and this module's own GET.
VERIFY_SSL = (os.environ.get("LAGO_VERIFY_SSL") or "true").strip().lower() not in (
    "0",
    "false",
    "no",
)

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
        verify=VERIFY_SSL,
    )
    r.raise_for_status()
    out: dict[str, float] = {}
    for c in r.json().get("customer_usage", {}).get("charges_usage", []) or []:
        code = c.get("billable_metric", {}).get("code", "")
        out[code] = float(c.get("units", 0) or 0)
    return out


def test_emit_then_reconcile_with_live_lago():
    """Send 5 known-shape events; assert input/output totals incremented correctly."""
    sdk = LagoSDK(
        api_key=API_KEY,
        api_url=API_URL,
        default_subscription_id=SUB_ID,
        verify_ssl=VERIFY_SSL,
    )

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

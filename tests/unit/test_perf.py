"""Wrap-overhead benchmark — fails if p99 > 5ms."""
from __future__ import annotations

import statistics
import time

from lago_agent_sdk import LagoSDK

from .test_wrapper_bedrock import FakeBedrockClient


def test_wrap_overhead_p99_under_5ms():
    """1000 mocked converse() calls; p99 wrap overhead vs unwrapped baseline must be ≤ 5ms."""
    sdk = LagoSDK(api_key="dummy", default_subscription_id="sub")
    sdk._queue._sender = lambda b: None  # type: ignore[attr-defined]

    fake = FakeBedrockClient()
    wrapped = sdk.wrap(fake)

    # baseline: unwrapped
    base = FakeBedrockClient()
    baseline_durs = []
    for _ in range(1000):
        t0 = time.perf_counter()
        base.converse(modelId="eu.amazon.nova-lite-v1:0", messages=[])
        baseline_durs.append(time.perf_counter() - t0)

    wrapped_durs = []
    for _ in range(1000):
        t0 = time.perf_counter()
        wrapped.converse(modelId="eu.amazon.nova-lite-v1:0", messages=[])
        wrapped_durs.append(time.perf_counter() - t0)

    sdk.shutdown(timeout=1.0)

    base_p99 = statistics.quantiles(baseline_durs, n=100)[98]
    wrap_p99 = statistics.quantiles(wrapped_durs, n=100)[98]
    overhead_ms = (wrap_p99 - base_p99) * 1000
    print(f"\np99 baseline={base_p99*1000:.3f}ms wrapped={wrap_p99*1000:.3f}ms overhead={overhead_ms:.3f}ms")
    assert overhead_ms < 5.0, f"p99 wrap overhead {overhead_ms:.2f}ms exceeds 5ms budget"

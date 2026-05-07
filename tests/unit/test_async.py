"""asyncio context propagation — each task sees its own subscription, no cross-talk."""
from __future__ import annotations

import asyncio
import collections

import pytest

from lago_agent_sdk import CanonicalUsage, LagoSDK


@pytest.mark.asyncio
async def test_subscription_isolated_per_asyncio_task():
    sdk = LagoSDK(api_key="dummy", default_subscription_id=None)
    received: list = []
    sdk._queue._sender = lambda b: received.append(list(b))  # type: ignore[attr-defined]

    async def task(sub_id: str, n: int):
        sdk.set_subscription(sub_id)
        for _ in range(n):
            sdk.emit(CanonicalUsage(input=1, model="m", provider="p", api="bedrock_invoke"))
            await asyncio.sleep(0)  # yield so tasks interleave

    try:
        await asyncio.gather(*[task(f"sub_{i}", 50) for i in range(10)])
        assert sdk.flush(timeout=5.0)
    finally:
        sdk.shutdown(timeout=2.0)

    flat = [e for batch in received for e in batch]
    assert len(flat) == 500  # 10 tasks × 50 emits × 1 nonzero field
    by_sub = collections.Counter(e["external_subscription_id"] for e in flat)
    for i in range(10):
        assert by_sub[f"sub_{i}"] == 50, f"sub_{i} got {by_sub[f'sub_{i}']} events, expected 50"


@pytest.mark.asyncio
async def test_concurrent_tasks_with_different_subs_do_not_leak():
    """Even with mixed timing, no event ends up under the wrong subscription."""
    sdk = LagoSDK(api_key="dummy", default_subscription_id=None)
    received: list = []
    sdk._queue._sender = lambda b: received.append(list(b))  # type: ignore[attr-defined]

    async def task(sub_id: str, payload_marker: int):
        sdk.set_subscription(sub_id)
        # Mark each event with the payload so we can verify sub<->payload mapping
        sdk.emit(CanonicalUsage(input=payload_marker, model=sub_id, provider="p", api="bedrock_invoke"))

    try:
        await asyncio.gather(*[task(f"sub_{i}", i + 1) for i in range(50)])
        assert sdk.flush(timeout=5.0)
    finally:
        sdk.shutdown(timeout=2.0)

    flat = [e for batch in received for e in batch]
    for e in flat:
        sub = e["external_subscription_id"]
        # The model field equals the sub_id we set just before emit
        assert e["properties"]["model"] == sub, f"event for {sub} carries wrong model={e['properties']['model']}"

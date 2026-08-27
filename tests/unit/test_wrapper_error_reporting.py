"""An adapter that throws must reach `on_error`, not just the log.

The adapter call sits OUTSIDE `sdk.emit()`, on the wrapper's side of the boundary, so
emit's own reporting never fires for a failure raised there — and that is the common
failure: provider drift changes a response shape and every subsequent call for that
provider goes unbilled.
"""

from __future__ import annotations

from typing import Any

import pytest

from lago_agent_sdk import LagoSDK
from lago_agent_sdk.canonical import CanonicalUsage
from lago_agent_sdk.config import LagoConfig


class _FakeResponse:
    """Message-like enough that the wrapper hands it to the adapter — which is where
    provider drift actually blows up."""

    def __init__(self) -> None:
        self.usage = {"input_tokens": 10, "output_tokens": 5}
        self.content: list[Any] = []


class _FakeMessages:
    def create(self, **kwargs: Any) -> Any:
        return _FakeResponse()


class _FakeAnthropic:
    # `detect_client_kind` keys on the class's module name.
    __module__ = "anthropic"

    def __init__(self) -> None:
        self.messages = _FakeMessages()


def test_adapter_failure_reaches_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider drift raises inside the ADAPTER, which runs on the wrapper's side of
    `sdk.emit()` — so emit's own error reporting never fires. The call still returns
    normally to the customer; only the billing gap is reported."""
    seen: list[tuple[str, str]] = []

    sdk = LagoSDK(
        api_key="k",
        config=LagoConfig(
            default_subscription_id="sub",
            on_error=lambda exc, where: seen.append((type(exc).__name__, where)),
        ),
    )
    sdk._queue._sender = lambda b: None

    import lago_agent_sdk.wrappers.anthropic as wrapper

    def boom(*args: Any, **kwargs: Any) -> CanonicalUsage:
        raise ValueError("unknown usage shape")

    monkeypatch.setattr(wrapper, "extract_anthropic_native", boom)

    client = sdk.wrap(_FakeAnthropic())
    result = client.messages.create(model="claude-haiku-4-5", messages=[])

    assert result is not None, "the customer's LLM call must still return"
    assert ("ValueError", "emit") in seen, f"on_error never fired for adapter drift: {seen}"
    sdk.shutdown(timeout=1.0)


# ----------------------------------------------------------------------
# connection reuse
# ----------------------------------------------------------------------

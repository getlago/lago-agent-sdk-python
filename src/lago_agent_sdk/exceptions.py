"""Error types for the Lago Agent SDK."""

from __future__ import annotations


class LagoSDKError(Exception):
    """Base class."""


class LagoConfigError(LagoSDKError):
    """Raised at wrap time — fails loud on misconfiguration."""


class LagoApiError(LagoSDKError):
    """HTTP non-2xx from Lago events endpoint."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Lago API error {status}: {body[:200]}")
        self.status = status
        self.body = body


class UnknownClientError(LagoConfigError):
    """`wrap()` received a client kind the SDK does not recognize."""

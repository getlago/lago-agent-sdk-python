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


class PricingUnavailableError(LagoSDKError):
    """Price mode could not resolve a price (table not warm yet, or model not
    matched). Surfaced via on_error; the SDK falls back to emitting token events."""

    def __init__(self, provider: str, model: str, api: str, detail: str | None = None) -> None:
        # `detail` is for the miss that is NOT "table cold / model unknown" — a Ramp
        # Router call served at a non-default tier is unpriced by decision, and the
        # customer needs to read that off the error rather than chase a model name.
        message = f"no price for provider={provider!r} model={model!r} api={api!r}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.api = api
        self.detail = detail

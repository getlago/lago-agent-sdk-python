"""Thin HTTP client to Lago."""

from __future__ import annotations

import json
from typing import Any

import requests

from .exceptions import LagoApiError


class LagoClient:
    def __init__(self, api_key: str, api_url: str, timeout: float = 10.0, verify_ssl: bool = True) -> None:
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        if not verify_ssl:
            # The customer explicitly opted out via config — they've already
            # accepted the risk; requests/urllib3's warning on every single
            # request would just be noise at that point, not new information.
            # Import urllib3 directly rather than through `requests.packages`,
            # which is a legacy compatibility shim that is not guaranteed to exist.
            # Wrapped because this is an optional convenience: suppressing a warning
            # must never be able to fail construction of the SDK itself. That is not
            # hypothetical — `verify_ssl=False` is now a first-class constructor
            # argument that the docstring recommends for local dev, so this line sits
            # on an advertised path, and an ImportError/AttributeError here would
            # take down `LagoSDK()` for the exact setup the flag was added to serve.
            try:
                import urllib3

                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:  # noqa: BLE001
                pass

    def __repr__(self) -> str:
        if not self.api_key:
            masked = "<unset>"
        elif len(self.api_key) <= 8:
            masked = "***"
        else:
            masked = f"***{self.api_key[-4:]}"
        return (
            f"LagoClient(api_key={masked!r}, api_url={self.api_url!r}, "
            f"timeout={self.timeout}, verify_ssl={self.verify_ssl})"
        )

    def send_batch(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        url = f"{self.api_url}/events/batch"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"events": events}
        resp = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=self.timeout, verify=self.verify_ssl
        )
        if not (200 <= resp.status_code < 300):
            raise LagoApiError(resp.status_code, resp.text)

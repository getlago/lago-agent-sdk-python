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
            # Access urllib3 via requests' own re-export — it's only a
            # transitive dependency for us, not one we declare directly.
            requests.packages.urllib3.disable_warnings(  # type: ignore[attr-defined]
                requests.packages.urllib3.exceptions.InsecureRequestWarning  # type: ignore[attr-defined]
            )

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

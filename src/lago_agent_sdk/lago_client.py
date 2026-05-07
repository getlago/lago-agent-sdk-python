"""Thin HTTP client to Lago."""
from __future__ import annotations

import json
from typing import Any

import requests

from .exceptions import LagoApiError


class LagoClient:
    def __init__(self, api_key: str, api_url: str, timeout: float = 10.0) -> None:
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def send_batch(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        url = f"{self.api_url}/events/batch"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"events": events}
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        if not (200 <= resp.status_code < 300):
            raise LagoApiError(resp.status_code, resp.text)

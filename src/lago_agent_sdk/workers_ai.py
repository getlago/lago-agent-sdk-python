"""Workers AI client — the SDK's own, because there is no third-party one to wrap.

Every other provider is instrumented by patching its official client in place. Workers
AI has no such target: the official `cloudflare` package (5.7.0, measured 2026-09-21)
percent-encodes the slash in every model name, so `client.ai.run("@cf/meta/...")` hits
`/ai/run/@cf%2Fmeta%2F...` and Cloudflare answers "No route for that URI" — for every
model, not just partner ones. Patching its generic `client.post(...)` instead would
instrument every Cloudflare API call the customer makes. So the SDK ships this client:
one method, two routes, chosen per model id.

Two kinds of model, two routes — measured, not chosen:

* `@cf/...` models go to the gateway host, `gateway.ai.cloudflare.com/v1/{acct}/{gw}/
  workers-ai/{model}`. That route answers with `cf-aig-cache-status` (a HIT means the
  model never ran and nothing is billed) and `cf-aig-log-id` (emitted as the `cf_log_id`
  dimension, so a Lago row can be put beside its Logs API entry — see
  `gateway/adapters/cloudflare_gateway.py`). The model-in-body variant of that host,
  `.../workers-ai/run`, logs `model: "run"`, which is why it is not used for them.
* Partner models (`typesafe/jev` — no `@`) can only be reached on the *unified* path,
  `api.cloudflare.com/.../ai/run` with `{"model", "input"}` in the body and the gateway
  named in a `cf-aig-gateway-id` header. That is the one route where the partner key the
  customer stored under the gateway's BYOK is consulted; the gateway host answers 402
  "Insufficient balance" for the same call even with the key stored (fixture 08). The
  unified path returns NO `cf-aig-*` headers — a cached replay looks exactly like a fresh
  call — so this client sends `cf-aig-skip-cache: true` there rather than risk billing
  the same answer twice. Pass `extra_headers={"cf-aig-skip-cache": "false"}` to opt back
  in, knowing a replay then bills again.

Billing is by token count from the response's own `usage` block. In price mode a
`@cf/...` model prices from Cloudflare's Workers AI catalog as usual; a partner model is
not listed there and prices from AI Gateway's own cost table instead, fetched per id on
the queue's next tick — so the first call to a partner model in a process bills tokens
and reports the miss through `on_error`, and every call after it bills dollars (see
`pricing.parse_cloudflare_gateway_cost`).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from .adapters.workers_ai_native import extract_workers_ai_native

logger = logging.getLogger("lago_agent_sdk.workers_ai")

_DIRECT_BASE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"
_GATEWAY_BASE = "https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/workers-ai"


def _is_catalog_model(model: str) -> bool:
    """`@cf/...` (and `@hf/...`) ids are Cloudflare-hosted; anything else is a partner model."""
    return model.startswith("@")


class WorkersAIError(Exception):
    """Cloudflare answered the run with an error status or `success: false`.

    Raised *before* any instrumentation, so a 402 (partner model with no BYOK key and no
    gateway credits), a 403 (model not on the account's Workers plan) or a 400 (no such
    model) reaches the caller exactly as the API reported it, and nothing is billed for it.
    """

    def __init__(self, status_code: int, errors: list[dict[str, Any]], body: dict[str, Any]) -> None:
        self.status_code = status_code
        self.errors = errors
        self.body = body
        detail = "; ".join(str(e.get("message", e)) for e in errors) if errors else "no error detail"
        super().__init__(f"Workers AI HTTP {status_code}: {detail}")


class WorkersAI:
    """Minimal Workers AI client with Lago instrumentation. Build one via `LagoSDK.workers_ai()`."""

    def __init__(
        self,
        sdk: Any,
        account_id: str,
        api_token: str,
        *,
        gateway_id: str | None = None,
        gateway_auth: str | None = None,
        timeout: float = 60.0,
        dimensions: dict[str, Any] | None = None,
        subscription: str | None = None,
    ) -> None:
        self._sdk = sdk
        self._timeout = timeout
        self._base_dims = dict(dimensions or {})
        self._base_sub = subscription
        self._gateway_id = gateway_id
        self._direct = _DIRECT_BASE.format(account_id=account_id)
        self._gateway = (
            _GATEWAY_BASE.format(account_id=account_id, gateway_id=gateway_id) if gateway_id else None
        )
        self._auth = {"Authorization": f"Bearer {api_token}"}
        self._gateway_auth = {"cf-aig-authorization": f"Bearer {gateway_auth}"} if gateway_auth else {}
        self._session = requests.Session()

    def url_for(self, model: str) -> str:
        """The endpoint a `run(model, ...)` posts to — see the module docstring for why two."""
        if _is_catalog_model(model):
            return f"{self._gateway}/{model}" if self._gateway else f"{self._direct}/{model}"
        return self._direct

    def _request_for(self, model: str, input: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
        """(headers, body) for the route `url_for(model)` picks."""
        if _is_catalog_model(model):
            # Path route: the body IS the input. Gateway auth only on the gateway host.
            return ({**self._auth, **(self._gateway_auth if self._gateway else {})}, input)
        headers = dict(self._auth)
        if self._gateway_id:
            headers["cf-aig-gateway-id"] = self._gateway_id
            headers["cf-aig-skip-cache"] = "true"  # no cache header on this path — see module docstring
        return (headers, {"model": model, "input": input})

    def run(
        self,
        model: str,
        input: dict[str, Any],
        *,
        extra_lago: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run `model` on `input` and bill the response's usage.

        `input` is whatever the model takes: `{"messages": [...]}` for a chat model,
        `{"state": ..., "questions": {...}}` for `typesafe/jev`. Returns Cloudflare's full
        response envelope (`result`, `success`, `errors`, `messages`) unchanged.

        `extra_lago` takes the same keys as the wrappers' per-call kwarg: `subscription`,
        `dimensions`, `mode`, `markup`. `extra_headers` reaches the request as-is and wins
        over the client's own — e.g. `{"cf-aig-cache-ttl": "300"}` to have the gateway
        cache a `@cf/...` call.

        Streaming (`input["stream"] = True`) is refused up front: the body would be an SSE
        stream this method does not parse, and mis-reading it as JSON would bill zero for
        a call that ran. Use the OpenAI-compatible `/compat` endpoint through
        `sdk.wrap(OpenAI(...))` for streamed chat.
        """
        if input.get("stream"):
            raise ValueError(
                "WorkersAI.run() does not support stream=True; for streamed chat completions wrap an "
                "OpenAI client against the gateway's /compat endpoint instead."
            )
        lago_opts = extra_lago or {}
        headers, body = self._request_for(model, input)
        headers.update(extra_headers or {})
        sub = self._sdk._resolve_subscription(lago_opts.get("subscription") or self._base_sub)
        if self._gateway_id and sub and "cf-aig-metadata" not in headers:
            # Attribution travels with the call: the gateway stores this on the log entry, so
            # the Logs API backfill resolves the same subscription this emit() uses. Honoured on
            # both routes (measured on the unified path: the log row carried it).
            headers["cf-aig-metadata"] = json.dumps({"lago_subscription": sub})

        resp = self._session.post(self.url_for(model), headers=headers, json=body, timeout=self._timeout)
        try:
            payload: dict[str, Any] = resp.json()
        except ValueError:
            payload = {}
        if resp.status_code >= 400 or payload.get("success") is False:
            raise WorkersAIError(resp.status_code, list(payload.get("errors") or []), payload)

        try:
            if resp.headers.get("cf-aig-cache-status") == "HIT":
                # The gateway answered from its cache; the model never ran and Cloudflare
                # billed nothing. Billing it would charge for a call that did not happen.
                return payload
            usage = extract_workers_ai_native(payload, model_id=model)
            dims = {**self._base_dims, **(lago_opts.get("dimensions") or {})}
            log_id = resp.headers.get("cf-aig-log-id")
            if log_id:
                dims["cf_log_id"] = log_id
            self._sdk.emit(
                usage,
                subscription=sub,
                dimensions=dims,
                mode=lago_opts.get("mode"),
                markup=lago_opts.get("markup"),
            )
        except Exception as exc:  # noqa: BLE001 — instrumentation never breaks the customer's call
            logger.warning("lago: workers_ai.run instrumentation failed: %s", exc)
            self._sdk._report_error(exc, "emit")
        return payload

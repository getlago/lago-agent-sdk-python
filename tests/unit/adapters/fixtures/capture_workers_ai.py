"""Capture real Workers AI `/ai/run` responses for the Workers AI adapter.

Saves to `workers_ai/<scenario>.json` as `{_model_id, _status, _headers, _response}` —
successful responses only; the client's error path is unit-tested against inline bodies.

Two routes are exercised, because they differ in what reaches a partner model:

  direct   POST api.cloudflare.com/.../ai/run/{model}   (`@cf/...` ids)
           POST api.cloudflare.com/.../ai/run           body {"model", "input"} (partner ids)
           With `cf-aig-gateway-id: <gateway>` the call is routed through that gateway,
           which is what lets a partner model (`typesafe/jev`) use the partner key stored
           under the gateway's BYOK — without it the call bills gateway credits.
  gateway  POST gateway.ai.cloudflare.com/v1/.../workers-ai/{model}
           Exposes `cf-aig-cache-status` / `cf-aig-log-id` response headers.

Reads CF_ACCOUNT_ID, CF_API_TOKEN (or CF_LOGS_TOKEN), CF_GATEWAY_ID, CF_GATEWAY_AUTH and,
for scenario 8, TYPESAFE_API_KEY from env. Bodies carry no account identifiers; only the
gateway headers the client reads (`cf-aig-cache-status`) are kept.

Pass scenario numbers to recapture a subset: `capture_workers_ai.py 3 4`.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import requests

OUT = pathlib.Path(__file__).parent / "workers_ai"
OUT.mkdir(parents=True, exist_ok=True)
KEPT_HEADERS = ("cf-aig-cache-status", "content-type")
LLAMA = "@cf/meta/llama-3.2-3b-instruct"


def save(name: str, model: str, resp: requests.Response, source: str | None = None) -> None:
    payload: dict = {
        "_model_id": model,
        "_status": resp.status_code,
        "_headers": {k: v for k, v in resp.headers.items() if k.lower() in KEPT_HEADERS},
    }
    if source:
        payload["_source"] = source
    payload["_response"] = resp.json()
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  ✓ saved {name}.json  HTTP {resp.status_code}")


JEV_INPUT = {
    "state": "Hi, I was charged twice for my subscription this month and I need the duplicate refunded "
    "before my card statement closes on Friday. This is the second time this has happened.",
    "questions": {
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this convey urgency?",
            "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
        },
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing questions, upgrades",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}
CHAT_INPUT = {
    "messages": [{"role": "user", "content": "Write one sentence about dolphins."}],
    "max_tokens": 40,
}


def main() -> int:
    only = {int(a) for a in sys.argv[1:] if a.isdigit()}

    def want(n: int) -> bool:
        return not only or n in only

    acct = os.environ.get("CF_ACCOUNT_ID")
    token = os.environ.get("CF_API_TOKEN") or os.environ.get("CF_LOGS_TOKEN")
    gw, gw_auth = os.environ.get("CF_GATEWAY_ID"), os.environ.get("CF_GATEWAY_AUTH")
    if not (acct and token):
        print("error: set CF_ACCOUNT_ID and CF_API_TOKEN", file=sys.stderr)
        return 2
    direct = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/run"
    hdr = {"Authorization": f"Bearer {token}"}

    def post(url: str, body: dict, extra: dict | None = None) -> requests.Response:
        return requests.post(url, headers={**hdr, **(extra or {})}, json=body, timeout=90)

    if want(1):
        print("\n[1] chat model, direct API, path route")
        save("01_chat_direct", LLAMA, post(f"{direct}/{LLAMA}", CHAT_INPUT))

    if want(2):
        print("\n[2] reasoning model, direct API — does usage break reasoning out?")
        m = "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"
        save("02_reasoning_direct", m, post(f"{direct}/{m}", {**CHAT_INPUT, "max_tokens": 60}))

    if want(3):
        print("\n[3] OpenAI-vendor model on Workers AI, direct API")
        m = "@cf/openai/gpt-oss-120b"
        save("03_gpt_oss_direct", m, post(f"{direct}/{m}", CHAT_INPUT))

    if want(4):
        print("\n[4] Mistral-vendor model on Workers AI, direct API")
        m = "@cf/mistralai/mistral-small-3.1-24b-instruct"
        save("04_mistral_small_direct", m, post(f"{direct}/{m}", CHAT_INPUT))

    if want(5):
        print("\n[5] partner model typesafe/jev, direct API routed through the BYOK gateway")
        extra = {"cf-aig-gateway-id": gw} if gw else None
        save(
            "05_jev_byok_gateway",
            "typesafe/jev",
            post(direct, {"model": "typesafe/jev", "input": JEV_INPUT}, extra),
        )

    if gw and gw_auth and (want(6) or want(7)):
        gwbase = f"https://gateway.ai.cloudflare.com/v1/{acct}/{gw}/workers-ai"
        cache = {"cf-aig-authorization": f"Bearer {gw_auth}", "cf-aig-cache-ttl": "300"}
        print("\n[6] chat model via gateway host (cache MISS), cache enabled for the pair")
        save("06_chat_gateway_miss", LLAMA, post(f"{gwbase}/{LLAMA}", CHAT_INPUT, cache))
        time.sleep(2)
        print("\n[7] identical call again — gateway cache HIT")
        save("07_chat_gateway_hit", LLAMA, post(f"{gwbase}/{LLAMA}", CHAT_INPUT, cache))

    if want(8) and os.environ.get("TYPESAFE_API_KEY"):
        # The model's response body from its own API, for comparison with what Cloudflare
        # wraps in `result.result` on scenario 5.
        print("\n[8] typesafe/jev from TypeSafe's own API (not Cloudflare)")
        r = requests.post(
            "https://api.typesafe.ai/v1/systemone",
            headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"},
            json={"model": "jev-latest", **JEV_INPUT},
            timeout=90,
        )
        save(
            "08_jev_typesafe_direct",
            "typesafe/jev",
            r,
            source="POST https://api.typesafe.ai/v1/systemone (model=jev-latest)",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

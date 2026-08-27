"""Capture Snowflake Cortex responses off the OpenAI-compatible endpoint.

Saves to:
  tests/unit/adapters/fixtures/openai_native/11_snowflake_cortex_plain_chat.json
  tests/unit/adapters/fixtures/openai_native/12_snowflake_cortex_cache_chat.json

These live under `openai_native/` on purpose: Cortex answers on an OpenAI-wire
endpoint, so `extract_openai_native` / `extractOpenAINative` is the adapter that
serves them. They are the surface that proves the `total_tokens` reconciliation
cannot assume OpenAI's subtractive cache convention — on Cortex, `cached_tokens`
sits OUTSIDE `prompt_tokens` and INSIDE `total_tokens`.

Three things about Cortex that this script encodes, all measured 2026-08-25:

  * Caching only happens with an explicit Anthropic-style `cache_control` part.
    The same 4,800-token prompt sent twice WITHOUT it reports `cached_tokens: 0`
    both times, so the "call1 then call2" pattern the OpenAI cache fixtures use
    captures nothing here.
  * One cold call is enough for fixture 12: unlike Anthropic's own wire, Cortex
    reports a cache CREATION under `cached_tokens` too (`cache_write_tokens`
    stays 0), so the first `cache_control` call already carries the cached
    block. Measured on a matched pair — both calls returned identical usage
    while the account-usage view logged one `cache_write_input` row and one
    `cache_read_input` row. No warm-up call, no 5-minute-TTL race on recapture.
  * `max_tokens` is rejected outright ("deprecated in favor of
    max_completion_tokens"), unlike OpenAI which still accepts it.

Requires a Snowflake account with the Cortex REST endpoint entitled — it returns
403 `003001` otherwise, which is an account-level grant no config can work around.

  SNOWFLAKE_HOST=<org>-<account>.snowflakecomputing.com \
  SNOWFLAKE_PAT=<programmatic access token> \
  python3 capture_snowflake_cortex.py

Idempotent: skips files that already exist. Re-run after deleting one to refresh it.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import requests

MODEL = "claude-sonnet-4-5"
OUT = pathlib.Path(__file__).parent / "openai_native"

# Long enough to clear Anthropic's minimum cacheable prefix. Deliberately dull,
# fixed text: the fixture must not carry anything account- or person-identifying.
CACHEABLE_PREFIX = "Reference notes on widget calibration tolerances, revision seven. " * 400


def call(host: str, pat: str, body: dict) -> dict:
    r = requests.post(
        f"https://{host}/api/v2/cortex/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {pat}",
            "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=body,
        timeout=120,
    )
    if r.status_code != 200:
        sys.exit(f"Cortex returned {r.status_code}: {r.text[:300]}")
    return r.json()


def save(name: str, body: dict, host: str, pat: str) -> None:
    # The existence check runs BEFORE the request: `save(..., call(...))` would
    # evaluate the call first and fire two live Cortex requests — the 4,800-token
    # cacheable one included — on a checkout where both fixtures already exist,
    # then print "skip". Idempotent means no request, not just no write.
    path = OUT / name
    if path.exists():
        print(f"skip {name} (exists)")
        return
    response = call(host, pat, body)
    path.write_text(json.dumps({"_model_id": MODEL, "_response": response}, indent=2) + "\n")
    print(f"wrote {name}")


def main() -> None:
    host = os.environ.get("SNOWFLAKE_HOST")
    pat = os.environ.get("SNOWFLAKE_PAT")
    if not host or not pat:
        sys.exit("set SNOWFLAKE_HOST and SNOWFLAKE_PAT")

    save(
        "11_snowflake_cortex_plain_chat.json",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is 2 + 2? Answer in one word."}],
            "max_completion_tokens": 32,
        },
        host,
        pat,
    )

    # The regression fixture. `cache_control` is what makes Cortex report a cached
    # block at all, and the resulting payload is the one that used to inflate
    # `output` by the whole cached count. One call suffices — see the docstring's
    # creation-reports-as-cached_tokens note.
    save(
        "12_snowflake_cortex_cache_chat.json",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": CACHEABLE_PREFIX,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": "Reply with one word."},
                    ],
                }
            ],
            "max_completion_tokens": 32,
        },
        host,
        pat,
    )


if __name__ == "__main__":
    main()

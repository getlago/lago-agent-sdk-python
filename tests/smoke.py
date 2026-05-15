"""Smoke test for the Lago Agent SDK against AWS Bedrock.

Reads credentials from .env (or the surrounding shell), wraps a Bedrock
client, makes one converse() call, and flushes the usage event to Lago.

This is NOT a pytest test — it costs real money (one Bedrock call + one
Lago event). The filename intentionally avoids pytest's `test_*.py` /
`*_test.py` discovery patterns so it's never auto-collected.

Run:
    uv run python tests/smoke.py

Required env vars:
    LAGO_API_KEY
    LAGO_EXTERNAL_SUBSCRIPTION_ID
    AWS_BEARER_TOKEN_BEDROCK
"""

from __future__ import annotations

import os
import pathlib
import sys

import boto3

from lago_agent_sdk import LagoSDK

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_dotenv(path: pathlib.Path | None = None) -> None:
    """Tiny no-deps .env loader. Skips comments, blanks, and already-set vars."""
    p = path or (_REPO_ROOT / ".env")
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().lstrip("export ").strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def main() -> int:
    _load_dotenv()

    try:
        api_key = os.environ["LAGO_API_KEY"]
        subscription = os.environ["LAGO_EXTERNAL_SUBSCRIPTION_ID"]
    except KeyError as missing:
        print(f"error: {missing} is required (set in .env or shell)", file=sys.stderr)
        return 1

    api_url = os.environ.get("LAGO_API_URL", "https://api.getlago.com/api/v1/")
    region = os.environ.get("BEDROCK_REGION", "eu-west-1")
    model_id = os.environ.get("BEDROCK_MODEL_ID", "eu.amazon.nova-lite-v1:0")

    sdk = LagoSDK(api_key=api_key, api_url=api_url)
    bedrock = sdk.wrap(boto3.client("bedrock-runtime", region_name=region))

    resp = bedrock.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": "what is the capital of France?"}]}],
        extra_lago={"subscription": subscription},
    )

    print(resp["output"]["message"]["content"][0]["text"])
    print("\ntokens:", resp["usage"])

    # Ship the usage event before exit — atexit would handle it too, but
    # explicit makes failures (network errors, etc.) visible synchronously.
    if not sdk.flush(timeout=10.0):
        print("warning: lago queue did not drain within 10s", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

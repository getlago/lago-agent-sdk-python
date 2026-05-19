"""Capture every callable Bedrock model — both Converse and InvokeModel.

Saves to:
  shared/fixtures/bedrock/converse/{callable_id}.json
  shared/fixtures/bedrock/invoke/{callable_id}.json

Idempotent: skips files that already exist. Re-run to add newly-released models.

Read by tests/unit/adapters/test_bedrock_*.py and test_all_models_sweep.py —
proves the dispatch picks the right adapter for every model and the adapter
handles its real shape.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import requests

REGION = "eu-west-1"
PROMPT = "One sentence about dolphins."
MAX_TOKENS = 40


def headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def discover(api_key: str) -> tuple[list[dict], dict[str, str]]:
    r = requests.get(
        f"https://bedrock.{REGION}.amazonaws.com/foundation-models", headers=headers(api_key), timeout=30
    )
    r.raise_for_status()
    models = r.json().get("modelSummaries", [])
    r = requests.get(
        f"https://bedrock.{REGION}.amazonaws.com/inference-profiles", headers=headers(api_key), timeout=30
    )
    r.raise_for_status()
    profiles = r.json().get("inferenceProfileSummaries", [])
    profile_by_arn: dict[str, str] = {}
    for p in profiles:
        pid = p.get("inferenceProfileId", "")
        for m in p.get("models", []):
            arn = m.get("modelArn")
            if arn and (arn not in profile_by_arn or pid.startswith("eu.")):
                profile_by_arn[arn] = pid
    return models, profile_by_arn


def callable_for(model: dict, profile_by_arn: dict[str, str]) -> str | None:
    inf = set(model.get("inferenceTypesSupported", []))
    if model.get("modelLifecycle", {}).get("status") == "LEGACY":
        return None
    if "TEXT" not in set(model.get("inputModalities", [])):
        return None
    if "TEXT" not in set(model.get("outputModalities", [])):
        return None
    mid = model.get("modelId", "")
    if "ON_DEMAND" in inf:
        return mid
    if "INFERENCE_PROFILE" in inf:
        return profile_by_arn.get(model.get("modelArn"))
    return None


def make_invoke_body(mid: str, prompt: str, max_tokens: int) -> dict | None:
    m = mid.lower()
    if "anthropic" in m:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
    if "nova" in m:
        return {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
    if "pixtral" in m:
        return {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    if "mistral-large-2402" in m or "mistral-7b" in m or "mixtral-8x7b" in m:
        return {"prompt": f"<s>[INST] {prompt} [/INST]", "max_tokens": max_tokens}
    if "mistral" in m or "mixtral" in m:
        return {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    return {"messages": [{"role": "user", "content": prompt}], "max_completion_tokens": max_tokens}


def safe_filename(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def main() -> int:
    api_key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if not api_key:
        print("error: set AWS_BEARER_TOKEN_BEDROCK", file=sys.stderr)
        return 2

    out_root = pathlib.Path(__file__).parent / "bedrock"
    (out_root / "converse").mkdir(parents=True, exist_ok=True)
    (out_root / "invoke").mkdir(parents=True, exist_ok=True)

    print("Discovering...")
    models, profile_by_arn = discover(api_key)
    callable_ids: list[str] = []
    for m in models:
        cid = callable_for(m, profile_by_arn)
        if cid:
            callable_ids.append(cid)
    callable_ids = sorted(set(callable_ids))
    print(f"  {len(callable_ids)} callable model ids")

    print("\nConverse:")
    for i, cid in enumerate(callable_ids, 1):
        out = out_root / "converse" / f"{safe_filename(cid)}.json"
        if out.exists():
            print(f"  [{i}/{len(callable_ids)}] {cid}  (cached)")
            continue
        url = f"https://bedrock-runtime.{REGION}.amazonaws.com/model/{cid}/converse"
        body = {
            "messages": [{"role": "user", "content": [{"text": PROMPT}]}],
            "inferenceConfig": {"maxTokens": MAX_TOKENS},
        }
        try:
            r = requests.post(url, headers=headers(api_key), json=body, timeout=60)
            if r.status_code == 200:
                out.write_text(json.dumps({"_model_id": cid, "_response": r.json()}, indent=2))
                print(f"  [{i}/{len(callable_ids)}] {cid}  ✓")
            else:
                print(f"  [{i}/{len(callable_ids)}] {cid}  HTTP {r.status_code}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(callable_ids)}] {cid}  error: {exc}")
        time.sleep(0.2)

    print("\nInvokeModel:")
    for i, cid in enumerate(callable_ids, 1):
        out = out_root / "invoke" / f"{safe_filename(cid)}.json"
        if out.exists():
            print(f"  [{i}/{len(callable_ids)}] {cid}  (cached)")
            continue
        body = make_invoke_body(cid, PROMPT, MAX_TOKENS)
        if body is None:
            continue
        url = f"https://bedrock-runtime.{REGION}.amazonaws.com/model/{cid}/invoke"
        try:
            r = requests.post(url, headers=headers(api_key), json=body, timeout=60)
            if r.status_code == 200:
                out.write_text(json.dumps({"_model_id": cid, "_response": r.json()}, indent=2))
                print(f"  [{i}/{len(callable_ids)}] {cid}  ✓")
            else:
                print(f"  [{i}/{len(callable_ids)}] {cid}  HTTP {r.status_code}: {r.text[:80]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(callable_ids)}] {cid}  error: {exc}")
        time.sleep(0.2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

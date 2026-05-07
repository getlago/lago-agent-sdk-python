"""Capture real Mistral API responses for adapter design.

Saves raw responses to shared/fixtures/mistral_native/<scenario>.json so we can
verify spec §4.3 mappings against reality before writing the adapter.

Reads MISTRAL_API_KEY from env.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

from mistralai.client import Mistral

OUT = pathlib.Path(__file__).parent / "mistral_native"
OUT.mkdir(parents=True, exist_ok=True)


def to_dict(response) -> dict:
    """Mistral SDK returns pydantic models — convert to plain dict for JSON."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return json.loads(response.json()) if hasattr(response, "json") else dict(response)


def save(name: str, model: str, payload: dict) -> None:
    path = OUT / f"{name}.json"
    path.write_text(json.dumps({"_model_id": model, "_response": payload}, indent=2, default=str))
    print(f"  ✓ saved {path.name}")


def main() -> int:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        print("error: set MISTRAL_API_KEY", file=sys.stderr)
        return 2

    client = Mistral(api_key=key)

    PROMPT = "Write one sentence about dolphins."

    # ----- 1. Plain call (small model) -----
    print("\n[1] plain — mistral-small-latest")
    r = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=80,
    )
    save("01_plain_small", "mistral-small-latest", to_dict(r))

    # ----- 2. Plain call on Mistral Large (for cache eligibility) -----
    print("\n[2] plain — mistral-large-latest")
    r = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=80,
    )
    save("02_plain_large", "mistral-large-latest", to_dict(r))

    # ----- 3. Tool use -----
    print("\n[3] tool use — mistral-small-latest with weather tool")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    r = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=tools,
        tool_choice="any",
        max_tokens=200,
    )
    save("03_tool_use", "mistral-small-latest", to_dict(r))

    # ----- 4. Reasoning (Magistral) -----
    print("\n[4] reasoning — magistral-small-latest")
    try:
        r = client.chat.complete(
            model="magistral-small-latest",
            messages=[{"role": "user", "content": "Solve: a train leaves Paris at 9am at 80km/h, another at 11am at 120km/h. When does the second catch up?"}],
            max_tokens=600,
        )
        save("04_reasoning_magistral", "magistral-small-latest", to_dict(r))
    except Exception as exc:  # noqa: BLE001
        print(f"  magistral-small-latest error: {exc}")

    # ----- 5. Streaming (final chunk should carry usage) -----
    print("\n[5] streaming — mistral-small-latest")
    chunks: list[dict] = []
    stream = client.chat.stream(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=60,
    )
    for event in stream:
        chunks.append(to_dict(event))
    save("05_stream", "mistral-small-latest", {"chunks": chunks})

    # ----- 6. Multi-turn -----
    print("\n[6] multi-turn — mistral-small-latest (3 turns)")
    convo = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "And times 3?"},
    ]
    r = client.chat.complete(model="mistral-small-latest", messages=convo, max_tokens=40)
    save("06_multi_turn", "mistral-small-latest", to_dict(r))

    # ----- 7. Cache hit attempt — repeat the same large prompt twice -----
    print("\n[7] cache attempt — mistral-large-latest (long prompt, twice)")
    long_prompt = ("You are a helpful assistant. " * 200) + "What is 1+1?"
    msg = [{"role": "user", "content": long_prompt}]
    r1 = client.chat.complete(model="mistral-large-latest", messages=msg, max_tokens=20)
    save("07_cache_attempt_call1", "mistral-large-latest", to_dict(r1))
    r2 = client.chat.complete(model="mistral-large-latest", messages=msg, max_tokens=20)
    save("07_cache_attempt_call2", "mistral-large-latest", to_dict(r2))

    print("\nDone. Inspect shared/fixtures/mistral_native/*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

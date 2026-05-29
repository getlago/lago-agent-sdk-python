"""Capture real OpenAI API responses for adapter design.

Saves raw responses to tests/unit/adapters/fixtures/openai_native/<scenario>.json
so we can verify the field mappings against reality before writing the adapter.

Covers both Chat Completions (`client.chat.completions.create`) and
the Responses API (`client.responses.create`) — they have different
usage shapes.

Reads OPENAI_API_KEY from env.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

from openai import OpenAI

OUT = pathlib.Path(__file__).parent / "openai_native"
OUT.mkdir(parents=True, exist_ok=True)


def to_dict(response) -> dict:
    """OpenAI SDK returns pydantic models — convert to plain dict for JSON."""
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
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("error: set OPENAI_API_KEY", file=sys.stderr)
        return 2

    client = OpenAI(api_key=key)
    PROMPT = "Write one sentence about dolphins."

    # =================================================================
    # Chat Completions API — client.chat.completions.create(...)
    # =================================================================

    # ----- 1. Plain chat completion -----
    print("\n[1] plain chat — gpt-4o-mini")
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT}],
        max_completion_tokens=80,
    )
    save("01_plain_chat", "gpt-4o-mini", to_dict(r))

    # ----- 2. Tool use (function calling) -----
    print("\n[2] tool use chat — gpt-4o-mini with weather tool")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        max_completion_tokens=200,
    )
    save("02_tool_use_chat", "gpt-4o-mini", to_dict(r))

    # ----- 3. Cache hit attempt — long prompt sent twice (OpenAI auto-caches >1024 tokens) -----
    print("\n[3] cache attempt — long prompt, call 1 then call 2")
    long_prompt = (
        "You are an extremely thorough expert tutor. Answer concisely and cite reasoning step by step. "
        * 200
    )
    msgs = [
        {"role": "system", "content": long_prompt},
        {"role": "user", "content": "What is 2+2?"},
    ]
    r1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        max_completion_tokens=20,
    )
    save("03_cache_call1_chat", "gpt-4o-mini", to_dict(r1))

    msgs2 = [
        {"role": "system", "content": long_prompt},
        {"role": "user", "content": "What is 3+3?"},
    ]
    r2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs2,
        max_completion_tokens=20,
    )
    save("04_cache_call2_chat", "gpt-4o-mini", to_dict(r2))

    # ----- 5. Streaming with usage included -----
    print("\n[5] streaming chat — gpt-4o-mini with stream_options.include_usage")
    chunks: list[dict] = []
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT}],
        max_completion_tokens=60,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        chunks.append(to_dict(chunk))
    save("05_streaming_chat", "gpt-4o-mini", {"chunks": chunks})

    # ----- 6. Reasoning model (o-series) — exposes reasoning_tokens -----
    print("\n[6] reasoning chat — o4-mini")
    try:
        r = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Prove that the sum of the first n cubes equals the square of the sum "
                        "of the first n positive integers. Show each step."
                    ),
                }
            ],
            max_completion_tokens=2000,
        )
        save("06_reasoning_chat", "o4-mini", to_dict(r))
    except Exception as exc:  # noqa: BLE001
        print(f"  o4-mini error: {str(exc)[:160]}")

    # ----- 7. Multi-turn -----
    print("\n[7] multi-turn chat — gpt-4o-mini")
    convo = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "And times 3?"},
    ]
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=convo,
        max_completion_tokens=40,
    )
    save("07_multi_turn_chat", "gpt-4o-mini", to_dict(r))

    # =================================================================
    # Responses API — client.responses.create(...)
    # =================================================================

    # ----- 8. Plain Responses API call -----
    print("\n[8] plain responses — gpt-4o-mini")
    try:
        r = client.responses.create(
            model="gpt-4o-mini",
            input=PROMPT,
            max_output_tokens=80,
        )
        save("08_plain_responses", "gpt-4o-mini", to_dict(r))
    except Exception as exc:  # noqa: BLE001
        print(f"  responses.create error: {str(exc)[:160]}")

    # ----- 9. Responses API with tool use -----
    print("\n[9] tool use responses — gpt-4o-mini")
    try:
        r = client.responses.create(
            model="gpt-4o-mini",
            input="What's the weather in Tokyo?",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
            tool_choice="required",
            max_output_tokens=200,
        )
        save("09_tool_use_responses", "gpt-4o-mini", to_dict(r))
    except Exception as exc:  # noqa: BLE001
        print(f"  responses tool use error: {str(exc)[:160]}")

    # ----- 10. Reasoning via Responses API -----
    print("\n[10] reasoning responses — o4-mini")
    try:
        r = client.responses.create(
            model="o4-mini",
            input=(
                "Prove that the sum of the first n cubes equals the square of the sum "
                "of the first n positive integers. Show each step."
            ),
            reasoning={"effort": "low"},
            max_output_tokens=2000,
        )
        save("10_reasoning_responses", "o4-mini", to_dict(r))
    except Exception as exc:  # noqa: BLE001
        print(f"  responses reasoning error: {str(exc)[:160]}")

    print("\nDone. Inspect tests/unit/adapters/fixtures/openai_native/*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

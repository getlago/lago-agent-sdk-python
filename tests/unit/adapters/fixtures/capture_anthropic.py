"""Capture real Anthropic native API responses for adapter design.

Saves raw responses to tests/unit/adapters/fixtures/anthropic_native/<scenario>.json
so we can verify mappings against reality before writing the adapter.

Reads ANTHROPIC_API_KEY from env.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

from anthropic import Anthropic

OUT = pathlib.Path(__file__).parent / "anthropic_native"
OUT.mkdir(parents=True, exist_ok=True)


def to_dict(response) -> dict:
    """Anthropic SDK returns pydantic models — convert to plain dict for JSON."""
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
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("error: set ANTHROPIC_API_KEY", file=sys.stderr)
        return 2

    client = Anthropic(api_key=key)
    PROMPT = "Write one sentence about dolphins."

    # Rename badge: the script header reads "Sonnet 4.5" but the API only exposes 4-6+ now.
    # ----- 1. Plain call (small model) -----
    print("\n[1] plain — claude-haiku-4-5-20251001")
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=80,
        messages=[{"role": "user", "content": PROMPT}],
    )
    save("01_plain_haiku", "claude-haiku-4-5-20251001", to_dict(r))

    # ----- 2. Plain call (Sonnet, larger) -----
    print("\n[2] plain — claude-sonnet-4-6")
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=80,
        messages=[{"role": "user", "content": PROMPT}],
    )
    save("02_plain_sonnet", "claude-sonnet-4-6", to_dict(r))

    # ----- 3. Tool use -----
    print("\n[3] tool use — claude-sonnet-4-6 with weather tool")
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        tools=tools,
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    )
    save("03_tool_use", "claude-sonnet-4-6", to_dict(r))

    # ----- 4. Cache create (5m default TTL) — long system prompt -----
    print("\n[4] cache create — long system + cache_control 5m default")
    LONG_TEXT = ("You are a helpful assistant. Answer concisely. " * 200) + (
        "Always cite step by step. " * 100
    )
    cached_body = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 30,
        "system": [{"type": "text", "text": LONG_TEXT, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "What's 2+2?"}],
    }
    r = client.messages.create(**cached_body)
    save("04_cache_create_5m", "claude-sonnet-4-6", to_dict(r))

    # ----- 5. Cache read (same long system, different user question) -----
    print("\n[5] cache read — same cached_control content, second call")
    cached_body["messages"] = [{"role": "user", "content": "What's 3+3?"}]
    r = client.messages.create(**cached_body)
    save("05_cache_read", "claude-sonnet-4-6", to_dict(r))

    # ----- 6. Cache 1h TTL -----
    print("\n[6] cache 1h — explicit ttl")
    cached_1h = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 30,
        "system": [
            {
                "type": "text",
                "text": LONG_TEXT + " (1h variant)",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        "messages": [{"role": "user", "content": "Hi"}],
    }
    try:
        r = client.messages.create(**cached_1h)
        save("06_cache_create_1h", "claude-sonnet-4-6", to_dict(r))
    except Exception as exc:  # noqa: BLE001
        print(f"  1h TTL not available on this account/region: {str(exc)[:160]}")

    # ----- 7. Extended thinking (reasoning) -----
    print("\n[7] extended thinking — claude-sonnet-4-6")
    try:
        r = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Prove that the sum of the first n cubes equals the square of the sum of "
                        "the first n positive integers. Show each algebraic step."
                    ),
                }
            ],
        )
        save("07_extended_thinking", "claude-sonnet-4-6", to_dict(r))
    except Exception as exc:  # noqa: BLE001
        print(f"  extended thinking error: {str(exc)[:160]}")

    # ----- 8. Streaming -----
    print("\n[8] streaming — claude-haiku-4-5-20251001")
    events: list[dict] = []
    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=60,
        messages=[{"role": "user", "content": PROMPT}],
    ) as stream:
        for event in stream:
            events.append(to_dict(event))
    save("08_stream", "claude-haiku-4-5-20251001", {"events": events})

    # ----- 9. Multi-turn -----
    print("\n[9] multi-turn — claude-haiku-4-5-20251001")
    convo = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "And times 3?"},
    ]
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=40,
        messages=convo,
    )
    save("09_multi_turn", "claude-haiku-4-5-20251001", to_dict(r))

    print("\nDone. Inspect tests/unit/adapters/fixtures/anthropic_native/*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

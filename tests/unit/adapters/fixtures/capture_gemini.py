"""Capture real Gemini API responses for adapter design.

Saves raw responses to tests/unit/adapters/fixtures/gemini_native/<scenario>.json
so we can verify the field mappings against reality before writing the adapter.

Uses the modern `google-genai` SDK: `from google import genai`.

Reads GEMINI_API_KEY from env.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

from google import genai
from google.genai import types

OUT = pathlib.Path(__file__).parent / "gemini_native"
OUT.mkdir(parents=True, exist_ok=True)


def to_dict(response) -> dict:
    """google-genai SDK returns pydantic models — convert to plain dict for JSON."""
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
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("error: set GEMINI_API_KEY", file=sys.stderr)
        return 2

    client = genai.Client(api_key=key)
    PROMPT = "Write one sentence about dolphins."

    # ----- 1. Plain call (cheap flash model) -----
    print("\n[1] plain — gemini-2.5-flash")
    r = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=PROMPT,
    )
    save("01_plain_flash", "gemini-2.5-flash", to_dict(r))

    # ----- 2. Tool use (function calling) -----
    print("\n[2] tool use — gemini-2.5-flash with weather function")
    weather_fn = types.FunctionDeclaration(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters=types.Schema(
            type="OBJECT",
            properties={"city": types.Schema(type="STRING")},
            required=["city"],
        ),
    )
    r = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="What's the weather in Tokyo?",
        config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[weather_fn])],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY"),
            ),
        ),
    )
    save("02_tool_use", "gemini-2.5-flash", to_dict(r))

    # ----- 3. Streaming with usage metadata -----
    print("\n[3] streaming — gemini-2.5-flash")
    chunks: list[dict] = []
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents="Count from 1 to 5, one number per line.",
    ):
        chunks.append(to_dict(chunk))
    save("03_streaming", "gemini-2.5-flash", {"chunks": chunks})

    # ----- 4. Thinking mode (Gemini 2.5 — emits thoughts_token_count) -----
    print("\n[4] thinking — gemini-2.5-flash with thinking_config")
    try:
        r = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                "Prove that the sum of the first n cubes equals the square of "
                "the sum of the first n positive integers. Show each step."
            ),
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=2048),
            ),
        )
        save("04_thinking", "gemini-2.5-flash", to_dict(r))
    except Exception as exc:  # noqa: BLE001
        print(f"  thinking config error: {str(exc)[:160]}")

    # ----- 5. Multi-turn -----
    print("\n[5] multi-turn — gemini-2.5-flash (3 turns)")
    convo = [
        types.Content(role="user", parts=[types.Part(text="What is 2+2?")]),
        types.Content(role="model", parts=[types.Part(text="2+2 equals 4.")]),
        types.Content(role="user", parts=[types.Part(text="And times 3?")]),
    ]
    r = client.models.generate_content(model="gemini-2.5-flash", contents=convo)
    save("05_multi_turn", "gemini-2.5-flash", to_dict(r))

    # ----- 6. Explicit cache (Gemini's CachedContent API) -----
    # Note: requires a sufficiently large prompt (>32k tokens for flash) so we skip
    # for the demo; documented but not part of the captured fixture set.
    print("\n[6] (explicit-cache fixture skipped — needs >32k-token prompt)")

    # ----- 7. Larger model for cross-shape comparison -----
    print("\n[7] plain — gemini-2.5-pro")
    try:
        r = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=PROMPT,
        )
        save("07_plain_pro", "gemini-2.5-pro", to_dict(r))
    except Exception as exc:  # noqa: BLE001
        print(f"  gemini-2.5-pro error: {str(exc)[:160]}")

    print("\nDone. Inspect tests/unit/adapters/fixtures/gemini_native/*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

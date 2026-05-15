"""Sweep every chat-capable Mistral model — same approach as Bedrock all-models capture.

For each model:
  - Plain text chat → save to mistral_native/all_models/<id>.json
  - If vision-capable, also try with a small image → save to <id>__vision.json

Skipped: pure embedding / OCR / audio-transcription / moderation models
(they don't accept chat.complete and use different endpoints).

Reads MISTRAL_API_KEY from env.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

from mistralai.client import Mistral

OUT = pathlib.Path(__file__).parent / "mistral_native" / "all_models"
OUT.mkdir(parents=True, exist_ok=True)

# Tiny 1×1 transparent PNG, base64-encoded — minimal valid image for vision tests.
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFhAJ/wlseKgAAAABJRU5ErkJggg=="
)


def safe_filename(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def to_dict(response) -> dict:
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:  # noqa: BLE001
            pass
    return {} if response is None else dict(response)


def main() -> int:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        print("error: set MISTRAL_API_KEY", file=sys.stderr)
        return 2

    client = Mistral(api_key=key)
    print("Listing models...")
    models = client.models.list().model_dump().get("data", [])
    print(f"  {len(models)} total")

    # Filter to chat-capable, skip those whose only capability is non-chat
    chat_models = [m for m in models if (m.get("capabilities") or {}).get("completion_chat")]
    print(f"  {len(chat_models)} chat-capable")

    # Dedupe by id but keep aliases visible
    seen: set[str] = set()
    queue = []
    for m in chat_models:
        mid = m.get("id")
        if mid and mid not in seen:
            seen.add(mid)
            queue.append(m)

    summary = {"text": 0, "text_failed": 0, "vision": 0, "vision_failed": 0}
    for i, m in enumerate(queue, 1):
        mid = m["id"]
        cap = m.get("capabilities") or {}
        path = OUT / f"{safe_filename(mid)}.json"
        if not path.exists():
            try:
                r = client.chat.complete(
                    model=mid,
                    messages=[{"role": "user", "content": "Write one sentence about dolphins."}],
                    max_tokens=40,
                )
                payload = to_dict(r)
                path.write_text(
                    json.dumps(
                        {"_model_id": mid, "_capabilities": cap, "_response": payload}, indent=2, default=str
                    )
                )
                summary["text"] += 1
                print(f"  [{i}/{len(queue)}] {mid}  ✓")
            except Exception as exc:  # noqa: BLE001
                summary["text_failed"] += 1
                print(f"  [{i}/{len(queue)}] {mid}  ✗ {str(exc)[:120]}")
        else:
            print(f"  [{i}/{len(queue)}] {mid}  (cached)")

        if cap.get("vision"):
            vpath = OUT / f"{safe_filename(mid)}__vision.json"
            if not vpath.exists():
                try:
                    r = client.chat.complete(
                        model=mid,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "What is in this image?"},
                                    {
                                        "type": "image_url",
                                        "image_url": f"data:image/png;base64,{TINY_PNG_B64}",
                                    },
                                ],
                            }
                        ],
                        max_tokens=40,
                    )
                    payload = to_dict(r)
                    vpath.write_text(
                        json.dumps(
                            {"_model_id": mid, "_capabilities": cap, "_kind": "vision", "_response": payload},
                            indent=2,
                            default=str,
                        )
                    )
                    summary["vision"] += 1
                    print("           └─ vision  ✓")
                except Exception as exc:  # noqa: BLE001
                    summary["vision_failed"] += 1
                    print(f"           └─ vision  ✗ {str(exc)[:100]}")
            else:
                print("           └─ vision  (cached)")

        time.sleep(0.15)

    print(f"\nSummary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

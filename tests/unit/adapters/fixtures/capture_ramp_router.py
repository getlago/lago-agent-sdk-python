"""Capture real Ramp Router responses, scrubbing every capture as it is written.

Router is an OpenAI-Responses-compatible gateway in front of OpenAI, Anthropic, Google
Vertex, Fireworks and xAI. Its docs describe the request surface but not the
billing-relevant response behaviour, so this script exists to MEASURE the seven
questions the adapter's mapping depends on rather than reason about them:

  P1  GET /v1/models  — does the catalog publish per-model PRICING? (Answered 2026-08-28:
                        yes, `router.pricing` — but every observed input rate is empty.)
  P2  buffered call   — does the Response's `model` report the requested alias or the
                        served `provider:provider-model[:service-tier]`?
  P3  models fallback — under a candidate list, does `model` name the SERVED candidate?
                        Candidates MUST be the catalog's `provider:model` ids — bare
                        display names 400 (measured; the error fixture shows it).
  P4  stream: true    — do the SSE events carry usage and the resolved model where
                        `_extract_stream_usage` already looks (`.response.usage`)?
  P5  prompt cache    — the money question, in two probes: `prompt_cache_key` alone
                        does NOT warm a cache (05/06, cached_tokens 0 both calls);
                        an explicit `cache_control` content part DOES, and the warm
                        call answered it (05b/06b): cached INSIDE input — Router
                        normalizes the NUMBERS to OpenAI semantics.
  P6  reasoning model — is `reasoning_tokens` inside `output_tokens`? (Yes — measured
                        with o4-mini at effort medium; a too-small call reports 0.)
  P7  error families  — the envelope shape, and that a failure carries no usage.

Raw `urllib`, not the OpenAI SDK, deliberately: this captures the wire JSON AND the
response headers, and no doc says whether Router signals service tier, cache state or
BYOK in a header. If it does, that is a billing signal we would otherwise never see.

Reads RAMP_ROUTER_API_KEY from the environment and nothing else. Model ids are
account-specific ("Never invent one or reuse a provider's public model name"), so they
are DISCOVERED from P1 rather than hardcoded; override with RAMP_ROUTER_MODEL,
RAMP_ROUTER_ANTHROPIC_MODEL, RAMP_ROUTER_REASONING_MODEL if the heuristics pick badly.

Run with: RAMP_ROUTER_API_KEY="..." python tests/unit/adapters/fixtures/capture_ramp_router.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

OUT = Path(__file__).parent / "ramp_router"
OUT.mkdir(exist_ok=True)

API_KEY = os.environ.get("RAMP_ROUTER_API_KEY")
if not API_KEY:
    print("RAMP_ROUTER_API_KEY is not set. Put it in a gitignored .env and export it.")
    sys.exit(1)
BASE_URL = (os.environ.get("RAMP_ROUTER_BASE_URL") or "https://api.router.com/v1").rstrip("/")

# Keep every probe to a few tokens. The whole run should cost cents, and the fixtures are
# read for their `usage` object, never for their prose.
MAX_OUTPUT_TOKENS = 16
PROMPT = "Reply with exactly one word: pong."

# ---------------------------------------------------------------------------
# Scrubbing. Runs on EVERY value before it reaches the tree, in the same step
# that writes it — there is no unscrubbed intermediate file to forget about.
#
# The bar is the one the Cloudflare fixtures set: remove credentials and anything
# account-identifying, keep opaque platform ids, timings, costs and token counts,
# because those are what the tests assert on.
# ---------------------------------------------------------------------------

#: Header names never written to a fixture, whatever their value.
_DROP_HEADERS = frozenset({"authorization", "x-api-key", "set-cookie", "cookie", "proxy-authorization"})

_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    # Router's own key shape, plus the generic provider-key shapes, in case a key is
    # ever echoed back inside an error message or a request-context field.
    (re.compile(r"\bsk-[A-Za-z0-9_-]{10,}"), "rr-test-key-REDACTED"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{10,}=*", re.I), "Bearer rr-test-key-REDACTED"),
    # Emails, except ones already in a reserved documentation domain (RFC 2606).
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@(?!example\.(?:com|org|net)\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "user@example.com",
    ),
    # Dotted quads. RFC 5737 documentation address.
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "203.0.113.10"),
    # Dashboard/console URLs carry account and key ids in the path.
    (
        re.compile(r"https://(?:app|dashboard|router)\.(?:router|ramp)\.com/[^\s\"']*"),
        "https://app.router.com/REDACTED",
    ),
]


def _scrub_string(s: str) -> str:
    """Redact one string.

    The live key is substituted by VALUE first, not only by pattern: a key whose shape
    the patterns above do not anticipate would otherwise sail through. Pattern matching
    is the backstop, not the primary defence.
    """
    out = s.replace(API_KEY, "rr-test-key-REDACTED")
    for pattern, replacement in _REDACTIONS:
        out = pattern.sub(replacement, out)
    return out


#: Keys whose value is model-generated or caller-supplied CONTENT, blanked, not shipped.
_CONTENT_KEYS = frozenset(
    {"text", "input", "instructions", "output_text", "content", "refusal", "summary_text"}
)


def _scrub(value: Any, key: str = "", parent: str = "") -> Any:
    """Deep-scrub a captured payload.

    Content keys are blanked to "" rather than deleted, so the shape a test reads is the
    shape Router really sent — deleting entries would change what the fixture proves.

    `parent` exists because a content key is only content in a content position. The
    catalog's `router.pricing.input` is a RATE, and blanking it shipped a fixture whose
    every input rate read "" — which was then taken for Router's own data and cited as
    a reason price mode could not be built. Nothing under `pricing` is ever content.
    """
    if isinstance(value, str):
        if key in _CONTENT_KEYS and parent != "pricing":
            return ""
        return _scrub_string(value)
    if isinstance(value, list):
        return [_scrub(v, key, parent) for v in value]
    if isinstance(value, dict):
        return {k: _scrub(v, k, key) for k, v in value.items() if k.lower() not in _DROP_HEADERS}
    return value


# Provenance is derived, never asserted. A fixture captured against a mock or a staging
# host must not claim to be a real one — the whole point of the `NN_real_*` naming
# convention in this repo is that it means something.
try:
    _HOST: str | None = urllib.parse.urlsplit(BASE_URL).hostname
except ValueError:
    _HOST = None
_IS_PRODUCTION = _HOST == "api.router.com"
_PROVENANCE = (
    "real capture against a live Ramp Router account"
    if _IS_PRODUCTION
    else f"NOT a production capture — taken against {_HOST}. Do not commit as NN_real_*."
)


def save(name: str, probe: str, question: str, payload: dict[str, Any]) -> None:
    """Write one fixture. The only path that touches the tree, so the only scrub site."""
    body = {
        "_probe": probe,
        "_question": question,
        "_captured": _PROVENANCE,
        "_scrubbed": 'credentials, emails, IPs, dashboard URLs; prompt and completion text blanked to ""',
        **_scrub(payload),
    }
    (OUT / name).write_text(json.dumps(body, indent=2) + "\n")
    print(f"  wrote {name}")


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def call(
    method: str, path: str, body: Any | None = None, extra_headers: dict[str, str] | None = None
) -> dict[str, Any]:
    """One request, captured whole.

    The body is parsed as JSON when it is JSON and kept as text when it is not:
    api.router.com sits behind Cloudflare bot management, and an unrecognized client
    gets an HTML challenge page instead of Router's documented error envelope. A capture
    script that assumed JSON would crash on exactly the case worth recording.
    """
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        method=method,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "lago-agent-sdk-capture/0.2.0",
            **(extra_headers or {}),
        },
        data=None if body is None else json.dumps(body).encode(),
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            status, headers, text = resp.status, dict(resp.headers), resp.read().decode()
    except urllib.error.HTTPError as err:
        status, headers, text = err.code, dict(err.headers), err.read().decode()
    header_map = {k.lower(): v for k, v in headers.items() if k.lower() not in _DROP_HEADERS}
    try:
        parsed: Any = json.loads(text)
        was_json = True
    except ValueError:
        parsed, was_json = text, False
    return {"_status": status, "_headers": header_map, "_body": parsed, "_body_was_json": was_json}


def call_stream(
    body: Any, path: str = "/responses", extra_headers: dict[str, str] | None = None
) -> dict[str, Any]:
    """A streamed request, captured as the ordered list of SSE events. Anthropic's SSE
    prefixes each `data:` line with an `event:` line; only the data lines are kept, which
    is also all the wrapper reads."""
    captured = call("POST", path, body, extra_headers=extra_headers)
    raw = captured["_body"] if isinstance(captured["_body"], str) else ""
    events: list[Any] = []
    for line in raw.split("\n"):
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except ValueError:
            events.append({"_unparseable": payload})
    return {**captured, "_body": None, "_body_was_json": False, "_events": events}


# ---------------------------------------------------------------------------
# Model discovery. Ids are account-specific, so nothing here is hardcoded.
# ---------------------------------------------------------------------------


def pick_model(models: list[dict[str, Any]], vendor_hints: list[str], name_hints: list[str]) -> str | None:
    by_vendor = [m for m in models if any(h in str(m.get("owned_by", "")).lower() for h in vendor_hints)]
    pool = by_vendor or models
    for hint in name_hints:
        hit = next((m for m in pool if hint in str(m.get("id", "")).lower()), None)
        if hit:
            return str(hit["id"])
    return str(pool[0]["id"]) if pool else None


def candidate_id(models: list[dict[str, Any]], model_id: str | None) -> str | None:
    """The catalog's `provider:model` candidate form for a model id.

    `models` entries MUST be this form — a bare display name 400s with
    "`models` entry 0 must be a provider:model string" (measured)."""
    for m in models:
        if m.get("id") == model_id:
            router = m.get("router")
            if isinstance(router, dict) and router.get("catalog_id"):
                return str(router["catalog_id"])
    return None


# ---------------------------------------------------------------------------
# P11-P15: Router's SECOND surface, `POST /v1/messages` (Anthropic-shaped), reached with an
# Anthropic client. Captured because it is the only surface that reports an
# Anthropic-served cache WRITE — `/v1/responses` has no field for it — and because nothing
# in these bodies says Router was in the path, so these fixtures are what the Anthropic
# wrapper's provider hint is tested against. Run alone with `--messages-only`.
# ---------------------------------------------------------------------------
# Required on this surface (a request without it is rejected, measured 2026-09-04).
_MESSAGES_HEADERS = {"anthropic-version": "2023-06-01"}


def _cache_prefix(n_lines: int = 220) -> str:
    """~2.6k tokens of synthetic ledger lines: above Haiku's 2,048-token minimum cacheable
    prefix, and cheap to write (a few tenths of a cent)."""
    return " ".join(
        f"Ledger entry {i}: account ACC-{(i * 7) % 9973:05d}, debit {(i * 11) % 503}.{(i * 13) % 100:02d} EUR, "
        f"memo 'batch {i // 12} settlement', region {chr(65 + (i * 5) % 26)}."
        for i in range(n_lines)
    )


def _probe_messages(cheap: str | None, anthropic: str | None) -> None:
    def messages_call(body: dict[str, Any]) -> dict[str, Any]:
        return call("POST", "/messages", body, extra_headers=_MESSAGES_HEADERS)

    user = [{"role": "user", "content": PROMPT}]
    if anthropic:
        print("[P11] /v1/messages, plain call, Anthropic-served")
        save(
            "11_real_messages_plain.json",
            "P11",
            "does the Anthropic-shaped surface report usage in Anthropic's native shape, service_tier inside usage?",
            messages_call({"model": anthropic, "max_tokens": MAX_OUTPUT_TOKENS, "messages": user}),
        )
        cached = {
            "model": anthropic,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "system": [{"type": "text", "text": _cache_prefix(), "cache_control": {"type": "ephemeral"}}],
            "messages": user,
        }
        print("[P12] /v1/messages, cache_control cold — is the WRITE count reported here?")
        save(
            "12_real_messages_cache_control_cold.json",
            "P12",
            "does a cold cache_control write report cache_creation_input_tokens, with the 5m/1h split?",
            messages_call(cached),
        )
        print("[P13] /v1/messages, cache_control warm")
        save(
            "13_real_messages_cache_control_warm.json",
            "P13",
            "warm repeat of P12 — cache_read_input_tokens beside input_tokens (additive)?",
            messages_call(cached),
        )
        print("[P14] /v1/messages, streamed")
        save(
            "14_real_messages_streamed.json",
            "P14",
            "where do usage and service_tier land across message_start / message_delta?",
            call_stream(
                {"model": anthropic, "max_tokens": MAX_OUTPUT_TOKENS, "stream": True, "messages": user},
                path="/messages",
                extra_headers=_MESSAGES_HEADERS,
            ),
        )
    if cheap:
        print("[P15] /v1/messages, OpenAI-served model")
        save(
            "15_real_messages_openai_served.json",
            "P15",
            "does a non-Anthropic vendor's usage arrive in Anthropic's additive shape on this surface?",
            messages_call({"model": cheap, "max_tokens": MAX_OUTPUT_TOKENS, "messages": user}),
        )


def main() -> None:
    # ---- P1: the catalog. Does it publish prices? ----------------------------------
    print("[P1] GET /v1/models")
    p1 = call("GET", "/models")
    save("01_real_models_catalog.json", "P1", "does GET /v1/models publish per-model pricing?", p1)

    body = p1["_body"] if isinstance(p1["_body"], dict) else {}
    catalog: list[dict[str, Any]] = body.get("data") if isinstance(body.get("data"), list) else []
    print(f"  {len(catalog)} models visible to this key")
    if not catalog:
        print("  no models — cannot run P2-P7. Check the key's catalog in the dashboard.")
        return

    # Report, loudly, whether P1 answered the pricing question.
    price_keys: set[str] = set()
    price_re = re.compile(r"(pric|cost|rate|per_m|per_million)", re.I)
    for m in catalog:
        for k, v in m.items():
            if price_re.search(k):
                price_keys.add(k)
            if isinstance(v, dict):
                price_keys.update(f"{k}.{nk}" for nk in v if price_re.search(nk))
    print(
        f"  P1 ANSWER: catalog carries price-shaped fields: {', '.join(sorted(price_keys))}"
        if price_keys
        else "  P1 ANSWER: no price-shaped field in the catalog — price mode must go through OpenRouter"
    )

    cheap = os.environ.get("RAMP_ROUTER_MODEL") or pick_model(
        catalog, ["openai"], ["nano", "mini", "4o-mini"]
    )
    anthropic = os.environ.get("RAMP_ROUTER_ANTHROPIC_MODEL") or pick_model(
        catalog, ["anthropic"], ["haiku", "sonnet"]
    )
    reasoning = os.environ.get("RAMP_ROUTER_REASONING_MODEL") or pick_model(
        catalog, ["openai"], ["o4-mini", "o3-mini", "o3", "gpt-5"]
    )
    print(f"  using: cheap={cheap} anthropic={anthropic} reasoning={reasoning}")

    if "--messages-only" in sys.argv:
        _probe_messages(cheap, anthropic)
        return

    # ---- P2: requested alias, or served candidate? ----------------------------------
    if cheap:
        print("[P2] buffered call, plain `model`")
        p2 = call(
            "POST",
            "/responses",
            {
                "model": cheap,
                "input": PROMPT,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "metadata": {"lago_subscription": "rr_gateway_test_sub"},
            },
        )
        save(
            "02_real_buffered_plain_model.json",
            "P2",
            "does response.model echo the requested id or the served candidate?",
            {"_requested_model": cheap, **p2},
        )
        served = p2["_body"].get("model") if isinstance(p2["_body"], dict) else None
        print(f'  P2 ANSWER: requested "{cheap}" -> response.model "{served}"')

    # ---- P3: which candidate answered? -----------------------------------------------
    cheap_cand, anthropic_cand = candidate_id(catalog, cheap), candidate_id(catalog, anthropic)
    if cheap_cand and anthropic_cand and cheap_cand != anthropic_cand:
        print("[P3] models fallback list")
        # Both candidates are real (an unroutable one fails the whole request rather
        # than falling back), so the question is only which one Router names.
        p3 = call(
            "POST",
            "/responses",
            {"models": [cheap_cand, anthropic_cand], "input": PROMPT, "max_output_tokens": MAX_OUTPUT_TOKENS},
        )
        save(
            "03_real_models_fallback.json",
            "P3",
            "under a candidate list, does response.model name the SERVED candidate?",
            {"_requested_models": [cheap_cand, anthropic_cand], **p3},
        )
        served = p3["_body"].get("model") if isinstance(p3["_body"], dict) else None
        print(f'  P3 ANSWER: candidates [{cheap_cand}, {anthropic_cand}] -> response.model "{served}"')

    # ---- P4: streaming ---------------------------------------------------------------
    if cheap:
        print("[P4] stream: true")
        p4 = call_stream(
            {"model": cheap, "input": PROMPT, "max_output_tokens": MAX_OUTPUT_TOKENS, "stream": True}
        )
        save(
            "04_real_streamed.json",
            "P4",
            "do SSE events carry usage and the resolved model under .response?",
            p4,
        )
        with_usage = [
            e
            for e in p4["_events"]
            if (
                isinstance(e, dict)
                and (e.get("usage") or (isinstance(e.get("response"), dict) and e["response"].get("usage")))
            )
        ]
        print(f"  P4 ANSWER: {len(p4['_events'])} events, {len(with_usage)} carrying usage")

    # ---- P5: the money question -------------------------------------------------------
    if anthropic:
        print("[P5] prompt cache, two probes")
        # Probe A: `prompt_cache_key` alone. Measured NOT to warm anything — kept
        # because a negative that cost a capture is worth not re-buying.
        filler = "You are a careful billing assistant. Answer in one word. " * 120
        cache_body: dict[str, Any] = {
            "model": anthropic,
            "instructions": filler,
            "input": PROMPT,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "prompt_cache_key": "lago-capture-p5",
        }
        write = call("POST", "/responses", cache_body)
        save(
            "05_real_cache_write.json",
            "P5",
            "first call — does it report cache_write/creation tokens?",
            write,
        )
        time.sleep(3)
        read = call("POST", "/responses", cache_body)
        save(
            "06_real_cache_read.json",
            "P5",
            "second call — is cache_read INSIDE input_tokens or additive?",
            read,
        )
        # Probe B: an explicit cache_control content part, which Router forwards and the
        # provider honours. THIS pair is the one that answered the question.
        big = "You are a precise assistant. " + " ".join(
            f"Fact {i}: item {i} maps to {(i * 7) % 1000}." for i in range(1300)
        )
        cc_body = {
            "model": anthropic,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": big, "cache_control": {"type": "ephemeral"}},
                        {"type": "input_text", "text": "What does item 3 map to? Number only."},
                    ],
                }
            ],
        }
        cold = call("POST", "/responses", cc_body)
        save(
            "05b_real_cache_control_cold.json",
            "P5b",
            "does Router forward an explicit cache_control part?",
            cold,
        )
        time.sleep(2)
        warm = call("POST", "/responses", cc_body)
        save(
            "06b_real_cache_control_warm.json", "P5b", "warm repeat of 05b — additive or inside-input?", warm
        )
        u = warm["_body"].get("usage") if isinstance(warm["_body"], dict) else None
        print(f"  P5 ANSWER: usage on the warm call = {json.dumps(u)}")
        print("  input unchanged with cached_tokens > 0 => cache_read is INSIDE input (OpenAI semantics).")

    # ---- P6: reasoning ---------------------------------------------------------------
    if reasoning:
        print("[P6] reasoning model")
        p6 = call(
            "POST",
            "/responses",
            {
                "model": reasoning,
                "input": "How many primes are there between 10 and 30? Answer with the count only.",
                "max_output_tokens": 400,
                "reasoning": {"effort": "medium"},
            },
        )
        save("07_real_reasoning.json", "P6", "is reasoning_tokens inside output_tokens?", p6)
        u = p6["_body"].get("usage") if isinstance(p6["_body"], dict) else None
        print(f"  P6 ANSWER: usage = {json.dumps(u)}")

    # ---- P7: error families -----------------------------------------------------------
    print("[P7] error families")
    errors: list[tuple[str, str, dict[str, Any]]] = [
        (
            "08_real_error_404_model_not_found.json",
            "an id not in this key's catalog",
            {"model": "definitely-not-a-model-id", "input": PROMPT, "max_output_tokens": MAX_OUTPUT_TOKENS},
        ),
        (
            "09_real_error_400_both_selectors.json",
            "both route selectors at once",
            {"model": cheap, "models": [cheap_cand], "input": PROMPT, "max_output_tokens": MAX_OUTPUT_TOKENS},
        ),
        (
            "10_real_error_400_no_selector.json",
            "neither route selector",
            {"input": PROMPT, "max_output_tokens": MAX_OUTPUT_TOKENS},
        ),
    ]
    for name, what, err_body in errors:
        captured = call("POST", "/responses", err_body)
        save(name, "P7", f"error envelope for: {what}", captured)
        print(f"  {name}: HTTP {captured['_status']} json={captured['_body_was_json']}")

    print("\nDone. Inspect tests/unit/adapters/fixtures/ramp_router/*.json")
    print("Then grep the directory for the key value, Bearer, non-example emails and routable IPs.")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:  # noqa: BLE001
        # Deliberately not dumping the error object whole: a failure can carry the
        # request — headers included — and this script holds a live credential.
        print(f"capture failed: {err.__class__.__name__}: {err}")
        sys.exit(1)

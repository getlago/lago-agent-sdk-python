"""CanonicalUsage — normalized usage shape emitted to Lago."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# The routing prefix Cloudflare's OpenAI-compatible `/compat` endpoint requires:
# "workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast". The same model therefore
# arrives under two spellings depending on which surface the customer used, and two
# unrelated layers need to agree on this string — `adapters/openai_native` decides
# the PROVIDER from it, and `pricing.lookup_cloudflare_workers_ai` strips it before
# matching, because Cloudflare's own catalog lists only the bare "@cf/..." form.
#
# It lives here rather than in either of them because they must never import each
# other (an adapter is a pure function of a provider response; pricing is SDK state),
# and because a drift between two copies is a silent unpriced call, not a crash. This
# module is the natural shared floor: it imports nothing from the package, so there is
# no cycle in either direction, and depending on it does not pull `pricing`'s ~50KB
# into a lightweight adapter.
WORKERS_AI_COMPAT_PREFIX = "workers-ai/"


@dataclass
class CanonicalUsage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    reasoning: int = 0
    tool_calls: int = 0
    image_input: int = 0
    audio_input: int = 0
    audio_output: int = 0
    model: str = ""
    provider: str = ""
    api: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    NUMERIC_FIELDS = (
        "input",
        "output",
        "cache_read",
        "cache_write",
        "cache_write_5m",
        "cache_write_1h",
        "reasoning",
        "tool_calls",
        "image_input",
        "audio_input",
        "audio_output",
    )

    def nonzero_numeric(self) -> dict[str, int]:
        """Fields with a POSITIVE count, i.e. the ones worth billing.

        `> 0`, not just truthy: a negative slipped through and was emitted verbatim
        as `value="-100"`, which Lago would sum into a negative billable quantity.
        Nothing upstream should produce one — every adapter clamps at extraction —
        but this is the last gate before an event is built, and the JS port already
        filtered on `> 0`, so the two disagreed on the same input.
        """
        return {k: v for k in self.NUMERIC_FIELDS if (v := getattr(self, k)) and v > 0}

    def negative_numeric(self) -> dict[str, int]:
        """Fields `nonzero_numeric` DROPPED for being negative, so the caller can
        report them.

        Reachable, unlike most defensive paths here: `CanonicalUsage` is exported and
        `emit()` takes one directly, which is the documented way to backfill usage the
        SDK did not intercept. A caller computing a delta wrongly can hand us a
        negative, and silently dropping it is the one drop path that never reached
        `on_error` — the same gap that was closed for queue overflow and for an
        unresolvable subscription. Kept as a separate pure query so `CanonicalUsage`
        stays a dumb dataclass with no notification channel of its own.
        """
        return {k: v for k in self.NUMERIC_FIELDS if (v := getattr(self, k)) and v < 0}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

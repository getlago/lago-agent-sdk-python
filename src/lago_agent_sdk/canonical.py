"""CanonicalUsage — normalized usage shape emitted to Lago."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

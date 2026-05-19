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
    )

    def nonzero_numeric(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in self.NUMERIC_FIELDS if getattr(self, k)}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

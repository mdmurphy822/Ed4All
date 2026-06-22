"""Qwen-specialist data contracts (Phase 0).

Concrete-shaped types that downstream phases (5-9b) plug into.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AdapterID(str, Enum):
    """Stable identifier of a Qwen LoRA adapter on disk.

    Strings (not Path) so the values round-trip through JSON without
    custom encoders. The on-disk path lives in qwen_specialists/config.yaml.
    """
    PROSE = "prose"
    TABLE = "table"
    MATH = "math"
    GAP_FILL = "gap_fill"


@dataclass(frozen=True)
class SpecialistRequest:
    """One inference request to a Qwen specialist.

    `payload` is intentionally typed-loosely; per-adapter shape is locked
    by Phase 5/6's prompt-construction modules. The base contract is just
    "string-keyed dict serializable to JSON".
    """
    adapter: AdapterID
    payload: dict[str, Any]
    request_id: str = ""              # caller-supplied; useful for retries / dedup
    sampling_overrides: dict[str, Any] = field(default_factory=dict)
    # ^ optional per-call overrides on top of qwen_specialists/config.yaml defaults.


@dataclass(frozen=True)
class Candidate:
    """One generated completion from a specialist.

    Specialists emit K candidates (default K=4 per architecture.md §4.2);
    Stage 8's per-region soft reranker picks the survivor.
    """
    adapter: AdapterID
    request_id: str
    text: str                        # the generated completion (HTML fragment, MathML, etc.)
    score: float | None = None       # specialist self-score; None if not emitted
    sampling_seed: int | None = None
    finish_reason: str = ""          # "stop" | "length" | "eos" | ...
    raw_metadata: dict[str, Any] = field(default_factory=dict)

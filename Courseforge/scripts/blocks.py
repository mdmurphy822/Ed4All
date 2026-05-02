"""Phase 2 intermediate block format.

Defines the canonical in-memory ``Block`` + ``Touch`` dataclasses that
``generate_course.py`` will project to (a) ``data-cf-*`` HTML attribute
strings on emit and (b) JSON-LD entries inside ``<script type=
"application/ld+json">`` blocks. Blocks are the stable intermediate
between the Phase 1 ``ContentGeneratorProvider`` (which currently
returns ``str`` HTML) and the legacy renderer surface; Phase 2 widens
the provider to return ``Block`` instances so the renderer composes
attribute strings from a typed object instead of regex-parsing back
out of the LLM's HTML.

The dataclass is intentionally frozen — Phase 2 mutations (touch chain
appends, validation-attempt increments, escalation marking) all return
new instances via ``dataclasses.replace``. Three feedback-driven fields
support Phase 3's per-block regeneration budget + escalation primitive:
``validation_attempts`` (incremented per failed validator pass) and
``escalation_marker`` (set when a block is escalated to the rewrite
tier after the outline-tier budget is exhausted).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = ["Block", "Touch", "BLOCK_TYPES"]


BLOCK_TYPES: frozenset = frozenset(
    {
        "objective",
        "concept",
        "example",
        "assessment_item",
        "explanation",
        "prereq_set",
        "activity",
        "misconception",
        "callout",
        "flip_card_grid",
        "self_check_question",
        "summary_takeaway",
        "reflection_prompt",
        "discussion_prompt",
        "chrome",
        "recap",
    }
)


_TOUCH_TIERS: frozenset = frozenset({"outline", "validation", "rewrite"})

_TOUCH_PROVIDERS: frozenset = frozenset(
    {"anthropic", "local", "together", "claude_session", "deterministic"}
)

_ESCALATION_MARKERS: frozenset = frozenset(
    {
        "outline_budget_exhausted",
        "structural_unfixable",
        "validator_consensus_fail",
    }
)


_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    """Lowercase + non-alnum-to-underscore + strip + max 40 chars.

    Mirrors Courseforge's existing slug helper (lowercase + space-to-
    underscore + non-alnum collapse) so block IDs concatenate cleanly
    with page IDs.
    """
    if not text:
        return ""
    lowered = text.strip().lower()
    collapsed = _SLUG_NON_ALNUM.sub("_", lowered).strip("_")
    return collapsed[:40]


@dataclass(frozen=True)
class Touch:
    """One revision attribution event in a Block's touch chain.

    Each tier (``outline`` / ``validation`` / ``rewrite``) emits a
    ``Touch`` when it modifies the block. The chain is cumulative — the
    audit value of the full chain is bounded by ~12k entries × ~80
    bytes ≈ ~1 MB JSON before gzip per course (well within IMSCC payload
    budgets), so retention is full per pre-resolved decision #2 in the
    Phase 2 plan.
    """

    model: str
    provider: str
    tier: str
    timestamp: str
    decision_capture_id: str
    purpose: str

    def __post_init__(self) -> None:
        if not self.decision_capture_id:
            raise ValueError(
                "Touch.decision_capture_id required (Wave 112 invariant)"
            )
        if self.tier not in _TOUCH_TIERS:
            raise ValueError(
                f"Touch.tier must be one of {sorted(_TOUCH_TIERS)}; got {self.tier!r}"
            )
        if self.provider not in _TOUCH_PROVIDERS:
            raise ValueError(
                f"Touch.provider must be one of {sorted(_TOUCH_PROVIDERS)}; "
                f"got {self.provider!r}"
            )

    def to_jsonld(self) -> Dict[str, Any]:
        """Wire shape — camelCase keys for JSON-LD emit."""
        return {
            "model": self.model,
            "provider": self.provider,
            "tier": self.tier,
            "timestamp": self.timestamp,
            "decisionCaptureId": self.decision_capture_id,
            "purpose": self.purpose,
        }


@dataclass(frozen=True)
class Block:
    """Canonical intermediate block.

    Owns its identity (``block_id`` / ``page_id`` / ``sequence``) plus
    the pedagogical metadata the renderer + JSON-LD builder consume.
    Frozen — mutations return a new instance via ``dataclasses.replace``.

    Phase 3 feedback-driven fields:
        ``validation_attempts``: incremented by the outline-tier
            regeneration router on every failed validator pass.
        ``escalation_marker``: set to a non-empty marker (one of
            ``_ESCALATION_MARKERS``) when the block exhausts its
            outline-tier budget and is escalated to the rewrite tier.
    Both stay default (``0`` / ``None``) for blocks emitted by the
    deterministic / Phase-1-provider paths in Phase 2.
    """

    block_id: str
    block_type: str
    page_id: str
    sequence: int
    content: Union[str, Dict[str, Any]]
    template_type: Optional[str] = None
    key_terms: Tuple[str, ...] = ()
    objective_ids: Tuple[str, ...] = ()
    bloom_level: Optional[str] = None
    bloom_verb: Optional[str] = None
    bloom_range: Optional[str] = None
    bloom_levels: Tuple[str, ...] = ()
    bloom_verbs: Tuple[str, ...] = ()
    cognitive_domain: Optional[str] = None
    teaching_role: Optional[str] = None
    content_type_label: Optional[str] = None
    purpose: Optional[str] = None
    component: Optional[str] = None
    source_ids: Tuple[str, ...] = ()
    source_primary: Optional[str] = None
    source_references: Tuple[Dict[str, Any], ...] = ()
    touched_by: Tuple[Touch, ...] = ()
    content_hash: Optional[str] = None
    validation_attempts: int = 0
    escalation_marker: Optional[str] = None

    def __post_init__(self) -> None:
        if self.block_type not in BLOCK_TYPES:
            raise ValueError(
                f"Block.block_type must be one of {sorted(BLOCK_TYPES)}; "
                f"got {self.block_type!r}"
            )
        if self.sequence < 0:
            raise ValueError(
                f"Block.sequence must be >= 0; got {self.sequence}"
            )
        if not self.page_id:
            raise ValueError("Block.page_id must be non-empty")
        if self.validation_attempts < 0:
            raise ValueError(
                f"Block.validation_attempts must be >= 0; "
                f"got {self.validation_attempts}"
            )
        if (
            self.escalation_marker is not None
            and self.escalation_marker not in _ESCALATION_MARKERS
        ):
            raise ValueError(
                f"Block.escalation_marker must be None or one of "
                f"{sorted(_ESCALATION_MARKERS)}; got {self.escalation_marker!r}"
            )

    @classmethod
    def stable_id(cls, page_id: str, block_type: str, slug: str, idx: int) -> str:
        """Position-based block ID per pre-resolved decision #1.

        Format: ``{page_id}#{block_type}_{slug}_{idx}``. Hash-based IDs
        are deferred — bottom-up migration produces stable orderings
        per renderer; reorder churn is rare.
        """
        return f"{page_id}#{block_type}_{slug}_{idx}"

    def with_touch(self, touch: Touch) -> "Block":
        """Return a new Block with ``touch`` appended to ``touched_by``.

        The content hash is unchanged — touches are audit-only and
        excluded from the canonical hash payload.
        """
        return dataclasses.replace(self, touched_by=self.touched_by + (touch,))

    def compute_content_hash(self) -> str:
        """SHA-256 hex of the canonical Block payload.

        Excludes ``touched_by``, ``sequence``, ``validation_attempts``,
        and ``escalation_marker`` so a touch-only or budget-only
        revision keeps a stable hash. The hash exists for re-execution
        drift detection — same content → same hash regardless of
        which tier authored it or how many times it was retried.
        """
        payload = {
            "content": self.content,
            "block_type": self.block_type,
            "key_terms": list(self.key_terms),
            "bloom_level": self.bloom_level,
            "objective_ids": list(self.objective_ids),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

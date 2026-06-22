"""Gate result data contracts (Phase 0).

`GateResult` is the eliminating verdict (pass/fail) plus a per-check
breakdown. The assembler's `DocCandidate` carries a `GateResult` slot
typed as a forward-string to avoid a circular import at definition time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GateCheck(str, Enum):
    """Individual checks any of the four gate stages can run."""
    AXE_WCAG22AA       = "axe_wcag22aa"
    HTML5_VALIDATOR    = "html5_validator"
    TEXT_PRESERVE      = "text_preserve"
    MATHML_VALIDITY    = "mathml_validity"
    TABLE_STRUCTURE    = "table_structure"
    HEADING_TREE       = "heading_tree"
    LANDMARK_PRESENCE  = "landmark_presence"
    REF_LINK_INTEGRITY = "ref_link_integrity"
    LANG_DECLARED      = "lang_declared"
    TITLE_PRESENT      = "title_present"


@dataclass(frozen=True)
class CheckOutcome:
    """One check's pass/fail plus diagnostic detail.

    ``skipped`` semantics: True means the check did not actually run
    against real source signal — pass should be interpreted as "no
    measurement", not "verified safe". Default False so existing
    constructors keep the old contract; aggregators that conflate
    skipped passes with measured passes silently mask gate-coverage
    holes (gap-fill regions with empty source text, lxml absent on
    math regions, doc with zero headings, etc.).
    """
    check: GateCheck
    passed: bool
    message: str = ""
    details: dict = field(default_factory=dict)
    skipped: bool = False


@dataclass(frozen=True)
class GateResult:
    """Aggregate verdict from a gate stage.

    `passed` is the AND of every CheckOutcome.passed. A gate that runs
    zero checks is treated as failing — empty checks is always a bug.
    """
    stage: str                            # "hard_region" | "hard_document" | "soft_region" | "soft_document"
    passed: bool
    checks: list[CheckOutcome]
    region_id: int | None = None         # None for document-level gates
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def from_checks(stage: str, checks: list[CheckOutcome],
                    region_id: int | None = None) -> "GateResult":
        if not checks:
            raise ValueError(f"{stage}: GateResult requires at least one CheckOutcome")
        return GateResult(
            stage=stage,
            passed=all(c.passed for c in checks),
            checks=list(checks),
            region_id=region_id,
        )

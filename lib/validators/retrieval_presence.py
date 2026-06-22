"""IB7.5b — RetrievalPresenceValidator.

Spacing-aware retrieval-presence floor. The framework's
``spacing-retrieval-in-exposition-floor`` gap (sev med): a content-bearing
module must carry at least one *low-stakes retrieval* block
(``self_check_question`` / ``reflection_prompt``) that is SPACED from the
exposition — i.e. NOT the very first block on the content page. A retrieval
opportunity buried at the top of a page (before any material is taught) is
not a retrieval-from-memory cue; it is a pre-quiz. And a content module that
ships zero low-stakes retrieval blocks at all gives the learner no practice
hook.

Different surface from ``InstructionalDepthValidator`` (per-page concept /
example TOKEN floors) and ``ExampleCompletenessValidator`` (per-``example``
body completeness): this validator audits ONLY the *presence + spacing* of
retrieval blocks across each content-bearing module (page group).

Per the IB7.5 spec (``plans/finegrain/ib7-planner-lifecycle-bloom-climb-
spacing-2026-06-22.md`` §IB7.5b): group the input ``blocks`` by ``page_id``;
a "content-bearing module" is a group with >=1 block whose ``block_type`` is
in the content set. For each such group assert >=1 retrieval block exists
AND that the first retrieval block is NOT the first block on the page (there
must be >=1 non-retrieval block before it). A miss emits a warning-severity
GateIssue carrying ``action="regenerate"``.

NO embeddings — pure block-type / page-grouping. Mirrors the
``example_completeness`` posture from CB5a (capture-emit pattern, empty-input
graceful pass, ``_ISSUE_LIST_CAP``).

Inputs (``inputs`` dict):

    blocks: List[Block]
        Rewrite- (or outline-) tier ``Courseforge.scripts.blocks.Block``
        instances. Only ``block_type`` + ``page_id`` are read, so the
        ``content`` shape (str rewrite-tier / dict outline-tier) is
        irrelevant here.

    decision_capture: Optional[Any]
        Optional ``DecisionCapture``-shaped instance. One
        ``content_selection`` event is emitted per audited content-bearing
        module (dynamic signals: page_id, n_blocks, n_content_blocks,
        n_retrieval_blocks, first_block_is_retrieval, verdict).

    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to
        ``"retrieval_presence"``).

References:
    - ``plans/finegrain/ib7-planner-lifecycle-bloom-climb-spacing-2026-06-22.md``
      §IB7.5b.
    - ``lib/validators/example_completeness.py`` — sibling CB5a gate
      (structure mirrored here).
"""

from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

# Import bridge for Block (mirror of example_completeness.py).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:  # pragma: no cover - import guard
    from blocks import Block  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - fallback for partial installs
    Block = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Block-type sets
# ---------------------------------------------------------------------------

#: Content-bearing block types. A page group carrying >=1 of these is a
#: content-bearing module subject to the retrieval-presence floor.
CONTENT_BLOCK_TYPES: frozenset = frozenset(
    {
        "concept",
        "explanation",
        "example",
        "worked_example",
        "formula",
        "vocab_card",
        "key_idea",
        "table",
        "scenario",
        "problem",
        "activity",
    }
)

#: Low-stakes retrieval block types. Presence of >=1 (spaced) satisfies the
#: floor.
RETRIEVAL_BLOCK_TYPES: frozenset = frozenset(
    {"self_check_question", "reflection_prompt"}
)

#: Cap on per-validate issue list to avoid runaway emit on many-module
#: corpora. Mirrors the cap used by sibling structural validators.
_ISSUE_LIST_CAP: int = 50


# ---------------------------------------------------------------------------
# Grouping helper
# ---------------------------------------------------------------------------


def _group_by_page(blocks: List[Any]) -> "OrderedDict[str, List[Any]]":
    """Group blocks by their ``page_id`` attribute, preserving block order.

    Order within a page is the input order; if blocks carry a ``sequence``
    attribute the input is assumed already sequence-ordered (mirrors the
    outline/rewrite emit contract). Page-group insertion order follows the
    first appearance of each ``page_id``.
    """
    groups: "OrderedDict[str, List[Any]]" = OrderedDict()
    for b in blocks:
        page_id = str(getattr(b, "page_id", "") or "")
        groups.setdefault(page_id, []).append(b)
    return groups


# ---------------------------------------------------------------------------
# Decision-capture emit
# ---------------------------------------------------------------------------


def _emit_decision(
    capture: Optional[Any],
    *,
    page_id: str,
    n_blocks: int,
    n_content_blocks: int,
    n_retrieval_blocks: int,
    first_block_is_retrieval: bool,
    spaced: bool,
    passed: bool,
) -> None:
    """Emit one ``content_selection`` event per audited content-bearing
    module.

    Rationale interpolates the dynamic signals (page_id, block / content /
    retrieval counts, first_block_is_retrieval) so post-hoc replay
    reconstructs the verdict without re-reading the gate result.
    """
    if capture is None:
        return
    verdict = "spaced_retrieval_present" if passed else "retrieval_floor_miss"
    rationale = (
        f"retrieval_presence audit of module page_id={page_id!r}: "
        f"n_blocks={n_blocks}, n_content_blocks={n_content_blocks}, "
        f"n_retrieval_blocks={n_retrieval_blocks}, "
        f"first_block_is_retrieval={first_block_is_retrieval}, "
        f"spaced={spaced} -> {verdict}. A content-bearing module must carry "
        f">=1 low-stakes retrieval block (self_check_question / "
        f"reflection_prompt) that is NOT the first block on the page."
    )
    ml_features = {
        "page_id": page_id,
        "n_content_blocks": int(n_content_blocks),
        "n_retrieval_blocks": int(n_retrieval_blocks),
        "spaced": bool(spaced),
        "passed": bool(passed),
    }
    try:
        capture.log_decision(
            decision_type="content_selection",
            decision=verdict,
            rationale=rationale,
            ml_features=ml_features,
        )
    except Exception as exc:  # noqa: BLE001 - audit emit is best-effort
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "retrieval_presence content_selection: %s",
            exc,
        )


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------


class RetrievalPresenceValidator:
    """IB7.5b — spacing-aware retrieval-presence floor.

    Validator-protocol-compatible class. Audits every content-bearing module
    (page group) for >=1 spaced low-stakes retrieval block. A module with
    zero retrieval blocks emits ``MODULE_NO_RETRIEVAL``; a module whose only
    retrieval block is the first block on the page emits
    ``RETRIEVAL_UNSPACED`` — both warning, both ``action="regenerate"``.
    Empty input / no content-bearing groups pass gracefully.
    """

    name = "retrieval_presence"
    version = "0.1.0"

    def __init__(
        self,
        *,
        decision_capture: Optional[Any] = None,
    ) -> None:
        self._decision_capture = decision_capture

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or self._decision_capture

        raw = inputs.get("blocks")
        if raw is None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="MISSING_BLOCKS_INPUT",
                        message=(
                            "inputs['blocks'] is required; expected a list "
                            "of Courseforge Block instances."
                        ),
                    )
                ],
                action="regenerate",
            )
        if not isinstance(raw, list):
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="critical",
                        code="INVALID_BLOCKS_INPUT",
                        message=(
                            f"inputs['blocks'] must be a list; got "
                            f"{type(raw).__name__}."
                        ),
                    )
                ],
                action="regenerate",
            )

        blocks: List[Any] = list(raw)
        groups = _group_by_page(blocks)

        issues: List[GateIssue] = []
        audited = 0
        missing = 0
        for page_id, group in groups.items():
            content_blocks = [
                b for b in group
                if getattr(b, "block_type", None) in CONTENT_BLOCK_TYPES
            ]
            if not content_blocks:
                # Not a content-bearing module → not audited.
                continue
            audited += 1
            retrieval_idxs = [
                i for i, b in enumerate(group)
                if getattr(b, "block_type", None) in RETRIEVAL_BLOCK_TYPES
            ]
            n_retrieval = len(retrieval_idxs)
            first_is_retrieval = bool(group) and (
                getattr(group[0], "block_type", None)
                in RETRIEVAL_BLOCK_TYPES
            )
            # Spaced iff there is >=1 retrieval block that is NOT the first
            # block on the page (i.e. some retrieval block sits at index>0).
            spaced = any(i > 0 for i in retrieval_idxs)
            passed_module = n_retrieval > 0 and spaced

            _emit_decision(
                capture,
                page_id=page_id,
                n_blocks=len(group),
                n_content_blocks=len(content_blocks),
                n_retrieval_blocks=n_retrieval,
                first_block_is_retrieval=first_is_retrieval,
                spaced=spaced,
                passed=passed_module,
            )

            if passed_module:
                continue
            missing += 1
            if len(issues) >= _ISSUE_LIST_CAP:
                continue
            if n_retrieval == 0:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="MODULE_NO_RETRIEVAL",
                        message=(
                            f"content-bearing module page_id={page_id!r} "
                            f"({len(content_blocks)} content block(s)) carries "
                            f"NO low-stakes retrieval block "
                            f"(self_check_question / reflection_prompt)."
                        ),
                        location=page_id,
                        suggestion=(
                            "Regenerate the module to include >=1 low-stakes "
                            "retrieval block (self_check_question or "
                            "reflection_prompt) spaced after the exposition "
                            "that taught the same objective."
                        ),
                    )
                )
            else:
                # Retrieval exists but only as the first block on the page.
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="RETRIEVAL_UNSPACED",
                        message=(
                            f"content-bearing module page_id={page_id!r} has a "
                            f"retrieval block as the FIRST block on the page "
                            f"(unspaced pre-quiz, not a retrieval-from-memory "
                            f"cue); no spaced retrieval block follows the "
                            f"exposition."
                        ),
                        location=page_id,
                        suggestion=(
                            "Regenerate the module so the retrieval block "
                            "(self_check_question / reflection_prompt) is "
                            "SPACED — place >=1 exposition block before it "
                            "rather than opening the page with the check."
                        ),
                    )
                )

        # No content-bearing modules → graceful pass.
        if audited == 0:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
                action=None,
            )

        passed = missing == 0
        score = round((audited - missing) / audited, 4) if audited else 1.0
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None if passed else "regenerate",
        )


__all__ = [
    "RetrievalPresenceValidator",
    "CONTENT_BLOCK_TYPES",
    "RETRIEVAL_BLOCK_TYPES",
]

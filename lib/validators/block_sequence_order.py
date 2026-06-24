"""FR-COURSE-02 — BlockSequenceOrderValidator.

Pedagogical block-sequence-order floor. The framework's worked-example →
guided-practice → independent-practice gradient (the "fade" from modelled to
independent performance) plus the spacing + opener constraints. Three
out-of-order signals, all warning-day-1 with a deferred critical-flip:

  * ``WORKED_BEFORE_GUIDED_OUT_OF_ORDER`` — a B05 worked_example that the
    framework's fade gradient expects BEFORE its B08 guided-practice
    (``problem`` / ``activity``) appears AFTER that guided block on the same
    page. Modelling must precede guided practice.
  * ``CHECK_NOT_SPACED`` — a B07 check (``self_check_question`` /
    ``assessment_item``) sits immediately adjacent to (massed against) the
    exposition that taught the SAME objective, with no intervening block —
    not spaced retrieval.
  * ``SCENARIO_OPENS_TO`` — a B09 ``scenario`` is the FIRST block of a
    terminal-objective module (page group), opening the TO with an
    application activity before any exposition has scaffolded it.

Cloned from ``lib/validators/retrieval_presence.py`` — same input shape
(``inputs['blocks']`` grouped by ``page_id``), same gating posture (runs
warning-day-1 regardless of any flag; no env gate), same capture-emit +
``_ISSUE_LIST_CAP`` + graceful-empty-pass conventions. NO embeddings — pure
block-type / objective_ids / page-grouping.

Inputs (``inputs`` dict):

    blocks: List[Block]
        Outline- or rewrite-tier ``Courseforge.scripts.blocks.Block``
        instances. Reads ``block_type``, ``page_id``, ``objective_ids``.

    decision_capture: Optional[Any]
        Optional ``DecisionCapture``-shaped instance. One ``content_selection``
        event per audited content-bearing module.

    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to
        ``"block_sequence_order"``).

References:
    - FRAMEWORK.pdf — worked-example fade gradient (B05→B08→B09), spacing.
    - ``lib/validators/retrieval_presence.py`` — sibling structural validator
      (structure cloned here).
"""

from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult

# Import bridge for Block (mirror of retrieval_presence.py).
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

#: Content-bearing block types (mirror of retrieval_presence). A page group
#: carrying >=1 of these is a content-bearing module subject to the order
#: floor.
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

#: B05 — modelled-performance block.
WORKED_EXAMPLE_TYPES: frozenset = frozenset({"worked_example"})

#: B08 — guided-practice block (learner practises with scaffolding).
GUIDED_PRACTICE_TYPES: frozenset = frozenset({"problem", "activity"})

#: B07 — check / retrieval block whose spacing from same-objective exposition
#: is audited.
CHECK_TYPES: frozenset = frozenset(
    {"self_check_question", "assessment_item"}
)

#: B09 — application scenario block. Should not OPEN a module.
SCENARIO_TYPES: frozenset = frozenset({"scenario"})

#: Exposition block types — the material a B07 check must be spaced from.
EXPOSITION_TYPES: frozenset = frozenset(
    {"concept", "explanation", "example", "worked_example", "formula",
     "vocab_card", "key_idea", "table"}
)

#: Cap on per-validate issue list to avoid runaway emit on many-module
#: corpora. Mirrors retrieval_presence.
_ISSUE_LIST_CAP: int = 50


# ---------------------------------------------------------------------------
# Grouping helper (mirror of retrieval_presence)
# ---------------------------------------------------------------------------


def _group_by_page(blocks: List[Any]) -> "OrderedDict[str, List[Any]]":
    """Group blocks by ``page_id``, preserving input (sequence) order."""
    groups: "OrderedDict[str, List[Any]]" = OrderedDict()
    for b in blocks:
        page_id = str(getattr(b, "page_id", "") or "")
        groups.setdefault(page_id, []).append(b)
    return groups


def _objective_set(block: Any) -> set:
    raw = getattr(block, "objective_ids", ()) or ()
    try:
        return {str(x) for x in raw if x}
    except TypeError:
        return set()


# ---------------------------------------------------------------------------
# Decision-capture emit
# ---------------------------------------------------------------------------


def _emit_decision(
    capture: Optional[Any],
    *,
    page_id: str,
    n_blocks: int,
    n_content_blocks: int,
    worked_before_guided: bool,
    check_unspaced: bool,
    scenario_opens: bool,
    passed: bool,
) -> None:
    """Emit one ``content_selection`` event per audited content module."""
    if capture is None:
        return
    verdict = "block_sequence_ok" if passed else "block_sequence_violation"
    rationale = (
        f"block_sequence_order audit of module page_id={page_id!r}: "
        f"n_blocks={n_blocks}, n_content_blocks={n_content_blocks}, "
        f"worked_before_guided_out_of_order={worked_before_guided}, "
        f"check_not_spaced={check_unspaced}, scenario_opens_to={scenario_opens}"
        f" -> {verdict}. The framework fade gradient expects worked_example "
        f"(B05) before guided practice (B08), spaced checks (B07), and no "
        f"scenario (B09) opening a TO."
    )
    ml_features = {
        "page_id": page_id,
        "n_content_blocks": int(n_content_blocks),
        "worked_before_guided_out_of_order": bool(worked_before_guided),
        "check_not_spaced": bool(check_unspaced),
        "scenario_opens_to": bool(scenario_opens),
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
            "DecisionCapture.log_decision raised on block_sequence_order "
            "content_selection: %s",
            exc,
        )


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------


class BlockSequenceOrderValidator:
    """FR-COURSE-02 — pedagogical block-sequence-order floor.

    Validator-protocol-compatible class cloned from
    ``RetrievalPresenceValidator``. Audits every content-bearing module
    (page group) for the three out-of-order signals. Runs warning-day-1
    regardless of any flag (mirrors retrieval_presence's no-flag posture).
    Empty input / no content-bearing groups pass gracefully.
    """

    name = "block_sequence_order"
    version = "0.1.0"

    def __init__(self, *, decision_capture: Optional[Any] = None) -> None:
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
                continue
            audited += 1

            types = [getattr(b, "block_type", None) for b in group]

            # (1) WORKED_BEFORE_GUIDED_OUT_OF_ORDER — a guided-practice block
            # appears before any worked_example, while a worked_example DOES
            # exist later on the page (i.e. modelling came after guided
            # practice).
            worked_idxs = [
                i for i, t in enumerate(types) if t in WORKED_EXAMPLE_TYPES
            ]
            guided_idxs = [
                i for i, t in enumerate(types) if t in GUIDED_PRACTICE_TYPES
            ]
            worked_before_guided = bool(worked_idxs) and bool(guided_idxs) and (
                min(guided_idxs) < min(worked_idxs)
            )

            # (2) CHECK_NOT_SPACED — a check block immediately follows the
            # exposition that taught the SAME objective with no intervening
            # block (massed, not spaced).
            check_unspaced = False
            for i, b in enumerate(group):
                if getattr(b, "block_type", None) not in CHECK_TYPES:
                    continue
                if i == 0:
                    continue
                prev = group[i - 1]
                if getattr(prev, "block_type", None) not in EXPOSITION_TYPES:
                    continue
                shared = _objective_set(b) & _objective_set(prev)
                if shared:
                    check_unspaced = True
                    break

            # (3) SCENARIO_OPENS_TO — the first block of the module is a
            # scenario (opens the TO with an application activity).
            scenario_opens = bool(group) and (
                getattr(group[0], "block_type", None) in SCENARIO_TYPES
            )

            passed_module = not (
                worked_before_guided or check_unspaced or scenario_opens
            )

            _emit_decision(
                capture,
                page_id=page_id,
                n_blocks=len(group),
                n_content_blocks=len(content_blocks),
                worked_before_guided=worked_before_guided,
                check_unspaced=check_unspaced,
                scenario_opens=scenario_opens,
                passed=passed_module,
            )

            if passed_module:
                continue
            missing += 1
            if worked_before_guided and len(issues) < _ISSUE_LIST_CAP:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="WORKED_BEFORE_GUIDED_OUT_OF_ORDER",
                        message=(
                            f"module page_id={page_id!r}: a guided-practice "
                            f"block (problem/activity) precedes the "
                            f"worked_example that should model it first "
                            f"(B05->B08 fade gradient violated)."
                        ),
                        location=page_id,
                        suggestion=(
                            "Reorder so the worked_example (B05) appears "
                            "BEFORE the guided-practice block (B08) — model "
                            "the performance before asking the learner to "
                            "practise it."
                        ),
                    )
                )
            if check_unspaced and len(issues) < _ISSUE_LIST_CAP:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="CHECK_NOT_SPACED",
                        message=(
                            f"module page_id={page_id!r}: a check block (B07) "
                            f"is massed directly against the exposition that "
                            f"taught the same objective (no intervening "
                            f"block) — not spaced retrieval."
                        ),
                        location=page_id,
                        suggestion=(
                            "Insert >=1 intervening block between the "
                            "exposition and the same-objective check so the "
                            "check is a spaced retrieval cue, not massed "
                            "re-reading."
                        ),
                    )
                )
            if scenario_opens and len(issues) < _ISSUE_LIST_CAP:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="SCENARIO_OPENS_TO",
                        message=(
                            f"module page_id={page_id!r}: a scenario (B09) is "
                            f"the FIRST block, opening the terminal objective "
                            f"with an application activity before any "
                            f"exposition scaffolds it."
                        ),
                        location=page_id,
                        suggestion=(
                            "Place exposition / worked_example before the "
                            "scenario so the application activity follows the "
                            "material it applies."
                        ),
                    )
                )

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
    "BlockSequenceOrderValidator",
    "CONTENT_BLOCK_TYPES",
    "WORKED_EXAMPLE_TYPES",
    "GUIDED_PRACTICE_TYPES",
    "CHECK_TYPES",
    "SCENARIO_TYPES",
]

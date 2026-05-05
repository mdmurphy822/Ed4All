"""Phase 4 Wave N2 — Category C embedding validator (Subtask 14).

Validates that every ``assessment_item`` Block aligns semantically with
the learning objective(s) it claims to assess. Embedding-tier sibling
of the Phase-3 ``BlockPageObjectivesValidator`` (which only checks
*structural* membership of an ``objective_id`` in a known set). This
validator embeds the assessment stem + answer-key surface and the
declared objective statements, then computes pair-wise cosine similarity.
A drop below the threshold (default 0.55) emits ``action="regenerate"``
so the rewrite-tier router re-rolls the assessment with sharper
objective targeting.

Inputs (``inputs`` dict, mirroring the Phase 3.5 inter_tier_gates'
``BlockPageObjectivesValidator`` shape):

    blocks: List[Block]
        Outline- or rewrite-tier ``Courseforge.scripts.blocks.Block``
        instances. Only ``block.block_type == "assessment_item"`` rows
        are audited; other block types are skipped.

    objective_statements: Optional[Dict[str, str]]
        Mapping of canonical objective_id -> objective statement text
        (e.g. ``{"TO-01": "Define federated identity"}``). The router
        / workflow runner populates this from the project's synthesised
        objectives JSON before dispatch (mirrors the
        ``valid_objective_ids`` asymmetry on
        ``BlockPageObjectivesValidator``).

    threshold: Optional[float]
        Override the default cosine-similarity floor (0.55). Below this
        value the gate emits a critical issue with action="regenerate".

    embedder: Optional[SentenceEmbedder]
        Test injection point. Defaults to ``try_load_embedder()`` per
        Wave N1's fallback contract (``EMBEDDING_DEPS_MISSING`` warning
        when extras unavailable, ``passed=True``, no action).

    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to
        ``"objective_assessment_similarity"``).

Behavior contract (Wave N1 fallback policy mirror):
    - Missing ``blocks`` input -> ``passed=False``, single critical
      issue, ``action="regenerate"``.
    - No assessment_item blocks -> ``passed=True``, ``action=None``,
      no issues (the validator is a no-op on a corpus without
      assessments).
    - Embedding extras missing -> single warning issue
      (``EMBEDDING_DEPS_MISSING``), ``passed=True``, ``action=None``.
      Mirrors the SHACL-deps-missing graceful-degrade from Subtask 10.
    - At least one assessment with ``min(per-pair cosine) < threshold``
      -> ``passed=False``, ``action="regenerate"`` with one critical
      issue per low-similarity pair (capped at 50 entries to avoid
      drowning the gate report).
    - Assessment with no resolvable objective statements -> warning
      issue, no action change.
    - Assessment with empty stem+answer surface -> warning issue,
      no action change.

References:
    - ``lib/validators/page_objectives.py`` — GateResult emit pattern.
    - ``lib/validators/courseforge_outline_shacl.py`` — Wave N1
      validator pattern + extras-missing fallback.
    - ``Courseforge/router/inter_tier_gates.py::BlockPageObjectivesValidator``
      — structural sibling that this embedding validator complements.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.embedding._math import cosine_similarity
from lib.embedding.sentence_embedder import (
    SentenceEmbedder,
    try_load_embedder,
)

# ``blocks.py`` lives at ``Courseforge/scripts/blocks.py``; mirror the
# import bridge used by ``Courseforge/router/inter_tier_gates.py:56``
# so ``from blocks import Block`` resolves regardless of how this
# module is loaded.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # type: ignore[import-not-found]  # noqa: E402

logger = logging.getLogger(__name__)


#: Default cosine-similarity floor below which a (assessment, objective)
#: pair is flagged as semantically misaligned. Calibrated against the
#: Wave N1 PoC corpus — well-aligned pairs cluster above 0.65; outright
#: misalignments (e.g. arithmetic question routed to a network-policy
#: objective) cluster below 0.40. The 0.55 threshold leaves a buffer
#: against the all-MiniLM-L6-v2 model's intrinsic similarity floor
#: while still catching real misalignment.
DEFAULT_THRESHOLD: float = 0.55

#: Cap on per-block issue list to keep gate reports readable when an
#: entire batch of assessments misaligns uniformly (matches
#: ``Courseforge/router/inter_tier_gates.py::_ISSUE_LIST_CAP``).
_ISSUE_LIST_CAP: int = 50


def _emit_decision(
    capture: Any,
    *,
    block_id: str,
    passed: bool,
    code: Optional[str],
    cosine: Optional[float],
    threshold: float,
    objective_id: Optional[str],
    objective_count: int,
    embedder_strict: bool,
) -> None:
    """Emit one ``objective_assessment_similarity_check`` decision per
    ``validate()`` invocation (per-block cardinality).

    Rationale interpolates dynamic signals so captures are replayable:
    block_id, declared-objective count, weakest-pair cosine + threshold +
    above/below flag, the offending objective_id, and the embedder
    strict-mode flag (so EMBEDDING_DEPS_MISSING graceful-degrade events
    are distinguishable from real low-similarity events).
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    cos_str = f"{cosine:.4f}" if cosine is not None else "n/a"
    above_threshold = (
        cosine is not None and cosine >= threshold
    )
    rationale = (
        f"Objective/assessment similarity check on Block {block_id!r}: "
        f"min_pair_cosine={cos_str}, threshold={threshold:.4f}, "
        f"above_threshold={above_threshold}, "
        f"weakest_objective={objective_id or 'n/a'}, "
        f"declared_objective_count={objective_count}, "
        f"embedder_strict_mode={embedder_strict}, "
        f"failure_code={code or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="objective_assessment_similarity_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "objective_assessment_similarity_check: %s",
            exc,
        )


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Block], Optional[GateIssue]]:
    """Pull a list of Block instances out of ``inputs["blocks"]``.

    Mirrors the ``_coerce_blocks`` helper in
    ``Courseforge/router/inter_tier_gates.py:74``. Returns
    ``(blocks, error_issue)``. Non-None ``error_issue`` is wrapped
    into a ``passed=False`` ``GateResult`` by the caller.
    """
    raw = inputs.get("blocks")
    if raw is None:
        return [], GateIssue(
            severity="critical",
            code="MISSING_BLOCKS_INPUT",
            message=(
                "inputs['blocks'] is required; expected a list of "
                "Courseforge Block instances."
            ),
        )
    if not isinstance(raw, list):
        return [], GateIssue(
            severity="critical",
            code="INVALID_BLOCKS_INPUT",
            message=(
                f"inputs['blocks'] must be a list; got {type(raw).__name__}."
            ),
        )
    return list(raw), None


def _extract_assessment_surface(block: Block) -> Optional[str]:
    """Return ``stem + " " + answer_key`` from the outline-tier dict
    shape, or ``None`` when the block isn't an outline-tier assessment.

    Outline-tier assessment_item blocks carry ``stem`` + ``answer_key``
    in ``content`` per the per-block-type JSON schema in
    ``Courseforge/generators/_outline_provider.py:411-415``.
    Rewrite-tier blocks carry an HTML string in ``content``; we strip
    tags via the same lightweight helper as the inter_tier_gates
    adapters and use the full visible text as the embedding surface.
    """
    content = block.content
    if isinstance(content, dict):
        stem = content.get("stem") or ""
        answer = content.get("answer_key") or ""
        surface = f"{stem} {answer}".strip()
        return surface or None
    if isinstance(content, str):
        # Reuse the same regex-based tag stripper the inter_tier_gates
        # adapters use; avoids pulling BeautifulSoup into this validator.
        import re

        text = re.sub(r"<[^>]+>", " ", content)
        text = re.sub(r"\s+", " ", text).strip()
        return text or None
    return None


def _resolve_objective_ids(block: Block) -> List[str]:
    """Return the canonical objective_id list a block declares.

    Mirrors ``_extract_objective_refs_from_block`` in the
    inter_tier_gates module: prefers the structural
    ``Block.objective_ids`` tuple, falls back to ``content["objective_ids"]``
    on outline-tier dicts.
    """
    structural = list(block.objective_ids or ())
    if structural:
        return structural
    content = block.content
    if isinstance(content, dict):
        raw = content.get("objective_ids") or content.get("objective_refs") or []
        return [o for o in raw if isinstance(o, str) and o]
    return []


class ObjectiveAssessmentSimilarityValidator:
    """Phase 4 Wave N2 — embedding-tier objective/assessment alignment gate.

    Validator-protocol-compatible class designed to be wired into
    ``inter_tier_validation`` (Subtask 17) and ``post_rewrite_validation``
    (Subtask 18). Severity is intentionally ``warning`` per Wave N2's
    PoC contract — the structural Phase-3
    ``BlockPageObjectivesValidator`` remains authoritative; this gate
    surfaces semantic-drift signals that the structural validator
    cannot detect.
    """

    name = "objective_assessment_similarity"
    version = "0.1.0"  # Phase 4 Wave N2 PoC

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        embedder: Optional[SentenceEmbedder] = None,
    ) -> None:
        self._threshold = threshold
        # Lazy-load on validate() so test injections take precedence
        # and a slim install doesn't pay the import cost at construction.
        self._embedder_override = embedder

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture")
        threshold = float(inputs.get("threshold", self._threshold))

        blocks, err = _coerce_blocks(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="regenerate",
            )

        assessments = [
            b for b in blocks if getattr(b, "block_type", None) == "assessment_item"
        ]
        if not assessments:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        from lib.embedding.sentence_embedder import is_strict_mode
        embedder_strict = is_strict_mode()

        embedder = self._embedder_override or try_load_embedder()
        if embedder is None:
            # Graceful-degrade — emit one passed=True capture per
            # audited assessment so the silent-degrade C5 signal lands
            # in the audit trail, then return the single warning gate
            # result.
            for block in assessments:
                _emit_decision(
                    capture,
                    block_id=block.block_id,
                    passed=True,
                    code="EMBEDDING_DEPS_MISSING",
                    cosine=None,
                    threshold=threshold,
                    objective_id=None,
                    objective_count=len(_resolve_objective_ids(block)),
                    embedder_strict=embedder_strict,
                )
            # Wave N1 graceful-degrade contract — warning, passed=True,
            # no action. Mirrors the SHACL-deps-missing path in
            # courseforge_outline_shacl.py:299-317.
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="warning",
                        code="EMBEDDING_DEPS_MISSING",
                        message=(
                            "sentence-transformers extras not installed; "
                            "Phase 4 PoC objective/assessment similarity "
                            "gate skipped. Install via "
                            "`pip install -e .[embedding]` to enable."
                        ),
                    )
                ],
            )

        objective_statements: Dict[str, str] = inputs.get(
            "objective_statements", {}
        ) or {}
        if not isinstance(objective_statements, dict):
            objective_statements = {}

        issues: List[GateIssue] = []
        per_pair_results: List[Tuple[str, str, float]] = []
        passed_count = 0
        audited = 0
        low_similarity_count = 0

        for block in assessments:
            audited += 1
            surface = _extract_assessment_surface(block)
            if not surface:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="ASSESSMENT_SURFACE_EMPTY",
                            message=(
                                f"Block {block.block_id!r} declares "
                                f"block_type='assessment_item' but carries no "
                                f"stem+answer_key text surface; embedding "
                                f"similarity skipped for this block."
                            ),
                            location=block.block_id,
                        )
                    )
                _emit_decision(
                    capture, block_id=block.block_id, passed=True,
                    code="ASSESSMENT_SURFACE_EMPTY", cosine=None,
                    threshold=threshold, objective_id=None,
                    objective_count=0, embedder_strict=embedder_strict,
                )
                continue

            obj_ids = _resolve_objective_ids(block)
            if not obj_ids:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="ASSESSMENT_NO_OBJECTIVE_REFS",
                            message=(
                                f"Block {block.block_id!r} declares no "
                                f"objective_ids; embedding similarity "
                                f"requires at least one declared objective."
                            ),
                            location=block.block_id,
                        )
                    )
                _emit_decision(
                    capture, block_id=block.block_id, passed=True,
                    code="ASSESSMENT_NO_OBJECTIVE_REFS", cosine=None,
                    threshold=threshold, objective_id=None,
                    objective_count=0, embedder_strict=embedder_strict,
                )
                continue

            try:
                stem_vec = embedder.encode(surface)
            except Exception as exc:  # noqa: BLE001 — log and degrade.
                logger.warning(
                    "embedder.encode raised on block %s: %s",
                    block.block_id,
                    exc,
                )
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="EMBEDDING_ENCODE_ERROR",
                            message=(
                                f"Failed to encode assessment surface for "
                                f"block {block.block_id!r}: {exc}"
                            ),
                            location=block.block_id,
                        )
                    )
                _emit_decision(
                    capture, block_id=block.block_id, passed=True,
                    code="EMBEDDING_ENCODE_ERROR", cosine=None,
                    threshold=threshold, objective_id=None,
                    objective_count=len(obj_ids),
                    embedder_strict=embedder_strict,
                )
                continue

            min_cos: Optional[float] = None
            min_obj: Optional[str] = None
            unresolved: List[str] = []
            for obj_id in obj_ids:
                statement = objective_statements.get(obj_id)
                if not statement or not isinstance(statement, str):
                    unresolved.append(obj_id)
                    continue
                try:
                    obj_vec = embedder.encode(statement)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "embedder.encode raised on objective %s: %s",
                        obj_id,
                        exc,
                    )
                    continue
                cos = cosine_similarity(stem_vec, obj_vec)
                per_pair_results.append((block.block_id, obj_id, cos))
                if min_cos is None or cos < min_cos:
                    min_cos = cos
                    min_obj = obj_id

            if unresolved and len(issues) < _ISSUE_LIST_CAP:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="OBJECTIVE_STATEMENT_UNRESOLVED",
                        message=(
                            f"Block {block.block_id!r} references objective_ids "
                            f"{unresolved!r} that don't appear in the supplied "
                            f"objective_statements map; semantic similarity "
                            f"skipped for those pairs."
                        ),
                        location=block.block_id,
                    )
                )

            if min_cos is None:
                # No resolvable objective statements at all — already
                # surfaced via the unresolved-warnings issue above.
                _emit_decision(
                    capture, block_id=block.block_id, passed=True,
                    code="OBJECTIVE_STATEMENT_UNRESOLVED", cosine=None,
                    threshold=threshold, objective_id=None,
                    objective_count=len(obj_ids),
                    embedder_strict=embedder_strict,
                )
                continue

            if min_cos < threshold:
                low_similarity_count += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="ASSESSMENT_OBJECTIVE_LOW_SIMILARITY",
                            message=(
                                f"Block {block.block_id!r} (assessment) has "
                                f"cosine similarity {min_cos:.4f} with its "
                                f"weakest declared objective {min_obj!r}, "
                                f"below threshold {threshold:.4f}. The "
                                f"assessment surface may not actually probe "
                                f"the declared learning outcome."
                            ),
                            location=block.block_id,
                            suggestion=(
                                "Re-roll the rewrite-tier provider with the "
                                "objective statement injected verbatim into "
                                "the prompt, or revise the block's "
                                "objective_ids to match what the stem "
                                "actually probes."
                            ),
                        )
                    )
                _emit_decision(
                    capture, block_id=block.block_id, passed=False,
                    code="ASSESSMENT_OBJECTIVE_LOW_SIMILARITY",
                    cosine=min_cos, threshold=threshold,
                    objective_id=min_obj,
                    objective_count=len(obj_ids),
                    embedder_strict=embedder_strict,
                )
            else:
                passed_count += 1
                _emit_decision(
                    capture, block_id=block.block_id, passed=True,
                    code=None, cosine=min_cos, threshold=threshold,
                    objective_id=min_obj,
                    objective_count=len(obj_ids),
                    embedder_strict=embedder_strict,
                )

        # Compute aggregate score: passing assessments / audited
        # assessments. Mirrors the inter_tier_gates per-block scoring
        # so downstream score-aggregation reads cleanly.
        score = 1.0 if audited == 0 else round(passed_count / audited, 4)
        passed = low_similarity_count == 0
        action: Optional[str] = None
        if low_similarity_count > 0:
            action = "regenerate"

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=action,
        )


__all__ = [
    "ObjectiveAssessmentSimilarityValidator",
    "DEFAULT_THRESHOLD",
]

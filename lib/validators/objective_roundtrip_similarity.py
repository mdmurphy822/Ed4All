"""Phase 4 Wave N2 — Category C embedding validator (Subtask 16).

Validates that every ``objective`` Block survives a paraphrase round-trip
without semantic drift. The validator dispatches a paraphrase request
through an injected ``paraphrase_fn`` (defaulting to the rewrite-tier
router), then embeds both the original objective statement and the
paraphrase, and computes cosine similarity. A drop below the threshold
(default 0.70) emits ``action="regenerate"`` because the objective
statement is unstable under paraphrase — typically a sign of vague
verbs, ambiguous targets, or LO statements that conflate two distinct
outcomes.

The roundtrip threshold is intentionally higher than Subtask 14
(0.55) and Subtask 15 (0.50): a paraphrase MUST preserve meaning, so
the cosine floor for "still about the same thing" sits well above the
floor for "topically related". The 0.70 default is calibrated against
the all-MiniLM-L6-v2 model — paraphrases that preserve meaning cluster
above 0.78; paraphrases that drift cluster below 0.65.

Dispatch surface (paraphrase_fn injection point):
    The validator's responsibility ends at the cosine-similarity
    check. Calling code (workflow runner / test) supplies a
    ``paraphrase_fn(text: str) -> Optional[str]`` callable. The plan
    permits a stub paraphrase function as the dispatch surface (per
    the §Blockers escalation) — production wiring threads this
    through the rewrite-tier ``CourseforgeRouter`` via the optional
    ``router`` constructor argument and the
    ``_paraphrase_via_router`` adapter helper. When ``paraphrase_fn``
    is None and no ``router`` is supplied, the validator returns a
    warning issue (``PARAPHRASE_NOT_CONFIGURED``) per Wave N1's
    fallback contract — same behavior as the embedding-deps-missing
    path.

Inputs (``inputs`` dict):

    blocks: List[Block]
        Outline- or rewrite-tier ``Courseforge.scripts.blocks.Block``
        instances. Only ``block.block_type == "objective"`` rows are
        audited; other block types are skipped.

    threshold: Optional[float]
        Override the default cosine-similarity floor (0.70).

    embedder: Optional[SentenceEmbedder]
        Test injection point for the embedding backend.

    paraphrase_fn: Optional[Callable[[str], Optional[str]]]
        Override the validator's per-instance ``paraphrase_fn``. Useful
        for per-call mocking without rebuilding the validator.

    gate_id: Optional[str]
        Override for ``GateResult.gate_id``.

Behavior contract (Wave N1 fallback policy mirror):
    - Missing ``blocks`` input -> ``passed=False``, single critical
      issue, ``action="regenerate"``.
    - No objective blocks -> ``passed=True``, ``action=None``, no
      issues.
    - Embedding extras missing -> warning issue
      (``EMBEDDING_DEPS_MISSING``), ``passed=True``, ``action=None``.
    - paraphrase_fn missing AND no router -> warning issue
      (``PARAPHRASE_NOT_CONFIGURED``), ``passed=True``, ``action=None``.
    - paraphrase dispatch fails per-block -> warning issue
      (``PARAPHRASE_DISPATCH_FAILED``), block skipped (no action
      change).
    - Roundtrip cosine < threshold -> ``passed=False``,
      ``action="regenerate"`` with one critical issue per low-similarity
      objective (capped at 50 entries).

References:
    - ``lib/validators/objective_assessment_similarity.py`` —
      sibling validator built in Subtask 14.
    - ``lib/validators/concept_example_similarity.py`` —
      sibling validator built in Subtask 15.
    - ``Courseforge/router/router.py::CourseforgeRouter.route`` —
      production paraphrase dispatch surface (rewrite tier).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.embedding._math import cosine_similarity
from lib.embedding.sentence_embedder import (
    SentenceEmbedder,
    try_load_embedder,
)

# Import bridge for Block (mirror of Subtasks 14 + 15).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # type: ignore[import-not-found]  # noqa: E402

logger = logging.getLogger(__name__)


#: Default cosine-similarity floor for the (original, paraphrase) pair
#: roundtrip check. See module docstring for calibration rationale.
DEFAULT_THRESHOLD: float = 0.70

#: Cap on per-block issue list (matches Subtasks 14 + 15).
_ISSUE_LIST_CAP: int = 50

#: Default rewrite-tier prompt template applied by the
#: ``_paraphrase_via_router`` helper. Kept short on purpose so any
#: rewrite-tier provider — local 7B, Together OSS, Anthropic — can
#: handle it without bespoke template wiring.
_DEFAULT_PARAPHRASE_PROMPT: str = "Paraphrase preserving meaning"


ParaphraseFn = Callable[[str], Optional[str]]


def _emit_decision(
    capture: Any,
    *,
    block_id: str,
    passed: bool,
    code: Optional[str],
    cosine: Optional[float],
    threshold: float,
    statement_len: int,
    paraphrase_len: int,
    embedder_strict: bool,
) -> None:
    """Emit one ``objective_roundtrip_similarity_check`` decision per
    audited objective block.

    Rationale interpolates dynamic signals — block_id, cosine +
    threshold + above/below flag, original-statement + paraphrase
    surface lengths, and the embedder strict-mode flag (so
    EMBEDDING_DEPS_MISSING graceful-degrade events are distinguishable
    from real low-similarity events in the audit trail).
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    cos_str = f"{cosine:.4f}" if cosine is not None else "n/a"
    above_threshold = (cosine is not None and cosine >= threshold)
    rationale = (
        f"Objective-roundtrip similarity check on Block {block_id!r}: "
        f"roundtrip_cosine={cos_str}, threshold={threshold:.4f}, "
        f"above_threshold={above_threshold}, "
        f"statement_len={statement_len}, "
        f"paraphrase_len={paraphrase_len}, "
        f"embedder_strict_mode={embedder_strict}, "
        f"failure_code={code or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="objective_roundtrip_similarity_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "objective_roundtrip_similarity_check: %s",
            exc,
        )


def _coerce_blocks(
    inputs: Dict[str, Any],
) -> Tuple[List[Block], Optional[GateIssue]]:
    """Pull a list of Block instances out of ``inputs["blocks"]``.

    Mirror of the helper in Subtasks 14 + 15.
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


def _extract_objective_statement(block: Block) -> Optional[str]:
    """Return the human-readable objective statement from an ``objective``
    Block.

    Outline-tier dict shape: prefers ``content["statement"]`` (the
    canonical field for the objective text). Falls back to
    ``content["body"]`` and ``content["text"]`` for older/heuristic
    shapes.
    Rewrite-tier str shape: strips HTML and uses the visible text.
    """
    content = block.content
    if isinstance(content, dict):
        for key in ("statement", "body", "text"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    if isinstance(content, str):
        import re

        text = re.sub(r"<[^>]+>", " ", content)
        text = re.sub(r"\s+", " ", text).strip()
        return text or None
    return None


def _paraphrase_via_router(
    router: Any,
    prompt_template: str = _DEFAULT_PARAPHRASE_PROMPT,
) -> ParaphraseFn:
    """Build a ``paraphrase_fn`` adapter on top of a CourseforgeRouter.

    Production wiring threads this through the rewrite-tier dispatch
    so the same router that authors page HTML also services the
    paraphrase round-trip. The plan documents that the validator's
    responsibility ends at the cosine-similarity check; this adapter
    keeps the router-coupling surface contained in one helper instead
    of leaking into the validator class.

    The adapter is intentionally thin: it constructs an ephemeral
    ``Block`` carrying the input text in ``content["statement"]``, then
    calls ``router.route(block, tier="rewrite", overrides={"prompt_template":
    prompt_template})``. If the router signature drifts, the adapter
    catches the exception and returns ``None`` so the validator
    surfaces ``PARAPHRASE_DISPATCH_FAILED`` for the affected block
    instead of crashing the gate.
    """

    def _adapter(text: str) -> Optional[str]:
        if not text:
            return None
        try:
            stub_block = Block(
                block_id="paraphrase_roundtrip#0",
                block_type="objective",
                page_id="paraphrase_roundtrip",
                sequence=0,
                content={"statement": text},
            )
            paraphrased = router.route(
                stub_block,
                tier="rewrite",
                overrides={"prompt_template": prompt_template},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "paraphrase router dispatch failed (%s); returning None.",
                exc,
            )
            return None

        # Pull the paraphrased text back out of whatever shape the
        # router returns. Most rewrite-tier providers emit a Block
        # with a string ``content``; some emit a dict with a
        # ``statement`` field. Defensive against both.
        if paraphrased is None:
            return None
        result_content = getattr(paraphrased, "content", None)
        if isinstance(result_content, str):
            return result_content.strip() or None
        if isinstance(result_content, dict):
            for key in ("statement", "body", "text"):
                value = result_content.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    return _adapter


class ObjectiveRoundtripSimilarityValidator:
    """Phase 4 Wave N2 — embedding-tier paraphrase-roundtrip gate.

    Validator-protocol-compatible class designed to be wired into
    ``inter_tier_validation`` (Subtask 17) and ``post_rewrite_validation``
    (Subtask 18). Severity is intentionally ``warning`` per Wave N2's
    PoC contract — no Phase-3 structural validator covers paraphrase
    stability, so this gate is purely additive.

    The validator's responsibility ends at the cosine-similarity check;
    paraphrase dispatch is delegated to an injected ``paraphrase_fn``
    callable so production wiring (rewrite-tier router) and test
    mocks (stub paraphrase function) share the same interface.
    """

    name = "objective_roundtrip_similarity"
    version = "0.1.0"  # Phase 4 Wave N2 PoC

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        embedder: Optional[SentenceEmbedder] = None,
        paraphrase_fn: Optional[ParaphraseFn] = None,
        router: Optional[Any] = None,
        prompt_template: str = _DEFAULT_PARAPHRASE_PROMPT,
    ) -> None:
        self._threshold = threshold
        self._embedder_override = embedder
        self._prompt_template = prompt_template
        # Resolution priority (high → low):
        #   1. ``paraphrase_fn`` constructor kwarg (tests / explicit override)
        #   2. ``router`` constructor kwarg (production wiring)
        #   3. None → validator emits PARAPHRASE_NOT_CONFIGURED warning.
        if paraphrase_fn is not None:
            self._paraphrase_fn: Optional[ParaphraseFn] = paraphrase_fn
        elif router is not None:
            self._paraphrase_fn = _paraphrase_via_router(router, prompt_template)
        else:
            self._paraphrase_fn = None

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

        objectives = [
            b for b in blocks if getattr(b, "block_type", None) == "objective"
        ]
        if not objectives:
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
            for block in objectives:
                stmt = _extract_objective_statement(block) or ""
                _emit_decision(
                    capture, block_id=block.block_id, passed=True,
                    code="EMBEDDING_DEPS_MISSING", cosine=None,
                    threshold=threshold,
                    statement_len=len(stmt), paraphrase_len=0,
                    embedder_strict=embedder_strict,
                )
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
                            "Phase 4 PoC objective-roundtrip similarity "
                            "gate skipped. Install via "
                            "`pip install -e .[embedding]` to enable."
                        ),
                    )
                ],
            )

        # Per-call paraphrase_fn override takes precedence over the
        # constructor-time wiring (mirrors the threshold + embedder
        # input-side overrides).
        paraphrase_fn: Optional[ParaphraseFn] = (
            inputs.get("paraphrase_fn") or self._paraphrase_fn
        )
        if paraphrase_fn is None:
            for block in objectives:
                stmt = _extract_objective_statement(block) or ""
                _emit_decision(
                    capture, block_id=block.block_id, passed=True,
                    code="PARAPHRASE_NOT_CONFIGURED", cosine=None,
                    threshold=threshold,
                    statement_len=len(stmt), paraphrase_len=0,
                    embedder_strict=embedder_strict,
                )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="warning",
                        code="PARAPHRASE_NOT_CONFIGURED",
                        message=(
                            "No paraphrase_fn supplied and no router "
                            "wired into the validator constructor; "
                            "Phase 4 PoC objective-roundtrip similarity "
                            "gate skipped."
                        ),
                        suggestion=(
                            "Construct the validator with router=<router> "
                            "or paraphrase_fn=<callable>, or pass "
                            "paraphrase_fn directly via inputs."
                        ),
                    )
                ],
            )

        issues: List[GateIssue] = []
        passed_count = 0
        audited = 0
        low_similarity_count = 0

        for block in objectives:
            audited += 1
            statement = _extract_objective_statement(block)
            if not statement:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="OBJECTIVE_STATEMENT_EMPTY",
                            message=(
                                f"Block {block.block_id!r} declares "
                                f"block_type='objective' but carries no "
                                f"statement/body text surface; roundtrip "
                                f"similarity skipped for this block."
                            ),
                            location=block.block_id,
                        )
                    )
                _emit_decision(
                    capture, block_id=block.block_id, passed=True,
                    code="OBJECTIVE_STATEMENT_EMPTY", cosine=None,
                    threshold=threshold,
                    statement_len=0, paraphrase_len=0,
                    embedder_strict=embedder_strict,
                )
                continue

            try:
                paraphrase = paraphrase_fn(statement)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "paraphrase_fn raised on block %s: %s",
                    block.block_id,
                    exc,
                )
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="PARAPHRASE_DISPATCH_FAILED",
                            message=(
                                f"paraphrase_fn raised on block "
                                f"{block.block_id!r}: {exc}"
                            ),
                            location=block.block_id,
                        )
                    )
                _emit_decision(
                    capture, block_id=block.block_id, passed=True,
                    code="PARAPHRASE_DISPATCH_FAILED", cosine=None,
                    threshold=threshold,
                    statement_len=len(statement), paraphrase_len=0,
                    embedder_strict=embedder_strict,
                )
                continue

            if not paraphrase or not isinstance(paraphrase, str):
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="PARAPHRASE_DISPATCH_FAILED",
                            message=(
                                f"paraphrase_fn returned no usable text for "
                                f"block {block.block_id!r}; cannot compute "
                                f"roundtrip similarity."
                            ),
                            location=block.block_id,
                        )
                    )
                _emit_decision(
                    capture, block_id=block.block_id, passed=True,
                    code="PARAPHRASE_DISPATCH_FAILED", cosine=None,
                    threshold=threshold,
                    statement_len=len(statement), paraphrase_len=0,
                    embedder_strict=embedder_strict,
                )
                continue

            try:
                original_vec = embedder.encode(statement)
                paraphrase_vec = embedder.encode(paraphrase)
            except Exception as exc:  # noqa: BLE001
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
                                f"Failed to encode statement/paraphrase pair "
                                f"for block {block.block_id!r}: {exc}"
                            ),
                            location=block.block_id,
                        )
                    )
                _emit_decision(
                    capture, block_id=block.block_id, passed=True,
                    code="EMBEDDING_ENCODE_ERROR", cosine=None,
                    threshold=threshold,
                    statement_len=len(statement),
                    paraphrase_len=len(paraphrase),
                    embedder_strict=embedder_strict,
                )
                continue

            cos = cosine_similarity(original_vec, paraphrase_vec)

            if cos < threshold:
                low_similarity_count += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="OBJECTIVE_ROUNDTRIP_LOW_SIMILARITY",
                            message=(
                                f"Block {block.block_id!r} (objective) has "
                                f"paraphrase-roundtrip cosine similarity "
                                f"{cos:.4f}, below threshold {threshold:.4f}. "
                                f"The objective statement is unstable under "
                                f"paraphrase — typically a sign of vague "
                                f"verbs, ambiguous targets, or a statement "
                                f"that conflates two distinct outcomes."
                            ),
                            location=block.block_id,
                            suggestion=(
                                "Re-roll the rewrite-tier provider with "
                                "guidance to use a single Bloom verb + a "
                                "concrete observable outcome; split the "
                                "objective if it conflates two outcomes."
                            ),
                        )
                    )
                _emit_decision(
                    capture, block_id=block.block_id, passed=False,
                    code="OBJECTIVE_ROUNDTRIP_LOW_SIMILARITY",
                    cosine=cos, threshold=threshold,
                    statement_len=len(statement),
                    paraphrase_len=len(paraphrase),
                    embedder_strict=embedder_strict,
                )
            else:
                passed_count += 1
                _emit_decision(
                    capture, block_id=block.block_id, passed=True,
                    code=None, cosine=cos, threshold=threshold,
                    statement_len=len(statement),
                    paraphrase_len=len(paraphrase),
                    embedder_strict=embedder_strict,
                )

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
    "ObjectiveRoundtripSimilarityValidator",
    "DEFAULT_THRESHOLD",
    "ParaphraseFn",
    "_paraphrase_via_router",
]

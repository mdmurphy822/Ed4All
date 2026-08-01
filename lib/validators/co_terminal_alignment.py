"""WS3 — CO↔TO semantic-alignment detection gate (course_planning).

The structural ``terminal_objective_coverage`` gate is a *roll-up* check
only — "does every TO have ≥1 CO pointing at it / does every TO's chapter
resolve?". It NEVER measures whether the CO→TO assignment is *semantically*
meaningful. On a real broken full-book run all 62 COs carried a populated
``terminal_id`` (force-assigned floorlessly by
``lib/ontology/lo_backlink.py::backlink_cos_to_tos``), so the structural
gate passed via the genuine orphan-check path even though the assignments
were near-random — a silent-pass loophole.

This validator closes that loophole by recomputing
``cosine(co.statement, assigned_to.statement)`` per chapter objective and
flagging every link below a floor (``CO_TERMINAL_SEMANTIC_MISMATCH``), plus
a corpus-level warning when the recomputed weak-link rate exceeds a floor
(``CO_TERMINAL_WEAK_LINK_RATE_HIGH``). It is **detection, NOT the cure**
(WS1's bottom-up TO derivation is the cure). On the current broken run it
fires on ~80%+ of COs, so it lands **warning-severity** day-1 with a
``# TODO(calibration)`` deferred critical flip (see §2.3 of
``plans/finegrain/obj-synth-ws3-alignment-gate.md``).

Mirror (embedding-validator template):
``lib/validators/objective_assessment_similarity.py`` — the
``try_load_embedder()`` + ``EMBEDDING_DEPS_MISSING`` graceful-degrade,
``is_strict_mode()``, ``_emit_decision``, ``_ISSUE_LIST_CAP=50``,
``score=passed/audited``. Objectives IO mirrors
``terminal_objective_coverage.py`` (``_load_json`` / ``_coerce_dict``) +
``lib/ontology/terminal_coverage.py::flatten_chapter_objectives``.

Two failure surfaces that stay VISIBLY DISTINCT: missing ``[embedding]``
extras degrades to the warning ``EMBEDDING_DEPS_MISSING`` with
``passed=True`` (fail-closed only under ``TRAINFORGE_REQUIRE_EMBEDDINGS``),
whereas extras-present-but-requested-``ED4ALL_EMBEDDING_DEVICE``-absent
raises :class:`EmbeddingModelUnavailable` FATALLY regardless of that flag.
The embedder is ``preload()``-ed before the audit loop and the per-encode
guards re-raise the typed error, so a device failure can never be
downgraded to a passing ``EMBEDDING_ENCODE_ERROR``.

Inputs (``inputs`` dict, fed by the EXISTING
``_build_chapter_objective_coverage_inputs`` builder — no new builder):

    synthesized_objectives_path (str) OR synthesized_objectives (dict)
        The course's synthesized objectives doc. Resolved via the
        ``_coerce_dict`` helper (inline dict for test ergonomics, path
        for the live builder). Unresolvable / empty → graceful no-op pass.

    textbook_structure_path / textbook_structure
        Accepted for builder-shape compat but UNUSED (CO/TO statements are
        self-contained). Accept-and-ignore.

    threshold (float, TOP-LEVEL config key)
        Per-CO cosine floor (default 0.45). The executor merges the gate's
        top-level ``config`` block into ``inputs`` via shallow
        ``setdefault`` (``MCP/hardening/validation_gates.py``), so these
        are read as ``inputs.get("threshold")`` — a NESTED
        ``config.thresholds.*`` block would NOT be read.

    weak_rate_warn (float, TOP-LEVEL config key)
        Corpus-level weak-link-rate warn floor (default 0.30).

    embedder (Optional[SentenceEmbedder])
        Test seam. Defaults to ``try_load_embedder()``.

    decision_capture / capture
        Injected by the executor. One ``co_terminal_alignment_check``
        decision per audited CO.

References:
    - ``lib/validators/objective_assessment_similarity.py`` — class template.
    - ``lib/validators/terminal_objective_coverage.py`` — objectives IO.
    - ``lib/ontology/terminal_coverage.py`` — flatten + parent-terminal keys.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.embedding._math import cosine_similarity
from lib.embedding.sentence_embedder import (
    EmbeddingModelUnavailable,
    SentenceEmbedder,
    is_strict_mode,
    try_load_embedder,
)
from lib.ontology.terminal_coverage import (
    _PARENT_TERMINAL_KEYS,
    flatten_chapter_objectives,
)
from lib.validators.terminal_objective_coverage import _coerce_dict

logger = logging.getLogger(__name__)

#: Per-CO cosine floor below which a (CO, assigned-TO) link is flagged as
#: semantically misaligned. Matches the WS2 ``ED4ALL_TO_BACKLINK_FLOOR``
#: default — a CO whose statement is below this cosine to its assigned TO
#: statement is unlikely to genuinely roll up to it.
DEFAULT_THRESHOLD: float = 0.45

#: Corpus-level weak-link-rate floor above which the gate emits the
#: ``CO_TERMINAL_WEAK_LINK_RATE_HIGH`` aggregate warning.
DEFAULT_WEAK_RATE_WARN: float = 0.30

#: Cap on the emitted issue list so a uniformly-misaligned objectives set
#: doesn't drown the gate report (matches
#: ``objective_assessment_similarity.py::_ISSUE_LIST_CAP``).
_ISSUE_LIST_CAP: int = 50

#: Validator-capture decision_type — NEW enum member (WS3).
_DECISION_TYPE: str = "co_terminal_alignment_check"


def _emit_decision(
    capture: Any,
    *,
    co_id: str,
    assigned_terminal: Optional[str],
    passed: bool,
    code: Optional[str],
    cosine: Optional[float],
    threshold: float,
    weak: bool,
    used_ws2: bool,
    embedder_strict: bool,
) -> None:
    """Emit one ``co_terminal_alignment_check`` decision per audited CO.

    Rationale interpolates dynamic signals (≥20 chars) so captures are
    replayable post-hoc: the CO id, its assigned terminal, the recomputed
    CO→TO cosine + threshold + above/below flag, the weak flag, whether the
    WS2 stamped-score fast-path was used, the embedder strict-mode flag (so
    EMBEDDING_DEPS_MISSING degrade events are distinguishable from real
    low-cosine events), and the failure code.
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    cos_str = f"{cosine:.4f}" if cosine is not None else "n/a"
    above_threshold = cosine is not None and cosine >= threshold
    rationale = (
        f"CO/TO semantic-alignment check on CO {co_id!r}: "
        f"assigned_terminal={assigned_terminal or 'n/a'}, "
        f"co_to_cosine={cos_str}, threshold={threshold:.4f}, "
        f"above_threshold={above_threshold}, weak={weak}, "
        f"used_ws2_flag={used_ws2}, "
        f"embedder_strict_mode={embedder_strict}, "
        f"failure_code={code or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "co_terminal_alignment_check: %s",
            exc,
        )


def _resolve_assigned_terminal(co: Dict[str, Any]) -> Optional[str]:
    """Return the lower-cased terminal id a CO is assigned to, or None.

    Mirrors ``terminal_coverage.co_parent_terminal`` (the canonical
    parent-terminal key precedence) but kept inline so the validator owns
    its skip semantics.
    """
    for key in _PARENT_TERMINAL_KEYS:
        val = co.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return None


def _co_statement(co: Dict[str, Any]) -> str:
    """Return the CO statement text, tolerating common field names."""
    for key in ("statement", "text", "objective", "description"):
        val = co.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


class CoTerminalAlignmentValidator:
    """WS3 — CO↔TO semantic-alignment detection gate at ``course_planning``.

    Validator-protocol-compatible class wired into
    ``textbook_to_course::course_planning::co_terminal_alignment``
    (warning / warn). Recompute is PRIMARY: it embeds each CO statement +
    its assigned TO statement, computes cosine, and derives the weak-link
    rate from the recomputed scores. The WS2 stamped-score read
    (``terminal_link_score``) is an OPTIONAL fast-path (dead until WS2
    lands); recompute is authoritative.

    Day-1 ALL issues are warning-severity → ``passed`` is always True
    (``critical-count == 0``), keeping the ~80%-weak broken run from
    hard-blocking. The critical flip is DEFERRED (``# TODO(calibration)``).
    """

    name = "co_terminal_alignment"
    version = "0.1.0"

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        weak_rate_warn: float = DEFAULT_WEAK_RATE_WARN,
        embedder: Optional[SentenceEmbedder] = None,
    ) -> None:
        self._threshold = threshold
        self._weak_rate_warn = weak_rate_warn
        # Lazy-load on validate() so test injections take precedence.
        self._embedder_override = embedder

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or inputs.get("capture")
        threshold = float(inputs.get("threshold", self._threshold))
        weak_rate_warn = float(
            inputs.get("weak_rate_warn", self._weak_rate_warn)
        )

        objectives, err = _coerce_dict(
            inputs,
            explicit_key="synthesized_objectives",
            path_key="synthesized_objectives_path",
            code_prefix="CO_TERMINAL_ALIGNMENT_OBJECTIVES",
        )
        # Unresolvable objectives input → graceful no-op pass (this gate is
        # detection-only; other gates own malformed-objectives). We do NOT
        # propagate the critical _coerce_dict error code — WS3 never blocks.
        if err is not None or objectives is None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        terminals = (
            objectives.get("terminal_objectives")
            or objectives.get("terminal_outcomes")
            or []
        )
        terminals = [t for t in terminals if isinstance(t, dict)]
        chapter_objectives = (
            objectives.get("chapter_objectives")
            or objectives.get("component_objectives")
            or []
        )
        cos = flatten_chapter_objectives(chapter_objectives)

        # Empty terminals OR empty chapter objectives → nothing to align.
        if not terminals or not cos:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        to_by_id: Dict[str, str] = {}
        for to in terminals:
            tid = to.get("id")
            if isinstance(tid, str) and tid.strip():
                stmt = _co_statement(to)
                to_by_id[tid.strip().lower()] = stmt

        embedder_strict = is_strict_mode()
        embedder = self._embedder_override or try_load_embedder()
        if embedder is None:
            # Graceful-degrade — emit one passed=True capture per CO so the
            # silent-degrade signal lands in the audit trail, then return
            # the single warning gate result. Mirrors
            # objective_assessment_similarity.py.
            for co in cos:
                co_id = str(co.get("id") or "?")
                _emit_decision(
                    capture,
                    co_id=co_id,
                    assigned_terminal=_resolve_assigned_terminal(co),
                    passed=True,
                    code="EMBEDDING_DEPS_MISSING",
                    cosine=None,
                    threshold=threshold,
                    weak=False,
                    used_ws2=False,
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
                            "WS3 CO<->TO semantic-alignment gate skipped. "
                            "Install via `pip install -e .[embedding]` to "
                            "enable."
                        ),
                    )
                ],
            )

        # Force the model load NOW so a device failure lands HERE, fatal.
        # EmbeddingModelUnavailable (extras present, the requested
        # ED4ALL_EMBEDDING_DEVICE absent) must never reach the per-encode
        # `except Exception` below, which would degrade it to a
        # warning-severity EMBEDDING_ENCODE_ERROR with passed=True and pass
        # this gate vacuously. Distinct from the missing-extras contract
        # above, which still degrades to a warning by default. ``getattr``
        # because the ``_embedder_override`` test seam injects encode-only
        # stubs; the per-encode re-raise below is the backstop.
        _preload = getattr(embedder, "preload", None)
        if callable(_preload):
            _preload()

        issues: List[GateIssue] = []
        # Memoize encode(to_stmt) by assigned_tid — 62 COs typically share
        # ~3 TOs, so avoid redundant encodes.
        to_vec_cache: Dict[str, Any] = {}
        audited = 0
        weak_count = 0
        passed_count = 0

        for co in cos:
            co_id = str(co.get("id") or "?")
            assigned_tid = _resolve_assigned_terminal(co)
            if not assigned_tid or assigned_tid not in to_by_id:
                # Missing / unresolved terminal assignment — owned by
                # terminal_objective_coverage; don't double-fail. Warn + skip
                # (NOT counted weak).
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="CO_NO_ASSIGNED_TERMINAL",
                            message=(
                                f"Chapter objective {co_id!r} carries no "
                                f"resolvable parent-terminal "
                                f"(parent_to/parent_terminal/terminal_id); "
                                f"CO<->TO cosine alignment skipped. Terminal "
                                f"assignment is owned by "
                                f"terminal_objective_coverage."
                            ),
                            location=f"chapter_objectives[id={co_id}]",
                        )
                    )
                _emit_decision(
                    capture,
                    co_id=co_id,
                    assigned_terminal=assigned_tid,
                    passed=True,
                    code="CO_NO_ASSIGNED_TERMINAL",
                    cosine=None,
                    threshold=threshold,
                    weak=False,
                    used_ws2=False,
                    embedder_strict=embedder_strict,
                )
                continue

            audited += 1

            # [Optional WS2 fast-path] — read a stamped terminal_link_score
            # instead of re-embedding. Dead until WS2 lands (backlink_cos_to_tos
            # stamps no such field today); recompute below is authoritative.
            used_ws2 = False
            cos_val: Optional[float] = None
            stamped = co.get("terminal_link_score")
            if isinstance(stamped, (int, float)) and not isinstance(
                stamped, bool
            ):
                cos_val = float(stamped)
                used_ws2 = True

            if cos_val is None:
                co_stmt = _co_statement(co)
                to_stmt = to_by_id[assigned_tid]
                to_vec = to_vec_cache.get(assigned_tid)
                if to_vec is None:
                    try:
                        to_vec = embedder.encode(to_stmt)
                    except EmbeddingModelUnavailable:
                        # Typed device unavailability is FATAL.
                        raise
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "embedder.encode raised on terminal %s: %s",
                            assigned_tid,
                            exc,
                        )
                        to_vec = None
                    to_vec_cache[assigned_tid] = to_vec
                if to_vec is None:
                    # Encode failed — skip scoring this CO (already logged).
                    _emit_decision(
                        capture,
                        co_id=co_id,
                        assigned_terminal=assigned_tid,
                        passed=True,
                        code="EMBEDDING_ENCODE_ERROR",
                        cosine=None,
                        threshold=threshold,
                        weak=False,
                        used_ws2=False,
                        embedder_strict=embedder_strict,
                    )
                    passed_count += 1
                    continue
                try:
                    co_vec = embedder.encode(co_stmt)
                except EmbeddingModelUnavailable:
                    # Typed device unavailability is FATAL.
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "embedder.encode raised on CO %s: %s", co_id, exc
                    )
                    _emit_decision(
                        capture,
                        co_id=co_id,
                        assigned_terminal=assigned_tid,
                        passed=True,
                        code="EMBEDDING_ENCODE_ERROR",
                        cosine=None,
                        threshold=threshold,
                        weak=False,
                        used_ws2=False,
                        embedder_strict=embedder_strict,
                    )
                    passed_count += 1
                    continue
                cos_val = cosine_similarity(co_vec, to_vec)

            weak = cos_val < threshold
            if weak:
                weak_count += 1
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="CO_TERMINAL_SEMANTIC_MISMATCH",
                            message=(
                                f"Chapter objective {co_id!r} has cosine "
                                f"similarity {cos_val:.4f} with its assigned "
                                f"terminal {assigned_tid!r}, below threshold "
                                f"{threshold:.4f}. The CO may not genuinely "
                                f"roll up to that terminal."
                            ),
                            location=f"chapter_objectives[id={co_id}]",
                            suggestion=(
                                "Re-derive TOs bottom-up (WS1) or raise "
                                "ED4ALL_TO_BACKLINK_FLOOR so weakly-linked "
                                "COs are not force-assigned to a "
                                "semantically-distant terminal."
                            ),
                        )
                    )
                _emit_decision(
                    capture,
                    co_id=co_id,
                    assigned_terminal=assigned_tid,
                    passed=False,
                    code="CO_TERMINAL_SEMANTIC_MISMATCH",
                    cosine=cos_val,
                    threshold=threshold,
                    weak=True,
                    used_ws2=used_ws2,
                    embedder_strict=embedder_strict,
                )
            else:
                passed_count += 1
                _emit_decision(
                    capture,
                    co_id=co_id,
                    assigned_terminal=assigned_tid,
                    passed=True,
                    code=None,
                    cosine=cos_val,
                    threshold=threshold,
                    weak=False,
                    used_ws2=used_ws2,
                    embedder_strict=embedder_strict,
                )

        weak_to_link_rate = (weak_count / audited) if audited else 0.0
        if audited and weak_to_link_rate > weak_rate_warn:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="CO_TERMINAL_WEAK_LINK_RATE_HIGH",
                    message=(
                        f"{weak_count}/{audited} chapter objective(s) "
                        f"({weak_to_link_rate:.2%}) link to their assigned "
                        f"terminal below cosine {threshold:.4f}, exceeding "
                        f"the warn floor {weak_rate_warn:.2%}. WS1 (bottom-up "
                        f"TO derivation) is the cure; this gate is detection "
                        f"only."
                    ),
                    location="chapter_objectives",
                )
            )

        # Day-1 ALL issues warning-severity → passed always True.
        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        score = 1.0 if audited == 0 else round(passed_count / audited, 4)

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=None,
        )


__all__ = [
    "CoTerminalAlignmentValidator",
    "DEFAULT_THRESHOLD",
    "DEFAULT_WEAK_RATE_WARN",
]

"""Worker W4.B — PairLearningOutcomeRefsValidator.

GPT Feedback v2 Wave 4 net-new validator. Closes Gap B in the Wave 4
symmetric-validation investigation: per-pair ``learning_outcome_refs[]``
must be a subset of the cited source chunk's ``learning_outcome_refs[]``.

Surface mirror of three precedents:

* :class:`lib.validators.assessment_objective_alignment.AssessmentObjectiveAlignmentValidator`
  (Trainforge-side LO-resolution analog) — `_collect_chunk_refs`
  case-insensitive normalization pattern (line 341-369). Trainforge
  emits chunk refs lowercase; Courseforge emits ``TO-NN`` / ``CO-NN``
  mixed-case. Comparison is case-insensitive while preserving emit case
  in the audit field.
* :class:`lib.validators.objective_source_refs.ObjectiveSourceRefValidator`
  (Courseforge-side objective-LO chunk-id resolution) — W4.B is the
  pair-side mirror.
* :class:`lib.validators.pair.promotion.TrainingPairPromotionValidator.validate_pair`
  (line 494-781) — same per-pair filter surface signature; mirrors the
  ``(promotion_status, rejection_reason, new_fields)`` return triple.

Two surfaces:

1. :meth:`validate_pair` — per-pair pre-write filter wired into
   ``Trainforge/synthesize_training.py`` AFTER W4.A's claim-support
   check. Returns ``(promotion_status, rejection_reason, new_fields)``
   where ``new_fields`` carries a ``pair_lo_resolution`` audit-trail
   payload.
2. :meth:`validate` — gate-runner surface walking
   ``training_specs/{instruction,preference}_pairs.jsonl`` and
   ``imscc_chunks/chunks.jsonl`` (or the legacy ``chunks.jsonl``).
   Critical-fails on any phantom pair LO ref.

Reject precedence:

* ``missing_pair_lo_refs`` — fires first when ``require_non_empty=True``
  and ``pair.learning_outcome_refs`` is empty / unset. Reasoning: a
  pair with no declared LO can't be subset-checked, and the schema's
  ``trainable``-conditional requires the field. Catching missing-LO
  before phantom-LO produces a more actionable rejection_reason for
  the operator (the upstream emitter forgot the field) vs.
  potentially attributing the failure to an empty intersection.
* ``phantom_pair_lo_refs`` — fires when the pair declares one or more
  LO refs that the cited chunk does not carry. The ``phantom_los``
  audit-trail field lists the offending refs (in their original
  emit-case) so post-hoc replay can identify the disjoint-LO-scheme
  failure mode.

Decision capture: every :meth:`validate_pair` invocation emits exactly
one ``pair_lo_refs_check`` event when a capture is provided. Rationale
interpolates ``pair_id`` (or ``chunk_id`` when no pair_id), ``kind``,
``declared_los``, ``chunk_los``, ``phantom_los``, ``resolved``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


def _normalize_lo_set(refs: Any) -> Tuple[List[str], Set[str]]:
    """Return ``(original_case_list, lowercase_set)`` for a learning_outcome_refs field.

    Filters None values and non-string entries defensively. Preserves
    emit order in the original-case list so the audit-trail field
    matches what the upstream emitter produced; the lowercase set is
    used for the subset comparison only.
    """
    out_orig: List[str] = []
    out_lower: Set[str] = set()
    if not isinstance(refs, list):
        return out_orig, out_lower
    seen: Set[str] = set()
    for r in refs:
        if not isinstance(r, str):
            continue
        s = r.strip()
        if not s:
            continue
        lower = s.lower()
        if lower in seen:
            continue
        seen.add(lower)
        out_orig.append(s)
        out_lower.add(lower)
    return out_orig, out_lower


def _emit_pair_lo_decision(
    capture: Any,
    *,
    pair_id: str,
    kind: str,
    declared_los: List[str],
    chunk_los: List[str],
    phantom_los: List[str],
    resolved: bool,
    rejection_reason: Optional[str],
) -> None:
    """Emit one ``pair_lo_refs_check`` decision per :meth:`validate_pair` call.

    Mirrors :func:`lib.validators.assessment_objective_alignment._emit_alignment_decision`
    rationale shape: identifier + declared/observed sets + resolved flag.
    Rationale ≥20 chars and interpolates dynamic per-pair signals.
    """
    if capture is None:
        return
    decision = "passed" if resolved else f"failed:{rejection_reason or 'unknown'}"
    rationale = (
        f"PairLearningOutcomeRefsValidator audited {kind} pair "
        f"{pair_id!r}: declared_los={declared_los!r}, "
        f"chunk_los={chunk_los!r}, phantom_los={phantom_los!r}, "
        f"resolved={resolved}, rejection_reason={rejection_reason or 'none'}."
    )
    metrics: Dict[str, Any] = {
        "pair_id": pair_id,
        "kind": kind,
        "declared_los": list(declared_los),
        "chunk_los": list(chunk_los),
        "phantom_los": list(phantom_los),
        "resolved": bool(resolved),
        "rejection_reason": rejection_reason,
    }
    try:
        capture.log_decision(
            decision_type="pair_lo_refs_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "pair_lo_refs_check: %s",
            exc,
        )


class PairLearningOutcomeRefsValidator:
    """Per-pair LO-refs subset gate.

    Asserts every emitted training pair's ``learning_outcome_refs[]``
    is a subset of the cited source chunk's ``learning_outcome_refs[]``
    under case-insensitive comparison.
    """

    name = "pair_lo_refs"
    version = "1.0.0"

    def __init__(
        self,
        *,
        require_non_empty: bool = True,
        decision_capture: Any = None,
    ) -> None:
        self._require_non_empty = bool(require_non_empty)
        self._decision_capture = decision_capture

    # ------------------------------------------------------------------ #
    # Per-pair filter — call site in synthesize_training.py
    # ------------------------------------------------------------------ #

    def validate_pair(
        self,
        pair: Dict[str, Any],
        *,
        kind: str,
        chunk: Optional[Dict[str, Any]] = None,
        decision_capture: Any = None,
    ) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """Audit one pair against the per-pair LO-subset rule.

        Returns ``(promotion_status, rejection_reason, new_fields)``:

        - ``promotion_status`` is ``"validated"`` on pass and
          ``"rejected"`` on any rejection trigger.
        - ``rejection_reason`` is one of ``"missing_pair_lo_refs"`` or
          ``"phantom_pair_lo_refs"`` on reject; ``None`` on pass.
        - ``new_fields`` always carries ``pair_lo_resolution``:
          ``{"declared_los": [...], "chunk_los": [...],
          "phantom_los": [...]}``. Lists are in original emit-case
          order so the audit trail records exactly what the upstream
          emitter produced. The caller MUST ``pair.update(new_fields)``
          to land the audit fields regardless of pass/fail.
        """
        capture = decision_capture or self._decision_capture

        # Pair-side canonical field is ``lo_refs`` (per
        # ``schemas/knowledge/instruction_pair.schema.json`` line 36 +
        # ``preference_pair.schema.json`` line 44 — both list ``lo_refs``
        # as required). ``synthesize_training.py:529-530`` populates
        # both ``lo_refs`` and ``learning_outcome_refs`` at pair-build
        # time, but downstream projection/serialisation may strip the
        # latter (the canonical schema doesn't carry it). Read both
        # keys with ``lo_refs`` taking precedence so this validator
        # works whether the upstream emit drops the legacy alias or
        # not. Chunk-side stays ``learning_outcome_refs`` per the
        # canonical chunk schema (``chunk_v4.schema.json``).
        pair_lo_field = (
            pair.get("lo_refs") or pair.get("learning_outcome_refs")
        )
        pair_los_orig, pair_los_lower = _normalize_lo_set(pair_lo_field)
        chunk_payload = chunk if isinstance(chunk, dict) else {}
        chunk_los_orig, chunk_los_lower = _normalize_lo_set(
            chunk_payload.get("learning_outcome_refs")
        )

        # Compute phantom set in lowercase, then project back to
        # original emit case from the pair's declared list so the audit
        # field surfaces what the emitter wrote.
        phantom_lower = pair_los_lower - chunk_los_lower
        phantom_orig: List[str] = [
            ref for ref in pair_los_orig if ref.lower() in phantom_lower
        ]

        new_fields: Dict[str, Any] = {
            "pair_lo_resolution": {
                "declared_los": list(pair_los_orig),
                "chunk_los": list(chunk_los_orig),
                "phantom_los": list(phantom_orig),
            }
        }

        # Identifier for the audit-trail event. Pair_id is preferred
        # but synthesize_training.py mints it after the validator runs
        # in some paths; fall back to chunk_id then "?".
        pair_id = str(
            pair.get("pair_id")
            or pair.get("id")
            or pair.get("chunk_id")
            or chunk_payload.get("chunk_id")
            or "?"
        )

        # ---- Reject precedence ---- #
        # missing_pair_lo_refs fires before phantom_pair_lo_refs because
        # an empty declared set has an empty phantom set (vacuously);
        # the more actionable rejection_reason for the operator is
        # "the upstream emitter forgot the field" vs. "the cited chunk
        # is wrong".
        rejection_reason: Optional[str] = None
        if self._require_non_empty and not pair_los_lower:
            rejection_reason = "missing_pair_lo_refs"
        elif phantom_lower:
            rejection_reason = "phantom_pair_lo_refs"

        promotion_status = (
            "rejected" if rejection_reason is not None else "validated"
        )
        resolved = rejection_reason is None

        _emit_pair_lo_decision(
            capture,
            pair_id=pair_id,
            kind=str(kind),
            declared_los=pair_los_orig,
            chunk_los=chunk_los_orig,
            phantom_los=phantom_orig,
            resolved=resolved,
            rejection_reason=rejection_reason,
        )

        return promotion_status, rejection_reason, new_fields

    # ------------------------------------------------------------------ #
    # Gate-runner surface — walks training_specs/*.jsonl + chunks.jsonl
    # ------------------------------------------------------------------ #

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Post-hoc audit walking pair files + chunks.jsonl on disk.

        Critical-fails on any phantom pair LO ref against its cited
        chunk's ``learning_outcome_refs[]``. Mirrors the
        :class:`AssessmentObjectiveAlignmentValidator` chunks-walk
        contract: builds a chunk_id → chunk_lo_set map up-front, then
        per-pair subset-checks against that map.

        Inputs:

        - ``instruction_pairs_path`` (str) — path to
          ``training_specs/instruction_pairs.jsonl``. Optional when
          ``course_dir`` is provided.
        - ``preference_pairs_path`` (str) — path to
          ``training_specs/preference_pairs.jsonl``. Optional when
          ``course_dir`` is provided.
        - ``chunks_path`` (str) — path to ``chunks.jsonl``. Required.
          Tolerates either ``imscc_chunks/chunks.jsonl`` (Wave 7c
          preferred) or the legacy top-level ``chunks.jsonl``.
        - ``course_dir`` (str) — fallback resolver: walks
          ``<course_dir>/training_specs/{instruction,preference}_pairs.jsonl``
          and resolves chunks via :func:`lib.libv2_storage.resolve_imscc_chunks_path`.

        Outputs:

        - ``passed=True`` when every pair on disk passes the subset
          check (or both pair files are absent / empty).
        - ``passed=False`` with ``PHANTOM_PAIR_LO_REFS`` /
          ``MISSING_PAIR_LO_REFS`` issues (capped at 50) otherwise.
        - Critical when chunks.jsonl is unresolvable / unparseable
          (mirror of the assessment-objective-alignment silent-zero
          guard).
        """
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or self._decision_capture

        # ---- Resolve pair file paths ---- #
        inst_path: Optional[Path] = None
        pref_path: Optional[Path] = None
        explicit_inst = inputs.get("instruction_pairs_path")
        explicit_pref = inputs.get("preference_pairs_path")
        course_dir_raw = inputs.get("course_dir")
        training_specs_dir_raw = inputs.get("training_specs_dir")

        if isinstance(explicit_inst, str) and explicit_inst:
            inst_path = Path(explicit_inst)
        if isinstance(explicit_pref, str) and explicit_pref:
            pref_path = Path(explicit_pref)
        if course_dir_raw and (inst_path is None or pref_path is None):
            cd = Path(course_dir_raw)
            if inst_path is None:
                inst_path = cd / "training_specs" / "instruction_pairs.jsonl"
            if pref_path is None:
                pref_path = cd / "training_specs" / "preference_pairs.jsonl"
        if (
            training_specs_dir_raw
            and (inst_path is None or pref_path is None)
        ):
            ts = Path(training_specs_dir_raw)
            if inst_path is None:
                inst_path = ts / "instruction_pairs.jsonl"
            if pref_path is None:
                pref_path = ts / "preference_pairs.jsonl"

        # ---- Resolve chunks.jsonl path ---- #
        chunks_path: Optional[Path] = None
        chunks_raw = inputs.get("chunks_path")
        if isinstance(chunks_raw, str) and chunks_raw:
            chunks_path = Path(chunks_raw)
        if chunks_path is None and course_dir_raw:
            try:
                from lib.libv2_storage import resolve_imscc_chunks_path
                chunks_path = resolve_imscc_chunks_path(
                    Path(course_dir_raw), "chunks.jsonl"
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "resolve_imscc_chunks_path raised: %s", exc
                )

        if chunks_path is None:
            return self._fail(
                gate_id,
                "MISSING_CHUNKS_PATH",
                (
                    "PairLearningOutcomeRefsValidator requires one of: "
                    "course_dir or chunks_path."
                ),
            )
        if not chunks_path.exists():
            return self._fail(
                gate_id,
                "CHUNKS_NOT_FOUND",
                f"Chunks file does not exist: {chunks_path}",
                location=str(chunks_path),
            )

        # ---- Build chunk_id → lowercase LO-set map ---- #
        chunk_lo_map = self._collect_chunk_lo_map(chunks_path)
        if chunk_lo_map is None:
            return self._fail(
                gate_id,
                "INVALID_CHUNKS_JSONL",
                f"Failed to parse chunks JSONL: {chunks_path}",
                location=str(chunks_path),
            )

        # Silent-degradation guard mirroring AssessmentObjectiveAlignmentValidator.
        # If every chunk dropped its LO refs, the subset check is
        # vacuously false for every non-empty pair_los; surface as the
        # upstream chunking failure rather than 100% phantom rejections.
        if chunk_lo_map and not any(v for v in chunk_lo_map.values()):
            return self._fail(
                gate_id,
                "PAIR_LO_NO_CHUNK_REFS",
                (
                    f"Chunks file at {chunks_path} contained zero "
                    "learning_outcome_refs across all chunks. Every "
                    "non-empty pair_los would be vacuously phantom; "
                    "fail closed rather than ship 100% phantom rejections. "
                    "Did the imscc_chunking phase run successfully?"
                ),
                location=str(chunks_path),
            )

        # ---- Walk pair files ---- #
        issues: List[GateIssue] = []
        audited = 0
        phantom_pairs = 0
        missing_lo_pairs = 0
        unresolvable_chunk_pairs = 0
        _ISSUE_CAP = 50

        for pair_path, kind in (
            (inst_path, "instruction"),
            (pref_path, "preference"),
        ):
            if pair_path is None or not pair_path.exists():
                continue
            try:
                with pair_path.open("r", encoding="utf-8") as fh:
                    for line_num, line in enumerate(fh, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            pair = json.loads(line)
                        except json.JSONDecodeError:
                            # Soft-skip malformed lines; upstream emit
                            # gates should already filter these.
                            continue
                        if not isinstance(pair, dict):
                            continue
                        audited += 1

                        pair_id = str(
                            pair.get("pair_id")
                            or pair.get("id")
                            or f"{pair_path.name}:{line_num}"
                        )
                        chunk_id = str(pair.get("chunk_id") or "").strip()

                        # Pair-side canonical field is ``lo_refs`` per
                        # the canonical pair schemas; legacy alias
                        # ``learning_outcome_refs`` is read as fallback.
                        _, pair_los_lower = _normalize_lo_set(
                            pair.get("lo_refs")
                            or pair.get("learning_outcome_refs")
                        )

                        # Missing-LO check first (mirror per-pair filter).
                        if self._require_non_empty and not pair_los_lower:
                            missing_lo_pairs += 1
                            if len(issues) < _ISSUE_CAP:
                                issues.append(GateIssue(
                                    severity="critical",
                                    code="MISSING_PAIR_LO_REFS",
                                    message=(
                                        f"{pair_path.name}:{line_num} "
                                        f"({kind}) pair {pair_id!r} has "
                                        f"empty / missing "
                                        f"learning_outcome_refs[] but the "
                                        f"validator requires non-empty refs."
                                    ),
                                    location=str(pair_path),
                                ))
                            continue

                        # Wave 8 Worker A: deterministic-template pairs
                        # (kg_metadata, violation, abstention,
                        # schema_translation) carry synthetic chunk_ids
                        # that don't resolve against chunks.jsonl by
                        # construction. The per-pair filter (W4.B's
                        # `filter_pair`) stamps
                        # ``pair_lo_resolution.skipped =
                        # "deterministic_template"`` on these pairs
                        # before emit so this walk knows the chunk-id
                        # resolution was intentionally bypassed —
                        # bypassing because the four deterministic
                        # generators are oracle-grounded by construction
                        # (pyshacl / graph-membership / fixture-pinned /
                        # concept-absence). Skip these pairs before the
                        # ``chunk_lo_lower is None`` branch fires
                        # ``PAIR_CHUNK_NOT_FOUND``.
                        pair_lo_resolution = pair.get("pair_lo_resolution")
                        if (
                            isinstance(pair_lo_resolution, dict)
                            and pair_lo_resolution.get("skipped")
                            == "deterministic_template"
                        ):
                            continue

                        # Resolve cited chunk's LO set.
                        chunk_lo_lower = chunk_lo_map.get(chunk_id)
                        if chunk_lo_lower is None:
                            # Pair cites a chunk_id not present in
                            # chunks.jsonl. Per the symmetric-validation
                            # contract, this is a critical failure: the
                            # subset check can't run against an absent
                            # chunk so the pair's LO claim is unverifiable.
                            unresolvable_chunk_pairs += 1
                            if len(issues) < _ISSUE_CAP:
                                issues.append(GateIssue(
                                    severity="critical",
                                    code="PAIR_CHUNK_NOT_FOUND",
                                    message=(
                                        f"{pair_path.name}:{line_num} "
                                        f"({kind}) pair {pair_id!r} cites "
                                        f"chunk_id {chunk_id!r} which is "
                                        f"absent from chunks.jsonl; the "
                                        f"LO subset check is unverifiable."
                                    ),
                                    location=str(pair_path),
                                    suggestion=(
                                        "Verify the pair's chunk_id "
                                        "matches an emitted chunk in "
                                        "imscc_chunks/chunks.jsonl, or "
                                        "re-run training_synthesis "
                                        "against a fresh corpus."
                                    ),
                                ))
                            continue

                        phantom_lower = pair_los_lower - chunk_lo_lower
                        if phantom_lower:
                            phantom_pairs += 1
                            if len(issues) < _ISSUE_CAP:
                                issues.append(GateIssue(
                                    severity="critical",
                                    code="PHANTOM_PAIR_LO_REFS",
                                    message=(
                                        f"{pair_path.name}:{line_num} "
                                        f"({kind}) pair {pair_id!r} "
                                        f"declares learning_outcome_refs "
                                        f"not present on cited chunk "
                                        f"{chunk_id!r}: phantom_los="
                                        f"{sorted(phantom_lower)!r}."
                                    ),
                                    location=str(pair_path),
                                    suggestion=(
                                        "Re-run training_synthesis "
                                        "with W4.B's per-pair filter "
                                        "wired in (see "
                                        "Trainforge/synthesize_training.py "
                                        "post-W4.A call site), or "
                                        "regenerate the IMSCC chunks "
                                        "so the chunk picks up the "
                                        "real TO-NN / CO-NN refs."
                                    ),
                                ))
            except OSError as exc:
                issues.append(GateIssue(
                    severity="critical",
                    code="PAIRS_FILE_READ_ERROR",
                    message=f"Failed to read {pair_path}: {exc}",
                    location=str(pair_path),
                ))

        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        action: Optional[str] = None if passed else "block"

        # Score: 1 - (rejected pairs / audited pairs). Both phantom and
        # missing-LO rejections weigh the same.
        rejected = phantom_pairs + missing_lo_pairs + unresolvable_chunk_pairs
        score: float
        if audited == 0:
            score = 1.0
        else:
            score = round((audited - rejected) / audited, 4)

        if capture is not None:
            try:
                capture.log_decision(
                    decision_type="pair_lo_refs_check",
                    decision=(
                        "passed"
                        if passed
                        else (
                            f"failed:phantom={phantom_pairs},"
                            f"missing={missing_lo_pairs},"
                            f"chunk_not_found={unresolvable_chunk_pairs}"
                        )
                    ),
                    rationale=(
                        f"PairLearningOutcomeRefsValidator on-disk audit: "
                        f"{audited} pair(s) audited across "
                        f"instruction + preference; "
                        f"{phantom_pairs} phantom, "
                        f"{missing_lo_pairs} missing-LO, "
                        f"{unresolvable_chunk_pairs} chunk-not-found. "
                        f"require_non_empty={self._require_non_empty}."
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "DecisionCapture.log_decision raised on "
                    "pair_lo_refs_check (gate path): %s",
                    exc,
                )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action=action,
        )

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _fail(
        gate_id: str,
        code: str,
        message: str,
        *,
        location: Optional[str] = None,
    ) -> GateResult:
        """Return a critical-failure GateResult with a single issue."""
        return GateResult(
            gate_id=gate_id,
            validator_name=PairLearningOutcomeRefsValidator.name,
            validator_version=PairLearningOutcomeRefsValidator.version,
            passed=False,
            issues=[GateIssue(
                severity="critical",
                code=code,
                message=message,
                location=location,
            )],
            action="block",
        )

    @staticmethod
    def _collect_chunk_lo_map(
        chunks_path: Path,
    ) -> Optional[Dict[str, Set[str]]]:
        """Build chunk_id → lowercase LO-set map from chunks.jsonl.

        Returns None on parse error. Empty dict means no chunks parsed
        (the silent-degradation guard above will catch this).

        Mirrors :meth:`AssessmentObjectiveAlignmentValidator._collect_chunk_refs`
        case-insensitive normalization (line 341-369) — Trainforge
        emits lowercase, Courseforge emits mixed-case; comparison is
        case-insensitive while the audit field preserves emit case.
        """
        out: Dict[str, Set[str]] = {}
        try:
            with chunks_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        # Soft-skip malformed lines; upstream emit gates
                        # should already filter these.
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    chunk_id_raw = chunk.get("chunk_id") or chunk.get("id")
                    if not isinstance(chunk_id_raw, str):
                        continue
                    chunk_id = chunk_id_raw.strip()
                    if not chunk_id:
                        continue
                    _, lo_lower = _normalize_lo_set(
                        chunk.get("learning_outcome_refs")
                    )
                    out[chunk_id] = lo_lower
        except OSError:
            return None
        return out


__all__ = ["PairLearningOutcomeRefsValidator"]

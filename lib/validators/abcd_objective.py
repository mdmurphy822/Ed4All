"""Phase 6 Subtask 4 — ABCD verb-Bloom alignment validator.

Gates a learning-objective set so every objective whose ``abcd`` field is
present declares a Bloom-aligned ``abcd.behavior.verb``. The verb-set
truth table comes from ``lib.ontology.learning_objectives.BLOOMS_VERBS``,
which is the immutable frozenset projection of the canonical
``schemas/taxonomies/bloom_verbs.json`` taxonomy.

Per-LO contract (mirrors the JSON-LD ``$defs.AbcdObjective`` schema in
``schemas/knowledge/courseforge_jsonld_v1.schema.json``):

1. **Verb mismatch.** When ``abcd`` is present and
   ``abcd.behavior.verb.lower()`` is NOT in
   ``BLOOMS_VERBS[lo.bloom_level]``, emit a warning-severity GateIssue
   with code ``ABCD_VERB_BLOOM_MISMATCH`` and route the validator
   ``action="regenerate"`` so the upstream content-generator / outliner
   can re-roll the LO. Also emit a ``decision_type="abcd_verb_bloom_mismatch"``
   DecisionCapture event when a capture is wired in.
2. **Missing ABCD on a LO that requires it.** When the LO declares
   ``requires_abcd=True`` (or the inputs declare ``require_abcd=True``
   at the top level — Phase 6 contract: every newly-emitted LO has
   ABCD), emit a warning-severity GateIssue with code ``ABCD_MISSING``.
   Phase 6 lands as warning-only; Phase 7+ promotes to critical once
   corpus calibration confirms safe per the schema's
   ``$defs.LearningObjective.abcd`` description.
3. **Malformed ABCD shape.** When ``abcd`` is present but missing a
   required sub-field (``audience``, ``behavior.verb``,
   ``behavior.action_object``, ``condition``, ``degree``), emit a
   critical-severity GateIssue with code ``ABCD_MALFORMED``. Routes
   ``action="block"`` because re-rolling the LO won't reshape a
   structurally broken emit — the upstream emitter is the bug.
4. **Bloom-level absent on a LO that has ABCD.** When ``abcd`` is
   present but the LO carries no usable ``bloom_level`` (the
   schema admits ``null``), the validator can't audit verb alignment;
   emit a warning-severity GateIssue with code ``ABCD_NO_BLOOM_LEVEL``,
   ``passed=True`` per-LO. No decision event — there's nothing
   actionable to capture.

Inputs contract:

* ``inputs["objectives"]`` — list of LO dicts. Either each LO is a
  bare dict carrying ``id`` / ``statement`` / ``bloom_level`` / ``abcd``
  fields, or LOs are nested under ``learning_outcomes`` /
  ``terminal_objectives`` / ``chapter_objectives`` (Courseforge
  synthesized form). The validator tries both surfaces.
* ``inputs["synthesized_objectives_path"]`` — alternative to
  ``objectives``; points at a ``synthesized_objectives.json`` file the
  validator loads + flattens via the same shape-tolerance rules.
* ``inputs["require_abcd"]`` — opt-in bool. When true, every LO must
  declare ``abcd``; when false (default), missing-ABCD is silently
  skipped to preserve backward compatibility with legacy LOs that
  pre-date Phase 6.
* ``inputs["decision_capture"]`` — optional ``DecisionCapture``
  instance. When provided, the validator emits one event per
  verb-mismatch + one event per pass (positive-path
  ``abcd_authored``).
* ``inputs["gate_id"]`` — optional override for the gate ID stamped
  on the returned ``GateResult``. Defaults to the validator name.

Both ``bloomLevel`` (camelCase, JSON-LD canonical) and ``bloom_level``
(snake_case, Python-side) field names are accepted on the LO dict so
the validator works against both the on-disk
``synthesized_objectives.json`` form and the in-memory dict form
``MCP/tools/_content_gen_helpers.py::_normalize_objective_entry``
emits.

Cross-references:

* ``schemas/knowledge/courseforge_jsonld_v1.schema.json::$defs.AbcdObjective``
  — canonical ABCD shape; this validator's input contract is its
  Python-side mirror.
* ``schemas/events/decision_event.schema.json::decision_type.enum``
  — the ``abcd_verb_bloom_mismatch`` and ``abcd_authored`` enum
  members were added alongside this validator.
* ``lib.ontology.learning_objectives.BLOOMS_VERBS`` — the verb-set
  truth table (Phase 6 ST 2; commit ``b46e433``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.ontology.learning_objectives import BLOOMS_VERBS

logger = logging.getLogger(__name__)


#: Cap the number of issues emitted so a uniformly-broken LO batch
#: doesn't drown the gate report. Mirrors the cap used by other Phase 4 /
#: Phase 6 validators (e.g. ``bloom_classifier_disagreement._ISSUE_LIST_CAP``).
_ISSUE_LIST_CAP: int = 50


#: Canonical decision-type strings emitted by this validator. Both
#: must appear in ``schemas/events/decision_event.schema.json::decision_type.enum``;
#: the alphabetised position is enforced by the schema's existing
#: alphabetical ordering contract (Phase 4.5 cleanup, commit ``3184f1a``).
_DECISION_TYPE_MISMATCH: str = "abcd_verb_bloom_mismatch"
_DECISION_TYPE_PASS: str = "abcd_authored"


def _bloom_level(lo: Mapping[str, Any]) -> Optional[str]:
    """Return the canonical lowercase Bloom level for an LO dict.

    Tolerates both ``bloom_level`` (snake_case Python emit) and
    ``bloomLevel`` (camelCase JSON-LD emit). Returns ``None`` when
    neither key is present or the value is empty / not-a-string.
    """
    raw = lo.get("bloom_level")
    if not isinstance(raw, str) or not raw.strip():
        raw = lo.get("bloomLevel")
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    return s if s else None


def _lo_id(lo: Mapping[str, Any]) -> str:
    """Return a printable LO ID for issue messages.

    Falls back to ``"<unknown-id>"`` when the LO has no canonical ``id``;
    the page-level integrity gates (``page_objectives``) catch that as
    a separate failure mode, so we don't redundantly flag it here.
    """
    raw = lo.get("id") or lo.get("objective_id") or ""
    raw = str(raw).strip()
    return raw or "<unknown-id>"


def _flatten_objectives(payload: Any) -> List[Dict[str, Any]]:
    """Pull a flat List[LO-dict] out of the various container shapes.

    Accepts:

    * A bare list of LO dicts.
    * A dict with ``learning_outcomes`` (Wave 24+ Courseforge synthesized
      form's flat list — preferred when present).
    * A dict with ``terminal_objectives`` + ``chapter_objectives`` (the
      grouped form; ``chapter_objectives`` may be a flat list of LO
      dicts OR a list of ``{"chapter": str, "objectives": [...]}``
      groups, which are flattened).
    * A dict with ``terminal_outcomes`` + ``component_objectives`` (Wave 75
      LibV2 archive form — same flatten rules as above).

    Returns ``[]`` for any unrecognised shape.
    """
    if isinstance(payload, list):
        return [lo for lo in payload if isinstance(lo, dict)]
    if not isinstance(payload, Mapping):
        return []
    # Preferred flat surface (Courseforge synthesized form).
    flat = payload.get("learning_outcomes")
    if isinstance(flat, list):
        return [lo for lo in flat if isinstance(lo, dict)]

    out: List[Dict[str, Any]] = []
    for key in ("terminal_objectives", "terminal_outcomes"):
        block = payload.get(key)
        if isinstance(block, list):
            out.extend(lo for lo in block if isinstance(lo, dict))
    for key in ("chapter_objectives", "component_objectives"):
        block = payload.get(key)
        if not isinstance(block, list):
            continue
        for entry in block:
            if not isinstance(entry, dict):
                continue
            if "objectives" in entry and isinstance(entry["objectives"], list):
                out.extend(lo for lo in entry["objectives"] if isinstance(lo, dict))
            else:
                out.append(entry)
    return out


def _coerce_objectives(
    inputs: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Optional[GateIssue]]:
    """Resolve the LO list from ``inputs``.

    Priority: ``inputs['objectives']`` > ``inputs['synthesized_objectives_path']``.
    Returns ``(los, error_issue)``; ``error_issue`` is non-None only on
    structural failures (path doesn't exist, JSON unparseable). An empty
    LO list is NOT an error — the validator's contract is a no-op pass
    on empty input.
    """
    explicit = inputs.get("objectives")
    if explicit is not None:
        return _flatten_objectives(explicit), None

    path_raw = inputs.get("synthesized_objectives_path")
    if not path_raw:
        return [], None

    p = Path(path_raw)
    if not p.exists():
        return [], GateIssue(
            severity="critical",
            code="ABCD_OBJECTIVES_PATH_MISSING",
            message=(
                f"synthesized_objectives_path {str(p)!r} does not exist; "
                f"cannot audit ABCD verb-Bloom alignment."
            ),
            location=str(p),
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], GateIssue(
            severity="critical",
            code="ABCD_OBJECTIVES_PATH_UNREADABLE",
            message=(
                f"Failed to parse synthesized_objectives_path {str(p)!r}: "
                f"{exc.__class__.__name__}: {exc}"
            ),
            location=str(p),
        )
    return _flatten_objectives(data), None


def _format_valid_verbs(level: str, *, max_verbs: int = 10) -> str:
    """Render a truncated, sorted preview of the valid verbs for a level.

    Sorted output keeps the message stable across runs — a frozenset
    has no canonical order. We cap at ``max_verbs`` (default 10) to
    avoid drowning the GateIssue message in a 100+ verb list when the
    validator hits a level like ``apply``.
    """
    verbs = sorted(BLOOMS_VERBS.get(level, frozenset()))
    if not verbs:
        return "<no canonical verbs registered>"
    preview = verbs[:max_verbs]
    suffix = f" (+{len(verbs) - max_verbs} more)" if len(verbs) > max_verbs else ""
    return ", ".join(preview) + suffix


def _emit_decision(
    capture: Any,
    *,
    decision_type: str,
    decision: str,
    rationale: str,
    context: Optional[str] = None,
    alternatives: Optional[List[str]] = None,
) -> None:
    """Emit one DecisionCapture event, swallowing any capture-side errors.

    The validator's correctness must not depend on the capture sidecar
    succeeding — capture is observability, not control flow. Errors are
    logged at warning level so postmortems can correlate but the gate
    result still ships.
    """
    if capture is None:
        return
    try:
        capture.log_decision(
            decision_type=decision_type,
            decision=decision,
            rationale=rationale,
            context=context,
            alternatives_considered=alternatives or None,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.warning(
            "AbcdObjectiveValidator: decision_capture.log_decision failed "
            "(decision_type=%s): %s",
            decision_type,
            exc,
        )


class AbcdObjectiveValidator:
    """Phase 6 ABCD verb-Bloom alignment gate.

    Validator-protocol-compatible class wired into
    ``textbook_to_course::course_planning::abcd_verb_alignment``.
    Emits regenerate-action GateIssues on Bloom-verb mismatch, warning
    GateIssues on missing-ABCD (when ``require_abcd=True``), and
    block-action GateIssues on malformed ABCD shape.
    """

    name = "abcd_objective"
    version = "0.1.0"  # Phase 6 ST 4 PoC

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)

        objectives, err = _coerce_objectives(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        # Empty LO set is a no-op pass. Phase 6 prep verifies this is
        # the contract the gate-wiring path tolerates (Subtask 4.5).
        if not objectives:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        require_abcd_top = bool(inputs.get("require_abcd", False))
        capture = inputs.get("decision_capture")

        issues: List[GateIssue] = []
        audited = 0
        passed_count = 0
        had_mismatch = False

        for lo in objectives:
            if not isinstance(lo, Mapping):
                continue
            audited += 1
            lo_id = _lo_id(lo)
            abcd = lo.get("abcd")
            require_abcd_lo = bool(lo.get("requires_abcd", False))
            require_abcd = require_abcd_top or require_abcd_lo

            # 1. Missing ABCD path.
            if abcd is None:
                if require_abcd:
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(
                            GateIssue(
                                severity="warning",
                                code="ABCD_MISSING",
                                message=(
                                    f"LO {lo_id!r} declares no ``abcd`` field "
                                    f"but the inputs require ABCD-shaped emit "
                                    f"(``require_abcd=True``). Phase 6 lands "
                                    f"as warning; Phase 7+ promotes to critical."
                                ),
                                location=lo_id,
                                suggestion=(
                                    "Have the course-outliner / content-generator "
                                    "emit the LO with the ABCD sub-object. See "
                                    "$defs.AbcdObjective in "
                                    "courseforge_jsonld_v1.schema.json."
                                ),
                            )
                        )
                else:
                    # Legacy LO without ABCD on a non-strict run is a
                    # silent pass — preserves backward compatibility
                    # with pre-Phase-6 corpora.
                    passed_count += 1
                continue

            if not isinstance(abcd, Mapping):
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="ABCD_MALFORMED",
                            message=(
                                f"LO {lo_id!r} carries an ``abcd`` field but "
                                f"its value is {type(abcd).__name__!r}, not a "
                                f"mapping. Expected $defs.AbcdObjective shape."
                            ),
                            location=lo_id,
                        )
                    )
                continue

            # 2. Malformed ABCD shape.
            behavior = abcd.get("behavior")
            if not isinstance(behavior, Mapping):
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="ABCD_MALFORMED",
                            message=(
                                f"LO {lo_id!r} ABCD is missing the "
                                f"``behavior`` mapping (got "
                                f"{type(behavior).__name__!r})."
                            ),
                            location=lo_id,
                            suggestion=(
                                "Emit ``abcd.behavior = {verb, action_object}`` "
                                "per $defs.AbcdObjective."
                            ),
                        )
                    )
                continue

            verb_raw = behavior.get("verb")
            if not isinstance(verb_raw, str) or not verb_raw.strip():
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="ABCD_MALFORMED",
                            message=(
                                f"LO {lo_id!r} ABCD.behavior.verb is "
                                f"missing or empty (value: {verb_raw!r})."
                            ),
                            location=lo_id,
                            suggestion=(
                                "Emit a non-empty Bloom-aligned verb in "
                                "abcd.behavior.verb."
                            ),
                        )
                    )
                continue

            verb = verb_raw.strip().lower()

            # 3. Bloom-level absent on a LO that has ABCD — can't audit.
            level = _bloom_level(lo)
            if level is None:
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="ABCD_NO_BLOOM_LEVEL",
                            message=(
                                f"LO {lo_id!r} has ``abcd`` set but no "
                                f"``bloom_level`` to audit verb alignment "
                                f"against. Skipping verb-Bloom check."
                            ),
                            location=lo_id,
                        )
                    )
                # No actionable signal — count as passing for the
                # score denominator so a corpus of legacy LOs missing
                # bloom_level doesn't drag the score to 0.
                passed_count += 1
                continue

            valid_verbs = BLOOMS_VERBS.get(level)
            if valid_verbs is None or not valid_verbs:
                # Bloom level outside the canonical 6-value set — the
                # taxonomy loader enforces this surface upstream, so
                # reaching here means an out-of-enum level snuck through.
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="ABCD_UNKNOWN_BLOOM_LEVEL",
                            message=(
                                f"LO {lo_id!r} declares "
                                f"bloom_level={level!r} which is not in "
                                f"the canonical Bloom verb taxonomy."
                            ),
                            location=lo_id,
                        )
                    )
                continue

            # 4. Verb-Bloom mismatch path — the core gate.
            if verb not in valid_verbs:
                had_mismatch = True
                if len(issues) < _ISSUE_LIST_CAP:
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="ABCD_VERB_BLOOM_MISMATCH",
                            message=(
                                f"LO {lo_id!r} declares "
                                f"bloom_level={level!r} but its "
                                f"abcd.behavior.verb={verb_raw!r} "
                                f"(normalized: {verb!r}) is not in the "
                                f"canonical verb set. Valid verbs for "
                                f"level {level!r}: "
                                f"{_format_valid_verbs(level)}."
                            ),
                            location=lo_id,
                            suggestion=(
                                f"Either pick a verb from the {level!r} "
                                f"verb set or update bloom_level to a "
                                f"level whose verb set contains "
                                f"{verb!r}."
                            ),
                        )
                    )
                _emit_decision(
                    capture,
                    decision_type=_DECISION_TYPE_MISMATCH,
                    decision=(
                        f"Flagged LO {lo_id} verb {verb!r} as not in "
                        f"BLOOMS_VERBS[{level!r}] (size "
                        f"{len(valid_verbs)})."
                    ),
                    rationale=(
                        f"AbcdObjectiveValidator audits verb-Bloom alignment "
                        f"against lib.ontology.learning_objectives.BLOOMS_VERBS. "
                        f"LO {lo_id} declared bloom_level={level!r} and "
                        f"abcd.behavior.verb={verb_raw!r}; normalized "
                        f"{verb!r} is not in the canonical verb set "
                        f"(first 10: {_format_valid_verbs(level)}). "
                        f"Routing action=regenerate so the upstream emitter "
                        f"re-rolls the LO."
                    ),
                    context=(
                        f"lo_id={lo_id}; bloom_level={level}; verb={verb}; "
                        f"valid_count={len(valid_verbs)}"
                    ),
                )
                continue

            # 5. Pass path — emit the positive-path decision event.
            passed_count += 1
            _emit_decision(
                capture,
                decision_type=_DECISION_TYPE_PASS,
                decision=(
                    f"LO {lo_id} ABCD verb {verb!r} aligned with "
                    f"bloom_level={level!r}."
                ),
                rationale=(
                    f"AbcdObjectiveValidator confirmed "
                    f"abcd.behavior.verb={verb_raw!r} (normalized "
                    f"{verb!r}) is a member of "
                    f"BLOOMS_VERBS[{level!r}] (size "
                    f"{len(valid_verbs)}). No regenerate action emitted."
                ),
                context=f"lo_id={lo_id}; bloom_level={level}; verb={verb}",
            )

        # Score = pass rate over audited LOs. ``passed`` is True iff no
        # critical-severity issues fired (warnings don't fail the gate
        # under the Phase 6 warning-severity wiring; the
        # ``ABCD_MALFORMED`` / ``ABCD_UNKNOWN_BLOOM_LEVEL`` paths emit
        # critical and DO fail-closed).
        critical_count = sum(1 for i in issues if i.severity == "critical")
        passed = critical_count == 0
        score = 1.0 if audited == 0 else round(passed_count / audited, 4)

        # Action signal:
        #   - critical issues  → "block"  (re-rolling won't fix shape).
        #   - verb mismatches  → "regenerate".
        #   - warnings only    → None.
        action: Optional[str]
        if critical_count > 0:
            action = "block"
        elif had_mismatch:
            action = "regenerate"
        else:
            action = None

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
    "AbcdObjectiveValidator",
]

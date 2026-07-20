"""C3-2 — DiscussionAssignmentGroundingValidator.

Per-item-type grounding for the B10 discussion + assignment assessment
surface. This is the real validator behind the ``discussion_assignment_grounded``
gate, replacing the ``# TODO(validator)`` stand-in that re-pointed at
``AssessmentObjectiveAlignmentValidator`` (which only re-ran the aggregate
objective-coverage check that the CRITICAL ``assessment_objective_alignment``
gate already enforces — zero independent signal).

What it actually checks
-----------------------

The aggregate ``assessment_objective_alignment`` gate audits EVERY synthesized
item's ``objective_id`` (quiz + assignment + discussion) at CRITICAL — a
missing-grounding discussion is indistinguishable in that report from a
missing-grounding quiz. This validator is the per-TYPE breakdown: it isolates
the discussion (B10) + assignment items and verifies, ONE ITEM AT A TIME, that
each item's prompt is grounded in real source chunks.

An item is GROUNDED when EITHER:

  * it carries an explicit non-empty ``source_chunk_ids`` whose ids resolve to
    real chunks in the chunkset; OR
  * its ``objective_id`` resolves to >= 1 source chunk — i.e. some chunk's
    ``learning_outcome_refs[]`` contains that objective id (the SAME chunk-side
    resolution surface :class:`AssessmentObjectiveAlignmentValidator` uses,
    optionally unioned with the synthesized objectives' id set). The prose
    authoring pass (``_synthesize_prose_prompt``) writes the item's prompt
    grounded ONLY in exactly those chunks, so an objective that resolves to a
    real chunk is the persisted-on-disk proxy for "the prompt traces to its
    cited source".

Flags (both WARNING — mirrors the stand-in's severity; deferred critical-flip):

  * ``DISCUSSION_UNGROUNDED`` — a discussion (B10) item lacks resolvable
    source grounding (no resolvable ``source_chunk_ids`` AND its
    ``objective_id`` resolves to no chunk).
  * ``ASSIGNMENT_UNGROUNDED`` — an assignment item lacks resolvable source
    grounding.

Anti-fabrication: only flags what real signals support. An item with NO
``objective_id`` AND no ``source_chunk_ids`` cannot be evaluated for grounding,
so it is flagged (a prompt with no traceable source is exactly the failure mode
this gate exists to surface) — but the empty-input / no-chunks / no-items cases
all GRACEFUL-SKIP (passed=True with a structured info/warning issue) rather than
inventing a verdict.

NO new model load — pure id / chunk-ref resolution (reuses the same lexical
resolution surface as the alignment + cumulative validators). No gating flag —
warning-day-1, mirroring ``cumulative_assessment``.

Inputs (``inputs`` dict)
------------------------

    discussion_items: Optional[List[dict]]
    assignment_items: Optional[List[dict]]
        Direct in-memory item lists (each dict: ``objective_id`` /
        ``source_chunk_ids`` / ``discussion_id`` | ``assignment_id`` /
        ``text``). Primary input. When absent, the builder reconstructs the
        item set from ``assessments_path`` (the ``06_assessments`` manifest +
        per-item XML).

    assessments_path / assessment_path: Optional[str]
        Path to the ``06_assessments`` manifest.json (``assessments[]`` with
        ``type=discussion|assignment`` + per-item ``file``). When the explicit
        item lists are absent, discussion/assignment items are reconstructed
        from this manifest (objective_id parsed from the per-item XML title +
        identifier).

    chunks_path: Optional[str]
        Path to ``chunks.jsonl`` (the resolution surface — every chunk's
        ``learning_outcome_refs[]`` + its id). Required for the grounding
        check; absent → graceful skip.

    synthesized_objectives_path: Optional[str]
        Optional. When provided + extant, the synthesized objectives' id set
        is unioned into the objective-resolution surface (mirrors the W5.E
        union arm of :class:`AssessmentObjectiveAlignmentValidator`). NOTE:
        the union ONLY matters for items carrying explicit chunk grounding;
        for the objective-resolves-to-a-chunk path the chunk side is
        authoritative.

    decision_capture: Optional[Any]
        Optional ``DecisionCapture``-shaped instance. One
        ``assessment_quality`` event per validate (dynamic signals:
        n_discussion / n_assignment / n_ungrounded per type / verdict).

    gate_id: Optional[str]
        Override for ``GateResult.gate_id`` (defaults to
        ``"discussion_assignment_grounded"``).

References:
    - ``lib/validators/assessment_objective_alignment.py`` — the aggregate
      CRITICAL sibling whose ``_collect_chunk_refs`` resolution surface this
      reuses.
    - ``lib/validators/cumulative_assessment.py`` — sibling assessment-surface
      validator (input shape + warning-day-1 posture mirrored here).
    - ``MCP/tools/pipeline_tools.py::_run_assessment_synthesis`` — emits the
      discussion / assignment items + the ``06_assessments`` manifest.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)

_CODE_DISCUSSION_UNGROUNDED = "DISCUSSION_UNGROUNDED"
_CODE_ASSIGNMENT_UNGROUNDED = "ASSIGNMENT_UNGROUNDED"

# A5 — flag-gated NLI TEXT-grounding arm (default OFF → legacy refs-only path).
# The legacy grounding check resolves an item purely by REFERENCE — its
# objective_id appears in some chunk's learning_outcome_refs, or its
# source_chunk_ids resolve to real chunks. That confirms the item POINTS at a
# real chunk, not that its prompt TEXT is actually supported by it. When
# ``ED4ALL_DISCUSSION_GROUNDING_NLI`` is truthy AND an item carries prompt text +
# resolvable cited chunk TEXTS + an NLI scorer is available, the arm makes the
# text-entailment verdict authoritative for that item (grounded iff the prompt
# is entailed by its cited chunks). Any item that can't run the NLI check
# (no text / no resolvable chunk text / scorer unavailable) falls back to the
# legacy refs-only verdict, so the arm only ever TIGHTENS a resolvable item and
# never fabricates a verdict. Off → byte-identical legacy path.
_GROUNDING_NLI_ENV = "ED4ALL_DISCUSSION_GROUNDING_NLI"
_TRUTHY_TOKENS = frozenset({"1", "true", "yes", "on"})


def _grounding_nli_enabled() -> bool:
    """Return True when ED4ALL_DISCUSSION_GROUNDING_NLI is truthy (default off)."""
    return os.environ.get(_GROUNDING_NLI_ENV, "").strip().lower() in _TRUTHY_TOKENS

# Extract an objective id (canonical TO-NN / CO-NN, or {COURSE}_OBJ_N legacy)
# from a reconstructed item title / identifier when the persisted manifest
# dropped the explicit objective_id field.
_OBJ_ID_RE = re.compile(r"\b([A-Z]{2,}-\d{2,}|[A-Z0-9]+_OBJ_\d+)\b")


# ---------------------------------------------------------------------------
# JSON / chunk loading
# ---------------------------------------------------------------------------


def _load_json(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        logger.debug(
            "discussion_assignment_grounding: failed to load %s: %s", path, exc
        )
        return None


def _collect_chunk_grounding(
    chunks_path: Optional[str],
    synthesized_objectives_path: Optional[str],
) -> Optional[Tuple[Set[str], Set[str]]]:
    """Return ``(grounded_objective_ids, chunk_ids)`` from the chunkset.

    ``grounded_objective_ids`` — every objective id (lower-cased) that appears
    in at least one chunk's ``learning_outcome_refs[]``, optionally unioned
    with the synthesized objectives' id set. ``chunk_ids`` — every chunk id
    (lower-cased) for resolving explicit ``source_chunk_ids``.

    Returns ``None`` when the chunks file is missing / unreadable so the caller
    can graceful-skip.
    """
    p = Path(chunks_path) if chunks_path else None
    if p is None or not p.exists() or not p.is_file():
        return None

    grounded: Set[str] = set()
    chunk_ids: Set[str] = set()
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                cid = chunk.get("id") or chunk.get("chunk_id")
                if isinstance(cid, str) and cid.strip():
                    chunk_ids.add(cid.strip().lower())
                los = chunk.get("learning_outcome_refs") or []
                if isinstance(los, list):
                    for lo in los:
                        if isinstance(lo, str) and lo.strip():
                            grounded.add(lo.strip().lower())
    except OSError:
        return None

    # Union the synthesized objectives' id set (W5.E parity). The canonical
    # _flatten_objectives loader handles both the Courseforge synthesized form
    # and the LibV2 archive form. Graceful: any failure leaves the chunk-side
    # surface untouched.
    if synthesized_objectives_path:
        sp = Path(synthesized_objectives_path)
        if sp.exists() and sp.is_file():
            try:
                from lib.validators.abcd_objective import _flatten_objectives

                payload = json.loads(sp.read_text(encoding="utf-8"))
                for lo in _flatten_objectives(payload):
                    lo_id = lo.get("id") or lo.get("objective_id")
                    if isinstance(lo_id, str) and lo_id.strip():
                        grounded.add(lo_id.strip().lower())
            except (OSError, ValueError, ImportError) as exc:
                logger.debug(
                    "discussion_assignment_grounding: synthesized-objectives "
                    "union skipped (%s)",
                    exc,
                )

    return grounded, chunk_ids


def _collect_chunk_texts(
    chunks_path: Optional[str],
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Return ``(chunk_text_by_id, obj_to_chunk_ids)`` for the A5 NLI arm.

    ``chunk_text_by_id`` maps a lower-cased chunk id → its text; ``obj_to_chunk_ids``
    maps a lower-cased objective id → the ids of chunks whose
    ``learning_outcome_refs[]`` contain it. Both empty when the chunks file is
    missing / unreadable (the NLI arm then degrades to the legacy path).
    """
    text_by_id: Dict[str, str] = {}
    obj_to_chunks: Dict[str, List[str]] = {}
    p = Path(chunks_path) if chunks_path else None
    if p is None or not p.exists() or not p.is_file():
        return text_by_id, obj_to_chunks
    try:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                cid = chunk.get("id") or chunk.get("chunk_id")
                if not (isinstance(cid, str) and cid.strip()):
                    continue
                cid_l = cid.strip().lower()
                text = chunk.get("text")
                if isinstance(text, str) and text.strip():
                    text_by_id[cid_l] = text
                los = chunk.get("learning_outcome_refs") or []
                if isinstance(los, list):
                    for lo in los:
                        if isinstance(lo, str) and lo.strip():
                            obj_to_chunks.setdefault(
                                lo.strip().lower(), []
                            ).append(cid_l)
    except OSError:
        return {}, {}
    return text_by_id, obj_to_chunks


# ---------------------------------------------------------------------------
# Item extraction
# ---------------------------------------------------------------------------


def _item_objective_id(item: Dict[str, Any]) -> str:
    """Resolve an item's objective id from the dict or its title/identifier."""
    oid = item.get("objective_id")
    if isinstance(oid, str) and oid.strip():
        return oid.strip()
    # Reconstructed-from-manifest fallback: parse the canonical id out of the
    # title / identifier / id fields.
    for key in ("title", "identifier", "discussion_id", "assignment_id", "id"):
        val = item.get(key)
        if isinstance(val, str):
            m = _OBJ_ID_RE.search(val)
            if m:
                return m.group(1)
    return ""


def _item_source_chunk_ids(item: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    raw = item.get("source_chunk_ids")
    if isinstance(raw, list):
        for c in raw:
            if isinstance(c, str) and c.strip():
                out.append(c.strip())
    return out


def _item_label(item: Dict[str, Any], kind: str, idx: int) -> str:
    for key in ("discussion_id", "assignment_id", "identifier", "id", "title"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return f"{kind}:idx:{idx}"


def _reconstruct_items_from_manifest(
    assessments_path: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Reconstruct (discussions, assignments) from a 06_assessments manifest.

    The persisted ``manifest.json`` ``assessments[]`` entries carry
    ``type`` / ``file`` / ``title`` / ``identifier`` (objective_id is dropped),
    so the objective id is recovered from the title / identifier (and, when the
    sibling XML is present, from its body). Items with no recoverable
    objective id are still returned (and will be flagged as ungrounded — a
    prompt with no traceable source is the failure mode this gate surfaces).
    """
    data = _load_json(assessments_path)
    if not isinstance(data, dict):
        return [], []
    entries = data.get("assessments")
    if not isinstance(entries, list):
        return [], []

    base_dir = Path(assessments_path).parent if assessments_path else None
    discussions: List[Dict[str, Any]] = []
    assignments: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        etype = str(entry.get("type") or "").strip().lower()
        if etype not in ("discussion", "assignment"):
            continue
        item: Dict[str, Any] = {
            "title": str(entry.get("title") or ""),
            "identifier": str(entry.get("identifier") or ""),
        }
        # Recover an objective id from the sibling XML body when the title /
        # identifier didn't carry one.
        if not _OBJ_ID_RE.search(item["title"] + " " + item["identifier"]):
            fname = entry.get("file")
            if base_dir is not None and isinstance(fname, str) and fname:
                try:
                    xml_text = (base_dir / fname).read_text(encoding="utf-8")
                    m = _OBJ_ID_RE.search(xml_text)
                    if m:
                        item["objective_id"] = m.group(1)
                except OSError:
                    pass
        if etype == "discussion":
            discussions.append(item)
        else:
            assignments.append(item)
    return discussions, assignments


# ---------------------------------------------------------------------------
# Decision-capture emit
# ---------------------------------------------------------------------------


def _emit_decision(
    capture: Optional[Any],
    *,
    n_discussion: int,
    n_assignment: int,
    n_discussion_ungrounded: int,
    n_assignment_ungrounded: int,
    n_grounding_objectives: int,
    applied: bool,
    passed: bool,
) -> None:
    if capture is None:
        return
    verdict = (
        "discussion_assignment_grounding_skipped" if not applied
        else (
            "discussion_assignment_grounded" if passed
            else "discussion_assignment_ungrounded"
        )
    )
    rationale = (
        f"discussion_assignment_grounding audit: n_discussion={n_discussion} "
        f"discussion (B10) item(s), n_assignment={n_assignment} assignment "
        f"item(s); resolution surface carries n_grounding_objectives="
        f"{n_grounding_objectives} chunk-grounded objective id(s); "
        f"n_discussion_ungrounded={n_discussion_ungrounded}, "
        f"n_assignment_ungrounded={n_assignment_ungrounded}; applied={applied} "
        f"-> {verdict}. Each discussion/assignment prompt must trace to a real "
        f"source chunk (resolvable source_chunk_ids OR an objective_id present "
        f"in some chunk's learning_outcome_refs)."
    )
    ml_features = {
        "n_discussion": int(n_discussion),
        "n_assignment": int(n_assignment),
        "n_discussion_ungrounded": int(n_discussion_ungrounded),
        "n_assignment_ungrounded": int(n_assignment_ungrounded),
        "n_grounding_objectives": int(n_grounding_objectives),
        "applied": bool(applied),
        "passed": bool(passed),
    }
    try:
        capture.log_decision(
            decision_type="assessment_quality",
            decision=verdict,
            rationale=rationale,
            ml_features=ml_features,
        )
    except Exception as exc:  # noqa: BLE001 - audit emit is best-effort
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "discussion_assignment_grounding assessment_quality: %s",
            exc,
        )


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------


class DiscussionAssignmentGroundingValidator:
    """C3-2 — per-item grounding for B10 discussion + assignment items.

    Validator-protocol-compatible class. Flags ``DISCUSSION_UNGROUNDED`` /
    ``ASSIGNMENT_UNGROUNDED`` (WARNING) when a discussion/assignment item's
    prompt lacks resolvable source grounding. Graceful-skip (passed=True) when
    the grounding inputs (chunks, items) are absent.
    """

    name = "discussion_assignment_grounding"
    version = "1.0.0"

    def __init__(self, *, decision_capture: Optional[Any] = None) -> None:
        self._decision_capture = decision_capture

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "discussion_assignment_grounded")
        capture = inputs.get("decision_capture") or self._decision_capture

        # 1. Resolve the item set — explicit lists first, manifest fallback.
        discussions = inputs.get("discussion_items")
        assignments = inputs.get("assignment_items")
        discussions = [d for d in discussions if isinstance(d, dict)] if isinstance(discussions, list) else None
        assignments = [a for a in assignments if isinstance(a, dict)] if isinstance(assignments, list) else None

        if discussions is None and assignments is None:
            assessments_path = (
                inputs.get("assessments_path") or inputs.get("assessment_path")
            )
            discussions, assignments = _reconstruct_items_from_manifest(
                assessments_path
            )
        else:
            discussions = discussions or []
            assignments = assignments or []

        n_disc = len(discussions)
        n_asgn = len(assignments)

        # No discussion/assignment items at all → structured skip (this is a
        # legitimate state — assessment_synthesis is optional / may emit only
        # quizzes). Never invent a failure.
        if n_disc == 0 and n_asgn == 0:
            _emit_decision(
                capture,
                n_discussion=0,
                n_assignment=0,
                n_discussion_ungrounded=0,
                n_assignment_ungrounded=0,
                n_grounding_objectives=0,
                applied=False,
                passed=True,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="info",
                        code="NO_DISCUSSION_ASSIGNMENT_ITEMS",
                        message=(
                            "No discussion (B10) or assignment items to audit "
                            "(quiz-only / skipped assessment synthesis); "
                            "per-type grounding check is a no-op."
                        ),
                    )
                ],
                action=None,
            )

        # 2. Build the grounding-resolution surface from the chunkset.
        chunks_path = inputs.get("chunks_path")
        grounding = _collect_chunk_grounding(
            chunks_path,
            inputs.get("synthesized_objectives_path"),
        )
        if grounding is None:
            # No chunks to resolve against → cannot audit grounding. Graceful
            # skip (warning, not critical — the gate is advisory day-1).
            _emit_decision(
                capture,
                n_discussion=n_disc,
                n_assignment=n_asgn,
                n_discussion_ungrounded=0,
                n_assignment_ungrounded=0,
                n_grounding_objectives=0,
                applied=False,
                passed=True,
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
                        code="GROUNDING_INPUTS_UNAVAILABLE",
                        message=(
                            "chunks_path missing / unreadable; cannot resolve "
                            "discussion/assignment grounding (skipped). "
                            f"({n_disc} discussion + {n_asgn} assignment item(s) "
                            "were not audited.)"
                        ),
                        location=str(chunks_path or ""),
                    )
                ],
                action=None,
            )

        grounded_objectives, chunk_ids = grounding

        # A5 — flag-gated NLI text-grounding arm. Only load the chunk TEXTS +
        # objective→chunk map (and resolve the scorer) when the arm is on; off →
        # these stay empty / None and the audit runs the legacy refs-only path.
        nli_on = _grounding_nli_enabled()
        chunk_text_by_id: Dict[str, str] = {}
        obj_to_chunk_ids: Dict[str, List[str]] = {}
        nli_scorer = None
        if nli_on:
            chunk_text_by_id, obj_to_chunk_ids = _collect_chunk_texts(chunks_path)
            nli_scorer = inputs.get("groundedness_scorer")
            if nli_scorer is None:
                try:
                    from lib.retrieval.groundedness import score_groundedness
                    nli_scorer = score_groundedness
                except Exception as exc:  # noqa: BLE001 — degrade to legacy
                    logger.debug(
                        "discussion_assignment_grounding: NLI scorer "
                        "unavailable (%s); arm degrades to refs-only.", exc,
                    )

        # 3. Per-item grounding audit.
        issues: List[GateIssue] = []
        n_disc_ungrounded = self._audit_items(
            items=discussions,
            kind="discussion",
            code=_CODE_DISCUSSION_UNGROUNDED,
            grounded_objectives=grounded_objectives,
            chunk_ids=chunk_ids,
            location=str(chunks_path or ""),
            issues=issues,
            nli_on=nli_on,
            nli_scorer=nli_scorer,
            chunk_text_by_id=chunk_text_by_id,
            obj_to_chunk_ids=obj_to_chunk_ids,
        )
        n_asgn_ungrounded = self._audit_items(
            items=assignments,
            kind="assignment",
            code=_CODE_ASSIGNMENT_UNGROUNDED,
            grounded_objectives=grounded_objectives,
            chunk_ids=chunk_ids,
            location=str(chunks_path or ""),
            issues=issues,
            nli_on=nli_on,
            nli_scorer=nli_scorer,
            chunk_text_by_id=chunk_text_by_id,
            obj_to_chunk_ids=obj_to_chunk_ids,
        )

        total = max(1, n_disc + n_asgn)
        ungrounded = n_disc_ungrounded + n_asgn_ungrounded
        # Both codes are WARNING — passed stays True (warning-day-1; the
        # GateConfig threshold gates on max_critical_issues=0, which warnings
        # never breach). score = grounded ratio for operator visibility.
        passed = True
        score = round((total - ungrounded) / total, 4)

        _emit_decision(
            capture,
            n_discussion=n_disc,
            n_assignment=n_asgn,
            n_discussion_ungrounded=n_disc_ungrounded,
            n_assignment_ungrounded=n_asgn_ungrounded,
            n_grounding_objectives=len(grounded_objectives),
            applied=True,
            passed=ungrounded == 0,
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            action="regenerate" if ungrounded else None,
        )

    # ----------------------------------------------------------------- helper

    @staticmethod
    def _nli_grounded(
        *,
        item: Dict[str, Any],
        src_ids: List[str],
        oid: str,
        scorer: Any,
        chunk_text_by_id: Dict[str, str],
        obj_to_chunk_ids: Dict[str, List[str]],
    ) -> Optional[bool]:
        """A5 NLI verdict for one item.

        Returns True (prompt text ENTAILED by cited chunks), False (not
        entailed / contradicted), or None when the check cannot run (no prompt
        text, no resolvable cited chunk texts, or the scorer degraded). None
        makes the caller fall back to the legacy refs-only verdict.
        """
        prompt = item.get("text") or item.get("prompt")
        if not (isinstance(prompt, str) and prompt.strip()):
            return None
        # Resolve cited chunk texts: explicit source_chunk_ids first, else the
        # item's objective_id → chunk-id map.
        cited_ids: List[str] = [c.lower() for c in src_ids]
        if not cited_ids and oid:
            cited_ids = list(obj_to_chunk_ids.get(oid.lower(), []))
        passages = [
            chunk_text_by_id[c] for c in cited_ids
            if c in chunk_text_by_id and chunk_text_by_id[c].strip()
        ]
        if not passages:
            return None
        try:
            report = scorer(prompt, passages, split_claims=False)
        except TypeError:
            report = scorer(prompt, passages)
        except Exception as exc:  # noqa: BLE001 — degrade to legacy
            logger.debug(
                "discussion_assignment_grounding: NLI scorer raised (%s); "
                "falling back to refs-only for this item.", exc,
            )
            return None
        if not getattr(report, "available", True):
            return None
        if int(getattr(report, "contradicted_count", 0) or 0) > 0:
            return False
        rate = float(getattr(report, "groundedness_rate", 0.0) or 0.0)
        return rate >= 1.0

    @staticmethod
    def _audit_items(
        *,
        items: List[Dict[str, Any]],
        kind: str,
        code: str,
        grounded_objectives: Set[str],
        chunk_ids: Set[str],
        location: str,
        issues: List[GateIssue],
        nli_on: bool = False,
        nli_scorer: Optional[Any] = None,
        chunk_text_by_id: Optional[Dict[str, str]] = None,
        obj_to_chunk_ids: Optional[Dict[str, List[str]]] = None,
    ) -> int:
        """Audit one item-type; append a WARNING per ungrounded item.

        Returns the count of ungrounded items. Legacy (refs-only) path: an item
        is grounded iff its explicit ``source_chunk_ids`` resolve to real chunks
        OR its ``objective_id`` is present in the chunk-grounded objective set.

        A5 NLI arm (``nli_on``): when the item carries prompt TEXT + resolvable
        cited chunk TEXTS + a scorer, the text-entailment verdict is
        authoritative for that item (grounded iff entailed). An item that can't
        run NLI falls back to the legacy verdict below.
        """
        chunk_text_by_id = chunk_text_by_id or {}
        obj_to_chunk_ids = obj_to_chunk_ids or {}
        n_ungrounded = 0
        for idx, item in enumerate(items):
            label = _item_label(item, kind, idx)
            src_ids = _item_source_chunk_ids(item)
            oid = _item_objective_id(item)

            # A5 — NLI text-grounding arm (authoritative when it can run).
            if nli_on and nli_scorer is not None:
                nli_verdict = DiscussionAssignmentGroundingValidator._nli_grounded(
                    item=item,
                    src_ids=src_ids,
                    oid=oid,
                    scorer=nli_scorer,
                    chunk_text_by_id=chunk_text_by_id,
                    obj_to_chunk_ids=obj_to_chunk_ids,
                )
                if nli_verdict is True:
                    continue  # text entailed by cited chunks → grounded.
                if nli_verdict is False:
                    n_ungrounded += 1
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code=code,
                            message=(
                                f"{kind} item {label!r} is ungrounded (NLI): its "
                                f"prompt text is not entailed by its cited source "
                                f"chunks."
                            ),
                            location=location,
                            suggestion=(
                                f"Re-author the {kind} so its prompt is supported "
                                f"by the cited source chunk text."
                            ),
                        )
                    )
                    continue
                # nli_verdict is None → could not run NLI; fall through to legacy.

            # (a) explicit source_chunk_ids resolving to real chunks.
            if src_ids and any(c.lower() in chunk_ids for c in src_ids):
                continue

            # (b) objective_id resolves to >= 1 source chunk.
            if oid and oid.lower() in grounded_objectives:
                continue

            # Ungrounded — neither path resolved.
            n_ungrounded += 1
            reason = (
                "carries no objective_id and no resolvable source_chunk_ids"
                if not oid and not src_ids
                else (
                    f"objective_id={oid!r} resolves to no source chunk's "
                    f"learning_outcome_refs"
                    if oid
                    else "declared source_chunk_ids resolve to no real chunk"
                )
            )
            issues.append(
                GateIssue(
                    severity="warning",
                    code=code,
                    message=(
                        f"{kind} item {label!r} is ungrounded: {reason}. A "
                        f"{kind} prompt must trace to its cited source chunks "
                        f"(real source_chunk_ids OR an objective_id present in "
                        f"some chunk's learning_outcome_refs)."
                    ),
                    location=location,
                    suggestion=(
                        f"Re-author the {kind} grounded in the objective's "
                        f"source chunks, or set the item's objective_id to a "
                        f"chunk-grounded objective."
                    ),
                )
            )
        return n_ungrounded


__all__ = ["DiscussionAssignmentGroundingValidator"]

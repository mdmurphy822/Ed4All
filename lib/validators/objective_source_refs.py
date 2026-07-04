"""Wave 1.6 W1.6.C — per-objective source-attribution consistency walk.

Gates a synthesized learning-objective set so every per-objective
``source_refs[]`` entry resolves against a real upstream universe — either
a chapter / section ID minted by the textbook structure extractor, or a
``dart:{slug}#{block_id}`` chunk reference harvested from the DART
chunkset emitted by the ``chunking`` workflow phase.

Pairs with W1.6.A (schema bump making ``source_refs[]`` a back-compat
``oneOf`` of legacy ``List[str]`` + structured ``List[{ref, chunk_ids[]}]``)
and W1.6.B (course-outliner emit + synthesizer Path-A/B emitting the
structured form). This validator closes the structural seam: a
synthesized objective referencing a non-existent chapter ID or DART
block ID is now flagged at the ``course_planning`` phase rather than
silently propagating to the rewrite tier.

Per-LO contract (mirrors the JSON-LD ``$defs.LearningObjective`` schema
in ``schemas/knowledge/courseforge_jsonld_v1.schema.json`` and the
back-compat ``oneOf`` in
``schemas/knowledge/objectives_v1.schema.json``):

1. **Legacy ``List[str]`` source_refs.** For each string entry, attempt
   resolution against the union of (chapter IDs ∪ section IDs ∪ DART
   chunk-id universe). Miss emits warning-severity GateIssue
   ``OBJECTIVE_SOURCE_REF_UNRESOLVED`` and routes
   ``action="regenerate"``. Wave 1.6 ships warning-only to preserve
   back-compat with the RDF/SHACL calibration corpus archive's existing emit.
2. **Structured ``{ref, chunk_ids[]}`` source_refs.**
   * ``ref`` not in ``textbook_structure.chapters[*].id`` ∪
     ``chapters[*].sections[*].id`` → warning-severity
     ``OBJECTIVE_SOURCE_NOT_IN_TEXTBOOK_STRUCTURE``,
     ``action="regenerate"``.
   * Any ``chunk_ids[*]`` entry not in the DART chunk-id universe →
     warning-severity ``OBJECTIVE_CHUNK_NOT_IN_DART_MANIFEST``,
     ``action="regenerate"``. Skipped when no
     ``dart_chunks_manifest_path`` is provided so legacy archives
     don't fail loud against an unavailable universe.
   * **Split-brain citation-resolution net (GAP 1).** In addition to
     the per-LO warning above, the validator aggregates every cited
     ``chunk_ids[*]`` across all structured objectives and, when a
     chunkset universe IS available (``dart_chunks_manifest_path``
     resolved + non-empty ``chunks.jsonl``), fires a single
     CRITICAL-severity ``ORPHANED_CITATIONS`` GateIssue the moment the
     unresolvable-citation rate exceeds 0. This catches the failure
     mode where a chunkset is renumbered AFTER objective synthesis so
     100% of cited chunk_ids point at ids that no longer exist — the
     old warning-only per-LO codes let that pass every gate. The
     aggregate issue carries counts (total cited, orphaned, rate,
     affected-LO count) + a sample of orphaned ids. It is skipped
     (graceful) whenever the chunkset is absent or empty, preserving
     legacy-archive behavior. NOTE: because the gate is wired at
     ``severity: warning`` / ``on_fail: warn`` in
     ``config/workflows.yaml`` today, ``passed=False`` from this
     critical issue is surfaced but does NOT yet block the workflow
     (executor only blocks on ``severity: critical`` gates). Flipping
     the gate to ``severity: critical`` / ``on_fail: block`` promotes
     this net to fail-closed.
3. **Empty ``source_refs[]``.**
   * On a chapter-level objective (CO-NN) → warning-severity
     ``OBJECTIVE_MISSING_SOURCE_REFS``, ``action="regenerate"``.
   * On a terminal objective (TO-NN) when
     ``require_to_attribution=False`` (the day-1 default) → silent
     pass to preserve back-compat with all existing archives where
     TOs have no source_refs at all.
   * On a TO when ``require_to_attribution=True`` → same warning fires.
4. **No upstream universes available** (neither
   ``textbook_structure_path`` nor ``dart_chunks_manifest_path``
   wired) → warning-severity ``OBJECTIVE_SOURCE_REFS_NO_UNIVERSE``
   GateIssue with ``passed=True, action=None``. Same graceful-degrade
   pattern ``RewriteSourceGroundingValidator`` uses for missing
   ``[embedding]`` extras.

Inputs contract:

* ``inputs["synthesized_objectives_path"]`` — required str / Path. Points
  at a ``synthesized_objectives.json`` file the validator loads + flattens
  via the same shape-tolerance helper ``AbcdObjectiveValidator`` uses
  (accepts both Courseforge synthesized form (``terminal_objectives`` /
  ``chapter_objectives``) AND LibV2-archive form (``terminal_outcomes`` /
  ``component_objectives``)).
* ``inputs["textbook_structure_path"]`` — optional str / Path. Points
  at a ``textbook_structure.json`` file. When absent the chapter / section
  universe is empty.
* ``inputs["dart_chunks_manifest_path"]`` — optional str / Path. Points
  at the ``LibV2/courses/<slug>/dart_chunks/manifest.json`` sidecar
  emitted by the ``chunking`` workflow phase. The validator loads the
  SIBLING ``chunks.jsonl`` (in the manifest's parent directory) and
  harvests chunk-id universe from each chunk's ``id`` field AND the
  union of every chunk's ``source.source_references[*].sourceId`` (the
  ``dart:{slug}#{block_id}`` shape that the synthesizer's
  ``_topic_objective_source_ref`` emits into structured
  ``chunk_ids[]``). Both forms are accepted as resolution targets so
  legacy emits and new emits both validate against the same universe.
* ``inputs["require_to_attribution"]`` — opt-in bool. Default ``False``.
  When ``True``, an empty ``source_refs[]`` on a TO fires
  ``OBJECTIVE_MISSING_SOURCE_REFS`` instead of silently passing.
* ``inputs["decision_capture"]`` — optional ``DecisionCapture`` instance.
  When provided, the validator emits one ``objective_source_ref_check``
  event per audited LO carrying the seven canonical signals
  (``lo_id``, ``hierarchy_level``, ``declared_refs_count``,
  ``unresolved_refs_count``, ``declared_chunk_ids_count``,
  ``unresolved_chunk_ids_count``, ``attribution_shape``).
* ``inputs["gate_id"]`` — optional override for the gate ID stamped on
  the returned ``GateResult``. Defaults to the validator name.

Cross-references:

* ``schemas/knowledge/objectives_v1.schema.json`` — the canonical
  back-compat ``oneOf`` for ``source_refs[]`` (W1.6.A).
* ``MCP/tools/_content_gen_helpers.py::_topic_objective_source_ref``
  (W1.6.B) — the synthesizer surface that emits the structured
  ``{ref, chunk_ids[]}`` shape this validator audits.
* ``schemas/events/decision_event.schema.json::decision_type.enum`` —
  the ``objective_source_ref_check`` enum member added alongside
  this validator.
* ``Courseforge/router/inter_tier_gates.py::BlockSourceRefValidator``
  — the Wave 1.5 W1.5.C precedent the algorithm shape mirrors
  (per-block universe resolution + per-claim attribution walk).
* ``lib/validators/abcd_objective.py::_flatten_objectives`` — the
  module-level shape-tolerance helper this validator imports
  directly so the LO container-shape contract is centralised.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

# Reuse the canonical shape-tolerance helper directly. ``_flatten_objectives``
# is a module-level function (not a method) on ``abcd_objective.py``
# (see ``lib/validators/abcd_objective.py:134``), so the import is
# clean — no duplication and no need to factor it out further. The
# helper accepts both Courseforge form (``terminal_objectives`` /
# ``chapter_objectives``) AND LibV2-archive form (``terminal_outcomes``
# / ``component_objectives``) so both surfaces flatten identically.
from lib.validators.abcd_objective import _flatten_objectives

logger = logging.getLogger(__name__)


#: Cap the number of issues emitted so a uniformly-broken LO batch
#: doesn't drown the gate report. Mirrors the cap used by
#: ``AbcdObjectiveValidator`` and the four ``Block*Validator`` adapters
#: in ``Courseforge/router/inter_tier_gates.py``.
_ISSUE_LIST_CAP: int = 50


#: Canonical decision-type string emitted by this validator. Must
#: appear in ``schemas/events/decision_event.schema.json::decision_type.enum``;
#: alphabetised position is enforced by the schema's ordering contract.
_DECISION_TYPE: str = "objective_source_ref_check"


#: Pattern for canonical terminal-objective IDs (``TO-NN``). The hierarchy
#: classification is purely lexical — TO-NN vs everything else (CO-NN
#: in practice). The validator never gates on the hierarchy directly;
#: it gates on the ``require_to_attribution`` flag against the inferred
#: hierarchy.
_TO_PREFIX: str = "TO-"


def _lo_id(lo: Mapping[str, Any]) -> str:
    """Return a printable LO ID for issue messages.

    Falls back to ``"<unknown-id>"`` when the LO has no canonical ``id``;
    integrity gates upstream catch missing IDs as a separate failure
    mode, so we don't redundantly flag it here.
    """
    raw = lo.get("id") or lo.get("objective_id") or ""
    raw = str(raw).strip()
    return raw or "<unknown-id>"


def _hierarchy_level(lo_id_value: str) -> str:
    """Return ``"terminal"`` for a TO-NN ID, ``"chapter"`` otherwise.

    The classification is lexical: an LO whose ID starts with ``"TO-"``
    is a terminal objective; everything else (``CO-NN`` in practice) is
    treated as chapter-level. The validator's ``require_to_attribution``
    semantics gates on this classification.
    """
    if lo_id_value.upper().startswith(_TO_PREFIX):
        return "terminal"
    return "chapter"


def _load_textbook_structure_universe(
    textbook_structure_path: Optional[Path],
) -> Set[str]:
    """Harvest the chapter + section ID universe from textbook_structure.json.

    Walks ``chapters[*].id`` and ``chapters[*].sections[*].id`` (recursive
    one level — sections themselves don't carry sub-sections in the
    Wave-24 emit; defensive for future shapes via a flat walk).

    Returns an empty set when the path is missing or unreadable. The
    caller decides how to handle an empty universe (graceful-degrade
    vs strict-fail).
    """
    if textbook_structure_path is None or not textbook_structure_path.exists():
        return set()

    try:
        data = json.loads(textbook_structure_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(
            "ObjectiveSourceRefValidator: failed to load "
            "textbook_structure_path %s: %s",
            textbook_structure_path,
            exc,
        )
        return set()

    universe: Set[str] = set()
    if not isinstance(data, Mapping):
        return universe

    chapters = data.get("chapters") or []
    if not isinstance(chapters, list):
        return universe

    for chapter in chapters:
        if not isinstance(chapter, Mapping):
            continue
        ch_id = chapter.get("id")
        if isinstance(ch_id, str) and ch_id.strip():
            universe.add(ch_id.strip())
        sections = chapter.get("sections") or []
        if not isinstance(sections, list):
            continue
        for section in sections:
            if not isinstance(section, Mapping):
                continue
            sec_id = section.get("id") or section.get("section_id")
            if isinstance(sec_id, str) and sec_id.strip():
                universe.add(sec_id.strip())

    return universe


def _load_dart_chunks_universe(
    dart_chunks_manifest_path: Optional[Path],
) -> Set[str]:
    """Harvest the DART chunk-id universe from the chunkset sidecar.

    The plan §2 Fix 1.6.C calls for harvesting from
    ``dart_chunks/manifest.json`` directly, but the canonical
    ``schemas/library/chunkset_manifest.schema.json`` does NOT list any
    ``chunks[]`` entries on the manifest itself — it carries
    ``chunks_sha256`` + ``chunker_version`` + ``chunkset_kind`` (+
    optional ``chunks_count`` / ``generated_at``) and nothing else.
    The actual chunk listing lives in the sibling ``chunks.jsonl``
    file in the same directory (per
    ``MCP/tools/pipeline_tools.py::_run_dart_chunking`` at ``:7589-7623``).

    To match the plan's intent, this helper resolves the sibling
    ``chunks.jsonl`` from the manifest's parent directory and harvests
    BOTH:

    * Each chunk's top-level ``"id"`` field (the per-chunk ID, e.g.
      ``"demo_course_chunk_00001"``).
    * The union of every chunk's ``source.source_references[*].sourceId``
      (the ``dart:{slug}#{block_id}`` shape that the synthesizer's
      ``_topic_objective_source_ref`` emits into structured
      ``chunk_ids[]`` per
      ``MCP/tools/_content_gen_helpers.py::_topic_objective_source_ref:1423``).

    Both forms are accepted as valid resolution targets so legacy
    chunk-id emits AND structured-source-ref emits both validate
    against the same merged universe.

    Returns an empty set when the manifest path is missing / unreadable
    OR the sibling chunks.jsonl is missing / unreadable. The caller
    decides how to handle an empty universe.
    """
    if dart_chunks_manifest_path is None:
        return set()
    if not dart_chunks_manifest_path.exists():
        return set()

    chunks_jsonl_path = dart_chunks_manifest_path.parent / "chunks.jsonl"
    if not chunks_jsonl_path.exists():
        return set()

    universe: Set[str] = set()
    try:
        with chunks_jsonl_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    # Defensive: a malformed chunk doesn't block the
                    # whole universe build; skip it and continue.
                    logger.debug(
                        "ObjectiveSourceRefValidator: malformed chunk on "
                        "line %d of %s; skipping.",
                        line_no, chunks_jsonl_path,
                    )
                    continue
                if not isinstance(chunk, Mapping):
                    continue
                chunk_id = chunk.get("id")
                if isinstance(chunk_id, str) and chunk_id.strip():
                    universe.add(chunk_id.strip())
                source = chunk.get("source")
                if not isinstance(source, Mapping):
                    continue
                src_refs = source.get("source_references") or []
                if not isinstance(src_refs, list):
                    continue
                for ref in src_refs:
                    if not isinstance(ref, Mapping):
                        continue
                    src_id = ref.get("sourceId")
                    if isinstance(src_id, str) and src_id.strip():
                        universe.add(src_id.strip())
    except OSError as exc:
        logger.warning(
            "ObjectiveSourceRefValidator: failed to read chunks.jsonl at "
            "%s: %s",
            chunks_jsonl_path,
            exc,
        )
        return set()

    return universe


def _emit_decision(
    capture: Any,
    *,
    decision: str,
    rationale: str,
    context: Optional[str] = None,
) -> None:
    """Emit one ``objective_source_ref_check`` decision event.

    Mirrors the swallow-on-error pattern from
    ``lib/validators/abcd_objective.py::_emit_decision``: capture
    failures must not break the gate. Errors are logged at warning
    level so postmortems can correlate but the gate result still ships.
    """
    if capture is None:
        return
    try:
        capture.log_decision(
            decision_type=_DECISION_TYPE,
            decision=decision,
            rationale=rationale,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.warning(
            "ObjectiveSourceRefValidator: decision_capture.log_decision "
            "failed: %s",
            exc,
        )


def _classify_attribution_shape(
    source_refs: Any,
) -> str:
    """Return the canonical ``attribution_shape`` enum value for an LO.

    Three values land in the decision-capture event so the audit trail
    records which arm of the W1.6.A ``oneOf`` the synthesizer emitted:

    * ``"empty"`` — no ``source_refs`` field, or an empty list.
    * ``"legacy_string_list"`` — ``List[str]`` (pre-W1.6.A back-compat).
    * ``"structured_per_ref"`` — ``List[{ref, chunk_ids[]}]`` (W1.6.B
      synthesizer + course-outliner emit).

    Mixed-shape entries fall through to ``"structured_per_ref"`` since
    the schema's ``oneOf`` upstream rejects mixed lists; the per-entry
    isinstance check inside ``validate()`` defends against any leak.
    """
    if not isinstance(source_refs, list) or not source_refs:
        return "empty"
    # Inspect the first entry's shape — the schema's ``oneOf``
    # guarantees the list is homogeneous, so a single sample is enough.
    first = source_refs[0]
    if isinstance(first, str):
        return "legacy_string_list"
    return "structured_per_ref"


def _coerce_objectives_path(
    inputs: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Optional[GateIssue]]:
    """Resolve the LO list from ``inputs["synthesized_objectives_path"]``.

    Returns ``(los, error_issue)``; ``error_issue`` is non-None only on
    structural failures (path missing, JSON unparseable). An empty LO
    list is NOT an error — the validator's contract is a no-op pass
    on empty input (back-compat with the empty-corpus path).
    """
    path_raw = inputs.get("synthesized_objectives_path")
    if not path_raw:
        # Per-call test ergonomics: tests can also pass ``inputs["objectives"]``
        # directly to skip the path-resolution layer.
        explicit = inputs.get("objectives")
        if explicit is not None:
            return _flatten_objectives(explicit), None
        return [], GateIssue(
            severity="critical",
            code="OBJECTIVE_SOURCE_REFS_PATH_MISSING",
            message=(
                "ObjectiveSourceRefValidator requires "
                "``synthesized_objectives_path`` in inputs (or an explicit "
                "``objectives`` list for test ergonomics)."
            ),
        )

    p = Path(path_raw)
    if not p.exists():
        return [], GateIssue(
            severity="critical",
            code="OBJECTIVE_SOURCE_REFS_PATH_MISSING",
            message=(
                f"synthesized_objectives_path {str(p)!r} does not exist; "
                f"cannot audit per-objective source attribution."
            ),
            location=str(p),
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], GateIssue(
            severity="critical",
            code="OBJECTIVE_SOURCE_REFS_PATH_UNREADABLE",
            message=(
                f"Failed to parse synthesized_objectives_path {str(p)!r}: "
                f"{exc.__class__.__name__}: {exc}"
            ),
            location=str(p),
        )
    return _flatten_objectives(data), None


class ObjectiveSourceRefValidator:
    """Wave 1.6 W1.6.C per-objective source-attribution gate.

    Validator-protocol-compatible class wired into
    ``textbook_to_course::course_planning::objective_source_refs``.
    Emits regenerate-action GateIssues on structural source-ref misses
    (LO references a chapter / section / chunk that doesn't exist
    upstream). Wave 1.6 ships at WARNING severity day-1 to preserve
    back-compat with the RDF/SHACL calibration corpus's existing
    emit; promotion to CRITICAL waits for future-wave calibration data
    (see plan §6.3 risk note).

    The action-resolution chain mirrors
    ``BlockSourceRefValidator._action_for_result`` (Wave 1.5 W1.5.C):

    * Any critical structural-miss code → ``action="block"`` (dormant
      day-1 since Wave 1.6 ships only warning-severity codes).
    * Any warning-level miss across any code → ``action="regenerate"``.
    * No misses → ``action=None``.
    """

    name = "objective_source_refs"
    version = "1.0.0"  # Wave 1.6 W1.6.C

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)

        objectives, err = _coerce_objectives_path(inputs)
        if err is not None:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[err],
                action="block",
            )

        # Empty LO set is a no-op pass (back-compat with empty-corpus path).
        if not objectives:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        # Resolve the dual universe of valid refs.
        ts_path_raw = inputs.get("textbook_structure_path")
        ts_path = Path(ts_path_raw) if ts_path_raw else None
        chunks_path_raw = inputs.get("dart_chunks_manifest_path")
        chunks_path = Path(chunks_path_raw) if chunks_path_raw else None

        textbook_universe = _load_textbook_structure_universe(ts_path)
        chunks_universe = _load_dart_chunks_universe(chunks_path)

        require_to = bool(inputs.get("require_to_attribution", False))
        capture = inputs.get("decision_capture") or inputs.get("capture")

        # Graceful-degrade path: neither universe is available. Emit a
        # warning, pass the gate, and route action=None so the
        # workflow doesn't fail-close on legacy / pre-Phase-7 archives.
        # Mirrors the empty-extras pattern in
        # ``RewriteSourceGroundingValidator`` (per-validator graceful
        # degrade is the project convention for missing optional
        # upstream universes).
        no_universes = (
            ts_path_raw is None
            and chunks_path_raw is None
        )
        if no_universes:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[GateIssue(
                    severity="warning",
                    code="OBJECTIVE_SOURCE_REFS_NO_UNIVERSE",
                    message=(
                        "ObjectiveSourceRefValidator received neither "
                        "``textbook_structure_path`` nor "
                        "``dart_chunks_manifest_path`` — no universe "
                        "available to resolve per-LO source_refs against. "
                        "Gate passes (graceful-degrade for legacy / "
                        "pre-Phase-7 archives)."
                    ),
                    suggestion=(
                        "Wire the upstream phase outputs "
                        "``objective_extraction.textbook_structure_path`` and "
                        "``chunking.dart_chunks_path`` (resolves to the "
                        "sibling manifest.json) into this gate's input "
                        "builder."
                    ),
                )],
                action=None,
            )

        # Walk the LO set.
        issues: List[GateIssue] = []
        audited = 0
        passed_count = 0
        # Track whether we saw at least one regenerate-worthy miss so
        # the action resolver can short-circuit at end of walk.
        had_warning_miss = False
        had_critical_miss = False
        # Union universe used by the legacy List[str] arm — a string
        # entry can be either a chapter / section ID OR a chunk-id, so
        # we accept resolution against either.
        union_universe: Set[str] = textbook_universe | chunks_universe

        # GAP 1 — split-brain citation-resolution net. Aggregate every
        # cited chunk_id across all structured source_refs so a
        # chunkset renumbered after synthesis (100% orphaned citations)
        # trips a single CRITICAL ORPHANED_CITATIONS issue instead of
        # slipping through the per-LO warning arm. Only meaningful when
        # a chunkset universe is actually available.
        chunks_universe_available = bool(
            chunks_path_raw is not None and chunks_universe
        )
        total_cited_chunk_ids = 0
        orphaned_citation_ids: List[str] = []
        orphaned_citation_lo_ids: Set[str] = set()

        for lo in objectives:
            if not isinstance(lo, Mapping):
                continue
            audited += 1
            lo_id_value = _lo_id(lo)
            hierarchy = _hierarchy_level(lo_id_value)

            source_refs = lo.get("source_refs")
            shape = _classify_attribution_shape(source_refs)

            declared_refs_count = (
                len(source_refs) if isinstance(source_refs, list) else 0
            )
            unresolved_refs_count = 0
            declared_chunk_ids_count = 0
            unresolved_chunk_ids_count = 0
            lo_passed = True

            if shape == "empty":
                # Apply the empty-source-refs gating per hierarchy.
                if hierarchy == "chapter" or require_to:
                    if len(issues) < _ISSUE_LIST_CAP:
                        issues.append(GateIssue(
                            severity="warning",
                            code="OBJECTIVE_MISSING_SOURCE_REFS",
                            message=(
                                f"LO {lo_id_value!r} ({hierarchy}-level) "
                                f"has no ``source_refs`` field, or it is "
                                f"empty. Wave 1.6 W1.6.B requires every "
                                f"chapter-level objective to declare at "
                                f"least one structured "
                                f"``{{ref, chunk_ids[]}}`` entry; "
                                f"terminal objectives are required when "
                                f"``require_to_attribution=True``."
                            ),
                            location=lo_id_value,
                            suggestion=(
                                "Re-run the synthesizer with the "
                                "Wave-1.6 W1.6.B emit path; ensure the "
                                "topic carries ``source_file`` + "
                                "``dart_block_ids`` so "
                                "``_topic_objective_source_ref`` can "
                                "build a structured entry."
                            ),
                        ))
                    had_warning_miss = True
                    lo_passed = False
                # else: TO with require_to_attribution=False → silent pass.
            elif shape == "legacy_string_list":
                # Each string entry must resolve against the union
                # universe. A miss emits OBJECTIVE_SOURCE_REF_UNRESOLVED.
                for entry in source_refs:
                    if not isinstance(entry, str):
                        # Mixed-shape entry — schema rejects upstream;
                        # defensive skip.
                        continue
                    entry_stripped = entry.strip()
                    if not entry_stripped:
                        continue
                    if not union_universe:
                        # Universe is the union; if both are empty (e.g.
                        # textbook_structure_path provided but the file
                        # is empty), we have nothing to resolve against
                        # and can't claim a miss.
                        continue
                    if entry_stripped not in union_universe:
                        unresolved_refs_count += 1
                        if len(issues) < _ISSUE_LIST_CAP:
                            issues.append(GateIssue(
                                severity="warning",
                                code="OBJECTIVE_SOURCE_REF_UNRESOLVED",
                                message=(
                                    f"LO {lo_id_value!r} legacy "
                                    f"``source_refs[]`` entry "
                                    f"{entry_stripped!r} does not "
                                    f"resolve against the union of "
                                    f"chapter / section IDs and DART "
                                    f"chunk-id universe (sizes: "
                                    f"textbook={len(textbook_universe)}, "
                                    f"chunks={len(chunks_universe)})."
                                ),
                                location=lo_id_value,
                                suggestion=(
                                    "Re-roll the upstream emitter so the "
                                    "ref matches a real chapter / section "
                                    "ID from textbook_structure.json or a "
                                    "chunk-id / dart:slug#block_id from "
                                    "the DART chunkset."
                                ),
                            ))
                if unresolved_refs_count > 0:
                    had_warning_miss = True
                    lo_passed = False
            else:
                # Structured shape: per-entry resolution.
                for entry in source_refs:
                    if not isinstance(entry, dict):
                        # Mixed-shape entry — defensive skip.
                        continue
                    ref_field = entry.get("ref")
                    chunk_ids = entry.get("chunk_ids") or []

                    # Resolve the ref against the textbook universe.
                    if (
                        isinstance(ref_field, str)
                        and ref_field.strip()
                        and textbook_universe
                        and ref_field.strip() not in textbook_universe
                    ):
                        unresolved_refs_count += 1
                        if len(issues) < _ISSUE_LIST_CAP:
                            issues.append(GateIssue(
                                severity="warning",
                                code="OBJECTIVE_SOURCE_NOT_IN_TEXTBOOK_STRUCTURE",
                                message=(
                                    f"LO {lo_id_value!r} structured "
                                    f"``source_refs[]`` entry "
                                    f"ref={ref_field.strip()!r} does not "
                                    f"resolve against "
                                    f"textbook_structure.chapters[*].id ∪ "
                                    f"chapters[*].sections[*].id "
                                    f"(universe size: "
                                    f"{len(textbook_universe)})."
                                ),
                                location=lo_id_value,
                                suggestion=(
                                    "Re-run the synthesizer; ensure the "
                                    "topic's chapter_id / heading slug "
                                    "matches a chapter or section in "
                                    "textbook_structure.json."
                                ),
                            ))
                        had_warning_miss = True
                        lo_passed = False

                    # Resolve each chunk_id against the DART chunks
                    # universe. Skip when no manifest path was provided
                    # so we don't fail loud against an unavailable
                    # universe.
                    if not isinstance(chunk_ids, list):
                        continue
                    declared_chunk_ids_count += len(chunk_ids)
                    if chunks_path_raw is None:
                        # Manifest absent → can't fail-close on chunk_ids.
                        continue
                    if not chunks_universe:
                        # Manifest path provided but universe is empty
                        # (e.g. chunks.jsonl missing or empty). Skip the
                        # check rather than firing a miss against an
                        # unavailable universe — the missing-universe
                        # case is already covered by the
                        # OBJECTIVE_SOURCE_REFS_NO_UNIVERSE path
                        # (which only fires when BOTH paths are absent).
                        continue
                    for chunk_id in chunk_ids:
                        if not isinstance(chunk_id, str):
                            continue
                        cid = chunk_id.strip()
                        if not cid:
                            continue
                        # Feed the GAP-1 aggregate: this chunk_id is a
                        # real citation resolved against a live universe.
                        total_cited_chunk_ids += 1
                        if cid not in chunks_universe:
                            unresolved_chunk_ids_count += 1
                            orphaned_citation_ids.append(cid)
                            orphaned_citation_lo_ids.add(lo_id_value)
                            if len(issues) < _ISSUE_LIST_CAP:
                                issues.append(GateIssue(
                                    severity="warning",
                                    code="OBJECTIVE_CHUNK_NOT_IN_DART_MANIFEST",
                                    message=(
                                        f"LO {lo_id_value!r} structured "
                                        f"``source_refs[]`` entry "
                                        f"ref={ref_field!r} cites "
                                        f"chunk_id {cid!r} which does "
                                        f"not resolve against the DART "
                                        f"chunkset universe (size: "
                                        f"{len(chunks_universe)})."
                                    ),
                                    location=lo_id_value,
                                    suggestion=(
                                        "Verify the topic's "
                                        "``dart_block_ids`` map to real "
                                        "DART block IDs that survived "
                                        "the chunking phase, or re-run "
                                        "stage_dart_outputs + chunking "
                                        "to refresh the chunkset."
                                    ),
                                ))
                            had_warning_miss = True
                            lo_passed = False

            if lo_passed:
                passed_count += 1

            # Emit one decision-capture event per audited LO carrying
            # the seven required signals.
            decision_label = (
                "passed" if lo_passed
                else f"failed:declared={declared_refs_count}"
            )
            rationale = (
                f"ObjectiveSourceRefValidator audited LO {lo_id_value} "
                f"({hierarchy}-level): attribution_shape={shape!r}, "
                f"declared_refs_count={declared_refs_count}, "
                f"unresolved_refs_count={unresolved_refs_count}, "
                f"declared_chunk_ids_count={declared_chunk_ids_count}, "
                f"unresolved_chunk_ids_count={unresolved_chunk_ids_count}. "
                f"Universes: textbook={len(textbook_universe)}, "
                f"chunks={len(chunks_universe)}. "
                f"Wave 1.6 W1.6.C structural seam — warning-severity "
                f"day-1 per plan §6.3 risk note."
            )
            context = (
                f"lo_id={lo_id_value}; hierarchy_level={hierarchy}; "
                f"declared_refs_count={declared_refs_count}; "
                f"unresolved_refs_count={unresolved_refs_count}; "
                f"declared_chunk_ids_count={declared_chunk_ids_count}; "
                f"unresolved_chunk_ids_count={unresolved_chunk_ids_count}; "
                f"attribution_shape={shape}"
            )
            _emit_decision(
                capture,
                decision=decision_label,
                rationale=rationale,
                context=context,
            )

        # GAP 1 — split-brain net. When a chunkset universe is available
        # and ANY cited chunk_id failed to resolve, fire a single
        # CRITICAL aggregate ORPHANED_CITATIONS issue with counts +
        # sample ids. This is the net that catches a chunkset renumbered
        # after synthesis (unresolvable rate up to 1.0) which the
        # per-LO warning arm alone let pass every gate.
        if (
            chunks_universe_available
            and total_cited_chunk_ids > 0
            and orphaned_citation_ids
        ):
            orphan_count = len(orphaned_citation_ids)
            unresolvable_rate = round(orphan_count / total_cited_chunk_ids, 4)
            # De-dupe for the sample while preserving first-seen order.
            unique_orphans = list(dict.fromkeys(orphaned_citation_ids))
            sample_ids = unique_orphans[:10]
            affected_los = len(orphaned_citation_lo_ids)
            issues.append(GateIssue(
                severity="critical",
                code="ORPHANED_CITATIONS",
                message=(
                    f"{orphan_count} of {total_cited_chunk_ids} cited "
                    f"chunk_ids ({unresolvable_rate:.1%}) across "
                    f"{affected_los} objective(s) do NOT resolve against "
                    f"the CURRENT DART chunkset (universe size "
                    f"{len(chunks_universe)}). This is the split-brain "
                    f"signature — the chunkset was renumbered / "
                    f"regenerated after these objectives were synthesized, "
                    f"orphaning their source citations. Sample orphaned "
                    f"chunk_ids: {sample_ids}."
                ),
                suggestion=(
                    "Re-synthesize the objectives against the current "
                    "chunkset, or re-run --reuse-objectives with the "
                    "chunkset that produced them. Never ship an "
                    "objectives artifact whose citations point at a "
                    "stale chunk numbering."
                ),
            ))
            _emit_decision(
                capture,
                decision=(
                    f"orphaned_citations:{orphan_count}/"
                    f"{total_cited_chunk_ids}"
                ),
                rationale=(
                    f"ObjectiveSourceRefValidator GAP-1 split-brain net: "
                    f"{orphan_count}/{total_cited_chunk_ids} cited chunk_ids "
                    f"({unresolvable_rate}) across {affected_los} objectives "
                    f"failed to resolve against the current chunkset "
                    f"(universe size {len(chunks_universe)}). Emitting "
                    f"CRITICAL ORPHANED_CITATIONS. Sample: {sample_ids}."
                ),
                context=(
                    f"total_cited_chunk_ids={total_cited_chunk_ids}; "
                    f"orphaned_chunk_ids={orphan_count}; "
                    f"unresolvable_rate={unresolvable_rate}; "
                    f"affected_los={affected_los}"
                ),
            )

        # Score = pass rate over audited LOs. ``passed`` is True iff no
        # critical-severity issues fired (warnings don't fail the gate
        # under the Wave 1.6 warning-severity wiring).
        critical_count = sum(1 for i in issues if i.severity == "critical")
        had_critical_miss = critical_count > 0
        passed = critical_count == 0
        score = 1.0 if audited == 0 else round(passed_count / audited, 4)

        # Action resolution mirrors BlockSourceRefValidator's W1.5.C
        # contract:
        # - critical structural miss → "block" (dormant at warning-only
        #   day-1; flips on when a future wave promotes the codes).
        # - warning miss → "regenerate".
        # - no miss → None.
        action: Optional[str]
        if had_critical_miss:
            action = "block"
        elif had_warning_miss:
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
    "ObjectiveSourceRefValidator",
]

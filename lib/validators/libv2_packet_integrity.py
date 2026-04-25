"""LibV2 Packet Integrity Validator (Wave 75 Worker D).

Runs SHACL-style integrity rules over a self-contained LibV2 archive
(`LibV2/courses/<slug>/`) and returns a typed result object describing
which rules passed and which issues fired.

This is a *post-hoc* operator tool — not a workflow gate. The
``LibV2ManifestValidator`` (Wave 23) already gates the
``libv2_archival`` phase of the ``textbook_to_course`` workflow on
manifest schema + on-disk artifact integrity. This validator drills
into the *internal consistency* of the archive: chunks, objectives,
graphs, and how they cross-reference each other.

Rules
-----

* ``unique_chunk_ids`` (critical) — every ``chunks.jsonl`` ``id`` is unique.
* ``refs_resolve`` (critical) — every chunk's ``learning_outcome_refs``
  resolves against ``objectives.json`` (case-insensitive).
* ``co_has_parent`` (critical) — every component objective has a
  ``parent_terminal`` and that terminal exists in
  ``terminal_outcomes``.
* ``no_comma_refs`` (critical) — no chunk ref entry contains a literal
  comma. (Post-Worker-A normalization should mean zero.)
* ``graph_edges_resolve`` (critical) — every edge ``source`` and
  ``target`` in ``concept_graph`` and ``pedagogy_graph`` resolves to a
  node id within that same graph.
* ``assessment_has_objective`` (warning) — every chunk with
  ``chunk_type=assessment_item`` has at least one
  ``learning_outcome_refs`` entry.
* ``to_has_teaching_and_assessment`` (warning) — every terminal
  outcome has at least one teaching chunk
  (``chunk_type ∈ {explanation, overview, summary, example}``) and at
  least one assessment chunk (``assessment_item``) — directly or via
  one of its component objectives.
* ``domain_concept_has_chunk`` (warning) — every ``concept_graph``
  node with ``class=DomainConcept`` appears in at least one chunk's
  ``concept_tags`` or text.
* ``scaffolding_not_assessed`` (warning) — concept_graph nodes whose
  class is pedagogical scaffolding
  (``PedagogicalMarker``, ``AssessmentOption``, ``LowSignal``,
  ``InstructionalArtifact``) never appear as the *target* of
  ``derived-from-objective`` or ``assesses`` edges in
  ``concept_graph_semantic.json``.

The validator returns ``ValidationResult`` (see below); the CLI
wrapper turns this into JSON or a human-readable summary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Result shape
# ---------------------------------------------------------------------- #


@dataclass
class ValidationIssue:
    """One issue raised by a packet integrity rule."""

    rule: str
    severity: str  # "critical" | "warning"
    issue_code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "issue_code": self.issue_code,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class ValidationResult:
    """Aggregate result for ``PacketIntegrityValidator.validate``."""

    archive_root: str
    rules_run: int = 0
    rules_passed: int = 0
    rules_failed: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "archive_root": self.archive_root,
            "rules_run": self.rules_run,
            "rules_passed": self.rules_passed,
            "rules_failed": self.rules_failed,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------- #
# Rule severity table
# ---------------------------------------------------------------------- #
#
# Intentionally module-level + public so tests + the CLI can introspect
# the catalog without instantiating the validator.

RULE_SEVERITY: Dict[str, str] = {
    "unique_chunk_ids": "critical",
    "refs_resolve": "critical",
    "co_has_parent": "critical",
    "no_comma_refs": "critical",
    "graph_edges_resolve": "critical",
    "assessment_has_objective": "warning",
    "to_has_teaching_and_assessment": "warning",
    "domain_concept_has_chunk": "warning",
    "scaffolding_not_assessed": "warning",
}

# Chunk types that *teach* their referenced LOs (vs. assess them or
# scaffold them as exercise prompts).
TEACHING_CHUNK_TYPES: Set[str] = {
    "explanation",
    "overview",
    "summary",
    "example",
}
ASSESSMENT_CHUNK_TYPE = "assessment_item"

# Concept-graph node classes that scaffold rather than represent
# domain content. ``scaffolding_not_assessed`` flags them when they
# appear as edge targets of pedagogical edges.
SCAFFOLDING_CLASSES: Set[str] = {
    "PedagogicalMarker",
    "AssessmentOption",
    "LowSignal",
    "InstructionalArtifact",
}

# Edge ``type`` values in concept_graph_semantic.json that should
# point at *content* (DomainConcept / Misconception / LearningObjective)
# rather than scaffolding.
PEDAGOGICAL_EDGE_TYPES: Set[str] = {
    "derived-from-objective",
    "assesses",
}


# ---------------------------------------------------------------------- #
# Validator
# ---------------------------------------------------------------------- #


class PacketIntegrityValidator:
    """Validates a LibV2 archive's internal consistency.

    Usage::

        validator = PacketIntegrityValidator()
        result = validator.validate(Path("LibV2/courses/<slug>"))
    """

    name = "libv2_packet_integrity"
    version = "1.0.0"

    def validate(self, archive_root: Path) -> ValidationResult:
        """Run every rule and return a ``ValidationResult``."""
        archive_root = Path(archive_root)
        result = ValidationResult(archive_root=str(archive_root))

        if not archive_root.exists() or not archive_root.is_dir():
            result.rules_run = 1
            result.rules_failed = 1
            result.issues.append(
                ValidationIssue(
                    rule="archive_exists",
                    severity="critical",
                    issue_code="ARCHIVE_NOT_FOUND",
                    message=f"Archive root does not exist: {archive_root}",
                    context={"archive_root": str(archive_root)},
                )
            )
            result.summary = {"error": "archive_not_found"}
            return result

        # ----- Load the four data sources -------------------------------
        chunks = self._load_chunks(archive_root, result)
        objectives = self._load_objectives(archive_root, result)
        concept_graph = self._load_json_dict(
            archive_root / "graph" / "concept_graph.json", result, "concept_graph"
        )
        concept_graph_semantic = self._load_json_dict(
            archive_root / "graph" / "concept_graph_semantic.json",
            result,
            "concept_graph_semantic",
        )
        pedagogy_graph = self._load_json_dict(
            archive_root / "graph" / "pedagogy_graph.json", result, "pedagogy_graph"
        )

        # ----- Run rules ------------------------------------------------
        rule_runners = [
            ("unique_chunk_ids", self._rule_unique_chunk_ids, (chunks,)),
            (
                "refs_resolve",
                self._rule_refs_resolve,
                (chunks, objectives),
            ),
            ("co_has_parent", self._rule_co_has_parent, (objectives,)),
            ("no_comma_refs", self._rule_no_comma_refs, (chunks,)),
            (
                "graph_edges_resolve",
                self._rule_graph_edges_resolve,
                (concept_graph, pedagogy_graph),
            ),
            (
                "assessment_has_objective",
                self._rule_assessment_has_objective,
                (chunks,),
            ),
            (
                "to_has_teaching_and_assessment",
                self._rule_to_has_teaching_and_assessment,
                (chunks, objectives),
            ),
            (
                "domain_concept_has_chunk",
                self._rule_domain_concept_has_chunk,
                (concept_graph, chunks),
            ),
            (
                "scaffolding_not_assessed",
                self._rule_scaffolding_not_assessed,
                (concept_graph, concept_graph_semantic),
            ),
        ]

        for rule_name, runner, args in rule_runners:
            result.rules_run += 1
            issues_before = len(result.issues)
            try:
                runner(rule_name, result, *args)
            except Exception as exc:  # pragma: no cover (defensive)
                logger.exception("Rule %s raised", rule_name)
                result.issues.append(
                    ValidationIssue(
                        rule=rule_name,
                        severity="critical",
                        issue_code="RULE_EXCEPTION",
                        message=f"Rule {rule_name} raised an exception: {exc}",
                        context={"exception": str(exc)},
                    )
                )
            issues_after = len(result.issues)
            if issues_after == issues_before:
                result.rules_passed += 1
            else:
                result.rules_failed += 1

        # ----- Summary --------------------------------------------------
        result.summary = self._build_summary(
            result,
            chunks=chunks,
            objectives=objectives,
            concept_graph=concept_graph,
            concept_graph_semantic=concept_graph_semantic,
            pedagogy_graph=pedagogy_graph,
        )
        return result

    # ------------------------------------------------------------------ #
    # Loaders
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_chunks(archive_root: Path, result: ValidationResult) -> List[Dict[str, Any]]:
        """Load ``corpus/chunks.jsonl`` (one JSON object per line)."""
        chunks_path = archive_root / "corpus" / "chunks.jsonl"
        if not chunks_path.exists():
            result.issues.append(
                ValidationIssue(
                    rule="load_chunks",
                    severity="critical",
                    issue_code="MISSING_CHUNKS_JSONL",
                    message=f"Required file missing: {chunks_path}",
                    context={"path": str(chunks_path)},
                )
            )
            return []
        chunks: List[Dict[str, Any]] = []
        try:
            with chunks_path.open("r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        result.issues.append(
                            ValidationIssue(
                                rule="load_chunks",
                                severity="critical",
                                issue_code="INVALID_CHUNK_JSON",
                                message=(
                                    f"Failed to parse chunk JSON at line "
                                    f"{lineno}: {exc}"
                                ),
                                context={"line": lineno, "path": str(chunks_path)},
                            )
                        )
        except OSError as exc:
            result.issues.append(
                ValidationIssue(
                    rule="load_chunks",
                    severity="critical",
                    issue_code="CHUNKS_READ_ERROR",
                    message=f"Failed to read chunks.jsonl: {exc}",
                    context={"path": str(chunks_path)},
                )
            )
        return chunks

    @staticmethod
    def _load_objectives(
        archive_root: Path, result: ValidationResult
    ) -> Dict[str, Any]:
        """Load ``objectives.json`` (Worker A) with ``course.json`` fallback.

        Returns a dict shaped like
        ``{"terminal_outcomes": [...], "component_outcomes": [...]}``.
        """
        objectives_path = archive_root / "objectives.json"
        if objectives_path.exists():
            try:
                data = json.loads(objectives_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                result.issues.append(
                    ValidationIssue(
                        rule="load_objectives",
                        severity="critical",
                        issue_code="OBJECTIVES_PARSE_ERROR",
                        message=f"Failed to parse objectives.json: {exc}",
                        context={"path": str(objectives_path)},
                    )
                )
                return {"terminal_outcomes": [], "component_outcomes": []}
            return {
                "terminal_outcomes": data.get("terminal_outcomes", []) or [],
                "component_outcomes": data.get("component_objectives", [])
                or data.get("component_outcomes", [])
                or [],
                "_source": "objectives.json",
            }

        # Fallback: course.json learning_outcomes
        course_json_path = archive_root / "course.json"
        if not course_json_path.exists():
            result.issues.append(
                ValidationIssue(
                    rule="load_objectives",
                    severity="critical",
                    issue_code="MISSING_OBJECTIVES_AND_COURSE_JSON",
                    message=(
                        "Neither objectives.json nor course.json exists in "
                        f"{archive_root}; cannot resolve LO refs."
                    ),
                    context={"archive_root": str(archive_root)},
                )
            )
            return {"terminal_outcomes": [], "component_outcomes": []}
        try:
            course = json.loads(course_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            result.issues.append(
                ValidationIssue(
                    rule="load_objectives",
                    severity="critical",
                    issue_code="COURSE_JSON_PARSE_ERROR",
                    message=f"Failed to parse course.json: {exc}",
                    context={"path": str(course_json_path)},
                )
            )
            return {"terminal_outcomes": [], "component_outcomes": []}

        terminals: List[Dict[str, Any]] = []
        components: List[Dict[str, Any]] = []
        for lo in course.get("learning_outcomes", []) or []:
            level = (lo.get("hierarchy_level") or "").lower()
            if level == "terminal":
                terminals.append(lo)
            elif level == "chapter" or level == "component":
                components.append(lo)
            else:
                # Unknown level — bucket by ID prefix.
                lo_id = (lo.get("id") or "").lower()
                if lo_id.startswith("to-"):
                    terminals.append(lo)
                elif lo_id.startswith("co-"):
                    components.append(lo)
        return {
            "terminal_outcomes": terminals,
            "component_outcomes": components,
            "_source": "course.json (fallback)",
        }

    @staticmethod
    def _load_json_dict(
        path: Path, result: ValidationResult, label: str
    ) -> Dict[str, Any]:
        """Best-effort JSON dict loader; missing/bad → empty dict + issue."""
        if not path.exists():
            # Missing graph files are not always critical — record an
            # internal info-style note via the summary, not the issue
            # list. Rules will simply skip.
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            result.issues.append(
                ValidationIssue(
                    rule=f"load_{label}",
                    severity="critical",
                    issue_code="GRAPH_PARSE_ERROR",
                    message=f"Failed to parse {label} at {path}: {exc}",
                    context={"path": str(path)},
                )
            )
            return {}

    # ------------------------------------------------------------------ #
    # Rules
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rule_unique_chunk_ids(
        rule: str, result: ValidationResult, chunks: List[Dict[str, Any]]
    ) -> None:
        """Every ``chunks.jsonl`` chunk.id is unique."""
        seen: Dict[str, int] = {}
        duplicates: List[str] = []
        for ch in chunks:
            cid = ch.get("id")
            if not cid:
                continue
            if cid in seen:
                duplicates.append(cid)
            seen[cid] = seen.get(cid, 0) + 1
        for dup in sorted(set(duplicates)):
            result.issues.append(
                ValidationIssue(
                    rule=rule,
                    severity=RULE_SEVERITY[rule],
                    issue_code="DUPLICATE_CHUNK_ID",
                    message=f"Chunk id appears more than once: {dup}",
                    context={"chunk_id": dup, "count": seen[dup]},
                )
            )

    @staticmethod
    def _rule_refs_resolve(
        rule: str,
        result: ValidationResult,
        chunks: List[Dict[str, Any]],
        objectives: Dict[str, Any],
    ) -> None:
        """Every chunk.learning_outcome_refs resolves against objectives."""
        valid_ids = _build_objective_id_set(objectives)
        for ch in chunks:
            refs = ch.get("learning_outcome_refs") or []
            for ref in refs:
                if not isinstance(ref, str):
                    continue
                if "," in ref:
                    # Comma refs are caught by no_comma_refs; skip here so
                    # the same chunk doesn't fire twice for one ref.
                    continue
                norm = ref.strip().lower()
                if not norm:
                    continue
                if norm not in valid_ids:
                    result.issues.append(
                        ValidationIssue(
                            rule=rule,
                            severity=RULE_SEVERITY[rule],
                            issue_code="UNRESOLVED_OBJECTIVE_REF",
                            message=(
                                f"Chunk {ch.get('id')} references unknown "
                                f"objective: {ref!r}"
                            ),
                            context={
                                "chunk_id": ch.get("id"),
                                "ref": ref,
                            },
                        )
                    )

    @staticmethod
    def _rule_co_has_parent(
        rule: str, result: ValidationResult, objectives: Dict[str, Any]
    ) -> None:
        """Every component objective has a non-empty parent_terminal that exists."""
        terminal_ids = {
            (t.get("id") or "").lower()
            for t in objectives.get("terminal_outcomes", []) or []
            if isinstance(t, dict)
        }
        for co in objectives.get("component_outcomes", []) or []:
            if not isinstance(co, dict):
                continue
            parent = (co.get("parent_terminal") or "").strip()
            cid = co.get("id") or "<unknown>"
            if not parent:
                result.issues.append(
                    ValidationIssue(
                        rule=rule,
                        severity=RULE_SEVERITY[rule],
                        issue_code="ORPHAN_COMPONENT_OBJECTIVE",
                        message=(
                            f"Component objective {cid!r} has no "
                            "parent_terminal."
                        ),
                        context={"co_id": cid, "parent_terminal": None},
                    )
                )
                continue
            if parent.lower() not in terminal_ids:
                result.issues.append(
                    ValidationIssue(
                        rule=rule,
                        severity=RULE_SEVERITY[rule],
                        issue_code="ORPHAN_COMPONENT_OBJECTIVE",
                        message=(
                            f"Component objective {cid!r} parent_terminal "
                            f"{parent!r} not in terminal_outcomes."
                        ),
                        context={"co_id": cid, "parent_terminal": parent},
                    )
                )

    @staticmethod
    def _rule_no_comma_refs(
        rule: str, result: ValidationResult, chunks: List[Dict[str, Any]]
    ) -> None:
        """No chunk has a learning_outcome_ref entry containing a comma."""
        for ch in chunks:
            refs = ch.get("learning_outcome_refs") or []
            for ref in refs:
                if isinstance(ref, str) and "," in ref:
                    result.issues.append(
                        ValidationIssue(
                            rule=rule,
                            severity=RULE_SEVERITY[rule],
                            issue_code="MALFORMED_COMMA_REF",
                            message=(
                                f"Chunk {ch.get('id')} has comma-delimited "
                                f"learning_outcome_ref: {ref!r}"
                            ),
                            context={
                                "chunk_id": ch.get("id"),
                                "ref": ref,
                            },
                        )
                    )

    @staticmethod
    def _rule_graph_edges_resolve(
        rule: str,
        result: ValidationResult,
        concept_graph: Dict[str, Any],
        pedagogy_graph: Dict[str, Any],
    ) -> None:
        """Every edge.source/target resolves to a node id in its own graph."""
        for graph_name, graph in (
            ("concept_graph", concept_graph),
            ("pedagogy_graph", pedagogy_graph),
        ):
            if not graph:
                continue
            node_ids: Set[str] = {
                n.get("id")
                for n in graph.get("nodes", []) or []
                if isinstance(n, dict) and n.get("id")
            }
            for edge in graph.get("edges", []) or []:
                if not isinstance(edge, dict):
                    continue
                src = edge.get("source")
                tgt = edge.get("target")
                if src is not None and src not in node_ids:
                    result.issues.append(
                        ValidationIssue(
                            rule=rule,
                            severity=RULE_SEVERITY[rule],
                            issue_code="DANGLING_EDGE",
                            message=(
                                f"{graph_name} edge has unresolved source: "
                                f"{src!r} -> {tgt!r}"
                            ),
                            context={
                                "graph": graph_name,
                                "side": "source",
                                "source": src,
                                "target": tgt,
                                "type": edge.get("type")
                                or edge.get("relation_type"),
                            },
                        )
                    )
                if tgt is not None and tgt not in node_ids:
                    result.issues.append(
                        ValidationIssue(
                            rule=rule,
                            severity=RULE_SEVERITY[rule],
                            issue_code="DANGLING_EDGE",
                            message=(
                                f"{graph_name} edge has unresolved target: "
                                f"{src!r} -> {tgt!r}"
                            ),
                            context={
                                "graph": graph_name,
                                "side": "target",
                                "source": src,
                                "target": tgt,
                                "type": edge.get("type")
                                or edge.get("relation_type"),
                            },
                        )
                    )

    @staticmethod
    def _rule_assessment_has_objective(
        rule: str, result: ValidationResult, chunks: List[Dict[str, Any]]
    ) -> None:
        """Every assessment_item chunk has ≥1 learning_outcome_refs entry."""
        for ch in chunks:
            if ch.get("chunk_type") != ASSESSMENT_CHUNK_TYPE:
                continue
            refs = ch.get("learning_outcome_refs") or []
            non_empty = [
                r for r in refs if isinstance(r, str) and r.strip()
            ]
            if not non_empty:
                result.issues.append(
                    ValidationIssue(
                        rule=rule,
                        severity=RULE_SEVERITY[rule],
                        issue_code="UNANCHORED_ASSESSMENT",
                        message=(
                            f"Assessment chunk {ch.get('id')!r} has no "
                            "learning_outcome_refs."
                        ),
                        context={"chunk_id": ch.get("id")},
                    )
                )

    @staticmethod
    def _rule_to_has_teaching_and_assessment(
        rule: str,
        result: ValidationResult,
        chunks: List[Dict[str, Any]],
        objectives: Dict[str, Any],
    ) -> None:
        """Every TO has ≥1 teaching chunk AND ≥1 assessment chunk.

        An assessment for a TO can come either directly (a chunk
        referencing the TO) or indirectly (a chunk referencing a CO
        whose ``parent_terminal`` is that TO).
        """
        terminals = objectives.get("terminal_outcomes", []) or []
        components = objectives.get("component_outcomes", []) or []

        # Map co_id -> parent_terminal_id, both lowercase.
        co_to_to: Dict[str, str] = {}
        for co in components:
            if not isinstance(co, dict):
                continue
            cid = (co.get("id") or "").lower()
            parent = (co.get("parent_terminal") or "").lower()
            if cid and parent:
                co_to_to[cid] = parent

        teaching_for_to: Dict[str, int] = {}
        assessment_for_to: Dict[str, int] = {}

        for ch in chunks:
            ctype = ch.get("chunk_type")
            refs = ch.get("learning_outcome_refs") or []
            if not refs:
                continue
            # Expand each ref to the TO that "owns" it.
            owning_tos: Set[str] = set()
            for ref in refs:
                if not isinstance(ref, str) or "," in ref:
                    continue
                norm = ref.strip().lower()
                if norm.startswith("to-"):
                    owning_tos.add(norm)
                elif norm.startswith("co-") and norm in co_to_to:
                    owning_tos.add(co_to_to[norm])
            if ctype in TEACHING_CHUNK_TYPES:
                for tid in owning_tos:
                    teaching_for_to[tid] = teaching_for_to.get(tid, 0) + 1
            if ctype == ASSESSMENT_CHUNK_TYPE:
                for tid in owning_tos:
                    assessment_for_to[tid] = assessment_for_to.get(tid, 0) + 1

        for to in terminals:
            if not isinstance(to, dict):
                continue
            tid = (to.get("id") or "").lower()
            if not tid:
                continue
            missing: List[str] = []
            if teaching_for_to.get(tid, 0) == 0:
                missing.append("teaching")
            if assessment_for_to.get(tid, 0) == 0:
                missing.append("assessment")
            if missing:
                result.issues.append(
                    ValidationIssue(
                        rule=rule,
                        severity=RULE_SEVERITY[rule],
                        issue_code="UNCOVERED_TERMINAL_OUTCOME",
                        message=(
                            f"Terminal outcome {tid!r} missing "
                            f"{' and '.join(missing)} chunk(s)."
                        ),
                        context={
                            "to_id": tid,
                            "missing": missing,
                            "teaching_count": teaching_for_to.get(tid, 0),
                            "assessment_count": assessment_for_to.get(tid, 0),
                        },
                    )
                )

    @staticmethod
    def _rule_domain_concept_has_chunk(
        rule: str,
        result: ValidationResult,
        concept_graph: Dict[str, Any],
        chunks: List[Dict[str, Any]],
    ) -> None:
        """Every DomainConcept node appears in concept_tags or text of a chunk."""
        if not concept_graph:
            return
        domain_nodes = [
            n
            for n in concept_graph.get("nodes", []) or []
            if isinstance(n, dict) and n.get("class") == "DomainConcept"
        ]
        if not domain_nodes:
            return

        # Build lookup tables once.
        all_concept_tags: Set[str] = set()
        text_blob_lower = ""
        text_segments: List[str] = []
        for ch in chunks:
            for tag in ch.get("concept_tags") or []:
                if isinstance(tag, str):
                    all_concept_tags.add(tag.strip().lower())
            text = ch.get("text") or ""
            if isinstance(text, str):
                text_segments.append(text.lower())
        text_blob_lower = "\n".join(text_segments)

        for n in domain_nodes:
            nid = n.get("id")
            if not nid:
                continue
            nid_lower = str(nid).strip().lower()
            label = n.get("label")
            label_lower = (
                str(label).strip().lower() if isinstance(label, str) else ""
            )
            # Match: concept_tags equality OR substring in any chunk text
            # (tag form is slug-like; text match is the looser fallback).
            tag_hit = nid_lower in all_concept_tags or (
                label_lower and label_lower in all_concept_tags
            )
            text_hit = False
            if not tag_hit:
                # Match on slug normalized to spaces (e.g. "rdf-graph" → "rdf graph"),
                # plus the slug form itself, plus the human label.
                candidates = {nid_lower, nid_lower.replace("-", " ")}
                if label_lower:
                    candidates.add(label_lower)
                for cand in candidates:
                    if cand and cand in text_blob_lower:
                        text_hit = True
                        break
            if not (tag_hit or text_hit):
                result.issues.append(
                    ValidationIssue(
                        rule=rule,
                        severity=RULE_SEVERITY[rule],
                        issue_code="ORPHAN_DOMAIN_CONCEPT",
                        message=(
                            f"DomainConcept node {nid!r} has no chunk "
                            "where it appears in concept_tags or text."
                        ),
                        context={"node_id": nid, "label": label},
                    )
                )

    @staticmethod
    def _rule_scaffolding_not_assessed(
        rule: str,
        result: ValidationResult,
        concept_graph: Dict[str, Any],
        concept_graph_semantic: Dict[str, Any],
    ) -> None:
        """Scaffolding nodes shouldn't be edge targets of pedagogical edges."""
        if not concept_graph or not concept_graph_semantic:
            return
        # Build class lookup from concept_graph.
        node_class: Dict[str, str] = {}
        for n in concept_graph.get("nodes", []) or []:
            if isinstance(n, dict) and n.get("id"):
                node_class[n["id"]] = n.get("class") or ""
        for edge in concept_graph_semantic.get("edges", []) or []:
            if not isinstance(edge, dict):
                continue
            etype = edge.get("type") or edge.get("relation_type")
            if etype not in PEDAGOGICAL_EDGE_TYPES:
                continue
            tgt = edge.get("target")
            if not tgt:
                continue
            klass = node_class.get(tgt)
            if klass in SCAFFOLDING_CLASSES:
                result.issues.append(
                    ValidationIssue(
                        rule=rule,
                        severity=RULE_SEVERITY[rule],
                        issue_code="SCAFFOLDING_AS_ASSESSED",
                        message=(
                            f"{etype} edge targets scaffolding node "
                            f"{tgt!r} (class={klass})."
                        ),
                        context={
                            "edge_type": etype,
                            "source": edge.get("source"),
                            "target": tgt,
                            "target_class": klass,
                        },
                    )
                )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def _build_summary(
        self,
        result: ValidationResult,
        *,
        chunks: List[Dict[str, Any]],
        objectives: Dict[str, Any],
        concept_graph: Dict[str, Any],
        concept_graph_semantic: Dict[str, Any],
        pedagogy_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the ``summary`` dict carried on the result."""
        from collections import Counter

        issues_by_code: Dict[str, int] = {}
        issues_by_rule: Dict[str, int] = {}
        for i in result.issues:
            issues_by_code[i.issue_code] = issues_by_code.get(i.issue_code, 0) + 1
            issues_by_rule[i.rule] = issues_by_rule.get(i.rule, 0) + 1

        return {
            "validator_version": self.version,
            "chunk_count": len(chunks),
            "chunk_types": dict(
                Counter(c.get("chunk_type", "?") for c in chunks)
            ),
            "terminal_outcome_count": len(
                objectives.get("terminal_outcomes", []) or []
            ),
            "component_outcome_count": len(
                objectives.get("component_outcomes", []) or []
            ),
            "objectives_source": objectives.get("_source", "unknown"),
            "concept_graph_node_count": len(
                (concept_graph or {}).get("nodes", []) or []
            ),
            "concept_graph_edge_count": len(
                (concept_graph or {}).get("edges", []) or []
            ),
            "concept_graph_semantic_edge_count": len(
                (concept_graph_semantic or {}).get("edges", []) or []
            ),
            "pedagogy_graph_node_count": len(
                (pedagogy_graph or {}).get("nodes", []) or []
            ),
            "pedagogy_graph_edge_count": len(
                (pedagogy_graph or {}).get("edges", []) or []
            ),
            "issues_by_code": issues_by_code,
            "issues_by_rule": issues_by_rule,
            "critical_count": result.critical_count,
            "warning_count": result.warning_count,
        }


# ---------------------------------------------------------------------- #
# Module helpers
# ---------------------------------------------------------------------- #


def _build_objective_id_set(objectives: Dict[str, Any]) -> Set[str]:
    """Return the lowercase set of valid LO IDs from ``objectives``."""
    valid: Set[str] = set()
    for bucket in ("terminal_outcomes", "component_outcomes"):
        for entry in objectives.get(bucket, []) or []:
            if not isinstance(entry, dict):
                continue
            lo_id = entry.get("id")
            if isinstance(lo_id, str) and lo_id.strip():
                valid.add(lo_id.strip().lower())
    return valid

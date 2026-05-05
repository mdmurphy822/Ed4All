"""LibV2 Packet Integrity Validator (Wave 75 Worker D + Wave 78).

Runs SHACL-style integrity rules over a self-contained LibV2 archive
(`LibV2/courses/<slug>/`) and returns a typed result object describing
which rules passed and which issues fired.

Wave 78 promotes the validator from a *post-hoc operator tool* to a
real workflow gate at ``libv2_archival``. The validator now exposes a
dual interface:

* ``validate(archive_root: Path) -> ValidationResult`` — the
  CLI-facing call shape (Wave 75). Returns the full structured result
  with rule-by-rule passes/failures plus a summary.
* ``validate(inputs: Dict[str, Any]) -> GateResult`` — the
  ``ValidationGateManager`` call shape. ``inputs`` carries
  ``manifest_path`` / ``course_dir`` (built by the gate input router)
  plus optional strict-mode toggles (``strict``, ``strict_coverage``,
  ``strict_typing``).

Rules
-----

Pre-Wave-78 rules (carried over):

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
* ``to_has_teaching_and_assessment`` (warning by default; **critical
  under --strict-coverage**) — every terminal outcome has at least
  one teaching chunk and one assessment chunk (directly or via one
  of its component objectives).
* ``domain_concept_has_chunk`` (warning by default; **critical under
  --strict-coverage**) — every ``concept_graph`` node with
  ``class=DomainConcept`` appears in at least one chunk's
  ``concept_tags`` or text.
* ``scaffolding_not_assessed`` (warning) — pedagogical-scaffolding
  concept_graph nodes never appear as targets of
  ``derived-from-objective`` or ``assesses`` edges in
  ``concept_graph_semantic.json``.

Wave 78 rules (new):

* ``every_objective_has_teaching`` (warning by default; **critical
  under --strict-coverage**) — every TO + CO has either a chunk
  whose ``learning_outcome_refs`` lists it, or a ``teaches`` edge
  in ``pedagogy_graph`` from a Chunk to that objective.
* ``every_objective_has_assessment`` (warning by default; **critical
  under --strict-coverage**) — every TO + CO has either an
  ``assessment_item`` chunk that references it, or an ``assesses``
  edge pointing to it. Terminal outcomes also count their child COs'
  assessment coverage as a rollup.
* ``edge_endpoint_typing`` (warning by default; **critical under
  --strict-typing**) — every edge in ``pedagogy_graph`` and
  ``concept_graph_semantic`` has source + target node classes that
  match the typed-endpoint contract per relation type.

Strictness modes
----------------

Default behavior (no flags) preserves Wave 75 semantics: coverage and
typing rules emit warnings only. ``--strict-coverage`` promotes the
four coverage rules to critical; ``--strict-typing`` promotes
``edge_endpoint_typing`` to critical; ``--strict`` implies both.

The ``LIBV2_RELAX_PACKET_INTEGRITY=true`` env var is a one-way
escape hatch for legacy archives — when set, all gated rules
downgrade to warnings regardless of the caller's strict flags.
"""

from __future__ import annotations

import json
import logging
import os
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
    # Wave 78 — coverage rules (default warning, critical under
    # --strict-coverage).
    "every_objective_has_teaching": "warning",
    "every_objective_has_assessment": "warning",
    # Wave 78 — typing rule (default warning, critical under
    # --strict-typing).
    "edge_endpoint_typing": "warning",
}

# Wave 78 — rules promoted to critical when the caller passes
# ``strict=True`` or the more granular flags. The CLI surfaces
# ``--strict-coverage`` and ``--strict-typing``; ``--strict`` implies
# both.
COVERAGE_RULES: Set[str] = {
    "to_has_teaching_and_assessment",
    "every_objective_has_teaching",
    "every_objective_has_assessment",
    "domain_concept_has_chunk",
}
TYPING_RULES: Set[str] = {
    "edge_endpoint_typing",
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


# Wave 78 — typed-endpoint contract for ``edge_endpoint_typing``.
# Maps edge.relation_type to (allowed_source_classes,
# allowed_target_classes). Outcome covers terminal outcomes;
# ComponentObjective covers component objectives. DomainConcept covers
# both pedagogy_graph "Concept" nodes (synonym) and concept_graph
# "DomainConcept" nodes — see ``EDGE_CLASS_SYNONYMS`` for the
# normalisation map.
EDGE_TYPING_CONTRACT: Dict[str, Tuple[Set[str], Set[str]]] = {
    "teaches": ({"Chunk"}, {"Outcome", "ComponentObjective"}),
    "assesses": ({"Chunk"}, {"Outcome", "ComponentObjective"}),
    "practices": ({"Chunk"}, {"Outcome", "ComponentObjective"}),
    "exemplifies": ({"Chunk"}, {"DomainConcept"}),
    "supports_outcome": ({"ComponentObjective"}, {"Outcome"}),
    "interferes_with": ({"Misconception"}, {"DomainConcept"}),
    "prerequisite_of": ({"DomainConcept"}, {"DomainConcept"}),
    "belongs_to_module": ({"Chunk"}, {"Module"}),
    "at_bloom_level": (
        {"Outcome", "ComponentObjective"},
        {"BloomLevel"},
    ),
    "follows": ({"Module"}, {"Module"}),
}

# Class synonyms — pedagogy_graph emits ``Concept`` for what the
# concept_graph emits as ``DomainConcept``; ``TerminalOutcome`` and
# ``Outcome`` are synonyms in different worker emits. Map every
# observed name to a canonical class so the typing contract is
# strict but tolerant of legitimate naming drift.
EDGE_CLASS_SYNONYMS: Dict[str, str] = {
    "Concept": "DomainConcept",
    "DomainConcept": "DomainConcept",
    "TerminalOutcome": "Outcome",
    "Outcome": "Outcome",
    "LearningObjective": "Outcome",
    "ComponentObjective": "ComponentObjective",
    "Chunk": "Chunk",
    "Module": "Module",
    "BloomLevel": "BloomLevel",
    "Misconception": "Misconception",
}


# Wave 78 — env var that downgrades all gated rules to warnings,
# regardless of strict-mode flags. Legacy archives only.
RELAX_ENV_VAR = "LIBV2_RELAX_PACKET_INTEGRITY"


def _env_relax() -> bool:
    """Return True iff ``LIBV2_RELAX_PACKET_INTEGRITY=true`` is set."""
    return (os.getenv(RELAX_ENV_VAR, "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# ---------------------------------------------------------------------- #
# Validator
# ---------------------------------------------------------------------- #


def _emit_packet_integrity_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    metrics: Dict[str, Any],
) -> None:
    """Emit one ``libv2_packet_integrity_check`` decision per gate-shape ``validate()`` (H3 W6b)."""
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    metric_strs = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
    rationale = (
        f"LibV2 packet integrity gate verdict={decision}, "
        f"failure_code={code or 'none'}, metrics=({metric_strs})."
    )
    enriched = dict(metrics)
    enriched["passed"] = bool(passed)
    enriched["failure_code"] = code
    try:
        capture.log_decision(
            decision_type="libv2_packet_integrity_check",
            decision=decision,
            rationale=rationale,
            context=str(enriched),
            metrics=enriched,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on libv2_packet_integrity_check: %s",
            exc,
        )


class PacketIntegrityValidator:
    """Validates a LibV2 archive's internal consistency.

    Usage (CLI shape, Wave 75)::

        validator = PacketIntegrityValidator()
        result = validator.validate(Path("LibV2/courses/<slug>"))

    Usage (gate shape, Wave 78)::

        validator = PacketIntegrityValidator()
        gate_result = validator.validate({
            "manifest_path": "LibV2/courses/<slug>/manifest.json",
            "course_dir": "LibV2/courses/<slug>",
            "strict": True,  # or strict_coverage / strict_typing
        })

    Strict-mode flags can also be passed to the constructor — useful
    when the caller is the CLI and wants the validator to compute
    severity once at instantiation time. Note that
    ``LIBV2_RELAX_PACKET_INTEGRITY=true`` overrides both: when set,
    every gated rule downgrades to warning regardless of caller
    intent.
    """

    name = "libv2_packet_integrity"
    version = "2.0.0"

    def __init__(
        self,
        *,
        strict_coverage: bool = False,
        strict_typing: bool = False,
    ) -> None:
        self.strict_coverage = bool(strict_coverage)
        self.strict_typing = bool(strict_typing)

    # ------------------------------------------------------------------ #
    # Dual-interface dispatcher
    # ------------------------------------------------------------------ #

    def validate(
        self,
        archive_or_inputs: Any,
    ) -> Any:  # noqa: ANN401 (dual-shape return)
        """Dual entry point — Path → ValidationResult; dict → GateResult.

        Path / str → CLI shape (returns ``ValidationResult``).
        Dict[str, Any] → gate-framework shape (returns ``GateResult``).
        Anything else raises ``TypeError`` so misuse is loud.
        """
        if isinstance(archive_or_inputs, dict):
            return self._validate_gate(archive_or_inputs)
        if isinstance(archive_or_inputs, (str, Path)):
            return self._validate_archive(Path(archive_or_inputs))
        raise TypeError(
            "PacketIntegrityValidator.validate accepts a Path/str (CLI) "
            "or a dict (gate framework); got "
            f"{type(archive_or_inputs).__name__}."
        )

    # ------------------------------------------------------------------ #
    # Strictness resolution
    # ------------------------------------------------------------------ #

    def _resolve_severity(self, rule_name: str) -> str:
        """Resolve a rule's effective severity given strict flags + env.

        Precedence (highest first):

        1. ``LIBV2_RELAX_PACKET_INTEGRITY=true`` — every coverage /
           typing rule downgrades to warning.
        2. ``--strict-coverage`` (instance flag) — promotes coverage
           rules to critical.
        3. ``--strict-typing`` (instance flag) — promotes typing
           rules to critical.
        4. Default — table value (which is ``warning`` for coverage
           + typing rules).
        """
        base = RULE_SEVERITY.get(rule_name, "warning")

        if _env_relax():
            if rule_name in COVERAGE_RULES or rule_name in TYPING_RULES:
                return "warning"

        if self.strict_coverage and rule_name in COVERAGE_RULES:
            return "critical"
        if self.strict_typing and rule_name in TYPING_RULES:
            return "critical"
        return base

    # ------------------------------------------------------------------ #
    # CLI shape — Path → ValidationResult
    # ------------------------------------------------------------------ #

    def _validate_archive(self, archive_root: Path) -> ValidationResult:
        """Run every rule and return a ``ValidationResult``."""
        return self._run_rules(archive_root)

    # ------------------------------------------------------------------ #
    # Gate shape — Dict → GateResult
    # ------------------------------------------------------------------ #

    def _validate_gate(self, inputs: Dict[str, Any]):
        """Adapt the gate framework's input/result contract.

        Reads ``manifest_path`` / ``course_dir`` from inputs (the
        ``GateInputRouter`` builder fills these in for the
        ``libv2_archival`` phase). Strict-mode flags can be passed in
        ``inputs`` (``strict``, ``strict_coverage``, ``strict_typing``)
        or via the gate's ``config`` block — the executor merges
        ``GateConfig.config`` into inputs before calling.
        """
        # Lazy import to keep the validator usable when MCP isn't on the
        # path (e.g. CLI-only environments).
        from MCP.hardening.validation_gates import GateIssue, GateResult

        gate_id = str(inputs.get("gate_id", "libv2_packet_integrity") or "libv2_packet_integrity")
        capture = inputs.get("decision_capture")

        # Strict flags: either gate-config (merged into inputs by the
        # executor) or per-call inputs. ``strict`` is sugar for both.
        strict_all = bool(inputs.get("strict", False))
        self.strict_coverage = bool(
            inputs.get("strict_coverage", False) or strict_all
            or self.strict_coverage
        )
        self.strict_typing = bool(
            inputs.get("strict_typing", False) or strict_all
            or self.strict_typing
        )

        # Resolve archive root: course_dir wins; manifest_path's parent
        # is the fallback.
        course_dir_raw = inputs.get("course_dir")
        manifest_path_raw = inputs.get("manifest_path")
        archive_root: Optional[Path] = None
        if course_dir_raw:
            archive_root = Path(course_dir_raw)
        elif manifest_path_raw:
            archive_root = Path(manifest_path_raw).parent

        if archive_root is None:
            _emit_packet_integrity_decision(
                capture,
                passed=False,
                code="MISSING_ARCHIVE_INPUTS",
                metrics={
                    "archive_root_str": None,
                    "issue_count": 1,
                    "rules_run": 0,
                    "strict_coverage": bool(self.strict_coverage),
                    "strict_typing": bool(self.strict_typing),
                },
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="MISSING_ARCHIVE_INPUTS",
                    message=(
                        "PacketIntegrityValidator requires either "
                        "'course_dir' or 'manifest_path' in gate inputs."
                    ),
                )],
            )

        result = self._run_rules(archive_root)

        # Map ValidationIssue → GateIssue.
        gate_issues: List[GateIssue] = []
        for issue in result.issues:
            gate_issues.append(GateIssue(
                severity=issue.severity,
                code=issue.issue_code,
                message=issue.message,
                location=str(archive_root),
                suggestion=None,
            ))

        critical_count = sum(1 for i in gate_issues if i.severity == "critical")
        warning_count = sum(1 for i in gate_issues if i.severity == "warning")
        passed = critical_count == 0
        score = max(0.0, 1.0 - len(gate_issues) * 0.05) if gate_issues else 1.0

        first_critical_code = next(
            (i.code for i in gate_issues if i.severity == "critical"), None
        )
        _emit_packet_integrity_decision(
            capture,
            passed=passed,
            code=first_critical_code,
            metrics={
                "archive_root_str": str(archive_root),
                "score": float(round(score, 4)),
                "critical_count": int(critical_count),
                "warning_count": int(warning_count),
                "issue_count": len(gate_issues),
                "rules_run": int(getattr(result, "rules_run", 0) or 0),
                "rules_failed": int(getattr(result, "rules_failed", 0) or 0),
                "strict_coverage": bool(self.strict_coverage),
                "strict_typing": bool(self.strict_typing),
            },
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=gate_issues,
        )

    # ------------------------------------------------------------------ #
    # Shared rule runner
    # ------------------------------------------------------------------ #

    def _run_rules(self, archive_root: Path) -> ValidationResult:
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
            # Wave 78 — coverage rules.
            (
                "every_objective_has_teaching",
                self._rule_every_objective_has_teaching,
                (chunks, objectives, pedagogy_graph),
            ),
            (
                "every_objective_has_assessment",
                self._rule_every_objective_has_assessment,
                (chunks, objectives, pedagogy_graph),
            ),
            # Wave 78 — typing rule.
            (
                "edge_endpoint_typing",
                self._rule_edge_endpoint_typing,
                (concept_graph, pedagogy_graph, concept_graph_semantic),
            ),
        ]

        for rule_name, runner, args in rule_runners:
            result.rules_run += 1
            issues_before = len(result.issues)
            severity = self._resolve_severity(rule_name)
            try:
                runner(rule_name, severity, result, *args)
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
        """Load ``imscc_chunks/chunks.jsonl`` (one JSON object per line).

        Phase 7c: prefers ``imscc_chunks/`` and falls back to legacy
        ``corpus/`` via the shim for unprovisioned archives.
        """
        from lib.libv2_storage import resolve_imscc_chunks_path

        chunks_path = resolve_imscc_chunks_path(archive_root, "chunks.jsonl")
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
        rule: str,
        severity: str,
        result: ValidationResult,
        chunks: List[Dict[str, Any]],
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
                    severity=severity,
                    issue_code="DUPLICATE_CHUNK_ID",
                    message=f"Chunk id appears more than once: {dup}",
                    context={"chunk_id": dup, "count": seen[dup]},
                )
            )

    @staticmethod
    def _rule_refs_resolve(
        rule: str,
        severity: str,
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
                            severity=severity,
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
        rule: str,
        severity: str,
        result: ValidationResult,
        objectives: Dict[str, Any],
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
                        severity=severity,
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
                        severity=severity,
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
        rule: str,
        severity: str,
        result: ValidationResult,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """No chunk has a learning_outcome_ref entry containing a comma."""
        for ch in chunks:
            refs = ch.get("learning_outcome_refs") or []
            for ref in refs:
                if isinstance(ref, str) and "," in ref:
                    result.issues.append(
                        ValidationIssue(
                            rule=rule,
                            severity=severity,
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
        severity: str,
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
                            severity=severity,
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
                            severity=severity,
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
        rule: str,
        severity: str,
        result: ValidationResult,
        chunks: List[Dict[str, Any]],
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
                        severity=severity,
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
        severity: str,
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
                        severity=severity,
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
        severity: str,
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
                        severity=severity,
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
        severity: str,
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
                        severity=severity,
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
    # Wave 78 — coverage rules
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rule_every_objective_has_teaching(
        rule: str,
        severity: str,
        result: ValidationResult,
        chunks: List[Dict[str, Any]],
        objectives: Dict[str, Any],
        pedagogy_graph: Dict[str, Any],
    ) -> None:
        """Every TO + CO has ≥1 teaching chunk OR ≥1 ``teaches`` edge.

        Wave 78 strict-coverage rule. Pre-Wave-78 the only coverage
        signal was ``to_has_teaching_and_assessment`` which (a)
        only inspected terminal outcomes and (b) was warning-only.
        This rule extends coverage to component objectives and
        accepts ``pedagogy_graph`` ``teaches`` edges as an
        alternative source — useful when chunk LO refs lag behind
        Worker B's pedagogy emit.
        """
        # Collect every objective id (TO + CO) at lowercase.
        objective_ids: Set[str] = set()
        for bucket in ("terminal_outcomes", "component_outcomes"):
            for entry in objectives.get(bucket, []) or []:
                if not isinstance(entry, dict):
                    continue
                lo_id = entry.get("id")
                if isinstance(lo_id, str) and lo_id.strip():
                    objective_ids.add(lo_id.strip().lower())

        if not objective_ids:
            return

        # Path A — chunk learning_outcome_refs.
        ref_coverage: Set[str] = set()
        for ch in chunks:
            for ref in ch.get("learning_outcome_refs") or []:
                if not isinstance(ref, str) or "," in ref:
                    continue
                norm = ref.strip().lower()
                if norm in objective_ids:
                    ref_coverage.add(norm)

        # Path B — pedagogy_graph ``teaches`` edges from a Chunk node.
        edge_coverage: Set[str] = set()
        if pedagogy_graph:
            chunk_node_ids: Set[str] = {
                n.get("id")
                for n in pedagogy_graph.get("nodes", []) or []
                if isinstance(n, dict)
                and (n.get("class") or "") == "Chunk"
                and n.get("id")
            }
            for edge in pedagogy_graph.get("edges", []) or []:
                if not isinstance(edge, dict):
                    continue
                etype = edge.get("type") or edge.get("relation_type")
                if etype != "teaches":
                    continue
                src = edge.get("source")
                tgt = edge.get("target")
                if src in chunk_node_ids and isinstance(tgt, str):
                    norm = tgt.strip().lower()
                    if norm in objective_ids:
                        edge_coverage.add(norm)

        covered = ref_coverage | edge_coverage
        for oid in sorted(objective_ids - covered):
            result.issues.append(
                ValidationIssue(
                    rule=rule,
                    severity=severity,
                    issue_code="OBJECTIVE_NO_TEACHING_CHUNK",
                    message=(
                        f"Objective {oid!r} has no teaching chunk and no "
                        "'teaches' edge from a Chunk in pedagogy_graph."
                    ),
                    context={
                        "objective_id": oid,
                        "ref_coverage": oid in ref_coverage,
                        "edge_coverage": oid in edge_coverage,
                    },
                )
            )

    @staticmethod
    def _rule_every_objective_has_assessment(
        rule: str,
        severity: str,
        result: ValidationResult,
        chunks: List[Dict[str, Any]],
        objectives: Dict[str, Any],
        pedagogy_graph: Dict[str, Any],
    ) -> None:
        """Every TO + CO has ≥1 assessment chunk OR ``assesses`` edge.

        Wave 78 strict-coverage rule. Terminal outcomes additionally
        roll up their child COs' assessment coverage — a TO is
        considered covered if any of its COs has an assessment.
        """
        # Build TO + CO id sets and CO → parent_terminal map.
        to_ids: Set[str] = set()
        co_ids: Set[str] = set()
        co_to_to: Dict[str, str] = {}
        for entry in objectives.get("terminal_outcomes", []) or []:
            if isinstance(entry, dict):
                tid = (entry.get("id") or "").strip().lower()
                if tid:
                    to_ids.add(tid)
        for entry in objectives.get("component_outcomes", []) or []:
            if isinstance(entry, dict):
                cid = (entry.get("id") or "").strip().lower()
                parent = (entry.get("parent_terminal") or "").strip().lower()
                if cid:
                    co_ids.add(cid)
                if cid and parent:
                    co_to_to[cid] = parent

        all_objective_ids = to_ids | co_ids
        if not all_objective_ids:
            return

        # Path A — assessment_item chunks referencing the objective.
        ref_coverage: Set[str] = set()
        for ch in chunks:
            if ch.get("chunk_type") != ASSESSMENT_CHUNK_TYPE:
                continue
            for ref in ch.get("learning_outcome_refs") or []:
                if not isinstance(ref, str) or "," in ref:
                    continue
                norm = ref.strip().lower()
                if norm in all_objective_ids:
                    ref_coverage.add(norm)

        # Path B — pedagogy_graph ``assesses`` edges.
        edge_coverage: Set[str] = set()
        if pedagogy_graph:
            for edge in pedagogy_graph.get("edges", []) or []:
                if not isinstance(edge, dict):
                    continue
                etype = edge.get("type") or edge.get("relation_type")
                if etype != "assesses":
                    continue
                tgt = edge.get("target")
                if isinstance(tgt, str):
                    norm = tgt.strip().lower()
                    if norm in all_objective_ids:
                        edge_coverage.add(norm)

        covered = ref_coverage | edge_coverage

        # TO rollup — a TO is also covered when any of its child COs
        # has assessment coverage.
        rollup_covered: Set[str] = set()
        for cid in co_ids:
            if cid in covered:
                parent = co_to_to.get(cid)
                if parent and parent in to_ids:
                    rollup_covered.add(parent)

        for oid in sorted(all_objective_ids):
            if oid in covered:
                continue
            if oid in to_ids and oid in rollup_covered:
                continue
            result.issues.append(
                ValidationIssue(
                    rule=rule,
                    severity=severity,
                    issue_code="OBJECTIVE_NO_ASSESSMENT",
                    message=(
                        f"Objective {oid!r} has no assessment_item chunk, "
                        "no 'assesses' edge, and (for TOs) no child CO "
                        "with assessment coverage."
                    ),
                    context={
                        "objective_id": oid,
                        "kind": "to" if oid in to_ids else "co",
                        "ref_coverage": oid in ref_coverage,
                        "edge_coverage": oid in edge_coverage,
                    },
                )
            )

    # ------------------------------------------------------------------ #
    # Wave 78 — typing rule
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rule_edge_endpoint_typing(
        rule: str,
        severity: str,
        result: ValidationResult,
        concept_graph: Dict[str, Any],
        pedagogy_graph: Dict[str, Any],
        concept_graph_semantic: Dict[str, Any],
    ) -> None:
        """Validate edge endpoint classes against the typed-endpoint contract.

        Wave 78 strict-typing rule. For every relation type listed in
        ``EDGE_TYPING_CONTRACT``, the source and target node must
        belong to one of the allowed classes. Class names are
        normalized via ``EDGE_CLASS_SYNONYMS`` so legitimate naming
        drift (Concept ↔ DomainConcept, TerminalOutcome ↔ Outcome)
        doesn't trip the rule.

        Edges whose relation_type isn't in the contract are silently
        skipped — they're either custom relations or relations that
        Wave 78 hasn't pinned a typing contract for yet.
        """
        # Build a unified node-class index across all three graphs so
        # cross-graph references (rare but legal) don't mis-fire.
        node_class: Dict[str, str] = {}
        for graph in (concept_graph, pedagogy_graph, concept_graph_semantic):
            if not graph:
                continue
            for n in graph.get("nodes", []) or []:
                if isinstance(n, dict) and n.get("id"):
                    raw_class = n.get("class") or ""
                    canonical = EDGE_CLASS_SYNONYMS.get(raw_class, raw_class)
                    # Don't overwrite an already-canonicalized class
                    # with an empty one from a different graph.
                    if canonical or n["id"] not in node_class:
                        node_class[n["id"]] = canonical

        for graph_name, graph in (
            ("pedagogy_graph", pedagogy_graph),
            ("concept_graph_semantic", concept_graph_semantic),
        ):
            if not graph:
                continue
            for edge in graph.get("edges", []) or []:
                if not isinstance(edge, dict):
                    continue
                etype = edge.get("type") or edge.get("relation_type")
                contract = EDGE_TYPING_CONTRACT.get(etype)
                if contract is None:
                    continue
                allowed_src, allowed_tgt = contract
                src = edge.get("source")
                tgt = edge.get("target")
                src_class = node_class.get(src) if src else None
                tgt_class = node_class.get(tgt) if tgt else None

                src_ok = bool(src_class and src_class in allowed_src)
                tgt_ok = bool(tgt_class and tgt_class in allowed_tgt)

                if src_ok and tgt_ok:
                    continue

                violations: List[str] = []
                if not src_ok:
                    violations.append(
                        f"source class {src_class!r} not in {sorted(allowed_src)}"
                    )
                if not tgt_ok:
                    violations.append(
                        f"target class {tgt_class!r} not in {sorted(allowed_tgt)}"
                    )

                result.issues.append(
                    ValidationIssue(
                        rule=rule,
                        severity=severity,
                        issue_code="EDGE_ENDPOINT_TYPE_MISMATCH",
                        message=(
                            f"{graph_name} '{etype}' edge {src!r} -> {tgt!r} "
                            f"violates typed-endpoint contract: "
                            + "; ".join(violations)
                        ),
                        context={
                            "graph": graph_name,
                            "edge_type": etype,
                            "source": src,
                            "target": tgt,
                            "source_class": src_class,
                            "target_class": tgt_class,
                            "allowed_source_classes": sorted(allowed_src),
                            "allowed_target_classes": sorted(allowed_tgt),
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

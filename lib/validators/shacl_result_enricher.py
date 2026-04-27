"""Phase 6 — Editor-grade SHACL ValidationReport enrichment.

Consumes a pyshacl ``sh:ValidationReport`` graph (produced by Phase 4's
``lib/validators/shacl_runner.py``) and joins each result against:

1. **Shape-source provenance** — built by ``ShapeSourceIndex`` over the
   Turtle shape files. Each ``sh:sourceShape`` IRI looks up to a
   ``(file_path, line_number)`` pair so editors and CI can render
   inline feedback.
2. **Data-source provenance** — caller-supplied
   ``Dict[str, DataProvenance]`` keyed on focus-node IRI. Phase 4's
   Courseforge case knows which HTML file produced each focus node;
   the enricher accepts the mapping as a parameter rather than trying
   to discover it. ``data_block_id`` is reserved for a future
   ``data-cf-block-id``-style provenance triple emitter; the MVP
   passes ``None`` until such an emitter exists.

Output: ``List[EnrichedValidationResult]`` plus a JSON-LD report
projection via ``report_to_jsonld``.

Side effects (sub-plan §2.4 + §2.5):

* When ``GITHUB_ACTIONS=true`` is in the environment, emit one
  ``::error file=...,line=...,col=1::message`` (or the ``::warning`` /
  ``::notice`` equivalents) per result to stderr. Side-effect-only;
  the function still returns the EnrichedResult list either way.
* When a ``DecisionCapture`` instance is provided, emit one
  ``decision_type="shacl_validation"`` event per Violation-severity
  result. Rationale interpolates the source shape, focus node, and
  message — well above the 20-character minimum the project's main
  CLAUDE.md "Decision Capture" §"Required Fields" mandates.

Phase 6 explicitly does NOT modify the Phase 4 runner; an additive
sibling accessor ``run_shacl_with_report_graph`` is exported from
``shacl_runner`` so this module can get at the raw rdflib graph.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from lib.validators.shape_provenance import ShapeSourceIndex, ShapeSourceLocation


__all__ = [
    "EnrichedValidationResult",
    "DataProvenance",
    "enrich_validation_report",
    "report_to_jsonld",
    "GITHUB_ACTIONS_ENV",
    "DECISION_TYPE_SHACL",
]


GITHUB_ACTIONS_ENV = "GITHUB_ACTIONS"
DECISION_TYPE_SHACL = "shacl_validation"

# SHACL namespace + result-level predicates. Mirrors the constants the
# Phase 4 runner uses; not imported from there to keep the enricher's
# import surface independent of the runner.
_SH_NS = "http://www.w3.org/ns/shacl#"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

_SEVERITY_TO_STR = {
    f"{_SH_NS}Violation": "critical",
    f"{_SH_NS}Warning": "warning",
    f"{_SH_NS}Info": "info",
}

# Severity -> GitHub annotation command. Annotation command names are
# defined by the GitHub workflow-command spec (`::error`, `::warning`,
# `::notice`); ``info`` SHACL severities map to ``notice``.
_SEVERITY_TO_GH = {
    "critical": "error",
    "warning": "warning",
    "info": "notice",
}


@dataclass(frozen=True)
class DataProvenance:
    """Where a focus node came from, in source-file terms.

    For Phase 4's Courseforge case, ``file_path`` is the
    ``week_*/*.html`` page that carried the JSON-LD block which pyld
    materialised into the focus-node IRI. ``block_id`` is reserved for
    a future emitter that stamps ``data-cf-block-id`` triples onto the
    data graph; the MVP leaves it ``None``.
    """

    file_path: Optional[Path] = None
    block_id: Optional[str] = None


@dataclass
class EnrichedValidationResult:
    """One row of an enriched validation report (sub-plan §2.3).

    Carries the standard SHACL ValidationResult fields PLUS shape-source
    provenance (``shape_file``, ``shape_line``) joined via
    ``ShapeSourceIndex`` and data-source provenance (``data_file``,
    ``data_block_id``) joined via the caller-supplied mapping.

    All optional fields tolerate ``None`` per the parent task's
    constraint: a ValidationReport that lacks ``sh:value`` /
    ``sh:resultPath`` (NodeShape constraints, ``sh:not``, etc. — Q42)
    must not break the enricher.
    """

    rule_id: Optional[str]
    shape_file: Optional[str]
    shape_line: Optional[int]
    data_file: Optional[str]
    data_block_id: Optional[str]
    focus_node: str
    value: Optional[str]
    path: Optional[str]
    message: str
    severity: str
    source_constraint_component: Optional[str]
    source_shape: Optional[str]

    def to_jsonld(self) -> Dict[str, Any]:
        """Project to the ``ed4all:EnrichedValidationResult`` JSON-LD shape.

        Inlines the @context for self-containedness; consumers that
        already carry an Ed4All context can strip it. Field naming
        matches sub-plan §2.3 verbatim.
        """
        doc: Dict[str, Any] = {
            "@context": {
                "ed4all": "https://ed4all.io/vocab/",
                "prov": "http://www.w3.org/ns/prov#",
                "sh": _SH_NS,
                "ruleId": "ed4all:ruleId",
                "shapeFile": "prov:atLocation",
                "shapeLine": "ed4all:sourceLine",
                "dataFile": "ed4all:dataFile",
                "dataBlockId": "ed4all:dataBlockId",
                "focusNode": "sh:focusNode",
                "value": "sh:value",
                "path": "sh:resultPath",
                "message": "sh:resultMessage",
                "severity": "sh:resultSeverity",
                "sourceConstraintComponent": "sh:sourceConstraintComponent",
                "sourceShape": "sh:sourceShape",
            },
            "@type": "ed4all:EnrichedValidationResult",
            "ruleId": self.rule_id,
            "shapeFile": self.shape_file,
            "shapeLine": self.shape_line,
            "dataFile": self.data_file,
            "dataBlockId": self.data_block_id,
            "focusNode": self.focus_node,
            "value": self.value,
            "path": self.path,
            "message": self.message,
            "severity": self.severity,
            "sourceConstraintComponent": self.source_constraint_component,
            "sourceShape": self.source_shape,
        }
        return doc


def enrich_validation_report(
    results_graph: Any,
    *,
    shape_source_index: ShapeSourceIndex,
    shape_files: Iterable[Path] = (),
    shapes_graph: Optional[Any] = None,
    data_provenance: Optional[Dict[str, DataProvenance]] = None,
    decision_capture: Optional[Any] = None,
) -> List[EnrichedValidationResult]:
    """Walk a pyshacl ValidationReport graph and emit enriched rows.

    Args:
        results_graph: An ``rdflib.Graph`` containing one or more
            ``sh:ValidationResult`` nodes (the standard pyshacl output).
        shape_source_index: Pre-built or empty
            :class:`ShapeSourceIndex`. The function calls
            ``build_for_file`` on each path in ``shape_files`` before
            walking the report, so the typical caller passes the
            shape-file paths their runner used.
        shape_files: Iterable of Turtle shape file paths to scan. Empty
            is fine — the enricher just won't be able to populate
            ``shape_file`` / ``shape_line``.
        shapes_graph: Optional ``rdflib.Graph`` containing the SHACL
            shapes used during validation. When pyshacl reports
            ``sh:sourceShape`` as a blank node (inline
            ``sh:property [...]`` PropertyShape), the enricher walks
            upward via ``sh:property`` to find the owning NodeShape's
            IRI and uses THAT for the line-number lookup. Without the
            shapes graph the enricher can only resolve named-shape
            sources; inline PropertyShape violations stay
            ``shape_file=None, shape_line=None``.
        data_provenance: ``{focus_node_iri: DataProvenance}`` map. Empty
            / ``None`` is fine; results emit ``data_file=None``.
        decision_capture: Optional ``DecisionCapture`` instance. When
            provided, emits one ``decision_type="shacl_validation"``
            event per Violation-severity result.

    Returns:
        A list of :class:`EnrichedValidationResult`, one per
        ``sh:ValidationResult`` node in the graph.

    Side effects:
        * GitHub annotation lines on stderr when
          ``GITHUB_ACTIONS=true``.
        * DecisionCapture log events when ``decision_capture`` is
          provided (Violations only — Warnings / Infos do not trigger
          captures, per sub-plan §2.5 + Q45).

    Tolerance contract:
        * Results that lack ``sh:value`` / ``sh:resultPath`` emit
          ``None`` for those fields rather than raising.
        * Results whose ``sh:sourceShape`` IRI isn't in the cache
          emit ``shape_file=None, shape_line=None``.
        * Blank-node ``sh:sourceShape`` values map to
          ``source_shape=None`` (we serialise blank nodes as their
          ``_:`` strings; the index returns ``None`` for those).
    """
    # rdflib is a hard project dependency; importing here keeps the
    # module-level import surface light for callers that only need the
    # dataclasses.
    from rdflib import Namespace, URIRef
    from rdflib.term import Literal

    SH = Namespace(_SH_NS)
    RDF = Namespace(_RDF_NS)

    # Eagerly build shape-source index for the supplied shape files.
    for path in shape_files:
        shape_source_index.build_for_file(Path(path))

    data_provenance = data_provenance or {}
    enriched: List[EnrichedValidationResult] = []
    github_actions = os.environ.get(GITHUB_ACTIONS_ENV) == "true"

    for result_node in results_graph.subjects(RDF.type, SH.ValidationResult):
        focus = results_graph.value(result_node, SH.focusNode)
        path = results_graph.value(result_node, SH.resultPath)
        sev = results_graph.value(result_node, SH.resultSeverity)
        msg = results_graph.value(result_node, SH.resultMessage)
        src_shape = results_graph.value(result_node, SH.sourceShape)
        src_cc = results_graph.value(result_node, SH.sourceConstraintComponent)
        value = results_graph.value(result_node, SH.value)

        focus_str = str(focus) if focus is not None else ""
        path_str = str(path) if path is not None else None
        msg_str = (
            str(msg)
            if isinstance(msg, Literal)
            else (str(msg) if msg is not None else "")
        )
        sev_str = _SEVERITY_TO_STR.get(str(sev), "critical")
        src_shape_str = (
            str(src_shape) if isinstance(src_shape, URIRef) else None
        )
        src_cc_str = str(src_cc) if isinstance(src_cc, URIRef) else None
        value_str = str(value) if value is not None else None

        # Shape-source provenance.
        # Two cases: (a) sh:sourceShape is a named NodeShape IRI -> direct
        # lookup. (b) sh:sourceShape is a blank node (inline PropertyShape
        # inside ``sh:property [...]``) -> walk the shapes graph upward
        # via ``sh:property`` to find the owning NodeShape, then look up
        # THAT IRI. Phase 4's PageObjectivesMinCountShape is case (b);
        # the canonical fixture exercises this path.
        shape_file_str: Optional[str] = None
        shape_line_int: Optional[int] = None
        lookup_iri: Optional[str] = src_shape_str
        if (
            lookup_iri is None
            and shapes_graph is not None
            and src_shape is not None
        ):
            lookup_iri = _resolve_owning_nodeshape(
                shapes_graph, src_shape, results_graph
            )
        if lookup_iri is not None and not lookup_iri.startswith("_:"):
            loc = shape_source_index.lookup(lookup_iri)
            if loc is not None:
                shape_file_str = str(loc.file_path)
                shape_line_int = loc.line_number

        # Data-source provenance.
        data_file_str: Optional[str] = None
        data_block_id: Optional[str] = None
        prov = data_provenance.get(focus_str)
        if prov is not None:
            data_file_str = (
                str(prov.file_path) if prov.file_path is not None else None
            )
            data_block_id = prov.block_id

        rule_id = _extract_rule_id(msg_str)

        result = EnrichedValidationResult(
            rule_id=rule_id,
            shape_file=shape_file_str,
            shape_line=shape_line_int,
            data_file=data_file_str,
            data_block_id=data_block_id,
            focus_node=focus_str,
            value=value_str,
            path=path_str,
            message=msg_str,
            severity=sev_str,
            source_constraint_component=src_cc_str,
            source_shape=src_shape_str,
        )
        enriched.append(result)

        if github_actions:
            _emit_github_annotation(result)

        if decision_capture is not None and sev_str == "critical":
            _emit_decision_capture(decision_capture, result)

    return enriched


def report_to_jsonld(
    results: Sequence[EnrichedValidationResult],
    *,
    conforms: bool,
) -> Dict[str, Any]:
    """Project a list of EnrichedValidationResult into a top-level
    ``ed4all:EnrichedValidationReport`` JSON-LD envelope (sub-plan §2.3).

    The envelope mirrors stock SHACL's ``sh:ValidationReport`` shape but
    with the enriched per-result projection. ``conforms`` is the
    pyshacl-supplied conformance boolean; consumers that don't carry
    that signal can pass ``conforms=False`` whenever
    ``len(violations) > 0`` (Q45 — conforms is governed by Violations,
    not by the result count).
    """
    return {
        "@context": {
            "ed4all": "https://ed4all.io/vocab/",
            "sh": _SH_NS,
            "conforms": "sh:conforms",
            "violationCount": "ed4all:violationCount",
            "results": "sh:result",
        },
        "@type": "ed4all:EnrichedValidationReport",
        "conforms": conforms,
        "violationCount": sum(1 for r in results if r.severity == "critical"),
        "results": [r.to_jsonld() for r in results],
    }


# ---------------------------------------------------------------------- #
# Internals
# ---------------------------------------------------------------------- #


def _extract_rule_id(message: str) -> Optional[str]:
    """Pull the leading ``XX-NNN`` code prefix from a result message.

    Matches the same convention the Phase 4 runner uses
    (``ShaclViolation.to_gate_issue`` extracts the same prefix). Phase
    6 surfaces it as ``ruleId`` on the EnrichedValidationResult; the
    Phase 4 runner exposes it as ``GateIssue.code``.
    """
    import re

    if not message:
        return None
    m = re.match(r"^([A-Z]{2,}-\d{2,})\b", message)
    return m.group(1) if m else None


def _resolve_owning_nodeshape(
    shapes_graph: Any,
    report_blank_shape: Any,
    report_graph: Any,
) -> Optional[str]:
    """Find the named NodeShape IRI that owns the inline PropertyShape
    pyshacl reported.

    pyshacl emits ``sh:sourceShape`` as a freshly-minted blank node in
    the *report* graph carrying ``sh:path`` / ``sh:minCount`` / etc.
    That blank node is NOT the one in the original shapes graph (rdflib
    blank-node identity isn't preserved across graphs), so we cannot
    pivot on identity. Instead we extract the constraint's "shape
    signature" — the ``sh:path`` plus the relevant constraint
    component value — and walk the shapes graph for a PropertyShape
    matching that signature, then walk upward to the NodeShape that
    owns it.

    For Phase 4 the signature is ``(sh:path schema:teaches, sh:minCount 1)``
    → matches the inline PropertyShape inside
    ``cfshapes:PageObjectivesMinCountShape``, which owns it via
    ``sh:property``.

    Returns the NodeShape's IRI string, or ``None`` if no unique match.
    """
    from rdflib import URIRef
    from rdflib.namespace import Namespace

    SH = Namespace(_SH_NS)

    # Pull the report-side PropertyShape signature.
    report_path = report_graph.value(report_blank_shape, SH.path)
    if report_path is None:
        return None

    # Collect candidate PropertyShape blank nodes in the shapes graph
    # that share the path and any pyshacl-relevant constraint values.
    constraint_predicates = (
        SH.minCount,
        SH.maxCount,
        SH.datatype,
        SH.nodeKind,
        getattr(SH, "in"),
        SH.hasValue,
        SH.pattern,
    )
    report_constraints = {}
    for pred in constraint_predicates:
        val = report_graph.value(report_blank_shape, pred)
        if val is not None:
            report_constraints[str(pred)] = val

    candidates = []
    for prop_node in shapes_graph.subjects(SH.path, report_path):
        # Verify the candidate's constraint set matches the report.
        ok = True
        for pred, expected in report_constraints.items():
            actual = shapes_graph.value(prop_node, URIRef(pred))
            if actual is None:
                continue
            if str(actual) != str(expected):
                ok = False
                break
        if ok:
            candidates.append(prop_node)

    if not candidates:
        return None

    # Walk upward: each candidate is an ``sh:property`` value of some
    # NodeShape. Find the URIRef owner.
    for prop_node in candidates:
        for owner in shapes_graph.subjects(SH.property, prop_node):
            if isinstance(owner, URIRef):
                return str(owner)
            # Anonymous owner — uncommon for Phase 4-style shapes but
            # possible; fall back to a single-step further walk.
            for grand in shapes_graph.subjects(SH.property, owner):
                if isinstance(grand, URIRef):
                    return str(grand)
    return None


def _emit_github_annotation(result: EnrichedValidationResult) -> None:
    """Write one GitHub workflow command line to stderr.

    Format (from GitHub Actions docs):

        ::<level> file=<path>,line=<n>,col=1::<message>

    When ``shape_file`` / ``shape_line`` aren't known, omit the file/line
    qualifiers; the annotation still shows in the workflow log, just
    without inline source rendering.
    """
    level = _SEVERITY_TO_GH.get(result.severity, "error")
    # GitHub interprets ``%``, ``\r``, and ``\n`` as command separators
    # inside annotation strings. Escape them so multi-line messages
    # don't collapse into multiple commands.
    safe_message = (
        result.message.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
    )

    if result.shape_file is not None and result.shape_line is not None:
        line = (
            f"::{level} file={result.shape_file},"
            f"line={result.shape_line},col=1::{safe_message}"
        )
    else:
        line = f"::{level}::{safe_message}"
    sys.stderr.write(line + "\n")


def _emit_decision_capture(
    decision_capture: Any,
    result: EnrichedValidationResult,
) -> None:
    """Log one DecisionCapture event for a Violation-severity result.

    Rationale interpolates the source shape, focus node, and message —
    typically 100-300 characters, well above the 20-char minimum the
    project main ``CLAUDE.md`` "Decision Capture" §"Required Fields"
    requires. ``decision_type="shacl_validation"`` is the canonical
    value for this call site.
    """
    rule_label = result.rule_id or result.source_constraint_component or "SHACL"
    decision_summary = (
        f"SHACL violation {rule_label} on focusNode {result.focus_node}"
    )
    rationale = (
        f"Shape {result.source_shape} fired {rule_label} on focus node "
        f"{result.focus_node}: {result.message[:240]}"
    )
    alternatives = []
    if result.shape_file is not None and result.shape_line is not None:
        alternatives.append(
            f"Edit shape at {result.shape_file}:{result.shape_line}"
        )
    if result.data_file is not None:
        alternatives.append(f"Edit data file {result.data_file}")

    decision_capture.log_decision(
        decision_type=DECISION_TYPE_SHACL,
        decision=decision_summary,
        rationale=rationale,
        alternatives_considered=alternatives or None,
        context=f"severity={result.severity}; sourceShape={result.source_shape}",
    )

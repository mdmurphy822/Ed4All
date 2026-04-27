"""SHACL runner for Phase 4 PoC validators (proof of concept).

Wraps pyshacl with the established Ed4All conventions (``inference="none"``,
``advanced=True``, ``meta_shacl=False``, ``js=False``) and a small
``ShaclViolation`` shape that downstream consumers (gate framework,
decision capture, future Phase 6 enrichment) can route on without
reaching into rdflib internals.

The runner is deliberately small: one shape file + one data graph in,
``(conforms, [ShaclViolation, ...])`` out. Phase 4.5+ work (multi-shape
shape graphs, SHACL Rules, SPARQL constraints) extends this surface; the
PoC keeps it minimal so the parallel ``page_objectives_shacl`` gate is
the only thing that breaks if the runner's API changes during the
graduation window.

Pipeline parity with ``LibV2/tools/libv2/_shacl_validator.py``:

    JSON-LD payload(s)
        └── pyld.jsonld.to_rdf  (apply Wave 62 @context, get N-Quads)
            └── rdflib.Graph    (ingest as nquads)
                └── pyshacl.validate (against the named shape file)
                    └── ShaclViolation(...) per sh:ValidationResult

Public API:

* ``run_shacl(shapes_path, data_graph) -> (conforms, [ShaclViolation, ...])``
* ``jsonld_payloads_to_graph(payloads) -> rdflib.Graph``
* ``parse_shacl_report(results_graph) -> [ShaclViolation, ...]``
* ``PageObjectivesShaclValidator`` — Validator-protocol-compatible class
  that the gate framework dispatches via the canonical
  ``lib.validators.shacl_runner.PageObjectivesShaclValidator`` import path.

Severity routing (sub-plan §5):

    sh:Violation -> "critical"
    sh:Warning   -> "warning"
    sh:Info      -> "info"

so the resulting ``GateIssue.severity`` strings match the rest of the
validator framework verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from MCP.hardening.validation_gates import GateIssue, GateResult

#: Logical IRI for the Courseforge JSON-LD @context. Mirrors
#: ``lib.ontology.jsonld_context_loader.CANONICAL_COURSEFORGE_CONTEXT_URL``.
#: Re-exported here so callers don't have to import the loader module
#: directly when they only need the URL constant.
CANONICAL_COURSEFORGE_CONTEXT_URL = "https://ed4all.dev/ns/courseforge/v1"


#: Where Phase 4 (and any future phase) ships its SHACL shape files.
#: Lookup convention: ``SHAPES_DIR / f"{gate_id}.ttl"``.
SHAPES_DIR = Path(__file__).resolve().parent / "shacl"


_SH_NS = "http://www.w3.org/ns/shacl#"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


_SEVERITY_MAP = {
    f"{_SH_NS}Violation": "critical",
    f"{_SH_NS}Warning": "warning",
    f"{_SH_NS}Info": "info",
}


__all__ = [
    "CANONICAL_COURSEFORGE_CONTEXT_URL",
    "SHAPES_DIR",
    "ShaclDepsMissing",
    "ShaclViolation",
    "PageObjectivesShaclValidator",
    "jsonld_payloads_to_graph",
    "parse_shacl_report",
    "run_shacl",
    "run_shacl_with_report_graph",
]


class ShaclDepsMissing(ImportError):
    """Raised when pyld / pyshacl / rdflib aren't importable.

    Mirrors the LibV2 sandbox's ``ShaclDepsMissing`` so callers across
    the repo can pattern-match the same exception type when they need
    to skip-on-missing instead of fail-loud.
    """


@dataclass
class ShaclViolation:
    """One row of a parsed SHACL ValidationReport.

    Field choice mirrors the canonical sh:ValidationResult shape (Q41,
    Q42): ``focus_node`` + ``path`` + ``severity`` + ``message`` are
    enough to locate any violation in the data graph; ``source_shape``
    + ``source_constraint_component`` are the metadata Phase 4 calls
    out as "richer diagnostics for free" relative to the Python gate's
    ``GateIssue`` shape.
    """

    focus_node: str
    path: Optional[str]
    severity: str  # "critical" / "warning" / "info"
    message: str
    source_shape: Optional[str] = None
    source_constraint_component: Optional[str] = None
    value: Optional[str] = None

    def to_gate_issue(self) -> GateIssue:
        """Project a ShaclViolation onto the Validator-framework GateIssue.

        Code prefix derives from the leading ``XX-NNN`` token in the
        message when present (per Q43 authoring rules — sub-plan §6
        requires a stable code prefix); falls back to the constraint
        component's local segment otherwise.
        """
        code = _extract_code_prefix(self.message) or _local_segment(
            self.source_constraint_component
        ) or "SHACL_VIOLATION"
        return GateIssue(
            severity=self.severity,
            code=code,
            message=self.message,
            location=self.focus_node,
            suggestion=None,
        )


def _ensure_deps():
    """Import pyld, pyshacl, rdflib or raise :class:`ShaclDepsMissing`."""
    try:
        import pyld  # noqa: F401
        import pyshacl  # noqa: F401
        import rdflib  # noqa: F401
    except ImportError as exc:  # pragma: no cover — exercised when extras missing
        raise ShaclDepsMissing(str(exc)) from exc


def _extract_code_prefix(message: str) -> Optional[str]:
    """Pull the leading ``XX-NNN`` (or longer) code prefix from a message.

    Authoring convention from sub-plan §6 + Q43: shape messages start
    with a stable code like ``PO-001:``. The runner harvests that
    prefix back into ``GateIssue.code`` so downstream tooling has a
    stable identifier to route on. Returns ``None`` if no prefix
    matches.
    """
    m = re.match(r"^([A-Z]{2,}-\d{2,})\b", message)
    return m.group(1) if m else None


def _local_segment(iri: Optional[str]) -> Optional[str]:
    """Return the local segment of an IRI, or ``None`` if not derivable."""
    if not iri:
        return None
    for sep in ("#", "/"):
        if sep in iri:
            tail = iri.rsplit(sep, 1)[1]
            if tail:
                return tail
    return iri


def jsonld_payloads_to_graph(
    payloads: Iterable[Dict[str, Any]],
    *,
    context_url: str = CANONICAL_COURSEFORGE_CONTEXT_URL,
) -> "rdflib.Graph":
    """Materialize a sequence of JSON-LD dicts as a single rdflib.Graph.

    Each payload gets ``@context: context_url`` injected, then is
    expanded via pyld to N-Quads, then merged into the same graph
    instance. The Wave 64 local document loader is registered up front
    so the canonical context resolves from disk.
    """
    _ensure_deps()
    from pyld import jsonld
    from rdflib import Graph

    # Wave 64 loader — register once, idempotent. Imported here so the
    # validators module doesn't pay the import cost at module load.
    from lib.ontology.jsonld_context_loader import register_local_loader

    register_local_loader()

    graph = Graph()
    for payload in payloads:
        with_context = dict(payload)
        with_context["@context"] = context_url
        nq = jsonld.to_rdf(with_context, {"format": "application/n-quads"})
        if nq:
            graph.parse(data=nq, format="nquads")
    return graph


def parse_shacl_report(results_graph: "rdflib.Graph") -> List[ShaclViolation]:
    """Walk a pyshacl ValidationReport graph; emit ShaclViolation rows.

    Implementation note: pyshacl returns the report as an rdflib.Graph
    keyed by ``sh:ValidationResult`` nodes; we enumerate every result,
    pull the standard fixed-property set the spec guarantees (Q41), and
    map ``sh:resultSeverity`` into the validator-framework severity
    strings.
    """
    _ensure_deps()
    from rdflib import Namespace, URIRef
    from rdflib.term import Literal

    SH = Namespace(_SH_NS)
    RDF = Namespace(_RDF_NS)

    results: List[ShaclViolation] = []
    for result_node in results_graph.subjects(RDF.type, SH.ValidationResult):
        focus = results_graph.value(result_node, SH.focusNode)
        path = results_graph.value(result_node, SH.resultPath)
        sev = results_graph.value(result_node, SH.resultSeverity)
        msg = results_graph.value(result_node, SH.resultMessage)
        src_shape = results_graph.value(result_node, SH.sourceShape)
        src_cc = results_graph.value(result_node, SH.sourceConstraintComponent)
        value = results_graph.value(result_node, SH.value)

        sev_str = _SEVERITY_MAP.get(str(sev), "critical")
        msg_str = (
            str(msg)
            if isinstance(msg, Literal)
            else (str(msg) if msg is not None else "")
        )
        focus_str = str(focus) if focus is not None else ""
        path_str = str(path) if path is not None else None
        src_shape_str = str(src_shape) if isinstance(src_shape, URIRef) else None
        src_cc_str = str(src_cc) if isinstance(src_cc, URIRef) else None
        value_str = str(value) if value is not None else None

        results.append(
            ShaclViolation(
                focus_node=focus_str,
                path=path_str,
                severity=sev_str,
                message=msg_str,
                source_shape=src_shape_str,
                source_constraint_component=src_cc_str,
                value=value_str,
            )
        )
    return results


def run_shacl(
    shapes_path: Path,
    data_graph: Union["rdflib.Graph", str],
    *,
    inference: str = "none",
) -> Tuple[bool, List[ShaclViolation]]:
    """Validate a data graph against a SHACL shapes file.

    Args:
        shapes_path: Path to a Turtle shapes file. Must exist; a missing
            file raises :class:`FileNotFoundError`.
        data_graph: Either a fully-loaded ``rdflib.Graph`` (preferred —
            the caller controls how triples are assembled) or an
            N-Quads string the runner will parse internally.
        inference: pyshacl ``inference`` mode; defaults to ``"none"`` to
            match the LibV2 / Wave 67 convention (closed-world,
            shape-author-asserted-only validation).

    Returns:
        ``(conforms, violations)``. ``conforms=True`` means the shapes
        graph reports no violations on the data graph. ``violations``
        is the parsed report list — empty when ``conforms=True``.

    Raises:
        FileNotFoundError: If ``shapes_path`` does not point at a
            readable file.
        ShaclDepsMissing: If the SHACL toolchain isn't importable.
    """
    _ensure_deps()
    import pyshacl
    from rdflib import Graph

    shapes_path = Path(shapes_path)
    if not shapes_path.exists():
        raise FileNotFoundError(
            f"SHACL shapes file not found: {shapes_path}. "
            f"Phase 4 PoC ships shapes under {SHAPES_DIR}."
        )

    if isinstance(data_graph, str):
        graph = Graph()
        graph.parse(data=data_graph, format="nquads")
    else:
        graph = data_graph

    shapes_graph = Graph()
    shapes_graph.parse(shapes_path, format="turtle")

    conforms, results_graph, _results_text = pyshacl.validate(
        data_graph=graph,
        shacl_graph=shapes_graph,
        inference=inference,
        abort_on_first=False,
        meta_shacl=False,
        advanced=True,
        js=False,
        debug=False,
    )

    violations = parse_shacl_report(results_graph)
    return bool(conforms), violations


def run_shacl_with_report_graph(
    shapes_path: Path,
    data_graph: Union["rdflib.Graph", str],
    *,
    inference: str = "none",
) -> Tuple[bool, List[ShaclViolation], "rdflib.Graph"]:
    """Phase 6 sibling of :func:`run_shacl` that ALSO returns the raw
    pyshacl ``ValidationReport`` graph so the
    ``shacl_result_enricher`` module can re-walk it for shape-source /
    data-source provenance enrichment.

    The behavior mirrors :func:`run_shacl` exactly — same pyshacl
    invocation, same defaults (``inference="none"``,
    ``advanced=True``, ``meta_shacl=False``, ``js=False``,
    ``abort_on_first=False``). Phase 4's ``run_shacl`` is unchanged;
    this is a strict superset. See
    ``plans/phase-6-shacl-result-enrichment.md`` § 4.3 for the rationale.

    Returns:
        ``(conforms, violations, results_graph)``. ``results_graph`` is
        an ``rdflib.Graph`` containing the SHACL ValidationReport;
        callers that don't need it should keep using
        :func:`run_shacl` to avoid carrying a graph reference around.
    """
    _ensure_deps()
    import pyshacl
    from rdflib import Graph

    shapes_path = Path(shapes_path)
    if not shapes_path.exists():
        raise FileNotFoundError(
            f"SHACL shapes file not found: {shapes_path}. "
            f"Phase 4 PoC ships shapes under {SHAPES_DIR}."
        )

    if isinstance(data_graph, str):
        graph = Graph()
        graph.parse(data=data_graph, format="nquads")
    else:
        graph = data_graph

    shapes_graph = Graph()
    shapes_graph.parse(shapes_path, format="turtle")

    conforms, results_graph, _results_text = pyshacl.validate(
        data_graph=graph,
        shacl_graph=shapes_graph,
        inference=inference,
        abort_on_first=False,
        meta_shacl=False,
        advanced=True,
        js=False,
        debug=False,
    )

    violations = parse_shacl_report(results_graph)
    return bool(conforms), violations, results_graph


# --------------------------------------------------------------------- #
# Validator-protocol class for the page_objectives_shacl gate
# --------------------------------------------------------------------- #


_JSON_LD_RE = re.compile(
    r'<script\s+type="application/ld\+json"\s*>\s*(\{.*?\})\s*</script>',
    re.DOTALL | re.IGNORECASE,
)


def _extract_jsonld_blocks(html: str) -> List[Dict[str, Any]]:
    """Mirror of Courseforge/scripts/validate_page_objectives.py extractor.

    Re-implemented in this module rather than imported to avoid the
    Courseforge-scripts ``sys.path`` surgery the existing wrapper does;
    keeps the SHACL runner's import surface clean.
    """
    import json

    blocks: List[Dict[str, Any]] = []
    for match in _JSON_LD_RE.finditer(html):
        try:
            blocks.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    return blocks


def _discover_html_pages(content_dir: Path) -> List[Path]:
    """Return every week_*/*.html under ``content_dir`` for SHACL scanning.

    Matches the Python gate's behavior: only ``week_*`` paths are
    in-scope; project docs and non-week HTML aren't expected to carry
    LO metadata.
    """
    if content_dir.is_file():
        return [content_dir]
    pages: List[Path] = []
    for p in sorted(content_dir.rglob("*.html")):
        if any(part.startswith("week_") for part in p.parts):
            pages.append(p)
    return pages


class PageObjectivesShaclValidator:
    """Phase 4 PoC: SHACL parallel of PageObjectivesValidator.

    Wired as a ``warning``-severity, parallel-not-replacement gate
    alongside the existing Python ``page_objectives`` gate. Reuses the
    same gate-input builder (``_build_page_objectives``) so workflow
    configuration is identical except for the validator class path.

    Validate flow:

    1. Walk ``content_dir`` for ``week_*/*.html`` pages.
    2. Extract every JSON-LD block from each page (Courseforge stamps
       one ``application/ld+json`` block per page in <head>).
    3. Materialize the blocks as a single RDF graph through the Wave 62
       @context.
    4. Run them through ``page_objectives_shacl.ttl``.
    5. Project SHACL violations onto ``GateIssue`` rows.

    Behavior contract sub-plan §8:
        - SHACL deps missing -> single warning issue, ``passed=True``.
          Cannot block the workflow during PoC.
        - ``content_dir`` missing -> single error issue, ``passed=False``.
        - Empty corpus (no week_* pages) -> ``passed=True``, no issues.
        - Real violations -> projected to GateIssue list, severity
          mapped from sh:Violation/Warning per the runner.
    """

    name = "page_objectives_shacl"
    version = "0.1.0"  # PoC

    def __init__(
        self,
        *,
        shapes_path: Optional[Path] = None,
    ) -> None:
        self._shapes_path = (
            Path(shapes_path)
            if shapes_path is not None
            else SHAPES_DIR / "page_objectives_shacl.ttl"
        )

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "page_objectives_shacl")

        content_dir_raw = inputs.get("content_dir")
        if not content_dir_raw:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="MISSING_CONTENT_DIR",
                        message=(
                            "content_dir is required for "
                            "PageObjectivesShaclValidator"
                        ),
                    )
                ],
            )

        content_dir = Path(content_dir_raw)
        if not content_dir.exists():
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="CONTENT_DIR_NOT_FOUND",
                        message=f"content_dir does not exist: {content_dir}",
                    )
                ],
            )

        # SHACL deps may not be present in every environment (pyld /
        # pyshacl are dev-extras). Degrade gracefully so the PoC gate
        # never blocks a run on missing extras — sub-plan §8 PoC
        # severity is `warning` regardless.
        try:
            _ensure_deps()
        except ShaclDepsMissing as exc:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[
                    GateIssue(
                        severity="warning",
                        code="SHACL_DEPS_MISSING",
                        message=(
                            f"SHACL toolchain not importable: {exc}. "
                            "Phase 4 PoC gate skipped; the Python "
                            "page_objectives gate is still authoritative."
                        ),
                    )
                ],
            )

        pages = _discover_html_pages(content_dir)
        if not pages:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        payloads: List[Dict[str, Any]] = []
        for page in pages:
            try:
                html = page.read_text(encoding="utf-8")
            except OSError:
                continue
            payloads.extend(_extract_jsonld_blocks(html))

        if not payloads:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                score=1.0,
                issues=[],
            )

        graph = jsonld_payloads_to_graph(payloads)
        try:
            conforms, violations = run_shacl(self._shapes_path, graph)
        except FileNotFoundError as exc:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="SHAPE_FILE_MISSING",
                        message=str(exc),
                    )
                ],
            )

        issues = [v.to_gate_issue() for v in violations]
        critical = sum(1 for i in issues if i.severity == "critical")
        score = 1.0 if not violations else max(
            0.0, 1.0 - len(violations) / max(1, len(payloads))
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=conforms or critical == 0,
            score=score,
            issues=issues,
        )

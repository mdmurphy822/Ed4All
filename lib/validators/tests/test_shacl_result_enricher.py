"""Phase 6 tests — SHACL result enrichment + GitHub annotations + DecisionCapture.

Layered per ``plans/phase-6-shacl-result-enrichment.md`` § 3:

* §3.1 Unit tests of ``ShapeSourceIndex`` and ``EnrichedValidationResult``.
* §3.2 Integration test: full Phase 4 corpus -> ``run_shacl_with_report_graph``
  -> ``enrich_validation_report``; assert at least one violation gets
  ``shape_line == 41`` (the actual line where the canonical Phase 4
  ``cfshapes:PageObjectivesMinCountShape`` is declared in
  ``lib/validators/shacl/page_objectives_shacl.ttl``).
* §3.3 GitHub annotation tests (env var + stderr capture).
* §3.4 DecisionCapture tests (real instance, scoped to a tmp dir).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Add repo root for sibling-module imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# rdflib is a hard project dependency (Wave 2 promoted), but the SHACL
# extras (pyld + pyshacl) are dev-extras. Skip the integration tests
# when the SHACL toolchain isn't installed; the unit tests of the line
# scanner and the JSON-LD projection only need rdflib.
rdflib = pytest.importorskip("rdflib")


from lib.validators.shacl_result_enricher import (  # noqa: E402
    DECISION_TYPE_SHACL,
    DataProvenance,
    EnrichedValidationResult,
    enrich_validation_report,
    report_to_jsonld,
)
from lib.validators.shape_provenance import (  # noqa: E402
    ShapeSourceIndex,
    ShapeSourceLocation,
)


PHASE4_SHAPE_FILE = (
    _REPO_ROOT / "lib" / "validators" / "shacl" / "page_objectives_shacl.ttl"
)


# ---------------------------------------------------------------------- #
# §3.1 — Unit tests
# ---------------------------------------------------------------------- #


def test_shape_source_index_resolves_phase4_shape():
    """The canonical Phase 4 NodeShape is declared at line 41."""
    idx = ShapeSourceIndex()
    mapping = idx.build_for_file(PHASE4_SHAPE_FILE)
    target = "https://ed4all.dev/ns/courseforge/v1/shapes#PageObjectivesMinCountShape"
    assert target in mapping, (
        f"ShapeSourceIndex did not index the canonical Phase 4 NodeShape; "
        f"found: {list(mapping)}"
    )
    loc = mapping[target]
    assert isinstance(loc, ShapeSourceLocation)
    assert loc.file_path == PHASE4_SHAPE_FILE
    assert loc.line_number == 41, (
        f"PageObjectivesMinCountShape declared at line {loc.line_number}, "
        f"expected 41 (lib/validators/shacl/page_objectives_shacl.ttl)."
    )


def test_shape_source_index_lookup_caches_per_file(tmp_path):
    idx = ShapeSourceIndex()
    idx.build_for_file(PHASE4_SHAPE_FILE)
    target = "https://ed4all.dev/ns/courseforge/v1/shapes#PageObjectivesMinCountShape"
    loc1 = idx.lookup(target)
    loc2 = idx.lookup(target, hint_file=PHASE4_SHAPE_FILE)
    assert loc1 == loc2
    assert loc1 is not None
    assert loc1.line_number == 41


def test_shape_source_index_returns_none_for_unknown_iri():
    idx = ShapeSourceIndex()
    idx.build_for_file(PHASE4_SHAPE_FILE)
    assert idx.lookup("https://example.org/UnknownShape") is None
    # Blank-node IRIs (serialised as ``_:b1``) must also return None
    # rather than raising — this is the inline-PropertyShape case.
    assert idx.lookup("_:b0") is None
    assert idx.lookup("") is None


def test_shape_source_index_tolerates_malformed_turtle(tmp_path):
    """Garbage Turtle yields an empty mapping but does not raise."""
    bad = tmp_path / "garbage.ttl"
    bad.write_text("this is not turtle :: at all\n", encoding="utf-8")
    idx = ShapeSourceIndex()
    mapping = idx.build_for_file(bad)
    assert mapping == {}


def test_shape_source_index_handles_full_iri_form(tmp_path):
    """Shape declared with ``<full IRI>`` instead of a CURIE — both
    inline (single-line) and multi-line styles are indexed; the
    indexed line number points at the IDENTIFIER line per the
    canonical Phase 4 convention."""
    # Multi-line style (Phase 4's pattern): IRI on its own line, anchor
    # follows on the next non-blank line.
    ttl = tmp_path / "full_iri_shape.ttl"
    ttl.write_text(
        "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
        "<https://example.org/MyShape>\n"
        "  a sh:NodeShape ;\n"
        "  sh:targetClass <https://example.org/MyClass> .\n",
        encoding="utf-8",
    )
    idx = ShapeSourceIndex()
    mapping = idx.build_for_file(ttl)
    assert "https://example.org/MyShape" in mapping
    # Identifier sits on line 2; the anchor is on line 3. The scanner
    # commits the IDENTIFIER line so editors land on the IRI, not the
    # type predicate.
    assert mapping["https://example.org/MyShape"].line_number == 2

    # Inline style: IRI and anchor on the same line.
    ttl_inline = tmp_path / "inline_iri_shape.ttl"
    ttl_inline.write_text(
        "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
        "<https://example.org/MyShape> a sh:NodeShape ;\n"
        "  sh:targetClass <https://example.org/MyClass> .\n",
        encoding="utf-8",
    )
    idx2 = ShapeSourceIndex()
    mapping2 = idx2.build_for_file(ttl_inline)
    assert "https://example.org/MyShape" in mapping2
    assert mapping2["https://example.org/MyShape"].line_number == 2


def test_enriched_result_to_jsonld_carries_all_fields():
    """``to_jsonld`` round-trips every plan field including @context."""
    result = EnrichedValidationResult(
        rule_id="PO-001",
        shape_file="lib/validators/shacl/page_objectives_shacl.ttl",
        shape_line=41,
        data_file="tmp/week_01/week_01_overview.html",
        data_block_id=None,
        focus_node="https://ed4all.io/cm/0",
        value=None,
        path="http://schema.org/teaches",
        message="PO-001: every CourseModule must declare ...",
        severity="critical",
        source_constraint_component=(
            "http://www.w3.org/ns/shacl#MinCountConstraintComponent"
        ),
        source_shape=(
            "https://ed4all.dev/ns/courseforge/v1/shapes#PageObjectivesMinCountShape"
        ),
    )
    doc = result.to_jsonld()
    assert "@context" in doc
    assert doc["@type"] == "ed4all:EnrichedValidationResult"
    assert doc["ruleId"] == "PO-001"
    assert doc["shapeFile"].endswith("page_objectives_shacl.ttl")
    assert doc["shapeLine"] == 41
    assert doc["focusNode"] == "https://ed4all.io/cm/0"
    assert doc["severity"] == "critical"
    # Ensure JSON-serialisable end-to-end.
    json.dumps(doc)


def test_report_to_jsonld_envelope_carries_violation_count():
    results = [
        EnrichedValidationResult(
            rule_id="PO-001",
            shape_file=None,
            shape_line=None,
            data_file=None,
            data_block_id=None,
            focus_node="https://ed4all.io/cm/0",
            value=None,
            path=None,
            message="PO-001: ...",
            severity="critical",
            source_constraint_component=None,
            source_shape=None,
        ),
        EnrichedValidationResult(
            rule_id="X-002",
            shape_file=None,
            shape_line=None,
            data_file=None,
            data_block_id=None,
            focus_node="https://ed4all.io/cm/1",
            value=None,
            path=None,
            message="X-002: warn",
            severity="warning",
            source_constraint_component=None,
            source_shape=None,
        ),
    ]
    envelope = report_to_jsonld(results, conforms=False)
    assert envelope["@type"] == "ed4all:EnrichedValidationReport"
    assert envelope["conforms"] is False
    assert envelope["violationCount"] == 1  # only the critical row
    assert len(envelope["results"]) == 2


def test_enricher_handles_missing_value_and_path(tmp_path):
    """Synthetic ValidationReport without sh:value / sh:resultPath emits
    ``value=None, path=None`` rather than raising (Q42 contract)."""
    g = rdflib.Graph()
    SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")
    RDF = rdflib.RDF

    result_node = rdflib.BNode()
    g.add((result_node, RDF.type, SH.ValidationResult))
    g.add((result_node, SH.focusNode, rdflib.URIRef("https://ed4all.io/cm/0")))
    g.add((result_node, SH.resultSeverity, SH.Violation))
    g.add(
        (
            result_node,
            SH.resultMessage,
            rdflib.Literal("PO-001: synthetic minCount violation."),
        )
    )
    # Intentionally no sh:value / sh:resultPath / sh:sourceShape.

    idx = ShapeSourceIndex()
    enriched = enrich_validation_report(g, shape_source_index=idx)
    assert len(enriched) == 1
    r = enriched[0]
    assert r.value is None
    assert r.path is None
    assert r.shape_file is None
    assert r.shape_line is None
    assert r.severity == "critical"
    assert r.rule_id == "PO-001"


def test_enricher_handles_unknown_shape_iri():
    """sourceShape IRI not in the cache -> shape_file/line both None."""
    g = rdflib.Graph()
    SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")
    RDF = rdflib.RDF

    result_node = rdflib.BNode()
    g.add((result_node, RDF.type, SH.ValidationResult))
    g.add((result_node, SH.focusNode, rdflib.URIRef("https://ed4all.io/cm/0")))
    g.add((result_node, SH.resultSeverity, SH.Violation))
    g.add(
        (
            result_node,
            SH.sourceShape,
            rdflib.URIRef("https://example.org/UnknownShape"),
        )
    )
    g.add((result_node, SH.resultMessage, rdflib.Literal("X-001: ...")))

    idx = ShapeSourceIndex()
    idx.build_for_file(PHASE4_SHAPE_FILE)  # cache populated, but with the
    # Phase 4 shape, NOT the synthetic UnknownShape.
    enriched = enrich_validation_report(g, shape_source_index=idx)
    assert len(enriched) == 1
    r = enriched[0]
    assert r.source_shape == "https://example.org/UnknownShape"
    assert r.shape_file is None
    assert r.shape_line is None


# ---------------------------------------------------------------------- #
# §3.2 — Integration test: full Phase 4 stack
# ---------------------------------------------------------------------- #

# Skip the integration tests when SHACL extras aren't installed.
pyld = pytest.importorskip("pyld")
pyshacl = pytest.importorskip("pyshacl")


from lib.validators.shacl_runner import (  # noqa: E402
    jsonld_payloads_to_graph,
    run_shacl_with_report_graph,
)


def _phase4_shapes_graph() -> "rdflib.Graph":
    """Parse the canonical Phase 4 shape file once per test invocation.

    The enricher's blank-node -> NodeShape upward walk needs the shapes
    graph; pyshacl reports ``sh:sourceShape`` as the inline
    PropertyShape blank node for Phase 4's shape, so without the
    shapes graph the line lookup misses.
    """
    g = rdflib.Graph()
    g.parse(PHASE4_SHAPE_FILE, format="turtle")
    return g


def _make_empty_lo_payload(page_id: str = "week_01_overview") -> Dict[str, Any]:
    """Mirror of test_shacl_runner._make_page_payload(lo_ids=[]).

    Inlined here so we don't cross-import test modules (pytest discovery
    can be finicky about that). Builds a minimal CourseModule with an
    explicitly empty learningObjectives list — fires the Phase 4
    minCount=1 shape.
    """
    return {
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": page_id,
        "learningObjectives": [],
    }


def test_integration_phase4_violation_carries_shape_line(tmp_path):
    """Real Phase 4 violation -> enricher attaches shape_line=41.

    This is the central integration test: build an empty-LO CourseModule
    payload, run it through the Phase 4 SHACL pipeline, capture the
    raw rdflib report graph via ``run_shacl_with_report_graph``, pass
    it into the enricher, and assert at least one violation got fully
    enriched (shape_file matches the Phase 4 TTL, shape_line == 41).
    """
    payload = _make_empty_lo_payload()
    data_graph = jsonld_payloads_to_graph([payload])
    conforms, violations, results_graph = run_shacl_with_report_graph(
        PHASE4_SHAPE_FILE, data_graph
    )
    assert not conforms, "Empty-LO payload should fail Phase 4 minCount."
    assert len(violations) >= 1

    idx = ShapeSourceIndex()
    shapes_graph = rdflib.Graph()
    shapes_graph.parse(PHASE4_SHAPE_FILE, format="turtle")
    enriched = enrich_validation_report(
        results_graph,
        shape_source_index=idx,
        shape_files=[PHASE4_SHAPE_FILE],
        shapes_graph=shapes_graph,
    )
    assert len(enriched) >= 1, "Enricher produced no rows."

    # At least one enriched row should have full shape provenance.
    fully_enriched = [
        r for r in enriched if r.shape_file is not None and r.shape_line is not None
    ]
    assert fully_enriched, (
        "No enriched result carried both shape_file + shape_line; "
        f"results: {[(r.shape_file, r.shape_line, r.source_shape) for r in enriched]}"
    )
    sample = fully_enriched[0]
    assert sample.shape_file == str(PHASE4_SHAPE_FILE)
    assert sample.shape_line == 41
    assert sample.severity == "critical"
    assert sample.rule_id == "PO-001"


def test_integration_data_provenance_join(tmp_path):
    """Caller-supplied data_provenance map populates data_file + data_block_id."""
    payload = _make_empty_lo_payload()
    data_graph = jsonld_payloads_to_graph([payload])
    _conforms, _violations, results_graph = run_shacl_with_report_graph(
        PHASE4_SHAPE_FILE, data_graph
    )

    # We don't know the focus-node IRI a priori (pyld mints it deterministically
    # from the payload but the exact form depends on the @context loader). Pull
    # focus nodes from the graph and build a provenance map for them.
    SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")
    RDF = rdflib.RDF
    focus_iris: List[str] = []
    for result_node in results_graph.subjects(RDF.type, SH.ValidationResult):
        focus = results_graph.value(result_node, SH.focusNode)
        if focus is not None:
            focus_iris.append(str(focus))
    assert focus_iris, "No focus nodes in the report graph."

    fake_html = tmp_path / "week_01" / "week_01_overview.html"
    fake_html.parent.mkdir(parents=True, exist_ok=True)
    fake_html.write_text("<html></html>", encoding="utf-8")
    provenance = {
        focus_iris[0]: DataProvenance(file_path=fake_html, block_id="block-12"),
    }

    idx = ShapeSourceIndex()
    enriched = enrich_validation_report(
        results_graph,
        shape_source_index=idx,
        shape_files=[PHASE4_SHAPE_FILE],
        shapes_graph=_phase4_shapes_graph(),
        data_provenance=provenance,
    )
    matched = [r for r in enriched if r.focus_node == focus_iris[0]]
    assert matched
    assert matched[0].data_file == str(fake_html)
    assert matched[0].data_block_id == "block-12"


# ---------------------------------------------------------------------- #
# §3.3 — GitHub annotation tests
# ---------------------------------------------------------------------- #


def test_github_annotation_emitted_under_env_flag(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    payload = _make_empty_lo_payload()
    data_graph = jsonld_payloads_to_graph([payload])
    _c, _v, results_graph = run_shacl_with_report_graph(
        PHASE4_SHAPE_FILE, data_graph
    )

    idx = ShapeSourceIndex()
    enrich_validation_report(
        results_graph,
        shape_source_index=idx,
        shape_files=[PHASE4_SHAPE_FILE],
        shapes_graph=_phase4_shapes_graph(),
    )

    captured = capsys.readouterr()
    err_lines = [ln for ln in captured.err.splitlines() if ln.startswith("::")]
    assert err_lines, (
        "No GitHub annotation lines emitted on stderr; "
        f"stderr was: {captured.err!r}"
    )
    # At least one ``::error`` line for the Violation severity.
    assert any(ln.startswith("::error ") for ln in err_lines), (
        f"Expected at least one ::error line; got: {err_lines}"
    )
    # That line should carry file= and line= qualifiers (shape provenance hit).
    error_lines = [ln for ln in err_lines if ln.startswith("::error ")]
    assert any("line=41" in ln for ln in error_lines), (
        f"Expected line=41 qualifier; got: {error_lines}"
    )
    # And the canonical PO-001 prefix in the message.
    assert any("PO-001" in ln for ln in error_lines)


def test_no_github_annotation_when_env_unset(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    payload = _make_empty_lo_payload()
    data_graph = jsonld_payloads_to_graph([payload])
    _c, _v, results_graph = run_shacl_with_report_graph(
        PHASE4_SHAPE_FILE, data_graph
    )

    idx = ShapeSourceIndex()
    enrich_validation_report(
        results_graph,
        shape_source_index=idx,
        shape_files=[PHASE4_SHAPE_FILE],
        shapes_graph=_phase4_shapes_graph(),
    )

    captured = capsys.readouterr()
    err_lines = [ln for ln in captured.err.splitlines() if ln.startswith("::")]
    assert not err_lines, (
        f"GitHub annotation emitted when GITHUB_ACTIONS unset: {err_lines}"
    )


# ---------------------------------------------------------------------- #
# §3.4 — DecisionCapture tests
# ---------------------------------------------------------------------- #


@pytest.fixture
def isolated_decision_capture(tmp_path, monkeypatch):
    """Yield a real DecisionCapture instance scoped to ``tmp_path``.

    LibV2Storage tries to create training-capture directories under the
    course root; pointing the legacy training dir at tmp_path keeps the
    test hermetic. The capture's ``decisions`` list is populated as
    log_decision is called, so tests can assert on it directly.
    """
    monkeypatch.setenv("LEGACY_TRAINING_DIR", str(tmp_path / "training"))
    from lib.decision_capture import DecisionCapture

    capture = DecisionCapture(
        course_code="TEST_101",
        phase="validation",
        tool="trainforge",
        streaming=False,  # no file IO; we only inspect .decisions in-memory
    )
    yield capture
    try:
        capture.close()
    except Exception:
        pass


def test_decision_capture_one_event_per_violation(isolated_decision_capture):
    """One log_decision call per Violation-severity result."""
    payload = _make_empty_lo_payload()
    data_graph = jsonld_payloads_to_graph([payload])
    _c, _v, results_graph = run_shacl_with_report_graph(
        PHASE4_SHAPE_FILE, data_graph
    )

    idx = ShapeSourceIndex()
    enriched = enrich_validation_report(
        results_graph,
        shape_source_index=idx,
        shape_files=[PHASE4_SHAPE_FILE],
        decision_capture=isolated_decision_capture,
    )

    violation_count = sum(1 for r in enriched if r.severity == "critical")
    assert violation_count >= 1
    # Filter to shacl_validation events (the capture may carry other events
    # if LibV2 storage default-logs anything; right now it doesn't, but be
    # defensive).
    shacl_events = [
        d
        for d in isolated_decision_capture.decisions
        if d.get("decision_type") == DECISION_TYPE_SHACL
    ]
    assert len(shacl_events) == violation_count, (
        f"Expected {violation_count} shacl_validation captures, got "
        f"{len(shacl_events)}; decisions: "
        f"{[d.get('decision_type') for d in isolated_decision_capture.decisions]}"
    )


def test_decision_capture_rationale_meets_minimum_length(
    isolated_decision_capture,
):
    """Rationale must be ≥20 chars and reference the focus node + message."""
    payload = _make_empty_lo_payload()
    data_graph = jsonld_payloads_to_graph([payload])
    _c, _v, results_graph = run_shacl_with_report_graph(
        PHASE4_SHAPE_FILE, data_graph
    )

    idx = ShapeSourceIndex()
    enriched = enrich_validation_report(
        results_graph,
        shape_source_index=idx,
        shape_files=[PHASE4_SHAPE_FILE],
        decision_capture=isolated_decision_capture,
    )

    shacl_events = [
        d
        for d in isolated_decision_capture.decisions
        if d.get("decision_type") == DECISION_TYPE_SHACL
    ]
    assert shacl_events, "No shacl_validation events captured."
    for event in shacl_events:
        rationale = event.get("rationale", "")
        assert len(rationale) >= 20, (
            f"Rationale too short ({len(rationale)} chars): {rationale!r}"
        )
        # Rationale references either the focus node or the PO-001 prefix.
        focus_iris = {r.focus_node for r in enriched if r.severity == "critical"}
        assert any(focus in rationale for focus in focus_iris) or "PO-001" in rationale, (
            f"Rationale doesn't reference focus node or PO-001: {rationale!r}"
        )


def test_decision_capture_skips_warning_severity(
    isolated_decision_capture,
):
    """Synthetic Warning result -> no DecisionCapture event."""
    g = rdflib.Graph()
    SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")
    RDF = rdflib.RDF

    result_node = rdflib.BNode()
    g.add((result_node, RDF.type, SH.ValidationResult))
    g.add((result_node, SH.focusNode, rdflib.URIRef("https://ed4all.io/cm/0")))
    g.add((result_node, SH.resultSeverity, SH.Warning))
    g.add(
        (
            result_node,
            SH.resultMessage,
            rdflib.Literal("X-001: synthetic warning."),
        )
    )

    idx = ShapeSourceIndex()
    enriched = enrich_validation_report(
        g,
        shape_source_index=idx,
        decision_capture=isolated_decision_capture,
    )
    assert len(enriched) == 1
    assert enriched[0].severity == "warning"
    shacl_events = [
        d
        for d in isolated_decision_capture.decisions
        if d.get("decision_type") == DECISION_TYPE_SHACL
    ]
    assert shacl_events == [], (
        f"Warning-severity result should not trigger a capture; got: {shacl_events}"
    )

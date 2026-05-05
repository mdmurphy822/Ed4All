"""Phase 4 PoC tests — SHACL runner + page_objectives_shacl shape.

Layered per sub-plan §9 (plans/phase-4-validators-to-shacl.md):

* §9.1 Unit tests: shape file resolution, parsing, severity routing,
  error handling.
* §9.2 Equivalence tests vs Python PageObjectivesValidator: build a
  fixture corpus, run both gates, compare accept/reject decisions.
  Documents the empty-LO case where SHACL is intentionally STRICTER
  than the Python gate (richer-diagnostics-for-free win — sub-plan
  §9.2).
* §9.3 SHACL meta-validation: the shape file itself parses cleanly
  and declares well-formed sh:NodeShape / sh:property structure (i.e.
  no typo'd predicate paths, every NodeShape carries at least one
  rdfs:label, every PropertyShape has both sh:path and sh:minCount or
  another constraint component).

The PoC stays small on purpose. Once the gate graduates from
informational `warning` severity to `critical`, this file is the
contract that protects against shape regressions during the
graduation window.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add repo root for sibling-module imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Skip the entire module when SHACL extras aren't installed; these
# tests can't run meaningfully without the toolchain.
pyld = pytest.importorskip(
    "pyld",
    reason="pyld is required for SHACL tests; install with `pip install pyld`.",
)
pyshacl = pytest.importorskip(
    "pyshacl",
    reason="pyshacl is required for SHACL tests; install with `pip install pyshacl`.",
)
rdflib = pytest.importorskip("rdflib", reason="rdflib comes with pyshacl.")

from rdflib import Graph, Namespace  # noqa: E402

from lib.validators.shacl_runner import (  # noqa: E402
    SHAPES_DIR,
    PageObjectivesShaclValidator,
    ShaclViolation,
    jsonld_payloads_to_graph,
    parse_shacl_report,
    run_shacl,
)
from lib.validators.page_objectives import PageObjectivesValidator  # noqa: E402

SHACL_NS = Namespace("http://www.w3.org/ns/shacl#")
SHAPE_FILE = SHAPES_DIR / "page_objectives_shacl.ttl"


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def shape_path() -> Path:
    return SHAPE_FILE


def _make_page_payload(*, course_code: str = "TEST_101", week: int = 1,
                       page_id: str = "week_01_overview",
                       lo_ids: list[str] | None = None) -> dict:
    """Build a minimal CourseModule JSON-LD payload.

    ``lo_ids=None`` produces a page WITH learningObjectives (the SHACL
    minCount=1 shape accepts it). ``lo_ids=[]`` produces an empty
    learningObjectives list, which violates minCount=1.
    """
    payload = {
        "@type": "CourseModule",
        "courseCode": course_code,
        "weekNumber": week,
        "moduleType": "overview",
        "pageId": page_id,
    }
    if lo_ids is None:
        payload["learningObjectives"] = [
            {
                "@type": "LearningObjective",
                "id": "https://example.org/los/CO-01",
                "statement": "Apply the framework to sample data.",
            }
        ]
    else:
        payload["learningObjectives"] = [
            {
                "@type": "LearningObjective",
                "id": f"https://example.org/los/{i}",
                "statement": f"Statement {i}",
            }
            for i in lo_ids
        ]
    return payload


def _write_html_page(path: Path, payload: dict) -> None:
    """Write a minimal HTML file with one JSON-LD block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "<!DOCTYPE html><html><head>"
        f'<script type="application/ld+json">{json.dumps(payload)}</script>'
        "</head><body><h1>Test page</h1></body></html>"
    )
    path.write_text(body, encoding="utf-8")


# --------------------------------------------------------------------- #
# §9.1 Unit tests of the runner
# --------------------------------------------------------------------- #


def test_shape_file_exists_at_canonical_location():
    assert SHAPE_FILE.exists(), (
        f"Phase 4 PoC shape file not found at {SHAPE_FILE}. "
        "Sub-plan §7 requires SHAPES_DIR / '<gate_id>.ttl' lookup."
    )


def test_shape_file_parses_as_turtle(shape_path):
    g = Graph()
    g.parse(shape_path, format="turtle")
    assert len(g) > 0, "Shape file parsed to empty graph."


def test_shape_declares_target_class(shape_path):
    """Shape MUST target ed4all:CourseModule per sub-plan §3."""
    g = Graph()
    g.parse(shape_path, format="turtle")
    targets = list(g.objects(predicate=SHACL_NS.targetClass))
    assert any(
        str(t) == "https://ed4all.dev/ns/courseforge/v1#CourseModule"
        for t in targets
    ), f"NodeShape must target ed4all:CourseModule; found targets: {targets}"


def test_run_shacl_accepts_well_formed_payload(shape_path):
    payload = _make_page_payload()
    graph = jsonld_payloads_to_graph([payload])
    conforms, violations = run_shacl(shape_path, graph)
    assert conforms, f"Well-formed payload rejected: {violations}"
    assert violations == []


def test_run_shacl_rejects_empty_lo_payload(shape_path):
    """SHACL minCount 1 MUST fire on learningObjectives: []."""
    payload = _make_page_payload(lo_ids=[])
    graph = jsonld_payloads_to_graph([payload])
    conforms, violations = run_shacl(shape_path, graph)
    assert not conforms, "Empty-LO payload incorrectly accepted."
    assert len(violations) == 1
    v = violations[0]
    # Severity routing per sub-plan §5.
    assert v.severity == "critical", (
        f"sh:Violation must map to 'critical', got {v.severity!r}."
    )
    # Message authoring per sub-plan §6 (Q43).
    assert "PO-001" in v.message, (
        f"Result message must carry the PO-001 code prefix; got {v.message!r}."
    )
    assert "schema:teaches" in v.message
    # Constraint component metadata (Q41 — richer diagnostics for free).
    assert v.source_constraint_component is not None
    assert "MinCountConstraintComponent" in v.source_constraint_component


def test_run_shacl_missing_shape_raises_file_not_found(tmp_path):
    bogus = tmp_path / "does_not_exist.ttl"
    with pytest.raises(FileNotFoundError):
        run_shacl(bogus, Graph())


def test_run_shacl_accepts_nquads_string(shape_path):
    """Runner must accept either an rdflib.Graph or an N-Quads string.

    rdflib's plain ``Graph`` only serializes to N-Triples (no graph
    name slot); we round-trip through nt and parse it back via the
    runner's nquads parser, which accepts triple-shaped lines.
    """
    payload = _make_page_payload()
    graph = jsonld_payloads_to_graph([payload])
    nq = graph.serialize(format="nt")
    conforms, _ = run_shacl(shape_path, nq)
    assert conforms


def test_parse_shacl_report_handles_empty_graph():
    """An empty results graph yields zero violations."""
    assert parse_shacl_report(Graph()) == []


def test_shacl_violation_to_gate_issue_carries_code_prefix():
    v = ShaclViolation(
        focus_node="https://example.org/page1",
        path="http://schema.org/teaches",
        severity="critical",
        message="PO-001: every CourseModule must declare ...",
    )
    issue = v.to_gate_issue()
    assert issue.code == "PO-001"
    assert issue.severity == "critical"
    assert issue.location == "https://example.org/page1"


# --------------------------------------------------------------------- #
# §9.2 Equivalence tests vs the Python gate
# --------------------------------------------------------------------- #


def _build_minimal_corpus(root: Path, *, with_los: bool) -> Path:
    """Create a content_dir with one week_01 page that does or does not
    carry a learningObjectives entry.

    Also writes a minimal ``course.json`` at the content-dir root so the
    Python ``PageObjectivesValidator`` has a canonical objectives source
    to validate against. Post-silent-degradation-cleanup the validator
    fail-closes on missing objectives_path / course.json (was warn-and-
    pass), so the SHACL ↔ Python equivalence tests need real objectives
    fixtures to exercise the LO-content branch instead of tripping the
    upstream-contract-failure branch. The course.json's
    chapter_objectives entry covers ``Week 1`` and declares the same
    ``CO-01`` URI that ``_make_page_payload`` emits in
    ``learningObjectives``.
    """
    page = root / "week_01" / "week_01_overview.html"
    payload = _make_page_payload() if with_los else _make_page_payload(lo_ids=[])
    _write_html_page(page, payload)
    course_json = {
        "terminal_objectives": [],
        "chapter_objectives": [
            {
                "chapter": "Week 1",
                "objectives": [
                    {
                        "id": "https://example.org/los/CO-01",
                        "statement": "Apply the framework to sample data.",
                        "bloomLevel": "apply",
                    }
                ],
            }
        ],
    }
    (root / "course.json").write_text(json.dumps(course_json), encoding="utf-8")
    return root


def test_equivalence_accept_path(tmp_path):
    """Page WITH learningObjectives -> both gates accept."""
    content_dir = _build_minimal_corpus(tmp_path, with_los=True)

    py_result = PageObjectivesValidator().validate(
        {"content_dir": str(content_dir)}
    )
    py_critical = sum(1 for i in py_result.issues if i.severity == "critical")

    shacl_result = PageObjectivesShaclValidator().validate(
        {"content_dir": str(content_dir)}
    )
    shacl_critical = sum(1 for i in shacl_result.issues if i.severity == "critical")

    assert py_critical == 0, f"Python gate unexpectedly rejected: {py_result.issues}"
    assert shacl_critical == 0, (
        f"SHACL gate unexpectedly rejected: {shacl_result.issues}"
    )
    assert py_result.passed
    assert shacl_result.passed


def test_equivalence_reject_path_documented_asymmetry(tmp_path):
    """Page with learningObjectives: [] -> SHACL rejects, Python accepts.

    Sub-plan §9.2: the Python validator currently treats empty-LO pages
    as pass (it walks `extract_lo_ids` and returns the page's allowed
    set without flagging length=0). The SHACL minCount=1 shape rejects
    them. This is a richer-diagnostics-for-free win per the parent
    plan; the test documents the asymmetry as expected, not as a bug.

    If this assertion ever flips (Python catches empty-LO too), the
    PoC has graduated and the SHACL gate severity can be raised to
    `critical` with confidence — update the assertions and remove
    this test, or replace with a strict-equivalence assertion.
    """
    content_dir = _build_minimal_corpus(tmp_path, with_los=False)

    py_result = PageObjectivesValidator().validate(
        {"content_dir": str(content_dir)}
    )
    py_critical = sum(1 for i in py_result.issues if i.severity == "critical")

    shacl_result = PageObjectivesShaclValidator().validate(
        {"content_dir": str(content_dir)}
    )
    shacl_critical = sum(1 for i in shacl_result.issues if i.severity == "critical")
    shacl_codes = {i.code for i in shacl_result.issues}

    # Documented asymmetry — see docstring.
    assert py_critical == 0, (
        "Python gate now catches empty-LO pages — PoC has graduated; "
        "update test_shacl_runner.py per sub-plan §9.2."
    )
    assert shacl_critical >= 1, (
        f"SHACL gate must fire on empty-LO page; got issues: {shacl_result.issues}"
    )
    assert "PO-001" in shacl_codes


def test_shacl_validator_handles_missing_content_dir():
    result = PageObjectivesShaclValidator().validate({})
    assert not result.passed
    assert any(i.code == "MISSING_CONTENT_DIR" for i in result.issues)


def test_shacl_validator_handles_nonexistent_content_dir(tmp_path):
    result = PageObjectivesShaclValidator().validate(
        {"content_dir": str(tmp_path / "nope")}
    )
    assert not result.passed
    assert any(i.code == "CONTENT_DIR_NOT_FOUND" for i in result.issues)


def test_shacl_validator_passes_on_empty_corpus(tmp_path):
    """No week_* pages -> nothing to validate -> passed=True with no issues."""
    (tmp_path / "project_docs").mkdir()
    result = PageObjectivesShaclValidator().validate(
        {"content_dir": str(tmp_path)}
    )
    assert result.passed
    assert result.issues == []


# --------------------------------------------------------------------- #
# §9.3 SHACL meta-validation
# --------------------------------------------------------------------- #


def test_shape_file_well_formed_nodeshape_structure(shape_path):
    """The shape file must declare at least one well-formed sh:NodeShape.

    Lightweight structural meta-check: every NodeShape carries
    rdfs:label + sh:targetClass + at least one sh:property. pyshacl's
    full meta-shape validation requires loading the SHACL spec
    file; the structural check here catches the most common typos
    (missing rdf:type sh:NodeShape, malformed property blank nodes)
    without that download.
    """
    g = Graph()
    g.parse(shape_path, format="turtle")
    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")

    nodeshapes = list(
        g.subjects(
            predicate=Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#").type,
            object=SHACL_NS.NodeShape,
        )
    )
    assert nodeshapes, "Shape file declares no sh:NodeShape — meta-fail."
    for ns in nodeshapes:
        labels = list(g.objects(ns, RDFS.label))
        assert labels, f"NodeShape {ns} has no rdfs:label."
        targets = list(g.objects(ns, SHACL_NS.targetClass))
        assert targets, f"NodeShape {ns} has no sh:targetClass."
        properties = list(g.objects(ns, SHACL_NS.property))
        assert properties, f"NodeShape {ns} declares no sh:property."
        for prop in properties:
            paths = list(g.objects(prop, SHACL_NS.path))
            assert paths, (
                f"PropertyShape on {ns} has no sh:path — malformed."
            )


def test_shape_file_passes_pyshacl_meta_validation(shape_path):
    """Full meta-validation: pyshacl validates the shape file against
    its internal SHACL Core meta-shape via meta_shacl=True.

    This is the strict version of the structural check above; if the
    shape file has any structural defect pyshacl recognizes, it fires
    here. Run it on a trivial empty data graph so we're meta-validating
    the shape, not the data.
    """
    shapes_g = Graph()
    shapes_g.parse(shape_path, format="turtle")
    data_g = Graph()  # empty data graph
    conforms, results_g, results_text = pyshacl.validate(
        data_graph=data_g,
        shacl_graph=shapes_g,
        inference="none",
        abort_on_first=False,
        meta_shacl=True,  # the meta-validation switch
        advanced=True,
        js=False,
        debug=False,
    )
    assert conforms, (
        f"Shape file failed pyshacl meta-validation:\n{results_text}"
    )

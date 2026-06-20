"""W10 Phase 2 (packaging portion) — assessment-surface packaging tests.

Exercises the ``package_multifile_imscc.py`` extension that emits QTI /
discussion / assignment XML found in a ``06_assessments/`` sibling of the
content dir as the canonical IMS CC resource types.

Asserted contracts (spec §4):

1. With a ``06_assessments/`` dir carrying a QTI + imsdt + assignment XML, the
   produced cartridge's manifest contains a ``<resource>`` with EACH of the
   three canonical ``type=`` strings, an organization ``<item identifierref>``
   references each, and the XML files are present in the produced zip.
2. ABSENT ``06_assessments/`` → manifest byte-identical to today (no
   regression).
3. A malformed XML in ``06_assessments/`` is DROPPED with a warning, and the
   package still builds (the well-formed siblings ship).
4. The optional ``06_assessments/manifest.json`` sidecar drives type / title /
   week / identifier classification, and ``packaging_report.json`` grows an
   ``assessment_coverage`` block.

All fixtures are built IN-TEST — no course slug, no disk dependency on a real
archived course. The valid XML fixtures are produced by the in-tree
``qti_emitter`` so they round-trip the parser + the vendored XSDs.
"""

import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from package_multifile_imscc import (  # noqa: E402
    _ASSESSMENT_RES_TYPE,
    build_manifest,
    package_imscc,
)
from qti_emitter import (  # noqa: E402
    assessment_to_qti,
    assignment_to_resource,
    discussion_to_imsdt,
)

# IMS CC 1.3 namespace used by the manifest (matches build_manifest).
_CC_NS = "http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"


# --------------------------------------------------------------------------- #
# Fixture XML builders (valid, via the in-tree emitter)
# --------------------------------------------------------------------------- #


def _qti_xml() -> str:
    return assessment_to_qti({
        "assessment_id": "ASMT_week3",
        "title": "Week 3 Quiz",
        "questions": [
            {
                "question_id": "Q1",
                "question_type": "multiple_choice",
                "stem": "What is 2 + 2?",
                "objective_id": "CO-01",
                "points": 1,
                "choices": [
                    {"id": "A", "text": "3"},
                    {"id": "B", "text": "4"},
                    {"id": "C", "text": "5"},
                ],
                "correct_answer": "B",
            },
            {
                "question_id": "Q2",
                "question_type": "true_false",
                "stem": "The sum is associative.",
                "objective_id": "CO-02",
                "points": 1,
                "correct_answer": "True",
            },
        ],
    })


def _discussion_xml() -> str:
    return discussion_to_imsdt({
        "title": "Week 3 Discussion",
        "objective_id": "TO-01",
        "text": "Discuss why associativity matters in algebra.",
    })


def _assignment_xml() -> str:
    return assignment_to_resource({
        "assignment_id": "ASN_week3",
        "title": "Week 3 Assignment",
        "objective_id": "CO-03",
        "text": "Solve five linear equations and show your work.",
        "points": 10,
    })


# --------------------------------------------------------------------------- #
# Content-dir fixtures
# --------------------------------------------------------------------------- #


def _make_week_pages(content_dir: Path) -> None:
    """A minimal two-week nested-layout content dir (no LO validation)."""
    (content_dir / "week_03").mkdir(parents=True, exist_ok=True)
    (content_dir / "week_03" / "week_03_overview.html").write_text(
        "<html><body><h1>Week 3 Overview</h1></body></html>", encoding="utf-8",
    )


@pytest.fixture
def content_dir_with_assessments(tmp_path):
    """Content dir + a 06_assessments/ sibling with one of each resource type."""
    _make_week_pages(tmp_path)
    adir = tmp_path / "06_assessments"
    adir.mkdir()
    (adir / "week_03_quiz.xml").write_text(_qti_xml(), encoding="utf-8")
    (adir / "week_03_discussion.xml").write_text(_discussion_xml(), encoding="utf-8")
    (adir / "week_03_assignment.xml").write_text(_assignment_xml(), encoding="utf-8")
    return tmp_path


@pytest.fixture
def content_dir_no_assessments(tmp_path):
    """Content dir with NO 06_assessments/ sibling (no-regression baseline)."""
    _make_week_pages(tmp_path)
    return tmp_path


@pytest.fixture
def sibling_layout_content_dir(tmp_path):
    """The REAL pipeline layout: content_dir == ``<export>/03_content_development``
    and the synthesized ``06_assessments/`` lives as a SIBLING at
    ``<export>/06_assessments`` (NOT a child of content_dir). Exercises the
    sibling-fallback discovery so the packager finds what
    ``_run_assessment_synthesis`` actually writes."""
    content_dir = tmp_path / "03_content_development"
    _make_week_pages(content_dir)
    adir = tmp_path / "06_assessments"
    adir.mkdir()
    (adir / "week_03_quiz.xml").write_text(_qti_xml(), encoding="utf-8")
    (adir / "week_03_discussion.xml").write_text(_discussion_xml(), encoding="utf-8")
    (adir / "week_03_assignment.xml").write_text(_assignment_xml(), encoding="utf-8")
    return content_dir


def _zip_names(output: Path):
    with zipfile.ZipFile(output) as zf:
        return zf.namelist()


def _manifest_str(output: Path) -> str:
    with zipfile.ZipFile(output) as zf:
        return zf.read("imsmanifest.xml").decode("utf-8")


def _resource_types(manifest_xml: str):
    root = ET.fromstring(manifest_xml)
    return [
        r.get("type")
        for r in root.iter(f"{{{_CC_NS}}}resource")
    ]


def _item_refs(manifest_xml: str):
    root = ET.fromstring(manifest_xml)
    return {
        i.get("identifierref")
        for i in root.iter(f"{{{_CC_NS}}}item")
        if i.get("identifierref")
    }


# --------------------------------------------------------------------------- #
# 1. Loose discovery → three resource types + org items + zip payload
# --------------------------------------------------------------------------- #


class TestAssessmentPackaging:

    def test_three_resource_types_emitted(
        self, content_dir_with_assessments, tmp_path,
    ):
        output = tmp_path / "out.imscc"
        package_imscc(
            content_dir_with_assessments, output, "TEST_101", "Test Course",
            skip_validation=True,
        )
        assert output.exists()
        manifest_xml = _manifest_str(output)
        types = _resource_types(manifest_xml)
        for kind, type_str in _ASSESSMENT_RES_TYPE.items():
            assert type_str in types, (
                f"expected a resource type={type_str} for {kind}; got {types}"
            )
        # Webcontent resources (week pages) still present.
        assert "webcontent" in types

    def test_sibling_layout_discovered(
        self, sibling_layout_content_dir, tmp_path,
    ):
        """Pipeline seam: content_dir=03_content_development, assessments at
        the <export>/06_assessments SIBLING — all three resource types must
        still land in the cartridge via the sibling-fallback discovery."""
        output = tmp_path / "out.imscc"
        package_imscc(
            sibling_layout_content_dir, output, "TEST_101", "Test Course",
            skip_validation=True,
        )
        assert output.exists()
        types = _resource_types(_manifest_str(output))
        for kind, type_str in _ASSESSMENT_RES_TYPE.items():
            assert type_str in types, (
                f"sibling layout: expected resource type={type_str} for "
                f"{kind}; got {types}"
            )

    def test_org_items_reference_each_assessment(
        self, content_dir_with_assessments, tmp_path,
    ):
        output = tmp_path / "out.imscc"
        package_imscc(
            content_dir_with_assessments, output, "TEST_101", "Test Course",
            skip_validation=True,
        )
        manifest_xml = _manifest_str(output)
        # Each assessment resource id must be referenced by an org <item>.
        root = ET.fromstring(manifest_xml)
        res_ids = {
            r.get("identifier")
            for r in root.iter(f"{{{_CC_NS}}}resource")
            if r.get("type") in _ASSESSMENT_RES_TYPE.values()
        }
        assert len(res_ids) == 3
        refs = _item_refs(manifest_xml)
        for rid in res_ids:
            assert rid in refs, f"resource {rid} has no org <item identifierref>"

    def test_xml_files_present_in_zip(
        self, content_dir_with_assessments, tmp_path,
    ):
        output = tmp_path / "out.imscc"
        package_imscc(
            content_dir_with_assessments, output, "TEST_101", "Test Course",
            skip_validation=True,
        )
        names = _zip_names(output)
        for fname in (
            "06_assessments/week_03_quiz.xml",
            "06_assessments/week_03_discussion.xml",
            "06_assessments/week_03_assignment.xml",
        ):
            assert fname in names, f"expected {fname} in zip; got {names}"

    def test_assessments_nested_under_week_when_inferable(
        self, content_dir_with_assessments, tmp_path,
    ):
        """A ``week_03_*`` file name infers week 3 → nested under WEEK_3 item."""
        output = tmp_path / "out.imscc"
        package_imscc(
            content_dir_with_assessments, output, "TEST_101", "Test Course",
            skip_validation=True,
        )
        root = ET.fromstring(_manifest_str(output))
        # Find the WEEK_3 item and confirm an assessment ref lives under it.
        week_item = None
        for item in root.iter(f"{{{_CC_NS}}}item"):
            if item.get("identifier") == "WEEK_3":
                week_item = item
                break
        assert week_item is not None
        child_refs = {
            c.get("identifierref")
            for c in week_item.iter(f"{{{_CC_NS}}}item")
            if c.get("identifierref")
        }
        # At least one assessment resource id is nested under WEEK_3.
        assert any("assessment" in (r or "") for r in child_refs), (
            f"no assessment item nested under WEEK_3; got {child_refs}"
        )


# --------------------------------------------------------------------------- #
# 2. No-regression: absent 06_assessments/ → byte-identical manifest
# --------------------------------------------------------------------------- #


class TestNoRegression:

    def test_absent_assessments_manifest_byte_identical(
        self, content_dir_no_assessments,
    ):
        """No 06_assessments/ dir → build_manifest output is unchanged whether
        or not the assessments arg is threaded (empty list == None)."""
        baseline = build_manifest(
            content_dir_no_assessments, "TEST_101", "Test Course",
        )
        with_arg = build_manifest(
            content_dir_no_assessments, "TEST_101", "Test Course",
            assessments=[],
        )
        assert baseline == with_arg
        # And no assessment resource types leak in.
        assert not any(
            t in _resource_types(baseline)
            for t in _ASSESSMENT_RES_TYPE.values()
        )

    def test_absent_assessments_zip_has_no_assessment_dir(
        self, content_dir_no_assessments, tmp_path,
    ):
        output = tmp_path / "out.imscc"
        package_imscc(
            content_dir_no_assessments, output, "TEST_101", "Test Course",
            skip_validation=True,
        )
        names = _zip_names(output)
        assert not any(n.startswith("06_assessments/") for n in names)


# --------------------------------------------------------------------------- #
# 3. Malformed XML dropped with a warning; package still builds
# --------------------------------------------------------------------------- #


class TestMalformedDropped:

    def test_malformed_xml_dropped_package_builds(
        self, content_dir_with_assessments, tmp_path, capsys,
    ):
        # Add a not-well-formed XML alongside the valid ones.
        adir = content_dir_with_assessments / "06_assessments"
        (adir / "broken.xml").write_text(
            "<questestinterop><assessment>UNCLOSED",
            encoding="utf-8",
        )
        output = tmp_path / "out.imscc"
        package_imscc(
            content_dir_with_assessments, output, "TEST_101", "Test Course",
            skip_validation=True,
        )
        assert output.exists(), "package must still build with one bad item"
        captured = capsys.readouterr().out
        assert "broken.xml" in captured
        # The malformed file is NOT in the zip; the three valid ones are.
        names = _zip_names(output)
        assert "06_assessments/broken.xml" not in names
        assert "06_assessments/week_03_quiz.xml" in names

    def test_schema_invalid_xml_dropped(
        self, content_dir_with_assessments, tmp_path, capsys,
    ):
        """A well-formed but XSD-invalid QTI is dropped (best-effort, when lxml
        is available); package still builds. Skips gracefully when lxml absent."""
        pytest.importorskip("lxml")
        adir = content_dir_with_assessments / "06_assessments"
        (adir / "bad_schema.xml").write_text(
            '<?xml version="1.0"?>'
            '<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2">'
            "<bogusElement/></questestinterop>",
            encoding="utf-8",
        )
        output = tmp_path / "out.imscc"
        package_imscc(
            content_dir_with_assessments, output, "TEST_101", "Test Course",
            skip_validation=True,
        )
        assert output.exists()
        captured = capsys.readouterr().out
        assert "bad_schema.xml" in captured
        assert "XSD validation" in captured
        names = _zip_names(output)
        assert "06_assessments/bad_schema.xml" not in names
        # The three valid ones still ship.
        assert "06_assessments/week_03_quiz.xml" in names

    def test_unclassifiable_root_skipped(
        self, content_dir_with_assessments, tmp_path, capsys,
    ):
        """A well-formed XML whose root is none of the known types is skipped."""
        adir = content_dir_with_assessments / "06_assessments"
        (adir / "stray.xml").write_text(
            "<somethingElse><child/></somethingElse>", encoding="utf-8",
        )
        output = tmp_path / "out.imscc"
        package_imscc(
            content_dir_with_assessments, output, "TEST_101", "Test Course",
            skip_validation=True,
        )
        captured = capsys.readouterr().out
        assert "stray.xml" in captured
        names = _zip_names(output)
        assert "06_assessments/stray.xml" not in names


# --------------------------------------------------------------------------- #
# 4. Sidecar-driven classification + coverage sidecar
# --------------------------------------------------------------------------- #


class TestSidecarAndCoverage:

    def test_sidecar_overrides_type_and_week_and_title(
        self, tmp_path,
    ):
        """The manifest.json sidecar drives type / title / week / identifier."""
        _make_week_pages(tmp_path)
        adir = tmp_path / "06_assessments"
        adir.mkdir()
        # File name carries no week token; sidecar supplies week + type.
        (adir / "graded_thing.xml").write_text(_qti_xml(), encoding="utf-8")
        (adir / "manifest.json").write_text(
            json.dumps({
                "schema_version": "v1",
                "assessments": [
                    {
                        "file": "graded_thing.xml",
                        "type": "qti",
                        "title": "Custom Quiz Title",
                        "week": 3,
                        "identifier": "RES_custom_quiz",
                    },
                ],
            }),
            encoding="utf-8",
        )
        output = tmp_path / "out.imscc"
        package_imscc(
            tmp_path, output, "TEST_101", "Test Course", skip_validation=True,
        )
        manifest_xml = _manifest_str(output)
        root = ET.fromstring(manifest_xml)
        # Resource carries the sidecar identifier + qti type.
        qti_res = [
            r for r in root.iter(f"{{{_CC_NS}}}resource")
            if r.get("type") == _ASSESSMENT_RES_TYPE["qti"]
        ]
        assert len(qti_res) == 1
        assert qti_res[0].get("identifier") == "RES_custom_quiz"
        # The custom title is on the org item.
        titles = [
            t.text for t in root.iter(f"{{{_CC_NS}}}title")
        ]
        assert "Custom Quiz Title" in titles
        # Nested under WEEK_3 (sidecar-supplied week).
        week_item = next(
            (i for i in root.iter(f"{{{_CC_NS}}}item")
             if i.get("identifier") == "WEEK_3"),
            None,
        )
        assert week_item is not None
        nested = {
            c.get("identifierref")
            for c in week_item.iter(f"{{{_CC_NS}}}item")
            if c.get("identifierref")
        }
        assert "RES_custom_quiz" in nested

    def test_unmapped_week_lands_under_top_level_assessments(
        self, tmp_path,
    ):
        """No week token + no sidecar week → top-level Assessments org item."""
        _make_week_pages(tmp_path)
        adir = tmp_path / "06_assessments"
        adir.mkdir()
        (adir / "standalone_discussion.xml").write_text(
            _discussion_xml(), encoding="utf-8",
        )
        output = tmp_path / "out.imscc"
        package_imscc(
            tmp_path, output, "TEST_101", "Test Course", skip_validation=True,
        )
        root = ET.fromstring(_manifest_str(output))
        assessments_item = next(
            (i for i in root.iter(f"{{{_CC_NS}}}item")
             if i.get("identifier") == "ASSESSMENTS"),
            None,
        )
        assert assessments_item is not None, "expected a top-level Assessments item"
        nested = {
            c.get("identifierref")
            for c in assessments_item.iter(f"{{{_CC_NS}}}item")
            if c.get("identifierref")
        }
        assert len(nested) == 1

    def test_coverage_sidecar_has_assessment_coverage_block(
        self, content_dir_with_assessments, tmp_path,
    ):
        output = tmp_path / "out.imscc"
        report_path = tmp_path / "packaging_report.json"
        package_imscc(
            content_dir_with_assessments, output, "TEST_101", "Test Course",
            skip_validation=True,
            coverage_sidecar_path=report_path,
        )
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "assessment_coverage" in report
        cov = report["assessment_coverage"]
        assert cov["assessments_authored"] == 3
        assert cov["assessments_packaged"] == 3
        assert cov["assessments_dropped"] == 0
        assert cov["by_type"] == {
            "qti": 1, "discussion": 1, "assignment": 1,
        }

    def test_coverage_block_absent_when_no_assessments(
        self, content_dir_no_assessments, tmp_path,
    ):
        output = tmp_path / "out.imscc"
        report_path = tmp_path / "packaging_report.json"
        package_imscc(
            content_dir_no_assessments, output, "TEST_101", "Test Course",
            skip_validation=True,
            coverage_sidecar_path=report_path,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "assessment_coverage" not in report

    def test_malformed_drop_reflected_in_coverage(
        self, content_dir_with_assessments, tmp_path,
    ):
        adir = content_dir_with_assessments / "06_assessments"
        (adir / "broken.xml").write_text("<topic>UNCLOSED", encoding="utf-8")
        output = tmp_path / "out.imscc"
        report_path = tmp_path / "packaging_report.json"
        package_imscc(
            content_dir_with_assessments, output, "TEST_101", "Test Course",
            skip_validation=True,
            coverage_sidecar_path=report_path,
        )
        cov = json.loads(report_path.read_text(encoding="utf-8"))["assessment_coverage"]
        assert cov["assessments_authored"] == 4
        assert cov["assessments_packaged"] == 3
        assert cov["assessments_dropped"] == 1

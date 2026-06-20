"""W10 Phase 6 — end-to-end assessment-surface seam test.

Spec: ``plans/finegrain/w10-assessments-qti-discussions-2026-06.md`` §11 Phase 6.

Proves the DETERMINISTIC assessment data path connects end to end — the seam
that matters once the (LLM-driven, not-CI-safe) ``_run_assessment_synthesis``
phase has produced its output:

    qti_emitter  →  package_multifile_imscc  →  quiz_service

The synthesis step itself (AssessmentGenerator + prose provider) is unit-tested
with fixture data in ``MCP/tests/test_assessment_synthesis.py``; this test does
NOT re-exercise it. Instead it builds fixture assessment data IN-TEST (standing
in for synthesis output), emits the canonical IMS CC resource XML via the REAL
emitter, runs the REAL packager, then reads the packaged QTI back through the
REAL quiz_service and grades it.

Hermetic: no LLM, no GPU, no course slug, no on-disk corpus — everything lives
in a tmp dir. Each landed component is imported + stitched, never modified.
"""

from __future__ import annotations

import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The packager adds its own scripts dir to sys.path at import time (for its
# sibling render_learning_objectives_page / validate_page_objectives imports).
from Courseforge.scripts import qti_emitter
from Courseforge.scripts.package_multifile_imscc import package_imscc
from gui.services import quiz_service


# IMS CC 1.3 resource-type strings the packager emits (mirrors
# package_multifile_imscc._ASSESSMENT_RES_TYPE — asserted, not imported, so a
# drift in the packager surfaces here).
_QTI_RES_TYPE = "imsqti_xmlv1p2/imscc_xmlv1p3/assessment"
_IMSDT_RES_TYPE = "imsdt_xmlv1p3"
_ASSIGNMENT_RES_TYPE = "associatedcontent/imscc_xmlv1p3/learning-application-resource"

_IMSCP_NS = "http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"


# --------------------------------------------------------------------------- #
# Fixture assessment data (built in-test — no course slug, no disk corpus)
# --------------------------------------------------------------------------- #


def _assessment_data() -> dict:
    """One QTI assessment: a multiple_choice + a numeric fill_in_blank item.

    Mirrors the ``AssessmentData.to_dict()`` shape the emitter consumes
    (assessment_id / title / questions[] of QuestionData dicts).
    """
    return {
        "assessment_id": "week_03_quiz",
        "title": "Week 3 Quiz",
        "course_code": "PHYS_101",
        "questions": [
            {
                "question_id": "q_mc_01",
                "question_type": "multiple_choice",
                "stem": "Which SI unit measures force?",
                "bloom_level": "remember",
                "objective_id": "CO-03",
                "choices": [
                    {"id": "A", "text": "Newton"},
                    {"id": "B", "text": "Joule"},
                    {"id": "C", "text": "Watt"},
                    {"id": "D", "text": "Pascal"},
                ],
                "correct_answer": "A",
                "points": 1.0,
            },
            {
                "question_id": "q_fib_01",
                "question_type": "fill_in_blank",
                "stem": "What fraction of a turn is 180 degrees? (decimal)",
                "bloom_level": "understand",
                "objective_id": "CO-03",
                "choices": [],
                "correct_answer": "0.5",
                "points": 1.0,
            },
        ],
    }


def _discussion_data() -> dict:
    return {
        "title": "Week 3 Discussion",
        "objective_id": "TO-01",
        "text": "Discuss a real-world situation where measuring force matters.",
    }


def _assignment_data() -> dict:
    return {
        "assignment_id": "week_03_assignment",
        "title": "Week 3 Assignment",
        "objective_id": "CO-03",
        "text": "Compute the net force on a 2 kg object accelerating at 3 m/s^2.",
        "points": 10.0,
    }


def _build_content_dir(root: Path) -> Path:
    """Build a minimal content dir + a 06_assessments/ sibling with emitted XML.

    Returns the content dir (``03_content_development``) to hand the packager.
    """
    content_dir = root / "03_content_development"
    week_dir = content_dir / "week_03"
    week_dir.mkdir(parents=True)
    # Minimal week overview page (full-mode packaging needs ≥1 real page).
    (week_dir / "week_03_overview.html").write_text(
        "<!DOCTYPE html><html><head><title>Week 3</title></head>"
        "<body><h1>Week 3 Overview: Forces</h1>"
        "<p>An introduction to forces and units.</p></body></html>",
        encoding="utf-8",
    )

    # Emit the three assessment resource XMLs via the REAL emitter into the
    # 06_assessments/ sibling (the convention the packager's _discover_assessments
    # honours: sibling at content_dir.parent OR child of content_dir).
    assessments_dir = root / "06_assessments"
    assessments_dir.mkdir()

    qti_xml = qti_emitter.assessment_to_qti(_assessment_data())
    (assessments_dir / "week_03_quiz.xml").write_text(qti_xml, encoding="utf-8")

    imsdt_xml = qti_emitter.discussion_to_imsdt(_discussion_data())
    (assessments_dir / "week_03_discussion.xml").write_text(imsdt_xml, encoding="utf-8")

    assignment_xml = qti_emitter.assignment_to_resource(_assignment_data())
    (assessments_dir / "week_03_assignment.xml").write_text(
        assignment_xml, encoding="utf-8"
    )

    # manifest.json sidecar per the §4 contract.
    import json

    sidecar = {
        "schema_version": "v1",
        "assessments": [
            {
                "file": "week_03_quiz.xml",
                "type": "qti",
                "title": "Week 3 Quiz",
                "week": 3,
                "identifier": "RES_week_03_quiz",
            },
            {
                "file": "week_03_discussion.xml",
                "type": "discussion",
                "title": "Week 3 Discussion",
                "week": 3,
                "identifier": "RES_week_03_discussion",
            },
            {
                "file": "week_03_assignment.xml",
                "type": "assignment",
                "title": "Week 3 Assignment",
                "week": 3,
                "identifier": "RES_week_03_assignment",
            },
        ],
    }
    (assessments_dir / "manifest.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    return content_dir


# --------------------------------------------------------------------------- #
# The e2e
# --------------------------------------------------------------------------- #


class TestW10AssessmentE2E:
    """emitter → packager → quiz_service, proven on a hermetic fixture."""

    @pytest.fixture(scope="class")
    def packaged(self, tmp_path_factory) -> dict:
        """Build + package once; the assertions share the produced cartridge."""
        root = tmp_path_factory.mktemp("w10_e2e")
        content_dir = _build_content_dir(root)
        output = root / "PHYS_101.imscc"

        # The REAL packager. skip_validation=True keeps the test off the
        # per-week LO contract (not what this seam exercises).
        package_imscc(
            content_dir,
            output,
            "PHYS_101",
            "Physics 101",
            skip_validation=True,
            emit_objectives_page=False,
        )
        assert output.exists(), "packager did not produce the .imscc"
        return {"imscc": output}

    # -- Step 3: manifest carries the canonical resource types ----------- #

    def test_manifest_carries_assessment_resource_types(self, packaged):
        with zipfile.ZipFile(packaged["imscc"]) as zf:
            names = set(zf.namelist())
            manifest_xml = zf.read("imsmanifest.xml").decode("utf-8")

        # The three XML payloads are inside the zip under 06_assessments/.
        assert "06_assessments/week_03_quiz.xml" in names
        assert "06_assessments/week_03_discussion.xml" in names
        assert "06_assessments/week_03_assignment.xml" in names

        root = ET.fromstring(manifest_xml)
        res_types = {
            r.get("type")
            for r in root.iter(f"{{{_IMSCP_NS}}}resource")
        }
        assert _QTI_RES_TYPE in res_types, (
            f"QTI resource type missing; saw {res_types}"
        )
        assert _IMSDT_RES_TYPE in res_types, (
            f"discussion (imsdt) resource type missing; saw {res_types}"
        )
        assert _ASSIGNMENT_RES_TYPE in res_types, (
            f"assignment resource type missing; saw {res_types}"
        )

        # The QTI resource's href points at the packaged XML.
        qti_hrefs = {
            r.get("href")
            for r in root.iter(f"{{{_IMSCP_NS}}}resource")
            if r.get("type") == _QTI_RES_TYPE
        }
        assert "06_assessments/week_03_quiz.xml" in qti_hrefs

    # -- Step 4: learner JSON leaks no answer key ------------------------ #

    def test_learner_json_omits_answer_key(self, packaged):
        with zipfile.ZipFile(packaged["imscc"]) as zf:
            qti_bytes = zf.read("06_assessments/week_03_quiz.xml")

        assessment = quiz_service.parse_quiz(qti_bytes)
        learner = quiz_service.to_learner_json(assessment)

        assert learner["assessment_id"] == "week_03_quiz"
        assert learner["title"] == "Week 3 Quiz"
        # Two questions surface to the learner.
        assert len(learner["questions"]) == 2

        # Anti-leak: no correct_answer / varequal / is_correct anywhere in the
        # learner payload. Check structurally AND by serialized scan.
        import json

        blob = json.dumps(learner).lower()
        assert "varequal" not in blob
        assert "correct_answer" not in blob
        assert "correct_response" not in blob
        assert "is_correct" not in blob

        for q in learner["questions"]:
            assert set(q.keys()) == {"question_id", "stem", "type", "choices"}
            for choice in q["choices"]:
                assert set(choice.keys()) == {"id", "text"}

    # -- Step 5: server-side grading incl. numeric equivalence ----------- #

    def test_grade_full_score_on_correct_answers(self, packaged):
        with zipfile.ZipFile(packaged["imscc"]) as zf:
            qti_text = zf.read("06_assessments/week_03_quiz.xml").decode("utf-8")

        assessment = quiz_service.parse_quiz(qti_text)
        # Submit the exact correct answers: MC choice id "A", numeric "0.5".
        result = quiz_service.grade_submission(
            assessment,
            {"q_mc_01": "A", "q_fib_01": "0.5"},
            quiz_xml=qti_text,
        )
        assert result["max_score"] == 2
        assert result["score"] == 2, f"expected full score, got {result}"
        per = {item["question_id"]: item["correct"] for item in result["per_item"]}
        assert per["q_mc_01"] is True
        assert per["q_fib_01"] is True

    def test_grade_numeric_fraction_equivalence(self, packaged):
        """'1/2' must grade correct against the '0.5' key (Fraction equivalence).

        This is the "gradable in the learner UI" proof: the Studio grader is
        smarter than QTI exact-match (spec §9.1).
        """
        with zipfile.ZipFile(packaged["imscc"]) as zf:
            qti_text = zf.read("06_assessments/week_03_quiz.xml").decode("utf-8")

        assessment = quiz_service.parse_quiz(qti_text)
        result = quiz_service.grade_submission(
            assessment,
            {"q_mc_01": "A", "q_fib_01": "1/2"},  # equivalent form of 0.5
            quiz_xml=qti_text,
        )
        per = {item["question_id"]: item["correct"] for item in result["per_item"]}
        assert per["q_fib_01"] is True, (
            f"'1/2' should grade correct vs key '0.5'; got {result}"
        )
        assert result["score"] == 2

    def test_grade_wrong_answer_scores_zero_for_that_item(self, packaged):
        with zipfile.ZipFile(packaged["imscc"]) as zf:
            qti_text = zf.read("06_assessments/week_03_quiz.xml").decode("utf-8")

        assessment = quiz_service.parse_quiz(qti_text)
        result = quiz_service.grade_submission(
            assessment,
            {"q_mc_01": "B", "q_fib_01": "0.25"},  # both wrong
            quiz_xml=qti_text,
        )
        per = {item["question_id"]: item["correct"] for item in result["per_item"]}
        assert per["q_mc_01"] is False
        assert per["q_fib_01"] is False
        assert result["score"] == 0

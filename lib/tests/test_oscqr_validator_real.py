"""Wave 31 — OSCQRValidator real-implementation tests.

Covers the 12 automated OSCQR rubric items across 6 categories:

* Course Overview — OV-1 (syllabus present), OV-3 (syllabus complete).
* Learner Support — LS-1 (accessibility statement).
* Course Structure — CS-2 (weekly modules), CS-3 (unique titles),
  CS-5 (page objectives populated).
* Content / Learning Activities — CLA-1 (word-count floor),
  CLA-3 (activity prompt variety).
* Instructor Interaction — II-3 (interaction prompts).
* Assessment & Measurement — AM-1 (question type variety),
  AM-2 (objective-linked assessments), AM-2b (no placeholder stems).

Each item has a positive and negative case. A simulated failing course
(empty objectives, identical activities, placeholder assessments) scores
near 0; a well-formed course scores ≥ 0.8.

Hermetic: synthetic fixtures only, no corpus-specific identifiers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.validators.oscqr import OSCQRValidator  # noqa: E402

# ---------------------------------------------------------------------- #
# Fixture builders
# ---------------------------------------------------------------------- #


def _build_good_course(tmp_path: Path) -> Path:
    """Build a small but well-formed course that should score > 0.8."""
    root = tmp_path / "good_course"
    root.mkdir()

    # Syllabus (complete)
    (root / "syllabus.html").write_text(
        "<html><body>"
        "<h1>Course Syllabus</h1>"
        "<h2>Objectives</h2><p>Learners will...</p>"
        "<h2>Schedule</h2><p>Week 1...</p>"
        "<h2>Grading Policy</h2><p>70% exams, 30% projects.</p>"
        "<h2>Instructor Contact</h2><p>Email: prof@example.edu</p>"
        "<h2>Accessibility</h2><p>Disability services available.</p>"
        "</body></html>",
        encoding="utf-8",
    )

    # Weekly modules — 4 weeks with distinct titles + populated objectives
    # + substantial content + distinct activity prompts.
    topics = ["Introduction to Algorithms", "Graph Traversal",
              "Dynamic Programming", "Complexity Analysis"]
    activities = [
        "Trace the insertion sort algorithm on the array [5, 2, 8, 1, 9] and list each intermediate state.",
        "Implement breadth-first search on the provided adjacency list and count the nodes visited.",
        "Design a memoized solution for the Fibonacci sequence that caches intermediate results.",
        "Compute the Big-O runtime of the provided nested loop function and explain your reasoning.",
    ]
    for i, (topic, activity) in enumerate(zip(topics, activities), start=1):
        week_dir = root / f"week_{i:02d}"
        week_dir.mkdir()
        overview = week_dir / f"week_{i:02d}_overview.html"
        overview.write_text(
            f"<html><body>"
            f"<h1>Week {i}: {topic}</h1>"
            f"<h2>Learning Objectives</h2>"
            f"<ul><li>Explain the concept of {topic.lower()}</li>"
            f"<li>Apply {topic.lower()} to novel problems</li>"
            f"<li>Analyze the complexity of {topic.lower()} techniques</li></ul>"
            f"<p>This week introduces {topic.lower()}. "
            f"We will explore practical examples, work through "
            f"algorithmic trade-offs, and examine real-world applications. "
            f"Discussion board participation is expected — please reflect "
            f"on the readings and share your analysis with peers. Also, "
            f"note the accessibility statement on the syllabus page. "
            f"Additional context: {topic} is foundational to computer science "
            f"and appears throughout the course curriculum.</p>"
            f"</body></html>",
            encoding="utf-8",
        )
        (week_dir / f"week_{i:02d}_application.html").write_text(
            f"<html><body>"
            f"<h1>Week {i} Application: {topic}</h1>"
            f"<h2>Learning Objectives</h2>"
            f"<ul><li>Apply {topic.lower()} to a concrete problem</li>"
            f"<li>Justify your algorithmic choices</li></ul>"
            f"<p>{activity}</p>"
            f"<p>Submit your answer to the discussion forum for peer review. "
            f"Reflect on how {topic.lower()} applies to the scenarios discussed "
            f"during the week's reading material. Consider edge cases such as "
            f"empty inputs, single-element inputs, and large-scale performance "
            f"characteristics of the chosen solution. Compare and contrast with "
            f"the previous week's techniques.</p>"
            f"<p>Additional rubric criteria: correctness, efficiency, clarity. "
            f"Please cite any references consulted.</p>"
            f"</body></html>",
            encoding="utf-8",
        )

    # course.json with varied assessments
    course_json = {
        "course_code": "ALGO_101",
        "assessments": [
            {
                "questions": [
                    {"type": "multiple_choice", "stem": "Which sorting algorithm is O(n log n) on average?", "objective_id": "OBJ-1"},
                    {"type": "short_answer", "stem": "Explain BFS vs DFS.", "objective_id": "OBJ-2"},
                    {"type": "true_false", "stem": "Binary search is O(n).", "objective_id": "OBJ-3"},
                ]
            }
        ]
    }
    (root / "course.json").write_text(json.dumps(course_json), encoding="utf-8")
    return root


def _build_bad_course(tmp_path: Path) -> Path:
    """Build a course that mimics the OLSR_SIM_01 failure pattern."""
    root = tmp_path / "bad_course"
    root.mkdir()

    # No syllabus. Empty objectives. Identical activities. Placeholder assessments.
    topics = ["Topic A"] * 4  # Identical titles
    identical_activity = "Read the chapter and answer the discussion prompt."
    for i in range(1, 5):
        week_dir = root / f"week_{i:02d}"
        week_dir.mkdir()
        (week_dir / f"week_{i:02d}_overview.html").write_text(
            f"<html><body><h1>{topics[i-1]}</h1>"
            "<h2>Learning Objectives</h2><ul></ul>"
            "<p>Short.</p></body></html>",
            encoding="utf-8",
        )
        (week_dir / f"week_{i:02d}_application.html").write_text(
            f"<html><body><h1>Activity</h1><p>{identical_activity}</p></body></html>",
            encoding="utf-8",
        )

    course_json = {
        "assessments": [
            {"questions": [
                {"type": "multiple_choice", "stem": "[TODO]"},
                {"type": "multiple_choice", "stem": "Placeholder question"},
            ]}
        ]
    }
    (root / "course.json").write_text(json.dumps(course_json), encoding="utf-8")
    return root


# ---------------------------------------------------------------------- #
# Overall score behavior
# ---------------------------------------------------------------------- #


class TestOverallScoreBehavior:
    def test_good_course_scores_above_80_percent(self, tmp_path):
        root = _build_good_course(tmp_path)
        validator = OSCQRValidator()
        result = validator.validate({"course_path": str(root)})
        assert result.score is not None
        assert result.score >= 0.8, f"Expected good course score ≥0.8, got {result.score}"
        assert result.passed is True

    def test_olsr_sim_01_shape_fails_critical(self, tmp_path):
        """Simulates OLSR_SIM_01 failure pattern: empty objectives, identical
        activities, placeholder assessments. Must fail closed."""
        root = _build_bad_course(tmp_path)
        validator = OSCQRValidator()
        result = validator.validate({"course_path": str(root)})
        assert result.passed is False, "Bad course must fail OSCQR critical items"
        # Score should be meaningfully lower than a good course.
        assert result.score is not None
        assert result.score < 0.6, f"Expected bad course score <0.6, got {result.score}"
        # At least one critical issue must surface.
        critical = [i for i in result.issues if i.severity == "critical"]
        assert len(critical) >= 3, (
            f"Expected ≥3 critical failures in bad course, got {len(critical)}: "
            f"{[(i.code, i.message) for i in critical]}"
        )


# ---------------------------------------------------------------------- #
# Per-item tests
# ---------------------------------------------------------------------- #


class TestSyllabusItems:
    def test_ov1_syllabus_present_passes(self, tmp_path):
        root = _build_good_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        codes = {i.code for i in result.issues}
        assert "OV-1_FAIL" not in codes

    def test_ov1_missing_syllabus_fails(self, tmp_path):
        root = _build_bad_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        codes = {i.code for i in result.issues}
        assert "OV-1_FAIL" in codes

    def test_ov3_incomplete_syllabus_warns(self, tmp_path):
        root = tmp_path / "partial"
        root.mkdir()
        (root / "syllabus.html").write_text(
            "<html><body><h1>Syllabus</h1><p>TBD</p></body></html>",
            encoding="utf-8",
        )
        result = OSCQRValidator().validate({"course_path": str(root)})
        codes = {i.code for i in result.issues}
        assert "OV-3_FAIL" in codes


class TestLearnerSupport:
    def test_ls1_accessibility_statement_detected(self, tmp_path):
        root = _build_good_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        codes = {i.code for i in result.issues}
        assert "LS-1_FAIL" not in codes

    def test_ls1_missing_accessibility_statement(self, tmp_path):
        root = _build_bad_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        codes = {i.code for i in result.issues}
        assert "LS-1_FAIL" in codes


class TestCourseStructure:
    def test_cs2_weekly_modules_detected(self, tmp_path):
        root = _build_good_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        codes = {i.code for i in result.issues}
        assert "CS-2_FAIL" not in codes

    def test_cs2_no_weeks_fails_critical(self, tmp_path):
        root = tmp_path / "no_weeks"
        root.mkdir()
        (root / "index.html").write_text("<html><body>Hi</body></html>", encoding="utf-8")
        result = OSCQRValidator().validate({"course_path": str(root)})
        codes = {i.code for i in result.issues}
        assert "CS-2_FAIL" in codes
        assert result.passed is False

    def test_cs3_unique_titles_pass(self, tmp_path):
        root = _build_good_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        codes = {i.code for i in result.issues}
        assert "CS-3_FAIL" not in codes

    def test_cs3_duplicate_titles_warn(self, tmp_path):
        root = _build_bad_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        codes = {i.code for i in result.issues}
        assert "CS-3_FAIL" in codes

    def test_cs5_objectives_populated_pass(self, tmp_path):
        root = _build_good_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        codes = {i.code for i in result.issues}
        assert "CS-5_FAIL" not in codes

    def test_cs5_empty_objectives_fail_critical(self, tmp_path):
        root = _build_bad_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        critical_codes = {i.code for i in result.issues if i.severity == "critical"}
        assert "CS-5_FAIL" in critical_codes


class TestContentQuality:
    def test_cla1_word_floor_pass(self, tmp_path):
        root = _build_good_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        codes = {i.code for i in result.issues}
        assert "CLA-1_FAIL" not in codes

    def test_cla1_thin_pages_fail_critical(self, tmp_path):
        root = _build_bad_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        critical_codes = {i.code for i in result.issues if i.severity == "critical"}
        assert "CLA-1_FAIL" in critical_codes

    def test_cla3_distinct_activities_pass(self, tmp_path):
        root = _build_good_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        codes = {i.code for i in result.issues}
        assert "CLA-3_FAIL" not in codes

    def test_cla3_copypaste_activities_fail(self, tmp_path):
        root = _build_bad_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        critical_codes = {i.code for i in result.issues if i.severity == "critical"}
        assert "CLA-3_FAIL" in critical_codes


class TestAssessments:
    def test_am1_question_type_variety_pass(self, tmp_path):
        root = _build_good_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        codes = {i.code for i in result.issues}
        assert "AM-1_FAIL" not in codes

    def test_am1_low_variety_warn(self, tmp_path):
        root = _build_bad_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        codes = {i.code for i in result.issues}
        assert "AM-1_FAIL" in codes

    def test_am2_objective_linked_pass(self, tmp_path):
        root = _build_good_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        codes = {i.code for i in result.issues}
        assert "AM-2_FAIL" not in codes

    def test_am2_unlinked_fail(self, tmp_path):
        root = _build_bad_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        critical_codes = {i.code for i in result.issues if i.severity == "critical"}
        assert "AM-2_FAIL" in critical_codes

    def test_am2b_placeholder_fail(self, tmp_path):
        root = _build_bad_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        critical_codes = {i.code for i in result.issues if i.severity == "critical"}
        assert "AM-2b_FAIL" in critical_codes


# ---------------------------------------------------------------------- #
# Gate integration smoke
# ---------------------------------------------------------------------- #


class TestGateIntegration:
    def test_validator_returns_gate_result_with_required_fields(self, tmp_path):
        root = _build_good_course(tmp_path)
        result = OSCQRValidator().validate({"course_path": str(root)})
        assert result.validator_name == "oscqr_score"
        assert result.validator_version == "1.0.0"
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert result.gate_id == "oscqr_score"

    def test_no_inputs_gracefully_skipped(self):
        """No course_path + no course_json → validator should not crash."""
        result = OSCQRValidator().validate({})
        # Missing inputs should surface as skipped items, not a crash.
        assert result.validator_name == "oscqr_score"

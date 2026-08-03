"""Instructor answer-key emitter + packaging-inclusion tests (Phase 2 / C.2).

Exercises ``Courseforge/scripts/packaging/answer_key_emitter.py`` (the deterministic,
LLM-free renderer) and the ``package_multifile_imscc.py`` change that ships the
rendered ``answer_key.html`` as a CC ``webcontent`` resource.

Asserted contracts:

1. Every ``item_subtype`` the diversified tier emits (mc_single,
   mc_multiple_response, tf, fib_numeric, error_analysis, matching_mc,
   ordering_mc, two_tier_answer, two_tier_reason) reshapes into a well-formed
   answer-key record.
2. Correct-answer derivation precedence: plural ``correct_answers`` →
   scalar ``correct_answer`` → the ``is_correct`` choice texts.
3. Worked-solution steps carry the item's stamped ``source_chunks`` (grounding
   contract) and split a ``<li>`` feedback body into one step per item.
4. Per-distractor rationale surfaces the misconception note, and an ABSENT
   note renders as an honest absence — never a fabricated value.
5. Numeric-FIB rows stamp the LMS-tolerance instructor note.
6. Essay / discussion / assignment rubric scaffolds are deterministic, keyed
   off the objective's Bloom level, with the criterion row citing the objective
   text; rubrics live ONLY in the answer key.
7. Absent fields render as explicitly-absent, never fabricated.
8. The renderer accepts BOTH ``QuestionData`` dataclass instances and their
   ``to_dict()`` dict form.
9. The HTML meets the WCAG posture: semantic headings, tables with ``scope``,
   no color-only signaling, escaped content.
10. ``emit_answer_key`` writes both files and returns the manifest row.
11. The packager ships ``answer_key.html`` as a ``webcontent`` resource in the
    cartridge (manifest resource + org item + zip payload).

All fixtures are built IN-TEST — no course slug, no model, no network.
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

from answer_key_emitter import (  # noqa: E402
    answer_key_manifest_entry,
    build_answer_key,
    emit_answer_key,
    render_answer_key_html,
)
from package_multifile_imscc import (  # noqa: E402
    _ASSESSMENT_RES_TYPE,
    _assessment_res_type_for_kind,
    _classify_assessment,
    package_imscc,
)

_CC_NS = "http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"


# --------------------------------------------------------------------------- #
# Fixture question builders (dict / to_dict() form)
# --------------------------------------------------------------------------- #
def _q_mc_single():
    return {
        "question_id": "q_mc", "question_type": "multiple_choice",
        "stem": "<p>Which definition matches X?</p>", "bloom_level": "remember",
        "objective_id": "CO-01", "item_subtype": "mc_single",
        "choices": [
            {"text": "<p>the right def</p>", "is_correct": True},
            {"text": "<p>wrong def</p>", "is_correct": False,
             "misconception_note": "confuses X with Y"},
            {"text": "<p>other def</p>", "is_correct": False},
        ],
        "feedback": "<p>X is the right def.</p>", "source_chunks": ["c1"],
    }


def _q_mr():
    return {
        "question_id": "q_mr", "question_type": "multiple_response",
        "stem": "<p>Select all steps.</p>", "bloom_level": "apply",
        "objective_id": "CO-01", "item_subtype": "mc_multiple_response",
        "choices": [
            {"text": "<p>step 1</p>", "is_correct": True},
            {"text": "<p>step 2</p>", "is_correct": True},
            {"text": "<p>bad step</p>", "is_correct": False,
             "misconception_note": "add-exponents-on-multiply"},
        ],
        "correct_answers": ["step 1", "step 2"],
        "feedback": "<p>The correct steps are drawn from the worked solution.</p>",
        "source_chunks": ["c2"],
    }


def _q_tf():
    return {
        "question_id": "q_tf", "question_type": "true_false",
        "stem": "<p>The claim is true.</p>", "bloom_level": "remember",
        "objective_id": "CO-02", "item_subtype": "tf",
        "choices": [
            {"text": "True", "is_correct": True},
            {"text": "False", "is_correct": False},
        ],
        "feedback": "<p>This is correct.</p>", "source_chunks": ["c3"],
    }


def _q_fib_numeric():
    return {
        "question_id": "q_num", "question_type": "fill_in_blank",
        "stem": "<p>Solve for x: 2x = 10.</p>", "bloom_level": "apply",
        "objective_id": "CO-02", "item_subtype": "fib_numeric",
        "correct_answer": "5",
        "feedback": "<p>x = 5 (verified by substitution).</p>",
        "source_chunks": ["c4"],
    }


def _q_error_analysis():
    return {
        "question_id": "q_err", "question_type": "multiple_choice",
        "stem": "<p>Which step is wrong?</p>", "bloom_level": "analyze",
        "objective_id": "CO-03", "item_subtype": "error_analysis",
        "choices": [
            {"text": "<p>Step 2</p>", "is_correct": True,
             "misconception_note": "sign-drop"},
            {"text": "<p>Step 1</p>", "is_correct": False},
            {"text": "<p>Step 3</p>", "is_correct": False},
        ],
        "feedback": "<p>Step 2 is incorrect: sign-drop.</p>",
        "source_chunks": ["c5"],
    }


def _q_matching():
    return {
        "question_id": "q_match", "question_type": "multiple_choice",
        "stem": "<p>Which definition matches the term?</p>",
        "bloom_level": "understand", "objective_id": "CO-03",
        "item_subtype": "matching_mc",
        "choices": [
            {"text": "<p>correct def</p>", "is_correct": True},
            {"text": "<p>sibling def 1</p>", "is_correct": False},
            {"text": "<p>sibling def 2</p>", "is_correct": False},
        ],
        "feedback": "<p>The term means the correct def.</p>",
        "source_chunks": ["c6"],
    }


def _q_ordering():
    return {
        "question_id": "q_order", "question_type": "multiple_choice",
        "stem": "<p>Which shows the correct order?</p>", "bloom_level": "apply",
        "objective_id": "CO-04", "item_subtype": "ordering_mc",
        "choices": [
            {"text": "<p>(1) a → (2) b</p>", "is_correct": True},
            {"text": "<p>(1) b → (2) a</p>", "is_correct": False},
        ],
        "feedback": "<p>The correct sequence follows the worked solution.</p>",
        "source_chunks": ["c7"],
    }


def _q_two_tier():
    ans = {
        "question_id": "q_tt_a", "question_type": "multiple_choice",
        "stem": "<p>What is the answer?</p>", "bloom_level": "understand",
        "objective_id": "CO-04", "item_subtype": "two_tier_answer",
        "linked_item_id": "q_tt_r",
        "choices": [
            {"text": "<p>right</p>", "is_correct": True},
            {"text": "<p>wrong</p>", "is_correct": False},
        ],
        "feedback": "<p>right is correct.</p>", "source_chunks": ["c8"],
    }
    reason = {
        "question_id": "q_tt_r", "question_type": "multiple_choice",
        "stem": "<p>What is the reason?</p>", "bloom_level": "understand",
        "objective_id": "CO-04", "item_subtype": "two_tier_reason",
        "linked_item_id": "q_tt_a",
        "choices": [
            {"text": "<p>because Z</p>", "is_correct": True},
            {"text": "<p>because W</p>", "is_correct": False},
        ],
        "feedback": "<p>because Z is the reason.</p>", "source_chunks": ["c8"],
    }
    return ans, reason


def _q_essay():
    return {
        "question_id": "q_essay", "question_type": "essay",
        "stem": "<p>Evaluate the tradeoffs.</p>", "bloom_level": "evaluate",
        "objective_id": "TO-01",
        "feedback": "<ul><li>identifies principle</li><li>proposes application</li></ul>",
        "source_chunks": [],
    }


def _all_quiz_questions():
    tt_a, tt_r = _q_two_tier()
    return [
        _q_mc_single(), _q_mr(), _q_tf(), _q_fib_numeric(), _q_error_analysis(),
        _q_matching(), _q_ordering(), tt_a, tt_r, _q_essay(),
    ]


def _assessment(questions, week=1):
    return {"assessment_id": "w1_quiz", "title": "Week 1 Quiz",
            "questions": questions, "week": week}


_OBJ_TEXT = {
    "TO-01": "Analyze the tradeoffs between two designs",
    "CO-01": "Apply the method to a new problem",
}


# --------------------------------------------------------------------------- #
# 1. Every item_subtype reshapes into a well-formed record
# --------------------------------------------------------------------------- #
def test_every_item_subtype_reshapes():
    doc = build_answer_key(
        assessments=[_assessment(_all_quiz_questions())],
        objective_text=_OBJ_TEXT, generated_at="2026-01-01T00:00:00Z",
    )
    # Every diversified subtype is present; the essay item legitimately
    # carries no subtype (None) and is not counted here.
    subtypes = {it["item_subtype"] for it in doc["items"] if it["item_subtype"]}
    assert subtypes == {
        "mc_single", "mc_multiple_response", "tf", "fib_numeric",
        "error_analysis", "matching_mc", "ordering_mc",
        "two_tier_answer", "two_tier_reason",
    }
    for it in doc["items"]:
        assert set(it) >= {
            "item_id", "objective_id", "type", "item_subtype",
            "correct_answers", "worked_solution_steps", "per_distractor",
            "source_chunk_ids",
        }
    assert doc["item_count"] == 10


def _q_written(subtype="extended_response"):
    return {
        "question_id": "q_written", "question_type": "essay",
        "stem": "<p>Solve <em>the linear system</em>. Show your work.</p>",
        "bloom_level": "apply", "objective_id": "TO-01",
        "item_subtype": subtype,
        "feedback": "<ol><li>Isolate x</li><li>Substitute</li></ol>",
        "source_chunks": ["c_proc"],
        "rubric": {
            "criteria": [
                {"criterion": "Shows and justifies isolating x.",
                 "cites": ["c_proc"],
                 "levels": [{"score": 2, "descriptor": "Full."},
                            {"score": 1, "descriptor": "Partial."},
                            {"score": 0, "descriptor": "None."}]},
                {"criterion": "Shows the substitution step.",
                 "cites": ["c_proc"],
                 "levels": [{"score": 2, "descriptor": "Full."},
                            {"score": 0, "descriptor": "None."}]},
            ],
            "deductions": [
                {"error": "sign_drop", "note": "Dropped a negative sign.",
                 "points": -1.0, "cites": ["c_proc"]},
            ],
        },
    }


def test_written_item_structured_rubric_in_json():
    doc = build_answer_key(assessments=[_assessment([_q_written()])],
                           generated_at="t")
    item = doc["items"][0]
    assert item["item_subtype"] == "extended_response"
    rub = item["rubric"]
    assert isinstance(rub, dict)
    assert len(rub["criteria"]) == 2
    # Every criterion cites its source chunk (grounded by construction).
    for row in rub["criteria"]:
        assert row["cites"] == ["c_proc"]
        assert row["levels"][0]["score"] == 2
    assert rub["deductions"][0]["points"] == -1.0
    # Model answer travels as worked-solution steps citing the chunk.
    assert item["worked_solution_steps"]
    assert item["worked_solution_steps"][0]["cites"] == ["c_proc"]


def test_written_item_structured_rubric_in_html():
    doc = build_answer_key(assessments=[_assessment([_q_written()])],
                           generated_at="t")
    html = render_answer_key_html(doc)
    assert "Score 2" in html
    assert "Common error" in html
    assert "Dropped a negative sign." in html
    assert "c_proc" in html  # source citation rendered


def test_two_tier_linked_item_id_preserved():
    doc = build_answer_key(
        assessments=[_assessment(_all_quiz_questions())],
        generated_at="2026-01-01T00:00:00Z",
    )
    by_id = {it["item_id"]: it for it in doc["items"]}
    assert by_id["q_tt_a"]["linked_item_id"] == "q_tt_r"
    assert by_id["q_tt_r"]["linked_item_id"] == "q_tt_a"


# --------------------------------------------------------------------------- #
# 2. Correct-answer derivation precedence
# --------------------------------------------------------------------------- #
def test_correct_answer_plural_wins():
    doc = build_answer_key(assessments=[_assessment([_q_mr()])],
                           generated_at="t")
    assert doc["items"][0]["correct_answers"] == ["step 1", "step 2"]


def test_correct_answer_scalar():
    doc = build_answer_key(assessments=[_assessment([_q_fib_numeric()])],
                           generated_at="t")
    assert doc["items"][0]["correct_answers"] == ["5"]


def test_correct_answer_from_is_correct_choices():
    # mc_single carries neither plural nor scalar → derive from choices.
    doc = build_answer_key(assessments=[_assessment([_q_mc_single()])],
                           generated_at="t")
    assert doc["items"][0]["correct_answers"] == ["the right def"]


# --------------------------------------------------------------------------- #
# 3. Worked-solution steps cite the stamped source chunks
# --------------------------------------------------------------------------- #
def test_worked_solution_steps_cite_source_chunks():
    doc = build_answer_key(assessments=[_assessment([_q_mc_single()])],
                           generated_at="t")
    steps = doc["items"][0]["worked_solution_steps"]
    assert steps == [{"text": "X is the right def.", "cites": ["c1"]}]


def test_worked_solution_li_body_splits_into_steps():
    doc = build_answer_key(assessments=[_assessment([_q_essay()])],
                           generated_at="t")
    steps = doc["items"][0]["worked_solution_steps"]
    assert [s["text"] for s in steps] == [
        "identifies principle", "proposes application"
    ]
    # Essay has no source chunks → steps cite an empty list (honest).
    assert all(s["cites"] == [] for s in steps)


# --------------------------------------------------------------------------- #
# 4. Per-distractor rationale + honest absence
# --------------------------------------------------------------------------- #
def test_per_distractor_surfaces_misconception():
    doc = build_answer_key(assessments=[_assessment([_q_mc_single()])],
                           generated_at="t")
    dists = doc["items"][0]["per_distractor"]
    # Two distractors: one with a note, one without.
    notes = {d["choice"]: d["misconception"] for d in dists}
    assert notes == {"wrong def": "confuses X with Y", "other def": None}


# --------------------------------------------------------------------------- #
# 5. Numeric FIB instructor note
# --------------------------------------------------------------------------- #
def test_numeric_fib_stamps_tolerance_note():
    doc = build_answer_key(assessments=[_assessment([_q_fib_numeric()])],
                           generated_at="t")
    assert "tolerance" in doc["items"][0]["instructor_note"].lower()


def test_non_numeric_has_no_instructor_note():
    doc = build_answer_key(assessments=[_assessment([_q_mc_single()])],
                           generated_at="t")
    assert "instructor_note" not in doc["items"][0]


# --------------------------------------------------------------------------- #
# 6. Rubric scaffolds (essay / discussion / assignment)
# --------------------------------------------------------------------------- #
def test_essay_item_carries_rubric():
    doc = build_answer_key(assessments=[_assessment([_q_essay()])],
                           generated_at="t")
    rubric = doc["items"][0]["rubric"]
    assert len(rubric) == 3
    for row in rubric:
        assert set(row) == {"criterion", "levels"}
        assert row["levels"] and all(
            set(l) == {"level", "descriptor"} for l in row["levels"]
        )


def test_discussion_assignment_rubric_items():
    disc = [{"discussion_id": "D1", "objective_id": "TO-01",
             "title": "Disc", "text": "<p>Compare X and Y</p>", "week": 1}]
    asgn = [{"assignment_id": "A1", "objective_id": "CO-01",
             "title": "Asgn", "text": "<p>Build a thing</p>", "week": 1}]
    doc = build_answer_key(assessments=[], discussions=disc, assignments=asgn,
                           objective_text=_OBJ_TEXT, generated_at="t")
    assert doc["rubric_item_count"] == 2
    kinds = {r["kind"] for r in doc["rubric_items"]}
    assert kinds == {"discussion", "assignment"}
    disc_row = next(r for r in doc["rubric_items"] if r["kind"] == "discussion")
    # Bloom derived from the objective statement verb ("Analyze ...").
    assert disc_row["bloom_level"] == "analyze"
    # Criterion cites the objective text verbatim.
    assert _OBJ_TEXT["TO-01"] in disc_row["rubric"][0]["criterion"]


def test_rubric_lives_only_in_answer_key_not_quiz_json():
    # A non-essay quiz item never carries a rubric key.
    doc = build_answer_key(assessments=[_assessment([_q_mc_single()])],
                           generated_at="t")
    assert "rubric" not in doc["items"][0]


# --------------------------------------------------------------------------- #
# 7. Absent-field honesty
# --------------------------------------------------------------------------- #
def test_absent_fields_render_explicitly_absent():
    bare = {
        "question_id": "q_bare", "question_type": "multiple_choice",
        "stem": "<p>?</p>", "bloom_level": "remember", "objective_id": "CO-09",
        "item_subtype": "mc_single",
        # no choices, no correct_answer(s), no feedback, no source_chunks.
    }
    doc = build_answer_key(assessments=[_assessment([bare])], generated_at="t")
    it = doc["items"][0]
    assert it["correct_answers"] == []
    assert it["worked_solution_steps"] == []
    assert it["per_distractor"] == []
    assert it["source_chunk_ids"] == []
    html = render_answer_key_html(doc)
    assert "not recorded" in html.lower()
    # No fabricated answer text appears.
    assert "q_bare" in html


# --------------------------------------------------------------------------- #
# 8. Accepts QuestionData dataclass instances
# --------------------------------------------------------------------------- #
def test_accepts_questiondata_dataclass_instances():
    from Trainforge.generators.assessment_generator import (
        AssessmentData, QuestionData,
    )
    q = QuestionData(
        question_id="qd1", question_type="multiple_response",
        stem="<p>select all</p>", bloom_level="apply", objective_id="CO-01",
        choices=[
            {"text": "<p>k1</p>", "is_correct": True},
            {"text": "<p>d1</p>", "is_correct": False,
             "misconception_note": "err"},
        ],
        correct_answers=["k1"], item_subtype="mc_multiple_response",
        feedback="<p>solution</p>", source_chunks=["cc1"],
    )
    a = AssessmentData(assessment_id="a1", title="A1", course_code="X",
                       questions=[q])
    doc = build_answer_key(assessments=[a], generated_at="t")
    it = doc["items"][0]
    assert it["item_id"] == "qd1"
    assert it["correct_answers"] == ["k1"]
    assert it["worked_solution_steps"][0]["cites"] == ["cc1"]
    assert it["per_distractor"] == [{"choice": "d1", "misconception": "err"}]


# --------------------------------------------------------------------------- #
# 9. Accessible HTML rendering
# --------------------------------------------------------------------------- #
def test_html_is_accessible_and_escaped():
    q = _q_mc_single()
    q["stem"] = "<p>tag & <script>x</script></p>"
    doc = build_answer_key(assessments=[_assessment([q, _q_essay()])],
                           discussions=[{"discussion_id": "D1",
                                         "objective_id": "TO-01",
                                         "title": "Disc",
                                         "text": "<p>Compare</p>"}],
                           objective_text=_OBJ_TEXT, generated_at="t")
    html = render_answer_key_html(doc)
    assert html.startswith("<!DOCTYPE html>")
    assert '<html lang="en">' in html
    assert "<h1>" in html and "<h2>" in html and "<h3>" in html
    # Tables carry scope on header cells (no layout-only tables).
    assert 'scope="col"' in html
    assert 'scope="row"' in html
    # Correct answer signaled by a TEXT label, not color alone.
    assert "Correct answer" in html
    # No raw script injection survived escaping.
    assert "<script>x</script>" not in html


# --------------------------------------------------------------------------- #
# 10. emit_answer_key writes both files + returns the manifest row
# --------------------------------------------------------------------------- #
def test_emit_writes_files_and_returns_manifest_row(tmp_path):
    out = tmp_path / "06_assessments"
    entry = emit_answer_key(
        out, assessments=[_assessment(_all_quiz_questions())],
        discussions=[{"discussion_id": "D1", "objective_id": "TO-01",
                      "title": "Disc", "text": "<p>Compare</p>"}],
        objective_text=_OBJ_TEXT, course_code="TEST",
        generated_at="2026-01-01T00:00:00Z",
    )
    assert (out / "answer_key.json").is_file()
    assert (out / "answer_key.html").is_file()
    doc = json.loads((out / "answer_key.json").read_text(encoding="utf-8"))
    assert doc["schema_version"] == "v1"
    assert doc["item_count"] == 10
    assert entry == {
        "file": "answer_key.html", "type": "answer_key",
        "title": "TEST — Instructor Answer Key",
        "identifier": "RES_answer_key",
    }


def test_manifest_entry_shape():
    entry = answer_key_manifest_entry()
    assert entry["file"] == "answer_key.html"
    assert entry["type"] == "answer_key"
    # The packager maps the answer_key type to a CC webcontent resource, but
    # keeps it OUT of the distinctly-assessment QTI-type set.
    assert _assessment_res_type_for_kind("answer_key") == "webcontent"
    assert "answer_key" not in _ASSESSMENT_RES_TYPE


# --------------------------------------------------------------------------- #
# 11. Packaging inclusion — answer_key.html ships as a webcontent resource
# --------------------------------------------------------------------------- #
def test_classify_accepts_answer_key_type(tmp_path):
    adir = tmp_path / "06_assessments"
    adir.mkdir()
    (adir / "answer_key.html").write_text("<html></html>", encoding="utf-8")
    a = _classify_assessment(
        xml_path=adir / "answer_key.html", assessments_dir=adir,
        declared_type="answer_key", declared_title="Instructor Answer Key",
        declared_week=None, declared_id="RES_answer_key",
    )
    assert a is not None
    assert a.kind == "answer_key"
    assert a.rel_path == "06_assessments/answer_key.html"


def _make_content_dir_with_answer_key(root: Path) -> Path:
    """Pipeline layout: content dir + a 06_assessments/ sibling carrying an
    answer_key.html + a manifest.json sidecar declaring the answer_key row."""
    content_dir = root / "03_content_development"
    (content_dir / "week_01").mkdir(parents=True, exist_ok=True)
    (content_dir / "week_01" / "week_01_overview.html").write_text(
        "<html><body><h1>Week 1</h1></body></html>", encoding="utf-8",
    )
    adir = root / "06_assessments"
    entry = emit_answer_key(
        adir, assessments=[_assessment(_all_quiz_questions())],
        objective_text=_OBJ_TEXT, course_code="TEST",
        generated_at="2026-01-01T00:00:00Z",
    )
    (adir / "manifest.json").write_text(
        json.dumps({"schema_version": "v1", "assessments": [entry]}),
        encoding="utf-8",
    )
    return content_dir


def test_packager_ships_answer_key_as_webcontent(tmp_path):
    content_dir = _make_content_dir_with_answer_key(tmp_path)
    output = tmp_path / "out.imscc"
    package_imscc(content_dir, output, "TEST_101", "Test Course",
                  skip_validation=True)
    assert output.exists()
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
        manifest_xml = zf.read("imsmanifest.xml").decode("utf-8")
    # The answer_key.html landed in the zip at the manifest rel_path.
    assert "06_assessments/answer_key.html" in names
    # A manifest <resource> references it as webcontent.
    root = ET.fromstring(manifest_xml)
    hrefs = {
        r.get("href"): r.get("type")
        for r in root.iter(f"{{{_CC_NS}}}resource")
    }
    assert hrefs.get("06_assessments/answer_key.html") == "webcontent"
    # And an org <item identifierref> points at it.
    refs = {
        i.get("identifierref")
        for i in root.iter(f"{{{_CC_NS}}}item")
        if i.get("identifierref")
    }
    assert "RES_answer_key" in refs


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))

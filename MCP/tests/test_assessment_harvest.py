"""Regression net for `ED4ALL_TRAINFORGE_ASSESSMENT_HARVEST`.

`trainforge_assessment` historically ran a SECOND generation pass over the
same content `assessment_synthesis` already authored and packaging already
shipped. Harvest mode reads those items back out of the packaged IMSCC and
re-keys them onto the IMSCC chunk universe instead.

Contracts pinned here:

* default OFF (the legacy generate path is untouched);
* chunk re-keying is PRINCIPLED — an item's `source_chunks` are the IMSCC
  chunks whose `learning_outcome_refs` carry its `objective_id`, which is
  the same relation the fail-closed coverage rule asserts;
* anti-fabrication — an item whose objective resolves to no chunk is
  DROPPED and counted, never given an invented citation;
* a non-QTI / unreadable archive degrades to an empty harvest (the caller
  turns that into a loud structured error, not a silent fallback).
"""

import zipfile

import pytest

from Courseforge.scripts.qti_emitter import assessment_to_qti
from MCP.tools.pipeline_tools import (
    _harvest_questions_from_imscc,
    _resolve_assessment_harvest,
)

ENV = "ED4ALL_TRAINFORGE_ASSESSMENT_HARVEST"


def _question(qid="Q-1", objective="CO-01"):
    return {
        "question_id": qid,
        "question_type": "multiple_choice",
        "stem": "<p>Identify the additive identity.</p>",
        "choices": [
            {"id": "A", "text": "0", "is_correct": True},
            {"id": "B", "text": "1"},
            {"id": "C", "text": "-1"},
        ],
        "correct_answer": "0",
        "objective_id": objective,
        "bloom_level": "remember",
    }


def _imscc(tmp_path, questions, name="course.imscc"):
    xml = assessment_to_qti(
        {"assessment_id": "A-1", "title": "Quiz", "questions": questions}
    )
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("assessments/quiz.xml", xml)
    return p


def _chunks(*objectives):
    return [
        {"id": f"imscc_chunk_{i:05d}", "learning_outcome_refs": [o]}
        for i, o in enumerate(objectives)
    ]


# ── flag ────────────────────────────────────────────────────────────────────
def test_defaults_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert _resolve_assessment_harvest() is False


@pytest.mark.parametrize("raw", ["1", "true", "YES", "on"])
def test_truthy(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    assert _resolve_assessment_harvest() is True


@pytest.mark.parametrize("raw", ["", "0", "false", "off", "garbage"])
def test_falsey_or_garbage(monkeypatch, raw):
    monkeypatch.setenv(ENV, raw)
    assert _resolve_assessment_harvest() is False


# ── harvest ─────────────────────────────────────────────────────────────────
def test_harvests_items_and_rekeys_chunks(tmp_path):
    imscc = _imscc(tmp_path, [_question()])
    qs, stats = _harvest_questions_from_imscc(imscc, _chunks("CO-01"))
    assert stats["items_kept"] == 1 and stats["items_seen"] == 1
    q = qs[0]
    assert q.objective_id == "CO-01"
    assert q.source_chunks == ["imscc_chunk_00000"]
    assert q.question_type == "multiple_choice"
    assert len(q.choices) == 3


def test_objective_with_no_covering_chunk_is_dropped_not_fabricated(tmp_path):
    """Anti-fabrication: no covering chunk -> drop + count, never invent."""
    imscc = _imscc(tmp_path, [_question(objective="CO-99")])
    qs, stats = _harvest_questions_from_imscc(imscc, _chunks("CO-01"))
    assert qs == []
    assert stats["dropped_no_chunk"] == 1
    assert stats["items_kept"] == 0


def test_multiple_chunks_for_one_objective_all_attach(tmp_path):
    imscc = _imscc(tmp_path, [_question()])
    chunks = [
        {"id": "c1", "learning_outcome_refs": ["CO-01"]},
        {"id": "c2", "learning_outcome_refs": ["CO-01", "CO-02"]},
        {"id": "c3", "learning_outcome_refs": ["CO-02"]},
    ]
    qs, _ = _harvest_questions_from_imscc(imscc, chunks)
    assert qs[0].source_chunks == ["c1", "c2"]


def test_objective_match_is_case_insensitive(tmp_path):
    imscc = _imscc(tmp_path, [_question(objective="co-01")])
    qs, _ = _harvest_questions_from_imscc(imscc, _chunks("CO-01"))
    assert len(qs) == 1


def test_multiple_qti_files_are_all_read(tmp_path):
    xml_a = assessment_to_qti(
        {"assessment_id": "A", "title": "A", "questions": [_question("Q-1", "CO-01")]}
    )
    xml_b = assessment_to_qti(
        {"assessment_id": "B", "title": "B", "questions": [_question("Q-2", "CO-02")]}
    )
    p = tmp_path / "c.imscc"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("a/q1.xml", xml_a)
        zf.writestr("a/q2.xml", xml_b)
    qs, stats = _harvest_questions_from_imscc(p, _chunks("CO-01", "CO-02"))
    assert stats["qti_files"] == 2 and len(qs) == 2


def test_non_qti_xml_entries_are_ignored(tmp_path):
    p = tmp_path / "c.imscc"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("imsmanifest.xml", "<manifest><resources/></manifest>")
        zf.writestr("notes.txt", "not xml")
        zf.writestr(
            "a/q.xml",
            assessment_to_qti(
                {"assessment_id": "A", "title": "A", "questions": [_question()]}
            ),
        )
    qs, stats = _harvest_questions_from_imscc(p, _chunks("CO-01"))
    assert stats["qti_files"] == 1 and len(qs) == 1


def test_missing_archive_yields_empty_harvest(tmp_path):
    """Caller turns an empty harvest into a loud error; this must not raise."""
    qs, stats = _harvest_questions_from_imscc(
        tmp_path / "does-not-exist.imscc", _chunks("CO-01")
    )
    assert qs == [] and stats["items_kept"] == 0


def test_corrupt_archive_yields_empty_harvest(tmp_path):
    p = tmp_path / "broken.imscc"
    p.write_bytes(b"not a zip file")
    qs, _ = _harvest_questions_from_imscc(p, _chunks("CO-01"))
    assert qs == []


def test_chunks_without_outcome_refs_cover_nothing(tmp_path):
    imscc = _imscc(tmp_path, [_question()])
    qs, stats = _harvest_questions_from_imscc(
        imscc, [{"id": "c1"}, {"id": "c2", "learning_outcome_refs": []}]
    )
    assert qs == [] and stats["dropped_no_chunk"] == 1

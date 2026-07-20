"""D1 regression — harvested QTI ``assessment_item`` chunks are tagged
``practice_bank=True``.

Owner directive 2026-07-19: the harvested end-of-section assessment_item
chunks are verbatim OpenStax content demoted to an internal practice bank.
They stay in the chunkset (archival coverage stands) but must carry the
``practice_bank`` marker so the graded-QTI / SFT-source / answer paths exclude
them by default. This pins the tag on the harvest site.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import _harvest_qti_assessment_chunks  # noqa: E402


def test_harvested_assessment_item_chunks_are_practice_bank_tagged() -> None:
    from Courseforge.scripts.qti_emitter import assessment_to_qti

    quiz_xml = assessment_to_qti({
        "assessment_id": "ASM-1",
        "title": "Week 1 Quiz",
        "questions": [
            {"question_id": "q-1", "question_type": "multiple_choice",
             "stem": "<p>Pick one.</p>", "objective_id": "CO-01",
             "bloom_level": "understand",
             "choices": [
                 {"id": "A", "text": "Right", "is_correct": True},
                 {"id": "B", "text": "Wrong", "is_correct": False},
             ], "correct_answer": "A"},
        ],
    })

    def _fake_create_chunk(**kwargs):
        return {
            "id": kwargs["chunk_id"],
            "chunk_type": kwargs["chunk_type"],
            "learning_outcome_refs": kwargs["item"]["objective_refs"],
        }

    chunks = _harvest_qti_assessment_chunks(
        [{"path": "06_assessments/week_01_quiz.xml", "content": quiz_xml}],
        create_chunk=_fake_create_chunk,
        existing_chunks=[],
        course_code="DEMO",
    )
    assert chunks, "expected one harvested assessment_item chunk"
    for c in chunks:
        assert c["chunk_type"] == "assessment_item"
        assert c["practice_bank"] is True

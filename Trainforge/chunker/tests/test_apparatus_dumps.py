"""Unit tests for the answer-key / ToC apparatus-dump chunk filter.

Positive fixtures are lifted verbatim from a real 3-chapter OpenStax Elementary
Algebra SCAN chunkset (``semantik_chunks/chunks.jsonl``):

  * ``_chunk_00008`` — an end-of-chapter answer-key run
    ("46. 550 47. 22,335 48. 39,075 ...").
  * ``_chunk_00001`` / ``_00080`` / ``_00121`` — chapter-outline / ToC chunks
    ("Chapter Outline 1.2 Use the Language of Algebra 1.3 ...").

Negative fixtures model real math-dense exercise/example prose that MUST NOT be
dropped (LaTeX + legitimate embedded numbers).
"""

from __future__ import annotations

from Trainforge.chunker.apparatus_dumps import (
    FLAG_ENV,
    classify_apparatus_dump,
    partition_apparatus_dumps,
)


_ANSWER_KEY = (
    "46. 550 47. 22,335 48. 39,075 37. 84 38. 9,696 39. 75 40. 78 41. 900 "
    "42. 800 43. 986 44. 942 45. 350 100. 12 101. 44 102. 9 103. 7"
)
_OUTLINE = (
    "Chapter Outline 1.2 Use the Language of Algebra \n 1.3 Add and Subtract "
    "Integers \n 1.4 Multiply and Divide Integers \n 1.5 Add and Subtract "
    "Fractions"
)
_TOC = (
    "Table of Contents 2.1 Solve Equations \n 2.2 Use a Formula \n 2.3 Solve "
    "Applications \n 2.4 Model Real Life"
)
# Real math exercise prose — item numbers followed by LaTeX, NOT bare numbers.
_EXERCISE = (
    "In the following exercises, simplify using the distributive property. "
    "$8(4y + 9)$ $9(3w + 7)$ $6(c - 13)$ $7(y - 4)$ 554. $\\frac{3}{4}$ "
    "555. $\\frac{8}{5}$"
)
_EXAMPLE = (
    "Round 23,658 to the nearest hundred. Solution Step 1. Locate the given "
    "place value. Step 2. The digit to the right is 5, so round up to 23,700."
)
_MENTION = (
    "This chapter outline is a useful planning tool, but the real work begins "
    "in section 1.2 where we build the language of algebra from the ground up."
)


def test_answer_key_dump_dropped():
    v = classify_apparatus_dump({"id": "c8", "text": _ANSWER_KEY})
    assert v.is_apparatus is True
    assert "answer_key_dump" in v.reason


def test_chapter_outline_dropped():
    v = classify_apparatus_dump({"id": "c1", "text": _OUTLINE})
    assert v.is_apparatus is True
    assert "chapter_outline_toc" in v.reason


def test_table_of_contents_dropped():
    v = classify_apparatus_dump({"id": "c2", "text": _TOC})
    assert v.is_apparatus is True


def test_real_exercise_prose_kept():
    assert classify_apparatus_dump({"id": "e1", "text": _EXERCISE}).is_apparatus is False


def test_worked_example_kept():
    assert classify_apparatus_dump({"id": "x1", "text": _EXAMPLE}).is_apparatus is False


def test_outline_mention_in_prose_kept():
    # "chapter outline" as a leading header needs corroborating N.M enumerations;
    # a prose sentence that merely opens with it is NOT dropped.
    assert classify_apparatus_dump({"id": "m1", "text": _MENTION}).is_apparatus is False


def test_empty_chunk_kept():
    assert classify_apparatus_dump({"id": "z", "text": ""}).is_apparatus is False


def test_partition_splits_and_preserves_order():
    chunks = [
        {"id": "keep1", "text": _EXAMPLE},
        {"id": "ak", "text": _ANSWER_KEY},
        {"id": "keep2", "text": _EXERCISE},
        {"id": "toc", "text": _OUTLINE},
    ]
    kept, dropped = partition_apparatus_dumps(chunks)
    assert [c["id"] for c in kept] == ["keep1", "keep2"]
    assert [v.chunk_id for v in dropped] == ["ak", "toc"]
    assert all(v.rationale() for v in dropped)


def test_flag_env_name():
    assert FLAG_ENV == "TRAINFORGE_DROP_APPARATUS_DUMPS"

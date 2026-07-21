"""Unit tests for the answer-key / ToC apparatus-dump chunk filter.

Positive fixtures are SYNTHETIC, modeling the two apparatus-dump families a
scanned algebra textbook chunkset produces (invented numbers + invented section
titles that preserve every discriminating shape the classifier keys on):

  * an end-of-chapter answer-key run — a dominant bare "NN. NNNN" number soup.
  * chapter-outline / Table-of-Contents chunks — a leading header corroborated
    by several "N.M <Title>" section enumerations.

Negative fixtures model math-dense exercise/example prose that MUST NOT be
dropped (LaTeX + legitimate embedded numbers).
"""

from __future__ import annotations

from Trainforge.chunker.apparatus_dumps import (
    FLAG_ENV,
    classify_apparatus_dump,
    partition_apparatus_dumps,
)


_ANSWER_KEY = (
    "12. 480 13. 1,024 14. 96 15. 7,205 16. 512 17. 88 18. 3,640 19. 275 "
    "20. 6,018 21. 144 22. 909 23. 42 24. 1,360 25. 58 26. 704 27. 233"
)
_OUTLINE = (
    "Chapter Outline 1.2 Reading Numeric Expressions \n 1.3 Combining Signed "
    "Quantities \n 1.4 Scaling and Partitioning Values \n 1.5 Working with "
    "Part-Whole Numbers"
)
_TOC = (
    "Table of Contents 2.1 Building Simple Equations \n 2.2 Applying a Known "
    "Formula \n 2.3 Translating Word Problems \n 2.4 Estimating Everyday "
    "Quantities"
)
# Math exercise prose — item numbers followed by LaTeX, NOT bare numbers.
_EXERCISE = (
    "In the following exercises, simplify using the distributive property. "
    "$5(3x + 8)$ $7(2w + 4)$ $9(b - 12)$ $6(y - 5)$ 331. $\\frac{2}{7}$ "
    "332. $\\frac{9}{4}$"
)
_EXAMPLE = (
    "Round 47,182 to the nearest hundred. Solution Step 1. Locate the given "
    "place value. Step 2. The digit to the right is 8, so round up to 47,200."
)
_MENTION = (
    "This chapter outline is a useful planning tool, but the real work begins "
    "in section 1.2 where we build core algebraic reasoning from the ground up."
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

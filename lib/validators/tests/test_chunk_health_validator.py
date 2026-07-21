"""Tests for ``ChunkHealthValidator`` (pre-synthesis chunk-health gate).

The gate ports the ``chunk_health_preflight.py`` prototype and blocks
``course_planning`` synthesis when the emitted chunkset / structure is poisoned.

Fixtures are SYNTHETIC / invented — no real course slug or on-disk corpus path
is hardcoded (project rule: ``no_course_data_references``). The "poisoned"
fixture mirrors the observed MC3 defect shape (phantom chapters from a collapsed
``contiguous_ward`` resegment + a low instructional-prose share + an empty
chunk) without referencing the real course.

Cases:

1. ``test_skip_with_pass_when_gate_off`` — default (flag unset) → no-op pass,
   even with poisoned inputs.
2. ``test_poisoned_corpus_blocks`` — with the gate ON, the phantom-chapter /
   arbitrary-resegment / section-explosion / instructional-starvation / empty
   defects fire CRITICAL and block.
3. ``test_healthy_corpus_passes`` — a well-formed synthetic corpus passes clean.
4. ``test_missing_chunkset_fails_closed`` — gate ON + no chunks_path → critical
   fail-closed (never synthesize from nothing).
5. ``test_structure_optional`` — a chunk-only import (no structure) runs the
   C-class checks and passes when the chunks are healthy.
6. ``test_decision_capture_emitted`` — exactly one ``content_structure_check``
   event per run.
7. ``test_env_threshold_override`` — an env override widens the instructional
   floor so a borderline corpus flips verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.validators.chunk_health import ChunkHealthValidator  # noqa: E402


class _StubDecisionCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


def _codes(result) -> List[str]:
    return [i.code for i in result.issues]


def _crit_codes(result) -> List[str]:
    return [i.code for i in result.issues if i.severity == "critical"]


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #


def _healthy_chunks() -> List[Dict[str, Any]]:
    """A synthetic instructional-rich chunkset (>= 40 chunks, mostly prose)."""
    rows: List[Dict[str, Any]] = []
    body = (
        "This section explains the core idea in full prose sentences so that a "
        "learner can understand the concept and its motivation before any drill. "
        "It builds up the reasoning step by step with connected explanation."
    )
    for i in range(40):
        ctype = "explanation" if i % 3 else "example"
        rows.append(
            {
                "id": f"chunk-{i:03d}",
                "chunk_type": ctype,
                "text": f"{body} (part {i})",
                "html": f"<p>{body}</p>",
                "word_count": 42,
                "source": {"module_title": "Chapter Two Linear Functions"},
            }
        )
    return rows


def _healthy_structure() -> Dict[str, Any]:
    """A synthetic well-formed structure: 3 sources → 4 chapters, real sections."""
    chapters = []
    for c in range(4):
        chapters.append(
            {
                "id": f"ch{c + 1}",
                "headingText": f"Chapter {c + 1}",
                "source_file": f"book_part_{c % 3}.html",
                "sections": [
                    {"id": f"s{c}_{s}", "headingText": f"{c + 1}.{s + 1} Topic"}
                    for s in range(6)
                ],
            }
        )
    return {
        "source_files": ["book_part_0.html", "book_part_1.html", "book_part_2.html"],
        "chapters": chapters,
        "structureDiagnostics": {"resegmented": False},
    }


def _worked_example_chunks() -> List[Dict[str, Any]]:
    """A worked-example-driven workbook: mostly ``example`` chunks WITH worked
    solutions + a couple explanation chunks.

    This mirrors the real OpenStax algebra shape (~80% example/exercise), where
    the examples are genuine WORKED examples (they carry a solution / step /
    "Try It" marker). Under the corrected C2 definition — worked examples TEACH
    and count as instructional — this corpus is synthesis-ready and must PASS.
    Under the OLD definition (only explanation/overview count) its instructional
    share would be ~15% and it would false-block.
    """
    rows: List[Dict[str, Any]] = []
    solution_body = (
        "Name the number 8,165,432,098,710 using words. Solution: Start at the "
        "left and name the number in each period. Step 1. Identify each period. "
        "Step 2. Write the words. So the answer is eight trillion and so on."
    )
    prose_body = (
        "This section explains the reasoning behind place value in full prose "
        "sentences so a learner understands the concept before working examples."
    )
    for i in range(40):
        if i % 8 == 0:
            # ~5 genuine instructional-prose chunks.
            rows.append(
                {
                    "id": f"we-{i:03d}",
                    "chunk_type": "explanation",
                    "text": f"{prose_body} (part {i})",
                    "html": f"<p>{prose_body}</p>",
                    "word_count": 30,
                    "source": {"module_title": "Chapter Two Whole Numbers"},
                }
            )
        else:
            # ~35 WORKED example chunks (carry a Solution / Step marker).
            rows.append(
                {
                    "id": f"we-{i:03d}",
                    "chunk_type": "example",
                    "text": f"Example {i}. {solution_body}",
                    "html": f"<p>{solution_body}</p>",
                    "word_count": 44,
                    "source": {"module_title": "Chapter Two Whole Numbers"},
                }
            )
    return rows


def _bare_exercise_chunks() -> List[Dict[str, Any]]:
    """A corpus with genuinely NO teaching content: only BARE exercises /
    answer-dumps (no worked solutions, no prose) + one empty chunk.

    Even under the corrected C2 definition (worked examples count) this must
    still FAIL C2 — the examples/exercises carry no solution marker, so nothing
    teaches. This guards the genuine starvation defect the check exists for.
    """
    rows: List[Dict[str, Any]] = []
    for i in range(50):
        # Bare drill: a problem statement + an answer-key dump, no solution walk.
        ctype = "example" if i % 2 else "exercise"
        rows.append(
            {
                "id": f"bare-{i:03d}",
                "chunk_type": ctype,
                "text": f"Exercise {i}: 1) 4 2) 8 3) 12 4) 16 answer key values.",
                "html": "",
                "word_count": 11,
                "source": {"module_title": "Chapter 1 Foundations 83"},
            }
        )
    rows.append(
        {
            "id": "bare-empty",
            "chunk_type": "exercise",
            "text": "   ",
            "html": "",
            "word_count": 0,
            "source": {"module_title": "Chapter 1 Foundations 84"},
        }
    )
    return rows


def _poisoned_chunks() -> List[Dict[str, Any]]:
    """Apparatus-dominated chunkset (low instructional share) + one empty chunk."""
    rows: List[Dict[str, Any]] = []
    for i in range(50):
        # Only ~12% instructional; the rest is example/exercise apparatus.
        ctype = "explanation" if i % 8 == 0 else ("example" if i % 2 else "exercise")
        rows.append(
            {
                "id": f"chunk-{i:03d}",
                "chunk_type": ctype,
                "text": f"Example {i}: 1) 4 2) 8 3) 12 4) 16 answer key values here.",
                "html": "",
                "word_count": 11,
                "source": {"module_title": "Chapter 1 Foundations 83"},
            }
        )
    # One genuinely empty-text chunk (C7 critical).
    rows.append(
        {
            "id": "chunk-empty",
            "chunk_type": "example",
            "text": "   ",
            "html": "",
            "word_count": 0,
            "source": {"module_title": "Chapter 1 Foundations 84"},
        }
    )
    return rows


def _poisoned_structure() -> Dict[str, Any]:
    """Collapsed structure: 3 sources → 85 phantom chapters via ward resegment."""
    chapters = []
    for c in range(85):
        chapters.append(
            {
                "id": f"ch{c + 1}",
                "headingText": f"Chapter {c + 1}",
                "source_file": f"book_part_{c % 3}.html",
                "sections": [
                    {"id": f"s{c}_{s}", "headingText": f"Example {s + 1}"}
                    for s in range(18)
                ],
            }
        )
    return {
        "source_files": ["book_part_0.html", "book_part_1.html", "book_part_2.html"],
        "chapters": chapters,
        "structureDiagnostics": {
            "resegmented": True,
            "method": "contiguous_ward",
            "k": 85,
            "original_section_count": 1555,
        },
    }


def _write(tmp_path: Path, name: str, obj: Any, jsonl: bool = False) -> str:
    p = tmp_path / name
    if jsonl:
        p.write_text(
            "\n".join(json.dumps(r) for r in obj) + "\n", encoding="utf-8"
        )
    else:
        p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_skip_with_pass_when_gate_off(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ED4ALL_CHUNK_HEALTH_GATE", raising=False)
    chunks = _write(tmp_path, "chunks.jsonl", _poisoned_chunks(), jsonl=True)
    ts = _write(tmp_path, "structure.json", _poisoned_structure())
    result = ChunkHealthValidator().validate(
        {"chunks_path": chunks, "textbook_structure_path": ts}
    )
    # Even with a poisoned corpus, the flag-off gate is a no-op pass.
    assert result.passed is True
    assert result.issues == []
    assert result.action is None


def test_poisoned_corpus_blocks(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ED4ALL_CHUNK_HEALTH_GATE", "1")
    chunks = _write(tmp_path, "chunks.jsonl", _poisoned_chunks(), jsonl=True)
    ts = _write(tmp_path, "structure.json", _poisoned_structure())
    result = ChunkHealthValidator().validate(
        {"chunks_path": chunks, "textbook_structure_path": ts}
    )
    assert result.passed is False
    assert result.action == "block"
    crit = _crit_codes(result)
    # The MC3-shape defect classes all fire critical. (S3 section explosion
    # deliberately absent: this fixture's 18 sections/chapter is NOT a
    # per-chapter explosion — S3 scales by chapters since the 2026-07-21
    # whole-book fix; its own defect shape is covered by
    # test_section_explosion_per_chapter below.)
    assert "CHUNK_HEALTH_CHAPTER_EXPLOSION" in crit
    assert "CHUNK_HEALTH_ARBITRARY_RESEGMENT" in crit
    assert "CHUNK_HEALTH_INSTRUCTIONAL_STARVED" in crit
    assert "CHUNK_HEALTH_EMPTY_CHUNKS" in crit


def test_section_explosion_per_chapter(monkeypatch, tmp_path) -> None:
    """True S3 shape: few REAL chapters, each shattered into ~60 sections."""
    monkeypatch.setenv("ED4ALL_CHUNK_HEALTH_GATE", "1")
    chunks = _write(tmp_path, "chunks.jsonl", _healthy_chunks(), jsonl=True)
    structure = {
        "source_files": ["book.html"],
        "chapters": [
            {
                "id": f"ch{c + 1}",
                "headingText": f"Chapter {c + 1}",
                "source_file": "book.html",
                "sections": [
                    {"id": f"s{c}_{s}", "headingText": f"Frag {s + 1}"}
                    for s in range(62)
                ],
            }
            for c in range(4)
        ],
    }
    ts = _write(tmp_path, "structure.json", structure)
    result = ChunkHealthValidator().validate(
        {"chunks_path": chunks, "textbook_structure_path": ts}
    )
    assert "CHUNK_HEALTH_SECTION_EXPLOSION" in _crit_codes(result)


def test_whole_book_single_pdf_structure_passes_s3(monkeypatch, tmp_path) -> None:
    """2026-07-21 canary regression: ONE source file with 19 real chapters of
    ~4 sections each must NOT trip S3 (the old source-file scale basis did)."""
    monkeypatch.setenv("ED4ALL_CHUNK_HEALTH_GATE", "1")
    chunks = _write(tmp_path, "chunks.jsonl", _healthy_chunks(), jsonl=True)
    structure = {
        "source_files": ["whole_book.html"],
        "chapters": [
            {
                "id": f"ch{c + 1}",
                "headingText": f"Chapter {c + 1}",
                "source_file": "whole_book.html",
                "sections": [
                    {"id": f"s{c}_{s}", "headingText": f"{c + 1}.{s + 1} Topic"}
                    for s in range(4)
                ],
            }
            for c in range(19)
        ],
    }
    ts = _write(tmp_path, "structure.json", structure)
    result = ChunkHealthValidator().validate(
        {"chunks_path": chunks, "textbook_structure_path": ts}
    )
    codes = {i.code for i in result.issues}
    assert "CHUNK_HEALTH_SECTION_EXPLOSION" not in codes


def test_healthy_corpus_passes(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ED4ALL_CHUNK_HEALTH_GATE", "1")
    chunks = _write(tmp_path, "chunks.jsonl", _healthy_chunks(), jsonl=True)
    ts = _write(tmp_path, "structure.json", _healthy_structure())
    result = ChunkHealthValidator().validate(
        {"chunks_path": chunks, "textbook_structure_path": ts}
    )
    assert result.passed is True, _codes(result)
    assert _crit_codes(result) == []
    assert result.action is None


def test_missing_chunkset_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("ED4ALL_CHUNK_HEALTH_GATE", "1")
    result = ChunkHealthValidator().validate({})
    assert result.passed is False
    assert result.action == "block"
    assert "CHUNK_HEALTH_CHUNKS_NOT_FOUND" in _crit_codes(result)


def test_structure_optional(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ED4ALL_CHUNK_HEALTH_GATE", "1")
    chunks = _write(tmp_path, "chunks.jsonl", _healthy_chunks(), jsonl=True)
    # No textbook_structure_path — a chunk-only import. C-class checks still run.
    result = ChunkHealthValidator().validate({"chunks_path": chunks})
    assert result.passed is True, _codes(result)
    # No structure means no S-class codes.
    assert not any(c.startswith("CHUNK_HEALTH_CHAPTER") for c in _codes(result))


def test_decision_capture_emitted(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ED4ALL_CHUNK_HEALTH_GATE", "1")
    chunks = _write(tmp_path, "chunks.jsonl", _healthy_chunks(), jsonl=True)
    cap = _StubDecisionCapture()
    ChunkHealthValidator().validate(
        {"chunks_path": chunks, "decision_capture": cap}
    )
    assert len(cap.events) == 1
    assert cap.events[0]["decision_type"] == "content_structure_check"
    assert len(cap.events[0]["rationale"]) >= 20


def test_env_threshold_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ED4ALL_CHUNK_HEALTH_GATE", "1")
    # A borderline corpus: ~30% instructional — below the 0.40 warn band but
    # above the 0.20 fail floor → warning by default.
    rows: List[Dict[str, Any]] = []
    for i in range(40):
        ctype = "explanation" if i % 3 == 0 else "example"
        rows.append(
            {
                "id": f"c{i}",
                "chunk_type": ctype,
                "text": "Prose explanation sentence with enough words to count here.",
                "word_count": 40,
                "source": {"module_title": "Clean Title"},
            }
        )
    chunks = _write(tmp_path, "chunks.jsonl", rows, jsonl=True)

    default = ChunkHealthValidator().validate({"chunks_path": chunks})
    assert default.passed is True
    assert "CHUNK_HEALTH_INSTRUCTIONAL_STARVED" in _codes(default)

    # Raise the FAIL floor above the observed share → now CRITICAL / blocks.
    monkeypatch.setenv("ED4ALL_CHUNK_HEALTH_INSTRUCTIONAL_FAIL", "0.50")
    strict = ChunkHealthValidator().validate({"chunks_path": chunks})
    assert strict.passed is False
    assert "CHUNK_HEALTH_INSTRUCTIONAL_STARVED" in _crit_codes(strict)


# --------------------------------------------------------------------------- #
# C2 recalibration: worked examples count as instructional
# --------------------------------------------------------------------------- #


def test_worked_example_workbook_passes_c2(monkeypatch, tmp_path) -> None:
    """A worked-example-heavy workbook (mostly ``example`` chunks WITH solution
    markers) is synthesis-ready and must PASS C2 — worked examples TEACH."""
    monkeypatch.setenv("ED4ALL_CHUNK_HEALTH_GATE", "1")
    monkeypatch.delenv(
        "ED4ALL_CHUNK_HEALTH_WORKED_EXAMPLE_INSTRUCTIONAL", raising=False
    )
    chunks = _write(tmp_path, "chunks.jsonl", _worked_example_chunks(), jsonl=True)
    result = ChunkHealthValidator().validate({"chunks_path": chunks})
    assert result.passed is True, _codes(result)
    assert "CHUNK_HEALTH_INSTRUCTIONAL_STARVED" not in _codes(result)
    assert result.action is None


def test_worked_example_opt_out_reverts_to_starved(monkeypatch, tmp_path) -> None:
    """With the worked-example credit OPTED OUT, the same example-heavy corpus
    reverts to the old behaviour and fires the C2 critical."""
    monkeypatch.setenv("ED4ALL_CHUNK_HEALTH_GATE", "1")
    monkeypatch.setenv("ED4ALL_CHUNK_HEALTH_WORKED_EXAMPLE_INSTRUCTIONAL", "0")
    chunks = _write(tmp_path, "chunks.jsonl", _worked_example_chunks(), jsonl=True)
    result = ChunkHealthValidator().validate({"chunks_path": chunks})
    assert result.passed is False
    assert "CHUNK_HEALTH_INSTRUCTIONAL_STARVED" in _crit_codes(result)


def test_bare_exercises_still_fail_c2(monkeypatch, tmp_path) -> None:
    """A corpus of BARE exercises / answer-dumps (no worked solutions) still
    fails C2 critical even under the corrected definition — nothing teaches."""
    monkeypatch.setenv("ED4ALL_CHUNK_HEALTH_GATE", "1")
    monkeypatch.delenv(
        "ED4ALL_CHUNK_HEALTH_WORKED_EXAMPLE_INSTRUCTIONAL", raising=False
    )
    chunks = _write(tmp_path, "chunks.jsonl", _bare_exercise_chunks(), jsonl=True)
    result = ChunkHealthValidator().validate({"chunks_path": chunks})
    assert result.passed is False
    assert "CHUNK_HEALTH_INSTRUCTIONAL_STARVED" in _crit_codes(result)


# --------------------------------------------------------------------------- #
# C10 recalibration: apparatus numbering is excluded
# --------------------------------------------------------------------------- #


def _instructional_prose_chunk(idx: int, text: str) -> Dict[str, Any]:
    return {
        "id": f"prose-{idx:03d}",
        "chunk_type": "explanation",
        "text": text,
        "html": "",
        "word_count": max(20, len(text.split())),
        "source": {"module_title": "Chapter Two Whole Numbers"},
    }


def test_c10_apparatus_numbering_not_flagged(monkeypatch, tmp_path) -> None:
    """Non-monotonic N.M numbering INSIDE apparatus chunks (example/exercise)
    is legitimate drill numbering, NOT a reading-order scramble → not flagged."""
    monkeypatch.setenv("ED4ALL_CHUNK_HEALTH_GATE", "1")
    # A healthy instructional-prose base (monotonic / no ordinals) so C2 passes.
    rows: List[Dict[str, Any]] = [
        _instructional_prose_chunk(
            i,
            "This section explains the reasoning behind the concept in full "
            "prose sentences so a learner understands it before drilling.",
        )
        for i in range(35)
    ]
    # Apparatus chunks whose EXERCISE/EXAMPLE numbering is non-monotonic.
    scramble = "Simplify 5.5 then 4.4 then 3.3 then 2.2 then 1.1 as drill items."
    rows.append(
        {
            "id": "ex-scramble",
            "chunk_type": "exercise",
            "text": scramble,
            "html": "",
            "word_count": 12,
            "source": {"module_title": "Chapter Two Whole Numbers"},
        }
    )
    rows.append(
        {
            "id": "example-scramble",
            "chunk_type": "example",
            "text": f"Example. Solution: {scramble}",
            "html": "",
            "word_count": 14,
            "source": {"module_title": "Chapter Two Whole Numbers"},
        }
    )
    chunks = _write(tmp_path, "chunks.jsonl", rows, jsonl=True)
    result = ChunkHealthValidator().validate({"chunks_path": chunks})
    assert "CHUNK_HEALTH_READING_ORDER" not in _codes(result), _codes(result)


def test_c10_prose_scramble_flagged(monkeypatch, tmp_path) -> None:
    """A GENUINE reading-order scramble in an instructional-PROSE chunk IS
    flagged (WARN) — non-monotonic ordinals in continuous prose is a real defect."""
    monkeypatch.setenv("ED4ALL_CHUNK_HEALTH_GATE", "1")
    rows: List[Dict[str, Any]] = [
        _instructional_prose_chunk(
            i,
            "This section explains the reasoning behind the concept in full "
            "prose sentences so a learner understands it before drilling.",
        )
        for i in range(35)
    ]
    # A prose chunk whose section cross-references run out of order.
    rows.append(
        _instructional_prose_chunk(
            999,
            "As shown in section 5.5 and revisited in 4.4, we then return to 3.3 "
            "and finally 2.2, which reverses the natural reading order badly.",
        )
    )
    chunks = _write(tmp_path, "chunks.jsonl", rows, jsonl=True)
    result = ChunkHealthValidator().validate({"chunks_path": chunks})
    reading = [i for i in result.issues if i.code == "CHUNK_HEALTH_READING_ORDER"]
    assert reading, _codes(result)
    assert reading[0].severity == "warning"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))

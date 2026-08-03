"""Phase IA WS1.4 — unit tests for lib/retrieval/gold_set.py.

Tier 1 (always-on, hermetic): builds a course layout in tmp_path from the
shared mini-course fixture (Executor A) and exercises every loader issue code
incl. the fail-without-fix corrupted-sha case.

The suite never reads operator course archives.  Its end-to-end checks use the
tracked neutral mini-course copied into ``tmp_path``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MINI_COURSE = PROJECT_ROOT / "tests" / "fixtures" / "retrieval" / "mini_course"

from lib.retrieval.gold_set import (  # noqa: E402
    GoldSetIssue,
    load_gold_set,
    validate_gold_set,
    has_critical_issues,
    critical_issues,
)


def _codes(issues):
    return {i.code for i in issues}


@pytest.fixture()
def mini_course_dir(tmp_path: Path) -> Path:
    """A writable copy of the mini-course fixture.

    The shared fixture ships its chunkset under the ``semantik_chunks/`` layout;
    the loader resolves it via the pinned ``chunkset.chunks_path``.
    """
    dst = tmp_path / "mini-course"
    (dst / "semantik_chunks").mkdir(parents=True)
    (dst / "retrieval_eval").mkdir(parents=True)
    shutil.copy(
        MINI_COURSE / "semantik_chunks" / "chunks.jsonl",
        dst / "semantik_chunks" / "chunks.jsonl",
    )
    shutil.copy(
        MINI_COURSE / "retrieval_eval" / "gold_set.json",
        dst / "retrieval_eval" / "gold_set.json",
    )
    return dst


# ---------------------------------------------------------------- clean load


def test_clean_load_zero_issues(mini_course_dir: Path):
    gold, issues = load_gold_set(mini_course_dir, verify=True)
    assert gold.get("schema_version") == "1.0"
    assert issues == [], [(_codes(issues))]
    assert not has_critical_issues(issues)


def test_verify_false_skips_chunkset_io(mini_course_dir: Path):
    # Remove the chunkset entirely; verify=False must still load + schema-check.
    (mini_course_dir / "semantik_chunks" / "chunks.jsonl").unlink()
    gold, issues = load_gold_set(mini_course_dir, verify=False)
    assert gold.get("schema_version") == "1.0"
    assert "GOLD_SET_CHUNKSET_NOT_FOUND" not in _codes(issues)
    assert not has_critical_issues(issues)


# ---------------------------------------------------------------- not found / json


def test_gold_set_not_found(tmp_path: Path):
    gold, issues = load_gold_set(tmp_path / "no-such-course", verify=True)
    assert gold == {}
    assert "GOLD_SET_NOT_FOUND" in _codes(issues)
    assert has_critical_issues(issues)


def test_invalid_json(mini_course_dir: Path):
    (mini_course_dir / "retrieval_eval" / "gold_set.json").write_text(
        "{ not valid json", encoding="utf-8"
    )
    gold, issues = load_gold_set(mini_course_dir, verify=True)
    assert gold == {}
    assert "GOLD_SET_INVALID_JSON" in _codes(issues)


# ---------------------------------------------------------------- sha mismatch


def test_corrupted_sha_detected(mini_course_dir: Path):
    """Fail-without-fix: tampering the pinned chunks.jsonl (changing its bytes
    without re-pinning) must surface GOLD_SET_CHUNKSET_SHA_MISMATCH. This is
    the contract WS2's eval harness consumes — an eval refuses to run against
    a drifted corpus."""
    chunks_path = mini_course_dir / "semantik_chunks" / "chunks.jsonl"
    # Append a byte so the recomputed sha no longer matches the pinned one.
    with chunks_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "extra", "text": "x", "source": {}}) + "\n")
    gold, issues = load_gold_set(mini_course_dir, verify=True)
    assert "GOLD_SET_CHUNKSET_SHA_MISMATCH" in _codes(issues)
    assert has_critical_issues(issues)


def test_chunkset_not_found(mini_course_dir: Path):
    (mini_course_dir / "semantik_chunks" / "chunks.jsonl").unlink()
    gold, issues = load_gold_set(mini_course_dir, verify=True)
    assert "GOLD_SET_CHUNKSET_NOT_FOUND" in _codes(issues)


# ---------------------------------------------------------------- unknown chunk


def test_unknown_chunk_id(mini_course_dir: Path):
    gold_path = mini_course_dir / "retrieval_eval" / "gold_set.json"
    doc = json.loads(gold_path.read_text(encoding="utf-8"))
    doc["questions"][0]["relevant_passages"][0]["chunk_id"] = "no_such_chunk"
    gold_path.write_text(json.dumps(doc), encoding="utf-8")
    _, issues = load_gold_set(mini_course_dir, verify=True)
    assert "GOLD_SET_UNKNOWN_CHUNK_ID" in _codes(issues)


# ---------------------------------------------------------------- quote not in chunk


def test_quote_not_in_chunk(mini_course_dir: Path):
    gold_path = mini_course_dir / "retrieval_eval" / "gold_set.json"
    doc = json.loads(gold_path.read_text(encoding="utf-8"))
    doc["questions"][0]["relevant_passages"][0]["anchor"]["text_quote"] = (
        "this exact sentence is definitely not present in any mini chunk text at all"
    )
    gold_path.write_text(json.dumps(doc), encoding="utf-8")
    _, issues = load_gold_set(mini_course_dir, verify=True)
    assert "GOLD_SET_QUOTE_NOT_IN_CHUNK" in _codes(issues)


# ---------------------------------------------------------------- duplicate id


def test_duplicate_question_id(mini_course_dir: Path):
    gold_path = mini_course_dir / "retrieval_eval" / "gold_set.json"
    doc = json.loads(gold_path.read_text(encoding="utf-8"))
    # Duplicate the first question (same id appears twice).
    doc["questions"].append(json.loads(json.dumps(doc["questions"][0])))
    gold_path.write_text(json.dumps(doc), encoding="utf-8")
    _, issues = load_gold_set(mini_course_dir, verify=True)
    assert "GOLD_SET_DUPLICATE_QUESTION_ID" in _codes(issues)


# ---------------------------------------------------------------- warnings


def test_type_imbalance_warning_fires_at_scale():
    """GOLD_SET_TYPE_IMBALANCE fires only once len(questions) >= 30 and some
    type is < 10%. Pure-function test on validate_gold_set."""
    questions = []
    # 29 factual_recall + 1 where_covered + ... build 31 total with where_covered
    # at 1/31 < 10%.
    for n in range(30):
        questions.append(
            {
                "question_id": f"gq-x-{n:04d}",
                "question_text": "long enough question text",
                "question_type": "factual_recall",
                "relevant_passages": [],
            }
        )
    questions.append(
        {
            "question_id": "gq-x-0030",
            "question_text": "long enough question text",
            "question_type": "where_covered",
            "relevant_passages": [],
        }
    )
    gold = {"chunkset": {"chunks_sha256": "a" * 64}, "questions": questions}
    issues = validate_gold_set(gold, chunks_by_id={}, chunks_sha256="a" * 64)
    codes = _codes(issues)
    assert "GOLD_SET_TYPE_IMBALANCE" in codes
    # conceptual_synthesis = 0% and where_covered ~3% both flagged
    imbalance = [i for i in issues if i.code == "GOLD_SET_TYPE_IMBALANCE"]
    assert imbalance and all(i.severity == "warning" for i in imbalance)


def test_type_imbalance_not_fired_for_small_seed():
    """A 10-question seed set is NOT flagged for imbalance (below the 30-q
    scale gate)."""
    questions = [
        {
            "question_id": f"gq-x-{n:04d}",
            "question_text": "long enough question text",
            "question_type": "factual_recall",
            "relevant_passages": [],
        }
        for n in range(10)
    ]
    gold = {"chunkset": {"chunks_sha256": "a" * 64}, "questions": questions}
    issues = validate_gold_set(gold, chunks_by_id={}, chunks_sha256="a" * 64)
    assert "GOLD_SET_TYPE_IMBALANCE" not in _codes(issues)


def test_ambiguous_quote_warning():
    """A text_quote contained in > 3 chunks of the pinned set fires
    GOLD_SET_AMBIGUOUS_QUOTE (warning)."""
    shared = "this boilerplate sentence appears verbatim in many different chunks here"
    chunks_by_id = {
        f"c{i}": {"id": f"c{i}", "text": f"intro {shared} outro {i}"}
        for i in range(5)
    }
    gold = {
        "chunkset": {"chunks_sha256": "a" * 64},
        "questions": [
            {
                "question_id": "gq-x-0001",
                "question_text": "a sufficiently long question",
                "question_type": "factual_recall",
                "relevant_passages": [
                    {
                        "chunk_id": "c0",
                        "relevance": "primary",
                        "anchor": {"item_path": "p.html", "text_quote": shared},
                    }
                ],
            }
        ],
    }
    issues = validate_gold_set(gold, chunks_by_id, chunks_sha256="a" * 64)
    codes = _codes(issues)
    assert "GOLD_SET_AMBIGUOUS_QUOTE" in codes
    amb = [i for i in issues if i.code == "GOLD_SET_AMBIGUOUS_QUOTE"]
    assert amb and all(i.severity == "warning" for i in amb)
    # the ambiguity finding itself is non-critical (it doesn't carry a
    # critical code), so it never fails the load on its own.
    assert not any(i.code == "GOLD_SET_AMBIGUOUS_QUOTE" and i.severity == "critical" for i in issues)


def test_tracked_seed_gold_set_zero_critical(mini_course_dir: Path):
    """The neutral tracked seed loads and resolves without operator data."""
    gold, issues = load_gold_set(mini_course_dir, verify=True)
    crit = critical_issues(issues)
    assert not crit, [f"{i.code}: {i.message}" for i in crit]
    assert len(gold["questions"]) >= 1
    assert gold.get("schema_version") in {"1.0", "1.1", "1.2"}
    assert gold["chunkset"]["kind"] in {"imscc", "semantik", "dart", "corpus"}


def test_tracked_seed_quotes_resolve_via_text_helpers(mini_course_dir: Path):
    """Independently verify every fixture text_quote is normalized-contained in
    its cited chunk using the citation-anchor _text helpers (the same
    normalization the sweep/eval harness uses)."""
    from lib.retrieval._text import normalize_ws

    course_dir = mini_course_dir
    gold_path = course_dir / "retrieval_eval" / "gold_set.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    chunks_rel = gold["chunkset"]["chunks_path"]
    chunks_by_id = {}
    with (course_dir / chunks_rel).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                chunks_by_id[rec["id"]] = rec

    n_quotes = 0
    for q in gold["questions"]:
        for p in q["relevant_passages"]:
            cid = p["chunk_id"]
            quote = p["anchor"]["text_quote"]
            assert len(quote) >= 40, f"{q['question_id']} quote < 40 chars"
            assert cid in chunks_by_id, f"{q['question_id']} unknown chunk {cid}"
            chunk_text = normalize_ws(chunks_by_id[cid]["text"]).lower()
            assert normalize_ws(quote).lower() in chunk_text, (
                f"{q['question_id']} quote not in {cid}: {quote!r}"
            )
            n_quotes += 1
    # >= one primary per question; count is variable post-repin/scale-up.
    assert n_quotes >= len(gold["questions"])


def test_tracked_seed_questions_present(mini_course_dir: Path):
    gp = mini_course_dir / "retrieval_eval" / "gold_set.json"
    assert len(json.loads(gp.read_text(encoding="utf-8"))["questions"]) >= 1

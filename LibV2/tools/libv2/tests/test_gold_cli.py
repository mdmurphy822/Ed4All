"""retrieval-answer-eval-set P0/P1 — CliRunner tests for the gold-set CLI surface.

Covers ``libv2 gold-validate`` (+ --coverage) and ``libv2 gold-repin``. Builds a
synthetic union-corpus course in tmp_path (no real course data, no LLM, no
index) and drives the commands via the established CliRunner pattern.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LibV2.tools.libv2.cli import main as libv2_main  # noqa: E402
from lib.retrieval._text import normalize_ws  # noqa: E402
from lib.utils import sha256_file  # noqa: E402

_SLUG = "demo-union-101"


def _csha(text: str) -> str:
    return hashlib.sha256(normalize_ws(text).lower().encode("utf-8")).hexdigest()


_TEXT_A = "A vector store indexes high-dimensional embedding vectors for retrieval."
_TEXT_B = "To add or subtract fractions, they must share a common denominator."


def _make_course(repo_root: Path, *, kind="corpus", rel="corpus/chunks.jsonl",
                 stale_pin=False, frozen=False) -> Path:
    cdir = repo_root / "courses" / _SLUG
    chunks_dir = cdir / Path(rel).parent
    chunks_dir.mkdir(parents=True)
    chunks = [
        {"id": "c001", "text": _TEXT_A,
         "source": {"item_path": "week_01/a.html"}, "learning_outcome_refs": ["to-01"]},
        {"id": "c002", "text": _TEXT_B,
         "source": {"item_path": "textbook.html",
                    "source_references": [{"sourceId": "dart:tb#1"}]},
         "learning_outcome_refs": ["co-01"]},
    ]
    chunks_path = cdir / rel
    with chunks_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")

    pin_sha = "0" * 64 if stale_pin else sha256_file(chunks_path)
    gold = {
        "schema_version": "1.1",
        "course_slug": _SLUG,
        "chunkset": {"kind": kind, "chunks_path": rel, "chunks_sha256": pin_sha},
        "authored_at": "2026-06-11T00:00:00Z",
        "frozen": frozen,
        "questions": [
            {"question_id": "gq-demo-union-101-0001", "question_type": "factual_recall",
             "question_text": "what does a vector store index?", "difficulty": "easy",
             "relevant_passages": [{
                 "chunk_id": "c001", "relevance": "primary",
                 "anchor": {"item_path": "week_01/a.html", "text_quote": _TEXT_A}}],
             "authoring": {"method": "manual", "author": "@t",
                           "reviewed_by": "PENDING_REVIEW", "status": "seed"}},
            {"question_id": "gq-demo-union-101-0002", "question_type": "factual_recall",
             "question_text": "what must fractions share?", "difficulty": "medium",
             "relevant_passages": [{
                 "chunk_id": "c002", "relevance": "primary",
                 "anchor": {"item_path": "textbook.html", "text_quote": _TEXT_B}}],
             "authoring": {"method": "manual", "author": "@t",
                           "reviewed_by": "PENDING_REVIEW", "status": "seed"}},
        ],
    }
    eval_dir = cdir / "retrieval_eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "gold_set.json").write_text(json.dumps(gold))
    (cdir / "manifest.json").write_text(json.dumps({"classification": {}}))
    return cdir


def _invoke(repo_root: Path, *args: str):
    return CliRunner().invoke(libv2_main, ["--repo", str(repo_root), *args])


# ---------------------------------------------------------------- gold-validate


def test_gold_validate_clean(tmp_path: Path):
    _make_course(tmp_path)
    res = _invoke(tmp_path, "gold-validate", "--course", _SLUG)
    assert res.exit_code == 0, res.output
    assert "gold set valid" in res.output
    assert "schema_version 1.1" in res.output


def test_gold_validate_stale_pin_fails(tmp_path: Path):
    _make_course(tmp_path, stale_pin=True)
    res = _invoke(tmp_path, "gold-validate", "--course", _SLUG)
    assert res.exit_code == 1
    assert "GOLD_SET_CHUNKSET_SHA_MISMATCH" in res.output


def test_gold_validate_unknown_course(tmp_path: Path):
    res = _invoke(tmp_path, "gold-validate", "--course", "no-such")
    assert res.exit_code == 1
    assert "course not found" in res.output


def test_gold_validate_coverage(tmp_path: Path):
    course = _make_course(tmp_path)
    res = _invoke(tmp_path, "gold-validate", "--course", _SLUG, "--coverage")
    assert res.exit_code == 0, res.output
    assert "Coverage matrix" in res.output
    # coverage artifact written
    arts = list((course / "retrieval_eval").glob("gold_coverage_*.json"))
    assert arts


def test_gold_validate_coverage_no_write(tmp_path: Path):
    course = _make_course(tmp_path)
    res = _invoke(tmp_path, "gold-validate", "--course", _SLUG,
                  "--coverage", "--no-coverage-write")
    assert res.exit_code == 0, res.output
    assert "Coverage matrix" in res.output
    assert not list((course / "retrieval_eval").glob("gold_coverage_*.json"))


# ---------------------------------------------------------------- gold-repin


def test_gold_repin_dart_to_corpus(tmp_path: Path):
    # course pinned to a (stale) corpus already; repin to its own corpus.
    course = _make_course(tmp_path, stale_pin=True)
    res = _invoke(tmp_path, "gold-repin", "--course", _SLUG, "--kind", "corpus")
    assert res.exit_code == 0, res.output
    assert "re-pin" in res.output
    # now validates clean
    res2 = _invoke(tmp_path, "gold-validate", "--course", _SLUG)
    assert res2.exit_code == 0, res2.output


def test_gold_repin_dry_run(tmp_path: Path):
    course = _make_course(tmp_path, stale_pin=True)
    before = (course / "retrieval_eval" / "gold_set.json").read_text()
    res = _invoke(tmp_path, "gold-repin", "--course", _SLUG, "--kind", "corpus", "--dry-run")
    assert res.exit_code == 0, res.output
    assert "dry-run" in res.output
    assert (course / "retrieval_eval" / "gold_set.json").read_text() == before


def test_gold_repin_frozen_refused(tmp_path: Path):
    _make_course(tmp_path, frozen=True, stale_pin=True)
    res = _invoke(tmp_path, "gold-repin", "--course", _SLUG, "--kind", "corpus")
    assert res.exit_code == 1
    assert "GOLD_SET_FROZEN" in res.output


def test_gold_repin_frozen_unfreeze(tmp_path: Path):
    _make_course(tmp_path, frozen=True, stale_pin=True)
    res = _invoke(tmp_path, "gold-repin", "--course", _SLUG, "--kind", "corpus",
                  "--unfreeze-for-repin")
    assert res.exit_code == 0, res.output

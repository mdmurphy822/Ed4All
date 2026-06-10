"""Tests for the read-only chunk-size/overlap retrieval sweep harness.

Hermetic: every test materialises the shared mini-course fixture into a
``tmp_path`` LibV2 layout and points ``ED4ALL_LIBV2_ROOT`` at it, so the sweep
resolves the course exactly as it would a real archived course without
touching the committed fixture tree. Deterministic, no LLM, no network.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from Trainforge.scripts import retrieval_chunk_sweep as sweep

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "retrieval"
    / "mini_course"
)
COURSE_SLUG = "mini-retrieval-101"
COURSE_CODE = "mini-retrieval-101"


@pytest.fixture()
def mini_course(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy the mini-course fixture into a tmp LibV2 layout; return course dir."""
    if not FIXTURE_ROOT.exists():  # pragma: no cover - fixture is committed
        pytest.skip(f"mini-course fixture missing at {FIXTURE_ROOT}")
    libv2_root = tmp_path / "LibV2"
    course_dir = libv2_root / "courses" / COURSE_SLUG
    shutil.copytree(FIXTURE_ROOT, course_dir)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    return course_dir


def _run_sweep(course_dir: Path, *, overlaps=(0, 1)) -> dict:
    params = sweep.SweepParams(
        course_code=COURSE_CODE,
        course_slug=COURSE_SLUG,
        source_kind="dart",
        max_chunk_sizes=(120, 200),
        target_chunk_sizes=(60,),
        overlap_sentences=tuple(overlaps),
        ks=(1, 3, 5),
        min_chunk_size=20,
    )
    gold_path = course_dir / "retrieval_eval" / "gold_set.json"
    return sweep.run_sweep(params, gold_path)


# --------------------------------------------------------------------------- #
# 1. Sweep runs on the mini fixture
# --------------------------------------------------------------------------- #
def test_sweep_runs_on_mini_fixture(mini_course: Path):
    """A 2×1×2 grid over the mini course + its 3-q gold set → 4 rows, metrics in [0,1]."""
    report = _run_sweep(mini_course)
    assert report["schema_version"] == "1.0"
    assert report["course_slug"] == COURSE_SLUG
    assert report["n_questions"] == 3
    assert len(report["results"]) == 4  # 2 max × 1 target × 2 overlap
    for row in report["results"]:
        assert row["n_chunks"] >= 1
        assert 0.0 <= row["mrr"] <= 1.0
        for v in row["recall_at"].values():
            assert 0.0 <= v <= 1.0
        assert set(row["recall_at"].keys()) == {"1", "3", "5"}
    # Baseline from the canonical on-disk chunkset is present (fixture ships
    # dart_chunks/chunks.jsonl).
    assert report["baseline"] is not None
    assert report["baseline"]["params"] == "canonical-on-disk"
    # Two-consumer gate note is surfaced.
    assert "KG eval" in report["note"]


# --------------------------------------------------------------------------- #
# 2. Sweep never touches the canonical chunkset
# --------------------------------------------------------------------------- #
def test_sweep_never_touches_canonical_chunkset(mini_course: Path):
    """sha256 of the fixture's dart_chunks/chunks.jsonl is identical before/after."""
    chunks_path = mini_course / "dart_chunks" / "chunks.jsonl"
    before = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    _run_sweep(mini_course)
    after = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    assert before == after
    # And it matches the gold set's frozen pin (Executor A's fixture).
    assert before == "fc8a1c651e2cdf7eb6d5aecb57ddc7244f71eebf805d34a88684210685b03a0f"


# --------------------------------------------------------------------------- #
# 3. No chunker-default mutation
# --------------------------------------------------------------------------- #
def test_no_chunker_default_mutation(mini_course: Path):
    """The chunker module size constants are unchanged after import + run."""
    import Trainforge.chunker as ck
    import Trainforge.chunker.chunker as ckmod

    before = (
        ck.MIN_CHUNK_SIZE,
        ck.MAX_CHUNK_SIZE,
        ck.TARGET_CHUNK_SIZE,
        ckmod.MIN_CHUNK_SIZE,
        ckmod.MAX_CHUNK_SIZE,
        ckmod.TARGET_CHUNK_SIZE,
    )
    _run_sweep(mini_course)
    after = (
        ck.MIN_CHUNK_SIZE,
        ck.MAX_CHUNK_SIZE,
        ck.TARGET_CHUNK_SIZE,
        ckmod.MIN_CHUNK_SIZE,
        ckmod.MAX_CHUNK_SIZE,
        ckmod.TARGET_CHUNK_SIZE,
    )
    assert before == after
    # Canonical defaults are the documented values (sanity that we read the
    # real constants, not a shadow copy).
    assert (ck.MIN_CHUNK_SIZE, ck.MAX_CHUNK_SIZE, ck.TARGET_CHUNK_SIZE) == (100, 800, 500)


# --------------------------------------------------------------------------- #
# 4. Overlap post-pass unit test
# --------------------------------------------------------------------------- #
def _mk_chunk(cid: str, item_path: str, text: str) -> dict:
    return {
        "id": cid,
        "text": text,
        "word_count": len(text.split()),
        "source": {"item_path": item_path},
    }


def test_overlap_post_pass_identity():
    """n_sentences=0 is the identity (text unchanged) and never mutates input."""
    chunks = [
        _mk_chunk("c1", "p.html", "First. Second. Third."),
        _mk_chunk("c2", "p.html", "Fourth. Fifth."),
    ]
    original_texts = [c["text"] for c in chunks]
    out = sweep._apply_sentence_overlap(chunks, 0)
    assert [c["text"] for c in out] == original_texts
    # Input chunks are not mutated.
    assert [c["text"] for c in chunks] == original_texts


def test_overlap_post_pass_prepends_previous_sentence():
    """n_sentences=1: chunk i's text starts with the last sentence of i-1."""
    chunks = [
        _mk_chunk("c1", "p.html", "Alpha one. Alpha two. Alpha three."),
        _mk_chunk("c2", "p.html", "Beta one. Beta two."),
    ]
    out = sweep._apply_sentence_overlap(chunks, 1)
    assert out[0]["text"] == "Alpha one. Alpha two. Alpha three."  # first chunk unchanged
    assert out[1]["text"].startswith("Alpha three.")
    assert "Beta one." in out[1]["text"]
    assert out[1]["word_count"] == len(out[1]["text"].split())


def test_overlap_post_pass_respects_item_path_boundary():
    """Overlap never crosses an item_path boundary (no cross-page bleed)."""
    chunks = [
        _mk_chunk("c1", "page_a.html", "A one. A two."),
        _mk_chunk("c2", "page_b.html", "B one. B two."),
    ]
    out = sweep._apply_sentence_overlap(chunks, 1)
    # c2 is on a different page → its text must NOT be prefixed with page_a text.
    assert out[1]["text"] == "B one. B two."
    assert "A two." not in out[1]["text"]


def test_overlap_negative_rejected():
    with pytest.raises(ValueError):
        sweep._apply_sentence_overlap([], -1)


# --------------------------------------------------------------------------- #
# 5. Report determinism
# --------------------------------------------------------------------------- #
def _strip_volatile(report: dict) -> dict:
    """Drop generated_at + timing fields for a stable equality check."""
    out = json.loads(json.dumps(report))  # deep copy
    out.pop("generated_at", None)
    for row in out.get("results", []):
        row.pop("index_seconds", None)
        row.pop("query_seconds_p50", None)
    if out.get("baseline"):
        out["baseline"].pop("index_seconds", None)
        out["baseline"].pop("query_seconds_p50", None)
    return out


def test_report_deterministic(mini_course: Path):
    """Two runs produce identical reports modulo generated_at + timing fields."""
    r1 = _run_sweep(mini_course)
    r2 = _run_sweep(mini_course)
    assert _strip_volatile(r1) == _strip_volatile(r2)


# --------------------------------------------------------------------------- #
# 6. Parameter validation
# --------------------------------------------------------------------------- #
def test_parse_int_csv_rejects_non_positive():
    with pytest.raises(ValueError):
        sweep._parse_int_csv("400,-1", flag="--max-chunk-sizes")


def test_parse_int_csv_allows_zero_overlap():
    assert sweep._parse_int_csv("0,1,2", flag="--overlap-sentences", positive=False) == [
        0,
        1,
        2,
    ]


def test_parse_int_csv_rejects_empty():
    with pytest.raises(ValueError):
        sweep._parse_int_csv("", flag="--ks")


def test_parse_int_csv_rejects_non_int():
    with pytest.raises(ValueError):
        sweep._parse_int_csv("abc", flag="--ks")


def test_parse_params_rejects_target_above_max(monkeypatch: pytest.MonkeyPatch):
    parser = sweep.build_arg_parser()
    args = parser.parse_args(
        [
            "--course-code",
            "x",
            "--max-chunk-sizes",
            "400",
            "--target-chunk-sizes",
            "800",
        ]
    )
    with pytest.raises(ValueError, match="exceeds"):
        sweep.parse_params(args)


def test_main_missing_gold_set_returns_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A non-existent gold set is an operator error (exit 2), not a crash."""
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(tmp_path / "LibV2"))
    rc = sweep.main(["--course-code", "does-not-exist"])
    assert rc == 2


def test_cli_writes_report(mini_course: Path, tmp_path: Path):
    """End-to-end CLI: main() writes a parseable report and returns 0."""
    out_path = tmp_path / "report.json"
    rc = sweep.main(
        [
            "--course-code",
            COURSE_CODE,
            "--source-kind",
            "dart",
            "--max-chunk-sizes",
            "120,200",
            "--target-chunk-sizes",
            "60",
            "--overlap-sentences",
            "0,1",
            "--ks",
            "1,3,5",
            "--min-chunk-size",
            "20",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    report = json.loads(out_path.read_text())
    assert report["schema_version"] == "1.0"
    assert len(report["results"]) == 4

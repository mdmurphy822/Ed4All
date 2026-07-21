"""Unit tests for scripts/ocr_recall_ab.py.

All fixtures are tiny synthetic gold chunks / extraction dicts built in-test — no
course data is read from disk (repo rule: tracked tests must not reference course
data). The .json-vs-.json path needs NO SemantiK deps; the PDF path's
``extract_shared_cached`` boundary is faked via sys.modules so the env-hygiene
test runs without SemantiK installed.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ocr_recall_ab as ab  # noqa: E402
import gold_compare as gc  # noqa: E402


# ---- synthetic fixtures --------------------------------------------------

SEC_A = (
    "The counting numbers are one two three and so on used to count objects. "
    "Whole numbers add zero to the counting numbers forming a complete set."
)
SEC_B = (
    "A variable is a letter that stands for a number we do not yet know. "
    "An algebraic expression combines variables and constants using operations."
)


def _chunk(cid, text, chapter, section):
    return {
        "id": cid,
        "text": text,
        "source": {
            "item_path": f"synthetic-book-ch{chapter:02d}_accessible.html",
            "source_references": [
                {
                    "sourceId": f"semantik:synthetic-book-ch{chapter:02d}_accessible#{section}",
                    "role": "primary",
                }
            ],
        },
    }


def _write_gold(tmp_path) -> Path:
    p = tmp_path / "gold.jsonl"
    rows = [
        _chunk("g1", SEC_A, 1, "1-1-whole-numbers"),
        _chunk("g2", SEC_B, 1, "1-2-variables"),
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def _shared(text: str) -> dict:
    return {"pages": [{"merged": {"text_blocks": [{"text": text}]}}]}


def _write_json_side(tmp_path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(_shared(text)), encoding="utf-8")
    return p


# ---- extraction_text -----------------------------------------------------


def test_extraction_text_joins_merged():
    shared = {
        "pages": [
            {"merged": {"text_blocks": [{"text": "hello"}, {"text": "world"}]}},
            {"merged": {"text_blocks": [{"text": "again"}]}},
        ]
    }
    out = ab.extraction_text(shared)
    assert "hello" in out and "world" in out and "again" in out


def test_extraction_text_fallback_when_no_merged():
    shared = {"pages": [{"tesseract": {"text_blocks": [{"text": "ocr text here"}]}}]}
    assert "ocr text here" in ab.extraction_text(shared)


# ---- scoring + gate ------------------------------------------------------


def test_score_side_recall_and_word_coverage(tmp_path):
    gold = gc.load_gold(_write_gold(tmp_path))
    gch = gold[1]
    full_text = SEC_A + " " + SEC_B
    score = ab.score_side(gch, full_text)
    assert score["recall_mean"] == pytest.approx(1.0, abs=1e-6)
    assert score["word_coverage"] == pytest.approx(1.0, abs=1e-6)


def test_compare_gate_sign_and_invariant(tmp_path):
    gold_path = _write_gold(tmp_path)
    full = SEC_A + " " + SEC_B
    degraded = "the counting numbers are one"  # loses almost every gold shingle

    side_worse = (degraded, {"label": "A", "source": "a"})
    side_better = (full, {"label": "B", "source": "b"})
    report = ab.compare(gold_path, 1, side_worse, side_better)
    assert report["schema_version"] == ab.SCHEMA_VERSION
    assert report["deltas"]["recall_mean_delta"] > 0.03
    assert report["gate"]["passed"] is True
    # gate is exactly the +3pt rule.
    assert report["gate"]["passed"] == (
        report["deltas"]["recall_mean_delta"] >= ab.GATE_THRESHOLD
    )

    # identical sides -> zero delta -> gate fails (below +3pt).
    same = ab.compare(gold_path, 1, (full, {"label": "A", "source": "a"}),
                      (full, {"label": "B", "source": "b"}))
    assert same["deltas"]["recall_mean_delta"] == pytest.approx(0.0, abs=1e-9)
    assert same["gate"]["passed"] is False


def test_compare_missing_chapter_hard_errors(tmp_path):
    gold_path = _write_gold(tmp_path)
    with pytest.raises(SystemExit):
        ab.compare(gold_path, 99, ("x", {}), ("y", {}))


# ---- json-side end-to-end (no SemantiK deps) -----------------------------


def test_json_side_e2e(tmp_path):
    gold_path = _write_gold(tmp_path)
    full = SEC_A + " " + SEC_B
    a = _write_json_side(tmp_path, "a.json", "the counting numbers are one")
    b = _write_json_side(tmp_path, "b.json", full)

    side_a = ab.run_side(a, None, None, None)
    side_b = ab.run_side(b, None, None, None)
    assert side_a[1]["kind"] == "json"
    assert side_b[1]["kind"] == "json"
    side_a[1]["label"] = "A"
    side_b[1]["label"] = "B"

    report = ab.compare(gold_path, 1, side_a, side_b)
    assert report["deltas"]["recall_mean_delta"] > 0
    assert set(report["sides"]) == {"a", "b"}
    assert "word_coverage_delta" in report["deltas"]


# ---- side_cache_dir ------------------------------------------------------


def test_side_cache_dir_default_when_empty():
    assert ab.side_cache_dir(None, "", None) is None


def test_side_cache_dir_distinct_for_distinct_configs(tmp_path):
    d1 = ab.side_cache_dir(tmp_path, "--oem 1", None)
    d2 = ab.side_cache_dir(tmp_path, "--oem 0", None)
    assert d1 is not None and d2 is not None
    assert d1 != d2


def test_side_cache_dir_stable_for_equal_configs(tmp_path):
    d1 = ab.side_cache_dir(tmp_path, "--oem 1", "words.txt")
    d2 = ab.side_cache_dir(tmp_path, "--oem 1", "words.txt")
    assert d1 == d2


# ---- env hygiene (PDF side) ----------------------------------------------


def _install_fake_extract(monkeypatch, capture: dict, *, raise_after: bool = False):
    """Inject a fake semantik_structure.extract_shared with a capturing
    extract_shared_cached, so run_side's lazy import resolves without SemantiK."""
    pkg = types.ModuleType("semantik_structure")
    mod = types.ModuleType("semantik_structure.extract_shared")

    def extract_shared_cached(pdf_path, cache_dir=None):
        capture["config"] = os.environ.get(ab._ENV_CONFIG)
        capture["user_words"] = os.environ.get(ab._ENV_USER_WORDS)
        capture["cache_dir"] = cache_dir
        if raise_after:
            raise RuntimeError("boom")
        return _shared("some extracted text")

    mod.extract_shared_cached = extract_shared_cached
    pkg.extract_shared = mod
    monkeypatch.setitem(sys.modules, "semantik_structure", pkg)
    monkeypatch.setitem(sys.modules, "semantik_structure.extract_shared", mod)


def test_run_side_pdf_sets_and_restores_env(monkeypatch, tmp_path):
    capture: dict = {}
    _install_fake_extract(monkeypatch, capture)
    monkeypatch.setenv(ab._ENV_CONFIG, "PRIOR_CONFIG")
    monkeypatch.delenv(ab._ENV_USER_WORDS, raising=False)

    pdf = tmp_path / "scan.pdf"  # never opened by the fake
    text, meta = ab.run_side(pdf, "--oem 1", None, tmp_path)
    # the config reached extract at call time
    assert capture["config"] == "--oem 1"
    assert capture["user_words"] is None
    # prior env restored afterward
    assert os.environ[ab._ENV_CONFIG] == "PRIOR_CONFIG"
    assert meta["kind"] == "pdf"
    assert "some extracted text" in text


def test_run_side_pdf_restores_env_on_failure(monkeypatch, tmp_path):
    capture: dict = {}
    _install_fake_extract(monkeypatch, capture, raise_after=True)
    monkeypatch.setenv(ab._ENV_CONFIG, "PRIOR_CONFIG")

    pdf = tmp_path / "scan.pdf"
    with pytest.raises(RuntimeError):
        ab.run_side(pdf, "--oem 1", None, tmp_path)
    # restored even though extract raised
    assert os.environ[ab._ENV_CONFIG] == "PRIOR_CONFIG"


# ---- reuse guard: shingle primitives ARE gold_compare's own --------------


def test_shingle_primitives_are_gold_compare_objects():
    # ocr_recall_ab must IMPORT gold_compare's shingle math, not re-implement it.
    assert ab.gc is gc
    assert ab.gc.metric_recall is gc.metric_recall
    assert ab.gc.ngram_set is gc.ngram_set
    assert ab.gc.tokenize is gc.tokenize

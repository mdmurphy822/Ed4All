"""Tests for cross-course hard-negative mining (W4.4).

Deterministic + CI-safe: NO model, NO network, NO embedding index. Course dirs
are tiny synthetic chunksets in tmp; the optional target dry-runs use an
injected deterministic fake retriever.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.retrieval import cross_course_negatives as ccn
from lib.tests.test_answer_composer import SpyCapture


def _write_course(libv2_root: Path, slug: str, *, domain: str, chunks):
    course_dir = libv2_root / "courses" / slug
    (course_dir / "dart_chunks").mkdir(parents=True, exist_ok=True)
    with (course_dir / "dart_chunks" / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    (course_dir / "manifest.json").write_text(
        json.dumps({"classification": {"primary_domain": domain}}), encoding="utf-8"
    )
    return course_dir


def _chunk(cid, topic, *, ctype="explanation", wc=40):
    return {
        "id": cid,
        "text": f"This passage explains {topic} in detail with several sentences.",
        "chunk_type": ctype,
        "word_count": wc,
        "concept_tags": [topic],
        "source": {"section_heading": f"About {topic}"},
    }


@pytest.fixture()
def library(tmp_path):
    root = tmp_path / "LibV2"
    _write_course(root, "target", domain="physics",
                  chunks=[_chunk("t0", "kinematics")])
    _write_course(root, "chem", domain="chemistry",
                  chunks=[_chunk("chem0", "titration"), _chunk("chem1", "moles"),
                          _chunk("chem2", "acids"), _chunk("chem3", "bases")])
    _write_course(root, "hist", domain="history",
                  chunks=[_chunk("h0", "renaissance"), _chunk("h1", "reformation")])
    return root


# --------------------------------------------------------------------------- #
# Flag
# --------------------------------------------------------------------------- #


def test_resolve_default_off(monkeypatch):
    monkeypatch.delenv(ccn.ENV_CROSS_COURSE_NEGATIVES, raising=False)
    assert ccn.resolve_cross_course_negatives() is False
    monkeypatch.setenv(ccn.ENV_CROSS_COURSE_NEGATIVES, "junk")
    assert ccn.resolve_cross_course_negatives() is False
    monkeypatch.setenv(ccn.ENV_CROSS_COURSE_NEGATIVES, "true")
    assert ccn.resolve_cross_course_negatives() is True
    assert ccn.resolve_cross_course_negatives(False) is False


# --------------------------------------------------------------------------- #
# Mining: real chunks, labeled, out-of-domain preferred
# --------------------------------------------------------------------------- #


def test_mine_produces_labeled_real_negatives(library):
    cap = SpyCapture()
    cands = ccn.mine_cross_course_negatives(
        library / "courses" / "target", libv2_root=library, n=6, capture=cap
    )
    assert cands, "expected cross-course negatives"
    for c in cands:
        assert c["category"] == "cross_course_negative"
        # provenance: a REAL foreign chunk id + course.
        assert c["source_course_slug"] in {"chem", "hist"}
        assert c["source_chunk_id"]
        assert c["source_excerpt"]
        assert c["authoring"]["reviewed_by"] == "PENDING_REVIEW"
        # never the target course itself.
        assert c["source_course_slug"] != "target"
    # decision capture fired.
    assert any(
        e["decision_type"] == ccn.DECISION_TYPE_CROSS_COURSE for e in cap.events
    )


def test_source_chunk_ids_are_real(library):
    cands = ccn.mine_cross_course_negatives(
        library / "courses" / "target", libv2_root=library, n=6
    )
    # Every mined source id must exist in its source course's chunkset.
    for c in cands:
        slug = c["source_course_slug"]
        path = library / "courses" / slug / "dart_chunks" / "chunks.jsonl"
        ids = {json.loads(ln)["id"] for ln in path.read_text().splitlines() if ln.strip()}
        assert c["source_chunk_id"] in ids


def test_round_robin_max_per_course(library):
    cands = ccn.mine_cross_course_negatives(
        library / "courses" / "target", libv2_root=library, n=10, max_per_course=2
    )
    from collections import Counter
    per = Counter(c["source_course_slug"] for c in cands)
    assert all(v <= 2 for v in per.values())
    # both foreign courses contribute (round-robin spread).
    assert set(per) == {"chem", "hist"}


def test_question_text_is_deterministic_template(library):
    cands = ccn.mine_cross_course_negatives(
        library / "courses" / "target", libv2_root=library, n=4
    )
    for c in cands:
        assert c["source_topic"] and c["source_topic"] in c["question_text"]


# --------------------------------------------------------------------------- #
# End-to-end doc + dry-runs
# --------------------------------------------------------------------------- #


def _fake_retrieve(libv2_root, query, *, course_slug, engine, limit):
    # Target course scores every cross-course probe low (out-of-domain).
    class _R:
        chunk_id = "t0"
        text = "kinematics body"
        score = 0.01
    return [_R()]


def test_generate_writes_doc_with_dry_runs(library):
    doc, path = ccn.generate_cross_course_negatives(
        library / "courses" / "target", libv2_root=library, n=4,
        retrieve_fn=_fake_retrieve, write=True,
    )
    assert path is not None and path.is_file()
    assert doc["target_course_slug"] == "target"
    assert doc["schema_version"] == "1.0"
    assert doc["authoring_run"]["n_probes"] == len(doc["probe_candidates"])
    for c in doc["probe_candidates"]:
        assert c["probe_id"].startswith("ccn-target-")
        # target dry-runs attached (should score low → refuse).
        assert "target_dry_runs" in c
        assert all(dr["top_score"] <= 0.5 for dr in c["target_dry_runs"])
    # on-disk doc is valid JSON.
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["target_course_slug"] == "target"


def test_no_other_courses_yields_empty(tmp_path):
    root = tmp_path / "LibV2"
    _write_course(root, "solo", domain="physics", chunks=[_chunk("s0", "x")])
    cands = ccn.mine_cross_course_negatives(
        root / "courses" / "solo", libv2_root=root, n=5
    )
    assert cands == []

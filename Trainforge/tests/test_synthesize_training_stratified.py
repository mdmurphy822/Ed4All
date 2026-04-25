#!/usr/bin/env python3
"""
Wave 77: stratified-sampling and misconception-DPO emission for
``Trainforge.synthesize_training``.

Covered contracts:
  - ``--stratify bloom`` produces a roughly-uniform bloom-level distribution
    across emitted pairs (drawn from the rdf-shacl-550 LibV2 archive).
  - ``--include-dpo-from-misconceptions`` emits >=67 DPO pairs for the
    rdf-shacl-550 corpus (which carries 147 valid editorial misconception
    entries -- the >=67 floor matches the spec's quoted minimum and absorbs
    any future corpus rebalancing).
  - Misconception DPO pair shape: ``{prompt, chosen, rejected}`` with all
    fields non-empty.
  - Same ``--seed`` regenerates the same output JSONL bytes.
  - ``--max-pairs N`` caps each artifact at N records.
  - LibV2-archive entry path (``run_synthesis_from_libv2``) reads
    ``corpus/chunks.jsonl`` straight from the archive without re-running
    the pipeline.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.synthesize_training import (  # noqa: E402
    _build_misconception_dpo_pair,
    _resolve_libv2_corpus_dir,
    _stratified_sample,
    _stratify_key,
    run_synthesis,
    run_synthesis_from_libv2,
)


RDF_SHACL_SLUG_CANDIDATES = (
    "rdf-shacl-550-rdf-shacl-550",
    "rdf-shacl-550",
)


def _rdf_shacl_archive() -> Path:
    libv2_root = PROJECT_ROOT / "LibV2" / "courses"
    for slug in RDF_SHACL_SLUG_CANDIDATES:
        candidate = libv2_root / slug
        if candidate.exists():
            return candidate
    pytest.skip("rdf-shacl-550 archive not present; integration test skipped")


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Unit tests for the new helpers (don't require the LibV2 archive)
# ---------------------------------------------------------------------------

def test_stratify_key_extracts_bloom_chunk_type_outcome():
    chunk = {
        "bloom_level": "Apply",
        "chunk_type": "Explanation",
        "learning_outcome_refs": ["co-01", "co-02"],
        "difficulty": "Foundational",
    }
    assert _stratify_key(chunk, "bloom") == "apply"
    assert _stratify_key(chunk, "chunk_type") == "explanation"
    assert _stratify_key(chunk, "outcome") == "co-01"
    assert _stratify_key(chunk, "difficulty") == "foundational"
    assert _stratify_key({}, "bloom") == "unknown"


def test_stratified_sample_balances_buckets():
    import random as _rand
    chunks = (
        [{"id": f"a{i}", "bloom_level": "remember"} for i in range(10)]
        + [{"id": f"b{i}", "bloom_level": "understand"} for i in range(2)]
        + [{"id": f"c{i}", "bloom_level": "apply"} for i in range(20)]
    )
    rng = _rand.Random(0)
    picked = _stratified_sample(chunks, ["bloom"], target_count=6, rng=rng)
    counts = Counter(c["bloom_level"] for c in picked)
    # Round-robin draws one per bucket per pass, so 6 picks across 3 buckets
    # MUST be exactly 2 per bucket.
    assert counts == {"remember": 2, "understand": 2, "apply": 2}, counts


def test_stratified_sample_falls_back_when_buckets_drain():
    import random as _rand
    chunks = (
        [{"id": f"a{i}", "bloom_level": "remember"} for i in range(5)]
        + [{"id": f"b{i}", "bloom_level": "understand"} for i in range(1)]
    )
    rng = _rand.Random(0)
    # Ask for 5 samples; bucket 'understand' empties after pass 1.
    picked = _stratified_sample(chunks, ["bloom"], target_count=5, rng=rng)
    counts = Counter(c["bloom_level"] for c in picked)
    assert counts["understand"] == 1
    assert counts["remember"] == 4
    assert sum(counts.values()) == 5


def test_build_misconception_dpo_pair_has_required_fields():
    chunk = {
        "id": "chunk_42",
        "bloom_level": "understand",
        "concept_tags": ["rdf-graph"],
        "learning_outcome_refs": ["co-01"],
    }
    mc = {
        "misconception": "RDF triples have implicit type columns like a SQL row.",
        "correction": "RDF triples carry no schema; types are explicit via rdf:type.",
    }
    pair = _build_misconception_dpo_pair(chunk, mc, pair_index=0)
    assert pair is not None
    for k in ("prompt", "chosen", "rejected", "chunk_id", "misconception_id"):
        assert pair[k], f"empty field {k!r} on misconception DPO pair"
    assert pair["chosen"] == mc["correction"]
    assert pair["rejected"] == mc["misconception"]
    assert pair["chosen"] != pair["rejected"]


def test_build_misconception_dpo_pair_drops_empty_sides():
    chunk = {"id": "x", "learning_outcome_refs": ["co-01"]}
    assert _build_misconception_dpo_pair(chunk, {"misconception": "", "correction": "y"}, 0) is None
    assert _build_misconception_dpo_pair(chunk, {"misconception": "x", "correction": ""}, 1) is None


def test_unknown_stratify_dimension_raises(tmp_path):
    """A typo in --stratify must raise ValueError, not silently no-op."""
    # Build a tiny ad-hoc corpus.
    corpus_dir = tmp_path / "course"
    (corpus_dir / "corpus").mkdir(parents=True)
    (corpus_dir / "training_specs").mkdir()
    chunk = {
        "id": "c1",
        "chunk_type": "explanation",
        "text": "x" * 200,
        "learning_outcome_refs": ["co-01"],
        "bloom_level": "remember",
        "concept_tags": ["a"],
    }
    with (corpus_dir / "corpus" / "chunks.jsonl").open("w") as fh:
        fh.write(json.dumps(chunk) + "\n")
    with pytest.raises(ValueError):
        run_synthesis(
            corpus_dir=corpus_dir,
            course_code="TEST_101",
            stratify=["nonsense"],
        )


# ---------------------------------------------------------------------------
# Integration tests against the rdf-shacl-550 LibV2 archive
# ---------------------------------------------------------------------------

def test_libv2_resolver_finds_rdf_shacl_archive():
    archive = _rdf_shacl_archive()
    # Resolver must find it whether the slug is canonical or doubled.
    resolved = _resolve_libv2_corpus_dir(archive.name)
    assert resolved == archive
    assert (resolved / "corpus" / "chunks.jsonl").exists()


def test_stratify_bloom_balances_pair_distribution(tmp_path):
    """--stratify bloom must yield a roughly-uniform bloom distribution.

    The rdf-shacl-550 corpus is heavily skewed toward 'apply' and
    'understand'; without stratification the output mirrors that skew.
    With round-robin stratification the bucket counts must be within 1 of
    each other (the round-robin invariant), bounded by the smallest
    bucket's size.
    """
    archive = _rdf_shacl_archive()
    out_dir = tmp_path / "out"

    # First read the corpus's per-bucket population so we can compute the
    # round-robin invariant properly for buckets that exhaust early.
    bucket_supply: dict[str, int] = Counter()
    with (archive / "corpus" / "chunks.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if not c.get("learning_outcome_refs"):
                continue
            bucket_supply[(c.get("bloom_level") or "unknown").lower()] += 1
    assert bucket_supply, "fixture has no eligible chunks"

    target = 60
    stats = run_synthesis_from_libv2(
        slug=archive.name,
        course_code="RDF_SHACL_550",
        provider="mock",
        seed=42,
        stratify=["bloom"],
        max_pairs=target,
        output_dir=out_dir,
    )
    bloom_dist = stats.stratify_distribution.get("bloom", {})
    assert bloom_dist, "stratify_distribution missing the bloom dimension"

    # Round-robin invariant for stratified sampling with bucket exhaustion:
    # in pass k each non-empty bucket has donated exactly k chunks, so
    # picked = min(supply_b, ceil_or_floor_of_passes). The largest bucket's
    # picked count must therefore be at most 1 above the next-largest
    # bucket whose supply was not exhausted, AND no bucket may exceed its
    # own supply.
    for bucket, picked in bloom_dist.items():
        assert picked <= bucket_supply[bucket], (
            f"bucket {bucket}: picked {picked} > supply {bucket_supply[bucket]}"
        )
    # Every non-empty supply bucket must be represented.
    assert set(bloom_dist.keys()) == set(bucket_supply.keys()), (
        f"missing buckets: supply={set(bucket_supply)} picked={set(bloom_dist)}"
    )

    # The strongest distribution-uniformity claim we can make under
    # exhaustion: among buckets whose supply was NOT exhausted, all picked
    # counts must be within 1 of each other (the live round-robin
    # invariant). Buckets with picked == supply have been drained and are
    # exempted.
    live = {b: n for b, n in bloom_dist.items() if n < bucket_supply[b]}
    if live:
        live_counts = sorted(live.values())
        assert live_counts[-1] - live_counts[0] <= 1, (
            f"live buckets not balanced: {live}; full dist={bloom_dist}"
        )


def test_include_dpo_from_misconceptions_meets_floor(tmp_path):
    """rdf-shacl-550 has >=67 editorial misconception/correction pairs, so
    --include-dpo-from-misconceptions must emit >=67 misconception DPO
    pairs. (The corpus actually carries 147; the >=67 floor matches the
    spec and is robust to corpus rebalancing.)"""
    archive = _rdf_shacl_archive()
    out_dir = tmp_path / "out"
    stats = run_synthesis_from_libv2(
        slug=archive.name,
        course_code="RDF_SHACL_550",
        provider="mock",
        seed=42,
        include_dpo_from_misconceptions=True,
        max_pairs=10000,  # Don't cap; we want every misconception pair.
        output_dir=out_dir,
    )
    assert stats.misconception_dpo_pairs_emitted >= 67, (
        f"Expected >=67 misconception DPO pairs from rdf-shacl-550; "
        f"got {stats.misconception_dpo_pairs_emitted}"
    )

    pref_path = out_dir / "preference_pairs.jsonl"
    pref_records = _load_jsonl(pref_path)
    misc_records = [r for r in pref_records if r.get("source") == "misconception_editorial"]
    assert len(misc_records) >= 67, (
        f"Expected >=67 misconception_editorial DPO records on disk; "
        f"got {len(misc_records)}"
    )
    # Shape check: prompt / chosen / rejected non-empty on every record.
    for rec in misc_records:
        assert rec.get("prompt"), f"empty prompt on {rec.get('id')}"
        assert rec.get("chosen"), f"empty chosen on {rec.get('id')}"
        assert rec.get("rejected"), f"empty rejected on {rec.get('id')}"
        assert rec["chosen"] != rec["rejected"]


def test_seed_42_regenerates_byte_identical_output(tmp_path):
    archive = _rdf_shacl_archive()
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        run_synthesis_from_libv2(
            slug=archive.name,
            course_code="RDF_SHACL_550",
            provider="mock",
            seed=42,
            stratify=["bloom"],
            include_dpo_from_misconceptions=True,
            max_pairs=80,
            output_dir=out,
        )

    def _canonical(path: Path) -> list[dict]:
        recs = _load_jsonl(path)
        # decision_capture_id is session-bound; strip before comparing.
        for r in recs:
            r.pop("decision_capture_id", None)
        return recs

    assert _canonical(out_a / "instruction_pairs.jsonl") == \
        _canonical(out_b / "instruction_pairs.jsonl")
    assert _canonical(out_a / "preference_pairs.jsonl") == \
        _canonical(out_b / "preference_pairs.jsonl")


def test_max_pairs_50_caps_each_artifact(tmp_path):
    archive = _rdf_shacl_archive()
    out_dir = tmp_path / "out"
    stats = run_synthesis_from_libv2(
        slug=archive.name,
        course_code="RDF_SHACL_550",
        provider="mock",
        seed=42,
        include_dpo_from_misconceptions=True,
        max_pairs=50,
        output_dir=out_dir,
    )
    inst = _load_jsonl(out_dir / "instruction_pairs.jsonl")
    pref = _load_jsonl(out_dir / "preference_pairs.jsonl")
    assert len(inst) <= 50, f"instruction_pairs not capped: {len(inst)}"
    assert len(pref) <= 50, f"preference_pairs not capped: {len(pref)}"
    assert stats.capped_at_max_pairs is True


def test_difficulty_curriculum_orders_foundational_first(tmp_path):
    archive = _rdf_shacl_archive()
    out_dir = tmp_path / "out"
    run_synthesis_from_libv2(
        slug=archive.name,
        course_code="RDF_SHACL_550",
        provider="mock",
        seed=42,
        difficulty_curriculum=True,
        max_pairs=120,
        output_dir=out_dir,
    )
    # Read back the chunk -> difficulty map so we can verify ordering on
    # the emitted instruction_pairs.
    chunks_by_id = {}
    with (archive / "corpus" / "chunks.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            chunks_by_id[c["id"]] = c

    inst = _load_jsonl(out_dir / "instruction_pairs.jsonl")
    assert inst, "no instruction pairs emitted"

    # Map difficulty to its rank; pairs must appear in non-decreasing rank
    # order (foundational <= intermediate <= advanced <= unknown).
    rank = {"foundational": 0, "intermediate": 1, "advanced": 2}
    last_rank = -1
    for rec in inst:
        chunk = chunks_by_id.get(rec["chunk_id"])
        diff = (chunk or {}).get("difficulty", "unknown")
        r = rank.get(diff, len(rank))
        assert r >= last_rank, (
            f"difficulty curriculum violated at chunk {rec['chunk_id']}: "
            f"rank {r} after {last_rank}"
        )
        last_rank = r

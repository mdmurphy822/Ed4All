"""Tests for the W5 generation-quality eval harness (fakes only; no real model).

Mirrors ``grounded_eval``'s test posture: a scripted fake NLI + fake embeddings;
the real ~750MB models never load in CI. Cases map to spec §5.2 (1, 4-9, 11) +
§5.4 (review sample).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from blocks import Block  # noqa: E402
from lib.objectives.generation_quality_eval import (  # noqa: E402
    SCHEMA_VERSION,
    run_generation_quality_eval,
)


# --------------------------------------------------------------------------- #
# Fakes (reuse the established interfaces)
# --------------------------------------------------------------------------- #

class _FakeScore:
    def __init__(self, ent: float, con: float = 0.0):
        self.entailment = ent
        self.neutral = max(0.0, 1.0 - ent - con)
        self.contradiction = con


class _ScriptedNLI:
    """Scripted fake NLI. ``score_batch`` returns high entailment unless the
    hypothesis (the answer sentence) contains a planted ``FABRICATED`` token, in
    which case it returns high contradiction. A ``LOW`` token returns a
    below-floor neutral verdict (unsupported)."""

    _revision = "fake-nli"
    device = "cpu"

    def score_batch(self, *, pairs):
        out = []
        for _premise, hypothesis in pairs:
            h = str(hypothesis)
            if "FABRICATED" in h:
                out.append(_FakeScore(0.05, con=0.95))
            elif "LOWGROUND" in h:
                out.append(_FakeScore(0.10, con=0.10))
            else:
                out.append(_FakeScore(0.95))
        return out


class _FakeEmbedder:
    """Deterministic sha256-seeded unit vectors; identical text → identical
    vector (so two near-identical objectives collide for the dedup test)."""

    provider_name = "fake"
    model_id = "fake-embed"

    def encode_batch(self, texts):
        import hashlib

        import numpy as np

        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            arr = np.frombuffer(h[:32], dtype=np.uint8).astype(np.float32)
            norm = np.linalg.norm(arr) or 1.0
            vecs.append(arr / norm)
        return np.asarray(vecs, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

def _chunks():
    return {
        "c1": {"id": "c1", "text": "Photosynthesis converts light into chemical energy.",
               "concept_tags": ["photosynthesis"], "learning_outcome_refs": ["CO-01"]},
        "c2": {"id": "c2", "text": "Cellular respiration releases energy from glucose.",
               "concept_tags": ["respiration"], "learning_outcome_refs": ["CO-02"]},
    }


def _objectives():
    return {
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Explain energy flow in cells.", "bloom_level": "understand"},
        ],
        "chapter_objectives": [
            {"id": "CO-01", "statement": "Describe photosynthesis light conversion.",
             "bloom_level": "understand",
             "source_refs": [{"ref": "ch1", "chunk_ids": ["c1"]}]},
            {"id": "CO-02", "statement": "FABRICATED claim about respiration.",
             "bloom_level": "analyze",
             "source_refs": [{"ref": "ch2", "chunk_ids": ["c2"]}]},
            {"id": "CO-03", "statement": "An objective with no usable chunk ids.",
             "bloom_level": "apply", "source_refs": []},
        ],
    }


def _blocks():
    return [
        Block(block_id="b1", block_type="example", page_id="p1", sequence=0,
              content="Photosynthesis converts light into chemical energy in plants.",
              objective_ids=("CO-01",), source_ids=("c1",)),
        Block(block_id="b2", block_type="self_check_question", page_id="p1", sequence=1,
              content="Cellular respiration releases energy from glucose effectively.",
              objective_ids=("CO-02",), source_ids=("c2",)),
        Block(block_id="b3", block_type="explanation", page_id="p1", sequence=2,
              content="Plants make food via photosynthesis using sunlight.",
              objective_ids=("CO-01",), source_ids=("c1",)),
        Block(block_id="b4", block_type="concept", page_id="p1", sequence=3,
              content="FABRICATED unsupported claim that conflicts with the source.",
              objective_ids=("CO-02",), source_ids=("c2",)),
        # Scaffolding block — skipped.
        Block(block_id="b5", block_type="objective", page_id="p1", sequence=4,
              content="CO-01: understand energy.", objective_ids=("CO-01",)),
    ]


def _manifests(*, partial=False, phantom=False, mismatch=False):
    m = {
        "b1": {"source_chunk_ids": ["c1"], "model": "qwen", "params": {"t": 0.0},
               "concept_tags": ["photosynthesis"], "template_id": "tmpl", "slots": {}},
        "b2": {"source_chunk_ids": ["c2"], "model": "qwen", "params": {"t": 0.0},
               "concept_tags": ["respiration"], "free_scaffold": True},
        "b3": {"source_chunk_ids": ["c1"], "model": "qwen", "params": {"t": 0.0},
               "concept_tags": [], "template_id": "tmpl", "slots": {}},
        "b4": {"source_chunk_ids": ["c2"], "model": "qwen", "params": {"t": 0.0},
               "concept_tags": [], "free_scaffold": True},
    }
    if partial:
        del m["b4"]  # 3/4 blocks have a manifest
    if phantom:
        m["b1"]["source_chunk_ids"] = ["does-not-exist"]
    if mismatch:
        m["b2"]["source_chunk_ids"] = ["c1"]  # manifest cites c1 but block cites c2
    return m


def _run(**kw):
    defaults = dict(
        objectives=_objectives(),
        blocks=_blocks(),
        chunks_by_id=_chunks(),
        manifests=_manifests(),
        nli=_ScriptedNLI(),
        embedder=_FakeEmbedder(),
        technique_config="C5",
        write=False,
    )
    defaults.update(kw)
    return run_generation_quality_eval(_REPO_ROOT, "test-course", **defaults)


# --------------------------------------------------------------------------- #
# Case 1 — report shape + schema validity
# --------------------------------------------------------------------------- #

def test_report_shape_and_schema_validity():
    report = _run()
    assert report["schema_version"] == "1.0" == SCHEMA_VERSION
    for key in (
        "course_slug", "technique_config", "generated_at", "provenance",
        "counts", "headline", "hierarchy_gates", "provenance_audit",
        "downstream", "diagnostics", "blocks", "objectives",
        "success_condition", "review_sample",
    ):
        assert key in report, f"missing top-level key {key}"
    h = report["headline"]
    for rate_key in ("block_prose_grounding_rate", "objective_grounding_rate"):
        val = h[rate_key]
        assert val is None or 0.0 <= val <= 1.0

    # Validate against the JSON schema.
    schema_path = _REPO_ROOT / "schemas" / "eval" / "generation_quality_eval.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(report, schema)


# --------------------------------------------------------------------------- #
# Case 4 — metrics aggregate correctly
# --------------------------------------------------------------------------- #

def test_metrics_aggregate_correctly():
    report = _run()
    h = report["headline"]
    # 4 scored blocks (b5 objective skipped); b4 is FABRICATED → contradicted.
    assert report["counts"]["scored_blocks"] == 4
    assert report["counts"]["skipped_scaffold_blocks"] == 1
    assert h["contradicted_block_count"] == 1
    # 3 of 4 grounded (b1,b2,b3 grounded; b4 contradicted) → 0.75.
    assert h["block_prose_grounding_rate"] == pytest.approx(0.75)
    # Objectives: CO-01 grounded, CO-02 FABRICATED (contradicted); CO-03 (empty
    # refs) + TO-01 (no source_refs) are both unscorable → 2 unscorable.
    assert report["counts"]["objectives_unscorable"] == 2
    assert report["counts"]["scored_objectives"] == 2  # CO-01 + CO-02
    assert h["objective_contradicted_count"] == 1
    assert h["objective_grounding_rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Case 5 — provenance completeness
# --------------------------------------------------------------------------- #

def test_provenance_completeness_full():
    report = _run(manifests=_manifests())
    assert report["headline"]["provenance_completeness_rate"] == pytest.approx(1.0)


def test_provenance_completeness_partial():
    report = _run(manifests=_manifests(partial=True))
    # 3/4 blocks have a complete manifest.
    assert report["headline"]["provenance_completeness_rate"] == pytest.approx(0.75)


def test_provenance_phantom_chunk_drops_resolution():
    report = _run(manifests=_manifests(phantom=True))
    assert report["provenance_audit"]["manifest_chunk_resolution_rate"] < 1.0


def test_provenance_mismatch_drops_matches():
    report = _run(manifests=_manifests(mismatch=True))
    assert report["provenance_audit"]["manifest_matches_cited_rate"] < 1.0


def test_provenance_none_is_null_and_not_met():
    report = _run(manifests=None)
    assert report["headline"]["provenance_completeness_rate"] is None
    assert report["provenance"].get("provenance_reason") == "manifests_not_supplied"
    assert report["success_condition"]["met"] is False
    assert report["success_condition"]["pillars"]["every_block_provenanced"]["met"] is False


# --------------------------------------------------------------------------- #
# Case 6 — graceful degrade
# --------------------------------------------------------------------------- #

def test_graceful_degrade_no_nli(monkeypatch):
    # Force NLI + embedder absence (the singleton may otherwise load a real
    # model when the extras happen to be installed in the test env).
    import lib.objectives.generation_quality_eval as harness

    monkeypatch.setattr(harness, "_resolve_nli", lambda nli: None)
    monkeypatch.setattr(harness, "_resolve_embedder", lambda embedder: None)
    report = _run(nli=None, embedder=None)
    h = report["headline"]
    assert h["block_prose_grounding_rate"] is None
    assert h["objective_grounding_rate"] is None
    # Structural gates still run.
    assert isinstance(report["hierarchy_gates"], list) and report["hierarchy_gates"]


def test_strict_mode_raises_without_nli(monkeypatch):
    import lib.objectives.generation_quality_eval as harness

    monkeypatch.setattr(harness, "_resolve_nli", lambda nli: None)
    monkeypatch.setattr(harness, "_resolve_embedder", lambda embedder: None)
    monkeypatch.setenv("TRAINFORGE_REQUIRE_EMBEDDINGS", "true")
    with pytest.raises(RuntimeError):
        _run(nli=None, embedder=None)


# --------------------------------------------------------------------------- #
# Case 7 — Bloom collapse detection
# --------------------------------------------------------------------------- #

def test_bloom_collapse_detection():
    collapsed = {
        "chapter_objectives": [
            {"id": "CO-01", "statement": "Understand x.", "bloom_level": "understand"},
            {"id": "CO-02", "statement": "Understand y.", "bloom_level": "understand"},
            {"id": "CO-03", "statement": "Understand z.", "bloom_level": "understand"},
        ]
    }
    report = _run(objectives=collapsed)
    bloom = report["headline"]["bloom"]
    assert bloom["collapse"] is True
    assert bloom["distinct_levels"] == 1


# --------------------------------------------------------------------------- #
# Case 8 — dedup detection
# --------------------------------------------------------------------------- #

def test_dedup_detection():
    dup = {
        "chapter_objectives": [
            {"id": "CO-01", "statement": "Describe photosynthesis light conversion.",
             "bloom_level": "understand"},
            {"id": "CO-02", "statement": "Describe photosynthesis light conversion.",
             "bloom_level": "apply"},
        ]
    }
    report = _run(objectives=dup)
    dedup = report["headline"]["objective_dedup"]
    assert dedup["near_dup_pairs"] >= 1
    assert dedup["dedup_clean"] is False


def test_dedup_ignores_cross_level_to_co_clone():
    """A terminal objective sharing text with a child chapter objective is an
    abstraction smell on a different axis — NOT a redundant-objective dup.

    Dedup is measured WITHIN a hierarchy level, so an identical TO/CO pair
    keeps ``dedup_clean`` True while the cross-level cosine stays visible as a
    transparency diagnostic. Regression for the integration success-condition:
    a 1-section smoke corpus cloned TO-01≈CO-01 (cosine 1.0) and wrongly
    failed the ``objectives_grounded_valid`` pillar.
    """
    clone = {
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Understand chapter one structure and content.",
             "bloom_level": "understand", "hierarchy_level": "terminal"},
        ],
        "chapter_objectives": [
            {"id": "CO-01", "statement": "Understand chapter one structure and content.",
             "bloom_level": "remember", "hierarchy_level": "chapter"},
            {"id": "CO-02", "statement": "Solve linear equations using inverse operations.",
             "bloom_level": "apply", "hierarchy_level": "chapter"},
        ],
    }
    report = _run(objectives=clone)
    dedup = report["headline"]["objective_dedup"]
    # CO-01 vs CO-02 are distinct → no within-level near-dup.
    assert dedup["near_dup_pairs"] == 0
    assert dedup["dedup_clean"] is True
    # The TO≈CO identity is preserved as a cross-level diagnostic, not a failure.
    assert dedup["cross_level_max_cosine"] is not None
    assert dedup["cross_level_max_cosine"] > 0.9


# --------------------------------------------------------------------------- #
# Case 9 — success-condition wiring
# --------------------------------------------------------------------------- #

def test_success_condition_clears_all_pillars():
    # Build a clean fixture: all blocks grounded, all manifests complete,
    # objectives grounded, no contradictions, distinct bloom, no dups.
    clean_objs = {
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Explain energy flow.", "bloom_level": "understand",
             "source_refs": [{"ref": "t", "chunk_ids": ["c1"]}]},
        ],
        "chapter_objectives": [
            {"id": "CO-01", "statement": "Describe photosynthesis conversion.",
             "bloom_level": "remember", "source_refs": [{"ref": "c", "chunk_ids": ["c1"]}]},
            {"id": "CO-02", "statement": "Analyze respiration energy release.",
             "bloom_level": "analyze", "source_refs": [{"ref": "c", "chunk_ids": ["c2"]}]},
        ],
    }
    clean_blocks = [
        Block(block_id="b1", block_type="example", page_id="p1", sequence=0,
              content="Photosynthesis converts light into chemical energy.",
              objective_ids=("CO-01",), source_ids=("c1",)),
    ]
    clean_manifests = {
        "b1": {"source_chunk_ids": ["c1"], "model": "qwen", "params": {},
               "concept_tags": [], "template_id": "t", "slots": {}},
    }
    report = _run(objectives=clean_objs, blocks=clean_blocks, manifests=clean_manifests)
    sc = report["success_condition"]
    assert sc["pinned_at"] == "MEASURE-FIRST"
    assert sc["single_corpus_caveat"] is True
    # grounded=1.0, provenance=1.0, no contradictions.
    assert report["headline"]["block_prose_grounding_rate"] == pytest.approx(1.0)
    assert report["headline"]["contradicted_block_count"] == 0
    assert sc["pillars"]["every_block_grounded"]["met"] is True
    assert sc["pillars"]["every_block_provenanced"]["met"] is True


def test_success_condition_fails_when_pillar_below_threshold():
    # The default fixture has a contradicted block → fails hard-zero.
    report = _run()
    assert report["success_condition"]["met"] is False


# --------------------------------------------------------------------------- #
# Case 11 — hierarchy gates wired
# --------------------------------------------------------------------------- #

def test_hierarchy_gates_invoked():
    report = _run()
    gate_ids = {g["gate_id"] for g in report["hierarchy_gates"]}
    assert gate_ids == {
        "abcd_verb_alignment", "objective_source_refs",
        "chapter_objective_coverage", "terminal_objective_coverage",
        "block_objective_delivery",
    }
    assert isinstance(report["headline"]["hierarchy_valid"], bool)


# --------------------------------------------------------------------------- #
# §5.4 — operator-review sample determinism
# --------------------------------------------------------------------------- #

def test_review_sample_deterministic_and_empty_fields():
    r1 = _run()
    r2 = _run()
    s1 = r1["review_sample"]
    s2 = r2["review_sample"]
    assert s1["samples"] == s2["samples"]  # deterministic across runs
    assert s1["sample_seed"] == "sha256(course_slug::item_id)"
    n_items = s1["n_items"]
    assert s1["n_sampled"] == min(max(10, round(0.2 * n_items)), n_items)
    for row in s1["samples"]:
        rf = row["reviewer_fields"]
        assert rf["objective_correct"] is None
        assert rf["grounding_correct"] is None
        assert rf["notes"] == ""


def test_write_emits_file(tmp_path):
    out = tmp_path / "report.json"
    report = run_generation_quality_eval(
        _REPO_ROOT, "test-course",
        objectives=_objectives(), blocks=_blocks(), chunks_by_id=_chunks(),
        manifests=_manifests(), nli=_ScriptedNLI(), embedder=_FakeEmbedder(),
        write=True, output_path=out,
    )
    assert out.exists()
    assert report["_written"]["report_path"] == str(out)
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "1.0"


def test_write_creates_missing_output_path_parent(tmp_path):
    """A caller-supplied output_path whose parent dir is absent must not crash.

    Regression for the local-generation integration harness, which passes an
    explicit output_path under a not-yet-created ``retrieval_eval/`` dir.
    """
    out = tmp_path / "retrieval_eval" / "nested" / "report.json"
    assert not out.parent.exists()
    report = run_generation_quality_eval(
        _REPO_ROOT, "test-course",
        objectives=_objectives(), blocks=_blocks(), chunks_by_id=_chunks(),
        manifests=_manifests(), nli=_ScriptedNLI(), embedder=_FakeEmbedder(),
        write=True, output_path=out,
    )
    assert out.exists()
    assert report["_written"]["report_path"] == str(out)

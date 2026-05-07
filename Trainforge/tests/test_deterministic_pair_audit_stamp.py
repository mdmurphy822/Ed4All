"""Wave 8 Worker A — deterministic-pair audit-stamp regression suite.

Tests cover:

- Each of the four deterministic generators (kg_metadata, violation,
  abstention, schema_translation) emits pairs whose on-disk JSONL
  carries the seven audit fields the post-emit gate-runner walks
  read (``promotion_status``, ``per_claim_support``,
  ``claim_support_rate``, ``claim_contradicted_rate``, ``deps_missing``,
  ``pair_lo_resolution`` w/ ``skipped == "deterministic_template"``,
  ``pair_objective_alignment``, ``pair_objective_alignment_pass_rate``).
- ``PairLearningOutcomeRefsValidator.validate`` (W4.B walk) skips
  deterministic-template pairs via the ``skipped`` discriminator —
  passes the gate even when the pair's ``chunk_id`` does NOT resolve
  against ``imscc_chunks/chunks.jsonl`` (the bug closed by Wave 8 Worker A).
- ``TrainingPairPromotionValidator.validate`` (W2.E walk) recognises
  the stamped ``promotion_status="validated"`` so the gate passes.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.synthesize_training import (  # noqa: E402
    _stamp_deterministic_pair_audit_fields,
    run_synthesis,
)


FIXTURE_ROOT = (
    Path(__file__).resolve().parent / "fixtures" / "mini_course_training"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _make_working_copy(tmp_path: Path) -> Path:
    """Copy the read-only fixture into a tmp dir so run_synthesis can write."""
    dst = tmp_path / "mini_course_training"
    shutil.copytree(FIXTURE_ROOT, dst)
    for stale in (
        dst / "training_specs" / "instruction_pairs.jsonl",
        dst / "training_specs" / "preference_pairs.jsonl",
    ):
        if stale.exists():
            stale.unlink()
    return dst


def _write_pedagogy_graph(working: Path) -> Path:
    """Write a minimal pedagogy_graph.json that exercises kg_metadata +
    abstention generator paths.

    Mirrors ``Trainforge/tests/test_kg_metadata_generator.py::_five_triple_graph``
    augmented so the abstention generator (which sweeps concepts a chunk
    DOES NOT address) finds at least one chunk with both authored
    concepts AND silent-on concepts to probe.
    """
    graph_dir = working / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "nodes": [
            {"id": "chunk_alpha", "class": "Chunk"},
            {"id": "chunk_beta", "class": "Chunk"},
            {"id": "chunk_gamma", "class": "Chunk"},
            {"id": "chunk_delta", "class": "Chunk"},
            {"id": "concept_a", "class": "Concept"},
            {"id": "concept_b", "class": "Concept"},
            {"id": "concept_c", "class": "Concept"},
            {"id": "concept_d", "class": "Concept"},
            {"id": "module:week_01", "class": "Module"},
            {"id": "module:week_02", "class": "Module"},
            {"id": "bloom:remember", "class": "BloomLevel"},
            {"id": "bloom:apply", "class": "BloomLevel"},
        ],
        "edges": [
            {
                "source": "chunk_alpha",
                "target": "concept_a",
                "relation_type": "assesses",
            },
            {
                "source": "chunk_beta",
                "target": "concept_b",
                "relation_type": "assesses",
            },
            {
                "source": "chunk_gamma",
                "target": "concept_c",
                "relation_type": "assesses",
            },
            {
                "source": "chunk_delta",
                "target": "concept_d",
                "relation_type": "assesses",
            },
            {
                "source": "chunk_alpha",
                "target": "module:week_01",
                "relation_type": "belongs_to_module",
            },
            {
                "source": "chunk_beta",
                "target": "module:week_01",
                "relation_type": "belongs_to_module",
            },
            {
                "source": "chunk_gamma",
                "target": "module:week_02",
                "relation_type": "belongs_to_module",
            },
            {
                "source": "chunk_delta",
                "target": "module:week_02",
                "relation_type": "belongs_to_module",
            },
            {
                "source": "chunk_alpha",
                "target": "bloom:remember",
                "relation_type": "at_bloom_level",
            },
            {
                "source": "chunk_beta",
                "target": "bloom:apply",
                "relation_type": "at_bloom_level",
            },
        ],
    }
    path = graph_dir / "pedagogy_graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# Audit fields the seven post-emit walks read on every pair. The walks
# use ``"per_claim_support" in row`` membership semantics so an empty
# list passes (NOT None — the membership check still recognises the
# key, which signals the validator ran).
_REQUIRED_FIELDS = (
    "promotion_status",
    "per_claim_support",
    "claim_support_rate",
    "claim_contradicted_rate",
    "deps_missing",
    "pair_lo_resolution",
    "pair_objective_alignment",
    "pair_objective_alignment_pass_rate",
)


def _assert_pair_carries_audit_fields(pair: Dict[str, Any]) -> None:
    for f in _REQUIRED_FIELDS:
        assert f in pair, (
            f"deterministic pair missing audit field {f!r}; "
            f"pair keys={sorted(pair.keys())}"
        )
    assert pair["promotion_status"] == "validated"
    assert pair["per_claim_support"] == []
    assert pair["claim_support_rate"] is None
    assert pair["claim_contradicted_rate"] is None
    assert pair["deps_missing"] is False
    plr = pair["pair_lo_resolution"]
    assert isinstance(plr, dict)
    assert plr.get("skipped") == "deterministic_template"
    assert isinstance(plr.get("declared_los"), list)
    assert plr.get("chunk_los") == []
    assert plr.get("phantom_los") == []
    assert pair["pair_objective_alignment"] is None
    assert pair["pair_objective_alignment_pass_rate"] is None


def _filter_by_template_prefix(
    rows: List[Dict[str, Any]], prefix: str
) -> List[Dict[str, Any]]:
    return [r for r in rows if str(r.get("template_id", "")).startswith(prefix)]


# ---------------------------------------------------------------------------
# Helper unit test (covers the four field-shape contracts in one path)
# ---------------------------------------------------------------------------


class _FakeCapture:
    """Minimal DecisionCapture stand-in for the helper unit test."""

    def __init__(self) -> None:
        self.decisions: List[Dict[str, Any]] = []
        self._counter = 0

    def log_decision(self, **kwargs: Any) -> None:
        self._counter += 1
        record = dict(kwargs)
        record["event_id"] = f"EVT_{self._counter:06d}"
        self.decisions.append(record)


def test_stamp_helper_writes_required_audit_fields_and_emits_decision():
    capture = _FakeCapture()
    pair = {
        "template_id": "kg_metadata.assesses_yes",
        "chunk_id": "chunk_alpha",
        "lo_refs": ["kg-metadata"],
        "kg_metadata_polarity": "yes",
    }
    _stamp_deterministic_pair_audit_fields(pair, capture=capture)

    _assert_pair_carries_audit_fields(pair)
    # declared_los carries the pair's lo_refs verbatim.
    assert pair["pair_lo_resolution"]["declared_los"] == ["kg-metadata"]

    # Exactly one decision emitted; canonical decision_type registered
    # in schemas/events/decision_event.schema.json.
    assert len(capture.decisions) == 1
    evt = capture.decisions[0]
    assert evt["decision_type"] == "deterministic_pair_audit_stamp"
    rationale = evt["rationale"]
    assert "kg_metadata.assesses_yes" in rationale
    assert "chunk_alpha" in rationale
    # 20-char minimum is enforced by the schema; spot-check the floor.
    assert len(rationale) >= 20


# ---------------------------------------------------------------------------
# Per-generator on-disk emit tests
# ---------------------------------------------------------------------------


def test_kg_metadata_pair_carries_audit_fields_after_run_synthesis(tmp_path):
    working = _make_working_copy(tmp_path)
    _write_pedagogy_graph(working)

    run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=17,
        with_kg_metadata=True,
        kg_metadata_max_pairs=20,
    )

    inst = _load_jsonl(working / "training_specs" / "instruction_pairs.jsonl")
    kg_pairs = _filter_by_template_prefix(inst, "kg_metadata.")
    assert kg_pairs, "expected kg_metadata pairs in emitted instruction_pairs.jsonl"
    for pair in kg_pairs:
        _assert_pair_carries_audit_fields(pair)


def test_abstention_pair_carries_audit_fields_after_run_synthesis(tmp_path):
    working = _make_working_copy(tmp_path)
    _write_pedagogy_graph(working)

    run_synthesis(
        corpus_dir=working,
        course_code="MINI_TRAINING_101",
        provider="mock",
        seed=23,
        with_abstention=True,
        abstention_max_pairs=20,
    )

    inst = _load_jsonl(working / "training_specs" / "instruction_pairs.jsonl")
    ab_pairs = _filter_by_template_prefix(inst, "abstention.")
    assert ab_pairs, "expected abstention pairs in emitted instruction_pairs.jsonl"
    for pair in ab_pairs:
        _assert_pair_carries_audit_fields(pair)


def test_violation_pair_carries_audit_fields_after_run_synthesis(tmp_path):
    """Violation generator requires pyshacl + a SHACL property manifest.

    The mini_course_training fixture has no manifest; the rdf_shacl
    family slug resolves against the canonical
    ``schemas/training/property_manifest.rdf_shacl.yaml`` — driving the
    course_code through that family slug lets the synthesizer load a
    real SHACL manifest. pyshacl is the only optional dep; skip when
    absent rather than fail-closed.
    """
    pytest.importorskip("pyshacl")
    pytest.importorskip("rdflib")

    working = _make_working_copy(tmp_path)
    _write_pedagogy_graph(working)

    # course_code resolves to the rdf_shacl family slug via
    # `_family_slug` in lib.ontology.property_manifest.
    run_synthesis(
        corpus_dir=working,
        course_code="rdf-shacl-test",
        slug="rdf-shacl-test",
        provider="mock",
        seed=29,
        with_violation_detection=True,
        violation_detection_max_pairs=20,
    )

    inst = _load_jsonl(working / "training_specs" / "instruction_pairs.jsonl")
    vio_pairs = _filter_by_template_prefix(inst, "violation_detection.")
    assert vio_pairs, (
        "expected violation_detection pairs in emitted "
        "instruction_pairs.jsonl"
    )
    for pair in vio_pairs:
        _assert_pair_carries_audit_fields(pair)


def test_schema_translation_pair_carries_audit_fields_after_run_synthesis(
    tmp_path,
):
    """Schema-translation requires a property manifest. Same trick as
    violation: drive course_code through the rdf_shacl family slug."""
    working = _make_working_copy(tmp_path)
    _write_pedagogy_graph(working)

    run_synthesis(
        corpus_dir=working,
        course_code="rdf-shacl-test",
        slug="rdf-shacl-test",
        provider="mock",
        seed=31,
        with_schema_translation=True,
        schema_translation_max_pairs=12,
    )

    inst = _load_jsonl(working / "training_specs" / "instruction_pairs.jsonl")
    st_pairs = _filter_by_template_prefix(inst, "schema_translation.")
    assert st_pairs, (
        "expected schema_translation pairs in emitted instruction_pairs.jsonl"
    )
    for pair in st_pairs:
        _assert_pair_carries_audit_fields(pair)


# ---------------------------------------------------------------------------
# Gate-runner walk tests — the contract Wave 8 Worker A actually closes
# ---------------------------------------------------------------------------


def test_pair_lo_refs_validate_walk_skips_deterministic_pairs(tmp_path):
    """W4.B walk MUST pass on a corpus whose deterministic pairs cite
    synthetic chunk_ids absent from chunks.jsonl. The skip-discriminator
    on ``pair_lo_resolution.skipped == "deterministic_template"`` is
    THE mechanism that flips the previous critical
    ``PAIR_CHUNK_NOT_FOUND`` failure to a non-issue."""
    pytest.importorskip("pyshacl")
    pytest.importorskip("rdflib")

    working = _make_working_copy(tmp_path)
    _write_pedagogy_graph(working)

    run_synthesis(
        corpus_dir=working,
        course_code="rdf-shacl-test",
        slug="rdf-shacl-test",
        provider="mock",
        seed=37,
        with_kg_metadata=True,
        kg_metadata_max_pairs=10,
        with_violation_detection=True,
        violation_detection_max_pairs=10,
        with_abstention=True,
        abstention_max_pairs=10,
        with_schema_translation=True,
        schema_translation_max_pairs=10,
    )

    from lib.validators.pair_lo_refs import PairLearningOutcomeRefsValidator

    validator = PairLearningOutcomeRefsValidator()
    # The mini_course fixture lays chunks at corpus/chunks.jsonl —
    # explicitly hand the validator the chunks_path + the
    # training_specs path so the resolver doesn't go searching for
    # imscc_chunks/.
    chunks_path = working / "corpus" / "chunks.jsonl"
    inst_path = working / "training_specs" / "instruction_pairs.jsonl"
    pref_path = working / "training_specs" / "preference_pairs.jsonl"
    result = validator.validate({
        "instruction_pairs_path": str(inst_path),
        "preference_pairs_path": str(pref_path),
        "chunks_path": str(chunks_path),
    })
    # No critical PAIR_CHUNK_NOT_FOUND fired despite the four
    # deterministic generators emitting pairs whose chunk_ids
    # (concept_alpha, shacl_test_graph_001, etc.) don't resolve against
    # corpus/chunks.jsonl.
    assert result.passed, (
        f"PairLearningOutcomeRefsValidator.validate failed unexpectedly: "
        f"issues={[(i.code, i.message) for i in result.issues]}"
    )
    # Defense in depth: assert no PAIR_CHUNK_NOT_FOUND issue was logged
    # against any deterministic chunk_id.
    chunk_not_found = [
        i for i in result.issues if i.code == "PAIR_CHUNK_NOT_FOUND"
    ]
    assert not chunk_not_found, (
        f"Expected zero PAIR_CHUNK_NOT_FOUND issues; got: "
        f"{[i.message for i in chunk_not_found]}"
    )


def test_pair_promotion_validate_walk_passes_deterministic_pairs(tmp_path):
    """W2.E walk MUST pass on the same all-four-generators corpus.
    Every pair on disk carries ``promotion_status="validated"``, so the
    walk's ``MISSING_PROMOTION_STATUS`` check finds nothing to flag."""
    pytest.importorskip("pyshacl")
    pytest.importorskip("rdflib")

    working = _make_working_copy(tmp_path)
    _write_pedagogy_graph(working)

    run_synthesis(
        corpus_dir=working,
        course_code="rdf-shacl-test",
        slug="rdf-shacl-test",
        provider="mock",
        seed=41,
        with_kg_metadata=True,
        kg_metadata_max_pairs=10,
        with_violation_detection=True,
        violation_detection_max_pairs=10,
        with_abstention=True,
        abstention_max_pairs=10,
        with_schema_translation=True,
        schema_translation_max_pairs=10,
    )

    from lib.validators.training_pair_promotion import (
        TrainingPairPromotionValidator,
    )

    validator = TrainingPairPromotionValidator()
    inst_path = working / "training_specs" / "instruction_pairs.jsonl"
    pref_path = working / "training_specs" / "preference_pairs.jsonl"
    result = validator.validate({
        "instruction_pairs_path": str(inst_path),
        "preference_pairs_path": str(pref_path),
    })
    assert result.passed, (
        f"TrainingPairPromotionValidator.validate failed unexpectedly: "
        f"issues={[(i.code, i.message) for i in result.issues]}"
    )
    missing_status = [
        i for i in result.issues if i.code == "MISSING_PROMOTION_STATUS"
    ]
    assert not missing_status, (
        f"Expected zero MISSING_PROMOTION_STATUS issues; got: "
        f"{[i.message for i in missing_status]}"
    )

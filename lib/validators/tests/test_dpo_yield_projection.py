"""DpoYieldProjectionValidator + project_dpo_yield — Bloom-ladder addendum AD-02.

Covers ``lib/validators/dpo_yield_projection.py``:

- ``project_dpo_yield`` arithmetic against REAL fixtures walked through the
  real production predicates (``resolve_chunk_misconceptions``,
  ``focus_chunk_on_canonical_objective``, ``pair_eligibility``,
  ``micro_preference_eligibility``, ``is_dpo_editorial_record`` — nothing
  reimplemented): cards_found / per-rung breakdown / arm_a_admitted /
  projected_admissible distinguish an Arm-A-survivable card from one that
  is recovered but never admitted (source-unbacked correction);
- the gate fires (warning, below floor) / passes (at/above floor);
- structured skip on missing chunks_path / objectives_path;
- structured skip on zero misconception cards with ``ED4ALL_BLOOM_LADDER``
  off (legacy corpus), and the real warning firing instead once the flag is
  on;
- ``passed`` stays True regardless (warning-only, never blocks);
- three-place wiring: the ``dpo_yield_projection`` gate_id in
  ``config/workflows.yaml::textbook_to_course::training_synthesis`` resolves
  a REGISTERED builder in ``MCP/hardening/gate_input_routing.py`` (never
  ``__no_builder_registered__``), and that builder actually resolves real
  inputs off real phase outputs.

No course slugs anywhere — every fixture is synthetic and built inline or
under ``tmp_path``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lib.generation.bloom_ladder_blocks import ENV_BLOOM_LADDER
from lib.validators.dpo_yield_projection import (
    DEFAULT_MIN_DPO_PAIRS,
    DpoYieldProjectionValidator,
    project_dpo_yield,
)

_OBJ_STATEMENT_UNDERSTAND = (
    "Explain how absolute value represents distance magnitude for a "
    "number line position"
)
_OBJ_STATEMENT_CREATE = (
    "Derive the commuting proof steps for advanced algebraic identities"
)

_OBJECTIVES = {
    "to-01": {
        "statement": _OBJ_STATEMENT_UNDERSTAND,
        "bloom_level": "understand",
        "bloom_verb": "explain",
    },
    "to-02": {
        "statement": _OBJ_STATEMENT_CREATE,
        "bloom_level": "create",
        "bloom_verb": "derive",
    },
}

_CLAIM_A = (
    "absolute value equals the original number for every number line position"
)
_CORRECTION_A = (
    "absolute value equals the non negative magnitude for every number "
    "line position"
)
_FILLER_A = " ".join([
    "Distance magnitude concepts recur across many worked examples in "
    "this unit for number line position study."
] * 6)

_CLAIM_B = (
    "derived commuting proof steps skip the base case entirely for "
    "advanced identities"
)
_CORRECTION_B = (
    "derived commuting proof steps always verify the base case first "
    "for advanced identities"
)
_FILLER_B = " ".join([
    "Advanced algebraic identity derivations recur across many "
    "commuting proof steps in this unit study."
] * 6)


def _admitted_chunk() -> dict:
    """A chunk whose card is fully source-backed -> Arm-A admitted."""
    return {
        "id": "chunk-admitted",
        "text": _FILLER_A + " " + _CLAIM_A + ". " + _CORRECTION_A + ".",
        "bloom_level": "understand",
        "learning_outcome_refs": ["to-01"],
        "misconceptions": [{
            "misconception": _CLAIM_A,
            "correction": _CORRECTION_A,
            "mechanism_evidence": _CORRECTION_A,
            "bloom_level": "understand",
        }],
    }


def _recovered_but_unbacked_chunk() -> dict:
    """A card is recovered (counts toward cards_found) but its correction
    never appears in the chunk's own text -> Arm A rejects it as
    source_unbacked. Proves cards_found != arm_a_admitted."""
    return {
        "id": "chunk-unbacked",
        # Deliberately omits _CORRECTION_B from the source text.
        "text": _FILLER_B + " " + _CLAIM_B + ".",
        "bloom_level": "create",
        "learning_outcome_refs": ["to-02"],
        "misconceptions": [{
            "misconception": _CLAIM_B,
            "correction": _CORRECTION_B,
            "mechanism_evidence": _CORRECTION_B,
            "bloom_level": "create",
        }],
    }


def _no_card_chunk() -> dict:
    """A plain chunk with prose but no misconception markup at all."""
    return {
        "id": "chunk-plain",
        "text": _FILLER_A,
        "bloom_level": "understand",
        "learning_outcome_refs": ["to-01"],
    }


# --------------------------------------------------------------------- #
# project_dpo_yield arithmetic (real predicates, no monkeypatching)
# --------------------------------------------------------------------- #


def test_admitted_card_counts_toward_every_tally():
    report = project_dpo_yield([_admitted_chunk()], _OBJECTIVES, min_dpo_pairs=1)
    assert report["chunk_count"] == 1
    assert report["cards_found"] == 1
    assert report["cards_by_rung"] == {"understand": 1}
    assert report["arm_a_admitted"] == 1
    assert report["projected_admissible"] == 1
    assert report["projected_admissible_by_rung"] == {"understand": 1}
    assert report["shortfall"] == 0
    assert report["deficit"] is False


def test_recovered_card_that_fails_source_backing_is_not_admitted():
    """cards_found counts it; arm_a_admitted / projected_admissible do not —
    this is the core "eligibility != admission" distinction the famine
    guard exists to catch."""
    report = project_dpo_yield(
        [_recovered_but_unbacked_chunk()], _OBJECTIVES, min_dpo_pairs=1,
    )
    assert report["cards_found"] == 1
    assert report["cards_by_rung"] == {"create": 1}
    assert report["arm_a_admitted"] == 0
    assert report["projected_admissible"] == 0
    assert report["deficit"] is True


def test_mixed_corpus_distinguishes_found_from_admitted():
    report = project_dpo_yield(
        [_admitted_chunk(), _recovered_but_unbacked_chunk()],
        _OBJECTIVES,
        min_dpo_pairs=5,
    )
    assert report["chunk_count"] == 2
    assert report["cards_found"] == 2
    assert report["cards_by_rung"] == {"create": 1, "understand": 1}
    assert report["arm_a_admitted"] == 1
    assert report["projected_admissible"] == 1
    assert report["projected_admissible_by_rung"] == {"understand": 1}
    assert report["shortfall"] == 4
    assert report["deficit"] is True


def test_chunk_with_no_misconception_markup_contributes_zero_cards():
    report = project_dpo_yield([_no_card_chunk()], _OBJECTIVES, min_dpo_pairs=1)
    assert report["chunk_count"] == 1
    assert report["cards_found"] == 0
    assert report["arm_a_admitted"] == 0
    assert report["projected_admissible"] == 0


def test_meets_floor_when_admitted_count_reaches_min_dpo_pairs():
    chunks = [dict(_admitted_chunk(), id=f"chunk-{i}") for i in range(3)]
    report = project_dpo_yield(chunks, _OBJECTIVES, min_dpo_pairs=3)
    assert report["projected_admissible"] == 3
    assert report["shortfall"] == 0
    assert report["deficit"] is False


def test_default_min_dpo_pairs_matches_trainer_floor():
    assert DEFAULT_MIN_DPO_PAIRS == 50


def test_empty_objectives_map_projects_zero_admissible():
    """Mirrors the real synthesis contract: no canonical objectives, no
    focus, no pairs — every chunk is ineligible."""
    report = project_dpo_yield([_admitted_chunk()], {}, min_dpo_pairs=1)
    assert report["cards_found"] == 1
    assert report["arm_a_admitted"] == 0
    assert report["projected_admissible"] == 0


def test_authored_misconceptions_are_read_untouched():
    """Authored non-empty chunk.misconceptions wins over recovery — same
    contract as resolve_chunk_misconceptions itself."""
    chunk = _admitted_chunk()
    chunk["misconceptions"] = [dict(chunk["misconceptions"][0], id="authored-id")]
    report = project_dpo_yield([chunk], _OBJECTIVES, min_dpo_pairs=1)
    assert report["cards_found"] == 1


# --------------------------------------------------------------------- #
# DpoYieldProjectionValidator — gate wrapper
# --------------------------------------------------------------------- #


def _write_chunks(path: Path, chunks: list) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk) + "\n")


def _write_objectives(path: Path) -> None:
    path.write_text(json.dumps([
        {
            "id": "TO-01",
            "statement": _OBJ_STATEMENT_UNDERSTAND,
            "bloom_level": "understand",
        },
        {
            "id": "TO-02",
            "statement": _OBJ_STATEMENT_CREATE,
            "bloom_level": "create",
        },
    ]))


@pytest.fixture(autouse=True)
def _clear_bloom_ladder_env(monkeypatch):
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    yield


def test_gate_skips_when_chunks_path_missing():
    validator = DpoYieldProjectionValidator()
    result = validator.validate({})
    assert result.passed is True
    assert result.metadata["skipped"] is True
    assert result.metadata["skip_reason"] == "chunks_path_missing"


def test_gate_skips_when_chunks_path_does_not_exist(tmp_path: Path):
    validator = DpoYieldProjectionValidator()
    result = validator.validate({
        "chunks_path": str(tmp_path / "does_not_exist.jsonl"),
    })
    assert result.passed is True
    assert result.metadata["skip_reason"] == "chunks_path_absent"


def test_gate_skips_when_objectives_path_missing(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, [_admitted_chunk()])
    validator = DpoYieldProjectionValidator()
    result = validator.validate({"chunks_path": str(chunks_path)})
    assert result.passed is True
    assert result.metadata["skip_reason"] == "objectives_path_missing"


def test_gate_skips_on_zero_cards_when_bloom_ladder_off(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, [_no_card_chunk()])
    objectives_path = tmp_path / "objectives.json"
    _write_objectives(objectives_path)

    validator = DpoYieldProjectionValidator()
    result = validator.validate({
        "chunks_path": str(chunks_path),
        "objectives_path": str(objectives_path),
    })
    assert result.passed is True
    assert result.metadata["skipped"] is True
    assert result.metadata["skip_reason"] == "no_cards_legacy_corpus"
    assert result.issues[0].severity == "info"


def test_gate_fires_real_warning_on_zero_cards_when_bloom_ladder_on(
    tmp_path: Path, monkeypatch,
):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, [_no_card_chunk()])
    objectives_path = tmp_path / "objectives.json"
    _write_objectives(objectives_path)

    validator = DpoYieldProjectionValidator()
    result = validator.validate({
        "chunks_path": str(chunks_path),
        "objectives_path": str(objectives_path),
    })
    # Warning-only: never blocks, but the deficit issue is real (not a
    # structured skip) once the operator opted into ladder-driven authoring.
    assert result.passed is True
    assert result.metadata["skipped"] is False
    assert len(result.issues) == 1
    assert result.issues[0].severity == "warning"
    assert result.issues[0].code == "DPO_YIELD_PROJECTION_BELOW_FLOOR"


def test_gate_fires_warning_below_floor(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, [_admitted_chunk(), _recovered_but_unbacked_chunk()])
    objectives_path = tmp_path / "objectives.json"
    _write_objectives(objectives_path)

    validator = DpoYieldProjectionValidator()
    result = validator.validate({
        "chunks_path": str(chunks_path),
        "objectives_path": str(objectives_path),
        "min_dpo_pairs": 5,
    })
    assert result.passed is True  # warning-only, never blocks
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.severity == "warning"
    assert issue.code == "DPO_YIELD_PROJECTION_BELOW_FLOOR"
    assert "projected_admissible" not in issue.message  # human prose, not a key dump
    assert result.metadata["projected_admissible"] == 1
    assert result.metadata["min_dpo_pairs"] == 5


def test_gate_passes_cleanly_at_or_above_floor(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, [_admitted_chunk()])
    objectives_path = tmp_path / "objectives.json"
    _write_objectives(objectives_path)

    validator = DpoYieldProjectionValidator()
    result = validator.validate({
        "chunks_path": str(chunks_path),
        "objectives_path": str(objectives_path),
        "min_dpo_pairs": 1,
    })
    assert result.passed is True
    assert result.issues == []
    assert result.metadata["deficit"] is False


def test_min_dpo_pairs_threshold_override_via_thresholds_block(tmp_path: Path):
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunks_path, [_admitted_chunk()])
    objectives_path = tmp_path / "objectives.json"
    _write_objectives(objectives_path)

    validator = DpoYieldProjectionValidator()
    result = validator.validate({
        "chunks_path": str(chunks_path),
        "objectives_path": str(objectives_path),
        "thresholds": {"min_dpo_pairs": 2},
    })
    assert result.metadata["min_dpo_pairs"] == 2
    assert result.metadata["deficit"] is True


def test_validator_name_and_version():
    validator = DpoYieldProjectionValidator()
    assert validator.name == "dpo_yield_projection"
    assert isinstance(validator.version, str)


# --------------------------------------------------------------------- #
# Three-place wiring — config/workflows.yaml + gate_input_routing.py
# --------------------------------------------------------------------- #

_VALIDATOR_DOTTED_PATH = (
    "lib.validators.dpo_yield_projection.DpoYieldProjectionValidator"
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOWS_YAML = _REPO_ROOT / "config" / "workflows.yaml"


def _load_training_synthesis_gates() -> dict:
    import yaml

    with _WORKFLOWS_YAML.open() as fh:
        workflows = yaml.safe_load(fh)
    phase = next(
        p
        for p in workflows["workflows"]["textbook_to_course"]["phases"]
        if p["name"] == "training_synthesis"
    )
    return {g["gate_id"]: g for g in phase["validation_gates"]}


def test_dpo_yield_projection_gate_wired_at_training_synthesis():
    gates = _load_training_synthesis_gates()
    assert "dpo_yield_projection" in gates, sorted(gates)
    gate_row = gates["dpo_yield_projection"]
    assert gate_row["validator"] == _VALIDATOR_DOTTED_PATH
    assert gate_row["severity"] == "warning"


def test_dpo_yield_projection_gate_id_resolves_a_registered_builder():
    """Regression guard against a silent ``__no_builder_registered__``
    skip — an unregistered validator makes the executor fall through to
    ``({}, ['__no_builder_registered__'])`` and the (warning-severity) gate
    passes as a vacuous no-op every run."""
    from MCP.hardening.gate_input_routing import default_router

    router = default_router()
    assert _VALIDATOR_DOTTED_PATH in router.builders, (
        f"AD-02 regression: no builder registered for "
        f"{_VALIDATOR_DOTTED_PATH}; gate will silently skip via "
        "__no_builder_registered__."
    )


def test_dpo_yield_projection_builder_resolves_real_inputs():
    from MCP.hardening.gate_input_routing import default_router

    router = default_router()
    phase_outputs = {
        "training_synthesis": {
            "imscc_chunks_path": "/tmp/dpo-yield-fixture/chunks.jsonl",
        },
        "course_planning": {
            "synthesized_objectives_path": "/tmp/dpo-yield-fixture/objectives.json",
        },
    }
    inputs, missing = router.build(_VALIDATOR_DOTTED_PATH, phase_outputs, {})
    assert missing == [], f"unexpected required_missing: {missing}"
    assert inputs["chunks_path"] == "/tmp/dpo-yield-fixture/chunks.jsonl"
    assert inputs["objectives_path"] == "/tmp/dpo-yield-fixture/objectives.json"


def test_dpo_yield_projection_builder_structurally_skips_without_chunks():
    from MCP.hardening.gate_input_routing import default_router

    router = default_router()
    inputs, missing = router.build(_VALIDATOR_DOTTED_PATH, {}, {})
    assert missing == ["chunks_path"]

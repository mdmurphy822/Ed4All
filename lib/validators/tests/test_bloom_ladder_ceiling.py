"""BloomLadderCeilingValidator unit tests — Bloom-ladder initiative WI-15.

Covers ``lib/validators/bloom_ladder_ceiling.py``'s ``ED4ALL_BLOOM_LADDER``-
gated ceiling/completeness gate:

- structured skip (byte-stable) when the flag is off — a single INFO
  ``BLOOM_LADDER_DISABLED`` issue, ``metadata["skipped"] = True``;
- NEVER-above-ceiling: a block's ``bloom_level`` / ``ladder_rung`` ranking
  above its target objective's own declared ``bloom_level`` fires
  ``BLOOM_LADDER_ABOVE_CEILING``;
- a compliant block (at or below its objective's ceiling) passes cleanly;
- ladder completeness: a missing rung between ``remember`` and the
  objective's own ceiling fires ``BLOOM_LADDER_INCOMPLETE_RUNG``;
- a fully-covered ladder passes cleanly;
- the objective-ceiling projection is the tolerant ``bloom_level`` /
  ``bloomLevel`` fallback (stripped + lowercased), same shape as
  ``lib.generation.block_planner._bloom_ladder_objective_ceiling``;
- ``blocks_final_path`` / ``synthesized_objectives_path`` JSONL/JSON
  hydration off disk;
- missing-blocks-input structural error stays critical + blocking
  regardless of the warning-day-1 severity of the two real checks;
- warning severity day-1: ``passed`` stays True even when
  ceiling/completeness issues fire (only a critical issue would flip it).

No course slugs anywhere — every fixture is a synthetic block/objective
built as a plain dict or a minimal attribute-bearing object.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from lib.generation.bloom_ladder_blocks import ENV_BLOOM_LADDER
from lib.validators.bloom_ladder_ceiling import (
    BloomLadderCeilingValidator,
    _CODE_ABOVE_CEILING,
    _CODE_DISABLED,
    _CODE_INCOMPLETE_RUNG,
    _CODE_MISSING_BLOCKS,
    _coerce_objectives,
    _objective_ceiling,
)


# --------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------- #


def _block(
    *,
    block_id: str,
    objective_ids: Optional[List[str]] = None,
    bloom_level: Optional[str] = None,
    ladder_rung: Optional[str] = None,
) -> Dict[str, Any]:
    """A rewrite-tier block dict row (validator accepts dict OR Block)."""
    row: Dict[str, Any] = {"block_id": block_id, "block_type": "concept"}
    if objective_ids is not None:
        row["objective_ids"] = objective_ids
    if bloom_level is not None:
        row["bloom_level"] = bloom_level
    if ladder_rung is not None:
        row["ladder_rung"] = ladder_rung
    return row


def _objective(lo_id: str, bloom_level: str) -> Dict[str, Any]:
    return {"id": lo_id, "statement": f"Do the thing for {lo_id}.", "bloom_level": bloom_level}


def _rungs_for(ceiling: str) -> List[str]:
    """Ordered rung levels from ``remember`` up to ``ceiling`` inclusive."""
    from lib.ontology.bloom import BLOOM_LEVELS

    idx = BLOOM_LEVELS.index(ceiling)
    return list(BLOOM_LEVELS[: idx + 1])


def _fully_covered_blocks(oid: str, ceiling: str) -> List[Dict[str, Any]]:
    """One block per rung from remember up to ``ceiling`` targeting ``oid``."""
    return [
        _block(block_id=f"{oid}-{level}", objective_ids=[oid], bloom_level=level)
        for level in _rungs_for(ceiling)
    ]


def _validate(monkeypatch, *, on: bool, **inputs: Any):
    if on:
        monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    else:
        monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    return BloomLadderCeilingValidator().validate(inputs)


# --------------------------------------------------------------------- #
# Flag-off structured skip
# --------------------------------------------------------------------- #


def test_flag_off_is_structured_skip(monkeypatch):
    blocks = [_block(block_id="b1", objective_ids=["CO-01"], bloom_level="create")]
    objectives = [_objective("CO-01", "remember")]
    result = _validate(monkeypatch, on=False, blocks=blocks, objectives=objectives)

    assert result.passed is True
    assert result.score == 1.0
    assert result.action is None
    assert len(result.issues) == 1
    assert result.issues[0].code == _CODE_DISABLED
    assert result.issues[0].severity == "info"
    assert result.metadata == {"skipped": True, "enabled": False}


def test_flag_off_never_touches_blocks_or_objectives_input(monkeypatch):
    # Missing/invalid blocks input would normally be a critical error — but
    # the flag-off path returns before ever coercing blocks, so garbage
    # input is a pure no-op.
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    result = BloomLadderCeilingValidator().validate({"blocks": "not-a-list"})
    assert result.passed is True
    assert result.metadata["skipped"] is True


# --------------------------------------------------------------------- #
# Empty-input no-op (flag on, nothing to audit)
# --------------------------------------------------------------------- #


def test_flag_on_empty_objectives_is_noop_pass(monkeypatch):
    blocks = [_block(block_id="b1", objective_ids=["CO-01"], bloom_level="apply")]
    result = _validate(monkeypatch, on=True, blocks=blocks, objectives=[])
    assert result.passed is True
    assert result.score == 1.0
    assert result.issues == []
    assert result.metadata["ceiling_audited"] == 0
    assert result.metadata["completeness_audited"] == 0


def test_flag_on_empty_blocks_is_noop_pass(monkeypatch):
    objectives = [_objective("CO-01", "apply")]
    result = _validate(monkeypatch, on=True, blocks=[], objectives=objectives)
    # No covering blocks at all -> every rung up to the ceiling is an
    # incomplete-rung issue; this is NOT a no-op (completeness still audits).
    assert result.passed is True  # warning-day-1: still passes
    assert result.metadata["completeness_audited"] == 3  # remember/understand/apply
    assert result.metadata["completeness_passed"] == 0
    codes = {i.code for i in result.issues}
    assert codes == {_CODE_INCOMPLETE_RUNG}


# --------------------------------------------------------------------- #
# (a) NEVER-above-ceiling
# --------------------------------------------------------------------- #


def test_above_ceiling_via_bloom_level_fires(monkeypatch):
    blocks = [_block(block_id="b1", objective_ids=["CO-01"], bloom_level="create")]
    objectives = [_objective("CO-01", "remember")]
    result = _validate(monkeypatch, on=True, blocks=blocks, objectives=objectives)

    breach = [i for i in result.issues if i.code == _CODE_ABOVE_CEILING]
    assert len(breach) == 1
    assert breach[0].severity == "warning"
    assert breach[0].location == "b1"
    assert "create" in breach[0].message
    assert "remember" in breach[0].message
    assert result.metadata["ceiling_audited"] == 1
    assert result.metadata["ceiling_passed"] == 0
    # Warning-day-1: still passes overall.
    assert result.passed is True


def test_above_ceiling_via_ladder_rung_fires(monkeypatch):
    blocks = [
        _block(block_id="b1", objective_ids=["CO-01"], ladder_rung="analyze"),
    ]
    objectives = [_objective("CO-01", "understand")]
    result = _validate(monkeypatch, on=True, blocks=blocks, objectives=objectives)

    breach = [i for i in result.issues if i.code == _CODE_ABOVE_CEILING]
    assert len(breach) == 1
    assert "ladder_rung" in breach[0].message


def test_at_ceiling_is_compliant(monkeypatch):
    blocks = [_block(block_id="b1", objective_ids=["CO-01"], bloom_level="apply")]
    objectives = [_objective("CO-01", "apply")]
    result = _validate(monkeypatch, on=True, blocks=blocks, objectives=objectives)

    assert not any(i.code == _CODE_ABOVE_CEILING for i in result.issues)
    assert result.metadata["ceiling_audited"] == 1
    assert result.metadata["ceiling_passed"] == 1


def test_below_ceiling_is_compliant(monkeypatch):
    blocks = [_block(block_id="b1", objective_ids=["CO-01"], bloom_level="remember")]
    objectives = [_objective("CO-01", "create")]
    result = _validate(monkeypatch, on=True, blocks=blocks, objectives=objectives)

    assert not any(i.code == _CODE_ABOVE_CEILING for i in result.issues)
    assert result.metadata["ceiling_passed"] == 1


def test_multi_objective_block_flags_any_breach(monkeypatch):
    # Block targets two objectives; breaches ONLY the lower-ceiling one.
    blocks = [
        _block(
            block_id="b1",
            objective_ids=["CO-01", "CO-02"],
            bloom_level="apply",
        ),
    ]
    objectives = [
        _objective("CO-01", "create"),   # apply <= create: fine
        _objective("CO-02", "remember"),  # apply > remember: breach
    ]
    result = _validate(monkeypatch, on=True, blocks=blocks, objectives=objectives)

    breach = [i for i in result.issues if i.code == _CODE_ABOVE_CEILING]
    assert len(breach) == 1
    assert "CO-02" in breach[0].message
    assert "CO-01" not in breach[0].message


def test_block_with_no_objective_ids_is_not_audited(monkeypatch):
    blocks = [_block(block_id="b1", bloom_level="create")]
    objectives = [_objective("CO-01", "remember")]
    result = _validate(monkeypatch, on=True, blocks=blocks, objectives=objectives)

    assert result.metadata["ceiling_audited"] == 0
    assert not any(i.code == _CODE_ABOVE_CEILING for i in result.issues)


def test_block_with_unresolvable_objective_is_not_ceiling_audited(monkeypatch):
    # objective_ids references an objective absent from the objectives map.
    blocks = [_block(block_id="b1", objective_ids=["CO-99"], bloom_level="create")]
    objectives = [_objective("CO-01", "remember")]
    result = _validate(monkeypatch, on=True, blocks=blocks, objectives=objectives)

    assert result.metadata["ceiling_audited"] == 0


def test_garbage_bloom_level_is_ignored_not_a_crash(monkeypatch):
    blocks = [_block(block_id="b1", objective_ids=["CO-01"], bloom_level="not-a-level")]
    objectives = [_objective("CO-01", "remember")]
    result = _validate(monkeypatch, on=True, blocks=blocks, objectives=objectives)

    # No observed (valid) level -> not ceiling-auditable, no crash, no
    # ceiling issue (the completeness check independently still flags the
    # objective's uncovered rungs -- the garbage bloom_level never counts
    # as covering "remember").
    assert result.metadata["ceiling_audited"] == 0
    assert not any(i.code == _CODE_ABOVE_CEILING for i in result.issues)


# --------------------------------------------------------------------- #
# (b) ladder completeness
# --------------------------------------------------------------------- #


def test_incomplete_rung_fires_for_missing_middle_rung(monkeypatch):
    # Ceiling "apply" needs remember/understand/apply; only remember+apply present.
    blocks = [
        _block(block_id="b-remember", objective_ids=["CO-01"], bloom_level="remember"),
        _block(block_id="b-apply", objective_ids=["CO-01"], bloom_level="apply"),
    ]
    objectives = [_objective("CO-01", "apply")]
    result = _validate(monkeypatch, on=True, blocks=blocks, objectives=objectives)

    incomplete = [i for i in result.issues if i.code == _CODE_INCOMPLETE_RUNG]
    assert len(incomplete) == 1
    assert incomplete[0].location == "CO-01"
    assert "understand" in incomplete[0].message
    assert result.metadata["completeness_audited"] == 3
    assert result.metadata["completeness_passed"] == 2


def test_fully_covered_ladder_has_no_incomplete_issues(monkeypatch):
    blocks = _fully_covered_blocks("CO-01", "apply")
    objectives = [_objective("CO-01", "apply")]
    result = _validate(monkeypatch, on=True, blocks=blocks, objectives=objectives)

    assert not any(i.code == _CODE_INCOMPLETE_RUNG for i in result.issues)
    assert result.metadata["completeness_audited"] == 3
    assert result.metadata["completeness_passed"] == 3
    assert result.score == 1.0
    assert result.passed is True


def test_completeness_uses_ladder_rung_too(monkeypatch):
    # A block with no resolvable `bloom_level` but a valid `ladder_rung`
    # still covers its rung.
    blocks = [
        _block(block_id="b1", objective_ids=["CO-01"], ladder_rung="remember"),
        _block(block_id="b2", objective_ids=["CO-01"], ladder_rung="understand"),
    ]
    objectives = [_objective("CO-01", "understand")]
    result = _validate(monkeypatch, on=True, blocks=blocks, objectives=objectives)

    assert not any(i.code == _CODE_INCOMPLETE_RUNG for i in result.issues)
    assert result.metadata["completeness_audited"] == 2
    assert result.metadata["completeness_passed"] == 2


def test_objective_with_no_ceiling_is_not_completeness_audited(monkeypatch):
    objectives = [{"id": "CO-01", "statement": "no bloom_level here"}]
    result = _validate(monkeypatch, on=True, blocks=[], objectives=objectives)
    assert result.metadata["completeness_audited"] == 0
    assert result.issues == []


# --------------------------------------------------------------------- #
# Objective-ceiling projection (bloom_level / bloomLevel tolerant fallback)
# --------------------------------------------------------------------- #


def test_objective_ceiling_reads_bloom_level():
    assert _objective_ceiling({"bloom_level": "apply"}) == "apply"


def test_objective_ceiling_falls_back_to_camel_case():
    assert _objective_ceiling({"bloomLevel": "Analyze"}) == "analyze"


def test_objective_ceiling_strips_and_lowercases():
    assert _objective_ceiling({"bloom_level": "  Apply  "}) == "apply"


def test_objective_ceiling_none_for_unrecognized_level():
    assert _objective_ceiling({"bloom_level": "not-a-level"}) is None


def test_objective_ceiling_none_when_absent():
    assert _objective_ceiling({}) is None


# --------------------------------------------------------------------- #
# blocks_final_path / synthesized_objectives_path disk hydration
# --------------------------------------------------------------------- #


def test_hydrates_blocks_from_jsonl_path(monkeypatch, tmp_path):
    path = tmp_path / "blocks_final.jsonl"
    rows = [
        _block(block_id="b1", objective_ids=["CO-01"], bloom_level="create"),
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    objectives = [_objective("CO-01", "remember")]
    result = _validate(
        monkeypatch, on=True, blocks_final_path=str(path), objectives=objectives
    )
    assert any(i.code == _CODE_ABOVE_CEILING for i in result.issues)


def test_hydrates_objectives_from_synthesized_objectives_path(monkeypatch, tmp_path):
    path = tmp_path / "synthesized_objectives.json"
    payload = {
        "terminal_objectives": [],
        "chapter_objectives": [_objective("CO-01", "remember")],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    blocks = [_block(block_id="b1", objective_ids=["CO-01"], bloom_level="create")]
    result = _validate(
        monkeypatch,
        on=True,
        blocks=blocks,
        synthesized_objectives_path=str(path),
    )
    assert any(i.code == _CODE_ABOVE_CEILING for i in result.issues)


def test_synthesized_objectives_path_missing_is_empty_map_not_error(monkeypatch, tmp_path):
    missing = tmp_path / "does_not_exist.json"
    blocks = [_block(block_id="b1", objective_ids=["CO-01"], bloom_level="create")]
    result = _validate(
        monkeypatch,
        on=True,
        blocks=blocks,
        synthesized_objectives_path=str(missing),
    )
    assert result.passed is True
    assert result.issues == []
    assert _coerce_objectives({"synthesized_objectives_path": str(missing)}) == {}


# --------------------------------------------------------------------- #
# Missing-blocks structural error (critical, blocking, regardless of the
# warning-day-1 severity of the two real checks)
# --------------------------------------------------------------------- #


def test_missing_blocks_input_is_critical_and_blocking(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    result = BloomLadderCeilingValidator().validate({"objectives": []})
    assert result.passed is False
    assert result.action == "block"
    assert len(result.issues) == 1
    assert result.issues[0].code == _CODE_MISSING_BLOCKS
    assert result.issues[0].severity == "critical"


# --------------------------------------------------------------------- #
# Attribute-bearing (dataclass-shaped) blocks — not just plain dicts
# --------------------------------------------------------------------- #


def test_accepts_attribute_bearing_block_objects(monkeypatch):
    block = SimpleNamespace(
        block_id="b1",
        block_type="concept",
        objective_ids=("CO-01",),
        bloom_level="create",
        ladder_rung=None,
        content="irrelevant",
    )
    objectives = [_objective("CO-01", "remember")]
    result = _validate(monkeypatch, on=True, blocks=[block], objectives=objectives)
    assert any(i.code == _CODE_ABOVE_CEILING for i in result.issues)


# --------------------------------------------------------------------- #
# gate_id passthrough + version/name identity
# --------------------------------------------------------------------- #


def test_gate_id_defaults_to_validator_name(monkeypatch):
    result = _validate(monkeypatch, on=True, blocks=[], objectives=[])
    assert result.gate_id == "bloom_ladder_ceiling"
    assert result.validator_name == "bloom_ladder_ceiling"


def test_gate_id_override_is_honored(monkeypatch):
    result = _validate(
        monkeypatch, on=True, blocks=[], objectives=[], gate_id="custom_gate_id"
    )
    assert result.gate_id == "custom_gate_id"


# --------------------------------------------------------------------- #
# Combined score across both checks
# --------------------------------------------------------------------- #


def test_score_combines_both_checks(monkeypatch):
    # Ceiling check: 1 audited, 1 passed (compliant).
    # Completeness check: ceiling "understand" -> 2 rungs (remember,
    # understand); only "remember" covered -> 1/2.
    blocks = [
        _block(block_id="b1", objective_ids=["CO-01"], bloom_level="understand"),
    ]
    objectives = [_objective("CO-01", "understand")]
    result = _validate(monkeypatch, on=True, blocks=blocks, objectives=objectives)

    assert result.metadata["ceiling_audited"] == 1
    assert result.metadata["ceiling_passed"] == 1
    assert result.metadata["completeness_audited"] == 2
    assert result.metadata["completeness_passed"] == 1
    # total: (1 + 1) / (1 + 2) = 2/3, rounded to 4dp by the validator.
    assert result.score == round(2 / 3, 4)

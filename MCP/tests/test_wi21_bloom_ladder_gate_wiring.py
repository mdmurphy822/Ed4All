"""Bloom-ladder initiative WI-21 — gate wiring regression tests.

WI-21 wires two validators built ahead of this landing (WI-15's
``BloomLadderCeilingValidator`` / WI-16's ``RejectedClaimEntailmentValidator``)
into ``config/workflows.yaml`` + ``MCP/hardening/gate_input_routing.py``. This
suite locks in the three-place contract:

1. ``config/workflows.yaml`` parses and carries the gate rows at the
   documented phases.
2. Both validator dotted paths resolve a REGISTERED builder (never
   ``__no_builder_registered__``) in every phase they appear.
3. The root ``CLAUDE.md`` per-workflow gate-count table matches a fresh
   recount straight off ``config/workflows.yaml`` (the YAML is the source
   of truth — this test would fail loudly on drift in either direction).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from MCP.hardening.gate_input_routing import default_router

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS_YAML = _REPO_ROOT / "config" / "workflows.yaml"
_ROOT_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"

_LADDER_VALIDATOR = "lib.validators.bloom_ladder_ceiling.BloomLadderCeilingValidator"
_REJECTED_VALIDATOR = (
    "lib.validators.pair.rejected_claim_entailment.RejectedClaimEntailmentValidator"
)


def _load_workflows() -> Dict[str, Any]:
    with _WORKFLOWS_YAML.open() as fh:
        return yaml.safe_load(fh)


def _phases_by_name(workflows: Dict[str, Any], workflow_name: str) -> Dict[str, Any]:
    return {p["name"]: p for p in workflows["workflows"][workflow_name]["phases"]}


def _gate_ids(phase: Dict[str, Any]) -> Dict[str, str]:
    """Return ``{gate_id: validator_dotted_path}`` for a phase."""
    return {
        g["gate_id"]: g["validator"]
        for g in phase.get("validation_gates", [])
    }


# ---------------------------------------------------------------------- #
# 1. YAML parses + gate rows land where the spec says
# ---------------------------------------------------------------------- #


def test_workflows_yaml_parses():
    workflows = _load_workflows()
    assert "textbook_to_course" in workflows["workflows"]
    assert "course_generation" in workflows["workflows"]


@pytest.mark.parametrize("workflow_name", ["course_generation", "textbook_to_course"])
@pytest.mark.parametrize("phase_name", ["inter_tier_validation", "post_rewrite_validation"])
def test_bloom_ladder_ceiling_gate_wired_at_both_seams(workflow_name, phase_name):
    """``bloom_ladder_ceiling`` (warning) fires at both two-pass seams in
    BOTH ladder-bearing workflows."""
    workflows = _load_workflows()
    phase = _phases_by_name(workflows, workflow_name)[phase_name]
    gates = _gate_ids(phase)
    assert "bloom_ladder_ceiling" in gates, (
        f"{workflow_name}::{phase_name} is missing the bloom_ladder_ceiling "
        f"gate; saw {sorted(gates)}."
    )
    assert gates["bloom_ladder_ceiling"] == _LADDER_VALIDATOR
    gate_row = next(
        g for g in phase["validation_gates"] if g["gate_id"] == "bloom_ladder_ceiling"
    )
    assert gate_row["severity"] == "warning"


def test_rejected_claim_entailment_gate_wired_at_training_synthesis():
    """``rejected_claim_entailment`` (warning) fires alongside the existing
    pair-tier gates at ``textbook_to_course::training_synthesis`` only —
    ``course_generation`` has no ``training_synthesis`` phase."""
    workflows = _load_workflows()
    tc_phases = _phases_by_name(workflows, "textbook_to_course")
    assert "training_synthesis" in tc_phases
    gates = _gate_ids(tc_phases["training_synthesis"])
    assert "rejected_claim_entailment" in gates, sorted(gates)
    assert gates["rejected_claim_entailment"] == _REJECTED_VALIDATOR
    gate_row = next(
        g
        for g in tc_phases["training_synthesis"]["validation_gates"]
        if g["gate_id"] == "rejected_claim_entailment"
    )
    assert gate_row["severity"] == "warning"

    cg_phases = _phases_by_name(workflows, "course_generation")
    assert "training_synthesis" not in cg_phases


# ---------------------------------------------------------------------- #
# 2. Both gate_ids resolve a REGISTERED builder — never
#    __no_builder_registered__ — in every phase they appear.
# ---------------------------------------------------------------------- #


def test_bloom_ladder_ceiling_and_rejected_claim_entailment_have_builders():
    """Both WI-21 validators must resolve to a real builder.

    Regression guard against silent ``__no_builder_registered__`` skips —
    an unregistered validator makes the executor fall through to
    ``({}, ['__no_builder_registered__'])`` and the (warning-severity) gate
    passes as a vacuous no-op every run.
    """
    r = default_router()
    assert _LADDER_VALIDATOR in r.builders, (
        f"WI-21 regression: no builder registered for {_LADDER_VALIDATOR}; "
        "gate will silently skip via __no_builder_registered__."
    )
    assert _REJECTED_VALIDATOR in r.builders, (
        f"WI-21 regression: no builder registered for {_REJECTED_VALIDATOR}; "
        "gate will silently skip via __no_builder_registered__."
    )


@pytest.mark.parametrize("workflow_name", ["course_generation", "textbook_to_course"])
@pytest.mark.parametrize("phase_name", ["inter_tier_validation", "post_rewrite_validation"])
def test_bloom_ladder_ceiling_builder_resolves_at_every_wired_phase(
    workflow_name, phase_name
):
    """Every ``config/workflows.yaml`` gate row referencing a validator
    dotted path must resolve to a builder registered under that EXACT
    dotted path — walks the YAML rather than a hardcoded phase list, so a
    future phase move / rename is caught."""
    workflows = _load_workflows()
    phase = _phases_by_name(workflows, workflow_name)[phase_name]
    gates = _gate_ids(phase)
    assert "bloom_ladder_ceiling" in gates
    validator_path = gates["bloom_ladder_ceiling"]
    r = default_router()
    assert validator_path in r.builders, (
        f"{workflow_name}::{phase_name}::bloom_ladder_ceiling points at "
        f"{validator_path!r}, which has no registered builder — the gate "
        "will silently skip via __no_builder_registered__."
    )


def test_rejected_claim_entailment_builder_resolves_at_training_synthesis():
    workflows = _load_workflows()
    phase = _phases_by_name(workflows, "textbook_to_course")["training_synthesis"]
    gates = _gate_ids(phase)
    validator_path = gates["rejected_claim_entailment"]
    r = default_router()
    assert validator_path in r.builders, (
        f"textbook_to_course::training_synthesis::rejected_claim_entailment "
        f"points at {validator_path!r}, which has no registered builder."
    )


def test_bloom_ladder_ceiling_builder_actually_runs_at_post_rewrite_validation(
    tmp_path: Path,
):
    """Non-degenerate check: the builder must produce REAL inputs (not just
    be registered) once ``content_generation_rewrite`` has emitted
    ``blocks_final_path`` — the seam where the gate runs for real on a
    normal full build."""
    blocks_final = tmp_path / "blocks_final.jsonl"
    blocks_final.write_text('{"block_id": "b1", "block_type": "concept"}\n')
    objectives = tmp_path / "synthesized_objectives.json"
    objectives.write_text('{"objectives": []}')

    phase_outputs = {
        "content_generation_rewrite": {"blocks_final_path": str(blocks_final)},
        "course_planning": {"synthesized_objectives_path": str(objectives)},
    }
    r = default_router()
    inputs, missing = r.build(_LADDER_VALIDATOR, phase_outputs, {})
    assert missing == [], f"unexpected required_missing: {missing}"
    assert inputs["blocks_final_path"] == str(blocks_final)
    assert inputs["synthesized_objectives_path"] == str(objectives)


def test_bloom_ladder_ceiling_builder_structurally_skips_before_rewrite_emits():
    """At ``inter_tier_validation`` time on a normal full run,
    ``content_generation_rewrite`` has not produced ``blocks_final_path``
    yet — the builder must return a structured ``required_missing``, not a
    silent empty pass."""
    r = default_router()
    inputs, missing = r.build(_LADDER_VALIDATOR, {}, {})
    assert missing == ["blocks_final_path"]


# ---------------------------------------------------------------------- #
# 3. Recount vs CLAUDE.md's "Active Gates" summary table
# ---------------------------------------------------------------------- #


def _recount_gates_from_yaml() -> Dict[str, Dict[str, int]]:
    workflows = _load_workflows()["workflows"]
    counts: Dict[str, Dict[str, int]] = {}
    for wf_name in ("course_generation", "rag_training", "textbook_to_course", "trainforge_train"):
        crit = warn = 0
        for phase in workflows[wf_name]["phases"]:
            for gate in phase.get("validation_gates", []):
                sev = gate.get("severity")
                if sev == "critical":
                    crit += 1
                elif sev == "warning":
                    warn += 1
                else:
                    raise AssertionError(
                        f"Unexpected/unknown gate severity {sev!r} on "
                        f"{wf_name}::{phase['name']}::{gate.get('gate_id')}"
                    )
        counts[wf_name] = {"critical": crit, "warning": warn, "total": crit + warn}
    return counts


def _parse_claude_md_gate_table() -> Dict[str, Dict[str, int]]:
    text = _ROOT_CLAUDE_MD.read_text(encoding="utf-8")
    row_re = re.compile(
        r"\|\s*`(course_generation|rag_training|textbook_to_course|trainforge_train)`"
        r"\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    )
    out: Dict[str, Dict[str, int]] = {}
    for match in row_re.finditer(text):
        name, crit, warn, total = match.groups()
        out[name] = {
            "critical": int(crit),
            "warning": int(warn),
            "total": int(total),
        }
    return out


def test_claude_md_gate_table_matches_yaml_recount():
    """The root CLAUDE.md "Active Gates" summary table must match a fresh
    recount straight off ``config/workflows.yaml`` — the YAML is the
    source of truth. Fails loudly on drift in either direction (a gate
    added to the YAML without updating the table, or vice versa)."""
    from_yaml = _recount_gates_from_yaml()
    from_doc = _parse_claude_md_gate_table()
    assert set(from_yaml) <= set(from_doc), (
        f"CLAUDE.md gate table is missing workflow rows: "
        f"{set(from_yaml) - set(from_doc)}"
    )
    for wf_name, expected in from_yaml.items():
        assert from_doc[wf_name] == expected, (
            f"CLAUDE.md gate-count row for {wf_name!r} is stale: doc says "
            f"{from_doc[wf_name]}, config/workflows.yaml recounts to "
            f"{expected}."
        )


def test_wi21_landing_matches_projected_counts():
    """Pins the exact per-workflow gate counts so a future unrelated gate
    addition doesn't accidentally paper over a WI-21 regression.

    Snapshot advanced once for the Bloom-ladder addendum AD-02 landing
    (``dpo_yield_projection``, warning, ``textbook_to_course::
    training_synthesis``): textbook_to_course warnings 78 -> 79, grand
    warning total 114 -> 115. Any FUTURE gate landing must advance this
    pin again in the same change that recounts the root CLAUDE.md table.
    """
    counts = _recount_gates_from_yaml()
    assert counts["course_generation"] == {"critical": 33, "warning": 33, "total": 66}
    assert counts["textbook_to_course"] == {"critical": 64, "warning": 79, "total": 143}
    assert counts["rag_training"] == {"critical": 4, "warning": 3, "total": 7}
    assert counts["trainforge_train"] == {"critical": 2, "warning": 0, "total": 2}
    grand_critical = sum(c["critical"] for c in counts.values())
    grand_warning = sum(c["warning"] for c in counts.values())
    assert grand_critical == 103
    assert grand_warning == 115
    assert grand_critical + grand_warning == 218

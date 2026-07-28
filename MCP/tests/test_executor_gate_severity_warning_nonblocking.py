"""Gate-severity contract for ``TaskExecutor.execute_phase``.

Two contracts are pinned here:

1. **Warning failures never block; critical ones do.** The executor's
   only ``gates_passed = False`` site requires ``GateSeverity.CRITICAL``
   — a failing warning gate must leave ``gates_passed=True``.

2. **Persisted / returned gate results are severity-auditable.**
   ``GateResult`` carries no ``severity`` field; without the executor
   stamping the declared ``GateConfig`` severity onto each result dict,
   a persisted checkpoint cannot distinguish a blocking (critical)
   failure from an advisory (warning) one — and an all-warning failure
   set reads as an unexplained phase-level ``gates_passed=False``. These
   tests assert the severity is present on every persisted/returned
   result so that diagnosis stays possible.

In-process against a synthetic ``TaskExecutor`` — no live workflow, no
checkpoint manager, no filesystem.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MCP.core.executor import TaskExecutor
from MCP.hardening.validation_gates import GateResult


class _AllFailManager:
    """Gate manager whose gates all FAIL — isolates severity handling."""

    def run_gate(self, gate, merged_inputs):  # noqa: D401
        return GateResult(
            gate_id=gate.gate_id,
            validator_name=gate.validator_path,
            validator_version="all-fail-manager",
            passed=False,
            score=0.0,
            issues=[],
        )


class _NullRouter:
    """Router that resolves every validator with an empty input blob
    (never reports missing inputs, so no gate is skipped)."""

    def build(self, validator_path, phase_outputs, workflow_params):
        return {}, []


def _executor() -> TaskExecutor:
    ex = TaskExecutor(tool_registry={}, max_retries=0)
    ex.gate_manager = _AllFailManager()
    ex.gate_input_router = _NullRouter()
    ex.checkpoint_manager = None  # no filesystem persistence in-test
    return ex


async def _run(
    gate_configs: List[Dict[str, Any]],
    *,
    phase_name: str = "post_rewrite_validation",
    workflow_params: Dict[str, Any] | None = None,
    router=None,
):
    ex = _executor()
    if router is not None:
        ex.gate_input_router = router

    async def _fake_parallel(*_a, **_k):
        return {}

    with patch.object(ex, "_execute_parallel", _fake_parallel):
        _results, gates_passed, gate_results = await ex.execute_phase(
            workflow_id="W_gate_sev_001",
            phase_name=phase_name,
            phase_index=1,
            tasks=[],
            gate_configs=gate_configs,
            max_concurrent=1,
            phase_outputs={},
            workflow_params=workflow_params or {},
            extract_phase_outputs_fn=None,
        )
    return gates_passed, gate_results


@pytest.mark.asyncio
async def test_required_training_synthesis_missing_gate_input_blocks():
    class _MissingRouter:
        def build(self, validator_path, phase_outputs, workflow_params):
            return {}, ["instruction_pairs_path"]

    gates_passed, gate_results = await _run(
        [{
            "gate_id": "synthesis_quota",
            "validator": "lib.validators.synthesis_quota.SynthesisQuotaValidator",
            "severity": "warning",
            "threshold": {},
        }],
        phase_name="training_synthesis",
        workflow_params={"with_training": True},
        router=_MissingRouter(),
    )

    assert gates_passed is False
    result = _by_id(gate_results)["synthesis_quota"]
    assert result["passed"] is False
    assert result["waiver_info"]["reason"] == "instruction_pairs_path"


def _by_id(gate_results):
    return {gr.get("gate_id"): gr for gr in (gate_results or [])}


# --------------------------------------------------------------------- #
# 1. Warning-only failures do NOT block, and the verdict is
#    severity-auditable.
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_warning_only_failures_do_not_block_and_are_auditable():
    gate_configs = [
        {
            "gate_id": "rewrite_source_grounding",
            "validator": "lib.validators.x.X",
            "severity": "warning",
            "threshold": {},
        },
        {
            "gate_id": "retrieval_presence",
            "validator": "lib.validators.y.Y",
            "severity": "warning",
            "threshold": {},
        },
    ]
    gates_passed, gate_results = await _run(gate_configs)

    # Contract 1: warning failures never block.
    assert gates_passed is True, (
        "A phase whose only failing gates are warning-severity MUST "
        "yield gates_passed=True."
    )

    # Contract 2: each persisted/returned result carries its severity so
    # a post-hoc reader can prove the failures were advisory, not
    # blocking (anti-diagnosis-blinding).
    by_id = _by_id(gate_results)
    assert set(by_id) == {"rewrite_source_grounding", "retrieval_presence"}
    for gid, gr in by_id.items():
        assert gr.get("passed") is False
        assert gr.get("severity") == "warning", (
            f"gate {gid} result must record severity='warning' so a "
            "blocking-vs-advisory diagnosis is possible from the "
            "persisted gate_results alone"
        )


# --------------------------------------------------------------------- #
# 2. A failing critical gate blocks.
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_critical_failure_blocks_and_is_labelled():
    gate_configs = [
        {
            "gate_id": "block_prose_entailment",
            "validator": "lib.validators.x.X",
            "severity": "critical",
            "threshold": {},
        },
    ]
    gates_passed, gate_results = await _run(gate_configs)
    assert gates_passed is False
    assert _by_id(gate_results)["block_prose_entailment"]["severity"] == "critical"


# --------------------------------------------------------------------- #
# 3. Mixed critical + warning failures block (on the critical), and
#    BOTH severities are recorded.
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mixed_critical_and_warning_blocks_on_critical():
    gate_configs = [
        {
            "gate_id": "rewrite_source_grounding",
            "validator": "lib.validators.x.X",
            "severity": "warning",
            "threshold": {},
        },
        {
            "gate_id": "block_prose_entailment",
            "validator": "lib.validators.y.Y",
            "severity": "critical",
            "threshold": {},
        },
    ]
    gates_passed, gate_results = await _run(gate_configs)
    assert gates_passed is False, "a failing CRITICAL gate must block"
    by_id = _by_id(gate_results)
    assert by_id["rewrite_source_grounding"]["severity"] == "warning"
    assert by_id["block_prose_entailment"]["severity"] == "critical"


# --------------------------------------------------------------------- #
# 4. A gate dict that OMITS severity defaults to CRITICAL (fail-closed)
#    AND logs loudly — a silent omission must never quietly block.
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_missing_severity_defaults_critical_and_warns(caplog):
    gate_configs = [
        {
            "gate_id": "gate_without_severity",
            "validator": "lib.validators.x.X",
            "threshold": {},
        },
    ]
    with caplog.at_level("WARNING"):
        gates_passed, gate_results = await _run(gate_configs)

    assert gates_passed is False, "missing severity is fail-closed (critical)"
    assert _by_id(gate_results)["gate_without_severity"]["severity"] == "critical"
    assert any(
        "gate_without_severity" in rec.getMessage()
        and "severity" in rec.getMessage().lower()
        for rec in caplog.records
    ), "a gate omitting severity must emit a loud WARNING"


# --------------------------------------------------------------------- #
# 5. The YAML ``behavior:`` block is forwarded into the parsed
#    GateConfig.
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_behavior_block_forwarded_into_gateconfig():
    from MCP.hardening.validation_gates import GateBehavior

    captured: List[Any] = []

    class _CapturingManager:
        def run_gate(self, gate, merged_inputs):
            captured.append(gate)
            return GateResult(
                gate_id=gate.gate_id,
                validator_name=gate.validator_path,
                validator_version="capture",
                passed=True,
                score=1.0,
                issues=[],
            )

    ex = _executor()
    ex.gate_manager = _CapturingManager()

    gate_configs = [
        {
            "gate_id": "gate_with_behavior",
            "validator": "lib.validators.x.X",
            "severity": "warning",
            "threshold": {},
            "behavior": {"on_fail": "warn", "on_error": "warn"},
        },
    ]

    async def _fake_parallel(*_a, **_k):
        return {}

    with patch.object(ex, "_execute_parallel", _fake_parallel):
        await ex.execute_phase(
            workflow_id="W_gate_sev_beh",
            phase_name="post_rewrite_validation",
            phase_index=1,
            tasks=[],
            gate_configs=gate_configs,
            max_concurrent=1,
            phase_outputs={},
            workflow_params={},
            extract_phase_outputs_fn=None,
        )

    assert captured, "gate must have run"
    gate = captured[0]
    assert gate.behavior_on_fail == GateBehavior.WARN, (
        "the YAML behavior.on_fail must be forwarded into GateConfig; "
        "hand-rolling GateConfig drops it"
    )
    assert gate.behavior_on_error == GateBehavior.WARN


# --------------------------------------------------------------------- #
# 6. Forwarding must NOT mutate the caller's shared phase config dict
#    (GateConfig.from_dict pops the ``behavior`` key).
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_parse_does_not_mutate_shared_config_dict():
    gate_configs = [
        {
            "gate_id": "g_no_mutate",
            "validator": "lib.validators.x.X",
            "severity": "warning",
            "threshold": {},
            "behavior": {"on_fail": "block", "on_error": "fail_closed"},
        },
    ]
    await _run(gate_configs)
    assert "behavior" in gate_configs[0], (
        "the shared phase.validation_gates dict must not be mutated by "
        "gate parsing (from_dict pops 'behavior' — parse must copy first)"
    )

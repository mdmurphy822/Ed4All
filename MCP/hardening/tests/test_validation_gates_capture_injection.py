"""H3 Worker S0.5 regression: gate-runner injects ``decision_capture``
and the legacy ``capture`` alias into the inputs dict every validator's
``validate(inputs)`` sees.

Pre-S0.5 contract: the orchestrator's ``TaskExecutor`` carried a
live ``self.capture`` instance, but never threaded it into the
``merged_inputs`` blob handed to validators. Pattern A emitters
(``KGQualityValidator``, ``RewriteHtmlShapeValidator``,
``RewriteSourceGroundingValidator``, ``ABCDObjectiveValidator``,
``SHACLResultEnricher``) read ``inputs.get("decision_capture")``;
Pattern B emitters (``FamilyCompletenessValidator``,
``EvalGatingValidator``) read ``inputs.get("capture")``. In production
both lookups returned ``None`` and the ``if capture is None: return``
guard short-circuited 100% of emit code. This test pins the dual-key
injection at both gate-runner seams (executor + GateManager) and the
``setdefault`` semantics that preserve any explicit per-call override.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from MCP.core.executor import TaskExecutor
from MCP.hardening.validation_gates import (
    GateBehavior,
    GateConfig,
    GateResult,
    GateSeverity,
    ValidationGateManager,
)


class _StubValidator:
    """Minimal validator stub that snapshots the inputs dict it sees.

    Bypasses the ``ALLOWED_VALIDATOR_PREFIXES`` allowlist by being
    pre-inserted into ``ValidationGateManager._validators`` under a
    sentinel dotted path; ``load_validator`` short-circuits on the
    cache hit.
    """

    name = "stub_capture_injection"
    version = "1.0.0"

    def __init__(self) -> None:
        self.captured_inputs: List[Dict[str, Any]] = []

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        # Snapshot a shallow copy so a subsequent mutation by
        # the gate manager doesn't poison the assertion.
        self.captured_inputs.append(dict(inputs))
        return GateResult(
            gate_id="stub",
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
        )


def _make_gate(stub_path: str = "lib.validators.kg_quality.StubValidator") -> GateConfig:
    """Build a minimal GateConfig pointing at a sentinel dotted path.

    The path is never imported because the stub validator is
    pre-injected into the manager's ``_validators`` cache.
    """
    return GateConfig(
        gate_id="stub_gate",
        validator_path=stub_path,
        severity=GateSeverity.WARNING,
        behavior_on_fail=GateBehavior.WARN,
        behavior_on_error=GateBehavior.WARN,
    )


# ---------------------------------------------------------------------
# ValidationGateManager direct-invocation path
# ---------------------------------------------------------------------


def test_validation_gate_manager_run_gate_injects_capture_keys() -> None:
    """When ``ValidationGateManager`` is constructed with a capture, the
    direct-invocation ``run_gate`` path injects BOTH ``decision_capture``
    and ``capture`` into the inputs the validator sees.
    """
    capture = MagicMock(name="DecisionCapture")
    manager = ValidationGateManager(capture=capture)

    stub = _StubValidator()
    sentinel_path = "lib.validators.kg_quality.StubValidatorOne"
    manager._validators[sentinel_path] = stub

    gate = _make_gate(sentinel_path)
    manager.run_gate(gate, inputs={"some_artifact": "x"})

    assert stub.captured_inputs, "stub validator was not invoked"
    seen = stub.captured_inputs[0]
    assert seen.get("decision_capture") is capture
    assert seen.get("capture") is capture
    # Original input keys still present.
    assert seen.get("some_artifact") == "x"


def test_validation_gate_manager_setdefault_preserves_per_call_capture() -> None:
    """If the caller passed an explicit ``decision_capture`` / ``capture``
    in ``inputs``, ``setdefault`` MUST NOT clobber it with the manager's
    stored capture.
    """
    manager_capture = MagicMock(name="ManagerCapture")
    explicit_capture = MagicMock(name="ExplicitCapture")
    manager = ValidationGateManager(capture=manager_capture)

    stub = _StubValidator()
    sentinel_path = "lib.validators.kg_quality.StubValidatorTwo"
    manager._validators[sentinel_path] = stub

    gate = _make_gate(sentinel_path)
    manager.run_gate(
        gate,
        inputs={
            "decision_capture": explicit_capture,
            "capture": explicit_capture,
        },
    )

    seen = stub.captured_inputs[0]
    assert seen["decision_capture"] is explicit_capture
    assert seen["capture"] is explicit_capture
    assert seen["decision_capture"] is not manager_capture


def test_validation_gate_manager_no_capture_keeps_legacy_contract() -> None:
    """When ``ValidationGateManager()`` is constructed with no capture
    (legacy / pre-S0.5 callers), neither key is injected so existing
    tests that assert "capture is None" semantics still pass.
    """
    manager = ValidationGateManager()  # no capture kwarg
    stub = _StubValidator()
    sentinel_path = "lib.validators.kg_quality.StubValidatorThree"
    manager._validators[sentinel_path] = stub

    gate = _make_gate(sentinel_path)
    manager.run_gate(gate, inputs={"some_artifact": "x"})

    seen = stub.captured_inputs[0]
    assert "decision_capture" not in seen
    assert "capture" not in seen


# ---------------------------------------------------------------------
# TaskExecutor._run_validation_gates seam (execute_phase)
# ---------------------------------------------------------------------


def _run_phase(executor: TaskExecutor, gate_configs: list) -> None:
    """Drive ``execute_phase`` synchronously for the test suite.

    The phase has zero tasks so the executor skips straight to gate
    evaluation against an empty ``results`` dict.
    """
    asyncio.get_event_loop().run_until_complete(
        executor.execute_phase(
            workflow_id="WF-S0.5-CAPTURE-TEST",
            phase_name="stub_phase",
            phase_index=0,
            tasks=[],
            gate_configs=gate_configs,
            max_concurrent=1,
        )
    )


def test_executor_run_validation_gates_injects_capture_keys() -> None:
    """``TaskExecutor`` constructs ``ValidationGateManager`` with its
    own ``self.capture`` and the gate-runner seam stamps both keys into
    ``merged_inputs`` before ``run_gate`` fires.
    """
    capture = MagicMock(name="ExecutorCapture")
    executor = TaskExecutor(capture=capture)
    if executor.gate_manager is None:
        pytest.skip("HARDENING_VALIDATION_GATES disabled — gate manager not constructed")

    stub = _StubValidator()
    sentinel_path = "lib.validators.kg_quality.StubValidatorExecutor"
    executor.gate_manager._validators[sentinel_path] = stub

    # Disable the gate-input router so the executor falls back to the
    # fallback_inputs blob path (router=None branch in execute_phase).
    # That path is what real legacy gates exercise when no builder is
    # registered, and it's the path that flows through the new
    # ``setdefault`` injection.
    executor.gate_input_router = None

    gate_configs = [
        {
            "gate_id": "stub_executor_gate",
            "validator": sentinel_path,
            "severity": "warning",
            "behavior": {"on_fail": "warn", "on_error": "warn"},
        }
    ]

    _run_phase(executor, gate_configs)

    assert stub.captured_inputs, "executor seam never invoked stub validator"
    seen = stub.captured_inputs[0]
    assert seen.get("decision_capture") is capture
    assert seen.get("capture") is capture


def test_executor_run_validation_gates_preserves_explicit_per_call_override() -> None:
    """If a future builder injects ``decision_capture`` explicitly into
    the per-gate inputs (e.g., a phase-scoped sub-capture), the
    executor's ``setdefault`` MUST preserve that override rather than
    clobber it with ``self.capture``.

    Today, no builder injects either key, so this is a forward-compat
    contract test driven via the GateManager mirror — same ``setdefault``
    semantics are asserted at both seams in series so a future
    regression at either site fails loudly.
    """
    executor_capture = MagicMock(name="ExecutorCaptureOuter")
    builder_capture = MagicMock(name="BuilderInjectedCapture")
    executor = TaskExecutor(capture=executor_capture)
    if executor.gate_manager is None:
        pytest.skip("HARDENING_VALIDATION_GATES disabled — gate manager not constructed")

    # Stub the manager's run_gate so we can directly observe the
    # merged_inputs dict produced by the executor seam — the manager's
    # own setdefault is exercised by the manager-level tests above.
    captured_merged: Dict[str, Any] = {}

    def _spy_run_gate(gate_config: GateConfig, inputs: Dict[str, Any]) -> GateResult:
        captured_merged.update(inputs)
        return GateResult(
            gate_id=gate_config.gate_id,
            validator_name=gate_config.validator_path,
            validator_version="spy",
            passed=True,
        )

    executor.gate_manager.run_gate = _spy_run_gate  # type: ignore[assignment]

    # Inject the builder override via the gate input router. Easiest
    # path: monkey-patch the router to return the override map for our
    # sentinel validator path. We bypass the router by setting it to
    # None — then ``inputs = dict(fallback_inputs)`` runs and we instead
    # smuggle the override through the fallback_inputs blob, which is
    # populated from task results' ``artifacts`` / ``result`` dicts.
    # Cleaner: directly call the executor's gate-input merge by hand
    # is impossible from the public surface, so we install a synthetic
    # router that emits the override.
    class _OverrideRouter:
        def build(self, validator_path: str, phase_outputs: Dict, workflow_params: Dict):
            return (
                {"decision_capture": builder_capture, "capture": builder_capture},
                [],
            )

    executor.gate_input_router = _OverrideRouter()  # type: ignore[assignment]

    gate_configs = [
        {
            "gate_id": "stub_override_gate",
            "validator": "lib.validators.kg_quality.StubValidatorOverride",
            "severity": "warning",
            "behavior": {"on_fail": "warn", "on_error": "warn"},
        }
    ]

    _run_phase(executor, gate_configs)

    assert captured_merged.get("decision_capture") is builder_capture
    assert captured_merged.get("capture") is builder_capture
    assert captured_merged.get("decision_capture") is not executor_capture

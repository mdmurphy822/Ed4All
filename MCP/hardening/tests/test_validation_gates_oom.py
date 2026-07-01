"""W2.1 regression: a CUDA OOM raised inside a validator is NEVER a silent pass.

Bug: ``ValidationGateManager.run_gate`` caught every validator exception in one
broad ``except`` and, when the gate is configured ``behavior_on_error=warn``,
rewrote the result to ``passed=True`` — so a ``torch.cuda.OutOfMemoryError``
raised during an NLI / embedding forward pass on a VRAM-starved box became a
SILENT auto-pass (buried in a generic ``VALIDATOR_ERROR`` that also fires for
any validator bug). The loud OOM diagnostic existed only on the executor's
task path.

Fix: detect the OOM (shared ``lib.llm.oom.is_cuda_oom``) and emit a DISTINCT
``VALIDATOR_OOM`` issue + a DecisionCapture. Default OFF still honours the
gate's ``behavior_on_error`` for pass/block (no NEW blocking) but the OOM is
now greppable, not silent. The opt-in ``ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM``
escalates to fail-closed/block regardless of ``behavior_on_error``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from MCP.hardening.validation_gates import (
    GateBehavior,
    GateConfig,
    GateResult,
    GateSeverity,
    ValidationGateManager,
)


# A class literally named ``OutOfMemoryError`` so ``is_cuda_oom`` detects it
# torch-free via its class-name fallback (mirrors the executor test's tactic).
class OutOfMemoryError(Exception):  # noqa: A001 - deliberately shadow builtin name
    """Stand-in for ``torch.cuda.OutOfMemoryError`` (no torch in CI)."""


class _OOMRaisingValidator:
    name = "oom_raiser"
    version = "1.0.0"

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def validate(self, inputs):  # noqa: ANN001, ANN201
        raise self._exc


class _PlainErrorValidator:
    name = "plain_error"
    version = "1.0.0"

    def validate(self, inputs):  # noqa: ANN001, ANN201
        raise ValueError("some ordinary validator bug")


def _install(manager: ValidationGateManager, validator, path: str) -> None:
    manager._validators[path] = validator


def _oom_gate(path: str, on_error: GateBehavior = GateBehavior.WARN) -> GateConfig:
    return GateConfig(
        gate_id="oom_gate",
        validator_path=path,
        severity=GateSeverity.WARNING,
        behavior_on_fail=GateBehavior.WARN,
        behavior_on_error=on_error,
    )


def _issue_codes(result: GateResult):
    return {i.code for i in result.issues}


def test_oom_default_warn_emits_distinct_issue_not_silent_pass(monkeypatch):
    """Default (flag off) + on_error=warn: passes (non-blocking) BUT surfaces a
    distinct VALIDATOR_OOM warning — never the pre-fix silent generic pass."""
    monkeypatch.delenv("ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM", raising=False)
    capture = MagicMock(name="capture")
    manager = ValidationGateManager(capture=capture)
    path = "lib.validators.kg_quality.OOMStubWarn"
    _install(manager, _OOMRaisingValidator(OutOfMemoryError("CUDA out of memory")), path)

    with patch(
        "MCP.hardening.validation_gates.probe_free_vram_mib", return_value=137
    ):
        result = manager.run_gate(_oom_gate(path), inputs={})

    # Honours behavior_on_error=warn → non-blocking, but NOT silent.
    assert result.passed is True
    assert "VALIDATOR_OOM" in _issue_codes(result)
    assert "VALIDATOR_ERROR" not in _issue_codes(result)
    assert result.validator_version == "oom"
    oom_issue = next(i for i in result.issues if i.code == "VALIDATOR_OOM")
    assert oom_issue.severity == "warning"
    assert "137 MiB free" in oom_issue.message
    # DecisionCapture fired on the OOM branch.
    assert capture.log_decision.called
    kwargs = capture.log_decision.call_args.kwargs
    assert kwargs["decision_type"] == "validation_result"
    assert "OOM" in kwargs["decision"]


def test_oom_default_fail_closed_behavior_blocks(monkeypatch):
    """Default (flag off) + on_error=fail_closed: OOM fails the gate closed."""
    monkeypatch.delenv("ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM", raising=False)
    manager = ValidationGateManager()
    path = "lib.validators.kg_quality.OOMStubFC"
    _install(
        manager,
        _OOMRaisingValidator(RuntimeError("CUDA out of memory. Tried to allocate 2GiB")),
        path,
    )

    with patch(
        "MCP.hardening.validation_gates.probe_free_vram_mib", return_value=None
    ):
        result = manager.run_gate(
            _oom_gate(path, on_error=GateBehavior.FAIL_CLOSED), inputs={}
        )

    assert result.passed is False
    oom_issue = next(i for i in result.issues if i.code == "VALIDATOR_OOM")
    assert oom_issue.severity == "critical"
    assert "free VRAM unprobeable" in oom_issue.message


def test_oom_flag_forces_fail_closed_even_when_behavior_warn(monkeypatch):
    """ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM=1: OOM blocks despite on_error=warn."""
    monkeypatch.setenv("ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM", "1")
    manager = ValidationGateManager()
    path = "lib.validators.kg_quality.OOMStubFlag"
    _install(manager, _OOMRaisingValidator(OutOfMemoryError("cuda oom")), path)

    with patch(
        "MCP.hardening.validation_gates.probe_free_vram_mib", return_value=42
    ):
        result = manager.run_gate(_oom_gate(path), inputs={})

    assert result.passed is False
    oom_issue = next(i for i in result.issues if i.code == "VALIDATOR_OOM")
    assert oom_issue.severity == "critical"


def test_oom_flag_garbage_value_stays_off(monkeypatch):
    """Garbage flag value → off (parse-with-fallback): warn behavior honoured."""
    monkeypatch.setenv("ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM", "banana")
    manager = ValidationGateManager()
    path = "lib.validators.kg_quality.OOMStubGarbage"
    _install(manager, _OOMRaisingValidator(OutOfMemoryError("CUDA out of memory")), path)

    with patch(
        "MCP.hardening.validation_gates.probe_free_vram_mib", return_value=1
    ):
        result = manager.run_gate(_oom_gate(path), inputs={})

    assert result.passed is True  # warn honoured, flag off
    assert "VALIDATOR_OOM" in _issue_codes(result)


def test_non_oom_exception_unchanged(monkeypatch):
    """A non-OOM validator exception still routes to the generic VALIDATOR_ERROR
    path — the OOM branch must not hijack ordinary errors."""
    monkeypatch.delenv("ED4ALL_VALIDATOR_FAIL_CLOSED_ON_OOM", raising=False)
    manager = ValidationGateManager()
    path = "lib.validators.kg_quality.PlainErr"
    _install(manager, _PlainErrorValidator(), path)

    # on_error=warn → generic error is treated as a warning-pass (unchanged
    # legacy behaviour), with the generic VALIDATOR_ERROR code.
    result = manager.run_gate(_oom_gate(path), inputs={})

    assert "VALIDATOR_ERROR" in _issue_codes(result)
    assert "VALIDATOR_OOM" not in _issue_codes(result)
    assert result.validator_version == "error"
    assert result.passed is True  # on_error=warn

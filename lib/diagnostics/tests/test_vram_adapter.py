"""Unit tests for :mod:`lib.diagnostics.vram` — the VRAM check adapter.

Hermetic + GPU-free: ``snapshot_vram`` / ``fit_check`` are monkeypatched
to return constructed :class:`VramSnapshot` / :class:`FitVerdict` objects,
so the suite never touches a GPU, NVML, or an ollama server. Asserts the
outcome→severity mapping, the snapshot-summary result, and the
never-raising contract.
"""

from __future__ import annotations

import pytest

from lib.diagnostics import vram as vram_adapter
from lib.diagnostics.core import (
    CheckContext,
    Severity,
    clear_registry,
    registered_checks,
    resolve_exit_code,
)
from lib.llm.vram_doctor import FitVerdict, VramSnapshot


@pytest.fixture(autouse=True)
def _isolate_registry() -> None:
    clear_registry()
    yield
    clear_registry()


def _snapshot(**kw) -> VramSnapshot:
    defaults = dict(
        free_mib=6000,
        total_mib=8000,
        cuda_available=True,
        probe_source="nvml",
        resident_models=[{"name": "qwen2.5:7b", "vram_mib": 5300}],
        error=None,
    )
    defaults.update(kw)
    return VramSnapshot(**defaults)


def _verdict(consumer: str, outcome: str, **kw) -> FitVerdict:
    defaults = dict(
        consumer=consumer,
        device_requested="cuda",
        need_mib=900,
        free_mib=6000,
        outcome=outcome,
        detail=f"{consumer} {outcome}",
    )
    defaults.update(kw)
    return FitVerdict(**defaults)


def _patch(monkeypatch, snapshot, verdicts):
    monkeypatch.setattr(
        "lib.llm.vram_doctor.snapshot_vram", lambda base_url=None: snapshot
    )
    monkeypatch.setattr("lib.llm.vram_doctor.fit_check", lambda snap: verdicts)


# --------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------- #


def test_register_gpu_checks_no_import_side_effect() -> None:
    # Importing the module must not have registered anything.
    assert registered_checks() == []
    vram_adapter.register_gpu_checks()
    pairs = registered_checks()
    assert [g for g, _ in pairs] == ["gpu"]
    assert pairs[0][1] is vram_adapter.gpu_checks


# --------------------------------------------------------------------- #
# Snapshot-summary result
# --------------------------------------------------------------------- #


def test_emits_snapshot_summary_result(monkeypatch) -> None:
    _patch(monkeypatch, _snapshot(), [_verdict("nli", "fits")])
    results = vram_adapter.gpu_checks(CheckContext())
    summary = results[0]
    assert summary.name == "gpu_snapshot"
    assert summary.group == "gpu"
    assert summary.severity is Severity.OK
    assert summary.data["free_mib"] == 6000
    assert summary.data["total_mib"] == 8000
    assert summary.data["probe_source"] == "nvml"
    assert summary.data["resident_models"]


# --------------------------------------------------------------------- #
# Outcome → Severity mapping
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "outcome,expected",
    [
        ("would_oom", Severity.FAIL),
        ("would_fail_closed", Severity.FAIL),
        ("would_fallback_cpu", Severity.WARN),
        ("unknown", Severity.INFO),
        ("fits", Severity.OK),
        ("cpu_requested", Severity.OK),
        ("would_evict_then_fit", Severity.OK),
    ],
)
def test_outcome_severity_mapping(monkeypatch, outcome, expected) -> None:
    _patch(monkeypatch, _snapshot(), [_verdict("nli", outcome)])
    results = vram_adapter.gpu_checks(CheckContext())
    fit = next(r for r in results if r.name == "gpu_fit_nli")
    assert fit.severity is expected
    assert fit.data["outcome"] == outcome
    assert fit.data["consumer"] == "nli"
    assert fit.data["need_mib"] == 900


def test_unknown_outcome_is_info_with_clear_summary(monkeypatch) -> None:
    # ``unknown`` (the WSL2 free-VRAM probe quirk) is INFO, not WARN — it is
    # not evidence of a problem and must carry a clear, non-alarming summary.
    _patch(monkeypatch, _snapshot(free_mib=None), [_verdict("nli", "unknown", free_mib=None)])
    results = vram_adapter.gpu_checks(CheckContext())
    fit = next(r for r in results if r.name == "gpu_fit_nli")
    assert fit.severity is Severity.INFO
    assert "not necessarily" in fit.summary
    # INFO never renders remediation.
    assert fit.remediation == ""


def test_unknown_outcome_does_not_escalate_exit_code(monkeypatch) -> None:
    # An all-``unknown`` fit set (plus the OK snapshot summary) must exit 0:
    # INFO is below WARN and never pushes resolve_exit_code above 0.
    _patch(
        monkeypatch,
        _snapshot(free_mib=None),
        [_verdict("nli", "unknown", free_mib=None), _verdict("embedding", "unknown", free_mib=None)],
    )
    results = vram_adapter.gpu_checks(CheckContext())
    assert all(
        r.severity is Severity.INFO for r in results if r.name.startswith("gpu_fit_")
    )
    assert resolve_exit_code(results) == 0


def test_unrecognized_outcome_defaults_to_warn(monkeypatch) -> None:
    # A genuinely unexpected (unmapped) outcome still flags WARN — only the
    # well-defined ``unknown`` is demoted to INFO.
    _patch(monkeypatch, _snapshot(), [_verdict("nli", "some_new_outcome")])
    results = vram_adapter.gpu_checks(CheckContext())
    fit = next(r for r in results if r.name == "gpu_fit_nli")
    assert fit.severity is Severity.WARN
    assert resolve_exit_code(results) == 1


def test_remediation_populated_for_warn_and_fail(monkeypatch) -> None:
    _patch(
        monkeypatch,
        _snapshot(),
        [_verdict("nli", "would_oom"), _verdict("embedding", "would_fallback_cpu")],
    )
    results = vram_adapter.gpu_checks(CheckContext())
    oom = next(r for r in results if r.name == "gpu_fit_nli")
    fallback = next(r for r in results if r.name == "gpu_fit_embedding")
    assert oom.remediation
    assert fallback.remediation


def test_fail_closed_and_fallback_render_differently(monkeypatch) -> None:
    """The two cuda-unavailable outcomes must stay distinguishable in the UI.

    ``would_fallback_cpu`` (NLI — a blessed CPU fallback) is a WARN whose
    remediation offers to accept a slowdown. ``would_fail_closed``
    (embedding — the backend RAISES, no automatic CUDA→CPU downgrade) is a
    FAIL whose remediation names the operator's explicit CPU opt-out. Sharing
    the fallback's text told the operator the run was safe when it was not.
    """
    _patch(
        monkeypatch,
        _snapshot(cuda_available=False, free_mib=None),
        [
            _verdict("nli", "would_fallback_cpu", free_mib=None),
            _verdict("embedding", "would_fail_closed", free_mib=None),
        ],
    )
    results = vram_adapter.gpu_checks(CheckContext())
    fallback = next(r for r in results if r.name == "gpu_fit_nli")
    fail_closed = next(r for r in results if r.name == "gpu_fit_embedding")

    assert fallback.severity is Severity.WARN
    assert "fall back to CPU" in fallback.remediation

    assert fail_closed.severity is Severity.FAIL
    assert "NO CPU fallback" in fail_closed.remediation
    assert "ED4ALL_EMBEDDING_DEVICE=cpu" in fail_closed.remediation
    assert fail_closed.remediation != fallback.remediation
    # A fail-closed prediction must push the doctor's exit code to the hard
    # failure code, not the warning one.
    assert resolve_exit_code(results) == 2


def test_remediation_empty_for_ok(monkeypatch) -> None:
    _patch(monkeypatch, _snapshot(), [_verdict("nli", "fits")])
    results = vram_adapter.gpu_checks(CheckContext())
    fit = next(r for r in results if r.name == "gpu_fit_nli")
    assert fit.remediation == ""


def test_one_result_per_verdict_plus_snapshot(monkeypatch) -> None:
    _patch(
        monkeypatch,
        _snapshot(),
        [_verdict("nli", "fits"), _verdict("embedding", "would_oom")],
    )
    results = vram_adapter.gpu_checks(CheckContext())
    names = [r.name for r in results]
    assert names == ["gpu_snapshot", "gpu_fit_nli", "gpu_fit_embedding"]


def test_base_url_threaded_to_snapshot(monkeypatch) -> None:
    seen = {}

    def fake_snapshot(base_url=None):
        seen["base_url"] = base_url
        return _snapshot()

    monkeypatch.setattr("lib.llm.vram_doctor.snapshot_vram", fake_snapshot)
    monkeypatch.setattr("lib.llm.vram_doctor.fit_check", lambda s: [])
    vram_adapter.gpu_checks(CheckContext(base_url="http://host:11434"))
    assert seen["base_url"] == "http://host:11434"


# --------------------------------------------------------------------- #
# Never-raises contract
# --------------------------------------------------------------------- #


def test_degraded_snapshot_does_not_raise(monkeypatch) -> None:
    degraded = _snapshot(
        free_mib=None,
        total_mib=None,
        cuda_available=False,
        probe_source="unavailable",
        resident_models=[],
        error="ollama /api/ps unreachable",
    )
    _patch(monkeypatch, degraded, [])
    results = vram_adapter.gpu_checks(CheckContext())
    summary = results[0]
    assert summary.severity is Severity.OK
    assert summary.data["free_mib"] is None
    assert "unreachable" in (summary.detail or "")


def test_snapshot_raising_is_swallowed(monkeypatch) -> None:
    def boom(base_url=None):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr("lib.llm.vram_doctor.snapshot_vram", boom)
    results = vram_adapter.gpu_checks(CheckContext())
    assert len(results) == 1
    assert results[0].name == "gpu_snapshot"
    assert results[0].severity is Severity.WARN
    assert "probe exploded" in results[0].summary


def test_fit_check_raising_is_swallowed(monkeypatch) -> None:
    def boom(snap):
        raise RuntimeError("fit exploded")

    monkeypatch.setattr("lib.llm.vram_doctor.snapshot_vram", lambda base_url=None: _snapshot())
    monkeypatch.setattr("lib.llm.vram_doctor.fit_check", boom)
    results = vram_adapter.gpu_checks(CheckContext())
    # snapshot summary still emitted, plus a fit-error warning
    assert results[0].name == "gpu_snapshot"
    err = next(r for r in results if r.name == "gpu_fit_error")
    assert err.severity is Severity.WARN
    assert "fit exploded" in err.summary

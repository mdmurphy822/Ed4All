from __future__ import annotations

import builtins

from ci.integrity_check import (
    check_hash_chains,
    check_path_security,
    check_sample_finalization,
    check_tool_registry,
)


def test_sample_finalization_uses_current_report_contract(tmp_path) -> None:
    run_path = tmp_path / "synthetic-run"
    run_path.mkdir()

    result = check_sample_finalization(tmp_path)

    assert result.passed is True
    assert result.details == {
        "test_run": "synthetic-run",
        "finalization_valid": True,
        "hash_chain_valid": True,
        "artifact_count": 0,
    }


def test_sample_finalization_verbose_uses_current_report_contract(tmp_path) -> None:
    (tmp_path / "synthetic-run").mkdir()

    result = check_sample_finalization(tmp_path, verbose=True)

    assert result.passed is True
    assert result.errors == []
    assert result.details["hash_chain_valid"] is True


def test_sample_finalization_exception_fails(tmp_path, monkeypatch) -> None:
    (tmp_path / "synthetic-run").mkdir()

    from lib import run_finalizer

    def raise_from_verify_only(self):
        raise RuntimeError("verification exploded")

    monkeypatch.setattr(run_finalizer.RunFinalizer, "verify_only", raise_from_verify_only)

    result = check_sample_finalization(tmp_path)

    assert result.passed is False
    assert result.errors == ["Finalization error: verification exploded"]


def test_path_security_passes_with_current_constants_contract() -> None:
    result = check_path_security()

    assert result.passed is True
    assert result.details["project_root_exists"] is True
    assert result.errors == []


def test_path_security_import_failure_fails(monkeypatch) -> None:
    real_import = builtins.__import__

    def reject_path_constants(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "lib.path_constants":
            raise ImportError("simulated missing constants")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_path_constants)

    result = check_path_security()

    assert result.passed is False
    assert result.errors == [
        "Path constants not available: simulated missing constants"
    ]


def test_empty_tool_registry_is_explicitly_vacuous(monkeypatch) -> None:
    from lib import tool_registry

    monkeypatch.setattr(tool_registry, "_global_registry", tool_registry.ToolRegistry())

    result = check_tool_registry()

    assert result.passed is True
    assert result.details["tool_count"] == 0
    assert result.message == "Registry valid but empty; no required tools configured"
    assert result.warnings == [result.message]


def test_runs_without_hash_chains_are_reported_as_zero_work(tmp_path) -> None:
    (tmp_path / "run-one").mkdir()
    (tmp_path / "run-two").mkdir()

    result = check_hash_chains(tmp_path)

    assert result.passed is True
    assert result.details == {
        "run_count": 2,
        "chain_count": 0,
        "verified_count": 0,
    }
    assert result.message == "No hash chains found across 2 runs"
    assert result.warnings == [result.message]

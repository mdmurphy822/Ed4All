from __future__ import annotations

import builtins
import json
import logging
from types import SimpleNamespace

from ci.integrity_check import (
    PROJECT_ROOT,
    _integrity_environment,
    check_hash_chains,
    check_libv2_taxonomy_resolution,
    check_path_security,
    check_sample_finalization,
    check_tool_registry,
)


def test_integrity_environment_does_not_disclose_checkout_root(tmp_path) -> None:
    environment = _integrity_environment(
        PROJECT_ROOT / "runtime" / "state" / "runs",
        PROJECT_ROOT / "schemas",
        tmp_path / "private-config",
    )

    serialized = json.dumps(environment)
    assert environment["runs_path"] == "runtime/state/runs"
    assert environment["schemas_path"] == "schemas"
    assert environment["config_path"] == "<external-path>"
    assert str(PROJECT_ROOT) not in serialized


def test_sample_finalization_uses_current_report_contract(tmp_path) -> None:
    run_path = tmp_path / "synthetic-run"
    run_path.mkdir()

    result = check_sample_finalization(tmp_path)

    assert result.passed is True
    assert result.details == {
        "test_run_selected": True,
        "finalization_valid": True,
        "hash_chain_valid": True,
        "artifact_count": 0,
    }


def test_sample_finalization_verbose_does_not_log_private_run_name(
    tmp_path, caplog
) -> None:
    private_name = "private-course-run-123"
    (tmp_path / private_name).mkdir()

    with caplog.at_level(logging.INFO, logger="ci.integrity_check"):
        result = check_sample_finalization(tmp_path, verbose=True)

    assert result.passed is True
    assert result.errors == []
    assert result.details["hash_chain_valid"] is True
    assert private_name not in caplog.text
    assert private_name not in json.dumps(result.to_dict(), sort_keys=True)


def test_sample_finalization_redacts_embedded_report_errors(
    tmp_path, monkeypatch, caplog
) -> None:
    private_name = "private-course-run-123"
    run_path = tmp_path / private_name
    run_path.mkdir()

    from lib import run_finalizer

    def failed_report(self):
        return SimpleNamespace(
            success=False,
            all_chains_valid=False,
            artifact_count=1,
            errors=[f"artifact under {run_path}/output for {private_name} failed"],
        )

    monkeypatch.setattr(run_finalizer.RunFinalizer, "verify_only", failed_report)

    with caplog.at_level(logging.INFO, logger="ci.integrity_check"):
        result = check_sample_finalization(tmp_path, verbose=True)

    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert result.passed is False
    assert "<private-run>" in serialized
    assert private_name not in serialized
    assert str(run_path) not in serialized
    assert private_name not in caplog.text


def test_sample_finalization_exception_fails(tmp_path, monkeypatch) -> None:
    private_name = "private-course-run-123"
    run_path = tmp_path / private_name
    run_path.mkdir()

    from lib import run_finalizer

    def raise_from_verify_only(self):
        raise RuntimeError(f"verification exploded for {private_name} at {run_path}")

    monkeypatch.setattr(run_finalizer.RunFinalizer, "verify_only", raise_from_verify_only)

    result = check_sample_finalization(tmp_path)

    assert result.passed is False
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert "<private-run>" in serialized
    assert private_name not in serialized
    assert str(run_path) not in serialized


def test_path_security_passes_with_current_constants_contract() -> None:
    result = check_path_security()

    assert result.passed is True
    assert result.details["project_root_exists"] is True
    assert result.errors == []
    assert str(PROJECT_ROOT.resolve()) not in json.dumps(
        result.to_dict(), sort_keys=True
    )


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


def test_hash_chain_skip_logs_and_result_do_not_expose_run_names(
    tmp_path, caplog
) -> None:
    private_names = ["private-course-run-123", "another-private-run-456"]
    for private_name in private_names:
        (tmp_path / private_name).mkdir()

    with caplog.at_level(logging.INFO, logger="ci.integrity_check"):
        result = check_hash_chains(tmp_path, verbose=True)

    serialized = json.dumps(result.to_dict(), sort_keys=True)
    for private_name in private_names:
        assert private_name not in caplog.text
        assert private_name not in serialized


def test_hash_chain_failure_does_not_expose_run_name(tmp_path, monkeypatch) -> None:
    private_name = "private-course-run-123"
    run_path = tmp_path / private_name
    run_path.mkdir()
    (run_path / "hash_chain.jsonl").write_text("{}\n", encoding="utf-8")

    from lib import replay_engine

    monkeypatch.setattr(
        replay_engine.ReplayEngine,
        "verify_run_integrity",
        lambda self, run_id: {"checks": {"hash_chain_valid": False}},
    )

    result = check_hash_chains(tmp_path)
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert result.passed is False
    assert private_name not in serialized
    assert str(run_path) not in serialized
    assert result.errors == ["run #1: Hash chain integrity failed"]


def test_hash_chain_exception_redacts_run_name_and_path(tmp_path, monkeypatch) -> None:
    private_name = "private-course-run-123"
    run_path = tmp_path / private_name
    run_path.mkdir()
    (run_path / "hash_chain.jsonl").write_text("{}\n", encoding="utf-8")

    from lib import replay_engine

    def raise_private_error(self, run_id):
        raise RuntimeError(f"failure inside {private_name} at {run_path}")

    monkeypatch.setattr(
        replay_engine.ReplayEngine, "verify_run_integrity", raise_private_error
    )

    result = check_hash_chains(tmp_path)
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert result.passed is False
    assert "<private-run>" in serialized
    assert private_name not in serialized
    assert str(run_path) not in serialized


def test_libv2_taxonomy_reports_repository_relative_path() -> None:
    result = check_libv2_taxonomy_resolution()

    assert result.passed is True
    assert result.details["canonical_path"] == "schemas/taxonomies/bloom_verbs.json"
    assert str(PROJECT_ROOT.resolve()) not in json.dumps(
        result.to_dict(), sort_keys=True
    )


def test_libv2_taxonomy_redacts_external_path(tmp_path, monkeypatch) -> None:
    from LibV2.tools.libv2 import _bloom_verbs

    external_path = tmp_path / "private-taxonomies" / "bloom_verbs.json"
    monkeypatch.setattr(_bloom_verbs, "_CANONICAL_PATH", external_path)
    monkeypatch.setattr(_bloom_verbs, "get_verbs_list", lambda: ["apply"])

    result = check_libv2_taxonomy_resolution()
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert result.passed is False
    assert result.details["canonical_path"] == "<external-path>"
    assert str(tmp_path) not in serialized

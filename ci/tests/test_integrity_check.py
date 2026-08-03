from __future__ import annotations

from ci.integrity_check import check_sample_finalization


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

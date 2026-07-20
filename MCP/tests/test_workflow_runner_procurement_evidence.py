"""EVAL-C stitch — ``WorkflowRunner._maybe_write_procurement_evidence`` wiring.

The post-loop advisory procurement-evidence exporter is best-effort: it writes
the bundle ONLY when a LibV2 course dir is resolvable, keys it to the
promotion-chain report, and NEVER alters ``final_status`` (any failure logs a
warning and returns ``None``). The exporter itself
(``lib.governance.procurement_evidence.write_evidence_bundle``) is covered by
``lib/governance/tests/test_procurement_evidence.py``; this test guards the
runner-side wiring only.

Fully offline / deterministic — no network, no model, no course slugs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.core.workflow_runner import WorkflowRunner  # noqa: E402


def _runner() -> WorkflowRunner:
    # The helper only reads its explicit kwargs — executor/config are unused.
    return WorkflowRunner(executor=object(), config=object())


def test_no_libv2_course_dir_returns_none(tmp_path, monkeypatch):
    """No ``libv2_archival.course_dir`` in phase_outputs → no bundle, None."""
    called = {"n": 0}

    def _fake_write(*_args, **_kwargs):
        called["n"] += 1
        return tmp_path / "should_not_write.json"

    import lib.governance.procurement_evidence as pe
    monkeypatch.setattr(pe, "write_evidence_bundle", _fake_write)

    out = _runner()._maybe_write_procurement_evidence(
        workflow_id="WF-TEST",
        workflow_params={"course_name": "UNIT_TEST_101"},
        phase_outputs={},  # no libv2_archival
        promotion_chain_path=None,
    )
    assert out is None
    assert called["n"] == 0


def test_wires_course_dir_and_promotion_chain(tmp_path, monkeypatch):
    """With a resolvable course_dir the helper forwards course_dir / course_slug /
    run_id / promotion_chain_path into write_evidence_bundle and returns its
    path."""
    course_dir = tmp_path / "phys-101"
    course_dir.mkdir()
    chain_path = tmp_path / "courseforge_promotion_chain_report.json"
    bundle_path = course_dir / "retrieval_eval" / "procurement_evidence_bundle.json"

    seen = {}

    def _fake_write(course, *, course_code, course_slug, run_id,
                    promotion_chain_path):
        seen.update({
            "course": Path(course),
            "course_code": course_code,
            "course_slug": course_slug,
            "run_id": run_id,
            "promotion_chain_path": promotion_chain_path,
        })
        return bundle_path

    import lib.governance.procurement_evidence as pe
    monkeypatch.setattr(pe, "write_evidence_bundle", _fake_write)

    out = _runner()._maybe_write_procurement_evidence(
        workflow_id="WF-XYZ",
        workflow_params={"course_name": "PHYS_101"},
        phase_outputs={"libv2_archival": {"course_dir": str(course_dir)}},
        promotion_chain_path=chain_path,
    )

    assert out == bundle_path
    assert seen["course"] == course_dir
    assert seen["course_code"] == "PHYS_101"
    assert seen["course_slug"] == "phys-101"
    assert seen["run_id"] == "WF-XYZ"
    assert seen["promotion_chain_path"] == chain_path


def test_exporter_failure_is_swallowed(tmp_path, monkeypatch):
    """A write_evidence_bundle exception is best-effort — the helper returns None
    (never propagates, never alters final_status)."""
    course_dir = tmp_path / "chem-201"
    course_dir.mkdir()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("disk full")

    import lib.governance.procurement_evidence as pe
    monkeypatch.setattr(pe, "write_evidence_bundle", _boom)

    out = _runner()._maybe_write_procurement_evidence(
        workflow_id="WF-BOOM",
        workflow_params={},
        phase_outputs={"libv2_archival": {"course_dir": str(course_dir)}},
        promotion_chain_path=None,
    )
    assert out is None

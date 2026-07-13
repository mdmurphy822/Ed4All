"""Regression net for the task-#43 ``unit_coverage`` + ``page_arranger``
end-to-end plumbing (cascade result → bridge/in-process seam → operator
artifacts).

Both keys are emitted at the cascade result-dict top level
(``result["unit_coverage"]`` / ``result["page_arranger"]``) but were DROPPED
before the operator artifacts: the cross-venv bridge serializer
(``run_cascade_json._build_bridge_dict``) whitelisted keys and never forwarded
them, and the ``MCP/tools/pipeline_tools.py`` emit path wrote neither into the
``{stem}_accessible.cascade_ir.json`` nor surfaced the coverage gate into
``{stem}.quality.json``. These tests pin the fixed thread:

  1. the bridge serializer forwards both keys (JSON round-trip);
  2. ``_SemantikBridgeResult`` surfaces them as attributes;
  3. the pipeline_tools resolvers pull them off BOTH seam arms (bridge attr /
     in-process ``.cascade`` dict) and return ``None`` when absent;
  4. ``_apply_unit_coverage_to_quality`` folds a below-floor page into the
     quality sidecar as a first-class critical issue + flag;
  5. a FULL ``_run_semantik_v2_conversion`` run (cascade stubbed — no models)
     writes BOTH keys into the emitted cascade_ir.json AND the coverage
     verdict into the emitted quality.json.

Built + run with NO models / GPU (the cascade is stubbed via ``sys.modules``).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Fixtures — audit dicts + a duck-typed cascade result.
# ---------------------------------------------------------------------------


def _unit_coverage_report(*, passed: bool) -> dict:
    """A SEMANTIK_UNIT_COVERAGE_GATE report DICT (the ``run_unit_coverage_gate``
    shape). ``passed=False`` drops page 2 below the hard floor and page 3 below
    the warn floor."""
    if passed:
        pages = [
            {"page": 1, "coverage": 0.99, "n_tokens": 100, "n_missing": 1,
             "below_min": False, "below_floor": False, "missing_spans": []},
        ]
        return {
            "gate": "unit_coverage", "enabled": True, "min_coverage": 0.90,
            "page_floor": 0.70, "n_pages": 1, "document_passed": True,
            "warned_pages": [], "failed_pages": [], "pages": pages,
        }
    pages = [
        {"page": 1, "coverage": 0.99, "n_tokens": 100, "n_missing": 1,
         "below_min": False, "below_floor": False, "missing_spans": []},
        {"page": 2, "coverage": 0.55, "n_tokens": 200, "n_missing": 90,
         "below_min": True, "below_floor": True,
         "missing_spans": ["EXAMPLE 3.2 Solve the worked example"]},
        {"page": 3, "coverage": 0.82, "n_tokens": 150, "n_missing": 27,
         "below_min": True, "below_floor": False,
         "missing_spans": ["a short warned span"]},
    ]
    return {
        "gate": "unit_coverage", "enabled": True, "min_coverage": 0.90,
        "page_floor": 0.70, "n_pages": 3, "document_passed": False,
        "warned_pages": [3], "failed_pages": [2], "pages": pages,
    }


def _page_arranger_audit() -> dict:
    """A SEMANTIK_PAGE_ARRANGER audit DICT (scan-lane structure-authority)."""
    return {
        "schema": "page_arranger/1", "model": "qwen2.5vl:7b",
        "pages": 3, "valid_pages": 3, "failed_pages": 0,
        "coercions": 1, "repairs": 0, "attempts": {"1": 3},
    }


def _region_provenance() -> list[dict]:
    return [
        {"region_index": 0, "region_kind": "heading", "role": "heading",
         "confidence": 0.95, "wcag_status": "passed", "first_raw_block_index": 0,
         "pages": [1], "heading_text": "Chapter 1: Foundations", "level": 1,
         "figure_alt": None, "raw_text": "Chapter 1: Foundations"},
        {"region_index": 1, "region_kind": "paragraph", "role": "body",
         "confidence": 0.7, "wcag_status": "passed", "first_raw_block_index": 1,
         "pages": [1], "heading_text": None, "level": None, "figure_alt": None,
         "raw_text": "The order of operations is PEMDAS."},
    ]


class _FakeCascadeResult:
    """Duck-typed in-process ``PipelineV2Result`` carrying BOTH task-#43 keys
    at the ``.cascade`` result-dict top level (the real in-process shape)."""

    def __init__(self, *, coverage_passes: bool):
        self.pdf = "sample.pdf"
        self.html = "<main></main>"
        self.wcag_status = "passed"
        self.exit_action = "ship_with_confidence"
        self.theta_score = 0.91
        self.flags: list[str] = []
        self.lane_used = "fast-lane"
        self.lang = "en"
        self.region_provenance = _region_provenance()
        self.heading_tree = [(1, "Chapter 1: Foundations")]
        self.cascade = {
            "runtime_mode": "real",
            "unit_coverage": _unit_coverage_report(passed=coverage_passes),
            "page_arranger": _page_arranger_audit(),
        }


def _import_bridge_builder():
    scripts_dir = _REPO_ROOT / "SemantiK" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import run_cascade_json  # type: ignore[import-not-found]

    return run_cascade_json


# ---------------------------------------------------------------------------
# 1. Bridge serializer forwards BOTH keys + _SemantikBridgeResult surfaces them.
# ---------------------------------------------------------------------------


def test_bridge_forwards_unit_coverage_and_page_arranger():
    rcj = _import_bridge_builder()
    result = _FakeCascadeResult(coverage_passes=False)

    bridge = rcj._build_bridge_dict(result, pdf="x.pdf")
    # JSON-serializable (the bridge is written to disk + read back).
    bridge = json.loads(json.dumps(bridge))

    assert bridge["unit_coverage"] is not None
    assert bridge["unit_coverage"]["failed_pages"] == [2]
    assert bridge["page_arranger"] is not None
    assert bridge["page_arranger"]["valid_pages"] == 3

    from MCP.tools.pipeline_tools import _SemantikBridgeResult

    br = _SemantikBridgeResult(bridge)
    assert br.unit_coverage["document_passed"] is False
    assert br.page_arranger["pages"] == 3


def test_bridge_absent_keys_forward_null():
    rcj = _import_bridge_builder()

    class _Bare:
        cascade = {"runtime_mode": "real"}

    bridge = json.loads(json.dumps(rcj._build_bridge_dict(_Bare(), pdf="x.pdf")))
    assert bridge["unit_coverage"] is None
    assert bridge["page_arranger"] is None

    from MCP.tools.pipeline_tools import _SemantikBridgeResult

    br = _SemantikBridgeResult(bridge)
    assert br.unit_coverage is None
    assert br.page_arranger is None


# ---------------------------------------------------------------------------
# 2. Resolvers read BOTH seam arms (bridge attribute / in-process cascade dict).
# ---------------------------------------------------------------------------


def test_resolvers_read_in_process_cascade_arm():
    from MCP.tools.pipeline_tools import (
        _semantik_resolve_page_arranger,
        _semantik_resolve_unit_coverage,
    )

    result = _FakeCascadeResult(coverage_passes=True)  # no bridge attribute
    uc = _semantik_resolve_unit_coverage(result)
    pa = _semantik_resolve_page_arranger(result)
    assert uc is not None and uc["document_passed"] is True
    assert pa is not None and pa["model"] == "qwen2.5vl:7b"


def test_resolvers_read_bridge_attribute_arm():
    from MCP.tools.pipeline_tools import (
        _SemantikBridgeResult,
        _semantik_resolve_page_arranger,
        _semantik_resolve_unit_coverage,
    )

    rcj = _import_bridge_builder()
    bridge = json.loads(
        json.dumps(
            rcj._build_bridge_dict(_FakeCascadeResult(coverage_passes=False), pdf="x")
        )
    )
    br = _SemantikBridgeResult(bridge)
    assert _semantik_resolve_unit_coverage(br)["failed_pages"] == [2]
    assert _semantik_resolve_page_arranger(br)["valid_pages"] == 3


def test_resolvers_return_none_when_absent():
    from MCP.tools.pipeline_tools import (
        _semantik_resolve_page_arranger,
        _semantik_resolve_unit_coverage,
    )

    class _Bare:
        cascade = {"runtime_mode": "real"}

    assert _semantik_resolve_unit_coverage(_Bare()) is None
    assert _semantik_resolve_page_arranger(_Bare()) is None


# ---------------------------------------------------------------------------
# 3. Quality-sidecar fold — a below-floor page is a first-class quality issue.
# ---------------------------------------------------------------------------


def test_apply_unit_coverage_folds_failed_and_warned_into_quality():
    from lib.semantik.adapter import build_quality_sidecar
    from MCP.tools.pipeline_tools import _apply_unit_coverage_to_quality

    quality = build_quality_sidecar(
        "<main></main>", title="T", slug="t", wcag_status="passed"
    )
    assert quality["compliant"] is True  # healthy WCAG baseline

    _apply_unit_coverage_to_quality(quality, _unit_coverage_report(passed=False))

    # Summary block with per-page coverage scores.
    assert quality["unit_coverage"]["document_passed"] is False
    assert quality["unit_coverage"]["page_coverage"][2] == 0.55
    # The below-hard-floor page flips compliance + adds a critical issue + flag.
    assert "UNIT_COVERAGE_FAILED" in quality["flags"]
    assert quality["compliant"] is False
    assert quality["quality_score"] == 0.0
    codes = {i["code"] for i in quality["issues"]}
    assert "UNIT_COVERAGE_BELOW_FLOOR" in codes  # page 2
    assert "UNIT_COVERAGE_BELOW_MIN" in codes  # page 3
    floor_issue = next(
        i for i in quality["issues"] if i["code"] == "UNIT_COVERAGE_BELOW_FLOOR"
    )
    assert floor_issue["page"] == 2 and floor_issue["severity"] == "critical"
    assert quality["total_issues"] == len(quality["issues"])


def test_apply_unit_coverage_healthy_report_only_adds_summary():
    from lib.semantik.adapter import build_quality_sidecar
    from MCP.tools.pipeline_tools import _apply_unit_coverage_to_quality

    quality = build_quality_sidecar(
        "<main></main>", title="T", slug="t", wcag_status="passed"
    )
    _apply_unit_coverage_to_quality(quality, _unit_coverage_report(passed=True))
    assert quality["unit_coverage"]["document_passed"] is True
    assert quality["compliant"] is True  # unchanged
    assert "UNIT_COVERAGE_FAILED" not in quality["flags"]
    assert quality["issues"] == []


# ---------------------------------------------------------------------------
# 4. FULL emit path — the cascade is stubbed (no models); assert the WRITTEN
#    cascade_ir.json + quality.json carry the two keys / the coverage verdict.
# ---------------------------------------------------------------------------


@pytest.fixture
def _stub_cascade_modules(monkeypatch):
    """Replace the three ``SemantiK.semantik_structure`` submodules the seam
    imports at call time with lightweight stubs (no torch / GGUF / Chromium).

    ``run_pipeline_v2`` returns the injected fake result; the whole downstream
    emit path (build_chapters_ir → normalize_cascade_to_ed4all → sidecar/IR
    writes) is the REAL pure-Python code under test.
    """
    holder: dict = {}

    def _install(result):
        holder["result"] = result

        cascade_mod = types.ModuleType("SemantiK.semantik_structure.cascade")
        cascade_mod.run_pipeline_v2 = lambda *a, **k: holder["result"]  # type: ignore[attr-defined]

        v2cfg_mod = types.ModuleType("SemantiK.semantik_structure.v2_config")
        v2cfg_mod.resolve_local_v2_config = lambda: None  # type: ignore[attr-defined]

        stop_mod = types.ModuleType("SemantiK.semantik_structure.stop_seam")

        class _CascadeStopRequested(RuntimeError):
            pass

        stop_mod.CascadeStopRequested = _CascadeStopRequested  # type: ignore[attr-defined]

        monkeypatch.setitem(
            sys.modules, "SemantiK.semantik_structure.cascade", cascade_mod
        )
        monkeypatch.setitem(
            sys.modules, "SemantiK.semantik_structure.v2_config", v2cfg_mod
        )
        monkeypatch.setitem(
            sys.modules, "SemantiK.semantik_structure.stop_seam", stop_mod
        )

    return _install


def test_full_conversion_writes_ir_and_quality_with_both_keys(
    tmp_path, _stub_cascade_modules
):
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    _stub_cascade_modules(_FakeCascadeResult(coverage_passes=False))

    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")  # never parsed (cascade is stubbed)
    out_html = tmp_path / "sample_accessible.html"

    res = _run_semantik_v2_conversion(str(pdf), str(out_html))
    assert res.get("success") is not None  # ran to the contract return

    # (a) The emitted cascade_ir.json carries BOTH keys at top level.
    ir_path = tmp_path / "sample_accessible.cascade_ir.json"
    assert ir_path.exists(), "cascade_ir.json was not written"
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    assert ir["unit_coverage"]["failed_pages"] == [2]
    assert ir["page_arranger"]["valid_pages"] == 3

    # (b) The emitted quality.json carries the coverage verdict as a quality
    #     issue + flag (a below-floor page is first-class).
    quality_path = tmp_path / "sample_accessible.quality.json"
    assert quality_path.exists(), "quality.json was not written"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert "UNIT_COVERAGE_FAILED" in quality["flags"]
    assert quality["unit_coverage"]["document_passed"] is False
    codes = {i["code"] for i in quality["issues"]}
    assert "UNIT_COVERAGE_BELOW_FLOOR" in codes


def test_full_conversion_healthy_coverage_no_failure_flag(
    tmp_path, _stub_cascade_modules
):
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    _stub_cascade_modules(_FakeCascadeResult(coverage_passes=True))

    pdf = tmp_path / "clean.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    out_html = tmp_path / "clean_accessible.html"
    _run_semantik_v2_conversion(str(pdf), str(out_html))

    ir = json.loads(
        (tmp_path / "clean_accessible.cascade_ir.json").read_text(encoding="utf-8")
    )
    assert ir["unit_coverage"]["document_passed"] is True
    assert ir["page_arranger"]["valid_pages"] == 3
    quality = json.loads(
        (tmp_path / "clean_accessible.quality.json").read_text(encoding="utf-8")
    )
    assert "UNIT_COVERAGE_FAILED" not in quality["flags"]

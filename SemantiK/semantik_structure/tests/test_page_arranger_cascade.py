"""Cascade-level integration — the SEMANTIK_PAGE_ARRANGER seam wired into
``run_full_cascade`` (task #43).

Mirrors ``test_reasoning_qc_cascade`` harness: drives the REAL
``run_full_cascade`` to completion in ``mock`` runtime with the heavy upstream
mocked, so Stages 6-13 (mock authoring, real assembler/gates/theta) run for real.

Pinned here:
  * flag-ON + OCR-lane route → the arranger OWNS structure: ``run_council`` is
    monkeypatched to RAISE if called (council heads never load on the route);
    ``result['page_arranger']`` audit arm present; every ``region_provenance``
    row carries ``typing_authority='vlm-arranger'`` (payload-first).
  * flag-OFF → byte-identical through the seam: ``run_council`` runs, no
    ``page_arranger`` key, no ``typing_authority`` key.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def _install_axe_stubs() -> None:
    if "axe_playwright_python" not in sys.modules:
        try:
            import axe_playwright_python  # noqa: F401
        except Exception:  # noqa: BLE001
            pkg = types.ModuleType("axe_playwright_python")
            sub = types.ModuleType("axe_playwright_python.sync_playwright")
            sub.Axe = object
            pkg.sync_playwright = sub
            sys.modules["axe_playwright_python"] = pkg
            sys.modules["axe_playwright_python.sync_playwright"] = sub
    if "playwright" not in sys.modules:
        try:
            import playwright  # noqa: F401
        except Exception:  # noqa: BLE001
            pw = types.ModuleType("playwright")
            pw_sync = types.ModuleType("playwright.sync_api")
            pw_sync.sync_playwright = lambda *a, **k: None
            pw.sync_api = pw_sync
            sys.modules["playwright"] = pw
            sys.modules["playwright.sync_api"] = pw_sync


_install_axe_stubs()

from semantik_structure import cascade  # noqa: E402
from semantik_structure import page_arranger  # noqa: E402
from semantik_structure.council import base as council_base  # noqa: E402
from semantik_structure.council.types import CouncilState  # noqa: E402
from semantik_structure.structure_graph import Region  # noqa: E402
from semantik_structure.types import FeatureBlock, RawBlock  # noqa: E402


def _fb(text: str, page: int = 1) -> FeatureBlock:
    return FeatureBlock(
        raw=RawBlock(text=text, page=page, bbox=(0.0, 0.0, 10.0, 10.0),
                     page_width=100.0, page_height=100.0, source="tesseract"),
        size_bucket="md", gap_above=None, is_top_of_page=False, is_centered=False,
        caps=None, indent_bucket=0, relative_font_ratio=1.0, provenance="tesseract",
    )


def _fbs():
    return [
        _fb("Chapter 1 Whole Numbers"),
        _fb("To add whole numbers, line up the columns and add."),
    ]


def _arranger_regions():
    return [
        Region(
            kind="heading",
            feature_block_indices=(0,),
            payload={"text": "Chapter 1 Whole Numbers", "level_hint": 1,
                     "typing_authority": "vlm-arranger"},
            provenance={"pass": "page_arranger"},
        ),
        Region(
            kind="paragraph",
            feature_block_indices=(1,),
            payload={"text": "To add whole numbers, line up the columns and add.",
                     "typing_authority": "vlm-arranger"},
            provenance={"pass": "page_arranger"},
        ),
    ]


class _FakeRoute:
    def __init__(self, fbs, regions):
        self._fbs = fbs
        self._regions = regions

    def council_substitute(self):
        return CouncilState(outputs={}), [], self._fbs

    def arrange_regions(self, feature_blocks, *, log=None, **_k):
        return list(self._regions), {"pass": "page_arranger", "pages": 1,
                                     "pages_valid": 1, "pages_failed": 0}


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path / "cache_root"))
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "0")
    monkeypatch.setenv("SEMANTIK_STRUCTURE_CLEAN", "0")
    monkeypatch.delenv("SEMANTIK_REASONING_QC", raising=False)
    monkeypatch.delenv("SEMANTIK_STRUCTURE_REVIEW", raising=False)
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.delenv("SEMANTIK_DETECT_FIGURES", raising=False)
    monkeypatch.delenv("SEMANTIK_SECOND_PASS", raising=False)
    monkeypatch.setattr(council_base, "release_council_gpu", lambda: None)


def _run(tmp_path):
    return cascade.run_full_cascade(
        tmp_path / "doc.pdf", validator=types.SimpleNamespace(),
        runtime_mode="mock", return_html=True,
    )


def test_flag_on_route_bypasses_council_and_stamps_authority(monkeypatch, tmp_path):
    fbs = _fbs()
    route = _FakeRoute(fbs, _arranger_regions())
    monkeypatch.setattr(page_arranger, "resolve_page_arranger_route", lambda _pdf: route)

    def _boom(_pdf):
        raise AssertionError("run_council MUST NOT run on the arranger route")

    monkeypatch.setattr(cascade, "run_council", _boom)

    result = _run(tmp_path)
    assert "page_arranger" in result
    assert result["page_arranger"]["pages_valid"] == 1
    prov = result["region_provenance"]
    assert prov, "region_provenance must be non-empty"
    assert all(e.get("typing_authority") == "vlm-arranger" for e in prov)
    # never the collapsed-BERT authority
    assert all(e.get("typing_authority") != "bert" for e in prov)


def test_flag_off_byte_identical_through_seam(monkeypatch, tmp_path):
    fbs = _fbs()
    regions = _arranger_regions()
    # route returns None (flag off / born-digital) → council path owns structure.
    monkeypatch.setattr(page_arranger, "resolve_page_arranger_route", lambda _pdf: None)
    called = {"council": 0}

    def _council(_pdf):
        called["council"] += 1
        return CouncilState(outputs={}), [], fbs

    monkeypatch.setattr(cascade, "run_council", _council)
    monkeypatch.setattr(cascade, "arbitrate", lambda _s, _r: object())
    # strip the arranger-only payload key so the council path's provenance is clean
    plain = [
        Region(kind=r.kind, feature_block_indices=r.feature_block_indices,
               payload={k: v for k, v in r.payload.items() if k != "typing_authority"})
        for r in regions
    ]
    monkeypatch.setattr(cascade, "build_structure_graph", lambda *a, **k: list(plain))

    result = _run(tmp_path)
    assert called["council"] == 1  # council ran on the off path
    assert "page_arranger" not in result
    prov = result["region_provenance"]
    assert all("typing_authority" not in e for e in prov)  # byte-stable: key absent

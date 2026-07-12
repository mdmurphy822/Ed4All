"""Cascade-level integration — the Stage-9b reasoning-QC seam wired into
``run_full_cascade`` (SEMANTIK_REASONING_QC).

Drives the REAL ``run_full_cascade`` to completion in ``mock`` runtime with the
heavy upstream (council / arbitrate / structure-graph) mocked out — the exact
harness ``test_cascade_stop_seam`` uses — so Stages 6-13 (mock authoring, real
assembler, real gates, real theta) run for real and the result dict is genuine.

Pinned here:
  * SHADOW mode with the VLM judgment MOCKED (no real inference — the
    ``_run_qc_judgment`` / seat / capture seams are monkeypatched): the cascade
    result carries ``result['reasoning_qc']`` (ran, mode=shadow), AND the
    emitted HTML + ``region_provenance`` are BYTE-IDENTICAL to a flag-OFF
    control run (shadow applies nothing → ``capped`` rides through unchanged).
  * flag-OFF does NOT import / resolve the QC SEAT (``_resolve_qc_seat`` is
    never called; no ``reasoning_qc`` result key).

GPU lifecycle is forced OFF so the seam never touches the live ollama/GPU; the
reasoning-QC re-type channel builds no endpoint seat in shadow (nothing is
applied), so no network is ever reached.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Import scaffold — stub axe_playwright_python + playwright for IMPORT ONLY if
# the slim venv lacks them (mirrors test_cascade_stage5d / _stop_seam). The
# assembler/gate path uses the REAL modules when present.
# ---------------------------------------------------------------------------
def _install_axe_stubs() -> None:
    if "axe_playwright_python" not in sys.modules:
        try:
            import axe_playwright_python  # noqa: F401
        except Exception:  # noqa: BLE001 — absent → stub for import only
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
from semantik_structure import reasoning_qc  # noqa: E402
from semantik_structure.council import base as council_base  # noqa: E402
from semantik_structure.council.types import CouncilState  # noqa: E402
from semantik_structure.structure_graph import Region  # noqa: E402
from semantik_structure.types import FeatureBlock, RawBlock  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic document (2 FBs on one physical page → one QC window).
# ---------------------------------------------------------------------------
def _fb(text: str, page: int = 1) -> FeatureBlock:
    return FeatureBlock(
        raw=RawBlock(
            text=text,
            page=page,
            bbox=(0.0, 0.0, 10.0, 10.0),
            page_width=100.0,
            page_height=100.0,
        ),
        size_bucket="md",
        gap_above=None,
        is_top_of_page=False,
        is_centered=False,
        caps=None,
        indent_bucket=0,
        relative_font_ratio=1.0,
    )


def _regions_and_fbs():
    fbs = [
        _fb("Chapter 1 Whole Numbers", page=1),
        _fb("To add whole numbers, line up the columns and add.", page=1),
    ]
    regions = [
        Region(
            kind="heading",
            feature_block_indices=(0,),
            payload={"text": "Chapter 1 Whole Numbers", "level_hint": 1},
        ),
        Region(
            kind="paragraph",
            feature_block_indices=(1,),
            payload={"text": "To add whole numbers, line up the columns and add."},
        ),
    ]
    return regions, fbs


def _prime_cascade(monkeypatch):
    """Mock Stages 1-5 (council/arbitrate/structure-graph) + council release,
    force GPU lifecycle OFF, disable the deterministic clean pass + optional
    upstream passes so the two runs differ ONLY by SEMANTIK_REASONING_QC. Stages
    6-13 (mock authoring, real assembler/gate/theta) then run for real."""
    monkeypatch.setenv("SEMANTIK_STRUCTURE_CLEAN", "0")
    monkeypatch.delenv("SEMANTIK_STRUCTURE_REVIEW", raising=False)
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.delenv("SEMANTIK_DETECT_FIGURES", raising=False)
    monkeypatch.delenv("SEMANTIK_SECOND_PASS", raising=False)
    monkeypatch.delenv("SEMANTIK_OCR_CONFUSABLE_REPAIR", raising=False)
    monkeypatch.setenv("ED4ALL_GPU_LIFECYCLE", "0")

    regions, fbs = _regions_and_fbs()
    monkeypatch.setattr(
        cascade, "run_council", lambda _pdf: (CouncilState(outputs={}), [], fbs)
    )
    monkeypatch.setattr(cascade, "arbitrate", lambda _s, _r: object())
    monkeypatch.setattr(
        cascade, "build_structure_graph", lambda *a, **k: list(regions)
    )
    monkeypatch.setattr(council_base, "release_council_gpu", lambda: None)


def _mock_qc_vlm(monkeypatch):
    """Monkeypatch the reasoning-QC VLM/seat/capture seams so NO real inference
    (torch/ollama/weights) is ever touched. Returns a ``seat_calls`` list that
    records every ``_resolve_qc_seat`` resolution."""
    seat_calls: list[bool] = []

    def _seat():
        seat_calls.append(True)
        return object()

    monkeypatch.setattr(reasoning_qc, "_resolve_qc_seat", _seat)
    monkeypatch.setattr(reasoning_qc, "_unload_seat", lambda seat: None)
    monkeypatch.setattr(reasoning_qc, "_build_reasoning_qc_capture", lambda: None)
    monkeypatch.setattr(
        reasoning_qc,
        "resolve_reasoning_qc_model",
        lambda default_model=None: "qc:test",
    )
    # Mocked VLM judgment — a confident verdict with NO reconcile ops (shadow
    # applies nothing regardless, so byte-identity holds; this exercises the
    # per-window judgment + audit path without real inference).
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf_path, page_num, block_texts: {"confidence": 0.9},
    )
    return seat_calls


def _run(monkeypatch, tmp_path):
    return cascade.run_full_cascade(
        tmp_path / "doc.pdf",
        validator=types.SimpleNamespace(),
        runtime_mode="mock",
        return_html=True,
    )


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
def test_shadow_reasoning_qc_present_and_byte_identical(monkeypatch, tmp_path):
    # Control: flag OFF → no reasoning_qc key, seat NEVER resolved.
    _prime_cascade(monkeypatch)
    seat_calls = _mock_qc_vlm(monkeypatch)
    monkeypatch.delenv("SEMANTIK_REASONING_QC", raising=False)
    off = _run(monkeypatch, tmp_path)

    assert "reasoning_qc" not in off, "flag-OFF must not add the reasoning_qc arm"
    assert seat_calls == [], "flag-OFF must not import/resolve the QC seat"

    off_html = off["html"]
    off_prov = off["region_provenance"]

    # SHADOW: same harness, flip the flag on, VLM judgment mocked.
    _prime_cascade(monkeypatch)
    seat_calls = _mock_qc_vlm(monkeypatch)
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    shadow = _run(monkeypatch, tmp_path)

    # (a) result['reasoning_qc'] appears and reflects a real shadow run.
    assert "reasoning_qc" in shadow
    qc = shadow["reasoning_qc"]
    assert qc["ran"] is True
    assert qc["mode"] == "shadow"
    assert qc["model"] == "qc:test"
    assert qc["retype"]["applied"] == 0
    assert qc["merge_move"]["applied"] == 0
    # The QC seat WAS resolved this time (the pass actually ran).
    assert seat_calls == [True]

    # (b) HTML + region_provenance byte-identical to the flag-OFF control
    # (shadow verifies + audits but applies NOTHING → capped rides through).
    assert shadow["html"] == off_html
    assert shadow["region_provenance"] == off_prov


def test_off_mode_does_not_resolve_qc_seat(monkeypatch, tmp_path):
    """Explicit garbage / falsey SEMANTIK_REASONING_QC resolves off — no seat,
    no reasoning_qc key (byte-stable to unset)."""
    _prime_cascade(monkeypatch)
    seat_calls = _mock_qc_vlm(monkeypatch)
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "0")
    res = _run(monkeypatch, tmp_path)
    assert "reasoning_qc" not in res
    assert seat_calls == []


def test_seam_precedes_ocr_repair_and_provenance_in_source():
    """Source-order guard: the reasoning-QC seam must run BEFORE the OCR-repair
    pass (region-index-keyed) AND before _build_region_provenance."""
    import inspect

    src = inspect.getsource(cascade.run_full_cascade)
    seam = src.index("running Stage 9b reasoning-QC pass")
    ocr = src.index("running OCR-confusable micro-repair pass")
    prov = src.index("region_provenance = _build_region_provenance(")
    assert seam < ocr, "reasoning-QC seam must precede the OCR-repair pass"
    assert seam < prov, "reasoning-QC seam must precede _build_region_provenance"

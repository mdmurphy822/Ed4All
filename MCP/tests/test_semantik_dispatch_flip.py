"""SemantiK dispatch tests — the LIVE PDF→HTML conversion runs SemantiK.

Run with NO models / GPU. We MOCK ``_run_semantik_v2_conversion`` at the
``MCP.tools.pipeline_tools`` module level so the dispatch wiring is exercised
without touching SemantiK's heavy runtime deps.

Covers (migration plan §5 step 1/2, §3.7 fail-closed; updated for the DART
retirement that deleted the legacy ``_raw_text_to_accessible_html`` path):
  * The ``dart_conversion`` phase (textbook_to_course / course_generation
    PDF-ingest path) CALLS the SemantiK seam.
  * A seam ``success=False`` propagates as a conversion failure — NO silent
    fall-through to any legacy DART path (that path was retired).
  * The seam params (figures_dir / canonical_course_code / reuse_conversion)
    are threaded from the task kwargs into the seam.
  * Scope: any non-``dart_conversion`` caller does NOT reach the SemantiK
    seam — the dispatch is keyed strictly on ``phase == "dart_conversion"``;
    every other phase fails CLOSED with ``method == "unsupported_phase"``
    (the legacy DART converter is gone, so there is nothing to fall back to).

Run:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
    python -m pytest MCP/tests/test_semantik_dispatch_flip.py -q
"""

from __future__ import annotations

import json

import pytest


def _registry_tool():
    from MCP.tools.pipeline_tools import _build_tool_registry

    registry = _build_tool_registry()
    return registry["extract_and_convert_pdf"]


# ---------------------------------------------------------------------------
# (a) dart_conversion phase → calls the SemantiK seam.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dart_conversion_phase_calls_semantik_seam(monkeypatch, tmp_path):
    import MCP.tools.pipeline_tools as pt

    seam_calls: list = []

    def _fake_seam(pdf_path, output_path, **kwargs):
        seam_calls.append((pdf_path, output_path, kwargs))
        return {
            "success": True,
            "output_path": output_path,
            "output_paths": [output_path],
            "html_path": output_path,
            "html_paths": [output_path],
            "html_length": 42,
            "method": "semantik_v2",
            "certification_status": "certified",
        }

    monkeypatch.setattr(pt, "_run_semantik_v2_conversion", _fake_seam)

    tool = _registry_tool()
    out = await tool(
        pdf_path="sample_text.pdf",
        course_code="ALG_9",
        phase="dart_conversion",
        canonical_course_code="ALG_9",
        figures_dir=str(tmp_path / "figs"),
        reuse_conversion=True,
        output_dir=str(tmp_path),
    )
    payload = json.loads(out)

    # The SemantiK seam was called.
    assert len(seam_calls) == 1, "SemantiK seam not invoked on dart_conversion"
    assert payload["success"] is True
    assert payload["method"] == "semantik_v2"

    # Seam params threaded through.
    _pdf, _outpath, kw = seam_calls[0]
    assert _pdf == "sample_text.pdf"
    assert kw["canonical_course_code"] == "ALG_9"
    assert kw["figures_dir"] == str(tmp_path / "figs")
    assert kw["reuse_conversion"] is True


# ---------------------------------------------------------------------------
# (b) seam success=False propagates — NO silent DART fallback.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seam_failure_propagates_no_dart_fallback(monkeypatch, tmp_path):
    import MCP.tools.pipeline_tools as pt

    def _fake_seam(pdf_path, output_path, **kwargs):
        return {
            "success": False,
            "error": "SemantiK runtime deps not provisioned: no module 'llama_cpp'",
            "method": "semantik_v2",
            "output_path": output_path,
            "html_path": output_path,
        }

    monkeypatch.setattr(pt, "_run_semantik_v2_conversion", _fake_seam)

    tool = _registry_tool()
    out = await tool(
        pdf_path="doc.pdf",
        phase="dart_conversion",
        output_dir=str(tmp_path),
    )
    payload = json.loads(out)

    # Conversion FAILS with the SemantiK reason; there is no legacy DART
    # converter to fall back to.
    assert payload["success"] is False
    assert "not provisioned" in payload["error"]
    assert payload["method"] == "semantik_v2"


# ---------------------------------------------------------------------------
# (c) scope — a non-dart_conversion phase fails closed (no legacy path).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_conversion_phase_fails_closed(monkeypatch, tmp_path):
    """A non-``dart_conversion`` phase must NOT reach the SemantiK seam — the
    dispatch is keyed strictly on ``phase == "dart_conversion"``. The legacy
    DART converter was retired, so any other phase fails CLOSED with
    ``method == "unsupported_phase"`` rather than silently converting."""
    import MCP.tools.pipeline_tools as pt

    seam_calls: list = []

    def _fake_seam(*args, **kwargs):
        seam_calls.append((args, kwargs))
        return {"success": True}

    monkeypatch.setattr(pt, "_run_semantik_v2_conversion", _fake_seam)

    tool = _registry_tool()
    out = await tool(
        pdf_path="doc.pdf",
        phase="staging",  # non-conversion phase — unsupported
        output_dir=str(tmp_path),
    )
    payload = json.loads(out)

    # SemantiK seam was NOT touched; the tool fails closed.
    assert seam_calls == [], "SemantiK seam reached from a non-dart_conversion phase"
    assert payload["success"] is False
    assert payload["method"] == "unsupported_phase"


@pytest.mark.asyncio
async def test_no_phase_kwarg_fails_closed(monkeypatch, tmp_path):
    """A caller with no ``phase`` kwarg (direct invocation) is unsupported —
    only the explicit ``dart_conversion`` phase runs the SemantiK seam."""
    import MCP.tools.pipeline_tools as pt

    seam_calls: list = []
    monkeypatch.setattr(
        pt,
        "_run_semantik_v2_conversion",
        lambda *a, **k: seam_calls.append((a, k)) or {"success": True},
    )

    tool = _registry_tool()
    out = await tool(pdf_path="doc.pdf", output_dir=str(tmp_path))
    payload = json.loads(out)
    assert seam_calls == [], "seam reached without a dart_conversion phase"
    assert payload["success"] is False
    assert payload["method"] == "unsupported_phase"

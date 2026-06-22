"""P3c tests — the LIVE PDF→HTML conversion dispatch is flipped to SemantiK.

Run with NO models / GPU. We MOCK ``_run_semantik_v2_conversion`` at the
``MCP.tools.pipeline_tools`` module level so the dispatch wiring is exercised
without touching SemantiK's heavy runtime deps.

Covers (migration plan §5 step 1/2, §2.5 carryover, §3.7 fail-closed):
  * The ``dart_conversion`` phase (textbook_to_course / course_generation
    PDF-ingest path) now CALLS the SemantiK seam, NOT
    ``_raw_text_to_accessible_html``.
  * A seam ``success=False`` propagates as a conversion failure — NO silent
    fall-through to the legacy DART path.
  * The seam params (figures_dir / canonical_course_code / reuse_conversion)
    are threaded from the task kwargs into the seam.
  * Carryover scope: the ``batch_dart`` ``multi_source_synthesis`` phase (and
    any non-``dart_conversion`` caller) does NOT reach the SemantiK seam — it
    stays on the legacy DART path.

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
# (a) dart_conversion phase → calls the SemantiK seam (not legacy DART).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dart_conversion_phase_calls_semantik_seam(monkeypatch, tmp_path):
    import MCP.tools.pipeline_tools as pt

    seam_calls: list = []
    legacy_calls: list = []

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

    def _fake_legacy(*args, **kwargs):
        legacy_calls.append((args, kwargs))
        return "<html>legacy</html>"

    monkeypatch.setattr(pt, "_run_semantik_v2_conversion", _fake_seam)
    monkeypatch.setattr(pt, "_raw_text_to_accessible_html", _fake_legacy)

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

    # The SemantiK seam was called; the legacy DART path was NOT.
    assert len(seam_calls) == 1, "SemantiK seam not invoked on dart_conversion"
    assert legacy_calls == [], "legacy _raw_text_to_accessible_html was called"
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

    legacy_calls: list = []

    def _fake_seam(pdf_path, output_path, **kwargs):
        return {
            "success": False,
            "error": "SemantiK runtime deps not provisioned: no module 'llama_cpp'",
            "method": "semantik_v2",
            "output_path": output_path,
            "html_path": output_path,
        }

    def _fake_legacy(*args, **kwargs):
        legacy_calls.append((args, kwargs))
        return "<html>legacy</html>"

    monkeypatch.setattr(pt, "_run_semantik_v2_conversion", _fake_seam)
    monkeypatch.setattr(pt, "_raw_text_to_accessible_html", _fake_legacy)

    tool = _registry_tool()
    out = await tool(
        pdf_path="doc.pdf",
        phase="dart_conversion",
        output_dir=str(tmp_path),
    )
    payload = json.loads(out)

    # Conversion FAILS with the SemantiK reason; legacy DART never runs.
    assert payload["success"] is False
    assert "not provisioned" in payload["error"]
    assert payload["method"] == "semantik_v2"
    assert legacy_calls == [], (
        "silent DART fallback: legacy path ran after seam failure"
    )


# ---------------------------------------------------------------------------
# (c) carryover scope — non-dart_conversion phase stays on legacy DART.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_source_synthesis_phase_stays_legacy(monkeypatch, tmp_path):
    """batch_dart's multi_source_synthesis must NOT reach the SemantiK seam."""
    import MCP.tools.pipeline_tools as pt

    seam_calls: list = []

    def _fake_seam(*args, **kwargs):
        seam_calls.append((args, kwargs))
        return {"success": True}

    monkeypatch.setattr(pt, "_run_semantik_v2_conversion", _fake_seam)

    # Make the legacy pdftotext path deterministic: feed enough raw text so
    # the legacy branch reaches _raw_text_to_accessible_html.
    import subprocess as _sp

    class _FakeProc:
        stdout = "Algebra foundations. " * 50

    monkeypatch.setattr(
        _sp, "run", lambda *a, **k: _FakeProc()
    )

    legacy_calls: list = []

    def _fake_legacy(raw_text, title, **kwargs):
        legacy_calls.append((raw_text, title, kwargs))
        return "<html><body><main role='main'>legacy</main></body></html>"

    monkeypatch.setattr(pt, "_raw_text_to_accessible_html", _fake_legacy)

    tool = _registry_tool()
    out = await tool(
        pdf_path="doc.pdf",
        phase="multi_source_synthesis",  # batch_dart phase — NOT flipped
        output_dir=str(tmp_path),
    )
    payload = json.loads(out)

    # Legacy DART path ran; SemantiK seam was NOT touched.
    assert seam_calls == [], "SemantiK seam reached from a non-dart_conversion phase"
    assert len(legacy_calls) == 1
    assert payload["success"] is True
    assert payload["method"] == "pdftotext_to_html"


@pytest.mark.asyncio
async def test_no_phase_kwarg_stays_legacy(monkeypatch, tmp_path):
    """A caller with no ``phase`` kwarg (direct/legacy invocation) stays on
    the DART path — only the explicit ``dart_conversion`` phase flips."""
    import MCP.tools.pipeline_tools as pt

    seam_calls: list = []
    monkeypatch.setattr(
        pt,
        "_run_semantik_v2_conversion",
        lambda *a, **k: seam_calls.append((a, k)) or {"success": True},
    )

    import subprocess as _sp

    class _FakeProc:
        stdout = "Algebra foundations. " * 50

    monkeypatch.setattr(_sp, "run", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(
        pt,
        "_raw_text_to_accessible_html",
        lambda *a, **k: "<html><body><main role='main'>x</main></body></html>",
    )

    tool = _registry_tool()
    out = await tool(pdf_path="doc.pdf", output_dir=str(tmp_path))
    payload = json.loads(out)
    assert seam_calls == [], "seam reached without a dart_conversion phase"
    assert payload["method"] == "pdftotext_to_html"

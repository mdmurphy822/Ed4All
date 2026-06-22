"""P3b tests for the SemantiK v2 conversion seam + inline chunk wiring.

Run with NO models / GPU. The cascade (``run_pipeline_v2``) is MOCKED via a
synthetic ``SemantiK.dart_semantic.pipeline_v2`` module injected into
``sys.modules`` so the seam's LAZY import resolves to the mock — proving the
seam never touches SemantiK's heavy runtime deps (axe_playwright_python /
llama_cpp / torch).

Run:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
    python -m pytest MCP/tests/test_semantik_v2_seam.py -q

Covers (migration plan §5 step 1 / §3.7 / §3.3a item 2 / §4.5):
  * seam writes {stem}_accessible.html + both sidecars; returns all required
    config/workflows.yaml:898 keys.
  * exit_action → success mapping (ship_with_confidence / ship_with_flag /
    non_certified_stamp = success=True; unknown = success=False).
  * R4 mock-trap: a non-real runtime_mode returns success=False (fail-closed).
  * --reuse-conversion: with prior artifacts the mocked cascade is NOT called.
  * inline _run_dart_chunking stamps the 6 SemantiK §4 chunk-root fields from a
    fixture adapter-output HTML + a doc-level quality sidecar.
"""

from __future__ import annotations

import json
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Synthetic cascade-result fixture (region_provenance, reused P3a shape).
# ---------------------------------------------------------------------------


def _region_provenance() -> list[dict]:
    return [
        {
            "region_index": 0,
            "region_kind": "heading",
            "role": "heading",
            "confidence": 0.95,
            "wcag_status": "passed",
            "first_raw_block_index": 0,
            "pages": [1],
            "heading_text": "Chapter 1: Foundations",
            "level": 1,
            "figure_alt": None,
            "raw_text": "Chapter 1: Foundations",
        },
        {
            "region_index": 1,
            "region_kind": "paragraph",
            "role": "body",
            "confidence": 0.6,
            "wcag_status": "passed",
            "first_raw_block_index": 1,
            "pages": [2, 3],
            "heading_text": None,
            "level": None,
            "figure_alt": None,
            "raw_text": "The order of operations is PEMDAS.",
        },
    ]


class _MockPipelineResult:
    """Duck-typed ``PipelineV2Result`` with a P3a-surfaced region_provenance."""

    def __init__(self, *, exit_action="ship_with_confidence", runtime_mode="real"):
        self.pdf = "sample_text_ch1.pdf"
        self.wcag_status = "passed"
        self.exit_action = exit_action
        self.theta_score = 0.91
        self.flags: list[str] = []
        self.lane_used = "fast-lane"
        self.lang = "en"
        self.runtime_mode = runtime_mode
        self.region_provenance = _region_provenance()
        self.heading_tree = [(1, "Chapter 1: Foundations")]


def _install_mock_cascade(monkeypatch, *, result=None, recorder=None):
    """Inject a fake ``SemantiK.dart_semantic.cascade`` carrying a mocked
    ``run_pipeline_v2`` so the seam's lazy import resolves to it. (The seam's
    in-process import is ``from SemantiK.dart_semantic.cascade import
    run_pipeline_v2`` — run_pipeline_v2 lives in cascade.py, not pipeline_v2.py.)"""
    res = result if result is not None else _MockPipelineResult()

    def _fake_run_pipeline_v2(pdf_path, *args, **kwargs):
        if recorder is not None:
            recorder.append(pdf_path)
        return res

    pkg = types.ModuleType("SemantiK")
    pkg.__path__ = []  # mark as package
    sub = types.ModuleType("SemantiK.dart_semantic")
    sub.__path__ = []
    mod = types.ModuleType("SemantiK.dart_semantic.cascade")
    mod.run_pipeline_v2 = _fake_run_pipeline_v2
    monkeypatch.setitem(sys.modules, "SemantiK", pkg)
    monkeypatch.setitem(sys.modules, "SemantiK.dart_semantic", sub)
    monkeypatch.setitem(sys.modules, "SemantiK.dart_semantic.cascade", mod)
    return res


# ---------------------------------------------------------------------------
# (a) Seam: writes HTML + sidecars + returns all required keys.
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "output_path",
    "output_paths",
    "html_path",
    "html_paths",
    "success",
    "html_length",
}


def test_a_seam_writes_html_and_sidecars(monkeypatch, tmp_path):
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    _install_mock_cascade(monkeypatch)
    out = tmp_path / "sample_text_ch1_accessible.html"
    result = _run_semantik_v2_conversion(
        "sample_text_ch1.pdf", str(out)
    )

    assert result["success"] is True
    assert _REQUIRED_KEYS.issubset(result.keys()), (
        f"missing required keys: {_REQUIRED_KEYS - set(result.keys())}"
    )
    assert result["method"] == "semantik_v2"
    assert result["html_length"] == len(out.read_text(encoding="utf-8"))

    # HTML + both sidecars on disk.
    assert out.is_file()
    synth = tmp_path / "sample_text_ch1_accessible_synthesized.json"
    quality = tmp_path / "sample_text_ch1_accessible.quality.json"
    assert synth.is_file()
    assert quality.is_file()
    # Sidecars are valid JSON carrying the canonical shapes.
    synth_doc = json.loads(synth.read_text(encoding="utf-8"))
    assert synth_doc["sections"]
    quality_doc = json.loads(quality.read_text(encoding="utf-8"))
    assert quality_doc["exit_action"] == "ship_with_confidence"
    assert quality_doc["certification_status"] == "certified"

    # The normalized HTML passes the dart_markers gate.
    from lib.validators.dart_markers import DartMarkersValidator

    html = out.read_text(encoding="utf-8")
    res = DartMarkersValidator().validate({"html_content": html})
    assert res.passed, [i.code for i in res.issues if i.severity == "critical"]


# ---------------------------------------------------------------------------
# (b) exit_action → success mapping.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exit_action,expect_success,expect_cert",
    [
        ("ship_with_confidence", True, "certified"),
        ("ship_with_flag", True, "flagged"),
        ("non_certified_stamp", True, "non_certified"),
        ("some_unknown_action", False, "error"),
        (None, False, "error"),
    ],
)
def test_b_exit_action_success_mapping(
    monkeypatch, tmp_path, exit_action, expect_success, expect_cert
):
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    _install_mock_cascade(
        monkeypatch,
        result=_MockPipelineResult(exit_action=exit_action, runtime_mode="real"),
    )
    out = tmp_path / "doc_accessible.html"
    result = _run_semantik_v2_conversion("doc.pdf", str(out))

    assert result["success"] is expect_success
    assert result["certification_status"] == expect_cert


# ---------------------------------------------------------------------------
# (c) R4 mock trap — non-real runtime_mode fails closed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runtime_mode", ["mock", None, "stub"])
def test_c_mock_runtime_fails_closed(monkeypatch, tmp_path, runtime_mode):
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    _install_mock_cascade(
        monkeypatch,
        result=_MockPipelineResult(runtime_mode=runtime_mode),
    )
    out = tmp_path / "doc_accessible.html"
    result = _run_semantik_v2_conversion("doc.pdf", str(out))

    assert result["success"] is False
    assert "real mode" in result["error"]
    # Must NOT have written HTML — mock output never ships.
    assert not out.is_file()


def test_c_runtime_mode_from_cascade_dict(monkeypatch, tmp_path):
    """runtime_mode resolves from a ``.cascade`` dict too (not just attr)."""
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    res = _MockPipelineResult(runtime_mode="real")
    # Hide the top-level attr; carry it in a .cascade dict instead.
    delattr(res, "runtime_mode")
    res.cascade = {"runtime_mode": "real"}
    _install_mock_cascade(monkeypatch, result=res)
    out = tmp_path / "doc_accessible.html"
    result = _run_semantik_v2_conversion("doc.pdf", str(out))
    assert result["success"] is True


# ---------------------------------------------------------------------------
# (d) missing-dep ImportError → clear error dict (no silent fallback).
# ---------------------------------------------------------------------------


def test_d_missing_semantik_deps_clear_error(monkeypatch, tmp_path):
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    # Force the lazy import to fail by injecting a module whose attr access
    # raises ImportError on run_pipeline_v2.
    pkg = types.ModuleType("SemantiK")
    pkg.__path__ = []
    sub = types.ModuleType("SemantiK.dart_semantic")
    sub.__path__ = []
    mod = types.ModuleType("SemantiK.dart_semantic.pipeline_v2")
    # No run_pipeline_v2 attribute → ImportError on `from ... import`.
    monkeypatch.setitem(sys.modules, "SemantiK", pkg)
    monkeypatch.setitem(sys.modules, "SemantiK.dart_semantic", sub)
    monkeypatch.setitem(sys.modules, "SemantiK.dart_semantic.pipeline_v2", mod)

    out = tmp_path / "doc_accessible.html"
    result = _run_semantik_v2_conversion("doc.pdf", str(out))
    assert result["success"] is False
    assert "not provisioned" in result["error"]
    assert not out.is_file()


# ---------------------------------------------------------------------------
# (e) --reuse-conversion — prior artifacts reused, cascade NOT called.
# ---------------------------------------------------------------------------


def test_e_reuse_conversion_skips_cascade(monkeypatch, tmp_path):
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    out = tmp_path / "doc_accessible.html"
    # Pre-seed prior artifacts (a flagged ship so we also prove provenance
    # round-trips through reuse).
    out.write_text(
        "<!DOCTYPE html><html><body><main role='main'>"
        "<p>prior</p></main></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "doc_accessible.quality.json").write_text(
        json.dumps(
            {
                "compliant": True,
                "wcag_status": "passed",
                "theta_score": 0.88,
                "exit_action": "ship_with_flag",
                "certification_status": "flagged",
                "flags": ["truncated_math"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "doc_accessible_synthesized.json").write_text(
        json.dumps({"sections": []}), encoding="utf-8"
    )

    called: list = []
    _install_mock_cascade(monkeypatch, recorder=called)

    result = _run_semantik_v2_conversion(
        "doc.pdf", str(out), reuse_conversion=True
    )

    assert called == [], "cascade was invoked despite reuse-conversion"
    assert result["reused_conversion"] is True
    assert result["success"] is True
    assert result["certification_status"] == "flagged"
    assert result["exit_action"] == "ship_with_flag"
    assert result["theta_score"] == 0.88
    assert result["flags"] == ["truncated_math"]
    assert _REQUIRED_KEYS.issubset(result.keys())


def test_e_reuse_conversion_env_var(monkeypatch, tmp_path):
    """ED4ALL_REUSE_CONVERSION env mirrors the flag."""
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    out = tmp_path / "doc_accessible.html"
    out.write_text("<html><body><main>x</main></body></html>", encoding="utf-8")
    monkeypatch.setenv("ED4ALL_REUSE_CONVERSION", "1")
    called: list = []
    _install_mock_cascade(monkeypatch, recorder=called)
    result = _run_semantik_v2_conversion("doc.pdf", str(out))
    assert called == []
    assert result["reused_conversion"] is True


def test_e_reuse_conversion_no_prior_runs_cascade(monkeypatch, tmp_path):
    """reuse ON but no prior artifacts → cascade runs (no silent skip)."""
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    out = tmp_path / "doc_accessible.html"  # does not exist yet
    called: list = []
    _install_mock_cascade(monkeypatch, recorder=called)
    result = _run_semantik_v2_conversion(
        "doc.pdf", str(out), reuse_conversion=True
    )
    assert called == ["doc.pdf"], "cascade should run when no prior artifacts"
    assert result["success"] is True
    assert out.is_file()


# ---------------------------------------------------------------------------
# (f) Inline chunk 6-field mirror — _run_dart_chunking stamps from HTML +
#     doc-level quality sidecar.
# ---------------------------------------------------------------------------


def _adapter_html_fixture() -> str:
    """A small SemantiK-adapter-shaped HTML with the data-dart enrichment.

    Reuses the real adapter's render so the data-dart-block-role /
    -confidence / -wcag attrs land on the SAME element as data-dart-block-id
    (the chunker's same-element pairing requirement).
    """
    from lib.semantik.adapter import _AdapterBlock, _AdapterChapter, _render_html

    blocks = [
        _AdapterBlock(
            html="<p>The order of operations is PEMDAS, a key foundation.</p>",
            region_kind="paragraph",
            raw_block_index=1,
            raw_text="The order of operations is PEMDAS, a key foundation.",
            heading_text="Order of Operations",
            pages=(2, 3),
            confidence=0.6,
            block_role="body",
            wcag_status="passed",
        ),
    ]
    chapter = _AdapterChapter(title="Chapter 1: Foundations", blocks=blocks)
    return _render_html([chapter], title="Chapter 1", lang="en")


@pytest.mark.asyncio
async def test_f_inline_chunk_stamps_six_fields(monkeypatch, tmp_path):
    """The inline _run_dart_chunking path stamps the 6 SemantiK §4 fields."""
    from MCP.tools.pipeline_tools import _build_tool_registry

    staging = tmp_path / "staging"
    staging.mkdir()
    html = _adapter_html_fixture()
    html_file = staging / "sample_text_ch1_accessible.html"
    html_file.write_text(html, encoding="utf-8")
    # Doc-level quality sidecar next to the HTML (the seam writes this).
    (staging / "sample_text_ch1_accessible.quality.json").write_text(
        json.dumps(
            {
                "compliant": True,
                "wcag_status": "passed",
                "theta_score": 0.91,
                "exit_action": "ship_with_confidence",
                "certification_status": "certified",
            }
        ),
        encoding="utf-8",
    )

    libv2_root = tmp_path / "libv2"
    registry = _build_tool_registry()
    run_dart_chunking = registry["run_dart_chunking"]

    out = await run_dart_chunking(
        staging_dir=str(staging),
        course_name="sample_text",
        libv2_root=str(libv2_root),
    )
    payload = json.loads(out)
    assert payload.get("success"), payload
    chunks_path = payload["dart_chunks_path"]
    chunks = [
        json.loads(line)
        for line in open(chunks_path, encoding="utf-8")
        if line.strip()
    ]
    assert chunks, "no chunks emitted"

    # At least one chunk carries the harvested + doc-level enrichment.
    role_chunks = [c for c in chunks if c.get("source_block_role")]
    assert role_chunks, "no chunk stamped source_block_role"
    c = role_chunks[0]
    assert c["source_block_role"] == "body"
    assert c["source_block_confidence"] == pytest.approx(0.6)
    assert c["wcag_block_status"] == "passed"
    # Doc-level signals threaded onto every chunk from the quality sidecar.
    assert c["semantic_preservation_score"] == pytest.approx(0.91)
    assert c["certification_status"] == "certified"


@pytest.mark.asyncio
async def test_f_inline_chunk_omits_fields_without_enrichment(
    monkeypatch, tmp_path
):
    """Legacy / non-SemantiK HTML (no enrichment, no quality sidecar) stamps
    NONE of the 6 fields (back-compat, byte-stable)."""
    from MCP.tools.pipeline_tools import _build_tool_registry

    staging = tmp_path / "staging"
    staging.mkdir()
    # Plain DART HTML with block-id but no role/confidence/wcag, no sidecar.
    html = (
        "<!DOCTYPE html><html lang='en'><body>"
        "<a class='skip-link' href='#m'>skip</a>"
        "<main id='m' role='main' class='dart-document'>"
        "<article role='doc-chapter'><h2>Ch 1</h2>"
        "<section class='dart-section' aria-labelledby='s1' "
        "data-dart-block-id='s1' data-dart-source='synthesized'>"
        "<h3 id='s1'>Topic</h3>"
        "<p>Some grounded body content about algebra foundations here.</p>"
        "</section></article></main></body></html>"
    )
    (staging / "legacy_accessible.html").write_text(html, encoding="utf-8")

    libv2_root = tmp_path / "libv2"
    registry = _build_tool_registry()
    out = await registry["run_dart_chunking"](
        staging_dir=str(staging),
        course_name="legacy",
        libv2_root=str(libv2_root),
    )
    payload = json.loads(out)
    assert payload.get("success"), payload
    chunks = [
        json.loads(line)
        for line in open(payload["dart_chunks_path"], encoding="utf-8")
        if line.strip()
    ]
    assert chunks
    for c in chunks:
        for field in (
            "source_block_role",
            "source_block_confidence",
            "wcag_block_status",
            "semantic_preservation_score",
            "certification_status",
        ):
            assert field not in c, f"unexpected {field} on legacy chunk"

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


# ---------------------------------------------------------------------------
# (g) LOCAL-real wiring — the seam builds a POPULATED V2Config from the
#     on-disk vendored weights and passes it into run_pipeline_v2 so the
#     cascade auto-derives runtime_mode="real" (not the all-None default that
#     silently resolves to "mock"). Regression for the local-real gap: the
#     in-process LOCAL path previously called run_pipeline_v2(pdf) with no
#     config → DEFAULT_V2_CONFIG (all None) → is_qwen_specialist_loaded()
#     False → runtime_mode="mock" → R4 mock-trap refuses to ship.
#
#     The heavy GGUF/council loads are NOT exercised here (CI has no GPU): the
#     resolver is unit-tested against a fake on-disk model dir, and the seam
#     test asserts the resolved config is THREADED THROUGH to run_pipeline_v2.
# ---------------------------------------------------------------------------


def test_g_resolver_populates_from_on_disk_weights(tmp_path):
    """V2Config.from_model_dir / resolve_local_v2_config flips the
    runtime_mode gate ONLY when the artifacts exist on disk."""
    from SemantiK.dart_semantic.v2_config import (
        DEFAULT_V2_CONFIG,
        V2Config,
        resolve_local_v2_config,
    )

    # Empty model dir → every slot None → gate stays mock (graceful partial).
    empty = tmp_path / "empty_models"
    empty.mkdir()
    cfg_empty = V2Config.from_model_dir(empty)
    assert cfg_empty.is_qwen_specialist_loaded() is False
    assert cfg_empty.is_council_loaded() is False
    assert cfg_empty.theta_semantic_scorer_path is None

    # Control: the all-None default never flips the gate.
    assert DEFAULT_V2_CONFIG.is_qwen_specialist_loaded() is False

    # Fabricate the canonical vendored layout: one council adapter dir, one
    # Qwen GGUF (named per the real config.yaml), and the theta head dir.
    models = tmp_path / "models"
    (models / "council" / "semantic" / "final").mkdir(parents=True)
    (models / "theta" / "semantic_preservation" / "v8").mkdir(parents=True)
    prose_dir = models / "qwen_specialists" / "prose" / "v1"
    prose_dir.mkdir(parents=True)
    (prose_dir / "prose.q4_k_m.gguf").write_bytes(b"\x00")  # presence only

    cfg = V2Config.from_model_dir(models)
    # The real config.yaml's prose adapter_path resolves under this root.
    assert cfg.qwen_prose_adapter is not None
    assert cfg.qwen_prose_adapter.exists()
    assert cfg.is_qwen_specialist_loaded() is True  # <-- flips runtime_mode→real
    assert cfg.council_semantic_adapter is not None
    assert cfg.is_council_loaded() is True
    assert cfg.theta_semantic_scorer_path is not None
    # A specialist whose GGUF is absent on disk stays None (partial install).
    assert cfg.qwen_table_adapter is None

    # resolve_local_v2_config(model_dir) is the same builder.
    assert resolve_local_v2_config(models).is_qwen_specialist_loaded() is True


def _install_mock_cascade_with_config(monkeypatch, *, captured, runtime_mode="real"):
    """Like _install_mock_cascade, but the synthetic SemantiK.dart_semantic
    subpackage ALSO carries a v2_config module exposing a recording
    resolve_local_v2_config, and the fake run_pipeline_v2 records the
    ``config`` kwarg the seam passes. Proves the seam (a) imports the resolver
    and (b) threads its output into the cascade call."""
    res = _MockPipelineResult(runtime_mode=runtime_mode)

    # A sentinel "populated" config object the recorder can identify.
    class _SentinelConfig:
        def is_qwen_specialist_loaded(self):  # parity with the real API
            return True

    sentinel = _SentinelConfig()

    def _fake_run_pipeline_v2(pdf_path, *args, config=None, **kwargs):
        captured["pdf"] = pdf_path
        captured["config"] = config
        return res

    def _fake_resolve_local_v2_config(model_dir=None):
        captured["resolver_called"] = True
        return sentinel

    pkg = types.ModuleType("SemantiK")
    pkg.__path__ = []
    sub = types.ModuleType("SemantiK.dart_semantic")
    sub.__path__ = []
    casc = types.ModuleType("SemantiK.dart_semantic.cascade")
    casc.run_pipeline_v2 = _fake_run_pipeline_v2
    vcfg = types.ModuleType("SemantiK.dart_semantic.v2_config")
    vcfg.resolve_local_v2_config = _fake_resolve_local_v2_config
    monkeypatch.setitem(sys.modules, "SemantiK", pkg)
    monkeypatch.setitem(sys.modules, "SemantiK.dart_semantic", sub)
    monkeypatch.setitem(sys.modules, "SemantiK.dart_semantic.cascade", casc)
    monkeypatch.setitem(sys.modules, "SemantiK.dart_semantic.v2_config", vcfg)
    return sentinel


def test_g_seam_threads_local_config_into_cascade(monkeypatch, tmp_path):
    """The in-process seam arm passes the resolved local V2Config to
    run_pipeline_v2 (not the bare no-config call)."""
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    captured: dict = {}
    sentinel = _install_mock_cascade_with_config(monkeypatch, captured=captured)
    out = tmp_path / "doc_accessible.html"
    result = _run_semantik_v2_conversion("doc.pdf", str(out))

    assert result["success"] is True
    assert captured.get("resolver_called") is True, "resolver must be invoked"
    assert captured.get("config") is sentinel, (
        "seam must thread the resolved local config into run_pipeline_v2 "
        "(else runtime_mode silently resolves to mock)"
    )


def test_g_seam_degrades_when_resolver_raises(monkeypatch, tmp_path):
    """A resolver failure never crashes the seam — it falls back to the
    no-config call (run still completes; runtime_mode may then be mock, which
    the R4 trap handles separately)."""
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    captured: dict = {}
    _install_mock_cascade_with_config(monkeypatch, captured=captured)

    # Replace the resolver with one that raises.
    def _boom(model_dir=None):
        raise RuntimeError("simulated resolver failure")

    monkeypatch.setitem(
        sys.modules["SemantiK.dart_semantic.v2_config"].__dict__,
        "resolve_local_v2_config",
        _boom,
    )
    out = tmp_path / "doc_accessible.html"
    result = _run_semantik_v2_conversion("doc.pdf", str(out))

    # Cascade still ran (with config=None) and the mocked real result shipped.
    assert "config" in captured and captured["config"] is None
    assert result["success"] is True


# ---------------------------------------------------------------------------
# (h) In-process runtime_mode resolution — run_full_cascade does NOT put a
#     top-level "runtime_mode" key on its result dict (only the cross-venv
#     bridge does). The canonical in-process location is
#     cascade["conformance_audit"]["provenance"]["runtime_mode"]. The seam's
#     resolver must read it there, else a REAL in-process run trips the R4
#     mock-trap and fails closed. Regression for the second local-real gap.
# ---------------------------------------------------------------------------


def test_h_runtime_mode_from_conformance_audit_provenance():
    """The seam resolves runtime_mode from the conformance-audit provenance
    block (the real in-process PipelineV2Result shape)."""
    from MCP.tools.pipeline_tools import (
        _semantik_resolve_runtime_mode,
        _SEMANTIK_REAL_RUNTIME_MODES,
    )

    class _RealResult:
        # No top-level .runtime_mode (matches the real PipelineV2Result).
        runtime_mode = None
        cascade = {
            # No top-level "runtime_mode" key (matches run_full_cascade's
            # result dict) — only the nested provenance block carries it.
            "conformance_audit": {"provenance": {"runtime_mode": "real"}},
        }

    resolved = _semantik_resolve_runtime_mode(_RealResult())
    assert resolved == "real"
    assert resolved in _SEMANTIK_REAL_RUNTIME_MODES

    # A mock in-process run is still caught (provenance says mock).
    class _MockResult:
        runtime_mode = None
        cascade = {"conformance_audit": {"provenance": {"runtime_mode": "mock"}}}

    assert _semantik_resolve_runtime_mode(_MockResult()) == "mock"

    # Top-level key still wins when present (the bridge arm's shape).
    class _BridgeResult:
        runtime_mode = None
        cascade = {"runtime_mode": "real", "conformance_audit": {}}

    assert _semantik_resolve_runtime_mode(_BridgeResult()) == "real"


def test_h_seam_ships_real_in_process_result(monkeypatch, tmp_path):
    """End-to-end seam: a real in-process result whose runtime_mode lives ONLY
    in the conformance-audit provenance block ships success=True (does not trip
    the mock-trap)."""
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    res = _MockPipelineResult(runtime_mode="real")
    # Strip the convenience top-level attr so resolution MUST come from the
    # conformance-audit provenance block (the real in-process shape).
    delattr(res, "runtime_mode")
    res.cascade = {
        "conformance_audit": {"provenance": {"runtime_mode": "real"}},
    }

    captured: dict = {}
    _install_mock_cascade_with_config(monkeypatch, captured=captured)
    # Point the fake run_pipeline_v2 at our provenance-shaped result.
    sys.modules["SemantiK.dart_semantic.cascade"].run_pipeline_v2 = (
        lambda pdf_path, *a, **k: res
    )

    out = tmp_path / "doc_accessible.html"
    result = _run_semantik_v2_conversion("doc.pdf", str(out))
    assert result["success"] is True, result.get("error")
    assert out.is_file()


# ---------------------------------------------------------------------------
# (i) Frozen-result tolerance — the in-process PipelineV2Result is a FROZEN
#     dataclass, so the seam's chapters-IR stamp must use object.__setattr__
#     (a plain setattr raises FrozenInstanceError and aborts a real run AFTER
#     the mock-trap). Regression for the third local-real gap.
# ---------------------------------------------------------------------------


def test_i_seam_handles_frozen_pipeline_result(monkeypatch, tmp_path):
    """A frozen-dataclass cascade result (the real in-process shape) does not
    raise FrozenInstanceError on the chapters-IR stamp."""
    import dataclasses

    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    @dataclasses.dataclass(frozen=True)
    class _FrozenResult:
        pdf: str = "doc.pdf"
        html: str = "<p>x</p>"
        wcag_status: str = "passed"
        exit_action: str = "ship_with_confidence"
        theta_score: float = 0.9
        flags: tuple = ()
        lane_used: str = "fast-lane"
        lang: str = "en"
        runtime_mode: str = "real"
        region_provenance: tuple = ()
        heading_tree: tuple = ()

    frozen = _FrozenResult(region_provenance=tuple(_region_provenance()))

    # Sanity: a plain setattr on this instance WOULD raise (proves the test
    # exercises the frozen path the fix guards).
    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.chapters = []  # type: ignore[attr-defined]

    captured: dict = {}
    _install_mock_cascade_with_config(monkeypatch, captured=captured)
    sys.modules["SemantiK.dart_semantic.cascade"].run_pipeline_v2 = (
        lambda pdf_path, *a, **k: frozen
    )

    out = tmp_path / "doc_accessible.html"
    result = _run_semantik_v2_conversion("doc.pdf", str(out))
    assert result["success"] is True, result.get("error")
    assert out.is_file()


# ---------------------------------------------------------------------------
# GAP A — figure-sidecar relocation (`_relocate_figure_sidecars`).
#
# The cascade writes `{pdf_stem}_figures/` next to the SOURCE PDF and stamps
# each `<img src>` as `./{pdf_stem}_figures/fig-N.png`. When the HTML lands in
# a DIFFERENT directory (staging / convert-slice), that relative `src` would
# dangle. The seam relocates the dir to `{html_stem}_figures/` next to the
# HTML and rewrites the `src` so the written dir == the emitted src.
# ---------------------------------------------------------------------------
def _write_fig_dir(base, dirname, names):
    d = base / dirname
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"\x89PNG\r\n\x1a\n" + n.encode())
    return d


def test_gap_a_relocates_figures_and_rewrites_src(tmp_path):
    from MCP.tools.pipeline_tools import _relocate_figure_sidecars
    from pathlib import Path

    # PDF under inputs/, HTML under inputs/slice-out/ — the divergent case.
    pdf_dir = tmp_path / "inputs"
    html_dir = pdf_dir / "slice-out"
    pdf_dir.mkdir(parents=True)
    html_dir.mkdir(parents=True)
    pdf = pdf_dir / "slice-mini.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    html_path = html_dir / "mini.html"

    # Cascade wrote the PNGs next to the PDF, stem = pdf stem.
    src_dir = _write_fig_dir(pdf_dir, "slice-mini_figures", ["fig-1.png", "fig-17.png"])

    html = (
        '<figure><img alt="Figure." src="./slice-mini_figures/fig-1.png"></figure>'
        '<figure><img alt="Figure." src="./slice-mini_figures/fig-17.png"></figure>'
    )
    out = _relocate_figure_sidecars(html, pdf_path=pdf, html_path=html_path)

    # Dir relocated to {html_stem}_figures next to the HTML; source removed.
    dst_dir = html_dir / "mini_figures"
    assert dst_dir.is_dir()
    assert (dst_dir / "fig-1.png").is_file()
    assert (dst_dir / "fig-17.png").is_file()
    assert not src_dir.exists()

    # src rewritten to the HTML-relative dir.
    assert "./mini_figures/fig-1.png" in out
    assert "./mini_figures/fig-17.png" in out
    assert "slice-mini_figures" not in out

    # Every <img src> resolves to a written file (relative to the HTML dir).
    import re as _re
    for m in _re.finditer(r'src="\./([^"]+)"', out):
        assert (html_dir / m.group(1)).is_file()


def test_gap_a_noop_when_pdf_and_html_colocated_same_stem(tmp_path):
    """Standalone run: PDF + HTML co-located with the same stem → no move."""
    from MCP.tools.pipeline_tools import _relocate_figure_sidecars

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    html_path = tmp_path / "doc.html"  # same stem, same dir
    src_dir = _write_fig_dir(tmp_path, "doc_figures", ["fig-2.png"])

    html = '<figure><img src="./doc_figures/fig-2.png"></figure>'
    out = _relocate_figure_sidecars(html, pdf_path=pdf, html_path=html_path)

    # Untouched: dir stays put, src unchanged.
    assert src_dir.is_dir()
    assert (src_dir / "fig-2.png").is_file()
    assert out == html


def test_gap_a_noop_when_no_figures_written(tmp_path):
    """Figure path off / no figures → source dir absent → HTML unchanged."""
    from MCP.tools.pipeline_tools import _relocate_figure_sidecars

    pdf = (tmp_path / "inputs"); pdf.mkdir()
    pdf_file = pdf / "slice.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")
    html_path = pdf / "out" / "slice.html"
    html_path.parent.mkdir(parents=True)

    html = "<p>no figures here</p>"
    out = _relocate_figure_sidecars(html, pdf_path=pdf_file, html_path=html_path)
    assert out == html
    assert not (html_path.parent / "slice_figures").exists()

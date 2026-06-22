"""Cross-venv SemantiK subprocess-bridge tests (migration plan §5 / Section 8 M4).

Run with NO models / GPU / real cascade. The subprocess is MOCKED via
``monkeypatch`` on ``subprocess.run`` so the seam's bridge arm is exercised
without ever spawning a real process or touching SemantiK's heavy deps:

  * SEMANTIK_PYTHON set + in-process import fails → the seam shells out, reads
    the (mock-written) bridge JSON, builds the IR + adapter, writes
    ``{stem}_accessible.html`` + sidecars, returns success + required keys.
  * SEMANTIK_PYTHON unset + in-process import fails → fail-closed clear error
    (NO DART fallback), no HTML written.
  * subprocess non-zero exit → fail-closed.
  * bridge JSON carrying ``{"error": ...}`` → fail-closed.

Run:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
    python -m pytest MCP/tests/test_semantik_bridge_subprocess.py -q
"""

from __future__ import annotations

import json
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Synthetic bridge-JSON (the P3a region_provenance shape, two chapters).
# ---------------------------------------------------------------------------


def _bridge_json(*, runtime_mode: str = "real", exit_action: str = "ship_with_confidence") -> dict:
    return {
        "pdf": "sample_text_ch1.pdf",
        "html": "",  # the seam re-derives HTML from the IR via the adapter
        "region_provenance": [
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
                "raw_text": "The order of operations is PEMDAS, a key foundation.",
            },
        ],
        "heading_tree": [[1, "Chapter 1: Foundations"]],
        "exit_action": exit_action,
        "theta_score": 0.91,
        "wcag_status": "passed",
        "flags": [],
        "lane_used": "fast-lane",
        "runtime_mode": runtime_mode,
    }


def _force_inprocess_import_fail(monkeypatch):
    """Inject a SemantiK.dart_semantic.pipeline_v2 WITHOUT run_pipeline_v2 so
    the seam's in-process lazy import raises ImportError → bridge arm."""
    pkg = types.ModuleType("SemantiK")
    pkg.__path__ = []
    sub = types.ModuleType("SemantiK.dart_semantic")
    sub.__path__ = []
    mod = types.ModuleType("SemantiK.dart_semantic.pipeline_v2")
    monkeypatch.setitem(sys.modules, "SemantiK", pkg)
    monkeypatch.setitem(sys.modules, "SemantiK.dart_semantic", sub)
    monkeypatch.setitem(sys.modules, "SemantiK.dart_semantic.pipeline_v2", mod)


class _FakeCompleted:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def _mock_subprocess_run(monkeypatch, *, bridge=None, returncode=0, stderr="", record=None):
    """Patch MCP.tools.pipeline_tools subprocess.run to drop a bridge JSON.

    The seam invokes ``subprocess.run([py, script, '--pdf', pdf, '--out-json',
    tmp, '--runtime', ...], cwd=...)``. We parse out the ``--out-json`` path and
    (when ``bridge`` is given) write it, then return a fake CompletedProcess.
    """
    def _fake_run(cmd, *args, **kwargs):
        if record is not None:
            record.append({"cmd": cmd, "cwd": kwargs.get("cwd")})
        # Find --out-json target.
        out_json = None
        for i, tok in enumerate(cmd):
            if tok == "--out-json" and i + 1 < len(cmd):
                out_json = cmd[i + 1]
        if bridge is not None and out_json is not None:
            from pathlib import Path

            Path(out_json).write_text(json.dumps(bridge), encoding="utf-8")
        return _FakeCompleted(returncode=returncode, stderr=stderr)

    # The seam imports subprocess INSIDE the helper (``import subprocess``),
    # so it resolves the stdlib module at call time — patch that.
    import subprocess as _stdlib_subprocess

    monkeypatch.setattr(_stdlib_subprocess, "run", _fake_run, raising=False)


_REQUIRED_KEYS = {
    "output_path",
    "output_paths",
    "html_path",
    "html_paths",
    "success",
    "html_length",
}


# ---------------------------------------------------------------------------
# (a) Bridge happy path — subprocess writes JSON, seam builds IR + adapter.
# ---------------------------------------------------------------------------


def test_a_bridge_writes_html_and_returns_keys(monkeypatch, tmp_path):
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    monkeypatch.setenv("SEMANTIK_PYTHON", "/fake/venv/python")
    monkeypatch.setenv("SEMANTIK_RUNTIME_DIR", str(tmp_path / "semantik_repo"))
    _force_inprocess_import_fail(monkeypatch)
    record: list = []
    _mock_subprocess_run(monkeypatch, bridge=_bridge_json(), record=record)

    out = tmp_path / "sample_text_ch1_accessible.html"
    result = _run_semantik_v2_conversion("sample_text_ch1.pdf", str(out))

    assert result["success"] is True, result
    assert _REQUIRED_KEYS.issubset(result.keys())
    assert result["method"] == "semantik_v2"
    assert result["runtime_mode"] == "real"
    # HTML + both sidecars written on the Ed4All side.
    assert out.is_file()
    assert (tmp_path / "sample_text_ch1_accessible_synthesized.json").is_file()
    assert (tmp_path / "sample_text_ch1_accessible.quality.json").is_file()
    # The HTML the adapter produced passes the dart_markers gate.
    from lib.validators.dart_markers import DartMarkersValidator

    html = out.read_text(encoding="utf-8")
    res = DartMarkersValidator().validate({"html_content": html})
    assert res.passed, [i.code for i in res.issues if i.severity == "critical"]

    # The subprocess was invoked with cwd = SEMANTIK_RUNTIME_DIR.
    assert record, "subprocess.run was not called"
    assert record[0]["cwd"] == str(tmp_path / "semantik_repo")
    assert "--pdf" in record[0]["cmd"]
    assert "--out-json" in record[0]["cmd"]


# ---------------------------------------------------------------------------
# (b) SEMANTIK_PYTHON unset + import fails → fail-closed clear (NO fallback).
# ---------------------------------------------------------------------------


def test_b_no_bridge_env_fails_closed(monkeypatch, tmp_path):
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    monkeypatch.delenv("SEMANTIK_PYTHON", raising=False)
    monkeypatch.delenv("SEMANTIK_RUNTIME_DIR", raising=False)
    _force_inprocess_import_fail(monkeypatch)
    # subprocess.run must NEVER be called on this path.
    record: list = []
    _mock_subprocess_run(monkeypatch, bridge=_bridge_json(), record=record)

    out = tmp_path / "doc_accessible.html"
    result = _run_semantik_v2_conversion("doc.pdf", str(out))

    assert result["success"] is False
    assert "not provisioned" in result["error"]
    assert "SEMANTIK_PYTHON" in result["error"]
    assert not out.is_file()
    assert record == [], "subprocess must not run without bridge env"


# ---------------------------------------------------------------------------
# (c) subprocess non-zero exit → fail-closed (no silent DART fallback).
# ---------------------------------------------------------------------------


def test_c_subprocess_nonzero_fails_closed(monkeypatch, tmp_path):
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    monkeypatch.setenv("SEMANTIK_PYTHON", "/fake/venv/python")
    monkeypatch.setenv("SEMANTIK_RUNTIME_DIR", str(tmp_path))
    _force_inprocess_import_fail(monkeypatch)
    # No bridge file written; non-zero exit + stderr.
    _mock_subprocess_run(
        monkeypatch, bridge=None, returncode=1, stderr="boom: GGUF missing"
    )

    out = tmp_path / "doc_accessible.html"
    result = _run_semantik_v2_conversion("doc.pdf", str(out))

    assert result["success"] is False
    assert "exited 1" in result["error"] or "boom" in result["error"]
    assert not out.is_file()


# ---------------------------------------------------------------------------
# (d) bridge JSON carrying {"error": ...} → fail-closed (runner caught a
#     cascade/import failure inside the SemantiK venv).
# ---------------------------------------------------------------------------


def test_d_bridge_error_json_fails_closed(monkeypatch, tmp_path):
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    monkeypatch.setenv("SEMANTIK_PYTHON", "/fake/venv/python")
    monkeypatch.setenv("SEMANTIK_RUNTIME_DIR", str(tmp_path))
    _force_inprocess_import_fail(monkeypatch)
    # Runner exited 3 (cascade failure) AND wrote an {"error"} JSON.
    _mock_subprocess_run(
        monkeypatch,
        bridge={"error": "SemantiK cascade failed: StageThirteenStubRequired", "pdf": "doc.pdf"},
        returncode=3,
    )

    out = tmp_path / "doc_accessible.html"
    result = _run_semantik_v2_conversion("doc.pdf", str(out))

    assert result["success"] is False
    assert "StageThirteenStubRequired" in result["error"]
    assert not out.is_file()


# ---------------------------------------------------------------------------
# (e) R4 mock-trap over the bridge — runtime_mode != real fails closed even
#     when the subprocess succeeds (mock HTML never ships as real).
# ---------------------------------------------------------------------------


def test_e_bridge_mock_runtime_fails_closed(monkeypatch, tmp_path):
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    monkeypatch.setenv("SEMANTIK_PYTHON", "/fake/venv/python")
    monkeypatch.setenv("SEMANTIK_RUNTIME_DIR", str(tmp_path))
    _force_inprocess_import_fail(monkeypatch)
    _mock_subprocess_run(monkeypatch, bridge=_bridge_json(runtime_mode="mock"))

    out = tmp_path / "doc_accessible.html"
    result = _run_semantik_v2_conversion("doc.pdf", str(out))

    assert result["success"] is False
    assert "real mode" in result["error"]
    assert not out.is_file()


# ---------------------------------------------------------------------------
# (f) reuse-conversion honored BEFORE the subprocess (cascade not invoked).
# ---------------------------------------------------------------------------


def test_f_reuse_skips_bridge_subprocess(monkeypatch, tmp_path):
    from MCP.tools.pipeline_tools import _run_semantik_v2_conversion

    monkeypatch.setenv("SEMANTIK_PYTHON", "/fake/venv/python")
    monkeypatch.setenv("SEMANTIK_RUNTIME_DIR", str(tmp_path))
    _force_inprocess_import_fail(monkeypatch)
    record: list = []
    _mock_subprocess_run(monkeypatch, bridge=_bridge_json(), record=record)

    out = tmp_path / "doc_accessible.html"
    out.write_text(
        "<!DOCTYPE html><html><body><main role='main'><p>prior</p></main></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "doc_accessible.quality.json").write_text(
        json.dumps(
            {
                "compliant": True,
                "wcag_status": "passed",
                "theta_score": 0.88,
                "exit_action": "ship_with_confidence",
                "certification_status": "certified",
            }
        ),
        encoding="utf-8",
    )

    result = _run_semantik_v2_conversion("doc.pdf", str(out), reuse_conversion=True)

    assert result.get("reused_conversion") is True
    assert result["success"] is True
    assert record == [], "subprocess must not run on the reuse path"


# ---------------------------------------------------------------------------
# (g) bridge dict → build_chapters_ir / from_bridge_json coercion proof.
# ---------------------------------------------------------------------------


def test_g_from_bridge_json_coerces_plain_dict():
    from lib.semantik.cascade_ir import build_chapters_ir, from_bridge_json

    bridge = _bridge_json()
    via_helper = from_bridge_json(bridge)
    via_build = build_chapters_ir(bridge)
    assert [c.title for c in via_helper] == [c.title for c in via_build]
    assert len(via_helper) == 1
    assert via_helper[0].title == "Chapter 1: Foundations"
    # The headingless body block carried its first_raw_block_index verbatim.
    anchors = {b.raw_block_index for c in via_helper for b in c.blocks}
    assert 1 in anchors

    with pytest.raises(TypeError):
        from_bridge_json(["not", "a", "mapping"])  # type: ignore[arg-type]

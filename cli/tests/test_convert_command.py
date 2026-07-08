"""``ed4all convert`` CLI tests (B1 — the accessible-HTML remediation slice).

Synthetic fixtures only (no corpus names / slugs / paths). The SemantiK cascade
seam and the vendor-ingest seam are monkeypatched for the CLI-behavior tests
(arg validation, detection dispatch, output handling, exit codes); a single
end-to-end test drives the REAL vendor-ingest path over a tiny synthetic HTML
directory (no GPU, no SemantiK runtime venv).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from click.testing import CliRunner

import MCP.tools.pipeline_tools as pt
from cli.main import cli


def _ok_result(html_path: Path) -> Dict[str, Any]:
    return {
        "success": True,
        "output_path": str(html_path),
        "html_path": str(html_path),
        "html_length": 42,
        "method": "stub",
    }


def _fail_result(html_path: Path, reason: str) -> Dict[str, Any]:
    return {
        "success": False,
        "error": reason,
        "output_path": str(html_path),
        "html_path": str(html_path),
        "method": "stub",
    }


def test_command_registered():
    result = CliRunner().invoke(cli, ["convert", "--help"])
    assert result.exit_code == 0
    assert "accessible html" in result.output.lower()


def test_unknown_input_fails_closed(tmp_path: Path):
    junk = tmp_path / "note.txt"
    junk.write_text("not a corpus", encoding="utf-8")
    out = tmp_path / "out"
    result = CliRunner().invoke(cli, ["convert", str(junk), "--output", str(out)])
    assert result.exit_code == 1
    assert "unrecognized input" in result.output.lower()


def test_missing_input_errors(tmp_path: Path):
    result = CliRunner().invoke(
        cli, ["convert", str(tmp_path / "nope.pdf"), "--output", str(tmp_path / "o")]
    )
    # click.Path(exists=True) rejects a nonexistent input before dispatch.
    assert result.exit_code != 0


def test_single_pdf_dispatches_to_cascade(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "chapter.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    out = tmp_path / "out"

    calls: List[Tuple[str, str]] = []

    def _fake_semantik(pdf_path, output_path, **kwargs):
        calls.append((pdf_path, output_path))
        return _ok_result(Path(output_path))

    monkeypatch.setattr(pt, "_run_semantik_v2_conversion", _fake_semantik)
    # A cascade-only test must never touch the vendor seam.
    monkeypatch.setattr(
        pt,
        "_run_vendor_ingest_conversion",
        lambda *a, **k: pytest.fail("vendor seam called for a PDF input"),
    )

    result = CliRunner().invoke(cli, ["convert", str(pdf), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0][0] == str(pdf)
    assert calls[0][1].endswith("chapter_accessible.html")
    assert "converted 1/1" in result.output.lower()


def test_pdf_directory_converts_each_pdf(tmp_path: Path, monkeypatch):
    src = tmp_path / "pdfs"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"%PDF a")
    (src / "b.pdf").write_bytes(b"%PDF b")
    out = tmp_path / "out"

    seen: List[str] = []

    def _fake_semantik(pdf_path, output_path, **kwargs):
        seen.append(Path(pdf_path).name)
        return _ok_result(Path(output_path))

    monkeypatch.setattr(pt, "_run_semantik_v2_conversion", _fake_semantik)

    result = CliRunner().invoke(cli, ["convert", str(src), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert sorted(seen) == ["a.pdf", "b.pdf"]
    assert "converted 2/2" in result.output.lower()


def test_partial_failure_exit_code_2(tmp_path: Path, monkeypatch):
    src = tmp_path / "pdfs"
    src.mkdir()
    (src / "good.pdf").write_bytes(b"%PDF g")
    (src / "bad.pdf").write_bytes(b"%PDF b")
    out = tmp_path / "out"

    def _fake_semantik(pdf_path, output_path, **kwargs):
        if Path(pdf_path).name == "bad.pdf":
            return _fail_result(Path(output_path), "cascade ran in mock mode")
        return _ok_result(Path(output_path))

    monkeypatch.setattr(pt, "_run_semantik_v2_conversion", _fake_semantik)

    result = CliRunner().invoke(cli, ["convert", str(src), "--output", str(out)])
    assert result.exit_code == 2, result.output
    assert "converted 1/2" in result.output.lower()
    assert "cascade ran in mock mode" in result.output
    assert "bad.pdf" in result.output


def test_total_failure_exit_code_1(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF x")
    out = tmp_path / "out"

    monkeypatch.setattr(
        pt,
        "_run_semantik_v2_conversion",
        lambda p, o, **k: _fail_result(Path(o), "unprovisioned cascade"),
    )

    result = CliRunner().invoke(cli, ["convert", str(pdf), "--output", str(out)])
    assert result.exit_code == 1, result.output
    assert "unprovisioned cascade" in result.output


def test_single_html_file_dispatches_to_vendor(tmp_path: Path, monkeypatch):
    page = tmp_path / "manual.html"
    page.write_text("<h1>Manual</h1><p>Body.</p>", encoding="utf-8")
    out = tmp_path / "out"

    calls: List[Tuple[str, str]] = []

    def _fake_vendor(path, output_path, **kwargs):
        calls.append((path, output_path))
        return _ok_result(Path(output_path))

    monkeypatch.setattr(pt, "_run_vendor_ingest_conversion", _fake_vendor)
    monkeypatch.setattr(
        pt,
        "_run_semantik_v2_conversion",
        lambda *a, **k: pytest.fail("cascade called for a vendor HTML input"),
    )

    result = CliRunner().invoke(cli, ["convert", str(page), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0][0] == str(page)
    assert calls[0][1].endswith("manual_accessible.html")


def test_reuse_flag_threaded_through(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "y.pdf"
    pdf.write_bytes(b"%PDF y")
    out = tmp_path / "out"

    captured: Dict[str, Any] = {}

    def _fake_semantik(pdf_path, output_path, **kwargs):
        captured.update(kwargs)
        return _ok_result(Path(output_path))

    monkeypatch.setattr(pt, "_run_semantik_v2_conversion", _fake_semantik)

    result = CliRunner().invoke(
        cli,
        ["convert", str(pdf), "--output", str(out), "--reuse-conversion"],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("reuse_conversion") is True


def test_vendor_ingest_end_to_end_real(tmp_path: Path):
    """Drive the REAL vendor-ingest seam over a tiny synthetic HTML dir."""
    src = tmp_path / "pages"
    src.mkdir()
    (src / "01-intro.html").write_text(
        "<html><body><h1>Introduction</h1>"
        "<p>This is a short accessible paragraph.</p></body></html>",
        encoding="utf-8",
    )
    (src / "02-body.html").write_text(
        "<html><body><h2>Details</h2>"
        "<p>A second page of readable prose.</p></body></html>",
        encoding="utf-8",
    )
    out = tmp_path / "out"

    result = CliRunner().invoke(cli, ["convert", str(src), "--output", str(out)])
    assert result.exit_code == 0, result.output

    produced = list(out.glob("*_accessible.html"))
    assert len(produced) == 1
    assert produced[0].name == "pages_accessible.html"
    html = produced[0].read_text(encoding="utf-8")
    assert "Introduction" in html
    # Sidecars land as siblings (the standard conversion contract).
    assert (out / "pages_accessible_synthesized.json").exists()
    assert (out / "pages_accessible.quality.json").exists()

"""``ed4all import-docs`` CLI tests.

Synthetic Markdown fixtures only (no corpus names / slugs / paths). Verifies the
command is registered, converts a docs tree, honors mkdocs-nav ordering, and
records the license note / provenance tag in the manifest.
"""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from cli.main import cli


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_command_registered():
    result = CliRunner().invoke(cli, ["import-docs", "--help"])
    assert result.exit_code == 0
    assert "clean accessible HTML" in result.output.lower() or "docs tree" in result.output.lower()


def test_import_docs_end_to_end(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write(src, "intro.md", "# Intro\n\nWelcome.\n")
    _write(src, "topics/detail.md", "# Detail\n\nDeep dive.\n")

    result = CliRunner().invoke(
        cli,
        [
            "import-docs",
            str(src),
            "--output",
            str(out),
            "--source-name",
            "Sample Docs",
            "--license-note",
            "CC BY-SA 4.0",
            "--provenance-tag",
            "sample-tag",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Imported 2 document(s)" in result.output

    manifest = json.loads((out / "import_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_name"] == "Sample Docs"
    assert manifest["license_note"] == "CC BY-SA 4.0"
    assert manifest["provenance_tag"] == "sample-tag"
    assert manifest["document_count"] == 2
    # Plain .html pages (vendor-ingest picks these up, not *_accessible.html).
    html_files = list(out.glob("*.html"))
    assert len(html_files) == 2
    assert not any(f.name.endswith("_accessible.html") for f in html_files)


def test_import_docs_reports_nav_ordering(tmp_path: Path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    _write(src, "docs/first.md", "# First\n\n1.\n")
    _write(src, "docs/second.md", "# Second\n\n2.\n")
    _write(src, "mkdocs.yml", "nav:\n  - First: first.md\n  - Second: second.md\n")

    result = CliRunner().invoke(
        cli, ["import-docs", str(src), "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "mkdocs-nav" in result.output
    manifest = json.loads((out / "import_manifest.json").read_text(encoding="utf-8"))
    assert [d["title"] for d in manifest["documents"]] == ["First", "Second"]


def test_import_docs_empty_dir_exits_nonzero(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli, ["import-docs", str(src), "--output", str(out)]
    )
    assert result.exit_code == 1
    assert "nothing written" in result.output.lower()

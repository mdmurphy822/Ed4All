"""CliRunner tests for ``libv2 remove`` confirm-gating (Marketable-v1 D5).

Synthetic ``tmp_path`` courses only — no real LibV2 dir is touched. Covers the
``--yes`` flag, interactive confirm (accept + decline), the missing-course /
escaping-slug refusals, and catalog-entry cleanup through the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from LibV2.tools.libv2.cli import main as libv2_main


def _make_course(root: Path, slug: str) -> Path:
    cdir = root / "courses" / slug
    (cdir / "imscc_chunks").mkdir(parents=True)
    (cdir / "imscc_chunks" / "chunks.jsonl").write_bytes(b"x" * 256)
    (cdir / "manifest.json").write_text(json.dumps({"slug": slug}), encoding="utf-8")
    # A catalog/ dir so get_repo_root resolves and --repo is honored.
    (root / "catalog").mkdir(parents=True, exist_ok=True)
    return cdir


def test_remove_with_yes_deletes(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_course(root, "demo-101")
    runner = CliRunner()
    result = runner.invoke(libv2_main, ["--repo", str(root), "remove", "demo-101", "--yes"])
    assert result.exit_code == 0, result.output
    assert not cdir.exists()
    assert "Removed course: demo-101" in result.output


def test_remove_interactive_confirm_accepts(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_course(root, "demo-101")
    runner = CliRunner()
    result = runner.invoke(libv2_main, ["--repo", str(root), "remove", "demo-101"], input="y\n")
    assert result.exit_code == 0, result.output
    assert not cdir.exists()


def test_remove_interactive_confirm_declines(tmp_path):
    root = tmp_path / "libv2"
    cdir = _make_course(root, "demo-101")
    runner = CliRunner()
    result = runner.invoke(libv2_main, ["--repo", str(root), "remove", "demo-101"], input="n\n")
    assert result.exit_code == 1
    assert cdir.exists(), "declining must NOT delete the course"
    assert "Aborted" in result.output


def test_remove_prints_summary(tmp_path):
    root = tmp_path / "libv2"
    _make_course(root, "demo-101")
    runner = CliRunner()
    result = runner.invoke(libv2_main, ["--repo", str(root), "remove", "demo-101"], input="n\n")
    assert "Disk size:" in result.output
    assert "demo-101" in result.output
    assert "no undo" in result.output.lower()


def test_remove_missing_course_errors(tmp_path):
    root = tmp_path / "libv2"
    (root / "courses").mkdir(parents=True)
    (root / "catalog").mkdir(parents=True)
    runner = CliRunner()
    result = runner.invoke(libv2_main, ["--repo", str(root), "remove", "nope-999", "--yes"])
    assert result.exit_code == 1
    assert "course_not_found" in result.output


def test_remove_escaping_slug_refused(tmp_path):
    root = tmp_path / "libv2"
    _make_course(root, "demo-101")
    runner = CliRunner()
    result = runner.invoke(libv2_main, ["--repo", str(root), "remove", "../evil", "--yes"])
    assert result.exit_code == 1
    assert (root / "courses" / "demo-101").exists()


def test_remove_cleans_catalog_entry(tmp_path):
    root = tmp_path / "libv2"
    _make_course(root, "demo-101")
    (root / "catalog" / "master_catalog.json").write_text(
        json.dumps({"total_courses": 1, "courses": [{"slug": "demo-101", "title": "A"}]}),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(libv2_main, ["--repo", str(root), "remove", "demo-101", "--yes"])
    assert result.exit_code == 0, result.output
    master = json.loads((root / "catalog" / "master_catalog.json").read_text())
    assert master["courses"] == []
    assert "Pruned catalog entries" in result.output

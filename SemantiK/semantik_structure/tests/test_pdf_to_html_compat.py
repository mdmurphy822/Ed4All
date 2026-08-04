"""Verify the relocated PDF-to-HTML command and compatibility facade."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from SemantiK.scripts import pdf_to_html as legacy
from SemantiK.scripts.ops import pdf_to_html as canonical

SEMANTIK_ROOT = Path(__file__).resolve().parents[2]


def test_canonical_command_resolves_semantik_root() -> None:
    assert canonical.REPO_ROOT == SEMANTIK_ROOT


def test_legacy_module_reuses_canonical_main() -> None:
    assert legacy.main is canonical.main


def _help_for(relative_path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SEMANTIK_ROOT / relative_path), "--help"],
        cwd=SEMANTIK_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_canonical_direct_path_help() -> None:
    result = _help_for("scripts/ops/pdf_to_html.py")
    assert result.returncode == 0, result.stderr
    assert "--invalidate-extract-cache" in result.stdout


def test_legacy_direct_path_help() -> None:
    result = _help_for("scripts/pdf_to_html.py")
    assert result.returncode == 0, result.stderr
    assert "--invalidate-extract-cache" in result.stdout

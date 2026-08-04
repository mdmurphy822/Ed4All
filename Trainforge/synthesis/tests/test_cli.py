"""Command-line contracts for training-pair synthesis."""

from __future__ import annotations

import subprocess
import sys

def test_canonical_module_cli_exposes_synthesis_options() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "Trainforge.synthesis.synthesize_training", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--corpus" in result.stdout
    assert "--slug" in result.stdout
    assert "--course-code" in result.stdout
    assert "--provider" in result.stdout
    assert "--max-concurrent" in result.stdout

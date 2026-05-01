"""Wave 137d-3: tests for the show_form_data_coverage operator CLI.

Three tests pin the contract:

1. ``test_show_latest_table_format`` — populated JSONL with one row =>
   table render contains the row's CURIE-ish identifiers.
2. ``test_show_all_rows_in_json_format`` — ``--all --format json`` emits
   a JSON array containing every row.
3. ``test_show_exits_2_when_checkpoint_absent`` — missing checkpoint
   path => exit 2 with stderr message.
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.scripts import show_form_data_coverage as cli  # noqa: E402


def _write_checkpoint(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(r) for r in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def _make_row(model_id: str, **fields: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "timestamp": "2026-05-01T12:00:00Z",
        "model_id": model_id,
        "course_slug": "rdf-shacl-551-2",
        "family": "rdf_shacl",
        "manifest_coverage_pct": 0.50,
        "complete_count": 5,
        "degraded_count": 5,
        "family_coverage_map": {
            "cardinality": {
                "complete": 1,
                "total": 2,
                "status": "partial",
                "curies": ["sh:minCount", "sh:maxCount"],
            }
        },
        "promotion_decision": "passed",
        "promotion_block_reasons": [],
    }
    base.update(fields)
    return base


# ----------------------------------------------------------------------
# 1. Latest table format
# ----------------------------------------------------------------------


def test_show_latest_table_format(tmp_path: Path) -> None:
    checkpoint = tmp_path / "form_data_coverage_checkpoint.jsonl"
    _write_checkpoint(checkpoint, [
        _make_row("test-v1"),
        _make_row("test-v2", manifest_coverage_pct=0.75, complete_count=8),
    ])

    out = io.StringIO()
    with redirect_stdout(out):
        rc = cli.main([
            "--course-code", "rdf-shacl-551-2",
            "--checkpoint-path", str(checkpoint),
        ])
    assert rc == 0
    rendered = out.getvalue()
    # Default: latest row only.
    assert "FORM_DATA COVERAGE CHECKPOINT" in rendered
    assert "test-v2" in rendered
    # Latest row's coverage_pct is 0.75 => "75.0%".
    assert "75.0%" in rendered
    # Family map line surfaces.
    assert "cardinality" in rendered
    assert "rdf_shacl" in rendered
    # Older row's model_id MUST NOT appear (default is latest-only).
    assert "test-v1" not in rendered


# ----------------------------------------------------------------------
# 2. --all --format json => JSON array containing every row.
# ----------------------------------------------------------------------


def test_show_all_rows_in_json_format(tmp_path: Path) -> None:
    checkpoint = tmp_path / "form_data_coverage_checkpoint.jsonl"
    _write_checkpoint(checkpoint, [
        _make_row("test-v1", manifest_coverage_pct=0.40, complete_count=4),
        _make_row("test-v2", manifest_coverage_pct=0.60, complete_count=6),
        _make_row("test-v3", manifest_coverage_pct=0.80, complete_count=8),
    ])

    out = io.StringIO()
    with redirect_stdout(out):
        rc = cli.main([
            "--course-code", "rdf-shacl-551-2",
            "--checkpoint-path", str(checkpoint),
            "--all",
            "--format", "json",
        ])
    assert rc == 0
    parsed = json.loads(out.getvalue())
    assert isinstance(parsed, list)
    assert len(parsed) == 3
    model_ids = [r["model_id"] for r in parsed]
    assert model_ids == ["test-v1", "test-v2", "test-v3"]


# ----------------------------------------------------------------------
# 3. Missing checkpoint => exit 2.
# ----------------------------------------------------------------------


def test_show_exits_2_when_checkpoint_absent(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_file.jsonl"
    err = io.StringIO()
    with redirect_stderr(err):
        rc = cli.main([
            "--course-code", "rdf-shacl-551-2",
            "--checkpoint-path", str(missing),
        ])
    assert rc == 2
    err_text = err.getvalue()
    assert "checkpoint" in err_text.lower()
    assert str(missing) in err_text

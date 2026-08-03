"""Build #23 — subclass pilot / corpus-apply driver (dry-run, mocked).

Exercises the pilot's real responsibilities — IR discovery, mock-client
selection, the render + subclass pass, per-chapter + corpus report aggregation,
distribution merge, bucket-collapse detection, and --apply HTML write. The
well-tested IR→chapters bridge (``build_chapters_ir``, owned by
``scripts/ops/semantik_rerender``) is stubbed to a chapter carrying a worked-example
composite unit so the render seam has a real unit to subclass. No GPU / network.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.semantik.adapter import _AdapterBlock, _AdapterChapter
from scripts.archive import subclass_pilot as pilot


def _worked_example_chapter():
    def _opener(text, idx, role):
        return _AdapterBlock(
            html="", region_kind="heading", raw_block_index=idx, heading_level=4,
            raw_text=text, heading_text=text, block_role=role,
        )

    def _para(text, idx, pages):
        return _AdapterBlock(
            html=f"<p>{text}</p>", region_kind="paragraph", raw_block_index=idx,
            raw_text=text, heading_text=None, pages=pages,
        )

    return _AdapterChapter(
        title="Roots",
        blocks=[
            _opener("Example 9.1", 0, "worked_example"),
            _para("Simplify sqrt 36.", 1, [3]),
            _opener("Solution", 2, "solution"),
            _para("Since 6^2 = 36.", 3, [4]),
        ],
    )


@pytest.fixture(autouse=True)
def _stub_ir_bridge(monkeypatch):
    """Stub the IR→chapters bridge to a unit-bearing chapter (render stays real)."""
    monkeypatch.setattr(
        pilot, "build_chapters_ir", lambda ir_doc: [_worked_example_chapter()]
    )


def _ir_sidecar(tmp_path: Path, stem: str) -> Path:
    p = tmp_path / f"{stem}_accessible.cascade_ir.json"
    p.write_text(json.dumps({"pdf": stem}), encoding="utf-8")
    return p


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_pilot_dry_run_report_shape(tmp_path):
    _ir_sidecar(tmp_path, "book-ch06")
    _ir_sidecar(tmp_path, "book-ch09")
    out = tmp_path / "report.json"
    rc = pilot.main([
        "--input-dir", str(tmp_path),
        "--chapters", "ch06,ch09",
        "--dry-run",
        "--json-out", str(out),
    ])
    assert rc == 0
    report = _load_report(out)
    assert report["mode"] == "dry-run-report"
    assert report["dry_run"] is True
    assert report["chapters_processed"] == 2
    for key in (
        "status_totals", "corpus_label_distribution",
        "corpus_bucket_collapse", "new_labels", "fold_events", "per_chapter",
    ):
        assert key in report
    # A worked_example unit was subclassed in each chapter.
    assert "worked_example" in report["corpus_label_distribution"]
    assert report["status_totals"].get("assigned", 0) >= 2


def test_pilot_apply_writes_annotated_html(tmp_path):
    _ir_sidecar(tmp_path, "book-ch06")
    out_dir = tmp_path / "applied"
    rc = pilot.main([
        "--input-dir", str(tmp_path),
        "--dry-run", "--apply", "--output-dir", str(out_dir),
    ])
    assert rc == 0
    html_files = list(out_dir.glob("*_accessible.html"))
    assert html_files
    html = html_files[0].read_text(encoding="utf-8")
    assert "data-semantik-subclass=" in html


def test_pilot_chapters_filter_selects_by_token(tmp_path):
    _ir_sidecar(tmp_path, "book-ch06")
    _ir_sidecar(tmp_path, "book-ch09")
    _ir_sidecar(tmp_path, "book-ch12")
    out = tmp_path / "r.json"
    pilot.main([
        "--input-dir", str(tmp_path), "--chapters", "ch06", "--dry-run",
        "--json-out", str(out),
    ])
    assert _load_report(out)["chapters_processed"] == 1


def test_pilot_bucket_collapse_detected_via_helper():
    dist = {"worked_example": {"drill": 15}}
    flagged = pilot.detect_bucket_collapse(dist)
    assert flagged and flagged[0]["unit_type"] == "worked_example"


def test_pilot_no_irs_returns_error(tmp_path):
    rc = pilot.main(["--input-dir", str(tmp_path), "--dry-run"])
    assert rc == 2

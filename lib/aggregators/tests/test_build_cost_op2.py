"""Roadmap OP2 — :class:`BuildCostAggregator` tests.

Coverage:

1. Per-phase wall-clock from checkpoints (started_at / completed_at delta).
2. Phases sorted by phase_index; totals roll up measured phases only.
3. GPU-residency section omitted when vram_trajectory.jsonl is absent.
4. GPU-residency ts→phase-window join when present.
5. LLM-usage section omitted when llm_usage.jsonl is absent.
6. LLM-usage tally (global + per-provider + per-model) when present.
7. Schema validation of the emitted report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from lib.aggregators.build_cost import BuildCostAggregator, SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPO_ROOT / "schemas" / "aggregators" / "build_cost.schema.json"


def _write_checkpoint(
    run_dir: Path,
    phase_name: str,
    phase_index: int,
    started_at: str,
    completed_at: str | None,
    status: str = "completed",
) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / f"{phase_name}_checkpoint.json").write_text(
        json.dumps(
            {
                "phase_name": phase_name,
                "phase_index": phase_index,
                "status": status,
                "started_at": started_at,
                "completed_at": completed_at,
            }
        ),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_phase_wall_clock_and_totals(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_checkpoint(
        run_dir, "staging", 1,
        "2026-07-07T10:00:00", "2026-07-07T10:00:30",
    )
    _write_checkpoint(
        run_dir, "course_planning", 5,
        "2026-07-07T10:01:00", "2026-07-07T10:03:00",
    )
    # In-progress phase (no completed_at) → wall_clock null, not measured.
    _write_checkpoint(
        run_dir, "content_generation", 6,
        "2026-07-07T10:03:00", None, status="started",
    )
    report = BuildCostAggregator(run_dir=run_dir, run_id="R").build()

    phases = report["phases"]
    # Sorted by phase_index.
    assert [p["phase"] for p in phases] == [
        "staging", "course_planning", "content_generation"
    ]
    assert phases[0]["wall_clock_seconds"] == 30.0
    assert phases[1]["wall_clock_seconds"] == 120.0
    assert phases[2]["wall_clock_seconds"] is None

    totals = report["totals"]
    assert totals["phase_count"] == 3
    assert totals["measured_phase_count"] == 2
    assert totals["total_wall_clock_seconds"] == 150.0

    # No trajectory / usage files → sections omitted.
    assert "gpu_residency" not in report
    assert "llm_usage" not in report


def test_gpu_residency_join(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_checkpoint(
        run_dir, "content_generation", 6,
        "2026-07-07T10:00:00", "2026-07-07T10:05:00",
    )
    _write_jsonl(
        run_dir / "vram_trajectory.jsonl",
        [
            {
                "ts": "2026-07-07T10:01:00+00:00",
                "phase": "content_generation",
                "resident_models": [{"name": "m", "vram_mib": 4000}],
            },
            {
                "ts": "2026-07-07T10:04:00+00:00",
                "phase": "content_generation",
                "resident_models": [{"name": "m", "vram_mib": 5000}],
            },
        ],
    )
    report = BuildCostAggregator(run_dir=run_dir, run_id="R").build()
    gpu = report["gpu_residency"]
    assert gpu["total_samples"] == 2
    assert gpu["peak_resident_mib"] == 5000
    per_phase = {p["phase"]: p for p in gpu["per_phase"]}
    cg = per_phase["content_generation"]
    assert cg["sample_count"] == 2
    assert cg["resident_sample_count"] == 2
    assert cg["residency_span_seconds"] == 180.0
    assert cg["peak_resident_mib"] == 5000


def test_llm_usage_tally(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_checkpoint(
        run_dir, "staging", 1,
        "2026-07-07T10:00:00", "2026-07-07T10:00:05",
    )
    _write_jsonl(
        run_dir / "llm_usage.jsonl",
        [
            {
                "ts": "2026-07-07T10:00:01+00:00",
                "provider": "local",
                "model": "qwen",
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "duration_ms": 1500.0,
            },
            {
                "ts": "2026-07-07T10:00:02+00:00",
                "provider": "local",
                "model": "qwen",
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "duration_ms": 500.0,
            },
            {
                "ts": "2026-07-07T10:00:03+00:00",
                "provider": "together",
                "model": "llama",
                "prompt_tokens": 200,
                "completion_tokens": 40,
                "duration_ms": 800.0,
            },
        ],
    )
    report = BuildCostAggregator(run_dir=run_dir, run_id="R").build()
    llm = report["llm_usage"]
    assert llm["total_calls"] == 3
    assert llm["total_prompt_tokens"] == 350
    assert llm["total_completion_tokens"] == 70
    assert llm["total_tokens"] == 420
    assert llm["total_duration_ms"] == 2800.0
    assert llm["by_provider"]["local"]["calls"] == 2
    assert llm["by_provider"]["local"]["prompt_tokens"] == 150
    assert llm["by_provider"]["together"]["calls"] == 1
    assert llm["by_model"]["qwen"]["completion_tokens"] == 30


def test_llm_usage_ttft_percentiles(tmp_path: Path):
    """Task #10 — rows carrying ttft_ms surface p50/p95 + sample count."""
    run_dir = tmp_path / "run"
    _write_checkpoint(
        run_dir, "staging", 1,
        "2026-07-07T10:00:00", "2026-07-07T10:00:05",
    )
    _write_jsonl(
        run_dir / "llm_usage.jsonl",
        [
            {
                "provider": "local", "model": "qwen",
                "prompt_tokens": 10, "completion_tokens": 2,
                "duration_ms": 100.0, "ttft_ms": 200.0,
            },
            {
                "provider": "local", "model": "qwen",
                "prompt_tokens": 10, "completion_tokens": 2,
                "duration_ms": 100.0, "ttft_ms": 400.0,
            },
            {
                "provider": "local", "model": "qwen",
                "prompt_tokens": 10, "completion_tokens": 2,
                "duration_ms": 100.0, "ttft_ms": 600.0,
            },
        ],
    )
    llm = BuildCostAggregator(run_dir=run_dir, run_id="R").build()["llm_usage"]
    assert llm["ttft_sample_count"] == 3
    # p50 of [200,400,600]: rank (3-1)*0.5=1 → 400.
    # p95: rank (3-1)*0.95=1.9 → 400 + (600-400)*0.9 = 580.
    assert llm["ttft_ms_p50"] == 400.0
    assert llm["ttft_ms_p95"] == 580.0


def test_llm_usage_no_ttft_omits_percentiles(tmp_path: Path):
    """Rows without ttft_ms → the TTFT block is omitted (byte-identical legacy
    llm_usage section)."""
    run_dir = tmp_path / "run"
    _write_checkpoint(
        run_dir, "staging", 1,
        "2026-07-07T10:00:00", "2026-07-07T10:00:05",
    )
    _write_jsonl(
        run_dir / "llm_usage.jsonl",
        [
            {
                "provider": "local", "model": "qwen",
                "prompt_tokens": 10, "completion_tokens": 2,
                "duration_ms": 100.0,
            },
        ],
    )
    llm = BuildCostAggregator(run_dir=run_dir, run_id="R").build()["llm_usage"]
    assert "ttft_sample_count" not in llm
    assert "ttft_ms_p50" not in llm
    assert "ttft_ms_p95" not in llm


def test_malformed_jsonl_lines_skipped(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_checkpoint(
        run_dir, "staging", 1,
        "2026-07-07T10:00:00", "2026-07-07T10:00:05",
    )
    usage = run_dir / "llm_usage.jsonl"
    usage.parent.mkdir(parents=True, exist_ok=True)
    usage.write_text(
        '{"provider":"local","model":"m","prompt_tokens":10,'
        '"completion_tokens":2,"duration_ms":1.0}\n'
        "not json at all\n"
        "\n",
        encoding="utf-8",
    )
    report = BuildCostAggregator(run_dir=run_dir, run_id="R").build()
    assert report["llm_usage"]["total_calls"] == 1


def test_missing_run_dir_is_safe(tmp_path: Path):
    report = BuildCostAggregator(
        run_dir=tmp_path / "does_not_exist", run_id="R"
    ).build()
    assert report["phases"] == []
    assert report["totals"]["total_wall_clock_seconds"] == 0
    assert "gpu_residency" not in report
    assert "llm_usage" not in report


def test_emitted_report_validates_against_schema(tmp_path: Path):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    run_dir = tmp_path / "run"
    _write_checkpoint(
        run_dir, "staging", 1,
        "2026-07-07T10:00:00", "2026-07-07T10:00:30",
    )
    _write_jsonl(
        run_dir / "vram_trajectory.jsonl",
        [
            {
                "ts": "2026-07-07T10:00:10+00:00",
                "phase": "staging",
                "resident_models": [{"name": "m", "vram_mib": 4000}],
            }
        ],
    )
    _write_jsonl(
        run_dir / "llm_usage.jsonl",
        [
            {
                "ts": "2026-07-07T10:00:11+00:00",
                "provider": "local",
                "model": "qwen",
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "duration_ms": 5.0,
            }
        ],
    )
    out = tmp_path / "build_cost_report.json"
    agg = BuildCostAggregator(run_dir=run_dir, course_code="C", run_id="R")
    agg.write(out)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema_version"] == SCHEMA_VERSION
    jsonschema.validate(report, schema)

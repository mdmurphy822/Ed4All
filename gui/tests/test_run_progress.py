"""Tests for the stage-tracker progress endpoint + service.

``gui.services.progress_service`` merges the run's ``config/workflows.yaml``
phase plan with its orchestrator workflow state, checkpoint wall-clocks, and a
bounded tail of the OP2 ``llm_usage.jsonl`` tap. These tests pin:

- the sliding-window tok/s math (pure ``usage_window_stats``),
- the happy-path payload (states, groups, wall-clocks, usage totals),
- ``stats`` nulls for a run with no usage rows yet,
- a typed 404 for an unknown run id,
- skipped-phase mapping (``_skipped`` markers + params-driven prediction),
- a workflow-agnostic phase list (``course_generation`` renders ITS OWN config
  phases, never the 21-phase textbook list),
- real in-phase unit progress (``stats.phase_units``) counted from the
  pipeline's per-unit resume-checkpoint sidecars: present → counts and ticks
  on append; absent → the field is OMITTED (honesty contract, no fabricated
  counts or estimated totals); corrupt/partial rows never crash the
  newline-based counter,
- the no-evidence → pending contract for env-conditional phases (the serving
  process env is never a skip prediction),
- the consolidated section grouping (one generation header spanning the whole
  authoring slice; every section header exactly once) and the rail SECTION
  order for the one sequenced build→training pipeline (conversion → planning →
  generation → validation → packaging → archive → training → finalization),
- resume RE-STAMP wall-clocks (identical started/completed timestamps)
  suppressed, never "0s", while a genuinely-fast phase keeps its real
  sub-second span ("<1s", never blank),
- the in-progress phase node carries a live ``elapsed_s`` (now − start),
- the live output tail (``output_tail`` + ``GET /api/runs/{id}/output-tail``):
  bounded seek-from-end read, HTML-strip + truncation marker, corrupt/partial
  rows skipped, absent sidecar → honest ``rows: []``.

State isolated via ``state_dir``; the endpoint tests need fastapi (opt-in
``gui`` extra) and are skipped without it. No network: the seat registry env is
cleared so no probe is attempted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from gui import shared_state
from gui.services import progress_service

# The rail's canonical SECTION order for the one sequenced build→training
# pipeline. The rail buckets phases by `group` in FIRST-OCCURRENCE order
# (stage-rail.js::renderRail), so this list is the order the derived groups
# must appear in — and a group must never appear twice, which would mean a
# later phase rendering inside an earlier section.
CANONICAL_GROUP_ORDER = [
    "conversion",
    "planning",
    "generation",
    "validation",
    "packaging",
    "archive",
    "training",
    "finalization",
]


def _rail_section_order(phases: List[Dict[str, Any]]) -> List[str]:
    """Group sections in the order stage-rail.js::renderRail would render them.

    Mirrors that function's bucketing EXACTLY: first occurrence of a `group`
    creates its section; later phases of the same group join the existing
    bucket wherever it already sits.
    """
    order: List[str] = []
    for p in phases:
        g = p.get("group") or "other"
        if g not in order:
            order.append(g)
    return order


@pytest.fixture(autouse=True)
def _no_seat_probes(monkeypatch: pytest.MonkeyPatch):
    """No registered seats → the service never opens a socket in tests."""
    monkeypatch.delenv("ED4ALL_SEAT_BASE_URLS", raising=False)
    # The two-pass env must not leak into skip prediction from the dev shell.
    monkeypatch.delenv("COURSEFORGE_TWO_PASS", raising=False)
    # Fresh usage accumulators + sidecar-count + vram caches per test
    # (path-keyed module caches).
    progress_service._USAGE_ACCUMULATORS.clear()
    progress_service._UNIT_COUNT_CACHE.clear()
    progress_service._VRAM_CACHE.clear()


def _seed_run(
    state_dir: Path,
    *,
    run_id: str = "GUI-prog-0001",
    workflow_id: str = "WF-20260101-prog0001",
    workflow: str = "textbook_to_course",
    gui_status: str = "running",
    wf_status: str = "RUNNING",
    orch_run_id: str = "TTC_prog_20260101_000000",  # synthetic run id, slug-guard: allow
    params: Optional[Dict[str, Any]] = None,
    phase_outputs: Optional[Dict[str, Any]] = None,
    failed_phase: Optional[str] = None,
    failure_reason: Optional[str] = None,
    usage_rows: Optional[List[Dict[str, Any]]] = None,
    checkpoints: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Register a GUI run + write its workflow state / run-dir artifacts."""
    shared_state.register_run(
        {
            "run_id": run_id,
            "kind": "pipeline",
            "workflow": workflow,
            "workflow_id": workflow_id,
            "course_name": "PHYS_101",
            "status": gui_status,
            "started_at": "2026-01-01T00:00:00",
        }
    )
    wf_doc: Dict[str, Any] = {
        "id": workflow_id,
        "type": workflow,
        "status": wf_status,
        "started_at": "2026-01-01T00:00:00",
        "params": {"run_id": orch_run_id, **(params or {})},
        "phase_outputs": phase_outputs or {},
    }
    if failed_phase:
        wf_doc["failed_phase"] = failed_phase
        wf_doc["failure_reason"] = failure_reason or "gate failure"
    wf_dir = state_dir / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{workflow_id}.json").write_text(json.dumps(wf_doc), encoding="utf-8")

    run_dir = state_dir / "runs" / orch_run_id
    if usage_rows is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "llm_usage.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in usage_rows), encoding="utf-8"
        )
    for phase, ckpt in (checkpoints or {}).items():
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / f"{phase}_checkpoint.json").write_text(
            json.dumps({"phase_name": phase, **ckpt}), encoding="utf-8"
        )
    return run_id


def _phase(payload: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(p for p in payload["phases"] if p["name"] == name)


# ------------------------------------------------- usage window (pure math)


def test_usage_window_stats_tok_s_math():
    """The three window figures, each with its own denominator:

    - tok_s (AGGREGATE): window completion tokens over the WALL span
      (first in-window row ts → now) — the seat's overall output rate.
    - per_stream_tok_s: tokens over the calls' own summed generation seconds
      (duration_ms) — what one request experiences under concurrency.
    - streams: sum(duration_ms)/1000 over the wall span (Little's law).
    """
    now = 1_000_000.0

    def _iso(epoch: float) -> str:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

    rows = [
        # Outside the 180s window — must be excluded from every figure.
        {"ts": _iso(now - 3600), "completion_tokens": 99999, "duration_ms": 1000.0},
        # In-window: 750 tokens; wall span (now-100 → now) = 100s;
        # generation seconds 10 + 10 = 20.
        {"ts": _iso(now - 100), "completion_tokens": 500, "duration_ms": 10000.0,
         "ttft_ms": 100.0},
        {"ts": _iso(now - 10), "completion_tokens": 250, "duration_ms": 10000.0,
         "ttft_ms": 300.0},
    ]
    out = progress_service.usage_window_stats(rows, now_ts=now, window_s=180.0)
    # Aggregate: 750 tokens / 100s wall span — NOT the per-stream 37.5.
    assert out["tok_s"] == 7.5
    # Per-stream: 750 tokens / 20 generation seconds.
    assert out["per_stream_tok_s"] == 37.5
    # In-flight estimate: 20 generation seconds / 100s wall span.
    assert out["streams"] == 0.2
    assert out["window_calls"] == 2
    # p50 over ALL rows carrying ttft samples (100, 300) → 200.
    assert out["ttft_p50_ms"] == 200.0


def test_usage_window_stats_empty_window_is_null():
    """No rows inside the window → every figure None (never fabricated)."""
    rows = [{"ts": "2020-01-01T00:00:00+00:00", "completion_tokens": 10, "duration_ms": 1000.0}]
    out = progress_service.usage_window_stats(rows, now_ts=2_000_000_000.0)
    assert out["tok_s"] is None
    assert out["per_stream_tok_s"] is None
    assert out["streams"] is None
    assert out["ttft_p50_ms"] is None
    assert progress_service.usage_window_stats([], now_ts=1.0)["tok_s"] is None


def test_usage_window_stats_no_durations_still_aggregates():
    """Rows without duration_ms: the aggregate wall-span tok_s still computes;
    per_stream and streams are honestly None (no generation seconds)."""
    now = 1_000_000.0

    def _iso(epoch: float) -> str:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

    rows = [
        {"ts": _iso(now - 50), "completion_tokens": 100},
        {"ts": _iso(now - 10), "completion_tokens": 150},
    ]
    out = progress_service.usage_window_stats(rows, now_ts=now, window_s=180.0)
    assert out["tok_s"] == 5.0  # 250 tokens / 50s wall span
    assert out["per_stream_tok_s"] is None
    assert out["streams"] is None


# ------------------------------------------------------- service-level merge


def test_run_progress_happy_path(state_dir):
    """Done/current/pending states, groups, wall-clocks, and usage totals."""
    from datetime import datetime, timezone

    from datetime import timedelta

    now = datetime.now(timezone.utc)
    # In-window rows 100s and 10s ago: 500 tokens over a ~100s wall span
    # (aggregate ~5 tok/s), 10 generation seconds (per-stream 50 tok/s,
    # ~0.1 streams in flight).
    run_id = _seed_run(
        state_dir,
        params={"skip_training": True},
        phase_outputs={
            "semantik_conversion": {"_completed": True},
            "heading_judge": {"_completed": True},
            "staging": {"_completed": True},
        },
        checkpoints={
            "staging": {
                "status": "completed",
                "started_at": "2026-01-01T00:01:00",
                "completed_at": "2026-01-01T00:01:30",
            },
        },
        usage_rows=[
            {"ts": (now - timedelta(seconds=100)).isoformat(), "provider": "local",
             "model": "m", "prompt_tokens": 100,
             "completion_tokens": 400, "duration_ms": 8000.0, "ttft_ms": 500.0},
            {"ts": (now - timedelta(seconds=10)).isoformat(), "provider": "local",
             "model": "m", "prompt_tokens": 50,
             "completion_tokens": 100, "duration_ms": 2000.0},
        ],
    )

    payload = progress_service.run_progress(run_id)
    assert payload is not None
    assert payload["workflow"] == "textbook_to_course"
    assert payload["status"] == "running"

    # Phase list is the CONFIG plan (21 phases for textbook_to_course), ordered.
    names = [p["name"] for p in payload["phases"]]
    assert names[0] == "semantik_conversion"
    assert "packaging" in names and "finalization" in names
    assert [p["index"] for p in payload["phases"]] == sorted(
        p["index"] for p in payload["phases"]
    )

    assert _phase(payload, "semantik_conversion")["state"] == "done"
    assert _phase(payload, "staging")["state"] == "done"
    # wall-clock from the checkpoint pair: 30s.
    assert _phase(payload, "staging")["wallclock_s"] == 30.0
    # The first unresolved phase is current.
    assert payload["current_phase"] == "chunking"
    assert _phase(payload, "chunking")["state"] == "current"
    assert _phase(payload, "packaging")["state"] == "pending"
    # --skip-training prediction → training_synthesis renders skipped.
    assert _phase(payload, "training_synthesis")["state"] == "skipped"

    # Groups derive from names, not indices. inter_tier_validation belongs to
    # the consolidated "generation" section (owner design — see the grouping
    # test below).
    assert _phase(payload, "semantik_conversion")["group"] == "conversion"
    assert _phase(payload, "course_planning")["group"] == "planning"
    assert _phase(payload, "inter_tier_validation")["group"] == "generation"
    assert _phase(payload, "packaging")["group"] == "packaging"
    assert _phase(payload, "libv2_archival")["group"] == "archive"
    # finalization is genuinely LAST in the sequenced pipeline, so it owns its
    # own trailing section — folding it into "archive" (whose first member is
    # much earlier) would render it visually BEFORE the training tail.
    assert _phase(payload, "finalization")["group"] == "finalization"

    # Usage totals + windowed throughput. The band's tok_s is AGGREGATE seat
    # throughput: 500 tokens over the ~100s wall span (first row ts → the
    # service's own now, hence the small tolerance band) — NOT the per-stream
    # 50.0 the band used to show.
    stats = payload["stats"]
    assert stats["calls"] == 2
    assert stats["prompt_tokens"] == 150
    assert stats["completion_tokens"] == 500
    # Lower bound allows up to ~40s of loaded-runner lag between seeding the
    # rows and the service taking its own "now" (the span denominator grows
    # with real elapsed time; observed flaking at 4.5 under a full-suite run).
    assert 3.5 <= stats["tok_s"] <= 5.0
    # Estimated in-flight requests: 10 gen-seconds / ~100s wall span.
    assert stats["streams"] == 0.1
    assert stats["ttft_p50_ms"] == 500.0
    assert stats["seat"] is None  # no registered seats in tests
    # The old band number lives in the detail matrix, clearly named:
    # 500 tokens / 10 generation seconds.
    assert stats["detail"]["throughput"]["per_stream_tok_s"] == 50.0
    # detail.window_tok_s mirrors the band's aggregate semantics (computed a
    # few ms later, so allow one rounding step).
    assert abs(stats["detail"]["throughput"]["window_tok_s"] - stats["tok_s"]) <= 0.1


def test_run_progress_no_usage_rows_yet(state_dir):
    """A run with no llm_usage.jsonl → stats nulls / zeros, no crash."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-nousage",
        workflow_id="WF-20260101-prog0002",
        orch_run_id="TTC_prog_nousage",
        usage_rows=None,
    )
    payload = progress_service.run_progress(run_id)
    assert payload is not None
    stats = payload["stats"]
    assert stats["tok_s"] is None
    assert "streams" not in stats  # omitted, never a fabricated 0
    assert stats["ttft_p50_ms"] is None
    assert stats["calls"] == 0
    assert stats["prompt_tokens"] == 0
    assert stats["completion_tokens"] == 0
    # Backward-compatible empty shape: the page-metered keys are ALWAYS
    # present (None/0 defaults), never key-errors on an old-style run.
    assert stats["pages"] == 0
    assert stats["pages_rows"] == 0
    assert stats["pages_per_hr"] is None


def test_run_progress_unknown_run_is_none(state_dir):
    assert progress_service.run_progress("GUI-does-not-exist") is None


# --------------------------------------------------- page-metered usage rows
#
# The SemantiK GLM-OCR conversion lane meters HONEST ZERO tokens — its usage
# rows carry a positive-int ``pages`` field + duration_ms instead. These tests
# pin the pages aggregation, the honest pages/hr derivation (wall-span of the
# page rows' own timestamps preferred; summed-duration fallback — never this
# process's wall clock), the zero-token+pages payload, a mixed-provider ledger
# mirroring a real run, and the pure rate helper.

# Four GLM-OCR batch rows straight off a real ledger
# (state/runs/.../llm_usage.jsonl): concurrent batches — completions land
# within ~3 min while each row's own duration is ~193-371 s, so the wall span
# (earliest start = ts − duration → latest completion) is ~370.6 s, NOT the
# ~954 s summed duration.
_GLMOCR_ROWS = [
    {"ts": "2026-07-22T19:32:22.533828+00:00", "provider": "semantik-glmocr",
     "model": "glm-ocr", "prompt_tokens": 0, "completion_tokens": 0,
     "duration_ms": 193481.01, "phase": "semantik_conversion", "pages": 59},
    {"ts": "2026-07-22T19:32:24.423731+00:00", "provider": "semantik-glmocr",
     "model": "glm-ocr", "prompt_tokens": 0, "completion_tokens": 0,
     "duration_ms": 194859.238, "phase": "semantik_conversion", "pages": 59},
    {"ts": "2026-07-22T19:32:25.013446+00:00", "provider": "semantik-glmocr",
     "model": "glm-ocr", "prompt_tokens": 0, "completion_tokens": 0,
     "duration_ms": 195001.934, "phase": "semantik_conversion", "pages": 59},
    {"ts": "2026-07-22T19:35:19.343905+00:00", "provider": "semantik-glmocr",
     "model": "glm-ocr", "prompt_tokens": 0, "completion_tokens": 0,
     "duration_ms": 370593.875, "phase": "semantik_conversion", "pages": 59},
]


def test_usage_pages_zero_token_payload(state_dir):
    """All-GLM-OCR ledger → pages stats populated, token totals honestly 0."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-pages0",
        workflow_id="WF-20260101-pages001",
        orch_run_id="TTC_prog_pages0",
        usage_rows=list(_GLMOCR_ROWS),
    )
    stats = progress_service.run_progress(run_id)["stats"]
    assert stats["calls"] == 4
    assert stats["prompt_tokens"] == 0
    assert stats["completion_tokens"] == 0
    assert stats["pages"] == 236
    assert stats["pages_rows"] == 4
    # Wall-span rate: 236 pages over ~370.594 s → ~2292 pages/hr. The
    # summed-duration rate would be ~890/hr (4-way concurrency understates
    # wall-clock throughput ~2.6x) — pin the WALL-SPAN figure.
    assert stats["pages_per_hr"] is not None
    assert 2290 <= stats["pages_per_hr"] <= 2295


def test_usage_pages_mixed_provider_ledger(state_dir):
    """GLM-OCR page rows + alt-text token rows: pages count ONLY the
    page-bearing rows; token totals count only the token rows; both coexist."""
    rows = list(_GLMOCR_ROWS) + [
        {"ts": "2026-07-22T19:35:33.640584+00:00", "provider": "semantik-alttext",
         "model": "qwen3-vl-30b", "prompt_tokens": 396, "completion_tokens": 54,
         "duration_ms": 12629.479, "phase": "semantik_conversion",
         "finish_reason": "stop"},
        {"ts": "2026-07-22T19:35:34.153757+00:00", "provider": "semantik-alttext",
         "model": "qwen3-vl-30b", "prompt_tokens": 431, "completion_tokens": 58,
         "duration_ms": 13147.202, "phase": "semantik_conversion",
         "finish_reason": "stop"},
    ]
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-pagesmix",
        workflow_id="WF-20260101-pages002",
        orch_run_id="TTC_prog_pagesmix",
        usage_rows=rows,
    )
    stats = progress_service.run_progress(run_id)["stats"]
    assert stats["calls"] == 6
    assert stats["prompt_tokens"] == 827
    assert stats["completion_tokens"] == 112
    assert stats["pages"] == 236
    assert stats["pages_rows"] == 4  # token rows never counted as page rows
    assert 2290 <= stats["pages_per_hr"] <= 2295  # rate over page rows only


def test_usage_pages_summed_duration_fallback(state_dir):
    """Page rows with NO parseable ts → rate falls back to the rows' own
    summed durations (still never this process's wall clock); bogus ``pages``
    values (0, garbage string, JSON true) never contribute."""
    rows = [
        {"provider": "semantik-glmocr", "model": "glm-ocr", "prompt_tokens": 0,
         "completion_tokens": 0, "duration_ms": 3600000.0, "pages": 10},
        {"ts": "not-a-timestamp", "provider": "semantik-glmocr",
         "model": "glm-ocr", "prompt_tokens": 0, "completion_tokens": 0,
         "duration_ms": 3600000.0, "pages": 10},
        # None of these are page-bearing rows:
        {"provider": "semantik-glmocr", "model": "glm-ocr", "prompt_tokens": 0,
         "completion_tokens": 0, "duration_ms": 1000.0, "pages": 0},
        {"provider": "semantik-glmocr", "model": "glm-ocr", "prompt_tokens": 0,
         "completion_tokens": 0, "duration_ms": 1000.0, "pages": "garbage"},
        {"provider": "semantik-glmocr", "model": "glm-ocr", "prompt_tokens": 0,
         "completion_tokens": 0, "duration_ms": 1000.0, "pages": True},
    ]
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-pagesfb",
        workflow_id="WF-20260101-pages003",
        orch_run_id="TTC_prog_pagesfb",
        usage_rows=rows,
    )
    stats = progress_service.run_progress(run_id)["stats"]
    assert stats["calls"] == 5
    assert stats["pages"] == 20
    assert stats["pages_rows"] == 2
    # 20 pages over 2h of summed generation time → 10 pages/hr.
    assert stats["pages_per_hr"] == 10


def test_pages_per_hr_pure_helper():
    """Wall-span preferred; summed-duration fallback; None when nothing is
    honestly computable."""
    # Wall span wins when timestamps parsed: 100 pages over 0.5h → 200/hr
    # even though summed durations would say 100/hr.
    assert progress_service._pages_per_hr(100, 2, 3600000.0, 1000.0, 2800.0) == 200
    # No timestamps → summed-duration fallback: 100 pages over 1h → 100/hr.
    assert progress_service._pages_per_hr(100, 2, 3600000.0, None, None) == 100
    # Zero/negative wall span (clock skew) → falls through to summed.
    assert progress_service._pages_per_hr(100, 2, 3600000.0, 2800.0, 2800.0) == 100
    # No page rows / no pages / no denominator at all → None, never 0.
    assert progress_service._pages_per_hr(0, 0, 0.0, None, None) is None
    assert progress_service._pages_per_hr(0, 3, 3600000.0, None, None) is None
    assert progress_service._pages_per_hr(100, 2, 0.0, None, None) is None


def test_run_progress_skipped_marker_and_env_verdict(state_dir):
    """A branchy ``_skipped`` marker HIDES the not-taken side (owner design);
    the OBSERVED env skip re-enables the complementary two-pass tiers even
    when the serving process env differs."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-skips",
        workflow_id="WF-20260101-prog0003",
        orch_run_id="TTC_prog_skips",
        phase_outputs={
            "semantik_conversion": {"_completed": True},
            "heading_judge": {"_completed": True},
            "staging": {"_completed": True},
            "chunking": {"_completed": True},
            "objective_extraction": {"_completed": True},
            "source_mapping": {"_completed": True},
            "course_planning": {"_completed": True},
            "concept_extraction": {"_completed": True},
            # The runner skipped the single-pass phase → the run had
            # COURSEFORGE_TWO_PASS=true, whatever THIS process env says.
            "content_generation": {"_completed": True, "_skipped": True},
        },
    )
    payload = progress_service.run_progress(run_id)
    # The provably-not-taken branch row is hidden entirely, not shown dimmed.
    assert "content_generation" not in [p["name"] for p in payload["phases"]]
    # The observed verdict enables the two-pass tiers: outline is current,
    # the downstream tiers pending — NOT predicted-skipped off the test env.
    assert payload["current_phase"] == "content_generation_outline"
    assert _phase(payload, "inter_tier_validation")["state"] == "pending"
    assert _phase(payload, "content_generation_rewrite")["state"] == "pending"


def test_single_pass_evidence_shows_row_and_hides_tiers(state_dir):
    """The complement direction: when the single-pass phase provably RUNS, it
    renders as today (current via its in-flight checkpoint, then done) and
    the two-pass tiers hide once their skip is observed."""
    # Evidence = in-flight checkpoint → the row appears as "current".
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-singlepass",
        workflow_id="WF-20260101-prog0040",
        orch_run_id="TTC_prog_singlepass",
        phase_outputs={
            name: {"_completed": True}
            for name in (
                "semantik_conversion",
                "heading_judge",
                "staging",
                "chunking",
                "objective_extraction",
                "source_mapping",
                "course_planning",
                "concept_extraction",
            )
        },
        checkpoints={
            "content_generation": {
                "status": "started",
                "started_at": "2026-01-01T00:10:00",
                "completed_at": None,
            },
        },
    )
    payload = progress_service.run_progress(run_id)
    assert payload["current_phase"] == "content_generation"
    assert _phase(payload, "content_generation")["state"] == "current"

    # Evidence = completed (ran, not skipped) → shown done; the observed
    # verdict skips the two-pass tiers, which HIDE (not-taken branch).
    run_id2 = _seed_run(
        state_dir,
        run_id="GUI-prog-singlepass2",
        workflow_id="WF-20260101-prog0041",
        orch_run_id="TTC_prog_singlepass2",
        phase_outputs={
            **{
                name: {"_completed": True}
                for name in (
                    "semantik_conversion",
                    "heading_judge",
                    "staging",
                    "chunking",
                    "objective_extraction",
                    "source_mapping",
                    "course_planning",
                    "concept_extraction",
                )
            },
            "content_generation": {"_completed": True},
        },
    )
    payload = progress_service.run_progress(run_id2)
    assert _phase(payload, "content_generation")["state"] == "done"
    names = [p["name"] for p in payload["phases"]]
    for tier in (
        "content_generation_outline",
        "inter_tier_validation",
        "content_generation_rewrite",
        "post_rewrite_validation",
    ):
        assert tier not in names, f"{tier} is the not-taken branch — hidden"
    # assessment_synthesis is optional, not env-branchy — always rendered.
    assert "assessment_synthesis" in names


def test_env_conditional_no_evidence_renders_pending(
    state_dir, monkeypatch: pytest.MonkeyPatch
):
    """HONESTY: an env-conditional phase with NO observational evidence renders
    "pending", never "skipped" — the serving process env is NOT a prediction
    source (owner-verified inversion: a live two-pass run pre-outline rendered
    its tiers "skipped" off the GUI env)."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-noev",
        workflow_id="WF-20260101-prog0030",
        orch_run_id="TTC_prog_noev",
        phase_outputs={
            "semantik_conversion": {"_completed": True},
            "heading_judge": {"_completed": True},
            "staging": {"_completed": True},
            "chunking": {"_completed": True},
            "objective_extraction": {"_completed": True},
            "source_mapping": {"_completed": True},
            "course_planning": {"_completed": True},
            # concept_extraction in flight — NO generation-branch marker yet.
        },
    )
    for env_value in (None, "true", "false"):
        if env_value is None:
            monkeypatch.delenv("COURSEFORGE_TWO_PASS", raising=False)
        else:
            monkeypatch.setenv("COURSEFORGE_TWO_PASS", env_value)
        payload = progress_service.run_progress(run_id)
        assert payload["current_phase"] == "concept_extraction"
        for name in (
            "content_generation_outline",
            "inter_tier_validation",
            "content_generation_rewrite",
            "post_rewrite_validation",
        ):
            assert _phase(payload, name)["state"] == "pending", (
                f"{name} must be pending (env={env_value!r}), never a guessed skip"
            )
        # The single-pass fallback row is HIDDEN while it has no evidence of
        # running (owner design: the generation group starts at the outline
        # tier) — never shown as a guessed state, in either env.
        names = [p["name"] for p in payload["phases"]]
        assert "content_generation" not in names


def test_generation_section_consolidated_grouping(state_dir):
    """Owner design: ONE generation section (single-pass phase + both tiers +
    inter-tier validators + assessment synthesis), post_rewrite_validation in
    validation, and the payload's group sequence renders each section header
    exactly once, in the canonical pipeline order."""
    assert progress_service.phase_group("content_generation") == "generation"
    assert progress_service.phase_group("content_generation_outline") == "generation"
    assert progress_service.phase_group("inter_tier_validation") == "generation"
    assert progress_service.phase_group("content_generation_rewrite") == "generation"
    assert progress_service.phase_group("assessment_synthesis") == "generation"
    assert progress_service.phase_group("post_rewrite_validation") == "validation"

    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-groups",
        workflow_id="WF-20260101-prog0031",
        orch_run_id="TTC_prog_groups",
    )
    payload = progress_service.run_progress(run_id)
    groups = [p["group"] for p in payload["phases"]]
    collapsed = [g for i, g in enumerate(groups) if i == 0 or g != groups[i - 1]]
    # Contiguous sections, each exactly once, in the canonical order. Compared
    # against the canonical order FILTERED to the groups this workflow's plan
    # actually contains, so the assertion states the contract ("no group ever
    # repeats, and sections appear in canonical order") rather than pinning
    # which optional tail phases config/workflows.yaml currently declares.
    assert collapsed == [g for g in CANONICAL_GROUP_ORDER if g in set(groups)]
    assert len(collapsed) == len(set(collapsed))
    # ...and the rail's own first-occurrence bucketing yields the same order,
    # i.e. no section is dragged backwards by a late phase joining an early
    # bucket (see test_rail_section_order_for_sequenced_build_then_training).
    assert _rail_section_order(payload["phases"]) == collapsed
    # Within-group phase order is the plan order. The single-pass
    # content_generation row is hidden on an evidence-free run (owner
    # design: generation starts at the outline tier); it appears only once
    # it provably runs (see the single-pass evidence test).
    gen = [p["name"] for p in payload["phases"] if p["group"] == "generation"]
    assert gen == [
        "content_generation_outline",
        "inter_tier_validation",
        "content_generation_rewrite",
        "assessment_synthesis",
    ]
    val = [p["name"] for p in payload["phases"] if p["group"] == "validation"]
    assert val == ["post_rewrite_validation"]


def test_rail_section_order_for_sequenced_build_then_training():
    """The rail's SECTION order for the sequenced build→training pipeline.

    build→training is ONE pipeline: the training tail runs after archival and
    finalization is genuinely last. Because the rail buckets by `group` in
    FIRST-OCCURRENCE order, a group whose earliest phase sits early renders
    early no matter what else it contains — so mapping `training` into
    "generation" (or `finalization` into "archive", whose first member is
    trainforge_assessment) would drag those post-build phases visually
    BACKWARDS. This pins the grouping against the phase order itself rather
    than against config/workflows.yaml, so it states the contract the rail
    depends on even while the plan is being re-sequenced.
    """
    phase_order = [
        "semantik_conversion",
        "heading_judge",
        "staging",
        "chunking",
        "objective_extraction",
        "source_mapping",
        "course_planning",
        "concept_extraction",
        "content_generation_outline",
        "inter_tier_validation",
        "content_generation_rewrite",
        "assessment_synthesis",
        "post_rewrite_validation",
        "packaging",
        "imscc_chunking",
        "trainforge_assessment",
        "training_synthesis",
        "libv2_archival",
        "vector_indexing",
        "training",
        "post_training_validation",
        "evaluation",
        "finalization",
    ]
    phases = [{"name": n, "group": progress_service.phase_group(n)} for n in phase_order]
    sections = _rail_section_order(phases)
    assert sections == CANONICAL_GROUP_ORDER
    # Each section exactly once — the property that makes first-occurrence
    # bucketing equal true execution order.
    assert len(sections) == len(set(sections))
    # No phase falls through to the catch-all bucket.
    assert "other" not in sections
    # The post-build tail is its own section, not a fold-back into an earlier
    # one (the specific defect this test exists to prevent).
    for name in ("training", "post_training_validation", "evaluation"):
        assert progress_service.phase_group(name) == "training", name
    assert progress_service.phase_group("finalization") == "finalization"


def test_restore_restamp_suppressed_but_fast_phase_shown(state_dir):
    """Only the resume RE-STAMP signature (IDENTICAL started/completed
    timestamps → exact 0.0s span measuring the restore) is suppressed; a
    genuinely-executed phase keeps its real span, however small, so a fast
    deterministic phase is never blank ("<1s" at the renderer)."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-restored",
        workflow_id="WF-20260101-prog0032",
        orch_run_id="TTC_prog_restored",
        phase_outputs={
            "semantik_conversion": {"_completed": True},
            "heading_judge": {"_completed": True},
            "staging": {"_completed": True},
        },
        checkpoints={
            # Resume re-stamp: IDENTICAL start/end (exact 0.0s span) → suppressed.
            "semantik_conversion": {
                "status": "completed",
                "started_at": "2026-01-01T00:01:00",
                "completed_at": "2026-01-01T00:01:00",
            },
            # Genuinely-fast phase: a real 0.4s span (differing timestamps) now
            # SHOWS — the owner contract is "never blank" for fast phases.
            "heading_judge": {
                "status": "completed",
                "started_at": "2026-01-01T00:01:01",
                "completed_at": "2026-01-01T00:01:01.400000",
            },
            # Genuinely executed: 30s span still renders.
            "staging": {
                "status": "completed",
                "started_at": "2026-01-01T00:02:00",
                "completed_at": "2026-01-01T00:02:30",
            },
        },
    )
    payload = progress_service.run_progress(run_id)
    assert _phase(payload, "semantik_conversion")["state"] == "done"
    # Restore re-stamp (identical timestamps) → suppressed, never "0s".
    assert _phase(payload, "semantik_conversion")["wallclock_s"] is None
    # Genuinely-fast phase → its real sub-second span survives (renderer "<1s").
    assert _phase(payload, "heading_judge")["state"] == "done"
    assert _phase(payload, "heading_judge")["wallclock_s"] == 0.4
    assert _phase(payload, "staging")["wallclock_s"] == 30.0


def test_fast_deterministic_phase_shows_subsecond_wallclock(state_dir):
    """The reported symptom: staging + source_mapping are near-instant, so their
    checkpoint span is sub-second (differing sub-second timestamps). They must
    keep a real, positive wallclock_s (renderer formats it "<1s"), NOT blank."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-fast",
        workflow_id="WF-20260101-prog0033",
        orch_run_id="TTC_prog_fast",
        phase_outputs={
            "semantik_conversion": {"_completed": True},
            "heading_judge": {"_completed": True},
            "staging": {"_completed": True},
            "chunking": {"_completed": True},
            "objective_extraction": {"_completed": True},
            "source_mapping": {"_completed": True},
        },
        checkpoints={
            # Real fast-phase spans lifted from live run state (~0.28s / ~0.15s).
            "staging": {
                "status": "completed",
                "started_at": "2026-01-01T00:01:00.640000",
                "completed_at": "2026-01-01T00:01:00.917000",
            },
            "source_mapping": {
                "status": "completed",
                "started_at": "2026-01-01T00:02:00.244000",
                "completed_at": "2026-01-01T00:02:00.449000",
            },
        },
    )
    payload = progress_service.run_progress(run_id)
    staging = _phase(payload, "staging")
    src = _phase(payload, "source_mapping")
    assert staging["state"] == "done" and src["state"] == "done"
    # Non-blank + honest: a real positive sub-second span, never None, never 0s.
    assert isinstance(staging["wallclock_s"], float) and 0.0 < staging["wallclock_s"] < 1.0
    assert isinstance(src["wallclock_s"], float) and 0.0 < src["wallclock_s"] < 1.0


def test_current_phase_carries_live_elapsed(state_dir):
    """The in-progress phase node carries a live elapsed_s (now − phase start),
    recomputed each call (not frozen at a server-render instant)."""
    import time
    from datetime import datetime, timedelta

    # An in-flight checkpoint (status "started", no completed_at) is the
    # authoritative current-phase signal; its naive host-local started_at
    # anchors elapsed_s (~5s ago).
    started = (datetime.now() - timedelta(seconds=5)).isoformat()
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-live",
        workflow_id="WF-20260101-prog0034",
        orch_run_id="TTC_prog_live",
        phase_outputs={
            "semantik_conversion": {"_completed": True},
            "heading_judge": {"_completed": True},
            "staging": {"_completed": True},
            "chunking": {"_completed": True},
            "objective_extraction": {"_completed": True},
            "source_mapping": {"_completed": True},
        },
        checkpoints={
            "course_planning": {"status": "started", "started_at": started},
        },
    )
    p1 = progress_service.run_progress(run_id)
    node = _phase(p1, "course_planning")
    assert node["state"] == "current"
    assert p1["current_phase"] == "course_planning"
    # Live elapsed present, positive, and ~ the 5s since the anchor start.
    assert isinstance(node.get("elapsed_s"), float) and node["elapsed_s"] >= 5.0
    # A completed phase never carries elapsed_s (that field is current-only).
    assert "elapsed_s" not in _phase(p1, "staging")
    # Not frozen: a later poll recomputes a non-decreasing value.
    time.sleep(0.05)
    p2 = progress_service.run_progress(run_id)
    assert _phase(p2, "course_planning")["elapsed_s"] >= node["elapsed_s"]


def test_stage_rail_component_renders_subsecond_and_live_elapsed():
    """Static-asset assertion: the rail renders a sub-second phase honestly
    ("<1s", never blank) and sources the in-progress node's time from the live
    ``elapsed_s`` field rather than freezing it blank until completion."""
    js = (
        Path(__file__).resolve().parents[1]
        / "static"
        / "shared"
        / "components"
        / "stage-rail.js"
    ).read_text(encoding="utf-8")
    # Sub-second phases are never blank.
    assert "'<1s'" in js
    assert "if (seconds < 1)" in js
    # The in-progress node's time comes from the live elapsed_s field.
    assert "p.elapsed_s" in js
    assert "state === 'current'" in js


def test_stage_rail_component_buckets_groups_once():
    """Static-asset assertion: the rail BUCKETS phases by server-derived group
    (first-occurrence order) so each section header renders exactly once even
    when the phase order interleaves groups."""
    js = (
        Path(__file__).resolve().parents[1]
        / "static"
        / "shared"
        / "components"
        / "stage-rail.js"
    ).read_text(encoding="utf-8")
    assert "const byGroup = new Map()" in js
    assert "exactly ONCE" in js
    # The old consecutive-run grouping is gone.
    assert "last.group === g" not in js


def test_run_progress_failed_run_static(state_dir):
    """A failed run: failed phase marked, no current phase (static render)."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-failed",
        workflow_id="WF-20260101-prog0004",
        orch_run_id="TTC_prog_failed",
        gui_status="failed",
        wf_status="FAILED",
        phase_outputs={
            "semantik_conversion": {"_completed": True},
            "heading_judge": {"_completed": True},
            "staging": {"_completed": True},
            "chunking": {"_completed": True},
        },
        failed_phase="objective_extraction",
        failure_reason="failed validation gate(s): chunk_health",
    )
    payload = progress_service.run_progress(run_id)
    assert payload["status"] == "failed"
    assert payload["current_phase"] is None
    assert _phase(payload, "objective_extraction")["state"] == "failed"
    assert payload["failed_phase"] == "objective_extraction"
    assert "chunk_health" in payload["failure_reason"]
    # Nothing after the failure is current; the rest stays pending/skipped.
    assert _phase(payload, "packaging")["state"] in ("pending", "skipped")
    assert payload["stats"]["seat"] is None  # terminal → no probe


def test_run_progress_workflow_agnostic_phase_list(state_dir):
    """course_generation renders ITS OWN config phases — never the 21-phase
    textbook list (the workflow-agnostic contract)."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-cg",
        workflow_id="WF-20260101-prog0005",
        workflow="course_generation",
        orch_run_id="TTC_prog_cg",
        phase_outputs={"planning": {"_completed": True}},
    )
    payload = progress_service.run_progress(run_id)
    names = [p["name"] for p in payload["phases"]]
    assert names[0] == "planning"
    assert "semantik_conversion" not in names
    assert "packaging" in names and "validation" in names
    assert len(names) < 15  # course_generation's own (10-phase) plan
    assert _phase(payload, "planning")["state"] == "done"
    assert _phase(payload, "planning")["group"] == "planning"
    assert _phase(payload, "validation")["group"] == "validation"


def test_checkpoint_start_marks_current_and_completion_marks_done(state_dir):
    """An in-flight ``status: started`` checkpoint names the current phase
    (ahead of the phase-boundary state re-save); a ``completed`` checkpoint
    marks done even before the workflow state file catches up."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-ckpt",
        workflow_id="WF-20260101-prog0006",
        orch_run_id="TTC_prog_ckpt",
        phase_outputs={
            "semantik_conversion": {"_completed": True},
        },
        checkpoints={
            # State file hasn't re-saved yet, but the checkpoint completed.
            "heading_judge": {
                "status": "completed",
                "started_at": "2026-01-01T00:02:00",
                "completed_at": "2026-01-01T00:02:10",
            },
            # …and staging is in flight right now.
            "staging": {
                "status": "started",
                "started_at": "2026-01-01T00:02:11",
                "completed_at": None,
            },
        },
    )
    payload = progress_service.run_progress(run_id)
    assert _phase(payload, "heading_judge")["state"] == "done"
    assert _phase(payload, "heading_judge")["wallclock_s"] == 10.0
    assert payload["current_phase"] == "staging"
    assert payload["stats"]["phase_elapsed_s"] is not None


# ------------------------------------------------- per-phase unit progress


def _two_pass_outputs_through_content_generation(
    export_dir: Path,
) -> Dict[str, Any]:
    """Phase outputs putting the run INSIDE ``content_generation_outline``.

    Mirrors the real state shape: ``objective_extraction`` carries the
    Courseforge export ``project_path`` and the single-pass phase is stamped
    ``_skipped`` (the observed two-pass verdict), so the outline tier resolves
    as the current phase.
    """
    return {
        "semantik_conversion": {"_completed": True},
        "heading_judge": {"_completed": True},
        "staging": {"_completed": True},
        "chunking": {"_completed": True},
        "objective_extraction": {
            "_completed": True,
            "project_path": str(export_dir),
        },
        "source_mapping": {"_completed": True},
        "course_planning": {"_completed": True},
        "concept_extraction": {"_completed": True},
        "content_generation": {"_completed": True, "_skipped": True},
    }


def _append_sidecar(path: Path, rows: int, *, start: int = 0) -> None:
    """Append ``rows`` complete JSONL rows shaped like real checkpoint rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for i in range(start, start + rows):
            fh.write(
                json.dumps(
                    {
                        "block_id": f"blk-{i:04d}",
                        "block_type": "concept",
                        "sequence": i,
                        "checkpoint_fingerprint": f"fp-{i:04d}",
                    }
                )
                + "\n"
            )


def test_phase_units_from_outline_sidecar_counts_and_updates(state_dir, tmp_path):
    """Sidecar present → stats.phase_units surfaces the row count, sourced from
    the export dir resolved out of phase_outputs (never a hardcoded pattern);
    appended rows tick the count on the next poll."""
    export_dir = tmp_path / "export"
    sidecar = export_dir / "01_outline" / ".blocks_outline_checkpoint.jsonl"
    _append_sidecar(sidecar, 3)
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-units",
        workflow_id="WF-20260101-prog0010",
        orch_run_id="TTC_prog_units",
        phase_outputs=_two_pass_outputs_through_content_generation(export_dir),
    )

    payload = progress_service.run_progress(run_id)
    assert payload["current_phase"] == "content_generation_outline"
    units = payload["stats"]["phase_units"]
    assert units["count"] == 3
    assert units["label"] == "blocks done"
    assert units["source"] == ".blocks_outline_checkpoint.jsonl"
    assert isinstance(units["updated_at"], str) and units["updated_at"]
    # NO fabricated total: none of the writers emits a planned-unit count.
    assert "total" not in units

    # Append two more completed units → the incremental count follows.
    _append_sidecar(sidecar, 2, start=3)
    payload = progress_service.run_progress(run_id)
    assert payload["stats"]["phase_units"]["count"] == 5


def test_phase_units_absent_sidecar_omits_field(state_dir, tmp_path):
    """HONESTY: no sidecar file on disk → the field is omitted entirely."""
    export_dir = tmp_path / "export"
    (export_dir / "01_outline").mkdir(parents=True)
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-units-none",
        workflow_id="WF-20260101-prog0011",
        orch_run_id="TTC_prog_units_none",
        phase_outputs=_two_pass_outputs_through_content_generation(export_dir),
    )
    payload = progress_service.run_progress(run_id)
    assert payload["current_phase"] == "content_generation_outline"
    assert "phase_units" not in payload["stats"]


def test_phase_units_unmapped_phase_omits_field(state_dir, tmp_path):
    """A current phase with no known sidecar (staging) never gets the field."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-units-unmapped",
        workflow_id="WF-20260101-prog0012",
        orch_run_id="TTC_prog_units_unmapped",
        phase_outputs={
            "semantik_conversion": {"_completed": True},
            "heading_judge": {"_completed": True},
        },
    )
    payload = progress_service.run_progress(run_id)
    assert payload["current_phase"] == "staging"
    assert "phase_units" not in payload["stats"]


def test_phase_units_corrupt_and_partial_rows_do_not_crash(state_dir, tmp_path):
    """Counting is newline-based: a garbage row still counts as a checkpointed
    line (bounded, no JSON parse), while a mid-append PARTIAL tail line without
    its newline is NOT counted."""
    export_dir = tmp_path / "export"
    sidecar = export_dir / "01_outline" / ".blocks_outline_checkpoint.jsonl"
    _append_sidecar(sidecar, 2)
    with sidecar.open("ab") as fh:
        fh.write(b"\x00\xff{not json at all\n")  # corrupt but complete row
        fh.write(b'{"block_id": "blk-trunc')  # mid-append partial tail
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-units-corrupt",
        workflow_id="WF-20260101-prog0013",
        orch_run_id="TTC_prog_units_corrupt",
        phase_outputs=_two_pass_outputs_through_content_generation(export_dir),
    )
    payload = progress_service.run_progress(run_id)
    assert payload["stats"]["phase_units"]["count"] == 3

    # The partial tail completes → it counts on the next poll.
    with sidecar.open("ab") as fh:
        fh.write(b'"}\n')
    payload = progress_service.run_progress(run_id)
    assert payload["stats"]["phase_units"]["count"] == 4


def test_phase_units_libv2_anchor_for_concept_extraction(state_dir, tmp_path):
    """The concept_extraction sidecar anchors on the LibV2 course dir derived
    from the chunking output path (real state, no hardcoded slug)."""
    course_dir = tmp_path / "libv2" / "courses" / "unit-test-course"
    chunks_path = course_dir / "semantik_chunks" / "chunks.jsonl"
    sidecar = course_dir / "concept_graph" / ".concept_extraction_checkpoint.jsonl"
    _append_sidecar(sidecar, 4)
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-units-concept",
        workflow_id="WF-20260101-prog0014",
        orch_run_id="TTC_prog_units_concept",
        phase_outputs={
            "semantik_conversion": {"_completed": True},
            "heading_judge": {"_completed": True},
            "staging": {"_completed": True},
            "chunking": {
                "_completed": True,
                "semantik_chunks_path": str(chunks_path),
            },
            "objective_extraction": {"_completed": True},
            "source_mapping": {"_completed": True},
            "course_planning": {"_completed": True},
        },
    )
    payload = progress_service.run_progress(run_id)
    assert payload["current_phase"] == "concept_extraction"
    units = payload["stats"]["phase_units"]
    assert units["count"] == 4
    assert units["label"] == "windows done"
    assert units["source"] == ".concept_extraction_checkpoint.jsonl"


def test_phase_units_count_cache_reuses_size_key(state_dir, tmp_path):
    """Same size → cached count (no re-read); growth counts only appended
    bytes; a shrunk (rotated) file triggers a full recount."""
    sidecar = tmp_path / ".blocks_outline_checkpoint.jsonl"
    _append_sidecar(sidecar, 3)
    size = sidecar.stat().st_size
    assert progress_service._count_sidecar_rows(sidecar, size) == 3
    # Cached result at the same size.
    assert progress_service._UNIT_COUNT_CACHE[str(sidecar)] == (size, 3)
    assert progress_service._count_sidecar_rows(sidecar, size) == 3
    # Growth: incremental count over the appended span only.
    _append_sidecar(sidecar, 2, start=3)
    size = sidecar.stat().st_size
    assert progress_service._count_sidecar_rows(sidecar, size) == 5
    # Shrink (rotation / restart wrote a fresh sidecar): full recount.
    sidecar.write_text('{"block_id": "blk-0000"}\n', encoding="utf-8")
    size = sidecar.stat().st_size
    assert progress_service._count_sidecar_rows(sidecar, size) == 1


# ------------------------------------- unit sidecars: new phase coverage


def _outputs_through_trainforge_assessment(
    trainforge_out: Dict[str, Any],
) -> Dict[str, Any]:
    """Phase outputs putting a single-pass run INSIDE ``training_synthesis``."""
    outs: Dict[str, Any] = {
        name: {"_completed": True}
        for name in (
            "semantik_conversion",
            "heading_judge",
            "staging",
            "chunking",
            "objective_extraction",
            "source_mapping",
            "course_planning",
            "concept_extraction",
            "content_generation",
            "packaging",
            "imscc_chunking",
        )
    }
    for name in (
        "content_generation_outline",
        "inter_tier_validation",
        "content_generation_rewrite",
        "assessment_synthesis",
        "post_rewrite_validation",
    ):
        outs[name] = {"_completed": True, "_skipped": True}
    outs["trainforge_assessment"] = {"_completed": True, **trainforge_out}
    return outs


def test_training_synthesis_units_and_tail_from_pairs_checkpoint(
    state_dir, tmp_path
):
    """training_synthesis maps to the per-pair resume sidecar
    ``<corpus>/training_specs/.synthesis_pairs_checkpoint.jsonl``
    (Trainforge/synthesize_training.py::_append_synthesis_pairs_checkpoint).
    The corpus anchors off the trainforge_assessment output exactly like the
    registry handler: assessments_path one parent up, chunks_path two."""
    corpus = tmp_path / "trainforge-corpus"
    sidecar = corpus / "training_specs" / ".synthesis_pairs_checkpoint.jsonl"
    _append_content_rows(
        sidecar,
        [
            {
                "schema_version": 1,
                "chunk_id": f"chunk-{i:04d}",
                "kind": "instruction",
                "variant_index": 0,
                "pair": {"instruction": f"Explain concept {i}", "output": "…"},
                "provider": "local",
                "seed": 7,
            }
            for i in range(3)
        ],
    )
    run_id = _seed_run(
        state_dir,
        run_id="GUI-tail-pairs",
        workflow_id="WF-20260101-tail0030",
        orch_run_id="TTC_tail_pairs",
        phase_outputs=_outputs_through_trainforge_assessment(
            {"assessments_path": str(corpus / "assessments.json")}
        ),
    )
    payload = progress_service.run_progress(run_id)
    assert payload["current_phase"] == "training_synthesis"
    units = payload["stats"]["phase_units"]
    assert units["count"] == 3
    assert units["label"] == "pairs done"
    assert units["source"] == ".synthesis_pairs_checkpoint.jsonl"

    tail = progress_service.output_tail(run_id)
    assert tail["row_count"] == 3
    assert tail["rows"][-1]["label"] == "chunk-0002"  # chunk_id is the label
    assert "Explain concept 2" in tail["rows"][-1]["text"]

    # chunks_path variant: <corpus>/corpus/chunks.jsonl → two parents up.
    run_id2 = _seed_run(
        state_dir,
        run_id="GUI-tail-pairs2",
        workflow_id="WF-20260101-tail0031",
        orch_run_id="TTC_tail_pairs2",
        phase_outputs=_outputs_through_trainforge_assessment(
            {"chunks_path": str(corpus / "corpus" / "chunks.jsonl")}
        ),
    )
    tail = progress_service.output_tail(run_id2)
    assert tail["row_count"] == 3
    assert tail["label"] == "pairs done"


def test_training_synthesis_anchor_ignores_same_named_earlier_phase_path(
    state_dir, tmp_path
):
    """The tail follows trainforge_assessment, not the first same-named key."""
    wrong = tmp_path / "earlier-assessments"
    corpus = tmp_path / "trainforge-corpus"
    sidecar = corpus / "training_specs" / ".synthesis_pairs_checkpoint.jsonl"
    _append_content_rows(
        sidecar,
        [{"chunk_id": "right-chunk", "pair": {"instruction": "right"}}],
    )
    outputs = _outputs_through_trainforge_assessment(
        {"assessments_path": str(corpus / "assessments.json")}
    )
    outputs["assessment_synthesis"] = {
        "_completed": True,
        "assessments_path": str(wrong / "assessments.json"),
    }
    run_id = _seed_run(
        state_dir,
        run_id="GUI-tail-pairs-producer",
        workflow_id="WF-20260101-tail0032",
        orch_run_id="TTC_tail_pairs_producer",
        phase_outputs=outputs,
    )
    payload = progress_service.run_progress(run_id)
    assert payload["stats"]["phase_units"]["count"] == 1
    tail = progress_service.output_tail(run_id)
    assert tail["row_count"] == 1
    assert tail["rows"][0]["label"] == "right-chunk"


def test_heading_judge_units_and_tail_from_judgments_dir(state_dir, tmp_path):
    """heading_judge is a growing DIRECTORY (one {stem}.heading_judgments.json
    per chapter under state/runs/<run_id>/heading_judge/ —
    MCP/tools/pipeline_tools.py::_run_heading_judge), not one appended file:
    units = matching files, tail = newest files with their (bounded) JSON
    payload; a corrupt unit file yields a label-only row."""
    import os as _os

    orch = "TTC_tail_hj"
    run_id = _seed_run(
        state_dir,
        run_id="GUI-tail-hj",
        workflow_id="WF-20260101-tail0032",
        orch_run_id=orch,
        phase_outputs={"semantik_conversion": {"_completed": True}},
    )
    judge_dir = state_dir / "runs" / orch / "heading_judge"
    judge_dir.mkdir(parents=True)
    base = 1_700_000_000
    for i, payload in ((1, {"n_pending": 12, "applied": 9, "kept": 3}),
                       (2, {"n_pending": 4, "applied": 4, "kept": 0})):
        f = judge_dir / f"ch{i:02d}.heading_judgments.json"
        f.write_text(json.dumps(payload), encoding="utf-8")
        _os.utime(f, (base + i, base + i))
    # An unrelated file in the dir must not count (glob-scoped).
    (judge_dir / "ch01_accessible.html").write_text("<p></p>", encoding="utf-8")

    payload = progress_service.run_progress(run_id)
    assert payload["current_phase"] == "heading_judge"
    units = payload["stats"]["phase_units"]
    assert units["count"] == 2
    assert units["label"] == "chapters judged"
    assert units["source"] == "heading_judge"

    tail = progress_service.output_tail(run_id)
    assert tail["source"] == "heading_judge"
    assert tail["row_count"] == 2
    assert [r["label"] for r in tail["rows"]] == ["ch01", "ch02"]  # mtime order
    assert '"applied": 9' in tail["rows"][0]["text"]

    # A corrupt judgments file still surfaces as a label-only row.
    bad = judge_dir / "ch03.heading_judgments.json"
    bad.write_text("{not json", encoding="utf-8")
    _os.utime(bad, (base + 3, base + 3))
    tail = progress_service.output_tail(run_id)
    assert tail["row_count"] == 3
    assert tail["rows"][-1] == {"seq": None, "label": "ch03", "text": ""}


def test_heading_judge_empty_dir_omits_units(state_dir):
    """HONESTY: the judge dir existing with zero judged chapters → phase_units
    omitted and an empty tail (no fabricated 0-progress)."""
    orch = "TTC_tail_hj_empty"
    run_id = _seed_run(
        state_dir,
        run_id="GUI-tail-hj-empty",
        workflow_id="WF-20260101-tail0033",
        orch_run_id=orch,
        phase_outputs={"semantik_conversion": {"_completed": True}},
    )
    (state_dir / "runs" / orch / "heading_judge").mkdir(parents=True)
    payload = progress_service.run_progress(run_id)
    assert payload["current_phase"] == "heading_judge"
    assert "phase_units" not in payload["stats"]
    tail = progress_service.output_tail(run_id)
    assert tail["rows"] == [] and tail["source"] is None


def test_atomic_phases_stay_unmapped():
    """Owner honesty contract, pinned per investigation: these phases write
    their artifacts atomically at phase end (chunks.jsonl is streamed from a
    complete in-memory list; the IMSCC/index/manifest are whole-file emits)
    OR their output dir is not resolvable from the workflow state while
    running (semantik_conversion) — so they must NOT carry a unit-sidecar
    mapping (mapping one would fabricate progress)."""
    for name in (
        "semantik_conversion",
        "staging",
        "chunking",
        "imscc_chunking",
        "objective_extraction",
        "source_mapping",
        "packaging",
        "trainforge_assessment",
        "libv2_archival",
        "vector_indexing",
        "finalization",
        "post_training_validation",
    ):
        assert name not in progress_service._PHASE_UNIT_SIDECARS, name


# ------------------------------- trainforge_train (the adapter-training tail)


def test_trainforge_train_plan_renders_and_band_degrades(state_dir):
    """The LoRA workflow renders end-to-end off its own config plan (the
    workflow-agnostic contract): two phases, sensible groups, current phase;
    the stats band degrades honestly with no llm_usage rows (nulls/zeros,
    streams omitted, no crash)."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-tft-plan",
        workflow_id="WF-20260101-tft0001",
        workflow="trainforge_train",
        orch_run_id="TFT_plan_0001",
        params={"course_name": "TST_101", "base_model": "nemotron-3-nano"},
    )
    payload = progress_service.run_progress(run_id)
    names = [p["name"] for p in payload["phases"]]
    assert names == ["training", "post_training_validation"]
    # Both phases sit in the post-build "training" section. The train phase
    # used to map to "generation" and its validator to "validation" — two
    # groups whose first occurrence in the BUILD plan is far earlier, which
    # rendered the training tail inside those earlier sections once the two
    # plans render on one rail.
    assert _phase(payload, "training")["group"] == "training"
    assert _phase(payload, "post_training_validation")["group"] == "training"
    assert payload["current_phase"] == "training"
    # No llm_usage.jsonl for a training run → honest nulls, nothing invented.
    stats = payload["stats"]
    assert stats["tok_s"] is None
    assert stats["calls"] == 0
    assert "streams" not in stats
    assert "detail" not in stats


def test_trainforge_train_progress_units_snapshot_and_tail(state_dir, libv2_root):
    """The training phase tails its distinct SFT/DPO telemetry stream at
    <libv2>/courses/<slug>/models/<model_id>/training_telemetry.v1.jsonl — the
    NEWEST matching stream, since <model_id> is minted at run start. The
    anchor mirrors TrainingRunner._resolve_course_dir (canonical
    libv2_course_slug off params.course_name)."""
    import os as _os

    course = libv2_root / "courses" / "tst-101"  # libv2_course_slug("TST_101")
    old = course / "models" / "model-old" / "training_telemetry.v1.jsonl"
    new = course / "models" / "model-new" / "training_telemetry.v1.jsonl"
    _append_content_rows(
        old, [{"schema_version": 1, "event": "stage_start", "stage": "sft"}]
    )
    _append_content_rows(
        new,
        [
            {"schema_version": 1, "event": "stage_start", "stage": "sft",
             "status": "running", "global_step": 0, "max_steps": 40,
             "metrics": {}},
            {"schema_version": 1, "event": "progress", "stage": "sft",
             "status": "running", "global_step": 5, "max_steps": 40,
             "metrics": {"loss": 1.25, "eta_seconds": 70.0}},
        ],
    )
    latest = new.parent / "training_telemetry.latest.v1.json"
    latest.write_text(json.dumps({
        "schema_version": 1, "event": "progress", "stage": "sft",
        "status": "running", "global_step": 5, "max_steps": 40,
        "metrics": {"loss": 1.25, "eta_seconds": 70.0},
    }), encoding="utf-8")
    base = 1_700_000_000
    _os.utime(old, (base, base))
    _os.utime(new, (base + 60, base + 60))
    _os.utime(latest, (base + 60, base + 60))

    run_id = _seed_run(
        state_dir,
        run_id="GUI-tft-eval",
        workflow_id="WF-20260101-tft0002",
        workflow="trainforge_train",
        orch_run_id="TFT_eval_0002",
        params={"course_name": "TST_101"},
    )
    payload = progress_service.run_progress(run_id)
    assert payload["current_phase"] == "training"
    units = payload["stats"]["phase_units"]
    assert units["count"] == 2  # rows in the NEWEST stream only
    assert units["label"] == "training events"
    assert units["source"] == "training_telemetry.v1.jsonl"
    telemetry = payload["stats"]["training_telemetry"]
    assert telemetry["stage"] == "sft"
    assert telemetry["global_step"] == 5
    assert telemetry["metrics"]["eta_seconds"] == 70.0

    tail = progress_service.output_tail(run_id)
    assert tail["row_count"] == 2
    assert [r["label"] for r in tail["rows"]] == ["stage_start", "progress"]
    assert '"global_step": 5' in tail["rows"][-1]["text"]


def test_trainforge_train_no_training_stream_yet_omits_units(
    state_dir, libv2_root
):
    """HONESTY: before training mints its stream, no progress is fabricated."""
    (libv2_root / "courses" / "tst-101" / "models").mkdir(parents=True)
    run_id = _seed_run(
        state_dir,
        run_id="GUI-tft-noeval",
        workflow_id="WF-20260101-tft0003",
        workflow="trainforge_train",
        orch_run_id="TFT_eval_0003",
        params={"course_name": "TST_101"},
    )
    payload = progress_service.run_progress(run_id)
    assert payload["current_phase"] == "training"
    assert "phase_units" not in payload["stats"]
    tail = progress_service.output_tail(run_id)
    assert tail["rows"] == [] and tail["source"] is None


# ---------------------------------------------------------- live output tail


def _append_content_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_output_tail_maps_truncates_and_strips(state_dir, tmp_path):
    """The tail returns the LAST 15 complete rows mapped to bounded display
    records: label = the row's natural id, text = the content field
    HTML-stripped and truncated to ~500 chars with a trailing ellipsis."""
    export_dir = tmp_path / "export"
    sidecar = export_dir / "01_outline" / ".blocks_outline_checkpoint.jsonl"
    rows = [
        {
            "block_id": f"blk-{i:04d}",
            "block_type": "concept",
            "sequence": i,
            "checkpoint_fingerprint": f"fp-{i:04d}",
            "content": f"<p>Block <strong>{i}</strong> prose.</p>",
        }
        for i in range(19)
    ]
    rows.append(
        {
            "block_id": "blk-0019",
            "sequence": 19,
            "content": "<p>" + ("word " * 300) + "</p>",  # ~1500 chars
        }
    )
    _append_content_rows(sidecar, rows)
    run_id = _seed_run(
        state_dir,
        run_id="GUI-tail-map",
        workflow_id="WF-20260101-tail0001",
        orch_run_id="TTC_tail_map",
        phase_outputs=_two_pass_outputs_through_content_generation(export_dir),
    )
    tail = progress_service.output_tail(run_id)
    assert tail["phase"] == "content_generation_outline"
    assert tail["source"] == ".blocks_outline_checkpoint.jsonl"
    assert tail["label"] == "blocks done"
    assert tail["row_count"] == 20
    assert len(tail["rows"]) == 15  # last 15 only, oldest → newest
    assert tail["rows"][0]["label"] == "blk-0005"
    assert tail["rows"][0]["seq"] == 5
    # HTML stripped, whitespace collapsed.
    assert tail["rows"][0]["text"] == "Block 5 prose."
    last = tail["rows"][-1]
    assert last["label"] == "blk-0019"
    # Truncated with an explicit marker — never a raw multi-KB blob.
    assert len(last["text"]) <= 500
    assert last["text"].endswith("…")


def test_output_tail_row_without_content_field_renders_compact_json(
    state_dir, tmp_path
):
    """A window/concept-shaped row (no single content field) renders a compact
    JSON of its payload minus resume-plumbing keys (fingerprints, schema
    stamps), labelled by its unit_id."""
    export_dir = tmp_path / "export"
    sidecar = export_dir / "01_outline" / ".blocks_outline_checkpoint.jsonl"
    _append_content_rows(
        sidecar,
        [
            {
                "schema_version": "v1",
                "site_id": "concept_extraction",
                "unit_id": "ch0#w0",
                "fingerprint": "1587c39c",
                "window_index": 0,
                "concepts": [{"canonical": "distributed system"}],
            }
        ],
    )
    run_id = _seed_run(
        state_dir,
        run_id="GUI-tail-json",
        workflow_id="WF-20260101-tail0002",
        orch_run_id="TTC_tail_json",
        phase_outputs=_two_pass_outputs_through_content_generation(export_dir),
    )
    tail = progress_service.output_tail(run_id)
    (row,) = tail["rows"]
    assert row["label"] == "ch0#w0"
    assert row["seq"] == 0
    assert "distributed system" in row["text"]
    # Bookkeeping keys elided from the fallback rendering.
    assert "fingerprint" not in row["text"]
    assert "schema_version" not in row["text"]


def test_output_tail_absent_sidecar_and_unmapped_phase_are_empty(
    state_dir, tmp_path
):
    """HONESTY: no sidecar on disk → rows []; a current phase with no mapped
    sidecar (staging) → rows []. Nothing fabricated either way."""
    export_dir = tmp_path / "export"
    (export_dir / "01_outline").mkdir(parents=True)
    run_id = _seed_run(
        state_dir,
        run_id="GUI-tail-none",
        workflow_id="WF-20260101-tail0003",
        orch_run_id="TTC_tail_none",
        phase_outputs=_two_pass_outputs_through_content_generation(export_dir),
    )
    tail = progress_service.output_tail(run_id)
    assert tail["rows"] == []
    assert tail["source"] is None
    assert tail["row_count"] == 0

    unmapped = _seed_run(
        state_dir,
        run_id="GUI-tail-unmapped",
        workflow_id="WF-20260101-tail0004",
        orch_run_id="TTC_tail_unmapped",
        phase_outputs={
            "semantik_conversion": {"_completed": True},
            "heading_judge": {"_completed": True},
        },
    )
    tail = progress_service.output_tail(unmapped)
    assert tail["phase"] == "staging"
    assert tail["rows"] == []


def test_output_tail_skips_corrupt_and_partial_rows(state_dir, tmp_path):
    """A corrupt (complete) line is skipped; a mid-append PARTIAL tail line
    without its newline is never surfaced."""
    export_dir = tmp_path / "export"
    sidecar = export_dir / "01_outline" / ".blocks_outline_checkpoint.jsonl"
    _append_content_rows(
        sidecar,
        [{"block_id": "blk-ok", "sequence": 1, "content": "<p>fine</p>"}],
    )
    with sidecar.open("ab") as fh:
        fh.write(b"\x00\xff{not json at all\n")  # corrupt but complete row
        fh.write(b'{"block_id": "blk-partial", "content": "<p>trunc')  # partial
    run_id = _seed_run(
        state_dir,
        run_id="GUI-tail-corrupt",
        workflow_id="WF-20260101-tail0005",
        orch_run_id="TTC_tail_corrupt",
        phase_outputs=_two_pass_outputs_through_content_generation(export_dir),
    )
    tail = progress_service.output_tail(run_id)
    labels = [r["label"] for r in tail["rows"]]
    assert labels == ["blk-ok"]


def test_output_tail_bounded_read_drops_partial_head(
    state_dir, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """The read is bounded (seek-from-end cap): only rows inside the byte
    window return, and the partial head line the seek landed in is dropped."""
    export_dir = tmp_path / "export"
    sidecar = export_dir / "01_outline" / ".blocks_outline_checkpoint.jsonl"
    rows = [
        {"block_id": f"blk-{i:04d}", "sequence": i, "content": f"<p>row {i}</p>"}
        for i in range(30)
    ]
    _append_content_rows(sidecar, rows)
    monkeypatch.setattr(progress_service, "_TAIL_MAX_BYTES", 300)
    run_id = _seed_run(
        state_dir,
        run_id="GUI-tail-bounded",
        workflow_id="WF-20260101-tail0006",
        orch_run_id="TTC_tail_bounded",
        phase_outputs=_two_pass_outputs_through_content_generation(export_dir),
    )
    tail = progress_service.output_tail(run_id)
    assert 0 < len(tail["rows"]) < 30
    # Every surfaced row parsed cleanly from a COMPLETE line in the window —
    # newest last, no garbled partial-head remnant.
    assert tail["rows"][-1]["label"] == "blk-0029"
    for row in tail["rows"]:
        assert row["label"].startswith("blk-")
        assert row["text"].startswith("row ")


def test_output_tail_unknown_run_is_none(state_dir):
    assert progress_service.output_tail("GUI-does-not-exist") is None


def test_stage_rail_component_renders_output_tail_panel():
    """Static-asset assertions (the established headless-JS pattern): the rail
    exposes updateTail, hides the panel when the server sent no rows, sits the
    panel BELOW the (collapsed) detail disclosure, and auto-sticks to the
    newest row; create.js polls the endpoint on the shared cadence; the CSS
    bounds the panel with LOCAL vertical scroll in monospace."""
    static = Path(__file__).resolve().parents[1] / "static"
    js = (static / "shared" / "components" / "stage-rail.js").read_text(
        encoding="utf-8"
    )
    assert "function updateTail(payload)" in js
    assert "tailHost.hidden = true" in js  # no rows → hidden, no empty shell
    assert "nearBottom" in js  # auto-stick unless the user scrolled up
    # Panel mounts AFTER the detail disclosure (stats stay reachable above).
    assert js.index("detailHost,") < js.index("tailHost,")
    create = (static / "studio" / "create.js").read_text(encoding="utf-8")
    assert "/output-tail" in create
    assert "updateTail" in create
    css = (static / "shared" / "components" / "components.css").read_text(
        encoding="utf-8"
    )
    assert ".stage-output-scroll" in css
    assert "overflow-y: auto" in css
    assert "max-height: 300px" in css
    assert "var(--mono, monospace)" in css


# ------------------------------------------------------ detail stats matrix


def _detail_usage_rows() -> List[Dict[str, Any]]:
    """Mixed-model usage rows with known latency/health figures.

    3 local/qwen calls + 1 nvidia/big call; one truncation
    (``finish_reason: length``), one absent-defaulted usage row
    (``stream_usage_present: false``). Totals: 4 calls, 350 in, 1000 out over
    10 generation seconds → avg 100 tok/s; ttft samples [100, 200, 300].
    """
    return [
        {"ts": "2026-01-01T00:00:05+00:00", "provider": "local", "model": "qwen",
         "prompt_tokens": 100, "completion_tokens": 300, "duration_ms": 3000.0,
         "ttft_ms": 100.0},
        {"ts": "2026-01-01T00:00:10+00:00", "provider": "local", "model": "qwen",
         "prompt_tokens": 100, "completion_tokens": 300, "duration_ms": 3000.0,
         "ttft_ms": 200.0},
        {"ts": "2026-01-01T00:00:15+00:00", "provider": "local", "model": "qwen",
         "prompt_tokens": 100, "completion_tokens": 200, "duration_ms": 2000.0,
         "ttft_ms": 300.0, "finish_reason": "length"},
        {"ts": "2026-01-01T00:00:20+00:00", "provider": "nvidia", "model": "big",
         "prompt_tokens": 50, "completion_tokens": 200, "duration_ms": 2000.0,
         "stream_usage_present": False},
    ]


def test_detail_totals_latency_health_by_model(state_dir):
    """Full-file aggregation: totals, cumulative avg tok/s, ttft p50/p95,
    duration mean/median, both health tripwires, and the per-(provider, model)
    breakdown — with a corrupt line in the file tolerated (skipped)."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-detail",
        workflow_id="WF-20260101-prog0020",
        orch_run_id="TTC_prog_detail",
        usage_rows=_detail_usage_rows(),
    )
    usage_path = state_dir / "runs" / "TTC_prog_detail" / "llm_usage.jsonl"
    with usage_path.open("ab") as fh:
        fh.write(b"{corrupt not json\n")

    detail = progress_service.run_progress(run_id)["stats"]["detail"]
    assert detail["totals"] == {
        "calls": 4,
        "prompt_tokens": 350,
        "completion_tokens": 1000,
        "total_tokens": 1350,
    }
    # Cumulative avg: 1000 completion tokens / 10 generation seconds.
    assert detail["throughput"]["avg_tok_s"] == 100.0
    # The window figures keep the band's rolling-window membership (rows are
    # years outside the window here → honestly None for both).
    assert detail["throughput"]["window_tok_s"] is None
    assert detail["throughput"]["per_stream_tok_s"] is None
    lat = detail["latency"]
    assert lat["ttft_p50_ms"] == 200.0
    assert lat["ttft_p95_ms"] == 290.0  # linear interp over [100, 200, 300]
    assert lat["duration_mean_ms"] == 2500.0
    assert lat["duration_median_ms"] == 2500.0
    assert detail["health"] == {"truncated_calls": 1, "usage_missing_calls": 1}
    assert detail["by_model"] == [
        {"provider": "local", "model": "qwen", "calls": 3,
         "prompt_tokens": 300, "completion_tokens": 800},
        {"provider": "nvidia", "model": "big", "calls": 1,
         "prompt_tokens": 50, "completion_tokens": 200},
    ]
    # No vram_trajectory.jsonl → the vram section is omitted entirely.
    assert "vram" not in detail

    # Incremental cache: an appended row updates totals on the next poll
    # without a full re-read (offset-keyed accumulator).
    with usage_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(
            {"ts": "2026-01-01T00:00:25+00:00", "provider": "local",
             "model": "qwen", "prompt_tokens": 10, "completion_tokens": 20,
             "duration_ms": 1000.0}) + "\n")
    detail2 = progress_service.run_progress(run_id)["stats"]["detail"]
    assert detail2["totals"]["calls"] == 5
    assert detail2["totals"]["prompt_tokens"] == 360
    assert detail2["totals"]["completion_tokens"] == 1020


def test_detail_by_phase_attribution_and_unattributed(state_dir):
    """Usage rows bucket into the checkpoint wall-clock windows via the
    per-writer-frame epoch join (usage ts aware-UTC, checkpoints naive
    host-local); a row in an inter-phase gap lands in the explicit
    'unattributed' bucket, never guessed into a phase."""
    from datetime import datetime, timezone

    base = 1_700_000_000  # any fixed epoch; both frames derive from it

    def _local(epoch: float) -> str:
        return datetime.fromtimestamp(epoch).isoformat()  # naive host-local

    def _utc(epoch: float) -> str:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-phases",
        workflow_id="WF-20260101-prog0021",
        orch_run_id="TTC_prog_phases",
        checkpoints={
            "semantik_conversion": {
                "status": "completed",
                "started_at": _local(base),
                "completed_at": _local(base + 100),
            },
            # Gap [base+100, base+200) between the two windows.
            "staging": {
                "status": "completed",
                "started_at": _local(base + 200),
                "completed_at": _local(base + 300),
            },
        },
        usage_rows=[
            {"ts": _utc(base + 50), "provider": "local", "model": "m",
             "prompt_tokens": 10, "completion_tokens": 100, "duration_ms": 1000.0},
            {"ts": _utc(base + 250), "provider": "local", "model": "m",
             "prompt_tokens": 20, "completion_tokens": 200, "duration_ms": 1000.0},
            # In the inter-phase gap → unattributed (never guessed).
            {"ts": _utc(base + 150), "provider": "local", "model": "m",
             "prompt_tokens": 30, "completion_tokens": 300, "duration_ms": 1000.0},
        ],
    )
    by_phase = progress_service.run_progress(run_id)["stats"]["detail"]["by_phase"]
    assert by_phase == [
        {"phase": "semantik_conversion", "calls": 1,
         "prompt_tokens": 10, "completion_tokens": 100},
        {"phase": "staging", "calls": 1,
         "prompt_tokens": 20, "completion_tokens": 200},
        {"phase": "unattributed", "calls": 1,
         "prompt_tokens": 30, "completion_tokens": 300},
    ]


def test_detail_explicit_phase_wins_over_checkpoint_window(state_dir):
    """Writer-stamped phase identity is not discarded by a stale time join."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-explicit-phase",
        workflow_id="WF-20260101-prog0021b",
        orch_run_id="TTC_prog_explicit_phase",
        checkpoints={},
        usage_rows=[
            {
                "ts": "2026-01-01T00:00:00+00:00",
                "provider": "local",
                "model": "model",
                "phase": "content_generation_outline",
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "duration_ms": 1000,
            },
            {
                "ts": "not-a-time",
                "provider": "local",
                "model": "model",
                "prompt_tokens": 30,
                "completion_tokens": 40,
                "duration_ms": 1000,
            },
        ],
    )
    by_phase = progress_service.run_progress(run_id)["stats"]["detail"]["by_phase"]
    assert by_phase == [
        {
            "phase": "content_generation_outline",
            "calls": 1,
            "prompt_tokens": 10,
            "completion_tokens": 20,
        },
        {
            "phase": "unattributed",
            "calls": 1,
            "prompt_tokens": 30,
            "completion_tokens": 40,
        },
    ]


def test_detail_omitted_without_any_source(state_dir):
    """No llm_usage.jsonl and no vram_trajectory.jsonl → no stats.detail."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-nodetail",
        workflow_id="WF-20260101-prog0022",
        orch_run_id="TTC_prog_nodetail",
        usage_rows=None,
    )
    assert "detail" not in progress_service.run_progress(run_id)["stats"]


def test_detail_vram_section_from_trajectory_file(state_dir):
    """vram_trajectory.jsonl present (real writer row shape,
    lib/llm/vram_doctor.py::append_trajectory_row) → latest sample + count;
    the unbounded resident_models list is not surfaced."""
    orch = "TTC_prog_vram"
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-vram",
        workflow_id="WF-20260101-prog0023",
        orch_run_id=orch,
        usage_rows=None,
    )
    run_dir = state_dir / "runs" / orch
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"run_id": orch, "phase": "semantik_conversion", "when": "before",
         "ts": "2026-01-01T00:00:00+00:00", "event": "phase_boundary",
         "free_mib": 100000, "total_mib": 120000, "probe_source": "nvml",
         "resident_models": [{"name": "big-model"}], "cuda_available": True},
        {"run_id": orch, "phase": "semantik_conversion", "when": "after",
         "ts": "2026-01-01T00:10:00+00:00", "event": "phase_boundary",
         "free_mib": 90000, "total_mib": 120000, "probe_source": "nvml",
         "resident_models": [], "cuda_available": True},
    ]
    (run_dir / "vram_trajectory.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    detail = progress_service.run_progress(run_id)["stats"]["detail"]
    vram = detail["vram"]
    assert vram["samples"] == 2
    assert vram["latest"]["free_mib"] == 90000
    assert vram["latest"]["total_mib"] == 120000
    assert vram["latest"]["when"] == "after"
    assert vram["latest"]["probe_source"] == "nvml"
    assert "resident_models" not in vram["latest"]
    # Usage file absent → the usage-derived sections are omitted alongside.
    assert "totals" not in detail


def test_stage_rail_component_renders_detail_disclosure():
    """Static-asset assertions for the collapsible detail matrix in the SHARED
    stage-rail component (native details/summary; hidden when the server
    omitted stats.detail; tables in a local overflow wrapper)."""
    js = (
        Path(__file__).resolve().parents[1]
        / "static"
        / "shared"
        / "components"
        / "stage-rail.js"
    ).read_text(encoding="utf-8")
    assert "Detailed stats" in js
    assert "detailHost.hidden = true" in js  # omission → hidden, no empty shell
    assert "stats || {}).detail" in js
    assert "by_model" in js and "by_phase" in js
    assert "stage-detail-scroll" in js  # local overflow, no page h-scroll
    assert "truncated (finish=length)" in js
    # The throughput trio, each explicitly labeled (aggregate vs per-stream
    # vs cumulative — the concurrency-misread fix).
    assert "window tok/s (aggregate)" in js
    assert "per-stream tok/s" in js
    assert "avg tok/s (cumulative)" in js
    css = (
        Path(__file__).resolve().parents[1]
        / "static"
        / "shared"
        / "components"
        / "components.css"
    ).read_text(encoding="utf-8")
    assert ".stage-detail-scroll { overflow-x: auto;" in css


def test_stage_rail_component_renders_phase_units():
    """Static-asset assertion (the established pattern for headless-only JS):
    the SHARED stage-rail component — used by both the normal and the degraded
    CLI-run progress paths — renders stats.phase_units."""
    js = (
        Path(__file__).resolve().parents[1]
        / "static"
        / "shared"
        / "components"
        / "stage-rail.js"
    ).read_text(encoding="utf-8")
    assert "s.phase_units" in js
    assert "phase_units.count" in js
    # Absence renders nothing (no fabricated 0) — the render is guarded.
    assert "if (s.phase_units && Number.isFinite(s.phase_units.count))" in js


def test_stage_rail_component_renders_streams_stat():
    """Static-asset assertion: the band renders the stats.streams in-flight
    estimate ("N in flight"), guarded so the server's omission renders
    nothing (never a fabricated 0)."""
    js = (
        Path(__file__).resolve().parents[1]
        / "static"
        / "shared"
        / "components"
        / "stage-rail.js"
    ).read_text(encoding="utf-8")
    assert "if (typeof s.streams === 'number')" in js
    assert "in flight" in js


# ----------------------------------------------------------------- endpoint


@pytest.fixture
def client(state_dir, libv2_root):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from gui.app import create_app  # noqa: PLC0415

    return TestClient(create_app())


def test_endpoint_happy_path(client, state_dir):
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-http",
        workflow_id="WF-20260101-prog0007",
        orch_run_id="TTC_prog_http",
        phase_outputs={"semantik_conversion": {"_completed": True}},
    )
    resp = client.get(f"/api/runs/{run_id}/progress")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id
    assert body["workflow"] == "textbook_to_course"
    assert {"name", "index", "state", "group", "wallclock_s"} <= set(
        body["phases"][0].keys()
    )
    assert body["current_phase"] == "heading_judge"
    assert "updated_at" in body and "stats" in body


def test_endpoint_unknown_run_404(client):
    resp = client.get("/api/runs/GUI-no-such-run/progress")
    assert resp.status_code == 404
    assert resp.json() == {"error": "unknown_run", "detail": "GUI-no-such-run"}


def test_endpoint_output_tail(client, state_dir, tmp_path):
    """GET /api/runs/{id}/output-tail: 200 with the tail payload for a known
    run; typed 404 for an unknown one (mirrors /progress)."""
    export_dir = tmp_path / "export"
    sidecar = export_dir / "01_outline" / ".blocks_outline_checkpoint.jsonl"
    _append_content_rows(
        sidecar,
        [{"block_id": "blk-http", "sequence": 0, "content": "<p>hello</p>"}],
    )
    run_id = _seed_run(
        state_dir,
        run_id="GUI-tail-http",
        workflow_id="WF-20260101-tail0007",
        orch_run_id="TTC_tail_http",
        phase_outputs=_two_pass_outputs_through_content_generation(export_dir),
    )
    resp = client.get(f"/api/runs/{run_id}/output-tail")
    assert resp.status_code == 200
    body = resp.json()
    assert body["phase"] == "content_generation_outline"
    assert body["rows"] == [{"seq": 0, "label": "blk-http", "text": "hello"}]

    resp = client.get("/api/runs/GUI-no-such-run/output-tail")
    assert resp.status_code == 404
    assert resp.json() == {"error": "unknown_run", "detail": "GUI-no-such-run"}


# --------------------------------------------- course/book identity (header)
# The build-page header displays the run's course/book name. The workflow
# state's params are the FRESHEST source — an --auto-name run starts under a
# provisional slug and workflow_runner._maybe_apply_auto_name REBINDS
# params.course_name (+ display_title) mid-run, persisting back into
# state/workflows/<WF-id>.json — while the GUI run record keeps its
# creation-time name. run_progress must surface the live name (never-raise;
# unknown → None so the client omits the element).


def test_course_name_prefers_workflow_params_over_record(state_dir):
    """The rebound params.course_name wins over the record's creation name."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-cname-0001",
        workflow_id="WF-20260101-cname001",
        orch_run_id="TTC_cname_0001",
        params={"course_name": "tst-rebound-20260101-0000"},
    )
    payload = progress_service.run_progress(run_id)
    assert payload["course_name"] == "tst-rebound-20260101-0000"


def test_course_name_falls_back_to_record(state_dir):
    """No params.course_name → the GUI record's creation-time name."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-cname-0002",
        workflow_id="WF-20260101-cname002",
        orch_run_id="TTC_cname_0002",
    )
    payload = progress_service.run_progress(run_id)
    assert payload["course_name"] == "PHYS_101"  # the record's name


def test_course_name_updates_when_auto_name_rebinds(state_dir):
    """A mid-run rebind (workflow JSON rewritten) shows up on the next poll."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-cname-0003",
        workflow_id="WF-20260101-cname003",
        orch_run_id="TTC_cname_0003",
        params={"course_name": "tst-provisional-0003"},
    )
    assert (
        progress_service.run_progress(run_id)["course_name"]
        == "tst-provisional-0003"
    )
    # Simulate _maybe_apply_auto_name persisting the rebound identity.
    wf_path = state_dir / "workflows" / "WF-20260101-cname003.json"
    doc = json.loads(wf_path.read_text(encoding="utf-8"))
    doc["params"]["provisional_course_name"] = doc["params"]["course_name"]
    doc["params"]["course_name"] = "tst-final-0003"
    doc["params"]["display_title"] = "Test Final Title"
    wf_path.write_text(json.dumps(doc), encoding="utf-8")
    payload = progress_service.run_progress(run_id)
    assert payload["course_name"] == "tst-final-0003"
    assert payload["display_title"] == "Test Final Title"


def test_course_name_none_when_unknown_cli_run(state_dir):
    """A CLI-observed run whose params carry no course_name → None (omitted
    client-side), never a fabricated name. display_title likewise."""
    wf_id = "WF-20260101-cname004"
    wf_dir = state_dir / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{wf_id}.json").write_text(
        json.dumps(
            {
                "id": wf_id,
                "type": "textbook_to_course",
                "status": "RUNNING",
                "started_at": "2026-01-01T00:00:00",
                "params": {"run_id": "TTC_cname_0004"},
                "phase_outputs": {},
            }
        ),
        encoding="utf-8",
    )
    payload = progress_service.run_progress(wf_id)
    assert payload is not None
    assert payload["course_name"] is None
    assert payload["display_title"] is None


def test_display_title_blank_or_nonstring_is_none(state_dir):
    """Garbage display_title values degrade to None (parse-with-fallback)."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-cname-0005",
        workflow_id="WF-20260101-cname005",
        orch_run_id="TTC_cname_0005",
        params={"course_name": "TST_105", "display_title": "   "},
    )
    payload = progress_service.run_progress(run_id)
    assert payload["course_name"] == "TST_105"
    assert payload["display_title"] is None


# ============================================ seat-swap / cold-start activity
#
# The stage tracker attributes only a phase's OWN compute time, so a long vLLM
# seat swap AT a phase boundary (the schedule tears down the conversion seats
# and COLD-STARTS the next seat — up to ~10 min) renders as the phase hanging.
# `stats.seat_activity` names that wait WITHOUT misattributing it as phase
# compute. Derived purely from the shared cached GLOBAL seat_overview() probe
# (monkeypatched here — no network, no docker, no recursion).
#
# `heading_judge` is the concrete incident phase and declares `seats:
# [spark-super]` in the real config/workflows.yaml, so seeding a run parked on
# it exercises the true wiring.

from gui.services import seat_service  # noqa: E402

_SUPER = "spark-super"  # the real heading_judge seat (config-declared)


def _global_overview(seats):
    """A GLOBAL-shape seat_overview payload (what _seat_activity consumes)."""
    return {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "seats": list(seats),
        "registry_configured": True,
        "docker_available": True,
    }


def _seat(name, state, *, since_ms=None, since=None, container="vllm-super"):
    return {
        "name": name,
        "base_url": "http://localhost:8001/v1",
        "live": state == "live",
        "state": state,
        "container": container,
        "model": "some-model" if state == "live" else None,
        "since": since,
        "since_ms": since_ms,
    }


def _run_parked_on_heading_judge(state_dir, run_id, orch):
    """A running textbook_to_course parked on heading_judge (seats:[spark-super])."""
    return _seed_run(
        state_dir,
        run_id=run_id,
        workflow_id=f"WF-{run_id}",
        orch_run_id=orch,
        phase_outputs={"semantik_conversion": {"_completed": True}},
    )


def test_seat_activity_present_when_needed_seat_loading(state_dir, monkeypatch):
    """A COLD-STARTING declared seat surfaces as patient (concern=False)."""
    monkeypatch.setattr(
        seat_service,
        "seat_overview",
        lambda *a, **k: _global_overview([_seat(_SUPER, "loading", since_ms=568_000)]),
    )
    run_id = _run_parked_on_heading_judge(state_dir, "GUI-sa-load", "TTC_sa_load")
    payload = progress_service.run_progress(run_id)
    assert payload["current_phase"] == "heading_judge"
    act = payload["stats"]["seat_activity"]
    assert act["phase"] == "heading_judge"
    assert act["concern"] is False  # loading is the ~9-min PATIENT cold start
    assert len(act["seats"]) == 1
    s = act["seats"][0]
    assert s["seat"] == _SUPER
    assert s["state"] == "loading"
    assert s["concern"] is False
    assert s["elapsed_ms"] == 568_000  # the swap age, carried honestly


def test_seat_activity_absent_when_all_needed_seats_live(state_dir, monkeypatch):
    """Every needed seat live → the field is OMITTED (normal phases unaffected)."""
    monkeypatch.setattr(
        seat_service,
        "seat_overview",
        lambda *a, **k: _global_overview([_seat(_SUPER, "live", since_ms=1000)]),
    )
    run_id = _run_parked_on_heading_judge(state_dir, "GUI-sa-live", "TTC_sa_live")
    payload = progress_service.run_progress(run_id)
    assert payload["current_phase"] == "heading_judge"
    assert "seat_activity" not in payload["stats"]


def test_run_owned_seat_overlay_recognizes_live_standalone_seat(
    state_dir, monkeypatch
):
    """A CLI runner's attributed registry overrides the GUI's stale env."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-seat-run-overlay",
        workflow_id="WF-20260101-seat-overlay",
        orch_run_id="TTC_seat_run_overlay",
        phase_outputs={"semantik_conversion": {"_completed": True}},
    )
    monkeypatch.setattr(
        progress_service,
        "_active_run_seat_registry",
        lambda *_args: {_SUPER: "http://runner-seat.invalid:8123"},
    )
    monkeypatch.setattr(
        progress_service,
        "_probe_seat_model",
        lambda url: "served-snapshot" if "runner-seat" in url else None,
    )
    monkeypatch.setattr(
        "gui.services.seat_service.seat_overview",
        lambda: {"seats": []},
    )
    payload = progress_service.run_progress(run_id)
    assert payload["stats"]["seat"] == {
        "name": _SUPER,
        "url": "http://runner-seat.invalid:8123",
        "model": "served-snapshot",
    }
    assert "seat_activity" not in payload["stats"]


def test_seat_activity_never_mutates_phase_durations(state_dir, monkeypatch):
    """The 9-min swap must NOT bleed into any phase's compute wall-clock.

    The concrete incident: heading_judge logged ~2s of real work but ~9.5min of
    wall-clock elapsed during the cold start. The done phase keeps its true 2s
    compute bar; the swap age lives ONLY under stats.seat_activity.
    """
    monkeypatch.setattr(
        seat_service,
        "seat_overview",
        lambda *a, **k: _global_overview([_seat(_SUPER, "loading", since_ms=568_000)]),
    )
    run_id = _seed_run(
        state_dir,
        run_id="GUI-sa-nomut",
        workflow_id="WF-sa-nomut",
        orch_run_id="TTC_sa_nomut",
        phase_outputs={"semantik_conversion": {"_completed": True}},
        checkpoints={
            "semantik_conversion": {
                "status": "completed",
                "started_at": "2026-01-01T00:00:00",
                "completed_at": "2026-01-01T00:00:02",  # a true 2s compute span
            },
        },
    )
    payload = progress_service.run_progress(run_id)
    conv = _phase(payload, "semantik_conversion")
    assert conv["wallclock_s"] == 2.0  # unchanged by the seat swap
    judge = _phase(payload, "heading_judge")
    # The current phase carries NO fabricated compute bar for the swap.
    assert judge.get("wallclock_s") is None
    # seat_activity is a SEPARATE stats field — never a phase attribute.
    assert "seat_activity" not in judge
    assert payload["stats"]["seat_activity"]["seats"][0]["elapsed_ms"] == 568_000


def test_seat_activity_down_seat_is_alarming_mismatch(state_dir, monkeypatch):
    """A DOWN needed seat is a concern (container not running), unlike loading."""
    monkeypatch.setattr(
        seat_service,
        "seat_overview",
        lambda *a, **k: _global_overview([_seat(_SUPER, "down", since_ms=4000)]),
    )
    run_id = _run_parked_on_heading_judge(state_dir, "GUI-sa-down", "TTC_sa_down")
    act = progress_service.run_progress(run_id)["stats"]["seat_activity"]
    assert act["concern"] is True
    assert act["seats"][0]["state"] == "down"
    assert act["seats"][0]["concern"] is True


def test_seat_activity_unregistered_named_seat_is_alarming(state_dir, monkeypatch):
    """A phase naming a seat absent from the registry is a loud mismatch."""
    monkeypatch.setattr(
        seat_service,
        "seat_overview",
        lambda *a, **k: _global_overview([_seat("some-other-seat", "live")]),
    )
    run_id = _run_parked_on_heading_judge(state_dir, "GUI-sa-unreg", "TTC_sa_unreg")
    act = progress_service.run_progress(run_id)["stats"]["seat_activity"]
    assert act["concern"] is True
    s = act["seats"][0]
    assert s["seat"] == _SUPER and s["state"] == "unregistered" and s["concern"] is True


def test_seat_activity_unknown_state_is_alarming_not_patient(state_dir, monkeypatch):
    """`unknown` (docker unavailable) is NOT patient — only loading is."""
    monkeypatch.setattr(
        seat_service,
        "seat_overview",
        lambda *a, **k: _global_overview([_seat(_SUPER, "unknown")]),
    )
    run_id = _run_parked_on_heading_judge(state_dir, "GUI-sa-unk", "TTC_sa_unk")
    act = progress_service.run_progress(run_id)["stats"]["seat_activity"]
    assert act["concern"] is True
    assert act["seats"][0]["state"] == "unknown"


def test_seat_activity_never_raises_when_seat_overview_errors(state_dir, monkeypatch):
    """A raising / degraded seat probe never breaks the progress payload."""
    def _boom(*_a, **_k):
        raise RuntimeError("seat probe exploded")

    monkeypatch.setattr(seat_service, "seat_overview", _boom)
    run_id = _run_parked_on_heading_judge(state_dir, "GUI-sa-boom", "TTC_sa_boom")
    payload = progress_service.run_progress(run_id)
    assert payload is not None
    assert payload["current_phase"] == "heading_judge"
    assert "seat_activity" not in payload["stats"]  # degraded → omitted, never raised


def test_seat_activity_absent_for_a_terminal_run(state_dir, monkeypatch):
    """A finished run does no seat probing and carries no seat_activity."""
    calls = []
    monkeypatch.setattr(
        seat_service, "seat_overview", lambda *a, **k: calls.append(1) or _global_overview([])
    )
    run_id = _seed_run(
        state_dir,
        run_id="GUI-sa-term",
        workflow_id="WF-sa-term",
        orch_run_id="TTC_sa_term",
        gui_status="completed",
        wf_status="COMPLETED",
        phase_outputs={"semantik_conversion": {"_completed": True}},
    )
    payload = progress_service.run_progress(run_id)
    assert payload["stats"]["seat"] is None
    assert "seat_activity" not in payload["stats"]
    assert calls == []  # terminal → the seat branch is skipped entirely


# ------------------------------------------------------- _seat_activity (pure)
def test_seat_activity_helper_no_phase_or_no_seats_is_none(monkeypatch):
    """No current phase, or a phase declaring no seats → None (never probes)."""
    probed = []
    monkeypatch.setattr(
        seat_service, "seat_overview", lambda *a, **k: probed.append(1) or _global_overview([])
    )
    assert progress_service._seat_activity(None, [_SUPER]) is None
    assert progress_service._seat_activity("heading_judge", []) is None
    assert probed == []  # short-circuits before any probe


def test_seat_activity_helper_mixed_seats_reports_each(monkeypatch):
    """A phase needing two seats surfaces every not-live one, concern OR'd."""
    monkeypatch.setattr(
        seat_service,
        "seat_overview",
        lambda *a, **k: _global_overview(
            [_seat("seat-a", "loading", since_ms=1000), _seat("seat-b", "down")]
        ),
    )
    act = progress_service._seat_activity("some_phase", ["seat-a", "seat-b"])
    assert [s["seat"] for s in act["seats"]] == ["seat-a", "seat-b"]
    assert [s["concern"] for s in act["seats"]] == [False, True]
    assert act["concern"] is True  # any alarming seat raises the top-level flag


def test_stage_run_keeps_two_pass_tiers_visible(state_dir):
    """A courseforge-* stage run skips BOTH branch sides — hide NEITHER tier.

    Regression: the env-branch hiding rule hid any branchy row that resolved
    "skipped". Its premise is "the branch didn't run — its COMPLEMENT did",
    but a ``courseforge-rewrite`` stage subcommand skips phases via the
    ``courseforge_stage`` whitelist, so both sides skip and the two-pass
    tiers vanished from the rail entirely (observed live: 18 phases instead
    of 20, no content_generation_outline / inter_tier_validation).
    """
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-stagerun",
        workflow_id="WF-20260101-prog0040",
        orch_run_id="TTC_stagerun",  # synthetic run id, slug-guard: allow
        phase_outputs={
            **{
                name: {"_completed": True}
                for name in (
                    "semantik_conversion", "heading_judge", "staging",
                    "chunking", "objective_extraction", "source_mapping",
                    "course_planning", "concept_extraction",
                )
            },
            # The stage whitelist skipped BOTH sides of the env branch.
            "content_generation": {"_skipped": True},
            "content_generation_outline": {"_skipped": True},
            "inter_tier_validation": {"_skipped": True},
            # ...only the rewrite tier actually ran.
            "content_generation_rewrite": {"_completed": True},
        },
    )
    payload = progress_service.run_progress(run_id)
    names = [p["name"] for p in payload["phases"]]
    for tier in ("content_generation_outline", "inter_tier_validation"):
        assert tier in names, (
            f"{tier} was skipped by the stage whitelist, NOT by the env "
            "branch — its complement never ran, so it must stay on the rail"
        )
        assert _phase(payload, tier)["group"] == "generation"
    # The single-pass fallback row stays hidden — under two-pass it is noise.
    assert "content_generation" not in names


def test_single_pass_run_still_hides_two_pass_tiers(state_dir):
    """The original contract holds: complement RAN -> hide the not-taken side."""
    run_id = _seed_run(
        state_dir,
        run_id="GUI-prog-singlepass2",
        workflow_id="WF-20260101-prog0041",
        orch_run_id="TTC_singlepass2",  # synthetic run id, slug-guard: allow
        phase_outputs={
            **{
                name: {"_completed": True}
                for name in (
                    "semantik_conversion", "heading_judge", "staging",
                    "chunking", "objective_extraction", "source_mapping",
                    "course_planning", "concept_extraction",
                )
            },
            "content_generation": {"_completed": True},
            "content_generation_outline": {"_skipped": True},
            "inter_tier_validation": {"_skipped": True},
        },
    )
    payload = progress_service.run_progress(run_id)
    names = [p["name"] for p in payload["phases"]]
    assert _phase(payload, "content_generation")["state"] == "done"
    for tier in ("content_generation_outline", "inter_tier_validation"):
        assert tier not in names, f"{tier} not-taken and complement ran — hide"

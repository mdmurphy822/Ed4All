"""Seat-schedule enforcement tests for MCP.core.workflow_runner.

Covers:
  * ``_phase_seats`` workflow-scoped lookup (no cross-workflow name collision).
  * Boundary transitions computed correctly across the REAL two-pass
    textbook_to_course phase order (mocked docker/probe): the Super seat starts
    exactly ONCE and is never cold-restarted between rewrite and validation
    (the owner's 4-7x-starvation fix).
  * A coherence-probe failure fails the phase LOUDLY
    (``SeatScheduleProbeError``), never silent.
  * Flag OFF → pure no-op (no docker/probe calls; seat state carried unchanged).

NO live docker / NO live network — the seat primitives are monkeypatched.
"""

from __future__ import annotations

import asyncio

import pytest

import lib.vllm_container_lifecycle as vcl
from MCP.core.config import OrchestratorConfig
from MCP.core.workflow_runner import (
    SeatScheduleProbeError,
    WorkflowRunner,
    _phase_seats,
)

_SEATS = (
    "spark-super=http://localhost:8001,"
    "spark-glm=http://localhost:8002,"
    "spark-qwen=http://localhost:8003"
)
_CONTAINERS = (
    "http://localhost:8001=vllm-super,"
    "http://localhost:8002=vllm-glm,"
    "http://localhost:8003=vllm-qwen"
)


@pytest.fixture(autouse=True)
def _reset_warn_flags():
    vcl._REGISTRY_WARNED = False
    vcl._SEAT_REGISTRY_WARNED = False
    yield


@pytest.fixture
def _seat_env(monkeypatch):
    monkeypatch.setenv(vcl.ENV_SEAT_SCHEDULE, "1")
    monkeypatch.setenv(vcl.ENV_SEAT_BASE_URLS, _SEATS)
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, _CONTAINERS)


def _runner():
    r = WorkflowRunner.__new__(WorkflowRunner)

    class _Ex:
        run_id = "seat-test-run"

    r.executor = _Ex()
    return r


# --------------------------------------------------------------------------
# _phase_seats — workflow-scoped, no cross-workflow name collision
# --------------------------------------------------------------------------
def test_phase_seats_workflow_scoped():
    # assessment_synthesis exists in BOTH course_generation and
    # textbook_to_course; the textbook one carries [spark-super].
    assert _phase_seats("textbook_to_course", "assessment_synthesis") == ["spark-super"]
    assert _phase_seats("textbook_to_course", "post_rewrite_validation") == []
    assert _phase_seats("textbook_to_course", "course_planning") == ["spark-super"]
    assert _phase_seats("textbook_to_course", "semantik_conversion") == [
        "spark-glm",
        "spark-qwen",
    ]
    # A deterministic middle phase declares no opinion.
    assert _phase_seats("textbook_to_course", "staging") is None
    assert _phase_seats("textbook_to_course", "chunking") is None


def _two_pass_running_order():
    """The two-pass textbook_to_course dispatch order, minus the single-pass
    ``content_generation`` phase (skipped at runtime under COURSEFORGE_TWO_PASS)."""
    cfg = OrchestratorConfig.load()
    wf = cfg.get_workflow("textbook_to_course")
    r = WorkflowRunner.__new__(WorkflowRunner)
    ordered = WorkflowRunner._topological_sort(r, wf.phases)
    return [p.name for p in ordered if p.name != "content_generation"]


# --------------------------------------------------------------------------
# Boundary transitions across the real two-pass order
# --------------------------------------------------------------------------
def test_seat_transitions_across_two_pass_order(monkeypatch, _seat_env):
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    starts, stops = [], []
    monkeypatch.setattr(vcl, "start_seat", lambda seat, **k: starts.append(seat) or 1.0)
    monkeypatch.setattr(vcl, "stop_seat", lambda seat: stops.append(seat) or True)
    monkeypatch.setattr(vcl, "coherence_probe", lambda base_url, **k: True)

    order = _two_pass_running_order()
    seat_state = None
    for name in order:
        desired = _phase_seats("textbook_to_course", name)
        if desired is None:
            continue  # no opinion — enforcement carries state forward
        seat_state = WorkflowRunner._apply_seat_schedule_blocking(
            name, set(desired), seat_state, None
        )

    # Super starts EXACTLY once (at course_planning) and is never cold-restarted
    # for assessment after validation — the whole point of the schedule.
    assert starts.count("spark-super") == 1
    assert starts.count("spark-glm") == 1
    assert starts.count("spark-qwen") == 1
    # Super is stopped when the seat-free validation range begins.
    assert "spark-super" in stops
    # The conversion seats are swapped out exactly once (at course_planning).
    assert stops.count("spark-glm") == 1
    assert stops.count("spark-qwen") == 1
    # Final seat-state is empty (post_rewrite..finalization declare []).
    assert seat_state == set()


def test_first_phase_clean_start_stops_undesired_registered_seats(monkeypatch, _seat_env):
    """seat_state=None (fresh run): the first scheduled phase stops every
    REGISTERED seat not desired before starting its own."""
    starts, stops = [], []
    monkeypatch.setattr(vcl, "start_seat", lambda seat, **k: starts.append(seat) or 1.0)
    monkeypatch.setattr(vcl, "stop_seat", lambda seat: stops.append(seat) or True)
    monkeypatch.setattr(vcl, "coherence_probe", lambda base_url, **k: True)

    # First phase wants only the glm/qwen conversion seats.
    new = WorkflowRunner._apply_seat_schedule_blocking(
        "semantik_conversion", {"spark-glm", "spark-qwen"}, None, None
    )
    assert new == {"spark-glm", "spark-qwen"}
    assert set(starts) == {"spark-glm", "spark-qwen"}
    # spark-super is registered but not desired → stopped on the clean start.
    assert stops == ["spark-super"]


# --------------------------------------------------------------------------
# Coherence-probe failure fails the phase LOUDLY
# --------------------------------------------------------------------------
def test_probe_failure_raises_loud(monkeypatch, _seat_env):
    monkeypatch.setattr(vcl, "start_seat", lambda seat, **k: 2.0)
    monkeypatch.setattr(vcl, "stop_seat", lambda seat: True)
    monkeypatch.setattr(vcl, "coherence_probe", lambda base_url, **k: False)
    with pytest.raises(SeatScheduleProbeError) as ei:
        WorkflowRunner._apply_seat_schedule_blocking(
            "course_planning", {"spark-super"}, {"spark-glm"}, None
        )
    assert "coherence probe" in str(ei.value).lower()


def test_start_failure_raises_loud(monkeypatch, _seat_env):
    monkeypatch.setattr(vcl, "start_seat", lambda seat, **k: None)  # could not come up
    monkeypatch.setattr(vcl, "stop_seat", lambda seat: True)
    monkeypatch.setattr(vcl, "coherence_probe", lambda base_url, **k: pytest.fail("probed"))
    with pytest.raises(SeatScheduleProbeError):
        WorkflowRunner._apply_seat_schedule_blocking(
            "course_planning", {"spark-super"}, {"spark-glm"}, None
        )


def test_unmapped_seat_is_config_gap_not_loud(monkeypatch):
    """A declared seat missing from ED4ALL_SEAT_BASE_URLS is a config gap:
    warn + skip, never a run-failing raise."""
    monkeypatch.setenv(vcl.ENV_SEAT_SCHEDULE, "1")
    monkeypatch.setenv(vcl.ENV_SEAT_BASE_URLS, "")  # nothing mapped
    monkeypatch.setenv(vcl.ENV_VLLM_CONTAINERS, "")
    monkeypatch.setattr(vcl, "start_seat", lambda seat, **k: pytest.fail("started"))
    monkeypatch.setattr(vcl, "coherence_probe", lambda base_url, **k: pytest.fail("probed"))
    # Does not raise; returns the desired set.
    new = WorkflowRunner._apply_seat_schedule_blocking(
        "course_planning", {"spark-super"}, None, None
    )
    assert new == {"spark-super"}


# --------------------------------------------------------------------------
# Flag OFF → pure no-op (async wrapper carries state unchanged, no docker)
# --------------------------------------------------------------------------
def test_flag_off_is_noop(monkeypatch):
    monkeypatch.delenv(vcl.ENV_SEAT_SCHEDULE, raising=False)
    monkeypatch.setattr(vcl, "start_seat", lambda *a, **k: pytest.fail("started"))
    monkeypatch.setattr(vcl, "stop_seat", lambda *a, **k: pytest.fail("stopped"))
    r = _runner()
    prior = {"spark-glm"}
    out = asyncio.run(
        r._apply_seat_schedule("textbook_to_course", "course_planning", prior)
    )
    assert out is prior  # unchanged, byte-identical carry-through


def test_no_opinion_phase_carries_state(monkeypatch, _seat_env):
    monkeypatch.setattr(vcl, "start_seat", lambda *a, **k: pytest.fail("started"))
    monkeypatch.setattr(vcl, "stop_seat", lambda *a, **k: pytest.fail("stopped"))
    r = _runner()
    prior = {"spark-glm", "spark-qwen"}
    out = asyncio.run(
        r._apply_seat_schedule("textbook_to_course", "staging", prior)
    )
    assert out is prior  # no ``seats`` key → carry state forward, no docker

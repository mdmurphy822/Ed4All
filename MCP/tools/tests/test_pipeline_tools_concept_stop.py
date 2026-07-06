"""concept_extraction Stage-3 window-loop graceful-stop invariants.

The ``concept_extraction`` Stage-3 per-window ``synthesize_concepts`` loop must
checkpoint on command: an armed stop sentinel makes it persist every COMPLETED
window and raise ``GracefulStopRequested`` (which the Stage-3 caller re-raises
AS-IS rather than swallowing it into the "best-effort" empty-seed fallback), so
resume re-runs only the un-attempted windows.

Same two mechanisms as Stage-2 (``test_pipeline_tools_stage2_stop.py``):
between-batches ``check_stop`` (real sentinel) and the P3.0 in-flight
``STOP_MARKER`` return (blocker-#1: ``sidecar records == provider calls``).

Hermetic: a fake ``TextbookSynthesisProvider`` (no GPU / ollama / network),
driven with an empty staging dir (one chapter_text window per chapter). Sentinel
isolation via ``state_runs_isolated`` + a synthetic ``ED4ALL_RUN_ID``. No
hardcoded course slug — the slug is derived from a test-constant course name.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MCP.tools.pipeline_tools as pt  # noqa: E402
from lib.generation import stop_control  # noqa: E402
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402

_RUN_ID = "STOP_CONCEPT_TESTRUN"
_COURSE = "CONCEPT_STOP"


# ---------------------------------------------------------------------------
# Fake provider (constructed internally as TextbookSynthesisProvider(capture=)).
# Per-run knobs + the call log live at CLASS level, reset by the autouse fixture.
# ---------------------------------------------------------------------------
class _FakeConceptProvider:
    model: str = "test-model-v1"
    batch_size: int = 10
    calls: List[str] = []
    #: Arm the REAL run-scoped sentinel after this many calls (0 = never).
    arm_after: int = 0

    def __init__(self, *, capture: Any = None) -> None:
        self._model = type(self).model
        self._provider = "local"
        self._max_tokens = 4096

    def batch_chapters(self, items, batch_size: int = 10):
        bs = type(self).batch_size
        return [items[i:i + bs] for i in range(0, len(items), bs)]

    def synthesize_concepts(self, spec, *, course_name):
        cls = type(self)
        cid = str(spec.get("id") or "")
        widx = spec.get("window_index")
        cls.calls.append(f"{cid}#w{widx}")
        if cls.arm_after and len(cls.calls) == cls.arm_after:
            stop_control.request_stop(scope="run", reason="test", source="test")
        return {
            "concepts": [
                {
                    "canonical": f"Concept {cid} window {widx}",
                    "aliases": [f"alias {cid}"],
                    "chapter_ids": [cid],
                    "definition_hint": f"hint for {cid}",
                }
            ]
        }


class _GatedStopProbe:
    """Deterministic in-flight probe: first ``false_for`` → False, rest → True."""

    def __init__(self, false_for: int) -> None:
        self.false_for = false_for
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, run_id: Optional[str] = None) -> bool:
        with self._lock:
            self.calls += 1
            return self.calls > self.false_for


@pytest.fixture(autouse=True)
def _reset_fake_provider():
    _FakeConceptProvider.model = "test-model-v1"
    _FakeConceptProvider.batch_size = 10
    _FakeConceptProvider.calls = []
    _FakeConceptProvider.arm_after = 0
    yield


@pytest.fixture
def _armed_env(state_runs_isolated, monkeypatch):
    monkeypatch.setenv("ED4ALL_RUN_ID", _RUN_ID)
    stop_control.clear_stop(include_global=True)
    yield
    stop_control.clear_stop(include_global=True)


def _sidecar_for(tmp_path: Path) -> Path:
    slug = _COURSE.lower().replace("_", "-").replace(" ", "-")
    return (
        tmp_path / "libv2" / "courses" / slug / "concept_graph"
        / pt._CONCEPT_EXTRACTION_CHECKPOINT_NAME
    )


def _run_phase(tmp_path: Path, monkeypatch, *, batch_size: int = 10):
    """Drive run_concept_extraction through Stage-3, hermetically."""
    import Courseforge.generators._textbook_synthesis_provider as _tsp

    monkeypatch.setattr(_tsp, "TextbookSynthesisProvider", _FakeConceptProvider)
    monkeypatch.setenv("TEXTBOOK_SYNTHESIS_PROVIDER", "local")

    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)  # empty → no chunks → chapter_text windows

    ts_path = tmp_path / "textbook_structure.json"
    ts_path.write_text(
        json.dumps({
            "chapters": [
                {"id": "ch1", "chapter_text": "Alpha chapter body."},
                {"id": "ch2", "chapter_text": "Bravo chapter body."},
            ]
        }),
        encoding="utf-8",
    )

    _FakeConceptProvider.batch_size = batch_size
    registry = pt._build_tool_registry()
    tool = registry["run_concept_extraction"]
    result = asyncio.run(tool(
        project_id="",
        course_name=_COURSE,
        staging_dir=str(staging),
        textbook_structure_path=str(ts_path),
        libv2_root=str(tmp_path / "libv2"),
    ))
    return json.loads(result)


# ---------------------------------------------------------------------------
# (1) between-batches: stop after call N → exactly N calls + N sidecar records
# ---------------------------------------------------------------------------
def test_stop_between_batches_exact_n(tmp_path, monkeypatch, _armed_env):
    _FakeConceptProvider.arm_after = 1
    with pytest.raises(GracefulStopRequested):
        _run_phase(tmp_path, monkeypatch, batch_size=1)

    # Batch 2's loop-top check_stop raised: exactly 1 window on disk.
    assert _FakeConceptProvider.calls == ["ch1#w0"]
    records = pt._load_concept_windows_checkpoint(_sidecar_for(tmp_path))
    assert len(records) == 1


# ---------------------------------------------------------------------------
# (2) resume after a stop → only the un-attempted window re-runs
# ---------------------------------------------------------------------------
def test_resume_after_stop_completes(tmp_path, monkeypatch, _armed_env):
    _FakeConceptProvider.arm_after = 1
    with pytest.raises(GracefulStopRequested):
        _run_phase(tmp_path, monkeypatch, batch_size=1)
    assert _FakeConceptProvider.calls == ["ch1#w0"]
    assert len(pt._load_concept_windows_checkpoint(_sidecar_for(tmp_path))) == 1

    # Resume: clear the sentinel, rerun → only ch2 dispatches; total == 2.
    stop_control.clear_stop(include_global=True)
    _FakeConceptProvider.calls = []
    _FakeConceptProvider.arm_after = 0
    payload = _run_phase(tmp_path, monkeypatch, batch_size=1)
    assert payload["success"] is True, payload
    assert _FakeConceptProvider.calls == ["ch2#w0"]  # only the missing window
    assert payload["concept_windows_reused_from_checkpoint"] == 1
    assert len(pt._load_concept_windows_checkpoint(_sidecar_for(tmp_path))) == 2


# ---------------------------------------------------------------------------
# (3) pre-armed sentinel → zero provider calls
# ---------------------------------------------------------------------------
def test_pre_armed_sentinel_zero_calls(tmp_path, monkeypatch, _armed_env):
    stop_control.request_stop(scope="run", reason="test", source="test")
    with pytest.raises(GracefulStopRequested):
        _run_phase(tmp_path, monkeypatch)
    assert _FakeConceptProvider.calls == []
    assert not _sidecar_for(tmp_path).exists()


# ---------------------------------------------------------------------------
# (4) P3.0 marker pattern (blocker-#1): completed in-flight window IS persisted
# ---------------------------------------------------------------------------
def test_marker_pattern_in_flight_completed_persisted(
    tmp_path, monkeypatch, _armed_env
):
    gate = _GatedStopProbe(false_for=1)
    # _one_window probes pt.stop_requested (in-flight refusal); the batch
    # loop-top check_stop uses the REAL (un-armed) sentinel, so the raise comes
    # from the post-gather marker sweep.
    monkeypatch.setattr(pt, "stop_requested", gate)

    with pytest.raises(GracefulStopRequested):
        _run_phase(tmp_path, monkeypatch)  # one batch of 2 windows

    # Exactly the 1 gate-passed window completed AND is on disk.
    assert len(_FakeConceptProvider.calls) == 1
    records = pt._load_concept_windows_checkpoint(_sidecar_for(tmp_path))
    assert len(records) == len(_FakeConceptProvider.calls) == 1

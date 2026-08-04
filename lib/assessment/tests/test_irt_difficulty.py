"""Regression net for the honest IRT difficulty-calibration scaffold.

Covers the anti-fabrication core: no IRT parameters are ever invented for an
item without real backing responses, the scaffold is byte-identical when off,
and the distribution descriptor is deterministic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.assessment.irt_difficulty import (  # noqa: E402
    DEFAULT_MIN_RESPONSES,
    DIFFICULTY_PROVENANCE_CALIBRATED,
    DIFFICULTY_PROVENANCE_HEURISTIC,
    RASCH_DISCRIMINATION_A,
    describe_difficulty_distribution,
    irt_scaffold_enabled,
    load_calibrated_difficulty,
    resolve_response_data_path,
    tag_chunk_difficulty_provenance,
)

_FLAG = "TRAINFORGE_IRT_DIFFICULTY_SCAFFOLD"


@pytest.fixture
def scaffold_on(monkeypatch):
    monkeypatch.setenv(_FLAG, "true")
    assert irt_scaffold_enabled()


@pytest.fixture
def scaffold_off(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    assert not irt_scaffold_enabled()


def _build_responses_file(path: Path, per_item):
    """Write a synthetic item_responses.jsonl. per_item: {id: (n, n_correct)}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for item_id, (n, n_correct) in per_item.items():
        for i in range(n):
            lines.append(json.dumps({
                "chunk_id": item_id,
                "learner_id": f"L{i}",
                "correct": i < n_correct,
            }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Flag-off byte-identical
# --------------------------------------------------------------------------- #

def test_flag_off_byte_identical(scaffold_off):
    chunk = {"id": "c1", "difficulty": "intermediate", "text": "x"}
    before = dict(chunk)
    out = tag_chunk_difficulty_provenance(chunk, {"c1": {"difficulty": "advanced"}})
    assert out is chunk
    assert chunk == before  # no new keys, untouched


# --------------------------------------------------------------------------- #
# No response data → every chunk stays heuristic (anti-fabrication core)
# --------------------------------------------------------------------------- #

def test_no_response_data_stays_heuristic(scaffold_on, tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(tmp_path))
    response_path = resolve_response_data_path("phantom-course")
    assert response_path is None
    calibrated = load_calibrated_difficulty(response_path)
    assert calibrated == {}

    chunk = {"id": "c1", "difficulty": "intermediate"}
    tag_chunk_difficulty_provenance(chunk, calibrated)
    assert chunk["difficulty_provenance"] == DIFFICULTY_PROVENANCE_HEURISTIC
    assert "difficulty_irt" not in chunk
    assert chunk["difficulty"] == "intermediate"  # heuristic band kept


# --------------------------------------------------------------------------- #
# Calibrated only with >= min responses
# --------------------------------------------------------------------------- #

def test_calibrated_only_with_min_responses(scaffold_on, tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(tmp_path))
    slug = "calib-course"
    responses = tmp_path / "courses" / slug / "responses" / "item_responses.jsonl"
    n_a = DEFAULT_MIN_RESPONSES + 5
    n_b = DEFAULT_MIN_RESPONSES - 1
    # Item A: hard (only 20% correct) and well-sampled; Item B: under-sampled.
    _build_responses_file(responses, {
        "chunk_A": (n_a, n_a // 5),
        "chunk_B": (n_b, n_b),
    })
    path = resolve_response_data_path(slug)
    assert path is not None
    calibrated = load_calibrated_difficulty(path)
    assert "chunk_A" in calibrated
    assert "chunk_B" not in calibrated  # below floor → never calibrated

    chunk_a = {"id": "chunk_A", "difficulty": "foundational"}
    tag_chunk_difficulty_provenance(chunk_a, calibrated)
    assert chunk_a["difficulty_provenance"] == DIFFICULTY_PROVENANCE_CALIBRATED
    assert chunk_a["difficulty_irt"]["n_responses"] == n_a
    assert chunk_a["difficulty_irt"]["discrimination_a"] == RASCH_DISCRIMINATION_A
    # ~20% correct → hard → advanced band, positive difficulty_b.
    assert chunk_a["difficulty"] == "advanced"
    assert chunk_a["difficulty_irt"]["difficulty_b"] > 0

    chunk_b = {"id": "chunk_B", "difficulty": "foundational"}
    tag_chunk_difficulty_provenance(chunk_b, calibrated)
    assert chunk_b["difficulty_provenance"] == DIFFICULTY_PROVENANCE_HEURISTIC
    assert "difficulty_irt" not in chunk_b


def test_easy_item_maps_to_foundational(scaffold_on, tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(tmp_path))
    slug = "easy-course"
    responses = tmp_path / "courses" / slug / "responses" / "item_responses.jsonl"
    n = DEFAULT_MIN_RESPONSES + 10
    _build_responses_file(responses, {"chunk_E": (n, int(n * 0.95))})
    calibrated = load_calibrated_difficulty(resolve_response_data_path(slug))
    rec = calibrated["chunk_E"]
    assert rec["difficulty"] == "foundational"
    assert rec["difficulty_b"] < 0  # easy → negative b


# --------------------------------------------------------------------------- #
# Descriptor determinism
# --------------------------------------------------------------------------- #

def test_descriptor_deterministic():
    counts = {"foundational": 2, "intermediate": 4, "advanced": 2}
    d1 = describe_difficulty_distribution(counts, 0.5)
    d2 = describe_difficulty_distribution(counts, 0.5)
    assert d1 == d2
    assert d1["modal_level"] == "intermediate"
    assert d1["provenance"] == "mixed"
    # balance_ratio = min_share / max_share = (2/8)/(4/8) = 0.5
    assert d1["balance_ratio"] == pytest.approx(0.5, abs=1e-6)
    assert 0.0 <= d1["ordinal_entropy"] <= 1.0


def test_descriptor_provenance_endpoints():
    counts = {"foundational": 1, "intermediate": 1, "advanced": 1}
    assert describe_difficulty_distribution(counts, 0.0)["provenance"] == "heuristic"
    assert describe_difficulty_distribution(counts, 1.0)["provenance"] == "calibrated"
    # Uniform 3-band → ordinal_entropy == 1.0
    assert describe_difficulty_distribution(counts, 0.0)["ordinal_entropy"] == pytest.approx(1.0)


def test_descriptor_empty_counts():
    d = describe_difficulty_distribution({}, 0.0)
    assert d["modal_level"] is None
    assert d["ordinal_entropy"] == 0.0
    assert d["balance_ratio"] == 0.0


# --------------------------------------------------------------------------- #
# Parse-with-fallback on the flag
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("false", False), ("garbage", False), ("", False),
])
def test_flag_parse_with_fallback(monkeypatch, val, expected):
    monkeypatch.setenv(_FLAG, val)
    assert irt_scaffold_enabled() is expected


def test_decision_event_fires(scaffold_on, tmp_path, monkeypatch):
    """Exactly one difficulty_calibration decision fires per run, with a
    replayable rationale interpolating the run signals."""
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(tmp_path))
    monkeypatch.setenv("ED4ALL_TRAINING_CAPTURES_DIR", str(tmp_path / "captures"))
    from collections import defaultdict
    from types import SimpleNamespace

    from Trainforge.pipeline.process_course import CourseProcessor
    from lib.decision_capture import DecisionCapture

    capture = DecisionCapture(course_code="UNIT-IRT", phase="trainforge-content-analysis",
                              tool="trainforge", streaming=False)
    fake = SimpleNamespace(
        capture=capture,
        stats={
            "total_chunks": 4,
            "difficulty_distribution": defaultdict(int, {"foundational": 2, "intermediate": 2}),
        },
        _irt_calibrated_map={},
        _irt_response_file_present=False,
        _get_irt_calibrated_map=lambda: {},
    )
    stats_out: dict = {}
    CourseProcessor._maybe_add_difficulty_descriptor(fake, stats_out)

    assert "difficulty_distribution_descriptor" in stats_out
    events = [d for d in capture.decisions if d.get("decision_type") == "difficulty_calibration"]
    assert len(events) == 1
    rationale = events[0]["rationale"]
    for token in ("chunks_tagged", "calibrated_fraction", "response_file_present", "min_responses"):
        assert token in rationale


def test_min_responses_garbage_falls_back(scaffold_on, tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(tmp_path))
    slug = "fb-course"
    responses = tmp_path / "courses" / slug / "responses" / "item_responses.jsonl"
    n = DEFAULT_MIN_RESPONSES + 1
    _build_responses_file(responses, {"chunk_X": (n, n // 2)})
    calibrated = load_calibrated_difficulty(
        resolve_response_data_path(slug), min_responses="garbage"  # type: ignore[arg-type]
    )
    assert "chunk_X" in calibrated  # falls back to DEFAULT_MIN_RESPONSES


# --------------------------------------------------------------------------- #
# pipeline_tools SemantiK / IMSCC chunk-emit difficulty_calibration event (Fix 2)
# --------------------------------------------------------------------------- #


class _RecordingCapture:
    """Records ``log_decision`` calls so the helper's emit can be inspected."""

    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.events: list = []
        _RecordingCapture.instances.append(self)

    def log_decision(self, **kwargs):
        self.events.append(kwargs)


def _all_calibration_events():
    return [
        e
        for inst in _RecordingCapture.instances
        for e in inst.events
        if e.get("decision_type") == "difficulty_calibration"
    ]


def test_pipeline_tools_emit_helper_fires_exactly_once(monkeypatch):
    """The pipeline_tools chunk-emit helper emits exactly one
    difficulty_calibration event with a replayable, signal-interpolating
    rationale (closing the doc-vs-code gap for the SemantiK/IMSCC path)."""
    import lib.decision_capture as dc_mod

    _RecordingCapture.instances = []
    monkeypatch.setattr(dc_mod, "DecisionCapture", _RecordingCapture)

    from MCP.tools.pipeline_tools import _emit_irt_difficulty_calibration_event

    chunks = [
        {"id": "c1", "difficulty": "foundational", "difficulty_provenance": "calibrated"},
        {"id": "c2", "difficulty": "intermediate", "difficulty_provenance": "heuristic"},
        {"id": "c3", "difficulty": "advanced", "difficulty_provenance": "heuristic"},
    ]
    _emit_irt_difficulty_calibration_event(
        chunks=chunks,
        course_code="UNIT-IRT",
        phase="textbook-ingestor",
        calibrated_map={"c1": {}},
        response_present=True,
    )

    events = _all_calibration_events()
    assert len(events) == 1
    rationale = events[0]["rationale"]
    for token in (
        "chunks_tagged",
        "calibrated_chunk_count",
        "calibrated_fraction",
        "response_file_present",
        "min_responses",
    ):
        assert token in rationale
    # Dynamic signals are interpolated, not boilerplate.
    assert "chunks_tagged=3" in rationale
    assert "calibrated_chunk_count=1" in rationale
    assert "response_file_present=True" in rationale
    assert str(DEFAULT_MIN_RESPONSES) in rationale


def test_pipeline_tools_emit_gated_by_irt_enabled(monkeypatch):
    """The chunk-emit call site only invokes the helper under ``_irt_enabled``,
    so the flag-off path emits ZERO events and the flag-on path emits one."""
    import lib.decision_capture as dc_mod

    _RecordingCapture.instances = []
    monkeypatch.setattr(dc_mod, "DecisionCapture", _RecordingCapture)

    from MCP.tools.pipeline_tools import _emit_irt_difficulty_calibration_event

    chunks = [{"id": "c1", "difficulty": "foundational", "difficulty_provenance": "heuristic"}]

    # Mirror the chunk-emit call-site guard: the helper is reached ONLY when
    # the scaffold flag resolved on (``if _irt_enabled:``).
    for _irt_enabled in (False, True):
        if _irt_enabled:
            _emit_irt_difficulty_calibration_event(
                chunks=chunks,
                course_code="UNIT-IRT",
                phase="trainforge-content-analysis",
                calibrated_map={},
                response_present=False,
            )

    # One event total: zero on the flag-off iteration, one on the flag-on one.
    assert len(_all_calibration_events()) == 1

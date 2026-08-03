"""``--auto-name`` resolution-point rebind (``_maybe_apply_auto_name``).

Locks the workflow-runner seam that resolves the H1-derived,
run-timestamped FINAL course slug immediately after the conversion
(+ heading_judge) phases and BEFORE ``staging`` consumes course identity:

* opt-in only — a run without ``auto_name`` is byte-identical (params
  untouched, nothing persisted, no capture);
* pre-identity phases (conversion + heading_judge) never trigger it;
* the happy path rebinds ``course_name`` / ``canonical_course_code``,
  records ``display_title`` + ``provisional_course_name``, persists the
  params, emits ONE ``course_identity_rebind`` capture, and downstream
  ``_route_params`` routes the FINAL slug;
* every fallback arm keeps the provided name (honest, reason recorded);
* the resolution is once-per-run (idempotent across later phases/resume).

All fixtures are synthetic — no course-data references (project rule).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import lib.decision_capture as decision_capture_module
from MCP.core.workflow_runner import WorkflowRunner

RUN_CREATED_AT = "2026-07-22T07:04:33"
FINAL_SLUG = "principles-of-sample-systems-20260722-0704"


class _CaptureRecorder:
    """Stands in for DecisionCapture — records instead of writing."""

    instances = []

    def __init__(self, course_code, phase, tool="courseforge", **kwargs):
        self.course_code = course_code
        self.phase = phase
        self.tool = tool
        self.decisions = []
        _CaptureRecorder.instances.append(self)

    def log_decision(self, decision_type, decision, rationale, **kwargs):
        self.decisions.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
                **kwargs,
            }
        )


@pytest.fixture(autouse=True)
def _patch_capture(monkeypatch):
    _CaptureRecorder.instances = []
    monkeypatch.setattr(
        decision_capture_module, "DecisionCapture", _CaptureRecorder
    )
    yield


@pytest.fixture
def runner() -> WorkflowRunner:
    return WorkflowRunner(executor=object(), config=object())


def _mk_state(tmp_path: Path, params: dict) -> tuple[dict, Path]:
    state = {
        "id": "WF-20260722-testcafe",
        "type": "textbook_to_course",
        "params": params,
        "status": "RUNNING",
        "created_at": RUN_CREATED_AT,
    }
    path = tmp_path / "WF-20260722-testcafe.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    return state, path


def _mk_conversion_output(tmp_path: Path, title="Principles Of Sample Systems"):
    html = tmp_path / "book_accessible.html"
    html.write_text(f"<html><body><h1>{title}</h1></body></html>", encoding="utf-8")
    return {
        "semantik_conversion": {
            "_completed": True,
            "output_paths": [str(html)],
        }
    }


def _base_params(**extra):
    params = {
        "course_name": "fixture-source",
        "auto_name": True,
        "run_id": "TTC_fixture-source_20260722_070433",
        "corpus": "inputs/synthetic/book.pdf",
    }
    params.update(extra)
    return params


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_rebind_happy_path(tmp_path, runner):
    params = _base_params()
    state, path = _mk_state(tmp_path, params)
    outputs = _mk_conversion_output(tmp_path)

    runner._maybe_apply_auto_name("staging", params, outputs, state, path)

    assert params["course_name"] == FINAL_SLUG
    assert params["provisional_course_name"] == "fixture-source"
    assert params["display_title"] == "Principles Of Sample Systems"
    assert params["auto_name_resolved"] is True
    assert params["auto_name_reason"] == "h1_resolved"
    # canonical_course_code re-pinned from the FINAL name.
    assert params["canonical_course_code"] == (
        decision_capture_module.normalize_course_code(FINAL_SLUG)
    )
    # Persisted so a --resume sees the same identity.
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["params"]["course_name"] == FINAL_SLUG
    assert persisted["params"]["provisional_course_name"] == "fixture-source"
    # Exactly ONE capture, recording provisional -> final with signals.
    assert len(_CaptureRecorder.instances) == 1
    cap = _CaptureRecorder.instances[0]
    assert cap.tool == "orchestrator"
    assert len(cap.decisions) == 1
    dec = cap.decisions[0]
    assert dec["decision_type"] == "course_identity_rebind"
    assert "fixture-source" in dec["decision"] and FINAL_SLUG in dec["decision"]
    assert "Principles Of Sample Systems" in dec["rationale"]
    assert "2026-07-22" in dec["rationale"]
    assert len(dec["rationale"]) >= 20


def test_route_params_carries_final_slug_after_rebind(tmp_path, runner):
    params = _base_params()
    state, path = _mk_state(tmp_path, params)
    outputs = _mk_conversion_output(tmp_path)

    runner._maybe_apply_auto_name("staging", params, outputs, state, path)
    routed = runner._route_params("staging", params, outputs)
    assert routed["course_name"] == FINAL_SLUG


def test_run_id_and_timestamp_use_run_init_not_resolution_time(tmp_path, runner):
    # created_at unparseable -> falls back to the run_id INIT suffix, never
    # datetime.now() (the directive: run-init time, not resolution time).
    params = _base_params()
    state, path = _mk_state(tmp_path, params)
    state["created_at"] = "garbage"
    outputs = _mk_conversion_output(tmp_path)

    runner._maybe_apply_auto_name("staging", params, outputs, state, path)
    assert params["course_name"].endswith("-20260722-0704")


# ---------------------------------------------------------------------------
# Opt-in / once-per-run / pre-identity guards
# ---------------------------------------------------------------------------


def test_auto_name_off_is_byte_identical(tmp_path, runner):
    params = {"course_name": "fixture-source", "run_id": "TTC_fixture-source_20260722_070433"}
    before = dict(params)
    state, path = _mk_state(tmp_path, params)
    on_disk_before = path.read_text(encoding="utf-8")
    outputs = _mk_conversion_output(tmp_path)

    runner._maybe_apply_auto_name("staging", params, outputs, state, path)

    assert params == before
    assert path.read_text(encoding="utf-8") == on_disk_before
    assert _CaptureRecorder.instances == []


@pytest.mark.parametrize(
    "phase", ["semantik_conversion", "dart_conversion", "heading_judge"]
)
def test_pre_identity_phases_never_trigger(tmp_path, runner, phase):
    params = _base_params()
    state, path = _mk_state(tmp_path, params)
    outputs = _mk_conversion_output(tmp_path)

    runner._maybe_apply_auto_name(phase, params, outputs, state, path)

    assert params["course_name"] == "fixture-source"
    assert "auto_name_resolved" not in params


def test_resolution_is_once_per_run(tmp_path, runner):
    params = _base_params()
    state, path = _mk_state(tmp_path, params)
    outputs = _mk_conversion_output(tmp_path)

    runner._maybe_apply_auto_name("staging", params, outputs, state, path)
    first = params["course_name"]
    # Later phases (and a resume re-entering the loop) must not re-resolve.
    runner._maybe_apply_auto_name("chunking", params, outputs, state, path)
    runner._maybe_apply_auto_name("staging", params, outputs, state, path)

    assert params["course_name"] == first
    assert len(_CaptureRecorder.instances) == 1


# ---------------------------------------------------------------------------
# Fallback arms — the provided name is KEPT, reason recorded
# ---------------------------------------------------------------------------


def _assert_fallback(params, reason):
    assert params["course_name"] == "fixture-source"
    assert params["auto_name_resolved"] is True
    assert params["auto_name_reason"] == reason
    assert "provisional_course_name" not in params
    assert "display_title" not in params


def test_fallback_multi_file_corpus(tmp_path, runner):
    a = tmp_path / "a_accessible.html"
    b = tmp_path / "b_accessible.html"
    a.write_text("<h1>Title A</h1>", encoding="utf-8")
    b.write_text("<h1>Title B</h1>", encoding="utf-8")
    params = _base_params()
    state, path = _mk_state(tmp_path, params)
    outputs = {
        "semantik_conversion": {
            "_completed": True,
            "output_paths": [str(a), str(b)],
        }
    }

    runner._maybe_apply_auto_name("staging", params, outputs, state, path)
    _assert_fallback(params, "multi_file_corpus")
    # A fallback still captures the (kept) decision once.
    assert len(_CaptureRecorder.instances) == 1
    dec = _CaptureRecorder.instances[0].decisions[0]
    assert "kept provisional" in dec["decision"]


def test_fallback_comma_joined_skip_conversion_paths(tmp_path, runner):
    # --skip-conversion synthesizes output_paths as ONE comma-joined string;
    # two files in it must be honestly counted as a multi-file corpus.
    a = tmp_path / "a_accessible.html"
    b = tmp_path / "b_accessible.html"
    a.write_text("<h1>Title A</h1>", encoding="utf-8")
    b.write_text("<h1>Title B</h1>", encoding="utf-8")
    params = _base_params()
    state, path = _mk_state(tmp_path, params)
    outputs = {
        "semantik_conversion": {
            "_completed": True,
            "output_paths": f"{a},{b}",
        }
    }

    runner._maybe_apply_auto_name("staging", params, outputs, state, path)
    _assert_fallback(params, "multi_file_corpus")


def test_fallback_garbage_h1(tmp_path, runner):
    params = _base_params()
    state, path = _mk_state(tmp_path, params)
    outputs = _mk_conversion_output(tmp_path, title="Chapter 7")

    runner._maybe_apply_auto_name("staging", params, outputs, state, path)
    _assert_fallback(params, "h1_structural")


def test_fallback_missing_conversion_output(tmp_path, runner):
    params = _base_params()
    state, path = _mk_state(tmp_path, params)

    runner._maybe_apply_auto_name("staging", params, {}, state, path)
    _assert_fallback(params, "no_conversion_output")


def test_legacy_conversion_phase_key_still_resolves(tmp_path, runner):
    # Dual-read: an old paused run persisted under the legacy phase name.
    html = tmp_path / "book_accessible.html"
    html.write_text("<h1>Principles Of Sample Systems</h1>", encoding="utf-8")
    params = _base_params()
    state, path = _mk_state(tmp_path, params)
    outputs = {
        "dart_conversion": {"_completed": True, "output_paths": [str(html)]}
    }

    runner._maybe_apply_auto_name("staging", params, outputs, state, path)
    assert params["course_name"] == FINAL_SLUG

"""I1 composed objectives-review flow — service, router, and a11y coverage.

The Create wizard gains a "Pause for objectives review" checkpoint that launches
with ``stop_after=course_planning``; the run pauses (orchestrator ``paused`` →
GUI ``paused`` status), the progress view renders a review panel (edit
objectives + resume), and a plain resume with ``clear_stop_after`` continues the
build past the checkpoint.

This file covers, lane-distinct:

* ``run_service`` — the launch-with-stop-after option threading, the paused-run
  status mapping + paused-phase exposure, ``paused_review_info``,
  ``_clear_stop_after``, and the ``resume_run`` clear-stop-after arm.
* the router — ``GET /api/runs/{id}/review`` + the WS ``paused`` terminal frame.
* the Studio a11y gate — the reconstructed review panel + configure checkbox are
  WCAG-clean (zero CRITICAL/HIGH), plus create.js wiring assertions.

State is isolated via ``state_dir``. Synthetic only — the orchestrator is
monkeypatched (no real model / network).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Tuple

import pytest

from gui import shared_state
from gui.services import run_service


# --------------------------------------------------------------------------- #
# Service layer — launch-with-stop-after option threading
# --------------------------------------------------------------------------- #


def test_launch_with_pause_for_review_threads_stop_after(state_dir, monkeypatch):
    """A pause-for-review launch must pass stop_after=course_planning downstream."""
    import MCP.tools.pipeline_tools as pt

    recorded = {}

    async def fake_create(**kwargs):
        recorded["kwargs"] = kwargs
        # Mirror the real create contract: params echo stop_after when present.
        return json.dumps({
            "workflow_id": "WF-REVIEW-1",
            "status": "created",
            "params": {
                "course_name": kwargs.get("course_name"),
                "stop_after": kwargs.get("stop_after"),
            },
        })

    monkeypatch.setattr(pt, "create_textbook_pipeline", fake_create)

    async def fake_drive(run_id, workflow_id, **kw):
        return None

    monkeypatch.setattr(run_service, "_drive_pipeline", fake_drive)

    corpus = state_dir / "fixture.pdf"
    corpus.write_bytes(b"%PDF-1.4 test")

    req = {
        "workflow": "textbook_to_course",
        "course_name": "PHYS_101",
        "corpus": str(corpus),
        "options": {"stop_after": "course_planning"},
    }
    result = asyncio.run(run_service.launch_pipeline(req))
    assert result["status"] == "queued"
    # The real create fn got the halt point (persisted so a bare resume honors it).
    assert recorded["kwargs"]["stop_after"] == "course_planning"
    # The run record echoes stop_after in its params (the progress view reads it).
    record = shared_state.read_run(result["run_id"])
    assert record["params"].get("stop_after") == "course_planning"


# --------------------------------------------------------------------------- #
# Service layer — paused-run status mapping + paused-phase exposure
# --------------------------------------------------------------------------- #


def _write_workflow_state(workflow_id: str, state: dict) -> Path:
    path = run_service._workflow_state_file(workflow_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


def test_drive_pipeline_maps_paused_status_and_persists_paused_phase(
    state_dir, monkeypatch
):
    """orchestrator status=paused → GUI status=paused (NOT failed) + paused_phase."""
    import MCP.orchestrator as orch_pkg

    class PausedResult:
        def to_dict(self):
            return {
                "status": "paused",
                "gates_passed": True,
                "phase_results": {
                    "course_planning": {"gates_passed": True, "failed": 0},
                },
            }

    class PausedOrchestrator:
        def __init__(self, mode, backend_spec):
            pass

        async def run(self, workflow_id):
            return PausedResult()

    monkeypatch.setattr(orch_pkg, "PipelineOrchestrator", PausedOrchestrator)

    workflow_id = "WF-PAUSE-1"
    # The runner persists the --stop-after halt marker on the state file.
    _write_workflow_state(
        workflow_id,
        {
            "stopped_after": "course_planning",
            "phase_outputs": {"course_planning": {"_completed": True}},
        },
    )

    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run(
        {"run_id": run_id, "kind": "pipeline", "workflow": "textbook_to_course",
         "status": "queued", "course_name": "PHYS_101",
         "params": {"stop_after": "course_planning"}}
    )
    asyncio.run(
        run_service._drive_pipeline(
            run_id, workflow_id, mode="api", provider="local", model=None
        )
    )
    record = shared_state.read_run(run_id)
    assert record["status"] == "paused", "a paused orchestrator result is NOT a failure"
    assert record["paused_phase"] == "course_planning"
    # No failure surface leaks onto a paused run.
    assert record.get("failed_phase") is None
    assert record.get("error") is None


def test_drive_pipeline_paused_emits_run_paused_event(state_dir, monkeypatch):
    """A paused run emits a distinct run_paused activity event."""
    import MCP.orchestrator as orch_pkg

    class PausedResult:
        def to_dict(self):
            return {"status": "paused", "gates_passed": True, "phase_results": {}}

    class PausedOrchestrator:
        def __init__(self, mode, backend_spec):
            pass

        async def run(self, workflow_id):
            return PausedResult()

    monkeypatch.setattr(orch_pkg, "PipelineOrchestrator", PausedOrchestrator)
    _write_workflow_state("WF-PAUSE-2", {"stopped_after": "course_planning"})
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run({"run_id": run_id, "kind": "pipeline", "status": "queued"})
    asyncio.run(
        run_service._drive_pipeline(
            run_id, "WF-PAUSE-2", mode="api", provider="local", model=None
        )
    )
    events = shared_state.read_events(0)
    kinds = {e.get("kind") for e in events}
    assert "run_paused" in kinds and "run_failed" not in kinds


# --------------------------------------------------------------------------- #
# Service layer — paused_review_info
# --------------------------------------------------------------------------- #


def test_paused_review_info_exposes_paused_state(state_dir):
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run(
        {"run_id": run_id, "kind": "pipeline", "workflow": "textbook_to_course",
         "status": "paused", "course_name": "PHYS_101", "paused_phase": "course_planning",
         "params": {"stop_after": "course_planning"}}
    )
    info = run_service.paused_review_info(run_id)
    assert info is not None
    assert info["paused"] is True
    assert info["paused_phase"] == "course_planning"
    assert info["stop_after"] == "course_planning"
    assert info["course_name"] == "PHYS_101"
    # course_id degrades to the course_name when no export resolves (no export
    # on disk in this synthetic state), never raising.
    assert info["course_id"] == "PHYS_101"
    assert info["objectives_available"] is False


def test_paused_review_info_unknown_run_is_none(state_dir):
    assert run_service.paused_review_info("GUI-nope") is None


# --------------------------------------------------------------------------- #
# Service layer — _clear_stop_after
# --------------------------------------------------------------------------- #


def test_clear_stop_after_strips_marker(state_dir):
    workflow_id = "WF-CLEAR-1"
    _write_workflow_state(
        workflow_id,
        {"stopped_after": "course_planning",
         "params": {"stop_after": "course_planning", "course_name": "PHYS_101"}},
    )
    assert run_service._clear_stop_after(workflow_id) is True
    state = json.loads(run_service._workflow_state_file(workflow_id).read_text())
    assert "stop_after" not in state["params"]
    assert "stopped_after" not in state
    # Other params survive.
    assert state["params"]["course_name"] == "PHYS_101"
    # Idempotent: a second clear is a no-op (nothing left to strip).
    assert run_service._clear_stop_after(workflow_id) is False


def test_clear_stop_after_missing_state_is_false(state_dir):
    assert run_service._clear_stop_after("WF-DOES-NOT-EXIST") is False


# --------------------------------------------------------------------------- #
# Service layer — resume_run with clear_stop_after
# --------------------------------------------------------------------------- #


def test_resume_run_clear_stop_after_strips_marker_before_drive(state_dir, monkeypatch):
    """resume_run(clear_stop_after=True) removes the halt marker before re-driving."""
    workflow_id = "WF-RESUME-CLEAR-1"
    _write_workflow_state(
        workflow_id,
        {"stopped_after": "course_planning",
         "params": {"stop_after": "course_planning"},
         "phase_outputs": {"course_planning": {"_completed": True}}},
    )
    prior_id = shared_state.new_run_id("GUI")
    shared_state.register_run(
        {"run_id": prior_id, "kind": "pipeline", "workflow": "textbook_to_course",
         "workflow_id": workflow_id, "status": "paused", "course_name": "PHYS_101",
         "params": {"stop_after": "course_planning"}}
    )

    driven = {}

    async def fake_drive(run_id, wf_id, **kw):
        # Capture the state file AS the drive sees it (stop marker already cleared).
        driven["run_id"] = run_id
        driven["state"] = json.loads(
            run_service._workflow_state_file(wf_id).read_text()
        )

    monkeypatch.setattr(run_service, "_drive_pipeline", fake_drive)

    result = asyncio.run(
        run_service.resume_run(prior_id, {"clear_stop_after": True})
    )
    assert result["status"] == "queued"
    # By the time the re-drive runs, the halt marker is gone.
    assert "stop_after" not in driven["state"]["params"]
    assert "stopped_after" not in driven["state"]
    # The resumed run record no longer advertises the halt point.
    new_record = shared_state.read_run(result["run_id"])
    assert (new_record.get("params") or {}).get("stop_after") is None


def test_resume_run_without_clear_keeps_stop_after(state_dir, monkeypatch):
    """A plain resume (no clear flag) leaves the persisted stop marker in place."""
    workflow_id = "WF-RESUME-KEEP-1"
    _write_workflow_state(
        workflow_id,
        {"stopped_after": "course_planning",
         "params": {"stop_after": "course_planning"},
         "phase_outputs": {"course_planning": {"_completed": True}}},
    )
    prior_id = shared_state.new_run_id("GUI")
    shared_state.register_run(
        {"run_id": prior_id, "kind": "pipeline", "workflow": "textbook_to_course",
         "workflow_id": workflow_id, "status": "paused", "course_name": "PHYS_101",
         "params": {"stop_after": "course_planning"}}
    )

    async def fake_drive(run_id, wf_id, **kw):
        return None

    monkeypatch.setattr(run_service, "_drive_pipeline", fake_drive)
    asyncio.run(run_service.resume_run(prior_id, {}))
    state = json.loads(run_service._workflow_state_file(workflow_id).read_text())
    assert state["params"]["stop_after"] == "course_planning"


# --------------------------------------------------------------------------- #
# Router — GET /api/runs/{id}/review + WS paused terminal frame
# --------------------------------------------------------------------------- #

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from gui.app import create_app  # noqa: E402


@pytest.fixture
def client(state_dir, libv2_root):
    return TestClient(create_app())


def test_review_endpoint_returns_paused_info(client, state_dir):
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run(
        {"run_id": run_id, "kind": "pipeline", "workflow": "textbook_to_course",
         "status": "paused", "course_name": "PHYS_101", "paused_phase": "course_planning",
         "params": {"stop_after": "course_planning"}}
    )
    resp = client.get(f"/api/runs/{run_id}/review")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["paused"] is True
    assert body["paused_phase"] == "course_planning"
    assert body["course_name"] == "PHYS_101"


def test_review_endpoint_unknown_run_is_404(client):
    resp = client.get("/api/runs/GUI-nope/review")
    assert resp.status_code == 404
    assert resp.json()["error"] == "unknown_run"


def test_launch_request_accepts_clear_stop_after_field(client, state_dir, monkeypatch):
    """The launch body model accepts clear_stop_after (resume-flow field)."""
    # A resume with an unknown prior run fails closed (422) — but the point here
    # is that clear_stop_after is a recognized field (no 422 for an unknown key).
    resp = client.post(
        "/api/runs",
        json={"resume_run_id": "GUI-nope", "clear_stop_after": True},
    )
    # Unknown prior run → typed 422 (resume can't resolve), NOT a request-schema
    # rejection of clear_stop_after.
    assert resp.status_code == 422
    assert "cannot resume" in resp.json()["detail"]


def test_ws_sends_paused_status_frame_and_closes(client, state_dir):
    """The WS treats paused as terminal: it sends a status frame then closes."""
    run_id = shared_state.new_run_id("GUI")
    shared_state.register_run(
        {"run_id": run_id, "kind": "pipeline", "workflow": "textbook_to_course",
         "status": "paused", "course_name": "PHYS_101", "paused_phase": "course_planning"}
    )
    shared_state.append_log(run_id, "[iso] paused for review after phase course_planning\n")
    with client.websocket_connect(f"/api/ws/runs/{run_id}") as ws:
        frame = None
        for _ in range(20):
            frame = ws.receive_json()
            if frame.get("type") == "status":
                break
        assert frame is not None and frame["type"] == "status"
        assert frame["status"] == "paused"


# --------------------------------------------------------------------------- #
# Studio a11y gate — review panel + configure checkbox (WCAG clean) + wiring
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_DIR = REPO_ROOT / "gui" / "static" / "studio"
STUDIO_INDEX = STUDIO_DIR / "index.html"


def _bs4():
    from bs4 import BeautifulSoup  # noqa: PLC0415

    return BeautifulSoup


def _soup(html: str):
    return _bs4()(html, "html.parser")


def _shell_with_view(view_inner: str) -> str:
    soup = _soup(STUDIO_INDEX.read_text(encoding="utf-8"))
    v = soup.find(id="view")
    assert v is not None, "studio index.html must carry #view"
    v.append(_bs4()(view_inner, "html.parser"))
    return str(soup)


def _gate(html: str) -> Tuple[List, List]:
    from lib.validators.wcag import IssueSeverity, WCAGValidator  # noqa: PLC0415

    blocking_sev = {IssueSeverity.CRITICAL, IssueSeverity.HIGH}
    report = WCAGValidator().validate(html)
    blocking = [i for i in report.issues if i.severity in blocking_sev]
    diagnostics = [i for i in report.issues if i.severity not in blocking_sev]
    return blocking, diagnostics


def _assert_clean(variant: str, html: str) -> None:
    blocking, _ = _gate(html)
    assert not blocking, (
        f"{variant}: {len(blocking)} CRITICAL/HIGH WCAG finding(s) — "
        + "; ".join(f"{i.severity.value} {i.criterion} {i.message}" for i in blocking)
    )


# The configure step with the review checkpoint field, reconstructed exactly as
# create.js ``reviewCheckpointField`` builds it (a labelled checkbox + hint).
_CONFIGURE_REVIEW_CHECK_INNER = """
<h1>Create a course</h1>
<div class="wizard-panel">
  <h2>Step 2: Configure</h2>
  <form class="wizard-form" novalidate>
    <div class="field check review-check">
      <input id="pause-review-x" type="checkbox" aria-describedby="pause-review-h">
      <label for="pause-review-x">Pause for objectives review</label>
      <p id="pause-review-h" class="field-hint">Stop the build after planning the learning objectives so you can review and edit them before the rest of the course is generated. You can resume from the build screen.</p>
    </div>
  </form>
</div>
"""

# The progress view PAUSED at the review checkpoint, with the review panel
# rendered into the final-box exactly as create.js ``renderReviewPanel`` builds
# it (labelled section, objectives-file disclosure, edit + resume actions).
_PROGRESS_PAUSED_INNER = """
<h1>Building PHYS_101</h1>
<p class="muted"><span>Run GUI-x</span><span class="sep" aria-hidden="true"> · </span><span class="elapsed">finished</span></p>
<ol class="phase-checklist" aria-label="Course build steps">
  <li class="phase-row is-done" data-phase="course_planning"><span class="phase-icon" aria-hidden="true">●</span><span class="phase-label">Plan learning objectives</span><span class="phase-state">Done</span></li>
  <li class="phase-row is-pending" data-phase="content_generation"><span class="phase-icon" aria-hidden="true">○</span><span class="phase-label">Generate course content</span><span class="phase-state">Pending</span></li>
</ol>
<div class="final-box" aria-live="polite">
  <section class="review-panel" aria-labelledby="review-h">
    <h2 id="review-h" class="review-title">Review the learning objectives</h2>
    <p class="review-intro">The build paused after planning the learning objectives so you can review and edit them before the rest of the course is generated. When you are done, resume the build below.</p>
    <p class="review-path"><span class="review-path-label">Objectives file: </span><code class="kv">/tmp/exports/PROJ-PHYS_101-x/01_learning_objectives/synthesized_objectives.json</code></p>
    <div class="review-actions">
      <a class="btn" href="/advanced/#/courses" target="_blank" rel="noopener" aria-label="Edit objectives (opens the Advanced objectives editor in a new tab)">Edit objectives</a>
      <button type="button" class="btn primary">Resume build</button>
    </div>
    <p class="review-hint muted"><span>In the objectives editor, open the course </span><code class="kv">PROJ-PHYS_101-x</code><span>, save your edits, then return here and resume the build.</span></p>
  </section>
</div>
"""


@pytest.mark.parametrize(
    "label,inner",
    [
        ("configure-review-check", _CONFIGURE_REVIEW_CHECK_INNER),
        ("progress-paused-review-panel", _PROGRESS_PAUSED_INNER),
    ],
)
def test_review_flow_views_zero_aa_findings(label, inner):
    _assert_clean(f"studio-{label}", _shell_with_view(inner))


def test_review_panel_is_labelled_section_with_edit_and_resume():
    soup = _soup(_shell_with_view(_PROGRESS_PAUSED_INNER))
    panel = soup.find("section", class_="review-panel")
    assert panel is not None, "paused build must render a review panel"
    lbl = panel.get("aria-labelledby")
    assert lbl and soup.find(id=lbl) is not None, "review panel needs an accessible name"
    # Single-h1 invariant holds (panel title is an h2).
    assert len(soup.find_all("h1")) == 1
    # The edit-objectives deep link is a real anchor to the Advanced editor,
    # announcing the new tab (WCAG 3.2.5).
    edit = panel.find("a", class_="btn", href="/advanced/#/courses")
    assert edit is not None, "review panel must link to the objectives editor"
    assert edit.get("target") == "_blank" and "noopener" in (edit.get("rel") or [])
    assert "new tab" in (edit.get("aria-label") or "")
    # A real resume control.
    resume = panel.find("button", string=lambda s: s and "Resume build" in s)
    assert resume is not None, "review panel must offer a Resume build button"
    # The objectives file path is surfaced (which path will be used).
    assert panel.find("p", class_="review-path") is not None


def test_configure_review_checkbox_is_labelled():
    soup = _soup(_shell_with_view(_CONFIGURE_REVIEW_CHECK_INNER))
    box = soup.find("input", id="pause-review-x")
    assert box is not None and box.get("type") == "checkbox"
    assert soup.find("label", attrs={"for": "pause-review-x"}) is not None
    desc = box.get("aria-describedby")
    assert desc and soup.find(id=desc) is not None, "checkbox must describe the checkpoint"


def test_create_js_wires_review_checkpoint_and_resume():
    js = (STUDIO_DIR / "create.js").read_text(encoding="utf-8")
    # The configure checkpoint launches with stop_after=course_planning.
    assert "reviewCheckpointField" in js, "create.js must render the review checkpoint field"
    assert "Pause for objectives review" in js
    assert "options.stop_after = 'course_planning'" in js
    # The paused progress view renders the review panel + a plain resume.
    assert "renderReviewPanel" in js, "create.js must render the objectives-review panel"
    assert "resumeAfterReview" in js
    assert "resume_run_id" in js and "clear_stop_after" in js, (
        "resume must re-POST /api/runs with resume_run_id + clear_stop_after"
    )
    assert "/api/runs/${encodeURIComponent(record.run_id)}/review" in js, (
        "review panel must fetch the paused-review endpoint"
    )
    assert "/advanced/#/courses" in js, "review panel must link to the objectives editor"

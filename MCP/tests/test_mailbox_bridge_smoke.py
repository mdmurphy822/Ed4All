"""Wave 34 end-to-end smoke: 2-week synthetic content_generation phase
via the mailbox bridge.

The smoke exercises the full file-based plumbing from the dispatcher's
side: LocalDispatcher puts pending tasks, a MockWatcher (in-process,
running in a background thread, standing in for the outer Claude Code
session) claims them and writes synthetic completions. Final assertion:
both week dispatches return status="ok" with HTML page artifacts
recorded.

This smoke does NOT exercise a real Agent tool — that requires the
outer Claude Code session and is not hermetic. The bridge plumbing
itself is verified here.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.orchestrator.content_prompts import (  # noqa: E402
    build_content_generation_prompt,
)
from MCP.orchestrator.local_dispatcher import LocalDispatcher  # noqa: E402
from MCP.orchestrator.task_mailbox import TaskMailbox  # noqa: E402
from MCP.orchestrator.worker_contracts import PhaseInput, PhaseOutput  # noqa: E402


class MockWatcher:
    """Same test double used in the unit suite, duplicated here to keep
    the smoke file self-contained."""

    def __init__(self, mailbox: TaskMailbox, *, envelope_factory, poll: float = 0.02):
        self.mailbox = mailbox
        self.envelope_factory = envelope_factory
        self.poll = poll
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.claimed: List[Dict[str, Any]] = []

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        while not self._stop.is_set():
            for task_id in self.mailbox.list_pending():
                try:
                    spec = self.mailbox.claim(task_id)
                except Exception:  # noqa: BLE001
                    continue
                self.claimed.append(spec)
                env = self.envelope_factory(task_id, spec)
                self.mailbox.complete(task_id, env)
            time.sleep(self.poll)


def _synthetic_html(week_n: int, page_kind: str, lo_id: str) -> str:
    return (
        f"<main data-cf-role='{page_kind}' "
        f"data-cf-objective-ids='{lo_id}' "
        f"data-cf-bloom-level='understand' "
        f"data-cf-source-ids='dart-block-w{week_n}-01'>"
        f"<h1>Week {week_n} {page_kind.title()}</h1>"
        f"<p>Synthetic body for week {week_n} covering {lo_id}.</p>"
        f"</main>"
    )


def _make_envelope_factory(run_id: str):
    """Return a factory that fabricates realistic content-generation
    completions for each claimed week-scoped task."""

    def factory(task_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        phase_name = spec["phase_input"]["phase_name"]
        params = spec["phase_input"].get("params", {})
        week_n = params.get("week_n", 0)
        lo_id = params.get("lo_id", "TO-00")
        pages = [
            {
                "filename": f"week_{week_n}_{kind}.html",
                "html": _synthetic_html(week_n, kind, lo_id),
                "source_ids": [f"dart-block-w{week_n}-01"],
            }
            for kind in ("overview", "content", "application", "summary")
        ]
        return {
            "success": True,
            "result": {
                "run_id": run_id,
                "phase_name": phase_name,
                "status": "ok",
                "outputs": {"pages": pages},
                "metrics": {"week_n": week_n, "lo_id": lo_id},
            },
        }

    return factory


@pytest.mark.asyncio
async def test_two_week_content_generation_via_mailbox_bridge(tmp_path: Path, monkeypatch):
    """Dispatch two synthetic content_generation tasks (one per week)
    via LocalDispatcher's mailbox bridge. A MockWatcher stands in for
    the outer session. Both phase outputs must report status="ok"
    with 4 HTML pages each."""

    monkeypatch.delenv("LOCAL_DISPATCHER_ALLOW_STUB", raising=False)
    run_id = "RUN_W34_SMOKE"
    runs_root = tmp_path / "runs"

    mailbox = TaskMailbox(run_id=run_id, base_dir=runs_root)
    watcher = MockWatcher(mailbox, envelope_factory=_make_envelope_factory(run_id))
    watcher.start()

    try:
        dispatcher = LocalDispatcher(
            project_root=tmp_path,
            mailbox_base_dir=runs_root,
            mailbox_timeout_seconds=5.0,
            mailbox_poll_interval=0.02,
        )

        weeks = [
            {"week_n": 1, "lo_id": "TO-01", "chapter": "<p>Ch1</p>"},
            {"week_n": 2, "lo_id": "TO-02", "chapter": "<p>Ch2</p>"},
        ]

        async def dispatch_week(w):
            # Build the real content-generation prompt (exercise
            # build_content_generation_prompt in the bridge path).
            out_dir = tmp_path / "weeks" / f"week_{w['week_n']}"
            prompt = build_content_generation_prompt(
                week_n=w["week_n"],
                chapter_html=w["chapter"],
                planned_los=[
                    {"id": w["lo_id"], "statement": f"LO for week {w['week_n']}"}
                ],
                output_dir=out_dir,
            )
            phase_input = PhaseInput(
                run_id=run_id,
                workflow_type="textbook_to_course",
                phase_name=f"content_generation_week_{w['week_n']}",
                phase_config={"agents": ["content-generator"]},
                params={"week_n": w["week_n"], "lo_id": w["lo_id"], "prompt_len": len(prompt)},
                mode="local",
            )
            return await dispatcher.dispatch_phase(phase_input)

        results = await asyncio.gather(*[dispatch_week(w) for w in weeks])

    finally:
        watcher.stop()

    # Both weeks succeeded.
    assert all(isinstance(r, PhaseOutput) for r in results)
    assert [r.status for r in results] == ["ok", "ok"]

    # Each result carries 4 pages with the expected HTML shape.
    for i, result in enumerate(results):
        week_n = i + 1
        pages = result.outputs.get("pages") or []
        assert len(pages) == 4, f"week {week_n}: expected 4 pages, got {len(pages)}"
        filenames = [p["filename"] for p in pages]
        assert filenames == [
            f"week_{week_n}_overview.html",
            f"week_{week_n}_content.html",
            f"week_{week_n}_application.html",
            f"week_{week_n}_summary.html",
        ]
        for page in pages:
            html = page["html"]
            assert "data-cf-role" in html
            assert "data-cf-source-ids" in html
            assert f"Week {week_n}" in html
        assert result.metrics.get("week_n") == week_n

    # Watcher saw exactly two claimed tasks. Each spec carries the
    # dispatcher-built prompt (agent-spec wrapper + routed params) and
    # the phase_input.params the smoke set (week_n, lo_id, prompt_len).
    assert len(watcher.claimed) == 2
    seen_weeks = set()
    for spec in watcher.claimed:
        assert spec["subagent_type"] == "content-generator"
        # Dispatcher-built prompt has the phase header + agent-spec section.
        assert "# Phase: content_generation_week_" in spec["prompt"]
        assert "## Agent spec: content-generator" in spec["prompt"]
        # Params from the PhaseInput (including the builder-produced
        # prompt_len) round-trip through the mailbox.
        params = spec["phase_input"]["params"]
        assert params["prompt_len"] > 0
        seen_weeks.add(params["week_n"])
    assert seen_weeks == {1, 2}

    # Mailbox is empty (dispatcher cleaned up each task after success).
    assert mailbox.list_pending() == []
    assert mailbox.list_in_progress() == []
    assert mailbox.list_completed() == []

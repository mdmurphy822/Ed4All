#!/usr/bin/env python3
"""Service an ed4all --mode local run's agent-task mailbox.

Under ``ED4ALL_AGENT_DISPATCH=true`` every phase task is routed through the
file mailbox as a ``kind=agent_task`` spec, and the run blocks on
``await_completion_async`` until an outer operator writes a completion
envelope. Most ed4all phase tasks are backed by *deterministic* registry
tools (``plan_course_structure``, ``build_source_module_map``,
``package_imscc``, ``archive_to_libv2`` …) — those we can execute in-process
here and complete immediately. Genuinely generative tasks (the
``content-generator`` courseforge-synthesis tasks, one agent = one file) are
left claimed and printed as ``NEEDS_AGENT`` so the outer Claude session can
dispatch a real subagent and complete them via ``--complete``.

Usage:
    python scripts/mailbox_servicer.py --run-id RID            # drain deterministic
    python scripts/mailbox_servicer.py --run-id RID --watch 90 # keep draining for Ns
    python scripts/mailbox_servicer.py --run-id RID --complete TASK_ID --result-file F
    python scripts/mailbox_servicer.py --run-id RID --list     # just list, no execute
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from MCP.orchestrator.local_dispatcher import GRACEFUL_STOP_ERROR_CODE  # noqa: E402
from MCP.orchestrator.task_mailbox import TaskMailbox  # noqa: E402
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402
from lib.generation.stop_control import (  # noqa: E402
    GracefulStopRequested,
    stop_requested,
)

# Tools that are pure deterministic transforms (no LLM completion needed).
DETERMINISTIC_TOOLS = {
    "stage_dart_outputs",
    "extract_textbook_structure",
    "build_source_module_map",
    "run_concept_extraction",
    "plan_course_structure",
    "package_imscc",
    "run_imscc_chunking",
    "run_inter_tier_validation",
    "run_post_rewrite_validation",
    "archive_to_libv2",
    "analyze_imscc_content",
    # Courseforge content emission is deterministic templating
    # (Courseforge.scripts.generate_course.generate_week) — no live LLM,
    # one weekly module per task (1 agent = 1 file).
    "generate_course_content",
    "run_content_generation_outline",
    "run_content_generation_rewrite",
}

REGISTRY = _build_tool_registry()


def _mailbox(run_id: str) -> TaskMailbox:
    # TaskMailbox appends ``{run_id}/mailbox`` to base_dir.
    return TaskMailbox(run_id=run_id, base_dir=str(ROOT / "state" / "runs"))


def _all_task_specs(mb: TaskMailbox) -> list[dict]:
    """Read specs from both pending/ and in_progress/ (already-claimed)."""
    specs = []
    for d, claimed in ((mb.pending_dir, False), (mb.in_progress_dir, True)):
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix != ".json" or p.name.startswith("."):
                continue
            try:
                spec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            spec["_task_id"] = p.stem
            spec["_claimed"] = claimed
            specs.append(spec)
    return specs


async def _run_tool(tool_name: str, params: dict) -> dict:
    fn = REGISTRY.get(tool_name)
    if fn is None:
        return {"success": False, "error": f"tool {tool_name!r} not in registry",
                "error_code": "UNKNOWN_TOOL"}
    raw = await fn(**params)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"success": True, "outputs": {"text": raw}, "artifacts": []}
    return {"success": True, "outputs": {}, "artifacts": []}


def _complete(mb: TaskMailbox, task_id: str, tool_dict: dict) -> None:
    ok = bool(tool_dict.get("success", True))
    mb.complete(task_id, {"success": ok, "result": tool_dict})


async def drain(run_id: str, watch_seconds: float) -> int:
    mb = _mailbox(run_id)
    deadline = time.time() + max(0.0, watch_seconds)
    serviced = 0
    needs_agent: list[dict] = []
    first = True
    stopped = False
    while first or time.time() < deadline:
        first = False
        specs = _all_task_specs(mb)
        progressed = False
        for spec in specs:
            tid = spec["_task_id"]
            tool = spec.get("tool_name") or ""
            agent = spec.get("agent_type") or ""
            params = dict(spec.get("task_params") or {})
            params.pop("id", None)  # not a tool kwarg
            # Pre-claim graceful-stop check (P3.M): once a stop sentinel is up,
            # stop claiming / executing NEW tasks. We check at the top of the
            # loop body — before claiming this task — so the in-flight task
            # (the one we just finished writing a completion for) is never torn
            # up; we simply refuse to start the next one. Resolved through
            # stop_control (get_state_runs_dir(), never CWD — risk R2). Prefer
            # the task's own run_id, falling back to the drain run_id.
            task_run_id = spec.get("run_id") or run_id
            if stop_requested(task_run_id):
                stopped = True
                break
            if tool in DETERMINISTIC_TOOLS:
                # claim if still pending
                if not spec["_claimed"]:
                    try:
                        mb.claim(tid)
                    except Exception:
                        pass
                print(f"[exec] {agent}/{tool} task={tid} ...", flush=True)
                try:
                    result = await _run_tool(tool, params)
                except GracefulStopRequested as exc:
                    # The tool observed a stop sentinel at a unit boundary,
                    # checkpointed the in-flight unit, and raised. Marshal it
                    # into a GRACEFUL_STOP completion envelope (carrying the
                    # exception's site_id / units_completed) BEFORE the
                    # catch-all below — otherwise it would be recorded as
                    # TOOL_RAISED, string-classified transient, retried 3x, and
                    # stamp the phase FAILED. local_dispatcher re-raises this
                    # code as the pause channel so the executor stamps PAUSED.
                    result = {
                        "success": False,
                        "error": str(exc),
                        "error_code": GRACEFUL_STOP_ERROR_CODE,
                        "site_id": getattr(exc, "site_id", None),
                        "units_completed": getattr(exc, "units_completed", 0),
                    }
                    _complete(mb, tid, result)
                    print(f"       -> PAUSED (graceful stop) "
                          f"units_completed={result['units_completed']}",
                          flush=True)
                    serviced += 1
                    stopped = True
                    break
                except Exception as exc:  # noqa: BLE001
                    result = {"success": False, "error": repr(exc),
                              "error_code": "TOOL_RAISED"}
                _complete(mb, tid, result)
                ok = result.get("success", True)
                print(f"       -> success={ok} "
                      f"{'keys=' + ','.join(list(result)[:6]) if ok else result.get('error','')[:200]}",
                      flush=True)
                serviced += 1
                progressed = True
            else:
                if tid not in {n["_task_id"] for n in needs_agent}:
                    needs_agent.append(spec)
        if stopped:
            print("\n=== GRACEFUL_STOP — sentinel observed; not claiming new "
                  "tasks ===", flush=True)
            break
        if needs_agent:
            break
        if not progressed:
            await asyncio.sleep(2.0)
    if needs_agent:
        print("\n=== NEEDS_AGENT (content-generator / generative tasks) ===")
        for spec in needs_agent:
            print(json.dumps({k: v for k, v in spec.items()
                              if not k.startswith("_")} | {"task_id": spec["_task_id"]},
                             indent=2))
    print(f"\nServiced {serviced} deterministic task(s); "
          f"{len(needs_agent)} need a real agent.", flush=True)
    return serviced


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--watch", type=float, default=0.0,
                    help="keep draining for N seconds (catches newly dispatched tasks)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--complete", metavar="TASK_ID")
    ap.add_argument("--result-file")
    args = ap.parse_args()
    mb = _mailbox(args.run_id)

    if args.list:
        for spec in _all_task_specs(mb):
            print(f"{spec['_task_id']:48} claimed={spec['_claimed']} "
                  f"agent={spec.get('agent_type')} tool={spec.get('tool_name')}")
        return
    if args.complete:
        tool_dict = json.loads(Path(args.result_file).read_text(encoding="utf-8"))
        _complete(mb, args.complete, tool_dict)
        print(f"completed {args.complete}")
        return
    asyncio.run(drain(args.run_id, args.watch))


if __name__ == "__main__":
    main()

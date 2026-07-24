"""Typed tool whitelist for the operator assistant. No shell, no free paths.

The model reaches these through the OpenAI tools/function-calling API; the
ONLY dispatch surface is :func:`dispatch_tool`, which looks the requested
name up in :data:`TOOL_REGISTRY` — any other name gets a refusal string
routed back to the model.

Sandbox rules (extend, never weaken):

* The model NEVER supplies filesystem paths, shell strings, or unvalidated
  ids. Run ids validate against :data:`RUN_ID_RE` (``WF-YYYYMMDD-xxxxxxxx``)
  or, where relevant, the executor form :data:`TTC_RUN_ID_RE`; course slugs
  must exist in the campaign manifest or ``LibV2/courses/``; report names
  come from fixed enums; log tails are capped at 200 lines and resolve ONLY
  through the two whitelisted patterns (campaign slug log / run-id log
  under ``state/runs``).
* Subprocess invocations use fixed argv lists (never ``shell=True``); the
  only model-supplied argv members are ids/slugs validated above.
* Every tool result is SUMMARIZED for a 32k-context model: hard cap
  :data:`MAX_TOOL_RESULT_CHARS` (~4000 chars) with a truncation note —
  never a raw JSON blob dump.
* Anything that could echo env/config values (doctor output, logs) is
  routed through ``lib.secrets_filter.redact`` before it reaches the model.

Tool families: status (campaign/run/seat), reports (workflows, run/gate
reports, logs, doctor, build cost, aggregators, flags), library (courses,
objectives, grounded ask), actions (start/resume/stop, seats, support
bundle), help.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from lib.paths import LIBV2_COURSES, PROJECT_ROOT, STATE_PATH, campaign_dir
from lib.retrieval.answer_backend import _is_loopback_host
from lib.secrets_filter import redact as _secrets_redact
from lib.vllm_container_lifecycle import (
    ENV_SEAT_BASE_URLS,
    ENV_SEAT_COHERENCE_ATTEMPTS,
    ENV_SEAT_LAUNCH_SPECS,
    ENV_SEAT_LOAD_TIMEOUT_SECONDS,
    ENV_SEAT_SCHEDULE,
    ENV_VLLM_CONTAINERS,
    _first_model_id,
    _probe_ready,
    parse_container_registry,
    parse_seat_launch_specs,
    parse_seat_registry,
    resolve_seat_coherence_attempts,
    resolve_seat_load_timeout,
    resolve_seat_schedule_mode,
)

logger = logging.getLogger(__name__)

#: Canonical workflow run-id shape (``WF-YYYYMMDD-xxxxxxxx``). Anything else
#: is refused BEFORE a subprocess argv or filesystem path is built.
RUN_ID_RE = re.compile(r"^WF-[0-9]{8}-[a-f0-9]{8}$")

#: Executor-form run id (``TTC_<course>_<ts>``) — accepted where relevant
#: (log resolution under ``state/runs``), never for state/workflows reads.
TTC_RUN_ID_RE = re.compile(r"^TTC_[A-Za-z0-9_-]{1,100}$")

#: Course-slug shape guard (defense in depth BEFORE the existence check —
#: no separators that could escape a directory).
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

#: Operator campaign-harness directory (manifest, per-book logs, review
#: queue). Site-configurable via ``ED4ALL_CAMPAIGN_DIR``; defaults to the
#: neutral repo-relative ``plans/campaign``. Resolved once at import so a
#: test can monkeypatch this module attribute directly.
CAMPAIGN_DIR = campaign_dir()

#: Docs the flag_lookup tool is allowed to grep (fixed set, read-only).
BEHAVIOR_FLAGS_MD = PROJECT_ROOT / "docs" / "operations" / "behavior-flags.md"
ROOT_CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"

#: Workflow config parsed (once, cached) by list_workflows.
WORKFLOWS_YAML = PROJECT_ROOT / "config" / "workflows.yaml"

#: Max recent runs surfaced by ``run_status``.
_RUN_STATUS_LIMIT = 10

#: Hard cap on any single tool result (chars) — keeps a 32k-context model
#: from drowning in one tool round.
MAX_TOOL_RESULT_CHARS = 4000

#: tail_log line clamp.
MAX_LOG_TAIL_LINES = 200

#: ``ed4all doctor`` group enum (mirrors lib/diagnostics registrations).
DOCTOR_GROUPS = ("environment", "provider", "postmortem", "gpu_profile", "gpu", "window")

#: Fixed enum of aggregator report names → candidate paths relative to the
#: LibV2 course dir. ``validation_report`` is special-cased (Courseforge
#: export glob). The model can only name a key from this dict.
AGGREGATOR_REPORTS: Dict[str, Tuple[str, ...]] = {
    "promotion_chain": ("courseforge_promotion_chain_report.json",),
    "coverage_map": ("coverage_map.json",),
    "accessibility_conformance": ("quality/accessibility_conformance.json",),
    "edge_consensus": (
        "graph/edge_consensus_report.json",
        "quality/edge_consensus_report.json",
    ),
    "block_quality_rollup": ("block_quality_rollup_report.json",),
    "concept_coverage": ("concept_coverage.json",),
    "intelligence_level": ("intelligence_level_report.json",),
    "assessment_quality": ("quality/trainforge_assessment_quality_report.json",),
    "provenance_resolution": ("quality/provenance_resolution_report.json",),
    "kg_quality": ("quality/kg_quality_report.json",),
    "build_cost": ("build_cost_report.json",),
    "validation_report": (),  # Courseforge export glob — special-cased
}

#: Per-seat vLLM ``--gpu-memory-utilization`` fractions, mirrored from
#: ``seats/launch-*.sh`` + the co-residency law in
#: the operator seat-boot harness (that ops harness exports no
#: importable fraction table, so the values are replicated here minimally;
#: keep them in sync with the launch scripts). Rules enforced by
#: ``start_seat``: (1) ``spark-super`` is NEVER co-resident with anything;
#: (2) the summed fractions of live seats + the target must stay ≤ 0.95.
SEAT_GPU_FRACTION: Dict[str, float] = {
    "spark-super": 0.7292,
    "spark-qwen": 0.40,
    "spark-glm": 0.25,
    "spark-nano": 0.30,
}
#: Conservative fraction assumed for a registered seat with no table row.
_UNKNOWN_SEAT_FRACTION = 0.5
#: Co-residency budget ceiling (see boot_seats.py rationale).
SEAT_GPU_BUDGET = 0.95

_TRUTHY_TOKENS = frozenset({"1", "true", "yes", "on"})


# --------------------------------------------------------------------------- #
# Shared summarizer / validation helpers
# --------------------------------------------------------------------------- #


def _clip(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Hard-cap a tool result with an explicit truncation note."""
    text = str(text)
    if len(text) <= limit:
        return text
    kept = text[: max(0, limit - 60)].rstrip()
    return f"{kept}\n…[truncated: showing {len(kept)} of {len(text)} chars]"

def _snip(text: str, limit: int) -> str:
    """Compact inline truncation (single-field values inside a summary)."""
    text = str(text).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _redact(text: str) -> str:
    """Redact secrets/env echoes; never let a filter failure eat the result."""
    try:
        return str(_secrets_redact(str(text)))
    except Exception:  # noqa: BLE001 — redaction is best-effort, loud in log
        logger.exception("assistant: secrets redaction failed")
        return "[redaction failed — output withheld]"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validated_run_id(run_id: Any, *, allow_ttc: bool = False) -> Optional[str]:
    """Return the canonical run id or None. The injection guard for every
    run-id-consuming tool: nothing invalid ever reaches a path or argv."""
    run_id = str(run_id or "").strip()
    if RUN_ID_RE.match(run_id):
        return run_id
    if allow_ttc and TTC_RUN_ID_RE.match(run_id):
        return run_id
    return None


def _campaign_manifest_books() -> List[Dict[str, Any]]:
    manifest_path = CAMPAIGN_DIR / "manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = _load_json(manifest_path)
    except (OSError, ValueError):
        return []
    books = manifest.get("books")
    return [b for b in books if isinstance(b, dict)] if isinstance(books, list) else []


def _known_course_slugs() -> set:
    """Slug whitelist: campaign-manifest slugs ∪ LibV2/courses dir names."""
    slugs = {str(b.get("slug")) for b in _campaign_manifest_books() if b.get("slug")}
    try:
        if LIBV2_COURSES.is_dir():
            slugs.update(p.name for p in LIBV2_COURSES.iterdir() if p.is_dir())
    except OSError:
        pass
    return slugs


def _validated_slug(slug: Any) -> Optional[str]:
    """Return the slug iff it is shape-valid AND known (manifest or LibV2)."""
    slug = str(slug or "").strip()
    if not SLUG_RE.match(slug):
        return None
    if slug not in _known_course_slugs():
        return None
    return slug


def _slug_refusal(slug: Any) -> str:
    return (
        f"Refused: {str(slug)[:80]!r} is not a known course slug (must exist in "
        f"the campaign manifest or LibV2/courses/). Use library_courses or "
        f"campaign_status to list valid slugs."
    )


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _workflow_state_path(run_id: str) -> Path:
    """state/workflows/<WF-id>.json for a PRE-VALIDATED WF run id."""
    return STATE_PATH / "workflows" / f"{run_id}.json"


def _stop_all_present() -> bool:
    """True when the global STOP_ALL sentinel exists (operator-owned)."""
    for candidate in (STATE_PATH / "runs" / "STOP_ALL", STATE_PATH / "STOP_ALL"):
        try:
            if candidate.exists():
                return True
        except OSError:
            return True  # unreadable state dir → do not act
    return False


# --------------------------------------------------------------------------- #
# Read-only tools — status family
# --------------------------------------------------------------------------- #


def campaign_status() -> str:
    """Summarize the campaign manifest (counts + running slugs).

    Reads ``<campaign dir>/manifest.json`` — see ``lib.paths.campaign_dir``
    (``ED4ALL_CAMPAIGN_DIR``, default ``plans/campaign``).
    """
    manifest_path = CAMPAIGN_DIR / "manifest.json"
    if not manifest_path.is_file():
        return "no campaign configured (campaign manifest.json not found)"
    try:
        manifest = _load_json(manifest_path)
    except (OSError, ValueError) as exc:
        return f"campaign manifest unreadable ({type(exc).__name__}: {exc})"

    books = manifest.get("books") or []
    counts: Dict[str, int] = {}
    running: List[str] = []
    for book in books:
        status = str(book.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
        if status == "running":
            running.append(str(book.get("slug", "?")))

    lines = [
        f"Campaign {manifest.get('campaign', '?')!r}: {len(books)} book(s).",
        "By status: "
        + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"),
    ]
    lines.append(
        "Currently running: " + (", ".join(running) if running else "none")
    )
    return _clip("\n".join(lines))


def run_status() -> str:
    """List active/recent runs from state/runs (read-only, newest first)."""
    runs_dir = STATE_PATH / "runs"
    if not runs_dir.is_dir():
        return f"no runs found (no {runs_dir} directory)"

    rows: List[str] = []
    run_dirs = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir()), reverse=True
    )[:_RUN_STATUS_LIMIT]
    for run_dir in run_dirs:
        status = "unknown"
        workflow = "?"
        created = "?"
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = _load_json(manifest_path)
                status = str(manifest.get("status", "completed"))
                workflow = str(manifest.get("workflow_type") or "?")
                created = str(manifest.get("created_at") or "?")[:19]
            except (OSError, ValueError):
                status = "manifest-unreadable"
        rows.append(f"{run_dir.name}  status={status}  workflow={workflow}  created={created}")

    if not rows:
        return "no runs found"
    return _clip(f"Recent runs (newest {len(rows)}):\n" + "\n".join(rows))


def seat_status() -> str:
    """Probe every seat in ED4ALL_SEAT_BASE_URLS; report liveness + model ids."""
    registry = parse_seat_registry()
    if not registry:
        return (
            f"no seats registered ({ENV_SEAT_BASE_URLS} is unset/empty) — "
            f"nothing to probe"
        )
    rows: List[str] = []
    for seat_name in sorted(registry):
        base_url = registry[seat_name]
        if _probe_ready(base_url):
            model_id = _first_model_id(base_url) or "?"
            rows.append(f"{seat_name}  SERVING  {base_url}  model={model_id}")
        else:
            rows.append(f"{seat_name}  DOWN     {base_url}")
    return _clip("Registered seats:\n" + "\n".join(rows))


# --------------------------------------------------------------------------- #
# Read-only tools — reports family
# --------------------------------------------------------------------------- #

_WORKFLOWS_CACHE: Dict[str, Any] = {}


def _workflows_config() -> Optional[Dict[str, Any]]:
    """Parse config/workflows.yaml once; re-parse only on mtime change."""
    try:
        mtime = WORKFLOWS_YAML.stat().st_mtime
    except OSError:
        return None
    key = f"{WORKFLOWS_YAML}:{mtime}"
    if _WORKFLOWS_CACHE.get("key") == key:
        return _WORKFLOWS_CACHE.get("config")
    try:
        import yaml  # noqa: PLC0415 — heavy import stays lazy

        config = yaml.safe_load(WORKFLOWS_YAML.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — surfaced as a tool message
        logger.warning("assistant: workflows.yaml parse failed: %s", exc)
        return None
    _WORKFLOWS_CACHE["key"] = key
    _WORKFLOWS_CACHE["config"] = config
    return config


def list_workflows() -> str:
    """Workflow names + ordered phase lists from config/workflows.yaml."""
    config = _workflows_config()
    if not isinstance(config, dict) or not isinstance(config.get("workflows"), dict):
        return "workflow config unreadable (config/workflows.yaml missing or malformed)"
    lines: List[str] = []
    for name, spec in config["workflows"].items():
        phases = spec.get("phases") if isinstance(spec, dict) else None
        phase_names = [
            str(p.get("name", "?")) for p in phases if isinstance(p, dict)
        ] if isinstance(phases, list) else []
        lines.append(f"{name} ({len(phase_names)} phases):")
        lines.append("  " + (" -> ".join(phase_names) or "no phases declared"))
    return _clip("\n".join(lines))


def _read_workflow_doc(run_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """(doc, error). Only for a PRE-VALIDATED WF id."""
    path = _workflow_state_path(run_id)
    if not path.is_file():
        return None, f"no workflow state for {run_id} (state/workflows/{run_id}.json missing)"
    try:
        doc = _load_json(path)
    except (OSError, ValueError) as exc:
        return None, f"workflow state for {run_id} unreadable ({type(exc).__name__}: {exc})"
    if not isinstance(doc, dict):
        return None, f"workflow state for {run_id} is not a JSON object"
    return doc, ""


def run_report(run_id: str) -> str:
    """Summarize one run: status, failed/paused phase, failure reason,
    per-phase completion + gate summary from state/workflows/<id>.json."""
    valid = _validated_run_id(run_id)
    if valid is None:
        return (
            f"Refused: {str(run_id)[:80]!r} is not a valid run id "
            f"(expected WF-YYYYMMDD-xxxxxxxx)."
        )
    doc, err = _read_workflow_doc(valid)
    if doc is None:
        return err

    params = doc.get("params") or {}
    lines = [
        f"Run {valid}  status={doc.get('status', '?')}  type={doc.get('type', '?')}",
        f"course={params.get('course_name', '?')}  "
        f"created={str(doc.get('created_at', '?'))[:19]}  "
        f"updated={str(doc.get('updated_at', '?'))[:19]}",
    ]
    for key in ("failed_phase", "paused_phase"):
        if doc.get(key):
            lines.append(f"{key}={doc[key]}")
    if doc.get("failure_reason"):
        lines.append("failure_reason: " + _snip(str(doc["failure_reason"]), 400))

    tasks = doc.get("tasks") or []
    task_counts: Dict[str, int] = {}
    for task in tasks:
        st = str(task.get("status", "?")) if isinstance(task, dict) else "?"
        task_counts[st] = task_counts.get(st, 0) + 1
    if task_counts:
        lines.append(
            "tasks: " + ", ".join(f"{k}={v}" for k, v in sorted(task_counts.items()))
        )

    phase_outputs = doc.get("phase_outputs") or {}
    if isinstance(phase_outputs, dict) and phase_outputs:
        lines.append("Phases (from phase_outputs):")
        for phase, out in phase_outputs.items():
            if not isinstance(out, dict):
                continue
            gates = out.get("_gate_results") or []
            failed_gates = sum(
                1 for g in gates if isinstance(g, dict) and not g.get("passed", True)
            )
            lines.append(
                f"  {phase}: completed={bool(out.get('_completed'))} "
                f"gates_passed={out.get('_gates_passed', '?')} "
                f"gate_results={len(gates)} failing={failed_gates}"
            )
    else:
        lines.append("Phases: no phase_outputs recorded yet.")
    return _clip(_redact("\n".join(lines)))


def gate_report(run_id: str, phase: str = "") -> str:
    """Gate results for a run (optionally one phase): gate_id, pass/fail,
    severity, top issue codes + truncated messages. Failing gates first."""
    valid = _validated_run_id(run_id)
    if valid is None:
        return (
            f"Refused: {str(run_id)[:80]!r} is not a valid run id "
            f"(expected WF-YYYYMMDD-xxxxxxxx)."
        )
    doc, err = _read_workflow_doc(valid)
    if doc is None:
        return err
    phase_outputs = doc.get("phase_outputs") or {}
    if not isinstance(phase_outputs, dict) or not phase_outputs:
        return f"Run {valid}: no phase_outputs (no gates recorded yet)."

    phase = str(phase or "").strip()
    if phase:
        if phase not in phase_outputs:
            return (
                f"Run {valid}: no phase {phase!r} in phase_outputs "
                f"(have: {', '.join(sorted(phase_outputs))})."
            )
        selected = {phase: phase_outputs[phase]}
    else:
        selected = phase_outputs

    lines = [f"Gate report for {valid}" + (f" phase={phase}" if phase else "")]
    shown = 0
    for phase_name, out in selected.items():
        gates = out.get("_gate_results") if isinstance(out, dict) else None
        if not isinstance(gates, list) or not gates:
            continue
        passed = [g for g in gates if isinstance(g, dict) and g.get("passed", True)]
        failing = [g for g in gates if isinstance(g, dict) and not g.get("passed", True)]
        lines.append(
            f"{phase_name}: {len(gates)} gate(s), {len(passed)} passed, "
            f"{len(failing)} FAILED"
        )
        for gate in failing + passed:
            issues = gate.get("issues") or []
            sev_counts: Dict[str, int] = {}
            for issue in issues:
                sev = str(issue.get("severity", "?")) if isinstance(issue, dict) else "?"
                sev_counts[sev] = sev_counts.get(sev, 0) + 1
            interesting = (not gate.get("passed", True)) or sev_counts
            if not interesting or shown >= 12:
                continue
            shown += 1
            lines.append(
                f"  [{'FAIL' if not gate.get('passed', True) else 'pass'}] "
                f"{gate.get('gate_id', '?')} severity={gate.get('severity', '?')} "
                f"issues=" + (
                    ",".join(f"{k}:{v}" for k, v in sorted(sev_counts.items())) or "0"
                )
            )
            for issue in issues[:3]:
                if isinstance(issue, dict):
                    lines.append(
                        f"    {issue.get('code', '?')}: "
                        + _snip(str(issue.get("message", "")), 160)
                    )
            if len(issues) > 3:
                lines.append(f"    …and {len(issues) - 3} more issue(s)")
    if len(lines) == 1:
        lines.append("No gate results recorded in any phase yet.")
    return _clip(_redact("\n".join(lines)))


def _tail_file(path: Path, lines: int) -> str:
    """Bounded tail: read only the trailing bytes, return the last N lines."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            # 400 bytes/line generous upper bound keeps the read bounded.
            handle.seek(max(0, size - lines * 400))
            data = handle.read().decode("utf-8", "replace")
    except OSError as exc:
        return f"log unreadable ({type(exc).__name__}: {exc})"
    tail = data.splitlines()[-lines:]
    return "\n".join(tail)


def tail_log(target: str, lines: int = 50) -> str:
    """Bounded log tail. ``target`` is a campaign slug (resolves to
    ``<campaign dir>/logs/<slug>.log``) or a run id (resolves to *.log
    files under state/runs/<id>/). Path resolution goes ONLY through these
    two whitelisted patterns; lines clamped to 1..200."""
    lines = max(1, min(MAX_LOG_TAIL_LINES, _coerce_int(lines, 50)))
    target = str(target or "").strip()

    run_id = _validated_run_id(target, allow_ttc=True)
    if run_id is not None:
        run_dir = STATE_PATH / "runs" / run_id
        candidates = sorted(
            (p for p in run_dir.glob("**/*.log") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if run_dir.is_dir() else []
        if not candidates:
            return f"no log files found for run {run_id} under state/runs/{run_id}/"
        chosen = candidates[0]
        body = _tail_file(chosen, lines)
        return _clip(_redact(f"[{chosen.name}] last {lines} line(s):\n{body}"))

    if SLUG_RE.match(target):
        log_path = CAMPAIGN_DIR / "logs" / f"{target}.log"
        if not log_path.is_file():
            return (
                f"no campaign log for {target!r} "
                f"(logs/{target}.log not found under the campaign dir)"
            )
        body = _tail_file(log_path, lines)
        return _clip(_redact(f"[{target}.log] last {lines} line(s):\n{body}"))

    return (
        f"Refused: {target[:80]!r} is neither a valid run id "
        f"(WF-YYYYMMDD-xxxxxxxx / TTC_...) nor a valid campaign slug."
    )


#: Max WARN/FAIL findings surfaced in one doctor payload (loud but bounded —
#: keeps a fully-degraded box from blowing the tool-result char cap).
_DOCTOR_MAX_FINDINGS = 25


def _summarize_doctor_json(payload: Any, returncode: int) -> str:
    """Turn the ``ed4all doctor --json`` payload into a compact, LLM-sized
    report. Keeps EVERY warn/fail (name + severity + summary + remediation,
    bounded), counts the OK/INFO checks without dumping their bodies, and
    surfaces the exit-code verdict. Raises on a shape it cannot read so the
    caller falls back to the text path."""
    results = payload["results"]  # KeyError → caller falls back to text
    if not isinstance(results, list):
        raise ValueError("doctor --json 'results' is not a list")
    exit_code = payload.get("exit_code", returncode)
    verdict = {0: "OK", 1: "DEGRADED (warnings)", 2: "DANGER (failures)"}.get(
        exit_code, f"exit {exit_code}"
    )

    counts: Dict[str, int] = {}
    findings: List[Dict[str, Any]] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        sev = str(row.get("severity", "?")).lower()
        counts[sev] = counts.get(sev, 0) + 1
        if sev in ("warn", "fail"):
            findings.append(row)

    # fail before warn, then stable by group/name for a deterministic report.
    findings.sort(key=lambda r: (
        0 if str(r.get("severity", "")).lower() == "fail" else 1,
        str(r.get("group", "")), str(r.get("name", "")),
    ))

    lines = [
        f"doctor verdict: {verdict} "
        + "(" + (", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no checks") + ")"
    ]
    summary = payload.get("summary")
    if summary:
        lines.append(_snip(str(summary), 200))

    if not findings:
        lines.append("No WARN/FAIL checks — all groups nominal.")
        return _clip(_redact("\n".join(lines)))

    lines.append(f"WARN/FAIL findings ({len(findings)}):")
    for row in findings[:_DOCTOR_MAX_FINDINGS]:
        sev = str(row.get("severity", "?")).upper()
        name = str(row.get("name", "?"))
        grp = str(row.get("group", "?"))
        lines.append(f"[{sev}] {name} ({grp}): " + _snip(str(row.get("summary", "")), 200))
        remediation = str(row.get("remediation", "")).strip()
        if remediation:
            lines.append("  fix: " + _snip(remediation, 200))
    if len(findings) > _DOCTOR_MAX_FINDINGS:
        lines.append(f"…and {len(findings) - _DOCTOR_MAX_FINDINGS} more WARN/FAIL finding(s)")
    return _clip(_redact("\n".join(lines)))


def doctor(group: str = "") -> str:
    """Run ``ed4all doctor --json`` (fixed argv; optional group from the known
    enum) and return a structured, secret-filtered finding summary sized for
    the model: the exit-code verdict + severity counts, every WARN/FAIL with
    its remediation (bounded), and OK/INFO counted but not dumped. Falls back
    to the formatted-text path if ``--json`` is unavailable/unparseable (older
    CLI)."""
    group = str(group or "").strip().lower()
    base_argv = [sys.executable, "-m", "cli.main", "doctor"]
    if group:
        if group not in DOCTOR_GROUPS:
            return (
                f"Refused: {group!r} is not a doctor group. "
                f"Valid groups: {', '.join(DOCTOR_GROUPS)}."
            )
        base_argv += ["-g", group]

    # Preferred path: structured --json.
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, enum-validated group
            base_argv + ["--json"], capture_output=True, text=True,
            timeout=180, cwd=str(PROJECT_ROOT),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"`ed4all doctor` failed to execute: {type(exc).__name__}: {exc}"
    try:
        payload = json.loads(proc.stdout or "")
        if not isinstance(payload, dict):
            raise ValueError("doctor --json output is not an object")
        return _summarize_doctor_json(payload, proc.returncode)
    except (ValueError, KeyError, TypeError) as exc:
        logger.info("assistant: doctor --json unparseable (%s); text fallback", exc)

    # Fallback: the formatted-text path (older CLI with no --json).
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, enum-validated group
            base_argv, capture_output=True, text=True, timeout=180, cwd=str(PROJECT_ROOT)
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"`ed4all doctor` failed to execute: {type(exc).__name__}: {exc}"
    output = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    verdict = {0: "OK", 1: "DEGRADED (warnings)", 2: "DANGER (failures)"}.get(
        proc.returncode, f"exit {proc.returncode}"
    )
    return _clip(_redact(f"doctor verdict: {verdict}\n{output}"))


def _fmt_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "?"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def build_cost(course_slug: str) -> str:
    """Summarize build_cost_report.json for a course: per-phase wall clock
    (+ LLM calls/tokens when metered)."""
    slug = _validated_slug(course_slug)
    if slug is None:
        return _slug_refusal(course_slug)
    candidates = [
        LIBV2_COURSES / slug / "build_cost_report.json",
        PROJECT_ROOT / "Trainforge" / "courses" / slug / "build_cost_report.json",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return (
            f"no build_cost_report.json for {slug} (checked the LibV2 course dir "
            f"and the Trainforge fallback). The BuildCostAggregator writes it "
            f"post-run."
        )
    try:
        report = _load_json(path)
    except (OSError, ValueError) as exc:
        return f"build cost report unreadable ({type(exc).__name__}: {exc})"

    totals = report.get("totals") or {}
    lines = [
        f"Build cost for {slug} (run {report.get('run_id', '?')}):",
        f"totals: phases={totals.get('phase_count', '?')} "
        f"measured={totals.get('measured_phase_count', '?')} "
        f"wall_clock={_fmt_seconds(totals.get('total_wall_clock_seconds'))}",
    ]
    for key in ("total_llm_calls", "total_prompt_tokens", "total_completion_tokens"):
        if key in totals:
            lines[-1] += f" {key.replace('total_', '')}={totals[key]}"
    phases = report.get("phases") or []
    rows: List[Tuple[float, str]] = []
    for row in phases:
        if not isinstance(row, dict):
            continue
        name = str(row.get("phase") or row.get("phase_name") or row.get("name") or "?")
        seconds = 0.0
        for sec_key in ("wall_clock_seconds", "wall_seconds", "duration_seconds"):
            try:
                seconds = float(row.get(sec_key))
                break
            except (TypeError, ValueError):
                continue
        extra = ""
        llm = row.get("llm") if isinstance(row.get("llm"), dict) else {}
        if llm:
            extra = (
                f"  llm_calls={llm.get('calls', '?')} "
                f"tokens={llm.get('prompt_tokens', '?')}+{llm.get('completion_tokens', '?')}"
            )
        rows.append((seconds, f"  {name}: {_fmt_seconds(seconds)}{extra}"))
    rows.sort(key=lambda r: r[0], reverse=True)
    if rows:
        lines.append("Per-phase (slowest first):")
        lines.extend(line for _, line in rows[:20])
        if len(rows) > 20:
            lines.append(f"  …and {len(rows) - 20} more phase(s)")
    else:
        lines.append("No per-phase rows recorded.")
    return _clip("\n".join(lines))


def _summarize_json_doc(doc: Any) -> str:
    """Top-level status/counts summary of a report JSON — never a raw dump."""
    if not isinstance(doc, dict):
        return f"(top-level {type(doc).__name__}, length {len(doc) if hasattr(doc, '__len__') else '?'})"
    priority = (
        "schema_version", "course_status", "promotion_decision", "status",
        "passed", "generated_at", "course_code", "run_id",
    )
    lines: List[str] = []
    seen = set()
    for key in priority:
        if key in doc and not isinstance(doc[key], (dict, list)):
            lines.append(f"{key}: {_snip(str(doc[key]), 120)}")
            seen.add(key)
    for key, value in doc.items():
        if key in seen:
            continue
        if isinstance(value, list):
            lines.append(f"{key}: list[{len(value)}]")
        elif isinstance(value, dict):
            scalars = {
                k: v for k, v in value.items() if not isinstance(v, (dict, list))
            }
            preview = ", ".join(f"{k}={_snip(str(v), 40)}" for k, v in list(scalars.items())[:8])
            lines.append(f"{key}: dict[{len(value)}]" + (f" ({preview})" if preview else ""))
        else:
            lines.append(f"{key}: {_snip(str(value), 120)}")
        if len(lines) >= 30:
            lines.append(f"…and {len(doc) - len(seen) - 30} more top-level key(s)")
            break
    return "\n".join(lines)


def aggregator_report(course_slug: str, name: str) -> str:
    """Read + summarize one known aggregator report for a course. ``name``
    must be from the fixed enum (see the tool schema)."""
    slug = _validated_slug(course_slug)
    if slug is None:
        return _slug_refusal(course_slug)
    name = str(name or "").strip().lower()
    if name not in AGGREGATOR_REPORTS:
        return (
            f"Refused: {name!r} is not a known aggregator report. Valid names: "
            + ", ".join(sorted(AGGREGATOR_REPORTS)) + "."
        )

    path: Optional[Path] = None
    if name == "validation_report":
        exports = PROJECT_ROOT / "Courseforge" / "exports"
        matches = sorted(
            exports.glob(f"PROJ-{slug}-*/courseforge_validation_report.json"),
            reverse=True,
        ) if exports.is_dir() else []
        path = matches[0] if matches else None
    else:
        for rel in AGGREGATOR_REPORTS[name]:
            candidate = LIBV2_COURSES / slug / rel
            if candidate.is_file():
                path = candidate
                break
    if path is None:
        return f"no {name} report found for {slug} (it is written post-run; some reports are flag-gated)."
    try:
        doc = _load_json(path)
    except (OSError, ValueError) as exc:
        return f"{name} report unreadable ({type(exc).__name__}: {exc})"
    return _clip(_redact(f"{name} for {slug} ({path.name}):\n" + _summarize_json_doc(doc)))


def _flag_table_rows(path: Path) -> List[str]:
    try:
        return [
            line for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("| `")
        ]
    except OSError:
        return []


def flag_lookup(name_or_prefix: str) -> str:
    """Search the behavior-flag tables (docs/operations/behavior-flags.md +
    root CLAUDE.md index) for a flag name/prefix; up to 5 truncated rows."""
    query = str(name_or_prefix or "").strip().strip("`")
    if not re.match(r"^[A-Za-z0-9_.*-]{2,64}$", query):
        return "Refused: flag query must be a 2-64 char flag name or prefix (letters/digits/underscore)."
    needle = query.lower().rstrip("*")
    matches: List[str] = []
    seen_flags = set()
    for source in (BEHAVIOR_FLAGS_MD, ROOT_CLAUDE_MD):
        for row in _flag_table_rows(source):
            cells = [c.strip() for c in row.split("|")]
            # cells[1] is "`FLAG_NAME`"
            if len(cells) < 4:
                continue
            flag = cells[1].strip("`")
            if needle not in flag.lower() or flag in seen_flags:
                continue
            seen_flags.add(flag)
            default = cells[2]
            purpose = _snip(" ".join(cells[3:]).strip(), 400)
            matches.append(f"{flag} (default: {default})\n  {purpose}")
            if len(matches) >= 5:
                break
        if len(matches) >= 5:
            break
    if not matches:
        return (
            f"No behavior flag matching {query!r} in the root-owned tables. "
            f"Subsystem-prefixed flags (SEMANTIK_*, COURSEFORGE_*, TRAINFORGE_*) "
            f"live in their subsystem CLAUDE.md files."
        )
    return _clip("\n".join(matches))


# --------------------------------------------------------------------------- #
# Read-only tools — library family
# --------------------------------------------------------------------------- #


def library_courses() -> str:
    """List LibV2/courses slugs with manifest / chunkset / vector-index presence."""
    if not LIBV2_COURSES.is_dir():
        return f"no LibV2 course library (no {LIBV2_COURSES} directory)"
    rows: List[str] = []
    for course_dir in sorted(p for p in LIBV2_COURSES.iterdir() if p.is_dir()):
        manifest = (course_dir / "manifest.json").is_file() or (
            course_dir / "course_manifest.json"
        ).is_file()
        chunks = any(
            (course_dir / kind / "chunks.jsonl").is_file()
            for kind in ("semantik_chunks", "imscc_chunks", "dart_chunks")  # legacy dir fallback, legacy-token: allow
        )
        index = (course_dir / "vector_index" / "manifest.json").is_file()
        rows.append(
            f"{course_dir.name}  manifest={'y' if manifest else 'n'} "
            f"chunks={'y' if chunks else 'n'} index={'y' if index else 'n'}"
        )
    if not rows:
        return "LibV2 course library is empty."
    header = f"LibV2 courses ({len(rows)}):"
    if len(rows) > 50:
        rows = rows[:50] + [f"…and {len(rows) - 50} more course(s)"]
    return _clip("\n".join([header] + rows))


def _objectives_doc_for(slug: str) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """Locate the objectives doc: LibV2 archive shape first, then the
    newest Courseforge export shape."""
    libv2_path = LIBV2_COURSES / slug / "objectives.json"
    if libv2_path.is_file():
        try:
            return libv2_path, _load_json(libv2_path)
        except (OSError, ValueError):
            pass
    exports = PROJECT_ROOT / "Courseforge" / "exports"
    matches = sorted(
        exports.glob(f"PROJ-{slug}-*/01_learning_objectives/synthesized_objectives.json"),
        reverse=True,
    ) if exports.is_dir() else []
    for candidate in matches:
        try:
            return candidate, _load_json(candidate)
        except (OSError, ValueError):
            continue
    return None


def course_objectives(course_slug: str) -> str:
    """TO/CO counts + TO titles from the course's objectives doc (LibV2
    archive shape or Courseforge export shape)."""
    slug = _validated_slug(course_slug)
    if slug is None:
        return _slug_refusal(course_slug)
    located = _objectives_doc_for(slug)
    if located is None:
        return f"no objectives doc found for {slug} (neither the LibV2 archive nor a Courseforge export has one)."
    path, doc = located
    tos = (
        doc.get("terminal_objectives")
        or doc.get("terminal_outcomes")
        or []
    )
    cos = (
        doc.get("component_objectives")
        or doc.get("course_objectives")
        or doc.get("learning_outcomes")
        or []
    )
    lines = [
        f"Objectives for {slug} ({path.name}): "
        f"{len(tos)} terminal objective(s), {len(cos)} course objective(s)."
    ]
    for to in tos[:15]:
        if not isinstance(to, dict):
            continue
        lines.append(
            f"  {to.get('id', '?')} [{to.get('bloom_level', '?')}] "
            + _snip(str(to.get("statement", "")), 110)
        )
    if len(tos) > 15:
        lines.append(f"  …and {len(tos) - 15} more TO(s)")
    return _clip("\n".join(lines))


def ask_course(course_slug: str, question: str) -> str:
    """Grounded course Q&A through lib/retrieval/grounded_answer. Refuses
    when the course has no vector index or a pipeline run is active."""
    slug = _validated_slug(course_slug)
    if slug is None:
        return _slug_refusal(course_slug)
    question = str(question or "").strip()
    if not question:
        return "Refused: ask_course needs a non-empty question."
    question = question[:2000]

    index_manifest = LIBV2_COURSES / slug / "vector_index" / "manifest.json"
    if not index_manifest.is_file():
        return (
            f"Refused: course {slug} has no vector index "
            f"(vector_index/manifest.json missing) — the grounded-answer path "
            f"needs the retrieval index built by the vector_indexing phase."
        )
    if _ed4all_run_active():
        return (
            "Refused: a pipeline run is currently active (GPU contention rule "
            "— the run owns the card; grounded answering would compete with "
            "it). Try again after the run finishes."
        )

    try:
        from lib.retrieval.grounded_answer import answer_course_question  # noqa: PLC0415

        answer = answer_course_question(
            PROJECT_ROOT, slug, question, engine="hybrid-rrf", limit=6
        )
    except Exception as exc:  # noqa: BLE001 — typed retrieval errors surface loudly
        return f"ask_course failed: {type(exc).__name__}: {_snip(str(exc), 300)}"

    lines = [f"[{answer.status}] course={slug}"]
    if answer.answer_text:
        lines.append(_clip(str(answer.answer_text), 1800))
    if answer.refusal:
        lines.append("refusal: " + _snip(json.dumps(answer.refusal), 300))
    cited = [c.chunk_id for c in (answer.citations or [])]
    lines.append("cited chunks: " + (", ".join(cited[:10]) or "none"))
    confidence = answer.confidence or {}
    conf_bits = ", ".join(
        f"{k}={_snip(str(v), 40)}"
        for k, v in confidence.items()
        if not isinstance(v, (dict, list))
    )
    if conf_bits:
        lines.append("confidence: " + conf_bits)
    if answer.groundedness and isinstance(answer.groundedness, dict):
        verdict = answer.groundedness.get("verdict")
        if verdict is not None:
            lines.append(f"groundedness verdict: {verdict}")
    return _clip(_redact("\n".join(lines)))


# --------------------------------------------------------------------------- #
# Read-only tools — seat-swap environment setup family
# --------------------------------------------------------------------------- #

#: generate_seat_env validation shapes (the model supplies STRUCTURED data,
#: never raw env strings or paths that skip these guards).
SEAT_NAME_RE = re.compile(r"^[a-z0-9-]{1,32}$")
CONTAINER_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
#: Hard cap on seats in a generated env snippet.
MAX_GENERATED_SEATS = 8

#: Defaults mirrored from lib/vllm_container_lifecycle.py resolvers.
_DEFAULT_SEAT_LOAD_TIMEOUT = 1200
_DEFAULT_SEAT_COHERENCE_ATTEMPTS = 3


def _spec_script_path(spec: str) -> Optional[str]:
    """The launch-spec's script path when it looks like one (absolute first
    token), else None (a free-form command is not path-checked)."""
    first = str(spec).strip().split()[0] if str(spec).strip() else ""
    return first if first.startswith("/") else None


def seat_env_doctor() -> str:
    """READ-ONLY audit of the seat-swap env stack (ED4ALL_SEAT_SCHEDULE /
    _SEAT_BASE_URLS / _VLLM_CONTAINERS / _SEAT_LAUNCH_SPECS / timeout /
    coherence). Parses through the lifecycle lib's OWN parsers and
    cross-checks seat→container resolution, launch-spec paths, loopback
    URLs, and duplicate ports."""
    findings: List[str] = []

    def _add(kind: str, message: str) -> None:
        findings.append(f"[{kind}] {message}")

    schedule_on = resolve_seat_schedule_mode()
    _add(
        "ok" if schedule_on else "info",
        f"{ENV_SEAT_SCHEDULE}: "
        + ("ON (pipeline owns seat lifecycle at phase boundaries)" if schedule_on
           else "off/unset (seat schedule disabled — the other vars are inert until it is on)"),
    )

    seats = parse_seat_registry()
    containers = parse_container_registry()
    specs = parse_seat_launch_specs()

    if not seats:
        _add("missing", f"{ENV_SEAT_BASE_URLS} is unset/empty — no seats registered.")
    else:
        _add("ok", f"{ENV_SEAT_BASE_URLS}: {len(seats)} seat(s) parsed "
             f"({', '.join(sorted(seats))}).")
    if not containers:
        _add("missing", f"{ENV_VLLM_CONTAINERS} is unset/empty — no base_url→container map.")
    if not specs:
        _add(
            "missing" if schedule_on else "info",
            f"{ENV_SEAT_LAUNCH_SPECS} is unset/empty — no cold-recreate "
            f"self-heal (a mode-collapsed seat cannot be relaunched).",
        )

    ports_seen: Dict[str, str] = {}
    for seat_name in sorted(seats):
        base_url = seats[seat_name].rstrip("/")
        if not _is_loopback_host(base_url):
            _add("mismatch", f"seat {seat_name}: base_url {base_url} is NOT loopback "
                 f"(seats must be local).")
        port_match = re.search(r":(\d+)", base_url)
        if port_match:
            port = port_match.group(1)
            if port in ports_seen and ports_seen[port] != seat_name:
                _add("mismatch", f"seat {seat_name}: port {port} duplicates seat "
                     f"{ports_seen[port]} — two seats cannot share a port.")
            ports_seen.setdefault(port, seat_name)
        container = containers.get(base_url)
        if container:
            _add("ok", f"seat {seat_name}: {base_url} -> container {container}.")
        else:
            _add("mismatch", f"seat {seat_name}: {base_url} has NO entry in "
                 f"{ENV_VLLM_CONTAINERS} — lifecycle cannot start/stop it.")
        spec = specs.get(seat_name)
        if spec is None:
            if schedule_on:
                _add("mismatch", f"seat {seat_name}: no {ENV_SEAT_LAUNCH_SPECS} entry "
                     f"— cannot self-heal a mode-collapse.")
        else:
            script = _spec_script_path(spec)
            if script is not None and not Path(script).exists():
                _add("mismatch", f"seat {seat_name}: launch script {script} does not "
                     f"exist on disk.")
            elif script is not None:
                _add("ok", f"seat {seat_name}: launch script {script} exists.")
            else:
                _add("info", f"seat {seat_name}: launch spec is a free-form command "
                     f"(not path-checked).")
    for spec_seat in sorted(set(specs) - set(seats)):
        _add("mismatch", f"launch spec names seat {spec_seat!r} which is not in "
             f"{ENV_SEAT_BASE_URLS}.")

    timeout = resolve_seat_load_timeout()
    attempts = resolve_seat_coherence_attempts()
    timeout_src = "override" if os.environ.get(ENV_SEAT_LOAD_TIMEOUT_SECONDS, "").strip() else "default"
    attempts_src = "override" if os.environ.get(ENV_SEAT_COHERENCE_ATTEMPTS, "").strip() else "default"
    _add("ok", f"{ENV_SEAT_LOAD_TIMEOUT_SECONDS}: {timeout:g}s ({timeout_src}).")
    _add("ok", f"{ENV_SEAT_COHERENCE_ATTEMPTS}: {attempts} ({attempts_src}).")

    counts: Dict[str, int] = {}
    for finding in findings:
        kind = finding[1:finding.index("]")]
        counts[kind] = counts.get(kind, 0) + 1
    header = "Seat-env audit: " + ", ".join(
        f"{k}={v}" for k, v in sorted(counts.items())
    )
    return _clip(_redact("\n".join([header] + findings)))


#: Fixed docker-ps argv (+ the sg-docker group-wrap fallback pattern used by
#: the operator campaign-harness drivers). Read-only discovery.
_DOCKER_PS_FORMAT = "{{.Names}}\t{{.Status}}\t{{.Image}}"
_DOCKER_PS_ARGV = ["docker", "ps", "-a", "--format", _DOCKER_PS_FORMAT]
_DOCKER_PS_SG_ARGV = [
    "sg", "docker", "-c", f"docker ps -a --format '{_DOCKER_PS_FORMAT}'"
]


def list_gpu_containers() -> str:
    """READ-ONLY container discovery (docker ps -a, names/status/image only,
    max 30 rows) — helps map containers to seats for the seat-swap env."""
    output: Optional[str] = None
    errors: List[str] = []
    for argv in (_DOCKER_PS_ARGV, _DOCKER_PS_SG_ARGV):
        try:
            proc = subprocess.run(  # noqa: S603 — fixed argv, no model input
                argv, capture_output=True, text=True, timeout=20
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{argv[0]}: {type(exc).__name__}: {exc}")
            continue
        if proc.returncode == 0:
            output = proc.stdout
            break
        errors.append(f"{argv[0]}: exit {proc.returncode}")
    if output is None:
        return (
            "docker unavailable (" + "; ".join(errors) + ") — cannot list "
            "containers. Is docker installed and the user in the docker group?"
        )
    rows = [line for line in output.splitlines() if line.strip()]
    if not rows:
        return "no containers found (docker ps -a returned nothing)."
    shown = rows[:30]
    lines = [f"Containers ({len(rows)} total, showing {len(shown)}): name / status / image"]
    for row in shown:
        parts = row.split("\t")
        lines.append("  " + " / ".join(_snip(p, 60) for p in parts[:3]))
    return _clip(_redact("\n".join(lines)))


def generate_seat_env(
    seats: Any,
    load_timeout: Any = None,
    coherence_attempts: Any = None,
) -> str:
    """Generate a ready-to-source seat-swap env snippet from a VALIDATED
    seat structure. Returns the export block as TEXT — it writes NO file;
    the user saves it (e.g. next to run-env) and sources it before a run."""
    if not isinstance(seats, list) or not seats:
        return (
            "Refused: seats must be a non-empty list of "
            "{seat_name, base_url, container, launch_script?} objects."
        )
    if len(seats) > MAX_GENERATED_SEATS:
        return f"Refused: at most {MAX_GENERATED_SEATS} seats (got {len(seats)})."

    validated: List[Dict[str, str]] = []
    seen_names: set = set()
    seen_urls: set = set()
    for i, entry in enumerate(seats):
        if not isinstance(entry, dict):
            return f"Refused: seats[{i}] is not an object."
        name = str(entry.get("seat_name") or "").strip()
        if not SEAT_NAME_RE.match(name):
            return (
                f"Refused: seats[{i}].seat_name {name[:40]!r} is invalid "
                f"(need ^[a-z0-9-]{{1,32}}$)."
            )
        if name in seen_names:
            return f"Refused: duplicate seat_name {name!r}."
        base_url = str(entry.get("base_url") or "").strip().rstrip("/")
        if not re.match(r"^https?://", base_url) or not _is_loopback_host(base_url):
            return (
                f"Refused: seats[{i}].base_url {base_url[:60]!r} must be a "
                f"loopback http(s) URL (localhost / 127.0.0.1 / ::1) — seats "
                f"are local-only."
            )
        if base_url in seen_urls:
            return f"Refused: duplicate base_url {base_url!r}."
        container = str(entry.get("container") or "").strip()
        if not CONTAINER_NAME_RE.match(container):
            return (
                f"Refused: seats[{i}].container {container[:40]!r} is invalid "
                f"(need ^[A-Za-z0-9._-]{{1,64}}$)."
            )
        launch_script = entry.get("launch_script")
        if launch_script is not None and str(launch_script).strip():
            launch_script = str(launch_script).strip()
            if not launch_script.startswith("/") or not Path(launch_script).is_file():
                return (
                    f"Refused: seats[{i}].launch_script {launch_script[:80]!r} "
                    f"must be an EXISTING absolute file path (or omitted)."
                )
        else:
            launch_script = ""
        seen_names.add(name)
        seen_urls.add(base_url)
        validated.append(
            {
                "seat_name": name,
                "base_url": base_url,
                "container": container,
                "launch_script": launch_script,
            }
        )

    timeout = _coerce_int(load_timeout, _DEFAULT_SEAT_LOAD_TIMEOUT) if load_timeout is not None else _DEFAULT_SEAT_LOAD_TIMEOUT
    if timeout <= 0:
        timeout = _DEFAULT_SEAT_LOAD_TIMEOUT
    attempts = _coerce_int(coherence_attempts, _DEFAULT_SEAT_COHERENCE_ATTEMPTS) if coherence_attempts is not None else _DEFAULT_SEAT_COHERENCE_ATTEMPTS
    if attempts <= 0:
        attempts = _DEFAULT_SEAT_COHERENCE_ATTEMPTS

    base_urls_tok = ",".join(f"{s['seat_name']}={s['base_url']}" for s in validated)
    containers_tok = ",".join(f"{s['base_url']}={s['container']}" for s in validated)
    specs_tok = ";".join(
        f"{s['seat_name']}={s['launch_script']}"
        for s in validated
        if s["launch_script"]
    )

    lines = [
        "# ------------------------------------------------------------------",
        "# Generated seat-schedule env (pipeline-owned vLLM seat lifecycle).",
        "# Save this to a file (e.g. seat-schedule.env) and `source` it",
        "# alongside your run env BEFORE `ed4all run` — NEVER mid-build.",
        "# Canonical template: docs/operations/seat-schedule.env.example",
        "# ------------------------------------------------------------------",
        "",
        "# Master switch — enforce each phase's workflows.yaml `seats:` block.",
        f"export {ENV_SEAT_SCHEDULE}=1",
        "",
        "# Logical seat name -> vLLM base URL.",
        f'export {ENV_SEAT_BASE_URLS}="{base_urls_tok}"',
        "",
        "# base URL -> docker container name.",
        f'export {ENV_VLLM_CONTAINERS}="{containers_tok}"',
    ]
    if specs_tok:
        lines += [
            "",
            "# Seat -> launch SCRIPT (the cold-recreate self-heal path; ';'-separated).",
            f'export {ENV_SEAT_LAUNCH_SPECS}="{specs_tok}"',
        ]
    else:
        lines += [
            "",
            f"# No launch scripts given: {ENV_SEAT_LAUNCH_SPECS} omitted — a",
            "# mode-collapsed seat cannot self-heal (warm docker start only).",
        ]
    lines += [
        "",
        "# LIVENESS load-wait ceiling (s) — a ceiling, not a sleep.",
        f"export {ENV_SEAT_LOAD_TIMEOUT_SECONDS}={timeout}",
        "",
        "# Bounded content-coherence probe attempts once a seat is live.",
        f"export {ENV_SEAT_COHERENCE_ATTEMPTS}={attempts}",
        "",
        "# Verify after sourcing: seat_env_doctor, then seat_status.",
    ]
    return _clip("\n".join(lines), 3800)


# --------------------------------------------------------------------------- #
# Mutating tools (refusal-first guards on every one)
# --------------------------------------------------------------------------- #


def _ed4all_run_active() -> bool:
    """True when an ``ed4all run`` process is already active (pgrep -f)."""
    try:
        proc = subprocess.run(
            ["pgrep", "-f", "ed4all run"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        # Cannot verify → refuse to spawn (the caller reports the reason).
        logger.warning("assistant: pgrep probe failed (%s); treating as active", exc)
        return True
    return proc.returncode == 0


def _spawn_detached(argv: List[str], log_stem: str) -> str:
    """Spawn a fixed argv detached with output to the campaign logs dir."""
    log_dir = CAMPAIGN_DIR / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{log_stem}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        with open(log_path, "ab") as log_handle:
            proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
                argv,
                cwd=str(PROJECT_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as exc:
        return f"Failed to spawn: {type(exc).__name__}: {exc}"
    return f"spawned detached (pid {proc.pid}); log: {log_path}"


def start_next_book() -> str:
    """Spawn the campaign driver detached — IF safe to do so.

    Refuses (with the reason) when the driver script does not exist or an
    ``ed4all run`` process is already active. On success the driver runs in
    its own session (``start_new_session=True``) with output appended to the
    campaign logs dir.
    """
    driver = CAMPAIGN_DIR / "run_next.py"
    if not driver.is_file():
        return (
            f"Refused: campaign driver {driver} does not exist — "
            f"no campaign is configured on this checkout."
        )
    if _ed4all_run_active():
        return (
            "Refused: an `ed4all run` process is already active (or activity "
            "could not be verified). One build at a time — check run_status "
            "and try again after it finishes."
        )

    log_dir = CAMPAIGN_DIR / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"assistant_start_{time.strftime('%Y%m%d_%H%M%S')}.log"
        with open(log_path, "ab") as log_handle:
            proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
                [sys.executable, str(driver)],
                cwd=str(PROJECT_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as exc:
        return f"Failed to spawn campaign driver: {type(exc).__name__}: {exc}"
    return (
        f"Campaign driver spawned detached (pid {proc.pid}); "
        f"log: {log_path}. Use run_status / campaign_status to follow it."
    )


def start_book(slug: str) -> str:
    """Start ONE specific pending campaign book (validated manifest slug).
    Same no-active-run guard as start_next_book."""
    slug = str(slug or "").strip()
    if not SLUG_RE.match(slug):
        return f"Refused: {slug[:80]!r} is not a valid slug shape. Nothing was started."
    driver = CAMPAIGN_DIR / "run_next.py"
    if not driver.is_file():
        return (
            f"Refused: campaign driver {driver} does not exist — "
            f"no campaign is configured on this checkout."
        )
    books = {str(b.get("slug")): b for b in _campaign_manifest_books()}
    book = books.get(slug)
    if book is None:
        return (
            f"Refused: {slug!r} is not in the campaign manifest. "
            f"Use campaign_status to see the book list."
        )
    if str(book.get("status")) != "pending":
        return (
            f"Refused: book {slug!r} is status={book.get('status')!r}, not "
            f"'pending' — only a pending book can be started."
        )
    if _ed4all_run_active():
        return (
            "Refused: an `ed4all run` process is already active (or activity "
            "could not be verified). One build at a time."
        )
    result = _spawn_detached(
        [sys.executable, str(driver), "--slug", slug], f"assistant_start_{slug}"
    )
    if result.startswith("Failed"):
        return result
    return f"Campaign driver for book {slug!r} {result}. Follow with campaign_status / tail_log."


def resume_run(run_id: str) -> str:
    """Resume a paused run: detached ``ed4all run textbook-to-course
    --resume <id>`` (plain resume, never --force). Refuses when the id is
    invalid, another run is active, or STOP_ALL is present."""
    valid = _validated_run_id(run_id)
    if valid is None:
        return (
            f"Refused: {str(run_id)[:80]!r} is not a valid run id "
            f"(expected WF-YYYYMMDD-xxxxxxxx). Nothing was resumed."
        )
    if _stop_all_present():
        return (
            "Refused: the global STOP_ALL sentinel is present — it blocks "
            "new/resumed runs until the operator clears it "
            "(clear_stop_all after confirming with the human)."
        )
    if _ed4all_run_active():
        return (
            "Refused: an `ed4all run` process is already active (or activity "
            "could not be verified). One build at a time."
        )
    result = _spawn_detached(
        [
            sys.executable, "-m", "cli.main", "run", "textbook-to-course",
            "--resume", valid,
        ],
        f"assistant_resume_{valid}",
    )
    if result.startswith("Failed"):
        return result
    return f"Resume of {valid} {result}. Follow with run_report / tail_log."


def stop_run(run_id: str) -> str:
    """Gracefully stop one run via ``ed4all stop <run_id>`` (validated id only)."""
    run_id = str(run_id).strip()
    if not RUN_ID_RE.match(run_id):
        return (
            f"Refused: {run_id!r} is not a valid run id "
            f"(expected WF-YYYYMMDD-xxxxxxxx). Nothing was stopped."
        )
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, validated run_id
            [sys.executable, "-m", "cli.main", "stop", run_id],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"`ed4all stop {run_id}` failed to execute: {type(exc).__name__}: {exc}"
    output = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    if proc.returncode != 0:
        return f"`ed4all stop {run_id}` exited {proc.returncode}: {output}"
    return f"Stop sentinel dropped for {run_id}: {output or 'ok'}"


def pause_all() -> str:
    """Global pause: ``ed4all stop --all`` (STOP_ALL — pauses AND blocks
    all runs until cleared)."""
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv
            [sys.executable, "-m", "cli.main", "stop", "--all"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"`ed4all stop --all` failed to execute: {type(exc).__name__}: {exc}"
    output = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    if proc.returncode != 0:
        return f"`ed4all stop --all` exited {proc.returncode}: {output}"
    return (
        "Global STOP_ALL dropped: every run pauses at its next unit boundary "
        "and new/resumed runs are blocked until clear_stop_all. "
        + (output or "")
    ).strip()


def clear_stop_all(confirm: bool = False) -> str:
    """Clear the operator-owned global STOP_ALL — ONLY with explicit human
    confirmation (``confirm`` must be exactly true; ask the human first)."""
    if confirm is not True:
        return (
            "Refused: clearing STOP_ALL is operator-owned. Ask the human "
            "operator to confirm, then call clear_stop_all with confirm=true. "
            "Nothing was cleared."
        )
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv
            [sys.executable, "-m", "cli.main", "stop", "--clear-all"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"`ed4all stop --clear-all` failed to execute: {type(exc).__name__}: {exc}"
    output = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    if proc.returncode != 0:
        return f"`ed4all stop --clear-all` exited {proc.returncode}: {output}"
    return f"STOP_ALL cleared: {output or 'ok'}"


def _live_seats(registry: Dict[str, str]) -> List[str]:
    return [name for name, url in registry.items() if _probe_ready(url)]


#: Default assistant chat seat when ``ED4ALL_ASSISTANT_SEAT`` is unset — mirrors
#: ``lib.assistant.client.ASSISTANT_SEAT_NAME`` (a deployment default only).
_DEFAULT_ASSISTANT_SEAT = "spark-nano"


def _assistant_own_seat() -> str:
    """The assistant's OWN chat seat name (``ED4ALL_ASSISTANT_SEAT``, default
    ``spark-nano``). In campaign mode this is the ONLY seat the assistant may
    start: the Super seat (and every pipeline seat) is owned by the pipeline
    seat schedule, never the assistant. Resolves through the client helper so
    the resolution matches everywhere; a resolution failure falls back to the
    documented default rather than raising."""
    try:
        from lib.assistant.client import resolve_assistant_seat  # noqa: PLC0415

        return resolve_assistant_seat()
    except Exception:  # noqa: BLE001 — fall back to the documented default
        return _DEFAULT_ASSISTANT_SEAT


def start_seat(seat: str) -> str:
    """Start one registered vLLM seat (coherent-start path). Refuses when a
    pipeline run is active (the seat schedule owns seats then) or when
    starting it would break the GPU co-residency rule."""
    seat = str(seat or "").strip()
    registry = parse_seat_registry()
    if not registry:
        return f"Refused: no seats registered ({ENV_SEAT_BASE_URLS} unset/empty)."
    if seat not in registry:
        return (
            f"Refused: {seat[:60]!r} is not a registered seat. Registered: "
            + ", ".join(sorted(registry)) + "."
        )
    if _ed4all_run_active():
        return (
            "Refused: a pipeline run is active — the seat schedule owns seat "
            "lifecycle during a run. Never touch seats mid-build."
        )
    live_all = _live_seats(registry)
    if seat in live_all:
        return f"Seat {seat} is already serving — nothing to do."
    live = [s for s in live_all if s != seat]
    # Co-residency law (mirrors the operator seat-boot harness):
    # spark-super is NEVER co-resident with anything; otherwise the summed
    # GPU fractions of live seats + the target must stay <= SEAT_GPU_BUDGET.
    if seat == "spark-super" and live:
        return (
            f"Refused: spark-super must never be co-resident with another "
            f"seat (live: {', '.join(sorted(live))}). Stop them first."
        )
    if "spark-super" in live:
        return (
            "Refused: spark-super is serving and must never be co-resident "
            "with another seat. Stop it first (stop_seat spark-super)."
        )
    budget = SEAT_GPU_FRACTION.get(seat, _UNKNOWN_SEAT_FRACTION) + sum(
        SEAT_GPU_FRACTION.get(s, _UNKNOWN_SEAT_FRACTION) for s in live
    )
    if budget > SEAT_GPU_BUDGET:
        return (
            f"Refused: starting {seat} would put the summed GPU fractions at "
            f"{budget:.2f} > {SEAT_GPU_BUDGET} (live: "
            f"{', '.join(sorted(live)) or 'none'}). Stop a seat first."
        )
    try:
        from lib.vllm_container_lifecycle import start_seat_coherent  # noqa: PLC0415

        result = start_seat_coherent(seat)
    except Exception as exc:  # noqa: BLE001 — surfaced loudly
        return f"start_seat({seat}) failed: {type(exc).__name__}: {exc}"
    if getattr(result, "ok", False):
        return (
            f"Seat {seat} is up + coherent "
            f"(reason={getattr(result, 'reason', '?')}, "
            f"recreated={getattr(result, 'recreated', '?')})."
        )
    return (
        f"Seat {seat} FAILED to come up coherent "
        f"(reason={getattr(result, 'reason', 'unknown')}). Check its container logs."
    )


def stop_seat(seat: str) -> str:
    """Stop one registered vLLM seat. Refuses when a pipeline run is active
    (the seat schedule owns seats then)."""
    seat = str(seat or "").strip()
    registry = parse_seat_registry()
    if not registry:
        return f"Refused: no seats registered ({ENV_SEAT_BASE_URLS} unset/empty)."
    if seat not in registry:
        return (
            f"Refused: {seat[:60]!r} is not a registered seat. Registered: "
            + ", ".join(sorted(registry)) + "."
        )
    if _ed4all_run_active():
        return (
            "Refused: a pipeline run is active — the seat schedule owns seat "
            "lifecycle during a run. Never touch seats mid-build."
        )
    if not _probe_ready(registry[seat]):
        return f"Seat {seat} is already down — nothing to stop."
    try:
        from lib import vllm_container_lifecycle as _lifecycle  # noqa: PLC0415

        ok = _lifecycle.stop_seat(seat)
    except Exception as exc:  # noqa: BLE001 — surfaced loudly
        return f"stop_seat({seat}) failed: {type(exc).__name__}: {exc}"
    return f"Seat {seat} stop {'OK' if ok else 'FAILED'}."


def support_bundle() -> str:
    """Assemble a redacted support bundle via ``ed4all support-bundle``
    (fixed output path under state/support_bundles/); returns path + size."""
    out_dir = STATE_PATH / "support_bundles"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"support_bundle failed: cannot create {out_dir} ({exc})"
    out_path = out_dir / f"ed4all-support-{time.strftime('%Y%m%d_%H%M%S')}.tar.gz"
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, fixed output path
            [
                sys.executable, "-m", "cli.main", "support-bundle",
                "--output", str(out_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"`ed4all support-bundle` failed to execute: {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        output = (proc.stdout or "").strip() or (proc.stderr or "").strip()
        return f"`ed4all support-bundle` exited {proc.returncode}: {_clip(_redact(output), 500)}"
    try:
        size = out_path.stat().st_size
    except OSError:
        size = 0
    return f"Support bundle written: {out_path} ({size} bytes). Share this file with the maintainer."


# --------------------------------------------------------------------------- #
# Curated help (static data — the model never reads files)
# --------------------------------------------------------------------------- #

_HELP_TOPICS: Dict[str, str] = {
    "run": (
        "Start a build: `ed4all run textbook-to-course --corpus <PATH> "
        "--course-name <NAME> [--skip-training]`. `--dry-run` prints the "
        "planned phases without executing. Exit codes: 0 completed, 1 "
        "failed/gate-blocked, 2 usage error, 3 paused at a checkpoint."
    ),
    "resume": (
        "Resume a paused run with a PLAIN `ed4all run --resume <WF-id>` — "
        "never `--force` after a stop (force clears the resume sidecars). A "
        "run paused by `ed4all stop`, SIGTERM/Ctrl-C, or a batch timeout "
        "resumes losslessly; worst-case loss is one in-flight LLM call. The "
        "resume_run tool does this for you (with the active-run + STOP_ALL "
        "guards)."
    ),
    "stop": (
        "`ed4all stop <WF-id>` drops a stop sentinel: the run finishes its "
        "in-flight unit, checkpoints, and pauses (exit code 3) — never "
        "failed, never retried. `ed4all stop --all` is the global STOP_ALL "
        "(also blocks new/resumed runs) until `ed4all stop --clear-all`. "
        "Tools: stop_run / pause_all / clear_stop_all (human-confirmed)."
    ),
    "campaign": (
        "The operator campaign driver (under the campaign dir — "
        "ED4ALL_CAMPAIGN_DIR, default plans/campaign) builds ONE "
        "pending book per invocation from manifest.json (stage-A recipe: "
        "textbook-to-course --skip-training). Source the campaign env first "
        "for a manual launch; it refuses under STOP_ALL, missing seats, or "
        "a foreign GPU workload. Tools: campaign_status, start_next_book, "
        "start_book <slug>, tail_log <slug>."
    ),
    "seats": (
        "vLLM seats are declared in ED4ALL_SEAT_BASE_URLS "
        "(seat_name=base_url), containers in ED4ALL_VLLM_CONTAINERS, launch "
        "specs in ED4ALL_SEAT_LAUNCH_SPECS. ED4ALL_SEAT_SCHEDULE=1 lets the "
        "pipeline reconcile resident seats per phase (liveness poll up to "
        "ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS, bounded coherence probes, "
        "cold-recreate self-heal). Never enable it mid-build. Tools: "
        "seat_status, start_seat/stop_seat (refused during a run; "
        "co-residency budget enforced — spark-super never shares the card)."
    ),
    "reports": (
        "Diagnostics tools: run_report <WF-id> (status + per-phase summary), "
        "gate_report <WF-id> [phase] (validation-gate pass/fail + top issue "
        "codes), tail_log <slug-or-run-id> [lines] (bounded log tail), "
        "doctor [group] (preflight checks: environment/provider/postmortem/"
        "gpu_profile/gpu/window), build_cost <slug> (per-phase wall clock + "
        "tokens), aggregator_report <slug> <name> (promotion_chain, "
        "coverage_map, kg_quality, …), flag_lookup <name> (behavior-flag "
        "semantics), support_bundle (redacted diagnostics tarball)."
    ),
    "library": (
        "Course-library tools: library_courses (LibV2 slugs + manifest/"
        "chunks/index presence), course_objectives <slug> (TO/CO counts + "
        "TO titles), ask_course <slug> <question> (grounded, citation-gated "
        "Q&A over the course's vector index — refused while a run is active "
        "or when the course has no index)."
    ),
    "seat-setup": (
        "Setting up the seat-swap env (pipeline-owned seat lifecycle): "
        "(1) list_gpu_containers to discover vLLM containers; (2) decide "
        "seat names + ports and map each container; (3) generate_seat_env "
        "with the seat list — it RETURNS a ready-to-source export block "
        "(nothing is written; save it yourself, e.g. seat-schedule.env); "
        "(4) `source` it alongside your run env BEFORE `ed4all run` (never "
        "mid-build); (5) verify with seat_env_doctor, then seat_status. "
        "Canonical template: docs/operations/seat-schedule.env.example."
    ),
    "debug": (
        "Debug mode: `ed4all assistant --debug [--run WF-...]` seeds the "
        "session with a failed run's diagnostic context (run report, "
        "failing-phase gates, log tail, doctor post-mortem) and shifts the "
        "assistant to root-cause diagnosis. With "
        "ED4ALL_ASSISTANT_DEBUG_ON_FAILURE=1 the campaign driver records "
        "each failed book to <campaign dir>/last_failure.json and "
        "prints the exact --debug command to run."
    ),
}


def get_help(topic: str = "") -> str:
    """Curated static help. Unknown/empty topic lists the available topics."""
    key = str(topic or "").strip().lower()
    if key in _HELP_TOPICS:
        return _HELP_TOPICS[key]
    return (
        "Available help topics: " + ", ".join(sorted(_HELP_TOPICS)) + ". "
        "Ask get_help with one of these."
    )


# --------------------------------------------------------------------------- #
# Registry + dispatch
# --------------------------------------------------------------------------- #

TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    # status
    "campaign_status": campaign_status,
    "run_status": run_status,
    "seat_status": seat_status,
    # reports
    "list_workflows": list_workflows,
    "run_report": run_report,
    "gate_report": gate_report,
    "tail_log": tail_log,
    "doctor": doctor,
    "build_cost": build_cost,
    "aggregator_report": aggregator_report,
    "flag_lookup": flag_lookup,
    # library
    "library_courses": library_courses,
    "course_objectives": course_objectives,
    "ask_course": ask_course,
    # seat-swap env setup
    "seat_env_doctor": seat_env_doctor,
    "list_gpu_containers": list_gpu_containers,
    "generate_seat_env": generate_seat_env,
    # actions
    "start_next_book": start_next_book,
    "start_book": start_book,
    "resume_run": resume_run,
    "stop_run": stop_run,
    "pause_all": pause_all,
    "clear_stop_all": clear_stop_all,
    "start_seat": start_seat,
    "stop_seat": stop_seat,
    "support_bundle": support_bundle,
    # help
    "get_help": get_help,
}

#: Argument whitelist per tool — anything not listed is dropped before the
#: call, so the model cannot smuggle kwargs into a tool function.
_TOOL_ARG_WHITELIST: Dict[str, tuple] = {
    "campaign_status": (),
    "run_status": (),
    "seat_status": (),
    "list_workflows": (),
    "run_report": ("run_id",),
    "gate_report": ("run_id", "phase"),
    "tail_log": ("target", "lines"),
    "doctor": ("group",),
    "build_cost": ("course_slug",),
    "aggregator_report": ("course_slug", "name"),
    "flag_lookup": ("name_or_prefix",),
    "library_courses": (),
    "course_objectives": ("course_slug",),
    "ask_course": ("course_slug", "question"),
    "seat_env_doctor": (),
    "list_gpu_containers": (),
    "generate_seat_env": ("seats", "load_timeout", "coherence_attempts"),
    "start_next_book": (),
    "start_book": ("slug",),
    "resume_run": ("run_id",),
    "stop_run": ("run_id",),
    "pause_all": (),
    "clear_stop_all": ("confirm",),
    "start_seat": ("seat",),
    "stop_seat": ("seat",),
    "support_bundle": (),
    "get_help": ("topic",),
}

#: Required arguments per tool — missing one is a refusal BEFORE the call.
_TOOL_REQUIRED_ARGS: Dict[str, tuple] = {
    "run_report": ("run_id",),
    "gate_report": ("run_id",),
    "tail_log": ("target",),
    "build_cost": ("course_slug",),
    "aggregator_report": ("course_slug", "name"),
    "flag_lookup": ("name_or_prefix",),
    "course_objectives": ("course_slug",),
    "ask_course": ("course_slug", "question"),
    "generate_seat_env": ("seats",),
    "start_book": ("slug",),
    "resume_run": ("run_id",),
    "stop_run": ("run_id",),
    "start_seat": ("seat",),
    "stop_seat": ("seat",),
}


def _schema(name: str, description: str, params: Optional[Dict[str, Any]] = None,
            required: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": params or {},
                "required": required or [],
            },
        },
    }


_RUN_ID_PARAM = {
    "type": "string",
    "description": "The workflow run id, e.g. WF-20260420-abc12345.",
}
_SLUG_PARAM = {
    "type": "string",
    "description": "A course slug from the campaign manifest or LibV2/courses.",
}

#: OpenAI function-calling schemas served to the model.
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    # status
    _schema("campaign_status",
            "Summarize the book-build campaign manifest: counts by status and the currently running book."),
    _schema("run_status",
            "List active and recent pipeline runs (run id, status, workflow, created)."),
    _schema("seat_status",
            "Probe the registered vLLM seats and report which are serving and what model each serves."),
    # reports
    _schema("list_workflows",
            "List the configured workflows with their ordered phase names."),
    _schema("run_report",
            "Summarize one run: status, failed/paused phase, failure reason, per-phase completion + gate counts.",
            {"run_id": _RUN_ID_PARAM}, ["run_id"]),
    _schema("gate_report",
            "Validation-gate results for a run (optionally one phase): pass/fail, severity, top issue codes.",
            {"run_id": _RUN_ID_PARAM,
             "phase": {"type": "string", "description": "Optional phase name to filter to."}},
            ["run_id"]),
    _schema("tail_log",
            "Tail a build log (max 200 lines). target = a campaign book slug or a run id.",
            {"target": {"type": "string", "description": "Campaign slug or run id (WF-... / TTC_...)."},
             "lines": {"type": "integer", "description": "Lines to show, 1-200 (default 50)."}},
            ["target"]),
    _schema("doctor",
            "Run ed4all doctor diagnostics and return a structured finding summary: the "
            "verdict (OK/DEGRADED/DANGER), per-severity counts, and every WARN/FAIL check "
            "with its remediation (OK/INFO checks are counted, not detailed). Optional group: "
            "environment, provider, postmortem, gpu_profile, gpu, window.",
            {"group": {"type": "string", "description": "Optional check group name."}}),
    _schema("build_cost",
            "Summarize a course's build-cost report: per-phase wall clock and LLM tokens.",
            {"course_slug": _SLUG_PARAM}, ["course_slug"]),
    _schema("aggregator_report",
            "Summarize one aggregator report for a course. name: promotion_chain, coverage_map, "
            "accessibility_conformance, edge_consensus, block_quality_rollup, concept_coverage, "
            "intelligence_level, assessment_quality, provenance_resolution, kg_quality, build_cost, validation_report.",
            {"course_slug": _SLUG_PARAM,
             "name": {"type": "string", "description": "Report name from the fixed enum."}},
            ["course_slug", "name"]),
    _schema("flag_lookup",
            "Look up an ED4ALL_* behavior flag's default + semantics by name or prefix (up to 5 matches).",
            {"name_or_prefix": {"type": "string", "description": "Flag name or prefix, e.g. ED4ALL_NLI."}},
            ["name_or_prefix"]),
    # library
    _schema("library_courses",
            "List the LibV2 course library: slug + manifest/chunkset/vector-index presence."),
    _schema("course_objectives",
            "TO/CO counts and terminal-objective titles for a course.",
            {"course_slug": _SLUG_PARAM}, ["course_slug"]),
    _schema("ask_course",
            "Ask a grounded, citation-gated question about a course's content (needs its vector index; "
            "refused while a pipeline run is active).",
            {"course_slug": _SLUG_PARAM,
             "question": {"type": "string", "description": "The learner/operator question."}},
            ["course_slug", "question"]),
    # seat-swap env setup
    _schema("seat_env_doctor",
            "Audit the seat-swap env stack (ED4ALL_SEAT_SCHEDULE / _SEAT_BASE_URLS / _VLLM_CONTAINERS / "
            "_SEAT_LAUNCH_SPECS / timeouts): set/unset, seat->container resolution, script paths, loopback, "
            "duplicate ports."),
    _schema("list_gpu_containers",
            "List docker containers (name/status/image, read-only, max 30) to help map containers to seats."),
    _schema("generate_seat_env",
            "Generate a ready-to-source seat-schedule env snippet (returned as text — the USER saves and "
            "sources it; no file is written). Max 8 seats; loopback URLs only.",
            {"seats": {
                "type": "array",
                "description": "Seat entries.",
                "items": {
                    "type": "object",
                    "properties": {
                        "seat_name": {"type": "string", "description": "Lowercase seat name, e.g. from seat_status."},
                        "base_url": {"type": "string", "description": "Loopback vLLM base URL, e.g. http://localhost:8001."},
                        "container": {"type": "string", "description": "Docker container name (see list_gpu_containers)."},
                        "launch_script": {"type": "string", "description": "Optional existing absolute launch-script path."},
                    },
                    "required": ["seat_name", "base_url", "container"],
                },
            },
             "load_timeout": {"type": "integer", "description": "Optional liveness ceiling seconds (default 1200)."},
             "coherence_attempts": {"type": "integer", "description": "Optional coherence attempts (default 3)."}},
            ["seats"]),
    # actions
    _schema("start_next_book",
            "Start the next pending campaign book build (refuses if a run is already active or no campaign is configured)."),
    _schema("start_book",
            "Start ONE specific pending campaign book by slug (refuses if not pending or a run is active).",
            {"slug": {"type": "string", "description": "Pending campaign book slug."}}, ["slug"]),
    _schema("resume_run",
            "Resume a paused run (plain --resume, detached). Refuses when another run is active or STOP_ALL is set.",
            {"run_id": _RUN_ID_PARAM}, ["run_id"]),
    _schema("stop_run",
            "Gracefully stop one pipeline run at its next checkpoint. run_id must look like WF-YYYYMMDD-xxxxxxxx.",
            {"run_id": _RUN_ID_PARAM}, ["run_id"]),
    _schema("pause_all",
            "Global pause (ed4all stop --all): every run checkpoints + pauses, and new runs are blocked until cleared."),
    _schema("clear_stop_all",
            "Clear the global STOP_ALL. Operator-owned: ASK THE HUMAN FIRST, then call with confirm=true. "
            "Refused unless confirm is exactly true.",
            {"confirm": {"type": "boolean",
                         "description": "Must be true, and only after the human operator explicitly confirmed."}}),
    _schema("start_seat",
            "Start a registered vLLM seat (coherent start). Refused during a run; GPU co-residency budget enforced "
            "(spark-super never shares the card).",
            {"seat": {"type": "string", "description": "Registered seat name (list them with seat_status)."}}, ["seat"]),
    _schema("stop_seat",
            "Stop a registered vLLM seat. Refused while a pipeline run is active.",
            {"seat": {"type": "string", "description": "Registered seat name."}}, ["seat"]),
    _schema("support_bundle",
            "Assemble a redacted diagnostics tarball (ed4all support-bundle) and report its path + size."),
    # help
    _schema("get_help",
            "Curated Ed4All operator help. Topics: run, resume, stop, campaign, seats, seat-setup, reports, library, debug.",
            {"topic": {"type": "string", "description": "Help topic name."}}),
]


#: Base tools that do NOT mutate run / seat / campaign state — the OBSERVE +
#: REPORT subset. This is the set the pilot's restricted ``campaign-tick``
#: engine mode exposes (so the tick LLM turn cannot bypass the pilot's
#: deterministic policy by starting/resuming/stopping runs or seats). Every
#: name here is a read-only status / report / library / seat-env-audit tool.
READONLY_TOOL_NAMES = frozenset(
    {
        # status
        "campaign_status", "run_status", "seat_status",
        # reports
        "list_workflows", "run_report", "gate_report", "tail_log", "doctor",
        "build_cost", "aggregator_report", "flag_lookup",
        # library
        "library_courses", "course_objectives", "ask_course",
        # seat-swap env setup (all read-only / text-returning; no seat is started)
        "seat_env_doctor", "list_gpu_containers", "generate_seat_env",
        # help
        "get_help",
    }
)

#: The read-only base tool schemas served in the restricted ``campaign-tick``
#: mode — the ``READONLY_TOOL_NAMES`` slice of :data:`TOOL_SCHEMAS`.
READONLY_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    s for s in TOOL_SCHEMAS if s["function"]["name"] in READONLY_TOOL_NAMES
]


def dispatch_tool(
    name: str,
    arguments: Dict[str, Any],
    *,
    campaign_mode: bool = False,
    readonly: bool = False,
) -> str:
    """Dispatch one model-requested tool call through the whitelist.

    Unknown tool names get a refusal string (routed back to the model, never
    executed). Arguments are filtered to the per-tool whitelist and checked
    against the per-tool required set. A tool exception is surfaced as an
    error string in the conversation — visible to the operator, never a
    silent success.

    ``campaign_mode`` gates the merged campaign tool surface: in campaign mode
    the assistant may start ONLY its own chat seat (``ED4ALL_ASSISTANT_SEAT``,
    default ``spark-nano``) — never ``spark-super`` or any other pipeline seat
    (those are owned by the pipeline seat schedule). ``readonly`` (the pilot's
    ``campaign-tick`` surface) refuses every mutating tool so the tick LLM turn
    can only OBSERVE + REPORT. Both default False → the interactive operator /
    campaign path is byte-identical to before.
    """
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return (
            f"Refused: tool {name!r} is not in the assistant whitelist. "
            f"Available tools: {', '.join(sorted(TOOL_REGISTRY))}."
        )
    if readonly and name not in READONLY_TOOL_NAMES:
        return (
            f"Refused: {name} is a mutating tool and is not available in the "
            f"read-only campaign-tick surface — the pilot's deterministic "
            f"policy owns run/seat control. Use a report tool to escalate."
        )
    allowed = _TOOL_ARG_WHITELIST.get(name, ())
    kwargs = {
        k: v for k, v in (arguments or {}).items() if k in allowed
    }
    # Campaign-mode seat guard: the assistant may start ONLY its own chat seat.
    # The Super seat (and every pipeline seat) belongs to the pipeline seat
    # schedule — the assistant must NEVER start/autostart it.
    if campaign_mode and name == "start_seat":
        requested = str(kwargs.get("seat") or "").strip()
        own_seat = _assistant_own_seat()
        if requested != own_seat:
            return (
                f"Refused: in campaign mode the assistant may start ONLY its "
                f"own chat seat ({own_seat!r}, from ED4ALL_ASSISTANT_SEAT); "
                f"{requested[:60]!r} is not allowed. Only the pipeline seat "
                f"schedule may seat other seats (e.g. spark-super). Nothing "
                f"was started."
            )
    for required in _TOOL_REQUIRED_ARGS.get(name, ()):
        if required not in kwargs:
            return f"Refused: {name} requires a {required} argument."
    try:
        return fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 — surfaced loudly in-conversation
        logger.exception("assistant tool %s raised", name)
        return f"Tool {name} failed: {type(exc).__name__}: {exc}"


__all__ = [
    "AGGREGATOR_REPORTS",
    "CAMPAIGN_DIR",
    "CONTAINER_NAME_RE",
    "DOCTOR_GROUPS",
    "MAX_GENERATED_SEATS",
    "SEAT_NAME_RE",
    "MAX_LOG_TAIL_LINES",
    "MAX_TOOL_RESULT_CHARS",
    "READONLY_TOOL_NAMES",
    "READONLY_TOOL_SCHEMAS",
    "RUN_ID_RE",
    "SEAT_GPU_BUDGET",
    "SEAT_GPU_FRACTION",
    "SLUG_RE",
    "TOOL_REGISTRY",
    "TOOL_SCHEMAS",
    "TTC_RUN_ID_RE",
    "ask_course",
    "aggregator_report",
    "build_cost",
    "campaign_status",
    "clear_stop_all",
    "course_objectives",
    "dispatch_tool",
    "doctor",
    "flag_lookup",
    "gate_report",
    "generate_seat_env",
    "get_help",
    "library_courses",
    "list_gpu_containers",
    "list_workflows",
    "pause_all",
    "resume_run",
    "run_report",
    "run_status",
    "seat_env_doctor",
    "seat_status",
    "start_book",
    "start_next_book",
    "start_seat",
    "stop_run",
    "stop_seat",
    "support_bundle",
    "tail_log",
]

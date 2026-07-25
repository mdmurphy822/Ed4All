"""Environment precondition checks — the ``environment`` doctor group.

These checks probe the environment-level preconditions whose SILENT
absence is the documented root cause of several Ed4All failure modes:

* an ollama server that is down (or whose configured model was never
  pulled) — the local synthesis / grounded-answer path then fails with an
  obscure connection error mid-run;
* a missing optional extra — ``[embedding]`` absent silently degrades the
  statistical-tier validators to lexical scoring (``EMBEDDING_DEPS_MISSING``;
  load-bearing for grounded quality → WARN), ``[gui]`` absent makes
  ``ed4all gui`` unavailable (only the optional GUI, not a pipeline run →
  INFO), ``[semantik]`` absent removes PDF→HTML conversion (only the
  ``semantik_conversion`` phase — a from-existing-HTML run does not need it →
  INFO);
* a missing ``torch`` — NLI groundedness scoring + GPU VRAM probing are
  unavailable, so grounding validators degrade (load-bearing → WARN);
* the ``mode=local`` dispatch trap — a local-mode run WITHOUT
  ``ED4ALL_AGENT_DISPATCH=true`` falls through to in-process stubs
  (sub-minute wall-clock, all-null KG fields).

Design contract (inherited from :mod:`lib.diagnostics.core`): a doctor
that crashes is worse than no doctor. Every sub-check here is wrapped so a
single failing probe becomes its own WARN result and the rest still run;
:func:`environment_checks` NEVER raises. Severity is graded by how
load-bearing the missing precondition is for a real run: genuine run
blockers (ollama down / model not pulled / the local-mode dispatch trap)
and silently-degrading-quality gaps (``[embedding]`` / ``torch``) emit
WARN (→ exit 1 / DEGRADED); advisory, single-phase extras (``[gui]`` /
``[semantik]``) emit INFO (shown with ``ℹ`` but NEVER escalate the exit
code or verdict). We use the cheap
``importlib.util.find_spec`` spec-probe for every optional dependency
(precedent: ``MCP/core/workflow_runner.py`` ~L3398) so NO heavy ML stack
(torch / sentence-transformers / the NLI model) is ever imported and the
GPU is never touched. ``httpx`` is imported lazily inside the ollama probe
with a short 2s timeout. Registration is explicit
(:func:`register_environment_checks`), never at import time.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Callable, List, Optional

from lib.diagnostics.core import CheckContext, CheckResult, Severity, register

logger = logging.getLogger(__name__)

#: Short per-request timeout (seconds) for the ollama ``/api/tags`` probe.
#: This is a liveness ping, not a generation call — we must not stall the
#: doctor on a wedged server.
_OLLAMA_PROBE_TIMEOUT_SECONDS = 2.0

#: Effective-mode env (CLI ``--mode`` knob; default ``local``).
_LLM_MODE_ENV = "LLM_MODE"
_DEFAULT_MODE = "local"


def _resolve_local_model() -> str:
    """Resolve the configured local model name via the SHARED resolver.

    Finding #8 — there is ONE local-model-name source of truth,
    :func:`lib.diagnostics.serving_window.resolve_local_model` (it reads
    ``LOCAL_SYNTHESIS_MODEL`` and falls back to the documented canonical
    default). We import the sibling module and call the attribute at call
    time so a monkeypatch of ``serving_window.resolve_local_model`` is
    honored. Best-effort: if the resolver is unavailable for ANY reason we
    degrade to a neutral placeholder rather than raise (the ollama
    model-pulled sub-check must never crash the doctor).
    """
    try:
        from lib.diagnostics import serving_window

        model = (serving_window.resolve_local_model() or "").strip()
        if model:
            return model
    except Exception as exc:  # noqa: BLE001 — resolver import/read must not crash
        logger.debug(
            "diagnostics.environment: resolve_local_model unavailable: %s", exc
        )
    return "unknown"


def _run_subcheck(
    fn: Callable[[CheckContext], List[CheckResult]], ctx: CheckContext
) -> List[CheckResult]:
    """Run one sub-check, isolating a raise into its own WARN result.

    Belt-and-suspenders around the per-sub-check ``try`` blocks: even if a
    sub-check forgets to wrap something, the failure degrades to a single
    WARN (named ``environment_<fn>``) and the remaining sub-checks still run.
    """
    try:
        return fn(ctx) or []
    except Exception as exc:  # noqa: BLE001 — a doctor sub-check must never crash
        name = getattr(fn, "__name__", "subcheck").lstrip("_")
        logger.warning("diagnostics.environment: %s raised: %s", name, exc)
        return [
            CheckResult(
                name=f"environment_{name}",
                group="environment",
                severity=Severity.WARN,
                summary=f"environment check '{name}' errored: {exc}",
                detail=f"{type(exc).__name__}: {exc}",
                remediation="this is a doctor bug — the sub-check should never raise",
                data={"error": str(exc), "error_type": type(exc).__name__},
            )
        ]


# --------------------------------------------------------------------- #
# 1. ollama reachable + configured model pulled
# --------------------------------------------------------------------- #


def _fetch_ollama_tags(root: str, timeout: float) -> List[str]:
    """Return the model names ollama reports via ``GET {root}/api/tags``.

    Lazy ``httpx`` import + short timeout. Parses the documented shape
    ``{"models": [{"name": "qwen2.5:7b-...", ...}, ...]}`` into a list of
    name strings. Lets transport / JSON errors PROPAGATE so the caller can
    distinguish "unreachable" from "reachable but empty" — the caller wraps
    this best-effort.
    """
    import httpx  # type: ignore  # lazy — only when the ollama probe runs

    with httpx.Client(timeout=timeout) as client:
        resp = client.get(f"{root}/api/tags")
        resp.raise_for_status()
        body = resp.json()
    models = body.get("models") if isinstance(body, dict) else None
    if not isinstance(models, list):
        return []
    names: List[str] = []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("model")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _model_present(model: str, names: List[str]) -> bool:
    """True iff ``model`` matches a pulled tag (tolerant of the ``:tag`` part).

    Exact match first; otherwise compare the base (segment before the first
    ``:``) so a configured ``qwen2.5:7b`` still resolves against a list that
    carries a fuller tag like ``qwen2.5:7b-instruct-q4_K_M`` and vice versa.
    """
    if model in names:
        return True
    base = model.split(":", 1)[0]
    return any(name == model or name.split(":", 1)[0] == base for name in names)


def _check_local_synthesis_vllm(ctx: CheckContext, topo) -> List[CheckResult]:
    """vLLM-seat local-synthesis check (P0-1): probe ``/v1/models``, no ollama warn.

    When LOCAL_SYNTHESIS_BASE_URL resolves to a vLLM seat (a seat registry is
    configured), ollama is NOT the serving path — so the ollama ``/api/tags``
    model-pull WARN would be a FALSE DEGRADED. Instead we probe the seat's
    OpenAI-compatible ``/v1/models`` and report INFO (live/down is the seat
    group's concern — down at rest is not an error here). NEVER raises.
    """
    from lib.diagnostics.run_env import probe_v1_models

    root = topo.base_url_root
    seat_label = topo.seat_name or "unregistered seat"
    live, model_ids, error = probe_v1_models(root)
    model = _resolve_local_model()

    served_note = (
        f"; serving {len(model_ids)} model(s): {model_ids}" if live else ""
    )
    results: List[CheckResult] = [
        CheckResult(
            name="local_synthesis_backend",
            group="environment",
            severity=Severity.INFO,
            summary=(
                f"local-synthesis backend is a vLLM seat ({seat_label}) at {root} "
                + (f"— live{served_note}" if live else f"— not answering /v1/models ({error})")
            ),
            detail=(
                "LOCAL_SYNTHESIS_BASE_URL resolves to a vLLM seat (a seat registry "
                "is configured), so the ollama /api/tags model-pull check does not "
                "apply; seat liveness is owned by the 'seat' group (down at rest is "
                "not an error, e.g. between GPU-lifecycle phases)"
            ),
            data={
                "backend": "vllm",
                "base_url": root,
                "seat_name": topo.seat_name,
                "live": live,
                "served_model_ids": model_ids,
                "configured_model": model,
                "error": error,
            },
        )
    ]

    # Configured LOCAL_SYNTHESIS_MODEL vs the seat's served ids — INFO only
    # (vLLM --served-model-name conventions vary; never a false DEGRADED).
    if live and model and model != "unknown":
        present = any(
            model == mid or model.split(":", 1)[0] == str(mid).split(":", 1)[0]
            for mid in model_ids
        )
        results.append(
            CheckResult(
                name="local_synthesis_model",
                group="environment",
                severity=Severity.INFO,
                summary=(
                    f"configured LOCAL_SYNTHESIS_MODEL '{model}' "
                    + ("matches" if present else "is not among")
                    + f" the seat's served id(s) {model_ids}"
                ),
                data={"model": model, "served_model_ids": model_ids, "matched": present},
            )
        )
    return results


def _check_ollama(ctx: CheckContext) -> List[CheckResult]:
    """ollama liveness + configured-model-pulled check (NEVER raises).

    P0-1 topology awareness: when a seat registry is configured AND
    LOCAL_SYNTHESIS_BASE_URL resolves to a vLLM seat, the ollama probe is
    replaced by an OpenAI-compatible ``/v1/models`` check (no false ollama-pull
    WARN). With no seat registry the legacy ollama path runs byte-identical.
    """
    try:
        from lib.diagnostics.run_env import resolve_local_synthesis_topology

        topo = resolve_local_synthesis_topology()
        if topo.backend == "vllm":
            return _check_local_synthesis_vllm(ctx, topo)
    except Exception as exc:  # noqa: BLE001 — topology resolution must not crash the doctor
        logger.debug("diagnostics.environment: topology resolution failed: %s", exc)

    # resolve_ollama_root never raises on well-formed input, but keep the
    # import + call inside the guarded sub-check so an import failure on a
    # broken tree degrades to a WARN rather than crashing the doctor.
    from lib.llm.vram_reclaim import resolve_ollama_root

    root = resolve_ollama_root(ctx.base_url)
    model = _resolve_local_model()

    try:
        names = _fetch_ollama_tags(root, _OLLAMA_PROBE_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 — server down / httpx absent / bad JSON
        logger.debug("diagnostics.environment: ollama probe failed: %s", exc)
        return [
            CheckResult(
                name="ollama_reachable",
                group="environment",
                severity=Severity.WARN,
                summary=f"ollama not reachable at {root}",
                detail=f"{type(exc).__name__}: {exc}",
                remediation=(
                    "start ollama (e.g. `ollama serve`) / check "
                    "LOCAL_SYNTHESIS_BASE_URL"
                ),
                data={"root": root, "error": str(exc)},
            )
        ]

    results: List[CheckResult] = [
        CheckResult(
            name="ollama_reachable",
            group="environment",
            severity=Severity.OK,
            summary=f"ollama reachable at {root} ({len(names)} model(s) pulled)",
            data={"root": root, "tags": list(names)},
        )
    ]

    if _model_present(model, names):
        results.append(
            CheckResult(
                name="ollama_model_pulled",
                group="environment",
                severity=Severity.OK,
                summary=f"configured local model '{model}' is pulled",
                data={"model": model, "tags": list(names)},
            )
        )
    else:
        results.append(
            CheckResult(
                name="ollama_model_pulled",
                group="environment",
                severity=Severity.WARN,
                summary=f"model '{model}' not pulled",
                detail=f"{len(names)} model(s) present: {names}",
                remediation=f"run `ollama pull {model}`",
                data={"model": model, "tags": list(names)},
            )
        )
    return results


# --------------------------------------------------------------------- #
# 2. Optional extras present (find_spec only — no heavy imports)
# --------------------------------------------------------------------- #

#: One row per optional-extra probe, an 8-field tuple::
#:
#:     (result_name, importable_module, extra_label, absent_consequence,
#:      remediation, note, absent_severity, require_all)
#:
#: where:
#:   * ``result_name`` — the stable ``CheckResult.name`` (e.g. ``extra_gui``);
#:   * ``importable_module`` — a module-name ``str`` OR a tuple of candidate
#:     module names to spec-probe;
#:   * ``extra_label`` — the pip extra label (e.g. ``[gui]``) for summaries;
#:   * ``absent_consequence`` — the one-line "what breaks when absent" text
#:     interpolated into the ABSENT summary;
#:   * ``remediation`` — the ``pip install`` line shown on the absent result;
#:   * ``note`` — optional longer ``detail`` string (probe heuristics);
#:   * ``absent_severity`` — Finding #5: the ``Severity`` emitted when the
#:     extra is ABSENT, graded by how load-bearing it is for a real run
#:     (``[embedding]`` → WARN; advisory single-phase ``[gui]`` / ``[semantik]``
#:     → INFO). A PRESENT extra is always OK.
#:   * ``require_all`` — Finding #3: when ``importable_module`` is a tuple,
#:     whether ALL candidates must be importable (``True``) or ANY (``False``).
#:     ``[gui]`` needs BOTH uvicorn AND fastapi (a half-installed ``[gui]``
#:     still ``ImportError``s ``ed4all gui``), so ``require_all=True``;
#:     single-module probes leave it ``False``.
#:
#: Detection is a cheap ``find_spec`` spec-probe — NEVER an import of the
#: heavy dependency.
_EXTRA_PROBES = [
    (
        "extra_embedding",
        "sentence_transformers",
        "[embedding]",
        "embedding extra missing -> statistical-tier validators silently "
        "degrade to lexical (EMBEDDING_DEPS_MISSING)",
        "pip install -e .[embedding]",
        "",
        Severity.WARN,
        False,
    ),
    (
        "extra_gui",
        ("uvicorn", "fastapi"),
        "[gui]",
        "`ed4all gui` unavailable (only the optional GUI — not a pipeline run)",
        "pip install -e .[gui]",
        "[gui] needs BOTH uvicorn AND fastapi; a half-installed extra still "
        "ImportErrors `ed4all gui`.",
        Severity.INFO,
        True,
    ),
    (
        "extra_semantik",
        "pypdfium2",
        "[semantik]",
        "PDF->HTML conversion (semantik_conversion phase) unavailable — a "
        "from-existing-HTML run does not need it",
        "pip install -e .[semantik]",
        "SemantiK runs as a subprocess bridge, so this find_spec probe of "
        "one of its deps (pypdfium2) is a heuristic for the extra's presence.",
        Severity.INFO,
        False,
    ),
]


def _one_spec_present(name: str) -> bool:
    """True iff ``name`` is importable per ``find_spec`` (broken parent → False).

    Uses ``importlib.util.find_spec`` only — the cheap no-heavy-import probe.
    A candidate whose parent package is itself absent raises inside
    ``find_spec``; that is treated as "absent" (False) rather than propagated.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001 — a broken parent package == absent
        return False


def _spec_present(module: object, require_all: bool = False) -> bool:
    """True iff ``module`` (a str or tuple of candidates) is importable.

    ``require_all=False`` (default) → ANY candidate importable; with
    ``require_all=True`` → ALL candidates must be importable (the ``[gui]``
    all-of semantics). Single-module probes are unaffected by the flag.
    """
    candidates = module if isinstance(module, tuple) else (module,)
    flags = [_one_spec_present(name) for name in candidates]
    return all(flags) if require_all else any(flags)


def _check_extras(ctx: CheckContext) -> List[CheckResult]:
    """One result per optional extra (OK present / WARN|INFO absent).

    A present extra is OK. An ABSENT extra emits its configured
    ``absent_severity`` (Finding #5: load-bearing ``[embedding]`` → WARN;
    advisory ``[gui]`` / ``[semantik]`` → INFO). For a multi-module extra
    the absent result NAMES exactly which candidate(s) are missing. NEVER
    raises.
    """
    results: List[CheckResult] = []
    for (
        name,
        module,
        label,
        consequence,
        remediation,
        note,
        absent_severity,
        require_all,
    ) in _EXTRA_PROBES:
        probed = module if isinstance(module, tuple) else (module,)
        missing = [m for m in probed if not _one_spec_present(m)]
        present = not missing if require_all else len(missing) < len(probed)
        if present:
            results.append(
                CheckResult(
                    name=name,
                    group="environment",
                    severity=Severity.OK,
                    summary=f"{label} extra present",
                    detail=note,
                    data={"module": list(probed), "missing": [], "present": True},
                )
            )
        else:
            detail = note
            if len(probed) > 1:
                miss_str = ", ".join(missing)
                detail = (f"{note} " if note else "") + f"missing: {miss_str}"
            results.append(
                CheckResult(
                    name=name,
                    group="environment",
                    severity=absent_severity,
                    summary=f"{label} extra missing — {consequence}",
                    detail=detail,
                    remediation=remediation,
                    data={
                        "module": list(probed),
                        "missing": missing,
                        "present": False,
                    },
                )
            )
    return results


# --------------------------------------------------------------------- #
# 3. torch / NLI device (no model load)
# --------------------------------------------------------------------- #


def _check_torch_nli(ctx: CheckContext) -> List[CheckResult]:
    """torch presence + resolved NLI device (find_spec only). NEVER raises."""
    if not _spec_present("torch"):
        return [
            CheckResult(
                name="torch_nli",
                group="environment",
                severity=Severity.WARN,
                summary=(
                    "torch missing -> NLI groundedness scoring + GPU VRAM "
                    "probing unavailable; grounding validators degrade"
                ),
                remediation="pip install -e .[embedding] (or install torch)",
                data={"torch_present": False},
            )
        ]

    # torch is importable; report the RESOLVED (config-only) NLI device.
    # resolve_nli_device only reads env — it NEVER imports torch / touches
    # the GPU; actual CUDA availability is the gpu group's concern.
    device = "unknown"
    try:
        from lib.classifiers.nli_classifier import resolve_nli_device

        device = resolve_nli_device()
    except Exception as exc:  # noqa: BLE001 — resolver import/read must not crash
        logger.debug("diagnostics.environment: resolve_nli_device failed: %s", exc)

    detail = (
        f"resolved NLI device = {device!r}"
        + (
            " (cuda requested — actual CUDA availability is checked by the "
            "gpu group)"
            if device.startswith("cuda")
            else ""
        )
    )
    return [
        CheckResult(
            name="torch_nli",
            group="environment",
            severity=Severity.OK,
            summary=f"torch present; NLI device resolves to {device!r}",
            detail=detail,
            data={"torch_present": True, "nli_device": device},
        )
    ]


# --------------------------------------------------------------------- #
# 4. local-mode dispatch trap
# --------------------------------------------------------------------- #


def _check_dispatch_trap(ctx: CheckContext) -> List[CheckResult]:
    """local-mode-without-dispatch trap (reuses ``_agent_dispatch_enabled``)."""
    mode = (os.environ.get(_LLM_MODE_ENV) or "").strip().lower() or _DEFAULT_MODE

    if mode != "local":
        return [
            CheckResult(
                name="local_dispatch_trap",
                group="environment",
                severity=Severity.OK,
                summary=f"mode={mode} — local-mode dispatch trap does not apply",
                data={"mode": mode, "agent_dispatch": None},
            )
        ]

    # Reuse the canonical ED4ALL_AGENT_DISPATCH reader rather than re-parsing.
    from MCP.core.executor import _agent_dispatch_enabled

    dispatch_on = _agent_dispatch_enabled()
    if dispatch_on:
        return [
            CheckResult(
                name="local_dispatch_trap",
                group="environment",
                severity=Severity.OK,
                summary="mode=local with ED4ALL_AGENT_DISPATCH=true",
                data={"mode": mode, "agent_dispatch": True},
            )
        ]
    return [
        CheckResult(
            name="local_dispatch_trap",
            group="environment",
            severity=Severity.WARN,
            summary=(
                "mode=local without ED4ALL_AGENT_DISPATCH=true -> phases fall "
                "through to in-process stubs (sub-minute wall-clock, all-null "
                "KG fields)"
            ),
            remediation=(
                "export ED4ALL_AGENT_DISPATCH=true for real local-mode runs"
            ),
            data={"mode": mode, "agent_dispatch": False},
        )
    ]


# --------------------------------------------------------------------- #
# 5. Stale non-terminal workflow records (P1-4)
# --------------------------------------------------------------------- #

#: Terminal + resting statuses (normalized lowercase; ``complete`` folds to
#: ``completed``). Mirrors ``gui/services/liveness.py::_FINAL_STATUSES`` plus the
#: resting ``paused`` states (a paused run legitimately has no live process and
#: carries a resume sidecar — it is NOT a stale RUNNING record).
_TERMINAL_OR_RESTING_STATUSES = frozenset(
    {
        "completed",
        "failed",
        "cancelled",
        "canceled",
        "interrupted",
        "timeout",
        "error",
        "paused",
        "pausing",
    }
)

#: How many stale run ids to name inline before summarizing "+N more".
_STALE_RUN_LIST_CAP = 20


def _resolve_workflows_dir():
    """Resolve the ``state/workflows`` dir (honors the test ED4ALL_STATE_RUNS_DIR).

    ``ED4ALL_STATE_RUNS_DIR`` points at ``<state_root>/runs`` (the same override
    ``gui/shared_state`` + ``lib.paths`` honor), so its PARENT is the state root
    and ``<state_root>/workflows`` is the sibling. Falls back to
    ``lib.paths.STATE_WORKFLOWS``. Never raises.
    """
    from pathlib import Path

    env_runs = os.environ.get("ED4ALL_STATE_RUNS_DIR")
    if env_runs and env_runs.strip():
        return Path(env_runs.strip()).parent / "workflows"
    from lib.paths import STATE_WORKFLOWS

    return STATE_WORKFLOWS


def _scan_ed4all_run_processes() -> List[int]:
    """Return the pids of live ``ed4all run`` processes (ADJACENT-token match).

    A LOCAL COPY of ``gui/services/liveness.py::scan_pipeline_processes`` (the
    established no-diagnostics→gui-dependency pattern): a process qualifies only
    when ``/proc/<pid>/cmdline`` carries an exact ``ed4all`` (or ``.../ed4all``)
    argv token IMMEDIATELY followed by a ``run`` token — token-adjacency, never a
    ``pgrep -f`` substring match, so a ``bash -c '... ed4all run ...'`` wrapper
    (whose whole script is one argv token) does not false-positive. NEVER raises;
    a non-``/proc`` platform or an unreadable entry is skipped.
    """
    pids: List[int] = []
    me = os.getpid()
    try:
        entries = os.listdir("/proc")
    except OSError:
        return pids
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == me:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                raw = fh.read()
        except OSError:
            continue
        argv = [tok.decode("utf-8", "replace") for tok in raw.split(b"\0") if tok]
        for i in range(len(argv) - 1):
            tok = argv[i]
            if (tok == "ed4all" or tok.endswith("/ed4all")) and argv[i + 1] == "run":
                pids.append(pid)
                break
    return pids


def _check_stale_runs(ctx: CheckContext) -> List[CheckResult]:
    """Surface non-terminal ``state/workflows/WF-*.json`` records with no live run.

    The orchestrator has no reaper that stamps a crashed / killed CLI run
    terminal, so a ``WF-*.json`` can say ``RUNNING`` forever. This scans the
    workflow store for non-terminal (active-claimed) statuses and, when NO live
    ``ed4all run`` process exists to back them, WARNs with the stale ids. NEVER
    raises.
    """
    import glob
    import json as _json

    wf_dir = _resolve_workflows_dir()
    try:
        paths = sorted(glob.glob(str(wf_dir / "WF-*.json")))
    except Exception as exc:  # noqa: BLE001 — a bad path must not crash the doctor
        logger.debug("diagnostics.environment: workflows glob failed: %s", exc)
        return [
            CheckResult(
                name="stale_runs",
                group="environment",
                severity=Severity.OK,
                summary="stale-run scan skipped (workflows dir unavailable)",
                data={"workflows_dir": str(wf_dir), "error": str(exc)},
            )
        ]

    active: List[tuple] = []  # (run_id, normalized_status)
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = _json.load(fh)
        except Exception:  # noqa: BLE001 — a malformed record is skipped, not fatal
            continue
        status = str((doc.get("status") if isinstance(doc, dict) else "") or "").strip().lower()
        if status == "complete":
            status = "completed"
        if not status or status in _TERMINAL_OR_RESTING_STATUSES:
            continue
        run_id = (doc.get("id") if isinstance(doc, dict) else None) or os.path.basename(path)[:-5]
        active.append((str(run_id), status))

    if not active:
        return [
            CheckResult(
                name="stale_runs",
                group="environment",
                severity=Severity.OK,
                summary="no non-terminal workflow records in the store",
                data={"workflows_dir": str(wf_dir), "scanned": len(paths)},
            )
        ]

    live_pids = _scan_ed4all_run_processes()
    ids = [rid for rid, _ in active]
    shown = ids[:_STALE_RUN_LIST_CAP]
    more = len(ids) - len(shown)
    id_str = ", ".join(shown) + (f" (+{more} more)" if more > 0 else "")

    if live_pids:
        # Some ed4all run process(es) are alive — one MIGHT back a record here
        # (per-run attribution needs corpus/id argv tokens, out of scope for a
        # doctor summary), so report INFO rather than a false stale WARN.
        return [
            CheckResult(
                name="stale_runs",
                group="environment",
                severity=Severity.INFO,
                summary=(
                    f"{len(active)} non-terminal workflow record(s) with "
                    f"{len(live_pids)} live `ed4all run` process(es) present — "
                    "per-run attribution not performed here"
                ),
                detail=f"non-terminal ids: {id_str}",
                data={
                    "count": len(active),
                    "run_ids": ids,
                    "live_pids": live_pids,
                    "statuses": {rid: st for rid, st in active},
                },
            )
        ]

    return [
        CheckResult(
            name="stale_runs",
            group="environment",
            severity=Severity.WARN,
            summary=(
                f"{len(active)} non-terminal workflow record(s) but NO live "
                f"`ed4all run` process — stale RUNNING record(s): {id_str}"
            ),
            detail=(
                "the orchestrator does not reap a crashed / killed CLI run, so "
                "these records claim active work with no process behind them"
            ),
            remediation=(
                "mark each cancelled or resume it (`ed4all run --resume <id>`); "
                "GC stale state with `ed4all state prune`"
            ),
            data={
                "count": len(active),
                "run_ids": ids,
                "statuses": {rid: st for rid, st in active},
                "workflows_dir": str(wf_dir),
            },
        )
    ]


# --------------------------------------------------------------------- #
# Entry point + registration
# --------------------------------------------------------------------- #

#: Ordered sub-checks — each isolated by :func:`_run_subcheck`.
_SUBCHECKS: List[Callable[[CheckContext], List[CheckResult]]] = [
    _check_ollama,
    _check_extras,
    _check_torch_nli,
    _check_dispatch_trap,
    _check_stale_runs,
]


def environment_checks(ctx: CheckContext) -> List[CheckResult]:
    """Run every environment sub-check, flattening the results (NEVER raises).

    Each sub-check is isolated by :func:`_run_subcheck`, so one failing probe
    becomes its own WARN result and the remaining sub-checks still run.
    """
    results: List[CheckResult] = []
    for fn in _SUBCHECKS:
        results.extend(_run_subcheck(fn, ctx))
    return results


def register_environment_checks() -> None:
    """Register :func:`environment_checks` under ``group="environment"``.

    Called explicitly from the CLI bootstrap — NEVER at import time.
    """
    register("environment", environment_checks)


__all__ = ["environment_checks", "register_environment_checks"]

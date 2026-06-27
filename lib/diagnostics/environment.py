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
  ``dart_conversion`` phase — a from-existing-HTML run does not need it →
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


def _check_ollama(ctx: CheckContext) -> List[CheckResult]:
    """ollama liveness + configured-model-pulled check (NEVER raises)."""
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
        "PDF->HTML conversion (dart_conversion phase) unavailable — a "
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
# Entry point + registration
# --------------------------------------------------------------------- #

#: Ordered sub-checks — each isolated by :func:`_run_subcheck`.
_SUBCHECKS: List[Callable[[CheckContext], List[CheckResult]]] = [
    _check_ollama,
    _check_extras,
    _check_torch_nli,
    _check_dispatch_trap,
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

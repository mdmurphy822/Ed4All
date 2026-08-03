"""Serving-window diagnostics check — does the served context window hold
the num_ctx budgets the pipeline ASSUMES?

This check catches the *silent head-truncation* bug class. Several authoring
seats size their grounding prompts against an ASSUMED ``num_ctx`` budget
(``ED4ALL_REWRITE_NUM_CTX`` / ``ED4ALL_ANSWER_NUM_CTX`` /
``ED4ALL_CONTENT_PAGE_NUM_CTX``). Ollama (the common local backend) serves a
FIXED window and SILENTLY truncates the prompt HEAD when the rendered prompt
exceeds it — dropping the system-prompt authoring CONTRACT and letting the
model author ungrounded. When the model's actually-served window is SMALLER
than an assumed budget, every prompt above the served window is head-truncated
with no error. This check introspects the served window (NET-NEW ollama
``/api/show`` call — nothing in the repo speaks ``/api/show`` today) and
compares it against each assumed budget.

CRITICAL distinction (the load-bearing fix): the model's *architectural*
context length (``model_info.<arch>.context_length`` — e.g. qwen2.5 reports
32768) is the TRAINED ceiling, NOT what ollama serves a request. The window a
request actually gets is its ``num_ctx``, which defaults to the Modelfile's
baked ``num_ctx`` (visible in ``/api/show`` ``parameters``) or, if unset, the
server runtime default ``OLLAMA_CONTEXT_LENGTH`` (env) or, if THAT is unset,
ollama's own built-in default (~4096). Reading the 32768 ceiling as "served"
would make this check conclude a 24k-token rewrite prompt "fits" when ollama
will truncate it at ~4096. So the served window and the architectural ceiling
are resolved + reported SEPARATELY: the ceiling is informational (INFO, never
gating); the served value drives every truncation comparison.

Like every check in :mod:`lib.diagnostics`, the entry point
(:func:`serving_window_checks`) NEVER raises; the HTTP probe
(:func:`probe_served_window`) is short-timeout (latency-sensitive preflight),
lazily imports ``httpx``, and degrades any failure to ``error != None``
(the WARN path) rather than guessing. Registration is explicit
(:func:`register_serving_window_checks`), called by the CLI bootstrap — never
at import time.
"""

from __future__ import annotations

import logging
import os
from typing import List, NamedTuple, Optional, Tuple

from lib.diagnostics.core import CheckContext, CheckResult, Severity, register
from lib.llm.vram_reclaim import resolve_ollama_root

logger = logging.getLogger(__name__)

#: Short, latency-sensitive timeout (seconds) for the ``/api/show`` probe. A
#: doctor preflight must NOT block on a wedged server — a missing answer is the
#: WARN path, not a hang.
_API_SHOW_TIMEOUT_SECONDS = 2.0

#: Documented canonical local model default (mirrors
#: ``Trainforge.generators._local_provider.DEFAULT_SYNTHESIS_MODEL``; we resolve
#: it lazily/defensively in :func:`resolve_local_model` and fall back to this
#: literal so importing the diagnostics check never drags in the synthesis
#: provider stack).
_DEFAULT_LOCAL_MODEL = "nemotron-3-nano-30b-a3b"

#: Below-or-equal this served window, GROUNDED rewrite authoring is unsafe:
#: median grounded rewrite prompts run ~24k tok, so an 8192 window is a
#: detector-only window (it exists to TRIP the truncation tripwire, not to hold
#: a real authoring prompt). See ``ED4ALL_REWRITE_FIT_WINDOW``.
_DETECTOR_ONLY_WINDOW = 8192

#: Ollama's own built-in default request window when neither a Modelfile
#: ``num_ctx`` nor ``OLLAMA_CONTEXT_LENGTH`` is set. Ollama's current default is
#: ~4096 (historically 2048); we use 4096 as the conservative documented
#: comparison floor for the "served window unconfigured" case.
_OLLAMA_DEFAULT_CONTEXT = 4096

#: How the served window was resolved — surfaced in result.data so an operator
#: can see whether the number is a Modelfile override, a runtime env default, or
#: the unconfigured ollama fallback.
_SOURCE_PARAMETERS = "modelfile_num_ctx"  # baked into the model's Modelfile
_SOURCE_ENV = "ollama_context_length_env"  # OLLAMA_CONTEXT_LENGTH
_SOURCE_UNSET = "ollama_default_unset"  # neither set → ollama built-in default


class ServedWindow(NamedTuple):
    """Result of a :func:`probe_served_window` introspection.

    - ``served``: the resolved served window in tokens (Modelfile ``num_ctx``
      or ``OLLAMA_CONTEXT_LENGTH``), or ``None`` when neither is configured
      (the "unset → ollama default" case; ``source == _SOURCE_UNSET``).
    - ``ceiling``: the model's ARCHITECTURAL context length
      (``model_info.<arch>.context_length``), or ``None`` if absent.
      Informational only — NEVER the served number.
    - ``source``: one of ``_SOURCE_PARAMETERS`` / ``_SOURCE_ENV`` /
      ``_SOURCE_UNSET``, or ``None`` when the probe itself failed.
    - ``model``: the resolved model name introspected.
    - ``error``: ``None`` on a successful probe, else a short reason string
      (ollama down / 404 / timeout / unparseable body).
    """

    served: Optional[int]
    ceiling: Optional[int]
    source: Optional[str]
    model: str
    error: Optional[str]


def resolve_local_model() -> str:
    """Resolve the local model name to introspect.

    ``LOCAL_SYNTHESIS_MODEL`` wins; absent, the documented canonical default
    (``qwen2.5:7b-instruct-q4_K_M``). The default is sourced from
    ``Trainforge.generators._local_provider.DEFAULT_SYNTHESIS_MODEL`` when it
    imports cleanly, falling back to the local literal otherwise so this check
    never hard-depends on the synthesis provider stack.
    """
    raw = os.environ.get("LOCAL_SYNTHESIS_MODEL")
    if raw and raw.strip():
        return raw.strip()
    try:  # best-effort — keep the literal default if the import chain is heavy/broken
        from Trainforge.generators._local_provider import (  # noqa: PLC0415
            DEFAULT_SYNTHESIS_MODEL,
        )

        if isinstance(DEFAULT_SYNTHESIS_MODEL, str) and DEFAULT_SYNTHESIS_MODEL.strip():
            return DEFAULT_SYNTHESIS_MODEL.strip()
    except Exception:  # noqa: BLE001 — never let a provider import break the doctor
        pass
    return _DEFAULT_LOCAL_MODEL


def _parse_num_ctx_from_parameters(parameters: object) -> Optional[int]:
    """Parse ``num_ctx`` out of the ``/api/show`` ``parameters`` field.

    Ollama returns ``parameters`` either as a newline-delimited string
    (``"num_ctx 8192\\nstop ..."``) OR as a dict. Defensive on both shapes;
    returns ``None`` when no positive ``num_ctx`` is present.
    """
    if isinstance(parameters, dict):
        val = parameters.get("num_ctx")
        try:
            ival = int(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return ival if ival > 0 else None
    if isinstance(parameters, str):
        for line in parameters.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "num_ctx":
                try:
                    ival = int(parts[1])
                except (TypeError, ValueError):
                    return None
                return ival if ival > 0 else None
    return None


def _parse_ollama_context_length_env() -> Optional[int]:
    """Parse ``OLLAMA_CONTEXT_LENGTH`` (the server runtime default) from env.

    Returns the positive int, or ``None`` when unset / non-positive / garbage
    (parse-with-fallback — a garbage value is treated as unset).
    """
    raw = os.environ.get("OLLAMA_CONTEXT_LENGTH")
    if not raw or not raw.strip():
        return None
    try:
        ival = int(raw.strip())
    except (TypeError, ValueError):
        return None
    return ival if ival > 0 else None


def _parse_architectural_ceiling(body: object) -> Optional[int]:
    """Extract the model's ARCHITECTURAL context length from ``/api/show``.

    Scans ``model_info`` for a key ENDING in ``.context_length`` (the arch
    prefix varies, e.g. ``qwen2.context_length`` / ``llama.context_length``)
    with a positive int value. This is the model's TRAINED ceiling — reported
    informationally, NEVER used as the served window. Returns ``None`` on an
    unexpected shape or when no ceiling is present.
    """
    if not isinstance(body, dict):
        return None
    model_info = body.get("model_info")
    if isinstance(model_info, dict):
        for key, val in model_info.items():
            if isinstance(key, str) and key.endswith(".context_length"):
                try:
                    ival = int(val)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if ival > 0:
                    return ival
    return None


def _resolve_served_window(body: object) -> Tuple[Optional[int], Optional[int], str]:
    """Resolve ``(served, ceiling, source)`` from an ``/api/show`` body.

    Served-window resolution order (the architectural ceiling is NEVER used as
    the served number):

    1. ``parameters`` ``num_ctx`` if present → the Modelfile-baked override
       (``source == _SOURCE_PARAMETERS``).
    2. ELSE ``OLLAMA_CONTEXT_LENGTH`` env if set + positive → the server runtime
       default (``source == _SOURCE_ENV``).
    3. ELSE ``None`` flagged ``_SOURCE_UNSET`` → ollama serves its built-in
       default (~4096). This is NOT an error — it is the common real failure
       case (a long-context model whose Modelfile never set ``num_ctx``).

    ``ceiling`` is the architectural context length, parsed separately for
    informational reporting.
    """
    ceiling = _parse_architectural_ceiling(body)

    parameters = body.get("parameters") if isinstance(body, dict) else None
    num_ctx = _parse_num_ctx_from_parameters(parameters)
    if num_ctx is not None:
        return num_ctx, ceiling, _SOURCE_PARAMETERS

    env_ctx = _parse_ollama_context_length_env()
    if env_ctx is not None:
        return env_ctx, ceiling, _SOURCE_ENV

    return None, ceiling, _SOURCE_UNSET


def probe_served_window(
    base_url: Optional[str] = None, model: Optional[str] = None
) -> ServedWindow:
    """Introspect the local model's served context window via ``/api/show``.

    NET-NEW ollama introspection — nothing else in the repo calls ``/api/show``
    today. POSTs ``{resolve_ollama_root(base_url)}/api/show`` with body
    ``{"name": <model>}`` under a SHORT timeout
    (:data:`_API_SHOW_TIMEOUT_SECONDS`), lazily importing ``httpx``. The served
    window + architectural ceiling are resolved SEPARATELY by
    :func:`_resolve_served_window` (Modelfile ``num_ctx`` → ``OLLAMA_CONTEXT_LENGTH``
    → unset for the served value; ``model_info.<arch>.context_length`` for the
    informational ceiling).

    Returns a :class:`ServedWindow`:

    - ``(served, ceiling, source, model, None)`` on a successful probe —
      ``served`` may be ``None`` with ``source == _SOURCE_UNSET`` (a VALID
      result: the model serves the ollama default ~4096),
    - ``(None, None, None, model, "<reason>")`` on ANY probe FAILURE (ollama
      down / model not found / non-200 / unparseable body).

    NEVER raises.
    """
    resolved_model = (model or "").strip() or resolve_local_model()

    try:
        import httpx  # noqa: PLC0415 — lazy; the doctor must not hard-dep httpx at import
    except Exception as exc:  # noqa: BLE001
        return ServedWindow(None, None, None, resolved_model, f"httpx unavailable: {exc}")

    try:
        root = resolve_ollama_root(base_url)
    except Exception as exc:  # noqa: BLE001 — belt-and-suspenders
        return ServedWindow(
            None, None, None, resolved_model, f"could not resolve ollama root: {exc}"
        )

    try:
        with httpx.Client(timeout=_API_SHOW_TIMEOUT_SECONDS) as client:
            resp = client.post(f"{root}/api/show", json={"name": resolved_model})
            resp.raise_for_status()
            body = resp.json()
    except Exception as exc:  # noqa: BLE001 — server down / 404 / timeout / bad JSON
        logger.debug(
            "serving_window: /api/show probe for %r at %s failed: %s",
            resolved_model,
            base_url or "<resolved>",
            exc,
        )
        return ServedWindow(None, None, None, resolved_model, f"{type(exc).__name__}: {exc}")

    if not isinstance(body, dict):
        return ServedWindow(
            None,
            None,
            None,
            resolved_model,
            "unparseable /api/show response (not a JSON object)",
        )

    served, ceiling, source = _resolve_served_window(body)
    return ServedWindow(served, ceiling, source, resolved_model, None)


def _resolve_assumed_budgets() -> List[Tuple[str, int]]:
    """Resolve the three assumed num_ctx budgets the pipeline sizes against.

    Each tuple is ``(seat_label, assumed_num_ctx)``. Resolver failures degrade
    to the documented default per resolver (the resolvers are themselves
    parse-with-fallback), and a hard import/raise is swallowed so the check
    still emits the served-window info even if a resolver is unavailable.
    """
    budgets: List[Tuple[str, int]] = []

    try:
        from Courseforge.generators.rewrite._rewrite_fit_window import (  # noqa: PLC0415
            resolve_rewrite_num_ctx,
        )

        budgets.append(("rewrite", int(resolve_rewrite_num_ctx())))
    except Exception as exc:  # noqa: BLE001
        logger.debug("serving_window: rewrite num_ctx resolver unavailable: %s", exc)

    try:
        from lib.retrieval._prompts import resolve_num_ctx  # noqa: PLC0415

        budgets.append(("answer", int(resolve_num_ctx())))
    except Exception as exc:  # noqa: BLE001
        logger.debug("serving_window: answer num_ctx resolver unavailable: %s", exc)

    try:
        from lib.generation.content_page_budget import (  # noqa: PLC0415
            resolve_content_page_num_ctx,
        )

        budgets.append(("content_page", int(resolve_content_page_num_ctx())))
    except Exception as exc:  # noqa: BLE001
        logger.debug("serving_window: content_page num_ctx resolver unavailable: %s", exc)

    return budgets


def serving_window_checks(ctx: CheckContext) -> List[CheckResult]:
    """Compare each assumed num_ctx budget against the model's SERVED window.

    Resolves the local model name + probes the served window (separately from
    the architectural ceiling). When the probe fails, emits ONE WARN (a missing
    answer must not abort the doctor). Otherwise:

    - if the served window is UNSET (no Modelfile ``num_ctx`` + no
      ``OLLAMA_CONTEXT_LENGTH``), emits a WARN that ollama will serve its
      unconfigured default (~4096) and compares budgets conservatively against
      4096 (the common, previously-masked real failure);
    - else emits an OK info result stating the served window and compares each
      assumed budget against it (``assumed > served`` → truncation WARN);
    - emits a separate INFO result reporting the architectural ceiling (never
      gating); and
    - emits an INFO note when the served window ``<= 8192`` (detector-only for
      GROUNDED rewrite authoring — informational, NOT a WARN).

    NEVER raises.
    """
    try:
        return _serving_window_checks(ctx)
    except Exception as exc:  # noqa: BLE001 — belt-and-suspenders (probe never raises)
        logger.warning("serving_window: check raised unexpectedly: %s", exc)
        return [
            CheckResult(
                name="serving_window_error",
                group="window",
                severity=Severity.WARN,
                summary=f"serving-window check errored: {exc}",
                detail=f"{type(exc).__name__}: {exc}",
                remediation="this is a doctor bug — the check should never raise",
                data={"error": str(exc), "error_type": type(exc).__name__},
            )
        ]


def _serving_window_vllm(ctx: CheckContext, topo) -> List[CheckResult]:
    """vLLM-seat served-window note (P0-1): no ollama ``/api/show`` truncation warn.

    ollama's ``/api/show`` served-window introspection is ollama-specific. A vLLM
    seat serves a FIXED context window (``--max-model-len``, set at launch) that
    ``/api/show`` cannot report, so probing it here mints a FALSE "served window
    unknown" WARN. Instead we probe ``/v1/models`` liveness and emit INFO: the
    ollama-only truncation comparison does not apply. NEVER raises.
    """
    from lib.diagnostics.run_env import probe_v1_models

    root = topo.base_url_root
    seat_label = topo.seat_name or "explicit endpoint"
    backend_label = (
        "a vLLM seat"
        if topo.backend == "vllm"
        else "an OpenAI-compatible endpoint"
    )
    live, model_ids, error = probe_v1_models(root)
    return [
        CheckResult(
            name="serving_window_vllm_seat",
            group="window",
            severity=Severity.INFO,
            summary=(
                f"local-synthesis backend is {backend_label} ({seat_label}) at {root}; "
                "Ollama /api/show served-window introspection does not apply "
                + ("(seat live)" if live else f"(seat not answering /v1/models: {error})")
            ),
            detail=(
                "a vLLM seat serves a FIXED context window (--max-model-len set at "
                "launch) that /api/show cannot report, so the ollama-only "
                "head-truncation comparison is skipped; seat liveness is owned by "
                "the 'seat' group"
            ),
            data={
                "backend": topo.backend,
                "base_url": root,
                "seat_name": topo.seat_name,
                "live": live,
                "served_model_ids": model_ids,
                "error": error,
            },
        )
    ]


def _serving_window_checks(ctx: CheckContext) -> List[CheckResult]:
    """Inner body for :func:`serving_window_checks` (the public wrapper catches)."""
    results: List[CheckResult] = []

    if ctx.local_synthesis_active is False:
        return [
            CheckResult(
                name="serving_window_inactive",
                group="window",
                severity=Severity.INFO,
                summary="local-synthesis served-window check is not active",
                detail=(
                    "No active local synthesis provider or explicit "
                    "LOCAL_SYNTHESIS_BASE_URL/_MODEL is configured, so an "
                    "Ollama /api/show probe would test an unused endpoint."
                ),
                data={"active": False},
            )
        ]

    # P0-1: when a seat registry is configured and LOCAL_SYNTHESIS_BASE_URL
    # resolves to a vLLM seat, skip the ollama /api/show probe (no false
    # "served window unknown" WARN). No seat registry → legacy path unchanged.
    try:
        from lib.diagnostics.run_env import resolve_local_synthesis_topology

        topo = resolve_local_synthesis_topology()
        if topo.backend in {"vllm", "openai_compatible"}:
            return _serving_window_vllm(ctx, topo)
    except Exception as exc:  # noqa: BLE001 — topology resolution must not crash the doctor
        logger.debug("serving_window: topology resolution failed: %s", exc)

    model = resolve_local_model()
    probe = probe_served_window(ctx.base_url, model)
    model = probe.model or model

    if probe.error is not None:
        results.append(
            CheckResult(
                name="serving_window_unknown",
                group="window",
                severity=Severity.WARN,
                summary=f"could not determine served window for {model}",
                detail=(f"probe error: {probe.error}" if probe.error else ""),
                remediation=(
                    "ensure ollama is running and the model is pulled; check "
                    "LOCAL_SYNTHESIS_BASE_URL/_MODEL"
                ),
                data={"model": model, "served_window": None, "error": probe.error},
            )
        )
        return results

    budgets = _resolve_assumed_budgets()
    ceiling = probe.ceiling

    # The served window UNSET case: ollama serves its built-in default (~4096).
    # The served value drives comparisons; carry both it and the ceiling so the
    # operator sees "currently serves ~4096, can support up to <ceiling>".
    if probe.source == _SOURCE_UNSET or probe.served is None:
        effective_served = _OLLAMA_DEFAULT_CONTEXT
        ceiling_hint = (
            f" (the model architecture supports up to {ceiling} tokens if you raise "
            "num_ctx)"
            if ceiling
            else ""
        )
        results.append(
            CheckResult(
                name="serving_window_unset_default",
                group="window",
                severity=Severity.WARN,
                summary=(
                    f"{model} has no Modelfile num_ctx and OLLAMA_CONTEXT_LENGTH is "
                    f"unset -> ollama serves its default ~{_OLLAMA_DEFAULT_CONTEXT}-token "
                    f"window; assumed budgets above ~{_OLLAMA_DEFAULT_CONTEXT} are "
                    "silently HEAD-truncated" + ceiling_hint
                ),
                detail=(
                    "neither a baked Modelfile num_ctx nor OLLAMA_CONTEXT_LENGTH was "
                    f"found, so the served window is the unconfigured ollama default "
                    f"(~{_OLLAMA_DEFAULT_CONTEXT}); budgets are compared conservatively "
                    f"against {_OLLAMA_DEFAULT_CONTEXT}"
                ),
                remediation=(
                    "set a Modelfile num_ctx or OLLAMA_CONTEXT_LENGTH >= the largest "
                    "assumed budget, or lower the ED4ALL_*_NUM_CTX knobs"
                ),
                data={
                    "model": model,
                    "served_window": None,
                    "served_source": _SOURCE_UNSET,
                    "effective_served_window": effective_served,
                    "architectural_ceiling": ceiling,
                    "assumed": {label: val for label, val in budgets},
                },
            )
        )
    else:
        effective_served = probe.served
        results.append(
            CheckResult(
                name="serving_window_served",
                group="window",
                severity=Severity.OK,
                summary=(
                    f"{model} serves a context window of {effective_served} tokens "
                    f"({probe.source})"
                ),
                detail=(
                    "assumed budgets: "
                    + ", ".join(f"{label}={val}" for label, val in budgets)
                    if budgets
                    else ""
                ),
                data={
                    "model": model,
                    "served_window": effective_served,
                    "served_source": probe.source,
                    "architectural_ceiling": ceiling,
                    "assumed": {label: val for label, val in budgets},
                },
            )
        )

    for label, assumed in budgets:
        if assumed > effective_served:
            results.append(
                CheckResult(
                    name=f"serving_window_fit_{label}",
                    group="window",
                    severity=Severity.WARN,
                    summary=(
                        f"{label} assumes num_ctx={assumed} but {model} serves "
                        f"{effective_served} -> prompts above {effective_served} are "
                        "silently HEAD-truncated (system-prompt contract dropped)"
                    ),
                    detail=(
                        f"the {label} seat sizes its grounding against {assumed} "
                        f"tokens; the served window is only {effective_served}"
                        + (
                            f" (unconfigured ollama default ~{_OLLAMA_DEFAULT_CONTEXT})"
                            if probe.source == _SOURCE_UNSET or probe.served is None
                            else ""
                        )
                    ),
                    remediation=(
                        "raise the model's Modelfile num_ctx / "
                        "OLLAMA_CONTEXT_LENGTH to >= the assumed value, or lower "
                        f"the ED4ALL_*_NUM_CTX knob for the {label} seat"
                    ),
                    data={
                        "seat": label,
                        "assumed": assumed,
                        "served_window": probe.served,
                        "effective_served_window": effective_served,
                        "served_source": probe.source,
                        "architectural_ceiling": ceiling,
                        "model": model,
                    },
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"serving_window_fit_{label}",
                    group="window",
                    severity=Severity.OK,
                    summary=(
                        f"{label} assumes num_ctx={assumed} <= {model} served "
                        f"window {effective_served} (fits)"
                    ),
                    data={
                        "seat": label,
                        "assumed": assumed,
                        "served_window": probe.served,
                        "effective_served_window": effective_served,
                        "served_source": probe.source,
                        "architectural_ceiling": ceiling,
                        "model": model,
                    },
                )
            )

    # Architectural ceiling — INFORMATIONAL ONLY (never gates the exit code).
    if ceiling:
        results.append(
            CheckResult(
                name="serving_window_ceiling",
                group="window",
                severity=Severity.INFO,
                summary=(
                    f"{model} architecture supports up to {ceiling} tokens "
                    f"(raise num_ctx / OLLAMA_CONTEXT_LENGTH to use it; currently "
                    f"serves {effective_served})"
                ),
                detail=(
                    "this is the model's TRAINED/architectural context ceiling, NOT "
                    "the window ollama serves a request — the served window is "
                    "governed by num_ctx"
                ),
                data={
                    "model": model,
                    "architectural_ceiling": ceiling,
                    "served_window": probe.served,
                    "effective_served_window": effective_served,
                    "served_source": probe.source,
                },
            )
        )

    if effective_served <= _DETECTOR_ONLY_WINDOW:
        results.append(
            CheckResult(
                name="serving_window_detector_only",
                group="window",
                severity=Severity.INFO,
                summary=(
                    f"{model} served window {effective_served} <= "
                    f"{_DETECTOR_ONLY_WINDOW} is detector-only for GROUNDED rewrite "
                    "authoring (median grounded rewrite prompts run ~24k tokens)"
                ),
                detail=(
                    "an 8192 window exists to TRIP the rewrite truncation tripwire, "
                    "not to hold a real grounded authoring prompt; see "
                    "ED4ALL_REWRITE_FIT_WINDOW"
                ),
                remediation=(
                    "for real grounded rewrite authoring, serve a larger context "
                    "window (Modelfile num_ctx / OLLAMA_CONTEXT_LENGTH), or enable "
                    "ED4ALL_REWRITE_FIT_WINDOW to shrink the prompt to fit"
                ),
                data={
                    "served_window": probe.served,
                    "effective_served_window": effective_served,
                    "detector_only_ceiling": _DETECTOR_ONLY_WINDOW,
                    "model": model,
                },
            )
        )

    return results


def register_serving_window_checks() -> None:
    """Register :func:`serving_window_checks` under ``group="window"``.

    Called explicitly from the CLI bootstrap, NEVER at import time (mirrors
    :func:`lib.diagnostics.vram.register_gpu_checks`).
    """
    register("window", serving_window_checks)


__all__ = [
    "ServedWindow",
    "probe_served_window",
    "serving_window_checks",
    "register_serving_window_checks",
    "resolve_local_model",
]

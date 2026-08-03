"""Per-call LLM usage metering tap for SemantiK's Super-120B call sites (P3).

A SMALL, dependency-free, STRICTLY best-effort meter that appends ONE JSON row
per LLM completion to an ``llm_usage.jsonl`` ledger, mirroring the Trainforge
OP2 usage tap (``Trainforge/generators/providers/_openai_compatible_client.py``
``_maybe_append_usage_row``) WITHOUT importing it — the vendored SemantiK tree
runs inside its own venv (which ships ``requests`` but not the Trainforge
package; see ``qwen_specialists/endpoint_runtime.py`` module docstring), so this
is a COPY of the pattern, never a cross-venv import.

Ledger location
---------------
* ``ED4ALL_RUN_ID`` set  → ``<state-runs>/<run_id>/llm_usage.jsonl`` (the same
  file the Trainforge tap writes, so a full pipeline run's SemantiK + Trainforge
  calls land in ONE ledger the ``BuildCostAggregator`` already reads).
* ``ED4ALL_RUN_ID`` unset → a per-run sidecar under the SemantiK data dir
  (``<SEMANTIK_DATA_DIR>/llm_usage/llm_usage.jsonl``) OR an explicit
  ``output_dir`` when a call site has one, so a STANDALONE SemantiK run (no
  Ed4All orchestration) still meters.

Row shape
---------
``{ts, site, model, prompt_tokens, completion_tokens, total_tokens,
duration_ms}`` — plus, ADDITIVELY (present-only, so a non-reasoning seat's row
is unchanged), the P1 measurement fields when the response exposes them:

* ``reasoning_tokens`` — ``usage.completion_tokens_details.reasoning_tokens``
  (some vLLM/OpenAI reasoning seats break this out; the live Nemotron-3 Super
  seat on vLLM 0.21.0 does NOT — its reasoning tokens are folded into
  ``completion_tokens`` and the reasoning TEXT rides ``message.reasoning``).
* ``reasoning_chars`` — ``len(choices[0].message.reasoning)`` when the seat
  returns a separate reasoning channel (the Nemotron-3 shape); a char-length
  PROXY for thinking effort when no reasoning-token count exists.

Strict best-effort
------------------
ANY failure (unresolvable dir, unwritable path, malformed response, serialization
error) is swallowed — metering must NEVER perturb or fail a real LLM call. This
module imports ONLY the standard library + ``semantik_structure.paths`` (itself
dependency-free), so importing it stays cheap and cross-venv-safe.
"""
from __future__ import annotations

import json as _json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RUN_ID_ENV = "ED4ALL_RUN_ID"
_STATE_RUNS_DIR_ENV = "ED4ALL_STATE_RUNS_DIR"
_ROOT_ENV = "ED4ALL_ROOT"

_LEDGER_NAME = "llm_usage.jsonl"


def _run_ledger_path() -> Optional[Path]:
    """Resolve ``<state-runs>/<run_id>/llm_usage.jsonl`` when a run id is set.

    Cross-venv-safe: tries the explicit ``ED4ALL_STATE_RUNS_DIR`` env first (the
    bridge exports it), then Ed4All's ``lib.paths`` when importable (in-process
    arm), then ``ED4ALL_ROOT/state/runs``. Returns ``None`` (→ sidecar fallback)
    when no run id or no base dir can be resolved.
    """
    run_id = os.environ.get(_RUN_ID_ENV, "").strip()
    if not run_id:
        return None
    base = os.environ.get(_STATE_RUNS_DIR_ENV, "").strip()
    if base:
        return Path(base) / run_id / _LEDGER_NAME
    try:  # in-process (Ed4All venv) — mirrors the Trainforge tap
        from lib.paths import get_state_runs_dir  # type: ignore

        return Path(get_state_runs_dir()) / run_id / _LEDGER_NAME
    except Exception:  # noqa: BLE001 — bridge venv lacks lib/ → fall through
        pass
    root = os.environ.get(_ROOT_ENV, "").strip()
    if root:
        return Path(root) / "state" / "runs" / run_id / _LEDGER_NAME
    return None


def _sidecar_ledger_path(output_dir: Optional[Any]) -> Optional[Path]:
    """Resolve the standalone-run sidecar ledger (no ``ED4ALL_RUN_ID``)."""
    if output_dir:
        return Path(output_dir) / _LEDGER_NAME
    try:
        from . import paths as _paths  # dependency-free, same package
    except Exception:  # noqa: BLE001
        try:
            from semantik_structure import paths as _paths  # type: ignore
        except Exception:  # noqa: BLE001
            return None
    try:
        return _paths.data_dir() / "llm_usage" / _LEDGER_NAME
    except Exception:  # noqa: BLE001
        return None


def _extract_reasoning_fields(data: Any) -> dict:
    """Pull the ADDITIVE (present-only) reasoning-effort fields off a response.

    Both are omitted when absent so a non-reasoning seat's row is byte-identical
    to the base shape. See the module docstring for the two channels.
    """
    fields: dict = {}
    if not isinstance(data, dict):
        return fields
    usage = data.get("usage")
    if isinstance(usage, dict):
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            rt = details.get("reasoning_tokens")
            if isinstance(rt, (int, float)):
                fields["reasoning_tokens"] = int(rt)
    try:
        message = data["choices"][0]["message"]
        reasoning = message.get("reasoning") if isinstance(message, dict) else None
        if isinstance(reasoning, str) and reasoning:
            fields["reasoning_chars"] = len(reasoning)
    except (KeyError, IndexError, TypeError):
        pass
    return fields


def record_provider_usage(
    *,
    provider: str,
    model: str,
    usage: Any = None,
    duration_ms: float,
    finish_reason: Optional[str] = None,
    phase: Optional[str] = None,
    extra: Optional[dict] = None,
    output_dir: Optional[Any] = None,
) -> None:
    """Append ONE PROVIDER-keyed metering row (strict best-effort, never raises).

    The OP2 / heading-judge row shape — ``{ts, provider, model, prompt_tokens,
    completion_tokens, duration_ms}`` plus ``finish_reason`` when reported —
    which is what the GUI stats band (``gui/services/progress_service.py``) and
    the ``BuildCostAggregator`` key on (they read ``row["provider"]``; the
    ``site``-keyed :func:`record_llm_usage` rows tally as provider "unknown").
    Used by the GLM-OCR lane call sites (``semantik-glmocr`` /
    ``semantik-alttext`` / the in-lane ``semantik-heading-judge`` fallback).

    ``usage`` is the server-reported usage dict (or ``None``); token counts come
    ONLY from it — absent usage → zeros, per the existing tap contract. ``extra``
    carries additive per-surface fields (e.g. ``{"pages": N}``). Ledger
    resolution is identical to :func:`record_llm_usage`: the run ledger when
    ``ED4ALL_RUN_ID`` resolves, else the standalone sidecar.
    """
    try:
        path = _run_ledger_path() or _sidecar_ledger_path(output_dir)
        if path is None:
            return
        usage = usage if isinstance(usage, dict) else {}
        row: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "provider": str(provider),
            "model": str(model),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "duration_ms": round(float(duration_ms), 3),
        }
        if phase:
            row["phase"] = str(phase)
        if finish_reason is not None:
            row["finish_reason"] = str(finish_reason)
        if isinstance(extra, dict):
            for key, value in extra.items():
                row.setdefault(str(key), value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(row) + "\n")
    except Exception as exc:  # noqa: BLE001 — metering must never crash a call
        logger.debug("llm_usage tap failed (ignoring): %s", exc)


def record_llm_usage(
    *,
    site: str,
    model: str,
    data: Any,
    duration_ms: float,
    output_dir: Optional[Any] = None,
) -> None:
    """Append ONE metering row for a completed LLM call (strict best-effort).

    ``data`` is the parsed response JSON (the tap reads ``data["usage"]`` and,
    present-only, the reasoning channel). ``site`` is the call-site tag
    (``super_resegment`` / ``endpoint_specialist`` / ``reasoning_qc`` /
    ``page_arranger``). ``output_dir`` is an OPTIONAL SemantiK output dir for the
    no-run-id sidecar. NEVER raises.
    """
    try:
        path = _run_ledger_path() or _sidecar_ledger_path(output_dir)
        if path is None:
            return
        usage = data.get("usage") if isinstance(data, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "site": str(site),
            "model": str(model),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "duration_ms": round(float(duration_ms), 3),
        }
        row.update(_extract_reasoning_fields(data))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(row) + "\n")
    except Exception as exc:  # noqa: BLE001 — metering must never crash a call
        logger.debug("llm_usage tap failed (ignoring): %s", exc)

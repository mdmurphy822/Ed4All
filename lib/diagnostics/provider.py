"""Provider / seat preflight checks — the ``provider`` doctor group.

These checks answer the one question that decides whether an ``ed4all run``
will quietly succeed, loudly fail, or silently route the WRONG way:

  *For the run the operator is about to launch, which provider does every
  authoring / synthesis / answer / embedding seat actually resolve to, and
  is that resolution correct + reachable?*

The heavy lifting (seat resolution + the run-env fanout modelling) lives in
the pure :mod:`lib.diagnostics.run_env` engine; this module is the doctor
adapter that turns its :class:`~lib.diagnostics.run_env.SeatResolution`
records into operator-facing :class:`~lib.diagnostics.core.CheckResult`.

Two modes, driven by ``ctx.run_config``:

* **RUN mode** (``run_config['workflow']`` is truthy) — model the ACTUAL
  run. Seat resolution happens INSIDE
  :func:`~lib.diagnostics.run_env.applied_run_env`, so the corpus-
  generalization + authoring-route fanout is applied exactly as a real
  ``ed4all run`` would apply it. This is the authoritative mode: the
  per-seat key-presence FAILs, the authoring-vs-synthesis split-brain WARN,
  and the training-synthesis licensing WARN are all gating here. A
  precomputed ``cloud_seat_preflight`` dict (legacy key: ``nvidia_preflight``;
  the CLI runs it under the same fanout) is mapped through verbatim.

* **BARE mode** (no ``workflow``) — resolve seats off the CURRENT
  ``os.environ`` with NO fanout. Split-brain / licensing are advisory-only
  here (they are only authoritative under the fanout) and are skipped.

The key-presence FAIL is gated by ONE universal rule in BOTH modes: only an
EXPLICITLY configured seat (``source != 'class_default'``) with a missing
required key is gating. A bare class-default seat is never a real bug — a
corpus-gen run's fanout flips most unset seats to ``local``, and a
non-corpus-gen workflow never runs the fanout so the seat just keeps a
class-default it never uses. A class-default with a missing key → at most
INFO, never FAIL.

The opt-in **ping** (``run_config['ping']``) is the ONLY thing in this
module that touches the network: a minimal reachability call per distinct
OpenAI-compatible provider, with a short timeout and a single attempt (no
backoff on a bad key). A ping only proves reachability + accepted
credentials: a tiny-``max_tokens`` truncation (``output_truncated``) is
SUCCESS, only auth (401/403) + transport errors are FAIL.

Design contract (inherited from :mod:`lib.diagnostics.core`): a doctor that
crashes the run is worse than no doctor. Every sub-section is isolated so a
single failing probe becomes its own result and the rest still run;
:func:`provider_checks` NEVER raises. The ping client + the run-env engine's
heavier paths are imported lazily. Registration is explicit
(:func:`register_provider_checks`), never at import time.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from lib.diagnostics import run_env
from lib.diagnostics.core import CheckContext, CheckResult, Severity, register

logger = logging.getLogger(__name__)

_GROUP = "provider"

#: Short per-request timeout (seconds) for the opt-in reachability ping. A
#: ping is a 1-token liveness call, not a generation — the doctor must not
#: stall on a wedged server.
_PING_TIMEOUT_SECONDS = 5.0

_TRUTHY = {"1", "true", "yes", "on"}

#: Authoring seats — the tiers ``--provider`` is meant to fill.
_AUTHORING_SEATS = {"rewrite", "outline", "content", "course_outline"}
#: The single corpus-synthesis seat the split-brain check compares authoring
#: against — pinned FIRST by the corpus-generalization setdefault, so
#: ``--provider`` does NOT reach it (the split-brain trap). NB:
#: ``training_synthesis`` is deliberately NOT here — the nvidia branch pins it
#: local for licensing, so a local training seat under cloud authoring is
#: correct, not split-brain (see :func:`_split_brain`).
_TEXTBOOK_SYNTHESIS_SEAT = "textbook_synthesis"

#: Providers whose training-data outputs are ToS-restricted (corpus taint).
#: This set MIRRORS the licensing posture in ``docs/LICENSING.md`` §
#: "Synthesis providers" (the single source of truth) — Anthropic
#: (Commercial/Consumer Terms) and NVIDIA (Llama-3.3 hosted tier) outputs are
#: NOT shippable as SLM training data. Keep it in sync when that doc changes;
#: drift between this constant and the doc is a documentation bug (mirrors the
#: doc-mirrored-constant convention used elsewhere, e.g. the richer-CSS token
#: prelude's checked-in mirror of templates/_base/variables.css).
_LICENSE_RESTRICTED = {"anthropic", "nvidia"}

#: OpenAI-compatible providers we can reach with a 1-token chat call.
_PINGABLE = {
    "local",
    "together",
    "nvidia",
    "nvidia-deepseek",
    "groq",
    "fireworks",
    "deepseek",
}

#: SDK-only providers — not pingable here (key presence is checked
#: separately).
_ANTHROPIC_FAMILY = {"anthropic", "claude_session"}

#: Providers that are "local-ish" for split-brain purposes (a synthesis seat
#: on one of these is the silently-stays-local half of the trap).
_LOCAL_LIKE = {"local", "mock"}

_BARE_CAVEAT = (
    "current env; a run applies a fanout that flips most unset seats to "
    "local — use `ed4all doctor --run <workflow>` to model an actual run"
)


# --------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------- #


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _join(parts: List[str]) -> str:
    return "; ".join(p for p in parts if p)


def _is_local_like(provider: Optional[str]) -> bool:
    return provider is None or provider in _LOCAL_LIKE


def _error(name: str, exc: Exception) -> CheckResult:
    """A doctor-bug result — a provider sub-check raised when it must not."""
    return CheckResult(
        name=name,
        group=_GROUP,
        severity=Severity.WARN,
        summary=f"provider check errored: {exc}",
        detail=f"{type(exc).__name__}: {exc}",
        remediation="this is a doctor bug — the provider check should never raise",
        data={"error": str(exc), "error_type": type(exc).__name__},
    )


def _guarded(fn: Callable[[], List[CheckResult]], name: str) -> List[CheckResult]:
    """Run ``fn`` isolated — any exception becomes a single WARN result."""
    try:
        return fn() or []
    except Exception as exc:  # noqa: BLE001 — a doctor must never crash
        logger.warning("provider: sub-check %r raised: %s", name, exc)
        return [_error(name, exc)]


# --------------------------------------------------------------------- #
# Ping client (lazy import so a bare import stays cheap + monkeypatchable)
# --------------------------------------------------------------------- #


def _make_ping_client(
    *,
    base_url: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    provider_label: str,
) -> Any:
    """Construct the OpenAI-compatible client used for the reachability ping.

    Factored out so tests monkeypatch THIS (rather than the lazily-imported
    symbol). A single attempt (``max_retries=1``) so a 401 / bad key fails
    fast instead of paying exponential backoff.
    """
    from Trainforge.generators.providers._openai_compatible_client import (  # lazy
        OpenAICompatibleClient,
    )

    return OpenAICompatibleClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        provider_label=provider_label,
        timeout=_PING_TIMEOUT_SECONDS,
        max_retries=1,
    )


# --------------------------------------------------------------------- #
# Provider-level captured records (key value + base_url + model)
# --------------------------------------------------------------------- #


def _capture_records(
    seats: List[Any], key_table: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Capture, per distinct effective provider, the data the ping needs.

    Reads ``os.environ`` for the raw API key value (``key_status`` reports
    only PRESENCE, not the value). MUST be called inside
    :func:`applied_run_env` in RUN mode so a fanout-set base-url is seen.
    Never raises.
    """
    records: Dict[str, Dict[str, Any]] = {}
    for seat in seats:
        provider = getattr(seat, "effective_provider", None)
        if not provider or provider in records:
            continue
        try:
            status = run_env.key_status(provider, key_table)
        except Exception:  # noqa: BLE001 — capture is best-effort
            status = {}
        row = key_table.get(provider, {}) if isinstance(key_table, dict) else {}
        key_env = row.get("key_env")
        api_key = os.environ.get(key_env) if key_env else None
        if api_key is not None:
            api_key = api_key.strip() or None
        # Prefer an explicitly-pinned seat model; fall back to the registry
        # default. (For a 1-token ping the model barely matters, but the
        # client refuses an empty model.)
        model = row.get("default_model")
        for other in seats:
            if getattr(other, "effective_provider", None) == provider and getattr(
                other, "model", None
            ):
                model = other.model
                break
        records[provider] = {
            "key_status": status,
            "api_key": api_key,
            "base_url": status.get("base_url") if isinstance(status, dict) else None,
            "model": model,
        }
    return records


# --------------------------------------------------------------------- #
# Per-seat resolution + key presence
# --------------------------------------------------------------------- #


def _one_seat(
    seat: Any, key_table: Dict[str, Dict[str, Any]], bare: bool
) -> List[CheckResult]:
    """Resolution INFO (+ key-presence FAIL / OK fold) for one seat."""
    provider = getattr(seat, "effective_provider", None)
    source = getattr(seat, "source", "?")
    model = getattr(seat, "model", None)
    note = getattr(seat, "note", "") or ""
    name = getattr(seat, "seat", "?")
    provider_env = getattr(seat, "provider_env", "")

    model_part = f" / {model}" if model else ""
    summary = f"seat '{name}' → {provider or 'off'} ({source}){model_part}"

    detail_parts: List[str] = []
    if note:
        detail_parts.append(note)
    if bare:
        detail_parts.append(_BARE_CAVEAT)

    data: Dict[str, Any] = {
        "seat": name,
        "provider_env": provider_env,
        "effective_provider": provider,
        "model": model,
        "source": source,
        "note": note,
    }

    # Key presence (only meaningful for a resolved provider).
    status: Dict[str, Any] = {}
    if provider:
        try:
            status = run_env.key_status(provider, key_table) or {}
        except Exception:  # noqa: BLE001 — never let a key probe crash
            status = {}
    key_required = bool(status.get("key_required"))
    key_present = bool(status.get("key_present", True))
    key_env = status.get("key_env")

    results: List[CheckResult] = []
    resolution_severity = Severity.INFO

    if provider and key_required and not key_present:
        if source == "class_default":
            # #5 — UNIFIED rule (both bare AND run mode): a class-default seat
            # is not an EXPLICITLY-configured provider. The workflow may never
            # use it — a corpus-gen run's fanout flips most unset seats to
            # local, and a non-corpus-gen workflow (rag_training,
            # trainforge_train, courseforge-*) never runs the fanout so the
            # seat just stays its bare anthropic class-default with no key it
            # never needs. Only an EXPLICITLY resolved seat (env/fanout/yaml)
            # with a missing required key is a real bug. Note it, never FAIL.
            detail_parts.append(
                f"{key_env} is unset, but seat '{name}' is only a class default "
                "(not explicitly configured via env/fanout/yaml) — not gating"
            )
        else:
            results.append(
                CheckResult(
                    name=f"seat_{name}_key_missing",
                    group=_GROUP,
                    severity=Severity.FAIL,
                    summary=(
                        f"seat '{name}' resolves to {provider} but {key_env} is "
                        "missing → that phase will fail/poison-pill at run time"
                    ),
                    detail=_join(detail_parts),
                    remediation=f"export {key_env}=<your key> before launching the run",
                    data={**data, "key_env": key_env, "key_present": False},
                )
            )
    elif provider and key_required and key_present:
        # Folded: a present required key upgrades the resolution line to OK.
        resolution_severity = Severity.OK

    resolution = CheckResult(
        name=f"seat_{name}",
        group=_GROUP,
        severity=resolution_severity,
        summary=summary,
        detail=_join(detail_parts),
        remediation="",
        data={**data, "key_env": key_env, "key_present": key_present},
    )
    # Resolution line first, then any key-missing FAIL.
    return [resolution, *results]


def _seat_results(
    seats: List[Any], key_table: Dict[str, Dict[str, Any]], bare: bool
) -> List[CheckResult]:
    out: List[CheckResult] = []
    for seat in seats:
        try:
            out.extend(_one_seat(seat, key_table, bare))
        except Exception as exc:  # noqa: BLE001 — isolate one bad seat
            out.append(_error(f"seat_{getattr(seat, 'seat', '?')}_error", exc))
    return out


# --------------------------------------------------------------------- #
# Split-brain + licensing (authoritative only under the fanout)
# --------------------------------------------------------------------- #


def _split_brain(seats: List[Any]) -> List[CheckResult]:
    """WARN when authoring routes cloud but textbook-synthesis stays local.

    The split-brain trap is narrow: ``--provider`` fills the AUTHORING seats,
    but the corpus-generalization setdefault pins ``textbook_synthesis``
    FIRST, so authoring goes cloud while textbook synthesis silently stays
    local. This compares authoring against ``textbook_synthesis`` ONLY.

    ``training_synthesis`` is DELIBERATELY excluded: the nvidia branch pins it
    local for licensing (``docs/LICENSING.md`` — the SLM training corpus must
    never route through a ToS-restricted teacher). A local training seat under
    cloud authoring is CORRECT, not split-brain, and advising the operator to
    route it cloud would taint the corpus.

    Only EXPLICITLY-resolved authoring seats (``source != 'class_default'``)
    count: an unset class-default authoring seat means the legacy no-LLM
    TEMPLATE path runs (no cloud call), so it is not a real cloud authoring
    seat.
    """
    authoring_cloud_set: set = set()
    for s in seats:
        if getattr(s, "seat", None) not in _AUTHORING_SEATS:
            continue
        if getattr(s, "source", None) == "class_default":
            continue  # #8 — an unset class-default seat runs the template path
        p = getattr(s, "effective_provider", None)
        if p and not _is_local_like(p):
            authoring_cloud_set.add(p)
    authoring_cloud = sorted(authoring_cloud_set)
    if not authoring_cloud:
        return []

    ts = next(
        (s for s in seats if getattr(s, "seat", None) == _TEXTBOOK_SYNTHESIS_SEAT),
        None,
    )
    if ts is None:
        return []
    ts_provider = getattr(ts, "effective_provider", None)
    if not _is_local_like(ts_provider):
        return []  # textbook-synthesis already follows authoring — no lag

    x = authoring_cloud[0]
    y = ts_provider or "local"
    return [
        CheckResult(
            name="provider_split_brain",
            group=_GROUP,
            severity=Severity.WARN,
            summary=(
                f"authoring={x} but textbook-synthesis={y} — split-brain: "
                "--provider fills authoring but corpus-generalization setdefault "
                "pins textbook-synthesis first, so it silently stays local"
            ),
            detail=(
                f"authoring cloud seats: {', '.join(authoring_cloud)}; "
                f"textbook_synthesis={y} (the seat that should follow "
                "authoring). training_synthesis is intentionally pinned local "
                "for licensing and is NOT part of this check."
            ),
            remediation=(
                f"export TEXTBOOK_SYNTHESIS_PROVIDER={x} to route textbook "
                "synthesis with authoring; leave TRAINFORGE_SYNTHESIS_PROVIDER "
                "local — the training corpus stays license-clean by design"
            ),
            data={
                "authoring_cloud": authoring_cloud,
                "textbook_synthesis": y,
            },
        )
    ]


def _licensing_guard(seats: List[Any]) -> List[CheckResult]:
    """WARN when the SLM training corpus routes through a ToS-restricted seat."""
    ts = next((s for s in seats if getattr(s, "seat", None) == "training_synthesis"), None)
    if ts is None:
        return []
    provider = getattr(ts, "effective_provider", None)
    if provider in _LICENSE_RESTRICTED:
        return [
            CheckResult(
                name="provider_training_synthesis_licensing",
                group=_GROUP,
                severity=Severity.WARN,
                summary=(
                    f"training-synthesis routes the SLM corpus through {provider} "
                    "(ToS-restricted) → corpus taint"
                ),
                detail=(
                    "the trained adapter is a derivative work of these outputs; "
                    "see docs/LICENSING.md § 'Synthesis providers'"
                ),
                remediation=(
                    "pin TRAINFORGE_SYNTHESIS_PROVIDER=local (Apache-2.0 Qwen) for a "
                    "license-clean corpus"
                ),
                data={"training_synthesis_provider": provider},
            )
        ]
    return [
        CheckResult(
            name="provider_training_synthesis_licensing",
            group=_GROUP,
            severity=Severity.OK,
            summary=(
                f"training-synthesis provider '{provider or 'mock'}' is license-clean "
                "for training data"
            ),
            detail="local / mock / together are training-permitted per docs/LICENSING.md",
            remediation="",
            data={"training_synthesis_provider": provider},
        )
    ]


# --------------------------------------------------------------------- #
# cloud-seat preflight passthrough (precomputed by the CLI under the fanout)
# --------------------------------------------------------------------- #


#: Map the precomputed cloud-seat-preflight ``level`` / ``verdict`` strings
#: onto a doctor Severity. The CLI's ``_cloud_seat_preflight``
#: (``cli/commands/run.py``) emits each check with a lower-case ``level`` ∈
#: {``pass``, ``warn``, ``error``} and a top-level upper-case ``verdict`` ∈
#: {``PASS``, ``WARN``, ``FAIL``}. We cover BOTH vocabularies (and the
#: synonyms ``ok`` / ``fail`` / ``info``) and resolve case-insensitively via
#: :func:`_nvidia_severity`. CRITICAL: ``error`` MUST land on FAIL — the old
#: map keyed only ``FAIL`` so an ``error`` fell through to INFO and a real
#: cloud-seat preflight failure never escalated the exit code.
_NVIDIA_LEVEL_MAP = {
    "pass": Severity.OK,
    "ok": Severity.OK,
    "warn": Severity.WARN,
    "fail": Severity.FAIL,
    "error": Severity.FAIL,
    "info": Severity.INFO,
}


def _nvidia_severity(level: Any) -> Severity:
    """Resolve an nvidia-preflight ``level``/``verdict`` to a Severity.

    Case-insensitive. An UNRECOGNIZED level maps to WARN (NOT INFO) — a real
    FAIL must never be silently downgraded, and a newly-introduced level
    should surface loudly rather than slip through as informational.
    """
    return _NVIDIA_LEVEL_MAP.get(str(level).strip().lower(), Severity.WARN)


def _cloud_seat_preflight_results(npf: Any) -> List[CheckResult]:
    """Map the precomputed cloud-seat preflight dict's checks to CheckResults."""
    if not isinstance(npf, dict):
        return []
    out: List[CheckResult] = []

    # Surface the top-level verdict so the operator (and the exit code) see
    # the aggregate, mapped through the SAME table as the per-check levels.
    verdict = npf.get("verdict")
    if verdict is not None:
        out.append(
            CheckResult(
                name="cloud_seat_preflight_verdict",
                group=_GROUP,
                severity=_nvidia_severity(verdict),
                summary=f"cloud-seat preflight verdict: {verdict}",
                detail=str(npf.get("note", "")),
                remediation="",
                data={"verdict": str(verdict)},
            )
        )

    checks = npf.get("checks")
    if not isinstance(checks, list):
        return out
    for entry in checks:
        if not isinstance(entry, dict):
            continue
        level = entry.get("level", "")
        severity = _nvidia_severity(level)
        cname = str(entry.get("name", "?"))
        out.append(
            CheckResult(
                name=f"cloud_seat_preflight_{cname}",
                group=_GROUP,
                severity=severity,
                summary=cname,
                detail=str(entry.get("detail", "")),
                remediation="",
                data={"level": str(level), "name": cname},
            )
        )
    return out


# --------------------------------------------------------------------- #
# Ping (opt-in, the only network I/O)
# --------------------------------------------------------------------- #


#: SynthesisProviderError codes that PROVE the server accepted the request and
#: started generating — i.e. the endpoint is reachable + the credentials were
#: accepted. A 1-token ping (``max_tokens`` tiny) almost always trips
#: ``output_truncated`` (``finish_reason='length'``) on a perfectly healthy
#: server, so treating that code as a FAIL inverts the check and fails EVERY
#: working provider. Reachability is all a ping needs to prove.
_PING_REACHABLE_CODES = {"output_truncated"}


def _ping_one(provider: str, record: Dict[str, Any]) -> CheckResult:
    base_url = record.get("base_url")
    model = record.get("model")
    api_key = record.get("api_key")
    ok = CheckResult(
        name=f"provider_ping_{provider}",
        group=_GROUP,
        severity=Severity.OK,
        summary=f"{provider} reachable (auth accepted)",
        detail=f"base_url={base_url}",
        remediation="",
        data={"provider": provider, "base_url": base_url},
    )
    try:
        client = _make_ping_client(
            base_url=base_url,
            model=model,
            api_key=api_key,
            provider_label=provider,
        )
        # A small (non-1) cap reduces truncation, but the real fix is below:
        # truncation is treated as reachability proof, not failure.
        client.chat_completion(
            [{"role": "user", "content": "ping"}],
            max_tokens=8,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 — a ping error is a FAIL, not a crash
        # A SynthesisProviderError whose code proves the request was accepted
        # and generation started (e.g. output_truncated from the tiny
        # max_tokens) means the server is reachable + the key is good → OK.
        # Only genuine auth failures (401/403) + transport/connection errors
        # are real reachability failures.
        code = getattr(exc, "code", None)
        if code in _PING_REACHABLE_CODES:
            return ok
        return CheckResult(
            name=f"provider_ping_{provider}",
            group=_GROUP,
            severity=Severity.FAIL,
            summary=f"{provider} ping failed: {exc}",
            detail=f"{type(exc).__name__}: {exc} (code={code})",
            remediation=(
                f"check {provider} reachability / API key / base_url ({base_url})"
            ),
            data={
                "provider": provider,
                "base_url": base_url,
                "error": str(exc),
                "code": code,
            },
        )
    # A clean (untruncated) response also proves reachability.
    return ok


def _ping_seats(
    seats: List[Any], records: Dict[str, Dict[str, Any]]
) -> List[CheckResult]:
    out: List[CheckResult] = []
    seen: set = set()
    for seat in seats:
        provider = getattr(seat, "effective_provider", None)
        if not provider or provider in seen:
            continue
        seen.add(provider)
        if provider in _ANTHROPIC_FAMILY:
            out.append(
                CheckResult(
                    name=f"provider_ping_{provider}",
                    group=_GROUP,
                    severity=Severity.INFO,
                    summary=(
                        f"{provider} uses the SDK, not pingable here; key presence "
                        "checked separately"
                    ),
                    detail="",
                    remediation="",
                    data={"provider": provider, "pingable": False},
                )
            )
            continue
        if provider not in _PINGABLE:
            continue
        out.append(_ping_one(provider, records.get(provider, {})))
    return out


# --------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------- #


def _run_mode(
    workflow: str, provider_hint: str, ping: bool, npf: Any
) -> List[CheckResult]:
    """Authoritative mode — resolve every seat UNDER the run fanout."""
    results: List[CheckResult] = []
    seats: List[Any] = []
    records: Dict[str, Dict[str, Any]] = {}
    try:
        with run_env.applied_run_env(workflow, provider_hint):
            two_pass = _truthy("COURSEFORGE_TWO_PASS")
            key_table = run_env.load_provider_key_table()
            seats = run_env.resolve_seats(two_pass)
            results.extend(_seat_results(seats, key_table, bare=False))
            records = _capture_records(seats, key_table)
    except Exception as exc:  # noqa: BLE001 — fanout / resolution must not crash
        logger.warning("provider: run-mode resolution failed: %s", exc)
        results.append(_error("provider_run_resolution_error", exc))
        return results

    results.extend(_guarded(lambda: _split_brain(seats), "provider_split_brain_error"))
    results.extend(_guarded(lambda: _licensing_guard(seats), "provider_licensing_error"))
    results.extend(
        _guarded(lambda: _cloud_seat_preflight_results(npf), "provider_cloud_seat_preflight_error")
    )
    if ping:
        results.extend(
            _guarded(lambda: _ping_seats(seats, records), "provider_ping_error")
        )
    return results


def _bare_mode(ping: bool) -> List[CheckResult]:
    """No fanout — seats show class-defaults a real run would flip."""
    results: List[CheckResult] = [
        CheckResult(
            name="provider_bare_caveat",
            group=_GROUP,
            severity=Severity.INFO,
            summary="provider seats shown off the CURRENT env (no run fanout applied)",
            detail=_BARE_CAVEAT,
            remediation="",
            data={},
        )
    ]
    seats: List[Any] = []
    records: Dict[str, Dict[str, Any]] = {}
    try:
        two_pass = _truthy("COURSEFORGE_TWO_PASS")
        key_table = run_env.load_provider_key_table()
        seats = run_env.resolve_seats(two_pass)
        results.extend(_seat_results(seats, key_table, bare=True))
        records = _capture_records(seats, key_table)
    except Exception as exc:  # noqa: BLE001 — resolution must not crash
        logger.warning("provider: bare-mode resolution failed: %s", exc)
        results.append(_error("provider_bare_resolution_error", exc))
        return results

    # Split-brain / licensing are authoritative only under the fanout — skip
    # them as gating in bare mode.
    if ping:
        results.extend(
            _guarded(lambda: _ping_seats(seats, records), "provider_ping_error")
        )
    return results


# --------------------------------------------------------------------- #
# Entry point + registration
# --------------------------------------------------------------------- #


def provider_checks(ctx: CheckContext) -> List[CheckResult]:
    """Provider / seat preflight (group ``provider``). NEVER raises.

    Dispatches RUN vs BARE on ``ctx.run_config['workflow']``. Every
    sub-section is isolated so one failing probe is its own result and the
    rest still run; the whole call is belt-and-suspenders wrapped.
    """
    try:
        raw = getattr(ctx, "run_config", None)
        rc = raw if isinstance(raw, dict) else {}
        workflow = rc.get("workflow")
        provider_hint = str(rc.get("provider_hint") or "")
        ping = bool(rc.get("ping"))
        # New key first, legacy ``nvidia_preflight`` fallback (old post-mortem
        # sidecars + plan dicts still carry the legacy key for one release).
        npf = rc.get("cloud_seat_preflight")
        if npf is None:
            npf = rc.get("nvidia_preflight")
        if workflow:
            return _run_mode(str(workflow), provider_hint, ping, npf)
        return _bare_mode(ping)
    except Exception as exc:  # noqa: BLE001 — a doctor must never crash
        logger.warning("provider: provider_checks raised: %s", exc)
        return [_error("provider_error", exc)]


def register_provider_checks() -> None:
    """Register :func:`provider_checks` under ``group='provider'``.

    Called explicitly from the CLI bootstrap — NEVER at import time.
    """
    register("provider", provider_checks)


__all__ = ["provider_checks", "register_provider_checks"]

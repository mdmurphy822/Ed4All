"""vLLM SEAT-topology diagnostics — the ``seat`` doctor group (P1-2 / P1-3).

Deployment-specific preflight for a vLLM-seat host: validate the seat registries
(``ED4ALL_SEAT_BASE_URLS`` / ``ED4ALL_VLLM_CONTAINERS`` / ``ED4ALL_SEAT_LAUNCH_SPECS``),
that seat base URLs are loopback-only and port-distinct, that each registered
container exists (``docker ps -a``), per-seat ``/v1/models`` liveness (INFO —
down at rest is not an error, e.g. between GPU-lifecycle phases), and that each
launch-spec path exists + is executable. Plus the assistant-seat sub-check: the
assistant base URL is loopback, its seat-priority walk resolves through the
registry, and the one seat the assistant may start has a registry entry + a
launch spec.

This group is NOT in the bare default set — seat topology is deployment-specific.
``cli/commands/doctor.py`` opts it in via ``-g seat`` / ``--run`` / a configured
seat registry (mirroring the ``provider`` group's opt-in pattern).

Design contract (inherited from :mod:`lib.diagnostics.core`): a doctor that
crashes is worse than no doctor. Every sub-check is isolated by
:func:`_run_subcheck`, so one failing probe becomes its own WARN result and the
rest still run; :func:`seat_checks` NEVER raises. It reuses the FAIL-SOFT
lifecycle parsers/resolvers from :mod:`lib.vllm_container_lifecycle`; it does NOT
import :mod:`lib.assistant` (off-limits + avoids a diagnostics→assistant
dependency) — the assistant envs are re-read directly with their documented
defaults. Registration is explicit (:func:`register_seat_checks`), never at
import time.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from lib.diagnostics.core import CheckContext, CheckResult, Severity, register

logger = logging.getLogger(__name__)

#: Bounded timeout (seconds) for the ``docker ps -a`` container-existence query.
_DOCKER_PS_TIMEOUT_SECONDS = 8.0

#: Assistant seat envs + documented defaults (re-read directly — NOT imported
#: from lib.assistant, which is off-limits). Kept in sync with
#: ``lib/assistant/client.py`` DEFAULT_BASE_URL / ASSISTANT_SEAT_NAME /
#: DEFAULT_SEAT_PRIORITY.
_ENV_ASSISTANT_BASE_URL = "ED4ALL_ASSISTANT_BASE_URL"
_DEFAULT_ASSISTANT_BASE_URL = "http://localhost:8004/v1"
_ENV_ASSISTANT_SEAT = "ED4ALL_ASSISTANT_SEAT"
_DEFAULT_ASSISTANT_SEAT = "spark-nano"
_ENV_ASSISTANT_SEAT_PRIORITY = "ED4ALL_ASSISTANT_SEAT_PRIORITY"
_DEFAULT_ASSISTANT_SEAT_PRIORITY = "spark-super,spark-nano"


def _run_subcheck(
    fn: Callable[[CheckContext], List[CheckResult]], ctx: CheckContext
) -> List[CheckResult]:
    """Run one sub-check, isolating a raise into its own WARN result.

    Mirrors :func:`lib.diagnostics.environment._run_subcheck` — a sub-check that
    forgets to wrap something degrades to a single WARN (named ``seat_<fn>``) and
    the remaining sub-checks still run.
    """
    try:
        return fn(ctx) or []
    except Exception as exc:  # noqa: BLE001 — a doctor sub-check must never crash
        name = getattr(fn, "__name__", "subcheck").lstrip("_")
        logger.warning("diagnostics.seat: %s raised: %s", name, exc)
        return [
            CheckResult(
                name=f"seat_{name}",
                group="seat",
                severity=Severity.WARN,
                summary=f"seat check '{name}' errored: {exc}",
                detail=f"{type(exc).__name__}: {exc}",
                remediation="this is a doctor bug — the sub-check should never raise",
                data={"error": str(exc), "error_type": type(exc).__name__},
            )
        ]


# --------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------- #


def _is_loopback(base_url: str) -> bool:
    """True iff ``base_url``'s host is loopback (localhost / 127.0.0.0/8 / ::1).

    Mirrors the grounded-answer / assistant loopback guard. A URL that will not
    parse is treated as NOT loopback (surfaced as a WARN) rather than raising.
    """
    try:
        host = (urlsplit(str(base_url)).hostname or "").strip().lower()
    except Exception:  # noqa: BLE001 — a malformed URL is not loopback
        return False
    if not host:
        return False
    return host == "localhost" or host == "::1" or host.startswith("127.")


def _audit_registry_tokens(raw: Optional[str], *, launch_spec: bool) -> Tuple[int, List[str]]:
    """Re-derive ``(valid_token_count, malformed_tokens)`` from a raw registry.

    The lifecycle parsers are FAIL-SOFT (they drop malformed tokens with a
    one-time warning and return only the valid pairs), so this re-splits the raw
    value with the SAME rules to surface which tokens were skipped. ``launch_spec``
    switches to the launch-spec grammar (``;`` sep when present, split on the
    FIRST ``=`` only). Never raises.
    """
    if not raw or not str(raw).strip():
        return 0, []
    sep = (";" if launch_spec and ";" in str(raw) else ",")
    valid = 0
    bad: List[str] = []
    for token in str(raw).split(sep):
        token = token.strip()
        if not token:
            continue
        if launch_spec:
            ok = "=" in token and all(part.strip() for part in token.split("=", 1))
        else:
            ok = token.count("=") == 1 and all(part.strip() for part in token.split("=", 1))
        if ok:
            valid += 1
        else:
            bad.append(token)
    return valid, bad


# --------------------------------------------------------------------- #
# 1. Registry parse validity
# --------------------------------------------------------------------- #


def _check_registry_parse(ctx: CheckContext) -> List[CheckResult]:
    """Validate the three seat registries; surface skipped malformed tokens."""
    from lib.vllm_container_lifecycle import (
        ENV_SEAT_BASE_URLS,
        ENV_SEAT_LAUNCH_SPECS,
        ENV_VLLM_CONTAINERS,
    )

    results: List[CheckResult] = []
    specs = [
        (ENV_SEAT_BASE_URLS, "seat_name=base_url", False),
        (ENV_VLLM_CONTAINERS, "base_url=container", False),
        (ENV_SEAT_LAUNCH_SPECS, "seat_name=<launch spec>", True),
    ]
    for env_name, shape, is_launch in specs:
        raw = os.environ.get(env_name)
        valid, bad = _audit_registry_tokens(raw, launch_spec=is_launch)
        if not raw or not str(raw).strip():
            results.append(
                CheckResult(
                    name=f"seat_registry_{env_name.lower()}",
                    group="seat",
                    severity=Severity.INFO,
                    summary=f"{env_name} is not configured",
                    data={"env": env_name, "configured": False},
                )
            )
        elif bad:
            results.append(
                CheckResult(
                    name=f"seat_registry_{env_name.lower()}",
                    group="seat",
                    severity=Severity.WARN,
                    summary=(
                        f"{env_name} has {len(bad)} malformed token(s) skipped "
                        f"(kept {valid} valid): {bad}"
                    ),
                    detail=f"expected '{shape}' per token",
                    remediation=f"fix the malformed {env_name} token(s): {bad}",
                    data={"env": env_name, "valid": valid, "malformed": bad},
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"seat_registry_{env_name.lower()}",
                    group="seat",
                    severity=Severity.OK,
                    summary=f"{env_name} parses cleanly ({valid} pair(s))",
                    data={"env": env_name, "valid": valid},
                )
            )
    return results


# --------------------------------------------------------------------- #
# 2. Loopback-only + duplicate-port
# --------------------------------------------------------------------- #


def _all_seat_base_urls() -> Dict[str, str]:
    """Return ``{label: base_url}`` for every seat/container base URL.

    Union of the ``seat_name=base_url`` registry (keyed by seat name) and any
    container base URL not already covered (keyed by the base URL itself).
    """
    from lib.vllm_container_lifecycle import parse_container_registry, parse_seat_registry

    out: Dict[str, str] = {}
    seats = parse_seat_registry()
    for name, burl in seats.items():
        out[name] = burl
    seen = set(seats.values())
    for burl in parse_container_registry():
        if burl not in seen:
            out[burl] = burl
    return out


def _check_loopback_and_ports(ctx: CheckContext) -> List[CheckResult]:
    """Loopback-only + duplicate-port detection over the seat base URLs."""
    base_urls = _all_seat_base_urls()
    if not base_urls:
        return [
            CheckResult(
                name="seat_loopback",
                group="seat",
                severity=Severity.INFO,
                summary="no seat base URLs configured (loopback/port check skipped)",
                data={"seats": {}},
            )
        ]

    results: List[CheckResult] = []
    non_loopback = {lbl: u for lbl, u in base_urls.items() if not _is_loopback(u)}
    if non_loopback:
        results.append(
            CheckResult(
                name="seat_loopback",
                group="seat",
                severity=Severity.WARN,
                summary=f"{len(non_loopback)} seat base URL(s) are NOT loopback: {non_loopback}",
                remediation=(
                    "seats must serve on loopback (localhost/127.0.0.1/::1); a "
                    "non-loopback seat exposes an unauthenticated LLM endpoint"
                ),
                data={"non_loopback": non_loopback},
            )
        )
    else:
        results.append(
            CheckResult(
                name="seat_loopback",
                group="seat",
                severity=Severity.OK,
                summary=f"all {len(base_urls)} seat base URL(s) are loopback",
                data={"seats": base_urls},
            )
        )

    # Duplicate-port: two distinct seats sharing the same host:port.
    by_netloc: Dict[str, List[str]] = {}
    for label, url in base_urls.items():
        try:
            parts = urlsplit(url)
            netloc = f"{(parts.hostname or '').lower()}:{parts.port}"
        except Exception:  # noqa: BLE001 — a malformed URL is skipped for the dup check
            continue
        by_netloc.setdefault(netloc, []).append(label)
    collisions = {nl: labels for nl, labels in by_netloc.items() if len(labels) > 1}
    if collisions:
        results.append(
            CheckResult(
                name="seat_port_collision",
                group="seat",
                severity=Severity.WARN,
                summary=f"seat base URLs collide on {len(collisions)} host:port: {collisions}",
                remediation="give each seat a distinct port",
                data={"collisions": collisions},
            )
        )
    else:
        results.append(
            CheckResult(
                name="seat_port_collision",
                group="seat",
                severity=Severity.OK,
                summary="no duplicate seat host:port bindings",
                data={"netlocs": {nl: labels for nl, labels in by_netloc.items()}},
            )
        )
    return results


# --------------------------------------------------------------------- #
# 3. Container existence (docker ps -a)
# --------------------------------------------------------------------- #


def _docker_container_names() -> Optional[set]:
    """Return the set of ALL container names (``docker ps -a``), or None.

    Bounded subprocess, NEVER raises. ``None`` means docker is unavailable
    (absent CLI / no perms / error) — the caller reports INFO, not FAIL. On a
    permission error the query is retried once through ``sg docker -c`` (the
    Spark docker-group wrapping), mirroring ``vllm_container_lifecycle._run_docker``.
    """
    cmd = ["docker", "ps", "-a", "--format", "{{.Names}}"]

    def _run(argv) -> Optional[Tuple[int, str, str]]:
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=_DOCKER_PS_TIMEOUT_SECONDS
            )
            return proc.returncode, proc.stdout or "", proc.stderr or ""
        except Exception as exc:  # noqa: BLE001 — docker absent / timeout / etc.
            logger.debug("diagnostics.seat: %r failed: %s", " ".join(argv), exc)
            return None

    res = _run(cmd)
    if res is not None and res[0] == 0:
        return {ln.strip() for ln in res[1].splitlines() if ln.strip()}
    # Permission-shaped failure → retry through the docker-group wrapper.
    if res is not None:
        if "permission denied" in res[2].lower() or "dial unix" in res[2].lower():
            wrapped = _run(["sg", "docker", "-c", " ".join(cmd)])
            if wrapped is not None and wrapped[0] == 0:
                return {ln.strip() for ln in wrapped[1].splitlines() if ln.strip()}
    return None


def _check_container_existence(ctx: CheckContext) -> List[CheckResult]:
    """Each registered container exists in ``docker ps -a`` (docker absent → INFO)."""
    from lib.vllm_container_lifecycle import parse_container_registry

    containers = parse_container_registry()  # {base_url: container}
    if not containers:
        return [
            CheckResult(
                name="seat_containers",
                group="seat",
                severity=Severity.INFO,
                summary="no container registry configured (container check skipped)",
                data={"containers": {}},
            )
        ]

    names = _docker_container_names()
    if names is None:
        return [
            CheckResult(
                name="seat_containers",
                group="seat",
                severity=Severity.INFO,
                summary="docker unavailable — container existence not checked",
                detail="docker CLI absent / no perms; this is not an error at rest",
                data={"containers": containers, "docker_available": False},
            )
        ]

    missing = {burl: c for burl, c in containers.items() if c not in names}
    if missing:
        return [
            CheckResult(
                name="seat_containers",
                group="seat",
                severity=Severity.WARN,
                summary=f"{len(missing)} registered container(s) do not exist: {missing}",
                remediation=(
                    "create the missing container(s) (their launch spec) or fix "
                    "the ED4ALL_VLLM_CONTAINERS mapping"
                ),
                data={"missing": missing, "present_count": len(containers) - len(missing)},
            )
        ]
    return [
        CheckResult(
            name="seat_containers",
            group="seat",
            severity=Severity.OK,
            summary=f"all {len(containers)} registered container(s) exist",
            data={"containers": containers},
        )
    ]


# --------------------------------------------------------------------- #
# 4. Per-seat /v1/models liveness (INFO — down at rest is not an error)
# --------------------------------------------------------------------- #


def _check_seat_liveness(ctx: CheckContext) -> List[CheckResult]:
    """Per-seat ``/v1/models`` liveness — INFO live/down (down at rest is fine)."""
    from lib.diagnostics.run_env import probe_v1_models

    base_urls = _all_seat_base_urls()
    if not base_urls:
        return []
    results: List[CheckResult] = []
    for label, url in base_urls.items():
        live, model_ids, error = probe_v1_models(url)
        results.append(
            CheckResult(
                name=f"seat_live_{label}",
                group="seat",
                severity=Severity.INFO,
                summary=(
                    f"seat {label} ({url}) "
                    + (f"live — {len(model_ids)} model(s): {model_ids}" if live else f"down ({error})")
                ),
                detail="down at rest is not an error (e.g. between GPU-lifecycle phases)",
                data={"seat": label, "base_url": url, "live": live, "served_model_ids": model_ids},
            )
        )
    return results


# --------------------------------------------------------------------- #
# 5. Launch-spec paths exist + executable
# --------------------------------------------------------------------- #


def _spec_is_path(spec: str) -> bool:
    """Heuristic: is ``spec`` a launch SCRIPT path (vs a full shell command)?

    A bare script path has no spaces and contains a ``/`` (absolute or relative);
    a full ``docker run …`` command has spaces. Only path-shaped specs are
    filesystem-checked; command specs are reported INFO (not path-checked).
    """
    s = spec.strip()
    return bool(s) and " " not in s and "/" in s


def _check_launch_specs(ctx: CheckContext) -> List[CheckResult]:
    """Each launch-spec PATH exists + is executable (command specs → INFO)."""
    from lib.vllm_container_lifecycle import parse_seat_launch_specs

    specs = parse_seat_launch_specs()  # {seat_name: spec}
    if not specs:
        return [
            CheckResult(
                name="seat_launch_specs",
                group="seat",
                severity=Severity.INFO,
                summary="no launch specs configured (a seat cannot self-heal a mode collapse)",
                data={"specs": {}},
            )
        ]

    results: List[CheckResult] = []
    for seat, spec in specs.items():
        if not _spec_is_path(spec):
            results.append(
                CheckResult(
                    name=f"seat_launch_{seat}",
                    group="seat",
                    severity=Severity.INFO,
                    summary=f"seat {seat} launch spec is a command (not path-checked): {spec!r}",
                    data={"seat": seat, "spec": spec, "kind": "command"},
                )
            )
            continue
        exists = os.path.isfile(spec)
        executable = exists and os.access(spec, os.X_OK)
        if executable:
            results.append(
                CheckResult(
                    name=f"seat_launch_{seat}",
                    group="seat",
                    severity=Severity.OK,
                    summary=f"seat {seat} launch script exists + is executable: {spec}",
                    data={"seat": seat, "spec": spec, "exists": True, "executable": True},
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f"seat_launch_{seat}",
                    group="seat",
                    severity=Severity.WARN,
                    summary=(
                        f"seat {seat} launch script {spec} "
                        + ("is not executable" if exists else "does not exist")
                    ),
                    remediation=(
                        f"chmod +x {spec}" if exists else f"create {spec} or fix the spec"
                    ),
                    data={"seat": seat, "spec": spec, "exists": exists, "executable": executable},
                )
            )
    return results


# --------------------------------------------------------------------- #
# 6. Assistant seat (P1-3) — envs re-read directly, NOT via lib.assistant
# --------------------------------------------------------------------- #


def _check_assistant_seat(ctx: CheckContext) -> List[CheckResult]:
    """Assistant base-URL loopback + priority-walk resolution + startable seat.

    Re-reads the assistant envs directly (lib.assistant is off-limits) and uses
    the lifecycle resolvers to check the registry. NEVER raises.
    """
    from lib.vllm_container_lifecycle import (
        resolve_seat_base_url,
        resolve_seat_launch_spec,
    )

    results: List[CheckResult] = []

    # (a) assistant base URL loopback.
    base_url = (os.environ.get(_ENV_ASSISTANT_BASE_URL) or _DEFAULT_ASSISTANT_BASE_URL).strip()
    if _is_loopback(base_url):
        results.append(
            CheckResult(
                name="assistant_base_url_loopback",
                group="seat",
                severity=Severity.OK,
                summary=f"assistant base URL is loopback ({base_url})",
                data={"base_url": base_url},
            )
        )
    else:
        results.append(
            CheckResult(
                name="assistant_base_url_loopback",
                group="seat",
                severity=Severity.WARN,
                summary=f"assistant base URL is NOT loopback ({base_url})",
                remediation=f"set {_ENV_ASSISTANT_BASE_URL} to a loopback host",
                data={"base_url": base_url},
            )
        )

    # (b) seat-priority walk resolves through ED4ALL_SEAT_BASE_URLS.
    raw_priority = (
        os.environ.get(_ENV_ASSISTANT_SEAT_PRIORITY) or _DEFAULT_ASSISTANT_SEAT_PRIORITY
    )
    priority = [p.strip() for p in raw_priority.split(",") if p.strip()]
    resolved = {p: resolve_seat_base_url(p) for p in priority}
    unresolved = [p for p, u in resolved.items() if not u]
    if not priority:
        pass  # empty priority is degenerate but not this check's concern
    elif not any(resolved.values()):
        results.append(
            CheckResult(
                name="assistant_seat_priority",
                group="seat",
                severity=Severity.WARN,
                summary=(
                    f"NONE of the assistant seat-priority names {priority} resolve "
                    f"through {('ED4ALL_SEAT_BASE_URLS')}"
                ),
                remediation="add at least one priority seat to ED4ALL_SEAT_BASE_URLS",
                data={"priority": priority, "resolved": resolved},
            )
        )
    elif unresolved:
        results.append(
            CheckResult(
                name="assistant_seat_priority",
                group="seat",
                severity=Severity.INFO,
                summary=(
                    f"assistant seat-priority: {len(priority) - len(unresolved)}/"
                    f"{len(priority)} resolve; unresolved (fallback-only): {unresolved}"
                ),
                data={"priority": priority, "resolved": resolved},
            )
        )
    else:
        results.append(
            CheckResult(
                name="assistant_seat_priority",
                group="seat",
                severity=Severity.OK,
                summary=f"all assistant seat-priority names {priority} resolve",
                data={"priority": priority, "resolved": resolved},
            )
        )

    # (c) the ONE seat the assistant may start needs a registry entry + launch spec.
    own_seat = (os.environ.get(_ENV_ASSISTANT_SEAT) or _DEFAULT_ASSISTANT_SEAT).strip()
    seat_url = resolve_seat_base_url(own_seat)
    seat_spec = resolve_seat_launch_spec(own_seat)
    if seat_url and seat_spec:
        results.append(
            CheckResult(
                name="assistant_seat_startable",
                group="seat",
                severity=Severity.OK,
                summary=f"assistant seat {own_seat!r} has a registry entry + launch spec",
                data={"seat": own_seat, "base_url": seat_url, "launch_spec": seat_spec},
            )
        )
    else:
        problems = []
        if not seat_url:
            problems.append("no ED4ALL_SEAT_BASE_URLS entry")
        if not seat_spec:
            problems.append("no ED4ALL_SEAT_LAUNCH_SPECS entry (cannot autostart / self-heal)")
        results.append(
            CheckResult(
                name="assistant_seat_startable",
                group="seat",
                severity=Severity.WARN,
                summary=f"assistant seat {own_seat!r}: {', '.join(problems)}",
                remediation=(
                    f"register {own_seat} in ED4ALL_SEAT_BASE_URLS + "
                    "ED4ALL_SEAT_LAUNCH_SPECS (the only seat the assistant may start)"
                ),
                data={"seat": own_seat, "base_url": seat_url, "launch_spec": seat_spec},
            )
        )
    return results


# --------------------------------------------------------------------- #
# Entry point + registration
# --------------------------------------------------------------------- #

#: Ordered sub-checks — each isolated by :func:`_run_subcheck`.
_SUBCHECKS: List[Callable[[CheckContext], List[CheckResult]]] = [
    _check_registry_parse,
    _check_loopback_and_ports,
    _check_container_existence,
    _check_seat_liveness,
    _check_launch_specs,
    _check_assistant_seat,
]


def seat_checks(ctx: CheckContext) -> List[CheckResult]:
    """Run every seat sub-check, flattening the results (NEVER raises)."""
    results: List[CheckResult] = []
    for fn in _SUBCHECKS:
        results.extend(_run_subcheck(fn, ctx))
    return results


def register_seat_checks() -> None:
    """Register :func:`seat_checks` under ``group="seat"`` (explicit — never at import)."""
    register("seat", seat_checks)


__all__ = ["seat_checks", "register_seat_checks"]

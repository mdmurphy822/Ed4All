"""Per-model vLLM CONTAINER lifecycle lease helper (Ed4All venv).

GOAL: **do NOT keep models resident** — a model
loads, does its job, and STOPS serving when done. This is the container-level
extension of :mod:`lib.gpu_lifecycle`: where that module hands the GPU card
between ollama / torch seats at a phase boundary (``keep_alive:0`` +
``empty_cache``), THIS module goes one level further and START/STOPs the vLLM
docker CONTAINERS themselves, so a hosted large-model seat that is not in use
holds ZERO VRAM (the container is stopped, not merely idle-resident). The
PIPELINE orchestrates this per-model lifecycle.

The seat map is data-driven (:data:`ENV_VLLM_CONTAINERS`): a comma-separated
``base_url=container`` registry, so a new vLLM seat is a registry entry, never
a code change (mirrors the OpenAI-compatible endpoint-registry house style).

The three verbs:

* :func:`ensure_serving` — probe ``GET {base_url}/v1/models``; if the seat is
  already ready, return ``0.0`` (no container start). Else ``docker start`` the
  mapped container and poll ``/v1/models`` until it answers or the timeout
  expires, returning the measured **load_seconds** (the load-time metering half
  of task #10). A timeout warns + returns ``None`` — the caller proceeds and
  its own HTTP call fails loudly rather than this helper masking the problem.
* :func:`release` / :func:`release_all` — ``docker stop`` one / every registered
  container so the seat frees its VRAM. ``release_all`` is the one wired
  integration point (workflow end = the unambiguous "done serving" boundary).
* :func:`record_load_event` — append a load-metering JSONL row to
  ``<run_dir>/model_load_events.jsonl`` (mirrors how ``vram_trajectory.jsonl``
  rows are written by :func:`lib.llm.vram_doctor.write_trajectory_row`).

Everything is best-effort / NEVER-raises (mirrors ``gpu_lifecycle`` /
``vram_reclaim``): docker missing, no docker-group perms, a wedged container, or
a probe timeout logs a warning and returns a sentinel (``None`` / ``False`` /
``0``) — a lifecycle failure must NEVER crash the phase it is observing. The
docker CLI is invoked directly (``docker ...``); on a permission error the call
is retried once through ``sg docker -c "docker ..."`` (the Spark docker-group
wrapping), so a box that needs the group-wrap still works fail-soft.

Flag: ``ED4ALL_VLLM_CONTAINER_LIFECYCLE`` — **default OFF**. When unset/falsey
EVERY function is a no-op (returns the off-sentinel WITHOUT touching docker or
the network), so a flag-off pipeline is byte-identical control flow. Parse:
truthy set ``{1, true, yes, on}`` (case-insensitive), anything else → off.

Follow-up (NOT built here): per-PHASE ensure/release seams — a phase-to-seat
need-map so a phase START calls :func:`ensure_serving` for the seat it needs and
a phase boundary releases the seats the NEXT phase does not. v1 wires only the
workflow-end :func:`release_all` (the single unambiguous done-serving boundary).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Union

from lib.file_lock import file_lock
from lib.paths import STATE_PATH

logger = logging.getLogger(__name__)

#: Master switch — **default OFF**. Only the truthy tokens (case-insensitive)
#: enable; unset / blank / falsey / garbage → off (a no-op, byte-identical
#: pipeline). Documented in root CLAUDE.md + docs/operations/behavior-flags.md.
ENV_VLLM_CONTAINER_LIFECYCLE = "ED4ALL_VLLM_CONTAINER_LIFECYCLE"

#: Comma-separated ``base_url=container`` registry, e.g.
#: ``http://localhost:8000=vllm-omni,http://localhost:8001=vllm-embed``. The
#: single source of truth for the seat map — a new vLLM seat is a registry
#: entry, never a subclass.
ENV_VLLM_CONTAINERS = "ED4ALL_VLLM_CONTAINERS"

#: Declarative per-phase SEAT SCHEDULE master switch — **default OFF**. When on,
#: ``MCP.core.workflow_runner`` enforces each phase's YAML-declared ``seats:``
#: annotation at the phase boundary (stop no-longer-needed seats, start + WAIT +
#: COHERENCE-PROBE newly-needed ones). Off → no seat enforcement at all (the
#: seat-schedule code path is a pure no-op, byte-identical). Truthy tokens only.
ENV_SEAT_SCHEDULE = "ED4ALL_SEAT_SCHEDULE"

#: Comma-separated ``seat_name=base_url`` registry mapping the LOGICAL seat names
#: used in the ``config/workflows.yaml`` ``seats:`` phase annotations (e.g.
#: ``spark-super``, ``spark-glm``, ``spark-qwen``) to the vLLM seat base URLs,
#: e.g. ``spark-super=http://localhost:8001,spark-glm=http://localhost:8002``.
#: The base_url then resolves to a docker container via
#: :data:`ENV_VLLM_CONTAINERS` (which stays the single container registry). A
#: A required seat name that does not resolve here is a CONFIG gap and the
#: workflow readiness boundary fails loudly before provider dispatch.
ENV_SEAT_BASE_URLS = "ED4ALL_SEAT_BASE_URLS"

#: Data-driven per-seat LAUNCH-SPEC registry (``seat_name=<launch spec>``) that
#: lets the seat schedule COLD-RECREATE a container (``docker rm -f`` + relaunch),
#: not merely warm ``docker start`` a pre-existing one — the self-heal path for a
#: seat that mode-collapses on a warm restart. The RHS spec is either an absolute
#: path to an ops-owned launch SCRIPT (``spark-super=/opt/seats/launch-super.sh``,
#: the RECOMMENDED form — keeps the long spec-decode vLLM command out of the env
#: and lets ops own the exact argv) OR a full shell command line. Tokens are
#: separated by ``;`` (preferred — a launch command may itself contain ``,``) or
#: by ``,`` when no ``;`` is present; each token is split on its FIRST ``=`` only,
#: so the spec may itself contain ``=`` (e.g. ``--env FOO=bar``). Malformed tokens
#: are SKIPPED with a one-time warning (mirrors the other registries). A seat with
#: NO launch spec can still warm-start but CANNOT self-heal a mode collapse.
ENV_SEAT_LAUNCH_SPECS = "ED4ALL_SEAT_LAUNCH_SPECS"

#: Env-tunable CEILING (seconds) on the seat load-wait poll loop — the maximum a
#: ``start_seat`` / ``recreate_seat`` health-check poll will wait for a seat to
#: become live AND coherent before giving up. It is a CEILING, not a fixed sleep:
#: the poll returns the instant the seat is actually ready, so a fast seat
#: proceeds in seconds and only a slow / stuck one approaches this bound. Default
#: **1200s (20 min)** — a 120B NVFP4 seat's cold load + swap is ~8-10 min observed
#: and the old 600s was cutting it close. Parse-with-fallback: a positive
#: int/float wins; garbage / non-positive → the 1200s default.
ENV_SEAT_LOAD_TIMEOUT_SECONDS = "ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS"

#: Default seat load-wait ceiling (seconds) — see
#: :data:`ENV_SEAT_LOAD_TIMEOUT_SECONDS`.
_DEFAULT_SEAT_LOAD_TIMEOUT_SECONDS = 1200.0

#: Env-tunable COUNT of content-coherence probe attempts once a seat is LIVE.
#: Coherence is a DIFFERENT gate from liveness: a mode-collapsed seat is
#: live-but-incoherent and detectable in SECONDS, so it must NOT ride out the
#: ``ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS`` liveness ceiling. After ``/v1/models``
#: answers, the coherence probe is retried at most this many times (short
#: interval between); if still incoherent the schedule cold-recreates IMMEDIATELY
#: (a live-but-incoherent seat self-heals in ~30-45s + reload, not ~20 min).
#: Default **3**. Parse-with-fallback: a positive int wins; garbage / non-positive
#: → 3.
ENV_SEAT_COHERENCE_ATTEMPTS = "ED4ALL_SEAT_COHERENCE_ATTEMPTS"

#: Default number of content-coherence probe attempts — see
#: :data:`ENV_SEAT_COHERENCE_ATTEMPTS`.
_DEFAULT_SEAT_COHERENCE_ATTEMPTS = 3

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Per-request timeout (seconds) for the ``/v1/models`` readiness probe. A probe
#: is a cheap liveness GET — short so a wedged seat doesn't stall the poll loop
#: (the OUTER ``timeout_seconds`` bounds the whole start+poll).
_PROBE_TIMEOUT_SECONDS = 5.0

#: Poll interval (seconds) between ``/v1/models`` readiness probes while waiting
#: for a freshly ``docker start``-ed container to come up.
_POLL_INTERVAL_SECONDS = 3.0

#: Timeout (seconds) for a single ``docker start`` / ``docker stop`` CLI call.
_DOCKER_CLI_TIMEOUT_SECONDS = 60.0

#: Only warn once per process about a garbage / unset registry so a flag-on run
#: with a typo'd registry doesn't spam the log every ``ensure_serving`` call.
_REGISTRY_WARNED = False

#: Sibling of :data:`_REGISTRY_WARNED` for the ``seat_name=base_url`` registry.
_SEAT_REGISTRY_WARNED = False

#: Sibling of :data:`_REGISTRY_WARNED` for the ``seat_name=launch_spec`` registry.
_LAUNCH_SPEC_WARNED = False

#: Interval (seconds) between the BOUNDED content-coherence probe attempts
#: (:func:`_coherence_check`). Short (~8s) so a just-loaded seat's first coherent
#: completion is picked up promptly; the number of attempts (not this interval)
#: bounds the coherence stage (see :data:`ENV_SEAT_COHERENCE_ATTEMPTS`).
_COHERENCE_POLL_INTERVAL_SECONDS = 8.0

#: How often (seconds) the load-wait poll loops emit a "still loading, N s
#: elapsed" progress line, so a human/operator can see a long wait is advancing
#: (not hung) without spamming the log every poll tick.
_PROGRESS_LOG_INTERVAL_SECONDS = 30.0

#: Per-request HTTP timeout (seconds) for the content-level COHERENCE PROBE
#: (``POST /v1/chat/completions``). Longer than the ``/v1/models`` liveness
#: probe: a just-warmed seat's first token can be slow, but a wedged / mode-
#: collapsed seat still resolves within this bound.
_COHERENCE_TIMEOUT_SECONDS = 45.0

#: The fixed prompt the coherence probe POSTs. Deliberately trivial: any coherent
#: seat can echo it; a mode-collapsed seat emits degenerate soup / null content.
_COHERENCE_PROMPT = (
    'Return exactly one JSON object matching this schema and no prose: '
    '{"seat_ok":true}'
)

#: Max completion tokens for the coherence probe (small — we only need enough to
#: tell coherent text from degenerate output).
_COHERENCE_MAX_TOKENS = 128


def resolve_vllm_container_lifecycle_mode(value: Optional[str] = None) -> bool:
    """Resolve whether the vLLM container-lifecycle lease is enabled.

    Resolution chain: explicit ``value`` arg → ``ED4ALL_VLLM_CONTAINER_LIFECYCLE``
    env → **default OFF**. Parse-with-fallback: only the truthy tokens
    (``1`` / ``true`` / ``yes`` / ``on``, case-insensitive) enable; anything
    else — unset / blank / falsey / garbage — is off (the house default-off
    convention; the complement of :func:`lib.gpu_lifecycle.
    resolve_gpu_lifecycle_mode`, which is default-ON).
    """
    raw = value if value is not None else os.environ.get(ENV_VLLM_CONTAINER_LIFECYCLE)
    if raw is None or not str(raw).strip():
        return False
    return str(raw).strip().lower() in _TRUTHY


def parse_container_registry(value: Optional[str] = None) -> Dict[str, str]:
    """Parse the ``base_url=container`` seat registry into ``{base_url: container}``.

    Reads ``ED4ALL_VLLM_CONTAINERS`` (or the explicit ``value``). Each
    comma-separated token is ``base_url=container``; the ``base_url`` is
    normalized by stripping a trailing ``/`` so a caller's ``.../v1`` vs
    ``.../v1/`` both match. Fail-soft: unset / blank returns ``{}``; a token
    without exactly one ``=`` (or an empty side) is SKIPPED with a one-time
    warning — a partly-garbage registry still yields its valid pairs rather than
    raising.
    """
    global _REGISTRY_WARNED
    raw = value if value is not None else os.environ.get(ENV_VLLM_CONTAINERS)
    if raw is None or not str(raw).strip():
        return {}

    registry: Dict[str, str] = {}
    bad_tokens = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if token.count("=") != 1:
            bad_tokens.append(token)
            continue
        base_url, container = token.split("=", 1)
        base_url = base_url.strip().rstrip("/")
        container = container.strip()
        if not base_url or not container:
            bad_tokens.append(token)
            continue
        registry[base_url] = container

    if bad_tokens and not _REGISTRY_WARNED:
        logger.warning(
            "vllm_container_lifecycle: ignored %d malformed %s token(s) %r "
            "(expected 'base_url=container'); using %d valid pair(s).",
            len(bad_tokens), ENV_VLLM_CONTAINERS, bad_tokens, len(registry),
        )
        _REGISTRY_WARNED = True
    return registry


def _lookup_container(base_url: str) -> Optional[str]:
    """Return the container mapped to ``base_url`` (trailing-``/`` tolerant), or None."""
    registry = parse_container_registry()
    if not registry:
        return None
    return registry.get(str(base_url).rstrip("/"))


def _probe_ready(base_url: str, *, timeout: float = _PROBE_TIMEOUT_SECONDS) -> bool:
    """Return True iff ``GET {base_url}/v1/models`` answers 2xx. Never raises."""
    url = f"{str(base_url).rstrip('/')}/v1/models"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
            return 200 <= int(code) < 300
    except Exception as exc:  # noqa: BLE001 — server down / refused / bad URL / timeout
        logger.debug("vllm_container_lifecycle: probe %s not ready (%s).", url, exc)
        return False


def _run_docker(args, *, timeout: float = _DOCKER_CLI_TIMEOUT_SECONDS) -> bool:
    """Run ``docker <args>``; on a permission error retry via ``sg docker -c``.

    Returns True on rc 0. Best-effort — NEVER raises: docker absent
    (``FileNotFoundError``), non-zero rc, or a CLI timeout all log a warning and
    return False. The ``sg docker -c "docker ..."`` fallback fires when the plain
    call hits a permission error (rc != 0 with a perms-shaped stderr, or a raised
    ``PermissionError``) — the Spark docker-group wrapping.
    """
    plain = ["docker", *args]
    try:
        proc = subprocess.run(
            plain, capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode == 0:
            return True
        stderr = (proc.stderr or "").lower()
        perms_shaped = (
            "permission denied" in stderr
            or "got permission denied" in stderr
            or "dial unix" in stderr
            or "connect: permission denied" in stderr
        )
        if not perms_shaped:
            logger.warning(
                "vllm_container_lifecycle: %r failed (rc=%d): %s",
                " ".join(plain), proc.returncode, (proc.stderr or "").strip(),
            )
            return False
        # fall through to the sg-docker retry
    except FileNotFoundError:
        logger.warning(
            "vllm_container_lifecycle: docker CLI not found; cannot run %r.",
            " ".join(plain),
        )
        return False
    except PermissionError:
        pass  # fall through to the sg-docker retry
    except subprocess.TimeoutExpired:
        logger.warning(
            "vllm_container_lifecycle: %r timed out after %ss.",
            " ".join(plain), timeout,
        )
        return False
    except Exception as exc:  # noqa: BLE001 — any other subprocess failure
        logger.warning("vllm_container_lifecycle: %r raised: %s", " ".join(plain), exc)
        return False

    # Retry through the docker-group wrapper.
    wrapped = ["sg", "docker", "-c", " ".join(plain)]
    try:
        proc = subprocess.run(
            wrapped, capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode == 0:
            return True
        logger.warning(
            "vllm_container_lifecycle: sg-docker fallback %r failed (rc=%d): %s",
            " ".join(wrapped), proc.returncode, (proc.stderr or "").strip(),
        )
        return False
    except Exception as exc:  # noqa: BLE001 — sg absent / timeout / etc.
        logger.warning(
            "vllm_container_lifecycle: sg-docker fallback %r raised: %s",
            " ".join(wrapped), exc,
        )
        return False


def ensure_serving(
    base_url: str,
    timeout_seconds: float = 600,
    *,
    run_dir: Optional[Union[str, Path]] = None,
) -> Optional[float]:
    """Ensure the vLLM seat at ``base_url`` is serving; return the load seconds.

    * Flag OFF or ``base_url`` unmapped → ``None`` (a pure no-op: no docker, no
      network).
    * Already serving (``/v1/models`` answers) → ``0.0`` (no ``docker start``).
    * Otherwise ``docker start`` the mapped container and poll ``/v1/models``
      until it answers or ``timeout_seconds`` elapses. Returns the measured
      **load_seconds** (wall-clock from just-before ``docker start`` to first
      ready probe) — the load-time metering half of task #10. When a container
      was actually started AND ``run_dir`` is known, a load event is appended
      via :func:`record_load_event`.
    * Timeout → warn + ``None`` (the caller proceeds; its own HTTP call to the
      seat will then fail loudly rather than this helper masking the stall).

    Best-effort — never raises.
    """
    if not resolve_vllm_container_lifecycle_mode():
        return None
    container = _lookup_container(base_url)
    if not container:
        return None

    if _probe_ready(base_url):
        logger.debug(
            "vllm_container_lifecycle: seat %s (%s) already serving.",
            base_url, container,
        )
        return 0.0

    start = time.monotonic()
    logger.info(
        "vllm_container_lifecycle: starting container %r for seat %s.",
        container, base_url,
    )
    if not _run_docker(["start", container]):
        logger.warning(
            "vllm_container_lifecycle: docker start %r failed; seat %s not "
            "brought up (caller's own call will fail loudly).",
            container, base_url,
        )
        return None

    deadline = start + float(timeout_seconds)
    while time.monotonic() < deadline:
        if _probe_ready(base_url):
            load_seconds = time.monotonic() - start
            logger.info(
                "vllm_container_lifecycle: seat %s (%s) ready after %.1fs.",
                base_url, container, load_seconds,
            )
            if run_dir is not None:
                record_load_event(run_dir, base_url, load_seconds)
            return load_seconds
        time.sleep(_POLL_INTERVAL_SECONDS)

    logger.warning(
        "vllm_container_lifecycle: seat %s (%s) not ready within %ss; returning "
        "None (caller's own call will fail loudly).",
        base_url, container, timeout_seconds,
    )
    return None


def release(base_url: str) -> bool:
    """``docker stop`` the container mapped to ``base_url``; return success.

    Flag OFF or ``base_url`` unmapped → ``False`` (no-op). Else ``docker stop``
    the mapped container (fail-soft via :func:`_run_docker`), returning True only
    when the stop succeeded. Stopping the container frees the seat's VRAM
    entirely (not merely idle-resident).
    """
    if not resolve_vllm_container_lifecycle_mode():
        return False
    container = _lookup_container(base_url)
    if not container:
        return False
    logger.info(
        "vllm_container_lifecycle: stopping container %r for seat %s.",
        container, base_url,
    )
    return _run_docker(["stop", container])


def release_all() -> int:
    """``docker stop`` every registered container; return the count stopped.

    Flag OFF → ``0`` (no-op). Else iterate the whole seat registry and stop each
    container, tallying the successful stops. Best-effort — one container's stop
    failure does not stop the others.
    """
    if not resolve_vllm_container_lifecycle_mode():
        return 0
    registry = parse_container_registry()
    stopped = 0
    for base_url, container in registry.items():
        logger.info(
            "vllm_container_lifecycle: release_all stopping container %r "
            "(seat %s).",
            container, base_url,
        )
        if _run_docker(["stop", container]):
            stopped += 1
    if registry:
        logger.info(
            "vllm_container_lifecycle: release_all stopped %d/%d container(s).",
            stopped, len(registry),
        )
    return stopped


def record_load_event(
    run_dir: Optional[Union[str, Path]],
    base_url: str,
    load_seconds: float,
) -> None:
    """Append a container-load metering row to ``<run_dir>/model_load_events.jsonl``.

    Row shape: ``{ts, base_url, container, load_seconds}`` (``ts`` = wall-clock
    UTC ISO-8601). Mirrors :func:`lib.llm.vram_doctor.write_trajectory_row` — the
    parent dir is created on first use and ANY failure (``run_dir`` None,
    unwritable path, serialization error) is logged and swallowed; a metering
    write must NEVER crash the run it observes. ``run_dir=None`` → skip.
    """
    if run_dir is None:
        return
    try:
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        target = run_path / "model_load_events.jsonl"
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "base_url": str(base_url).rstrip("/"),
            "container": _lookup_container(base_url),
            "load_seconds": round(float(load_seconds), 3),
        }
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception as exc:  # noqa: BLE001 — metering must never crash the run
        logger.debug(
            "vllm_container_lifecycle: load-event write failed for %s (ignoring): %s",
            base_url, exc,
        )


# ==========================================================================
# Declarative per-phase SEAT SCHEDULE primitives (ED4ALL_SEAT_SCHEDULE).
#
# These operate by LOGICAL SEAT NAME (the token used in the workflows.yaml
# ``seats:`` annotation) instead of raw base_url, and are gated by the caller's
# ``ED4ALL_SEAT_SCHEDULE`` decision rather than the container-lifecycle flag —
# the workflow_runner owns the on/off decision so the schedule can drive docker
# even when the standalone workflow-end ``ED4ALL_VLLM_CONTAINER_LIFECYCLE`` lease
# is off. Every primitive here is best-effort / NEVER-raises (mirrors the rest
# of this module); the ONE loud-failure signal — a seat that comes up but emits
# incoherent content (mode-collapse doctrine) — is surfaced by
# :func:`coherence_probe` returning ``False``, which the ORCHESTRATOR turns into
# a loud phase failure. This module never raises for it.
# ==========================================================================


def resolve_seat_schedule_mode(value: Optional[str] = None) -> bool:
    """Resolve whether the declarative per-phase seat schedule is enabled.

    Resolution chain: explicit ``value`` → ``ED4ALL_SEAT_SCHEDULE`` env →
    **default OFF**. Parse-with-fallback: only the truthy tokens
    (``1`` / ``true`` / ``yes`` / ``on``, case-insensitive) enable; anything
    else is off (the house default-off convention).
    """
    raw = value if value is not None else os.environ.get(ENV_SEAT_SCHEDULE)
    if raw is None or not str(raw).strip():
        return False
    return str(raw).strip().lower() in _TRUTHY


def parse_seat_registry(value: Optional[str] = None) -> Dict[str, str]:
    """Parse the ``seat_name=base_url`` registry into ``{seat_name: base_url}``.

    Reads ``ED4ALL_SEAT_BASE_URLS`` (or the explicit ``value``). Each
    comma-separated token is ``seat_name=base_url``; the ``base_url`` is
    normalized by stripping a trailing ``/`` so it matches the
    :data:`ENV_VLLM_CONTAINERS` keys. Fail-soft: unset / blank returns ``{}``; a
    token without exactly one ``=`` (or an empty side) is SKIPPED with a
    one-time warning — a partly-garbage registry still yields its valid pairs.
    """
    global _SEAT_REGISTRY_WARNED
    raw = value if value is not None else os.environ.get(ENV_SEAT_BASE_URLS)
    if raw is None or not str(raw).strip():
        return {}

    registry: Dict[str, str] = {}
    bad_tokens = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if token.count("=") != 1:
            bad_tokens.append(token)
            continue
        seat, base_url = token.split("=", 1)
        seat = seat.strip()
        base_url = base_url.strip().rstrip("/")
        if not seat or not base_url:
            bad_tokens.append(token)
            continue
        registry[seat] = base_url

    if bad_tokens and not _SEAT_REGISTRY_WARNED:
        logger.warning(
            "vllm_container_lifecycle: ignored %d malformed %s token(s) %r "
            "(expected 'seat_name=base_url'); using %d valid pair(s).",
            len(bad_tokens), ENV_SEAT_BASE_URLS, bad_tokens, len(registry),
        )
        _SEAT_REGISTRY_WARNED = True
    return registry


def resolve_seat_base_url(
    seat_name: str, *, value: Optional[str] = None
) -> Optional[str]:
    """Return the base_url mapped to ``seat_name``, or None if unmapped."""
    registry = parse_seat_registry(value)
    if not registry:
        return None
    return registry.get(str(seat_name).strip())


def all_registered_seat_names(value: Optional[str] = None) -> set:
    """Return every LOGICAL seat name declared in the seat registry."""
    return set(parse_seat_registry(value).keys())


def resolve_seat_load_timeout(value: Optional[str] = None) -> float:
    """Resolve the seat load-wait CEILING (seconds).

    Resolution chain: explicit ``value`` → ``ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS``
    env → **default 1200s (20 min)**. Parse-with-fallback: a positive int/float
    string wins; anything else — unset / blank / garbage / non-positive — falls
    back to the 1200s default (a 120B NVFP4 seat's cold load is ~8-10 min; the
    ceiling must comfortably exceed that). This is a CEILING on the health-check
    poll, NOT a fixed sleep: the poll returns the instant the seat is ready.
    """
    raw = value if value is not None else os.environ.get(ENV_SEAT_LOAD_TIMEOUT_SECONDS)
    if raw is None or not str(raw).strip():
        return _DEFAULT_SEAT_LOAD_TIMEOUT_SECONDS
    try:
        parsed = float(str(raw).strip())
    except (TypeError, ValueError):
        return _DEFAULT_SEAT_LOAD_TIMEOUT_SECONDS
    if parsed <= 0:
        return _DEFAULT_SEAT_LOAD_TIMEOUT_SECONDS
    return parsed


def resolve_seat_coherence_attempts(value: Optional[str] = None) -> int:
    """Resolve the bounded COUNT of content-coherence probe attempts.

    Resolution chain: explicit ``value`` → ``ED4ALL_SEAT_COHERENCE_ATTEMPTS``
    env → **default 3**. Parse-with-fallback: a positive int wins; anything else
    — unset / blank / garbage / non-positive — falls back to 3. Coherence is
    bounded (a mode-collapsed seat is detectable in seconds) and must NOT ride
    out the liveness ceiling: after this many failed attempts the schedule
    cold-recreates immediately.
    """
    raw = value if value is not None else os.environ.get(ENV_SEAT_COHERENCE_ATTEMPTS)
    if raw is None or not str(raw).strip():
        return _DEFAULT_SEAT_COHERENCE_ATTEMPTS
    try:
        parsed = int(str(raw).strip())
    except (TypeError, ValueError):
        return _DEFAULT_SEAT_COHERENCE_ATTEMPTS
    if parsed <= 0:
        return _DEFAULT_SEAT_COHERENCE_ATTEMPTS
    return parsed


def parse_seat_launch_specs(value: Optional[str] = None) -> Dict[str, str]:
    """Parse the ``seat_name=<launch spec>`` registry into ``{seat_name: spec}``.

    Reads ``ED4ALL_SEAT_LAUNCH_SPECS`` (or the explicit ``value``). Tokens are
    separated by ``;`` when present (a launch command may itself contain ``,``),
    else by ``,``. Each token is split on its FIRST ``=`` only, so the spec (the
    RHS — a script path or a full shell command) may contain further ``=`` (e.g.
    ``--env FOO=bar``). Fail-soft: unset / blank returns ``{}``; a token without
    a ``=`` (or with an empty seat name / empty spec) is SKIPPED with a one-time
    warning — a partly-garbage registry still yields its valid pairs.
    """
    global _LAUNCH_SPEC_WARNED
    raw = value if value is not None else os.environ.get(ENV_SEAT_LAUNCH_SPECS)
    if raw is None or not str(raw).strip():
        return {}

    sep = ";" if ";" in str(raw) else ","
    registry: Dict[str, str] = {}
    bad_tokens = []
    for token in str(raw).split(sep):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            bad_tokens.append(token)
            continue
        seat, spec = token.split("=", 1)
        seat = seat.strip()
        spec = spec.strip()
        if not seat or not spec:
            bad_tokens.append(token)
            continue
        registry[seat] = spec

    if bad_tokens and not _LAUNCH_SPEC_WARNED:
        logger.warning(
            "vllm_container_lifecycle: ignored %d malformed %s token(s) %r "
            "(expected 'seat_name=<launch script path or command>'); using %d "
            "valid pair(s).",
            len(bad_tokens), ENV_SEAT_LAUNCH_SPECS, bad_tokens, len(registry),
        )
        _LAUNCH_SPEC_WARNED = True
    return registry


def resolve_seat_launch_spec(
    seat_name: str, *, value: Optional[str] = None
) -> Optional[str]:
    """Return the launch spec mapped to ``seat_name``, or None if none configured."""
    registry = parse_seat_launch_specs(value)
    if not registry:
        return None
    return registry.get(str(seat_name).strip())


def start_seat(
    seat_name: str,
    *,
    timeout_seconds: Optional[float] = None,
    run_dir: Optional[Union[str, Path]] = None,
) -> Optional[float]:
    """Bring the logical ``seat_name`` up and WAIT until it answers ``/v1/models``.

    NOT gated by :func:`resolve_vllm_container_lifecycle_mode` — the seat-
    schedule caller owns the on/off decision. Resolves ``seat_name`` →
    base_url (:data:`ENV_SEAT_BASE_URLS`) → container (:data:`ENV_VLLM_CONTAINERS`).

    ``timeout_seconds`` is the load-wait CEILING: ``None`` → resolve from
    :func:`resolve_seat_load_timeout` (``ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS``,
    default 1200s). This is stage (a) of the two-stage readiness gate — a
    ``/v1/models`` liveness poll that returns the instant the swap/load is done;
    the caller (:func:`start_seat_coherent`) then runs stage (b), the content
    coherence poll. The loop emits a periodic "still loading, N s elapsed"
    progress line during long waits.

    Returns:
      * ``0.0``  — the seat was already serving (no ``docker start``).
      * ``float`` — measured load seconds after a cold ``docker start`` + poll.
      * ``None`` — the seat is UNRESOLVABLE (no base_url / no container mapping:
        a CONFIG gap) OR could not be brought up in time / ``docker start``
        failed (a runtime failure). The caller distinguishes the two by
        pre-resolving :func:`resolve_seat_base_url`.

    Best-effort — never raises.
    """
    if timeout_seconds is None:
        timeout_seconds = resolve_seat_load_timeout()
    base_url = resolve_seat_base_url(seat_name)
    if not base_url:
        return None
    container = _lookup_container(base_url)
    if not container:
        logger.warning(
            "vllm_container_lifecycle: seat %r resolves to base_url %s but that "
            "URL is not in %s; cannot start.",
            seat_name, base_url, ENV_VLLM_CONTAINERS,
        )
        return None

    if _probe_ready(base_url):
        logger.debug(
            "vllm_container_lifecycle: seat %r (%s / %s) already serving.",
            seat_name, base_url, container,
        )
        return 0.0

    start = time.monotonic()
    logger.info(
        "vllm_container_lifecycle: seat-schedule starting container %r for "
        "seat %r (%s).",
        container, seat_name, base_url,
    )
    if not _run_docker(["start", container]):
        logger.warning(
            "vllm_container_lifecycle: docker start %r failed for seat %r.",
            container, seat_name,
        )
        return None

    deadline = start + float(timeout_seconds)
    next_progress = start + _PROGRESS_LOG_INTERVAL_SECONDS
    while time.monotonic() < deadline:
        if _probe_ready(base_url):
            load_seconds = time.monotonic() - start
            logger.info(
                "vllm_container_lifecycle: seat %r (%s) live on /v1/models after "
                "%.1fs.",
                seat_name, base_url, load_seconds,
            )
            if run_dir is not None:
                record_load_event(run_dir, base_url, load_seconds)
            return load_seconds
        now = time.monotonic()
        if now >= next_progress:
            logger.info(
                "vllm_container_lifecycle: seat %r (%s) still loading, %ds "
                "elapsed (ceiling %ds)…",
                seat_name, base_url, int(now - start), int(timeout_seconds),
            )
            next_progress = now + _PROGRESS_LOG_INTERVAL_SECONDS
        time.sleep(_POLL_INTERVAL_SECONDS)

    logger.warning(
        "vllm_container_lifecycle: seat %r (%s / %s) not live within %ss.",
        seat_name, base_url, container, timeout_seconds,
    )
    return None


def stop_seat(seat_name: str) -> bool:
    """``docker stop`` the container backing logical ``seat_name``; return success.

    NOT gated by the container-lifecycle flag (the seat-schedule caller owns the
    gate). Unresolvable seat / container → ``False`` (no-op). Best-effort via
    :func:`_run_docker`.
    """
    base_url = resolve_seat_base_url(seat_name)
    if not base_url:
        return False
    container = _lookup_container(base_url)
    if not container:
        return False
    logger.info(
        "vllm_container_lifecycle: seat-schedule stopping container %r for "
        "seat %r (%s).",
        container, seat_name, base_url,
    )
    return _run_docker(["stop", container])


def _first_model_id(base_url: str, *, timeout: float = _PROBE_TIMEOUT_SECONDS) -> Optional[str]:
    """GET ``{base_url}/v1/models`` and return the first served model id, or None."""
    url = f"{str(base_url).rstrip('/')}/v1/models"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                mid = first.get("id")
                return str(mid) if mid else None
    except Exception as exc:  # noqa: BLE001 — server down / bad JSON / no models
        logger.debug("vllm_container_lifecycle: model-id fetch %s failed (%s).", url, exc)
    return None


def _looks_coherent(text: Optional[str]) -> bool:
    """Require the minimal structured probe contract, not merely sane text."""
    if not text:
        return False
    stripped = str(text).strip()
    if not stripped:
        return False
    try:
        payload = json.loads(stripped)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(payload, dict)
        and set(payload) == {"seat_ok"}
        and payload["seat_ok"] is True
    )


def coherence_probe(
    base_url: str,
    *,
    prompt: Optional[str] = None,
    timeout: float = _COHERENCE_TIMEOUT_SECONDS,
    max_tokens: int = _COHERENCE_MAX_TOKENS,
) -> bool:
    """CONTENT-level coherence probe: require a minimal JSON schema round-trip.

    A seat can pass the ``/v1/models`` liveness probe yet emit degenerate soup /
    null content after a ``docker start`` (mode-collapse doctrine). This probe
    POSTs :data:`_COHERENCE_PROMPT` to ``{base_url}/v1/chat/completions`` (model
    id auto-resolved from ``/v1/models``) and returns ``True`` only when the
    response carries the exact ``{"seat_ok": true}`` object
    (:func:`_looks_coherent`).

    NEVER raises — any failure (no model id, HTTP error, malformed JSON, empty /
    soup content) returns ``False``. The ORCHESTRATOR turns a ``False`` on a
    newly-started seat into a LOUD phase failure; this function itself is silent.
    """
    base = str(base_url).rstrip("/")
    model = _first_model_id(base, timeout=min(timeout, _PROBE_TIMEOUT_SECONDS * 2))
    if not model:
        logger.warning(
            "vllm_container_lifecycle: coherence probe could not resolve a model "
            "id at %s.", base,
        )
        return False
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt or _COHERENCE_PROMPT}],
            "max_tokens": int(max_tokens),
            "temperature": 0,
            "stream": False,
            # Reasoning-parsed seats (e.g. nemotron_v3) route think-tokens into
            # reasoning_content; without this a small probe budget is consumed
            # by think-scaffolding and content comes back EMPTY, mis-declaring a
            # healthy seat mode-collapsed. Templates that don't use the kwarg
            # ignore it (plain Jinja context var).
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode("utf-8")
    url = f"{base}/v1/chat/completions"
    try:
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            logger.warning(
                "vllm_container_lifecycle: coherence probe at %s returned no "
                "choices.", base,
            )
            return False
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        ok = _looks_coherent(content)
        if not ok and isinstance(message, dict):
            # A reasoning seat may spend the whole probe budget thinking (content
            # empty, reasoning_content populated). Sane reasoning text is direct
            # evidence AGAINST mode collapse (collapse = degenerate soup in any
            # channel), so accept it rather than cold-recreating a healthy seat.
            reasoning = message.get("reasoning_content")
            ok = _looks_coherent(reasoning)
        if not ok:
            logger.warning(
                "vllm_container_lifecycle: coherence probe at %s returned "
                "incoherent/empty content %r (possible mode collapse).",
                base, (content or "")[:80],
            )
        return ok
    except Exception as exc:  # noqa: BLE001 — HTTP down / timeout / bad JSON
        logger.warning(
            "vllm_container_lifecycle: coherence probe POST %s failed (%s).",
            url, exc,
        )
        return False


# ==========================================================================
# COLD-RECREATE self-heal for a mode-collapsed seat (ED4ALL_SEAT_LAUNCH_SPECS).
#
# ``start_seat`` can only WARM ``docker start`` a pre-existing container; a seat
# that mode-collapses on that warm restart (passes ``/v1/models`` yet emits
# soup/null — the documented failure) would otherwise fail the phase LOUDLY with
# no automatic recovery. These primitives add the cold-recreate path — ``docker
# rm -f`` the collapsed container + re-run its data-driven launch spec — so the
# manual "cold stop → start → re-probe" recovery becomes automatic pipeline
# behavior. Everything is best-effort / NEVER-raises (module doctrine); the loud
# failure stays owned by the orchestrator, which turns a non-``ok``
# :class:`SeatStartResult` into a ``SeatScheduleProbeError``.
# ==========================================================================


def _run_launch_spec(spec: str, *, timeout: float) -> bool:
    """Run a seat launch spec (script path or full command) via the shell.

    The spec is ops-owned config (a script path like ``/opt/seats/launch.sh`` or
    a full ``docker run …`` command line), so it is executed through the shell to
    honor its argv/redirection exactly as written. Best-effort — NEVER raises: a
    non-zero rc, a launch-command timeout, or any subprocess failure logs a
    warning and returns False. ``timeout`` bounds the launch INVOCATION (a
    ``docker run -d`` returns immediately; a script that blocks until ready is
    still bounded by the load-wait ceiling passed here).
    """
    try:
        proc = subprocess.run(
            spec, shell=True, capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode == 0:
            return True
        logger.warning(
            "vllm_container_lifecycle: launch spec %r failed (rc=%d): %s",
            spec, proc.returncode, (proc.stderr or "").strip(),
        )
        return False
    except subprocess.TimeoutExpired:
        logger.warning(
            "vllm_container_lifecycle: launch spec %r timed out after %ss.",
            spec, timeout,
        )
        return False
    except Exception as exc:  # noqa: BLE001 — any launch failure is non-fatal here
        logger.warning(
            "vllm_container_lifecycle: launch spec %r raised: %s", spec, exc
        )
        return False


def _emit_recreate_capture(
    seat_name: str,
    base_url: Optional[str],
    container: Optional[str],
    *,
    reason: str,
    load_seconds: Optional[float],
    launched: bool,
    run_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Emit a DecisionCapture recording a seat COLD-RECREATE (best-effort).

    Project law: a new decision path logs a DecisionCapture with a dynamic
    rationale interpolating the load-bearing signals (seat, base_url, container,
    ``reason``, the reload seconds, whether the relaunch command returned 0). A
    recreate is a genuine autonomous recovery decision, so it is captured for
    post-hoc replay. NEVER raises — a capture-write failure must not break the
    recovery it observes.
    """
    try:
        from lib.decision_capture import DecisionCapture

        run_identity = (
            Path(run_dir).name
            if run_dir is not None
            else os.environ.get("ED4ALL_RUN_ID", "").strip()
        )
        if not run_identity:
            logger.warning(
                "vllm_container_lifecycle: recreate capture skipped because "
                "no run-scoped identity is available."
            )
            return
        phase_identity = (
            os.environ.get("ED4ALL_PHASE_NAME", "").strip()
            or (
                "training_synthesis"
                if "synthesis_engine_dead:" in reason
                else "seat_schedule"
            )
        )
        dc = DecisionCapture(
            course_code=run_identity,
            phase=phase_identity,
            tool="orchestrator",
        )
        dc.log_decision(
            decision_type="seat_cold_recreate",
            decision=(
                f"cold-recreated vLLM seat {seat_name!r} "
                f"(container {container!r} at {base_url})"
            ),
            rationale=(
                f"seat {seat_name!r} ({base_url}, container {container!r}) "
                f"reason={reason}: the warm docker-start passed /v1/models "
                f"liveness but FAILED the content coherence probe (mode-collapse "
                f"doctrine), so the schedule cold-recreated it via docker rm -f + "
                f"its ED4ALL_SEAT_LAUNCH_SPECS launch spec "
                f"(relaunch_ok={launched}, reload_seconds={load_seconds}). "
                f"Automatic self-heal replacing the manual cold-restart."
            ),
            alternatives_considered=[
                {
                    "option": "Keep the warm-started container",
                    "reason_rejected": (
                        f"Seat {seat_name!r} at {base_url} failed the coherence "
                        f"probe with reason={reason!r}, so the warm container "
                        f"{container!r} cannot serve the scheduled workload."
                    ),
                },
                {
                    "option": "Raise without attempting recovery",
                    "reason_rejected": (
                        f"Seat {seat_name!r} has a configured cold-recreate path; "
                        f"the relaunch result was launched={launched} with "
                        f"reload_seconds={load_seconds}."
                    ),
                },
            ],
        )
        try:
            dc.save()
        except Exception:  # noqa: BLE001 — save is best-effort
            pass
    except Exception as exc:  # noqa: BLE001 — capture must never crash the recovery
        logger.debug(
            "vllm_container_lifecycle: recreate DecisionCapture failed for seat "
            "%r (ignoring): %s",
            seat_name, exc,
        )


def recreate_seat(
    seat_name: str,
    *,
    run_dir: Optional[Union[str, Path]] = None,
    reason: str = "warm_start_incoherent",
    timeout_seconds: Optional[float] = None,
) -> Optional[float]:
    """COLD-recreate ``seat_name``: ``docker rm -f`` + relaunch + poll to live.

    The self-heal path a warm ``docker start`` cannot provide. Resolves
    ``seat_name`` → base_url → container and → its launch spec
    (:func:`resolve_seat_launch_spec` / ``ED4ALL_SEAT_LAUNCH_SPECS``). Sequence:

      1. ``docker rm -f`` the (mode-collapsed) container — best-effort.
      2. Run the launch spec (script path or full command) via
         :func:`_run_launch_spec`. A launch failure → warn + ``None``.
      3. Poll ``/v1/models`` until live or the load-wait ceiling
         (``timeout_seconds`` → :func:`resolve_seat_load_timeout`, default 1200s),
         with a periodic progress line. Timeout → ``None``.

    Emits a DecisionCapture recording the recreate (dynamic rationale naming
    seat / base_url / ``reason``). Returns the measured reload seconds on success,
    else ``None`` (unresolvable seat / no launch spec / launch failed / timeout).
    Best-effort — never raises. The CALLER re-runs the coherence probe: a live
    reload does not by itself prove coherence.
    """
    if timeout_seconds is None:
        timeout_seconds = resolve_seat_load_timeout()
    base_url = resolve_seat_base_url(seat_name)
    if not base_url:
        return None
    container = _lookup_container(base_url)
    if not container:
        logger.warning(
            "vllm_container_lifecycle: seat %r resolves to base_url %s but that "
            "URL is not in %s; cannot recreate.",
            seat_name, base_url, ENV_VLLM_CONTAINERS,
        )
        return None
    spec = resolve_seat_launch_spec(seat_name)
    if not spec:
        logger.error(
            "vllm_container_lifecycle: seat %r has NO launch spec in %s; cannot "
            "cold-recreate (a warm start cannot self-heal a mode collapse).",
            seat_name, ENV_SEAT_LAUNCH_SPECS,
        )
        return None

    logger.warning(
        "vllm_container_lifecycle: COLD-RECREATING seat %r (container %r at %s) "
        "reason=%s — docker rm -f + relaunch via launch spec.",
        seat_name, container, base_url, reason,
    )
    start = time.monotonic()
    # 1. Remove the collapsed container (best-effort; a launch that re-`docker
    #    run --name <container>` needs the old name freed).
    _run_docker(["rm", "-f", container])
    # 2. Relaunch from the data-driven spec.
    launched = _run_launch_spec(spec, timeout=float(timeout_seconds))
    if not launched:
        _emit_recreate_capture(
            seat_name, base_url, container,
            reason=reason, load_seconds=None, launched=False,
            run_dir=run_dir,
        )
        logger.error(
            "vllm_container_lifecycle: seat %r launch spec did not succeed; "
            "recreate FAILED.",
            seat_name,
        )
        return None

    # 3. Poll to liveness within the ceiling.
    deadline = start + float(timeout_seconds)
    next_progress = start + _PROGRESS_LOG_INTERVAL_SECONDS
    reload_seconds: Optional[float] = None
    while time.monotonic() < deadline:
        if _probe_ready(base_url):
            reload_seconds = time.monotonic() - start
            logger.info(
                "vllm_container_lifecycle: seat %r (%s) live on /v1/models "
                "after cold recreate in %.1fs.",
                seat_name, base_url, reload_seconds,
            )
            if run_dir is not None:
                record_load_event(run_dir, base_url, reload_seconds)
            break
        now = time.monotonic()
        if now >= next_progress:
            logger.info(
                "vllm_container_lifecycle: seat %r (%s) recreate still loading, "
                "%ds elapsed (ceiling %ds)…",
                seat_name, base_url, int(now - start), int(timeout_seconds),
            )
            next_progress = now + _PROGRESS_LOG_INTERVAL_SECONDS
        time.sleep(_POLL_INTERVAL_SECONDS)

    _emit_recreate_capture(
        seat_name, base_url, container,
        reason=reason, load_seconds=reload_seconds, launched=True,
        run_dir=run_dir,
    )
    if reload_seconds is None:
        logger.warning(
            "vllm_container_lifecycle: seat %r (%s / %s) not live within %ss "
            "after cold recreate.",
            seat_name, base_url, container, timeout_seconds,
        )
    return reload_seconds


def _coherence_check(
    base_url: str,
    *,
    seat_name: str,
    attempts: Optional[int] = None,
    interval: float = _COHERENCE_POLL_INTERVAL_SECONDS,
) -> bool:
    """BOUNDED content-coherence check: at most ``attempts`` probes once LIVE.

    Stage (b) of the readiness gate, and DELIBERATELY bounded — unlike the
    liveness poll (which legitimately waits out the full load ceiling for a slow
    120B load), a mode-collapsed seat is live-but-incoherent and detectable in
    SECONDS, so it must NOT ride out ``ED4ALL_SEAT_LOAD_TIMEOUT_SECONDS``. Probes
    :func:`coherence_probe` up to ``attempts`` times (``None`` →
    :func:`resolve_seat_coherence_attempts`, default 3), sleeping a short
    ``interval`` between tries, and returns ``True`` on the first pass. After all
    attempts fail it returns ``False`` so the caller cold-recreates IMMEDIATELY
    (self-heal in ~30-45s, not ~20 min). NEVER raises (``coherence_probe`` is
    itself never-raising).
    """
    if attempts is None:
        attempts = resolve_seat_coherence_attempts()
    for attempt in range(1, attempts + 1):
        if coherence_probe(base_url):
            return True
        if attempt < attempts:
            logger.info(
                "vllm_container_lifecycle: seat %r (%s) live but not yet "
                "coherent (attempt %d/%d); retrying in %ss…",
                seat_name, base_url, attempt, attempts, interval,
            )
            time.sleep(interval)
    logger.warning(
        "vllm_container_lifecycle: seat %r (%s) STILL incoherent after %d "
        "bounded coherence attempt(s) — declaring mode-collapse.",
        seat_name, base_url, attempts,
    )
    return False


class SeatStartResult(NamedTuple):
    """Outcome of :func:`start_seat_coherent`.

    ``ok`` — the seat is up AND coherent (the phase may proceed). ``reason`` is a
    stable machine token the orchestrator maps to a ``SeatScheduleProbeError``
    message: ``already_serving`` / ``warm_start_coherent`` / ``start_failed`` /
    ``warm_incoherent_no_spec`` / ``cold_recreate_coherent`` / ``recreate_failed``
    / ``still_incoherent_after_recreate``. ``recreated`` records whether the cold
    self-heal path fired.
    """

    ok: bool
    reason: str
    load_seconds: Optional[float]
    recreated: bool
    base_url: Optional[str]


class SeatRecoveryResult(NamedTuple):
    """Outcome of a locked deep-health recovery attempt."""

    ok: bool
    reason: str
    seat_name: Optional[str]
    recreated: bool
    engine_signals: tuple[str, ...]
    load_seconds: Optional[float]


def _seat_for_base_url(base_url: str) -> Optional[str]:
    normalized = str(base_url).rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    matches = sorted(
        seat for seat, url in parse_seat_registry().items()
        if str(url).rstrip("/").removesuffix("/v1") == normalized
    )
    return matches[0] if matches else None


def _engine_failure_signals(seat_name: str) -> tuple[str, ...]:
    """Best-effort recent TRT fatal signals; never treats absence as health."""
    base_url = resolve_seat_base_url(seat_name) or ""
    normalized = str(base_url).rstrip("/").removesuffix("/v1")
    container = _lookup_container(normalized)
    if not container:
        return ()
    try:
        proc = subprocess.run(
            ["docker", "logs", "--tail", "400", container],
            capture_output=True, text=True, timeout=10, check=False,
        )
        text = f"{proc.stdout}\n{proc.stderr}"
    except Exception:  # noqa: BLE001
        return ()
    needles = (
        "Hang detected",
        "RequestError",
        "sampler_event.synchronize",
    )
    return tuple(token for token in needles if token in text)


def recover_seat_for_base_url(
    base_url: str,
    *,
    run_dir: Optional[Union[str, Path]] = None,
    reason: str,
) -> SeatRecoveryResult:
    """Singleflight cold recovery after a failed deep generation probe.

    Every waiter probes again *under* the cross-process seat lock.  Therefore
    only the first caller recreates; siblings observe the healed generation
    path and return ``already_recovered``.  A recreated seat must pass two
    independent chat completions before it is released to workload traffic.
    """

    seat_name = _seat_for_base_url(base_url)
    if not seat_name:
        return SeatRecoveryResult(
            False, "unmapped_base_url", None, False, (), None
        )
    lock_dir = STATE_PATH / "seat_recovery"
    lock_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in seat_name)
    try:
        # A Super cold load can take 8-10 minutes.  This lock must block for
        # the whole lifecycle operation: ``file_lock`` deliberately proceeds
        # unlocked after a finite timeout, which is unsafe for docker rm/run.
        with file_lock(lock_dir / f"{safe}.lock", timeout=None):
            mapped = resolve_seat_base_url(seat_name)
            if mapped and coherence_probe(mapped, max_tokens=32):
                return SeatRecoveryResult(
                    True, "already_recovered", seat_name, False, (), 0.0
                )
            signals = _engine_failure_signals(seat_name)
            reload_seconds = recreate_seat(
                seat_name, run_dir=run_dir, reason=reason
            )
            if reload_seconds is None:
                return SeatRecoveryResult(
                    False, "recreate_failed", seat_name, True, signals, None
                )
            healthy = bool(mapped) and all(
                coherence_probe(mapped, max_tokens=32) for _ in range(2)
            )
            return SeatRecoveryResult(
                healthy,
                "recovered" if healthy else "post_recreate_probe_failed",
                seat_name,
                True,
                signals,
                reload_seconds,
            )
    except Exception as exc:  # noqa: BLE001 - lifecycle surface never raises
        logger.error(
            "vllm_container_lifecycle: recovery lock/lifecycle failed for "
            "seat %r (%s): %s",
            seat_name,
            base_url,
            exc,
        )
        return SeatRecoveryResult(
            False,
            f"recovery_exception:{type(exc).__name__}",
            seat_name,
            False,
            (),
            None,
        )


def start_seat_coherent(
    seat_name: str,
    *,
    run_dir: Optional[Union[str, Path]] = None,
    timeout_seconds: Optional[float] = None,
) -> SeatStartResult:
    """Bring ``seat_name`` up, WAIT until live AND coherent, self-heal on collapse.

    The full reliable readiness gate the seat schedule needs, composed from the
    module primitives:

      1. :func:`start_seat` — warm ``docker start`` + poll ``/v1/models`` to
         LIVENESS (stage a), bounded by the load-wait ceiling
         (``timeout_seconds`` → :func:`resolve_seat_load_timeout`, default 1200s)
         — a genuinely slow 120B load legitimately needs this whole ceiling.
      2. :func:`_coherence_check` — a BOUNDED content-coherence check (stage b):
         at most :func:`resolve_seat_coherence_attempts` probes (default 3), NOT
         the liveness ceiling, because a mode-collapse is detectable in seconds.
      3. If the warm seat is live but INCOHERENT after those bounded attempts AND
         a launch spec is configured → :func:`recreate_seat` (``docker rm -f`` +
         relaunch) IMMEDIATELY (do NOT wait out the ceiling), then re-run stages
         a+b once more.

    Net effect: a slow load waits patiently (up to the ceiling for LIVENESS); a
    mode-collapse self-heals in ~bounded-coherence-attempts + the recreate load
    time, not ~20 min. Returns a :class:`SeatStartResult`; NEVER raises. The
    orchestrator turns a non-``ok`` result into a LOUD ``SeatScheduleProbeError``
    (loud-failure stays owned by the caller, per module doctrine). ``ok`` only
    when the seat both came up and passed coherence — warm, or after exactly one
    cold recreate.
    """
    if timeout_seconds is None:
        timeout_seconds = resolve_seat_load_timeout()
    base_url = resolve_seat_base_url(seat_name)

    # --- WARM: docker start + LIVENESS poll to the full ceiling (stage a) --
    load = start_seat(seat_name, timeout_seconds=timeout_seconds, run_dir=run_dir)
    if load is None:
        return SeatStartResult(
            ok=False, reason="start_failed", load_seconds=None,
            recreated=False, base_url=base_url,
        )
    # Stage b: BOUNDED coherence check (a mode-collapse is detectable in seconds
    # — do NOT let it ride out the liveness ceiling).
    if _coherence_check(base_url, seat_name=seat_name):
        return SeatStartResult(
            ok=True,
            reason=("already_serving" if load == 0.0 else "warm_start_coherent"),
            load_seconds=load, recreated=False, base_url=base_url,
        )

    # --- Warm seat is live but INCOHERENT → cold-recreate self-heal -------
    logger.warning(
        "vllm_container_lifecycle: seat %r warm-started but FAILED the bounded "
        "coherence check (possible mode collapse); attempting cold-recreate "
        "self-heal IMMEDIATELY (not waiting out the liveness ceiling).",
        seat_name,
    )
    spec = resolve_seat_launch_spec(seat_name)
    if not spec:
        logger.error(
            "vllm_container_lifecycle: seat %r incoherent and NO launch spec "
            "configured (%s) — cannot self-heal.",
            seat_name, ENV_SEAT_LAUNCH_SPECS,
        )
        return SeatStartResult(
            ok=False, reason="warm_incoherent_no_spec", load_seconds=load,
            recreated=False, base_url=base_url,
        )

    # --- COLD: docker rm -f + relaunch + LIVENESS poll to the ceiling -----
    reload = recreate_seat(
        seat_name, run_dir=run_dir,
        reason="warm_start_incoherent", timeout_seconds=timeout_seconds,
    )
    if reload is None:
        return SeatStartResult(
            ok=False, reason="recreate_failed", load_seconds=load,
            recreated=True, base_url=base_url,
        )
    # Stage b again: BOUNDED coherence check after the cold recreate.
    if all(coherence_probe(base_url) for _ in range(2)):
        logger.info(
            "vllm_container_lifecycle: seat %r COLD-RECREATED and now COHERENT "
            "after %.1fs (self-heal succeeded).",
            seat_name, reload,
        )
        return SeatStartResult(
            ok=True, reason="cold_recreate_coherent", load_seconds=reload,
            recreated=True, base_url=base_url,
        )
    logger.error(
        "vllm_container_lifecycle: seat %r STILL incoherent after a cold "
        "recreate — giving up (the orchestrator will fail the phase LOUDLY).",
        seat_name,
    )
    return SeatStartResult(
        ok=False, reason="still_incoherent_after_recreate", load_seconds=reload,
        recreated=True, base_url=base_url,
    )


__all__ = [
    "ENV_VLLM_CONTAINER_LIFECYCLE",
    "ENV_VLLM_CONTAINERS",
    "ENV_SEAT_SCHEDULE",
    "ENV_SEAT_BASE_URLS",
    "ENV_SEAT_LAUNCH_SPECS",
    "ENV_SEAT_LOAD_TIMEOUT_SECONDS",
    "ENV_SEAT_COHERENCE_ATTEMPTS",
    "resolve_vllm_container_lifecycle_mode",
    "parse_container_registry",
    "ensure_serving",
    "release",
    "release_all",
    "record_load_event",
    "resolve_seat_schedule_mode",
    "parse_seat_registry",
    "resolve_seat_base_url",
    "all_registered_seat_names",
    "resolve_seat_load_timeout",
    "resolve_seat_coherence_attempts",
    "parse_seat_launch_specs",
    "resolve_seat_launch_spec",
    "start_seat",
    "stop_seat",
    "coherence_probe",
    "recreate_seat",
    "start_seat_coherent",
    "SeatStartResult",
    "SeatRecoveryResult",
    "recover_seat_for_base_url",
]

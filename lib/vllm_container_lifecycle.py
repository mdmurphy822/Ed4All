"""Per-model vLLM CONTAINER lifecycle lease helper (Ed4All venv).

GOAL (owner directive, task #10): **do NOT keep models resident** — a model
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
from typing import Dict, Optional, Union

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
#: seat name that does not resolve here is a CONFIG gap → warn + skip (never a
#: run-failing error), so ``ED4ALL_SEAT_SCHEDULE`` on with an empty/partial map
#: degrades softly to no-op on the unmapped seats.
ENV_SEAT_BASE_URLS = "ED4ALL_SEAT_BASE_URLS"

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

#: Per-request HTTP timeout (seconds) for the content-level COHERENCE PROBE
#: (``POST /v1/chat/completions``). Longer than the ``/v1/models`` liveness
#: probe: a just-warmed seat's first token can be slow, but a wedged / mode-
#: collapsed seat still resolves within this bound.
_COHERENCE_TIMEOUT_SECONDS = 45.0

#: The fixed prompt the coherence probe POSTs. Deliberately trivial: any coherent
#: seat can echo it; a mode-collapsed seat emits degenerate soup / null content.
_COHERENCE_PROMPT = "Reply with exactly this word and nothing else: SEATOK"

#: Max completion tokens for the coherence probe (small — we only need enough to
#: tell coherent text from degenerate output).
_COHERENCE_MAX_TOKENS = 24


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


def start_seat(
    seat_name: str,
    *,
    timeout_seconds: float = 600,
    run_dir: Optional[Union[str, Path]] = None,
) -> Optional[float]:
    """Bring the logical ``seat_name`` up and WAIT until it answers ``/v1/models``.

    NOT gated by :func:`resolve_vllm_container_lifecycle_mode` — the seat-
    schedule caller owns the on/off decision. Resolves ``seat_name`` →
    base_url (:data:`ENV_SEAT_BASE_URLS`) → container (:data:`ENV_VLLM_CONTAINERS`).

    Returns:
      * ``0.0``  — the seat was already serving (no ``docker start``).
      * ``float`` — measured load seconds after a cold ``docker start`` + poll.
      * ``None`` — the seat is UNRESOLVABLE (no base_url / no container mapping:
        a CONFIG gap) OR could not be brought up in time / ``docker start``
        failed (a runtime failure). The caller distinguishes the two by
        pre-resolving :func:`resolve_seat_base_url`.

    Best-effort — never raises.
    """
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
    while time.monotonic() < deadline:
        if _probe_ready(base_url):
            load_seconds = time.monotonic() - start
            logger.info(
                "vllm_container_lifecycle: seat %r (%s) ready after %.1fs.",
                seat_name, base_url, load_seconds,
            )
            if run_dir is not None:
                record_load_event(run_dir, base_url, load_seconds)
            return load_seconds
        time.sleep(_POLL_INTERVAL_SECONDS)

    logger.warning(
        "vllm_container_lifecycle: seat %r (%s / %s) not ready within %ss.",
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
    """Heuristic: is ``text`` sane content vs mode-collapsed degenerate output?

    Rejects: ``None`` / empty / whitespace-only content, content with no
    alphanumeric character, and single-character-repeat soup (e.g. ``"!!!!!!"``
    or one glyph repeated). Deliberately permissive on WHAT the model said — we
    only need to tell coherent text from the null/soup a restarted-then-collapsed
    seat emits (memory: a seat can pass ``/v1/models`` yet return null content).
    """
    if not text:
        return False
    stripped = str(text).strip()
    if not stripped:
        return False
    if not any(ch.isalnum() for ch in stripped):
        return False
    # Degenerate single-glyph repetition (mode collapse) → not coherent.
    if len(set(stripped.replace(" ", ""))) <= 1:
        return False
    return True


def coherence_probe(
    base_url: str,
    *,
    prompt: Optional[str] = None,
    timeout: float = _COHERENCE_TIMEOUT_SECONDS,
    max_tokens: int = _COHERENCE_MAX_TOKENS,
) -> bool:
    """CONTENT-level coherence probe: POST a fixed prompt, require sane content.

    A seat can pass the ``/v1/models`` liveness probe yet emit degenerate soup /
    null content after a ``docker start`` (mode-collapse doctrine). This probe
    POSTs :data:`_COHERENCE_PROMPT` to ``{base_url}/v1/chat/completions`` (model
    id auto-resolved from ``/v1/models``) and returns ``True`` only when the
    response carries non-empty, non-degenerate assistant content
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


__all__ = [
    "ENV_VLLM_CONTAINER_LIFECYCLE",
    "ENV_VLLM_CONTAINERS",
    "ENV_SEAT_SCHEDULE",
    "ENV_SEAT_BASE_URLS",
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
    "start_seat",
    "stop_seat",
    "coherence_probe",
]

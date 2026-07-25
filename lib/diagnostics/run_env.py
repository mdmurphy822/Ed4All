"""Provider-resolution engine for the ``ed4all doctor`` provider/seat preflight.

This module answers two correctness-critical questions WITHOUT dispatching a
single LLM call:

1. **Per-seat provider resolution** (:func:`resolve_seats`) — for every
   authoring seat (single-pass content, two-pass outline/rewrite tiers,
   course-outliner, assessment, training-synthesis, textbook-synthesis,
   grounded-answer, embedding, objective-review, dynamic block planner),
   what provider WOULD the run resolve, off the CURRENT ``os.environ``, and
   from WHICH source (env var / block-routing YAML / fanout / class default /
   off)?

2. **Run-env fanout modelling** (:func:`applied_run_env`) — a hypothetical
   ``ed4all run`` mutates ``os.environ`` via two ``setdefault``-style helpers
   on :class:`MCP.core.workflow_runner.WorkflowRunner`
   (``_apply_corpus_generalization_defaults`` then
   ``_apply_authoring_route_env``, IN THAT ORDER). There is no pure variant of
   those helpers, so this context manager snapshots ``os.environ``, applies the
   two helpers, yields the merged deltas, and ALWAYS restores the environment
   in a ``finally`` (exception-safe).

Design contract:

* **Pure lib.** This module NEVER imports from ``cli/`` — the CLI consumes this
  engine, not the other way round. Heavier deps (MCP, ``yaml``, the endpoint
  registry) are LAZILY imported inside functions so a bare ``import
  lib.diagnostics.run_env`` stays cheap and side-effect-free.
* **Never raises.** Every public function is best-effort: a seat that cannot
  resolve gets ``effective_provider=None`` plus an explanatory ``note`` rather
  than an exception. The provider/key table degrades to ``{}`` on import
  failure.
* **No ``resolve_endpoint``.** The provider → key/base-url table is read
  STRUCTURALLY from :func:`lib.llm.endpoints.load_endpoint_registry` (the raw
  rows), NOT via :func:`lib.llm.endpoints.resolve_endpoint`, which RAISES on a
  missing required key.
"""

from __future__ import annotations

import contextlib
import json as _json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)

# Repo root: lib/diagnostics/run_env.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Hardcoded tier-provider fallbacks (router.py: outline ``OutlineProvider``
# default ``local``; rewrite ``RewriteProvider`` ``DEFAULT_PROVIDER='anthropic'``).
_HARDCODED_TIER_PROVIDER = {"outline": "local", "rewrite": "anthropic"}

_TRUTHY = {"1", "true", "yes", "on"}
_ANTHROPIC_FAMILY = {"anthropic", "claude_session"}


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class SeatResolution:
    """One authoring seat's resolved provider + provenance.

    ``source`` is one of ``env`` | ``yaml`` | ``fanout`` | ``class_default`` |
    ``off``. ``effective_provider`` is lowercased, or ``None`` when the seat is
    off / unset-and-no-default. ``note`` carries heterogeneity / two-pass /
    loopback / fail-closed caveats.
    """

    seat: str
    provider_env: str
    effective_provider: Optional[str]
    model: Optional[str]
    source: str
    note: str = ""


# ---------------------------------------------------------------------------
# Small env helpers
# ---------------------------------------------------------------------------


def _env(name: Optional[str]) -> Optional[str]:
    """Read a non-empty, stripped env var by name, else ``None``."""
    if not name:
        return None
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _truthy(name: str) -> bool:
    raw = os.environ.get(name, "")
    return raw.strip().lower() in _TRUTHY


def _join_notes(*parts: str) -> str:
    return "; ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Provider -> key / base-url table (registry-driven, never raises)
# ---------------------------------------------------------------------------


def load_provider_key_table() -> Dict[str, Dict[str, Any]]:
    """Return ``provider -> {key_env, key_required, base_url_env,
    base_url_default, model_env, default_model}``.

    Read STRUCTURALLY from :func:`lib.llm.endpoints.load_endpoint_registry`
    (the raw rows) — never via ``resolve_endpoint`` (which raises on a missing
    required key). Degrades to ``{}`` on any import / load failure.
    """
    try:
        from lib.llm.endpoints import load_endpoint_registry  # lazy
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning("run_env: endpoint registry import failed: %s", exc)
        return {}

    try:
        registry = load_endpoint_registry()
    except Exception as exc:
        logger.warning("run_env: endpoint registry load failed: %s", exc)
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    try:
        for name, row in registry.items():
            out[str(name)] = {
                "key_env": row.get("api_key_env"),
                "key_required": bool(row.get("api_key_required", False)),
                "base_url_env": row.get("base_url_env"),
                "base_url_default": row.get("base_url"),
                "model_env": row.get("model_env"),
                "default_model": row.get("default_model"),
            }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("run_env: endpoint registry shape unexpected: %s", exc)
        return {}
    return out


def key_status(
    provider: str, key_table: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Return ``{provider, key_env, key_required, key_present, base_url}``.

    Reads ``os.environ`` for the key + base-url env (NO ``resolve_endpoint``).
    A non-required-key provider (``local`` / unknown / ``None``) reports
    ``key_present=True``. Never raises.
    """
    table = key_table if key_table is not None else load_provider_key_table()
    row = table.get(provider, {}) if isinstance(table, dict) else {}

    key_env = row.get("key_env")
    key_required = bool(row.get("key_required", False))
    base_url_env = row.get("base_url_env")
    base_url_default = row.get("base_url_default")

    if not key_required:
        key_present = True
    else:
        key_present = bool(_env(key_env)) if key_env else False

    base_url = _env(base_url_env) or base_url_default

    return {
        "provider": provider,
        "key_env": key_env,
        "key_required": key_required,
        "key_present": key_present,
        "base_url": base_url,
    }


# ---------------------------------------------------------------------------
# Local-synthesis serving topology (ollama vs a vLLM seat) — the P0-1 fix.
#
# The environment / window doctor groups hardcoded the ollama LOCAL_SYNTHESIS
# topology (``/api/tags`` model-pull probe + ``/api/show`` served-window probe).
# On a vLLM-seat host — where LOCAL_SYNTHESIS_BASE_URL points at an OpenAI-
# compatible vLLM seat and a seat registry (ED4ALL_SEAT_BASE_URLS /
# ED4ALL_VLLM_CONTAINERS) is configured — those ollama probes fail for a model
# that is NOT the serving path, minting FALSE DEGRADED warns. This resolver
# tells the checks which backend actually serves local synthesis so they can
# probe the RIGHT endpoint (``/v1/models``) and skip the ollama-only warns.
#
# Legacy contract: when NO seat registry is configured, ``backend`` is always
# ``"ollama"`` and the checks take their byte-identical legacy path.
# ---------------------------------------------------------------------------

#: Ollama's default local root (the ``local`` endpoint-registry base_url without
#: ``/v1``). A LOCAL_SYNTHESIS_BASE_URL on port 11434 is treated as ollama.
_OLLAMA_DEFAULT_PORT_TOKEN = ":11434"

#: Short, latency-sensitive timeout (seconds) for the ``/v1/models`` liveness
#: probe. A doctor preflight must not stall on a wedged / down seat.
_V1_MODELS_TIMEOUT_SECONDS = 2.0


class LocalSynthTopology(NamedTuple):
    """Resolved local-synthesis serving topology (never-raising resolution).

    - ``seat_registry_configured``: ED4ALL_SEAT_BASE_URLS or ED4ALL_VLLM_CONTAINERS
      is non-empty (a vLLM seat topology is declared).
    - ``backend``: ``"ollama"`` (legacy default) | ``"vllm"`` (LOCAL_SYNTHESIS_BASE_URL
      resolves to a vLLM seat).
    - ``base_url_root``: the resolved local-synthesis base URL in ROOT form
      (trailing ``/`` and ``/v1`` stripped) — the endpoint the checks probe.
    - ``seat_name``: the registered logical seat name backing ``base_url_root``
      when it matched a registry entry, else ``None`` (e.g. a non-ollama base URL
      that is not in the seat registry — still treated as vLLM).
    - ``seats`` / ``containers``: the parsed registries (for the seat group).
    """

    seat_registry_configured: bool
    backend: str
    base_url_root: str
    seat_name: Optional[str]
    seats: Dict[str, str]
    containers: Dict[str, str]


def _normalize_root(url: Optional[str]) -> str:
    """Normalize a base URL to ROOT form: strip a trailing ``/`` then a ``/v1``.

    Mirrors :func:`lib.llm.vram_reclaim.resolve_ollama_root`'s ``/v1`` stripping
    so a ``.../v1`` client base URL and a ``root`` seat-registry base URL compare
    equal. Empty / ``None`` → ``""``.
    """
    u = (str(url).strip() if url else "").rstrip("/")
    if u.endswith("/v1"):
        u = u[: -len("/v1")].rstrip("/")
    return u


def _parse_seat_registries() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Best-effort parse of the two seat registries (never raises → ``({}, {})``)."""
    try:
        from lib.vllm_container_lifecycle import (  # lazy — cheap stdlib module
            parse_container_registry,
            parse_seat_registry,
        )

        return parse_seat_registry(), parse_container_registry()
    except Exception as exc:  # noqa: BLE001 — a registry import must not crash the doctor
        logger.debug("run_env: seat registry parse unavailable: %s", exc)
        return {}, {}


def resolve_local_synthesis_topology() -> LocalSynthTopology:
    """Resolve whether local synthesis is served by ollama or a vLLM seat.

    Reads the CURRENT ``os.environ`` (LOCAL_SYNTHESIS_BASE_URL + the seat
    registries). Backend rules:

    * No seat registry configured → ``ollama`` (legacy, byte-identical).
    * Seat registry configured AND the resolved local-synthesis root matches a
      registered seat/container base URL → ``vllm`` (with the matched seat name).
    * Seat registry configured AND the resolved root is a NON-ollama endpoint
      (not port 11434) → ``vllm`` (unregistered seat; ``seat_name=None``).
    * Otherwise (e.g. seat registry configured but LOCAL_SYNTHESIS_BASE_URL still
      points at ollama on 11434 — a mixed topology) → ``ollama``.

    NEVER raises.
    """
    seats, containers = _parse_seat_registries()
    seat_registry_configured = bool(seats) or bool(containers)

    raw = _env("LOCAL_SYNTHESIS_BASE_URL")
    # The ollama default when unset (the `local` endpoint registry base_url).
    local_root = _normalize_root(raw) if raw else f"http://localhost{_OLLAMA_DEFAULT_PORT_TOKEN}"

    # Map registered seat/container base URLs (root form) → seat name (or None).
    registered: Dict[str, Optional[str]] = {}
    for name, burl in seats.items():
        registered.setdefault(_normalize_root(burl), name)
    for burl in containers:
        registered.setdefault(_normalize_root(burl), None)

    backend = "ollama"
    seat_name: Optional[str] = None
    if seat_registry_configured:
        if local_root in registered:
            backend, seat_name = "vllm", registered[local_root]
        elif _OLLAMA_DEFAULT_PORT_TOKEN not in local_root:
            backend, seat_name = "vllm", None

    return LocalSynthTopology(
        seat_registry_configured=seat_registry_configured,
        backend=backend,
        base_url_root=local_root,
        seat_name=seat_name,
        seats=dict(seats),
        containers=dict(containers),
    )


def probe_v1_models(
    base_url_root: str, *, timeout: float = _V1_MODELS_TIMEOUT_SECONDS
) -> Tuple[bool, List[str], Optional[str]]:
    """Probe ``GET {base_url_root}/v1/models`` (stdlib urllib, bounded, never-raise).

    Returns ``(live, model_ids, error)``: ``live`` is True on a 2xx with a parsed
    body; ``model_ids`` are the served model ids (``data[].id``); ``error`` is a
    short reason string on failure (down / refused / timeout / bad JSON), else
    ``None``. Uses ``urllib`` (no httpx hard-dep) with a short timeout so a
    wedged / down seat never stalls the doctor.
    """
    url = f"{str(base_url_root).rstrip('/')}/v1/models"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = int(getattr(resp, "status", None) or resp.getcode())
            body = _json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — seat down / refused / timeout / bad JSON
        logger.debug("run_env: /v1/models probe %s failed: %s", url, exc)
        return False, [], f"{type(exc).__name__}: {exc}"
    if not (200 <= code < 300):
        return False, [], f"HTTP {code}"
    model_ids: List[str] = []
    data = body.get("data") if isinstance(body, dict) else None
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and entry.get("id"):
                model_ids.append(str(entry["id"]))
    return True, model_ids, None


# ---------------------------------------------------------------------------
# Block-routing YAML resolution (two-pass tiers)
# ---------------------------------------------------------------------------


def _active_block_routing_path() -> Path:
    """Resolve the active ``block_routing.yaml`` path.

    ``COURSEFORGE_BLOCK_ROUTING_PATH`` (the fanout sets it, possibly
    repo-relative) wins; else the default
    ``Courseforge/config/block_routing.yaml``.
    """
    override = _env("COURSEFORGE_BLOCK_ROUTING_PATH")
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        return path
    return _REPO_ROOT / "Courseforge" / "config" / "block_routing.yaml"


def _load_routing_yaml() -> tuple[Optional[Dict[str, Any]], str]:
    """Best-effort load of the active routing YAML. ``(doc | None, path_str)``."""
    path = _active_block_routing_path()
    try:
        if not path.exists():
            return None, str(path)
        import yaml  # lazy

        with path.open("r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        if not isinstance(doc, dict):
            return None, str(path)
        return doc, str(path)
    except Exception as exc:
        logger.warning("run_env: routing YAML load failed at %s: %s", path, exc)
        return None, str(path)


def _resolve_entry_spec(
    entry: Any, capability_tiers: Dict[str, Any]
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve a per-tier entry into ``(provider, model, escalation_provider)``.

    Honors a direct ``provider`` / ``model`` pair, else a ``capability_tier``
    reference (string or cascade list). A cascade list resolves the
    **FIRST** tier (``chain[0]``) — the starting-point spec the router
    actually dispatches on (``Courseforge/router/policy.py::
    _project_capability_aware_spec`` resolves ``chain[0]`` and only walks the
    rest of the chain on budget-exhaustion escalation). The LAST tier
    (escalation target) is returned separately so the seat note can record
    "starts on X, escalates to Y" without lying about the effective provider.
    """
    if not isinstance(entry, dict):
        return None, None, None
    if entry.get("provider"):
        prov = str(entry["provider"]).lower()
        model = str(entry["model"]) if entry.get("model") else None
        return prov, model, None
    ct = entry.get("capability_tier")
    escalation: Optional[str] = None
    if isinstance(ct, list):
        chain = [c for c in ct if c]
        if not chain:
            return None, None, None
        if len(chain) > 1:
            esc_cap = capability_tiers.get(chain[-1]) or {}
            esc_prov = esc_cap.get("provider") if isinstance(esc_cap, dict) else None
            escalation = str(esc_prov).lower() if esc_prov else None
        ct = chain[0]
    if not ct:
        return None, None, None
    cap = capability_tiers.get(ct) or {}
    if not isinstance(cap, dict):
        return None, None, None
    prov = cap.get("provider")
    model = cap.get("model")
    return (
        str(prov).lower() if prov else None,
        str(model) if model else None,
        escalation,
    )


def _resolve_entry_provider(
    entry: Any, capability_tiers: Dict[str, Any]
) -> Optional[str]:
    """Resolve a per-tier entry's provider (the FIRST/starting tier in a
    cascade list — matches the router's ``chain[0]`` dispatch)."""
    return _resolve_entry_spec(entry, capability_tiers)[0]


def _yaml_tier_spec(
    doc: Optional[Dict[str, Any]], tier: str
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve the ``defaults[tier]`` ``(provider, model, escalation)`` from the
    active YAML doc (cascade lists resolve the starting ``chain[0]`` tier)."""
    if not isinstance(doc, dict):
        return None, None, None
    caps = doc.get("capability_tiers") or {}
    defaults = doc.get("defaults") or {}
    entry = defaults.get(tier) if isinstance(defaults, dict) else None
    return _resolve_entry_spec(entry, caps if isinstance(caps, dict) else {})


def _yaml_tier_heterogeneous(
    doc: Optional[Dict[str, Any]], tier: str
) -> Optional[List[str]]:
    """Return the sorted distinct provider set IF the active YAML pins
    heterogeneous providers per ``block_type`` for ``tier``, else ``None``.
    """
    if not isinstance(doc, dict):
        return None
    caps = doc.get("capability_tiers") or {}
    blocks = doc.get("blocks") or {}
    if not isinstance(blocks, dict):
        return None
    providers = set()
    for entry in blocks.values():
        if not isinstance(entry, dict):
            continue
        tier_entry = entry.get(tier)
        if not isinstance(tier_entry, dict):
            continue
        prov = _resolve_entry_provider(tier_entry, caps if isinstance(caps, dict) else {})
        if prov:
            providers.add(prov)
    if len(providers) > 1:
        return sorted(providers)
    return None


# ---------------------------------------------------------------------------
# Seat resolution
# ---------------------------------------------------------------------------


def _simple_seat(
    seat: str,
    provider_env: str,
    model_env: Optional[str],
    *,
    default: Optional[str],
    default_source: str,
    note: str = "",
) -> SeatResolution:
    """Resolve a single-env seat: env value (source ``env``) else the seat's
    documented default (``default_source``) — or ``None`` + ``off`` when there
    is no default.
    """
    env_val = _env(provider_env)
    model = _env(model_env) if model_env else None
    if env_val:
        return SeatResolution(
            seat=seat,
            provider_env=provider_env,
            effective_provider=env_val.lower(),
            model=model,
            source="env",
            note=note,
        )
    if default is None:
        return SeatResolution(
            seat=seat,
            provider_env=provider_env,
            effective_provider=None,
            model=model,
            source="off",
            note=note,
        )
    return SeatResolution(
        seat=seat,
        provider_env=provider_env,
        effective_provider=default,
        model=model,
        source=default_source,
        note=note,
    )


def _resolve_tier_model(
    provider: Optional[str],
    model_env: str,
    yaml_model: Optional[str],
    key_table: Dict[str, Dict[str, Any]],
) -> tuple[Optional[str], str]:
    """Resolve the tier model the way the router actually would.

    The router gates the tier model env (``COURSEFORGE_OUTLINE_MODEL`` /
    ``COURSEFORGE_REWRITE_MODEL``) on ``spec.provider == 'local'``
    (``Courseforge/router/router.py``), so:

    * **local** tier → the tier model env applies (returned verbatim).
    * **cloud** tier → the tier model env is DEAD; the model comes from the
      active YAML tier ``model:`` field, else the provider's registry model
      env (``NVIDIA_LARGE_MODEL`` for ``nvidia``), else the registry default.
      Never presents the dead tier-env value as the effective model.
    * unresolved provider → no model.
    """
    if not provider:
        return None, ""
    if provider == "local":
        # Router applies the tier model env on local tiers (overriding the
        # YAML tier model), falling back to the YAML model when env is unset.
        return _env(model_env) or yaml_model, ""

    # Cloud / capability tier — tier model env is dead.
    dead = f"{model_env} is dead on the cloud/capability tier"
    if yaml_model:
        return yaml_model, (
            f"{dead}; model resolved from active block-routing YAML tier "
            f"model: (not {model_env})"
        )
    row = key_table.get(provider, {}) if isinstance(key_table, dict) else {}
    prov_model_env = row.get("model_env")
    env_model = _env(prov_model_env) if prov_model_env else None
    if env_model:
        return env_model, (
            f"{dead}; model resolved from {prov_model_env} (not {model_env})"
        )
    return row.get("default_model"), (
        f"{dead}; model resolved from {provider} registry default "
        f"(not {model_env})"
    )


def _resolve_tier(
    seat: str,
    provider_env: str,
    model_env: str,
    tier: str,
    doc: Optional[Dict[str, Any]],
    key_table: Dict[str, Dict[str, Any]],
) -> SeatResolution:
    """Resolve a two-pass tier: tier env > active-YAML tier default > hardcoded.

    Reports at TIER granularity, flags heterogeneous per-block_type provider
    pins in the active YAML (scanning ``chain[0]`` providers), and resolves the
    model via the router's local-only tier-model-env gate (see
    :func:`_resolve_tier_model`).
    """
    env_val = _env(provider_env)
    notes: List[str] = []

    yaml_provider, yaml_model, escalation = _yaml_tier_spec(doc, tier)

    if env_val:
        provider, source = env_val.lower(), "env"
    elif yaml_provider:
        provider, source = yaml_provider, "yaml"
        het = _yaml_tier_heterogeneous(doc, tier)
        if het:
            notes.append(
                f"active block-routing YAML pins heterogeneous {tier} "
                f"providers per block_type ({', '.join(het)}); reported "
                f"value is the tier default"
            )
    else:
        provider, source = _HARDCODED_TIER_PROVIDER.get(tier), "class_default"

    # Escalation provenance only when the ACTIVE tier default is a cascade.
    if source == "yaml" and escalation and escalation != provider:
        notes.append(
            f"starts on {provider}, escalates to {escalation} on budget exhaustion"
        )

    # Model attribution: the YAML model only describes the YAML-resolved
    # provider, so only feed it through when the provider also came from YAML.
    model, model_note = _resolve_tier_model(
        provider,
        model_env,
        yaml_model if source == "yaml" else None,
        key_table,
    )
    if model_note:
        notes.append(model_note)

    return SeatResolution(
        seat=seat,
        provider_env=provider_env,
        effective_provider=provider,
        model=model,
        source=source,
        note=_join_notes(*notes),
    )


def resolve_seats(two_pass: bool) -> List[SeatResolution]:
    """Resolve every authoring seat off the CURRENT ``os.environ``.

    The caller applies the fanout (:func:`applied_run_env`) FIRST when modelling
    a real run. Two-pass tiers (``outline`` / ``rewrite``) resolve only when
    ``two_pass`` is truthy; the single-pass ``content`` seat is always reported.
    Never raises.
    """
    seats: List[SeatResolution] = []

    # --- Single-pass content seat (always reported) --------------------
    seats.append(
        _simple_seat(
            "content",
            "COURSEFORGE_PROVIDER",
            None,
            default="anthropic",
            default_source="class_default",
            note=(
                "unset -> legacy deterministic-template path (no LLM); "
                "class default is 'anthropic'"
            ),
        )
    )

    # --- Two-pass tiers ------------------------------------------------
    if two_pass:
        doc, _path = _load_routing_yaml()
        key_table = load_provider_key_table()
        seats.append(
            _resolve_tier(
                "outline",
                "COURSEFORGE_OUTLINE_PROVIDER",
                "COURSEFORGE_OUTLINE_MODEL",
                "outline",
                doc,
                key_table,
            )
        )
        seats.append(
            _resolve_tier(
                "rewrite",
                "COURSEFORGE_REWRITE_PROVIDER",
                "COURSEFORGE_REWRITE_MODEL",
                "rewrite",
                doc,
                key_table,
            )
        )

    # --- Course outliner ------------------------------------------------
    seats.append(
        _simple_seat(
            "course_outline",
            "COURSEPLANNER_PROVIDER",
            "COURSEPLANNER_MODEL",
            default="anthropic",
            default_source="class_default",
        )
    )

    # --- Assessment -----------------------------------------------------
    seats.append(
        _simple_seat(
            "assessment",
            "TRAINFORGE_ASSESSMENT_PROVIDER",
            "TRAINFORGE_ASSESSMENT_MODEL",
            default="anthropic",
            default_source="class_default",
        )
    )

    # --- Training synthesis (honest effective default) -----------------
    ts_env = _env("TRAINFORGE_SYNTHESIS_PROVIDER")
    ts_model = _env("TRAINFORGE_SYNTHESIS_MODEL")
    if ts_env:
        ts_note = ""
        if ts_env.lower() in _ANTHROPIC_FAMILY:
            ts_note = (
                "Anthropic-family training synthesis FAILS CLOSED "
                "(SynthesisLicensingError) unless "
                "TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS=true"
            )
        seats.append(
            SeatResolution(
                seat="training_synthesis",
                provider_env="TRAINFORGE_SYNTHESIS_PROVIDER",
                effective_provider=ts_env.lower(),
                model=ts_model,
                source="env",
                note=ts_note,
            )
        )
    else:
        seats.append(
            SeatResolution(
                seat="training_synthesis",
                provider_env="TRAINFORGE_SYNTHESIS_PROVIDER",
                effective_provider="mock",
                model=ts_model,
                source="class_default",
                note=(
                    "kwarg default is 'mock' (NOT a real provider); pipeline "
                    "fanout setdefaults this to local-after-fanout; "
                    "Anthropic-family fails closed without "
                    "TRAINFORGE_ALLOW_ANTHROPIC_SYNTHESIS=true"
                ),
            )
        )

    # --- Textbook synthesis (workflow default OFF) ----------------------
    seats.append(
        _simple_seat(
            "textbook_synthesis",
            "TEXTBOOK_SYNTHESIS_PROVIDER",
            "TEXTBOOK_SYNTHESIS_MODEL",
            default=None,
            default_source="off",
            note="unset -> three-stage textbook synthesis phase short-circuits",
        )
    )

    # --- Grounded answer (default local, loopback-enforced) -------------
    seats.append(
        _simple_seat(
            "answer",
            "ED4ALL_ANSWER_PROVIDER",
            "ED4ALL_ANSWER_MODEL",
            default="local",
            default_source="class_default",
            note="LOOPBACK-ENFORCED — a non-loopback base_url raises AnswerProviderNotLocal",
        )
    )

    # --- Embedding (separate registry, default 'st') --------------------
    seats.append(
        _simple_seat(
            "embedding",
            "ED4ALL_EMBEDDING_PROVIDER",
            "ED4ALL_EMBEDDING_MODEL",
            default="st",
            default_source="class_default",
            note="separate embedding registry (lib/embedding/providers.py), NOT the LLM endpoint registry",
        )
    )

    # --- Objective review (default off) ---------------------------------
    seats.append(
        _simple_seat(
            "objective_review",
            "ED4ALL_OBJECTIVE_REVIEW_PROVIDER",
            "ED4ALL_OBJECTIVE_REVIEW_MODEL",
            default=None,
            default_source="off",
            note="unset -> objective-review pass is a strict no-op",
        )
    )

    # --- Dynamic block planner (gated by ED4ALL_DYNAMIC_BLOCK_PLAN) -----
    bp_env = _env("ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER")
    bp_model = _env("ED4ALL_DYNAMIC_BLOCK_PLAN_MODEL")
    if not _truthy("ED4ALL_DYNAMIC_BLOCK_PLAN"):
        seats.append(
            SeatResolution(
                seat="block_planner",
                provider_env="ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER",
                effective_provider=None,
                model=bp_model,
                source="off",
                note="feature gated off (ED4ALL_DYNAMIC_BLOCK_PLAN not truthy)",
            )
        )
    else:
        seats.append(
            SeatResolution(
                seat="block_planner",
                provider_env="ED4ALL_DYNAMIC_BLOCK_PLAN_PROVIDER",
                effective_provider=(bp_env or "nvidia").lower(),
                model=bp_model,
                source="env" if bp_env else "class_default",
                note="" if bp_env else "default seat is 'nvidia' (set =local for the license-clean Qwen seat)",
            )
        )

    return seats


# ---------------------------------------------------------------------------
# Run-env fanout (snapshot/restore, exception-safe)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def applied_run_env(workflow: str, provider_hint: str = "") -> Iterator[Dict[str, str]]:
    """Model the env a hypothetical ``ed4all run <workflow>`` would apply.

    Snapshots ``os.environ``, lazily constructs
    ``WorkflowRunner(executor=None, config=OrchestratorConfig.load())``, calls
    ``_apply_corpus_generalization_defaults(workflow)`` THEN
    ``_apply_authoring_route_env(workflow, provider_hint)`` in that order
    (mirroring ``run_workflow``), and yields the merged applied-deltas dict.

    ALWAYS restores ``os.environ`` in ``finally`` (clear + update from the
    snapshot) — even when the ``with``-body raises and even when the runner
    construction / config load fails (in which case ``{}`` is yielded).
    """
    snapshot = dict(os.environ)
    applied: Dict[str, str] = {}
    try:
        try:
            from MCP.core.config import OrchestratorConfig  # lazy
            from MCP.core.workflow_runner import WorkflowRunner  # lazy

            runner = WorkflowRunner(executor=None, config=OrchestratorConfig.load())
            corpus = runner._apply_corpus_generalization_defaults(workflow) or {}
            authoring = runner._apply_authoring_route_env(workflow, provider_hint) or {}
            applied = {**corpus, **authoring}
        except Exception as exc:
            logger.warning("run_env: applied_run_env setup failed: %s", exc)
            applied = {}
        yield applied
    finally:
        os.environ.clear()
        os.environ.update(snapshot)

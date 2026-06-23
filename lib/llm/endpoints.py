"""Unified LLM/API endpoint registry — loader + generic attach point.

Single source of truth: ``config/endpoints.yaml`` (schema
``schemas/config/endpoints.schema.json``). Every LLM/API endpoint the
codebase attaches to is one named row in that file; code attaches BY
NAME and the transport (``kind``) + identity (base_url, model, api_key
env, provenance value) flow from the row. Adding a new endpoint is a
ONE-ROW change to the YAML plus running the provenance codegen — zero
new code, zero new provider classes.

This module owns:

- :func:`load_endpoint_registry` — parse + schema-validate the YAML once
  (cached).
- :func:`endpoint_names` — the registry's endpoint name tuple.
- :func:`provenance_provider_names` — THE single authority the closed
  ``Touch.provider`` enum derives from (union of every row's
  ``provenance_provider`` + the ``"deterministic"`` sentinel). The three
  static provenance sites (blocks.py, the JSON-LD schema, the SHACL
  shapes) are kept in sync with this via
  ``scripts/codegen/sync_provenance_enum.py`` + a drift test.
- :func:`resolve_endpoint` — resolve one endpoint's effective config
  with the frozen precedence ``explicit kwarg > *_env var > registry
  default``.
- :func:`build_openai_compatible_client` — the ONE constructor that
  replaces every per-vendor ``_base.py`` if/elif branch +
  ``objective_review._build_review_client``: resolve an
  ``openai_compatible`` endpoint and build an
  ``OpenAICompatibleClient``.

Import hygiene (anti-cycle): this module imports only stdlib + ``yaml``
+ ``jsonschema`` + ``lib.paths``. It must NEVER import
``Courseforge.scripts.blocks`` / the router / a provider module — those
import FROM here (one direction). The ``OpenAICompatibleClient`` import
inside :func:`build_openai_compatible_client` is lazy (function-local)
so a bare ``import lib.llm.endpoints`` stays transport-free.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import yaml

from lib.paths import SCHEMAS_PATH, get_endpoints_path

__all__ = [
    "EndpointRegistryError",
    "UnknownEndpoint",
    "EndpointKeyRequired",
    "EndpointKindError",
    "ResolvedEndpoint",
    "load_endpoint_registry",
    "endpoint_names",
    "openai_compatible_legacy_registry",
    "provenance_provider_names",
    "resolve_endpoint",
    "build_openai_compatible_client",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EndpointRegistryError(RuntimeError):
    """Base error for the endpoint registry (parse / schema failures)."""


class UnknownEndpoint(EndpointRegistryError):
    """Raised when an endpoint name isn't in the registry."""


class EndpointKeyRequired(EndpointRegistryError):
    """Raised when an ``api_key_required`` endpoint resolves no key."""


class EndpointKindError(EndpointRegistryError):
    """Raised when an endpoint's ``kind`` doesn't match the call site.

    e.g. :func:`build_openai_compatible_client` against a
    ``claude_session`` row (no HTTP endpoint to build a client for).
    """


# Sentinel provenance value that is NOT an endpoint row — appended by
# :func:`provenance_provider_names`. Covers the Phase-1 deterministic
# templating paths that emit a ``Touch`` with no LLM behind it.
_DETERMINISTIC_PROVENANCE = "deterministic"


# ---------------------------------------------------------------------------
# Resolved view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedEndpoint:
    """Effective config for one endpoint after env + override resolution.

    Frozen — callers treat it as an immutable resolved snapshot. The
    resolution precedence per field is ``explicit kwarg > endpoint's
    *_env var > registry default``.
    """

    name: str
    kind: str
    base_url: Optional[str]
    api_key: Optional[str]
    model: str
    api_key_required: bool
    loopback_only: bool
    provenance_provider: str
    vision_capable: bool
    # OPTIONAL per-endpoint request-body extras (e.g.
    # ``{"chat_template_kwargs": {"thinking": false}}`` for a reasoning
    # model that must suppress chain-of-thought). Absent on the row →
    # ``None`` → byte-stable (no extras forwarded). A consumer forwards
    # this verbatim as the ``OpenAICompatibleClient``'s per-call
    # ``extra_payload`` so the keys land at the top level of the request
    # body. Schema field: ``schemas/config/endpoints.schema.json::
    # properties.extra_body``.
    extra_body: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Loader (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _load_validated(path_str: str) -> Dict[str, Any]:
    """Parse + schema-validate the YAML at ``path_str`` (cache key = path).

    Cached on the resolved path string so a test that monkeypatches
    ``ED4ALL_ENDPOINTS_PATH`` to a tmp file gets its own cache entry
    without colliding with the shipped registry.
    """
    try:
        raw_text = _read_text(path_str)
    except OSError as exc:
        raise EndpointRegistryError(
            f"endpoint registry not readable at {path_str!r}: {exc}"
        ) from exc
    try:
        doc = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise EndpointRegistryError(
            f"endpoint registry at {path_str!r} is not valid YAML: {exc}"
        ) from exc
    if not isinstance(doc, dict):
        raise EndpointRegistryError(
            f"endpoint registry at {path_str!r} must be a mapping, "
            f"got {type(doc).__name__}"
        )

    _validate_against_schema(doc, path_str)
    endpoints = doc.get("endpoints") or {}
    # Return a deep-ish copy so cached callers can't mutate the cache.
    return {name: dict(row) for name, row in endpoints.items()}


def _read_text(path_str: str) -> str:
    with open(path_str, "r", encoding="utf-8") as fh:
        return fh.read()


def _validate_against_schema(doc: Dict[str, Any], path_str: str) -> None:
    """Validate ``doc`` against the endpoints JSON Schema (fail-loud)."""
    schema_path = SCHEMAS_PATH / "config" / "endpoints.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except OSError as exc:  # pragma: no cover — defensive
        raise EndpointRegistryError(
            f"endpoint schema not readable at {schema_path}: {exc}"
        ) from exc
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover — jsonschema is a hard dep
        raise EndpointRegistryError(
            "jsonschema is required to load the endpoint registry"
        ) from exc
    try:
        jsonschema.validate(instance=doc, schema=schema)
    except jsonschema.ValidationError as exc:
        raise EndpointRegistryError(
            f"endpoint registry at {path_str!r} failed schema validation: "
            f"{exc.message} (at {'/'.join(str(p) for p in exc.absolute_path)})"
        ) from exc


def load_endpoint_registry() -> Dict[str, Dict[str, Any]]:
    """Return the parsed + schema-validated endpoint map (cached).

    Keyed by endpoint name; values are the raw row dicts (env-var names
    and defaults — NOT yet resolved). Path is resolved fresh each call
    via :func:`lib.paths.get_endpoints_path` (honoring
    ``ED4ALL_ENDPOINTS_PATH`` / ``ED4ALL_HOME``), but the parse +
    validation result is cached per resolved path.
    """
    path_str = str(get_endpoints_path())
    return _load_validated(path_str)


def endpoint_names() -> Tuple[str, ...]:
    """Return every registered endpoint name, registry order preserved."""
    return tuple(load_endpoint_registry().keys())


def openai_compatible_legacy_registry() -> Dict[str, Dict[str, Any]]:
    """Project the ``openai_compatible`` rows into the W-D12 legacy shape.

    The historical ``MCP/orchestrator/llm_backend.py::
    _OPENAI_COMPATIBLE_PROVIDERS`` dict (and every consumer of it — the
    answer path, the Courseforge / Trainforge synthesis providers,
    ``gui/env_catalog.py``) keys on a fixed per-entry shape
    (``base_url_env`` / ``base_url_default`` / ``api_key_env`` /
    ``api_key_default`` / ``model_env`` / ``model_default`` /
    ``api_key_required`` + optional ``vision_capable`` /
    ``vision_capable_env`` / ``unverified``). This builds that exact
    shape from ``config/endpoints.yaml`` so the YAML is the SINGLE source
    of truth — ``_OPENAI_COMPATIBLE_PROVIDERS`` becomes a projection, not
    a second hand-maintained registry. Only ``kind=openai_compatible``
    rows project (the ``anthropic`` SDK + ``claude_session`` dispatcher
    rows are not OpenAI-wire endpoints).

    A fresh dict (with per-row copies) is returned each call so a caller
    can hand it to ``monkeypatch.setitem`` / mutate it without poisoning
    the cached registry.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for name, row in load_endpoint_registry().items():
        if str(row.get("kind")) != "openai_compatible":
            continue
        entry: Dict[str, Any] = {
            "base_url_env": row.get("base_url_env"),
            "base_url_default": row.get("base_url"),
            "api_key_env": row.get("api_key_env"),
            "api_key_default": row.get("api_key_default"),
            "model_env": row.get("model_env"),
            "model_default": row.get("default_model"),
            "api_key_required": bool(row.get("api_key_required", False)),
        }
        # Optional fields only surface when present on the row so the
        # projection stays byte-equal to the historical literal for
        # entries that never declared them (e.g. ``together`` had no
        # ``vision_capable_env``; the stub providers carry ``unverified``).
        if "vision_capable" in row:
            entry["vision_capable"] = bool(row.get("vision_capable", False))
        if "vision_capable_env" in row:
            entry["vision_capable_env"] = row.get("vision_capable_env")
        if "unverified" in row:
            entry["unverified"] = bool(row.get("unverified", False))
        # OPTIONAL per-endpoint request-body extras (e.g. a reasoning
        # model's ``chat_template_kwargs:{thinking:false}``). Surfaced
        # only when present + a mapping so the projection stays byte-equal
        # to the historical literal for rows that never declared it.
        if isinstance(row.get("extra_body"), dict):
            entry["extra_body"] = dict(row["extra_body"])
        out[name] = entry
    return out


def provenance_provider_names() -> Tuple[str, ...]:
    """Return the closed ``Touch.provider`` set, sorted.

    THE single authority the three static provenance sites derive from:
    the sorted union of every endpoint row's ``provenance_provider``
    plus the ``"deterministic"`` sentinel (which is NOT an endpoint row —
    it covers Phase-1 deterministic templating Touches). Keeping this a
    closed, derived set is the governance contract: adding an endpoint
    whose ``provenance_provider`` is a NEW value grows this set, and the
    codegen + drift test then propagate it to blocks.py + the JSON-LD
    schema + the SHACL shapes in lockstep.
    """
    registry = load_endpoint_registry()
    values = {str(row["provenance_provider"]) for row in registry.values()}
    values.add(_DETERMINISTIC_PROVENANCE)
    return tuple(sorted(values))


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _env_value(env_name: Optional[str]) -> Optional[str]:
    """Read a non-empty env var by name, or ``None``."""
    if not env_name:
        return None
    raw = os.environ.get(env_name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _resolve_truthy(raw: Optional[str]) -> Optional[bool]:
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_endpoint(
    name: str,
    *,
    model_override: Optional[str] = None,
    api_key_override: Optional[str] = None,
    base_url_override: Optional[str] = None,
) -> ResolvedEndpoint:
    """Resolve ``name`` to an effective :class:`ResolvedEndpoint`.

    Per-field precedence (frozen): ``explicit kwarg > endpoint's *_env
    var > registry default``.

    Raises:
        UnknownEndpoint: ``name`` is not in the registry (message lists
            the sorted known names).
        EndpointKeyRequired: the row is ``api_key_required`` and no key
            resolves from override or env (the ``api_key_default`` floor,
            when present, satisfies the requirement — only ``local`` uses
            it).
    """
    registry = load_endpoint_registry()
    row = registry.get(name)
    if row is None:
        raise UnknownEndpoint(
            f"unknown endpoint {name!r}; registered endpoints: "
            f"{sorted(registry.keys())}"
        )

    kind = str(row["kind"])

    # --- base_url -------------------------------------------------------
    base_url = base_url_override
    if base_url is None:
        base_url = _env_value(row.get("base_url_env"))
    if base_url is None:
        base_url = row.get("base_url")  # may be None for SDK / dispatcher kinds
    if base_url is not None:
        base_url = str(base_url).rstrip("/")

    # --- model ----------------------------------------------------------
    model = model_override
    if model is None:
        model = _env_value(row.get("model_env"))
    if model is None:
        model = str(row["default_model"])

    # --- api_key --------------------------------------------------------
    api_key = api_key_override
    if api_key is None:
        api_key = _env_value(row.get("api_key_env"))
    if api_key is None:
        default_key = row.get("api_key_default")
        api_key = str(default_key) if default_key else None

    api_key_required = bool(row["api_key_required"])
    if api_key_required and not api_key:
        key_env = row.get("api_key_env") or "<no api_key_env>"
        raise EndpointKeyRequired(
            f"endpoint {name!r} requires an API key; set {key_env} or pass "
            f"api_key_override (tests inject a client instead)."
        )

    # --- vision_capable -------------------------------------------------
    vision = bool(row.get("vision_capable", False))
    vision_env = _resolve_truthy(_env_value(row.get("vision_capable_env")))
    if vision_env is not None:
        vision = vision_env

    # --- extra_body -----------------------------------------------------
    # OPTIONAL request-body extras. Surfaced verbatim from the row when
    # present and a mapping; otherwise ``None`` (byte-stable — no extras).
    # Copied so a caller mutating the resolved view can't poison the row.
    raw_extra_body = row.get("extra_body")
    extra_body = dict(raw_extra_body) if isinstance(raw_extra_body, dict) else None

    return ResolvedEndpoint(
        name=name,
        kind=kind,
        base_url=base_url,
        api_key=api_key,
        model=str(model),
        api_key_required=api_key_required,
        loopback_only=bool(row.get("loopback_only", False)),
        provenance_provider=str(row["provenance_provider"]),
        vision_capable=vision,
        extra_body=extra_body,
    )


# ---------------------------------------------------------------------------
# Generic attach point — the ONE openai-compatible constructor
# ---------------------------------------------------------------------------


def build_openai_compatible_client(
    name: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    json_mode: bool = True,
    timeout: Optional[float] = None,
    capture: Optional[Any] = None,
    client: Optional[Any] = None,
    provider_label: Optional[str] = None,
):
    """Resolve an ``openai_compatible`` endpoint and build its client.

    The ONE constructor that replaces every per-vendor ``_base.py``
    if/elif branch + ``objective_review._build_review_client``. Consumers
    pass an endpoint NAME; the transport + identity flow from the row.

    ``model`` / ``api_key`` / ``base_url`` are explicit overrides threaded
    into :func:`resolve_endpoint` (kwarg precedence). ``provider_label``
    defaults to the endpoint name (the audit string the
    ``OpenAICompatibleClient`` surfaces in decision-capture rationales).
    ``client`` injects a pre-built ``httpx.Client`` (tests / DI) and, when
    supplied, suppresses the ``api_key_required`` fail-loud (a test client
    needs no real key).

    Raises:
        EndpointKindError: the endpoint's ``kind`` is not
            ``openai_compatible`` (e.g. ``anthropic`` SDK or
            ``claude_session`` dispatcher — those aren't HTTP endpoints).
        UnknownEndpoint / EndpointKeyRequired: per :func:`resolve_endpoint`
            (the key check is skipped when a ``client`` is injected).
    """
    registry = load_endpoint_registry()
    row = registry.get(name)
    if row is None:
        raise UnknownEndpoint(
            f"unknown endpoint {name!r}; registered endpoints: "
            f"{sorted(registry.keys())}"
        )
    kind = str(row["kind"])
    if kind != "openai_compatible":
        raise EndpointKindError(
            f"endpoint {name!r} has kind={kind!r}; "
            f"build_openai_compatible_client requires kind='openai_compatible'. "
            f"({'use the Anthropic SDK path' if kind == 'anthropic' else 'this is a LocalDispatcher subagent, not an HTTP endpoint'})"
        )

    # When a client is injected (tests / DI) the real key is not needed —
    # mirror the per-vendor providers' "inject a client" escape hatch by
    # supplying an api_key override so the required-key check passes.
    resolve_key_override = api_key
    if resolve_key_override is None and client is not None:
        resolve_key_override = _env_value(row.get("api_key_env")) or "injected"

    resolved = resolve_endpoint(
        name,
        model_override=model,
        api_key_override=resolve_key_override,
        base_url_override=base_url,
    )

    if not resolved.base_url:  # pragma: no cover — schema guarantees non-null
        raise EndpointKindError(
            f"endpoint {name!r} resolved no base_url; cannot build an "
            f"OpenAI-compatible client."
        )

    # Lazy import keeps a bare `import lib.llm.endpoints` transport-free.
    from Trainforge.generators._openai_compatible_client import (
        OpenAICompatibleClient,
    )

    return OpenAICompatibleClient(
        base_url=resolved.base_url,
        model=resolved.model,
        api_key=resolved.api_key,
        capture=capture,
        timeout=timeout,
        provider_label=provider_label or name,
        client=client,
        json_mode=json_mode,
        vision_capable=resolved.vision_capable,
    )

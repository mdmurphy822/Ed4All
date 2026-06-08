"""Canonical settings persistence — ``state/gui/settings.json`` + env render.

This is the single source of truth for the GUI's env/API-key/model-routing
state. MCP tools and the GUI backend both read/write through here so a Claude
Code session and the GUI agree on what's configured.

No web deps: this module imports cleanly without FastAPI installed (it depends
only on ``gui.shared_state`` + ``gui.env_catalog`` + stdlib).

Settings doc schema (versioned) — see build spec §3:

    {
      "version": 1,
      "updated_at": "<iso>",
      "env": { "ANTHROPIC_API_KEY": "...", "LLM_MODE": "api", ... },
      "model_routing": { "global": {...}, "dart": {...}, ... },
      "retrieval": { "top_k": 5, "min_grounding_cosine": 0.45, ... },
      "flags": { "COURSEFORGE_TWO_PASS": false, ... }
    }
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

from gui import env_catalog, shared_state

# Secret values surfaced through GET are masked to one of these.
_SECRET_SET = "set"
_SECRET_UNSET = None


def _is_secret_key(key: str) -> bool:
    """True for env keys whose value is a credential (``*_API_KEY`` / ``*_KEY``)."""
    upper = key.upper()
    return upper.endswith("_API_KEY") or upper.endswith("_KEY")


# --------------------------------------------------------------------- load/save


def load_settings() -> Dict[str, Any]:
    """Return the persisted settings doc, or fresh defaults if none exists.

    The on-disk doc is overlaid onto the catalog defaults so a doc written by
    an older schema still carries every expected key.
    """
    path = shared_state.settings_path()
    defaults = env_catalog.default_settings()
    if not path.exists():
        return defaults
    try:
        import json  # noqa: PLC0415

        on_disk = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return defaults
    if not isinstance(on_disk, dict):
        return defaults
    return _deep_merge(defaults, on_disk)


def save_settings(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Validate, stamp ``updated_at``, and atomically persist the full doc.

    Returns the stored (unmasked) doc. Raises ``ValueError`` on validation
    failure (unknown provider, bad mode, malformed structure).
    """
    if not isinstance(doc, dict):
        raise ValueError("settings doc must be a dict")
    validated = _validate(doc)
    validated["version"] = int(validated.get("version", 1) or 1)
    validated["updated_at"] = shared_state.now_iso()
    shared_state._atomic_write_json(shared_state.settings_path(), validated)
    return validated


def update_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge ``patch`` into the current doc, then save. Returns stored doc."""
    if not isinstance(patch, dict):
        raise ValueError("patch must be a dict")
    current = load_settings()
    merged = _deep_merge(current, patch)
    return save_settings(merged)


# ---------------------------------------------------------------------- render


def render_env(doc: Dict[str, Any]) -> Dict[str, str]:
    """Flatten a settings doc to an env var map.

    Sources, in increasing precedence:
    1. ``flags`` -> bool env vars (lowercased "true"/"false").
    2. ``retrieval.require_embeddings`` -> ``TRAINFORGE_REQUIRE_EMBEDDINGS``.
    3. ``model_routing.<task>.<field>`` -> the canonical env var
       (``env_catalog.routing_to_env``).
    4. ``env`` -> raw passthrough of catalog-known keys.

    Null / empty values are dropped. Later sources win so an explicit ``env``
    value overrides a routing-derived one.
    """
    out: Dict[str, str] = {}
    if not isinstance(doc, dict):
        return out

    # 1. flags (bool) — only known catalog keys; emit lowercase true/false.
    flags = doc.get("flags") or {}
    if isinstance(flags, dict):
        for key, value in flags.items():
            if value is None:
                continue
            out[key] = "true" if _as_bool(value) else "false"

    # 2. retrieval.require_embeddings mirrors the trainforge flag.
    retrieval = doc.get("retrieval") or {}
    if isinstance(retrieval, dict) and retrieval.get("require_embeddings") is not None:
        out["TRAINFORGE_REQUIRE_EMBEDDINGS"] = (
            "true" if _as_bool(retrieval["require_embeddings"]) else "false"
        )

    # 3. model_routing -> env (provider/model/mode per task).
    out.update(env_catalog.routing_to_env(doc.get("model_routing") or {}))

    # 4. explicit env passthrough — only catalog-known keys, non-empty.
    known = set(env_catalog.catalog_keys())
    env = doc.get("env") or {}
    if isinstance(env, dict):
        for key, value in env.items():
            if key not in known:
                continue
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            out[key] = text

    return out


def apply_env(doc: Dict[str, Any]) -> Dict[str, str]:
    """Render ``doc``, write the resolved vars into ``os.environ``, and persist.

    The rendered map is written into ``os.environ`` (so launched runs inherit
    it) AND mirrored to ``state/gui/.env.rendered`` (informational). Returns the
    applied map (unmasked — caller masks for transport).
    """
    rendered = render_env(doc)
    for key, value in rendered.items():
        os.environ[key] = value
    _write_env_rendered(rendered)
    return rendered


def _write_env_rendered(rendered: Dict[str, str]) -> None:
    """Write ``state/gui/.env.rendered`` (dotenv form), atomically."""
    lines = ["# Rendered by gui.settings_store.apply_env — informational only.\n"]
    for key in sorted(rendered):
        value = rendered[key]
        # Mask secrets in the on-disk render so the file isn't a key leak.
        if _is_secret_key(key):
            value = _SECRET_SET
        lines.append(f"{key}={value}\n")
    shared_state._atomic_write_text(shared_state.env_rendered_path(), "".join(lines))


# ----------------------------------------------------------------- masking


def mask_secrets(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of ``doc`` with secret values replaced.

    Any ``env`` key matching ``*_API_KEY`` / ``*_KEY`` becomes ``"set"`` when a
    non-empty value is present, else ``None``. Raw values are write-only and
    never surface through a GET.
    """
    masked = copy.deepcopy(doc) if isinstance(doc, dict) else {}
    env = masked.get("env")
    if isinstance(env, dict):
        for key in list(env.keys()):
            if _is_secret_key(key):
                value = env[key]
                if value is None or (isinstance(value, str) and not value.strip()):
                    env[key] = _SECRET_UNSET
                else:
                    env[key] = _SECRET_SET
    return masked


# ---------------------------------------------------------------- validation


def _validate(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the doc against the catalog; return a normalized deep copy.

    Raises ``ValueError`` on: unknown ``LLM_MODE`` value, unknown provider in
    any routing/global provider slot, or unknown LLM mode in
    ``model_routing.global.mode``. Unknown ``env`` keys are dropped silently
    (forward-compat) rather than rejected.
    """
    normalized = _deep_merge(env_catalog.default_settings(), doc)
    valid_providers = set(env_catalog.provider_names())

    # global.mode must be local|api.
    routing = normalized.get("model_routing") or {}
    global_block = routing.get("global") or {}
    mode = global_block.get("mode")
    if mode is not None and mode not in ("local", "api"):
        raise ValueError(f"model_routing.global.mode must be 'local' or 'api', got {mode!r}")

    # Every provider slot across routing must name a registered provider.
    for task, block in routing.items():
        if not isinstance(block, dict):
            continue
        for field in ("provider", "vision_provider"):
            value = block.get(field)
            if value in (None, ""):
                continue
            if value not in valid_providers:
                raise ValueError(
                    f"model_routing.{task}.{field}={value!r} is not a registered "
                    f"provider (valid: {sorted(valid_providers)})"
                )

    # env.LLM_MODE / env.LLM_PROVIDER sanity (if set via the raw env block).
    env = normalized.get("env") or {}
    if isinstance(env, dict):
        llm_mode = env.get("LLM_MODE")
        if llm_mode not in (None, "") and llm_mode not in ("local", "api"):
            raise ValueError(f"env.LLM_MODE must be 'local' or 'api', got {llm_mode!r}")
        llm_provider = env.get("LLM_PROVIDER")
        if llm_provider not in (None, "") and llm_provider not in valid_providers:
            raise ValueError(
                f"env.LLM_PROVIDER={llm_provider!r} is not a registered provider"
            )

    return normalized


# --------------------------------------------------------------------- utils


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``patch`` over a deep copy of ``base``.

    Nested dicts merge key-by-key; every other value (including lists) is
    replaced wholesale by the patch value.
    """
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _as_bool(value: Any) -> bool:
    """Coerce a flag value to bool (accepts bools and truthy strings)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


__all__ = [
    "load_settings",
    "save_settings",
    "update_settings",
    "render_env",
    "apply_env",
    "mask_secrets",
]

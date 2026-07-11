"""Relocatable path resolution for the vendored SemantiK (Semantic v2) runtime.

Single source of truth for every MODEL / CACHE / DATA / CONFIG directory the
``semantik_structure`` package resolves at runtime. Mirrors Ed4All's
``lib/paths.py`` ``ED4ALL_HOME`` relocatable-root pattern so SemantiK loads
correctly under Ed4All's subagent / mailbox dispatch, where the current working
directory is unstable (migration plan risk R9).

The historic defaults were CWD-relative (``"models/..."`` / ``"data/..."``),
which silently break the moment the cascade is launched from a directory other
than the package root. This module replaces every such literal with a
PACKAGE-RELATIVE default so the resolved path no longer depends on the CWD.

Resolution precedence (per directory class), highest wins
---------------------------------------------------------
    1. Per-dir env override:
         ``SEMANTIK_MODEL_DIR``  (model weights / LoRA adapters / GGUFs)
         ``SEMANTIK_CACHE_DIR``  (extract / glm-ocr / prerender caches)
         ``SEMANTIK_DATA_DIR``   (eval reports, datasets, labels)
         ``SEMANTIK_CONFIG_DIR`` (calibration / config artifacts)
    2. ``SEMANTIK_HOME/<dirname>`` when ``SEMANTIK_HOME`` is set
       (relocatable data root; mirrors ``ED4ALL_HOME``).
    3. Package-relative default ``<SemantiK>/<dirname>`` — where
       ``<SemantiK>`` is the parent of the ``semantik_structure/`` package
       (``Path(__file__).resolve().parents[1]``).

When NO env var is set, every resolver returns the package-relative default,
which is byte-stable to the historic ``models/`` / ``data/`` layout *relative
to the package root* (only the CWD-dependence is removed — the on-disk layout
is unchanged). Existing tests / call paths that monkeypatch the module-level
``DEFAULT_ADAPTER_DIR`` constants keep working: those constants are now derived
from this resolver but remain ordinary module attributes.

Dev-runnability (env-only weights pointer)
------------------------------------------
The model weights are gitignored and are NOT present in the SemantiK subtree on
a dev checkout — they live OUTSIDE the repo. To point the runtime at the
existing weights WITHOUT that machine-specific absolute path ever appearing in
tracked code, set the env var only::

    export SEMANTIK_MODEL_DIR=/path/to/your/Semantic/models

``model_dir()`` then resolves there and ``resolve_model("council/structure/final")``
yields ``/path/to/your/Semantic/models/council/structure/final``. No absolute
path is committed; the pointer is supplied entirely through the environment.

Dependency-free by contract
---------------------------
This module imports ONLY ``os`` and ``pathlib`` — never torch / transformers /
llama-cpp / yaml. Importing ``semantik_structure.paths`` must stay cheap so a
CWD-independence smoke check (and the model-provisioning preflight) can resolve
directories without pulling in the heavy runtime extras.
"""
from __future__ import annotations

import os
from pathlib import Path

# Package root = the ``SemantiK/`` directory (parent of the ``semantik_structure``
# package). ``parents[0]`` is ``semantik_structure/``; ``parents[1]`` is ``SemantiK/``.
# Resolved once at import time; the package never moves at runtime.
PACKAGE_ROOT: Path = Path(__file__).resolve().parents[1]

# Directory-class basenames under the (package-relative or relocated) root. The
# basename matches the historic in-tree layout exactly so the package-relative
# default is byte-stable to the legacy ``models/`` / ``data/`` dirs.
_MODEL_BASENAME = "models"
_CACHE_BASENAME = "data"   # caches historically live UNDER data/ (data/*_cache)
_DATA_BASENAME = "data"
_CONFIG_BASENAME = "data"  # calibration/config artifacts live under data/ too


def semantik_home() -> Path | None:
    """Return the configured ``SEMANTIK_HOME`` data root, or ``None`` if unset.

    Read at call time (not import time) so a test can monkeypatch the env var
    and exercise the relocated layout without re-importing this module. An
    empty / whitespace value is treated as unset.
    """
    raw = os.environ.get("SEMANTIK_HOME", "").strip()
    return Path(raw) if raw else None


def _resolve(per_dir_env: str, basename: str) -> Path:
    """Resolve a directory honoring the documented 3-tier precedence.

        per-dir env override > SEMANTIK_HOME/<basename> > package-relative

    Read entirely at call time so tests / Docker can flip env vars without
    re-importing. Byte-stable to the package-relative default when no env set.
    """
    override = os.environ.get(per_dir_env, "").strip()
    if override:
        return Path(override)
    home = semantik_home()
    if home is not None:
        return home / basename
    return PACKAGE_ROOT / basename


def model_dir() -> Path:
    """Root directory for model weights (council LoRA adapters, Qwen GGUFs, theta).

    Precedence: ``SEMANTIK_MODEL_DIR`` > ``SEMANTIK_HOME/models`` >
    ``<SemantiK>/models``. Set ``SEMANTIK_MODEL_DIR`` to point at out-of-repo
    weights without committing the path (see the module docstring).
    """
    return _resolve("SEMANTIK_MODEL_DIR", _MODEL_BASENAME)


def cache_dir() -> Path:
    """Root directory for runtime caches (extract_cache, glm_ocr_cache, prerender_cache).

    Precedence: ``SEMANTIK_CACHE_DIR`` > ``SEMANTIK_HOME/data`` >
    ``<SemantiK>/data``. Historic caches lived under ``data/*_cache`` so the
    cache root coincides with the data root by default.
    """
    return _resolve("SEMANTIK_CACHE_DIR", _CACHE_BASENAME)


def data_dir() -> Path:
    """Root directory for data artifacts (eval reports, datasets, labels).

    Precedence: ``SEMANTIK_DATA_DIR`` > ``SEMANTIK_HOME/data`` >
    ``<SemantiK>/data``.
    """
    return _resolve("SEMANTIK_DATA_DIR", _DATA_BASENAME)


def config_dir() -> Path:
    """Root directory for config / calibration artifacts.

    Precedence: ``SEMANTIK_CONFIG_DIR`` > ``SEMANTIK_HOME/data`` >
    ``<SemantiK>/data``. Sibling-relative package configs (``config.yaml`` next
    to a module) keep using ``Path(__file__).parent`` and do NOT route here.
    """
    return _resolve("SEMANTIK_CONFIG_DIR", _CONFIG_BASENAME)


def resolve_model(rel: str | os.PathLike[str]) -> Path:
    """Resolve a model-relative subpath against :func:`model_dir`.

    ``resolve_model("council/structure/final")`` →
    ``<model_dir>/council/structure/final``. An absolute ``rel`` is returned
    unchanged (so an operator-supplied absolute adapter path is honored).
    """
    p = Path(rel)
    return p if p.is_absolute() else model_dir() / p


def resolve_model_literal(rel: str | os.PathLike[str]) -> Path:
    """Resolve a config-stored model path literal against :func:`model_dir`.

    Config files (``qwen_specialists/config.yaml``) and the figure-router store
    paths as ``models/<...>`` — i.e. WITH the ``models/`` prefix that names the
    model root itself. This strips a single leading ``models/`` segment before
    joining so the result is byte-stable to the legacy
    ``<root>/models/<...>`` layout while honoring ``SEMANTIK_MODEL_DIR``.

    An absolute ``rel`` is returned unchanged. A ``rel`` WITHOUT the ``models/``
    prefix is treated as already model-root-relative (joined directly).
    """
    p = Path(rel)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == _MODEL_BASENAME:
        p = Path(*p.parts[1:]) if len(p.parts) > 1 else Path()
    return model_dir() / p


def resolve_cache(rel: str | os.PathLike[str]) -> Path:
    """Resolve a cache-relative subpath against :func:`cache_dir`.

    The historic literals are ``data/extract_cache`` etc.; pass the basename
    AFTER ``data/`` (e.g. ``"extract_cache"``) — but ``data/extract_cache`` is
    accepted too: a leading ``data/`` segment is stripped so the result stays
    byte-stable to the legacy ``<root>/data/extract_cache`` layout.
    """
    p = Path(rel)
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == _CACHE_BASENAME:
        p = Path(*parts[1:]) if len(parts) > 1 else Path()
    return cache_dir() / p


def resolve_data(rel: str | os.PathLike[str]) -> Path:
    """Resolve a data-relative subpath against :func:`data_dir`.

    Mirrors :func:`resolve_cache` — a leading ``data/`` segment is stripped so
    the result is byte-stable to the legacy ``<root>/data/<rel>`` layout.
    """
    p = Path(rel)
    if p.is_absolute():
        return p
    parts = p.parts
    if parts and parts[0] == _DATA_BASENAME:
        p = Path(*parts[1:]) if len(parts) > 1 else Path()
    return data_dir() / p


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) on first use; return it. No-op if present."""
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "PACKAGE_ROOT",
    "semantik_home",
    "model_dir",
    "cache_dir",
    "data_dir",
    "config_dir",
    "resolve_model",
    "resolve_model_literal",
    "resolve_cache",
    "resolve_data",
    "ensure_dir",
]

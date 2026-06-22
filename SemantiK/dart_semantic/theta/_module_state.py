"""Module-level cached load of the semantic-preservation cross-encoder.

Phase 4c: lazy-loaded once per process. The theta artifact dir is
resolved through the single env-aware resolver
``paths.resolve_model("theta/semantic_preservation/v8")`` when no
override is set (SEMANTIK_MODEL_DIR / SEMANTIK_HOME / package-relative
honored there) — NOT a CWD-relative literal. The legacy theta-specific
override ``DART_SEMANTIC_MODEL_DIR`` still wins (highest precedence) but
now also flows through ``paths.resolve_model``. The loader is one-shot
and caches ``None`` (stub mode) on the first failed load when stub
mode is allowed, so every subsequent ``evaluate()`` call short-circuits
without re-attempting the heavy load.

**Strict-by-default policy.** Missing model dir or any load failure
(including sentinel-health-check failure) escalates to a
``RuntimeError`` by default. The ``stub_v1`` placeholder (returns 0.7
for every input) is opt-in via ``DART_ALLOW_THETA_STUB=1``. This
matches the project-wide "no silent fallbacks" preference: a placeholder
score that feeds into the composite ``theta_score`` would silently
degrade reported numbers without callers noticing. See
``feedback_no_silent_fallbacks.md`` in user memory.

History note: this env var used to be ``DART_REQUIRE_THETA_MODEL=1``
(opt-in strict). The sense was inverted 2026-05-13 after the user
explicitly rejected default-stub behavior.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .. import paths as _semantik_paths

if TYPE_CHECKING:  # pragma: no cover - type-only
    from .semantic_preservation import SemanticPreservationModel


logger = logging.getLogger(__name__)


# v8: full-FT DeBERTa-v3-small cross-encoder (cls pooling, BCE), trained on the
# qwen-style-inclusive dataset. Replaces the mode-collapsed v1 (val spearman 0.21,
# constant ~0.70 output that failed the discrimination sentinel). v8: val spearman
# 0.43, calibrated clean~0.77 vs unrelated~0.39 (gap ~0.38). See Plans/10 R8.
# Anchored to the SemantiK model root (CWD-independent) instead of the legacy
# CWD-relative ``models/...`` literal. ``DART_SEMANTIC_MODEL_DIR`` still wins.
_DEFAULT_MODEL_DIR = "theta/semantic_preservation/v8"

# Tri-state cache: missing key = "never attempted"; ``None`` value = "tried,
# falling back to stub"; populated value = "loaded, use it".
_cached: dict[str, "SemanticPreservationModel | None"] = {}


def _model_dir() -> Path:
    """Resolve the theta artifact dir through the SINGLE env-aware resolver.

    There is now ONE resolution path — ``paths.resolve_model`` — with
    precedence (high → low):
      ``DART_SEMANTIC_MODEL_DIR`` (legacy theta-specific override)
        > ``SEMANTIK_MODEL_DIR`` > ``SEMANTIK_HOME`` > package-relative.

    All four flow through ``paths.py``. The legacy override is treated as
    an absolute-or-cwd-relative theta artifact dir and routed through
    ``resolve_model`` (which returns an absolute path unchanged and joins
    a relative one onto ``model_dir()`` — so an absolute legacy override
    stays byte-stable while a bare relative one now resolves under the
    SEMANTIK_MODEL_DIR root instead of CWD, which is strictly more
    correct). When no override is set, returns
    ``resolve_model(_DEFAULT_MODEL_DIR)`` (byte-stable to the
    package-relative default).
    """
    override = os.environ.get("DART_SEMANTIC_MODEL_DIR", "").strip()
    if override:
        return _semantik_paths.resolve_model(override)
    return _semantik_paths.resolve_model(_DEFAULT_MODEL_DIR)


def _stub_allowed() -> bool:
    """True iff caller has opted in to the stub_v1 fallback.

    Default is False — strict-by-default. Set ``DART_ALLOW_THETA_STUB=1``
    to permit the 0.7 placeholder to flow into ``theta_score``.
    """
    return os.environ.get("DART_ALLOW_THETA_STUB", "").strip() == "1"


def _get_model() -> "SemanticPreservationModel | None":
    """Return the cached model, or ``None`` if stub fallback is permitted.

    The first call drives the load attempt; subsequent calls reuse the
    cached result. Use :func:`_reset_cache` from tests to force re-load.

    Raises ``RuntimeError`` when the model can't be loaded and stub
    fallback is not opted in (i.e. ``DART_ALLOW_THETA_STUB!=1``).
    """
    key = "active"
    if key in _cached:
        return _cached[key]

    model_dir = _model_dir()
    stub_allowed = _stub_allowed()

    # Lazy import keeps the stub path free of torch / transformers / peft.
    from .semantic_preservation import load as _load

    try:
        model = _load(model_dir)
    except Exception as exc:
        if not stub_allowed:
            # Don't poison the cache on strict failure — let the next
            # call retry (matches how ops would expect a "fix-then-retry"
            # workflow to work).
            raise RuntimeError(
                f"Theta semantic-preservation model failed to load from "
                f"{model_dir!s}: {exc!r}. Strict-by-default refuses to "
                f"silently fall back to the 0.7 stub. To override for "
                f"dev/eval, set DART_ALLOW_THETA_STUB=1 and accept that "
                f"theta_score will include placeholder values. To fix "
                f"the underlying issue, retrain the model (see "
                f"scripts/train_semantic_preservation.py)."
            ) from exc
        logger.warning(
            "Theta semantic-preservation model failed to load from %s "
            "(%r); falling back to stub_v1 because DART_ALLOW_THETA_STUB=1 "
            "is set. Composite theta_score will include 0.7 placeholder.",
            model_dir,
            exc,
        )
        _cached[key] = None
        return None

    if model is None and not stub_allowed:
        raise RuntimeError(
            f"No theta semantic-preservation model artifacts found under "
            f"{model_dir!s}. Strict-by-default refuses the 0.7 stub. To "
            f"override for dev/eval, set DART_ALLOW_THETA_STUB=1."
        )

    if model is None:
        logger.warning(
            "Theta semantic-preservation model not found at %s; "
            "falling back to stub_v1 because DART_ALLOW_THETA_STUB=1 "
            "is set.",
            model_dir,
        )

    _cached[key] = model
    return model


def _reset_cache() -> None:
    """Test hook — clears the cache so the next ``_get_model()`` re-loads."""
    _cached.clear()

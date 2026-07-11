"""Content-hash disk cache for the full-block reviewer's per-window op-lists.

Phase 4b of the SemantiK full-block structural-editor reviewer. The windowed
Stage-5d reviewer (``reviewer.py::_run_windowed_block_review``) fans ONE
idx-keyed op-list prompt per window of in-scope blocks. With greedy /
``temperature=0`` decoding the parsed op-list is a deterministic function of
the window input, so memoizing it by a content hash of that input makes
re-runs free and reproducible (design §6.8) — exactly mirroring the
extract / GLM-OCR disk caches.

Cache key: ``sha256`` over the FULL window input that determines the op-list —
the window prompt, the members' edge-record dicts, their ``council_kind``s,
the resolved reviewer model id, the dispatch temperature, and the prompt-/
scope-affecting flag set (``SEMANTIK_BLOCK_REVIEW_WINDOW`` /
``_EDGE_TOKENS`` / ``_CONF``). Value: the parsed op-list (``list[dict]``),
serialized as JSON. A sha256 over the full input makes collisions
statistically impossible, so a hit reproduces the EXACT same ops as a miss —
pure memoization, never a behavior change.

Storage layout (mirrors ``glm_ocr_cache``):
    data/block_review_ops/<first2>/<hash>.json

A corrupt / unreadable / schema-mismatched entry is treated as a MISS (re-
dispatch); a failed WRITE is best-effort (logged, swallowed). Neither ever
raises into the review.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .. import paths as _semantik_paths

logger = logging.getLogger(__name__)

# Bump when the op-list value shape or the key-payload composition changes so
# every entry produced by an older layout auto-invalidates (mirrors
# ``prerender_cache.RENDER_CONFIG_VERSION`` / ``glm_ocr_cache``'s implicit
# greedy-decode contract).
SCHEMA_VERSION = "v1"

# The cache namespace under the SemantiK cache root.
_CACHE_NAMESPACE = "block_review_ops"


def cache_dir() -> Path:
    """The block-review op-cache directory, resolved FRESH each call.

    Anchored under the SemantiK cache root (CWD-independent) via
    :func:`paths.resolve_cache` — so ``SEMANTIK_CACHE_DIR`` / ``SEMANTIK_HOME``
    redirect it (tests point it at a ``tmp_path``). Byte-stable to the historic
    ``<root>/data/block_review_ops`` layout when no ``SEMANTIK_*`` env is set.
    """
    return _semantik_paths.resolve_cache(_CACHE_NAMESPACE)


def make_key(
    *,
    prompt: str,
    records: list[dict[str, Any]],
    council_kinds: list[str],
    model: str,
    temperature: float,
    flags: dict[str, Any],
) -> str:
    """``sha256`` hexdigest over the FULL window input determining the op-list.

    Every argument feeds the hash; any change (an edited edge record, a member
    council_kind, the resolved model id, the dispatch temperature, or a prompt-
    affecting flag) yields a different key -> a cache MISS -> a fresh dispatch.
    Canonical JSON (``sort_keys=True``) so the digest is order-stable.
    """
    payload = {
        "v": SCHEMA_VERSION,
        "prompt": prompt,
        "records": records,
        "council_kinds": list(council_kinds),
        "model": model,
        "temperature": temperature,
        "flags": flags,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _path(key: str, cdir: Path) -> Path:
    return cdir / key[:2] / f"{key}.json"


def get(key: str, cdir: Path | None = None) -> list[dict[str, Any]] | None:
    """Return the cached parsed op-list for ``key``, or None on any miss.

    A missing file, an unreadable / corrupt entry, or a schema-version /
    shape mismatch ALL degrade to None (a MISS -> the caller re-dispatches).
    NEVER raises — mirrors ``glm_ocr_cache.get``."""
    cdir = cdir if cdir is not None else cache_dir()
    p = _path(key, cdir)
    try:
        if not p.exists():
            return None
        doc = json.loads(p.read_text())
    except Exception:  # noqa: BLE001 — any read/parse error == cache miss.
        return None
    if not isinstance(doc, dict) or doc.get("v") != SCHEMA_VERSION:
        return None  # schema mismatch -> miss (re-dispatch).
    ops = doc.get("ops")
    if not isinstance(ops, list):
        return None
    return [o for o in ops if isinstance(o, dict)]


def put(key: str, ops: list[dict[str, Any]], cdir: Path | None = None) -> None:
    """Store the parsed op-list for ``key`` (best-effort, atomic, no-raise).

    A failed write is logged and swallowed — the review never fails on a cache
    write. Writes to a temp sibling then atomically renames so a partial file
    is never mistaken for a hit (mirrors ``glm_ocr_cache.put`` /
    ``prerender_cache.get_or_render``)."""
    cdir = cdir if cdir is not None else cache_dir()
    p = _path(key, cdir)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = {"v": SCHEMA_VERSION, "ops": list(ops)}
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, default=str))
        tmp.replace(p)
    except Exception as exc:  # noqa: BLE001 — cache write is best-effort.
        logger.warning("block-review op-cache write failed (%s) -> skipping", exc)

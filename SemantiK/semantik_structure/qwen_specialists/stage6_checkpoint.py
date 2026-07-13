"""Content-addressed per-unit resume cache for Stage-6 authoring candidates.

The graceful-stop ("checkpoint on command") contract requires every
long-running LLM unit to persist its result so an ``ed4all stop`` / crash loses
at most the in-flight calls. Stage-6 authoring (the ~3h/chapter local-7B tier
and the batched hosted-endpoint tier) previously had NO per-region resume
sidecar — a stop mid-Stage-6 re-ran the whole document.

This module is the Stage-6 twin of ``reasoning_qc``'s per-unit sidecar
(``reasoning_qc.py`` §"Per-UNIT resume cache") and the block-reviewer's
``block_review_cache`` — a content-addressed disk cache keyed on the EXACT
generation input (prompt text + model + sampling + slot tag + optional draft),
so a resumed run serves every already-authored candidate from cache instead of
re-POSTing / re-generating.

Design invariants (mirroring ``reasoning_qc`` + ``block_review_cache``):

- **Site flag, default ON, family fallback.** ``SEMANTIK_STAGE6_CHECKPOINT``,
  WHEN SET (non-blank), wins (falsey token → off, else on); otherwise the
  family ``ED4ALL_GENERATION_CHECKPOINT`` decides (falsey token → off; unset /
  garbage → on). Off → byte-identical: no cache dir is ever created.
- **Content-addressed.** The fingerprint hashes the full generation input; any
  change (prompt, model, sampling, slot, draft) moves the key, so a stale
  sidecar is never served for a changed input.
- **Fail-soft.** A corrupt / unreadable sidecar is a MISS (re-generate); a
  failed write is best-effort (logged, swallowed). Neither raises into Stage 6.
- **Only genuine content is cached.** A ``None`` / empty-string result (the
  fail-soft sentinel a failed endpoint slot returns) is NEVER cached — it must
  re-run next time, never be memoized as "nothing produced".
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Bump when the sidecar value shape changes so older entries auto-invalidate.
SCHEMA_VERSION = "v1"

_CACHE_BASENAME = "stage6_candidate_cache"

# Site checkpoint flag (default ON) → falls back to the
# ED4ALL_GENERATION_CHECKPOINT family. Mirrors
# reasoning_qc.resolve_reasoning_qc_checkpoint verbatim (SemantiK's
# self-containment boundary forbids a lib/ import — the cross-venv conversion
# subprocess has no Ed4All lib on its path).
_CHECKPOINT_ENV = "SEMANTIK_STAGE6_CHECKPOINT"
_CHECKPOINT_FAMILY_ENV = "ED4ALL_GENERATION_CHECKPOINT"
_CHECKPOINT_FALSEY = frozenset({"0", "false", "no", "off"})


def resolve_stage6_checkpoint() -> bool:
    """Resolve the per-unit Stage-6 resume-cache gate (default ON). Read at
    CALL time.

    Precedence: the site env ``SEMANTIK_STAGE6_CHECKPOINT``, WHEN SET
    (non-blank), wins (falsey token → off, anything else → on); otherwise the
    family ``ED4ALL_GENERATION_CHECKPOINT`` decides (falsey token → off, unset /
    garbage → on). Off → byte-identical: neither reads nor writes the sidecar
    (no cache dir is ever created)."""
    import os

    site = os.environ.get(_CHECKPOINT_ENV, "")
    if site.strip():
        return site.strip().lower() not in _CHECKPOINT_FALSEY
    return (
        os.environ.get(_CHECKPOINT_FAMILY_ENV, "").strip().lower()
        not in _CHECKPOINT_FALSEY
    )


def candidate_fingerprint(
    prompt: str,
    *,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
    repeat_penalty: float,
    slot_tag: str,
    draft: str | None = None,
) -> str:
    """Content-address one Stage-6 candidate generation → a sha256 hex key.

    The fingerprint hashes the FULL input that determines the completion:

    * ``prompt`` — the region's generation prompt (verbatim);
    * ``model`` — the resolved seat model id (``getattr(rt, "_model", ...)``);
    * the sampling params (``temperature`` / ``top_p`` / ``max_tokens`` /
      ``repeat_penalty`` — sampling changes the output, so a cross-sampling hit
      would be a wrong candidate);
    * ``seed`` — the per-slot seed offset (``None`` when unseeded);
    * ``slot_tag`` — ``"{phase}:{adapter}:slot{k_idx}"``; the K-slot component is
      LOAD-BEARING because ``seed=None`` slots would otherwise collide on the
      same prompt and serve slot 0's candidate for every slot;
    * ``draft`` — the Phase-1 draft the batched-refine lane carries OUTSIDE the
      prompt (``None`` on the non-refine path); included so a refine candidate is
      keyed to its exact draft.

    Any change to any component moves the key, so a stale sidecar is never
    served for a changed input.
    """
    payload = {
        "v": SCHEMA_VERSION,
        "prompt": prompt,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "repeat_penalty": repeat_penalty,
        "slot_tag": slot_tag,
        "draft": draft,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_root() -> Path:
    """The content-addressed Stage-6 candidate sidecar root (CWD-independent).

    Resolves via ``semantik_structure.paths.resolve_cache`` so it honours
    ``SEMANTIK_HOME`` / ``SEMANTIK_CACHE_DIR`` and is stable regardless of the
    subprocess CWD (mirrors ``block_review_cache.cache_dir`` /
    ``reasoning_qc._qc_cache_root``)."""
    from .. import paths as _semantik_paths

    return _semantik_paths.resolve_cache(_CACHE_BASENAME)


def cache_path(key: str, root: Path) -> Path:
    return root / key[:2] / f"{key}.json"


def cacheable(text: Any) -> bool:
    """Only a genuine, non-empty completion string is persisted.

    ``None`` (the failed-slot sentinel) and ``""`` (a skip / empty draft) are
    NEVER cached — they must re-run next time, never be memoized as "nothing
    produced". Anything else (a non-empty str) is cacheable."""
    return isinstance(text, str) and bool(text)


def get(path: Path) -> str | None:
    """Read a cached candidate string (``None`` on miss / corrupt / IO —
    fail-soft)."""
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 — a corrupt/unreadable sidecar → miss
        logger.debug("stage6 cache read failed (non-fatal) %s: %s", path, exc)
        return None
    if not isinstance(obj, dict) or obj.get("v") != SCHEMA_VERSION:
        return None  # schema mismatch → miss (re-generate).
    text = obj.get("text")
    return text if isinstance(text, str) and text else None


def put(path: Path, text: str) -> None:
    """Persist a candidate atomically (temp+rename). Best-effort — never raises.

    Only call for a ``cacheable`` result; a caller that forgets is guarded here
    too (an empty / non-str value is silently skipped)."""
    if not cacheable(text):
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {"v": SCHEMA_VERSION, "text": text}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False))
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001 — a cache write must never break Stage 6
        logger.debug("stage6 cache write failed (non-fatal) %s: %s", path, exc)


__all__ = [
    "SCHEMA_VERSION",
    "cache_path",
    "cache_root",
    "cacheable",
    "candidate_fingerprint",
    "get",
    "put",
    "resolve_stage6_checkpoint",
]

"""Fingerprinted JSONL unit-checkpoint store for expensive LLM loops.

Generalizes the stage-2 objective-synthesis per-window resume sidecar (which
itself mirrors the training-synthesis ``.synthesis_pairs_checkpoint.jsonl``
per-item resume contract) into one shared mechanism usable by every expensive
per-unit LLM dispatch loop (stage-2 windows, bottom-up TO cluster authoring,
the Courseforge outline/rewrite per-block tiers).

Contract (identical across sites):

- One JSONL record per successfully-computed unit, carrying the unit's payload
  plus a deterministic FINGERPRINT of the LLM-relevant inputs.
- Append + flush + fsync per record so a mid-loop kill is recoverable.
- On re-entry a record is REUSED only when its stored fingerprint matches the
  recomputed one for the same unit id; any mismatch (changed inputs / model /
  window knobs / prompt) silently re-runs that unit (correct invalidation).
- Torn-line tolerance: a half-written trailing line (crash mid-append) is
  skipped with a warning, never breaking the load.
- Best-effort writes: an OSError appending only costs a re-run of that one
  unit on resume; it never aborts the loop.

Flag gating (ONE family flag, per-site overrides):

- ``ED4ALL_GENERATION_CHECKPOINT`` — the family flag governing every site.
  Default ON; an explicit falsey token (``0`` / ``false`` / ``no`` / ``off``,
  any case) disables all sites (parse-with-fallback).
- Each site may pass its own ``site_env`` (e.g. the pre-existing
  ``ED4ALL_OBJECTIVE_SYNTHESIS_CHECKPOINT`` /
  ``COURSEFORGE_OUTLINE_CHECKPOINT`` / ``COURSEFORGE_REWRITE_CHECKPOINT``).
  Precedence: **site flag (when SET) > family flag** — a set site env is
  authoritative for that site regardless of the family value; unset site env
  falls through to the family flag.

Selects no provider/model → no ``docs/LICENSING.md`` row.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

#: Family flag governing every fingerprinted LLM unit-checkpoint site.
FAMILY_ENV = "ED4ALL_GENERATION_CHECKPOINT"

#: Explicit falsey tokens (mirrors the Courseforge two-pass checkpoint flags).
FALSEY_TOKENS = frozenset({"0", "false", "no", "off"})

#: Record schema version. Bumping it invalidates every pre-bump sidecar
#: (records with a mismatched version are skipped at load).
SCHEMA_VERSION = "v1"

#: Keys reserved for the record envelope (payload keys must not collide).
_RESERVED_KEYS = frozenset({"schema_version", "site_id", "unit_id", "fingerprint"})


def checkpoint_enabled(site_env: Optional[str] = None) -> bool:
    """Resolve the checkpoint gate for a site (default ON).

    Precedence: the site's own env var, WHEN SET (non-empty), is authoritative
    (falsey token → off, anything else → on); otherwise the family
    ``ED4ALL_GENERATION_CHECKPOINT`` flag decides (falsey token → off,
    unset / garbage → on). Parse-with-fallback throughout. Read each call so
    tests can toggle inline.
    """
    if site_env:
        raw = os.environ.get(site_env, "")
        if raw.strip():
            return raw.strip().lower() not in FALSEY_TOKENS
    return (
        os.environ.get(FAMILY_ENV, "").strip().lower() not in FALSEY_TOKENS
    )


def compute_fingerprint(payload: Mapping[str, Any]) -> str:
    """Deterministic sha256 fingerprint of a JSON-able input-description dict.

    Canonical form: ``json.dumps(sort_keys=True, ensure_ascii=False,
    default=str)`` — the caller supplies a flat(ish) dict describing every
    LLM-relevant input axis (ids, content hashes, model id, window knobs,
    prompt hash, ...). Any change to any axis flips the fingerprint.
    """
    canon = json.dumps(
        dict(payload), sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def hash_text(text: str) -> str:
    """sha256 of a text blob (prompt / chunk-body / contract hashing helper)."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class CheckpointStore:
    """Fingerprinted JSONL unit-checkpoint store for one dispatch site.

    Typical batch-dispatch use (the stage-2 window pattern)::

        store = CheckpointStore(path, site_id="stage2_windows",
                                site_env="ED4ALL_OBJECTIVE_SYNTHESIS_CHECKPOINT")
        cache = store.load()
        for unit in units:
            fp = compute_fingerprint({...})
            payload = store.reuse(unit_id, fp, cache=cache)
            if payload is None:
                payload = expensive_llm_call(unit)
                store.append(unit_id, fp, payload)

    Or the inline single-unit form::

        payload = store.reuse_or_compute(unit_id, fp, fn)

    Record shape (one JSON line each)::

        {"schema_version": "v1", "site_id": ..., "unit_id": ...,
         "fingerprint": ..., **payload}

    The payload dict is merged into the record top-level (reserved envelope
    keys are protected), so site-specific fields stay first-class / greppable.
    """

    def __init__(
        self,
        path: Optional[Path],
        *,
        site_id: str,
        site_env: Optional[str] = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.site_id = site_id
        self.site_env = site_env

    # ------------------------------------------------------------------ #
    # Gating
    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        """True when this site's checkpoint is on AND a path was supplied."""
        return self.path is not None and checkpoint_enabled(self.site_env)

    # ------------------------------------------------------------------ #
    # Load / reuse
    # ------------------------------------------------------------------ #
    def load(self) -> Dict[str, Dict[str, Any]]:
        """Tolerant load: ``{unit_id: record}`` (last-line-wins per unit).

        Blank / malformed / schema-mismatched lines are skipped; a torn FINAL
        line (crash mid-append) is skipped with a warning so loading never
        breaks a resume. Missing / unreadable / disabled → empty map.
        """
        out: Dict[str, Dict[str, Any]] = {}
        if not self.enabled or not self.path.exists():
            return out
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return out
        lines = raw.splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                if i == len(lines) - 1:
                    logger.warning(
                        "llm_checkpoint[%s]: unparseable trailing line in %s "
                        "(torn crash write); skipping it.",
                        self.site_id, self.path,
                    )
                else:
                    logger.warning(
                        "llm_checkpoint[%s]: skipping malformed line %d in %s.",
                        self.site_id, i, self.path,
                    )
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("schema_version") != SCHEMA_VERSION:
                continue
            uid = rec.get("unit_id")
            if isinstance(uid, str) and uid:
                out[uid] = rec
        return out

    def reuse(
        self,
        unit_id: str,
        fingerprint: str,
        *,
        cache: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the cached payload for ``unit_id`` iff its fingerprint matches.

        ``cache`` is the (pre-``load()``-ed) record map; passing it avoids
        re-reading the file per unit. Returns the record's payload portion
        (envelope keys stripped) on a match; None on miss / mismatch /
        disabled — the caller then computes fresh.
        """
        if not self.enabled:
            return None
        records = cache if cache is not None else self.load()
        rec = records.get(unit_id)
        if not isinstance(rec, dict):
            return None
        if rec.get("fingerprint") != fingerprint:
            return None
        return {k: v for k, v in rec.items() if k not in _RESERVED_KEYS}

    # ------------------------------------------------------------------ #
    # Append / compute / remove
    # ------------------------------------------------------------------ #
    def append(
        self, unit_id: str, fingerprint: str, payload: Dict[str, Any],
    ) -> None:
        """Append one completed-unit record with flush+fsync (best-effort).

        An OSError is logged but NEVER raised — a missing line only costs a
        re-run of that one unit on resume. No-op when disabled.
        """
        if not self.enabled:
            return
        record: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "site_id": self.site_id,
            "unit_id": unit_id,
            "fingerprint": fingerprint,
        }
        for k, v in payload.items():
            if k not in _RESERVED_KEYS:
                record[k] = v
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str))
                fh.write("\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
        except OSError as exc:
            logger.warning(
                "llm_checkpoint[%s]: append failed for unit %r: %s",
                self.site_id, unit_id, exc,
            )

    def reuse_or_compute(
        self,
        unit_id: str,
        fingerprint: str,
        fn: Callable[[], Optional[Dict[str, Any]]],
        *,
        cache: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the cached payload, or compute + checkpoint it.

        ``fn`` returning None (fail-soft unit) is NOT checkpointed, so a
        resume re-attempts the failed unit (mirrors the Courseforge tiers'
        degraded-blocks-not-checkpointed contract).
        """
        cached = self.reuse(unit_id, fingerprint, cache=cache)
        if cached is not None:
            return cached
        result = fn()
        if result is not None:
            self.append(unit_id, fingerprint, dict(result))
        return result

    def remove(self) -> None:
        """Delete the sidecar (successful final write / ``--force`` clear).

        Best-effort: a failed unlink is logged, never raised. Runs regardless
        of the enabled gate (a ``--force`` clear must work even when the
        family flag is off).
        """
        if self.path is None:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(
                "llm_checkpoint[%s]: failed to remove sidecar %s: %s",
                self.site_id, self.path, exc,
            )

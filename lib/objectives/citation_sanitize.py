"""Deterministic write-time citation SANITIZER (default ON, safety net).

The stage-2 local synthesizer (7B / nano-omni) sometimes emits *fabricated*
citations: instead of a real chunk id (``<slug>_chunk_00001`` or a
``semantik:{slug}#sN`` sourceId) it echoes a
descriptive topic LABEL (``"Round Whole Numbers"``, ``"Order of Operations"``)
or the objective STATEMENT text itself into ``source_refs`` /
``source_chunk_ids``. Those ids resolve against NOTHING in the current
chunkset, so the ``objective_source_refs`` gate's aggregate
``ORPHANED_CITATIONS`` net fires CRITICAL and the (correctly-authored,
well-grounded) objectives artifact is HARD-BLOCKED at ``course_planning`` — the
sole failing gate on an otherwise-green multi-chapter run.

This pass is the deterministic, always-safe backstop. Immediately before
``synthesized_objectives.json`` is written it DROPS any citation that does not
resolve against the current chunkset / textbook-structure universe:

* a structured ``source_refs[].chunk_ids[*]`` entry not in the chunk-id
  universe → dropped from that entry's ``chunk_ids``;
* a fully-fabricated structured entry (``ref`` not in the textbook-structure
  universe AND no surviving ``chunk_ids``) → the whole entry is dropped;
* a legacy ``source_refs[]`` string not in the union universe → dropped;
* a flat ``source_chunk_ids[*]`` id not in the chunk-id universe → dropped.

A citation that is dropped becomes an absent / empty ref. The downstream gates
treat that as the BENIGN path — ``objective_source_refs`` no longer sees an
orphaned resolvable-universe citation (no ``ORPHANED_CITATIONS`` critical), and
``objective_entailment`` routes the now-uncited LO to its warning-severity
``OBJECTIVE_NO_GROUNDING_SOURCE`` pass instead of a failed entailment. Net: a
provably-wrong id can never hard-block the pipeline.

**Anti-fabrication.** The pass only ever REMOVES ids that fail set-membership
against a universe harvested from the real chunkset + textbook structure; it
never invents, re-points, or reorders a citation. It is the set-membership
complement of :mod:`lib.objectives.citation_reselect` (which cosine-re-derives
a BETTER real supporter): reselect improves provenance when it can build a
pool; the sanitizer guarantees the floor when reselect misses one or when the
``[embedding]`` extras are absent.

**Gate-parity (why removals match exactly what the gate would flag).** The drop
predicates mirror ``lib/validators/objective_source_refs.py`` arm-for-arm:

* structured ``chunk_ids`` are pruned ONLY when the chunk universe is
  non-empty (the gate skips the chunk-resolution check against an empty /
  unavailable universe — its ``chunks_universe`` guard);
* legacy strings are pruned ONLY when the union universe is non-empty (the
  gate's ``union_universe`` guard);
* a structured entry's ``ref`` is treated as fabricated ONLY when the
  textbook-structure universe is non-empty (the gate's ``textbook_universe``
  guard).

So on a legacy / no-universe run (both universes empty) the pass is a strict
no-op — byte-identical output — regardless of the flag.

**Flag / default.** Gated by ``ED4ALL_OBJECTIVE_SANITIZE_CITATIONS`` (resolver
:func:`resolve_sanitize_citations`), **default ON**. Default-ON is justified:
the pass removes ONLY provably-wrong ids, so on every healthy corpus (no
fabricated citations) it is a byte-identical no-op, and on a broken corpus it
converts a hard CRITICAL block into a benign warning rather than silently
shipping orphaned provenance. Opt out with ``0`` / ``false`` / ``no`` / ``off``
(any case) to restore the pre-sanitizer bytes verbatim. Selects no
provider / model → no ``docs/LICENSING.md`` row. Decision capture reuses the
existing ``objective_chunk_prune`` decision_type (no new enum member).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

logger = logging.getLogger(__name__)

ENV_SANITIZE_CITATIONS = "ED4ALL_OBJECTIVE_SANITIZE_CITATIONS"
_FALSEY = frozenset({"0", "false", "no", "off"})


def resolve_sanitize_citations(value: Optional[bool] = None) -> bool:
    """Resolve ``ED4ALL_OBJECTIVE_SANITIZE_CITATIONS`` (default **ON**).

    Explicit arg > env. Only the opt-out tokens ``0`` / ``false`` / ``no`` /
    ``off`` (any case) disable the sanitizer; unset / anything else keeps the
    default-ON safety net (parse-with-fallback, mirroring
    ``resolve_reselect_keep_original`` — this only removes provably-wrong ids).
    """
    if value is not None:
        return bool(value)
    raw = os.environ.get(ENV_SANITIZE_CITATIONS)
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSEY


def build_chunk_universe(chunks: Iterable[Mapping[str, Any]]) -> Set[str]:
    """Harvest the chunk-id resolution universe from a chunk iterable.

    Mirrors the ``lib/validators/objective_source_refs.py`` chunk-universe
    loader EXACTLY: each chunk's top-level ``id`` plus the union of every chunk's
    ``source.source_references[*].sourceId`` (the ``semantik:{slug}#sN`` shape).
    Both forms are valid citation targets so the sanitizer's set-membership
    check matches the gate's.
    """
    universe: Set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        cid = chunk.get("id") or chunk.get("chunk_id")
        if isinstance(cid, str) and cid.strip():
            universe.add(cid.strip())
        source = chunk.get("source")
        if not isinstance(source, Mapping):
            continue
        src_refs = source.get("source_references") or []
        if not isinstance(src_refs, list):
            continue
        for ref in src_refs:
            if not isinstance(ref, Mapping):
                continue
            src_id = ref.get("sourceId")
            if isinstance(src_id, str) and src_id.strip():
                universe.add(src_id.strip())
    return universe


@dataclass
class SanitizeResult:
    """Aggregate signals from one sanitize pass (LO dicts mutated in place)."""

    available: bool = False            # False → no-op (disabled / empty universe)
    los_scanned: int = 0
    los_mutated: int = 0               # LOs whose citations changed
    structured_chunk_ids_dropped: int = 0
    legacy_strings_dropped: int = 0
    flat_chunk_ids_dropped: int = 0
    entries_dropped: int = 0           # fully-fabricated structured entries removed
    per_lo_changes: List[Dict[str, Any]] = field(default_factory=list)


def _lo_id(lo: Mapping[str, Any]) -> str:
    raw = lo.get("id") or lo.get("objective_id") or ""
    return str(raw).strip() or "<unknown-id>"


def _sanitize_source_refs(
    source_refs: Any,
    *,
    chunk_universe: Set[str],
    structure_universe: Set[str],
    result: SanitizeResult,
) -> Any:
    """Return a sanitized ``source_refs`` value (or the input unchanged).

    Handles both W1.6.A arms: legacy ``List[str]`` and structured
    ``List[{ref, chunk_ids[]}]``. Removes only ids that fail set-membership
    against the relevant universe (gate-parity guards applied by the caller via
    empty-universe short-circuit).
    """
    if not isinstance(source_refs, list) or not source_refs:
        return source_refs

    union_universe = chunk_universe | structure_universe
    first = source_refs[0]

    # ---- Legacy List[str] arm. --------------------------------------------
    if isinstance(first, str):
        if not union_universe:
            return source_refs  # gate skips resolution against an empty union
        kept: List[str] = []
        for entry in source_refs:
            if not isinstance(entry, str):
                kept.append(entry)  # defensive: schema forbids mixed lists
                continue
            stripped = entry.strip()
            if not stripped:
                continue
            if stripped in union_universe:
                kept.append(entry)
            else:
                result.legacy_strings_dropped += 1
        return kept

    # ---- Structured List[{ref, chunk_ids[]}] arm. -------------------------
    new_refs: List[Any] = []
    for entry in source_refs:
        if not isinstance(entry, dict):
            new_refs.append(entry)  # defensive
            continue
        ref_field = entry.get("ref")
        ref_str = ref_field.strip() if isinstance(ref_field, str) else ""
        chunk_ids = entry.get("chunk_ids")

        kept_chunk_ids: List[Any] = []
        if isinstance(chunk_ids, list):
            for cid in chunk_ids:
                if not isinstance(cid, str):
                    kept_chunk_ids.append(cid)  # defensive
                    continue
                cs = cid.strip()
                if not cs:
                    continue
                # Gate-parity: only prune against a non-empty chunk universe.
                if not chunk_universe or cs in chunk_universe:
                    kept_chunk_ids.append(cid)
                else:
                    result.structured_chunk_ids_dropped += 1

        # Drop the WHOLE entry only when it is provably fully fabricated: no
        # surviving chunk_ids AND a ref we can prove is not in the (non-empty)
        # textbook-structure universe. When the structure universe is empty we
        # cannot prove the ref wrong, so the entry is retained (with its
        # chunk_ids already pruned) — the gate would only warn on it anyway.
        ref_is_fabricated = bool(
            structure_universe
            and ref_str
            and ref_str not in structure_universe
        )
        if not kept_chunk_ids and (ref_is_fabricated or not ref_str):
            result.entries_dropped += 1
            continue

        new_entry = dict(entry)
        if isinstance(chunk_ids, list):
            new_entry["chunk_ids"] = kept_chunk_ids
        new_refs.append(new_entry)
    return new_refs


def _sanitize_lo(
    lo: Dict[str, Any],
    *,
    chunk_universe: Set[str],
    structure_universe: Set[str],
    result: SanitizeResult,
) -> bool:
    """Sanitize one LO dict IN PLACE. Return True iff anything changed."""
    before = SanitizeResult(
        structured_chunk_ids_dropped=result.structured_chunk_ids_dropped,
        legacy_strings_dropped=result.legacy_strings_dropped,
        flat_chunk_ids_dropped=result.flat_chunk_ids_dropped,
        entries_dropped=result.entries_dropped,
    )

    source_refs = lo.get("source_refs")
    if isinstance(source_refs, list) and source_refs:
        sanitized = _sanitize_source_refs(
            source_refs,
            chunk_universe=chunk_universe,
            structure_universe=structure_universe,
            result=result,
        )
        if sanitized != source_refs:
            lo["source_refs"] = sanitized

    # Flat back-compat mirror.
    flat = lo.get("source_chunk_ids")
    if isinstance(flat, list) and flat and chunk_universe:
        kept_flat: List[Any] = []
        for cid in flat:
            if not isinstance(cid, str):
                kept_flat.append(cid)
                continue
            cs = cid.strip()
            if not cs:
                continue
            if cs in chunk_universe:
                kept_flat.append(cid)
            else:
                result.flat_chunk_ids_dropped += 1
        if kept_flat != flat:
            lo["source_chunk_ids"] = kept_flat

    changed = (
        result.structured_chunk_ids_dropped != before.structured_chunk_ids_dropped
        or result.legacy_strings_dropped != before.legacy_strings_dropped
        or result.flat_chunk_ids_dropped != before.flat_chunk_ids_dropped
        or result.entries_dropped != before.entries_dropped
    )
    if changed:
        result.per_lo_changes.append({
            "lo_id": _lo_id(lo),
            "structured_chunk_ids_dropped":
                result.structured_chunk_ids_dropped - before.structured_chunk_ids_dropped,
            "legacy_strings_dropped":
                result.legacy_strings_dropped - before.legacy_strings_dropped,
            "flat_chunk_ids_dropped":
                result.flat_chunk_ids_dropped - before.flat_chunk_ids_dropped,
            "entries_dropped":
                result.entries_dropped - before.entries_dropped,
        })
    return changed


def _iter_lo_dicts(synthesized: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield every LO dict across the on-disk synthesized-objectives shape.

    Walks ``terminal_objectives`` (flat list), ``chapter_objectives`` (either a
    flat list OR the WS5 group-of-groups shape with a nested ``objectives``
    list), and ``learning_outcomes`` (flat list). De-dupes by object identity
    so a dict referenced from two collections is counted once.
    """
    seen_ids: Set[int] = set()

    def _emit(item: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(item, dict) and id(item) not in seen_ids:
            seen_ids.add(id(item))
            yield item

    for key in ("terminal_objectives", "learning_outcomes"):
        coll = synthesized.get(key)
        if isinstance(coll, list):
            for item in coll:
                yield from _emit(item)

    chapter = synthesized.get("chapter_objectives")
    if isinstance(chapter, list):
        for item in chapter:
            if isinstance(item, dict) and isinstance(item.get("objectives"), list):
                for sub in item["objectives"]:
                    yield from _emit(sub)
            else:
                yield from _emit(item)


def _emit_capture(capture: Any, result: SanitizeResult) -> None:
    if capture is None:
        return
    try:
        capture.log_decision(
            decision_type="objective_chunk_prune",
            decision=(
                f"citation_sanitize: dropped {result.structured_chunk_ids_dropped} "
                f"structured chunk_id(s), {result.legacy_strings_dropped} legacy "
                f"string(s), {result.flat_chunk_ids_dropped} flat id(s), "
                f"{result.entries_dropped} fabricated entry/entries across "
                f"{result.los_mutated}/{result.los_scanned} LO(s)"
            ),
            rationale=(
                "Write-time citation sanitizer removed provably-unresolvable "
                "source_refs / source_chunk_ids entries (fabricated topic-label "
                "or statement-echo citations the local synthesizer emitted) that "
                "resolve against NOTHING in the current chunkset/textbook-"
                "structure universe (chunk_universe check + textbook_universe "
                "ref check), converting a would-be ORPHANED_CITATIONS critical "
                "block into the benign no-grounding warning path. "
                "Anti-fabrication: set-membership REMOVAL only — no id invented, "
                "re-pointed, or reordered."
            ),
            alternatives_considered=[
                {
                    "option": "Keep every supplied citation identifier",
                    "reason_rejected": (
                        f"Rejected because {result.entries_dropped} citation entries "
                        f"did not resolve for {result.los_mutated} objectives."
                    ),
                },
                {
                    "option": "Replace unresolved identifiers by cosine re-selection",
                    "reason_rejected": (
                        f"Rejected because deterministic membership removed "
                        f"{result.entries_dropped} entries across {result.los_scanned} "
                        "objectives and this write seam has no scored replacement pool."
                    ),
                },
            ],
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the phase
        logger.debug("citation_sanitize capture failed (%s); continuing", exc)


def sanitize_synthesized_objectives(
    synthesized: Dict[str, Any],
    *,
    chunk_universe: Optional[Set[str]] = None,
    structure_universe: Optional[Set[str]] = None,
    capture: Optional[Any] = None,
    enabled: Optional[bool] = None,
) -> SanitizeResult:
    """Sanitize the ``synthesized_objectives`` document IN PLACE.

    Drops every ``source_refs`` / ``source_chunk_ids`` citation that fails
    set-membership against ``chunk_universe`` (chunk ids + sourceIds) or, for
    legacy string refs, the union with ``structure_universe`` (chapter/section
    ids). No-op (``available=False``) when the gate is disabled or BOTH
    universes are empty (nothing can be proven wrong — byte-identical output).
    """
    result = SanitizeResult()
    if not resolve_sanitize_citations(enabled):
        return result
    chunk_universe = chunk_universe or set()
    structure_universe = structure_universe or set()
    if not chunk_universe and not structure_universe:
        # No universe to resolve against → mirror the gate's graceful-degrade
        # no-op (cannot prove any id fabricated). Byte-identical output.
        return result

    for lo in _iter_lo_dicts(synthesized):
        result.los_scanned += 1
        if _sanitize_lo(
            lo,
            chunk_universe=chunk_universe,
            structure_universe=structure_universe,
            result=result,
        ):
            result.los_mutated += 1

    result.available = True
    if result.los_mutated:
        _emit_capture(capture, result)
    return result


__all__ = [
    "ENV_SANITIZE_CITATIONS",
    "SanitizeResult",
    "build_chunk_universe",
    "resolve_sanitize_citations",
    "sanitize_synthesized_objectives",
]

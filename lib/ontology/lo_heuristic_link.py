"""W1b.5 — opt-in heuristic learning-outcome linker (EXISTING ids only).

The DART chunking phase emits chunks BEFORE course-planning mints the canonical
``TO-NN`` / ``CO-NN`` id set, so the chunker's per-chunk
``learning_outcome_refs[]`` is empty on initial emit. The canonical back-fill
(``MCP/tools/pipeline_tools.py::_backfill_dart_chunk_lo_refs``) re-scans the
on-disk chunks for LITERAL LO-id mentions via
``lib.ontology.learning_objectives.scan_lo_refs`` — but a DART chunk almost
never contains the literal string ``TO-03`` (those ids are minted downstream),
so the exact scan links very few chunks.

This module adds an OPT-IN heuristic arm. When ``ED4ALL_CHUNK_LO_HEURISTIC`` is
truthy, a chunk with no exact-id match is linked to an objective by
CONTENT-TOKEN overlap between the chunk text and the objective STATEMENT — but
the returned ids are drawn EXCLUSIVELY from the supplied objective set (a real,
already-minted id universe). Anti-fabrication is structural: the function can
only ever return an id that was passed in; it never invents a ``TO-NN`` /
``CO-NN``.

Default OFF → the resolver returns False and no caller runs the heuristic, so
every chunkset stays byte-identical to the exact-scan-only baseline.

Pure / deterministic / embedding-free (SBIR posture — no model load): a
whitespace tokeniser with the shared :data:`lib.ontology.lexical_concept_seeds.
FUNCTION_WORDS` stoplist, token-containment scoring, stable sort.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional

from lib.ontology.lexical_concept_seeds import FUNCTION_WORDS

__all__ = [
    "ENV_LO_HEURISTIC",
    "resolve_lo_heuristic_enabled",
    "heuristic_link_chunk",
    "heuristic_link_chunks",
    "DEFAULT_MIN_OVERLAP",
    "DEFAULT_MAX_LINKS",
    "MIN_OBJECTIVE_TOKENS",
]

#: Env var gating the heuristic linker (default OFF → byte-identical).
ENV_LO_HEURISTIC = "ED4ALL_CHUNK_LO_HEURISTIC"

#: A chunk links to an objective when this fraction of the objective's content
#: tokens appear in the chunk text (containment, objective-normalised).
DEFAULT_MIN_OVERLAP = 0.5

#: Never assign more than this many heuristic links to a single chunk.
DEFAULT_MAX_LINKS = 3

#: Objectives with fewer than this many content tokens are too generic to
#: link reliably (e.g. "Understand SHACL") — skipped to avoid false positives.
MIN_OBJECTIVE_TOKENS = 3

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_:-]+")
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_lo_heuristic_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """Resolve ``ED4ALL_CHUNK_LO_HEURISTIC`` (parse-with-fallback, default OFF)."""
    src = env if env is not None else os.environ
    return src.get(ENV_LO_HEURISTIC, "").strip().lower() in _TRUTHY


def _content_tokens(text: str) -> set:
    """Lowercase content tokens (len>2, minus function words)."""
    out: set = set()
    for tok in _TOKEN_RE.findall(text or ""):
        low = tok.lower()
        if len(low) > 2 and low not in FUNCTION_WORDS:
            out.add(low)
    return out


def heuristic_link_chunk(
    chunk_text: str,
    objectives: Iterable[Dict[str, Any]],
    *,
    allowed_ids: Optional[Iterable[str]] = None,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    max_links: int = DEFAULT_MAX_LINKS,
) -> List[str]:
    """Return the existing objective ids best matching ``chunk_text``.

    ``objectives`` is an iterable of ``{"id": str, "statement": str}`` dicts
    (the already-minted objective universe). For each objective with at least
    :data:`MIN_OBJECTIVE_TOKENS` content tokens, containment =
    ``|obj_tokens ∩ chunk_tokens| / |obj_tokens|`` is computed; objectives at or
    above ``min_overlap`` are returned, ranked by containment (id as the stable
    tie-break), capped at ``max_links``.

    Anti-fabrication: the returned ids are a subset of the ids present in
    ``objectives`` (and, when ``allowed_ids`` is given, additionally intersected
    with it) — the function can never emit an id it wasn't handed.
    """
    chunk_tokens = _content_tokens(chunk_text)
    if not chunk_tokens:
        return []
    allow = (
        {str(x).upper() for x in allowed_ids if x}
        if allowed_ids is not None
        else None
    )
    scored: List[tuple] = []
    for obj in objectives:
        if not isinstance(obj, dict):
            continue
        oid = str(obj.get("id") or "").strip()
        if not oid:
            continue
        if allow is not None and oid.upper() not in allow:
            continue
        obj_tokens = _content_tokens(str(obj.get("statement") or ""))
        if len(obj_tokens) < MIN_OBJECTIVE_TOKENS:
            continue
        overlap = len(obj_tokens & chunk_tokens) / len(obj_tokens)
        if overlap >= min_overlap:
            scored.append((overlap, oid))
    # Rank strongest-first; id as deterministic tie-break.
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [oid for _, oid in scored[: max(0, int(max_links))]]


def heuristic_link_chunks(
    chunks: List[Dict[str, Any]],
    objectives: Iterable[Dict[str, Any]],
    *,
    capture: Optional[Any] = None,
    only_when_empty: bool = True,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    max_links: int = DEFAULT_MAX_LINKS,
) -> Dict[str, Any]:
    """Populate ``learning_outcome_refs`` on ``chunks`` via the heuristic linker.

    Mutates each chunk in place, UNION-merging heuristic matches onto any
    existing ``learning_outcome_refs``. When ``only_when_empty`` (default), a
    chunk that already carries at least one ref is skipped (the exact scan wins;
    the heuristic only rescues un-linked chunks). Returns a summary dict.

    When a :class:`lib.decision_capture.DecisionCapture` is supplied, ONE
    ``content_selection`` decision event is logged (dynamic rationale: chunks
    scanned / linked, refs added, thresholds, objective-universe size) so the
    scan is auditable/replayable. Anti-fabrication: only ids present in
    ``objectives`` can land on a chunk.
    """
    obj_list = [o for o in objectives if isinstance(o, dict) and o.get("id")]
    allowed = {str(o["id"]).upper() for o in obj_list}
    scanned = 0
    linked = 0
    refs_added = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        existing = [str(r).upper() for r in (chunk.get("learning_outcome_refs") or []) if r]
        if only_when_empty and existing:
            continue
        scanned += 1
        matches = heuristic_link_chunk(
            str(chunk.get("text") or ""),
            obj_list,
            allowed_ids=allowed,
            min_overlap=min_overlap,
            max_links=max_links,
        )
        if not matches:
            continue
        merged = sorted(set(existing) | set(matches))
        added = len(merged) - len(existing)
        if added > 0:
            chunk["learning_outcome_refs"] = merged
            linked += 1
            refs_added += added

    if capture is not None:
        try:
            capture.log_decision(
                decision_type="content_selection",
                decision=f"heuristic_lo_link: linked={linked}",
                rationale=(
                    "W1b.5 heuristic LO linker (existing-ids-only): "
                    f"chunks_scanned={scanned}, chunks_linked={linked}, "
                    f"refs_added={refs_added}, objective_universe={len(obj_list)}, "
                    f"min_overlap={min_overlap}, max_links={max_links}, "
                    f"only_when_empty={only_when_empty}."
                ),
            )
        except Exception:  # noqa: BLE001 — capture must never break linking
            pass

    return {
        "chunks_scanned": scanned,
        "chunks_linked": linked,
        "refs_added": refs_added,
        "objective_universe": len(obj_list),
    }

"""Rule: derive ``corrected-by-chunk`` edges from chunks' misconception lists.

GPT Feedback (12 May 2026) item 4 promotes Misconception to a
first-class learning target by materialising three implicit pointers
that already live in the corpus as typed edges. This rule materialises
the chunk-corrects-misconception relation: every chunk that emits a
misconception in ``chunk.misconceptions[]`` ALSO supplies a non-empty
``correction`` (required by ``chunk_v4.schema.json::Misconception``).
The chunk IS the corrector-of-record::

    misconception_id --corrected-by-chunk--> chunk_id

Federation-by-convention: ``source`` is a misconception ID
(``mc_[0-9a-f]{16}`` per ``schemas/knowledge/misconception.schema.json``);
``target`` is a chunk ID. No new node types are added to the
concept-graph schema.

Confidence is ``1.0`` — the chunk emitting the misconception is
explicit in the upstream artifact.

**Signal availability.** Unlike the
``misconception_of_from_misconception_ref`` rule (which depends on
``misconception.concept_id`` being populated upstream — currently 0%
in the rdf-shacl-551 corpus per Wave 82's audit), this rule fires
on every chunk that carries ``misconceptions[]``. Production
corpora today emit those reliably, so this rule starts producing
edges immediately on the next graph rebuild.

Deterministic: output sorted by (source, target); duplicates on the
same (misconception_id, chunk_id) pair are collapsed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from lib.ontology.misconception_id import canonical_mc_id

RULE_NAME = "corrected_by_from_chunk_misconception"
RULE_VERSION = 1
EDGE_TYPE = "corrected-by-chunk"


def infer(
    chunks: List[Dict[str, Any]],
    course: Dict[str, Any] | None,
    concept_graph: Dict[str, Any],
    **_: Any,
) -> List[Dict[str, Any]]:
    """Emit ``corrected-by-chunk`` edges from chunk misconception lists.

    Args:
        chunks: Pipeline chunks. Each chunk may carry ``misconceptions[]``
            (Courseforge-emitted dict entries with required ``correction``
            + at least one of ``misconception`` / ``statement``; or plain
            strings from non-Courseforge IMSCC passthrough — string
            entries skip because there's no correction to anchor on).
        course: Unused; interface parity.
        concept_graph: Unused; endpoints reference external namespaces.

    Returns:
        A deterministically-ordered list of edge dicts.
    """
    del course, concept_graph  # unused; interface parity

    if not chunks:
        return []

    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = chunk.get("id")
        if not chunk_id:
            continue
        raw = chunk.get("misconceptions") or []
        for entry in raw:
            if not isinstance(entry, dict):
                # Non-Courseforge string-only misconceptions skip — there
                # is no correction text to anchor the edge on.
                continue
            # Wave 74 alias: subagents may emit ``statement`` instead of
            # ``misconception``. Mirror the chunk-schema anyOf.
            statement = (
                entry.get("misconception") or entry.get("statement") or ""
            ).strip()
            correction = (entry.get("correction") or "").strip()
            if not statement or not correction:
                continue
            # Wave 72 lock-step: bloom_level participates in the seed
            # whenever populated, matching ``_build_misconceptions_for_graph``.
            bloom_level = (entry.get("bloom_level") or "").strip().lower()
            mc_id = canonical_mc_id(statement, correction, bloom_level)
            key = (mc_id, chunk_id)
            if key in seen:
                continue
            seen[key] = {
                "source": mc_id,
                "target": chunk_id,
                "type": EDGE_TYPE,
                "confidence": 1.0,
                "provenance": {
                    "rule": RULE_NAME,
                    "rule_version": RULE_VERSION,
                    "evidence": {
                        "misconception_id": mc_id,
                        "chunk_id": chunk_id,
                    },
                },
            }

    return sorted(seen.values(), key=lambda e: (e["source"], e["target"]))

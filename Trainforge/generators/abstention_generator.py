"""Abstention SFT pair generator (audit 2026-04-30 fix — Wave 124).

The cc07cc76 SLM adapter scored hallucination_rate=0.63 / faithfulness=0.37
because the training corpus had zero pairs that taught the model to say
"the source does not establish X" — distinct from "X is false". The eval
harness probes negative_grounding by asking about concepts the chunk
does NOT address; without abstention training the adapter happily
hallucinates an affirmative answer.

This generator reads ``pedagogy_graph.json`` and emits deterministic
"absent-edge" probes:

  * For each chunk C, the graph encodes a small set of concepts C
    actually addresses (via ``assesses``, ``exemplifies``,
    ``derives_from_objective``, ``addresses_misconception`` edges).
  * The "silent" set is every other concept node in the graph minus
    that addressed set.
  * For each chunk we deterministically sample K silent concepts
    (seeded by ``seed + chunk_idx``) and emit a pair whose completion
    is a grounded "no, the chunk does not address X" — the model sees
    the truthful surface form for abstention rather than guessing yes.

The pair shape carries:

  * ``content_type="abstention_probe"`` — downstream filters /
    diversity scorers can isolate the cohort without re-parsing prompts.
  * ``bloom_level="understand"`` — abstention is recognising the
    boundary of what the source establishes, a comprehension-tier act
    rather than recall.
  * ``template_id="abstention.no_edge"`` — keeps the diversity
    distribution legible.
  * ``concept_tags=[K.id]`` — anchors the silent concept so a future
    eval can re-verify the abstention claim.
  * ``abstention_polarity="absent"`` — marker field for selective
    sampling (mirrors the kg_metadata_polarity convention).

Decision capture: one ``abstention_generation`` event per emitted pair,
rationale interpolating chunk_id, K.id, and the count of addressed
concepts (so audit replay can spot a chunk-loop that emits pairs against
an empty addressed set, which would degrade to a generic "I don't know"
fallback). The decision_capture_id on every pair anchors to the most
recent event.
"""
from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# Edges that count as "the chunk addresses this concept". The four
# Wave 91+ pedagogical edge types where the chunk is the source.
_ADDRESS_RELATIONS = (
    "assesses",
    "exemplifies",
    "derives_from_objective",
    "addresses_misconception",
)

# Default Bloom level for abstention pairs. Recognising the boundary of
# what the source establishes is a comprehension act, not pure recall.
_ABSTENTION_BLOOM_LEVEL = "understand"

# Default total cap. Mirrors the kg_metadata generator's cap policy.
DEFAULT_MAX_PAIRS = 1000

# Number of silent concepts sampled per chunk. Each sample becomes one
# abstention pair. Kept at 1 by default so the cohort doesn't dominate
# the corpus; raise via call site for adapter retraining experiments.
DEFAULT_SILENT_PER_CHUNK = 1


@dataclass
class AbstentionStats:
    """Counts returned from :func:`generate_abstention_pairs`."""

    chunks_total: int = 0
    chunks_with_silent: int = 0
    chunks_skipped_all_addressed: int = 0
    chunks_skipped_no_concepts: int = 0
    pairs_emitted: int = 0
    capped_at_max_pairs: bool = False
    per_chunk: Dict[str, int] = field(default_factory=dict)


def _load_concept_universe(graph: Dict[str, Any]) -> Dict[str, str]:
    """Map concept_id -> human-readable surface form.

    Picks every node whose class is "Concept" (case-insensitive). The
    surface form falls back to the id when no ``label`` / ``surface_form``
    field is present. Non-concept nodes (Chunk / Module / BloomLevel /
    Misconception) are ignored — they are never "silent concepts" the
    chunk could fail to address.
    """
    out: Dict[str, str] = {}
    for node in graph.get("nodes", []) or []:
        cls = str(node.get("class", "")).lower()
        if cls != "concept":
            continue
        nid = node.get("id")
        if not isinstance(nid, str) or not nid:
            continue
        label = node.get("label") or node.get("surface_form") or nid
        out[nid] = str(label)
    return out


def _load_chunk_addresses(
    graph: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Map chunk_id -> [concept_id, ...] addressed concepts.

    Iterates pedagogy_graph edges, picks the four edges where the chunk
    is the source and the relation is one of `_ADDRESS_RELATIONS`.
    Targets that aren't real concept nodes (e.g. a misconception node
    target on `addresses_misconception`) fall through silently — they
    are not part of the silent-concept universe.
    """
    out: Dict[str, List[str]] = {}
    for edge in graph.get("edges", []) or []:
        rt = edge.get("relation_type")
        s = edge.get("source")
        t = edge.get("target")
        if rt not in _ADDRESS_RELATIONS:
            continue
        if not (isinstance(s, str) and isinstance(t, str)):
            continue
        out.setdefault(s, []).append(t)
    return out


def _surface_form(concept_id: str, lookup: Dict[str, str]) -> str:
    """Render a concept's user-facing surface form."""
    return lookup.get(concept_id, concept_id)


def _addressed_surface_forms(
    addressed_ids: List[str],
    lookup: Dict[str, str],
    rng: random.Random,
    sample_size: int = 2,
) -> List[str]:
    """Pick up to `sample_size` addressed-concept surface forms.

    Deterministic via the supplied RNG. Returns fewer than
    `sample_size` items when the addressed list is short; returns
    an empty list when the addressed set is empty (caller swaps in a
    generic completion).
    """
    if not addressed_ids:
        return []
    pool = sorted(addressed_ids)
    rng.shuffle(pool)
    picks = pool[:sample_size]
    return [_surface_form(c, lookup) for c in picks]


def _build_completion(
    *,
    silent_surface: str,
    addressed_surfaces: List[str],
) -> str:
    """Render the abstention completion.

    Two shapes; the floor is 50 chars to satisfy the schema.

    * When `addressed_surfaces` has >=2 items, the completion grounds
      the abstention in real chunk content ("the chunk addresses A and
      B but does not establish a relationship with X").
    * When the addressed list is short (<2 items), the completion
      falls back to a chunk-anchored "no, based on the encoded edges"
      shape that still reads as honest abstention.
    """
    if len(addressed_surfaces) >= 2:
        a, b = addressed_surfaces[0], addressed_surfaces[1]
        return (
            f"No, there is no evidence in the source. The chunk "
            f"addresses {a} and {b} but does not establish a "
            f"relationship with {silent_surface}."
        )
    if len(addressed_surfaces) == 1:
        a = addressed_surfaces[0]
        return (
            f"No, there is no evidence in the source. The chunk "
            f"addresses {a} but does not establish a relationship "
            f"with {silent_surface}."
        )
    return (
        f"No, the chunk does not address {silent_surface} based on "
        f"the encoded edges in the pedagogy graph."
    )


def _build_pair(
    *,
    chunk_id: str,
    chunk_surface: str,
    silent_id: str,
    silent_surface: str,
    addressed_surfaces: List[str],
    decision_capture_id: str,
    seed: int,
) -> Dict[str, Any]:
    """Render one abstention instruction-pair record.

    Pair shape conforms to ``schemas/knowledge/instruction_pair.schema.json``.
    The completion is at least 50 chars by construction; the prompt is
    at least 40 chars when the chunk_surface + silent_surface combine
    (defensive caller already ensured both are non-empty strings).
    """
    prompt = (
        f"Does the chunk teaching {chunk_surface} address "
        f"{silent_surface}?"
    )
    completion = _build_completion(
        silent_surface=silent_surface,
        addressed_surfaces=addressed_surfaces,
    )
    pair: Dict[str, Any] = {
        "prompt": prompt,
        "completion": completion,
        "chunk_id": chunk_id,
        # Abstention pairs do not map to a particular LO — they teach
        # the boundary of what the chunk teaches. We carry an
        # "abstention" sentinel so the schema's minItems=1 is met
        # without conflating with real LO references.
        "lo_refs": ["abstention"],
        "bloom_level": _ABSTENTION_BLOOM_LEVEL,
        "content_type": "abstention_probe",
        "seed": seed,
        "decision_capture_id": decision_capture_id,
        "template_id": "abstention.no_edge",
        "provider": "mock",
        "schema_version": "v1",
        "requires_source_citation": False,
        "expected_response": "No.",
        "concept_tags": [silent_id],
        "abstention_polarity": "absent",
    }
    return pair


def _last_event_id(capture: Any) -> str:
    """Return the event_id of the most recent decision logged via `capture`.

    Mirrors `synthesize_training._last_event_id` so the emitted pairs
    carry valid `decision_capture_id` strings (Wave 112 invariant).
    """
    decisions = getattr(capture, "decisions", None) or []
    if not decisions:
        raise RuntimeError(
            "abstention_generator: capture has no logged decisions; "
            "log a stage-start decision before generating pairs."
        )
    last = decisions[-1]
    return str(last.get("event_id", "")) if isinstance(last, dict) else ""


def _validate_pair(pair: Dict[str, Any]) -> None:
    """Validate a single pair against `instruction_pair.schema.json`.

    Mirrors `kg_metadata_generator`'s schema-validate-on-emit policy:
    fail loud on shape drift rather than poison the corpus. Caller
    catches `jsonschema.ValidationError` at higher levels if needed.
    """
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - dev-test dep
        return
    schema_path = (
        PROJECT_ROOT / "schemas" / "knowledge" / "instruction_pair.schema.json"
    )
    if not schema_path.exists():  # pragma: no cover - missing schema
        return
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(pair, schema)


def generate_abstention_pairs(
    pedagogy_graph: Dict[str, Any],
    *,
    capture: Any,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    silent_per_chunk: int = DEFAULT_SILENT_PER_CHUNK,
    seed: int = 17,
) -> Tuple[List[Dict[str, Any]], AbstentionStats]:
    """Emit abstention SFT pairs from a pedagogy graph.

    Args:
        pedagogy_graph: Loaded `pedagogy_graph.json` (dict with `nodes`
            + `edges`).
        capture: A `DecisionCapture`-shaped object exposing
            `log_decision(...)` and a `decisions` list. Required: every
            emitted pair anchors `decision_capture_id` to the most
            recent event, and the generator emits one
            ``abstention_generation`` event per pair.
        max_pairs: Hard cap on emitted pairs. The chunk loop breaks
            cleanly mid-iteration when the cap is hit.
        silent_per_chunk: Number of silent concepts to sample per
            chunk. Default 1 keeps the cohort small (~1 pair per chunk
            with addressed concepts).
        seed: Base RNG seed. Each chunk's sample uses
            `random.Random(seed + chunk_idx)` so the same graph + same
            seed produce identical output.

    Returns:
        `(pairs, stats)` — the pair list (instruction_pair shape) and
        an `AbstentionStats` with per-chunk counts.
    """
    if not isinstance(pedagogy_graph, dict):
        raise TypeError("pedagogy_graph must be a dict")
    if max_pairs <= 0:
        raise ValueError(f"max_pairs must be > 0, got {max_pairs}")
    if silent_per_chunk <= 0:
        raise ValueError(
            f"silent_per_chunk must be > 0, got {silent_per_chunk}"
        )
    if capture is None:
        raise ValueError(
            "abstention_generator requires a DecisionCapture (got None); "
            "every emitted pair anchors decision_capture_id to a per-pair "
            "abstention_generation event."
        )

    concept_lookup = _load_concept_universe(pedagogy_graph)
    chunk_addresses = _load_chunk_addresses(pedagogy_graph)

    pairs: List[Dict[str, Any]] = []
    stats = AbstentionStats()

    if not concept_lookup or not chunk_addresses:
        return pairs, stats

    all_concept_ids = set(concept_lookup.keys())
    chunk_ids_sorted = sorted(chunk_addresses.keys())
    stats.chunks_total = len(chunk_ids_sorted)

    for chunk_idx, chunk_id in enumerate(chunk_ids_sorted):
        if stats.pairs_emitted >= max_pairs:
            stats.capped_at_max_pairs = True
            break

        addressed_ids = chunk_addresses.get(chunk_id, [])
        # Filter to addressed concepts that actually exist in the
        # concept universe (drops cross-class targets like Misconception
        # nodes addressed via `addresses_misconception`).
        addressed_concept_ids = [
            c for c in addressed_ids if c in all_concept_ids
        ]
        if not addressed_concept_ids:
            stats.chunks_skipped_no_concepts += 1
            continue

        silent_ids = sorted(all_concept_ids - set(addressed_concept_ids))
        if not silent_ids:
            stats.chunks_skipped_all_addressed += 1
            continue

        # Per-chunk RNG seeded by base seed + chunk index so the same
        # graph + same seed -> identical sample. Distinct seed per
        # chunk avoids a single skewed sample collapsing the whole
        # cohort onto the same silent concept.
        local_rng = random.Random(seed + chunk_idx)
        # Sample silent concepts (without replacement). When the
        # silent set is smaller than `silent_per_chunk`, we take all
        # of them.
        k = min(silent_per_chunk, len(silent_ids))
        sampled_silent = local_rng.sample(silent_ids, k)

        # Pick up to two addressed surface forms for the completion
        # using the same chunk-local RNG. The shuffle is done in
        # `_addressed_surface_forms` so it doesn't mutate
        # `addressed_concept_ids` here.
        addressed_surfaces = _addressed_surface_forms(
            addressed_concept_ids,
            concept_lookup,
            local_rng,
            sample_size=2,
        )

        chunk_surface = _surface_form(chunk_id, concept_lookup) \
            if chunk_id in concept_lookup else chunk_id

        per_chunk_emitted = 0
        for silent_id in sampled_silent:
            if stats.pairs_emitted >= max_pairs:
                stats.capped_at_max_pairs = True
                break

            silent_surface = _surface_form(silent_id, concept_lookup)

            # Per-emit decision. Rationale interpolates dynamic
            # signals so audit replay can distinguish a chunk that
            # legitimately addresses few concepts from one that has
            # bug-poisoned edges. Wave 22+ alternatives_considered
            # convention: list of {option, reason_rejected} dicts.
            capture.log_decision(
                decision_type="abstention_generation",
                decision=(
                    f"Emitting abstention pair for chunk={chunk_id!r}: "
                    f"silent_concept={silent_id!r}, "
                    f"addressed_count={len(addressed_concept_ids)}."
                ),
                rationale=(
                    f"The chunk's addressed-concept set "
                    f"({len(addressed_concept_ids)} concepts via "
                    f"{_ADDRESS_RELATIONS} edges) excludes "
                    f"{silent_id!r}; silent universe has "
                    f"{len(silent_ids)} candidate concepts, sampled "
                    f"{k} per chunk. seed={seed}+chunk_idx={chunk_idx}."
                ),
                alternatives_considered=[
                    {
                        "option": "use random non-graph concept ids as silent set",
                        "reason_rejected": (
                            "the model could learn to abstain on any "
                            "string it doesn't recognise rather than on "
                            "concepts truly absent from this chunk's "
                            "edges."
                        ),
                    },
                    {
                        "option": "skip chunks with fewer than 2 addressed concepts",
                        "reason_rejected": (
                            "drops abstention coverage on the very "
                            "chunks most likely to hallucinate; the "
                            "single-concept fallback completion still "
                            "teaches the abstention surface form."
                        ),
                    },
                ],
            )
            decision_id = _last_event_id(capture)

            pair = _build_pair(
                chunk_id=chunk_id,
                chunk_surface=chunk_surface,
                silent_id=silent_id,
                silent_surface=silent_surface,
                addressed_surfaces=addressed_surfaces,
                decision_capture_id=decision_id,
                seed=seed,
            )
            _validate_pair(pair)
            pairs.append(pair)
            per_chunk_emitted += 1
            stats.pairs_emitted += 1

        if per_chunk_emitted > 0:
            stats.chunks_with_silent += 1
            stats.per_chunk[chunk_id] = per_chunk_emitted

    return pairs, stats


__all__ = [
    "DEFAULT_MAX_PAIRS",
    "DEFAULT_SILENT_PER_CHUNK",
    "AbstentionStats",
    "generate_abstention_pairs",
]

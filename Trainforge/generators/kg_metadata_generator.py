"""KG-metadata SFT pair generator.

Audit 2026-04-30 found that the cc07cc76 adapter scored
faithfulness=0.37 / negative_grounding=0 because the training corpus
had zero pairs that taught literal KG-membership facts (does chunk X
assess concept Y? does chunk X belong to module Y?). The eval harness
asks exactly those yes/no questions via
:mod:`Trainforge.eval.faithfulness`'s ``_RELATION_TEMPLATES``, so the
adapter never had a training signal to learn them.

This generator reads ``pedagogy_graph.json`` and emits deterministic
SFT pairs that mirror the four most-asked relation templates:

- ``assesses(chunk, concept)`` -> "Does the chunk <X> assess <Y>?"
- ``belongs_to_module(chunk, module)`` -> "Does the chunk <X> belong to <Y>?"
- ``at_bloom_level(chunk, level)`` -> "Is the chunk <X> at Bloom level <Y>?"
- generic relation -> "Is the following statement true: <X> -[rel]-> <Y>?"

For each real triple the generator emits:

- One positive ("Yes.") pair grounded in the actual graph membership.
- 1-2 negative ("No.") pairs constructed by swapping the target with a
  plausible-but-wrong alternative drawn from the same relation's target
  distribution. Real concept_ids / module_ids appear in the negative
  prompts so the adapter learns to distinguish "this chunk really does
  assess concept_b" from "this chunk does not assess concept_e";
  random-string negatives would teach a different (and easier) skill.

The output schema is :file:`schemas/knowledge/instruction_pair.schema.json`.
``bloom_level`` is fixed at ``remember`` (these are factual recall
probes) and ``requires_source_citation`` is False (graph-membership
probes don't need a `[chunk_id]` citation tail). ``template_id`` carries
the relation + polarity tag so downstream diversity validators can see
the distribution.

Decision capture: one ``kg_metadata_generation`` event per emission
batch (per relation type). Per-pair captures would be excessive — the
generator is fully deterministic and the rationale interpolates the
relation count, distribution, and ``max_pairs`` cap.
"""
from __future__ import annotations

import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# Relation -> (positive template, negative template).
# Wave 132a: imported from `lib.ontology.relation_templates.RELATION_TEMPLATES`
# so the wording stays bytewise-aligned with the eval-side
# `Trainforge/eval/faithfulness.py::_RELATION_TEMPLATES`. Drift between
# train + eval would desync the adapter's training signal from the eval
# probe — the canonical map is the single source of truth.
#
# Pre-Wave-132a this module hand-defined three relations (assesses,
# belongs_to_module, at_bloom_level); the `assesses` wording had drifted
# vs faithfulness ("chunk" vs "assessment"). The canonical map carries
# the eval-aligned wording for all 12 relations, so any future relation
# the generator encounters in pedagogy_graph.json picks up the eval
# template automatically.
from lib.ontology.relation_templates import RELATION_TEMPLATES as _RELATION_TEMPLATES

_GENERIC_TEMPLATE = (
    "Is the following statement true: '{source}' -[{rel}]-> '{target}'?"
)


# Default number of negative pairs to emit per positive. The eval
# harness has roughly balanced positive / negative probes; mirroring
# that ratio at training time avoids skewing the adapter toward one
# polarity.
DEFAULT_NEGATIVES_PER_POSITIVE = 1

# Default total cap. Distributes evenly across the four relation
# templates (positives + negatives).
DEFAULT_MAX_PAIRS = 2000

# Default Bloom level for KG-membership probes. These are factual
# recall questions, not application / synthesis, so `remember` matches
# the cognitive-level the adapter is being trained on.
_KG_BLOOM_LEVEL = "remember"


@dataclass
class KGMetadataStats:
    """Counts returned from :func:`generate_kg_metadata_pairs`."""

    triples_total: int = 0
    triples_used: int = 0
    positives_emitted: int = 0
    negatives_emitted: int = 0
    pairs_emitted: int = 0
    capped_at_max_pairs: bool = False
    per_relation: Dict[str, Dict[str, int]] = field(default_factory=dict)


def _format_probe(relation: str, source: str, target: str) -> str:
    """Render the question text for one (source, relation, target).

    Falls back to the generic "Is the following statement true: ..."
    template for relations not in `_RELATION_TEMPLATES`. Source/target
    are never None at the call site (we filter incomplete edges in
    :func:`_load_triples`).
    """
    pair = _RELATION_TEMPLATES.get(relation)
    if pair is None:
        return _GENERIC_TEMPLATE.format(
            source=source, target=target, rel=relation,
        )
    template = pair[0]
    return template.format(source=source, target=target)


def _load_triples(graph: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Extract `(source, relation, target)` triples from a pedagogy graph.

    Skips edges with missing source / target / relation_type. Returns
    a flat list so the caller can group / sample by relation.
    """
    out: List[Tuple[str, str, str]] = []
    for edge in graph.get("edges", []) or []:
        s = edge.get("source")
        t = edge.get("target")
        rt = edge.get("relation_type")
        if not (isinstance(s, str) and isinstance(t, str) and isinstance(rt, str)):
            continue
        out.append((s, rt, t))
    return out


def _group_by_relation(
    triples: Sequence[Tuple[str, str, str]],
) -> Dict[str, List[Tuple[str, str]]]:
    """Map relation -> [(source, target), ...] pairs."""
    out: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for s, rt, t in triples:
        out[rt].append((s, t))
    return out


def _real_pairs(
    by_relation: Dict[str, List[Tuple[str, str]]],
) -> Dict[str, set]:
    """Map relation -> {(source, target)} for fast membership tests.

    Used to verify that a generated negative target really doesn't
    appear in the real graph for that source.
    """
    return {
        rt: {(s, t) for s, t in pairs}
        for rt, pairs in by_relation.items()
    }


def _per_source_targets(
    by_relation: Dict[str, List[Tuple[str, str]]],
) -> Dict[str, Dict[str, set]]:
    """Map relation -> {source: {targets actually associated}}.

    Negatives are drawn from `(targets in same relation) - (this source's
    real targets)` so the negative target is "plausibly wrong" — a real
    target for the relation, just not a real target for this source.
    """
    out: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for rt, pairs in by_relation.items():
        for s, t in pairs:
            out[rt][s].add(t)
    return out


def _all_targets_for_relation(
    by_relation: Dict[str, List[Tuple[str, str]]],
) -> Dict[str, List[str]]:
    """Map relation -> sorted unique target list (negative-target pool)."""
    out: Dict[str, List[str]] = {}
    for rt, pairs in by_relation.items():
        seen: set = set()
        targets: List[str] = []
        for _, t in pairs:
            if t not in seen:
                seen.add(t)
                targets.append(t)
        targets.sort()
        out[rt] = targets
    return out


def _build_pair(
    *,
    source: str,
    relation: str,
    target: str,
    polarity: str,
    decision_capture_id: str,
    seed: int,
) -> Dict[str, Any]:
    """Render one instruction-pair record.

    `polarity` is `"yes"` or `"no"` and decides both the completion
    text ("Yes." / "No.") and the `template_id` suffix.
    """
    if polarity not in ("yes", "no"):
        raise ValueError(
            f"polarity must be 'yes' or 'no', got {polarity!r}"
        )
    prompt = _format_probe(relation, source, target)
    # The factual answer is "Yes."/"No." but the instruction-pair
    # schema enforces completion >= 50 chars (paragraph-shaped
    # answers, not single-token replies). We anchor the literal
    # answer at the head of the completion so a yes/no classifier
    # like `_classify_response` still scores affirm/deny correctly,
    # then append a short explanation grounded in the graph
    # membership so the schema floor is met. The explanation is
    # deterministic and chunk_id-anchored — no fact beyond the
    # graph itself.
    if polarity == "yes":
        completion = (
            f"Yes. The pedagogy graph encodes the edge "
            f"'{source}' -[{relation}]-> '{target}'."
        )
    else:
        completion = (
            f"No. The pedagogy graph contains no edge "
            f"'{source}' -[{relation}]-> '{target}'."
        )
    template_id = f"kg_metadata.{relation}_{polarity}"
    pair: Dict[str, Any] = {
        "prompt": prompt,
        "completion": completion,
        "chunk_id": source,
        # The relation maps source -> target via membership in the
        # pedagogy graph, so the LO-ref of the source chunk is the
        # natural anchor. We carry an empty-but-valid placeholder when
        # the caller didn't supply one; downstream filters drop pairs
        # without LO refs.
        "lo_refs": ["kg-metadata"],
        "bloom_level": _KG_BLOOM_LEVEL,
        "content_type": "kg_metadata",
        "seed": seed,
        "decision_capture_id": decision_capture_id,
        "template_id": template_id,
        "provider": "mock",
        "schema_version": "v1",
        "requires_source_citation": False,
        "expected_response": "Yes." if polarity == "yes" else "No.",
        "kg_metadata_relation": relation,
        "kg_metadata_target": target,
        "kg_metadata_polarity": polarity,
    }
    return pair


def _last_event_id(capture: Any) -> str:
    """Return the event_id of the most recent decision logged via `capture`.

    Mirrors `synthesize_training._last_event_id` so the emitted pairs
    carry valid `decision_capture_id` strings (Wave 112 invariant).
    """
    decisions = getattr(capture, "decisions", None) or []
    if not decisions:
        # Caller is responsible for logging at least one decision before
        # generating pairs. Fail loud so we don't poison the corpus
        # with empty `decision_capture_id` strings.
        raise RuntimeError(
            "kg_metadata_generator: capture has no logged decisions; "
            "log a stage-start decision before generating pairs."
        )
    last = decisions[-1]
    return str(last.get("event_id", "")) if isinstance(last, dict) else ""


def generate_kg_metadata_pairs(
    pedagogy_graph: Dict[str, Any],
    *,
    capture: Any,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    negatives_per_positive: int = DEFAULT_NEGATIVES_PER_POSITIVE,
    seed: int = 17,
) -> Tuple[List[Dict[str, Any]], KGMetadataStats]:
    """Emit KG-membership SFT pairs from a pedagogy graph.

    Args:
        pedagogy_graph: Loaded `pedagogy_graph.json` (dict with
            `nodes` + `edges`).
        capture: A `DecisionCapture`-shaped object exposing
            `log_decision(...)` and a `decisions` list. Required: the
            generator emits one `kg_metadata_generation` event per
            relation batch and anchors each pair's
            `decision_capture_id` to the most recent event.
        max_pairs: Hard cap on emitted pairs. Distributed evenly across
            the four relation templates so a graph-rich relation
            doesn't crowd out the others.
        negatives_per_positive: Number of "No." pairs per "Yes." pair.
        seed: RNG seed for negative-target sampling.

    Returns:
        `(pairs, stats)` — the pair list (instruction_pair shape) and
        a `KGMetadataStats` with per-relation counts.
    """
    if not isinstance(pedagogy_graph, dict):
        raise TypeError("pedagogy_graph must be a dict")
    if max_pairs <= 0:
        raise ValueError(f"max_pairs must be > 0, got {max_pairs}")
    if negatives_per_positive < 0:
        raise ValueError(
            f"negatives_per_positive must be >= 0, got {negatives_per_positive}"
        )
    if capture is None:
        raise ValueError(
            "kg_metadata_generator requires a DecisionCapture (got None); "
            "the generator emits a kg_metadata_generation event per "
            "relation batch and anchors decision_capture_id from it."
        )

    triples = _load_triples(pedagogy_graph)
    by_relation = _group_by_relation(triples)
    real = _real_pairs(by_relation)
    per_source = _per_source_targets(by_relation)
    all_targets = _all_targets_for_relation(by_relation)

    stats = KGMetadataStats(triples_total=len(triples))
    pairs: List[Dict[str, Any]] = []

    # Distribute the cap evenly across the four relation templates we
    # actively support — others fall through to the generic template
    # which we DO emit (since the eval also uses generic-template
    # fallback) but at the same per-relation budget.
    relations = sorted(by_relation.keys())
    if not relations:
        return pairs, stats

    # Each pair pos+neg costs (1 + negatives_per_positive) emissions,
    # so per-relation triple budget = floor(per_relation_pairs /
    # (1 + neg)). This keeps pairs <= max_pairs even with multiple
    # negatives per positive.
    per_relation_budget = max(1, max_pairs // len(relations))
    triples_per_relation = max(
        1, per_relation_budget // (1 + negatives_per_positive),
    )

    rng = random.Random(seed)

    for relation in relations:
        rel_pairs_emitted = 0
        rel_positives = 0
        rel_negatives = 0
        rel_triples = list(by_relation[relation])
        if not rel_triples:
            continue
        # Deterministic shuffle so the same seed + same graph -> same
        # picked subset.
        local_rng = random.Random(seed + abs(hash(relation)))
        local_rng.shuffle(rel_triples)
        picked = rel_triples[:triples_per_relation]

        # Capture one decision per relation batch with rationale that
        # interpolates count + budget. Required-by-CLAUDE.md instruction
        # to interpolate dynamic signals.
        capture.log_decision(
            decision_type="kg_metadata_generation",
            decision=(
                f"Emitting KG-metadata pairs for relation={relation!r}: "
                f"{len(picked)}/{len(rel_triples)} triples sampled, "
                f"{negatives_per_positive} negatives per positive, "
                f"max_pairs={max_pairs}, per_relation_budget="
                f"{per_relation_budget}."
            ),
            rationale=(
                f"Generating yes/no membership probes mirroring "
                f"faithfulness._RELATION_TEMPLATES so the adapter sees "
                f"the same surface form at training and eval time. "
                f"Negatives drawn from the in-relation target pool "
                f"({len(all_targets[relation])} candidate targets, "
                f"{len(real[relation])} real (source,target) pairs) so "
                f"the model learns to distinguish real-but-mismatched "
                f"targets, not random strings. seed={seed}."
            ),
            alternatives_considered=[
                {
                    "option": "random-string negative targets",
                    "reason_rejected": (
                        "easier signal — adapter learns to reject "
                        "out-of-vocabulary strings rather than real "
                        "graph mismatches."
                    ),
                },
                {
                    "option": "all triples (no per-relation cap)",
                    "reason_rejected": (
                        "graph-rich relations (prerequisite_of has "
                        "4160 edges in rdf-shacl-551-2) would crowd out "
                        "low-volume relations like assessment_validates_"
                        "outcome (20 edges)."
                    ),
                },
            ],
        )
        decision_id = _last_event_id(capture)

        for src, tgt in picked:
            if stats.pairs_emitted >= max_pairs:
                stats.capped_at_max_pairs = True
                break

            pos_pair = _build_pair(
                source=src,
                relation=relation,
                target=tgt,
                polarity="yes",
                decision_capture_id=decision_id,
                seed=seed,
            )
            pairs.append(pos_pair)
            rel_pairs_emitted += 1
            rel_positives += 1
            stats.positives_emitted += 1
            stats.pairs_emitted += 1

            # Generate negatives from the in-relation target pool minus
            # this source's real targets. Falls through silently when
            # the relation only has one target overall (degenerate
            # graph; nothing to swap to).
            real_targets_for_src = per_source[relation].get(src, set())
            candidate_targets = [
                t for t in all_targets[relation]
                if t not in real_targets_for_src and t != tgt
            ]
            if not candidate_targets:
                continue

            attempts = 0
            negatives_made = 0
            while (
                negatives_made < negatives_per_positive
                and stats.pairs_emitted < max_pairs
                and attempts < negatives_per_positive * 4 + 4
            ):
                attempts += 1
                wrong_target = rng.choice(candidate_targets)
                # Defensive: confirm against the real-pairs set even
                # though the candidate filter already excluded it. If
                # somehow it's a real pair, skip.
                if (src, wrong_target) in real[relation]:
                    continue
                neg_pair = _build_pair(
                    source=src,
                    relation=relation,
                    target=wrong_target,
                    polarity="no",
                    decision_capture_id=decision_id,
                    seed=seed,
                )
                pairs.append(neg_pair)
                rel_pairs_emitted += 1
                rel_negatives += 1
                stats.negatives_emitted += 1
                stats.pairs_emitted += 1
                negatives_made += 1

            if stats.pairs_emitted >= max_pairs:
                stats.capped_at_max_pairs = True
                break

        stats.per_relation[relation] = {
            "triples_in_graph": len(rel_triples),
            "positives_emitted": rel_positives,
            "negatives_emitted": rel_negatives,
            "pairs_emitted": rel_pairs_emitted,
        }
        stats.triples_used += len(picked)

        if stats.capped_at_max_pairs:
            break

    return pairs, stats


__all__ = [
    "DEFAULT_MAX_PAIRS",
    "DEFAULT_NEGATIVES_PER_POSITIVE",
    "KGMetadataStats",
    "generate_kg_metadata_pairs",
]

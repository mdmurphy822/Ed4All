"""Cross-stratum edge consensus aggregator (GPT feedback 12-may, item 2).

The 12 inference rules under ``Trainforge/rag/inference_rules/`` (plus the
``intra_chunk_link`` co-location pass in ``lib/ontology/intra_chunk_linker.py``)
emit edges into ``concept_graph_semantic.json`` independently. GPT feedback
item 2 asked for more than a per-rule ``confidence`` float:

    "Not just confidence, but whether another rule contradicted or
     failed to confirm it."

**Why the original same-pair design was near-degenerate.** The first cut of
this aggregator looked for a DIFFERENT rule firing over the *same*
``(source, target)`` node pair (a confirmation signal) or the opposite
direction (a contradiction). On a real RAG-course graph this left
1891 of 1892 edges ``pending``: same-pair co-fire evidence is structurally
deleted upstream. ``typed_edge_inference._apply_precedence`` collapses
multiple rules' edges on a colliding ``(source, target)`` down to one
surviving edge (``is-a`` > tier-2 > ``related-to``), so by the time the graph
reaches this aggregator the "two rules agreed on the same pair" evidence is
already gone. The lone surviving same-pair confirmation in the measured graph
was a ``defined_by``×``exemplifies`` pair — and even that only confirmed one
of its two edges because the matrix was asymmetric.

**The recoverable signal is cross-stratum triangulation.** The persisted
per-edge provenance still records the *endpoints* each rule fired over — LO
ordinals, chunk IDs, concept slugs, objective IDs. Those survive precedence
because they live on edges of *different types* that never collide. This
aggregator reconstructs auxiliary indexes from that persisted provenance
(first-mention text order, LO→chunk mappings, concept→chunk mappings,
LO→concept targets, assessed (chunk, LO) pairs) and triangulates across
strata:

- **T1** prerequisite (LO order) corroborated / contradicted by first-mention
  text order.
- **T2** ``assesses`` (question→LO) × ``derived-from-objective`` (chunk→LO)
  closing a (chunk, LO) triangle.
- **T3** ``targets-concept`` (LO→C) × ``derived-from-objective`` (chunk→LO) ×
  ``defined-by`` (C→chunk) closing an LO–chunk–concept triangle.
- **T4** ``related-to`` co-location: two co-occurring concepts whose
  first-mention chunks overlap (a *support*-tier signal, never a
  ``confirmed``).

Each edge is stamped two schema-additive fields:

- ``edge_status`` ∈ ``{"pending", "supported", "confirmed", "contradicted",
  "retracted"}``. ``supported`` is the new graded tier for soft
  corroboration (co-location) that strengthens but does not confirm an edge;
  ``retracted`` is produced by the NLI extension (off by default behind
  ``TRAINFORGE_EDGE_NLI``) AND by the opt-in ``retract`` arm of the
  contradicted-edge policy (``TRAINFORGE_CONTRADICTED_EDGE_POLICY``, off by
  default — see :data:`_CONTRADICTED_POLICY_ENV`).
- ``consensus_signals: List[{other_rule, signal, confidence?, detail?}]`` —
  one entry per cross-rule / cross-stratum signal. ``signal`` ∈
  ``{"agree", "disagree", "support"}``.

``consensus_rate`` stays ``confirmed / total`` — ``supported`` is counted and
reported separately (``supported_rate``) so the soft tier never inflates the
headline consensus number. ``T1`` text-order conflicts flow into the existing
``contradictions[]`` list with the distinct detail string
``lo_order_vs_text_order_conflict`` so operators can tell a *soft* ordering
conflict from a *hard* cycle (``circular_prerequisite``).

The aggregator additionally writes ``edge_consensus_report.json`` to the same
directory as the source graph, summarising per-rule confirmed / supported /
contradicted / pending counts, listing every contradicted edge, and rolling
up an ``evidence_diversity`` block (multi-rule pair count + triangulated edge
count).

This module is intentionally pure aggregation — no LLM calls, no network I/O,
no schema validation pass. The schema admits the new fields on legacy graphs
(missing ``edge_status`` validates), and the NLI extension stays stubbed
behind ``TRAINFORGE_EDGE_NLI`` for a follow-on wave.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple, Union


logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"

#: Matrix version is interpolated into decision-capture rationales so
#: a future tweak to the rule-pair map bumps the audit trail
#: deterministically. v2: cross-stratum triangulation redesign.
MATRIX_VERSION = "2026-06-09.v2"

#: Edge types whose pair-equivalence-key is undirected — i.e. an edge
#: ``A → B`` and an edge ``B → A`` of the same type are treated as
#: signals over the SAME node pair. The rest are directed (``A → B``
#: differs from ``B → A``).
_UNDIRECTED_EDGE_TYPES: FrozenSet[str] = frozenset({"related-to", "is-a"})

#: Regex used to parse a chunk ordinal out of a chunk ID (``chunk_00270``,
#: ``demo_course_chunk_00270``, ``chunk-12``). Used by T1's first-mention
#: text-order index. Unparseable chunk IDs are skipped (graceful degrade).
_CHUNK_ORD_RE = re.compile(r"chunk[_-]?(\d+)")

#: Rule-pair confirmation matrix. ``_RULE_PAIR_MATRIX[rule]`` is a
#: 2-tuple ``(confirming_rules, contradicting_rules)``. ``contradicting
#: _rules`` enumerates OTHER rules that contradict; reverse-direction
#: same-rule cycles are detected separately because they're a
#: same-rule signal, not a cross-rule signal.
#:
#: Most same-pair confirmation entries are structurally unreachable on a
#: real graph because ``typed_edge_inference._apply_precedence`` deletes
#: colliding-pair edges upstream (see module docstring). They are retained
#: for the rare surviving same-pair co-fire (the measured
#: ``defined_by``×``exemplifies`` pair) and so the matrix stays a complete
#: registry of every rule name (drift-test target). The recoverable signal
#: is the cross-stratum triangulation in ``_compute_triangulation_signals``.
_RULE_PAIR_MATRIX: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]] = {
    "is_a_from_key_terms": (
        frozenset({"defined_by_from_first_mention", "targets_concept_from_lo"}),
        frozenset(),
    ),
    "prerequisite_from_lo_order": (
        frozenset(),
        frozenset(),
    ),
    "related_from_cooccurrence": (
        frozenset({"is_a_from_key_terms"}),
        frozenset(),
    ),
    "assesses_from_question_lo": (
        frozenset({"derived_from_lo_ref"}),
        frozenset(),
    ),
    "exemplifies_from_example_chunks": (
        frozenset({"defined_by_from_first_mention"}),
        frozenset(),
    ),
    "misconception_of_from_misconception_ref": (
        frozenset(),
        frozenset(),
    ),
    "derived_from_lo_ref": (
        frozenset({"assesses_from_question_lo"}),
        frozenset(),
    ),
    # Symmetry fix: the measured graph's one real same-pair co-fire was a
    # defined_by × exemplifies pair, but the matrix only confirmed the
    # exemplifies side. defined_by_from_first_mention must list exemplifies
    # in its confirming set so BOTH edges of the surviving pair land
    # confirmed.
    "defined_by_from_first_mention": (
        frozenset({"is_a_from_key_terms", "exemplifies_from_example_chunks"}),
        frozenset(),
    ),
    "targets_concept_from_lo": (
        frozenset({"is_a_from_key_terms", "defined_by_from_first_mention"}),
        frozenset(),
    ),
    # Completeness: the four remaining registered rules + the co-location
    # pass had no matrix entry. They take no same-pair confirmation today;
    # their cross-stratum signal (when any) lands via triangulation.
    "corrected_by_from_chunk_misconception": (
        frozenset(),
        frozenset(),
    ),
    "detected_by_from_distractor_misconception_id": (
        frozenset(),
        frozenset(),
    ),
    "interferes_with_outcome_from_misconception_lo": (
        frozenset(),
        frozenset(),
    ),
    # intra_chunk_link is co-location-derived: a co-location signal over its
    # own pair would be tautological, so its confirm set MUST stay empty and
    # T4 explicitly excludes it.
    "intra_chunk_link": (
        frozenset(),
        frozenset(),
    ),
}

#: Rules that emit directed edges where reverse-direction by the SAME
#: rule between the same nodes is a contradiction (cycle).
_CYCLE_DETECTING_RULES: FrozenSet[str] = frozenset({
    "prerequisite_from_lo_order",
    "is_a_from_key_terms",
})

#: NLI extension flag — when truthy AND a chunk-text lookup is supplied to
#: the aggregator, each chunk-anchored edge's predicate is rendered to a
#: hypothesis and scored (via the shared ``lib.classifiers.nli_classifier``
#: DeBERTa singleton) against the cited chunk text. Only a strong
#: CONTRADICTION (>= :data:`_NLI_CONTRADICTION_THRESHOLD`) emits a
#: ``disagree`` signal (mapped to ``edge_status: retracted`` by
#: :meth:`_status_from_signals`); entailment / neutral add no signal in v1
#: (conservative — the arm can only retract, never confirm). Default off →
#: the aggregator runs cross-rule-matrix + triangulation consensus only.
_NLI_FLAG_ENV: str = "TRAINFORGE_EDGE_NLI"

#: Per-run cap on the number of edges scored by the NLI extension (bounds
#: forward passes on the shared card / CPU). Satellite of ``TRAINFORGE_EDGE_NLI``.
_NLI_MAX_EDGES_ENV: str = "TRAINFORGE_EDGE_NLI_MAX_EDGES"
_NLI_DEFAULT_MAX_EDGES: int = 500

#: Contradiction probability at / above which the NLI extension emits a
#: ``disagree`` signal. Mirrors the established ``NliScore`` convention
#: (``contradiction >= 0.5`` — see ``lib.classifiers.nli_classifier``).
_NLI_CONTRADICTION_THRESHOLD: float = 0.5

#: Premise char cap so a long chunk doesn't blow the NLI serving window
#: (the tokenizer truncates too, but bound the input up front).
_NLI_PREMISE_CHAR_CAP: int = 2000

#: Chunk-anchored edge types the NLI extension scores. Each entry names
#: which endpoint carries the chunk id vs. the concept, and the hypothesis
#: template rendered against the cited chunk text. Only these types are
#: scored — the concept↔chunk predicates are the ones with a natural
#: text-entailment reading; LO-order / question→LO edges are skipped.
_NLI_EDGE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "defined_by_from_first_mention": {
        "chunk_endpoint": "target",
        "concept_endpoint": "source",
        "hypothesis": "This text defines {concept}.",
    },
    "exemplifies_from_example_chunks": {
        "chunk_endpoint": "source",
        "concept_endpoint": "target",
        "hypothesis": "This text gives an example of {concept}.",
    },
}

#: Chunkset locations searched by :func:`load_chunk_text_lookup`, relative
#: to a LibV2 course dir (first hit per id wins).
_CHUNK_JSONL_CANDIDATES: Tuple[str, ...] = (
    "dart_chunks/chunks.jsonl",
    "imscc_chunks/chunks.jsonl",
    "corpus/chunks.jsonl",
)

#: Opt-in retraction / decay policy for CONTRADICTED edges. Default unset →
#: stamp-only behaviour (byte-identical to the pre-flag output). The policy
#: is applied ONLY on :meth:`EdgeConsensusAggregator.apply_to_graph` — the
#: downstream complement to the upstream ``TRAINFORGE_PREREQ_LO_ADJACENT_ONLY``
#: mitigation (transitive reduction + 0.6→0.3 confidence demotion). The two
#: compose: the upstream flag thins + demotes the prerequisite closure so
#: fewer edges reach ``contradicted``; this flag governs what happens to the
#: ones that still do.
#:
#: Accepted values:
#:   ``decay``   — multiply a contradicted edge's ``confidence`` by
#:                 ``_DECAY_FACTOR`` (default 0.5). ``edge_status`` stays
#:                 ``contradicted`` (the contradiction evidence is preserved);
#:                 only the confidence weight is attenuated so retrieval /
#:                 pedagogy consumers that rank by confidence down-weight the
#:                 known-bad edge without losing its provenance.
#:   ``retract`` — set ``edge_status`` to ``retracted``. The edge is NOT
#:                 physically deleted (deletion breaks provenance / replay);
#:                 it stays in ``graph['edges']`` clearly statused so an
#:                 ``edge_status``-aware consumer can exclude it. Today's
#:                 in-tree consumers do not filter on ``edge_status``, so a
#:                 ``retracted`` edge remains physically present and
#:                 consumable — the status is the contract; physical removal
#:                 is explicitly out of scope.
_CONTRADICTED_POLICY_ENV: str = "TRAINFORGE_CONTRADICTED_EDGE_POLICY"
_VALID_CONTRADICTED_POLICIES: FrozenSet[str] = frozenset({"decay", "retract"})
_DECAY_FACTOR: float = 0.5

# Type alias for the equivalence key used to bucket edges that
# refer to the same node pair under the same edge type.
_PairKey = Tuple[str, Union[Tuple[str, str], FrozenSet[str]]]


def _canonical_pair_key(source: str, target: str, edge_type: str) -> _PairKey:
    """Compute the pair-equivalence-key for cross-rule confirmation.

    Cross-rule confirmation is "did a DIFFERENT rule fire over the
    same node pair?". We bucket by ``frozenset({source, target})``
    ignoring both edge_type AND direction.
    """
    _ = edge_type  # accepted for symmetry with the reverse key helper
    return ("__pair__", frozenset({source, target}))


def _reverse_pair_key(source: str, target: str, edge_type: str) -> Optional[_PairKey]:
    """The reverse-direction pair-key for cycle detection on directed
    types. Keyed on the specific edge_type because cycles only count
    when both directions share a type. Undirected types return None.
    """
    if edge_type in _UNDIRECTED_EDGE_TYPES:
        return None
    return (edge_type, (target, source))


#: Auxiliary cross-stratum index bundle built in one pass over edges. All
#: members are derived from immutable edge inputs only (endpoints +
#: evidence), so building them twice over the same edge list is byte-stable
#: (idempotency contract).
class _AuxIndexes:
    __slots__ = (
        "first_mention_ord",
        "chunks_by_lo",
        "concept_chunks",
        "targets_by_lo",
        "assessed_chunk_lo",
    )

    def __init__(self) -> None:
        # concept -> earliest chunk ordinal from defined_by edges.
        self.first_mention_ord: Dict[str, int] = {}
        # lo_lower -> {chunk_id} from derived_from_lo_ref edges.
        self.chunks_by_lo: Dict[str, Set[str]] = defaultdict(set)
        # concept -> {chunk_id} from defined_by + exemplifies endpoints.
        self.concept_chunks: Dict[str, Set[str]] = defaultdict(set)
        # lo_lower -> {concept} from targets_concept_from_lo edges.
        self.targets_by_lo: Dict[str, Set[str]] = defaultdict(set)
        # set of (source_chunk_id, lo_lower) from assesses edges' evidence.
        self.assessed_chunk_lo: Set[Tuple[str, str]] = set()


class EdgeConsensusAggregator:
    """Walks a semantic graph and stamps cross-stratum consensus signals.

    Construct with the path to ``concept_graph_semantic.json``; call
    :meth:`build` to return the report dict; call :meth:`write` to
    serialise it to disk; call :meth:`apply_to_graph` to mutate the
    graph in place with the new ``edge_status`` +
    ``consensus_signals[]`` fields.

    Parameters
    ----------
    semantic_graph_path:
        Path to ``concept_graph_semantic.json``. When the file is
        missing or unparseable, :meth:`build` returns the empty-summary
        shape (graceful degrade — never raises).
    course_slug:
        Operator-facing course slug. Surfaces verbatim in the report.
    run_id:
        Workflow / pipeline run identifier. Surfaces verbatim in the
        report and in the decision-capture rationale.
    decision_capture:
        Optional ``DecisionCapture`` instance. When wired, the
        aggregator emits one ``edge_consensus_resolution`` event per
        :meth:`build` call.
    chunk_text_lookup:
        Optional ``{chunk_id: text}`` mapping backing the NLI extension
        (``TRAINFORGE_EDGE_NLI``). Default ``None`` → the NLI extension
        no-ops even when the flag is on (graceful degrade — the aggregator
        only carries the graph path, whose edges reference chunk IDs, not
        chunk TEXT). Build it from the LibV2 course tree via
        :func:`load_chunk_text_lookup`.
    """

    def __init__(
        self,
        semantic_graph_path: Optional[Path],
        *,
        course_slug: str = "",
        run_id: str = "",
        decision_capture: Optional[Any] = None,
        chunk_text_lookup: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.semantic_graph_path = (
            Path(semantic_graph_path) if semantic_graph_path is not None else None
        )
        self.course_slug = str(course_slug or "")
        self.run_id = str(run_id or "")
        self.decision_capture = decision_capture
        # NLI-extension state (TRAINFORGE_EDGE_NLI). ``chunk_text_lookup``
        # None → the extension no-ops. Budget + classifier are (re)resolved
        # per build()/apply_to_graph() call via :meth:`_nli_setup`.
        self._chunk_text_lookup: Optional[Dict[str, str]] = (
            dict(chunk_text_lookup) if chunk_text_lookup else None
        )
        self._nli_budget_remaining: int = 0
        self._nli_classifier: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @staticmethod
    def _build_indexes(
        edges: List[Any],
    ) -> Tuple[
        Dict[_PairKey, List[Dict[str, Any]]],
        Dict[_PairKey, List[Dict[str, Any]]],
        _AuxIndexes,
    ]:
        """Build (pair_index, type_dir_index, aux) in one pass over edges.

        ``pair_index`` keys by node pair (ignoring edge_type) for the
        same-pair cross-rule matrix. ``type_dir_index`` keys by
        ``(edge_type, (source, target))`` for same-rule reverse-direction
        cycle detection. ``aux`` carries the cross-stratum triangulation
        indexes reconstructed from persisted provenance — endpoints +
        evidence only, never mutating the inputs (idempotency).
        """
        pair_index: Dict[_PairKey, List[Dict[str, Any]]] = defaultdict(list)
        type_dir_index: Dict[_PairKey, List[Dict[str, Any]]] = defaultdict(list)
        aux = _AuxIndexes()
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source = edge.get("source")
            target = edge.get("target")
            edge_type = edge.get("type")
            if not isinstance(source, str) or not isinstance(target, str):
                continue
            if not isinstance(edge_type, str):
                continue
            pair_index[_canonical_pair_key(source, target, edge_type)].append(edge)
            type_dir_index[(edge_type, (source, target))].append(edge)

            rule = _edge_rule(edge)
            if not rule:
                continue

            if rule == "defined_by_from_first_mention":
                # source=concept, target=chunk. Record concept->chunk and
                # concept->earliest chunk ordinal.
                aux.concept_chunks[source].add(target)
                ord_val = _chunk_ordinal(target)
                if ord_val is not None:
                    prior = aux.first_mention_ord.get(source)
                    if prior is None or ord_val < prior:
                        aux.first_mention_ord[source] = ord_val
            elif rule == "exemplifies_from_example_chunks":
                # source=chunk, target=concept. concept->chunk via target->source.
                aux.concept_chunks[target].add(source)
            elif rule == "derived_from_lo_ref":
                # source=chunk, target=lo. lo_lower -> {chunk}.
                aux.chunks_by_lo[target.lower()].add(source)
            elif rule == "targets_concept_from_lo":
                # source=lo, target=concept. lo_lower -> {concept}.
                aux.targets_by_lo[source.lower()].add(target)
            elif rule == "assesses_from_question_lo":
                # evidence carries source_chunk_id + objective_id.
                ev = _edge_evidence(edge)
                src_chunk = ev.get("source_chunk_id")
                obj_id = ev.get("objective_id")
                if isinstance(src_chunk, str) and src_chunk and isinstance(obj_id, str) and obj_id:
                    aux.assessed_chunk_lo.add((src_chunk, obj_id.lower()))
        return pair_index, type_dir_index, aux

    def build(self) -> Dict[str, Any]:
        """Build the consensus report dict.

        Reads the graph from disk (best-effort); computes per-edge
        signals; rolls into per-rule and corpus-wide summaries; emits
        the decision-capture event. Never mutates the graph on disk —
        :meth:`apply_to_graph` is the explicit mutation path.

        Consistency-attenuation coherence: ``summary['contradiction_rate']``
        (which ``KGQualityValidator.validate`` uses to attenuate the
        ``consistency`` axis by ``(1 - contradiction_rate)``) counts every
        edge whose CROSS-STRATUM VERDICT is ``contradicted`` — independent of
        the ``TRAINFORGE_CONTRADICTED_EDGE_POLICY`` retract/decay policy.
        ``build()`` deliberately does NOT apply that policy, so a decayed or
        retracted edge STILL counts toward ``contradiction_rate``: a
        contradicted edge remains evidence of contradiction even after its
        confidence is decayed or its status is re-stamped ``retracted``.
        Attenuating contradiction is not the same as erasing the
        contradiction signal, and the consistency axis must keep seeing it.
        """
        graph = self._load_graph()
        edges = _as_list(graph.get("edges"))

        self._nli_setup()
        pair_index, type_dir_index, aux = self._build_indexes(edges)

        per_rule_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {
                "edge_count": 0,
                "confirmed": 0,
                "supported": 0,
                "contradicted": 0,
                "pending": 0,
                "retracted": 0,
            }
        )
        confirmed_count = 0
        supported_count = 0
        contradicted_count = 0
        pending_count = 0
        retracted_count = 0
        contradictions: List[Dict[str, Any]] = []
        contradicting_pair_counter: Counter = Counter()

        # evidence-diversity rollup signals.
        multi_rule_pair_count = 0
        triangulated_edge_count = 0
        seen_pairs: Set[FrozenSet[str]] = set()

        nli_enabled = _flag_is_true(_NLI_FLAG_ENV)

        per_edge_results: List[Tuple[Dict[str, Any], str, List[Dict[str, Any]]]] = []

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source = edge.get("source")
            target = edge.get("target")
            edge_type = edge.get("type")
            if not isinstance(source, str) or not isinstance(target, str):
                continue
            if not isinstance(edge_type, str):
                continue
            this_rule = _edge_rule(edge)
            if not this_rule:
                per_edge_results.append((edge, "pending", []))
                pending_count += 1
                continue

            signals = self._compute_signals(
                edge=edge,
                this_rule=this_rule,
                source=source,
                target=target,
                edge_type=edge_type,
                pair_index=pair_index,
                type_dir_index=type_dir_index,
                aux=aux,
            )

            status = self._status_from_signals(signals)
            per_edge_results.append((edge, status, signals))

            if signals:
                triangulated_edge_count += 1
            pk = frozenset({source, target})

            per_rule_counts[this_rule]["edge_count"] += 1
            if status == "confirmed":
                confirmed_count += 1
                per_rule_counts[this_rule]["confirmed"] += 1
            elif status == "supported":
                supported_count += 1
                per_rule_counts[this_rule]["supported"] += 1
            elif status == "contradicted":
                contradicted_count += 1
                per_rule_counts[this_rule]["contradicted"] += 1
                # Record one contradiction entry per disagreeing signal so
                # the operator-facing report names each contradicting
                # (other_rule, detail) pair. Both hard cycles
                # (circular_prerequisite / type_hierarchy_cycle) and soft
                # T1 text-order conflicts (lo_order_vs_text_order_conflict)
                # flow here off sig["signal"] == "disagree".
                for sig in signals:
                    if sig.get("signal") != "disagree":
                        continue
                    contradicting_pair_counter[
                        (this_rule, sig.get("other_rule") or "")
                    ] += 1
                    contradictions.append({
                        "source": source,
                        "target": target,
                        "type": edge_type,
                        "rule": this_rule,
                        "contradicting_rule": sig.get("other_rule"),
                        "contradicting_signal": sig.get("signal"),
                        "detail": sig.get("detail") or "",
                    })
            elif status == "retracted":
                retracted_count += 1
                per_rule_counts[this_rule]["retracted"] += 1
            else:
                pending_count += 1
                per_rule_counts[this_rule]["pending"] += 1

            # Multi-rule-pair diversity: count a pair once when >1 distinct
            # rule fired over its node-pair bucket.
            if pk not in seen_pairs:
                seen_pairs.add(pk)
                rules_on_pair = {
                    _edge_rule(e)
                    for e in pair_index.get(
                        _canonical_pair_key(source, target, edge_type), ()
                    )
                    if _edge_rule(e)
                }
                if len(rules_on_pair) > 1:
                    multi_rule_pair_count += 1

        total_edges = (
            confirmed_count
            + supported_count
            + contradicted_count
            + pending_count
            + retracted_count
        )
        # consensus_rate stays confirmed / total — supported is reported
        # separately so the soft tier never inflates the headline number.
        consensus_rate = confirmed_count / total_edges if total_edges else 0.0
        supported_rate = supported_count / total_edges if total_edges else 0.0
        contradiction_rate = contradicted_count / total_edges if total_edges else 0.0
        multi_rule_pair_rate = (
            multi_rule_pair_count / len(seen_pairs) if seen_pairs else 0.0
        )

        top_contradictions = [
            f"{a}<->{b}={n}"
            for (a, b), n in contradicting_pair_counter.most_common(3)
        ]

        per_rule_rollup: List[Dict[str, Any]] = []
        for rule, counts in sorted(per_rule_counts.items()):
            per_rule_rollup.append({
                "rule": rule,
                **counts,
            })

        report: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "matrix_version": MATRIX_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "course_slug": self.course_slug,
            "run_id": self.run_id,
            "nli_extension_enabled": nli_enabled,
            "summary": {
                "total_edges": total_edges,
                "confirmed_count": confirmed_count,
                "supported_count": supported_count,
                "contradicted_count": contradicted_count,
                "pending_count": pending_count,
                "retracted_count": retracted_count,
                "consensus_rate": _round(consensus_rate),
                "supported_rate": _round(supported_rate),
                "contradiction_rate": _round(contradiction_rate),
            },
            "evidence_diversity": {
                "multi_rule_pair_count": multi_rule_pair_count,
                "multi_rule_pair_rate": _round(multi_rule_pair_rate),
                "triangulated_edge_count": triangulated_edge_count,
            },
            "per_rule": per_rule_rollup,
            "contradictions": contradictions,
        }

        self._emit_decision(report, top_contradictions=top_contradictions)

        self._last_per_edge_results = per_edge_results

        return report

    def write(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """Serialise :meth:`build`'s output to JSON.

        Defaults to writing alongside the source graph as
        ``edge_consensus_report.json``. Returns the resolved path on
        success, ``None`` when no destination resolves and no override
        was supplied (no-op rather than raise).
        """
        report = self.build()
        if output_path is None:
            if self.semantic_graph_path is None:
                logger.warning(
                    "EdgeConsensusAggregator.write: no output_path and no "
                    "semantic_graph_path; skipping emit."
                )
                return None
            output_path = self.semantic_graph_path.parent / "edge_consensus_report.json"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def apply_to_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate ``graph`` in place by stamping consensus fields.

        Computes per-edge consensus by walking the SAME ``graph['edges']``
        list passed in (NOT the on-disk graph). Idempotent: calling
        apply_to_graph twice produces byte-identical output (the matrix +
        triangulation indexes are deterministic over the same input edges,
        and the aux indexes are rebuilt from immutable endpoints/evidence,
        never from the ``edge_status`` / ``consensus_signals`` fields this
        method writes).

        Contradicted-edge policy (``TRAINFORGE_CONTRADICTED_EDGE_POLICY``):
        when set to ``decay`` / ``retract``, contradicted edges are
        attenuated (confidence × :data:`_DECAY_FACTOR`, status unchanged) or
        re-statused to ``retracted`` (edge retained, never deleted). Unset →
        stamp-only (byte-identical legacy output). An invalid value raises
        :class:`ContradictedEdgePolicyError` BEFORE any edge is mutated, so a
        bad flag never leaves the graph half-written.

        Composition with the upstream prerequisite mitigation
        (``TRAINFORGE_PREREQ_LO_ADJACENT_ONLY``): that flag thins the O(n²)
        prerequisite closure (transitive reduction) and demotes
        text-order-conflicting edges 0.6→0.3 with a
        ``lo_order_vs_text_order_conflict`` evidence flag upstream. Whatever
        survives and still computes to ``contradicted`` here is then governed
        by this policy — e.g. ``decay`` re-multiplies the already-demoted 0.3
        confidence by 0.5 to 0.15, compounding rather than conflicting.
        """
        # Resolve up front so an invalid flag fails closed before any
        # mutation lands (no half-stamped graph on a typo).
        policy = _resolve_contradicted_policy()

        self._nli_setup()
        edges = _as_list(graph.get("edges"))
        pair_index, type_dir_index, aux = self._build_indexes(edges)

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source = edge.get("source")
            target = edge.get("target")
            edge_type = edge.get("type")
            if not isinstance(source, str) or not isinstance(target, str):
                edge["edge_status"] = "pending"
                edge["consensus_signals"] = []
                continue
            if not isinstance(edge_type, str):
                edge["edge_status"] = "pending"
                edge["consensus_signals"] = []
                continue
            this_rule = _edge_rule(edge)
            if not this_rule:
                edge["edge_status"] = "pending"
                edge["consensus_signals"] = []
                continue
            signals = self._compute_signals(
                edge=edge,
                this_rule=this_rule,
                source=source,
                target=target,
                edge_type=edge_type,
                pair_index=pair_index,
                type_dir_index=type_dir_index,
                aux=aux,
            )
            status = self._status_from_signals(signals)
            edge["consensus_signals"] = signals
            if status == "contradicted" and policy is not None:
                status = self._apply_contradicted_policy(edge, policy)
            edge["edge_status"] = status
        return graph

    def _apply_contradicted_policy(
        self, edge: Dict[str, Any], policy: str
    ) -> str:
        """Apply the contradicted-edge policy to a single contradicted edge.

        Returns the (possibly re-mapped) ``edge_status`` to stamp. ``decay``
        attenuates ``confidence`` in place and keeps the status
        ``contradicted``; ``retract`` re-maps the status to ``retracted``
        WITHOUT deleting the edge (provenance / replay are preserved — a
        retracted edge stays physically present, clearly statused, so an
        ``edge_status``-aware consumer can exclude it).

        Idempotency: ``decay`` is NOT self-idempotent on its own (decaying a
        confidence twice would halve it twice). It is made idempotent via the
        ``consensus_confidence_decayed`` sentinel stamped on the edge — a
        second :meth:`apply_to_graph` call sees the sentinel and skips the
        multiply, so calling apply_to_graph twice is byte-identical (the
        aggregator's idempotency contract).
        """
        if policy == "decay":
            if not edge.get("consensus_confidence_decayed"):
                conf = edge.get("confidence")
                if isinstance(conf, (int, float)) and not isinstance(conf, bool):
                    # Record the pre-decay confidence so cross-edge
                    # consensus_signals[] keep reporting the ORIGINAL value
                    # across re-runs (idempotency: otherwise a second pass
                    # would index the already-decayed confidence and the
                    # signal entry's `confidence` field would drift).
                    edge["consensus_confidence_predecay"] = round(float(conf), 6)
                    edge["confidence"] = round(float(conf) * _DECAY_FACTOR, 6)
                    edge["consensus_confidence_decayed"] = True
            return "contradicted"
        # policy == "retract": status-only, edge retained.
        return "retracted"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _load_graph(self) -> Dict[str, Any]:
        """Load the semantic graph or return an empty graph dict."""
        if self.semantic_graph_path is None:
            return {}
        try:
            text = self.semantic_graph_path.read_text(encoding="utf-8")
        except (OSError, FileNotFoundError):
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "EdgeConsensusAggregator: failed to parse %s as JSON; "
                "returning empty summary.",
                self.semantic_graph_path,
            )
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _compute_signals(
        self,
        *,
        edge: Dict[str, Any],
        this_rule: str,
        source: str,
        target: str,
        edge_type: str,
        pair_index: Mapping[_PairKey, List[Dict[str, Any]]],
        type_dir_index: Mapping[_PairKey, List[Dict[str, Any]]],
        aux: _AuxIndexes,
    ) -> List[Dict[str, Any]]:
        """Compute the per-edge consensus_signals[] list."""
        matrix_entry = _RULE_PAIR_MATRIX.get(this_rule)
        if matrix_entry is None:
            confirming_rules: FrozenSet[str] = frozenset()
            cross_contradicting: FrozenSet[str] = frozenset()
        else:
            confirming_rules, cross_contradicting = matrix_entry

        signals: List[Dict[str, Any]] = []
        seen_other_rules: Set[Tuple[str, str]] = set()

        same_pair_key = _canonical_pair_key(source, target, edge_type)
        # Same-pair AGREE / DISAGREE checks against other rules that
        # share the equivalence key.
        for other_edge in pair_index.get(same_pair_key, ()):
            if other_edge is edge:
                continue
            other_rule = _edge_rule(other_edge)
            if not other_rule:
                continue
            if other_rule in confirming_rules:
                key = (other_rule, "agree")
                if key in seen_other_rules:
                    continue
                seen_other_rules.add(key)
                conf = _signal_confidence(other_edge)
                entry: Dict[str, Any] = {
                    "other_rule": other_rule,
                    "signal": "agree",
                }
                if isinstance(conf, (int, float)):
                    entry["confidence"] = round(float(conf), 4)
                signals.append(entry)
            elif other_rule in cross_contradicting:
                key = (other_rule, "disagree")
                if key in seen_other_rules:
                    continue
                seen_other_rules.add(key)
                entry = {
                    "other_rule": other_rule,
                    "signal": "disagree",
                    "detail": "cross_rule_contradiction",
                }
                conf = _signal_confidence(other_edge)
                if isinstance(conf, (int, float)):
                    entry["confidence"] = round(float(conf), 4)
                signals.append(entry)

        # Same-rule reverse-direction cycle check (e.g. circular
        # prerequisite). Only for directed types whose rule appears
        # in _CYCLE_DETECTING_RULES.
        if this_rule in _CYCLE_DETECTING_RULES and edge_type not in _UNDIRECTED_EDGE_TYPES:
            reverse_key = _reverse_pair_key(source, target, edge_type)
            if reverse_key is not None:
                for other_edge in type_dir_index.get(reverse_key, ()):
                    if other_edge is edge:
                        continue
                    other_rule = _edge_rule(other_edge)
                    if other_rule != this_rule:
                        continue
                    key = (this_rule, "disagree")
                    if key in seen_other_rules:
                        continue
                    seen_other_rules.add(key)
                    detail = (
                        "circular_prerequisite"
                        if this_rule == "prerequisite_from_lo_order"
                        else "type_hierarchy_cycle"
                    )
                    entry = {
                        "other_rule": this_rule,
                        "signal": "disagree",
                        "detail": detail,
                    }
                    conf = _signal_confidence(other_edge)
                    if isinstance(conf, (int, float)):
                        entry["confidence"] = round(float(conf), 4)
                    signals.append(entry)
                    break  # one reverse contradiction is enough to flip

        # Cross-stratum triangulation signals (T1-T4).
        signals.extend(
            self._compute_triangulation_signals(
                edge=edge,
                this_rule=this_rule,
                source=source,
                target=target,
                aux=aux,
            )
        )

        # NLI extension hook — flag-gated, deferred per plan.
        if _flag_is_true(_NLI_FLAG_ENV):
            try:
                nli_signal = self._nli_extension_signal(
                    edge=edge,
                    this_rule=this_rule,
                    source=source,
                    target=target,
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                logger.debug(
                    "EdgeConsensusAggregator NLI extension raised "
                    "(stub path): %s",
                    exc,
                )
                nli_signal = None
            if nli_signal is not None:
                signals.append(nli_signal)

        return signals

    def _compute_triangulation_signals(
        self,
        *,
        edge: Dict[str, Any],
        this_rule: str,
        source: str,
        target: str,
        aux: _AuxIndexes,
    ) -> List[Dict[str, Any]]:
        """Cross-stratum triangulation over persisted provenance (T1-T4).

        Fixed detector order; ``sorted()`` iteration over any set so output
        is deterministic regardless of dict/set insertion order.
        """
        out: List[Dict[str, Any]] = []

        # T1 — prerequisite (LO order) × first-mention text order.
        # Edge B->A means A (target) is the prerequisite of B (source). The
        # prerequisite should be mentioned FIRST in the text. fo_t < fo_s
        # corroborates the LO-order call; fo_t > fo_s conflicts.
        if this_rule == "prerequisite_from_lo_order":
            fo_t = aux.first_mention_ord.get(target)
            fo_s = aux.first_mention_ord.get(source)
            if fo_t is not None and fo_s is not None and fo_t != fo_s:
                if fo_t < fo_s:
                    out.append({
                        "other_rule": "defined_by_from_first_mention",
                        "signal": "agree",
                        "detail": "text_order_corroborates_lo_order",
                    })
                else:
                    out.append({
                        "other_rule": "defined_by_from_first_mention",
                        "signal": "disagree",
                        "detail": "lo_order_vs_text_order_conflict",
                    })

        # T2 — assesses (question->LO) × derived-from-objective (chunk->LO).
        if this_rule == "assesses_from_question_lo":
            ev = _edge_evidence(edge)
            src_chunk = ev.get("source_chunk_id")
            if isinstance(src_chunk, str) and src_chunk:
                lo_lower = target.lower()
                if src_chunk in aux.chunks_by_lo.get(lo_lower, set()):
                    out.append({
                        "other_rule": "derived_from_lo_ref",
                        "signal": "agree",
                        "detail": "chunk_lo_triangle",
                    })

        # T2 symmetric stamp on derived_from_lo_ref (chunk->LO) edges.
        if this_rule == "derived_from_lo_ref":
            lo_lower = target.lower()
            if (source, lo_lower) in aux.assessed_chunk_lo:
                out.append({
                    "other_rule": "assesses_from_question_lo",
                    "signal": "agree",
                    "detail": "chunk_lo_triangle",
                })

        # T3 — LO-chunk-concept triangle.
        # targets_concept_from_lo edge (LO->C): agree when the LO's chunks
        # intersect the concept's chunks.
        if this_rule == "targets_concept_from_lo":
            lo_chunks = aux.chunks_by_lo.get(source.lower(), set())
            concept_chunks = aux.concept_chunks.get(target, set())
            if lo_chunks and concept_chunks and (lo_chunks & concept_chunks):
                out.append({
                    "other_rule": "derived_from_lo_ref",
                    "signal": "agree",
                    "detail": "lo_chunk_concept_triangle",
                })

        # T3 symmetric on derived_from_lo_ref (chunk->LO) edges: agree when
        # some concept C targeted by this LO has the edge-source chunk in its
        # concept_chunks.
        if this_rule == "derived_from_lo_ref":
            for concept in sorted(aux.targets_by_lo.get(target.lower(), set())):
                if source in aux.concept_chunks.get(concept, set()):
                    out.append({
                        "other_rule": "targets_concept_from_lo",
                        "signal": "agree",
                        "detail": "lo_chunk_concept_triangle",
                    })
                    break

        # T4 — related-to co-location (support tier). related_from_cooccurrence
        # ONLY; intra_chunk_link is excluded (tautological — it IS a
        # co-location edge, so a co-location signal over itself is circular).
        if this_rule == "related_from_cooccurrence":
            sc = aux.concept_chunks.get(source, set())
            tc = aux.concept_chunks.get(target, set())
            if sc and tc and (sc & tc):
                out.append({
                    "other_rule": "defined_by_from_first_mention",
                    "signal": "support",
                    "detail": "co_located_first_mention",
                })

        return out

    def _status_from_signals(self, signals: List[Dict[str, Any]]) -> str:
        """Resolve the edge_status enum value from signals[].

        Precedence (strongest verdict wins):
            disagree(rule) -> contradicted
          > disagree(nli)  -> retracted
          > agree          -> confirmed
          > support        -> supported
          > (none)         -> pending
        """
        contradicted_by_nli = False
        contradicted_by_rule = False
        agree = False
        support = False
        for sig in signals:
            if not isinstance(sig, dict):
                continue
            signal = sig.get("signal")
            other = sig.get("other_rule")
            if signal == "disagree":
                if other == "nli_text_entailment":
                    contradicted_by_nli = True
                else:
                    contradicted_by_rule = True
            elif signal == "agree":
                agree = True
            elif signal == "support":
                support = True
        if contradicted_by_rule:
            return "contradicted"
        if contradicted_by_nli:
            return "retracted"
        if agree:
            return "confirmed"
        if support:
            return "supported"
        return "pending"

    def _nli_setup(self) -> None:
        """Resolve the per-run NLI budget + classifier (idempotent per call).

        Called at the top of :meth:`build` and :meth:`apply_to_graph` so both
        the report and the graph-mutation paths score the same edges under the
        same per-run cap. No-op (budget 0, classifier None) unless
        ``TRAINFORGE_EDGE_NLI`` is truthy AND a chunk-text lookup was supplied
        at construction. Resetting the budget every call keeps
        :meth:`apply_to_graph` idempotent — a second call re-scores the same
        edges deterministically. Best-effort: a classifier-load failure (deps
        missing / model load error) degrades to a no-op extension.
        """
        self._nli_budget_remaining = 0
        self._nli_classifier = None
        if not self._chunk_text_lookup:
            return
        if not _flag_is_true(_NLI_FLAG_ENV):
            return
        self._nli_budget_remaining = _resolve_nli_max_edges()
        try:
            from lib.classifiers.nli_classifier import NliClassifier
            self._nli_classifier = NliClassifier.get_or_load()
        except Exception as exc:  # noqa: BLE001 — graceful degrade
            logger.debug(
                "EdgeConsensusAggregator: NLI classifier load failed "
                "(extension no-ops): %s",
                exc,
            )
            self._nli_classifier = None

    def _nli_extension_signal(
        self,
        *,
        edge: Dict[str, Any],
        this_rule: str,
        source: str,
        target: str,
    ) -> Optional[Dict[str, Any]]:
        """NLI text-entailment extension (``TRAINFORGE_EDGE_NLI``).

        For a chunk-anchored edge (:data:`_NLI_EDGE_TEMPLATES`), render the
        edge predicate to a hypothesis and score it against the cited chunk
        text via the shared DeBERTa NLI singleton. Emits a single ``disagree``
        signal ONLY on a strong contradiction
        (``score.contradiction >= _NLI_CONTRADICTION_THRESHOLD``); entailment /
        neutral verdicts add no signal in v1 (conservative — the arm can only
        retract, never confirm). Returns ``None`` (no signal) when the flag is
        off, no chunk-text lookup was supplied, the classifier is unavailable,
        the per-run budget is exhausted, the edge type is not chunk-anchored,
        or the cited chunk text / concept label can't be resolved.
        """
        lookup = self._chunk_text_lookup
        classifier = self._nli_classifier
        if not lookup or classifier is None:
            return None
        if self._nli_budget_remaining <= 0:
            return None
        template = _NLI_EDGE_TEMPLATES.get(this_rule)
        if template is None:
            return None
        chunk_id = target if template["chunk_endpoint"] == "target" else source
        concept_id = source if template["concept_endpoint"] == "source" else target
        chunk_text = lookup.get(chunk_id)
        if not isinstance(chunk_text, str) or not chunk_text.strip():
            return None
        concept_label = _humanize_concept(concept_id)
        if not concept_label:
            return None
        hypothesis = template["hypothesis"].format(concept=concept_label)
        # Consume budget only when a real forward pass is about to run so the
        # cap maps to NLI inferences, not to skipped non-anchored edges.
        self._nli_budget_remaining -= 1
        try:
            score = classifier.score_pair(
                premise=chunk_text[:_NLI_PREMISE_CHAR_CAP],
                hypothesis=hypothesis,
            )
        except Exception as exc:  # noqa: BLE001 — never break the aggregation
            logger.debug(
                "EdgeConsensusAggregator: NLI score_pair raised for "
                "edge %s->%s (%s): %s",
                source, target, this_rule, exc,
            )
            return None
        contradiction = getattr(score, "contradiction", None)
        if not isinstance(contradiction, (int, float)) or isinstance(contradiction, bool):
            return None
        if float(contradiction) >= _NLI_CONTRADICTION_THRESHOLD:
            return {
                "other_rule": "nli_text_entailment",
                "signal": "disagree",
                "detail": "nli_contradiction",
                "confidence": round(float(contradiction), 4),
            }
        return None

    def _emit_decision(
        self,
        report: Dict[str, Any],
        *,
        top_contradictions: List[str],
    ) -> None:
        """Emit one ``edge_consensus_resolution`` decision per build()."""
        if self.decision_capture is None:
            return
        summary = report["summary"]
        passed = summary["contradicted_count"] == 0
        rationale = (
            f"Cross-stratum edge consensus for course={self.course_slug or 'n/a'} "
            f"run_id={self.run_id or 'n/a'} matrix={MATRIX_VERSION}: "
            f"total_edges={summary['total_edges']}, "
            f"confirmed={summary['confirmed_count']}, "
            f"supported={summary['supported_count']}, "
            f"contradicted={summary['contradicted_count']}, "
            f"pending={summary['pending_count']}, "
            f"retracted={summary['retracted_count']}, "
            f"consensus_rate={summary['consensus_rate']:.4f}, "
            f"supported_rate={summary['supported_rate']:.4f}, "
            f"contradiction_rate={summary['contradiction_rate']:.4f}, "
            f"top_contradicting_pairs="
            f"{','.join(top_contradictions) if top_contradictions else 'none'}, "
            f"nli_extension={'on' if report['nli_extension_enabled'] else 'off'}."
        )
        try:
            self.decision_capture.log_decision(
                decision_type="edge_consensus_resolution",
                decision="passed" if passed else "failed:contradictions_present",
                rationale=rationale,
            )
        except Exception as exc:  # noqa: BLE001 — capture must not break
            logger.debug(
                "DecisionCapture.log_decision raised on "
                "edge_consensus_resolution: %s",
                exc,
            )


# ----------------------------------------------------------------------
# Module helpers
# ----------------------------------------------------------------------


def _as_list(value: Any) -> List[Any]:
    """Return a list view; non-list values return [] for graceful degrade."""
    return list(value) if isinstance(value, list) else []


def _edge_rule(edge: Mapping[str, Any]) -> Optional[str]:
    """Pull the rule name off an edge's provenance, tolerating shape drift."""
    prov = edge.get("provenance") if isinstance(edge, Mapping) else None
    if not isinstance(prov, Mapping):
        return None
    rule = prov.get("rule")
    if isinstance(rule, str) and rule:
        return rule
    return None


def _signal_confidence(edge: Mapping[str, Any]) -> Any:
    """Return the confidence to report in a cross-edge consensus signal.

    Prefers the ``consensus_confidence_predecay`` snapshot when present (so a
    decay-policy re-run reports the ORIGINAL confidence on cross-edge signals
    rather than the already-decayed value — preserves apply_to_graph
    idempotency). Falls back to the live ``confidence``.
    """
    pre = edge.get("consensus_confidence_predecay") if isinstance(edge, Mapping) else None
    if isinstance(pre, (int, float)) and not isinstance(pre, bool):
        return pre
    return edge.get("confidence") if isinstance(edge, Mapping) else None


def _edge_evidence(edge: Mapping[str, Any]) -> Mapping[str, Any]:
    """Pull the evidence sub-dict off an edge's provenance; {} on drift."""
    prov = edge.get("provenance") if isinstance(edge, Mapping) else None
    if not isinstance(prov, Mapping):
        return {}
    ev = prov.get("evidence")
    return ev if isinstance(ev, Mapping) else {}


def _chunk_ordinal(chunk_id: str) -> Optional[int]:
    """Parse an integer chunk ordinal out of a chunk ID; None if unparseable."""
    if not isinstance(chunk_id, str):
        return None
    m = _CHUNK_ORD_RE.search(chunk_id)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def _flag_is_true(env_name: str) -> bool:
    """Truthy check matching the project's flag-resolution conventions."""
    raw = os.getenv(env_name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_nli_max_edges() -> int:
    """Per-run NLI edge-scoring cap. Garbage / non-positive → default."""
    raw = os.getenv(_NLI_MAX_EDGES_ENV, "")
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return _NLI_DEFAULT_MAX_EDGES
    return val if val > 0 else _NLI_DEFAULT_MAX_EDGES


def _humanize_concept(node_id: str) -> str:
    """Render a concept node id into a readable NLI hypothesis fragment.

    Strips a course-scoped ``course:slug`` prefix (keeps the last segment)
    and swaps ``-`` / ``_`` for spaces. Returns "" on non-str / empty input
    so the caller no-ops rather than build a degenerate hypothesis.
    """
    if not isinstance(node_id, str):
        return ""
    label = node_id.split(":")[-1].replace("-", " ").replace("_", " ").strip()
    return label


def load_chunk_text_lookup(course_dir: Optional[Any]) -> Optional[Dict[str, str]]:
    """Build a ``{chunk_id: text}`` lookup from a LibV2 course tree.

    Searches the canonical chunkset locations (:data:`_CHUNK_JSONL_CANDIDATES`)
    under ``course_dir``. Returns ``None`` when no chunkset is found (so the
    NLI extension no-ops exactly as if the lookup were never supplied).
    Best-effort: unreadable files / malformed JSONL lines are skipped.
    """
    if not course_dir:
        return None
    base = Path(course_dir)
    lookup: Dict[str, str] = {}
    for rel in _CHUNK_JSONL_CANDIDATES:
        path = base / rel
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            cid = obj.get("id") or obj.get("chunk_id")
            text = obj.get("text")
            if isinstance(cid, str) and cid and isinstance(text, str) and text:
                # First chunkset wins per id (dart before imscc before corpus).
                lookup.setdefault(cid, text)
    return lookup or None


def chunk_text_lookup_from_chunks(
    chunks: Optional[Any],
) -> Optional[Dict[str, str]]:
    """Build a ``{chunk_id: text}`` lookup from an in-memory chunk-dict list.

    Companion to :func:`load_chunk_text_lookup` for call sites that already
    hold the chunks in memory (e.g. the concept_extraction authoring path).
    Returns ``None`` when the input yields no usable ``(id, text)`` pairs so
    the NLI extension no-ops.
    """
    if not isinstance(chunks, list):
        return None
    lookup: Dict[str, str] = {}
    for obj in chunks:
        if not isinstance(obj, dict):
            continue
        cid = obj.get("id") or obj.get("chunk_id")
        text = obj.get("text")
        if isinstance(cid, str) and cid and isinstance(text, str) and text:
            lookup.setdefault(cid, text)
    return lookup or None


class ContradictedEdgePolicyError(ValueError):
    """Raised when ``TRAINFORGE_CONTRADICTED_EDGE_POLICY`` holds an
    unrecognised value (fail-closed — we do NOT silently fall back to
    stamp-only on a typo, which would mask an operator's intent)."""


def _resolve_contradicted_policy() -> Optional[str]:
    """Resolve the contradicted-edge policy from the env flag.

    Returns ``None`` (stamp-only, byte-identical legacy behaviour) when the
    flag is unset or empty. Returns the normalised ``decay`` / ``retract``
    string when a valid value is set. Raises
    :class:`ContradictedEdgePolicyError` on any other value so an operator
    typo fails closed rather than degrading silently to stamp-only.
    """
    raw = os.getenv(_CONTRADICTED_POLICY_ENV, "")
    value = raw.strip().lower()
    if not value:
        return None
    if value not in _VALID_CONTRADICTED_POLICIES:
        raise ContradictedEdgePolicyError(
            f"{_CONTRADICTED_POLICY_ENV}={raw!r} is invalid; accepted "
            f"values are {sorted(_VALID_CONTRADICTED_POLICIES)} "
            f"(unset = stamp-only, the default)."
        )
    return value


def _round(value: float) -> float:
    """Match the rounding convention used by other aggregators (4 dp)."""
    return round(float(value), 4)


__all__ = [
    "ContradictedEdgePolicyError",
    "EdgeConsensusAggregator",
    "MATRIX_VERSION",
    "SCHEMA_VERSION",
]

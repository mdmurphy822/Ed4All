"""Cross-rule edge consensus aggregator (GPT feedback 12-may, item 2).

The 9 inference rules under `Trainforge/rag/inference_rules/` emit
edges into `concept_graph_semantic.json` independently. Today each
edge carries a single per-rule `confidence` float plus the rule
provenance — but the graph has no record of whether ANOTHER rule
also fired over the same `(source, target)` pair (a confirmation
signal) or fired in the opposite direction (a contradiction
signal). GPT feedback item 2:

    "Not just confidence, but whether another rule contradicted or
     failed to confirm it."

This aggregator walks the emitted graph and stamps two new schema-
additive fields on each edge:

- ``edge_status`` ∈ ``{"pending", "confirmed", "contradicted",
  "retracted"}``. ``retracted`` is reserved for the NLI extension
  (off by default behind ``TRAINFORGE_EDGE_NLI``).
- ``consensus_signals: List[{other_rule, signal, confidence?,
  detail?}]`` — one entry per OTHER rule that fired over the same
  node pair.

The aggregator additionally writes ``edge_consensus_report.json`` to
the same directory as the source graph, summarising per-rule
confirmed / contradicted / pending counts and listing every
contradicted edge so operators can investigate cycles.

Cross-rule matrix (see ``_RULE_PAIR_MATRIX`` below + the wave plan
at ``plans/gptfeedback-may12-item2-edge-consensus-2026-05.md``):

- ``is_a_from_key_terms`` — confirmed by same-pair
  ``defined_by_from_first_mention`` or ``targets_concept_from_lo``;
  contradicted by reverse-direction same-rule (cycle).
- ``prerequisite_from_lo_order`` — contradicted by reverse-direction
  same-rule (circular prerequisite).
- ``related_from_cooccurrence`` — confirmed by same-pair
  ``is_a_from_key_terms`` (taxonomic anchor strengthens the
  related-to signal).
- ``assesses_from_question_lo`` — confirmed by same-pair
  ``derived_from_lo_ref`` (chunk both derives + assesses the LO).
- ``exemplifies_from_example_chunks`` — confirmed by same-pair
  ``defined_by_from_first_mention``.
- ``derived_from_lo_ref`` — confirmed by same-pair
  ``assesses_from_question_lo``.
- ``defined_by_from_first_mention`` — confirmed by same-pair
  ``is_a_from_key_terms``.
- ``targets_concept_from_lo`` — confirmed by same-target
  ``is_a_from_key_terms`` or ``defined_by_from_first_mention``.

Edge types ``related-to`` and ``is-a`` are treated as undirected
for the equivalence-key (their pair-key collapses
``{source, target}`` into a frozenset); every other edge type is
directed.

This module is intentionally pure aggregation — no LLM calls, no
network I/O, no schema validation pass. The schema is admitted
unchanged on legacy graphs (missing ``edge_status`` validates), and
the aggregator's NLI extension is stubbed behind
``TRAINFORGE_EDGE_NLI`` for a follow-on wave.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple, Union


logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"

#: Matrix version is interpolated into decision-capture rationales so
#: a future tweak to the rule-pair map bumps the audit trail
#: deterministically.
MATRIX_VERSION = "2026-05-12.v1"

#: Edge types whose pair-equivalence-key is undirected — i.e. an edge
#: ``A → B`` and an edge ``B → A`` of the same type are treated as
#: signals over the SAME node pair. The rest are directed (``A → B``
#: differs from ``B → A``).
_UNDIRECTED_EDGE_TYPES: FrozenSet[str] = frozenset({"related-to", "is-a"})

#: Rule-pair confirmation matrix. ``_RULE_PAIR_MATRIX[rule]`` is a
#: 2-tuple ``(confirming_rules, contradicting_rules)``. ``contradicting
#: _rules`` enumerates OTHER rules that contradict; reverse-direction
#: same-rule cycles are detected separately because they're a
#: same-rule signal, not a cross-rule signal.
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
    "defined_by_from_first_mention": (
        frozenset({"is_a_from_key_terms"}),
        frozenset(),
    ),
    "targets_concept_from_lo": (
        frozenset({"is_a_from_key_terms", "defined_by_from_first_mention"}),
        frozenset(),
    ),
}

#: Rules that emit directed edges where reverse-direction by the SAME
#: rule between the same nodes is a contradiction (cycle).
_CYCLE_DETECTING_RULES: FrozenSet[str] = frozenset({
    "prerequisite_from_lo_order",
    "is_a_from_key_terms",
})

#: NLI extension flag — schema admits the ``retracted`` value; full
#: NLI implementation deferred to a follow-on wave per the plan.
_NLI_FLAG_ENV: str = "TRAINFORGE_EDGE_NLI"

# Type alias for the equivalence key used to bucket edges that
# refer to the same node pair under the same edge type.
_PairKey = Tuple[str, Union[Tuple[str, str], FrozenSet[str]]]


def _canonical_pair_key(source: str, target: str, edge_type: str) -> _PairKey:
    """Compute the pair-equivalence-key for cross-rule confirmation.

    Cross-rule confirmation is "did a DIFFERENT rule fire over the
    same node pair?". We bucket by ``frozenset({source, target})``
    ignoring both edge_type AND direction — a same-pair ``is-a`` edge
    and a same-pair ``defined-by`` edge from different rules SHOULD
    share a bucket so the matrix can ask "did
    defined_by_from_first_mention confirm is_a_from_key_terms's call
    here?". Direction is intentionally collapsed because the
    semantic-direction story differs per edge_type (e.g. an LO that
    'targets' a concept reads forward A→B; a chunk that 'defines' a
    concept reads B←A but anchors the same pair).

    Cycle detection (same-rule reverse direction) uses the dedicated
    :func:`_reverse_pair_key` helper which keys on edge_type because a
    reverse same-rule cycle is type-specific AND direction-sensitive.
    """
    # ``edge_type`` accepted for symmetry with the reverse key helper
    # below; the cross-rule bucket itself is type+direction-agnostic.
    _ = edge_type
    return ("__pair__", frozenset({source, target}))


def _reverse_pair_key(source: str, target: str, edge_type: str) -> Optional[_PairKey]:
    """The reverse-direction pair-key for cycle detection on directed
    types. Keyed on the specific edge_type because cycles only count
    when both directions share a type (A is-a B AND B is-a A; A prerequisite-of B AND B prerequisite-of A).
    Undirected types return None — direction doesn't apply.
    """
    if edge_type in _UNDIRECTED_EDGE_TYPES:
        return None
    return (edge_type, (target, source))


class EdgeConsensusAggregator:
    """Walks a semantic graph and stamps cross-rule consensus signals.

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
    """

    def __init__(
        self,
        semantic_graph_path: Optional[Path],
        *,
        course_slug: str = "",
        run_id: str = "",
        decision_capture: Optional[Any] = None,
    ) -> None:
        self.semantic_graph_path = (
            Path(semantic_graph_path) if semantic_graph_path is not None else None
        )
        self.course_slug = str(course_slug or "")
        self.run_id = str(run_id or "")
        self.decision_capture = decision_capture

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @staticmethod
    def _build_indexes(
        edges: List[Any],
    ) -> Tuple[Dict[_PairKey, List[Dict[str, Any]]], Dict[_PairKey, List[Dict[str, Any]]]]:
        """Build (pair_index, type_dir_index).

        ``pair_index`` keys by ``(source, target)`` ignoring edge_type so
        the cross-rule matrix can ask "did ANY other rule fire over
        the same node pair?". ``type_dir_index`` keys by
        ``(edge_type, (source, target))`` so cycle detection can ask
        "is there a SAME-type B→A edge for this A→B?".
        """
        pair_index: Dict[_PairKey, List[Dict[str, Any]]] = defaultdict(list)
        type_dir_index: Dict[_PairKey, List[Dict[str, Any]]] = defaultdict(list)
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
        return pair_index, type_dir_index

    def build(self) -> Dict[str, Any]:
        """Build the consensus report dict.

        Reads the graph from disk (best-effort); computes per-edge
        signals; rolls into per-rule and corpus-wide summaries; emits
        the decision-capture event. Never mutates the graph on disk —
        :meth:`apply_to_graph` is the explicit mutation path.
        """
        graph = self._load_graph()
        edges = _as_list(graph.get("edges"))

        pair_index, type_dir_index = self._build_indexes(edges)

        # Walk every edge and compute its consensus verdict.
        per_rule_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"edge_count": 0, "confirmed": 0, "contradicted": 0, "pending": 0, "retracted": 0}
        )
        confirmed_count = 0
        contradicted_count = 0
        pending_count = 0
        retracted_count = 0
        contradictions: List[Dict[str, Any]] = []
        contradicting_pair_counter: Counter = Counter()

        # NLI extension is stubbed in this wave per the plan; the flag
        # is read once so the decision-capture rationale records which
        # mode the aggregator ran in.
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
                # Edge without a rule provenance — treat as pending
                # but don't index against the matrix.
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
            )

            status = self._status_from_signals(signals)
            per_edge_results.append((edge, status, signals))

            per_rule_counts[this_rule]["edge_count"] += 1
            if status == "confirmed":
                confirmed_count += 1
                per_rule_counts[this_rule]["confirmed"] += 1
            elif status == "contradicted":
                contradicted_count += 1
                per_rule_counts[this_rule]["contradicted"] += 1
                # Record one contradiction entry per disagreeing
                # signal so the operator-facing report names each
                # contradicting (other_rule, detail) pair.
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

        total_edges = (
            confirmed_count + contradicted_count + pending_count + retracted_count
        )
        consensus_rate = (
            confirmed_count / total_edges if total_edges else 0.0
        )
        contradiction_rate = (
            contradicted_count / total_edges if total_edges else 0.0
        )

        # Top three contradicting rule-pairs by frequency for the
        # decision-capture rationale.
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
                "contradicted_count": contradicted_count,
                "pending_count": pending_count,
                "retracted_count": retracted_count,
                "consensus_rate": _round(consensus_rate),
                "contradiction_rate": _round(contradiction_rate),
            },
            "per_rule": per_rule_rollup,
            "contradictions": contradictions,
        }

        self._emit_decision(report, top_contradictions=top_contradictions)

        # Stash the per-edge results on the instance so a caller that
        # wants to splice the consensus fields back onto the graph
        # (via apply_to_graph) doesn't pay the build cost twice.
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
        list passed in (NOT the on-disk graph), so the caller can pass a
        freshly-loaded dict and trust that the mutations land on its
        own edge dicts. Idempotent: calling apply_to_graph twice
        produces byte-identical output (the rule-pair matrix is
        deterministic over the same pair-index input).
        """
        edges = _as_list(graph.get("edges"))
        pair_index, type_dir_index = self._build_indexes(edges)

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
            )
            edge["edge_status"] = self._status_from_signals(signals)
            edge["consensus_signals"] = signals
        return graph

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
                conf = other_edge.get("confidence")
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
                conf = other_edge.get("confidence")
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
                    conf = other_edge.get("confidence")
                    if isinstance(conf, (int, float)):
                        entry["confidence"] = round(float(conf), 4)
                    signals.append(entry)
                    break  # one reverse contradiction is enough to flip

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

    def _status_from_signals(self, signals: List[Dict[str, Any]]) -> str:
        """Resolve the edge_status enum value from signals[].

        Any disagree → contradicted (or retracted when the disagree
        comes from the NLI extension). Otherwise any agree →
        confirmed. Otherwise pending.
        """
        contradicted_by_nli = False
        contradicted_by_rule = False
        agree = False
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
        if contradicted_by_rule:
            return "contradicted"
        if contradicted_by_nli:
            return "retracted"
        if agree:
            return "confirmed"
        return "pending"

    def _nli_extension_signal(
        self,
        *,
        edge: Dict[str, Any],
        this_rule: str,
        source: str,
        target: str,
    ) -> Optional[Dict[str, Any]]:
        """Stub for the NLI extension (TRAINFORGE_EDGE_NLI=true).

        Deferred per the wave plan. The schema admits the
        ``retracted`` edge_status value and the
        ``other_rule: "nli_text_entailment"`` signal shape so a
        follow-on wave can land the real implementation without
        re-bumping the schema. Returns ``None`` until that wave.
        """
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
            f"Cross-rule edge consensus for course={self.course_slug or 'n/a'} "
            f"run_id={self.run_id or 'n/a'} matrix={MATRIX_VERSION}: "
            f"total_edges={summary['total_edges']}, "
            f"confirmed={summary['confirmed_count']}, "
            f"contradicted={summary['contradicted_count']}, "
            f"pending={summary['pending_count']}, "
            f"retracted={summary['retracted_count']}, "
            f"consensus_rate={summary['consensus_rate']:.4f}, "
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


def _flag_is_true(env_name: str) -> bool:
    """Truthy check matching the project's flag-resolution conventions."""
    raw = os.getenv(env_name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _round(value: float) -> float:
    """Match the rounding convention used by other aggregators (4 dp)."""
    return round(float(value), 4)


__all__ = [
    "EdgeConsensusAggregator",
    "MATRIX_VERSION",
    "SCHEMA_VERSION",
]

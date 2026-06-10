"""KG-quality metric surface — four-dimension report over the asserted +
derived concept graphs and a SHACL ValidationReport.

This module exists to give the team concrete numbers (completeness,
consistency, accuracy, coverage) to drive every subsequent KG
improvement. It is pure aggregation:

* Reads ``concept_graph.json`` (asserted nodes + edges).
* Reads ``concept_graph_semantic.json`` (typed-edge inference output).
* Counts derived edges by inspecting per-edge provenance ``rule`` (the
  same surface a TriG named-graph diff would expose, since the IRI
  scheme ``https://ed4all.io/run/<run_id>/rule/<rule_name>`` —
  registered in ``Trainforge/rag/named_graph_writer.py`` — has one
  named graph per rule and the JSON form preserves the rule key).
* Walks a pyshacl ``ValidationReport`` (or any object with a
  ``results`` iterable carrying the canonical SHACL fields) to
  aggregate violations / warnings per source-shape.

No new SHACL evaluation pass is performed. Callers pass in the report
already produced by ``lib/validators/shacl_runner.py``. No LLM calls,
no DecisionCapture wiring — this is metric aggregation, not
classification.

Output: ``kg_quality_report.json`` — see :class:`KGQualityReporter`
docstring for the canonical shape.

Improvement #4 from the post-Wave 85 corpus-grounded gap analysis.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# Default required predicates for the completeness dimension. These
# correspond to the canonical shape of a node in concept_graph.json:
# every concept node is expected to carry an ``id`` and a ``label``.
# Callers can override via the ``required_predicates`` constructor
# argument when their corpus convention differs.
DEFAULT_REQUIRED_PREDICATES: List[str] = ["id", "label"]


# Named-graph IRI scheme prefix (see Trainforge/rag/named_graph_writer.py).
# Any quad whose graph IRI starts with this prefix represents a
# derived edge from the typed-edge inference rules.
RULE_GRAPH_IRI_PREFIX: str = "https://ed4all.io/run/"


class KGQualityReporter:
    """Computes the four KG-quality dimensions and writes the report.

    Dimensions:

    * **completeness** — ratio of focus nodes satisfying the required
      predicate set across asserted concepts. Numerator: nodes with
      every required predicate present and non-empty. Denominator:
      total node count. Score = numerator / denominator.

    * **consistency** — ``1 - (violation_count / total_focus_nodes)``
      where ``violation_count`` is the number of SHACL results with
      severity ``critical`` (sh:Violation) and ``total_focus_nodes``
      is the number of asserted nodes. Floored at 0.0.

    * **accuracy** — ``1 - (warning_count / total_focus_nodes)``,
      proxying type/range mismatches surfaced by SHACL warning-severity
      results. Floored at 0.0.

    * **coverage** — ``asserted / (asserted + derived)`` where
      ``asserted`` is the asserted-edge count from concept_graph.json
      and ``derived`` is the count of edges produced by inference rules
      (extracted from per-edge ``provenance.rule`` in
      concept_graph_semantic.json — the JSON-form analogue of the
      named-graph quads with IRI ``https://ed4all.io/run/*/rule/*``).

    The report shape:

    .. code-block:: json

        {
          "run_id": "...",
          "generated_at": "ISO-8601",
          "course_slug": "<course-slug>",
          "dimensions": {
            "completeness": {"score": 0.92, "metric": "...",
                             "denominator": 660, "numerator": 607},
            "consistency": {"score": 0.98, "metric": "...",
                            "violation_count": 14, "warning_count": 23},
            "accuracy": {"score": 0.95, "metric": "..."},
            "coverage": {"score": 0.83, "metric": "..."}
          },
          "per_shape": [
            {"shape_iri": "...", "violations": 3, "warnings": 0,
             "focus_nodes": 50}
          ],
          "rule_outputs": [
            {"rule_iri": "https://ed4all.io/run/<id>/rule/<name>",
             "edge_count": 424, "rule_version": "v1"}
          ]
        }
    """

    def __init__(
        self,
        course_slug: str,
        run_id: str,
        output_dir: Path,
        *,
        required_predicates: Optional[List[str]] = None,
    ) -> None:
        self.course_slug = course_slug
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.required_predicates = (
            list(required_predicates)
            if required_predicates is not None
            else list(DEFAULT_REQUIRED_PREDICATES)
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def compute(
        self,
        concept_graph: Path,
        semantic_graph: Path,
        validation_report: Any,
        *,
        pedagogy_graph: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Build the KG-quality report dict.

        Args:
            concept_graph: Path to ``concept_graph.json`` (asserted form).
            semantic_graph: Path to ``concept_graph_semantic.json``
                (typed-edge inference output).
            validation_report: Object with a ``results`` iterable. Each
                result must expose ``severity``, ``source_shape``, and
                ``focus_node`` attributes (or matching dict keys). This
                matches both pyshacl's report object and the
                ``ShaclViolation`` dataclass list produced by
                ``lib/validators/shacl_runner.py``.
            pedagogy_graph: Optional path to ``pedagogy_graph.json`` for
                future cross-graph completeness checks. Currently
                accepted but not used in metric math; carried in the
                report's metadata for downstream consumers.

        Returns:
            The full report dict (see class docstring for shape).
        """
        concept = _load_json(concept_graph) or {}
        semantic = _load_json(semantic_graph) or {}
        nodes = _as_list(concept.get("nodes"))
        asserted_edges = _as_list(concept.get("edges"))

        results = _normalize_results(validation_report)

        # ---- completeness
        denominator = len(nodes)
        numerator = sum(
            1 for n in nodes if _node_has_required_predicates(n, self.required_predicates)
        )
        completeness_score = (
            numerator / denominator if denominator else 1.0
        )

        # ---- consistency / accuracy
        violation_count = sum(1 for r in results if _severity(r) == "critical")
        warning_count = sum(1 for r in results if _severity(r) == "warning")
        total_focus = denominator if denominator else max(1, len(results))
        consistency_score = max(0.0, 1.0 - (violation_count / total_focus))
        accuracy_score = max(0.0, 1.0 - (warning_count / total_focus))

        # ---- coverage
        derived_edges, rule_outputs = _summarize_rule_outputs(semantic, self.run_id)
        derived_count = len(derived_edges)
        asserted_count = len(asserted_edges)
        denom_coverage = asserted_count + derived_count
        coverage_score = (
            asserted_count / denom_coverage if denom_coverage else 1.0
        )

        # ---- per-shape rollup
        per_shape = _rollup_per_shape(results)

        report: Dict[str, Any] = {
            "run_id": self.run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "course_slug": self.course_slug,
            "dimensions": {
                "completeness": {
                    "score": _round(completeness_score),
                    "metric": (
                        "ratio of focus nodes satisfying sh:minCount "
                        "across required predicates"
                    ),
                    "denominator": denominator,
                    "numerator": numerator,
                    "required_predicates": list(self.required_predicates),
                },
                "consistency": {
                    "score": _round(consistency_score),
                    "metric": "1 - (Violation count / total focus nodes)",
                    "violation_count": violation_count,
                    "warning_count": warning_count,
                    "total_focus_nodes": denominator,
                },
                "accuracy": {
                    "score": _round(accuracy_score),
                    "metric": (
                        "1 - (Warning count / total focus nodes), "
                        "proxies type / range mismatches"
                    ),
                    "warning_count": warning_count,
                    "total_focus_nodes": denominator,
                },
                "coverage": {
                    "score": _round(coverage_score),
                    "metric": (
                        "asserted triples / (asserted + expected-derived) "
                        "— derived count from named-graph diff"
                    ),
                    "asserted_count": asserted_count,
                    "derived_count": derived_count,
                },
            },
            "per_shape": per_shape,
            "rule_outputs": rule_outputs,
        }

        if pedagogy_graph is not None:
            report["pedagogy_graph_path"] = str(pedagogy_graph)

        return report

    def compute_metrics_only(
        self,
        semantic_graph: Dict[str, Any],
        *,
        contradiction_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute the four KG-quality dimensions from a SINGLE in-memory
        semantic graph, with no SHACL ValidationReport input.

        Authoring-time entry point. The gate-oriented :meth:`compute`
        expects two on-disk graphs (asserted ``concept_graph.json`` +
        derived ``concept_graph_semantic.json``) plus a pyshacl
        ValidationReport — none of which exist when
        ``_run_concept_extraction`` first authors the semantic graph.
        This method derives the same four dimensions from the freshly
        built ``concept_graph_semantic.json`` dict alone so the graph's
        ``kg_quality`` field can be populated at authoring time.

        Dimension derivation under the degenerate (report-less) input:

        * **completeness** — identical to :meth:`compute`: ratio of
          nodes carrying every required predicate (default ``id`` +
          ``label``). The semantic graph's ``nodes`` list IS the
          asserted node set at this phase.
        * **consistency** — no SHACL report exists at authoring time, so
          there is no violation count; the SHACL-derived base is ``1.0``.
          It is then attenuated by ``(1 - contradiction_rate)`` so this
          composes with — does NOT duplicate — the EdgeConsensusAggregator
          attenuation that ``KGQualityValidator.validate`` applies at the
          gate. ``contradiction_rate`` is taken from the caller when
          supplied; otherwise it is derived from the graph's per-edge
          ``edge_status`` field (``contradicted`` / total edges), which
          the consensus-stamping pass writes immediately upstream of this
          call. Floored at 0.0.
        * **accuracy** — no SHACL warning count exists either, so the
          base is ``1.0`` (no type/range mismatch evidence at authoring
          time).
        * **coverage** — chunk-anchored ``DomainConcept`` node-grounding:
          the share of concept vocabulary grounded in the corpus text.
          Numerator: ``DomainConcept``-or-classless nodes incident to ≥1
          edge whose ``provenance.rule`` is absent OR in
          :data:`_CHUNK_ANCHORED_RULES` (the seven chunk-evidenced rules
          — co-occurrence, intra-chunk link, defined-by, exemplifies,
          is-a, misconception-of, assesses). Denominator: all
          ``DomainConcept``-or-classless nodes. Empty denominator → 1.0.
          A frequency-0 ``lo_key_concept`` node (an LO-asserted concept
          the chunks never grounded) counts in the denominator and is
          covered only if some chunk-anchored edge touches it. This
          REPLACES the old asserted/(asserted+derived) edge-share metric,
          which grew a quadratic LO-order-derived denominator and
          anti-correlated with quality (the rdf-shacl calibration corpus
          fell to 0.047 under it). The old edge ratio is preserved
          unthresholded in the coverage detail dict as
          ``asserted_edge_share`` (informational).

        Args:
            semantic_graph: The in-memory ``concept_graph_semantic.json``
                dict (post edge-consensus stamping).
            contradiction_rate: Optional pre-computed contradiction rate
                in ``[0, 1]``. When ``None``, derived from the graph's
                stamped ``edge_status`` fields.

        Returns:
            A report dict in the same four-dimension shape :meth:`compute`
            emits (``dimensions`` + ``per_shape`` + ``rule_outputs``),
            with an extra ``derivation: "semantic_graph_metrics_only"``
            marker and the resolved ``contradiction_rate``.
        """
        semantic = semantic_graph if isinstance(semantic_graph, dict) else {}
        nodes = _as_list(semantic.get("nodes"))
        all_edges = _as_list(semantic.get("edges"))

        # ---- completeness (identical math to compute())
        denominator = len(nodes)
        numerator = sum(
            1 for n in nodes
            if _node_has_required_predicates(n, self.required_predicates)
        )
        completeness_score = numerator / denominator if denominator else 1.0

        # ---- coverage: chunk-anchored DomainConcept node-grounding.
        # Numerator: DomainConcept-or-classless nodes incident to ≥1
        # chunk-anchored edge (rule-less OR a rule in
        # _CHUNK_ANCHORED_RULES). Denominator: all DomainConcept-or-
        # classless nodes. Reads as "share of concept vocabulary (incl.
        # LO-asserted concepts) grounded in the corpus text"; empty
        # denominator → 1.0 (vacuously covered). ``rule_outputs`` keeps
        # the full per-rule rollup for the report. The legacy asserted/
        # derived edge ratio is preserved unthresholded as
        # ``asserted_edge_share`` (informational only).
        _, rule_outputs = _summarize_rule_outputs(semantic, self.run_id)
        grounded_count, concept_node_count = _node_grounding_coverage(
            nodes, all_edges,
        )
        coverage_score = (
            grounded_count / concept_node_count
            if concept_node_count else 1.0
        )
        asserted_count, derived_count = _split_asserted_derived(all_edges)
        denom_edge_share = asserted_count + derived_count
        asserted_edge_share = (
            asserted_count / denom_edge_share if denom_edge_share else 1.0
        )

        # ---- consistency: no SHACL report at authoring time → base 1.0,
        # then attenuate by (1 - contradiction_rate) so this composes
        # with the gate's EdgeConsensusAggregator attenuation rather than
        # duplicating it. Derive contradiction_rate from stamped
        # edge_status when the caller didn't supply it.
        if contradiction_rate is None:
            contradiction_rate = _contradiction_rate_from_edges(all_edges)
        contradiction_rate = max(0.0, min(1.0, float(contradiction_rate)))
        consistency_score = max(0.0, 1.0 * (1.0 - contradiction_rate))

        # ---- accuracy: no SHACL warning evidence at authoring time.
        accuracy_score = 1.0

        report: Dict[str, Any] = {
            "run_id": self.run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "course_slug": self.course_slug,
            "derivation": "semantic_graph_metrics_only",
            "contradiction_rate": _round(contradiction_rate),
            "dimensions": {
                "completeness": {
                    "score": _round(completeness_score),
                    "metric": (
                        "ratio of concept nodes carrying every required "
                        "predicate (authoring-time, semantic graph only)"
                    ),
                    "denominator": denominator,
                    "numerator": numerator,
                    "required_predicates": list(self.required_predicates),
                },
                "consistency": {
                    "score": _round(consistency_score),
                    "metric": (
                        "1.0 (no SHACL report at authoring time) attenuated "
                        "by (1 - contradiction_rate) from edge consensus"
                    ),
                    "violation_count": 0,
                    "warning_count": 0,
                    "total_focus_nodes": denominator,
                    "contradiction_rate": _round(contradiction_rate),
                },
                "accuracy": {
                    "score": _round(accuracy_score),
                    "metric": (
                        "1.0 (no SHACL warning evidence at authoring time)"
                    ),
                    "warning_count": 0,
                    "total_focus_nodes": denominator,
                },
                "coverage": {
                    "score": _round(coverage_score),
                    "metric": (
                        "chunk-anchored DomainConcept node-grounding: "
                        "DomainConcept-or-classless nodes incident to ≥1 "
                        "chunk-anchored edge / all DomainConcept-or-classless "
                        "nodes (share of concept vocabulary grounded in text)"
                    ),
                    "grounded_node_count": grounded_count,
                    "concept_node_count": concept_node_count,
                    "asserted_edge_share": _round(asserted_edge_share),
                    "asserted_count": asserted_count,
                    "derived_count": derived_count,
                },
            },
            "per_shape": [],
            "rule_outputs": rule_outputs,
        }
        return report

    def write(self, report: Dict[str, Any]) -> Path:
        """Write ``report`` as ``kg_quality_report.json`` under output_dir.

        Returns the written path. Creates ``output_dir`` if missing.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "kg_quality_report.json"
        out_path.write_text(
            json.dumps(report, indent=2, sort_keys=False),
            encoding="utf-8",
        )
        return out_path


# ---------------------------------------------------------------------- #
# Internals
# ---------------------------------------------------------------------- #


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Read a JSON file. Returns None on missing path / parse error."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _as_list(maybe_list: Any) -> List[Any]:
    return list(maybe_list) if isinstance(maybe_list, list) else []


#: Inference rules whose edges are CHUNK-ANCHORED — i.e. each such edge
#: is evidenced by one or more chunks of the source text, so an incident
#: concept node is "grounded in the text". These are the seven rules that
#: carry chunk-level evidence (verified against each rule module's
#: ``RULE_NAME`` constant in ``Trainforge/rag/inference_rules/`` and the
#: intra-chunk linker's ``RULE_NAME`` in
#: ``lib/ontology/intra_chunk_linker.py``):
#:
#: * ``related_from_cooccurrence`` — raw chunk co-occurrence.
#: * ``intra_chunk_link`` — co-located concepts within a single chunk.
#: * ``defined_by_from_first_mention`` — concept↔chunk definition anchor.
#: * ``exemplifies_from_example_chunks`` — concept↔example-chunk anchor.
#: * ``is_a_from_key_terms`` — taxonomic link evidenced by chunk key terms.
#: * ``misconception_of_from_misconception_ref`` — chunk misconception ref.
#: * ``assesses_from_question_lo`` — question↔LO assessment anchor.
#:
#: Edges carrying any OTHER rule (or a purely LO-order-derived rule such
#: as ``prerequisite_from_lo_order`` / ``derived_from_lo_ref`` /
#: ``targets_concept_from_lo``) are NOT chunk-anchored: they connect
#: concept vocabulary without grounding it in the corpus text, so they do
#: not count toward a node being "grounded".
_CHUNK_ANCHORED_RULES: frozenset = frozenset({
    "related_from_cooccurrence",
    "intra_chunk_link",
    "defined_by_from_first_mention",
    "exemplifies_from_example_chunks",
    "is_a_from_key_terms",
    "misconception_of_from_misconception_ref",
    "assesses_from_question_lo",
})


#: Node classes that count toward the concept-vocabulary coverage
#: denominator. A node with no ``class`` (legacy / classless) or
#: ``class == "DomainConcept"`` is concept vocabulary; pedagogical /
#: structural classes (``Chunk``, ``Outcome``, ``ComponentObjective``,
#: ``BloomLevel``, …) are NOT.
_CONCEPT_NODE_CLASSES: frozenset = frozenset({"DomainConcept"})


def _edge_is_chunk_anchored(edge: Any) -> bool:
    """A semantic-graph edge grounds its endpoints in the text iff it
    carries no ``provenance.rule`` OR a rule in
    :data:`_CHUNK_ANCHORED_RULES`.
    """
    if not isinstance(edge, dict):
        return False
    prov = edge.get("provenance") or {}
    rule = prov.get("rule") if isinstance(prov, dict) else None
    if not isinstance(rule, str) or not rule:
        return True
    return rule in _CHUNK_ANCHORED_RULES


def _is_concept_node(node: Any) -> bool:
    """A node counts toward the coverage denominator iff it is a
    DomainConcept (or classless / legacy node with no ``class``).
    """
    if not isinstance(node, dict):
        return False
    klass = node.get("class")
    if klass is None or klass == "":
        return True
    return klass in _CONCEPT_NODE_CLASSES


def _node_grounding_coverage(
    nodes: List[Any], edges: List[Any],
) -> tuple[int, int]:
    """Chunk-anchored DomainConcept node-grounding coverage.

    Returns ``(grounded_count, denominator)`` where the denominator is
    the count of DomainConcept-or-classless nodes and the numerator is
    the subset of those incident to ≥1 chunk-anchored edge. ``frequency``
    is irrelevant: a frequency-0 ``lo_key_concept`` node counts in the
    denominator and is covered only if some chunk-anchored edge touches
    it — i.e. coverage reads as "share of concept vocabulary (incl.
    LO-asserted concepts) grounded in the corpus text".
    """
    concept_ids = {
        node.get("id")
        for node in nodes
        if _is_concept_node(node) and node.get("id") is not None
    }
    grounded: set = set()
    for edge in edges:
        if not _edge_is_chunk_anchored(edge):
            continue
        for endpoint in (edge.get("source"), edge.get("target")):
            if endpoint in concept_ids:
                grounded.add(endpoint)
    return len(grounded), len(concept_ids)


def _split_asserted_derived(edges: List[Any]) -> tuple[int, int]:
    """Classify semantic-graph edges into (asserted, typed-derived).

    Retained for the informational ``asserted_edge_share`` figure carried
    in the coverage detail dict. Asserted: edges with no ``provenance.rule``
    OR a rule in :data:`_CHUNK_ANCHORED_RULES`. Typed-derived: every other
    rule (LO-order-derived inference — ``prerequisite_from_lo_order``,
    ``derived_from_lo_ref``, ``targets_concept_from_lo``, …).

    NOTE: this is no longer the coverage metric (which is now node
    grounding); it is preserved unthresholded so operators can still see
    the old asserted/derived edge ratio that anti-correlated with quality.
    """
    asserted = 0
    derived = 0
    for edge in edges:
        if _edge_is_chunk_anchored(edge):
            asserted += 1
        else:
            derived += 1
    return asserted, derived


def _contradiction_rate_from_edges(edges: List[Any]) -> float:
    """Derive the contradiction rate from per-edge ``edge_status``.

    The EdgeConsensusAggregator stamps ``edge_status`` on every edge
    immediately upstream of the authoring-time KG-quality computation.
    A ``contradicted`` status means a cross-rule (or same-rule reverse)
    disagreement fired over that edge's node pair. The rate is
    ``contradicted / total`` over edges that carry a stamped status;
    edges without a status (legacy / unstamped) don't count toward
    either numerator or denominator. Returns 0.0 when no edge carries a
    status (graceful degrade — un-stamped graphs read as no
    contradictions, leaving consistency at its 1.0 base).
    """
    stamped = 0
    contradicted = 0
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        status = edge.get("edge_status")
        if not isinstance(status, str) or not status:
            continue
        stamped += 1
        if status == "contradicted":
            contradicted += 1
    if stamped == 0:
        return 0.0
    return contradicted / stamped


def _node_has_required_predicates(
    node: Any, required: Iterable[str],
) -> bool:
    """A node satisfies completeness iff every required key is present
    and non-empty (None / empty-string / empty-list are treated as missing).
    """
    if not isinstance(node, dict):
        return False
    for pred in required:
        val = node.get(pred)
        if val is None:
            return False
        if isinstance(val, (str, list, dict)) and len(val) == 0:
            return False
    return True


def _normalize_results(validation_report: Any) -> List[Any]:
    """Return a list of result objects from a validation report.

    Tolerates: an object with ``.results`` attribute, a dict with
    ``"results"`` key, or a list of results passed directly.
    """
    if validation_report is None:
        return []
    if hasattr(validation_report, "results"):
        results = validation_report.results
    elif isinstance(validation_report, dict) and "results" in validation_report:
        results = validation_report["results"]
    elif isinstance(validation_report, list):
        results = validation_report
    else:
        return []
    return list(results) if results is not None else []


def _attr(result: Any, name: str) -> Any:
    """Fetch ``name`` from ``result`` whether it's an object or dict."""
    if isinstance(result, dict):
        return result.get(name)
    return getattr(result, name, None)


def _severity(result: Any) -> str:
    """Normalize a result's severity to one of critical / warning / info.

    Tolerates SHACL IRI strings (``http://www.w3.org/ns/shacl#Violation``)
    and the validator-framework strings used by ``ShaclViolation``.
    """
    sev = _attr(result, "severity")
    if sev is None:
        return "critical"
    s = str(sev)
    if s.endswith("#Violation") or s == "critical":
        return "critical"
    if s.endswith("#Warning") or s == "warning":
        return "warning"
    if s.endswith("#Info") or s == "info":
        return "info"
    return s.lower()


def _rollup_per_shape(results: List[Any]) -> List[Dict[str, Any]]:
    """Group results by ``source_shape`` IRI and tally violations / warnings.

    ``focus_nodes`` is the count of distinct focus_node IRIs that
    triggered any result for that shape — a lightweight proxy for the
    shape's denominator without re-running SHACL.
    """
    grouped: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"violations": 0, "warnings": 0, "focus_nodes": set()}
    )
    for r in results:
        shape = _attr(r, "source_shape")
        shape_iri = str(shape) if shape is not None else "(unbound)"
        sev = _severity(r)
        bucket = grouped[shape_iri]
        if sev == "critical":
            bucket["violations"] += 1
        elif sev == "warning":
            bucket["warnings"] += 1
        focus = _attr(r, "focus_node")
        if focus is not None:
            bucket["focus_nodes"].add(str(focus))

    rollup: List[Dict[str, Any]] = []
    for shape_iri, bucket in sorted(grouped.items()):
        rollup.append({
            "shape_iri": shape_iri,
            "violations": bucket["violations"],
            "warnings": bucket["warnings"],
            "focus_nodes": len(bucket["focus_nodes"]),
        })
    return rollup


def _summarize_rule_outputs(
    semantic: Dict[str, Any], run_id: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Walk the semantic graph; tally edges per rule, mint rule IRIs.

    The named-graph IRI scheme ``https://ed4all.io/run/<run_id>/rule/<rule>``
    matches ``Trainforge/rag/named_graph_writer.py``'s
    ``mint_rule_graph_iri``. We reconstruct it here from the per-edge
    ``provenance.rule`` field so the report works on the JSON form
    without requiring the TriG sibling artifact.

    Returns:
        ``(derived_edges, rule_outputs)``. ``derived_edges`` is the
        flat list of edges used for the coverage denominator;
        ``rule_outputs`` is the per-rule rollup [{rule_iri,
        edge_count, rule_version}, ...].
    """
    edges = _as_list(semantic.get("edges"))
    rule_versions = semantic.get("rule_versions") or {}

    derived_edges: List[Dict[str, Any]] = []
    counter: Counter = Counter()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        prov = edge.get("provenance") or {}
        if not isinstance(prov, dict):
            continue
        rule = prov.get("rule")
        if not isinstance(rule, str) or not rule:
            continue
        derived_edges.append(edge)
        counter[rule] += 1

    rule_outputs: List[Dict[str, Any]] = []
    for rule, count in sorted(counter.items()):
        rule_outputs.append({
            "rule_iri": f"{RULE_GRAPH_IRI_PREFIX}{run_id}/rule/{rule}",
            "edge_count": int(count),
            "rule_version": rule_versions.get(rule),
        })
    return derived_edges, rule_outputs


def _round(value: float) -> float:
    """Round to four decimal places for JSON readability."""
    return round(float(value), 4)


__all__ = [
    "KGQualityReporter",
    "DEFAULT_REQUIRED_PREDICATES",
    "RULE_GRAPH_IRI_PREFIX",
]

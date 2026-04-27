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
          "course_slug": "rdf-shacl-551-2",
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

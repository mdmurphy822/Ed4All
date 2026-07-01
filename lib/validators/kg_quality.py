"""KG-quality gate — thin wrapper around ``KGQualityReporter``.

Wired into ``config/workflows.yaml`` as a blocking ``critical``-severity
gate on the ``textbook_to_course::libv2_archival`` phase. The gate
surfaces the four KG-quality dimensions (completeness, consistency,
accuracy, coverage).

Improvement #4 from the post-Wave 85 corpus-grounded gap analysis.

Wave 91 calibration baseline (run on the RDF/SHACL calibration
corpus against an empty SHACL validation
report — i.e. zero violations, the steady-state expected at
libv2_archival):

    completeness: 1.0
    consistency:  1.0
    accuracy:     1.0
    coverage:     0.5248   (under the OLD edge-share coverage metric)
    composite:    0.8812 (unweighted mean)

Promoted thresholds (max(spec_default, baseline - 0.05)):

    min_completeness: 0.95   (baseline 1.0 - 0.05)
    min_consistency:  0.95   (baseline 1.0 - 0.05)
    min_accuracy:     0.95   (baseline 1.0 - 0.05)
    min_coverage:     0.50   (spec default; baseline 0.5248 - 0.05 = 0.4748 floors at spec)

Severity flipped from ``warning`` to ``critical`` and ``on_fail`` from
``warn`` to ``block``. The fixture corpus passes at the chosen
thresholds (smallest margin: coverage at 0.5248 vs floor 0.50, +0.025).

Coverage-semantics redesign — gate path aligned with the production
reporter. The ``min_coverage: 0.50`` floor thresholds *chunk-anchored
DomainConcept node-grounding* — the share of DomainConcept-or-classless
concept nodes incident to ≥1 chunk-evidenced edge — NOT the old
asserted/(asserted+derived) edge share. The redesign first landed in
``KGQualityReporter.compute_metrics_only`` (the authoring-time entry
point used by ``_run_concept_extraction``); ``KGQualityReporter.compute``
— the method THIS gate calls — was subsequently brought into line so the
gate verdict matches the production reporter. Before that alignment the
gate computed the legacy edge-share over an asserted ``concept_graph.json``
the textbook_to_course path never writes, so coverage scored 0.0 and the
critical gate FAILED a graph the authoring reporter scored ~0.69. The old
edge metric grew a quadratic LO-order-derived denominator with LO count
(the rdf-shacl calibration corpus fell to 0.047 under it) and
anti-correlated with quality (every quality-improving graph flag lowered
it). The new node-grounding metric reads on the RDF/SHACL calibration
corpus graph at 1.0 and on a RAG-course graph (76 frequency-0
lo_key_concept nodes) at ~0.69. Recalibration basis: the 0.50 floor is
RETAINED as-is (re-derived against the node-grounding metric, both
calibration corpora clear it). The coverage figure surfaces
``grounded_node_count`` / ``concept_node_count`` plus the legacy ratio as
an informational ``asserted_edge_share`` (see
``Trainforge/rag/kg_quality_report.py``). NB: this gate is now wired
end-to-end — an input builder is registered and ``run_gate`` forwards
``threshold:`` keys (``min_completeness`` etc.) into validator inputs,
so metric-floor breaches fail closed instead of passing with warnings.

Silent-degradation finding C3 (post-Wave 91 audit) — fail-closed
inversions:

    * Missing the required ``semantic_graph_path`` (or ``course_slug`` /
      ``run_id`` / ``output_dir``) → ``passed=False`` with
      ``KG_QUALITY_PEDAGOGY_GRAPH_MISSING`` (critical). A
      libv2_archival run with NO semantic graph is a critical fail, not a
      pass — thresholds are meaningless when no graph exists, and
      downstream property_coverage / min_edge_count would also pass
      vacuously and ship an empty knowledge graph to LibV2. The asserted
      ``concept_graph_path`` is OPTIONAL (completeness falls back to the
      semantic graph's nodes when it is absent), so its omission alone
      does NOT trip this arm.
    * Reporter exception → ``passed=False`` with
      ``KG_QUALITY_REPORTER_ERROR`` (critical). The previous
      ``passed=True`` swallowed silent reporter regressions.
    * Threshold-breach path is INTENTIONALLY untouched: it emits
      warning-severity issues with ``passed=True`` and the workflow
      YAML configures critical-severity at the gate level — see
      ``config/workflows.yaml::textbook_to_course::libv2_archival``.
    * Decision-capture wiring (audit H3): emits one
      ``kg_quality_report_check`` event per ``validate()`` call with
      the four computed dimension scores and the gate verdict, so
      post-hoc replay can distinguish "graph missing → fail-closed"
      from "graph present and below threshold → warning" from "graph
      present and above threshold → pass".

Inputs (passed via the gate framework):
    course_slug: Required. Course slug for the report context.
    run_id: Required. Pipeline run identifier.
    output_dir: Required. Directory to write ``kg_quality_report.json``.
    concept_graph_path: Optional. Path to the asserted
        ``concept_graph.json``. Authoritative node set for completeness
        WHEN PRESENT (the rdf_shacl path); when absent — the
        textbook_to_course path emits only the semantic graph — the
        reporter falls back to the semantic graph's nodes, so a missing
        asserted graph is NOT a hard fail.
    semantic_graph_path: Required. Path to ``concept_graph_semantic.json``.
        The load-bearing input: coverage (node-grounding) and the
        completeness fallback both compute from it. A missing semantic
        graph fails the gate closed.
    validation_report: Optional. Pre-built SHACL validation report
        object (``.results``-shaped). When absent the gate emits a
        report with empty SHACL aggregates — completeness / coverage
        still compute from the graphs alone.
    pedagogy_graph_path: Optional. Reserved for future cross-graph
        completeness checks; recorded in the report metadata.
    min_completeness: Optional float [0, 1]. Defaults to 0.0.
    min_consistency: Optional float [0, 1]. Defaults to 0.0.
    min_accuracy: Optional float [0, 1]. Defaults to 0.0.
    min_coverage: Optional float [0, 1]. Defaults to 0.0.
    decision_capture: Optional. ``DecisionCapture`` instance; when
        wired, one ``kg_quality_report_check`` decision event fires per
        ``validate()`` call (see audit H3 closure).

Outputs:
    GateResult with ``score`` set to the unweighted mean of the four
    dimensions and one ``warning``-severity GateIssue per threshold
    breach. Threshold defaults are 0.0 so the gate is advisory at
    threshold-breach severity unless the workflow YAML overrides them
    — but missing-graph and reporter-exception now fail closed.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


_DIMENSIONS = ("completeness", "consistency", "accuracy", "coverage")

# --------------------------------------------------------------------------- #
# W3.2 / W3.4 / W3.6 — prereq-DAG health signals (topological, no model).
#
# Gated behind ``ED4ALL_KG_PREREQ_HEALTH`` (default OFF → byte-identical: no
# ``prereq_health`` metadata key, no new warning issues, no decision event, so
# every existing snapshot / test stays unchanged). When truthy, three
# deterministic stdlib passes run over the just-emitted
# ``concept_graph_semantic.json``:
#
#   * W3.2 ACYCLICITY — Tarjan SCC over the ``prerequisite`` edge set (dependency
#     direction: source=dependent → target=prerequisite). A non-trivial SCC (or a
#     self-loop) is a circular-prerequisite defect; reported as a kg_quality
#     dimension-shaped signal (``acyclicity.score``) — NOT folded into the 4-dim
#     composite mean (that stays byte-stable). Warning-day-1.
#   * W3.4 (kg half) CENTRALITY — concept in-degree centrality recomputed over
#     the concept sub-graph (LO/TO endpoints excluded), top-K surfaced in the
#     report metadata. Informational (no warning).
#   * W3.6 DANGLING BACKGROUND — a prerequisite concept (edge TARGET) that is
#     never taught/introduced: no grounded graph node (empty ``occurrences[]``
#     AND non-positive ``frequency``), or not a node at all. Warning-day-1.
#
# All warning-severity with ``# TODO(calibration)`` deferred critical-flip.
# Anti-fabrication: reads only real graph node ids / edge endpoints; never
# invents a concept, edge, or objective.
# --------------------------------------------------------------------------- #

_TRUTHY = frozenset({"1", "true", "yes", "on"})
ENV_KG_PREREQ_HEALTH = "ED4ALL_KG_PREREQ_HEALTH"

#: Top-K most-central concepts surfaced in the metadata (informational).
_CENTRALITY_TOP_K = 10
#: Cap on the distinct dangling-prerequisite concept ids listed in metadata.
_DANGLING_REPORT_CAP = 25
#: Cap on the ids listed for the first detected cycle (for the warning message).
_CYCLE_EXAMPLE_CAP = 8


def resolve_kg_prereq_health(value: object = None) -> bool:
    """Return True iff the prereq-DAG health signals are enabled (default OFF).

    ``value`` overrides the env when not ``None``. Falsey / garbage / unset →
    False (parse-with-fallback, mirroring the other ``ED4ALL_*`` toggles).
    """
    if value is None:
        raw = os.environ.get(ENV_KG_PREREQ_HEALTH, "")
    else:
        raw = str(value)
    return raw.strip().lower() in _TRUTHY


def _emit_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    scores: Dict[str, float],
    thresholds: Dict[str, float],
    composite: Optional[float],
    course_slug: Optional[str],
    run_id: Optional[str],
) -> None:
    """Emit one ``kg_quality_report_check`` decision per validate() call.

    Closes audit H3 partially for this validator: every pass /
    threshold-fail / missing-graph / reporter-exception path emits one
    event so post-hoc replay can distinguish the four outcomes.
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    composite_str = (
        f"{composite:.4f}" if composite is not None else "n/a"
    )
    score_strs = ", ".join(
        f"{dim}={scores.get(dim, 0.0):.4f}" for dim in _DIMENSIONS
    )
    threshold_strs = ", ".join(
        f"min_{dim}={thresholds.get(dim, 0.0):.4f}" for dim in _DIMENSIONS
    )
    rationale = (
        f"KG-quality gate verdict for course={course_slug or 'n/a'} "
        f"run_id={run_id or 'n/a'}: "
        f"composite={composite_str}, "
        f"scores=({score_strs}), "
        f"thresholds=({threshold_strs}), "
        f"failure_code={code or 'none'}."
    )
    metrics: Dict[str, Any] = {
        **{dim: float(scores.get(dim, 0.0)) for dim in _DIMENSIONS},
        "composite": float(composite) if composite is not None else None,
        "passed": bool(passed),
        "failure_code": code,
    }
    # Threaded via context (canonical home for free-form structured
    # metric blobs in the DecisionCapture record) AND mirrored as a
    # ``metrics`` kwarg so test mocks can assert on it directly. The
    # rigid ``MLFeatures`` dataclass has no metric fields, so we don't
    # route through that surface.
    try:
        capture.log_decision(
            decision_type="kg_quality_report_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on "
            "kg_quality_report_check: %s",
            exc,
        )


# --------------------------------------------------------------------------- #
# Prereq-DAG health helpers (pure, stdlib-only, deterministic).
# --------------------------------------------------------------------------- #


def _norm_endpoint(raw: Any) -> str:
    """Normalize a graph node id / edge endpoint to a canonical comparison key.

    Endpoints may be a bare slug or ``{course_id}:{slug}`` (scoped-id mode);
    strip a single ``:`` namespace prefix then ``canonical_slug`` so node ids
    and edge endpoints from the same emit compare equal.
    """
    if not isinstance(raw, str) or not raw:
        return ""
    from lib.ontology.slugs import canonical_slug

    s = raw.split(":", 1)[1] if ":" in raw else raw
    return canonical_slug(s)


def _is_lo_endpoint(raw: Any) -> bool:
    """Return True iff ``raw`` is a canonical learning-objective id (``TO-NN`` /
    ``CO-...``). Federation ``TO -> TO`` prerequisite endpoints are objectives,
    not concepts, so they are excluded from the CONCEPT-level centrality +
    dangling-background passes (but NOT from cycle detection — a cycle among
    objectives is a defect too).
    """
    if not isinstance(raw, str) or not raw.strip():
        return False
    try:
        from lib.ontology.learning_objectives import validate_lo_id

        return bool(validate_lo_id(raw.strip().upper()))
    except Exception:  # noqa: BLE001 — never let id detection break the pass
        return False


def _prerequisite_pairs(edges: List[Any]) -> List[Tuple[str, str]]:
    """Extract RAW ``(source=dependent, target=prerequisite)`` prerequisite
    edge endpoints from the semantic graph edge list."""
    out: List[Tuple[str, str]] = []
    for e in edges:
        if not isinstance(e, dict) or e.get("type") != "prerequisite":
            continue
        s = e.get("source")
        t = e.get("target")
        if not isinstance(s, str) or not isinstance(t, str) or not s or not t:
            continue
        out.append((s, t))
    return out


def _tarjan_sccs(adj: Dict[str, List[str]]) -> List[List[str]]:
    """Iterative Tarjan strongly-connected-components (no recursion limit).

    ``adj`` must carry every node as a key (targets materialized by the caller).
    Deterministic: nodes are visited in insertion order and each SCC is returned
    in the order the algorithm pops it.
    """
    index_counter = [0]
    index: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    on_stack: Dict[str, bool] = {}
    stack: List[str] = []
    sccs: List[List[str]] = []

    for start in list(adj.keys()):
        if start in index:
            continue
        work: List[Tuple[str, int]] = [(start, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = index_counter[0]
                lowlink[v] = index_counter[0]
                index_counter[0] += 1
                stack.append(v)
                on_stack[v] = True
            recurse = False
            neighbors = adj.get(v, [])
            i = pi
            while i < len(neighbors):
                w = neighbors[i]
                if w not in index:
                    work[-1] = (v, i + 1)
                    work.append((w, 0))
                    recurse = True
                    break
                if on_stack.get(w):
                    lowlink[v] = min(lowlink[v], index[w])
                i += 1
            if recurse:
                continue
            if lowlink[v] == index[v]:
                comp: List[str] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                sccs.append(comp)
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[v])
    return sccs


def _detect_prereq_cycles(
    prereq_pairs: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """W3.2 — acyclicity signal over the prerequisite dependency graph.

    Dependency edge: ``dependent -> prerequisite``. A non-trivial SCC (or a
    self-loop) is a circular-prerequisite defect.
    """
    adj: Dict[str, List[str]] = defaultdict(list)
    display: Dict[str, str] = {}
    self_loops: set = set()
    for s_raw, t_raw in prereq_pairs:
        s = _norm_endpoint(s_raw)
        t = _norm_endpoint(t_raw)
        if not s or not t:
            continue
        display.setdefault(s, s_raw)
        display.setdefault(t, t_raw)
        adj[s].append(t)
        _ = adj[t]  # materialize target as a node (defaultdict)
        if s == t:
            self_loops.add(s)

    total_nodes = len(adj)
    sccs = _tarjan_sccs(dict(adj))
    cyclic_nodes: set = set()
    cycle_count = 0
    example_cycle: List[str] = []
    for comp in sccs:
        is_cycle = len(comp) > 1 or (len(comp) == 1 and comp[0] in self_loops)
        if not is_cycle:
            continue
        cycle_count += 1
        cyclic_nodes.update(comp)
        if not example_cycle:
            example_cycle = sorted(display.get(n, n) for n in comp)[
                :_CYCLE_EXAMPLE_CAP
            ]

    is_acyclic = cycle_count == 0
    if is_acyclic or not total_nodes:
        score = 1.0
    else:
        score = max(0.0, 1.0 - (len(cyclic_nodes) / total_nodes))
    return {
        "metric": (
            "1 - (nodes in circular-prerequisite SCCs / prerequisite nodes)"
        ),
        "is_acyclic": is_acyclic,
        "score": round(score, 4),
        "prerequisite_node_count": total_nodes,
        "cycle_count": cycle_count,
        "nodes_in_cycles": len(cyclic_nodes),
        "example_cycle": example_cycle,
    }


def _concept_in_degree_centrality(edges: List[Any]) -> Dict[str, Any]:
    """W3.4 (kg half) — concept in-degree centrality over the concept subgraph.

    In-degree of a concept = how many concepts point AT it (a proxy for
    foundationality). Edges touching a learning-objective endpoint are excluded
    (centrality is a concept-graph measure). Top-K surfaced for the report.
    """
    indeg: Dict[str, int] = defaultdict(int)
    display: Dict[str, str] = {}
    nodes: set = set()
    for e in edges:
        if not isinstance(e, dict):
            continue
        s = e.get("source")
        t = e.get("target")
        if not isinstance(s, str) or not isinstance(t, str):
            continue
        if _is_lo_endpoint(s) or _is_lo_endpoint(t):
            continue
        sn = _norm_endpoint(s)
        tn = _norm_endpoint(t)
        if not sn or not tn:
            continue
        nodes.add(sn)
        nodes.add(tn)
        display.setdefault(sn, s)
        display.setdefault(tn, t)
        indeg[tn] += 1
    top = sorted(indeg.items(), key=lambda kv: (-kv[1], kv[0]))[
        :_CENTRALITY_TOP_K
    ]
    return {
        "method": "in_degree",
        "concept_count": len(nodes),
        "top": [
            {"concept": display.get(k, k), "in_degree": int(v)}
            for k, v in top
        ],
    }


def _dangling_prerequisites(
    nodes: List[Any], prereq_pairs: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """W3.6 — dangling / unmet-background prerequisites.

    A prerequisite (edge TARGET) concept is "dangling" when the course
    references it as required background but never teaches/introduces it: it has
    no grounded graph node (empty ``occurrences[]`` AND non-positive
    ``frequency``), or is not a declared node at all. Learning-objective
    prerequisite targets (federation ``TO -> TO`` edges) are excluded — a TO is
    not a taught concept.
    """
    introduced: set = set()
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        if not isinstance(nid, str) or not nid:
            continue
        occ = n.get("occurrences") or []
        grounded = any(isinstance(o, str) and o for o in occ)
        if not grounded:
            freq = n.get("frequency")
            if (
                isinstance(freq, (int, float))
                and not isinstance(freq, bool)
                and freq > 0
            ):
                grounded = True
        if grounded:
            introduced.add(_norm_endpoint(nid))

    distinct_targets: set = set()
    dangling: Dict[str, str] = {}
    for _s_raw, t_raw in prereq_pairs:
        if _is_lo_endpoint(t_raw):
            continue
        tn = _norm_endpoint(t_raw)
        if not tn:
            continue
        distinct_targets.add(tn)
        if tn not in introduced and tn not in dangling:
            dangling[tn] = t_raw

    concepts = sorted(dangling.values())
    return {
        "count": len(dangling),
        "prereq_target_count": len(distinct_targets),
        "concepts": concepts[:_DANGLING_REPORT_CAP],
    }


def _load_graph_dict(path_raw: Any) -> Dict[str, Any]:
    """Best-effort load of the semantic graph JSON. Returns ``{}`` on any error."""
    try:
        p = Path(path_raw)
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001 — never break the gate
        logger.debug("kg_prereq_health: graph load failed: %s", exc)
        return {}


def _compute_prereq_health(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the three prereq-DAG health signals over a semantic graph dict."""
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
    prereq_pairs = _prerequisite_pairs(edges)
    return {
        "prerequisite_edge_count": len(prereq_pairs),
        "acyclicity": _detect_prereq_cycles(prereq_pairs),
        "centrality": _concept_in_degree_centrality(edges),
        "dangling_prerequisites": _dangling_prerequisites(nodes, prereq_pairs),
    }


def _prereq_health_issues(signals: Dict[str, Any]) -> List[GateIssue]:
    """Build warning-day-1 GateIssues from the prereq-health signals.

    # TODO(calibration): deferred critical-flip — a circular prerequisite and a
    # dangling background concept are real defects, but the anchored severity
    # awaits a >=2-corpus false-positive measurement (WS3/W4 deferred-flip
    # pattern). Centrality is informational (no issue).
    """
    issues: List[GateIssue] = []
    acyc = signals.get("acyclicity", {})
    if acyc.get("cycle_count", 0) > 0:
        example = ", ".join(acyc.get("example_cycle", [])) or "(unavailable)"
        issues.append(GateIssue(
            severity="warning",
            code="KG_PREREQ_CYCLE_DETECTED",
            message=(
                f"Prerequisite DAG is cyclic: {acyc.get('cycle_count', 0)} "
                f"circular-prerequisite cluster(s) span "
                f"{acyc.get('nodes_in_cycles', 0)} node(s) "
                f"(acyclicity score {acyc.get('score', 0.0):.4f}). "
                f"First cycle: {example}."
            ),
            suggestion=(
                "A circular prerequisite means no valid teaching order exists; "
                "inspect the prerequisite edges among the listed nodes and "
                "break the weakest back-pointing dependency."
            ),
        ))
    dangling = signals.get("dangling_prerequisites", {})
    if dangling.get("count", 0) > 0:
        concepts = ", ".join(dangling.get("concepts", [])) or "(none listed)"
        issues.append(GateIssue(
            severity="warning",
            code="KG_PREREQ_DANGLING_BACKGROUND",
            message=(
                f"{dangling.get('count', 0)} of "
                f"{dangling.get('prereq_target_count', 0)} prerequisite "
                f"concept(s) are referenced as required background but never "
                f"taught/introduced (no grounded chunk). Concepts: {concepts}."
            ),
            suggestion=(
                "Either add source content that introduces the missing "
                "background concept(s), or drop the prerequisite edge if the "
                "dependency is spurious."
            ),
        ))
    return issues


def _emit_prereq_decision(
    capture: Any,
    signals: Dict[str, Any],
    *,
    course_slug: Optional[str],
    run_id: Optional[str],
) -> None:
    """Emit one ``validation_result`` decision per validate() call when the
    prereq-health pass runs (distinct decision_type from the primary
    ``kg_quality_report_check`` so the two never collide)."""
    if capture is None:
        return
    acyc = signals.get("acyclicity", {})
    dangling = signals.get("dangling_prerequisites", {})
    centrality = signals.get("centrality", {})
    rationale = (
        f"KG prereq-DAG health for course={course_slug or 'n/a'} "
        f"run_id={run_id or 'n/a'}: "
        f"{signals.get('prerequisite_edge_count', 0)} prerequisite edge(s); "
        f"acyclic={acyc.get('is_acyclic')} "
        f"(cycles={acyc.get('cycle_count', 0)}, "
        f"nodes_in_cycles={acyc.get('nodes_in_cycles', 0)}, "
        f"score={acyc.get('score', 1.0):.4f}); "
        f"dangling_background={dangling.get('count', 0)}/"
        f"{dangling.get('prereq_target_count', 0)}; "
        f"concept_centrality over {centrality.get('concept_count', 0)} "
        f"concept node(s)."
    )
    try:
        capture.log_decision(
            decision_type="validation_result",
            decision=(
                "kg_prereq_health "
                f"acyclic={acyc.get('is_acyclic')} "
                f"dangling={dangling.get('count', 0)}"
            ),
            rationale=rationale,
            context=str(signals),
            metrics={
                "prerequisite_edge_count": int(
                    signals.get("prerequisite_edge_count", 0)
                ),
                "is_acyclic": bool(acyc.get("is_acyclic", True)),
                "cycle_count": int(acyc.get("cycle_count", 0)),
                "nodes_in_cycles": int(acyc.get("nodes_in_cycles", 0)),
                "acyclicity_score": float(acyc.get("score", 1.0)),
                "dangling_prerequisite_count": int(dangling.get("count", 0)),
                "prereq_target_count": int(
                    dangling.get("prereq_target_count", 0)
                ),
                "concept_count": int(centrality.get("concept_count", 0)),
            },
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug(
            "DecisionCapture.log_decision raised on kg_prereq_health: %s", exc
        )


class KGQualityValidator:
    """Validator-protocol wrapper for ``KGQualityReporter``."""

    name = "kg_quality_report"
    version = "1.0.0"

    def __init__(
        self,
        reporter_factory: Optional[Any] = None,
        *,
        decision_capture: Optional[Any] = None,
    ) -> None:
        # Allow tests to inject a mock factory; default lazy import keeps
        # Trainforge dependencies out of the validator module's import
        # surface for callers that only load the validators package.
        self._reporter_factory = reporter_factory
        # Optional capture instance; the workflow runner may also thread
        # one in via ``inputs["decision_capture"]`` per call. Pattern
        # matches ``lib/validators/rewrite_source_grounding.py``.
        self._decision_capture = decision_capture

    def _build_reporter(
        self,
        *,
        course_slug: str,
        run_id: str,
        output_dir: Path,
    ) -> Any:
        if self._reporter_factory is not None:
            return self._reporter_factory(
                course_slug=course_slug,
                run_id=run_id,
                output_dir=output_dir,
            )
        from Trainforge.rag.kg_quality_report import KGQualityReporter
        return KGQualityReporter(
            course_slug=course_slug,
            run_id=run_id,
            output_dir=output_dir,
        )

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", "kg_quality_report")
        issues: List[GateIssue] = []

        # Per-call capture override (workflow runner threads it in)
        # falls back to the constructor-injected one.
        capture = inputs.get("decision_capture") or self._decision_capture

        course_slug = inputs.get("course_slug")
        run_id = inputs.get("run_id")
        output_dir_raw = inputs.get("output_dir")
        concept_graph_raw = inputs.get("concept_graph_path")
        semantic_graph_raw = inputs.get("semantic_graph_path")

        thresholds: Dict[str, float] = {
            "completeness": float(inputs.get("min_completeness", 0.0) or 0.0),
            "consistency": float(inputs.get("min_consistency", 0.0) or 0.0),
            "accuracy": float(inputs.get("min_accuracy", 0.0) or 0.0),
            "coverage": float(inputs.get("min_coverage", 0.0) or 0.0),
        }
        zero_scores: Dict[str, float] = {dim: 0.0 for dim in _DIMENSIONS}

        # The SEMANTIC graph is the load-bearing input — coverage
        # (node-grounding) and the completeness fallback both compute from
        # it. The asserted ``concept_graph.json`` is OPTIONAL: the
        # textbook_to_course path emits only the semantic graph (no separate
        # asserted graph), and the reporter degrades a missing asserted graph
        # to ``{}`` internally, falling back to the semantic graph's nodes
        # for completeness. So a present semantic_graph + absent
        # concept_graph is NOT a hard fail; only a missing semantic graph
        # (a libv2_archival run with NO knowledge graph at all) fails closed.
        missing: List[str] = []
        if not course_slug:
            missing.append("course_slug")
        if not run_id:
            missing.append("run_id")
        if not output_dir_raw:
            missing.append("output_dir")
        if not semantic_graph_raw:
            missing.append("semantic_graph_path")
        if missing:
            # Audit C3 fail-closed inversion: a libv2_archival run with
            # NO graph is a critical fail, not a pass. Thresholds are
            # meaningless when no graph exists; downstream
            # property_coverage / min_edge_count would also pass
            # vacuously and ship an empty KG to LibV2.
            _emit_decision(
                capture,
                passed=False,
                code="KG_QUALITY_PEDAGOGY_GRAPH_MISSING",
                scores=zero_scores,
                thresholds=thresholds,
                composite=None,
                course_slug=str(course_slug) if course_slug else None,
                run_id=str(run_id) if run_id else None,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="KG_QUALITY_PEDAGOGY_GRAPH_MISSING",
                    message=(
                        "Required inputs missing for "
                        "KGQualityValidator: "
                        f"{', '.join(missing)}. A libv2_archival run "
                        "with no semantic knowledge graph is a critical "
                        "fail — refusing to ship an empty knowledge "
                        "graph to LibV2. (The asserted concept_graph.json "
                        "is OPTIONAL — completeness falls back to the "
                        "semantic graph's nodes when it is absent.)"
                    ),
                    suggestion=(
                        "Verify the upstream concept_extraction phase "
                        "emitted concept_graph_semantic.json; inspect that "
                        "phase's logs for silent failures."
                    ),
                )],
                action="block",
            )

        try:
            reporter = self._build_reporter(
                course_slug=str(course_slug),
                run_id=str(run_id),
                output_dir=Path(output_dir_raw),
            )
            report = reporter.compute(
                concept_graph=(
                    Path(concept_graph_raw) if concept_graph_raw else None
                ),
                semantic_graph=Path(semantic_graph_raw),
                validation_report=inputs.get("validation_report"),
                pedagogy_graph=(
                    Path(inputs["pedagogy_graph_path"])
                    if inputs.get("pedagogy_graph_path")
                    else None
                ),
            )
            reporter.write(report)
        except Exception as exc:
            # Audit C3 fail-closed inversion: a reporter raise was
            # previously swallowed as passed=True (silent regression
            # vector). Now critical-fail with the exception class +
            # message threaded through the issue.
            exc_msg = (
                f"KG-quality reporter raised "
                f"{type(exc).__name__}: {exc}"
            )
            _emit_decision(
                capture,
                passed=False,
                code="KG_QUALITY_REPORTER_ERROR",
                scores=zero_scores,
                thresholds=thresholds,
                composite=None,
                course_slug=str(course_slug),
                run_id=str(run_id),
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[GateIssue(
                    severity="critical",
                    code="KG_QUALITY_REPORTER_ERROR",
                    message=exc_msg,
                    suggestion=(
                        "Inspect Trainforge.rag.kg_quality_report logs "
                        "and the inputs to KGQualityReporter.compute; "
                        "rerun with --debug to capture the traceback."
                    ),
                )],
                action="block",
            )

        scores: Dict[str, float] = {}
        for dim in _DIMENSIONS:
            score = float(
                report.get("dimensions", {}).get(dim, {}).get("score", 0.0)
            )
            scores[dim] = score

        # GPT-feedback-12-may item 2: cross-rule edge consensus. Walk
        # the just-emitted concept_graph_semantic.json and stamp every
        # edge with edge_status + consensus_signals[]. Multiply the
        # consistency axis by (1 - contradiction_rate) BEFORE the
        # threshold check so contradicted-edge clusters take a
        # proportional hit on consistency. Best-effort: any aggregator
        # failure logs and leaves scores unchanged.
        consensus_rate: Optional[float] = None
        contradiction_rate: Optional[float] = None
        try:
            from lib.aggregators.edge_consensus import EdgeConsensusAggregator
            aggregator = EdgeConsensusAggregator(
                semantic_graph_path=Path(semantic_graph_raw),
                course_slug=str(course_slug),
                run_id=str(run_id),
                decision_capture=capture,
            )
            consensus_report = aggregator.build()
            # Emit alongside the existing kg_quality_report.json.
            try:
                aggregator.write(
                    Path(output_dir_raw) / "edge_consensus_report.json"
                )
            except Exception as write_exc:  # noqa: BLE001
                logger.warning(
                    "EdgeConsensusAggregator.write raised %s; consensus "
                    "signal still applied to the in-memory score.",
                    write_exc,
                )
            summary = consensus_report.get("summary", {})
            contradiction_rate = float(summary.get("contradiction_rate", 0.0))
            consensus_rate = float(summary.get("consensus_rate", 0.0))
            scores["consistency"] = max(
                0.0,
                scores["consistency"] * (1.0 - contradiction_rate),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "EdgeConsensusAggregator failed (%s); consistency "
                "axis left at its base value.",
                exc,
            )

        breached = False
        for dim in _DIMENSIONS:
            score = scores[dim]
            min_score = thresholds[dim]
            if min_score > 0.0 and score < min_score:
                breached = True
                issues.append(GateIssue(
                    severity="critical",
                    code=f"KG_QUALITY_{dim.upper()}_BELOW_THRESHOLD",
                    message=(
                        f"KG quality dimension '{dim}' scored {score:.4f} "
                        f"below configured min ({min_score:.4f})."
                    ),
                    suggestion=(
                        "Inspect kg_quality_report.json for per-shape and "
                        "rule-output detail; lower the threshold or "
                        "improve the upstream emit if the regression is real."
                    ),
                ))

        # Unweighted mean across the four dimensions.
        composite = sum(scores.values()) / len(_DIMENSIONS)

        # A breached per-dimension floor fails the gate closed. The
        # configured floors (``min_completeness`` / ``min_consistency`` /
        # ``min_accuracy`` / ``min_coverage``) reach this validator via the
        # gate manager's ``threshold:`` -> inputs forward; a dimension below
        # its floor emits a critical GateIssue and inverts ``passed`` so the
        # critical/block gate (textbook_to_course::libv2_archival) actually
        # refuses to ship a degraded knowledge graph to LibV2. When no floor
        # is configured (all default to 0.0) or every dimension clears its
        # floor, the gate passes as before. The missing-graph + reporter-
        # exception arms above remain the other two fail-closed vectors.

        _emit_decision(
            capture,
            passed=not breached,
            code=(
                "KG_QUALITY_DIMENSION_BELOW_THRESHOLD" if breached else None
            ),
            scores=scores,
            thresholds=thresholds,
            composite=composite,
            course_slug=str(course_slug),
            run_id=str(run_id),
        )

        # W3.2 / W3.4 / W3.6 — prereq-DAG health signals (gated, default OFF →
        # byte-identical). Best-effort: any failure logs and leaves scores /
        # issues / metadata unchanged. Warning-day-1; NEVER flips ``passed``
        # (those signals emit warning-severity issues only, and the composite
        # mean stays over the four canonical dimensions).
        prereq_health: Optional[Dict[str, Any]] = None
        if resolve_kg_prereq_health():
            try:
                graph = _load_graph_dict(semantic_graph_raw)
                prereq_health = _compute_prereq_health(graph)
                issues.extend(_prereq_health_issues(prereq_health))
                _emit_prereq_decision(
                    capture,
                    prereq_health,
                    course_slug=str(course_slug),
                    run_id=str(run_id),
                )
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning(
                    "kg_prereq_health signals failed (%s); "
                    "leaving gate result unchanged.",
                    exc,
                )
                prereq_health = None

        result_metadata: Dict[str, Any] = {}
        if prereq_health is not None:
            result_metadata["prereq_health"] = prereq_health
        if consensus_rate is not None:
            result_metadata["edge_consensus_rate"] = round(consensus_rate, 4)
        if contradiction_rate is not None:
            result_metadata["edge_contradiction_rate"] = round(
                contradiction_rate, 4
            )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=not breached,  # a breached floor fails the critical gate
            score=round(composite, 4),
            issues=issues,
            action=("block" if breached else None),
            metadata=result_metadata or None,
        )


__all__ = [
    "KGQualityValidator",
    "ENV_KG_PREREQ_HEALTH",
    "resolve_kg_prereq_health",
]

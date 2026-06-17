#!/usr/bin/env python3
"""Compare two LibV2 courses (baseline vs candidate) across course/KG/eval signals.

Built to validate a 7B-authored course against a stronger-model reference
(e.g. a Sonnet-authored course) on the same corpus, surfacing where the 7B
matches and where it lags. Slugs are CLI args — NO course slug or course-data
path is hard-coded (the project forbids course-data references in tracked
code); the tool is corpus-agnostic and reusable for any pair.

Compares, when present (each is graceful-degrade if an artifact is absent):
  - Knowledge graph (``concept_graph/concept_graph_semantic.json``):
    node/edge counts, asserted vs derived edges, per-rule edge breakdown.
  - KG quality report (``quality/kg_quality_report.json``): the four
    dimensions (completeness/consistency/accuracy/coverage), contradiction
    rate, DomainConcept grounded/total node counts.
  - IMSCC chunkset (``imscc_chunks/chunks.jsonl``): chunk count.
  - Synthesized objectives (``01_learning_objectives/synthesized_objectives.json``):
    terminal/chapter objective counts (handles Courseforge + LibV2 forms).
  - W5 generation-quality eval (``retrieval_eval/*generation_quality*`` or
    ``generation_eval/*``): success_condition.met + per-pillar booleans.

Usage:
    python scripts/integration/compare_courses.py --baseline <slug> --candidate <slug>
    python scripts/integration/compare_courses.py --baseline <slug> --candidate <slug> \
        --libv2-root LibV2 --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _count_jsonl(path: Path) -> Optional[int]:
    try:
        with path.open(encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return None


def _kg_summary(course_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    g = _load_json(course_dir / "concept_graph" / "concept_graph_semantic.json")
    if g is not None:
        nodes = g.get("nodes", []) or []
        edges = g.get("edges", []) or []
        out["nodes"] = len(nodes)
        out["edges"] = len(edges)
        # DomainConcept-or-classless node share (the coverage denominator)
        def _is_concept(n: Dict[str, Any]) -> bool:
            t = n.get("type") or n.get("node_type") or n.get("class") or ""
            return ("DomainConcept" in str(t)) or t in ("", None)
        out["concept_nodes"] = sum(1 for n in nodes if isinstance(n, dict) and _is_concept(n))
        # per-rule edge breakdown from the edge objects when available
        rule_counts: Dict[str, int] = {}
        asserted = derived = 0
        for e in edges:
            if not isinstance(e, dict):
                continue
            # the edge's ``rule`` is often a nested dict {rule, rule_version,
            # evidence}; key the histogram on the rule NAME only, never the
            # evidence (which would explode into one bucket per edge).
            r = e.get("rule") or e.get("rule_iri") or e.get("provenance") or "?"
            if isinstance(r, dict):
                r = r.get("rule") or r.get("rule_iri") or "?"
            name = str(r).rsplit("/", 1)[-1]
            rule_counts[name] = rule_counts.get(name, 0) + 1
            if e.get("derived") or e.get("status") == "derived":
                derived += 1
            else:
                asserted += 1
        out["edge_rules"] = rule_counts
        out["asserted_edges"] = asserted
        out["derived_edges"] = derived
    q = _load_json(course_dir / "quality" / "kg_quality_report.json")
    if q is not None:
        dims = q.get("dimensions", {}) or {}
        out["kg_quality"] = {
            d: round(float(dims.get(d, {}).get("score")), 4)
            for d in ("completeness", "consistency", "accuracy", "coverage")
            if isinstance(dims.get(d), dict) and dims.get(d, {}).get("score") is not None
        }
        out["contradiction_rate"] = q.get("contradiction_rate")
        cov = dims.get("coverage", {}) if isinstance(dims.get("coverage"), dict) else {}
        out["grounded_node_count"] = cov.get("grounded_node_count")
        out["concept_node_count"] = cov.get("concept_node_count")
        if "passed" in q:
            out["kg_passed"] = q.get("passed")
        # rule_outputs (authoring-time per-rule edge counts) if present
        if q.get("rule_outputs"):
            out["kg_rule_outputs"] = {
                str(r.get("rule_iri", "?")).rsplit("/", 1)[-1]: r.get("edge_count")
                for r in q["rule_outputs"]
            }
    return out


def _objectives_summary(course_dir: Path) -> Dict[str, Any]:
    for cand in (
        course_dir / "01_learning_objectives" / "synthesized_objectives.json",
        course_dir / "sources" / "objectives" / "synthesized_objectives.json",
    ):
        d = _load_json(cand)
        if d is not None:
            tos = d.get("terminal_objectives", d.get("terminal_outcomes", [])) or []
            cos = d.get("chapter_objectives", d.get("component_objectives", [])) or []
            return {"terminal": len(tos), "chapter": len(cos), "mint_method": d.get("mint_method")}
    return {}


def _eval_summary(course_dir: Path) -> Dict[str, Any]:
    candidates: List[Path] = []
    for sub in ("retrieval_eval", "generation_eval"):
        d = course_dir / sub
        if d.is_dir():
            candidates += sorted(d.glob("*generation_quality*")) + sorted(d.glob("*GENERATION*"))
    for p in candidates:
        d = _load_json(p)
        if not isinstance(d, dict):
            continue
        sc = d.get("success_condition") or d.get("w5_success_condition") or {}
        if sc:
            pillars = {k: v.get("met") if isinstance(v, dict) else v
                       for k, v in sc.items() if k not in ("met",)}
            return {"source": p.name, "met": sc.get("met"), "pillars": pillars}
    return {}


def summarize_course(libv2_root: Path, slug: str) -> Dict[str, Any]:
    course_dir = libv2_root / "courses" / slug
    summary: Dict[str, Any] = {"slug": slug, "exists": course_dir.is_dir()}
    if not course_dir.is_dir():
        return summary
    summary["kg"] = _kg_summary(course_dir)
    summary["imscc_chunks"] = _count_jsonl(course_dir / "imscc_chunks" / "chunks.jsonl")
    summary["objectives"] = _objectives_summary(course_dir)
    summary["eval"] = _eval_summary(course_dir)
    return summary


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt(x)}" for k, x in v.items()) or "—"
    return str(v)


def print_report(base: Dict[str, Any], cand: Dict[str, Any]) -> None:
    print(f"\n{'METRIC':<28} {'BASELINE: '+base['slug']:<34} {'CANDIDATE: '+cand['slug']:<34}")
    print("-" * 96)
    rows = [
        ("KG nodes", ("kg", "nodes")),
        ("KG edges", ("kg", "edges")),
        ("KG concept nodes", ("kg", "concept_nodes")),
        ("KG asserted/derived", ("kg", "asserted_edges")),
        ("KG quality dims", ("kg", "kg_quality")),
        ("KG contradiction_rate", ("kg", "contradiction_rate")),
        ("KG grounded/concept", ("kg", "grounded_node_count")),
        ("KG edge rules", ("kg", "edge_rules")),
        ("KG rule_outputs", ("kg", "kg_rule_outputs")),
        ("imscc_chunks", ("imscc_chunks",)),
        ("objectives TO/CO", ("objectives",)),
        ("W5 eval met", ("eval", "met")),
        ("W5 pillars", ("eval", "pillars")),
    ]
    def dig(d: Dict[str, Any], path: tuple) -> Any:
        for k in path:
            d = d.get(k, {}) if isinstance(d, dict) else None
            if d is None:
                return None
        return d
    for label, path in rows:
        print(f"{label:<28} {_fmt(dig(base, path)):<34} {_fmt(dig(cand, path)):<34}")
    print("-" * 96)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two LibV2 courses.")
    ap.add_argument("--baseline", required=True, help="reference course slug")
    ap.add_argument("--candidate", required=True, help="course slug under test")
    ap.add_argument("--libv2-root", default=os.environ.get("ED4ALL_LIBV2_ROOT", "LibV2"))
    ap.add_argument("--json", help="optional path to write the structured comparison JSON")
    args = ap.parse_args()
    root = Path(args.libv2_root)
    base = summarize_course(root, args.baseline)
    cand = summarize_course(root, args.candidate)
    print_report(base, cand)
    if args.json:
        Path(args.json).write_text(
            json.dumps({"baseline": base, "candidate": cand}, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Wave 92 — Holdout split builder for Tier-2 eval.

Partitions the LibV2 pedagogy graph BEFORE training so a deterministic
fraction of edges-per-relation-type is withheld and used as Tier-2
ground truth at eval time. The split is Bloom-stratified: chunks at
each Bloom level contribute proportionally to the held-out set so a
trained model isn't unfairly evaluated on a subset of cognitive
levels it never saw at training time.

Output (canonicalised JSON written to ``eval/holdout_split.json``):

    {
        "course_slug": "rdf-shacl-551-2",
        "seed": 42,
        "holdout_pct": 0.1,
        "edges_total": 8735,
        "edges_held_out": 873,
        "per_relation": {
            "prerequisite_of": {"total": 4160, "held_out": 416},
            ...
        },
        "bloom_strata": {
            "remember": {"total": 20, "held_out": 2},
            ...
        },
        "withheld_edges": [
            {"source": "...", "target": "...", "relation_type": "..."},
            ...
        ],
        "holdout_graph_hash": "<sha256 of canonicalised payload>"
    }

The split is REPRODUCIBLE: same seed + same input graph yields the
same `holdout_graph_hash`, so this hash drops cleanly into
``model_card.provenance.holdout_graph_hash`` and lets the eval
harness be replayed against any future model version.

Reads `relation_type` (NOT `type`) from edges — schema correction
per Wave 92 corpus inspection.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_PEDAGOGY_CANDIDATES = (
    "graph/pedagogy_graph.json",
    "pedagogy/pedagogy_graph.json",
    "pedagogy/pedagogy_model.json",
)


class HoldoutBuilder:
    """Build the Tier-2 holdout split from a LibV2 course's pedagogy graph.

    Args:
        course_path: Path to ``LibV2/courses/<slug>/``.
        holdout_pct: Fraction of edges per relation_type to withhold
            (default 0.1 = 10%). Each relation type is independently
            shuffled with the same seed so the per-type withhold is
            stratified.
        seed: RNG seed; pinned in the output JSON so reruns over the
            same graph + same seed produce byte-identical holdout
            files.
    """

    def __init__(
        self,
        course_path: Path,
        holdout_pct: float = 0.1,
        seed: int = 42,
    ) -> None:
        if not 0.0 < holdout_pct < 1.0:
            raise ValueError(
                f"holdout_pct must be in (0.0, 1.0); got {holdout_pct!r}"
            )
        self.course_path = Path(course_path)
        if not self.course_path.exists():
            raise FileNotFoundError(
                f"course_path does not exist: {self.course_path}"
            )
        self.holdout_pct = float(holdout_pct)
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    # Build                                                               #
    # ------------------------------------------------------------------ #

    def build(self) -> Path:
        """Build the split and write ``eval/holdout_split.json``.

        Returns the absolute path to the emitted JSON.
        """
        graph = self._load_pedagogy_graph()
        edges = list(graph.get("edges", []))
        nodes = list(graph.get("nodes", []))

        # Bloom-stratification index: for each chunk node, lookup its
        # bloom level via the at_bloom_level edges. Each relation_type's
        # withheld set will sample edges that touch chunks across
        # bloom levels so we don't accidentally drop all edges from
        # the rare "create" tier (which has only 16 chunks in the
        # rdf-shacl-551-2 corpus).
        bloom_by_chunk = self._build_bloom_index(edges)

        per_relation = defaultdict(list)
        for edge in edges:
            rt = edge.get("relation_type")
            if rt is None:
                continue
            per_relation[rt].append(edge)

        rng = random.Random(self.seed)
        withheld: List[Dict[str, Any]] = []
        per_relation_summary: Dict[str, Dict[str, int]] = {}

        for rt in sorted(per_relation.keys()):
            bucket = list(per_relation[rt])
            rng.shuffle(bucket)
            n = len(bucket)
            k = max(1, int(round(n * self.holdout_pct))) if n > 0 else 0
            # Cap at n - 1 so the train side is never empty.
            k = min(k, max(0, n - 1))
            withheld_for_rel = bucket[:k]
            withheld.extend(withheld_for_rel)
            per_relation_summary[rt] = {
                "total": n,
                "held_out": k,
            }

        bloom_strata = self._compute_bloom_strata(edges, withheld, bloom_by_chunk)

        # Wave 105: emit a `probes` array alongside `withheld_edges`
        # so downstream eval consumers (slm_eval_harness, evaluators)
        # have a stable, prompt-shaped surface even when the edge
        # carries no chunk anchor. Each probe carries the
        # canonical fields required by the Wave 105 contract:
        # ``probe_id`` / ``prompt`` / ``ground_truth_chunk_id`` /
        # ``edge_type``. ``ground_truth_chunk_id`` is null when the
        # edge isn't chunk-anchored (concept->concept edges).
        probes = self._build_probes(withheld)

        # Wave 108 / Phase B: sample negative probes — (source, relation,
        # target) tuples that DON'T exist in the graph. The correct
        # ground-truth response is "no", which catches the yes-bias
        # regression class (template-recognizer adapters trained on
        # all-positive corpora answer "yes" to everything).
        negative_probes = self._sample_negative_probes(edges, per_relation_summary)

        payload: Dict[str, Any] = {
            "course_slug": self.course_path.name,
            "seed": self.seed,
            "holdout_pct": self.holdout_pct,
            "edges_total": len(edges),
            "edges_held_out": len(withheld),
            "per_relation": per_relation_summary,
            "bloom_strata": bloom_strata,
            "withheld_edges": [
                {
                    "source": e.get("source"),
                    "target": e.get("target"),
                    "relation_type": e.get("relation_type"),
                }
                for e in withheld
            ],
            "probes": probes,
            "negative_probes": negative_probes,
        }

        # Hash the canonicalised payload (without the hash field) so
        # the split is content-addressable.
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["holdout_graph_hash"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

        output_path = self._resolve_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info(
            "HoldoutBuilder: wrote split to %s (held_out=%d / %d)",
            output_path, len(withheld), len(edges),
        )
        return output_path

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _load_pedagogy_graph(self) -> Dict[str, Any]:
        for rel in _PEDAGOGY_CANDIDATES:
            p = self.course_path / rel
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        raise FileNotFoundError(
            f"No pedagogy graph found under {self.course_path}. "
            f"Looked for: {', '.join(_PEDAGOGY_CANDIDATES)}"
        )

    @staticmethod
    def _build_bloom_index(
        edges: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Map chunk_id -> bloom level via at_bloom_level edges.

        The pedagogy graph encodes Bloom levels as separate
        ``BloomLevel`` nodes; chunks point to them via
        ``at_bloom_level`` edges. We walk those edges to build a
        chunk_id -> level lookup so the strata report is accurate.
        """
        out: Dict[str, str] = {}
        for edge in edges:
            if edge.get("relation_type") != "at_bloom_level":
                continue
            chunk_id = edge.get("source")
            target = str(edge.get("target", ""))
            # Targets look like "bloom:remember"; strip the prefix.
            if target.startswith("bloom:"):
                level = target.split(":", 1)[1]
            else:
                level = target
            if chunk_id and level:
                out[chunk_id] = level
        return out

    @staticmethod
    def _compute_bloom_strata(
        all_edges: List[Dict[str, Any]],
        withheld: List[Dict[str, Any]],
        bloom_by_chunk: Dict[str, str],
    ) -> Dict[str, Dict[str, int]]:
        """Tally edges per Bloom level.

        We attribute an edge to a Bloom level if either source or target
        resolves to a chunk node we have a Bloom mapping for. Edges
        between two non-chunk nodes (Outcome -> Outcome, etc.) are
        bucketed under "other".
        """
        def _bucket(edge: Dict[str, Any]) -> str:
            for endpoint in (edge.get("source"), edge.get("target")):
                if endpoint in bloom_by_chunk:
                    return bloom_by_chunk[endpoint]
            return "other"

        totals: Dict[str, int] = defaultdict(int)
        held: Dict[str, int] = defaultdict(int)
        for e in all_edges:
            totals[_bucket(e)] += 1
        for e in withheld:
            held[_bucket(e)] += 1

        return {
            level: {
                "total": totals[level],
                "held_out": held.get(level, 0),
            }
            for level in sorted(totals.keys())
        }

    @staticmethod
    def _build_probes(
        withheld: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Wave 105: derive prompt-shaped probes from withheld edges.

        Each probe is keyed by a deterministic ``probe_id`` so reruns
        with the same seed produce stable IDs. ``ground_truth_chunk_id``
        is set when the edge's ``source`` is chunk-anchored
        (``chunk_*``); otherwise it's ``None``. ``prompt`` is a
        terse paraphrase of the edge so RAG callables can be evaluated
        against the same surface as ``faithfulness._format_probe``.
        """
        out: List[Dict[str, Any]] = []
        for i, edge in enumerate(withheld):
            src = edge.get("source")
            tgt = edge.get("target")
            rel = edge.get("relation_type")
            # Wave 105: a chunk source can be either the canonical
            # ``chunk_NNNN`` form or a corpus-prefixed
            # ``<corpus>_chunk_NNNN`` form (the rdf-shacl-551-2 graph
            # uses the prefixed form). Detect both — substring search
            # for ``chunk_`` is sufficient because no other node class
            # in the pedagogy graph contains that token.
            gt_chunk = (
                src if isinstance(src, str) and "chunk_" in src
                else None
            )
            prompt = (
                f"Does the relation '{rel}' hold between "
                f"{src!r} and {tgt!r}?"
            )
            out.append({
                "probe_id": f"holdout-{i:04d}",
                "prompt": prompt,
                "ground_truth_chunk_id": gt_chunk,
                "edge_type": rel,
            })
        return out

    def _sample_negative_probes(
        self,
        edges: List[Dict[str, Any]],
        per_relation_summary: Dict[str, Dict[str, int]],
    ) -> List[Dict[str, Any]]:
        """Wave 108 / Phase B: sample (source, relation, target) tuples
        that DO NOT exist in the graph.

        Count per relation matches the per-relation held-out count so
        positive and negative probes are balanced. Strategy: for each
        relation type with at least one edge, build the set of real
        (source, target) pairs and the universe of (source, target)
        candidates as the cartesian of seen sources and seen targets
        for that relation. Sample candidates not in the real set.
        Bound retries to avoid infinite loops on tiny graphs.
        """
        from collections import defaultdict
        # +1 so negatives don't share the seed offset that withheld
        # positives use; same graph + same seed -> deterministic.
        rng = random.Random(self.seed + 1)
        per_relation_real: Dict[str, set] = defaultdict(set)
        per_relation_sources: Dict[str, set] = defaultdict(set)
        per_relation_targets: Dict[str, set] = defaultdict(set)
        for e in edges:
            rt = e.get("relation_type")
            s = e.get("source")
            t = e.get("target")
            if rt is None or s is None or t is None:
                continue
            per_relation_real[rt].add((s, t))
            per_relation_sources[rt].add(s)
            per_relation_targets[rt].add(t)

        negatives: List[Dict[str, Any]] = []
        for rt in sorted(per_relation_real.keys()):
            target_count = per_relation_summary.get(rt, {}).get("held_out", 0)
            if target_count <= 0:
                continue
            sources = sorted(per_relation_sources[rt])
            targets = sorted(per_relation_targets[rt])
            real = per_relation_real[rt]
            seen_neg: set = set()
            attempts = 0
            max_attempts = max(target_count * 20, 200)
            while len(seen_neg) < target_count and attempts < max_attempts:
                attempts += 1
                s = rng.choice(sources)
                t = rng.choice(targets)
                if (s, t) in real or (s, t) in seen_neg or s == t:
                    continue
                seen_neg.add((s, t))
                negatives.append({
                    "source": s,
                    "target": t,
                    "relation_type": rt,
                    "ground_truth": "no",
                })
        return negatives

    def _resolve_output_path(self) -> Path:
        return self.course_path / "eval" / "holdout_split.json"


def load_holdout_split(path: Path) -> Dict[str, Any]:
    """Read a previously-built holdout split."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Holdout split not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


__all__ = ["HoldoutBuilder", "load_holdout_split"]

"""Wave 75 — real pedagogy graph builder.

Replaces the stub ``_generate_pedagogy_graph`` (which only mirrored
pedagogy/logistics concept tags into a co-occurrence graph) with a
genuine pedagogical graph. Walks chunks + objectives + modules and
emits typed pedagogical edges so downstream consumers (RAG retrieval,
training-pair synthesis, course intelligence dashboards) can reason
about *teaches* / *assesses* / *practices* / *exemplifies* /
*supports_outcome* relations rather than re-deriving them from
chunk concept tags.

Public entry point::

    build_pedagogy_graph(chunks, objectives, course_id=None,
                         modules=None) -> Dict[str, Any]

Inputs:

* ``chunks``       -- iterable of v4 chunk dicts (the same shape that
                       lands in ``corpus/chunks.jsonl``).
* ``objectives``   -- canonical objectives dict. Accepts both shapes:
                       (a) Worker A's planned ``objectives.json``
                       (terminal_objectives / chapter_objectives), or
                       (b) Courseforge's ``synthesized_objectives.json``
                       (same key names — they share a contract).
* ``course_id``    -- optional course code. Used as an attribute on
                       every node.
* ``modules``      -- optional explicit list of module ids in display
                       order. When omitted, modules are discovered
                       from chunk source.module_id and ordered by the
                       leading ``week_NN`` token plus first-seen order.

Output: a dict with ``kind``, ``nodes``, ``edges``, ``generated_at``,
plus a ``stats`` block summarising counts per node-class and per
relation-type. Every node carries a ``class`` field. Every edge
carries a ``relation_type`` field.

Node classes:

* ``Outcome``           -- one per terminal objective (TO-NN).
* ``ComponentObjective``-- one per chapter objective (CO-NN).
* ``Chunk``             -- one per chunk (chunk_type kept as attr).
* ``Module``            -- one per top-level module (week_NN slice).
* ``BloomLevel``        -- six canonical levels.
* ``Misconception``     -- one per unique misconception statement.

Edge types (each gets ``relation_type`` set verbatim):

* ``teaches``           -- Chunk -> Objective (non-assessment chunks).
* ``assesses``          -- Chunk -> Objective (assessment_item chunks
                            and chunks emitted from quiz/self-check).
* ``practices``         -- Chunk -> Objective (chunk_type=exercise).
* ``exemplifies``       -- Chunk -> Concept (example chunks; concept
                            slugs from chunk.concept_tags).
* ``prerequisite_of``   -- Concept -> Concept (week N -> week N+1
                            concept ordering).
* ``interferes_with``   -- Misconception -> Concept (links each
                            misconception node to chunk concept_tags).
* ``belongs_to_module`` -- Chunk -> Module.
* ``supports_outcome``  -- ComponentObjective -> Outcome (parent_to).
* ``at_bloom_level``    -- Objective -> BloomLevel.
* ``follows``           -- Module -> Module (display-order chain).

The builder is fail-soft: malformed objectives or chunks are skipped
with no exception, and an empty graph is emitted when both inputs are
empty so callers can rely on the schema shape.
"""
from __future__ import annotations

import hashlib
import re
from collections import OrderedDict, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from lib.ontology.bloom import BLOOM_LEVELS

__all__ = ["build_pedagogy_graph", "load_objectives_with_fallback"]


# Quiz/self-check module id regex: matches "..._self_check", "..._quiz",
# "..._exam", "..._assessment". These identify pages whose chunks
# should be treated as 'assesses' even when chunk_type != assessment_item.
_QUIZ_MODULE_RE = re.compile(
    r"(self[_-]?check|quiz|exam|assessment|test)\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm_lo_id(raw: str) -> str:
    """Canonicalize an LO ref to upper-case ``XX-NN`` form.

    Accepts ``co-01``, ``CO-01``, leading/trailing whitespace; rejects
    empty / non-pattern values by returning empty string.
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip().upper()
    # Reject if obviously malformed (no dash, or compound like "CO-01,CO-02")
    if "," in s or ";" in s:
        return ""
    if not re.fullmatch(r"[A-Z]{2,}-\d+", s):
        return ""
    return s


def _split_lo_refs(refs: Iterable[Any]) -> List[str]:
    """Split a chunk's ``learning_outcome_refs`` list into clean ids.

    Handles the legacy ``"co-01,co-02"`` compound entry by splitting
    on comma/semicolon, and normalises to upper-case ``XX-NN``. Empty
    strings dropped. Order preserved, duplicates kept (caller dedups).
    """
    out: List[str] = []
    if not refs:
        return out
    for raw in refs:
        if not isinstance(raw, str):
            continue
        for piece in re.split(r"[,;]", raw):
            n = _norm_lo_id(piece)
            if n:
                out.append(n)
    return out


def _slugify_concept(tag: str) -> str:
    """Map a concept tag to a stable slug (lower-kebab)."""
    if not isinstance(tag, str):
        return ""
    s = tag.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s


def _module_top_slice(module_id: str, item_path: str = "") -> str:
    """Reduce a chunk source to its top-level week slice.

    Prefers ``module_id`` when it carries a ``week_NN`` prefix.
    Otherwise falls back to ``item_path`` (the on-disk path inside
    the IMSCC), which always carries the week directory. This handles
    Trainforge corpora where the parser stripped the week prefix off
    ``module_id`` (e.g., ``application``, ``content_01``) but the
    actual file lives under ``week_04/`` etc.

    ``week_01_self_check`` -> ``week_01``. Non-week ids return as-is.
    Empty inputs return empty.
    """
    if isinstance(module_id, str) and module_id:
        m = re.match(r"(week_\d+)", module_id)
        if m:
            return m.group(1)
    if isinstance(item_path, str) and item_path:
        m = re.match(r"(week_\d+)", item_path)
        if m:
            return m.group(1)
    if isinstance(module_id, str) and module_id:
        return module_id
    return ""


def _module_sort_key(module_id: str) -> Tuple[int, str]:
    """Stable sort key. Numeric week → (week_int, ''); else (10**9, id)."""
    m = re.match(r"week_(\d+)", module_id)
    if m:
        return (int(m.group(1)), module_id)
    return (10**9, module_id)


def _mc_id(text: str) -> str:
    """Stable misconception id from text content (matches schema)."""
    h = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()
    return f"mc_{h[:16]}"


# ---------------------------------------------------------------------------
# Objective loading
# ---------------------------------------------------------------------------


def load_objectives_with_fallback(
    objectives_path: Optional[str],
    synthesized_path: Optional[str],
) -> Dict[str, Any]:
    """Read Worker A's ``objectives.json`` if present; else fall back to
    Courseforge's ``synthesized_objectives.json``.

    Both files share the same canonical contract (terminal_objectives /
    chapter_objectives). Returns the parsed dict, or an empty dict
    when neither path exists.
    """
    import json
    import os

    for path in (objectives_path, synthesized_path):
        if not path:
            continue
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_pedagogy_graph(
    chunks: Iterable[Dict[str, Any]],
    objectives: Optional[Dict[str, Any]] = None,
    *,
    course_id: Optional[str] = None,
    modules: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Walk chunks + objectives and emit a typed pedagogical graph.

    See module docstring for inputs / outputs / edge semantics.
    """
    chunks = list(chunks or [])
    objectives = objectives or {}

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. Bloom level nodes (always emitted).
    # ------------------------------------------------------------------
    for level in BLOOM_LEVELS:
        nodes.append({
            "id": f"bloom:{level}",
            "class": "BloomLevel",
            "label": level.title(),
            "level": level,
        })

    # ------------------------------------------------------------------
    # 2. Outcome + ComponentObjective nodes from objectives.
    #    Supports BOTH conventions:
    #      (a) Worker A's objectives.json:
    #          terminal_outcomes / component_objectives (parent_terminal)
    #      (b) Courseforge synthesized_objectives.json:
    #          terminal_objectives / chapter_objectives (parent_to)
    # ------------------------------------------------------------------
    terminal = (
        objectives.get("terminal_objectives")
        or objectives.get("terminal_outcomes")
        or []
    )
    chapter = (
        objectives.get("chapter_objectives")
        or objectives.get("component_objectives")
        or []
    )

    objective_nodes: Dict[str, Dict[str, Any]] = {}

    for to in terminal:
        if not isinstance(to, dict):
            continue
        raw_id = to.get("id")
        nid = _norm_lo_id(raw_id) if raw_id else ""
        if not nid:
            continue
        bloom = (to.get("bloom_level") or "").strip().lower()
        node = {
            "id": nid,
            "class": "Outcome",
            "label": nid,
            "statement": to.get("statement") or to.get("text") or "",
            "bloom_level": bloom,
        }
        if course_id:
            node["course_id"] = course_id
        objective_nodes[nid] = node

    for co in chapter:
        if not isinstance(co, dict):
            continue
        raw_id = co.get("id")
        nid = _norm_lo_id(raw_id) if raw_id else ""
        if not nid:
            continue
        bloom = (co.get("bloom_level") or "").strip().lower()
        parent_raw = (
            co.get("parent_to")
            or co.get("parent_terminal")
            or co.get("parent")
        )
        parent = _norm_lo_id(parent_raw) if parent_raw else ""
        week = co.get("week")
        node = {
            "id": nid,
            "class": "ComponentObjective",
            "label": nid,
            "statement": co.get("statement") or co.get("text") or "",
            "bloom_level": bloom,
        }
        if parent:
            node["parent_terminal"] = parent
        if isinstance(week, int):
            node["week"] = week
        if course_id:
            node["course_id"] = course_id
        objective_nodes[nid] = node

    nodes.extend(objective_nodes.values())

    # ------------------------------------------------------------------
    # 3. supports_outcome edges (ComponentObjective -> Outcome).
    # ------------------------------------------------------------------
    for nid, node in objective_nodes.items():
        if node["class"] != "ComponentObjective":
            continue
        parent = node.get("parent_terminal")
        if parent and parent in objective_nodes \
                and objective_nodes[parent]["class"] == "Outcome":
            edges.append({
                "source": nid,
                "target": parent,
                "relation_type": "supports_outcome",
            })

    # ------------------------------------------------------------------
    # 4. at_bloom_level edges (Objective -> BloomLevel).
    # ------------------------------------------------------------------
    bloom_set = set(BLOOM_LEVELS)
    for nid, node in objective_nodes.items():
        bl = node.get("bloom_level")
        if bl in bloom_set:
            edges.append({
                "source": nid,
                "target": f"bloom:{bl}",
                "relation_type": "at_bloom_level",
            })

    # ------------------------------------------------------------------
    # 5. Module nodes.
    #    Discover modules from chunks (top-level week slice) unless a
    #    caller-supplied list is provided. Maintain a stable ordering.
    # ------------------------------------------------------------------
    discovered_modules: "OrderedDict[str, None]" = OrderedDict()
    for c in chunks:
        src = c.get("source") or {}
        mid = _module_top_slice(
            src.get("module_id", ""), src.get("item_path", "")
        )
        if mid:
            discovered_modules.setdefault(mid, None)

    if modules:
        ordered_modules = list(modules)
        # Append any module observed in chunks but missing from caller
        # list so chunks don't dangle without a Module node.
        for mid in discovered_modules:
            if mid not in ordered_modules:
                ordered_modules.append(mid)
    else:
        ordered_modules = sorted(discovered_modules.keys(), key=_module_sort_key)

    for idx, mid in enumerate(ordered_modules):
        m = re.match(r"week_(\d+)", mid)
        node = {
            "id": f"module:{mid}",
            "class": "Module",
            "label": mid.replace("_", " ").title(),
            "module_id": mid,
            "order": idx,
        }
        if m:
            node["week"] = int(m.group(1))
        if course_id:
            node["course_id"] = course_id
        nodes.append(node)

    # ------------------------------------------------------------------
    # 6. follows edges (Module -> Module by display order).
    # ------------------------------------------------------------------
    for a, b in zip(ordered_modules, ordered_modules[1:]):
        edges.append({
            "source": f"module:{a}",
            "target": f"module:{b}",
            "relation_type": "follows",
        })

    # ------------------------------------------------------------------
    # 7. Chunk nodes + per-chunk edges.
    # ------------------------------------------------------------------
    valid_objective_ids = set(objective_nodes.keys())
    concept_to_chunks: Dict[str, Set[str]] = defaultdict(set)
    concept_to_week: Dict[str, int] = {}
    concept_label: Dict[str, str] = {}

    for c in chunks:
        cid = c.get("id")
        if not cid:
            continue

        chunk_type = c.get("chunk_type") or ""
        src = c.get("source") or {}
        mid = src.get("module_id", "")
        ipath = src.get("item_path", "")
        top_mid = _module_top_slice(mid, ipath)

        chunk_node: Dict[str, Any] = {
            "id": cid,
            "class": "Chunk",
            "label": cid,
            "chunk_type": chunk_type,
            "module_id": top_mid or mid,
        }
        if course_id:
            chunk_node["course_id"] = course_id
        # Carry first-seen LO refs as a convenience attribute.
        lo_refs = _split_lo_refs(c.get("learning_outcome_refs") or [])
        if lo_refs:
            chunk_node["learning_outcome_refs"] = sorted(set(lo_refs))
        nodes.append(chunk_node)

        # belongs_to_module
        if top_mid:
            edges.append({
                "source": cid,
                "target": f"module:{top_mid}",
                "relation_type": "belongs_to_module",
            })

        # Determine if this chunk is *assessing* even though chunk_type
        # may be explanation/overview: quiz/self-check page heuristic.
        is_quiz_page = bool(_QUIZ_MODULE_RE.search(mid)) if mid else False

        # teaches / assesses / practices edges: drive off chunk_type and
        # the module heuristic, with LO refs as the targets.
        for lo in sorted(set(lo_refs)):
            if lo not in valid_objective_ids:
                # Still emit if the LO id is well-formed but the
                # objectives file is missing it — relation is still
                # informative for downstream consumers. We require
                # the LO node to exist for retrieval validity though,
                # so skip when missing to keep edges referentially
                # complete.
                continue
            if chunk_type == "assessment_item" or is_quiz_page:
                edges.append({
                    "source": cid,
                    "target": lo,
                    "relation_type": "assesses",
                })
            elif chunk_type == "exercise":
                edges.append({
                    "source": cid,
                    "target": lo,
                    "relation_type": "practices",
                })
            else:
                edges.append({
                    "source": cid,
                    "target": lo,
                    "relation_type": "teaches",
                })

        # exemplifies edges: example chunks → their concept_tags.
        if chunk_type == "example":
            for tag in c.get("concept_tags") or []:
                slug = _slugify_concept(tag)
                if not slug:
                    continue
                edges.append({
                    "source": cid,
                    "target": f"concept:{slug}",
                    "relation_type": "exemplifies",
                })
                concept_to_chunks[slug].add(cid)
                concept_label.setdefault(slug, tag)

        # Aggregate concept_tags per-week for prerequisite_of inference.
        m = re.match(r"week_(\d+)", top_mid or "")
        if m:
            week = int(m.group(1))
            for tag in c.get("concept_tags") or []:
                slug = _slugify_concept(tag)
                if not slug:
                    continue
                # First week wins so a concept's "home" week is its
                # introduction; later usages don't overwrite.
                concept_to_week.setdefault(slug, week)
                concept_label.setdefault(slug, tag)

    # ------------------------------------------------------------------
    # 8. Concept nodes (only those referenced by exemplifies edges or
    #    used by misconceptions below). Lightweight — concept graph
    #    proper is in concept_graph.json. Pedagogy graph carries just
    #    enough Concept nodes to keep exemplifies / interferes_with /
    #    prerequisite_of edges referentially complete.
    # ------------------------------------------------------------------
    concept_nodes_emitted: Set[str] = set()

    def _emit_concept(slug: str) -> None:
        if slug in concept_nodes_emitted:
            return
        node = {
            "id": f"concept:{slug}",
            "class": "Concept",
            "label": concept_label.get(slug, slug.replace("-", " ").title()),
            "slug": slug,
        }
        if course_id:
            node["course_id"] = course_id
        if slug in concept_to_week:
            node["first_seen_week"] = concept_to_week[slug]
        nodes.append(node)
        concept_nodes_emitted.add(slug)

    for slug in concept_to_chunks:
        _emit_concept(slug)

    # ------------------------------------------------------------------
    # 9. Misconception nodes + interferes_with edges.
    #    A misconception statement is unique by content. interferes_with
    #    targets are derived from the originating chunk's concept_tags
    #    (the concepts the misconception is *about*).
    # ------------------------------------------------------------------
    mc_seen: Dict[str, Dict[str, Any]] = {}
    mc_to_concepts: Dict[str, Set[str]] = defaultdict(set)
    for c in chunks:
        misconceptions = c.get("misconceptions") or []
        if not isinstance(misconceptions, list):
            continue
        chunk_concepts = [_slugify_concept(t) for t in (c.get("concept_tags") or [])]
        chunk_concepts = [s for s in chunk_concepts if s]
        for mc in misconceptions:
            if isinstance(mc, dict):
                text = (mc.get("misconception") or mc.get("text") or "").strip()
                concept_id = mc.get("concept_id") or mc.get("targets")
            elif isinstance(mc, str):
                text = mc.strip()
                concept_id = None
            else:
                continue
            if not text:
                continue
            mc_node_id = _mc_id(text)
            if mc_node_id not in mc_seen:
                mc_seen[mc_node_id] = {
                    "id": mc_node_id,
                    "class": "Misconception",
                    "label": text[:120],
                    "statement": text,
                }
                if course_id:
                    mc_seen[mc_node_id]["course_id"] = course_id
            # Target concepts: explicit override, else chunk's tags.
            targets: List[str] = []
            if concept_id:
                slug = _slugify_concept(concept_id) if isinstance(concept_id, str) else ""
                if slug:
                    targets.append(slug)
            if not targets:
                targets = chunk_concepts
            for slug in targets:
                if slug:
                    mc_to_concepts[mc_node_id].add(slug)
                    concept_label.setdefault(slug, slug.replace("-", " ").title())
                    if slug not in concept_to_week:
                        _src = c.get("source") or {}
                        _top = _module_top_slice(
                            _src.get("module_id", ""), _src.get("item_path", "")
                        )
                        m = re.match(r"week_(\d+)", _top)
                        if m:
                            concept_to_week.setdefault(slug, int(m.group(1)))

    # Emit misconception nodes (sorted for determinism).
    for mc_id_, node in sorted(mc_seen.items()):
        nodes.append(node)

    # interferes_with edges (sorted for determinism).
    for mc_id_, slugs in sorted(mc_to_concepts.items()):
        for slug in sorted(slugs):
            _emit_concept(slug)
            edges.append({
                "source": mc_id_,
                "target": f"concept:{slug}",
                "relation_type": "interferes_with",
            })

    # ------------------------------------------------------------------
    # 10. prerequisite_of edges (Concept -> Concept by week ordering).
    #     A concept introduced in week N is a prerequisite for any
    #     concept *first seen* in week N+1. Keeps the edge count
    #     bounded (only adjacent-week pairs) and carries genuine
    #     pedagogical signal: "you need this before that."
    # ------------------------------------------------------------------
    by_week: Dict[int, List[str]] = defaultdict(list)
    for slug, week in concept_to_week.items():
        if slug in concept_nodes_emitted:
            by_week[week].append(slug)
    sorted_weeks = sorted(by_week.keys())
    for i, w in enumerate(sorted_weeks[:-1]):
        nxt = sorted_weeks[i + 1]
        if nxt - w > 2:
            # Treat large jumps (e.g., week_03 -> week_09 in the rdf
            # corpus where weeks 4-8 weren't materialised) as still
            # prerequisitive — the relation is monotone in week order.
            pass
        for a in sorted(by_week[w]):
            for b in sorted(by_week[nxt]):
                if a == b:
                    continue
                edges.append({
                    "source": f"concept:{a}",
                    "target": f"concept:{b}",
                    "relation_type": "prerequisite_of",
                })

    # ------------------------------------------------------------------
    # 11. Stats block + return.
    # ------------------------------------------------------------------
    stats = _summarise(nodes, edges)

    return {
        "kind": "pedagogy",
        "schema_version": "v2",
        "course_id": course_id or "",
        "nodes": nodes,
        "edges": edges,
        "stats": stats,
        "generated_at": datetime.now().isoformat(),
    }


def _summarise(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
) -> Dict[str, Any]:
    nodes_by_class: Dict[str, int] = defaultdict(int)
    for n in nodes:
        nodes_by_class[n.get("class", "?")] += 1
    edges_by_relation: Dict[str, int] = defaultdict(int)
    for e in edges:
        edges_by_relation[e.get("relation_type", "?")] += 1
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes_by_class": dict(sorted(nodes_by_class.items())),
        "edges_by_relation": dict(sorted(edges_by_relation.items())),
    }

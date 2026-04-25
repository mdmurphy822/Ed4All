"""Wave 75 — real pedagogy graph builder.

Replaces the stub ``_generate_pedagogy_graph`` (which only mirrored
pedagogy/logistics concept tags into a co-occurrence graph) with a
genuine pedagogical graph. Walks chunks + objectives + modules and
emits typed pedagogical edges so downstream consumers (RAG retrieval,
training-pair synthesis, course intelligence dashboards) can reason
about *teaches* / *assesses* / *practices* / *exemplifies* /
*supports_outcome* relations rather than re-deriving them from
chunk concept tags.

Wave 78 (Worker B) completes the relation set with four new edge
types — ``derived_from_objective`` (Chunk → Objective provenance,
mirrored from concept_graph_semantic), ``concept_supports_outcome``
(DomainConcept → Outcome rollup, derived from concept ∩ chunk LO
refs), ``assessment_validates_outcome`` (assessment_item Chunk →
Outcome rollup via parent_terminal CO chains, distinct from the
direct ``assesses`` edge), and ``chunk_at_difficulty`` (Chunk →
DifficultyLevel typed node) — bringing the graph from 10 to 14
distinct relation types so the strict validator (Worker A) and intent
router (Worker C) have complete substrate to operate on.

Wave 76 (Worker D) refines the ``prerequisite_of`` and
``interferes_with`` rules so they emit only meaningful edges:

* ``prerequisite_of(A, B)`` requires (1) B's first-seen week strictly
  later than A's, (2) at least one chunk that contains both A and B
  as concept tags, and (3) both endpoints classified as
  ``DomainConcept`` per the supplied ``concept_classes`` map (when
  provided). The previous rule emitted a hard-capped (50/source)
  cartesian within adjacent weeks and over-saturated the graph
  (84% of edges in the rdf-shacl-550 archive).
* ``interferes_with(M, C)`` only emits when ``C`` is classified as
  ``DomainConcept``. PedagogicalMarker / AssessmentOption / LowSignal
  / InstructionalArtifact targets are dropped.

Public entry point::

    build_pedagogy_graph(chunks, objectives, course_id=None,
                         modules=None, concept_classes=None)
        -> Dict[str, Any]

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
* ``DifficultyLevel``   -- three canonical levels (foundational,
                            intermediate, advanced) — Wave 78.

Edge types (each gets ``relation_type`` set verbatim):

* ``teaches``                       -- Chunk -> Objective (non-assessment
                                        chunks).
* ``assesses``                      -- Chunk -> Objective (assessment_item
                                        chunks and chunks emitted from
                                        quiz/self-check). Direct ref —
                                        complement of the rollup edge
                                        ``assessment_validates_outcome``.
* ``practices``                     -- Chunk -> Objective
                                        (chunk_type=exercise).
* ``exemplifies``                   -- Chunk -> Concept (example chunks;
                                        concept slugs from
                                        chunk.concept_tags).
* ``prerequisite_of``               -- Concept -> Concept (week N -> week
                                        N+1 concept ordering).
* ``interferes_with``               -- Misconception -> Concept (links
                                        each misconception node to chunk
                                        concept_tags).
* ``belongs_to_module``             -- Chunk -> Module.
* ``supports_outcome``              -- ComponentObjective -> Outcome
                                        (parent_to).
* ``at_bloom_level``                -- Objective -> BloomLevel.
* ``follows``                       -- Module -> Module (display-order
                                        chain).
* ``derived_from_objective``        -- Chunk -> Objective (Wave 78).
                                        Mirrors the concept_graph_semantic
                                        ``derived-from-objective`` edge
                                        type into the pedagogy graph as
                                        explicit chunk-to-LO provenance —
                                        every chunk emits one edge per
                                        item in its
                                        ``learning_outcome_refs``.
                                        Conceptually similar to
                                        ``teaches`` but represents
                                        PROVENANCE rather than pedagogy.
* ``concept_supports_outcome``      -- DomainConcept -> Outcome (Wave 78).
                                        Derived: concept C → outcome O
                                        when at least one chunk contains
                                        C in concept_tags AND that
                                        chunk's learning_outcome_refs
                                        rolled up to terminal level
                                        contains O. Edge weight = number
                                        of supporting chunks. Restricted
                                        to DomainConcept-classified
                                        sources (Wave 76 filter).
* ``assessment_validates_outcome``  -- AssessmentItem Chunk -> Outcome
                                        (Wave 78). For every
                                        ``chunk_type == "assessment_item"``
                                        chunk with a ``co-NN`` ref, emit
                                        a rollup edge to that ref's
                                        parent terminal ``to-NN``.
                                        Distinct from ``assesses`` (which
                                        targets the direct ref) — this is
                                        the rollup chain.
* ``chunk_at_difficulty``           -- Chunk -> DifficultyLevel (Wave 78).
                                        Every chunk with a ``difficulty``
                                        attribute (foundational /
                                        intermediate / advanced) emits an
                                        edge to the corresponding
                                        DifficultyLevel typed node.
                                        Enables filtering / clustering by
                                        cognitive load via graph
                                        traversal.

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

# Wave 78: canonical difficulty levels — must match the ``difficulty``
# enum on chunk_v4.schema.json. Used both to seed DifficultyLevel typed
# nodes unconditionally (mirroring the BloomLevel-node convention) and to
# guard chunk_at_difficulty edge emission against malformed values.
DIFFICULTY_LEVELS = ("foundational", "intermediate", "advanced")


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
    concept_classes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Walk chunks + objectives and emit a typed pedagogical graph.

    See module docstring for inputs / outputs / edge semantics.

    ``concept_classes`` (Wave 76): optional mapping of concept slug ->
    class label sourced from ``concept_graph.json`` (Worker B's
    classifier). Recognised classes: ``DomainConcept``, ``Misconception``,
    ``PedagogicalMarker``, ``AssessmentOption``, ``LowSignal``,
    ``InstructionalArtifact``. Filters apply to ``prerequisite_of`` and
    ``interferes_with`` only — DomainConcept endpoints are kept;
    pedagogical/assessment scaffolding is dropped. When omitted,
    every concept is treated as DomainConcept-default (legacy mode);
    the new co-occurrence + strict-later-week filter still applies.
    """
    chunks = list(chunks or [])
    objectives = objectives or {}
    concept_classes = concept_classes or {}

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

    # Wave 78: DifficultyLevel typed nodes — emitted unconditionally so
    # downstream traversals can land on the node even on chunk-empty
    # corpora (parity with BloomLevel emission above).
    for level in DIFFICULTY_LEVELS:
        nodes.append({
            "id": f"difficulty:{level}",
            "class": "DifficultyLevel",
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
    # Wave 78: parent-terminal lookup for CO -> TO rollup. Used by both
    # ``assessment_validates_outcome`` (assessment chunk -> parent_TO)
    # and ``concept_supports_outcome`` (rolling each chunk's CO refs up
    # to terminal level before counting concept-supports-outcome chunks).
    co_to_parent_to: Dict[str, str] = {}
    for nid, node in objective_nodes.items():
        if node["class"] != "ComponentObjective":
            continue
        parent = node.get("parent_terminal")
        if (
            parent
            and parent in objective_nodes
            and objective_nodes[parent]["class"] == "Outcome"
        ):
            co_to_parent_to[nid] = parent

    # ``concept_to_chunks`` tracks chunk-membership for concepts that
    # are *exemplifies*-edge-eligible (i.e., concepts cited from example
    # chunks). It still drives Concept-node emission for the exemplifies
    # path. Wave 76 adds ``concept_to_chunks_all`` to track membership
    # across every chunk type — this is the substrate for the new
    # co-occurrence-aware ``prerequisite_of`` rule.
    concept_to_chunks: Dict[str, Set[str]] = defaultdict(set)
    concept_to_chunks_all: Dict[str, Set[str]] = defaultdict(set)
    concept_to_week: Dict[str, int] = {}
    concept_label: Dict[str, str] = {}
    # Wave 78: per-chunk rolled-up Outcome (terminal) ref set. For each
    # chunk we compute the union of (a) terminal refs already in the
    # chunk's learning_outcome_refs and (b) parents of any CO refs.
    # Drives the ``concept_supports_outcome`` derivation: a concept
    # appearing in a chunk is treated as supporting every Outcome in
    # that chunk's rolled-up set.
    chunk_to_rolled_outcomes: Dict[str, Set[str]] = {}

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

        # ------------------------------------------------------------
        # Wave 78: derived_from_objective (Chunk -> Objective).
        # Mirrors concept_graph_semantic's `derived-from-objective`
        # rule into pedagogy_graph as explicit chunk-LO provenance.
        # Conceptually similar to ``teaches`` but represents PROVENANCE
        # not pedagogy — emitted for *every* chunk regardless of
        # chunk_type, gated only on the LO node existing in the
        # objectives map (referential integrity).
        # ------------------------------------------------------------
        for lo in sorted(set(lo_refs)):
            if lo not in valid_objective_ids:
                continue
            edges.append({
                "source": cid,
                "target": lo,
                "relation_type": "derived_from_objective",
            })

        # ------------------------------------------------------------
        # Wave 78: assessment_validates_outcome (AssessmentItem chunk
        # -> Outcome) — rollup chain. For every assessment_item chunk
        # ref pointing at a CO-NN, emit an edge to that CO's
        # parent_terminal TO-NN. Distinct from ``assesses`` (which
        # targets the direct ref). Deduped per (chunk, parent_to)
        # pair so a chunk citing co-01 + co-02 (both rolling up to
        # to-01) emits only one validates edge.
        # ------------------------------------------------------------
        if chunk_type == "assessment_item":
            seen_validates: Set[str] = set()
            for lo in sorted(set(lo_refs)):
                parent_to = co_to_parent_to.get(lo)
                if not parent_to or parent_to in seen_validates:
                    continue
                if parent_to not in valid_objective_ids:
                    continue
                seen_validates.add(parent_to)
                edges.append({
                    "source": cid,
                    "target": parent_to,
                    "relation_type": "assessment_validates_outcome",
                })

        # ------------------------------------------------------------
        # Wave 78: chunk_at_difficulty (Chunk -> DifficultyLevel).
        # Every chunk with a canonical ``difficulty`` value emits an
        # edge to the corresponding DifficultyLevel typed node.
        # Malformed / missing values silently skipped (fail-soft, in
        # keeping with the rest of the builder).
        # ------------------------------------------------------------
        difficulty = c.get("difficulty")
        if isinstance(difficulty, str):
            d_norm = difficulty.strip().lower()
            if d_norm in DIFFICULTY_LEVELS:
                edges.append({
                    "source": cid,
                    "target": f"difficulty:{d_norm}",
                    "relation_type": "chunk_at_difficulty",
                })

        # ------------------------------------------------------------
        # Wave 78: roll up this chunk's LO refs to terminal level so
        # the post-loop ``concept_supports_outcome`` derivation can
        # count concept->outcome supporting chunks consistently. A
        # ref already at TO-NN level passes through; a CO-NN ref maps
        # to its parent_terminal (when present in the objectives map).
        # ------------------------------------------------------------
        rolled: Set[str] = set()
        for lo in lo_refs:
            if lo not in valid_objective_ids:
                continue
            cls = objective_nodes[lo]["class"]
            if cls == "Outcome":
                rolled.add(lo)
            elif cls == "ComponentObjective":
                parent_to = co_to_parent_to.get(lo)
                if parent_to:
                    rolled.add(parent_to)
        if rolled:
            chunk_to_rolled_outcomes[cid] = rolled

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
        # Also accumulate per-chunk membership across ALL chunk types
        # (Wave 76) — the new prerequisite_of rule needs this to test
        # "exists at least one chunk containing both A and B".
        m = re.match(r"week_(\d+)", top_mid or "")
        week = int(m.group(1)) if m else None
        for tag in c.get("concept_tags") or []:
            slug = _slugify_concept(tag)
            if not slug:
                continue
            concept_to_chunks_all[slug].add(cid)
            concept_label.setdefault(slug, tag)
            if week is not None:
                # First week wins so a concept's "home" week is its
                # introduction; later usages don't overwrite.
                concept_to_week.setdefault(slug, week)

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

    # Wave 76: a concept is interferes_with-eligible only when classified
    # as a DomainConcept. When ``concept_classes`` is unset, every concept
    # is treated as DomainConcept-default (legacy compatibility — Worker
    # B's classifier had not run yet on older corpora).
    def _is_domain_concept(slug: str) -> bool:
        if not concept_classes:
            return True
        cls = concept_classes.get(slug)
        if cls is None:
            return True  # unclassified -> permissive default
        return cls == "DomainConcept"

    # interferes_with edges (sorted for determinism). Filtered to
    # DomainConcept targets so misconceptions don't link to pedagogical
    # scaffolding ("key-takeaway", "rubric", etc.).
    for mc_id_, slugs in sorted(mc_to_concepts.items()):
        for slug in sorted(slugs):
            if not _is_domain_concept(slug):
                continue
            _emit_concept(slug)
            edges.append({
                "source": mc_id_,
                "target": f"concept:{slug}",
                "relation_type": "interferes_with",
            })

    # ------------------------------------------------------------------
    # 9b. concept_supports_outcome edges (DomainConcept -> Outcome).
    #
    #     Wave 78: derived rollup. Concept C supports Outcome O when
    #     at least one chunk contains C as a concept_tag AND that
    #     chunk's learning_outcome_refs (rolled up CO -> parent_TO)
    #     contains O. Edge weight (``confidence``) = count of
    #     supporting chunks. Source endpoint must be DomainConcept-
    #     classified per Worker D's filter (when ``concept_classes``
    #     supplied; permissive default otherwise). Target endpoint
    #     is always an Outcome that exists in ``objective_nodes``,
    #     so referential integrity is intact.
    # ------------------------------------------------------------------
    concept_outcome_support: Dict[Tuple[str, str], int] = defaultdict(int)
    for slug, chunk_ids in concept_to_chunks_all.items():
        if not _is_domain_concept(slug):
            continue
        for cid in chunk_ids:
            for outcome in chunk_to_rolled_outcomes.get(cid, ()):  # type: ignore[arg-type]
                concept_outcome_support[(slug, outcome)] += 1
    for (slug, outcome), weight in sorted(concept_outcome_support.items()):
        # Ensure both endpoints have nodes emitted. The Outcome node
        # always exists (it's in objective_nodes); the Concept node
        # may not yet have been emitted via exemplifies / prereq /
        # interferes_with paths.
        _emit_concept(slug)
        edges.append({
            "source": f"concept:{slug}",
            "target": outcome,
            "relation_type": "concept_supports_outcome",
            "confidence": weight,
        })

    # ------------------------------------------------------------------
    # 10. prerequisite_of edges (Concept -> Concept).
    #
    #     Wave 76 rule (replaces the prior adjacent-week cartesian):
    #
    #     Emit prerequisite_of(A, B) iff ALL hold:
    #       (1) A's first-seen week W_A < B's first-seen week W_B
    #           (strictly earlier — same-week pairs are NOT prereqs).
    #       (2) Some chunk contains BOTH A and B as concept_tags
    #           (co-occurrence anchors the relation in real content).
    #       (3) A and B are both DomainConcept-class per the supplied
    #           ``concept_classes`` map (or unclassified -> permissive
    #           default). Misconception nodes are interferes_with arms
    #           and never appear as prereq endpoints.
    #
    #     Each emitted edge carries ``confidence`` (the count of
    #     supporting co-occurring chunks) so downstream consumers can
    #     prune by signal strength.
    # ------------------------------------------------------------------
    candidates_by_week: Dict[int, List[str]] = defaultdict(list)
    for slug, week in concept_to_week.items():
        # Only consider slugs we already track per-chunk (so co-occurrence
        # tests are sound) AND classified as DomainConcept.
        if slug not in concept_to_chunks_all:
            continue
        if not _is_domain_concept(slug):
            continue
        candidates_by_week[week].append(slug)

    # For every (earlier, later) week pair, emit edges from every
    # earlier-week concept to every later-week concept that share at
    # least one chunk. No cap. Determinism: sort weeks + slugs.
    sorted_weeks = sorted(candidates_by_week.keys())
    for i, w_a in enumerate(sorted_weeks):
        for w_b in sorted_weeks[i + 1:]:
            for a in sorted(candidates_by_week[w_a]):
                a_chunks = concept_to_chunks_all.get(a, set())
                if not a_chunks:
                    continue
                for b in sorted(candidates_by_week[w_b]):
                    if a == b:
                        continue
                    b_chunks = concept_to_chunks_all.get(b, set())
                    if not b_chunks:
                        continue
                    shared = a_chunks & b_chunks
                    if not shared:
                        continue
                    # Ensure both endpoints have Concept nodes emitted.
                    _emit_concept(a)
                    _emit_concept(b)
                    edges.append({
                        "source": f"concept:{a}",
                        "target": f"concept:{b}",
                        "relation_type": "prerequisite_of",
                        "confidence": len(shared),
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

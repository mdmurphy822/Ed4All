"""
Alignment stage for Trainforge corpus pipeline.

Read-modify-write pass that enriches chunks with relational metadata:
- prereq_concepts: prerequisite concepts derived from concept graph + chunk ordering
- teaching_role: pedagogical function (introduce/elaborate/reinforce/assess/transfer/synthesize)
- learning_outcome_refs: semantic matching of chunks to learning outcomes via TF-IDF

Usage:
    python -m Trainforge.align_chunks \
        --corpus Trainforge/output/sample_101 \
        --objectives Courseforge/inputs/exam-objectives/SAMPLE_101_objectives.json \
        --llm-provider mock
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from MCP.orchestrator.llm_backend import LLMBackend
    from Trainforge.generators._curriculum_provider import (
        CurriculumAlignmentProvider,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROCEDURAL_TAGS = {"initial-post", "replies", "due"}
MAX_PREREQS_PER_CHUNK = 5
MAX_OUTCOMES_PER_CHUNK = 3
TFIDF_SIMILARITY_THRESHOLD = 0.15
VALID_ROLES = {"introduce", "elaborate", "reinforce", "assess", "transfer", "synthesize"}
WEEK_RE = re.compile(r"Week\s+(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Outcome:
    id: str
    statement: str
    bloom_level: str
    week_range: Tuple[int, int]  # (start_week, end_week)


# ---------------------------------------------------------------------------
# TF-IDF (reimplemented from LibV2/tools/libv2/outcome_linker.py)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words, stripping HTML."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.findall(r"\b[a-z][a-z0-9]+\b", text.lower())


class SimpleTFIDF:
    """Lightweight TF-IDF for outcome matching."""

    def __init__(self, documents: List[str]):
        self.doc_count = len(documents)
        self.df: Counter = Counter()
        self.doc_tfidf: List[Dict[str, float]] = []

        doc_tokens_list = []
        for doc in documents:
            tokens = tokenize(doc)
            doc_tokens_list.append(tokens)
            self.df.update(set(tokens))

        for tokens in doc_tokens_list:
            tf = Counter(tokens)
            total = len(tokens) or 1
            tfidf = {}
            for term, count in tf.items():
                tf_val = count / total
                idf_val = math.log((self.doc_count + 1) / (self.df[term] + 1)) + 1
                tfidf[term] = tf_val * idf_val
            self.doc_tfidf.append(tfidf)

    def search(self, query: str, limit: int = 5) -> List[Tuple[int, float]]:
        """Return (doc_index, similarity) sorted descending."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_tf = Counter(query_tokens)
        total = len(query_tokens)
        query_tfidf = {}
        for term, count in query_tf.items():
            tf_val = count / total
            idf_val = math.log((self.doc_count + 1) / (self.df.get(term, 0) + 1)) + 1
            query_tfidf[term] = tf_val * idf_val

        query_norm = math.sqrt(sum(v * v for v in query_tfidf.values()))
        scores = []
        for idx, doc_tfidf in enumerate(self.doc_tfidf):
            if not doc_tfidf:
                continue
            dot = sum(query_tfidf.get(t, 0) * w for t, w in doc_tfidf.items())
            doc_norm = math.sqrt(sum(v * v for v in doc_tfidf.values()))
            sim = dot / (query_norm * doc_norm) if query_norm and doc_norm else 0.0
            scores.append((idx, sim))

        scores.sort(key=lambda x: -x[1])
        return scores[:limit]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_corpus(corpus_dir: Path) -> Tuple[List[Dict], Dict]:
    """Load chunks.jsonl and concept_graph.json from a Trainforge output dir."""
    chunks_path = corpus_dir / "corpus" / "chunks.jsonl"
    graph_path = corpus_dir / "graph" / "concept_graph.json"

    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {chunks_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"concept_graph.json not found: {graph_path}")

    chunks = []
    with open(chunks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    with open(graph_path) as f:
        concept_graph = json.load(f)

    return chunks, concept_graph


def write_corpus(corpus_dir: Path, chunks: List[Dict]) -> None:
    """Write enriched chunks back to chunks.jsonl and chunks.json."""
    jsonl_path = corpus_dir / "corpus" / "chunks.jsonl"
    json_path = corpus_dir / "corpus" / "chunks.json"

    with open(jsonl_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    with open(json_path, "w") as f:
        json.dump(chunks, f, indent=2)


def load_objectives(objectives_path: Path) -> List[Outcome]:
    """Load outcomes from Trainforge-format objectives JSON."""
    with open(objectives_path) as f:
        doc = json.load(f)

    outcomes = []

    # Terminal objectives (course-level, all weeks)
    for to in doc.get("terminal_objectives", []):
        outcomes.append(Outcome(
            id=to["id"].lower(),
            statement=to["statement"],
            bloom_level=to.get("bloomLevel", "understand"),
            week_range=(1, 99),  # TOs apply to all weeks
        ))

    # Chapter objectives (week-scoped)
    for ch in doc.get("chapter_objectives", []):
        title = ch.get("chapter", "")
        # Extract week range: "Week 1-2: ..." → (1, 2)
        match = re.search(r"Week\s+(\d+)(?:\s*[-–]\s*(\d+))?", title)
        if match:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else start
        else:
            start, end = 1, 99

        for obj in ch.get("objectives", []):
            outcomes.append(Outcome(
                id=obj["id"].lower(),
                statement=obj["statement"],
                bloom_level=obj.get("bloomLevel", "understand"),
                week_range=(start, end),
            ))

    return outcomes


WEEK_SCOPED_ID_RE = re.compile(r"^w\d{2}-[a-z]{2}-\d{2,3}$", re.IGNORECASE)


def build_outcome_hierarchy(objectives_path: Path) -> Tuple[Dict[str, str], set]:
    """Return (parent_map, course_level_ids) from a dual-emission objectives file.

    parent_map maps every week-scoped ID (``w01-co-02``) to its course-level
    parent ID (``co-02``). When the objectives file pre-dates the dual-ID
    contract and carries no ``week_scoped_ids`` entries, parent_map is empty;
    week-scoped refs on chunks will then surface as orphans in the quality
    report (§2.1 orphan rule).
    """
    with open(objectives_path) as f:
        doc = json.load(f)

    parent_map: Dict[str, str] = {}
    course_level: set = set()

    for to in doc.get("terminal_objectives", []):
        parent_id = (to.get("id") or "").lower()
        if parent_id:
            course_level.add(parent_id)
        for ws in to.get("week_scoped_ids", []) or []:
            if ws and parent_id:
                parent_map[ws.lower()] = parent_id

    for ch in doc.get("chapter_objectives", []):
        for obj in ch.get("objectives", []):
            parent_id = (obj.get("id") or "").lower()
            if parent_id:
                course_level.add(parent_id)
            for ws in obj.get("week_scoped_ids", []) or []:
                if ws and parent_id:
                    parent_map[ws.lower()] = parent_id

    return parent_map, course_level


def partition_outcome_refs(
    chunks: List[Dict[str, Any]],
    parent_map: Dict[str, str],
    course_level_ids: set,
) -> int:
    """Split every chunk's learning_outcome_refs into course-level and pedagogical.

    Week-scoped IDs (``w0X-co-YY``) move from ``learning_outcome_refs`` into
    ``pedagogical_scope_refs``, each entry carrying its resolved parent or
    ``parent_id: null`` when the parent link is missing. The function returns
    the count of orphan refs encountered across the corpus — this is the
    number written to ``integrity.orphan_week_scoped_refs`` in the quality
    report (§2.1).

    Design choice (Option 2 from the plan): preserve-and-surface. We never
    silently drop week-scoped refs, never synthesise a parent ID, never raise.
    """
    orphan_count = 0
    for chunk in chunks:
        existing = chunk.get("learning_outcome_refs", []) or []
        course_refs: List[str] = []
        scope_refs: List[Dict[str, Any]] = []
        seen_scope_ids: set = set()

        for ref in existing:
            ref_lc = ref.lower()
            if WEEK_SCOPED_ID_RE.match(ref_lc):
                if ref_lc in seen_scope_ids:
                    continue
                seen_scope_ids.add(ref_lc)
                parent = parent_map.get(ref_lc)
                if parent:
                    scope_refs.append({
                        "id": ref_lc,
                        "parent_id": parent,
                        "status": "resolved",
                    })
                    if parent not in course_refs:
                        course_refs.append(parent)
                else:
                    scope_refs.append({
                        "id": ref_lc,
                        "parent_id": None,
                        "status": "orphan",
                    })
                    orphan_count += 1
            else:
                if ref_lc not in course_refs:
                    course_refs.append(ref_lc)

        chunk["learning_outcome_refs"] = course_refs
        if scope_refs:
            chunk["pedagogical_scope_refs"] = scope_refs

    return orphan_count


# ---------------------------------------------------------------------------
# Build chunk sequence
# ---------------------------------------------------------------------------

def build_chunk_sequence(chunks: List[Dict]) -> List[Dict]:
    """
    Order chunks by follows_chunk chain, assigning a 'position' key to each.
    Returns chunks in sequence order.
    """
    {c["id"]: c for c in chunks}

    # Find root(s) — chunks with no follows_chunk
    roots = [c for c in chunks if not c.get("follows_chunk")]
    visited = set()
    ordered = []

    # Walk chains from each root
    for root in roots:
        current = root
        while current and current["id"] not in visited:
            visited.add(current["id"])
            ordered.append(current)
            # Find the chunk that follows this one
            next_chunk = None
            for c in chunks:
                if c.get("follows_chunk") == current["id"] and c["id"] not in visited:
                    next_chunk = c
                    break
            current = next_chunk

    # Append any orphans (not reached via chain) by ID order
    for c in chunks:
        if c["id"] not in visited:
            ordered.append(c)

    # Assign positions
    for i, chunk in enumerate(ordered):
        chunk["_position"] = i

    return ordered


# ---------------------------------------------------------------------------
# Field 1: prereq_concepts
# ---------------------------------------------------------------------------

def compute_prereq_concepts(
    chunks: List[Dict], concept_graph: Dict, verbose: bool = False
) -> None:
    """
    Mutate chunks in-place to add prereq_concepts field.

    Algorithm: For each chunk's concept_tags, find graph-adjacent concepts
    that first appeared in earlier chunks. Rank by relevance, take top 5.
    """
    # Build node frequency map
    node_freq = {n["id"]: n["frequency"] for n in concept_graph.get("nodes", [])}

    # Build adjacency with edge weights
    # adjacency[a] = {b: weight, ...}
    adjacency: Dict[str, Dict[str, int]] = {}
    for edge in concept_graph.get("edges", []):
        s, t, w = edge["source"], edge["target"], edge["weight"]
        adjacency.setdefault(s, {})[t] = w
        adjacency.setdefault(t, {})[s] = w

    # Build concept_first_seen: {tag: earliest_position}
    concept_first_seen: Dict[str, int] = {}
    for chunk in chunks:
        pos = chunk["_position"]
        for tag in chunk.get("concept_tags", []):
            if tag not in concept_first_seen:
                concept_first_seen[tag] = pos

    # Compute prereqs for each chunk
    for chunk in chunks:
        pos = chunk["_position"]
        own_tags = set(chunk.get("concept_tags", []))

        # Collect candidates: (tag, relevance_score)
        candidates: Dict[str, float] = {}
        for tag in own_tags:
            if tag in PROCEDURAL_TAGS:
                continue
            neighbors = adjacency.get(tag, {})
            for neighbor, weight in neighbors.items():
                if neighbor in PROCEDURAL_TAGS:
                    continue
                if neighbor in own_tags:
                    continue
                first_seen = concept_first_seen.get(neighbor)
                if first_seen is None or first_seen >= pos:
                    continue
                # Relevance: edge weight / sqrt(node frequency)
                freq = node_freq.get(neighbor, 1)
                score = weight / math.sqrt(freq)
                if neighbor in candidates:
                    candidates[neighbor] = max(candidates[neighbor], score)
                else:
                    candidates[neighbor] = score

        # Sort by score descending, take top N
        ranked = sorted(candidates.items(), key=lambda x: -x[1])
        prereqs = [tag for tag, _ in ranked[:MAX_PREREQS_PER_CHUNK]]

        # Fallback: if graph traversal found nothing but chunk isn't first,
        # use the previous chunk's top concept_tags as sequence-based prereqs.
        # This handles chunks whose tags are all rare (not in top-50 graph).
        if not prereqs and pos > 0:
            prev_chunk = chunks[pos - 1] if pos < len(chunks) else None
            if prev_chunk:
                prev_tags = [
                    t for t in prev_chunk.get("concept_tags", [])
                    if t not in PROCEDURAL_TAGS and t not in own_tags
                ]
                prereqs = prev_tags[:MAX_PREREQS_PER_CHUNK]

        chunk["prereq_concepts"] = prereqs

        if verbose and prereqs:
            print(f"  {chunk['id']}: prereqs={prereqs}")


# ---------------------------------------------------------------------------
# Field 2: teaching_role
# ---------------------------------------------------------------------------

def _heuristic_role(chunk: Dict) -> Optional[str]:
    """Try to classify teaching role by deterministic rules. Returns None if ambiguous."""
    chunk_type = chunk.get("chunk_type", "")
    source = chunk.get("source", {})
    resource_type = source.get("resource_type", "")

    # Only actual assessment items get "assess" — explanatory preambles within
    # quiz pages (chunk_type=explanation, resource_type=quiz) should be
    # classified by content, not by their container.
    if chunk_type == "assessment_item":
        return "assess"

    if resource_type == "overview" and source.get("position_in_module", 0) == 0:
        return "introduce"

    if resource_type == "summary":
        return "synthesize"

    if resource_type == "application" or (
        chunk_type == "exercise" and resource_type != "quiz"
    ):
        return "transfer"

    return None


def _mock_role(chunk: Dict, concept_first_seen: Dict[str, int]) -> str:
    """Fallback classification for mock provider (no LLM)."""
    pos = chunk["_position"]
    tags = chunk.get("concept_tags", [])

    if not tags:
        return "introduce"

    # What fraction of this chunk's concepts appeared earlier?
    earlier_count = sum(
        1 for t in tags
        if concept_first_seen.get(t, pos) < pos
    )
    ratio = earlier_count / len(tags) if tags else 0

    # Check if concepts are from much earlier (3+ "weeks" back, ~14 chunks)
    if ratio > 0.5:
        earliest_prereq = min(
            (concept_first_seen.get(t, pos) for t in tags if concept_first_seen.get(t, pos) < pos),
            default=pos,
        )
        if pos - earliest_prereq > 14:
            return "reinforce"
        return "elaborate"

    return "introduce"


def _deterministic_role(chunk: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Try to resolve teaching_role deterministically from explicit metadata.

    Precedence:
      1. ``chunk["teaching_role_attr"]`` — surfaced by
         ``Trainforge/parsers/html_content_parser.py`` from a
         ``data-cf-teaching-role`` attribute on a flip-card / self-check /
         activity element (Courseforge REC-VOC-02 emit path).
      2. ``chunk["source"]["teaching_role"]`` — when the chunker propagates
         the parser's per-section role directly onto the chunk's source dict.
      3. ``chunk["source"]["section_teaching_roles"]`` — JSON-LD-derived
         section roles, used only when exactly one role is declared
         (ambiguous multi-value sections fall through).

    Returns ``(role, provenance)`` on success, ``(None, None)`` otherwise.
    Callers treat ``None`` as "no deterministic signal, continue to
    heuristic/LLM classifier".
    """
    # 1. Explicit per-chunk attribute
    attr_role = chunk.get("teaching_role_attr")
    if isinstance(attr_role, str) and attr_role in VALID_ROLES:
        return attr_role, "attr"

    source = chunk.get("source", {}) or {}
    if isinstance(source, dict):
        # 2. Chunker-propagated parser role
        src_role = source.get("teaching_role")
        if isinstance(src_role, str) and src_role in VALID_ROLES:
            return src_role, "source"

        # 3. JSON-LD section roles — unambiguous single-value case only
        section_roles = source.get("section_teaching_roles") or []
        if isinstance(section_roles, list) and len(section_roles) == 1:
            candidate = section_roles[0]
            if isinstance(candidate, str) and candidate in VALID_ROLES:
                return candidate, "jsonld"

    return None, None


def classify_teaching_roles(
    chunks: List[Dict],
    llm_provider: str = "mock",
    llm_model: str = "claude-haiku-4-5-20251001",
    verbose: bool = False,
    llm: Optional["LLMBackend"] = None,
    curriculum_provider: Optional["CurriculumAlignmentProvider"] = None,
) -> None:
    """Mutate chunks in-place to add teaching_role field.

    Precedence (REC-VOC-02, Wave 2):
      1. Deterministic signal from Courseforge-emitted metadata
         (``data-cf-teaching-role`` / JSON-LD ``teachingRole``). Skip all
         downstream classifiers.
      2. Existing deterministic heuristic (``_heuristic_role``).
      3. LLM classifier (anthropic) or mock fallback.
    The LLM path is preserved as-is for legacy IMSCCs that don't carry
    the deterministic attribute.

    Args:
        chunks: Chunks to classify in-place.
        llm_provider: ``"anthropic"`` to invoke the LLM for ambiguous chunks,
            anything else to stick with the mock/heuristic path.
        llm_model: Model identifier passed to the backend.
        verbose: Print per-chunk classifications.
        llm: Optional pre-built :class:`LLMBackend` instance. When provided,
            overrides the ``llm_provider`` path and routes LLM calls through
            the injected backend — enabling local / api / mock swap-in.
        curriculum_provider: Optional pre-built
            :class:`CurriculumAlignmentProvider`. When provided, the
            ambiguous-chunk fallback path routes through this LLM-agnostic
            provider (Anthropic / Together / Local selectable) instead of
            the legacy ``LLMBackend``-driven Anthropic path. Additive —
            when both ``llm`` and ``curriculum_provider`` are passed,
            ``curriculum_provider`` wins; when neither is passed, the
            existing legacy path is unchanged.
    """
    # Belt-and-suspenders: catch schema drift against the canonical
    # teaching_role enum. Soft-imports so standalone Trainforge installs
    # (without the repo root on sys.path) still work.
    try:
        from lib.ontology.teaching_roles import get_valid_roles as _canonical_valid_roles
        if VALID_ROLES != _canonical_valid_roles():
            print(
                "  WARNING: teaching_role schema drift: "
                f"align_chunks.VALID_ROLES={VALID_ROLES} vs "
                f"schemas/taxonomies/teaching_role.json={_canonical_valid_roles()}"
            )
    except Exception:
        pass  # standalone install without repo-level lib on sys.path

    # Build concept_first_seen for mock/heuristic fallback
    concept_first_seen: Dict[str, int] = {}
    for chunk in chunks:
        pos = chunk["_position"]
        for tag in chunk.get("concept_tags", []):
            if tag not in concept_first_seen:
                concept_first_seen[tag] = pos

    deterministic_count = 0
    heuristic_count = 0
    llm_count = 0
    ambiguous_chunks = []

    for chunk in chunks:
        # 1. Deterministic metadata (Courseforge REC-VOC-02 emit path)
        det_role, det_source = _deterministic_role(chunk)
        if det_role:
            chunk["teaching_role"] = det_role
            chunk["teaching_role_source"] = det_source
            deterministic_count += 1
            if verbose:
                print(f"  {chunk['id']}: role={det_role} (deterministic:{det_source})")
            continue

        # 2. Existing heuristic
        role = _heuristic_role(chunk)
        if role:
            chunk["teaching_role"] = role
            chunk["teaching_role_source"] = "heuristic"
            heuristic_count += 1
            if verbose:
                print(f"  {chunk['id']}: role={role} (heuristic)")
        else:
            ambiguous_chunks.append(chunk)

    # 3. Handle ambiguous chunks via LLM or mock fallback.
    # Precedence:
    #   - injected ``curriculum_provider`` (LLM-agnostic — Anthropic /
    #     Together / Local selectable) wins when present;
    #   - else the legacy ``LLMBackend`` path (anthropic-pinned) fires
    #     when ``llm`` is injected or ``llm_provider == "anthropic"``;
    #   - else the deterministic mock fallback runs.
    use_curriculum = curriculum_provider is not None
    use_llm = (llm is not None or llm_provider == "anthropic") and not use_curriculum
    if use_curriculum and ambiguous_chunks:
        _classify_with_curriculum_provider(
            ambiguous_chunks,
            concept_first_seen,
            curriculum_provider,
            verbose,
        )
        for chunk in ambiguous_chunks:
            chunk.setdefault("teaching_role_source", "llm")
        llm_count = len(ambiguous_chunks)
    elif use_llm and ambiguous_chunks:
        _classify_with_llm(
            ambiguous_chunks, concept_first_seen, llm_model, verbose, llm=llm
        )
        for chunk in ambiguous_chunks:
            chunk.setdefault("teaching_role_source", "llm")
        llm_count = len(ambiguous_chunks)
    else:
        # Mock: use heuristic fallback
        for chunk in ambiguous_chunks:
            role = _mock_role(chunk, concept_first_seen)
            chunk["teaching_role"] = role
            chunk["teaching_role_source"] = "mock"
            if verbose:
                print(f"  {chunk['id']}: role={role} (mock)")

    label = "LLM" if (use_curriculum or use_llm) else "mock"
    print(f"  Teaching roles: {deterministic_count} deterministic, "
          f"{heuristic_count} heuristic, "
          f"{llm_count or len(ambiguous_chunks)} {label}")


def _classify_with_llm(
    chunks: List[Dict],
    concept_first_seen: Dict[str, int],
    model: str,
    verbose: bool,
    llm: Optional["LLMBackend"] = None,
) -> None:
    """Classify ambiguous chunks using an LLM backend in batches.

    Prefers an injected :class:`LLMBackend`. When none is provided, lazily
    builds an :class:`AnthropicBackend` from the environment. The Anthropic
    SDK is never imported at module scope — it's loaded via the orchestrator
    backend module, which is the only place ``import anthropic`` lives.
    """
    if llm is None:
        try:
            from MCP.orchestrator.llm_backend import AnthropicBackend

            llm = AnthropicBackend(default_model=model)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  WARNING: could not initialize LLM backend ({exc}), "
                "falling back to mock"
            )
            for chunk in chunks:
                chunk["teaching_role"] = _mock_role(chunk, concept_first_seen)
            return

    batch_size = 12

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]

        # Build batch prompt
        chunk_descriptions = []
        for chunk in batch:
            words = chunk.get("text", "").split()[:200]
            excerpt = " ".join(words)
            chunk_descriptions.append(
                f"Chunk {chunk['id']} (position {chunk['_position']}):\n"
                f"  type: {chunk.get('chunk_type')}, resource: {chunk['source'].get('resource_type')}\n"
                f"  concept_tags: {chunk.get('concept_tags', [])}\n"
                f"  prereq_concepts: {chunk.get('prereq_concepts', [])}\n"
                f"  text excerpt: {excerpt}"
            )

        prompt = (
            "Classify the teaching role of each chunk below. "
            "Return ONLY a JSON array of objects with 'id' and 'role' fields.\n\n"
            "Roles:\n"
            "- introduce: First exposure to concepts in the course sequence\n"
            "- elaborate: Adds depth to previously introduced concepts\n"
            "- reinforce: Revisits concepts from earlier weeks in a new context\n"
            "- synthesize: Connects multiple concepts or summarizes\n\n"
            "Chunks:\n" + "\n\n".join(chunk_descriptions)
        )

        try:
            text = llm.complete_sync(
                system="",
                user=prompt,
                model=model,
                max_tokens=1024,
            )

            # Extract JSON array from response
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                results = json.loads(match.group())
                result_map = {r["id"]: r["role"] for r in results}
                for chunk in batch:
                    role = result_map.get(chunk["id"], "elaborate")
                    if role not in VALID_ROLES:
                        role = "elaborate"
                    chunk["teaching_role"] = role
                    if verbose:
                        print(f"  {chunk['id']}: role={role} (LLM)")
            else:
                # Fallback
                for chunk in batch:
                    chunk["teaching_role"] = _mock_role(chunk, concept_first_seen)
        except Exception as e:
            print(f"  WARNING: LLM batch failed ({e}), falling back to mock")
            for chunk in batch:
                chunk["teaching_role"] = _mock_role(chunk, concept_first_seen)


def _classify_with_curriculum_provider(
    chunks: List[Dict],
    concept_first_seen: Dict[str, int],
    provider: "CurriculumAlignmentProvider",
    verbose: bool,
) -> None:
    """Classify ambiguous chunks via the LLM-agnostic curriculum provider.

    Per-chunk dispatch (rather than the batch-of-12 the legacy
    ``_classify_with_llm`` uses) because the curriculum provider's
    role classification is a single-token output per call — batching
    via a JSON array prompt loses the four-class accuracy guarantee
    that the provider's invalid-role validation enforces. On any per-
    chunk failure (provider error, invalid response, transport
    error), fall back to the mock heuristic so the downstream pipeline
    keeps moving rather than failing the whole alignment stage.
    """
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        # Pull a small window of neighbors as pedagogical context. The
        # caller doesn't carry an explicit prev/next pointer; the
        # _position field built by build_chunk_sequence is the only
        # ordering surface we have.
        position = chunk.get("_position", 0)
        neighbors = [
            {
                "id": c.get("id"),
                "concept_tags": c.get("concept_tags") or [],
                "text": (c.get("text") or "")[:200],
            }
            for c in chunks
            if c is not chunk
            and abs(int(c.get("_position", 0)) - int(position)) <= 2
        ]
        try:
            role = provider.classify_teaching_role(
                str(chunk.get("text") or ""),
                chunk_id=chunk_id,
                neighbors=neighbors,
            )
            chunk["teaching_role"] = role
            if verbose:
                print(
                    f"  {chunk_id}: role={role} (curriculum-provider)"
                )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(
                    f"  WARNING: curriculum-provider failed for "
                    f"{chunk_id} ({exc}); falling back to mock"
                )
            chunk["teaching_role"] = _mock_role(chunk, concept_first_seen)


# ---------------------------------------------------------------------------
# Field 3: learning_outcome_refs
# ---------------------------------------------------------------------------

def match_learning_outcomes(
    chunks: List[Dict],
    objectives_path: Path,
    verbose: bool = False,
) -> None:
    """Mutate chunks in-place to enrich learning_outcome_refs via TF-IDF.

    Note: the existing chunk field is 'learning_outcome_refs' (set by the
    chunker from literal CO-XX matches). This pass enriches that same field
    with semantic TF-IDF matches, merging with any existing literal refs.
    """
    outcomes = load_objectives(objectives_path)
    if not outcomes:
        print("  WARNING: No outcomes loaded, skipping learning_outcome_refs")
        return

    # Build TF-IDF index over outcome statements
    outcome_texts = [o.statement for o in outcomes]
    index = SimpleTFIDF(outcome_texts)

    linked_count = 0
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            chunk.setdefault("learning_outcome_refs", [])
            continue

        # Determine chunk's week from lesson_title
        lesson_title = chunk.get("source", {}).get("lesson_title", "")
        week_match = WEEK_RE.search(lesson_title)
        chunk_week = int(week_match.group(1)) if week_match else None

        # Filter outcomes by week range
        if chunk_week is not None:
            candidate_indices = [
                i for i, o in enumerate(outcomes)
                if o.week_range[0] <= chunk_week <= o.week_range[1]
            ]
        else:
            candidate_indices = list(range(len(outcomes)))

        # Score chunk against all outcomes (TF-IDF), then filter by candidates
        all_matches = index.search(text, limit=len(outcomes))

        # Filter to candidates and apply threshold
        new_refs = []
        for outcome_idx, score in all_matches:
            if outcome_idx not in candidate_indices:
                continue
            if score < TFIDF_SIMILARITY_THRESHOLD:
                continue
            new_refs.append(outcomes[outcome_idx].id)
            if len(new_refs) >= MAX_OUTCOMES_PER_CHUNK:
                break

        # Merge with existing literal refs (preserve what's already there)
        existing = set(chunk.get("learning_outcome_refs", []))
        merged = list(existing)
        for ref in new_refs:
            if ref not in existing:
                merged.append(ref)
                existing.add(ref)

        # Fallback: if TF-IDF found nothing but we know the week, assign
        # the first CO from that week. Every chunk belongs to *some* outcome.
        if not merged and chunk_week is not None:
            week_cos = [
                outcomes[i].id for i in candidate_indices
                if outcomes[i].id.startswith("co-")
            ]
            if week_cos:
                merged = [week_cos[0]]

        # Cap at max
        chunk["learning_outcome_refs"] = merged[:MAX_OUTCOMES_PER_CHUNK]

        if chunk["learning_outcome_refs"]:
            linked_count += 1

        if verbose and chunk["learning_outcome_refs"]:
            print(f"  {chunk['id']}: outcomes={chunk['learning_outcome_refs']}")

    print(f"  Outcome linking: {linked_count}/{len(chunks)} chunks linked")


# ---------------------------------------------------------------------------
# Quality report update
# ---------------------------------------------------------------------------

def update_quality_report(
    corpus_dir: Path,
    chunks: List[Dict],
    valid_outcome_ids: Optional[set] = None,
    orphan_week_scoped_refs: int = 0,
) -> None:
    """Update quality_report.json with alignment field coverage metrics.

    ``learning_outcome_refs_coverage`` measures *referential integrity* under
    METRICS_SEMANTIC_VERSION=2: a chunk counts only if at least one of its
    ``learning_outcome_refs`` resolves to ``valid_outcome_ids``. When the
    caller doesn't pass a valid-ID set, the metric falls back to presence.
    """
    report_path = corpus_dir / "quality" / "quality_report.json"
    if not report_path.exists():
        return

    with open(report_path) as f:
        report = json.load(f)

    total = len(chunks) or 1

    # Alignment-specific metrics
    prereq_coverage = sum(1 for c in chunks if c.get("prereq_concepts")) / total
    role_coverage = sum(1 for c in chunks if c.get("teaching_role")) / total

    if valid_outcome_ids is not None:
        outcome_coverage = sum(
            1 for c in chunks
            if any(r in valid_outcome_ids for r in c.get("learning_outcome_refs", []))
        ) / total
        broken_refs = [
            {"chunk_id": c["id"], "ref": r}
            for c in chunks
            for r in c.get("learning_outcome_refs", [])
            if r not in valid_outcome_ids
        ]
    else:
        outcome_coverage = sum(1 for c in chunks if c.get("learning_outcome_refs")) / total
        broken_refs = []

    # Role consistency: check for type/role mismatches
    role_mismatches = 0
    for c in chunks:
        ct = c.get("chunk_type", "")
        role = c.get("teaching_role", "")
        if ct == "explanation" and role == "assess":
            role_mismatches += 1
        if ct == "assessment_item" and role != "assess":
            role_mismatches += 1
    role_consistency = 1.0 - (role_mismatches / total)

    role_dist = Counter(c.get("teaching_role", "none") for c in chunks)

    report["alignment"] = {
        "prereq_concepts_coverage": round(prereq_coverage, 3),
        "teaching_role_coverage": round(role_coverage, 3),
        "learning_outcome_refs_coverage": round(outcome_coverage, 3),
        "teaching_role_consistency": round(role_consistency, 3),
        "teaching_role_distribution": dict(role_dist),
    }

    # Referential-integrity findings live under ``integrity`` alongside the
    # base-pass integrity block (see process_course.py _generate_quality_report).
    integrity = report.setdefault("integrity", {})
    existing_broken = integrity.get("broken_refs", [])
    integrity["broken_refs"] = existing_broken + broken_refs
    integrity["orphan_week_scoped_refs"] = orphan_week_scoped_refs

    # Recompute overall score including alignment
    base_score = report.get("overall_quality_score", 0.0)
    alignment_score = (prereq_coverage + role_coverage + outcome_coverage + role_consistency) / 4
    # Weighted blend: 60% base corpus quality, 40% alignment quality
    report["overall_quality_score"] = round(base_score * 0.6 + alignment_score * 0.4, 3)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


# ---------------------------------------------------------------------------
# Stats reporting
# ---------------------------------------------------------------------------

def print_stats(chunks: List[Dict], fields: List[str]) -> None:
    """Print alignment statistics."""
    total = len(chunks)

    if "prereq_concepts" in fields:
        with_prereqs = sum(1 for c in chunks if c.get("prereq_concepts"))
        avg_prereqs = (
            sum(len(c.get("prereq_concepts", [])) for c in chunks) / total
            if total else 0
        )
        print(f"\n  prereq_concepts: {with_prereqs}/{total} chunks populated "
              f"(avg {avg_prereqs:.1f} per chunk)")

    if "teaching_role" in fields:
        with_role = sum(1 for c in chunks if c.get("teaching_role"))
        role_dist = Counter(c.get("teaching_role", "none") for c in chunks)
        print(f"  teaching_role: {with_role}/{total} chunks classified")
        for role, count in sorted(role_dist.items()):
            print(f"    {role}: {count}")

    if "learning_outcome_refs" in fields:
        with_refs = sum(1 for c in chunks if c.get("learning_outcome_refs"))
        outcome_dist = Counter()
        for c in chunks:
            for ref in c.get("learning_outcome_refs", []):
                outcome_dist[ref] += 1
        print(f"  learning_outcome_refs: {with_refs}/{total} chunks linked")
        if outcome_dist:
            top_5 = outcome_dist.most_common(5)
            print(f"    top outcomes: {', '.join(f'{oid}({cnt})' for oid, cnt in top_5)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Alignment stage: enrich Trainforge chunks with relational metadata",
    )
    p.add_argument("--corpus", required=True,
                   help="Path to Trainforge output directory")
    p.add_argument("--objectives",
                   help="Path to objectives JSON (required for learning_outcome_refs)")
    p.add_argument("--fields", default="prereq_concepts,teaching_role,learning_outcome_refs",
                   help="Comma-separated fields to compute (default: all)")
    p.add_argument("--llm-provider", default="mock", choices=["mock", "anthropic", "together", "local"],
                   help=(
                       "LLM provider for the legacy direct-classification path "
                       "(default: mock). For the license-clean teaching-role surface, "
                       "prefer --curriculum-provider local (or "
                       "CURRICULUM_ALIGNMENT_PROVIDER=local) — that route reuses the "
                       "same LOCAL_SYNTHESIS_* / TOGETHER_* env vars as synthesis."
                   ))
    p.add_argument("--llm-model", default="claude-haiku-4-5-20251001",
                   help="Model for LLM calls")
    p.add_argument(
        "--curriculum-provider",
        default=None,
        choices=["anthropic", "together", "local"],
        help=(
            "Curriculum-alignment provider for ambiguous-chunk teaching-role "
            "classification. Routes through "
            "Trainforge.generators._curriculum_provider.CurriculumAlignmentProvider. "
            "When unset, falls back to the CURRICULUM_ALIGNMENT_PROVIDER env "
            "var; when env is also unset, the legacy / mock path runs (no "
            "curriculum provider injected). Recommended setting for "
            "ToS-clean training corpora is 'local'."
        ),
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print stats without writing files")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-chunk decisions")
    return p


# ---------------------------------------------------------------------------
# Curriculum provider env-var resolution (Wave 137 followup)
# ---------------------------------------------------------------------------

CURRICULUM_PROVIDER_ENV = "CURRICULUM_ALIGNMENT_PROVIDER"


def _resolve_curriculum_provider_choice(args: argparse.Namespace) -> Optional[str]:
    """Pick the effective curriculum-provider value.

    Priority order:
      1. Explicit ``--curriculum-provider`` CLI flag if passed.
      2. ``CURRICULUM_ALIGNMENT_PROVIDER`` env var if set.
      3. ``None`` — no provider injected; legacy / mock path runs.

    The ``CurriculumAlignmentProvider`` class also reads the same env
    var inside its constructor, but only when its constructor is
    actually invoked. Pre-Wave-137-followup, ``align_chunks.main()``
    never instantiated a provider, which made the env var dead from
    the ``process_course.py`` invocation path. This helper closes that
    gap by reading the env var at the CLI surface.
    """
    cli_value = getattr(args, "curriculum_provider", None)
    if cli_value:
        return cli_value
    env_value = os.environ.get(CURRICULUM_PROVIDER_ENV)
    if env_value:
        return env_value
    return None


def _build_curriculum_provider(
    provider_choice: str,
    *,
    capture: Optional[Any] = None,
) -> Any:
    """Instantiate ``CurriculumAlignmentProvider`` for ``provider_choice``.

    Wraps the class import + construction so a bad provider string
    surfaces as a clean CLI error (exit 2) instead of a stack trace.
    Threads through the optional ``capture`` so every classification
    call emits a ``curriculum_alignment_call`` decision event.
    """
    from Trainforge.generators._curriculum_provider import (
        CurriculumAlignmentProvider,
    )
    return CurriculumAlignmentProvider(
        provider=provider_choice,
        capture=capture,
    )


def main(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    if args is None:
        args = build_parser().parse_args()

    corpus_dir = Path(args.corpus)
    fields = [f.strip() for f in args.fields.split(",")]

    # --- Wave 137 followup: resolve curriculum provider from CLI / env ---
    # Priority: --curriculum-provider CLI flag > CURRICULUM_ALIGNMENT_PROVIDER
    # env var > None (no provider injected; legacy / mock path runs).
    # The CurriculumAlignmentProvider constructor itself accepts
    # ``provider=None`` and falls back to the env, but it's never
    # instantiated unless this CLI surface fires it. The default of
    # ``DEFAULT_PROVIDER='anthropic'`` inside the class is intentionally
    # unchanged — backward compatibility for direct callers.
    curriculum_choice = _resolve_curriculum_provider_choice(args)
    curriculum_provider = None
    curriculum_capture = None
    if curriculum_choice is not None:
        # Wire a DecisionCapture for the curriculum_alignment_call events
        # the provider emits per classification. Soft-import so a
        # standalone Trainforge install (without lib/ on sys.path)
        # degrades to capture=None instead of failing.
        try:
            from lib.decision_capture import DecisionCapture
            course_code = (
                Path(args.corpus).resolve().name or "UNKNOWN"
            ).upper()
            curriculum_capture = DecisionCapture(
                course_code=course_code,
                phase="curriculum-alignment",
                tool="trainforge",
                streaming=True,
            )
        except Exception:
            curriculum_capture = None
        try:
            curriculum_provider = _build_curriculum_provider(
                curriculum_choice, capture=curriculum_capture,
            )
        except ValueError as exc:
            # Unknown provider string → clean exit 2.
            print(
                f"[Alignment] curriculum-provider error: {exc}",
                file=sys.stderr,
            )
            sys.exit(2)

    print(f"[Alignment] Loading corpus from {corpus_dir}")
    chunks, concept_graph = load_corpus(corpus_dir)
    print(f"  Loaded {len(chunks)} chunks, "
          f"{len(concept_graph.get('nodes', []))} graph nodes, "
          f"{len(concept_graph.get('edges', []))} graph edges")

    # Build sequence (adds _position to each chunk)
    chunks = build_chunk_sequence(chunks)
    print(f"  Sequence built: positions 0..{len(chunks) - 1}")

    # --- Field 1: prereq_concepts ---
    if "prereq_concepts" in fields:
        print("\n[1/3] Computing prereq_concepts...")
        compute_prereq_concepts(chunks, concept_graph, verbose=args.verbose)

    # --- Field 2: teaching_role ---
    if "teaching_role" in fields:
        print("\n[2/3] Classifying teaching_role...")
        classify_teaching_roles(
            chunks,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            verbose=args.verbose,
            curriculum_provider=curriculum_provider,
        )

    # --- Field 3: learning_outcome_refs ---
    orphan_count = 0
    if "learning_outcome_refs" in fields:
        if not args.objectives:
            print("\n[3/3] Skipping learning_outcome_refs (no --objectives provided)")
        else:
            # Partition first: move any week-scoped IDs (w01-co-02) onto the
            # chunk's pedagogical_scope_refs field with parent links. Orphans
            # are preserved with parent_id: null per §2.1.
            parent_map, course_level_ids = build_outcome_hierarchy(Path(args.objectives))
            orphan_count = partition_outcome_refs(chunks, parent_map, course_level_ids)
            if orphan_count:
                print(f"  Orphan week-scoped refs surfaced: {orphan_count}")

            print("\n[3/3] Matching learning_outcome_refs...")
            match_learning_outcomes(
                chunks, Path(args.objectives), verbose=args.verbose,
            )

    # --- Stats ---
    print("\n" + "=" * 50)
    print("ALIGNMENT SUMMARY")
    print("=" * 50)
    print_stats(chunks, fields)

    # --- Write ---
    if not args.dry_run:
        # Remove internal _position field before writing
        for chunk in chunks:
            chunk.pop("_position", None)

        write_corpus(corpus_dir, chunks)
        # Pass the valid-ID set and orphan count through so the updated
        # quality_report reflects referential integrity (§1.1) + §2.1 orphan
        # surfacing.
        valid_ids: Optional[set] = None
        if args.objectives and "learning_outcome_refs" in fields:
            parent_map, course_level_ids = build_outcome_hierarchy(Path(args.objectives))
            valid_ids = set(course_level_ids) | set(parent_map.keys())
        update_quality_report(
            corpus_dir, chunks,
            valid_outcome_ids=valid_ids,
            orphan_week_scoped_refs=orphan_count,
        )
        print(f"\n  Written to {corpus_dir / 'corpus'}")
    else:
        print("\n  [DRY RUN] No files written")
        # Still clean up _position
        for chunk in chunks:
            chunk.pop("_position", None)

    # Return stats for programmatic use
    return {
        "total_chunks": len(chunks),
        "prereq_populated": sum(1 for c in chunks if c.get("prereq_concepts")),
        "roles_populated": sum(1 for c in chunks if c.get("teaching_role")),
        "outcomes_populated": sum(1 for c in chunks if c.get("learning_outcome_refs")),
    }


if __name__ == "__main__":
    main()

"""End-to-end grounded-answer pipeline + citation gate (D9 / D5).

``answer_course_question`` is THE single entry point WS4 (and the wave-C
CLI) must use — it owns the refusal policy and the citation gate, so
bypassing it (e.g. calling ``compose_answer`` directly) is the
hallucination-by-construction path the plan forbids.

Flow (D9):

  retrieve (lazy LibV2 retriever; typed semantic errors propagate, NEVER a
            silent engine downgrade)
    -> evaluate confidence / refuse PRE-LLM (no client call on refusal)
    -> compose_answer (the LLM call site, owned by the composer)
    -> model-side ``not_in_course`` handling (refusal)
    -> CITATION GATE: every cited chunk must anchor acceptably via WS1's
       ``resolve_citation_anchor`` (``_RESOLVED_STATUSES``); an unresolvable
       citation BLOCKS emission (answer_text withheld) — no partial emission
       that silently drops the bad citation
    -> assemble ``GroundedAnswer`` + ``Citation`` objects (the frozen WS4
       rendering contract, § 3 / D5).

Two new decision-capture sites live here (composition's emit is the
composer's): ``grounded_answer_refusal`` (pre-LLM low-confidence AND
model-side not_in_course) and ``grounded_answer_citation_gate`` (pass or
block). Both interpolate dynamic, replayable signals.

No cloud call ever; no canned-answer fallback. Backend/index absence
surfaces as a typed error (``AnswerBackendUnavailable`` /
``SemanticIndexMissing`` / ...), not a fabricated answer.
"""
from __future__ import annotations

import hashlib
import inspect
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dataclasses import replace as _dc_replace

from lib.retrieval._text import normalize_ws
from lib.retrieval._prompts import COMPLETENESS_REMEDIATION_DIRECTIVE
from lib.retrieval.answer_completeness import analyze_completeness
from lib.retrieval.answer_composer import (
    AnswerComposeError,
    ComposedAnswer,
    RetrievedPassage,
    compose_answer,
)
from lib.retrieval.citation_attribution import (
    ADD_MIN_SHINGLE,
    ATTRIBUTION_SHINGLE_SIZE,
    MAX_ADDED_CITATIONS,
    MODE_OFF,
    MODE_ON,
    MODE_SHADOW,
    NLI_ADD_ENTAILMENT_FLOOR,
    NLI_ADD_MAX_ADDED_CITATIONS,
    NLI_ADD_TOKEN_COVERAGE_FLOOR,
    AttributionReport,
    attribute_citations,
    claim_numeric_literals,
    claim_numerics_present,
    passage_token_coverage,
    resolve_add_min_shingle,
    resolve_min_overlap,
    resolve_nli_add_mode,
    resolve_prune_mode,
)
from lib.retrieval.citation_anchor import (
    AnchorStatus,
    CitationAnchor,
    _RESOLVED_STATUSES,
    resolve_citation_anchor,
)
from lib.retrieval.refusal import (
    REASON_LOW_CONFIDENCE,
    REASON_NOT_IN_COURSE_MODEL,
    RefusalPolicy,
    evaluate_confidence,
    resolve_policy,
    should_refuse,
)
from lib.retrieval.reranker import (
    maybe_rerank,
    reranker_configured,
    resolve_candidate_pool,
)

__all__ = [
    "Citation",
    "GroundedAnswer",
    "answer_course_question",
    "page_label_for_source",
    "DECISION_TYPE_CITATION_PRUNE",
    "DECISION_TYPE_NLI_CITATION_ADD",
    "DECISION_TYPE_COMPLETENESS_RECHECK",
    # Status constants (WS4 + CLI map these to display copy / exit codes).
    "STATUS_ANSWERED",
    "STATUS_ANSWERED_WITH_WARNINGS",
    "STATUS_REFUSED_LOW_CONFIDENCE",
    "STATUS_REFUSED_NOT_IN_COURSE",
    "STATUS_BLOCKED_INVALID_CITATION",
    "STATUS_BLOCKED_CITATION_GATE",
]


# --------------------------------------------------------------------------- #
# Status enum (carried as data; WS4 owns the learner-facing copy)
# --------------------------------------------------------------------------- #

STATUS_ANSWERED = "answered"
STATUS_ANSWERED_WITH_WARNINGS = "answered_with_warnings"
STATUS_REFUSED_LOW_CONFIDENCE = "refused_low_confidence"
STATUS_REFUSED_NOT_IN_COURSE = "refused_not_in_course"
STATUS_BLOCKED_INVALID_CITATION = "blocked_invalid_citation"
STATUS_BLOCKED_CITATION_GATE = "blocked_citation_gate"

_ANSWERED_STATUSES = frozenset({STATUS_ANSWERED, STATUS_ANSWERED_WITH_WARNINGS})

DECISION_TYPE_REFUSAL = "grounded_answer_refusal"
DECISION_TYPE_CITATION_GATE = "grounded_answer_citation_gate"
DECISION_TYPE_CITATION_PRUNE = "grounded_answer_citation_prune"
DECISION_TYPE_NLI_CITATION_ADD = "grounded_answer_nli_citation_add"
DECISION_TYPE_COMPLETENESS_RECHECK = "grounded_answer_completeness_recheck"
DECISION_PHASE = "libv2-answer"

# Citation-prune/add outcomes (the capture `decision` suffix, plan §2.5).
_PRUNE_OUTCOME_PRUNED = "pruned"
_PRUNE_OUTCOME_NOOP = "noop"
_PRUNE_OUTCOME_PRUNED_ALL = "pruned_all_claimless"
_PRUNE_OUTCOME_NO_CLAIMS = "no_scorable_claims"
_PRUNE_OUTCOME_DISABLED = "disabled"


# --------------------------------------------------------------------------- #
# Citation — the frozen WS4 rendering contract (D5)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Citation:
    """A single passage-backed citation, the WS4 rendering contract (D5).

    ``to_dict()`` is consumed verbatim by WS4: ``page_label`` is the focusable
    link text ("Source: {page_label}"), ``link_target`` resolves to the
    rendered course page + fragment for focus management. ``char_span`` is
    forwarded into ``link_target`` ONLY for ``resolved_exact`` anchors (the WS1
    audit showed spans are otherwise untrustworthy), so the UI never scrolls to
    a fabricated offset.
    """

    chunk_id: str
    item_path: str
    section_heading: Optional[str]
    module_id: Optional[str]
    page_label: str
    anchor_status: str
    source_path: Optional[str]
    text_quote: Optional[str]
    link_target: Dict[str, Any] = field(default_factory=dict)
    # B4 provenance chain (additive, optional). ``source_block`` is the cited
    # chunk's primary DART block sourceId ("dart:{slug}#{block_id}");
    # ``pdf_pages`` are the original PDF page numbers that block came from.
    # Both are ``None`` / ``[]`` for legacy corpora whose chunks carry no
    # ``source.source_references`` (the foundation degrades gracefully).
    source_block: Optional[str] = None
    pdf_pages: List[int] = field(default_factory=list)
    # Human-readable module title from the chunk source (additive, optional).
    # ``module_id`` is a filename stem ("content_01") that repeats across
    # weeks, so renderers prefer this for display when present.
    module_title: Optional[str] = None
    # Claim-attribution surfacing (additive, optional). ``supporting_excerpt``
    # is the chunk sentence with maximal support for this citation's best-
    # attributed answer claim (``normalize_ws``'d, never fabricated — ``None``
    # when attribution found no supporting sentence or ran disabled).
    # ``supported_claim_count`` is how many answer claims this citation supports.
    # Both populated for KEPT and ADDED citations; the renderer prefers
    # ``supporting_excerpt`` over the legacy whole-answer ``text_quote``.
    supporting_excerpt: Optional[str] = None
    supported_claim_count: int = 0
    # Library-wide provenance (W4.2, additive/optional). The course this cited
    # chunk belongs to. ``None`` on the default single-course path — the key is
    # then OMITTED from ``to_dict`` so the single-course rendering contract stays
    # byte-identical. Populated only by the library-wide union so a renderer can
    # attribute every citation to its source course.
    course_slug: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "chunk_id": self.chunk_id,
            "item_path": self.item_path,
            "section_heading": self.section_heading,
            "module_id": self.module_id,
            "page_label": self.page_label,
            "anchor_status": self.anchor_status,
            "source_path": self.source_path,
            "text_quote": self.text_quote,
            "link_target": dict(self.link_target),
            "source_block": self.source_block,
            "pdf_pages": list(self.pdf_pages),
            "module_title": self.module_title,
            "supporting_excerpt": self.supporting_excerpt,
            "supported_claim_count": self.supported_claim_count,
        }
        # Additive: only surface course_slug when set (library-wide mode), so the
        # default single-course citation dict is byte-identical to the frozen
        # WS4 contract.
        if self.course_slug is not None:
            out["course_slug"] = self.course_slug
        return out


@dataclass
class GroundedAnswer:
    """The end-to-end answer result — THE WS4 + CLI JSON contract (D9).

    ``to_dict()`` is additive-only after freeze. ``citations`` is non-empty
    only for ``answered*`` statuses; ``answer_text`` is withheld (``None``) for
    every refused/blocked status.
    """

    status: str
    query: str
    course_slug: str
    engine: str
    answer_text: Optional[str]
    citations: List[Citation]
    refusal: Optional[Dict[str, Any]]
    confidence: Dict[str, Any]
    groundedness: Optional[Dict[str, Any]]
    warnings: List[str]
    model_id: Optional[str]
    prompt_version: Optional[str]
    generated_at: str
    latency_ms: float
    # NLI-ADD shadow diagnostics (additive, optional). Populated only when the
    # NLI-ADD arm ran in shadow/on mode (ED4ALL_ANSWER_NLI_ADD); ``None`` on the
    # default (off) path. Lets eval aggregate would-add counts WITHOUT parsing
    # warning strings. Shape: {"mode", "outcome", "would_add_ids", "added_ids",
    # "candidates":[{chunk_id, entailment, token_coverage, numerics_present,
    # passes_composite, anchor_resolved, ...}], "reason"}.
    nli_citation_add: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "query": self.query,
            "course_slug": self.course_slug,
            "engine": self.engine,
            "answer_text": self.answer_text,
            "citations": [c.to_dict() for c in self.citations],
            "refusal": dict(self.refusal) if self.refusal is not None else None,
            "confidence": dict(self.confidence),
            "groundedness": (
                dict(self.groundedness) if self.groundedness is not None else None
            ),
            "warnings": list(self.warnings),
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "generated_at": self.generated_at,
            "latency_ms": self.latency_ms,
            # Additive: present only when the NLI-ADD arm ran (shadow/on).
            "nli_citation_add": (
                dict(self.nli_citation_add)
                if self.nli_citation_add is not None
                else None
            ),
        }


# --------------------------------------------------------------------------- #
# Page label + text-quote helpers
# --------------------------------------------------------------------------- #


def page_label_for_source(source: Dict[str, Any]) -> str:
    """Human link text for a chunk source: heading, else prettified path stem.

    Prefers ``section_heading``; falls back to the ``item_path`` stem with
    separators turned into spaces and title-cased. Empty source → ``"Source"``.
    """
    heading = (source or {}).get("section_heading")
    if heading:
        return str(heading).strip()
    item_path = (source or {}).get("item_path")
    if item_path:
        stem = Path(str(item_path)).stem
        pretty = re.sub(r"[-_]+", " ", stem).strip()
        if pretty:
            return pretty.title()
    return "Source"


def _first_supporting_quote(
    answer_text: Optional[str], passage_text: str
) -> Optional[str]:
    """First sentence of the passage that overlaps the answer (never fabricated).

    Picks the passage sentence sharing the most normalized content tokens with
    the answer text. Returns ``None`` when there is no overlap or no answer —
    the quote is advisory preview copy, never invented.
    """
    if not answer_text or not passage_text:
        return None
    answer_tokens = set(_content_tokens(answer_text))
    if not answer_tokens:
        return None
    best_sentence: Optional[str] = None
    best_overlap = 0
    for sentence in _split_sentences(passage_text):
        sent_tokens = set(_content_tokens(sentence))
        overlap = len(answer_tokens & sent_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence.strip()
    if best_overlap == 0:
        return None
    return normalize_ws(best_sentence) if best_sentence else None


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _split_sentences(text: str) -> List[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _content_tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text) if len(t) > 2]


# --------------------------------------------------------------------------- #
# Retrieval (lazy namespace import; typed errors propagate)
# --------------------------------------------------------------------------- #


def _libv2_root(repo_root: Path) -> Path:
    """The retriever's ``repo_root`` — the LibV2 root.

    ``ED4ALL_LIBV2_ROOT`` wins (the WS1/WS2 fixture + GUI contract). Otherwise
    accept either a passed repo root that already *is* the LibV2 root (has a
    ``courses/`` child) or one whose ``LibV2/`` child is the root.
    """
    import os

    env = os.environ.get("ED4ALL_LIBV2_ROOT")
    if env:
        return Path(env)
    repo_root = Path(repo_root)
    if (repo_root / "courses").is_dir():
        return repo_root
    cand = repo_root / "LibV2"
    if cand.is_dir():
        return cand
    return repo_root


def _retrieve(
    libv2_root: Path,
    course_slug: str,
    query: str,
    *,
    engine: str,
    limit: int,
) -> List[Any]:
    """Call the live LibV2 retriever (lazy import; precedent: vector_index_manifest).

    The ``engine`` kwarg is passed ONLY when the live signature accepts it
    (``inspect.signature`` probe). A non-lexical engine requested against a
    pre-E3 tree that lacks the kwarg raises ``RuntimeError`` naming the missing
    dependency — NEVER a silent downgrade to lexical. Typed semantic errors
    (``SemanticIndexMissing`` / ``SemanticIndexStale`` / ...) propagate.
    """
    from LibV2.tools.libv2.retriever import retrieve_chunks

    params = inspect.signature(retrieve_chunks).parameters
    kwargs: Dict[str, Any] = {"course_slug": course_slug, "limit": limit}
    if "engine" in params:
        kwargs["engine"] = engine
    elif engine != "lexical":
        raise RuntimeError(
            f"engine={engine!r} requested but the installed "
            f"LibV2.tools.libv2.retriever.retrieve_chunks has no 'engine' "
            f"parameter (pre-WS2-E3 tree). The semantic/hybrid retrieval "
            f"dependency is not available; refusing to silently downgrade to "
            f"lexical."
        )
    return list(retrieve_chunks(libv2_root, query, **kwargs))


def _infer_chunkset_kind(libv2_root: Path, course_slug: str) -> str:
    """Infer the chunkset kind from which chunks dir resolves for the course.

    ``dart_chunks/`` -> ``"dart"``; ``imscc_chunks/`` -> ``"imscc"``; legacy
    ``corpus/`` -> ``"corpus"``. Falls back to ``"dart"`` (the common
    DART-staged pipeline shape) when nothing resolves.
    """
    course_dir = libv2_root / "courses" / course_slug
    if (course_dir / "dart_chunks" / "chunks.jsonl").is_file():
        return "dart"
    if (course_dir / "imscc_chunks" / "chunks.jsonl").is_file():
        return "imscc"
    if (course_dir / "corpus" / "chunks.jsonl").is_file():
        return "corpus"
    return "dart"


def _vector_index_chunkset_kind(libv2_root: Path, course_slug: str) -> Optional[str]:
    """Read ``chunkset_kind`` from the course's vector-index manifest, cheaply.

    For the semantic / hybrid-rrf engines the on-device vector index is the
    authoritative chunkset: retrieval, hydration, and the citation gate must
    all resolve against the SAME chunkset the index was built over. A course
    can carry both ``dart_chunks/`` and an ``imscc``-pinned index at once, so
    the directory-presence heuristic (:func:`_infer_chunkset_kind`) can pick
    the wrong kind and route the citation gate at the wrong source — every
    citation then fails ``source_page_missing`` and a correct answer is
    blocked. Reading the index manifest aligns the gate with the index.

    Reads ``vector_index/manifest.json`` directly (NOT the embeddings matrix)
    so this stays cheap. Returns ``None`` when no index / manifest exists or
    the field is absent; the caller then falls back to the directory
    heuristic.
    """
    import json

    manifest_path = (
        libv2_root / "courses" / course_slug / "vector_index" / "manifest.json"
    )
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    kind = data.get("chunkset_kind")
    return str(kind) if kind else None


def _vector_index_embedding_model_id(
    libv2_root: Path, course_slug: str
) -> Optional[str]:
    """Read ``embedding_model_id`` from the course's vector-index manifest.

    The refusal policy for a SEMANTIC engine is pinned PER EMBEDDING MODEL
    (a cosine threshold tuned for one embedder is meaningless for another), so
    :func:`resolve_policy` needs the model id the live index was built against.
    The vector-index manifest records it; reading the manifest (NOT the
    embeddings matrix) keeps this cheap. Returns ``None`` when no index /
    manifest exists or the field is absent — :func:`resolve_policy` then falls
    back to the v0-uncalibrated default rather than reusing a stale pin.

    Lexical retrieval carries no embedder; the caller passes ``None`` and never
    reaches here.
    """
    import json

    manifest_path = (
        libv2_root / "courses" / course_slug / "vector_index" / "manifest.json"
    )
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    model = data.get("embedding_model_id") or data.get("model_id")
    return str(model) if model else None


# --------------------------------------------------------------------------- #
# Citation gate (D5)
# --------------------------------------------------------------------------- #


def _passage_chunk_record(passage: RetrievedPassage) -> Dict[str, Any]:
    """Build the chunk-record shape the anchor resolver expects from a passage.

    The resolver keys off ``chunk["id"]``, ``chunk["text"]``, and
    ``chunk["source"]`` — all carried through on the retrieved passage, so the
    gate resolves the anchor without a second retrieval.
    """
    source = dict(passage.source or {})
    # Ensure the anchor-bearing keys are present even when the result object
    # surfaced them as top-level attributes rather than inside source.
    source.setdefault("item_path", passage.item_path)
    if passage.section_heading is not None:
        source.setdefault("section_heading", passage.section_heading)
    if passage.module_id is not None:
        source.setdefault("module_id", passage.module_id)
    return {"id": passage.chunk_id, "text": passage.text, "source": source}


def _provenance_from_source(
    source: Dict[str, Any]
) -> Tuple[Optional[str], List[int]]:
    """Pull the cited chunk's primary DART block sourceId + PDF pages (B4).

    Reads ``source.source_references`` — the additive Wave-10 provenance the
    chunker stamps as ``[{sourceId: "dart:{slug}#{block_id}", role, extractor,
    pages[]}]``. Prefers the ``role == "primary"`` ref (the block a chunk IS
    synthesized from); falls back to the first ref otherwise. Pages are the
    union over the SAME chosen ref. Returns ``(None, [])`` when no references
    are present (legacy corpora) so every downstream hop simply omits the
    final PDF link — the foundation degrades gracefully.
    """
    refs = (source or {}).get("source_references")
    if not isinstance(refs, list) or not refs:
        return None, []
    primary: Optional[Dict[str, Any]] = None
    first: Optional[Dict[str, Any]] = None
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        if first is None:
            first = ref
        if str(ref.get("role") or "") == "primary":
            primary = ref
            break
    chosen = primary or first
    if chosen is None:
        return None, []
    source_id = chosen.get("sourceId")
    source_block = str(source_id) if source_id else None
    pages_raw = chosen.get("pages")
    pages: List[int] = []
    if isinstance(pages_raw, list):
        for p in pages_raw:
            try:
                ip = int(p)
            except (TypeError, ValueError):
                continue
            if ip > 0 and ip not in pages:
                pages.append(ip)
    return source_block, sorted(pages)


def _build_citation(
    passage: RetrievedPassage,
    anchor: CitationAnchor,
    answer_text: Optional[str],
) -> Citation:
    source = dict(passage.source or {})
    source.setdefault("item_path", passage.item_path)
    if passage.section_heading is not None:
        source.setdefault("section_heading", passage.section_heading)
    page_label = page_label_for_source(source)
    quote = _first_supporting_quote(answer_text, passage.text)
    source_block, pdf_pages = _provenance_from_source(passage.source or {})

    fragment = _fragment_for(anchor, passage)
    char_span = (
        list(anchor.char_span)
        if (anchor.status is AnchorStatus.RESOLVED_EXACT and anchor.char_span)
        else None
    )
    link_target = {
        "kind": "course_page",
        "item_path": anchor.item_path or passage.item_path,
        "fragment": fragment,
        "char_span": char_span,
    }
    source_path = (
        str(anchor.source_path) if anchor.source_path is not None else None
    )
    return Citation(
        chunk_id=passage.chunk_id,
        item_path=anchor.item_path or passage.item_path,
        section_heading=passage.section_heading,
        module_id=passage.module_id,
        page_label=page_label,
        anchor_status=anchor.status.value,
        source_path=source_path,
        text_quote=quote,
        link_target=link_target,
        source_block=source_block,
        pdf_pages=pdf_pages,
        module_title=(
            str(source["module_title"]).strip()
            if source.get("module_title")
            else None
        ),
    )


def _fragment_for(
    anchor: CitationAnchor, passage: RetrievedPassage
) -> Dict[str, Any]:
    """The link fragment: xpath when present, else a heading slug, else None."""
    if anchor.html_xpath:
        return {"kind": "xpath", "value": anchor.html_xpath}
    heading = passage.section_heading
    if heading:
        slug = re.sub(r"[^a-z0-9]+", "-", str(heading).lower()).strip("-")
        if slug:
            return {"kind": "heading", "value": slug}
    return {"kind": None, "value": None}


def _run_citation_gate(
    cited_passages: Sequence[RetrievedPassage],
    course_dir: Path,
    *,
    chunkset_kind: str,
    answer_text: Optional[str],
    containment_threshold: float,
) -> Tuple[List[Citation], List[str], List[Tuple[str, str]], List[Tuple[str, float]]]:
    """Resolve every cited passage's anchor.

    Returns ``(citations, blocked_chunk_ids, per_citation_status,
    failure_containment)``. A passage whose anchor status is NOT in
    ``_RESOLVED_STATUSES`` is blocked: its chunk_id lands in
    ``blocked_chunk_ids`` and the pipeline withholds the answer (no partial
    emission with the bad citation dropped — dropping a citation changes the
    support story). ``failure_containment`` carries the measured containment
    rate of each blocked anchor for the capture rationale.
    """
    citations: List[Citation] = []
    blocked: List[str] = []
    statuses: List[Tuple[str, str]] = []
    failure_containment: List[Tuple[str, float]] = []
    for passage in cited_passages:
        chunk_record = _passage_chunk_record(passage)
        anchor = resolve_citation_anchor(
            chunk_record,
            course_dir,
            chunkset_kind=chunkset_kind,
            containment_threshold=containment_threshold,
        )
        statuses.append((passage.chunk_id, anchor.status.value))
        if anchor.status not in _RESOLVED_STATUSES:
            blocked.append(passage.chunk_id)
            failure_containment.append((passage.chunk_id, anchor.containment_rate))
        citations.append(_build_citation(passage, anchor, answer_text))
    return citations, blocked, statuses, failure_containment


# --------------------------------------------------------------------------- #
# Decision capture (refusal + citation gate; composition's emit is the composer's)
# --------------------------------------------------------------------------- #


def _query_sha(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]


def _emit_refusal(
    capture: Optional[Any],
    *,
    course_slug: str,
    query_sha: str,
    reason_code: str,
    signals: Dict[str, Any],
    policy_version: str,
    engine: str,
    embedding_model_id: Optional[str],
) -> None:
    if capture is None:
        return
    top = signals.get("top_score")
    n_above = signals.get("n_above_floor")
    if reason_code == REASON_LOW_CONFIDENCE:
        which = (
            f"top_score={top} below floor; n_above_floor={n_above}"
            if top is not None
            else "low retrieval confidence"
        )
    else:
        which = "model set not_in_course=true on the composed envelope"
    rationale = (
        f"refused query {query_sha} (course={course_slug or '?'}, "
        f"engine={engine}) reason={reason_code}: {which}; "
        f"policy_version={policy_version} "
        f"embedding_model_id={embedding_model_id} signals={signals}"
    )
    _safe_log(
        capture,
        decision_type=DECISION_TYPE_REFUSAL,
        decision=f"refuse:{reason_code}",
        rationale=rationale,
        context=f"reason={reason_code}",
    )


def _emit_citation_gate(
    capture: Optional[Any],
    *,
    course_slug: str,
    query_sha: str,
    chunkset_kind: str,
    per_citation_status: Sequence[Tuple[str, str]],
    blocked_chunk_ids: Sequence[str],
    failure_containment: Sequence[Tuple[str, float]],
) -> None:
    if capture is None:
        return
    summary = ", ".join(f"{cid}:{status}" for cid, status in per_citation_status)
    blocked_str = ", ".join(blocked_chunk_ids) if blocked_chunk_ids else "none"
    fail_cont = (
        ", ".join(f"{cid}@{rate}" for cid, rate in failure_containment)
        if failure_containment
        else "none"
    )
    outcome = "blocked" if blocked_chunk_ids else "passed"
    rationale = (
        f"citation gate {outcome} for query {query_sha} "
        f"(course={course_slug or '?'}, chunkset_kind={chunkset_kind}); "
        f"per_citation=[{summary}] blocked=[{blocked_str}] "
        f"failure_containment=[{fail_cont}]"
    )
    _safe_log(
        capture,
        decision_type=DECISION_TYPE_CITATION_GATE,
        decision=f"citation_gate:{outcome}",
        rationale=rationale,
        context=f"blocked={len(blocked_chunk_ids)}",
    )


def _emit_citation_prune(
    capture: Optional[Any],
    *,
    course_slug: str,
    query_sha: str,
    engine: str,
    mode: str,
    outcome: str,
    report: AttributionReport,
    kept_ids: Sequence[str],
    pruned_ids: Sequence[str],
    added_ids: Sequence[str],
    inline_vetoed_ids: Sequence[str],
    add_min_shingle: float = ADD_MIN_SHINGLE,
) -> None:
    """One capture covering the whole prune+add decision (plan §2.5).

    Rationale interpolates dynamic, replayable signals: query sha, course,
    engine, claim count, per-citation best-support ``[chunk_id@shingle/coverage
    *N]`` over the gate-eligible set, the thresholds + mode, and the kept /
    pruned / added / inline-vetoed id sets.
    """
    if capture is None:
        return

    def _support_str(cid: str) -> str:
        s = report.supports.get(cid)
        if s is None:
            return f"{cid}@?/?"
        flag = "cited" if s.cited else "uncited"
        return (
            f"{cid}@{s.best_shingle}/{s.best_coverage}"
            f"*{s.supported_claim_count}({flag})"
        )

    per_support = ", ".join(_support_str(cid) for cid in report.supports)
    kept_str = ", ".join(kept_ids) if kept_ids else "none"
    pruned_str = ", ".join(pruned_ids) if pruned_ids else "none"
    added_str = ", ".join(added_ids) if added_ids else "none"
    veto_str = ", ".join(inline_vetoed_ids) if inline_vetoed_ids else "none"
    rationale = (
        f"citation prune+add {outcome} for query {query_sha} "
        f"(course={course_slug or '?'}, engine={engine}, mode={mode}); "
        f"n_claims={report.n_claims} "
        f"min_overlap={report.min_overlap} shingle_size={ATTRIBUTION_SHINGLE_SIZE} "
        f"add_min_shingle={add_min_shingle}; "
        f"per_citation=[{per_support}] "
        f"kept=[{kept_str}] pruned=[{pruned_str}] added=[{added_str}] "
        f"inline_vetoed=[{veto_str}]"
    )
    _safe_log(
        capture,
        decision_type=DECISION_TYPE_CITATION_PRUNE,
        decision=f"citation_prune:{outcome}",
        rationale=rationale,
        context=f"mode={mode} pruned={len(pruned_ids)} added={len(added_ids)}",
    )


def _safe_log(capture: Any, **kwargs: Any) -> None:
    try:
        capture.log_decision(**kwargs)
    except Exception:  # pragma: no cover - capture must never break the path
        pass


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Attribution-driven prune + add (post-citation-gate; never changes verdict)
# --------------------------------------------------------------------------- #


def _apply_attribution_excerpts(
    citations: List[Citation], report: AttributionReport
) -> List[Citation]:
    """Stamp ``supporting_excerpt`` + ``supported_claim_count`` (+ ``text_quote``
    when attribution found a supporting sentence) onto each kept citation.

    The attribution excerpt supersedes the legacy whole-answer
    ``_first_supporting_quote`` heuristic for ``text_quote`` only when present;
    a citation with no attributed sentence keeps its legacy quote. Pure: returns
    a NEW list of new Citation objects (the dataclass is frozen).
    """
    out: List[Citation] = []
    for cit in citations:
        support = report.supports.get(cit.chunk_id)
        if support is None:
            out.append(cit)
            continue
        excerpt = support.supporting_excerpt
        out.append(
            _dc_replace(
                cit,
                supporting_excerpt=excerpt,
                supported_claim_count=support.supported_claim_count,
                text_quote=excerpt if excerpt else cit.text_quote,
            )
        )
    return out


def _select_additions(
    report: AttributionReport,
    gate_eligible_by_id: Dict[str, RetrievedPassage],
    add_candidate_passages: Sequence[RetrievedPassage],
    kept_citations: Sequence[Citation],
    *,
    answer_text: Optional[str],
    course_dir: Path,
    chunkset_kind: str,
    containment_threshold: float,
    add_min_shingle: float = ADD_MIN_SHINGLE,
) -> Tuple[List[Citation], List[str]]:
    """Build high-precision added citations from UNCITED gate-eligible passages.

    An uncited passage qualifies as an addition for some claim iff it (a) clears
    the high ``ADD_MIN_SHINGLE`` floor on that claim AND (b) out-supports (strict
    >) the strongest KEPT citation's shingle on that same claim — additions are
    relative + precision-first. A qualifying passage must ALSO resolve through
    the SAME citation-anchor machinery the model-cited set passed (any resolved
    status); an unresolvable add candidate is dropped (never blocks). Capped at
    :data:`MAX_ADDED_CITATIONS`, ordered by support strength.

    Returns ``(added_citations, added_ids)``.
    """
    if not add_candidate_passages:
        return [], []

    # Per-claim best shingle among the KEPT citations (the bar to beat).
    kept_ids = {c.chunk_id for c in kept_citations}
    kept_best_shingle_per_claim: Dict[int, float] = {}
    for cid in kept_ids:
        s = report.supports.get(cid)
        if s is None:
            continue
        for ci in s.supported_claim_indexes:
            # Approx: use the passage's overall best shingle as its per-claim
            # ceiling (the report keeps best_shingle, not per-claim shingles).
            prev = kept_best_shingle_per_claim.get(ci, 0.0)
            if s.best_shingle > prev:
                kept_best_shingle_per_claim[ci] = s.best_shingle

    # Rank uncited candidates by (claims supported desc, best_shingle desc,
    # rank asc). Only those that clear the ADD floor AND out-support the kept
    # set on at least one claim are eligible.
    candidates: List[Tuple[int, float, int, str]] = []
    for passage in add_candidate_passages:
        cid = passage.chunk_id
        s = report.supports.get(cid)
        if s is None or s.cited or s.is_claimless:
            continue
        if s.best_shingle < add_min_shingle:
            continue
        # Out-support the kept set on at least one of this passage's claims.
        beats_kept = any(
            s.best_shingle > kept_best_shingle_per_claim.get(ci, 0.0)
            for ci in s.supported_claim_indexes
        )
        if not beats_kept:
            continue
        candidates.append(
            (-s.supported_claim_count, -s.best_shingle, s.rank, cid)
        )

    candidates.sort()

    added: List[Citation] = []
    added_ids: List[str] = []
    for _, _, _, cid in candidates:
        if len(added) >= MAX_ADDED_CITATIONS:
            break
        passage = gate_eligible_by_id.get(cid)
        if passage is None:
            continue
        # The addition must pass the SAME anchor machinery the cited set passed.
        anchor = resolve_citation_anchor(
            _passage_chunk_record(passage),
            course_dir,
            chunkset_kind=chunkset_kind,
            containment_threshold=containment_threshold,
        )
        if anchor.status not in _RESOLVED_STATUSES:
            # Unresolvable add candidate: drop it (an addition never blocks the
            # answer and never surfaces an unanchored citation).
            continue
        cit = _build_citation(passage, anchor, answer_text)
        support = report.supports.get(cid)
        excerpt = support.supporting_excerpt if support else None
        cit = _dc_replace(
            cit,
            supporting_excerpt=excerpt,
            supported_claim_count=support.supported_claim_count if support else 0,
            text_quote=excerpt if excerpt else cit.text_quote,
        )
        added.append(cit)
        added_ids.append(cid)
    return added, added_ids


def _order_citations(citations: List[Citation]) -> List[Citation]:
    """Sort by (supported_claim_count desc, retrieval-stable order).

    Strongest supporter leads (fixes the case-1 "weak citation listed first"
    pattern). The input order is the citation-gate order (model-cited order,
    then appended additions), which preserves retrieval-rank monotonicity well
    enough for the secondary key, so a stable sort on the negated support count
    is sufficient and deterministic.
    """
    return sorted(
        citations, key=lambda c: -int(c.supported_claim_count or 0)
    )


def _apply_citation_attribution(
    *,
    citations: List[Citation],
    cited_passages: Sequence[RetrievedPassage],
    gate_eligible_passages: Sequence[RetrievedPassage],
    answer_text: Optional[str],
    course_dir: Path,
    chunkset_kind: str,
    containment_threshold: float,
    mode: str,
    min_overlap: float,
    add_min_shingle: float,
    capture: Optional[Any],
    course_slug: str,
    query_sha: str,
    engine: str,
) -> Tuple[List[Citation], List[str]]:
    """Run attribution over the gate-eligible set; prune + add per policy.

    Returns ``(final_citations, warnings)``. NEVER changes the answer verdict:
    pruning keeps >= 1 citation (all-prune is a no-op); additions only ever add
    anchor-resolved uncited passages. In ``shadow`` mode the report + capture +
    excerpts are produced but citations are NOT mutated (excerpts/counts still
    stamp — they are display refinement, not a verdict change).
    """
    warnings: List[str] = []
    if mode == MODE_OFF:
        return citations, warnings

    cited_ids = [p.chunk_id for p in cited_passages]
    report = attribute_citations(
        answer_text,
        gate_eligible_passages,
        cited_ids,
        min_overlap=min_overlap,
    )

    # Always stamp excerpts/counts onto the model-cited citations (display
    # refinement; happens in every non-off mode incl. shadow).
    enriched = _apply_attribution_excerpts(citations, report)

    # Zero scorable claims → skip prune/add entirely (no signal).
    if report.n_claims == 0:
        warnings.append("claimless_prune_no_scorable_claims")
        _emit_citation_prune(
            capture, course_slug=course_slug, query_sha=query_sha, engine=engine,
            mode=mode, outcome=_PRUNE_OUTCOME_NO_CLAIMS, report=report,
            kept_ids=[c.chunk_id for c in enriched], pruned_ids=[], added_ids=[],
            inline_vetoed_ids=[], add_min_shingle=add_min_shingle,
        )
        return enriched, warnings

    # Inline-marker defense (plan §2.4): a cited chunk_id appearing verbatim in
    # the answer text is referenced inline — never prune it.
    answer_norm = (answer_text or "")
    claimless = report.cited_claimless_ids()
    inline_vetoed = [cid for cid in claimless if cid and cid in answer_norm]
    prunable = [cid for cid in claimless if cid not in inline_vetoed]

    n_cited = len(cited_ids)
    # All cited citations claim-less: USER POLICY (2026-06-11) — "no sources
    # rather than a misleading one". The original plan no-op'd here to avoid
    # emitting zero citations; the operator chose display honesty instead:
    # claim-less citations are pruned even when that empties the cited set
    # (additions below may still repopulate from uncited supporters). The
    # caller flips answered → answered_with_warnings when the final set is
    # empty, so the learner sees the unverified-support advisory, never a
    # citation that backs nothing. Verdict (answered-family) still never
    # changes to a refusal/block.
    would_prune_all = bool(prunable) and len(prunable) >= n_cited

    # --- Additions (run over UNCITED gate-eligible passages) ---------------- #
    by_id = {p.chunk_id: p for p in gate_eligible_passages}
    # The "kept" set the additions must out-support: cited minus prunable
    # (unless all-prune no-op, in which case all cited stay).
    pruned_ids = list(prunable)
    kept_after_prune = [c for c in enriched if c.chunk_id not in pruned_ids]
    added_citations, added_ids = _select_additions(
        report,
        by_id,
        gate_eligible_passages,
        kept_after_prune,
        answer_text=answer_text,
        course_dir=course_dir,
        chunkset_kind=chunkset_kind,
        containment_threshold=containment_threshold,
        add_min_shingle=add_min_shingle,
    )

    # --- Outcome + warnings ------------------------------------------------- #
    if would_prune_all and not (kept_after_prune or added_ids):
        warnings.append("pruned_all_claimless_citations")
        outcome = _PRUNE_OUTCOME_PRUNED_ALL
    elif pruned_ids or added_ids:
        outcome = _PRUNE_OUTCOME_PRUNED
    else:
        outcome = _PRUNE_OUTCOME_NOOP
    for cid in pruned_ids:
        warnings.append(f"pruned_claimless_citation:{cid}")
    for cid in added_ids:
        warnings.append(f"added_supporting_citation:{cid}")

    final_kept_ids = [c.chunk_id for c in kept_after_prune] + added_ids
    _emit_citation_prune(
        capture, course_slug=course_slug, query_sha=query_sha, engine=engine,
        mode=mode, outcome=outcome, report=report,
        kept_ids=final_kept_ids, pruned_ids=pruned_ids, added_ids=added_ids,
        inline_vetoed_ids=inline_vetoed, add_min_shingle=add_min_shingle,
    )

    # Shadow mode: capture + warnings + excerpts produced, citations NOT
    # mutated (no prune, no add). Excerpts/counts already stamped on `enriched`.
    if mode == MODE_SHADOW:
        return _order_citations(enriched), warnings

    # On mode: apply prune + add, then order (strongest supporter first).
    final = kept_after_prune + added_citations
    return _order_citations(final), warnings


# --------------------------------------------------------------------------- #
# NLI-based citation ADD (2026-06-12 under-citing investigation; shadow-default)
# --------------------------------------------------------------------------- #
#
# The lexical (shingle) ADD arm above is unsalvageable on paraphrase answers
# (median cited shingle 0.000 — zero adds even at bar 0.10). This arm instead
# derives ADD candidates from the groundedness scorer's per-claim NLI verdicts
# (REUSED — NLI is never run twice) and gates them through a COMPOSITE criterion
# the 2026-06-12 hand sample found separates 4/4 genuine from 3/3 false adds:
# entailment >= NLI_ADD_ENTAILMENT_FLOOR AND token_coverage >=
# NLI_ADD_TOKEN_COVERAGE_FLOOR AND every claim numeric literal present in the
# chunk AND the chunk anchors/resolves AND <= NLI_ADD_MAX_ADDED_CITATIONS adds.


def _nli_add_candidate_signals(
    groundedness_report: Optional[Any],
    *,
    cited_chunk_ids: set,
    gate_eligible_by_id: Dict[str, RetrievedPassage],
) -> List[Dict[str, Any]]:
    """Per-candidate composite-criterion signals derived from NLI verdicts.

    REUSES the groundedness scorer's per-claim verdicts (``best_chunk_cited`` is
    ``False`` ⇒ the claim is entailed by an UNCITED corpus chunk — the exact
    under-citing signal). For each such claim whose entailment clears
    :data:`NLI_ADD_ENTAILMENT_FLOOR`, computes the two independent lexical legs
    (token coverage + numeric-literal presence) against the supporting chunk's
    text. Returns one dict per candidate (chunk_id) carrying every leg's value +
    a boolean ``passes_composite`` (anchor resolution is enforced separately at
    the call site — it needs the course dir). Pure: no anchor I/O, no mutation.
    """
    if groundedness_report is None or not getattr(groundedness_report, "available", False):
        return []
    # Aggregate per supporting-chunk: the strongest entailing uncited claim and
    # whether the lexical legs hold for SOME entailing claim on that chunk.
    by_chunk: Dict[str, Dict[str, Any]] = {}
    for verdict in getattr(groundedness_report, "claims", []) or []:
        if getattr(verdict, "verdict", None) != "entailed":
            continue
        # Only uncited supporters are ADD candidates (best_chunk_cited is False;
        # None means the cited/uncited split wasn't requested — not a candidate).
        if getattr(verdict, "best_chunk_cited", None) is not False:
            continue
        chunk_id = getattr(verdict, "best_chunk_id", None)
        if not chunk_id or chunk_id in cited_chunk_ids:
            continue
        passage = gate_eligible_by_id.get(chunk_id)
        if passage is None:
            # The supporter is outside the renderer-included gate-eligible set;
            # an ADD must be anchorable from that set, so skip it.
            continue
        entailment = float(getattr(verdict, "entailment", 0.0) or 0.0)
        claim_text = str(getattr(verdict, "claim_text", "") or "")
        chunk_text = str(getattr(passage, "text", "") or "")
        coverage = passage_token_coverage(claim_text, chunk_text)
        numerics = claim_numeric_literals(claim_text)
        numerics_ok = claim_numerics_present(claim_text, chunk_text)
        ent_ok = entailment >= NLI_ADD_ENTAILMENT_FLOOR
        cov_ok = coverage >= NLI_ADD_TOKEN_COVERAGE_FLOOR
        passes = bool(ent_ok and cov_ok and numerics_ok)
        prev = by_chunk.get(chunk_id)
        # Keep the strongest-entailment claim as the representative signal, but
        # OR the composite pass across claims (any qualifying claim qualifies the
        # chunk) — and prefer a passing representative when one exists.
        if (
            prev is None
            or (passes and not prev["passes_composite"])
            or (passes == prev["passes_composite"] and entailment > prev["entailment"])
        ):
            by_chunk[chunk_id] = {
                "chunk_id": chunk_id,
                "claim_text": claim_text,
                "entailment": round(entailment, 6),
                "token_coverage": round(coverage, 6),
                "claim_numerics": numerics,
                "numerics_present": numerics_ok,
                "entailment_ok": ent_ok,
                "coverage_ok": cov_ok,
                "passes_composite": passes,
            }
    # Deterministic order: strongest entailment first.
    return sorted(by_chunk.values(), key=lambda d: -d["entailment"])


def _apply_nli_citation_add(
    *,
    citations: List[Citation],
    groundedness_report: Optional[Any],
    cited_chunk_ids: set,
    gate_eligible_passages: Sequence[RetrievedPassage],
    answer_text: Optional[str],
    course_dir: Path,
    chunkset_kind: str,
    containment_threshold: float,
    mode: str,
    capture: Optional[Any],
    course_slug: str,
    query_sha: str,
    engine: str,
) -> Tuple[List[Citation], List[str], Optional[Dict[str, Any]]]:
    """NLI-ADD pass (shadow/on/off). Runs strictly AFTER the lexical prune+add.

    ``off`` → no-op. ``shadow`` → compute would-adds, emit the capture +
    additive warnings + the diagnostics block, mutate NOTHING. ``on`` → actually
    add the anchor-resolved would-adds (sorted after the existing citations,
    capped at :data:`NLI_ADD_MAX_ADDED_CITATIONS`); ships dark behind the knob.

    Reuses ``groundedness_report``'s per-claim verdicts — NLI is NEVER run a
    second time. When groundedness is unavailable (deps absent, with_groundedness
    off, or the report degraded) shadow mode does nothing and logs WHY. NEVER
    changes the answer verdict.

    Returns ``(final_citations, warnings, diagnostics_or_None)``; ``diagnostics``
    is ``None`` only in ``off`` mode (every shadow/on call emits a block, even a
    skip).
    """
    warnings: List[str] = []
    if mode == MODE_OFF:
        return citations, warnings, None

    gate_eligible_by_id = {p.chunk_id: p for p in gate_eligible_passages}
    existing_ids = {c.chunk_id for c in citations}
    # The "cited" set the candidates must be uncited against = the model's
    # citations UNION whatever the lexical arm already surfaced (so the NLI arm
    # never duplicates a citation the lexical add just credited).
    suppress_ids = set(cited_chunk_ids) | existing_ids

    available = bool(groundedness_report is not None and getattr(groundedness_report, "available", False))
    if not available:
        # Honest no-op: shadow mode logs WHY (the criterion cannot be evaluated
        # without per-claim NLI verdicts).
        reason = (
            "with_groundedness_off_or_nli_unavailable"
            if groundedness_report is None
            else "groundedness_report_unavailable"
        )
        warnings.append(f"nli_citation_add_skipped:{reason}")
        _emit_nli_citation_add(
            capture, course_slug=course_slug, query_sha=query_sha, engine=engine,
            mode=mode, outcome="skipped_no_nli", candidates=[],
            would_add_ids=[], added_ids=[], reason=reason,
        )
        diagnostics = {
            "mode": mode,
            "outcome": "skipped_no_nli",
            "reason": reason,
            "candidates": [],
            "would_add_ids": [],
            "added_ids": [],
        }
        return citations, warnings, diagnostics

    candidates = _nli_add_candidate_signals(
        groundedness_report,
        cited_chunk_ids=suppress_ids,
        gate_eligible_by_id=gate_eligible_by_id,
    )

    # Composite-criterion survivors, capped. Anchor resolution is the final leg
    # (it needs course I/O) — a candidate that passes the lexical+NLI legs but
    # fails to anchor is recorded as anchor_resolved=False and never added.
    would_add_ids: List[str] = []
    added_citations: List[Citation] = []
    added_ids: List[str] = []
    for cand in candidates:
        cid = cand["chunk_id"]
        if not cand["passes_composite"]:
            cand["anchor_resolved"] = None  # not evaluated (composite failed)
            continue
        passage = gate_eligible_by_id.get(cid)
        if passage is None:
            cand["anchor_resolved"] = False
            continue
        anchor = resolve_citation_anchor(
            _passage_chunk_record(passage),
            course_dir,
            chunkset_kind=chunkset_kind,
            containment_threshold=containment_threshold,
        )
        anchor_ok = anchor.status in _RESOLVED_STATUSES
        cand["anchor_resolved"] = bool(anchor_ok)
        if not anchor_ok:
            continue
        would_add_ids.append(cid)
        if len(added_ids) < NLI_ADD_MAX_ADDED_CITATIONS:
            cit = _build_citation(passage, anchor, answer_text)
            cit = _dc_replace(
                cit,
                supporting_excerpt=cit.text_quote,
                supported_claim_count=1,
            )
            added_citations.append(cit)
            added_ids.append(cid)

    # Cap the would-add list the same way the add list is capped, so the shadow
    # signal reports what ON mode WOULD have done (not an unbounded count).
    capped_would_add = would_add_ids[:NLI_ADD_MAX_ADDED_CITATIONS]

    # PRUNE-TO-EMPTY EXCLUSION (2026-06-12 under-citing investigation): on
    # answers whose citation set is EMPTY (the lexical prune-to-empty policy —
    # "no sources beats a misleading source"), the entailed-uncited claims were
    # hand-judged 3/3 question-induced false adds. ON mode NEVER restores a
    # citation to a zero-citation answer; shadow still computes the would-adds
    # but classifies them as a warning-class signal (a distinct outcome +
    # warning prefix) so eval aggregation reads them as suspect, not recovered.
    zero_citation_answer = not citations
    if zero_citation_answer:
        outcome = (
            "would_add_on_zero_citation_answer" if capped_would_add else "noop"
        )
        actually_added_ids = []
        added_citations = []
        for cid in capped_would_add:
            warnings.append(f"nli_would_add_on_zero_citation_answer:{cid}")
    else:
        outcome = "would_add" if capped_would_add else "noop"
        # ``added_ids`` reflects what was ACTUALLY added: empty in shadow
        # (shadow never mutates), the real add set in ON. ``would_add_ids`` is
        # what ON WOULD add (the shadow signal) and is identical to
        # ``added_ids`` under ON.
        actually_added_ids = added_ids if mode == MODE_ON else []

        if mode == MODE_SHADOW:
            for cid in capped_would_add:
                warnings.append(f"nli_would_add_citation:{cid}")
        else:  # MODE_ON
            for cid in actually_added_ids:
                warnings.append(f"nli_added_citation:{cid}")

    _emit_nli_citation_add(
        capture, course_slug=course_slug, query_sha=query_sha, engine=engine,
        mode=mode, outcome=outcome, candidates=candidates,
        would_add_ids=capped_would_add, added_ids=actually_added_ids, reason=None,
    )

    diagnostics = {
        "mode": mode,
        "outcome": outcome,
        "reason": None,
        "candidates": list(candidates),
        "would_add_ids": list(capped_would_add),
        "added_ids": list(actually_added_ids),
        "zero_citation_answer": zero_citation_answer,
    }

    if mode == MODE_SHADOW or not added_citations:
        return citations, warnings, diagnostics

    # On mode: append the NLI adds AFTER the existing citations (the lexical arm
    # already ordered the kept+lexical-add set strongest-first; the NLI adds are
    # the precision tail and land last).
    return citations + added_citations, warnings, diagnostics


def _emit_nli_citation_add(
    capture: Optional[Any],
    *,
    course_slug: str,
    query_sha: str,
    engine: str,
    mode: str,
    outcome: str,
    candidates: Sequence[Dict[str, Any]],
    would_add_ids: Sequence[str],
    added_ids: Sequence[str],
    reason: Optional[str],
) -> None:
    """One capture for the NLI-ADD decision (dynamic, replayable rationale).

    Rationale interpolates the per-candidate composite-criterion signals
    (entailment / coverage / numeric-literal result / anchor resolution), the
    thresholds actually used, the would-add / added id sets, and the mode — so a
    post-hoc reader can replay why each uncited supporter was or wasn't credited.
    """
    if capture is None:
        return

    def _cand_str(c: Dict[str, Any]) -> str:
        nums = "/".join(c.get("claim_numerics", [])) or "-"
        return (
            f"{c['chunk_id']}@ent={c['entailment']}"
            f"(ok={c['entailment_ok']}) cov={c['token_coverage']}"
            f"(ok={c['coverage_ok']}) nums=[{nums}]present={c['numerics_present']}"
            f" anchor={c.get('anchor_resolved')} composite={c['passes_composite']}"
        )

    per_cand = "; ".join(_cand_str(c) for c in candidates) or "none"
    would_str = ", ".join(would_add_ids) if would_add_ids else "none"
    added_str = ", ".join(added_ids) if added_ids else "none"
    reason_str = f" reason={reason}" if reason else ""
    rationale = (
        f"nli citation add {outcome} for query {query_sha} "
        f"(course={course_slug or '?'}, engine={engine}, mode={mode}){reason_str}; "
        f"composite thresholds ent>={NLI_ADD_ENTAILMENT_FLOOR} "
        f"cov>={NLI_ADD_TOKEN_COVERAGE_FLOOR} numerics_present cap={NLI_ADD_MAX_ADDED_CITATIONS}; "
        f"candidates=[{per_cand}] "
        f"would_add=[{would_str}] added=[{added_str}]"
    )
    _safe_log(
        capture,
        decision_type=DECISION_TYPE_NLI_CITATION_ADD,
        decision=f"nli_citation_add:{outcome}",
        rationale=rationale,
        context=f"mode={mode} would_add={len(would_add_ids)} added={len(added_ids)}",
    )


def _resolve_completeness_recheck() -> bool:
    """Resolve the completeness-recheck toggle (``ED4ALL_ANSWER_COMPLETENESS_RECHECK``).

    Default ON. A small (7B-Q4) model non-deterministically drops a sub-question
    of a multi-part question even when the grounding is its top passage; the
    recheck catches that and re-asks once. Falsey values (``0``/``false``/``no``/
    ``off``) disable. Parse-with-fallback (mirrors the other answer-path knobs).
    """
    import os

    raw = os.environ.get("ED4ALL_ANSWER_COMPLETENESS_RECHECK")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _emit_completeness_recheck(
    capture: Optional[Any],
    *,
    course_slug: str,
    query_sha: str,
    engine: str,
    outcome: str,
    parts: Sequence[Any],
    uncovered_before: Sequence[str],
    uncovered_after: Sequence[str],
    reasked: bool,
    adopted: bool,
    reason: Optional[str] = None,
) -> None:
    """One capture for the completeness-recheck decision (dynamic rationale).

    Interpolates the per-part addressed/grounded signals, the uncovered-but-
    grounded set that drove (or didn't drive) the re-ask, whether a re-ask fired
    and was adopted, and the residual uncovered set after — so a reader can
    replay why the answer was (or was not) extended.
    """
    if capture is None:
        return

    def _part_str(p: Any) -> str:
        sig = getattr(p, "addressed_signals", {}) or {}
        gsig = getattr(p, "grounded_signals", {}) or {}
        return (
            f"{getattr(p, 'text', '')[:48]!r}"
            f"(addr={getattr(p, 'addressed', None)}"
            f"@shingle={sig.get('shingle')},cov={sig.get('token_coverage')};"
            f"grnd={getattr(p, 'grounded', None)}"
            f"@cov={gsig.get('best_token_coverage')},chunk={gsig.get('best_chunk_id')})"
        )

    per_part = "; ".join(_part_str(p) for p in parts) or "none"
    before = ", ".join(uncovered_before) if uncovered_before else "none"
    after = ", ".join(uncovered_after) if uncovered_after else "none"
    reason_str = f" reason={reason}" if reason else ""
    rationale = (
        f"completeness recheck {outcome} for query {query_sha} "
        f"(course={course_slug or '?'}, engine={engine}){reason_str}; "
        f"reasked={reasked} adopted={adopted}; "
        f"uncovered_before=[{before}] uncovered_after=[{after}]; "
        f"parts=[{per_part}]"
    )
    _safe_log(
        capture,
        decision_type=DECISION_TYPE_COMPLETENESS_RECHECK,
        decision=f"completeness_recheck:{outcome}",
        rationale=rationale,
        context=f"reasked={reasked} adopted={adopted} uncovered={len(uncovered_before)}",
    )


def _merge_completed_answer(
    original: ComposedAnswer, reask: ComposedAnswer
) -> ComposedAnswer:
    """Merge a re-ask's missing-part answer onto the original (additive).

    The completeness directive is additive ("answer the missing part, do not
    restate the parts you already answered"), so the re-ask's ``answer_text``
    covers ONLY the previously-dropped sub-question. We append it to the
    original prose and union the citation ids (original order first), keeping
    the already-good first part verbatim. The merged answer then flows through
    the citation gate + prune + groundedness exactly like a single-pass answer.
    """
    merged_text = (original.answer_text or "").rstrip() + "\n\n" + (
        reask.answer_text or ""
    ).strip()
    merged_ids = list(
        dict.fromkeys([*original.cited_chunk_ids, *reask.cited_chunk_ids])
    )
    return _dc_replace(
        original,
        answer_text=merged_text.strip(),
        cited_chunk_ids=merged_ids,
        attempts=original.attempts + reask.attempts,
        latency_ms=original.latency_ms + reask.latency_ms,
        raw_response_len=original.raw_response_len + reask.raw_response_len,
    )


def _apply_completeness_recheck(
    *,
    query: str,
    composed: ComposedAnswer,
    passages: Sequence[RetrievedPassage],
    client: Any,
    capture: Optional[Any],
    course_slug: str,
    query_sha: str,
    engine: str,
    max_passages: int,
) -> Tuple[ComposedAnswer, List[str]]:
    """Single bounded re-ask when a multi-part question left a grounded part unanswered.

    Pure-lexical detection (no model load): a sub-question that the answer does
    NOT address but a passage DOES support triggers one re-ask through the
    composer with the additive completeness directive. NEVER regresses — a
    re-ask that refuses / returns no citations / errors keeps the original
    answered response. The re-ask is itself NOT rechecked (no recursion).
    """
    if not composed.answer_text:
        return composed, []

    result = analyze_completeness(query, composed.answer_text, passages)
    uncovered = result.uncovered_but_grounded
    if not result.is_multipart or not uncovered:
        _emit_completeness_recheck(
            capture,
            course_slug=course_slug,
            query_sha=query_sha,
            engine=engine,
            outcome="noop:single_part" if not result.is_multipart else "noop:all_addressed_or_ungrounded",
            parts=result.parts,
            uncovered_before=uncovered,
            uncovered_after=uncovered,
            reasked=False,
            adopted=False,
        )
        return composed, []

    directive = COMPLETENESS_REMEDIATION_DIRECTIVE.format(
        uncovered_parts="; ".join(uncovered)
    )
    try:
        reask = compose_answer(
            query,
            passages,
            client=client,
            capture=capture,
            course_code=course_slug,
            max_passages=max_passages,
            extra_directive=directive,
        )
    except Exception as exc:  # noqa: BLE001 - enhancement must never break a good answer
        _emit_completeness_recheck(
            capture,
            course_slug=course_slug,
            query_sha=query_sha,
            engine=engine,
            outcome="reask:failed",
            parts=result.parts,
            uncovered_before=uncovered,
            uncovered_after=uncovered,
            reasked=True,
            adopted=False,
            reason=type(exc).__name__,
        )
        return composed, ["completeness_reask_failed"]

    # Never regress an answered response: adopt only a strictly-valid re-ask.
    if reask.not_in_course or not reask.cited_chunk_ids or not reask.answer_text:
        _emit_completeness_recheck(
            capture,
            course_slug=course_slug,
            query_sha=query_sha,
            engine=engine,
            outcome="reask:rejected_no_extension",
            parts=result.parts,
            uncovered_before=uncovered,
            uncovered_after=uncovered,
            reasked=True,
            adopted=False,
        )
        return composed, []

    merged = _merge_completed_answer(composed, reask)
    post = analyze_completeness(query, merged.answer_text, passages)
    still = post.uncovered_but_grounded
    _emit_completeness_recheck(
        capture,
        course_slug=course_slug,
        query_sha=query_sha,
        engine=engine,
        outcome="reask:covered" if not still else "reask:still_uncovered",
        parts=post.parts,
        uncovered_before=uncovered,
        uncovered_after=still,
        reasked=True,
        adopted=True,
    )
    return merged, ["completeness_reasked"]


# --------------------------------------------------------------------------- #
# The pipeline (D9)
# --------------------------------------------------------------------------- #


def answer_course_question(
    repo_root: Path,
    course_slug: str,
    query: str,
    *,
    engine: str = "lexical",
    limit: int = 8,
    client: Optional[Any] = None,
    refusal_policy: Optional[RefusalPolicy] = None,
    validate_citations: bool = True,
    with_groundedness: bool = False,
    capture: Optional[Any] = None,
    chunkset_kind: Optional[str] = None,
    containment_threshold: float = 0.85,
    max_passages: int = 8,
    prune_mode: Optional[str] = None,
    prune_min_overlap: Optional[float] = None,
    completeness_recheck: Optional[bool] = None,
) -> GroundedAnswer:
    """Answer a single-course question, grounded + citation-gated.

    The single entry point WS4 + the CLI consume. Flow: retrieve (lazy LibV2
    import; typed semantic errors propagate, never a silent engine downgrade)
    -> evaluate confidence / refuse PRE-LLM -> compose_answer -> model-side
    ``not_in_course`` -> citation gate -> ``GroundedAnswer``. No canned-answer
    fallback; backend/index absence raises a typed error.
    """
    start = time.monotonic()
    query_sha = _query_sha(query)

    libv2_root = _libv2_root(repo_root)
    course_dir = libv2_root / "courses" / course_slug

    # Select the refusal policy. The SEMANTIC and HYBRID-RRF engines both pin PER
    # EMBEDDING MODEL, so we resolve the live index's embedding_model_id from its
    # manifest and route through resolve_policy((engine, model)). hybrid-rrf's
    # fused RRF score is NOT a cosine, but its semantic arm — and therefore the
    # whole fused-score distribution the threshold is measured against — is a
    # function of the embedder, so a hybrid pin measured on bge-large is no more
    # valid for MiniLM than a semantic one. Keying both by the index model keeps
    # the pin honest. A pinned pair returns the measured threshold; an unknown
    # model falls back to the v0-uncalibrated default (never a stale threshold);
    # lexical is model-agnostic (model id None). An explicit refusal_policy
    # override wins verbatim.
    if refusal_policy is not None:
        policy = refusal_policy
    else:
        embedding_model_id = (
            _vector_index_embedding_model_id(libv2_root, course_slug)
            if engine in ("semantic", "hybrid-rrf")
            else None
        )
        policy = resolve_policy(engine, embedding_model_id)
    if chunkset_kind is None:
        # For the semantic / hybrid-rrf engines the vector index's manifest is
        # the authoritative chunkset — read it so retrieval, hydration, and the
        # citation gate all resolve against the chunkset the index was built
        # over (a course may carry both dart_chunks/ and an imscc-pinned index;
        # the directory heuristic would otherwise pick the wrong one and block
        # every citation as source_page_missing). The lexical engine keeps the
        # directory-presence inference.
        if engine in ("semantic", "hybrid-rrf"):
            chunkset_kind = _vector_index_chunkset_kind(libv2_root, course_slug)
        if chunkset_kind is None:
            chunkset_kind = _infer_chunkset_kind(libv2_root, course_slug)

    # 1) Retrieve (typed errors propagate; no silent downgrade).
    #
    # Cross-encoder reranker hook (ED4ALL_RERANK_PROVIDER, default OFF). When
    # unset/empty this is a strict no-op: ``reranker_configured()`` returns
    # None, the over-fetch is skipped (retrieve_limit == limit), and the
    # retrieved order + native scores are byte-identical to the legacy path.
    # When configured, we over-fetch a candidate pool, re-score (query, passage)
    # pairs, and REORDER + TRIM back to ``limit`` STRICTLY BEFORE the refusal
    # gate — preserving each passage's native first-stage score so the per-
    # engine refusal threshold stays calibrated. Fail-OPEN.
    rerank_provider = reranker_configured()
    retrieve_limit = limit
    if rerank_provider is not None:
        retrieve_limit = max(limit, resolve_candidate_pool())
    results = _retrieve(
        libv2_root, course_slug, query, engine=engine, limit=retrieve_limit
    )
    passages = [
        RetrievedPassage.from_retrieval_result(r, engine=engine) for r in results
    ]
    if rerank_provider is not None:
        passages = maybe_rerank(
            query,
            passages,
            top_k=limit,
            capture=capture,
            course_slug=course_slug,
            query_sha=query_sha,
        )

    # 2) Pre-LLM refusal on low confidence (zero client calls on refusal).
    verdict = evaluate_confidence(passages, policy)
    confidence_payload = verdict.to_dict()
    refusal_check = should_refuse(passages, policy)
    if refusal_check.refuse:
        _emit_refusal(
            capture,
            course_slug=course_slug,
            query_sha=query_sha,
            reason_code=REASON_LOW_CONFIDENCE,
            signals=verdict.signals,
            policy_version=policy.policy_version,
            engine=engine,
            embedding_model_id=policy.embedding_model_id,
        )
        return _refused(
            status=STATUS_REFUSED_LOW_CONFIDENCE,
            query=query,
            course_slug=course_slug,
            engine=engine,
            refusal=refusal_check.to_dict(),
            confidence=confidence_payload,
            model_id=None,
            prompt_version=None,
            start=start,
        )

    # 3) Compose the answer (THE LLM call site, owned by the composer).
    if client is None:
        from lib.retrieval.answer_backend import build_answer_client

        client = build_answer_client(capture=capture)

    composed: ComposedAnswer = compose_answer(
        query,
        passages,
        client=client,
        capture=capture,
        course_code=course_slug,
        max_passages=max_passages,
    )

    # 4) Model-side not_in_course → refusal.
    if composed.not_in_course:
        model_refusal = {
            "refuse": True,
            "reason_code": REASON_NOT_IN_COURSE_MODEL,
            "signals": dict(verdict.signals),
            "policy_version": policy.policy_version,
            "engine": engine,
            "embedding_model_id": policy.embedding_model_id,
        }
        _emit_refusal(
            capture,
            course_slug=course_slug,
            query_sha=query_sha,
            reason_code=REASON_NOT_IN_COURSE_MODEL,
            signals=verdict.signals,
            policy_version=policy.policy_version,
            engine=engine,
            embedding_model_id=policy.embedding_model_id,
        )
        return _refused(
            status=STATUS_REFUSED_NOT_IN_COURSE,
            query=query,
            course_slug=course_slug,
            engine=engine,
            refusal=model_refusal,
            confidence=confidence_payload,
            model_id=composed.model_id,
            prompt_version=composed.prompt_version,
            start=start,
        )

    # An answer without citations is a contradiction in terms (D5).
    if not composed.cited_chunk_ids:
        return _blocked(
            status=STATUS_BLOCKED_INVALID_CITATION,
            query=query,
            course_slug=course_slug,
            engine=engine,
            confidence=confidence_payload,
            model_id=composed.model_id,
            prompt_version=composed.prompt_version,
            warnings=["empty_citations"],
            start=start,
        )

    # 4b) Completeness recheck — single bounded re-ask for a multi-part question
    #     that left a grounded sub-question unanswered (a 7B-Q4 failure mode even
    #     when the grounding is the top passage). Runs BEFORE the citation gate so
    #     the merged answer + unioned citations flow through gate → prune →
    #     groundedness exactly like a single-pass answer. Never regresses an
    #     answered response; pure-lexical detection (no extra model load).
    recheck_warnings: List[str] = []
    do_recheck = (
        _resolve_completeness_recheck()
        if completeness_recheck is None
        else completeness_recheck
    )
    if do_recheck:
        composed, recheck_warnings = _apply_completeness_recheck(
            query=query,
            composed=composed,
            passages=passages,
            client=client,
            capture=capture,
            course_slug=course_slug,
            query_sha=query_sha,
            engine=engine,
            max_passages=max_passages,
        )

    cited_set = set(composed.cited_chunk_ids)
    by_id = {p.chunk_id: p for p in passages}
    # Order-preserving cited passage list (composer already validated ids ∈
    # retrieved set, so every id resolves; defensive .get for robustness).
    cited_passages = [by_id[cid] for cid in composed.cited_chunk_ids if cid in by_id]

    # 5) Citation gate (test-only bypass via validate_citations=False).
    if not validate_citations:
        citations = [
            _build_citation(
                p,
                resolve_citation_anchor(
                    _passage_chunk_record(p),
                    course_dir,
                    chunkset_kind=chunkset_kind,
                    containment_threshold=containment_threshold,
                ),
                composed.answer_text,
            )
            for p in cited_passages
        ]
        return _answered(
            query=query,
            course_slug=course_slug,
            engine=engine,
            answer_text=composed.answer_text,
            citations=citations,
            confidence=confidence_payload,
            model_id=composed.model_id,
            prompt_version=composed.prompt_version,
            groundedness=None,
            warnings=list(recheck_warnings),
            start=start,
        )

    citations, blocked, statuses, failure_containment = _run_citation_gate(
        cited_passages,
        course_dir,
        chunkset_kind=chunkset_kind,
        answer_text=composed.answer_text,
        containment_threshold=containment_threshold,
    )
    _emit_citation_gate(
        capture,
        course_slug=course_slug,
        query_sha=query_sha,
        chunkset_kind=chunkset_kind,
        per_citation_status=statuses,
        blocked_chunk_ids=blocked,
        failure_containment=failure_containment,
    )

    if blocked:
        # Block emission: withhold the answer text AND the citations (the
        # contract: citations are non-empty only for answered* statuses). The
        # blocked chunk_ids ride the warnings + the capture stream so the
        # support story is auditable without surfacing a partial answer.
        return _blocked(
            status=STATUS_BLOCKED_CITATION_GATE,
            query=query,
            course_slug=course_slug,
            engine=engine,
            confidence=confidence_payload,
            model_id=composed.model_id,
            prompt_version=composed.prompt_version,
            warnings=[f"blocked_citation:{cid}" for cid in blocked],
            start=start,
        )

    # 6) Attribution-driven prune + add (post-gate; never changes the verdict).
    #    Runs over ALL gate-eligible passages — the renderer-INCLUDED set the
    #    model could have cited (composed.allowed_chunk_ids), in retrieval-rank
    #    order — so an uncited definitional supporter the model under-cited can
    #    be credited, and a model-cited chunk supporting zero claims dropped.
    warnings: List[str] = list(recheck_warnings)
    mode = resolve_prune_mode(prune_mode)
    min_overlap = resolve_min_overlap(prune_min_overlap)
    add_min_shingle = resolve_add_min_shingle()
    allowed = composed.allowed_chunk_ids or composed.cited_chunk_ids
    gate_eligible_passages = [by_id[cid] for cid in allowed if cid in by_id]
    citations, prune_warnings = _apply_citation_attribution(
        citations=citations,
        cited_passages=cited_passages,
        gate_eligible_passages=gate_eligible_passages,
        answer_text=composed.answer_text,
        course_dir=course_dir,
        chunkset_kind=chunkset_kind,
        containment_threshold=containment_threshold,
        mode=mode,
        min_overlap=min_overlap,
        add_min_shingle=add_min_shingle,
        capture=capture,
        course_slug=course_slug,
        query_sha=query_sha,
        engine=engine,
    )
    warnings.extend(prune_warnings)

    # 7) Optional groundedness (advisory; never blocks — D6 / scorer v2). The
    #    metric measures fabrication-vs-corpus, NOT citation-selection quality
    #    (that is measured separately by the attribution prune+add pass above).
    #    So the evidence pool is the FULL gate-eligible set — the renderer-
    #    included passages the model could have cited — and each claim's
    #    best_chunk_cited flags whether its supporting chunk was actually cited.
    #    A corpus supporter the model under-cited can still entail a claim
    #    (v2 wider-pool semantics, plan §2.2 honesty-over-flattery).
    groundedness_payload: Optional[Dict[str, Any]] = None
    groundedness_report: Optional[Any] = None
    status = STATUS_ANSWERED
    final_cited_ids = {c.chunk_id for c in citations}
    if with_groundedness:
        groundedness_passages = list(gate_eligible_passages) or cited_passages
        groundedness_payload, gw, groundedness_report = _score_groundedness(
            composed.answer_text,
            groundedness_passages,
            cited_chunk_ids=final_cited_ids,
        )
        warnings.extend(gw)
        # v2 contradicted_count is computed under windowed / single-topic
        # semantics (change 3 removed whole-chunk false contradictions), so this
        # flip now fires far less. Policy unchanged otherwise: a v2 contradicted
        # claim still escalates to answered_with_warnings.
        if "contradicted_claim" in gw:
            status = STATUS_ANSWERED_WITH_WARNINGS

    # 7b) NLI-based citation ADD (shadow-default; off on the default path).
    #     Runs strictly AFTER the lexical prune+add AND after groundedness, and
    #     REUSES the per-claim NLI verdicts (never re-runs NLI). In shadow mode
    #     it mutates nothing — only the capture + diagnostics + warnings fire.
    #     When with_groundedness is off (or NLI degraded) the report is None and
    #     shadow mode logs WHY. NEVER changes the answer verdict.
    nli_add_diagnostics: Optional[Dict[str, Any]] = None
    nli_add_mode = resolve_nli_add_mode()
    if nli_add_mode != MODE_OFF:
        citations, nli_warnings, nli_add_diagnostics = _apply_nli_citation_add(
            citations=citations,
            groundedness_report=groundedness_report,
            cited_chunk_ids=final_cited_ids,
            gate_eligible_passages=gate_eligible_passages,
            answer_text=composed.answer_text,
            course_dir=course_dir,
            chunkset_kind=chunkset_kind,
            containment_threshold=containment_threshold,
            mode=nli_add_mode,
            capture=capture,
            course_slug=course_slug,
            query_sha=query_sha,
            engine=engine,
        )
        warnings.extend(nli_warnings)

    # USER POLICY: an answered response whose every citation was pruned as
    # claim-less ships with NO sources + the unverified-support advisory
    # (answered_with_warnings) rather than a misleading citation.
    if not citations and "pruned_all_claimless_citations" in warnings:
        status = STATUS_ANSWERED_WITH_WARNINGS

    return _answered(
        query=query,
        course_slug=course_slug,
        engine=engine,
        answer_text=composed.answer_text,
        citations=citations,
        confidence=confidence_payload,
        model_id=composed.model_id,
        prompt_version=composed.prompt_version,
        groundedness=groundedness_payload,
        warnings=warnings,
        start=start,
        status=status,
        nli_citation_add=nli_add_diagnostics,
    )


def _score_groundedness(
    answer_text: Optional[str],
    cited_passages: Sequence[RetrievedPassage],
    *,
    cited_chunk_ids: Optional[set] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str], Optional[Any]]:
    """Optional per-answer groundedness (advisory). NLI absent → null block.

    Lazy import of E7's ``groundedness`` module so the ~750 MB DeBERTa load
    never sneaks onto the default query path (it is only reached under
    ``with_groundedness=True``). ``cited_chunk_ids`` (the model's actual
    citations) lets the v2 scorer flag each claim's supporting chunk as
    cited/uncited while scoring against the wider gate-eligible evidence pool.

    Returns ``(report_dict_or_None, warnings, report_obj_or_None)`` — the third
    element is the live :class:`GroundednessReport` (additive return) so the
    NLI-ADD pass can reuse its per-claim verdicts WITHOUT re-running NLI. It is
    ``None`` whenever the dict is ``None`` (deps absent / scoring raised).
    """
    try:
        from lib.retrieval.groundedness import score_groundedness
    except Exception:
        return None, [], None
    try:
        report = score_groundedness(
            answer_text or "", cited_passages, cited_chunk_ids=cited_chunk_ids
        )
    except Exception:
        return None, [], None
    report_dict = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    warnings: List[str] = []
    if report_dict.get("contradicted_count", 0):
        warnings.append("contradicted_claim")
    return report_dict, warnings, report


# --------------------------------------------------------------------------- #
# Result assembly helpers
# --------------------------------------------------------------------------- #


def _latency_ms(start: float) -> float:
    return round((time.monotonic() - start) * 1000.0, 3)


def _answered(
    *,
    query: str,
    course_slug: str,
    engine: str,
    answer_text: Optional[str],
    citations: List[Citation],
    confidence: Dict[str, Any],
    model_id: Optional[str],
    prompt_version: Optional[str],
    groundedness: Optional[Dict[str, Any]],
    warnings: List[str],
    start: float,
    status: str = STATUS_ANSWERED,
    nli_citation_add: Optional[Dict[str, Any]] = None,
) -> GroundedAnswer:
    return GroundedAnswer(
        status=status,
        query=query,
        course_slug=course_slug,
        engine=engine,
        answer_text=answer_text,
        citations=citations,
        refusal=None,
        confidence=confidence,
        groundedness=groundedness,
        warnings=list(warnings),
        model_id=model_id,
        prompt_version=prompt_version,
        generated_at=_utcnow_iso(),
        latency_ms=_latency_ms(start),
        nli_citation_add=nli_citation_add,
    )


def _refused(
    *,
    status: str,
    query: str,
    course_slug: str,
    engine: str,
    refusal: Dict[str, Any],
    confidence: Dict[str, Any],
    model_id: Optional[str],
    prompt_version: Optional[str],
    start: float,
) -> GroundedAnswer:
    return GroundedAnswer(
        status=status,
        query=query,
        course_slug=course_slug,
        engine=engine,
        answer_text=None,
        citations=[],
        refusal=refusal,
        confidence=confidence,
        groundedness=None,
        warnings=[],
        model_id=model_id,
        prompt_version=prompt_version,
        generated_at=_utcnow_iso(),
        latency_ms=_latency_ms(start),
    )


def _blocked(
    *,
    status: str,
    query: str,
    course_slug: str,
    engine: str,
    confidence: Dict[str, Any],
    model_id: Optional[str],
    prompt_version: Optional[str],
    warnings: List[str],
    start: float,
) -> GroundedAnswer:
    # The answer text AND citations are withheld for every blocked status (D5):
    # citations are non-empty only for answered* statuses.
    return GroundedAnswer(
        status=status,
        query=query,
        course_slug=course_slug,
        engine=engine,
        answer_text=None,
        citations=[],
        refusal=None,
        confidence=confidence,
        groundedness=None,
        warnings=list(warnings),
        model_id=model_id,
        prompt_version=prompt_version,
        generated_at=_utcnow_iso(),
        latency_ms=_latency_ms(start),
    )

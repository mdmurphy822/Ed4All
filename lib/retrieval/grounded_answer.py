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

from lib.retrieval._text import normalize_ws
from lib.retrieval.answer_composer import (
    ComposedAnswer,
    RetrievedPassage,
    compose_answer,
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

__all__ = [
    "Citation",
    "GroundedAnswer",
    "answer_course_question",
    "page_label_for_source",
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
DECISION_PHASE = "libv2-answer"


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

    def to_dict(self) -> Dict[str, Any]:
        return {
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
        }


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


def _safe_log(capture: Any, **kwargs: Any) -> None:
    try:
        capture.log_decision(**kwargs)
    except Exception:  # pragma: no cover - capture must never break the path
        pass


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    results = _retrieve(
        libv2_root, course_slug, query, engine=engine, limit=limit
    )
    passages = [
        RetrievedPassage.from_retrieval_result(r, engine=engine) for r in results
    ]

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
            warnings=[],
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

    # 6) Optional groundedness (advisory; never blocks — D6).
    groundedness_payload: Optional[Dict[str, Any]] = None
    warnings: List[str] = []
    status = STATUS_ANSWERED
    if with_groundedness:
        groundedness_payload, gw = _score_groundedness(
            composed.answer_text, cited_passages
        )
        warnings.extend(gw)
        if "contradicted_claim" in gw:
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
    )


def _score_groundedness(
    answer_text: Optional[str],
    cited_passages: Sequence[RetrievedPassage],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Optional per-answer groundedness (advisory). NLI absent → null block.

    Lazy import of E7's ``groundedness`` module so the ~750 MB DeBERTa load
    never sneaks onto the default query path (it is only reached under
    ``with_groundedness=True``). Returns ``(report_dict_or_None, warnings)``.
    """
    try:
        from lib.retrieval.groundedness import score_groundedness
    except Exception:
        return None, []
    try:
        report = score_groundedness(answer_text or "", cited_passages)
    except Exception:
        return None, []
    report_dict = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    warnings: List[str] = []
    if report_dict.get("contradicted_count", 0):
        warnings.append("contradicted_claim")
    return report_dict, warnings


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

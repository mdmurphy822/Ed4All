"""Assessment-aware answering guard (L2) — "the tutor won't do your homework".

The learner ask path should help a student *study*, not hand them the answer to
a graded assessment item. This module detects when an incoming learner question
closely matches a known course assessment **stem** and — in ``on`` mode — swaps
the composed answer for a *redirect-with-hint* envelope: it never hard-refuses,
it points the learner at the relevant course sections to study instead.

Rollout is governed by a three-valued env flag, mirroring the
``ED4ALL_ANSWER_CITATION_PRUNE`` precedent
(:func:`lib.retrieval.citation_attribution.resolve_prune_mode`):

* ``off``   (default, byte-identical unset): the guard is inert — no stem load,
  no comparison, no decision capture, no change to the answer path.
* ``shadow``: the answer is composed + returned **normally**, but the guard
  still loads stems, compares, and RECORDS the would-have-matched signal (as an
  additive ``assessment_guard`` key on the result dict + a decision capture) so
  an operator can measure hit-rate before enabling enforcement.
* ``on``: on a match, the compose is **skipped** and a redirect-with-hint answer
  envelope is returned instead (concept-page citations reused from the retrieval
  machinery). A no-match answers normally.

The guard needs **stems only, never answers**. The QTI path routes through
``gui.services.quiz_service`` (``to_learner_json`` strips the key by contract);
the ``training_specs/assessments.json`` path reads only the ``stem`` strings.

ANTI-HOMEWORK NOTE: in ``on`` mode the LLM compose is genuinely skipped on a
match — the redirect envelope's "sections to study" citations come from a
retrieval-only pass (no answer is composed, so no answer can leak).

Import-light by contract: heavy deps (``quiz_service`` → Trainforge QTI parser,
the grounded-answer retrieval stack, the embedding client) are all lazy-imported
inside the function bodies, so this module imports cleanly for a unit test that
injects stems directly.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Flag + knob names / defaults (mirrors the citation-prune precedent)
# --------------------------------------------------------------------------- #

#: Three-valued rollout flag (off | shadow | on).
ENV_ASSESSMENT_GUARD = "ED4ALL_ANSWER_ASSESSMENT_GUARD"
#: Float knob → the match threshold (overlap / cosine floor).
ENV_ASSESSMENT_GUARD_THRESHOLD = "ED4ALL_ANSWER_ASSESSMENT_GUARD_THRESHOLD"

GUARD_OFF = "off"
GUARD_SHADOW = "shadow"
GUARD_ON = "on"
#: Code default is OFF — the guard is byte-identical inert when the flag is
#: unset (an opt-in behavior gate, not a default-on posture).
DEFAULT_MODE = GUARD_OFF

#: Conservative default match floor. A near-verbatim quiz-stem paste scores at
#: or near 1.0 on the token-containment metric; a genuine study question sits
#: well below. Deliberately high so the guard is precise (a false redirect that
#: blocks a legitimate study question is worse than a missed homework paste).
DEFAULT_THRESHOLD = 0.75

#: Decision-capture ``decision_type`` for a guard evaluation. NOTE: strict
#: decision-schema enum parity (``schemas/events/decision_event.schema.json``)
#: needs a matching enum row; that schema is another lane's file, so it is a
#: followup — with ``VALIDATE_DECISIONS`` unset (default) the capture writes
#: fine, and this module never depends on the enum being present.
DECISION_TYPE_GUARD = "assessment_guard_match"

#: Cap on concept-page citations surfaced in the redirect hint.
_REDIRECT_MAX_CITATIONS = 5

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


# --------------------------------------------------------------------------- #
# Flag / knob resolution
# --------------------------------------------------------------------------- #


def resolve_guard_mode(explicit: Optional[str] = None) -> str:
    """Resolve the three-valued guard mode.

    Precedence: explicit arg > ``ED4ALL_ANSWER_ASSESSMENT_GUARD`` env >
    :data:`DEFAULT_MODE`. An unrecognized / garbage value falls back to the
    default (``off``) so a typo can never silently ENABLE enforcement.
    """
    raw = explicit if explicit is not None else os.environ.get(ENV_ASSESSMENT_GUARD)
    if raw is None:
        return DEFAULT_MODE
    val = str(raw).strip().lower()
    if val in (GUARD_OFF, GUARD_SHADOW, GUARD_ON):
        return val
    return DEFAULT_MODE


def resolve_guard_threshold(explicit: Optional[float] = None) -> float:
    """Resolve the match threshold knob.

    Precedence: explicit arg > ``ED4ALL_ANSWER_ASSESSMENT_GUARD_THRESHOLD`` env >
    :data:`DEFAULT_THRESHOLD`. Garbage / out-of-range (not in ``[0, 1]``) values
    fall back to the default (a misconfigured knob must never disable matching
    by pushing the floor out of band).
    """
    if explicit is not None:
        cand: Any = explicit
    else:
        raw = os.environ.get(ENV_ASSESSMENT_GUARD_THRESHOLD)
        if raw is None or str(raw).strip() == "":
            return DEFAULT_THRESHOLD
        cand = raw
    try:
        cand = float(cand)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD
    if 0.0 <= cand <= 1.0:
        return cand
    return DEFAULT_THRESHOLD


# --------------------------------------------------------------------------- #
# Stem model + loading (cached per course-slug + mtime)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AssessmentStem:
    """One assessment-item stem (never the answer)."""

    quiz_id: str
    question_id: str
    stem: str


# slug -> (signature, stems). ``signature`` is the assessment-source mtime tuple
# so a re-archived / edited course invalidates the cache without a restart.
_STEM_CACHE: Dict[str, Tuple[Tuple[Tuple[str, int], ...], List[AssessmentStem]]] = {}


def _assessment_signature(libv2_root: Path, slug: str) -> Tuple[Tuple[str, int], ...]:
    """A cache signature: ``(path, mtime_ns)`` over every assessment source.

    Covers loose ``06_assessments/*.xml``, ``training_specs/assessments.json``,
    and any packaged ``source/imscc/*.imscc``. Empty tuple when nothing exists
    (an unknown / assessment-less course still caches cleanly as ``[]``).
    """
    course_dir = libv2_root / "courses" / slug
    candidates: List[Path] = []
    adir = course_dir / "06_assessments"
    if adir.is_dir():
        candidates.extend(sorted(adir.glob("*.xml")))
    specs = course_dir / "training_specs" / "assessments.json"
    if specs.is_file():
        candidates.append(specs)
    imscc_dir = course_dir / "source" / "imscc"
    if imscc_dir.is_dir():
        candidates.extend(sorted(imscc_dir.glob("*.imscc")))
    sig: List[Tuple[str, int]] = []
    for p in candidates:
        try:
            sig.append((str(p), p.stat().st_mtime_ns))
        except OSError:
            continue
    return tuple(sig)


def load_assessment_stems(libv2_root: Path, slug: str) -> List[AssessmentStem]:
    """Load a course's assessment stems (STEMS only), cached per slug + mtime.

    Two fail-soft sources are unioned:

    1. Packaged QTI (via ``gui.services.quiz_service`` — ``to_learner_json``
       strips the answer key by contract).
    2. ``training_specs/assessments.json`` (``stem`` strings only).

    Any load failure degrades to ``[]`` (a missing / assessment-less course is a
    clean no-op, never an error on the answer path).
    """
    signature = _assessment_signature(libv2_root, slug)
    cached = _STEM_CACHE.get(slug)
    if cached is not None and cached[0] == signature:
        return cached[1]

    stems: List[AssessmentStem] = []
    seen: set = set()

    def _add(quiz_id: str, question_id: str, stem: str) -> None:
        text = (stem or "").strip()
        if not text:
            return
        key = (quiz_id, question_id, text)
        if key in seen:
            return
        seen.add(key)
        stems.append(AssessmentStem(quiz_id=quiz_id, question_id=question_id, stem=text))

    # 1) QTI via the quiz service (lazy import keeps this module gui-free).
    try:
        from gui.services import quiz_service  # noqa: PLC0415

        for blob in quiz_service.locate_quiz_xml(slug, libv2_root):
            try:
                assessment = quiz_service.parse_quiz(blob)
                learner = quiz_service.to_learner_json(assessment)
            except Exception:  # noqa: BLE001 — one bad quiz never sinks the rest
                continue
            quiz_id = str(learner.get("assessment_id") or "quiz")
            for q in learner.get("questions") or []:
                if isinstance(q, dict):
                    _add(quiz_id, str(q.get("question_id") or ""), str(q.get("stem") or ""))
    except Exception:  # noqa: BLE001 — quiz service unavailable → skip QTI source
        pass

    # 2) training_specs/assessments.json (stem strings only, defensive walk).
    specs = libv2_root / "courses" / slug / "training_specs" / "assessments.json"
    if specs.is_file():
        try:
            import json  # noqa: PLC0415

            data = json.loads(specs.read_text(encoding="utf-8"))
            for quiz_id, question_id, stem in _walk_spec_stems(data):
                _add(quiz_id, question_id, stem)
        except Exception:  # noqa: BLE001 — malformed specs → skip this source
            pass

    _STEM_CACHE[slug] = (signature, stems)
    return stems


def _walk_spec_stems(node: Any, quiz_id: str = "assessments") -> List[Tuple[str, str, str]]:
    """Recursively harvest ``(quiz_id, question_id, stem)`` from a spec doc.

    Shape-agnostic: any dict carrying a string ``stem`` (or ``prompt`` /
    ``question_text``) is treated as an item; ids come from ``id`` /
    ``question_id`` / ``ident`` when present. ONLY stem text is read — answer
    keys / options are never touched.
    """
    out: List[Tuple[str, str, str]] = []
    if isinstance(node, dict):
        stem = None
        for key in ("stem", "prompt", "question_text"):
            val = node.get(key)
            if isinstance(val, str) and val.strip():
                stem = val
                break
        if stem is not None:
            qid = ""
            for key in ("question_id", "id", "ident"):
                val = node.get(key)
                if isinstance(val, (str, int)) and str(val).strip():
                    qid = str(val)
                    break
            out.append((quiz_id, qid, stem))
        # Recurse; let a nested assessment id re-scope its children.
        local_quiz = quiz_id
        for key in ("assessment_id", "id", "ident"):
            val = node.get(key)
            if isinstance(val, (str, int)) and str(val).strip() and "questions" in node:
                local_quiz = str(val)
                break
        for val in node.values():
            if isinstance(val, (dict, list)):
                out.extend(_walk_spec_stems(val, local_quiz))
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                out.extend(_walk_spec_stems(item, quiz_id))
    return out


# --------------------------------------------------------------------------- #
# Matching (lexical primary; cosine arm only when an embedding client is present)
# --------------------------------------------------------------------------- #


def _content_tokens(text: str) -> set:
    """Lowercased content tokens (len > 2). Mirrors the grounded-answer lexical
    tokenizer so the guard's overlap scale matches the rest of the answer path;
    a local copy keeps this module free of the heavy grounded-answer import."""
    return {t.lower() for t in _WORD_RE.findall(text or "") if len(t) > 2}


def _containment(q_tokens: set, s_tokens: set) -> float:
    """Szymkiewicz–Simpson overlap: ``|Q ∩ S| / min(|Q|, |S|)``.

    Robust to length asymmetry — a short pasted stem inside a longer query (or
    vice-versa) still scores high, where Jaccard would understate it. A verbatim
    stem paste ⇒ ~1.0; an unrelated question ⇒ near 0.
    """
    if not q_tokens or not s_tokens:
        return 0.0
    inter = len(q_tokens & s_tokens)
    return inter / float(min(len(q_tokens), len(s_tokens)))


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    score: float
    quiz_id: str
    question_id: str
    stem: str
    method: str  # "lexical" | "cosine" | "none"


_NO_MATCH = MatchResult(False, 0.0, "", "", "", "none")


def match_question(
    query: str,
    stems: List[AssessmentStem],
    threshold: float,
    *,
    embedding_client: Optional[Any] = None,
) -> MatchResult:
    """Return the best-matching stem for ``query``.

    Lexical token-containment is the PRIMARY signal (always computed). When an
    ``embedding_client`` is supplied, a cosine arm is added and the per-stem
    score is ``max(lexical, cosine)`` (either arm can trip the floor). The
    winning ``method`` records which arm produced the best stem's score.
    """
    q_tokens = _content_tokens(query)
    if not stems or not q_tokens:
        return _NO_MATCH

    lexical = [_containment(q_tokens, _content_tokens(s.stem)) for s in stems]

    cosine: Optional[List[float]] = None
    if embedding_client is not None:
        cosine = _cosine_scores(query, stems, embedding_client)

    best_idx = -1
    best_score = -1.0
    best_method = "lexical"
    for i, s in enumerate(stems):
        lex = lexical[i]
        cos = cosine[i] if cosine is not None else -1.0
        if cos > lex:
            score, method = cos, "cosine"
        else:
            score, method = lex, "lexical"
        if score > best_score:
            best_score, best_idx, best_method = score, i, method

    if best_idx < 0:
        return _NO_MATCH
    best = stems[best_idx]
    return MatchResult(
        matched=best_score >= threshold,
        score=float(best_score),
        quiz_id=best.quiz_id,
        question_id=best.question_id,
        stem=best.stem,
        method=best_method,
    )


def _cosine_scores(
    query: str, stems: List[AssessmentStem], embedding_client: Any
) -> Optional[List[float]]:
    """Cosine(query, stem) for every stem, or ``None`` if the backend is down.

    The embedding client returns L2-normalized rows, so cosine reduces to a dot
    product. Any backend failure degrades to ``None`` (the caller falls back to
    lexical-only) — retrieval-adjacent scoring never hard-fails the answer path.
    """
    try:
        import numpy as np  # noqa: PLC0415

        texts = [query] + [s.stem for s in stems]
        vecs = embedding_client.encode_batch(texts)
        qv = vecs[0]
        sims = vecs[1:] @ qv
        return [float(x) for x in np.asarray(sims).tolist()]
    except Exception:  # noqa: BLE001 — deps missing / backend error → lexical only
        return None


def _build_embedding_client() -> Optional[Any]:
    """Best-effort default embedding client for the cosine arm, or ``None``.

    Resolves the registry-default provider and builds an offline (query-path)
    client. Any failure (``[embedding]`` extras absent, weights not cached,
    fake provider) → ``None`` so matching degrades to lexical-only.
    """
    try:
        from lib.embedding.providers import (  # noqa: PLC0415
            EmbeddingClient,
            resolve_embedding_provider,
        )

        resolved = resolve_embedding_provider()
        if getattr(resolved, "kind", None) == "fake":
            return None
        return EmbeddingClient(resolved, offline=True)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Guard evaluation + decision capture
# --------------------------------------------------------------------------- #


@dataclass
class GuardOutcome:
    mode: str
    evaluated: bool
    matched: bool
    score: float
    threshold: float
    quiz_id: str
    question_id: str
    method: str
    stem: str = ""

    def signal(self) -> Dict[str, Any]:
        """The additive ``assessment_guard`` payload stamped on a result dict."""
        return {
            "mode": self.mode,
            "matched": self.matched,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "quiz_id": self.quiz_id,
            "question_id": self.question_id,
            "method": self.method,
            # ``on`` + matched is the only case that actually redirects.
            "redirected": self.mode == GUARD_ON and self.matched,
        }


def _query_hash(query: str) -> str:
    return hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:12]


def evaluate_guard(
    libv2_root: Path,
    slug: str,
    query: str,
    mode: str,
    *,
    capture: Optional[Any] = None,
    embedding_client: Optional[Any] = None,
    use_embeddings: bool = True,
) -> GuardOutcome:
    """Load stems, compare, and EMIT a decision capture for the evaluation.

    Called only for ``shadow`` / ``on`` (the ``off`` path never reaches here, so
    the guard stays byte-identical inert when disabled). Fires exactly one
    :data:`DECISION_TYPE_GUARD` decision per evaluation — covering match,
    no-match, and shadow — with a dynamic rationale interpolating the query hash,
    matched quiz id, overlap score, and mode.
    """
    threshold = resolve_guard_threshold()
    stems = load_assessment_stems(libv2_root, slug)

    client = embedding_client
    if client is None and use_embeddings and stems:
        client = _build_embedding_client()

    result = match_question(query, stems, threshold, embedding_client=client)

    outcome = GuardOutcome(
        mode=mode,
        evaluated=True,
        matched=result.matched,
        score=result.score,
        threshold=threshold,
        quiz_id=result.quiz_id,
        question_id=result.question_id,
        method=result.method,
        stem=result.stem,
    )
    _emit_capture(capture, query, outcome, n_stems=len(stems))
    return outcome


def _emit_capture(
    capture: Optional[Any], query: str, outcome: GuardOutcome, *, n_stems: int
) -> None:
    """Emit the guard decision (best-effort; a capture failure never blocks)."""
    if capture is None:
        return
    if outcome.mode == GUARD_ON and outcome.matched:
        decision = "redirect_to_study_sections"
    elif outcome.mode == GUARD_SHADOW and outcome.matched:
        decision = "shadow_would_redirect"
    else:
        decision = "pass_through"
    qhash = _query_hash(query)
    rationale = (
        f"assessment_guard mode={outcome.mode} matched={outcome.matched} "
        f"score={outcome.score:.3f} threshold={outcome.threshold:.2f} "
        f"method={outcome.method} quiz={outcome.quiz_id or '-'} "
        f"question={outcome.question_id or '-'} qhash={qhash} stems={n_stems}"
    )
    try:
        capture.log_decision(
            decision_type=DECISION_TYPE_GUARD,
            decision=decision,
            rationale=rationale,
            confidence=round(outcome.score, 4),
        )
    except Exception:  # noqa: BLE001 — capture is advisory; never block the answer
        pass


# --------------------------------------------------------------------------- #
# Redirect envelope (reuses the retrieval machinery; NO compose)
# --------------------------------------------------------------------------- #

_REDIRECT_HEADING = (
    "This question closely matches a graded course assessment item, so I "
    "won't answer it directly — working through it yourself is how the "
    "assessment measures your learning."
)
_REDIRECT_BODY = (
    "Here are the course sections that cover this material. Review them, then "
    "try the question on your own."
)
_REDIRECT_BODY_NO_SOURCES = (
    "Review your course materials on this topic, then try the question on your "
    "own."
)

#: The redirect reuses the ``answered`` status so the WS4-owned answer renderer
#: (``gui/services/answer_render.py``) renders the hint text + Sources list (an
#: UNKNOWN status would fall back to the generic-error fragment).
REDIRECT_STATUS = "answered"


def build_redirect_envelope(
    libv2_root: Path,
    slug: str,
    query: str,
    engine: str,
    outcome: GuardOutcome,
) -> Dict[str, Any]:
    """Build the redirect-with-hint answer envelope (``GroundedAnswer``-shaped).

    Reuses the retrieval machinery to cite the concept pages to study, but does
    NOT compose an answer (no LLM call ⇒ no answer can leak). Mirrors
    ``GroundedAnswer.to_dict()`` keys so the router / renderer / drawer consume
    it unchanged, plus an additive ``assessment_guard`` signal.
    """
    from datetime import datetime, timezone  # noqa: PLC0415

    citations = _concept_citations(libv2_root, slug, query, engine, _REDIRECT_MAX_CITATIONS)
    body = _REDIRECT_BODY if citations else _REDIRECT_BODY_NO_SOURCES
    answer_text = f"{_REDIRECT_HEADING}\n\n{body}"
    warning = (
        f"assessment_guard: redirected (matched quiz {outcome.quiz_id or '-'} "
        f"question {outcome.question_id or '-'}, score={outcome.score:.3f}, "
        f"method={outcome.method})"
    )
    return {
        "status": REDIRECT_STATUS,
        "query": query,
        "course_slug": slug,
        "engine": engine,
        "answer_text": answer_text,
        "citations": citations,
        "refusal": None,
        "confidence": {"assessment_guard": outcome.signal()},
        "groundedness": None,
        "warnings": [warning],
        "model_id": None,
        "prompt_version": None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latency_ms": 0.0,
        "assessment_guard": outcome.signal(),
    }


def _concept_citations(
    libv2_root: Path, slug: str, query: str, engine: str, limit: int
) -> List[Dict[str, Any]]:
    """Retrieve concept pages and shape them as citation dicts (NO compose).

    Reuses the grounded-answer retriever + provenance helpers to build
    renderer-compatible citation dicts (``item_path`` + heading-slug fragment
    give a working "Source:" study link). Fail-soft: any retrieval error → the
    redirect still returns, just without a Sources list (falls back to lexical
    on a semantic-engine error so a course without a vector index still cites).
    """
    passages = _retrieve_passages(libv2_root, slug, query, engine, limit)
    citations: List[Dict[str, Any]] = []
    try:
        from lib.retrieval.grounded_answer import (  # noqa: PLC0415
            _provenance_from_source,
            page_label_for_source,
        )
    except Exception:  # noqa: BLE001
        return citations

    for p in passages:
        source = dict(getattr(p, "source", None) or {})
        source.setdefault("item_path", getattr(p, "item_path", ""))
        section_heading = getattr(p, "section_heading", None)
        if section_heading is not None:
            source.setdefault("section_heading", section_heading)
        try:
            page_label = page_label_for_source(source)
        except Exception:  # noqa: BLE001
            page_label = "Source"
        try:
            source_block, pdf_pages = _provenance_from_source(getattr(p, "source", None) or {})
        except Exception:  # noqa: BLE001
            source_block, pdf_pages = None, []
        item_path = getattr(p, "item_path", "") or ""
        citations.append(
            {
                "chunk_id": getattr(p, "chunk_id", ""),
                "item_path": item_path,
                "section_heading": section_heading,
                "module_id": getattr(p, "module_id", None),
                "page_label": page_label,
                "anchor_status": "unresolved",
                "source_path": None,
                "text_quote": _excerpt(getattr(p, "text", "")),
                "link_target": {
                    "kind": "course_page",
                    "item_path": item_path,
                    "fragment": _heading_fragment(section_heading),
                    "char_span": None,
                },
                "source_block": source_block,
                "pdf_pages": list(pdf_pages),
                "module_title": (
                    str(source["module_title"]).strip()
                    if source.get("module_title")
                    else None
                ),
                "supporting_excerpt": None,
                "supported_claim_count": 0,
            }
        )
    return citations


def _retrieve_passages(
    libv2_root: Path, slug: str, query: str, engine: str, limit: int
) -> List[Any]:
    """Retrieve passages via the grounded-answer retriever; fail-soft.

    Tries the resolved ``engine`` first; on any error (e.g. a semantic engine
    against a course with no vector index) falls back to ``lexical`` so the
    redirect still surfaces study sections. Returns ``[]`` if even lexical fails.
    """
    try:
        from lib.retrieval.grounded_answer import _retrieve  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return []
    for eng in (engine, "lexical"):
        try:
            return list(_retrieve(libv2_root, slug, query, engine=eng, limit=limit))
        except Exception:  # noqa: BLE001
            continue
    return []


def _heading_fragment(section_heading: Optional[str]) -> Dict[str, Any]:
    """A heading-slug link fragment (mirrors grounded_answer._fragment_for)."""
    if section_heading:
        slug = re.sub(r"[^a-z0-9]+", "-", str(section_heading).lower()).strip("-")
        if slug:
            return {"kind": "heading", "value": slug}
    return {"kind": None, "value": None}


def _excerpt(text: str, limit: int = 200) -> Optional[str]:
    """A short study-pointer excerpt of a passage (first ~200 chars)."""
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return None
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"


__all__ = [
    "ENV_ASSESSMENT_GUARD",
    "ENV_ASSESSMENT_GUARD_THRESHOLD",
    "GUARD_OFF",
    "GUARD_SHADOW",
    "GUARD_ON",
    "DEFAULT_THRESHOLD",
    "DECISION_TYPE_GUARD",
    "AssessmentStem",
    "MatchResult",
    "GuardOutcome",
    "resolve_guard_mode",
    "resolve_guard_threshold",
    "load_assessment_stems",
    "match_question",
    "evaluate_guard",
    "build_redirect_envelope",
]

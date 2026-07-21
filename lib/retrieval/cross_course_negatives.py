"""Cross-course hard-negative / out-of-domain refusal-probe mining (W4.4).

``ED4ALL_EVAL_CROSS_COURSE_NEGATIVES`` — an ADDITIVE eval arm that mines REAL
chunks from OTHER courses in the library as hard negatives for a target course's
eval set. A question grounded in a DIFFERENT course's material is (by
construction) out-of-domain for the target course, so a correctly-scoped target
corpus should REFUSE it. These are the hardest kind of refusal probe: they are
real, coherent, answerable educational questions — just answerable from a
different course.

Anti-fabrication (hard contract):

  * Every probe is derived from a REAL chunk in a REAL other course. The source
    course slug + source chunk id + a verbatim excerpt ride on the probe record
    so a reviewer can trace exactly where the negative came from.
  * The probe is CLEARLY LABELED cross-course (``category ==
    "cross_course_negative"`` + ``source_course_slug``). It is NEVER surfaced as
    the target course's own content — it is a labeled negative.
  * The question text is a DETERMINISTIC template over the foreign chunk's real
    concept tags / section heading (no LLM invents facts). The out-of-domain
    property is what makes it a negative, not any fabricated detail.

Domain awareness: courses whose ``primary_domain`` differs from the target are
preferred (a true out-of-domain negative); same-domain courses are used only to
top up when there are not enough foreign domains, and are flagged
``same_domain=True`` so the operator can drop the weaker negatives.

Optional target-course dry-runs (reusing ``probe_authoring.run_dry_runs``)
attach the empirical evidence that the TARGET corpus scores the probe low — the
same operator-review signal the refusal-probe candidates carry.

This module authors CANDIDATES (``reviewed_by = "PENDING_REVIEW"``); the final
``refusal_probes.json`` promotion stays an operator decision (R5 mislabeling
mitigation), exactly like :mod:`lib.retrieval.probe_authoring`.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from lib.retrieval.gold_set import RETRIEVAL_EVAL_SUBDIR, _load_chunks_by_id

__all__ = [
    "ENV_CROSS_COURSE_NEGATIVES",
    "CROSS_COURSE_NEGATIVES_FILENAME",
    "CROSS_COURSE_AUTHORING_VERSION",
    "DECISION_TYPE_CROSS_COURSE",
    "CROSS_COURSE_TEMPLATES",
    "resolve_cross_course_negatives",
    "mine_cross_course_negatives",
    "generate_cross_course_negatives",
]

ENV_CROSS_COURSE_NEGATIVES = "ED4ALL_EVAL_CROSS_COURSE_NEGATIVES"
CROSS_COURSE_NEGATIVES_FILENAME = "cross_course_negative_candidates.json"
CROSS_COURSE_AUTHORING_VERSION = "cross-course-neg.v1"
DECISION_TYPE_CROSS_COURSE = "cross_course_negative_authoring"

_TRUTHY = {"1", "true", "yes", "on"}

# Deterministic question templates over a foreign chunk's real topic. {topic} is
# a real concept tag / heading phrase from the source course — the question is a
# genuine, answerable question about THAT course, hence out-of-domain (refusal)
# for the target course.
CROSS_COURSE_TEMPLATES: Tuple[str, ...] = (
    "What does {topic} mean and how is it used?",
    "Explain the concept of {topic}.",
    "How is {topic} defined in this subject?",
    "What are the key ideas behind {topic}?",
)

# Content-bearing chunk types worth mining (mirrors the source-backfill filter):
# a real teachable idea, not front-matter / an exercise stem.
_CONTENT_CHUNK_TYPES = frozenset({"explanation", "concept", "content"})
_MIN_WORD_COUNT = 20
_MAX_PER_COURSE_DEFAULT = 3
_EXCERPT_CHARS = 240
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")


# --------------------------------------------------------------------------- #
# Flag resolution
# --------------------------------------------------------------------------- #


def resolve_cross_course_negatives(explicit: Optional[bool] = None) -> bool:
    """Resolve ``ED4ALL_EVAL_CROSS_COURSE_NEGATIVES`` (default OFF).

    An explicit ``bool`` wins; else read the env truthily. Parse-with-fallback:
    unset / ``0`` / garbage → off.
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get(ENV_CROSS_COURSE_NEGATIVES)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY


# --------------------------------------------------------------------------- #
# Course enumeration + domain lookup
# --------------------------------------------------------------------------- #


def _course_primary_domain(course_dir: Path) -> str:
    """Best-effort ``classification.primary_domain`` from a course manifest."""
    manifest_path = course_dir / "manifest.json"
    if not manifest_path.is_file():
        return "unknown"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    classification = (data or {}).get("classification") or {}
    return str(classification.get("primary_domain") or "unknown")


def _resolve_chunks_path(course_dir: Path) -> Optional[Path]:
    """The retrievable chunks.jsonl for a course, first hit wins.

    ``semantik_chunks`` is the current layout; the remaining dirs are
    dual-read fallbacks for legacy corpora.
    """
    for sub in ("semantik_chunks", "dart_chunks", "imscc_chunks", "corpus"):
        cand = course_dir / sub / "chunks.jsonl"
        if cand.is_file():
            return cand
    return None


def _enumerate_other_courses(
    libv2_root: Path, target_slug: str, *, explicit: Optional[Sequence[str]]
) -> List[str]:
    """Other course slugs carrying a retrievable chunkset (target excluded).

    ``explicit`` (operator list) wins verbatim; else scan ``courses/*/``.
    Deterministic (sorted). Never raises.
    """
    courses_root = libv2_root / "courses"
    out: List[str] = []
    seen: set = set()

    def _add(slug: str) -> None:
        slug = str(slug or "").strip()
        if not slug or slug == target_slug or slug in seen:
            return
        if _resolve_chunks_path(courses_root / slug) is not None:
            out.append(slug)
            seen.add(slug)

    try:
        if explicit is not None:
            for slug in explicit:
                _add(slug)
        elif courses_root.is_dir():
            for child in sorted(courses_root.iterdir()):
                if child.is_dir():
                    _add(child.name)
    except Exception:  # noqa: BLE001 — enumeration never fails authoring
        return out
    return out


# --------------------------------------------------------------------------- #
# Topic extraction + question derivation (deterministic; anti-fabrication)
# --------------------------------------------------------------------------- #


def _prettify(text: str) -> str:
    pretty = re.sub(r"[-_]+", " ", str(text or "")).strip()
    return pretty


def _chunk_topic(chunk: Dict[str, Any]) -> Optional[str]:
    """A real topic phrase for a chunk: first concept tag, else heading phrase.

    Returns ``None`` when neither a usable concept tag nor a heading resolves —
    the caller skips the chunk rather than inventing a topic.
    """
    tags = chunk.get("concept_tags") or []
    for tag in tags:
        if isinstance(tag, str) and len(tag.strip()) >= 3:
            return _prettify(tag)
    heading = ((chunk.get("source") or {}).get("section_heading"))
    if isinstance(heading, str) and len(heading.strip()) >= 3:
        return heading.strip()
    return None


def _excerpt(text: str) -> str:
    t = (text or "").strip().replace("\n", " ")
    return t[:_EXCERPT_CHARS]


def _content_chunks_sorted(
    chunks_by_id: Dict[str, Dict[str, Any]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """Content-bearing chunks (deterministic id order) worth mining."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    for cid, chunk in chunks_by_id.items():
        ctype = str(chunk.get("chunk_type") or "")
        wc = int(chunk.get("word_count") or 0)
        if ctype in _CONTENT_CHUNK_TYPES and wc >= _MIN_WORD_COUNT:
            if _chunk_topic(chunk) is not None:
                out.append((cid, chunk))
    out.sort(key=lambda kv: kv[0])
    return out


def _build_probe(
    *,
    target_slug: str,
    source_slug: str,
    source_domain: str,
    same_domain: bool,
    chunk_id: str,
    chunk: Dict[str, Any],
    template_index: int,
) -> Dict[str, Any]:
    topic = _chunk_topic(chunk) or ""
    template = CROSS_COURSE_TEMPLATES[template_index % len(CROSS_COURSE_TEMPLATES)]
    question = template.format(topic=topic)
    return {
        "question_text": question,
        "category": "cross_course_negative",
        "why_unanswerable": (
            f"Grounded in a DIFFERENT course ('{source_slug}', domain "
            f"'{source_domain}'); the target course '{target_slug}' does not "
            f"cover this material, so it must refuse. Real cross-course "
            f"content, clearly labeled as a negative (not target content)."
        ),
        # Cross-course provenance (anti-fabrication: a REAL foreign chunk).
        "source_course_slug": source_slug,
        "source_chunk_id": chunk_id,
        "source_domain": source_domain,
        "source_topic": topic,
        "source_excerpt": _excerpt(str(chunk.get("text") or "")),
        "same_domain": same_domain,
        "authoring": {
            "method": "cross_course_mined",
            "author": f"cross-course/{CROSS_COURSE_AUTHORING_VERSION}",
            "reviewed_by": "PENDING_REVIEW",
            "status": "draft",
        },
    }


# --------------------------------------------------------------------------- #
# Mining
# --------------------------------------------------------------------------- #


def mine_cross_course_negatives(
    target_course_dir: Path,
    *,
    libv2_root: Path,
    n: int = 10,
    other_slugs: Optional[Sequence[str]] = None,
    max_per_course: int = _MAX_PER_COURSE_DEFAULT,
    capture: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Mine up to ``n`` cross-course negative probe candidates.

    Enumerates OTHER courses (out-of-domain preferred), samples content-bearing
    real chunks from each (round-robin, capped per course for diversity), and
    derives a deterministic labeled negative per chunk. Returns the candidate
    list (unnumbered; :func:`generate_cross_course_negatives` wraps + writes).

    Emits one ``cross_course_negative_authoring`` decision (dynamic rationale:
    other-course count, out-of-domain count, mined count).
    """
    target_course_dir = Path(target_course_dir)
    libv2_root = Path(libv2_root)
    target_slug = target_course_dir.name
    target_domain = _course_primary_domain(target_course_dir)

    others = _enumerate_other_courses(
        libv2_root, target_slug, explicit=other_slugs
    )
    # Prefer out-of-domain courses first; same-domain courses top up.
    domain_by_slug: Dict[str, str] = {
        slug: _course_primary_domain(libv2_root / "courses" / slug)
        for slug in others
    }
    out_of_domain = [
        s for s in others if domain_by_slug[s] != target_domain
    ]
    same_domain = [s for s in others if domain_by_slug[s] == target_domain]
    ordered_courses = out_of_domain + same_domain

    # Pre-load each course's mineable chunk pool (deterministic order).
    pools: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for slug in ordered_courses:
        chunks_path = _resolve_chunks_path(libv2_root / "courses" / slug)
        if chunks_path is None:
            continue
        try:
            chunks_by_id = _load_chunks_by_id(chunks_path)
        except Exception:  # noqa: BLE001 — a bad chunkset is skipped
            continue
        pool = _content_chunks_sorted(chunks_by_id)[: max(0, int(max_per_course))]
        if pool:
            pools[slug] = pool

    # Round-robin across courses so the negatives span the library, not one
    # course; each course contributes at most ``max_per_course``.
    candidates: List[Dict[str, Any]] = []
    template_index = 0
    depth = 0
    active = [s for s in ordered_courses if s in pools]
    while len(candidates) < max(0, int(n)) and active:
        progressed = False
        for slug in list(active):
            if len(candidates) >= max(0, int(n)):
                break
            pool = pools.get(slug) or []
            if depth >= len(pool):
                active = [s for s in active if len(pools.get(s) or []) > depth + 1]
                continue
            chunk_id, chunk = pool[depth]
            candidates.append(
                _build_probe(
                    target_slug=target_slug,
                    source_slug=slug,
                    source_domain=domain_by_slug.get(slug, "unknown"),
                    same_domain=(domain_by_slug.get(slug) == target_domain),
                    chunk_id=chunk_id,
                    chunk=chunk,
                    template_index=template_index,
                )
            )
            template_index += 1
            progressed = True
        depth += 1
        if not progressed:
            break

    if capture is not None:
        try:
            capture.log_decision(
                decision_type=DECISION_TYPE_CROSS_COURSE,
                decision=(
                    f"mined {len(candidates)} cross-course negative probe(s) for "
                    f"'{target_slug}' from {len(pools)} other course(s)."
                ),
                rationale=(
                    f"Cross-course hard-negative mining ({CROSS_COURSE_AUTHORING_VERSION}) "
                    f"for target '{target_slug}' (domain '{target_domain}'): "
                    f"{len(others)} other course(s) available, "
                    f"{len(out_of_domain)} out-of-domain, {len(same_domain)} "
                    f"same-domain; requested={n}, mined={len(candidates)}, "
                    f"max_per_course={max_per_course}. Each negative is a REAL "
                    f"foreign chunk (source_course_slug + source_chunk_id carried) "
                    f"labeled cross_course_negative — the target corpus must refuse it."
                ),
            )
        except Exception:  # pragma: no cover — capture never breaks authoring
            pass
    return candidates


# --------------------------------------------------------------------------- #
# Assemble + write
# --------------------------------------------------------------------------- #


def _attach_target_dry_runs(
    candidates: Sequence[Dict[str, Any]],
    *,
    retrieve_fn: Callable[..., Sequence[Any]],
    libv2_root: Path,
    target_slug: str,
    limit: int = 8,
    engines: Sequence[str] = ("lexical", "semantic", "hybrid-rrf"),
) -> List[Dict[str, Any]]:
    """Attach TARGET-course read-only dry-runs (should score low → refuse).

    Reuses :func:`lib.retrieval.probe_authoring.run_dry_runs`. A retriever that
    raises records a zero-result dry-run (never fails the batch).
    """
    from lib.retrieval.probe_authoring import run_dry_runs

    out: List[Dict[str, Any]] = []
    for cand in candidates:
        entry = dict(cand)
        evidence = run_dry_runs(
            str(cand.get("question_text") or ""),
            retrieve_fn=retrieve_fn,
            libv2_root=libv2_root,
            course_slug=target_slug,
            limit=limit,
            engines=engines,
        )
        entry["target_dry_runs"] = [e.to_dict() for e in evidence]
        out.append(entry)
    return out


def build_cross_course_doc(
    target_slug: str, candidates: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Assemble + number the ``cross_course_negative_candidates.json`` doc."""
    numbered: List[Dict[str, Any]] = []
    for i, cand in enumerate(candidates, start=1):
        entry = dict(cand)
        entry["probe_id"] = f"ccn-{target_slug}-{i:04d}"
        numbered.append(entry)
    by_source: Dict[str, int] = {}
    for c in numbered:
        src = str(c.get("source_course_slug") or "unknown")
        by_source[src] = by_source.get(src, 0) + 1
    return {
        "schema_version": "1.0",
        "target_course_slug": target_slug,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authoring_run": {
            "n_probes": len(numbered),
            "by_source_course": by_source,
            "prompt_version": CROSS_COURSE_AUTHORING_VERSION,
        },
        "probe_candidates": numbered,
    }


def generate_cross_course_negatives(
    target_course_dir: Path,
    *,
    libv2_root: Path,
    n: int = 10,
    other_slugs: Optional[Sequence[str]] = None,
    max_per_course: int = _MAX_PER_COURSE_DEFAULT,
    retrieve_fn: Optional[Callable[..., Sequence[Any]]] = None,
    capture: Optional[Any] = None,
    write: bool = True,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """End-to-end W4.4: mine cross-course negatives, optionally attach target
    dry-runs, write ``cross_course_negative_candidates.json``.

    ``retrieve_fn`` (optional) drives the read-only TARGET-course dry-runs
    (inject a deterministic fake in tests / ``retrieve_chunks`` in production).
    Returns ``(doc, written_path)``.
    """
    target_course_dir = Path(target_course_dir)
    target_slug = target_course_dir.name

    candidates = mine_cross_course_negatives(
        target_course_dir,
        libv2_root=libv2_root,
        n=n,
        other_slugs=other_slugs,
        max_per_course=max_per_course,
        capture=capture,
    )

    if retrieve_fn is not None:
        candidates = _attach_target_dry_runs(
            candidates,
            retrieve_fn=retrieve_fn,
            libv2_root=libv2_root,
            target_slug=target_slug,
        )

    doc = build_cross_course_doc(target_slug, candidates)

    out_path: Optional[Path] = None
    if write:
        out_dir = target_course_dir / RETRIEVAL_EVAL_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / CROSS_COURSE_NEGATIVES_FILENAME
        out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc, out_path

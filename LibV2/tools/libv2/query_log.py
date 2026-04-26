"""Persist Claude's queries + synthesized answers alongside the queried corpus.

Each query/answer pair becomes a JSON artifact under
``LibV2/courses/<slug>/queries/<query_id>.json`` (per-course query) or
``LibV2/catalog/queries/<query_id>.json`` (cross-course query). A
companion ``queries_index.json`` in the same directory lets callers
list past queries without globbing.

Two-step flow: ``libv2 ask`` writes the query record with retrieved
chunks and ``status='open'``; after Claude reads the chunks and
synthesizes an answer, ``libv2 answer <query_id> <text>`` flips the
record to ``status='answered'``. A one-shot ``ask --answer`` path is
also supported for cases where the answer is already in hand.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_QUERIES_SUBDIR = "queries"
_INDEX_FILE = "queries_index.json"
_CROSS_COURSE_DIR = ("catalog", "queries")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mint_query_id(query_text: str, ts: Optional[datetime] = None) -> str:
    """Stable ID = ``q_<UTC date+time>_<sha1[:8] of normalized query>``.

    Same query text within the same second collapses to the same ID
    (so callers can detect dup submissions), but everyday spacing in
    time produces unique IDs.
    """
    ts = ts or datetime.now(timezone.utc)
    digest = hashlib.sha1(query_text.strip().lower().encode("utf-8")).hexdigest()[:8]
    return f"q_{ts.strftime('%Y%m%d_%H%M%S')}_{digest}"


def resolve_storage_dir(repo_root: Path, course_slug: Optional[str]) -> Path:
    """Return the directory where Q&A records live for this query.

    Per-course queries live next to the course corpus they queried.
    Cross-course queries (no ``course_slug``) live in the catalog
    namespace because there is no single source corpus to sit beside.
    """
    if course_slug:
        return repo_root / "courses" / course_slug / _QUERIES_SUBDIR
    return repo_root.joinpath(*_CROSS_COURSE_DIR)


def query_path(storage_dir: Path, query_id: str) -> Path:
    return storage_dir / f"{query_id}.json"


def index_path(storage_dir: Path) -> Path:
    return storage_dir / _INDEX_FILE


@dataclass
class CompactChunk:
    """Compact projection of a RetrievalResult for the query record.

    The full chunk text would bloat records and duplicate the corpus on
    disk. We keep an identifying handle (chunk_id), the rank+score, the
    section heading + concept tags so a reader can see *what* matched,
    and a snippet of the body so a reader can sanity-check without
    re-running retrieval.
    """

    rank: int
    chunk_id: str
    score: float
    course_slug: str
    section_heading: str
    module_id: str
    concept_tags: List[str]
    snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "score": round(float(self.score), 4),
            "course_slug": self.course_slug,
            "section_heading": self.section_heading,
            "module_id": self.module_id,
            "concept_tags": self.concept_tags,
            "snippet": self.snippet,
        }


def compact_retrieval_result(result: Any, rank: int, snippet_chars: int = 400) -> CompactChunk:
    """Project a ``RetrievalResult``-shaped object into a compact dict."""
    text = getattr(result, "text", "") or ""
    snippet = text[:snippet_chars].replace("\n", " ").strip()
    if len(text) > snippet_chars:
        snippet += "..."
    source = getattr(result, "source", None) or {}
    tags = list(getattr(result, "concept_tags", []) or [])[:8]
    return CompactChunk(
        rank=rank,
        chunk_id=getattr(result, "chunk_id", "") or "",
        score=float(getattr(result, "score", 0.0) or 0.0),
        course_slug=getattr(result, "course_slug", "") or "",
        section_heading=str(source.get("section_heading", "") or ""),
        module_id=str(source.get("module_id", "") or ""),
        concept_tags=tags,
        snippet=snippet,
    )


def write_query_record(
    repo_root: Path,
    course_slug: Optional[str],
    query_text: str,
    method: str,
    limit: int,
    retrieved: List[Dict[str, Any]],
    *,
    asked_by: str = "claude",
    query_id: Optional[str] = None,
    extra_filters: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist a new query record. Returns the file path written."""
    storage_dir = resolve_storage_dir(repo_root, course_slug)
    storage_dir.mkdir(parents=True, exist_ok=True)
    qid = query_id or mint_query_id(query_text)
    record = {
        "query_id": qid,
        "course_slug": course_slug,
        "scope": "course" if course_slug else "cross-course",
        "query_text": query_text,
        "asked_by": asked_by,
        "asked_at": _utc_now_iso(),
        "method": method,
        "limit": limit,
        "filters": extra_filters or {},
        "retrieved_chunks": retrieved,
        "answer": None,
        "answered_by": None,
        "answered_at": None,
        "status": "open",
    }
    path = query_path(storage_dir, qid)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    _upsert_index_entry(
        storage_dir,
        course_slug=course_slug,
        query_id=qid,
        query_text=query_text,
        status="open",
        asked_at=record["asked_at"],
        answered_at=None,
    )
    return path


def attach_answer(
    repo_root: Path,
    course_slug: Optional[str],
    query_id: str,
    answer: str,
    *,
    answered_by: str = "claude",
) -> Path:
    """Attach an answer to a previously-recorded query. Returns the path."""
    storage_dir = resolve_storage_dir(repo_root, course_slug)
    path = query_path(storage_dir, query_id)
    if not path.exists():
        raise FileNotFoundError(f"Query record not found: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["answer"] = answer
    record["answered_by"] = answered_by
    record["answered_at"] = _utc_now_iso()
    record["status"] = "answered"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    _upsert_index_entry(
        storage_dir,
        course_slug=course_slug,
        query_id=query_id,
        query_text=record.get("query_text", ""),
        status="answered",
        asked_at=record.get("asked_at"),
        answered_at=record["answered_at"],
    )
    return path


def list_queries(
    repo_root: Path,
    course_slug: Optional[str],
) -> List[Dict[str, Any]]:
    """Return the index entries for queries in the given scope."""
    storage_dir = resolve_storage_dir(repo_root, course_slug)
    idx = index_path(storage_dir)
    if not idx.exists():
        return []
    return json.loads(idx.read_text(encoding="utf-8")).get("queries", [])


def _normalize_query_text(query_text: str) -> str:
    """Normalize query text for cache lookup — lowercase + collapse whitespace.

    Two queries that differ only in case or surrounding whitespace are
    treated as the same cache key. Keeps the cache hit-rate high without
    semantic similarity machinery.
    """
    return " ".join(query_text.lower().split())


def find_answered_query(
    repo_root: Path,
    course_slug: Optional[str],
    query_text: str,
) -> Optional[Dict[str, Any]]:
    """Return the most-recently-answered record matching ``query_text``, or None.

    Match policy: normalized query text equality (lowercase + collapsed
    whitespace). Open (un-answered) records are NOT cache-eligible —
    only ``status='answered'`` qualifies. When multiple answered records
    match (re-asks over time), the most recent ``answered_at`` wins.

    Why a cache: queries against the corpus are cheap, but Claude's
    synthesis of an answer is expensive. The Q&A log already persists
    both; without a cache the answers would be invisible to future asks.
    """
    storage_dir = resolve_storage_dir(repo_root, course_slug)
    idx_file = index_path(storage_dir)
    if not idx_file.exists():
        return None
    idx = json.loads(idx_file.read_text(encoding="utf-8"))
    target = _normalize_query_text(query_text)
    matches = [
        q for q in idx.get("queries", [])
        if q.get("status") == "answered"
        and _normalize_query_text(q.get("query_text", "")) == target
    ]
    if not matches:
        return None
    # Sort ascending so equal-second timestamps preserve insertion order
    # (Python sort is stable). Newest-answered record is the last element —
    # which also matches "most recently re-upserted" when answered_at ties.
    matches.sort(key=lambda q: q.get("answered_at") or q.get("asked_at") or "")
    return load_record(repo_root, course_slug, matches[-1]["query_id"])


def load_record(
    repo_root: Path,
    course_slug: Optional[str],
    query_id: str,
) -> Dict[str, Any]:
    storage_dir = resolve_storage_dir(repo_root, course_slug)
    path = query_path(storage_dir, query_id)
    if not path.exists():
        raise FileNotFoundError(f"Query record not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _upsert_index_entry(
    storage_dir: Path,
    *,
    course_slug: Optional[str],
    query_id: str,
    query_text: str,
    status: str,
    asked_at: Optional[str],
    answered_at: Optional[str],
) -> None:
    idx = index_path(storage_dir)
    if idx.exists():
        data = json.loads(idx.read_text(encoding="utf-8"))
    else:
        data = {
            "scope": "course" if course_slug else "cross-course",
            "course_slug": course_slug,
            "queries": [],
        }
    data["queries"] = [q for q in data["queries"] if q.get("query_id") != query_id]
    data["queries"].append(
        {
            "query_id": query_id,
            "query_text": query_text,
            "status": status,
            "asked_at": asked_at,
            "answered_at": answered_at,
        }
    )
    data["queries"].sort(key=lambda q: q.get("asked_at") or "")
    idx.write_text(json.dumps(data, indent=2), encoding="utf-8")


__all__ = [
    "CompactChunk",
    "attach_answer",
    "compact_retrieval_result",
    "find_answered_query",
    "index_path",
    "list_queries",
    "load_record",
    "mint_query_id",
    "query_path",
    "resolve_storage_dir",
    "write_query_record",
]

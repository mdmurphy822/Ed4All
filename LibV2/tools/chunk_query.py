"""Typed faceted chunk query engine (Wave 77 Worker β).

Reusable backend for ``ed4all libv2 query`` and (later) MCP tool wrappers
that need read-only structured access to a LibV2 archive's chunk store.

Design notes
------------
* **Read-only.** The engine never mutates the archive; it just loads
  ``chunks.jsonl`` (canonical post-Wave-76) and applies in-memory filters.
* **Filter composition is AND.** Multi-value flags inside a single
  filter are OR-combined; cross-filter composition is AND.
* **TO/CO rollup.** When a terminal outcome (``to-NN``) is queried, the
  engine resolves its child component outcomes via ``objectives.json``
  and matches chunks tagged with the TO **or** any of its children.
* **Week extraction.** Weeks are derived from
  ``source.module_id`` (``week_NN_<suffix>``) and exposed as an int.
* **Stable sort.** Default ordering is ``(week ASC, chunk_id ASC)`` so
  that paginated output is deterministic.

The :func:`query_chunks` function is the single entry point; the CLI
command is a thin wrapper that adapts argparse-style flags to it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------- #
# Module-level constants                                                      #
# --------------------------------------------------------------------------- #

CHUNK_TYPES = (
    "explanation",
    "example",
    "exercise",
    "assessment_item",
    "overview",
    "summary",
)
BLOOM_LEVELS = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)
DIFFICULTY_LEVELS = ("foundational", "intermediate", "advanced")
SORT_KEYS = ("week", "chunk_id", "word_count", "bloom")

# Bloom levels in pedagogical order — used for the "bloom" sort key.
_BLOOM_ORDER: Dict[str, int] = {level: i for i, level in enumerate(BLOOM_LEVELS)}

_WEEK_RE = re.compile(r"^week_(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Public dataclasses                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QueryFilter:
    """Composable filter spec for :func:`query_chunks`.

    All list-valued filters use OR semantics within the list and AND
    semantics between filters. ``week_min`` / ``week_max`` form an
    inclusive range; pass equal values for a single-week query.
    """

    chunk_types: Optional[Sequence[str]] = None
    bloom_levels: Optional[Sequence[str]] = None
    difficulties: Optional[Sequence[str]] = None
    week_min: Optional[int] = None
    week_max: Optional[int] = None
    modules: Optional[Sequence[str]] = None
    outcomes: Optional[Sequence[str]] = None  # to-* or co-*; case-insensitive
    text_substring: Optional[str] = None  # case-insensitive
    limit: Optional[int] = None
    offset: int = 0
    sort_key: str = "week"  # one of SORT_KEYS


@dataclass
class QueryResult:
    """Result of a chunk query."""

    total_matches: int
    returned: int
    chunks: List[Dict[str, Any]]
    slug: str
    sort_key: str
    applied_filter: QueryFilter
    # Diagnostic — lets callers see which COs a queried TO rolled up to.
    expanded_outcomes: List[str] = field(default_factory=list)


class ChunkQueryError(Exception):
    """Base error for all query-engine failures."""


class UnknownSlugError(ChunkQueryError):
    """Raised when the slug doesn't resolve to a LibV2 archive."""


class MalformedArchiveError(ChunkQueryError):
    """Raised when the archive layout is missing required files."""


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #


def _archive_root(slug: str, courses_root: Path) -> Path:
    root = courses_root / slug
    if not root.is_dir():
        raise UnknownSlugError(
            f"No LibV2 archive found for slug {slug!r} under {courses_root}"
        )
    return root


def _load_chunks(archive_root: Path) -> List[Dict[str, Any]]:
    chunks_path = archive_root / "corpus" / "chunks.jsonl"
    if not chunks_path.is_file():
        raise MalformedArchiveError(
            f"Archive {archive_root} is missing corpus/chunks.jsonl"
        )
    chunks: List[Dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                chunks.append(json.loads(raw))
            except json.JSONDecodeError as exc:  # pragma: no cover - corruption
                raise MalformedArchiveError(
                    f"chunks.jsonl line {lineno}: {exc}"
                ) from exc
    return chunks


def _load_objectives(archive_root: Path) -> Optional[Dict[str, Any]]:
    """Load ``objectives.json`` if present; return ``None`` otherwise.

    Missing objectives.json is non-fatal — it just means TO→CO rollup
    isn't available and TO queries will only match chunks tagged with
    that exact TO.
    """
    obj_path = archive_root / "objectives.json"
    if not obj_path.is_file():
        return None
    try:
        return json.loads(obj_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - corruption
        raise MalformedArchiveError(f"objectives.json: {exc}") from exc


# --------------------------------------------------------------------------- #
# Outcome rollup                                                              #
# --------------------------------------------------------------------------- #


def _build_to_to_cos(objectives: Optional[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Map each terminal-outcome id (lowercase) to the set of its child CO ids."""
    to_to_cos: Dict[str, Set[str]] = {}
    if not objectives:
        return to_to_cos
    for co in objectives.get("component_objectives") or []:
        parent = (co.get("parent_terminal") or "").strip().lower()
        co_id = (co.get("id") or "").strip().lower()
        if parent and co_id:
            to_to_cos.setdefault(parent, set()).add(co_id)
    return to_to_cos


def _expand_outcomes(
    outcomes: Sequence[str],
    to_to_cos: Dict[str, Set[str]],
) -> Set[str]:
    """Expand ``to-NN`` ids to include their child ``co-NN`` ids.

    ``co-NN`` ids are returned verbatim. Comparison is case-insensitive
    and the returned set is lowercase.
    """
    expanded: Set[str] = set()
    for raw in outcomes:
        norm = (raw or "").strip().lower()
        if not norm:
            continue
        expanded.add(norm)
        if norm.startswith("to-") and norm in to_to_cos:
            expanded |= to_to_cos[norm]
    return expanded


# --------------------------------------------------------------------------- #
# Per-chunk accessors                                                         #
# --------------------------------------------------------------------------- #


def _chunk_week(chunk: Dict[str, Any]) -> Optional[int]:
    """Return the integer week parsed from ``source.module_id``, or ``None``."""
    module_id = (chunk.get("source") or {}).get("module_id") or ""
    match = _WEEK_RE.match(module_id)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:  # pragma: no cover - regex guarantees digits
        return None


def _chunk_module_id(chunk: Dict[str, Any]) -> str:
    return (chunk.get("source") or {}).get("module_id") or ""


def _chunk_outcomes(chunk: Dict[str, Any]) -> Set[str]:
    refs = chunk.get("learning_outcome_refs") or []
    return {(r or "").strip().lower() for r in refs if r}


# --------------------------------------------------------------------------- #
# Filter application                                                          #
# --------------------------------------------------------------------------- #


def _normalize_seq(values: Optional[Sequence[str]]) -> Optional[Set[str]]:
    if values is None:
        return None
    norm = {(v or "").strip().lower() for v in values if v is not None and str(v).strip()}
    return norm or None


def _matches(
    chunk: Dict[str, Any],
    *,
    chunk_types: Optional[Set[str]],
    bloom_levels: Optional[Set[str]],
    difficulties: Optional[Set[str]],
    week_min: Optional[int],
    week_max: Optional[int],
    modules: Optional[Set[str]],
    outcomes_expanded: Optional[Set[str]],
    text_lc: Optional[str],
) -> bool:
    """Apply all filters in selectivity order.

    Order rationale: the cheapest, most-selective filters run first to
    short-circuit on the common-case rejection.
    """
    # 1. Week (most selective for a single-week query).
    if week_min is not None or week_max is not None:
        wk = _chunk_week(chunk)
        if wk is None:
            return False
        if week_min is not None and wk < week_min:
            return False
        if week_max is not None and wk > week_max:
            return False

    # 2. Module id (exact match).
    if modules is not None:
        if _chunk_module_id(chunk).lower() not in modules:
            return False

    # 3. Chunk type.
    if chunk_types is not None:
        if (chunk.get("chunk_type") or "").lower() not in chunk_types:
            return False

    # 4. Difficulty.
    if difficulties is not None:
        if (chunk.get("difficulty") or "").lower() not in difficulties:
            return False

    # 5. Bloom level.
    if bloom_levels is not None:
        if (chunk.get("bloom_level") or "").lower() not in bloom_levels:
            return False

    # 6. Outcome (with TO→CO rollup already applied upstream).
    if outcomes_expanded is not None:
        chunk_outs = _chunk_outcomes(chunk)
        if not (chunk_outs & outcomes_expanded):
            return False

    # 7. Text substring (most expensive — last).
    if text_lc is not None:
        text = chunk.get("text") or ""
        if text_lc not in text.lower():
            return False

    return True


# --------------------------------------------------------------------------- #
# Sorting                                                                     #
# --------------------------------------------------------------------------- #


def _sort_chunks(
    chunks: List[Dict[str, Any]],
    sort_key: str,
) -> List[Dict[str, Any]]:
    """Stable-sort chunks by the requested key.

    Secondary key is always ``id`` ascending so output is deterministic.
    """
    key = (sort_key or "week").lower()
    if key not in SORT_KEYS:
        raise ChunkQueryError(
            f"Unknown sort_key={sort_key!r}; allowed: {', '.join(SORT_KEYS)}"
        )

    def _key_fn(chunk: Dict[str, Any]) -> Tuple[Any, ...]:
        chunk_id = chunk.get("id") or ""
        if key == "week":
            return (_chunk_week(chunk) or 9999, chunk_id)
        if key == "chunk_id":
            return (chunk_id,)
        if key == "word_count":
            return (chunk.get("word_count") or 0, chunk_id)
        if key == "bloom":
            level = (chunk.get("bloom_level") or "").lower()
            return (_BLOOM_ORDER.get(level, len(BLOOM_LEVELS)), chunk_id)
        return (chunk_id,)  # pragma: no cover

    return sorted(chunks, key=_key_fn)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def query_chunks(
    slug: str,
    query: QueryFilter,
    *,
    courses_root: Optional[Path] = None,
) -> QueryResult:
    """Run ``query`` against the chunks for ``slug`` and return a
    :class:`QueryResult`.

    Parameters
    ----------
    slug:
        LibV2 course slug (e.g. ``rdf-shacl-550-rdf-shacl-550``).
    query:
        Composed filter spec.
    courses_root:
        Override for the courses root (tests). Defaults to
        ``LibV2/courses/`` resolved off ``lib.paths.LIBV2_PATH``.

    Raises
    ------
    UnknownSlugError
        If ``slug`` does not resolve to a directory.
    MalformedArchiveError
        If the archive is missing required files.
    ChunkQueryError
        For other validation failures (bad sort_key, etc.).
    """
    # Resolve courses root lazily so test paths can override.
    if courses_root is None:
        from lib.paths import LIBV2_PATH  # noqa: WPS433 - intentional lazy import

        courses_root = LIBV2_PATH / "courses"

    archive = _archive_root(slug, courses_root)
    chunks = _load_chunks(archive)
    objectives = _load_objectives(archive)
    to_to_cos = _build_to_to_cos(objectives)

    # Normalize multi-value filters once.
    chunk_types = _normalize_seq(query.chunk_types)
    bloom_levels = _normalize_seq(query.bloom_levels)
    difficulties = _normalize_seq(query.difficulties)
    modules = _normalize_seq(query.modules)
    text_lc = (
        query.text_substring.lower()
        if query.text_substring is not None and query.text_substring != ""
        else None
    )

    expanded_outcomes: Optional[Set[str]] = None
    if query.outcomes:
        expanded_outcomes = _expand_outcomes(query.outcomes, to_to_cos)
        if not expanded_outcomes:
            expanded_outcomes = None

    matched: List[Dict[str, Any]] = [
        c
        for c in chunks
        if _matches(
            c,
            chunk_types=chunk_types,
            bloom_levels=bloom_levels,
            difficulties=difficulties,
            week_min=query.week_min,
            week_max=query.week_max,
            modules=modules,
            outcomes_expanded=expanded_outcomes,
            text_lc=text_lc,
        )
    ]

    sorted_chunks = _sort_chunks(matched, query.sort_key)

    total = len(sorted_chunks)
    offset = max(0, query.offset or 0)
    if query.limit is not None and query.limit >= 0:
        sliced = sorted_chunks[offset : offset + query.limit]
    else:
        sliced = sorted_chunks[offset:]

    return QueryResult(
        total_matches=total,
        returned=len(sliced),
        chunks=sliced,
        slug=slug,
        sort_key=query.sort_key,
        applied_filter=query,
        expanded_outcomes=sorted(expanded_outcomes) if expanded_outcomes else [],
    )


# --------------------------------------------------------------------------- #
# Helpers for week-range parsing (used by the CLI layer)                      #
# --------------------------------------------------------------------------- #


def parse_week_spec(spec: str) -> Tuple[int, int]:
    """Parse ``"7"`` or ``"1-12"`` into an inclusive ``(min, max)`` tuple.

    Raises :class:`ValueError` on malformed input.
    """
    if spec is None or not str(spec).strip():
        raise ValueError("week spec must be non-empty")
    s = str(spec).strip()
    if "-" in s:
        lo_s, hi_s = s.split("-", 1)
        lo = int(lo_s.strip())
        hi = int(hi_s.strip())
        if lo > hi:
            raise ValueError(f"week range {s!r} has min > max")
        return (lo, hi)
    val = int(s)
    return (val, val)


def parse_csv(value: Optional[str]) -> Optional[List[str]]:
    """Split a comma-separated string into trimmed non-empty tokens.

    Returns ``None`` for ``None`` input so callers can distinguish
    "filter not provided" from "filter provided but empty".
    """
    if value is None:
        return None
    parts = [p.strip() for p in str(value).split(",")]
    parts = [p for p in parts if p]
    return parts or None


def validate_choice(
    values: Optional[Sequence[str]],
    allowed: Sequence[str],
    flag_name: str,
) -> None:
    """Raise :class:`ValueError` if any value isn't in ``allowed``."""
    if not values:
        return
    allowed_set = {a.lower() for a in allowed}
    bad = [v for v in values if v.lower() not in allowed_set]
    if bad:
        raise ValueError(
            f"{flag_name}: invalid value(s) {bad!r}; "
            f"allowed: {', '.join(allowed)}"
        )


# --------------------------------------------------------------------------- #
# Convenience iterator (for callers that want streaming, not list output)    #
# --------------------------------------------------------------------------- #


def iter_chunks(
    slug: str,
    query: QueryFilter,
    *,
    courses_root: Optional[Path] = None,
) -> Iterable[Dict[str, Any]]:
    """Streaming variant of :func:`query_chunks` (yields matched chunks).

    Note: still loads the whole file (chunks.jsonl is small enough);
    the streaming surface is for ergonomics, not memory pressure.
    """
    result = query_chunks(slug, query, courses_root=courses_root)
    for chunk in result.chunks:
        yield chunk

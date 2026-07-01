"""W4.3 cross-course objective-library EXEMPLARS (default OFF).

When planning a NEW course, surface exemplar objectives drawn from the
catalog-enabled LibV2 library as advisory SUGGESTIONS / priors — a
provenance-labeled sidecar the synthesizer/operator can consult for
phrasing/Bloom-band precedent.

Hard anti-fabrication contract
------------------------------
* Exemplars are ADVISORY CONTEXT ONLY. This module NEVER injects another
  course's objective as THIS course's canonical objective — it only reads
  already-archived objectives from OTHER courses and returns them with an
  immutable ``provenance`` label (``libv2:<source_slug>#<objective_id>``).
* The synthesized objective ids/statements written to
  ``synthesized_objectives.json`` are untouched by this module; the
  exemplars land in a SEPARATE sidecar.
* Default OFF (``ED4ALL_OBJECTIVE_LIBRARY_EXEMPLARS`` unset) => strict
  no-op; ``synthesized_objectives.json`` byte-identical.

Everything here is pure-lexical (token Jaccard) — NO embeddings, NO model
load, no network — so it is cheap, deterministic, and safe to run inline
during course_planning.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env knobs (all parse-with-fallback, mirroring the other ED4ALL_* objective
# knobs — garbage/unset => the documented default).
# ---------------------------------------------------------------------------
EXEMPLARS_ENV = "ED4ALL_OBJECTIVE_LIBRARY_EXEMPLARS"
EXEMPLAR_LIMIT_ENV = "ED4ALL_OBJECTIVE_LIBRARY_EXEMPLAR_LIMIT"
EXEMPLAR_MIN_OVERLAP_ENV = "ED4ALL_OBJECTIVE_LIBRARY_EXEMPLAR_MIN_OVERLAP"

_DEFAULT_LIMIT = 8
_DEFAULT_MIN_OVERLAP = 0.05
_TRUTHY = {"1", "true", "yes", "on"}

# Discovery order per course dir: prefer the canonical LibV2 archive form,
# fall back to the staged Courseforge synthesized form.
_OBJECTIVE_REL_PATHS: Tuple[str, ...] = (
    "objectives.json",
    "sources/objectives/synthesized_objectives.json",
)

_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "into", "using",
        "use", "used", "their",
        "how", "why", "what", "which", "when", "where", "who", "will", "can",
        "are", "was", "were", "have", "has", "had", "its", "our", "out", "off",
        "your", "you", "them", "then", "than",
        "over", "under", "such", "both", "each", "some", "any", "all", "not",
        "but", "based", "given", "across", "within", "between", "about",
        "including", "include", "includes", "via", "per", "onto", "upon",
    }
)


def resolve_library_exemplars_enabled(value: Optional[Any] = None) -> bool:
    """True iff ``ED4ALL_OBJECTIVE_LIBRARY_EXEMPLARS`` is a truthy token.

    Parse-with-fallback: anything other than ``1/true/yes/on`` (case-
    insensitive) => off (the default-OFF, byte-identical posture).
    """
    raw = value if value is not None else os.environ.get(EXEMPLARS_ENV, "")
    return str(raw).strip().lower() in _TRUTHY


def resolve_exemplar_limit(value: Optional[Any] = None) -> int:
    """Top-K cap on surfaced exemplars (default 8). Garbage/<1 => default."""
    raw = value if value is not None else os.environ.get(EXEMPLAR_LIMIT_ENV, "")
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        return _DEFAULT_LIMIT
    return n if n >= 1 else _DEFAULT_LIMIT


def resolve_exemplar_min_overlap(value: Optional[Any] = None) -> float:
    """Jaccard floor for an exemplar to be surfaced (default 0.05).

    Out-of-range ``[0.0, 1.0]`` / garbage => default (parse-with-fallback,
    mirroring ``ED4ALL_OBJECTIVE_CHUNK_RELEVANCE_FLOOR``).
    """
    raw = value if value is not None else os.environ.get(
        EXEMPLAR_MIN_OVERLAP_ENV, ""
    )
    try:
        f = float(str(raw).strip())
    except (TypeError, ValueError):
        return _DEFAULT_MIN_OVERLAP
    if 0.0 <= f <= 1.0:
        return f
    return _DEFAULT_MIN_OVERLAP


def _tokenize(text: Any) -> set:
    """Lowercase word-token set (len>2, stopwords removed)."""
    toks = re.findall(r"[a-z0-9]+", str(text).lower())
    return {t for t in toks if len(t) > 2 and t not in _STOPWORDS}


def _iter_component_objectives(raw: Any) -> Iterable[Dict[str, Any]]:
    """Flatten the several chapter/component shapes into plain obj dicts.

    Handles (a) a list of group dicts ``[{"objectives": [...]}, ...]``,
    (b) a flat list of objective dicts, and (c) a dict-of-lists
    ``{"Chapter 1": [...]}`` (the OpenStax archive shape). Defensive: any
    non-dict entry is skipped.
    """
    if isinstance(raw, dict):
        for group in raw.values():
            yield from _iter_component_objectives(group)
        return
    if not isinstance(raw, list):
        return
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        nested = entry.get("objectives")
        if isinstance(nested, list):
            yield from _iter_component_objectives(nested)
        else:
            yield entry


def _to_exemplar(obj: Dict[str, Any], source_slug: str, kind: str) -> Optional[Dict[str, Any]]:
    """Project an archived objective dict to a provenance-labeled exemplar.

    Returns ``None`` when the objective has no usable statement (anti-
    fabrication: we only ever surface REAL archived statements).
    """
    statement = obj.get("statement")
    if not isinstance(statement, str) or not statement.strip():
        return None
    obj_id = str(obj.get("id") or "").strip() or "?"
    return {
        "source_course": source_slug,
        "source_objective_id": obj_id,
        "kind": kind,
        "statement": statement.strip(),
        "bloom_level": obj.get("bloom_level"),
        "bloom_verb": obj.get("bloom_verb"),
        # Immutable provenance label — the anti-fabrication contract: this
        # exemplar belongs to <source_slug>, NOT to the course being planned.
        "provenance": f"libv2:{source_slug}#{obj_id}",
    }


def _extract_objectives(doc: Dict[str, Any], source_slug: str) -> List[Dict[str, Any]]:
    """Pull terminal + component exemplars out of an archived objectives doc."""
    out: List[Dict[str, Any]] = []
    terminals = doc.get("terminal_outcomes") or doc.get("terminal_objectives") or []
    if isinstance(terminals, list):
        for t in terminals:
            if isinstance(t, dict):
                ex = _to_exemplar(t, source_slug, "terminal")
                if ex is not None:
                    out.append(ex)
    components = doc.get("component_objectives") or doc.get("chapter_objectives") or []
    for entry in _iter_component_objectives(components):
        ex = _to_exemplar(entry, source_slug, "component")
        if ex is not None:
            out.append(ex)
    return out


def _iter_library_objective_docs(
    libv2_root: Path, current_slug: Optional[str]
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield ``(slug, objectives_doc)`` for every OTHER catalog course.

    The current course (``current_slug``) is EXCLUDED so we never surface a
    course's own objectives back to itself. Unreadable / malformed docs are
    skipped silently (best-effort discovery).
    """
    courses_root = libv2_root / "courses"
    if not courses_root.is_dir():
        return
    for course_dir in sorted(courses_root.iterdir()):
        if not course_dir.is_dir() or course_dir.name.startswith("."):
            continue
        slug = course_dir.name
        if current_slug and slug == current_slug:
            continue
        for rel in _OBJECTIVE_REL_PATHS:
            path = course_dir / rel
            if not path.is_file():
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict):
                yield slug, doc
            break  # first present shape wins (archive form preferred)


def surface_exemplars(
    *,
    course_name: str,
    query_terms: Iterable[str],
    libv2_root: Optional[Any] = None,
    current_slug: Optional[str] = None,
    limit: Optional[int] = None,
    min_overlap: Optional[float] = None,
) -> Dict[str, Any]:
    """Surface top-K library exemplar objectives ranked by lexical overlap.

    Pure-lexical (token Jaccard) scoring against ``query_terms`` (derived
    from THIS course's synthesized objective statements). Never raises —
    any error/absence degrades to an empty exemplar set. The returned
    ``exemplars`` are advisory + provenance-labeled ONLY.
    """
    from lib.paths import libv2_path  # noqa: PLC0415 — call-time root resolve

    limit = resolve_exemplar_limit(limit)
    min_overlap = resolve_exemplar_min_overlap(min_overlap)
    qset = {t for t in (query_terms or []) if t}

    result: Dict[str, Any] = {
        "enabled": True,
        "course_name": course_name,
        "query_term_count": len(qset),
        "candidate_count": 0,
        "source_course_count": 0,
        "exemplar_count": 0,
        "limit": limit,
        "min_overlap": min_overlap,
        "exemplars": [],
        "note": (
            "advisory suggestions only; never injected as this course's "
            "canonical objectives"
        ),
    }
    if not qset:
        return result

    try:
        root = Path(libv2_root) if libv2_root else libv2_path()
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.warning("library-exemplars: root resolve failed (%s); skipping.", exc)
        return result

    scored: List[Tuple[float, Dict[str, Any]]] = []
    candidate = 0
    try:
        for slug, doc in _iter_library_objective_docs(root, current_slug):
            for ex in _extract_objectives(doc, slug):
                candidate += 1
                otoks = _tokenize(ex["statement"])
                if not otoks:
                    continue
                union = qset | otoks
                if not union:
                    continue
                overlap = len(qset & otoks) / len(union)
                if overlap < min_overlap:
                    continue
                scored.append((overlap, ex))
    except Exception as exc:  # noqa: BLE001 — best-effort discovery
        logger.warning(
            "library-exemplars: discovery failed (%s); returning no exemplars.",
            exc,
        )
        return result

    # Deterministic order: relevance desc, then stable by provenance.
    scored.sort(key=lambda pair: (-pair[0], pair[1]["provenance"]))
    top = scored[:limit]

    result["candidate_count"] = candidate
    result["source_course_count"] = len({ex["source_course"] for _, ex in top})
    result["exemplar_count"] = len(top)
    result["exemplars"] = [
        {**ex, "relevance": round(score, 4)} for score, ex in top
    ]
    return result


def build_query_terms(
    terminals: Iterable[Dict[str, Any]],
    chapter_objectives: Iterable[Dict[str, Any]],
) -> List[str]:
    """Token set (as a sorted list) from this course's objective statements.

    Used as the retrieval query for exemplar surfacing — the exemplars are
    ranked by lexical overlap against THIS course's own (grounded) objective
    phrasing, so the surfaced priors stay topically anchored.
    """
    terms: set = set()
    for obj in terminals or []:
        if isinstance(obj, dict):
            terms |= _tokenize(obj.get("statement", ""))
    for entry in _iter_component_objectives(list(chapter_objectives or [])):
        terms |= _tokenize(entry.get("statement", ""))
    return sorted(terms)

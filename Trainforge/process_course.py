#!/usr/bin/env python3
"""
Generic Course Corpus Pipeline

Processes any Courseforge IMSCC package into a Sourceforge-compatible
RAG corpus for LibV2 import.

Usage:
    python -m Trainforge.process_course \
        --imscc <IMSCC_PATH> \
        --course-code <course-code> \
        --division ARTS --domain education --subdomain instructional-design \
        --output <OUTPUT_DIR>

    # With objectives file for Bloom's-based difficulty mapping:
    python -m Trainforge.process_course \
        --imscc <IMSCC_PATH> \
        --objectives <OBJECTIVES_JSON> \
        --course-code <course-code> \
        --division ARTS --domain education \
        --output <OUTPUT_DIR> \
        --import-to-libv2
"""

import argparse
import hashlib
import html.parser
import json
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Make repository-owned packages importable during direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.decision_capture import DecisionCapture
from lib.ontology import concept_tagging as _concept_tagging
from lib.ontology.slugs import canonical_slug

# Delegate chunk construction to Trainforge.chunker while retaining wrappers
# and constant aliases as the public CourseProcessor compatibility surface.
from Trainforge.chunker import (
    CANONICAL_CHUNK_TYPES as _PKG_CANONICAL_CHUNK_TYPES,
)
from Trainforge.chunker import (
    MAX_CHUNK_SIZE as _PKG_MAX_CHUNK_SIZE,
)
from Trainforge.chunker import (
    MIN_CHUNK_SIZE as _PKG_MIN_CHUNK_SIZE,
)
from Trainforge.chunker import (
    TARGET_CHUNK_SIZE as _PKG_TARGET_CHUNK_SIZE,
)
from Trainforge.chunker import (
    ChunkerContext as _ChunkerContext,
)
from Trainforge.chunker import (
    chunk_content as _pkg_chunk_content,
)
from Trainforge.chunker import (
    chunk_text_block as _pkg_chunk_text_block,
)
from Trainforge.chunker import (
    merge_small_sections as _pkg_merge_small_sections,
)
from Trainforge.generators.postprocessing import summary_factory
from Trainforge.parsers.html_content_parser import HTMLContentParser, HTMLTextExtractor

# The canonical chunker owns XPath resolution for chunk boundaries.
from Trainforge.rag.boilerplate_detector import (
    BoilerplateConfig,
    contamination_rate,
    detect_repeated_ngrams,
)
from Trainforge.rag.wcag_canonical_names import canonicalize_sc_references

# Bumped whenever the semantics of quality_report.json metrics change.
# v1: field-presence metrics (legacy).
# v2: referential, structural, and content-sanity metrics.
# v3: adds outcome_reverse_coverage (metric) + integrity.uncovered_outcomes
#     (list); guaranteed bloom_level on every chunk via verb/default fallback;
#     pedagogy_model.json grows module_sequence, bloom_progression,
#     prerequisite_chain, prerequisite_violations. (Session 1)
# v4: adds five flow metrics that surface silent metadata drops:
#     content_type_label_coverage, key_terms_coverage,
#     key_terms_with_definitions_rate, misconceptions_present_rate,
#     interactive_components_rate. See docs/operations/flow-metrics.md.
# v5: adds top-level `package_completeness` aggregate — a flat mean of the
#     five enrichment coverage fractions. Answers "of the metadata this
#     package claims to provide, how much actually landed." NOT inside
#     `metrics`; NOT weighted into `overall_quality_score`. Separate
#     top-level key so consumers can read one honest number without
#     cross-referencing five metrics.
METRICS_SEMANTIC_VERSION = 5

# Chunk schema version for the enrichment and provenance contract. v4 adds:
#   - `summary`: 2–3 sentence extractive summary per chunk.
#   - `retrieval_text` (optional): summary + " " + key_terms_joined.
#   - `schema_version`: stamped on every chunk.
#   - `source.html_xpath` and `source.char_span`: audit-trail
#     provenance stamped on every chunk.
# The string also lands on manifest.json as `chunk_schema_version`. One bump
# per release train; see ADR-001 Contract 1 and docs/architecture/workers.md
# for the rebase protocol.
CHUNK_SCHEMA_VERSION = "v4"


# Canonical ChunkType enum gating ``data-cf-template-type`` propagation in
# ``_merge_small_sections``. When a
# Courseforge HTML section carries ``data-cf-template-type="<value>"`` and
# ``<value>`` is in this set, the chunker uses it as the chunk_type instead of
# the heading-keyword heuristic. Values outside the set fall back to the
# heuristic so off-spec corpora can't break downstream consumers that key off
# chunk_type. Source of truth:
# ``schemas/taxonomies/content_type.json::ChunkType``.
#
# lifted into the Trainforge.chunker package
# (``Trainforge.chunker.chunker.CANONICAL_CHUNK_TYPES``). Re-exported here so
# external importers (``scripts/archive/wave81_reclassify_chunks.py``) keep
# working without modification.
CANONICAL_CHUNK_TYPES = _PKG_CANONICAL_CHUNK_TYPES


# opt-in content-hash chunk IDs. When
# TRAINFORGE_CONTENT_HASH_IDS=true, chunk IDs are derived from
# sha256(text + source_locator + schema_version) so re-chunking the same
# source produces identical IDs; this keeps edge-evidence references that
# quote chunk IDs stable across re-runs. Default remains position-based for
# backward compatibility with already-ingested LibV2 courses.
USE_CONTENT_HASH_IDS = os.getenv("TRAINFORGE_CONTENT_HASH_IDS", "").lower() == "true"


# env-var-first target-models resolution for the
# operator-readable `dataset_config.json::target_models` list. Default
# preserves the previous hardcoded `["claude-opus-4-6",
# "claude-sonnet-4-6"]` pair; operators retraining against a different
# teacher set TRAINFORGE_TARGET_MODELS as a comma-separated list.
TARGET_MODELS_ENV = "TRAINFORGE_TARGET_MODELS"
TARGET_MODELS_DEFAULT = ("claude-opus-4-6", "claude-sonnet-4-6")


def _resolve_target_models() -> List[str]:
    """Pick the effective target-models list for ``dataset_config.json``.

    Priority order:
      1. ``TRAINFORGE_TARGET_MODELS`` env var (CSV) when set.
      2. ``TARGET_MODELS_DEFAULT`` (preserves legacy emit byte-shape).

    Whitespace per token is trimmed; empty tokens (e.g. trailing comma)
    are dropped. An env var that resolves to zero tokens falls back to
    the default rather than emitting an empty list, since downstream
    consumers expect ``target_models`` to be non-empty.
    """
    raw = os.environ.get(TARGET_MODELS_ENV)
    if raw is None:
        return list(TARGET_MODELS_DEFAULT)
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        return list(TARGET_MODELS_DEFAULT)
    return tokens


def _generate_chunk_id(prefix: str, start_id: int, text: str, source_locator: str) -> str:
    """Generate a chunk ID.

    Default (legacy): position-based ``f"{prefix}{start_id:05d}"``.

    When ``TRAINFORGE_CONTENT_HASH_IDS=true``: content-addressed
    ``f"{prefix}{sha256(text|source_locator|v4)[:16]}"``, stable across
    re-chunks. Reads the env var on each call so tests can flip it via
    ``monkeypatch.setenv`` without module reloads.
    """
    if os.getenv("TRAINFORGE_CONTENT_HASH_IDS", "").lower() == "true":
        payload = f"{text}|{source_locator}|{CHUNK_SCHEMA_VERSION}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}{digest}"
    return f"{prefix}{start_id:05d}"


# opt-in chunk validation against chunk_v4.schema.json.
# The schema plus its taxonomy $ref store is cached after first
# load. jsonschema is imported lazily so this module stays importable when the
# dependency is missing (same pattern as lib/validation.py::load_schema).
_CHUNK_VALIDATOR: Any = None
_CHUNK_SCHEMA_LOAD_FAILED: bool = False


def _load_chunk_validator() -> Any:
    """Build and cache a Draft202012Validator for chunk_v4.schema.json.

    The validator is wired up so every ``$ref`` — inline pointers like
    ``#/$defs/Source`` and external URIs like
    ``https://ed4all.dev/schemas/knowledge/source_reference.schema.json``
    — resolves offline against every schema under ``schemas/`` keyed by
    its ``$id``.

    The shared Draft 2020-12 validator builder resolves inline and external
    references deterministically against the offline schema registry.
    Returns ``None`` when validation is unavailable.
    """
    global _CHUNK_VALIDATOR, _CHUNK_SCHEMA_LOAD_FAILED
    if _CHUNK_VALIDATOR is not None:
        return _CHUNK_VALIDATOR
    if _CHUNK_SCHEMA_LOAD_FAILED:
        return None

    schemas_root = PROJECT_ROOT / "schemas"
    schema_path = schemas_root / "knowledge" / "chunk_v4.schema.json"
    if not schema_path.exists():
        _CHUNK_SCHEMA_LOAD_FAILED = True
        return None

    # W-D6: lift the Draft 2020-12 + ``referencing.Registry`` construction
    # to :func:`lib.utils.build_validator`. The deprecated ``RefResolver``
    # fallback is dropped — the project depends on ``referencing``
    # everywhere else, so the historical defensive branch is no longer
    # load-bearing.
    try:
        from lib.utils import build_validator

        validator = build_validator(schema_path, registry_root=schemas_root)
    except Exception:
        _CHUNK_SCHEMA_LOAD_FAILED = True
        return None

    if validator is None:
        _CHUNK_SCHEMA_LOAD_FAILED = True
        return None
    _CHUNK_VALIDATOR = validator
    return _CHUNK_VALIDATOR


def _validate_chunk(chunk: Dict[str, Any]) -> Optional[str]:
    """Validate a single chunk against chunk_v4.schema.json.

    Returns a formatted error string on first failure, or None on success.
    Also returns None when the schema/validator cannot be loaded (missing
    jsonschema dep, missing schema file) so the hook stays non-fatal during
    bootstrap.
    """
    validator = _load_chunk_validator()
    if validator is None:
        return None
    errors = sorted(
        validator.iter_errors(chunk), key=lambda e: list(e.absolute_path)
    )
    if not errors:
        return None
    first = errors[0]
    path = ".".join(str(p) for p in first.absolute_path) or "root"
    return f"{path}: {first.message}"


# Maps each temporary _metadata_trace value to the VERSIONING.md §4.4a
# hypothesis it diagnoses.
_HYPOTHESIS_BY_TRACE: Dict[str, str] = {
    "jsonld_section_match": "-",
    "jsonld_section_match_empty": "H3",  # short-circuit signature on key_terms
    "data_cf_fallback": "-",
    "none_no_jsonld_sections": "H2",
    "none_jsonld_parse_failed": "H5",
    "none_heading_mismatch": "H1",
    "none_no_sections_path": "H4",
    "section_jsonld": "-",
    "page_jsonld": "-",
    "lo_inherited": "-",
    "verbs": "-",
    "default": "-",
    "jsonld_page_misconceptions": "-",
    "none": "?",
}


class PipelineIntegrityError(RuntimeError):
    """Raised by :class:`CourseProcessor` in strict_mode when quality_report
    integrity invariants fail before writing final metadata.
    """

# ---------------------------------------------------------------------------
# Bloom's → difficulty mapping
# ---------------------------------------------------------------------------

BLOOM_TO_DIFFICULTY = {
    "remember": "foundational",
    "understand": "foundational",
    "apply": "intermediate",
    "analyze": "intermediate",
    "evaluate": "advanced",
    "create": "advanced",
}

# Numeric weights for median-based difficulty calculation
BLOOM_WEIGHT = {
    "remember": 1, "understand": 2, "apply": 3,
    "analyze": 4, "evaluate": 5, "create": 6,
}

# ---------------------------------------------------------------------------
# Bloom level + module_id canonicalization helpers.
# ---------------------------------------------------------------------------

# Canonical Bloom levels in ascending cognitive order. Mirrors
# lib.ontology.bloom.BLOOM_LEVELS; duplicated here to avoid circular import
# at module load.
_CANONICAL_BLOOM_LEVELS: Tuple[str, ...] = (
    "remember", "understand", "apply", "analyze", "evaluate", "create",
)
_BLOOM_RANK = {level: idx for idx, level in enumerate(_CANONICAL_BLOOM_LEVELS)}


def canonicalize_bloom_level(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a possibly-compound bloom_level value into (primary, secondary).

    External pipelines occasionally emit compound levels like
    ``"remember-apply"`` or ``"understand-analyze"`` to express that a
    chunk straddles two cognitive demands. The chunk_v4 schema only
    permits single canonical values, so we split on ``-`` and keep the
    HIGHER level (per Bloom's ordering) as the primary, storing the
    LOWER as ``bloom_level_secondary``. When
    the input is already a single canonical level, ``secondary`` is
    ``None``. Unknown / unparseable values pass through as-is with no
    secondary so the caller can decide whether to keep, drop, or warn.

    Returns:
        (primary, secondary). ``primary`` is ``None`` only when ``value``
        itself is None / empty.
    """
    if not value or not isinstance(value, str):
        return None, None
    raw = value.strip().lower()
    if not raw:
        return None, None
    if raw in _BLOOM_RANK:
        return raw, None
    if "-" in raw:
        parts = [p.strip() for p in raw.split("-") if p.strip()]
        canonical_parts = [p for p in parts if p in _BLOOM_RANK]
        if len(canonical_parts) >= 2:
            ordered = sorted(canonical_parts, key=lambda p: _BLOOM_RANK[p])
            # Higher Bloom level becomes primary; lower becomes secondary.
            return ordered[-1], ordered[0]
        if len(canonical_parts) == 1:
            return canonical_parts[0], None
    # Unknown value — pass through unchanged so downstream warn/log paths can
    # surface it without losing information.
    return raw, None


_WEEK_PREFIXED_RE = re.compile(r"^week_\d{2,}_")


def normalize_module_id(
    raw_module_id: Optional[str],
    item_path: Optional[str] = None,
    week_num: Optional[int] = None,
) -> Tuple[Optional[str], bool]:
    """Normalize a chunk's ``source.module_id`` to canonical ``week_NN_<slot>``.

    Strategy:
      * If ``raw_module_id`` already starts with ``week_NN_``, return it
        unchanged.
      * Otherwise, derive the week number from ``item_path`` (e.g.
        ``week_04/application.html``) or ``week_num`` and prepend
        ``week_NN_``.
      * If no week info is recoverable, return the input unchanged so
        no chunk is silently dropped — the caller is expected to warn.

    Returns ``(normalized_id, was_normalized)``. ``was_normalized`` is
    ``True`` only when the function actually rewrote the value.
    """
    if not raw_module_id:
        return raw_module_id, False
    mid = raw_module_id.strip()
    if not mid:
        return raw_module_id, False
    if _WEEK_PREFIXED_RE.match(mid):
        return mid, False
    # Try to recover the week number.
    week: Optional[int] = None
    if isinstance(week_num, int) and week_num > 0:
        week = week_num
    if week is None and item_path:
        m = re.search(r"week[_\-\s]?(\d+)", item_path, re.IGNORECASE)
        if m:
            try:
                week = int(m.group(1))
            except ValueError:
                week = None
    if week is None or week <= 0:
        return mid, False
    return f"week_{week:02d}_{mid}", True


def _assert_chunk_files_parity(jsonl_path: Path, json_path: Path) -> None:
    """Verify chunks.jsonl and chunks.json round-trip to the same chunk list.

    Raises ``RuntimeError`` when the ordered content differs between the
    streaming JSONL and bundled JSON representations.
    """
    jsonl_chunks: List[Dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                jsonl_chunks.append(json.loads(line))
    with open(json_path, encoding="utf-8") as f:
        json_chunks = json.load(f)
    if not isinstance(json_chunks, list):
        raise RuntimeError(
            f"chunks.json parity: expected top-level list, got "
            f"{type(json_chunks).__name__}"
        )
    if len(jsonl_chunks) != len(json_chunks):
        raise RuntimeError(
            f"chunks.json/.jsonl parity: line count mismatch — "
            f"jsonl={len(jsonl_chunks)} vs json={len(json_chunks)}"
        )
    for idx, (a, b) in enumerate(zip(jsonl_chunks, json_chunks, strict=False)):
        if a != b:
            raise RuntimeError(
                f"chunks.json/.jsonl parity: chunk index {idx} differs "
                f"(jsonl id={a.get('id')!r}, json id={b.get('id')!r})"
            )


# Resource types that cap difficulty one level below week max
# (overviews and summaries are inherently introductory)
INTRODUCTORY_RESOURCE_TYPES = {"overview", "summary"}

# ---------------------------------------------------------------------------
# Resource type classification patterns
# ---------------------------------------------------------------------------

RESOURCE_TYPE_PATTERNS = [
    # quiz / self-check / assessment
    (re.compile(r"self[_-]?check|quiz|assessment", re.I), "quiz"),
    # overview / introduction
    (re.compile(r"overview|introduction", re.I), "overview"),
    # summary / recap
    (re.compile(r"summary|recap", re.I), "summary"),
    # discussion
    (re.compile(r"discussion", re.I), "discussion"),
    # application / activity
    (re.compile(r"application|activity", re.I), "application"),
]


def classify_resource(path: str) -> Tuple[str, str, str]:
    """
    Classify an HTML resource and extract module info from its path.

    Returns:
        (resource_type, module_id, module_title)
    """
    stem = Path(path).stem
    path_lower = path.lower()

    # Determine resource type
    resource_type = "page"  # default
    for pattern, rtype in RESOURCE_TYPE_PATTERNS:
        if pattern.search(path_lower):
            resource_type = rtype
            break

    module_id = stem
    # Build a human-readable title from the stem
    # Strip leading week_XX_ or section_XX_ prefix, then the content_XX_ prefix
    title = stem
    title = re.sub(r"^(?:week|section)_\d+_", "", title)
    title = re.sub(r"^(?:content|module)_\d+_", "", title)
    module_title = title.replace("_", " ").strip().title() or stem.replace("_", " ").title()

    return resource_type, module_id, module_title


def extract_week_number(path: str) -> int:
    """Extract week/section number from path. Returns 0 if not found."""
    m = re.search(r"(?:week|section)[_-]?(\d+)", path, re.I)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Objectives loader
# ---------------------------------------------------------------------------

def load_objectives(objectives_path: Path) -> Dict[str, Any]:
    """
    Load objectives JSON and build week→bloom mapping.

    Returns dict with keys:
        terminal_objectives: list
        chapter_objectives: list
        week_bloom_map: {week_num: [bloom_levels]}
        bloom_distribution: {level: count}
        description: str

    Accept both supported objective-file schemas and expose a normalized
    shape to downstream graph and outcome consumers.
    """
    with open(objectives_path) as f:
        data = json.load(f)

    # Normalize both schemas to the field names consumed in this module.
    terminal_list = (
        data.get("terminal_objectives")
        or data.get("terminal_outcomes")
        or []
    )
    chapter_list = (
        data.get("chapter_objectives")
        or data.get("component_objectives")
        or []
    )

    week_bloom: Dict[int, List[str]] = defaultdict(list)

    for chapter in chapter_list:
        if not isinstance(chapter, dict):
            continue
        # Two shapes for a chapter entry:
        #   nested: {"chapter": "Week 1-2: ...", "objectives": [{...}]}
        #   flat:   {"id": "co-01", "week": 1, "bloom_level": "remember", ...}
        chapter_name = chapter.get("chapter", "")
        week_match = re.search(r"[Ww]eek\s+(\d+)(?:\s*-\s*(\d+))?", chapter_name)
        if week_match:
            start = int(week_match.group(1))
            end = int(week_match.group(2)) if week_match.group(2) else start
            weeks = list(range(start, end + 1))
        else:
            weeks = []

        if "objectives" in chapter and isinstance(chapter.get("objectives"), list):
            inner = chapter["objectives"]
        else:
            inner = [chapter]
            single_week = chapter.get("week")
            if isinstance(single_week, int):
                weeks = [single_week]

        for obj in inner:
            if not isinstance(obj, dict):
                continue
            bloom = (obj.get("bloomLevel") or obj.get("bloom_level") or "").lower()
            if bloom:
                for w in weeks:
                    week_bloom[w].append(bloom)

    return {
        "terminal_objectives": terminal_list,
        "chapter_objectives": chapter_list,
        "week_bloom_map": dict(week_bloom),
        "bloom_distribution": data.get("bloom_distribution", {}),
        "description": data.get("description", ""),
        "course_title": data.get("course_title", ""),
        # Optional per-course domain concept seeds. Shape:
        #   [{"id": "pour", "aliases": ["POUR", "perceivable operable"]}, ...]
        # CONCEPT_PATTERNS covers pedagogy terms only, so domain seeds are
        # the only text-based extraction path for course-specific vocabulary.
        "domain_concepts": data.get("domain_concepts", []),
    }


def compile_domain_concept_seeds(
    raw: List[Dict[str, Any]],
) -> List[Tuple[str, List[re.Pattern]]]:
    """Compile the domain_concepts block from an objectives file into
    (canonical_tag, [word-boundary regex]) pairs for fast matching.

    Aliases are matched case-insensitively with \\b word boundaries so that
    short tokens (``aria``, ``udl``) don't match inside longer words.
    """
    seeds: List[Tuple[str, List[re.Pattern]]] = []
    for entry in raw or []:
        canonical = normalize_tag(entry.get("id", ""))
        if not canonical:
            continue
        aliases = list(entry.get("aliases") or [])
        if entry.get("id") and entry["id"] not in aliases:
            aliases.append(entry["id"])
        patterns: List[re.Pattern] = []
        for alias in aliases:
            alias = str(alias).strip()
            if not alias:
                continue
            patterns.append(re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE))
        if patterns:
            seeds.append((canonical, patterns))
    return seeds


# ---------------------------------------------------------------------------
# learning_outcome_ref normalization
# ---------------------------------------------------------------------------

def normalize_outcome_refs(raw_refs: Any) -> List[str]:
    """Normalize an iterable of learning_outcome_refs.

    Split comma-delimited learning-outcome codes so downstream resolvers
    receive one reference per element. Preserve order and collapse duplicates.

    Accepts ``None``, a single string, or any iterable of strings.
    Returns ``[]`` for ``None`` / empty input. Non-string elements are
    coerced via ``str()`` defensively.
    """
    if raw_refs is None:
        return []
    if isinstance(raw_refs, str):
        raw_refs = [raw_refs]
    out: List[str] = []
    seen: Set[str] = set()
    for raw in raw_refs:
        if raw is None:
            continue
        if not isinstance(raw, str):
            raw = str(raw)
        if "," in raw:
            parts = [p.strip() for p in raw.split(",")]
        else:
            parts = [raw.strip()]
        for p in parts:
            if not p:
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Concept tag normalization
# ---------------------------------------------------------------------------

def normalize_tag(raw: str) -> str:
    """Normalize a concept string to lowercase-hyphenated tag.

    Delegates identity to ``lib.ontology.slugs.canonical_slug``, then applies
    Trainforge display constraints for LibV2 URL segments: HTML entity
    decoding, a four-token limit, and an alphabetic first character.
    """
    if raw is None:
        return ""
    # decode HTML entities BEFORE slugification so entity
    # glue tokens (mdash, ndash, hellip, etc.) never enter the slug.
    decoded = html.unescape(str(raw))
    tag = canonical_slug(decoded)
    # Limit to 4 words (display-layer cap specific to LibV2 tag URLs).
    parts = tag.split("-")
    if len(parts) > 4:
        tag = "-".join(parts[:4])
    # Tags must start with a letter (LibV2 lowercase-hyphenated format).
    if tag and not tag[0].isalpha():
        return ""
    return tag


# ---------------------------------------------------------------------------
# Enrichment fallbacks (v1.0 roadmap — see VERSIONING.md §6)
# ---------------------------------------------------------------------------

# Bloom's verb → level map. Populated with the canonical verbs per level.
# Used when JSON-LD / data-cf-* don't declare a bloom_level.
BLOOM_VERB_MAP: Dict[str, str] = {
    # Remember
    "define": "remember", "list": "remember", "recall": "remember",
    "identify": "remember", "name": "remember", "state": "remember",
    "recognize": "remember",
    # Understand
    "explain": "understand", "describe": "understand", "summarize": "understand",
    "interpret": "understand", "paraphrase": "understand", "classify": "understand",
    "compare": "understand",
    # Apply
    "apply": "apply", "demonstrate": "apply", "use": "apply",
    "solve": "apply", "implement": "apply", "execute": "apply",
    "illustrate": "apply",
    # Analyze
    "analyze": "analyze", "differentiate": "analyze", "examine": "analyze",
    "contrast": "analyze", "organize": "analyze", "deconstruct": "analyze",
    # Evaluate
    "evaluate": "evaluate", "assess": "evaluate", "critique": "evaluate",
    "judge": "evaluate", "justify": "evaluate", "argue": "evaluate",
    # Create
    "create": "create", "design": "create", "develop": "create",
    "construct": "create", "produce": "create", "formulate": "create",
}

# Stop-sets partitioning concept vs pedagogy nodes in the graph output.
# These are defensive: _extract_concept_tags already filters NON_CONCEPT_TAGS,
# but the graph-level partition is cheap and survives upstream drift.
PEDAGOGY_TAG_SET: Set[str] = {v for v in BLOOM_VERB_MAP}
LOGISTICS_TAG_SET: Set[str] = {
    "initial-post", "replies", "due", "guidelines",
    "correct", "incorrect", "submit", "deadline", "grading",
    "readings", "resources", "learning-objectives",
    "estimated-time", "time", "minutes", "hours",
    # "feedback" is legitimate pedagogy vocabulary (formative feedback in
    # course theory courses). For domain courses it reliably pollutes the
    # concept graph via boilerplate like "you'll receive immediate feedback"
    # in quiz intros. Routing it to pedagogy_graph keeps the signal without
    # polluting the domain graph.
    "feedback",
}

# Divs carrying these attribute prefixes are atomic — the chunker must not
# split through them regardless of word-count target.
ATOMIC_BLOCK_SELECTOR_PREFIXES: Tuple[str, ...] = (
    "data-cf-role", "data-cf-objective-id", "data-cf-content-type",
)

_MISCONCEPTION_PATTERNS = [
    re.compile(r"\b(?:Common\s+mistake|A\s+common\s+misconception|Students\s+often\s+think|Contrary\s+to\s+popular\s+belief)[:,]?\s+([^.]+\.)", re.IGNORECASE),
    re.compile(r"\b(?:It\s+is\s+a\s+myth\s+that|Many\s+learners\s+assume\s+that)\s+([^.]+\.)", re.IGNORECASE),
]

_KEY_TERM_TAG_RE = re.compile(
    r"<(?P<tag>strong|b|dfn)\b[^>]*>(?P<term>[^<]{2,60})</(?P=tag)>",
    re.IGNORECASE,
)
_DEF_SENTENCE_RE = re.compile(r"[^.]*\.")


def derive_bloom_from_verbs(text: str) -> Optional[str]:
    """Pick the dominant Bloom's level from verb frequencies in ``text``.

    Used as a fallback when JSON-LD / data-cf-* didn't specify a bloom level
    for the chunk. Returns None when no known Bloom verb appears.
    """
    if not text:
        return None
    counts: Dict[str, int] = defaultdict(int)
    for match in re.finditer(r"\b([a-zA-Z]+)\b", text):
        verb = match.group(1).lower()
        level = BLOOM_VERB_MAP.get(verb)
        if level:
            counts[level] += 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def extract_key_terms_from_html(html: str) -> List[Dict[str, str]]:
    """Extract bold/definition terms from an HTML fragment.

    Pairs each term with the sentence that contains it as a best-effort
    definition. Used as a fallback when JSON-LD keyTerms are absent.
    """
    if not html:
        return []
    seen: Set[str] = set()
    results: List[Dict[str, str]] = []
    # Build a plain-text sentence list for definition lookup
    extractor = HTMLTextExtractor()
    extractor.feed(html)
    plain = extractor.get_text()
    sentences = _DEF_SENTENCE_RE.findall(plain)

    for m in _KEY_TERM_TAG_RE.finditer(html):
        term = m.group("term").strip()
        if not term or len(term) < 2:
            continue
        low = term.lower()
        if low in seen:
            continue
        seen.add(low)
        definition = ""
        for sentence in sentences:
            if low in sentence.lower():
                definition = sentence.strip()
                break
        results.append({"term": term, "definition": definition})
    return results


_VOID_HTML_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


class _BalanceChecker(html.parser.HTMLParser):
    """Minimal stack-based HTML tag-balance checker.

    Returns True iff every opened non-void tag is closed in order. Self-closing
    forms (``<br/>``) and void elements (``<img>``) are not required to close.
    """

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self._stack: List[str] = []
        self._balanced = True

    @classmethod
    def check(cls, html_text: str) -> bool:
        inst = cls()
        try:
            inst.feed(html_text)
            inst.close()
        except Exception:
            return False
        return inst._balanced and not inst._stack

    @classmethod
    def unclosed(cls, html_text: str) -> List[str]:
        inst = cls()
        try:
            inst.feed(html_text)
            inst.close()
        except Exception:
            return ["<parse_error>"]
        return list(inst._stack)

    def handle_starttag(self, tag, attrs):
        if tag.lower() in _VOID_HTML_TAGS:
            return
        self._stack.append(tag.lower())

    def handle_startendtag(self, tag, attrs):
        return

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in _VOID_HTML_TAGS:
            return
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()
        elif tag in self._stack:
            # Tags closed out of order — pop until we find it.
            while self._stack and self._stack[-1] != tag:
                self._stack.pop()
            if self._stack:
                self._stack.pop()
            self._balanced = False
        else:
            self._balanced = False


def extract_misconceptions_from_text(text: str) -> List[Dict[str, str]]:
    """Regex-match common misconception prose patterns.

    Returns a list of ``{"misconception": ..., "correction": ""}`` dicts.
    """
    if not text:
        return []
    found: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for pattern in _MISCONCEPTION_PATTERNS:
        for m in pattern.finditer(text):
            statement = m.group(1).strip()
            if not statement:
                continue
            key = statement.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append({"misconception": statement, "correction": ""})
    return found


# token-overlap match for misconception → concept_tag routing.
# This prevents ``interferes_with`` edges from defaulting to the first concept
# tag when another tag is more relevant. Pure function; deterministic; ties
# break by tag-list position.
_ROUTING_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ROUTING_STOPWORDS = frozenset({
    # Conservative stopword list — drops only the highest-frequency
    # tokens that contribute zero discriminating signal. Domain-bearing
    # terms (rdf, owl, shacl) are not stopwords; they should match.
    "a", "an", "the", "of", "in", "on", "at", "by", "to", "for",
    "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "as", "if",
    "with", "from", "into", "than", "then", "so", "such", "not",
    "no", "do", "does", "did", "have", "has", "had", "will",
    "can", "may", "might", "should", "would", "could", "must",
    "you", "your", "we", "our", "they", "them", "their",
})


def _route_misconception_to_tag(
    statement: str, tags: List[str]
) -> Optional[str]:
    """Pick the chunk concept_tag whose slug tokens overlap ``statement`` most.

    Slug tokens are derived by splitting each tag on ``-`` and lowercasing.
    Statement tokens are word-extracted via :data:`_ROUTING_TOKEN_RE` minus
    stopwords. Both sides are also passed through
    :func:`lib.ontology.concept_classifier.singular_form` so a statement
    saying "triple" matches a ``triples`` tag (and vice versa). Score =
    number of tag tokens (deduplicated, across both surface + singular
    forms) that appear in the statement token set.

    Returns the highest-scoring tag (ties break by ``tags`` list order so
    the result is deterministic and reproduces the legacy first-tag
    behavior when no overlap exists). Returns the first tag in ``tags``
    when ``statement`` is empty or all candidate scores are zero.
    Returns None when ``tags`` is empty.
    """
    if not tags:
        return None
    if not statement:
        return tags[0]
    from lib.ontology.concept_classifier import singular_form

    raw_tokens = [
        t for t in _ROUTING_TOKEN_RE.findall(statement.lower())
        if t and t not in _ROUTING_STOPWORDS
    ]
    statement_tokens = set(raw_tokens) | {singular_form(t) for t in raw_tokens}
    if not statement_tokens:
        return tags[0]
    best_tag = tags[0]
    best_score = 0
    for tag in tags:
        raw_tag_tokens = [
            t for t in tag.lower().split("-")
            if t and t not in _ROUTING_STOPWORDS
        ]
        tag_tokens = set(raw_tag_tokens) | {
            singular_form(t) for t in raw_tag_tokens
        }
        if not tag_tokens:
            continue
        score = len(tag_tokens & statement_tokens)
        if score > best_score:
            best_score = score
            best_tag = tag
    return best_tag


# ---------------------------------------------------------------------------
# CourseProcessor
# ---------------------------------------------------------------------------

class CourseProcessor:
    """Generic processor that turns a Courseforge IMSCC into a Trainforge corpus."""

    # chunk-size constants are sourced from the
    # Trainforge.chunker package so the package + Trainforge can never drift.
    # The class-attribute aliases stay so existing call sites that read
    # ``self.MIN_CHUNK_SIZE`` / ``self.MAX_CHUNK_SIZE`` /
    # ``self.TARGET_CHUNK_SIZE`` keep working unchanged.
    TARGET_CHUNK_SIZE = _PKG_TARGET_CHUNK_SIZE
    MIN_CHUNK_SIZE = _PKG_MIN_CHUNK_SIZE  # Courseforge pages can be short (overviews, summaries)
    MAX_CHUNK_SIZE = _PKG_MAX_CHUNK_SIZE

    def __init__(
        self,
        imscc_path: str,
        output_dir: str,
        course_code: str,
        division: Optional[str] = None,
        domain: Optional[str] = None,
        subdomains: Optional[List[str]] = None,
        secondary_domains: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        objectives_path: Optional[str] = None,
        strict_mode: bool = False,
        typed_edges_llm: bool = False,
        concept_graph_path: Optional[str] = None,
        imscc_chunks_path: Optional[str] = None,
    ):
        # When strict_mode is True the pipeline refuses to write a final
        # artifact whose quality_report shows any broken_refs, any cross-lesson
        # follows_chunk link, or html_balance_violations above 5%. See §1.5 of
        # VERSIONING.md.
        self.strict_mode = strict_mode
        # When typed_edges_llm is True, the typed-edge concept-graph builder
        # calls an LLM escalation callable for edges no rule covered. Off by
        # default — the default deterministic path is byte-identical across
        # runs (ADR-001 Contract 3).
        self.typed_edges_llm = typed_edges_llm
        # optional path to a pre-built pedagogy graph emitted
        # upstream by the ``concept_extraction`` workflow phase
        # (``MCP/tools/pipeline_tools.py::_run_concept_extraction``). When
        # provided AND readable, ``_generate_pedagogy_graph`` short-circuits
        # the in-process ``build_pedagogy_graph`` call and loads the graph
        # from this path instead. Eliminates redundant graph rebuilds when
        # the upstream phase already materialised an authoritative graph at
        # ``LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json``.
        # When None (legacy / pre-Phase-6 corpora) the existing build path
        # runs unchanged. The fallback also fires on a missing/unreadable
        # path so a stale phase-output handoff degrades gracefully rather
        # than crashing the run.
        self.concept_graph_path: Optional[Path] = (
            Path(concept_graph_path) if concept_graph_path else None
        )
        # optional path to a pre-built IMSCC chunkset emitted
        # upstream by the ``imscc_chunking`` workflow phase
        # (``MCP/tools/pipeline_tools.py::_run_imscc_chunking``). When
        # provided AND readable, ``process()`` short-circuits the in-process
        # ``self._chunk_content(parsed_items)`` call and loads the
        # canonical chunks from this JSONL instead. The
        # ``imscc_chunking`` phase already runs the canonical chunker
        # against the packaged IMSCC zip and writes
        # ``LibV2/courses/<slug>/imscc_chunks/chunks.jsonl`` BEFORE
        # ``trainforge_assessment`` dispatches the CourseProcessor — so
        # re-running the chunker here is purely redundant work. When
        # None (legacy / pre-Phase-8 callers, e.g. ``python -m
        # Trainforge.process_course`` standalone) the existing
        # in-process build path runs unchanged. The fallback also fires
        # on a missing/unreadable path so a stale phase-output handoff
        # degrades gracefully rather than crashing the run.
        self.imscc_chunks_path: Optional[Path] = (
            Path(imscc_chunks_path) if imscc_chunks_path else None
        )
        self.imscc_path = Path(imscc_path)
        self.output_dir = Path(output_dir)
        self.course_code = course_code

        # ------------------------------------------------------------------
        # Classification resolution priority:
        # Priority:
        #   1. Explicit kwargs (non-None) from the caller/CLI — override.
        #   2. course_metadata.json stub at IMSCC root or alongside the file.
        #   3. Backward-compat defaults (division="STEM", domain="").
        # The loader runs before the fields are set so we can log the source.
        # ------------------------------------------------------------------
        stub = self._load_classification_stub() or {}
        stub_cls = stub.get("classification") if isinstance(stub, dict) else None
        stub_cls = stub_cls if isinstance(stub_cls, dict) else {}

        cli_has_division = division is not None
        cli_has_domain = domain is not None
        cli_has_subdomains = subdomains is not None
        cli_has_topics = topics is not None

        self.division = (
            division if cli_has_division
            else stub_cls.get("division") or "STEM"
        )
        self.domain = (
            domain if cli_has_domain
            else stub_cls.get("primary_domain") or ""
        )
        self.subdomains = (
            list(subdomains) if cli_has_subdomains
            else list(stub_cls.get("subdomains") or [])
        )
        self.topics = (
            list(topics) if cli_has_topics
            else list(stub_cls.get("topics") or [])
        )
        self.secondary_domains = list(secondary_domains or [])

        # Provenance log (observability — surface which path provided
        # classification so misconfiguration is trivially diagnosable).
        if stub_cls and not (cli_has_division or cli_has_domain or cli_has_subdomains or cli_has_topics):
            logger.info(
                "Using classification from course_metadata.json stub "
                "(division=%s, primary_domain=%s)",
                self.division, self.domain,
            )
        elif stub_cls and (cli_has_division or cli_has_domain or cli_has_subdomains or cli_has_topics):
            logger.info(
                "Using classification from CLI flags (override stub); "
                "resolved division=%s, primary_domain=%s",
                self.division, self.domain,
            )
        elif cli_has_division or cli_has_domain:
            logger.info(
                "Using classification from CLI flags "
                "(division=%s, primary_domain=%s)",
                self.division, self.domain,
            )
        else:
            logger.info("No classification provided; using defaults (division=STEM)")

        # Sub-directories
        # corpus/ renamed to imscc_chunks/. ``corpus_dir`` is
        # preserved as an alias on the class so existing references keep
        # working without churn; new writes target ``imscc_chunks_dir``.
        self.imscc_chunks_dir = self.output_dir / "imscc_chunks"
        self.corpus_dir = self.imscc_chunks_dir  # legacy alias
        self.graph_dir = self.output_dir / "graph"
        self.training_specs_dir = self.output_dir / "training_specs"
        self.pedagogy_dir = self.output_dir / "pedagogy"
        self.quality_dir = self.output_dir / "quality"

        # Objectives (optional)
        self.objectives: Optional[Dict[str, Any]] = None
        self.domain_concept_seeds: List[Tuple[str, List[re.Pattern]]] = []
        self._objectives_source: Optional[str] = None
        resolved_objectives_path: Optional[Path] = None
        if objectives_path:
            resolved_objectives_path = Path(objectives_path)
            self._objectives_source = "kwarg"
        else:
            # When no objectives_path is supplied, probe the
            # canonical auto-synthesized location the planner writes at
            # ``{project_path}/01_learning_objectives/synthesized_objectives.json``.
            # ``CourseProcessor`` is invoked with ``output_dir`` pointing
            # at the Trainforge nested workspace (usually
            # ``{project_path}/trainforge/``) — the synthesized objectives
            # live one level up so ``output_dir.parent`` is the first
            # candidate. For callers who pass the project root
            # directly we also probe ``output_dir`` itself.
            for _candidate_root in (self.output_dir.parent, self.output_dir):
                _candidate = (
                    _candidate_root
                    / "01_learning_objectives"
                    / "synthesized_objectives.json"
                )
                if _candidate.exists():
                    resolved_objectives_path = _candidate
                    self._objectives_source = "auto_synthesized"
                    logger.info(
                        "Auto-detected synthesized objectives at %s",
                        _candidate,
                    )
                    break

        if resolved_objectives_path is not None:
            try:
                self.objectives = load_objectives(resolved_objectives_path)
                self.domain_concept_seeds = compile_domain_concept_seeds(
                    self.objectives.get("domain_concepts", [])
                )
            except Exception as _obj_exc:  # noqa: BLE001 — defensive
                logger.warning(
                    "Failed to load objectives from %s: %s; "
                    "course.json will land as an empty-learning_outcomes shell",
                    resolved_objectives_path,
                    _obj_exc,
                )
                self.objectives = None
                self._objectives_source = "load_failed"

        # precompute the component->terminal parent map once so
        # the per-chunk retag pass in _create_chunk doesn't re-walk the
        # objectives payload. Empty when no objectives are loaded —
        # retag_chunk_outcomes degrades to a no-op for parent rollup
        # in that case but still applies the vocabulary retag rule.
        from Trainforge.retag_outcomes import build_parent_map as _bpm
        self._lo_parent_map: Dict[str, str] = _bpm(self.objectives)

        # Decision capture
        # Phase value must be in the canonical enum at
        # ``schemas/events/decision_event.schema.json`` (hyphenated). Prior
        # emit used the underscore form ``"content_extraction"`` which failed
        # closed under ``DECISION_VALIDATION_STRICT=true``. The canonical
        # enum value for Trainforge's first stage is
        # ``"trainforge-content-analysis"``.
        self.capture = DecisionCapture(
            course_code=course_code,
            phase="trainforge-content-analysis",
            tool="trainforge",
            streaming=True,
        )

        # HTML parser
        self.html_parser = HTMLContentParser()

        # Stats
        self.stats: Dict[str, Any] = {
            "total_chunks": 0,
            "total_words": 0,
            "total_tokens_estimate": 0,
            "chunk_types": defaultdict(int),
            "difficulty_distribution": defaultdict(int),
            "sections_processed": 0,
            "modules_processed": 0,
            "quizzes_processed": 0,
        }
        self._all_concept_tags: set = set()

        # Honest IRT difficulty-calibration scaffold
        # (TRAINFORGE_IRT_DIFFICULTY_SCAFFOLD). Lazily built once per run from
        # the optional learner-response seam; None until first resolved.
        self._irt_calibrated_map: Optional[Dict[str, Any]] = None
        self._irt_response_file_present: bool = False

        # Populated during processing; consumed by quality-report generation.
        self._boilerplate_spans: List[str] = []
        self._valid_outcome_ids: Set[str] = set()
        self._factual_flags: List[Dict[str, Any]] = []
        self._boilerplate_config = BoilerplateConfig()
        # Lesson IDs for pages whose JSON-LD declared at least one misconception.
        # Populated by _chunk_content; used as the denominator for
        # misconceptions_present_rate in _generate_quality_report.
        self._pages_with_misconceptions: Set[str] = set()

    # ------------------------------------------------------------------
    # Classification stub loader
    # ------------------------------------------------------------------

    def _load_classification_stub(self) -> Optional[Dict[str, Any]]:
        """Locate and parse ``course_metadata.json``, if present.

        Searches (in order):
          1. Inside the IMSCC zip at root — forward-compat for when the
             packager starts bundling the stub.
          2. Alongside the IMSCC file (``imscc_path.parent /
             course_metadata.json``) — today's Courseforge layout, where
             ``generate_course.py`` writes the stub to the content dir
             and the IMSCC is packaged to the same directory.

        Returns the parsed dict on success or ``None`` when no stub is
        found or parsing fails. A parse failure is logged but non-fatal
        so the pipeline falls back to CLI / defaults.
        """
        # 1. In-zip lookup.
        try:
            if self.imscc_path.exists():
                with zipfile.ZipFile(self.imscc_path, "r") as z:
                    if "course_metadata.json" in z.namelist():
                        try:
                            data = json.loads(
                                z.read("course_metadata.json").decode("utf-8")
                            )
                            if isinstance(data, dict):
                                return data
                        except Exception as e:
                            logger.warning(
                                "Failed to parse course_metadata.json "
                                "inside IMSCC zip (%s): %s",
                                self.imscc_path, e,
                            )
        except Exception as e:
            logger.debug("IMSCC stub lookup (zip) skipped: %s", e)

        # 2. Sibling lookup (current Courseforge layout).
        sibling = self.imscc_path.parent / "course_metadata.json"
        if sibling.exists():
            try:
                data = json.loads(sibling.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning(
                    "Failed to parse sibling course_metadata.json at %s: %s",
                    sibling, e,
                )
        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self) -> Dict[str, Any]:
        """Run the full 6-stage pipeline. Returns summary dict."""
        print(f"[Trainforge] Processing {self.imscc_path.name} → {self.output_dir}")

        self._create_directories()

        # Stage 1
        print("[1/6] Extracting IMSCC package...")
        title, html_files = self._extract_imscc()

        # Stage 2
        print("[2/6] Parsing HTML content...")
        parsed_items = self._parse_html(html_files)
        # Retain parsed_items so the semantic graph stage can reconstruct
        # objectives_metadata (list of LO dicts shaped like JSON-LD
        # learningObjectives[]) for the targets_concept_from_lo rule. Passing
        # objectives_metadata=None here instead leaves that rule firing on
        # empty input.
        self._parsed_items = parsed_items

        # Pre-chunking: detect corpus-wide boilerplate (footers / template chrome)
        # and build the set of valid outcome IDs for referential-integrity checks.
        self._boilerplate_spans = self._detect_corpus_boilerplate(parsed_items)
        self._valid_outcome_ids = self._build_valid_outcome_ids()

        # Stage 3
        print("[3/6] Chunking content into pedagogical units...")
        # short-circuit on upstream IMSCC chunkset.
        # When the ``imscc_chunking`` workflow phase has already
        # materialised a canonical chunkset at the supplied path,
        # consume it instead of re-running the chunker in-process.
        # Falls through to the in-process build on missing /
        # unreadable / corrupted upstream chunks (preserves
        # backward compat for legacy callers that don't thread the
        # path).
        chunks: Optional[List[Dict[str, Any]]] = None
        upstream_chunks_path = getattr(self, "imscc_chunks_path", None)
        if upstream_chunks_path is not None:
            chunks = self._load_chunks_from_jsonl(upstream_chunks_path)
        if chunks is None:
            chunks = self._chunk_content(parsed_items)

        # Stage 4
        print("[4/6] Writing chunks...")
        self._write_chunks(chunks)

        # Stage 5
        print("[5/6] Generating metadata...")
        concept_graph = self._generate_concept_graph(chunks)
        # Build the pedagogy graph after the concept graph so the
        # concept_classes map can flow
        # into the prerequisite_of / interferes_with /
        # concept_supports_outcome filters.
        # ``_generate_pedagogy_graph`` below.
        pedagogy_graph = self._generate_pedagogy_graph(
            chunks, concept_graph=concept_graph,
        )
        manifest = self._generate_manifest(title, concept_graph=concept_graph)
        corpus_stats = self._generate_corpus_stats()
        quality_report = self._generate_quality_report(chunks)
        # Typed-edge concept graph (additive to concept_graph). Rule-based
        # by default; LLM escalation opt-in via self.typed_edges_llm.
        # Always build course_data; the empty-LOs shell is
        # safe — semantic_graph_builder treats empty learning_outcomes
        # as "no typed-edge seeds" rather than crashing).
        course_data_for_semantic = self._build_course_json(manifest)
        semantic_graph = self._generate_semantic_concept_graph(
            chunks, course_data_for_semantic, concept_graph,
            parsed_items=parsed_items,
        )

        # Stage 6
        print("[6/6] Writing metadata files...")
        self._write_metadata(manifest, corpus_stats, concept_graph, quality_report,
                             pedagogy_graph=pedagogy_graph,
                             semantic_graph=semantic_graph,
                             chunks=chunks)

        summary = {
            "status": "success",
            "output_dir": str(self.output_dir),
            "course_code": self.course_code,
            "title": title,
            "stats": {k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in self.stats.items()},
        }

        print(f"\n[SUCCESS] Generated {self.stats['total_chunks']} chunks")
        print(f"  Total words: {self.stats['total_words']:,}")
        print(f"  Total tokens (est): {self.stats['total_tokens_estimate']:,}")
        print(f"  Output: {self.output_dir}")

        return summary

    # ------------------------------------------------------------------
    # Stage 1: Extract IMSCC
    # ------------------------------------------------------------------

    def _extract_imscc(self) -> Tuple[str, List[Dict[str, Any]]]:
        if not self.imscc_path.exists():
            raise FileNotFoundError(f"IMSCC not found: {self.imscc_path}")

        self.capture.log_decision(
            decision_type="imscc_extraction",
            decision=f"Extract {self.imscc_path.name}",
            rationale="Parse IMSCC manifest and HTML resources to build RAG corpus for LibV2 import",
        )

        html_files: List[Dict[str, Any]] = []
        title = self.course_code

        with zipfile.ZipFile(self.imscc_path, "r") as z:
            # Try to get title from manifest
            try:
                manifest_xml = z.read("imsmanifest.xml").decode("utf-8")
                root = ET.fromstring(manifest_xml)
                # Search for title across common namespaces
                for ns_uri in [
                    "http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest",
                    "http://ltsc.ieee.org/xsd/imsccv1p1/LOM/manifest",
                    "http://www.imsglobal.org/xsd/imsmd_v1p2",
                ]:
                    elem = root.find(f".//{{{ns_uri}}}title/{{{ns_uri}}}string")
                    if elem is not None and elem.text:
                        title = elem.text.strip()
                        break
                # Fallback: try unnamespaced
                if title == self.course_code:
                    for elem in root.iter():
                        if elem.tag.endswith("}string") or elem.tag == "string":
                            if elem.text and len(elem.text.strip()) > 5:
                                title = elem.text.strip()
                                break
            except Exception:
                pass

            # If we have an objectives file with a title, prefer that
            if self.objectives and self.objectives.get("course_title"):
                title = self.objectives["course_title"]

            # Extract HTML files
            for name in z.namelist():
                if name.endswith(".html") or name.endswith(".htm"):
                    try:
                        content = z.read(name).decode("utf-8", errors="ignore")
                        html_files.append({"path": name, "content": content, "id": Path(name).stem})
                    except Exception as e:
                        print(f"  Warning: Failed to read {name}: {e}")

        print(f"  Course title: {title}")
        print(f"  HTML files: {len(html_files)}")

        return title, html_files

    # ------------------------------------------------------------------
    # Stage 2: Parse HTML
    # ------------------------------------------------------------------

    def _parse_html(self, html_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parsed_items = []

        for item in html_files:
            if not item.get("content"):
                continue

            content = item["content"]
            resource_type, module_id, module_title = classify_resource(item["path"])

            # Strip assessment feedback from quiz HTML BEFORE parsing
            # so sections don't contain answer feedback text
            if resource_type == "quiz":
                content = self._strip_assessment_feedback(content)

            parsed = self.html_parser.parse(content)
            week_num = extract_week_number(item["path"])

            # diagnostic (§4.4a H5 detection): if a JSON-LD
            # <script type="application/ld+json"> tag is present in the raw
            # HTML but the parser returned no courseforge metadata, the
            # block either failed to parse (H5 signature) or parsed to a
            # non-dict payload. Distinguished here from H2 (tag genuinely
            # absent).
            jsonld_tag_present = bool(
                re.search(r'<script\s+type=["\']application/ld\+json["\']', content, re.IGNORECASE)
            )
            jsonld_parse_failed = (
                jsonld_tag_present and parsed.metadata.get("courseforge") is None
            )

            parsed_items.append({
                "item_id": item["id"],
                "item_path": item["path"],
                "title": parsed.title,
                "resource_type": resource_type,
                "module_id": module_id,
                "module_title": module_title,
                "week_num": week_num,
                "word_count": parsed.word_count,
                "sections": parsed.sections,
                "learning_objectives": parsed.learning_objectives,
                "key_concepts": parsed.key_concepts,
                "interactive_components": parsed.interactive_components,
                "raw_html": content,
                # New: metadata from JSON-LD / data-cf-* attributes
                "page_id": parsed.page_id,
                "misconceptions": parsed.misconceptions,
                "suggested_assessment_types": parsed.suggested_assessment_types,
                "courseforge_metadata": parsed.metadata.get("courseforge"),
                # page-level union of every
                # distinct data-cf-objective-ref on the page. Used as
                # fallback attachment when a chunk can't be mapped to a
                # specific section in _extract_objective_refs.
                "objective_refs": parsed.objective_refs,
                # page-level aggregated source_references (full
                # SourceReference dicts). Threaded into _create_chunk so
                # chunks carry source.source_references[] end-to-end.
                # Downstream treats an absent value as
                # "unknown", not an error.
                "source_references": parsed.source_references,
                # diagnostic flags (§4.4a H5 detection)
                "_jsonld_tag_present": jsonld_tag_present,
                "_jsonld_parse_failed": jsonld_parse_failed,
            })

        self.stats["modules_processed"] = len([p for p in parsed_items if p["resource_type"] == "page"])
        self.stats["quizzes_processed"] = len([p for p in parsed_items if p["resource_type"] == "quiz"])

        # Count unique weeks/sections
        weeks = {p["week_num"] for p in parsed_items if p["week_num"] > 0}
        self.stats["sections_processed"] = len(weeks)

        print(f"  Parsed {len(parsed_items)} items (modules={self.stats['modules_processed']}, quizzes={self.stats['quizzes_processed']}, weeks={len(weeks)})")
        return parsed_items

    # ------------------------------------------------------------------
    # Stage 3: Chunk content
    # ------------------------------------------------------------------

    def _load_chunks_from_jsonl(
        self, chunks_path: Path
    ) -> Optional[List[Dict[str, Any]]]:
        """Stream an upstream JSONL chunkset into a list of chunk dicts.

        Companion to the ``process()`` short-circuit on
        ``self.imscc_chunks_path``. The ``imscc_chunking`` workflow
        phase (``MCP/tools/pipeline_tools.py::_run_imscc_chunking``,
        writes the canonical chunks file at
        ``LibV2/courses/<slug>/imscc_chunks/chunks.jsonl`` BEFORE
        ``trainforge_assessment`` dispatches the CourseProcessor. This
        helper reads it, restores the side channels the in-process
        ``_chunk_content`` wrapper sets (``self.stats["total_chunks"]``
        and the operator-visible "Generated N chunks" log), and
        returns the chunks list shaped exactly like the in-process
        path.

        Returns ``None`` (signal to fall through to the in-process
        ``_chunk_content`` build) on:
        - path missing / not a file
        - any read error or JSONL parse failure
        - empty chunks list (defensive; an empty upstream emit is
          almost always a bug worth re-running the in-process chunker
          for, rather than silently emitting an empty corpus)

        Returns the chunks list on success. Invalid upstream artifacts return
        ``None`` so the caller can build chunks in process.
        """
        chunks_path = Path(chunks_path)
        if not chunks_path.exists() or not chunks_path.is_file():
            logger.warning(
                "imscc_chunks_path %s does not exist or "
                "is not a file; falling through to in-process "
                "_chunk_content build.",
                chunks_path,
            )
            return None
        try:
            loaded: List[Dict[str, Any]] = []
            with chunks_path.open("r", encoding="utf-8") as fh:
                for line_num, raw_line in enumerate(fh, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as parse_exc:
                        logger.warning(
                            "Malformed JSONL at %s "
                            "line %d (%s); falling through to in-"
                            "process _chunk_content build.",
                            chunks_path, line_num, parse_exc,
                        )
                        return None
                    if not isinstance(chunk, dict):
                        logger.warning(
                            "non-dict chunk at %s "
                            "line %d (got %s); falling through to "
                            "in-process _chunk_content build.",
                            chunks_path, line_num,
                            type(chunk).__name__,
                        )
                        return None
                    loaded.append(chunk)
        except OSError as read_exc:
            logger.warning(
                "Failed to read IMSCC chunks at %s "
                "(%s); falling through to in-process _chunk_content "
                "build.",
                chunks_path, read_exc,
            )
            return None
        if not loaded:
            logger.warning(
                "IMSCC chunks JSONL at %s parsed but "
                "is empty; falling through to in-process "
                "_chunk_content build (defensive — an empty upstream "
                "chunkset is almost always a bug).",
                chunks_path,
            )
            return None
        # Restore the side channels the in-process ``_chunk_content``
        # wrapper sets so downstream code (``_generate_quality_report``,
        # the CLI summary print) sees a consistent view regardless of
        # which path produced ``chunks``.
        self.stats["total_chunks"] = len(loaded)
        # ``_pages_with_misconceptions`` is the denominator for
        # ``misconceptions_present_rate`` in ``quality_report.json``.
        # The upstream chunker callback at
        # ``MCP/tools/pipeline_tools.py::_run_imscc_chunking`` doesn't
        # surface the per-page misconception accumulator (it's a
        # concept private to ``CourseProcessor._chunk_content``), so we
        # default to an empty set here. ``_generate_quality_report``
        # tolerates an empty set (the rate falls to 0 / total pages
        # rather than raising).
        if not hasattr(self, "_pages_with_misconceptions"):
            self._pages_with_misconceptions = set()
        logger.info(
            "Consuming upstream IMSCC chunkset from %s "
            "(chunks=%d); skipping in-process _chunk_content rebuild.",
            chunks_path, len(loaded),
        )
        print(f"  Loaded {len(loaded)} chunks from upstream IMSCC chunkset")
        return loaded

    def _chunk_content(self, parsed_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk parsed IMSCC items into a list of chunk dicts.

        Owns three Trainforge-specific concerns that ``Trainforge.chunker.chunk_content``
        deliberately doesn't:

        - Binds ``self._create_chunk`` as the chunker's per-chunk
          materialisation callback. ``_create_chunk`` reads
          ``self.capture``, ``self._lo_parent_map``,
          ``self.OBJECTIVE_CODE_RE``, etc., so it stays on this class.
        - Surfaces the chunker's side-channel return: stamps
          ``self._pages_with_misconceptions`` (denominator for
          ``misconceptions_present_rate`` in ``quality_report.json``)
          and ``self.stats["total_chunks"]``.
        - Preserves the legacy ``"  Generated N chunks"`` console line
          for operator-visible parity.
        """

        result = _pkg_chunk_content(
            parsed_items,
            self.course_code,
            self._boilerplate_spans,
            min_chunk_size=self.MIN_CHUNK_SIZE,
            max_chunk_size=self.MAX_CHUNK_SIZE,
            target_chunk_size=self.TARGET_CHUNK_SIZE,
            ctx=_ChunkerContext(create_chunk=self._create_chunk),
        )
        self._pages_with_misconceptions = result.pages_with_misconceptions
        self.stats["total_chunks"] = len(result.chunks)
        print(f"  Generated {len(result.chunks)} chunks")
        return result.chunks

    # role-precedence ranking for merging source_references across
    # multiple sections that collapse into one chunk. Lower integer = stronger
    # (primary overrides contributing, contributing overrides corroborating).
    _SOURCE_ROLE_PRECEDENCE = {"primary": 0, "contributing": 1, "corroborating": 2}

    def _merge_small_sections(
        self, sections
    ) -> List[Tuple[str, str, str, List[str]]]:
        """Merge adjacent sections below ``MIN_CHUNK_SIZE`` into combined blocks.

        Thin wrapper over ``Trainforge.chunker.merge_small_sections`` that
        threads ``self.MAX_CHUNK_SIZE`` (the chunker function is duck-typed
        on the section objects so no other state binding is needed). Kept
        for direct test callers
        (``Trainforge/tests/test_merge_small_sections_zero_word.py`` and
        ``test_html_content_parser_template_type.py`` invoke
        ``proc._merge_small_sections(sections)``).

        Returns list of (heading, combined_text, chunk_type,
        merged_source_ids, merged_headings) tuples.
        """

        return _pkg_merge_small_sections(
            sections,
            max_chunk_size=self.MAX_CHUNK_SIZE,
        )

    def _chunk_text_block(
        self, text: str, html: str, item: Dict[str, Any],
        heading: str, chunk_type: str, prefix: str, start_id: int,
        follows_chunk_id: Optional[str] = None,
        position_in_module: int = 0,
        section_source_ids: Optional[List[str]] = None,
        merged_headings: Optional[List[str]] = None,
        merged_key_claims: Optional[List[Dict[str, Any]]] = None,
        merged_objective_alignment: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Split a text block into one or more chunks (xpath/char-span provenance preserved).

        Thin wrapper over ``Trainforge.chunker.chunk_text_block`` that
        binds ``self._create_chunk`` (so the materialised chunks carry
        the full Trainforge metadata surface — concept_tags,
        objective_refs, bloom_level — that the callback resolves from
        ``self``-state) and threads ``self.MAX_CHUNK_SIZE`` /
        ``self.TARGET_CHUNK_SIZE``. Kept for the direct test caller at
        ``Trainforge/tests/test_provenance.py``.

        Passes ``merged_key_claims`` /
        ``merged_objective_alignment`` through to the chunker so chunks
        emitted from a merge boundary carry the union'd audit fields
        ``_create_chunk`` then stamps onto the chunk dict.

        See docs/reference/audit-trail.md for the round-trip contract
        on ``source.html_xpath`` / ``source.char_span``.
        """

        return _pkg_chunk_text_block(
            text=text,
            html=html,
            item=item,
            heading=heading,
            chunk_type=chunk_type,
            prefix=prefix,
            start_id=start_id,
            ctx=_ChunkerContext(create_chunk=self._create_chunk),
            follows_chunk_id=follows_chunk_id,
            position_in_module=position_in_module,
            section_source_ids=section_source_ids,
            merged_headings=merged_headings,
            merged_key_claims=merged_key_claims,
            merged_objective_alignment=merged_objective_alignment,
            max_chunk_size=self.MAX_CHUNK_SIZE,
            target_chunk_size=self.TARGET_CHUNK_SIZE,
        )

    def _create_chunk(
        self, chunk_id: str, text: str, html: str, item: Dict[str, Any],
        section_heading: str, chunk_type: str,
        follows_chunk_id: Optional[str] = None,
        position_in_module: int = 0,
        html_xpath: Optional[str] = None,
        char_span: Optional[List[int]] = None,
        section_source_ids: Optional[List[str]] = None,
        merged_headings: Optional[List[str]] = None,
        merged_key_claims: Optional[List[Dict[str, Any]]] = None,
        merged_objective_alignment: Optional[List[Dict[str, Any]]] = None,
        curie_anchors: Optional[List[str]] = None,
        forced_curie_anchors: Optional[List[str]] = None,
        dart_source_refs: Optional[List[Dict[str, Any]]] = None,
        composite_unit: Optional[str] = None,
        unit_roles: Optional[List[str]] = None,
        unit_subclass: Optional[str] = None,
    ) -> Dict[str, Any]:
        words = text.split()
        word_count = len(words)
        tokens_estimate = int(word_count * 1.3)

        # Canonicalise WCAG SC references in prose before concept-tag
        # extraction so text-based detection sees the single canonical form.
        text = canonicalize_sc_references(text)

        concept_tags = self._extract_concept_tags(text, item)
        difficulty = self._determine_difficulty(text, item)

        # normalize module_id to canonical week_NN_<slot> form.
        # The IMSCC inventory occasionally yields short slugs (``application``,
        # ``content_01``, ``summary``) when the file lives at
        # ``week_04/application.html`` instead of ``week_04/week_04_application.html``.
        # Both shapes describe the same module slot but downstream graphs key
        # off module_id, so we lift them to a single canonical form here.
        normalized_mid, mid_changed = normalize_module_id(
            item["module_id"],
            item_path=item.get("item_path"),
            week_num=item.get("week_num"),
        )
        if mid_changed:
            logger.debug(
                "module_id_normalized: %s -> %s (item_path=%s)",
                item["module_id"], normalized_mid, item.get("item_path"),
            )
        elif normalized_mid and not _WEEK_PREFIXED_RE.match(normalized_mid):
            logger.warning(
                "module_id_unprefixed: %s (no week info recoverable from "
                "item_path=%s, week_num=%s); keeping as-is",
                normalized_mid, item.get("item_path"), item.get("week_num"),
            )
        # Defensive heading-sanity filter (TRAINFORGE_HEADING_SANITY_FILTER,
        # default OFF → byte-identical legacy). Mirrors the inline
        # _create_chunk in MCP/tools/pipeline_tools.py: a section_heading
        # detected as upstream heading-classifier noise (answer-key /
        # exercise-prose / numeric) is repaired to the nearest clean ancestor
        # heading (merged_headings breadcrumb, then page title); the original
        # noise text stays in text/html. No clean ancestor → leave as-is +
        # stamp ``heading_suspect``.
        _heading_to_stamp = section_heading
        _heading_suspect_flag = False
        try:
            from lib.chunk_heading_sanity import (
                is_heading_sanity_filter_enabled as _hs_enabled,
            )
            from lib.chunk_heading_sanity import (
                repair_section_heading as _hs_repair,
            )
            if _hs_enabled():
                _hs = _hs_repair(
                    section_heading,
                    item=item,
                    merged_headings=merged_headings,
                )
                _heading_to_stamp = _hs["heading"]
                _heading_suspect_flag = _hs["suspect"] and not _hs["repaired"]
        except Exception:  # noqa: BLE001 — sanity filter is best-effort
            logger.warning(
                "heading-sanity filter failed for chunk %s; keeping original "
                "section_heading",
                chunk_id,
                exc_info=True,
            )
            _heading_to_stamp = section_heading
            _heading_suspect_flag = False
        source: Dict[str, Any] = {
            "course_id": self.course_code,
            "module_id": normalized_mid,
            "module_title": item["module_title"],
            "lesson_id": item["item_id"],
            "lesson_title": item["title"],
            "resource_type": item["resource_type"],
            "section_heading": _heading_to_stamp,
            "position_in_module": position_in_module,
        }
        if _heading_suspect_flag:
            source["heading_suspect"] = True
        # Audit-trail provenance (Section 508 / ADA Title II). Every chunk
        # ties back to the source IMSCC HTML element it was derived from.
        # See docs/reference/audit-trail.md for the round-trip contract.
        if html_xpath:
            source["html_xpath"] = html_xpath
        if char_span is not None:
            source["char_span"] = list(char_span)
        # Carry the IMSCC-relative path so auditors can open the source
        # file without walking imsmanifest.xml.
        if item.get("item_path"):
            source["item_path"] = item["item_path"]

        # GPT Feedback v2 (May 12 / item 3): thread an aggregate
        # source-document SHA into every chunk's source block when the
        # caller has stamped it onto the CourseProcessor instance via
        # ``self._source_document_sha256``. Same value across every
        # chunk in this run; gives downstream consumers a byte-stable
        # join key from chunk → upstream-source artifact. Off by
        # default (legacy parity): when the attribute is absent /
        # falsy, the field is simply omitted. See
        # schemas/knowledge/chunk_v4.schema.json::
        # $defs.Source.source_document_sha256.
        _src_doc_sha = getattr(self, "_source_document_sha256", None)
        if isinstance(_src_doc_sha, str) and _src_doc_sha:
            source["source_document_sha256"] = _src_doc_sha

        # fold DART source provenance into source.source_references[].
        # Precedence chain (same first-seen-wins policy as the parser):
        #   1. page-level JSON-LD sourceReferences (full shape)
        #   2. section-level JSON-LD sourceReferences (full shape)
        #   3. section-level data-cf-source-ids (stringified sourceId →
        #      synthesised {sourceId, role: 'contributing'})
        #   4. page-level data-cf-source-ids (same synthesis)
        # For merged sections (_merge_small_sections), ``section_source_ids``
        # already carries the unioned sourceId strings across every merged
        # section. Role-precedence between JSON-LD entries is preserved by
        # first-seen-wins: page-level JSON-LD overrides section-level, and
        # section-level JSON-LD overrides data-cf-* synthesis. When the
        # authoritative shape is missing (legacy corpora), the field is
        # omitted entirely — consumers treat absence as 'unknown'.
        resolved_refs = self._resolve_chunk_source_references(
            item=item,
            section_heading=section_heading,
            section_source_ids=section_source_ids or [],
            merged_headings=merged_headings,
        )
        if resolved_refs:
            source["source_references"] = resolved_refs

        chunk: Dict[str, Any] = {
            "id": chunk_id,
            "schema_version": CHUNK_SCHEMA_VERSION,
            "chunk_type": chunk_type,
            "text": text,
            "html": html,
            "follows_chunk": follows_chunk_id,
            "source": source,
            "concept_tags": concept_tags,
            # pass section_heading through
            # so the merge path can harvest section-scoped
            # data-cf-objective-ref values from activities/self-checks in
            # addition to the page-level learning_objectives list.
            "learning_outcome_refs": self._extract_objective_refs(
                item, section_heading=section_heading
            ),
            "difficulty": difficulty,
            "tokens_estimate": tokens_estimate,
            "word_count": word_count,
        }

        # Enrich from Courseforge metadata (JSON-LD / data-cf-*).
        # Resolution order: section JSON-LD → page JSON-LD → parsed LOs →
        # text verb heuristic → hardcoded default. Every chunk ends up with
        # a bloom_level; bloom_level_source records where it came from so
        # downstream consumers can weight low-confidence sources.
        (
            bloom_level,
            content_type_label,
            key_terms,
            section_metadata_extras,
            section_trace,
        ) = self._extract_section_metadata(
            item, section_heading, merged_headings=merged_headings,
        )
        bloom_source = "section_jsonld" if bloom_level else None

        # Merge structured JSON-LD keyTerms into concept_tags. These are the
        # highest-fidelity domain vocabulary Courseforge emits; leaving them
        # in chunk["key_terms"] only meant the concept graph missed them.
        # also filter through the concept classifier so JSON-LD
        # keyTerms can't reintroduce pedagogical scaffolding / fragments
        # that ``_extract_concept_tags`` already rejected.
        # also strip trailing LO-ref suffix (-co-NN / -to-NN)
        # so downstream consumers see clean concept slugs and don't
        # bleed `co 15` / `to 03` artifact tokens into paraphrase prompts.
        from lib.ontology.concept_classifier import (
            canonicalize_alias as _canon_alias,
        )
        from lib.ontology.concept_classifier import (
            classify_concept as _classify,
        )
        from lib.ontology.concept_classifier import (
            is_droppable_class as _is_droppable,
        )
        from lib.ontology.slugs import strip_lo_ref_suffix as _strip_lo_ref
        for kt in key_terms or []:
            term = kt.get("term") if isinstance(kt, dict) else kt
            tag = normalize_tag(term or "")
            if not tag or len(tag) < 3:
                continue
            tag = _strip_lo_ref(tag)
            if not tag:
                continue
            tag = _canon_alias(tag)
            if tag in concept_tags:
                continue
            if (self.OBJECTIVE_CODE_RE.match(tag)
                    or self.WEEK_PREFIX_RE.match(tag)
                    or tag in self.NON_CONCEPT_TAGS):
                continue
            if _is_droppable(_classify(tag)):
                continue
            concept_tags.append(tag)

        if not bloom_level:
            cf_meta = item.get("courseforge_metadata")
            if cf_meta and cf_meta.get("learningObjectives"):
                for lo in cf_meta["learningObjectives"]:
                    if lo.get("bloomLevel"):
                        bloom_level = lo["bloomLevel"]
                        bloom_source = "page_jsonld"
                        break
        if not bloom_level:
            for lo in item.get("learning_objectives", []):
                bl = lo.bloom_level if hasattr(lo, "bloom_level") else lo.get("bloom_level")
                if bl:
                    bloom_level = bl
                    bloom_source = "lo_inherited"
                    break
        if not bloom_level:
            derived = derive_bloom_from_verbs(text)
            if derived:
                bloom_level = derived
                bloom_source = "verbs"
        if not bloom_level:
            bloom_level = "understand"
            bloom_source = "default"

        # canonicalize compound bloom levels (e.g. "remember-apply").
        # The chunk_v4 schema only admits the six canonical Bloom levels; any
        # JSON-LD / data-cf-* / LO source that supplied a hyphenated value
        # would otherwise fail strict validation. Split into primary (HIGHER)
        # + secondary (LOWER) so no information is lost.
        primary_bloom, secondary_bloom = canonicalize_bloom_level(bloom_level)
        if primary_bloom is not None:
            bloom_level = primary_bloom
        chunk["bloom_level"] = bloom_level
        if secondary_bloom is not None:
            chunk["bloom_level_secondary"] = secondary_bloom
        # Only tag the source when it's below lo_inherited confidence;
        # authoritative chunks stay schema-identical to pre-fallback output.
        if bloom_source in ("verbs", "default"):
            chunk["bloom_level_source"] = bloom_source
            # Capture may be absent in unit tests that bypass ``__init__``
            # (e.g. test_provenance.py). Only log when it's present so this
            # non-test-facing observability doesn't turn into a hard failure.
            capture = getattr(self, "capture", None)
            if capture is not None:
                capture.log_decision(
                    decision_type="bloom_level_assignment",
                    decision=f"Assigned bloom_level={bloom_level} via {bloom_source}",
                    rationale=(
                        "No JSON-LD, data-cf-*, or parsed learning objective "
                        "supplied a bloom_level for this chunk; fell back to the "
                        "text verb heuristic (or the understand-level default) "
                        "so every chunk carries a level for downstream filters."
                    ),
                )

        # cognitive task-type axis. Orthogonal
        # to bloom_level — captures WHAT the learner is asked to do
        # (classify / compute / debug / critique / ...) rather than the
        # Bloom cognitive tier. Detector mirrors detect_bloom_level: first
        # whole-word match of a canonical task verb in the chunk text, or
        # None when no match. Optional field; skipped on no-match so legacy
        # chunks don't grow a null cognitive_task_type slot.
        from lib.ontology.cognitive_task import detect_cognitive_task_type
        cognitive_task_type = detect_cognitive_task_type(text)
        if cognitive_task_type:
            chunk["cognitive_task_type"] = cognitive_task_type
            capture = getattr(self, "capture", None)
            if capture is not None:
                stem_prefix = text[:80].replace("\n", " ").strip()
                capture.log_decision(
                    decision_type="cognitive_task_type_detection",
                    decision=f"Assigned cognitive_task_type={cognitive_task_type}",
                    rationale=(
                        f"detect_cognitive_task_type matched the canonical verb "
                        f"{cognitive_task_type!r} as a whole word in the chunk "
                        f"text (prefix={stem_prefix!r}, len={len(text)}). Adds "
                        f"an axis orthogonal to bloom_level={bloom_level!r} so "
                        "downstream consumers can audit per-task diversity."
                    ),
                )

        if content_type_label:
            chunk["content_type_label"] = content_type_label
        if key_terms:
            # Canonicalise SC references inside key-term metadata too.
            for kt in key_terms:
                if "term" in kt:
                    kt["term"] = canonicalize_sc_references(kt["term"])
                if "definition" in kt:
                    kt["definition"] = canonicalize_sc_references(kt["definition"])
            # chunk_v4 schema requires ``definition`` with minLength 1. Both
            # the data-cf-* fallback (``_extract_section_metadata`` emits
            # ``definition=""``) and occasional JSON-LD keyTerm entries with
            # empty/whitespace definitions trip schema validation when
            # ``TRAINFORGE_VALIDATE_CHUNKS=true``. Attempt a best-effort
            # extraction from the chunk's own text for each empty-definition
            # entry; drop entries when no definition can be recovered rather
            # than emit schema-invalid placeholders.
            key_terms = self._fill_or_drop_empty_key_term_definitions(key_terms, text)
            if key_terms:
                chunk["key_terms"] = key_terms

        # Stamp the per-block audit arrays harvested by
        # ``_extract_section_metadata`` from the matched JSON-LD
        # ``blocks[]`` entry. Conditional on truthy — empty list / None
        # means absent, preserving the back-compat contract that legacy
        # chunks (built from blocks without the audit arrays, or via the
        # data-cf-* fallback path) don't carry the new keys at all (not
        # even as ``[]``). The chunk_v4 schema admits both fields as
        # optional (W5.C) so strict validation is unaffected when absent.
        if section_metadata_extras.get("key_claims"):
            chunk["key_claims"] = section_metadata_extras["key_claims"]
        if section_metadata_extras.get("objective_alignment"):
            chunk["objective_alignment"] = section_metadata_extras["objective_alignment"]

        # W5.G — per-chunk reverse map: for every LO this chunk
        # claims to teach (chunk["learning_outcome_refs"]), look up the
        # source-chunk-IDs the page-level JSON-LD declares for that LO
        # and stamp a {lo_id: [chunk_id, ...]} projection. Lets downstream
        # consumers (training-pair synthesizer, eval harness) resolve
        # "which DART/Courseforge chunk(s) source TO-05?" without
        # re-loading synthesized_objectives.json off disk.
        reverse_map = self._build_learning_outcome_source_refs(
            item, chunk.get("learning_outcome_refs") or []
        )
        if reverse_map:
            chunk["learning_outcome_source_refs"] = reverse_map

        # (W5.F): when the chunk was emitted from a merge_small_sections
        # boundary, the merged audit arrays passed in via the chunker callback
        # take precedence (or supplement) the per-section JSON-LD harvest.
        # ``merge_small_sections`` collapses multiple ContentSections into one
        # text body; each section may carry its own per-block audit arrays
        # that the merge collects (concat for key_claims; first-seen-wins
        # dedupe for objective_alignment). Without this stamp, chunks emitted
        # from a merge boundary silently drop the audit arrays even when the
        # source sections carried them.
        if merged_key_claims:
            chunk["key_claims"] = list(merged_key_claims)
        if merged_objective_alignment:
            chunk["objective_alignment"] = list(merged_objective_alignment)

        # Page-level metadata
        misconceptions = item.get("misconceptions", [])
        if misconceptions:
            normalized_mis: List[Any] = []
            for m in misconceptions:
                if isinstance(m, dict) and "misconception" in m:
                    m = dict(m)
                    m["misconception"] = canonicalize_sc_references(m["misconception"])
                elif isinstance(m, str):
                    m = canonicalize_sc_references(m)
                normalized_mis.append(m)
            chunk["misconceptions"] = normalized_mis

        # vocabulary-driven retag + parent-outcome rollup.
        # Runs *after* the structured/JSON-LD/regex extraction in
        # _extract_objective_refs but *before* downstream consumers
        # (targeted_concepts propagation, summary, retrieval_text) so
        # they see the full set of refs. Both rules are additive —
        # never removes an existing ref. See Trainforge/retag_outcomes.py
        # for the rationale + vocabulary lists.
        from Trainforge.retag_outcomes import retag_chunk_outcomes
        retag_chunk_outcomes(chunk, parent_map=getattr(self, "_lo_parent_map", None))

        # propagate targetedConcepts[] from LOs onto chunks
        # whose learning_outcome_refs cite those LOs. Each chunk entry is
        # {"concept": <slug>, "bloom_level": <canonical level>} — a Bloom-
        # qualified LO→concept binding that downstream consumers (retrieval,
        # training-synthesis, SHACL validation) can key off of without
        # re-walking the LO list. Deduplicated across LOs by (concept,
        # bloom_level); preserves the first-seen bloom level when the same
        # concept shows up under multiple Bloom levels across different LOs
        # (matches the rule's first-wins dedup policy).
        lo_refs = chunk.get("learning_outcome_refs") or []
        if lo_refs:
            # Ref-resolution is case-insensitive (Trainforge convention).
            ref_set = {str(r).lower() for r in lo_refs if r}
            targeted: List[Dict[str, str]] = []
            seen_targeted: Set[tuple] = set()
            for lo in item.get("learning_objectives") or []:
                lo_id = getattr(lo, "id", None)
                if lo_id is None and isinstance(lo, dict):
                    lo_id = lo.get("id")
                if not isinstance(lo_id, str) or lo_id.lower() not in ref_set:
                    continue
                tc_list = getattr(lo, "targeted_concepts", None)
                if tc_list is None and isinstance(lo, dict):
                    tc_list = lo.get("targeted_concepts")
                for entry in tc_list or []:
                    if not isinstance(entry, dict):
                        continue
                    concept = entry.get("concept")
                    bloom = entry.get("bloom_level")
                    if not concept or not bloom:
                        continue
                    key = (concept, bloom)
                    if key in seen_targeted:
                        continue
                    seen_targeted.add(key)
                    targeted.append({
                        "concept": concept,
                        "bloom_level": bloom,
                    })
            if targeted:
                # Deterministic order: by (concept, bloom_level) so chunks
                # diff cleanly across runs.
                targeted.sort(key=lambda e: (e["concept"], e["bloom_level"]))
                chunk["targeted_concepts"] = targeted

        # Per-chunk summary for dense-retrieval recall augmentation (v4).
        # Add a deterministic extractive summary for retrieval augmentation.
        chunk["summary"] = summary_factory.generate(
            chunk["text"],
            key_terms=chunk.get("key_terms"),
            learning_outcome_refs=chunk.get("learning_outcome_refs"),
        )

        # Optional retrieval_text: summary + " " + key_terms_joined. Emitted
        # only when key_terms exist, since otherwise the field would just
        # duplicate `summary`. Benchmarked in
        # Trainforge/rag/retrieval_benchmark.py — on a representative real
        # course at commit time, retrieval_text lifted recall@5 from 0.0369
        # (text) to 0.0399 (retrieval_text); small but positive, so we ship it.
        kt = chunk.get("key_terms")
        if kt:
            kt_parts: List[str] = []
            for k in kt:
                if isinstance(k, dict):
                    term_s = k.get("term")
                    def_s = k.get("definition")
                    if term_s:
                        kt_parts.append(str(term_s))
                    if def_s:
                        kt_parts.append(str(def_s))
                elif isinstance(k, str):
                    kt_parts.append(k)
            kt_joined = " ".join(p for p in kt_parts if p).strip()
            if kt_joined:
                chunk["retrieval_text"] = f"{chunk['summary']} {kt_joined}".strip()

        # Record the source of each enrichment field for diagnostics.
        chunk["_metadata_trace"] = {
            "content_type_label": section_trace.get("content_type_label", "none"),
            "key_terms": section_trace.get("key_terms", "none"),
            "bloom_level": bloom_source or (
                "section_jsonld" if bloom_level and section_trace.get("content_type_label") == "jsonld_section_match" else "none"
            ),
            "misconceptions": "jsonld_page_misconceptions" if chunk.get("misconceptions") else (
                "none_jsonld_parse_failed" if item.get("_jsonld_parse_failed") else "none"
            ),
            # Trace entries for the audit fields.
            # Values: ``"jsonld_blocks_match"`` when the matched JSON-LD
            # block populated the field, ``"none"`` otherwise. Mirrors
            # the trace keys set in ``_extract_section_metadata``.
            "key_claims": section_trace.get("key_claims", "none"),
            "objective_alignment": section_trace.get("objective_alignment", "none"),
        }

        # Stamp the chunk schema version on every chunk so downstream
        # readers can gate on capabilities without re-reading manifest.json.
        chunk["schema_version"] = CHUNK_SCHEMA_VERSION

        # stamp run_id + created_at on every
        # newly-emitted chunk so downstream consumers can answer "all chunks
        # added after run R" and age out stale assertions at graph
        # granularity. `run_id` is sourced from the active DecisionCapture
        # ledger (same value that appears on decision_event.schema.json
        # records for this run). `created_at` is the emit timestamp in ISO
        # 8601 UTC. Both fields are optional at the schema level — legacy
        # chunks without them continue to validate.
        #
        # `capture` may be absent in unit tests that bypass __init__
        # (test_provenance.py pattern); mirror the defensive getattr used
        # for bloom_source logging at L1346. When capture is absent we
        # still stamp created_at (datetime.now is always available) but
        # skip run_id — a run_id requires a DecisionCapture instance.
        capture_for_run_id = getattr(self, "capture", None)
        if capture_for_run_id is not None:
            run_id = getattr(capture_for_run_id, "run_id", None)
            if run_id:
                chunk["run_id"] = run_id
        chunk["created_at"] = datetime.now(timezone.utc).isoformat()

        # CURIE anchoring (U3): fold the data-cf-curie tokens harvested
        # from the source HTML (force-injected CURIE anchor spans, per
        # the authoritative CURIE set) together with any CURIEs that appear
        # verbatim in the chunk prose (RDF/SHACL corpora) into the
        # chunk's ``curies`` / ``forced_curies`` fields. The downstream
        # ``curie_anchoring`` gate consumes ``curies`` as the
        # authoritative source-CURIE set for a chunk.
        #
        # Conditional stamping (mirrors source_document_sha256 /
        # key_terms): a chunk whose source HTML had no data-cf-curie
        # span AND no CURIE in prose emits neither key, so legacy / RDF
        # corpora stay byte-identical except where real CURIEs exist.
        # ``forced_curies`` is intersected with ``curies`` to enforce
        # the chunk_v4 subset invariant.
        from lib.ontology.curie_extraction import extract_curies as _extract_curies
        _prose_curies = _extract_curies(text)
        _all_curies = sorted(_prose_curies | set(curie_anchors or []))
        if _all_curies:
            chunk["curies"] = _all_curies
            _forced = sorted(set(forced_curie_anchors or []) & set(_all_curies))
            if _forced:
                chunk["forced_curies"] = _forced

        # SemantiK migration §4 — chunk-accompanying provenance enrichment.
        # Six optional chunk-root fields. All omit-when-absent (no null-filled
        # fields on legacy corpora — mirrors source_document_sha256 / key_terms):
        #   * 3 HTML-harvested (block_role / confidence / wcag): from the
        #     per-chunk ``dart_source_refs`` the chunker resolved off
        #     data-dart-block-role / data-dart-confidence / data-dart-wcag on
        #     the SAME element as data-dart-block-id (helpers.py same-element
        #     pairing). When a chunk spans several DART blocks, the FIRST
        #     resolved block (document order) supplies the role/confidence/wcag.
        #   * figure_alt: DOM-side recovery of <figcaption>/<img alt> for a
        #     figure chunk (retrieval-text augmentation + a11y audit).
        #   * 2 doc-level (semantic_preservation_score / certification_status):
        #     the SemantiK Stage-13 theta + exit-mode signals, stamped from the
        #     CourseProcessor instance (set by the P3 seam from the doc-level
        #     sidecar / PipelineV2Result), same value across every chunk of a
        #     doc (mirrors _source_document_sha256).
        self._stamp_semantik_chunk_enrichment(
            chunk, dart_source_refs=dart_source_refs, html=html,
            chunk_type=chunk_type,
        )

        # Honest IRT difficulty-calibration scaffold: stamp difficulty
        # provenance (+ optional IRT block from real learner responses) on the
        # chunk. No-op + byte-identical when TRAINFORGE_IRT_DIFFICULTY_SCAFFOLD
        # is off. The override can change the difficulty band, so re-read it for
        # the distribution counter below.
        try:
            from lib.assessment.irt_difficulty import (
                irt_scaffold_enabled,
                tag_chunk_difficulty_provenance,
            )
            if irt_scaffold_enabled():
                tag_chunk_difficulty_provenance(chunk, self._get_irt_calibrated_map())
                difficulty = chunk.get("difficulty", difficulty)
        except Exception as exc:  # noqa: BLE001 — scaffold must never break emit
            logger.warning("IRT difficulty provenance tagging failed: %s", exc)

        self.stats["total_words"] += word_count
        self.stats["total_tokens_estimate"] += tokens_estimate
        self.stats["chunk_types"][chunk_type] += 1
        self.stats["difficulty_distribution"][difficulty] += 1
        self._all_concept_tags.update(concept_tags)

        # Add pedagogical-role metadata harvested
        # from data-dart-unit / data-dart-flow / data-dart-opener). Omit-when-
        # absent so non-SemantiK IMSCC corpora stay byte-identical.
        if composite_unit:
            chunk["composite_unit"] = composite_unit
        if unit_roles:
            chunk["unit_roles"] = list(unit_roles)
        # Build #23 Tier-3: the composite-unit subclass (rides the unit). Omit-
        # when-absent so non-SemantiK / un-subclassed corpora stay byte-identical.
        if unit_subclass:
            chunk["unit_subclass"] = unit_subclass

        return chunk

    def _get_irt_calibrated_map(self) -> Dict[str, Any]:
        """Build the IRT calibrated-difficulty map once per run (lazy).

        Resolves the optional learner-response seam for ``self.course_code`` and
        computes a calibrated band only for items with real backing responses.
        Returns ``{}`` when no response file exists or no item clears the floor
        (anti-fabrication: never invents IRT parameters).
        """
        if self._irt_calibrated_map is not None:
            return self._irt_calibrated_map
        try:
            from lib.assessment.irt_difficulty import (
                load_calibrated_difficulty,
                resolve_response_data_path,
            )
            response_path = resolve_response_data_path(self.course_code)
            self._irt_response_file_present = response_path is not None
            self._irt_calibrated_map = load_calibrated_difficulty(response_path)
        except Exception as exc:  # noqa: BLE001 — scaffold must never break emit
            logger.warning("IRT calibrated-map build failed: %s", exc)
            self._irt_calibrated_map = {}
        return self._irt_calibrated_map

    # SemantiK migration §4 — WCAG block status enum (Stage-7 per-region gate).
    _SEMANTIK_WCAG_BLOCK_STATUSES = frozenset({"passed", "flagged", "skipped"})
    # SemantiK migration §4 — Stage-13 exit certification enum.
    _SEMANTIK_CERTIFICATION_STATUSES = frozenset(
        {"certified", "flagged", "non_certified"}
    )

    def _stamp_semantik_chunk_enrichment(
        self,
        chunk: Dict[str, Any],
        *,
        dart_source_refs: Optional[List[Dict[str, Any]]],
        html: str,
        chunk_type: str,
    ) -> None:
        """Populate the six SemantiK §4 chunk-accompanying-data fields.

        Omit-when-absent throughout — a legacy / non-SemantiK chunk (no
        enrichment on its source HTML and no doc-level signals on the
        instance) is stamped with NONE of these fields and stays
        byte-identical (back-compat, mirrors ``source_document_sha256``).
        """
        # --- 3 HTML-harvested fields (block_role / confidence / wcag) -------
        # The chunker resolved the DART block(s) overlapping this chunk; the
        # first (document order) supplies the per-block enrichment.
        first_ref: Optional[Dict[str, Any]] = None
        for ref in dart_source_refs or []:
            if isinstance(ref, dict):
                first_ref = ref
                break
        if first_ref is not None:
            role = first_ref.get("block_role")
            if isinstance(role, str) and role.strip():
                chunk["source_block_role"] = role.strip()
            conf = first_ref.get("confidence")
            if isinstance(conf, (int, float)) and not isinstance(conf, bool):
                conf_f = float(conf)
                if 0.0 <= conf_f <= 1.0:
                    chunk["source_block_confidence"] = conf_f
            wcag = first_ref.get("wcag_status")
            if (
                isinstance(wcag, str)
                and wcag.strip() in self._SEMANTIK_WCAG_BLOCK_STATUSES
            ):
                chunk["wcag_block_status"] = wcag.strip()

        # --- figure_alt (DOM-side, figure chunks only) ----------------------
        if chunk_type == "figure" or (html and "<figure" in html.lower()):
            alt = self._recover_figure_alt(html)
            if alt:
                chunk["figure_alt"] = alt

        # --- 2 doc-level fields (theta + exit certification) ----------------
        # Stamped from CourseProcessor instance attributes the P3 seam sets
        # from the doc-level sidecar / PipelineV2Result (same value across
        # every chunk of a doc). Absent attribute / out-of-range value -> omit.
        theta = getattr(self, "_semantic_preservation_score", None)
        if isinstance(theta, (int, float)) and not isinstance(theta, bool):
            theta_f = float(theta)
            if 0.0 <= theta_f <= 1.0:
                chunk["semantic_preservation_score"] = theta_f
        cert = getattr(self, "_certification_status", None)
        if (
            isinstance(cert, str)
            and cert.strip() in self._SEMANTIK_CERTIFICATION_STATUSES
        ):
            chunk["certification_status"] = cert.strip()

    @staticmethod
    def _recover_figure_alt(html: str) -> Optional[str]:
        """Best-effort DOM-side recovery of a figure chunk's alt / caption.

        Prefers a ``<figcaption>`` text, falling back to an ``<img alt="...">``
        value. Returns ``None`` when neither resolves (anti-fabrication —
        never invents alt text). Pure regex; no bs4 dependency added.
        """
        if not html:
            return None
        cap = re.search(
            r"<figcaption\b[^>]*>(.*?)</figcaption>",
            html,
            re.IGNORECASE | re.DOTALL,
        )
        if cap:
            text = re.sub(r"<[^>]+>", " ", cap.group(1))
            text = " ".join(text.split()).strip()
            if text:
                return text
        alt = re.search(
            r"<img\b[^>]*\balt\s*=\s*([\"'])(.*?)\1",
            html,
            re.IGNORECASE | re.DOTALL,
        )
        if alt:
            text = " ".join(alt.group(2).split()).strip()
            if text:
                return text
        return None

    def _resolve_chunk_source_references(
        self,
        *,
        item: Dict[str, Any],
        section_heading: str,
        section_source_ids: List[str],
        merged_headings: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Resolve the chunk's source_references array.

        Walks the precedence chain and returns a list of full
        SourceReference dicts (one per unique sourceId). Returns an empty
        list when no references are available (input without provenance or a
        chunk whose section carries no ``data-cf-source-ids`` and whose
        page JSON-LD carries no ``sourceReferences``).

        Precedence (first-seen wins on sourceId collision):
          1. Page-level JSON-LD ``sourceReferences`` (full shape).
          2. Section-level JSON-LD ``sourceReferences`` (matched by
             heading equality, case-insensitive; ``(part N)`` suffixes
             stripped to match _extract_section_metadata).
          3. Section-level ``data-cf-source-ids`` — the stringified ids
             that came through _merge_small_sections → auto-roled as
             ``contributing`` (P1 decision: HTML attrs lack role
             authority so they default to contributing).
          4. Page-level ``data-cf-source-ids`` fallback — not rebuilt
             here because _extract_sections already rolls them into
             item['source_references'] via _build_page_source_refs;
             already captured in step 1 above.

        Role-precedence across JSON-LD entries (primary > contributing >
        corroborating) is preserved because the parser's
        _build_page_source_refs already normalised the dedup; downstream
        reads them in first-seen order and uses their authoritative role.
        """
        refs: List[Dict[str, Any]] = []
        seen: set = set()

        def _add(entry: Dict[str, Any]) -> None:
            sid = entry.get("sourceId") if isinstance(entry, dict) else None
            if not isinstance(sid, str) or not sid:
                return
            if sid in seen:
                return
            seen.add(sid)
            refs.append(dict(entry))

        # 1. Page-level parsed refs (already aggregated by the parser:
        # page JSON-LD + section JSON-LD + HTML fallback merged with
        # JSON-LD precedence). item["source_references"] is a list of
        # full SourceReference dicts when input is present.
        for entry in item.get("source_references", []) or []:
            if isinstance(entry, dict):
                _add(entry)

        # 2. Section-level JSON-LD override: if a JSON-LD section matches
        # this chunk's heading and declares its own sourceReferences, add
        # them. The parser's _build_page_source_refs already merged these
        # into item["source_references"] so step 1 typically covers this,
        # but we re-walk here to ensure per-chunk specificity when
        # sections carry refs that aren't in the page-level set.
        # walk every merged heading, not just the anchor, so a
        # post-merge chunk picks up sourceReferences from any of the
        # sub-sections that collapsed into it.
        def _norm_h(h: str) -> str:
            return re.sub(r'\s*\(part\s+\d+\)\s*$', '', h or '').lower()

        candidate_headings: List[str] = [_norm_h(section_heading)]
        for h in (merged_headings or []):
            nh = _norm_h(h)
            if nh and nh not in candidate_headings:
                candidate_headings.append(nh)
        cf_meta = item.get("courseforge_metadata") or {}
        for cand in candidate_headings:
            for sec in cf_meta.get("sections", []) or []:
                if not isinstance(sec, dict):
                    continue
                if sec.get("heading", "").lower() != cand:
                    continue
                for entry in sec.get("sourceReferences", []) or []:
                    if isinstance(entry, dict):
                        _add(entry)
                break

        # 3. Section-level data-cf-source-ids (stringified; auto-roled).
        # These come from ``_merge_small_sections`` which already unioned
        # every merged section's attrs. Any id already captured in steps
        # 1-2 above keeps its authoritative role via first-seen-wins.
        for sid in section_source_ids or []:
            _add({"sourceId": sid, "role": "contributing"})

        return refs

    def _extract_section_metadata(
        self, item: Dict[str, Any], section_heading: str,
        *,
        merged_headings: Optional[List[str]] = None,
    ) -> Tuple[
        Optional[str],
        Optional[str],
        List[Dict[str, str]],
        Dict[str, Any],
        Dict[str, str],
    ]:
        """Extract bloom_level, content_type_label, and key_terms for a section.

        Checks JSON-LD sections metadata first, then falls back to
        ContentSection data-cf-* attributes.

        ``merged_headings`` is the ordered list of
        every section heading that collapsed into this chunk's buffer (via
        ``_merge_small_sections``). When provided, the JSON-LD section
        match tries each heading in order so post-merge chunks whose
        anchor heading drifted from the JSON-LD-keyed heading still find
        their metadata. Falls back to the single ``section_heading`` when
        ``merged_headings`` is empty (back-compat).

        Returns a 5-tuple:
            ``(bloom_level, content_type_label, key_terms,
               section_metadata_extras, trace)``.

        The returned metadata includes optional per-block audit arrays:
        ``section_metadata_extras`` carries the optional per-block audit
        arrays harvested from the JSON-LD ``blocks[]`` projection so the
        chunker can stamp them onto the emitted chunk dict without
        re-walking ``courseforge_metadata`` downstream. Shape:

          ``{
                "key_claims":         List[Dict[str, Any]],   # W1.5 / W5.A
                "objective_alignment": List[Dict[str, Any]],  # W1.7 / W5.A
            }``

        Empty lists when the matched block didn't carry the field, when
        no block matched, or when ``blocks[]`` is absent entirely. The
        chunker stamps each field onto the chunk dict only when truthy,
        preserving the back-compat contract that legacy chunks don't
        carry the new keys at all (not even as ``[]``).

        ``trace`` names the
        source path for each field. Values:

          - ``jsonld_blocks_match``         — JSON-LD ``blocks[]`` matched +
                                              populated (W5.B for the new
                                              ``key_claims`` / ``objective_alignment``
                                              entries; pre-existing usage on
                                              content_type_label / key_terms)
          - ``jsonld_section_match``        — JSON-LD section matched + populated
          - ``jsonld_section_match_empty``  — JSON-LD section matched but that
                                              specific field was empty on it
                                              (the H3 short-circuit signature
                                              for key_terms when contentType
                                              is present but keyTerms is not)
          - ``data_cf_fallback``            — data-cf-* section path populated
          - ``none_no_jsonld_sections``     — `cf_meta.sections` absent (H2)
          - ``none_jsonld_parse_failed``    — JSON-LD `<script>` present in
                                              raw HTML but parse failed (H5)
          - ``none_heading_mismatch``       — sections exist but no heading
                                              matched the chunk heading (H1)
          - ``none_no_sections_path``       — this chunk came via the
                                              `item["sections"] is empty`
                                              code path in `_chunk_content`
                                              (H4 signature — `section_heading`
                                              equals the page title, and no
                                              section with that heading exists
                                              in the JSON-LD `sections` list)
          - ``none``                        — residual missing
        """
        bloom_level: Optional[str] = None
        content_type_label: Optional[str] = None
        key_terms: List[Dict[str, str]] = []
        # Per-block audit arrays harvested from the matched
        # JSON-LD block. Empty lists by default — populated below when a
        # ``blocks[]`` entry matches the section heading and carries the
        # field, or via the data-cf-* fallback path (no-op day-1 — see
        # comment at the fallback site).
        section_metadata_extras: Dict[str, Any] = {
            "key_claims": [],
            "objective_alignment": [],
        }
        trace: Dict[str, str] = {
            "content_type_label": "none",
            "key_terms": "none",
            # Trace entries for the audit fields.
            # Values: "jsonld_blocks_match" when the matched block populated
            # the field, "none" otherwise. data-cf-* fallback never sets
            # these (no current attribute carries them) — symmetric path
            # left in place for future use.
            "key_claims": "none",
            "objective_alignment": "none",
        }

        # Normalize heading: strip "(part N)" suffix added by _chunk_text_block
        # so multi-part chunks still match their JSON-LD / data-cf-* metadata.
        def _norm(h: str) -> str:
            return re.sub(r'\s*\(part\s+\d+\)\s*$', '', h or '').lower()

        chunk_heading = _norm(section_heading)
        # ordered list of headings to try, in priority order.
        # Anchor heading first (so single-section chunks behave identically
        # to legacy), then any merged headings that aren't already in
        # the list.
        candidate_headings: List[str] = [chunk_heading]
        for h in (merged_headings or []):
            nh = _norm(h)
            if nh and nh not in candidate_headings:
                candidate_headings.append(nh)

        # Signals for hypothesis discrimination (instrumentation).
        cf_meta = item.get("courseforge_metadata")
        jsonld_has_sections = bool(cf_meta and cf_meta.get("sections"))
        jsonld_has_blocks = bool(cf_meta and cf_meta.get("blocks"))
        jsonld_parse_failed = bool(item.get("_jsonld_parse_failed"))

        # prefer JSON-LD ``blocks[]`` when present.
        # Courseforge with ``COURSEFORGE_EMIT_BLOCKS=true`` emits a
        # canonical Phase-2 ``blocks[]`` projection on every page; for
        # the section-type entries, the legacy ``_section_jsonld()``
        # shape (``{heading, contentType, keyTerms?, bloomRange?, ...}``)
        # is the wire shape. Walk those first and match by heading;
        # populate ``bloom_level``/``content_type_label``/``key_terms``
        # from the matched block. Trace value ``"jsonld_blocks_match"``.
        # When no block matches in ``blocks[]`` (legacy emit, or a
        # heading that drifts from the JSON-LD-keyed heading), fall
        # through to the existing ``cf_meta["sections"]`` path so
        # legacy corpora keep their pre-Phase-2 metadata-extraction
        # semantics unchanged.
        if jsonld_has_blocks:
            for cand in candidate_headings:
                matched_block = None
                for block in cf_meta["blocks"]:
                    if not isinstance(block, dict):
                        continue
                    if block.get("heading", "").lower() == cand:
                        matched_block = block
                        break
                if not matched_block:
                    continue
                blk_content_type = matched_block.get("contentType") or matched_block.get(
                    "contentTypeLabel"
                )
                if blk_content_type:
                    content_type_label = blk_content_type
                    trace["content_type_label"] = "jsonld_blocks_match"
                blk_bloom_range = matched_block.get("bloomRange", [])
                if blk_bloom_range:
                    bloom_level = (
                        blk_bloom_range[0]
                        if isinstance(blk_bloom_range, list)
                        else blk_bloom_range
                    )
                elif matched_block.get("bloomLevel"):
                    bloom_level = matched_block["bloomLevel"]
                for kt in matched_block.get("keyTerms", []):
                    if isinstance(kt, dict) and kt.get("term"):
                        key_terms.append(
                            {"term": kt["term"], "definition": kt.get("definition", "")}
                        )
                    elif isinstance(kt, str) and kt:
                        # Phase-2 ``_minimal_block_jsonld`` shape carries
                        # ``keyTerms`` as a string list (slugs); the
                        # legacy ``_section_jsonld`` shape carries it as
                        # ``[{term, definition}]`` dicts. Honor both so
                        # the consumer is robust to either emit shape.
                        key_terms.append({"term": kt, "definition": ""})
                if key_terms:
                    trace["key_terms"] = "jsonld_blocks_match"
                elif content_type_label:
                    trace["key_terms"] = "jsonld_blocks_match"
                # Harvest the ``keyClaims`` and
                # ``objectiveAlignment`` audit arrays from the matched
                # block. camelCase ↔ snake_case translation: JSON-LD wire
                # keys are camelCase, snake_case on the chunker side.
                # Defensive dict-only filter mirrors the posture in
                # ``html_content_parser.HTMLContentParser._content_sections_from_blocks``
                # so a malformed payload that slipped past schema
                # validation can't poison the chunker.
                raw_key_claims = matched_block.get("keyClaims") or []
                blk_key_claims: List[Dict[str, Any]] = [
                    kc for kc in raw_key_claims if isinstance(kc, dict)
                ]
                if blk_key_claims:
                    section_metadata_extras["key_claims"] = blk_key_claims
                    trace["key_claims"] = "jsonld_blocks_match"
                raw_objective_alignment = (
                    matched_block.get("objectiveAlignment") or []
                )
                blk_objective_alignment: List[Dict[str, Any]] = [
                    oa for oa in raw_objective_alignment if isinstance(oa, dict)
                ]
                if blk_objective_alignment:
                    section_metadata_extras["objective_alignment"] = (
                        blk_objective_alignment
                    )
                    trace["objective_alignment"] = "jsonld_blocks_match"
                break

        # Try JSON-LD sections metadata. Walk candidate headings in order —
        # the first matching JSON-LD section wins (anchor heading is
        # checked first to preserve back-compat).
        if jsonld_has_sections and not content_type_label:
            for cand in candidate_headings:
                matched_sec = None
                for sec in cf_meta["sections"]:
                    if sec.get("heading", "").lower() == cand:
                        matched_sec = sec
                        break
                if not matched_sec:
                    continue
                sec_content_type = matched_sec.get("contentType")
                if sec_content_type:
                    content_type_label = sec_content_type
                    trace["content_type_label"] = "jsonld_section_match"
                bloom_range = matched_sec.get("bloomRange", [])
                if bloom_range:
                    bloom_level = bloom_range[0] if isinstance(bloom_range, list) else bloom_range
                for kt in matched_sec.get("keyTerms", []):
                    if isinstance(kt, dict) and kt.get("term"):
                        key_terms.append({"term": kt["term"], "definition": kt.get("definition", "")})
                if key_terms:
                    trace["key_terms"] = "jsonld_section_match"
                elif content_type_label:
                    # H3 signature: section matched, contentType set,
                    # but keyTerms empty on the section. The data-cf-*
                    # fallback below is gated by `if not content_type_label`
                    # so it never runs — key_terms stays empty.
                    trace["key_terms"] = "jsonld_section_match_empty"
                break

        # Fallback: data-cf-* attributes from parsed sections.
        # Walk candidate headings the same way for symmetry with the
        # JSON-LD path — a merged sub-section's data-cf-* attributes
        # should still be reachable when the anchor heading drifts.
        if not content_type_label:
            for cand in candidate_headings:
                matched_section = None
                for section in item.get("sections", []):
                    if section.heading.lower() == cand:
                        matched_section = section
                        break
                if not matched_section:
                    continue
                if matched_section.content_type:
                    content_type_label = matched_section.content_type
                    trace["content_type_label"] = "data_cf_fallback"
                if matched_section.key_terms:
                    key_terms = [{"term": t, "definition": ""} for t in matched_section.key_terms]
                    trace["key_terms"] = "data_cf_fallback"
                # Symmetric data-cf-* fallback for the
                # audit arrays. Day-1 NO-OP — no current data-cf-*
                # attribute carries ``key_claims`` / ``objective_alignment``
                # so ``ContentSection.key_claims`` / ``.objective_alignment``
                # only ever populate via the JSON-LD ``blocks[]`` projection
                # path in
                # ``HTMLContentParser._content_sections_from_blocks``
                # (W5.A). The fallback path is wired in for symmetry with
                # the content_type / key_terms gates above so a future
                # data-cf-* attribute (e.g. ``data-cf-key-claims-json``)
                # would slot in cleanly. When the originating section
                # carries the field via ``ContentSection`` it lifts here
                # without further changes; back-compat preserved because
                # the dataclass defaults to ``[]``.
                section_key_claims = getattr(matched_section, "key_claims", None) or []
                if section_key_claims and not section_metadata_extras["key_claims"]:
                    section_metadata_extras["key_claims"] = section_key_claims
                    trace["key_claims"] = "data_cf_attr_match"
                section_objective_alignment = (
                    getattr(matched_section, "objective_alignment", None) or []
                )
                if (
                    section_objective_alignment
                    and not section_metadata_extras["objective_alignment"]
                ):
                    section_metadata_extras["objective_alignment"] = (
                        section_objective_alignment
                    )
                    trace["objective_alignment"] = "data_cf_attr_match"
                break

        # Categorize remaining `none` values by hypothesis so the trace report
        # can attribute each failure to H1/H2/H4/H5.
        if trace["content_type_label"] == "none":
            if jsonld_parse_failed:
                trace["content_type_label"] = "none_jsonld_parse_failed"
            elif not jsonld_has_sections:
                # H2 — JSON-LD for the page either absent or has an empty
                # `sections` array. No section metadata to match against.
                trace["content_type_label"] = "none_no_jsonld_sections"
            elif section_heading == item.get("title", ""):
                # H4 — chunk heading is the page title; JSON-LD sections
                # are keyed by section headings, so structurally no match
                # is possible on this path.
                trace["content_type_label"] = "none_no_sections_path"
            else:
                # H1 — JSON-LD sections populated but the heading drifted
                # (entity / whitespace / punctuation / case mismatch).
                trace["content_type_label"] = "none_heading_mismatch"
        if trace["key_terms"] == "none":
            # key_terms failure mirrors the content_type outcome for the
            # non-H3 cases, plus the H3-signature case handled above.
            if trace["content_type_label"].startswith("none_"):
                trace["key_terms"] = trace["content_type_label"]

        # Fallback: derive bloom_level from learning objectives
        if not bloom_level and item.get("learning_objectives"):
            for lo in item["learning_objectives"]:
                if lo.bloom_level:
                    bloom_level = lo.bloom_level
                    break

        return bloom_level, content_type_label, key_terms, section_metadata_extras, trace

    @staticmethod
    def _fill_or_drop_empty_key_term_definitions(
        key_terms: List[Dict[str, str]], section_text: str
    ) -> List[Dict[str, str]]:
        """Ensure every key_term entry has a non-empty ``definition``.

        chunk_v4 schema requires ``KeyTerm.definition`` with ``minLength: 1``.
        The data-cf-* fallback path in ``_extract_section_metadata`` synthesises
        ``{"term": t, "definition": ""}`` because data-cf-* attrs carry term
        slugs but no prose definition. Occasional JSON-LD entries with empty
        definitions also exist. For any entry lacking a definition, attempt
        to lift one from the chunk's own text by finding the first sentence
        that mentions the term. When extraction fails, drop the entry rather
        than emit a schema-invalid placeholder.
        """
        if not key_terms:
            return []

        # Cache sentence splits once per chunk text rather than per-term.
        sentences: List[str] = []
        if section_text:
            # Lightweight sentence split — enough for definition lookup. We
            # avoid pulling an NLP dependency; the heuristic only needs to
            # find "the sentence mentioning X" and return a single line.
            for raw in re.split(r"(?<=[.!?])\s+", section_text):
                s = raw.strip()
                if s:
                    sentences.append(s)

        def _find_definition(term: str) -> str:
            if not term:
                return ""
            term_norm = term.strip().lower()
            if not term_norm:
                return ""
            # Prefer sentences where the term appears (whole-word match).
            # Fall back to simple substring when the word boundary path
            # misses (multi-word terms, punctuation, hyphens).
            pattern = re.compile(
                r"(?<!\w)" + re.escape(term_norm) + r"(?!\w)",
                re.IGNORECASE,
            )
            for sentence in sentences:
                if pattern.search(sentence):
                    return sentence
            for sentence in sentences:
                if term_norm in sentence.lower():
                    return sentence
            return ""

        filled: List[Dict[str, str]] = []
        for kt in key_terms:
            if not isinstance(kt, dict):
                continue
            term = (kt.get("term") or "").strip()
            definition = (kt.get("definition") or "").strip()
            if not term:
                continue
            if definition:
                filled.append({"term": term, "definition": definition})
                continue
            derived = _find_definition(term)
            if derived:
                filled.append({"term": term, "definition": derived})
            # else: omit — never emit empty-string placeholder.
        return filled

    # ------------------------------------------------------------------
    # Chunk helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _type_from_resource(resource_type: str) -> str:
        # canonical implementation lives in
        # ``Trainforge.chunker.helpers.type_from_resource``. This wrapper
        # preserves the staticmethod surface so existing call sites
        # (``self._type_from_resource(...)`` and
        # ``CourseProcessor._type_from_resource(...)``) keep working.
        from Trainforge.chunker.helpers import type_from_resource

        return type_from_resource(resource_type)

    @staticmethod
    def _type_from_heading(heading: str) -> str:
        h = heading.lower()
        if any(kw in h for kw in ("example", "case study", "scenario")):
            return "example"
        if any(kw in h for kw in ("exercise", "activity", "practice", "application")):
            return "exercise"
        if any(kw in h for kw in ("summary", "recap", "key takeaway", "conclusion")):
            return "summary"
        if any(kw in h for kw in ("overview", "introduction", "welcome")):
            return "overview"
        if any(kw in h for kw in ("self-check", "self check", "knowledge check", "quiz", "check your")):
            return "assessment_item"
        if any(kw in h for kw in ("discussion", "reflection")):
            return "exercise"
        return "explanation"

    # Concept-tag extraction pipeline constants. The canonical
    # definitions now live in ``lib/ontology/concept_tagging.py`` so the
    # instance-free ``extract_concept_tags`` helper and ``CourseProcessor``
    # share one source of truth (no duplication). These class attributes
    # are kept as re-bindings because other ``CourseProcessor`` methods
    # reference ``self.OBJECTIVE_CODE_RE`` / ``self.WEEK_PREFIX_RE`` /
    # ``self.NON_CONCEPT_TAGS`` (e.g. ``_extract_objective_refs``).
    # Common educational concept patterns for text-based extraction.
    CONCEPT_PATTERNS: Dict[str, List[str]] = _concept_tagging.CONCEPT_PATTERNS
    # Pattern for course/terminal/learning objective codes (CO-01, TO-08, LO-003, etc.)
    OBJECTIVE_CODE_RE = _concept_tagging.OBJECTIVE_CODE_RE
    # Week prefix pattern (w01-, w02-) used by Courseforge JSON-LD but absent in course.json
    WEEK_PREFIX_RE = _concept_tagging.WEEK_PREFIX_RE
    # Non-concept tags to filter out (generic metadata, not knowledge concepts)
    NON_CONCEPT_TAGS = _concept_tagging.NON_CONCEPT_TAGS

    def _extract_concept_tags(self, text: str, item: Dict[str, Any]) -> List[str]:
        # Delegates to the instance-free ``extract_concept_tags`` helper
        # in ``lib/ontology/concept_tagging.py``. The helper was lifted
        # out of this method so the canonical DART chunkset path
        # (``MCP/tools/pipeline_tools.py::_run_dart_chunking``) can emit
        # real ``concept_tags`` instead of ``[]`` — it has no
        # ``CourseProcessor`` instance, so the per-course
        # ``domain_concept_seeds`` is threaded as a parameter rather
        # than read off ``self``.
        #
        # Pipeline shape:
        #   normalize_tag (HTML-decoded, slugified)
        #   → strip_lo_ref_suffix (drop -co-NN / -to-NN suffix)
        #   → canonicalize_alias (rdfxml → rdf-xml, ttl → turtle, …)
        #   → classify_concept → is_droppable_class → tag list
        #     (dedup-aware, singular-preferred).
        return _concept_tagging.extract_concept_tags(
            text,
            item,
            domain_concept_seeds=self.domain_concept_seeds,
        )

    def _extract_objective_refs(
        self,
        item: Dict[str, Any],
        section_heading: Optional[str] = None,
    ) -> List[str]:
        """Extract learning objective reference codes for a chunk.

        Resolution order:
          1. Structured IDs from JSON-LD / parsed ``LearningObjective.id``
             on the page-level ``learning_objectives`` list.
          2. Regex extraction of CO/TO codes from ``key_concepts`` as
             fallback when no structured IDs were present.
          3. ``data-cf-objective-ref`` on
             ``.activity-card`` / ``.self-check`` elements. Preferred
             section-scoped (matching the chunk's heading) with page-level
             fallback when no section matches.

        Case policy: controlled by ``TRAINFORGE_PRESERVE_LO_CASE``. Default
        (unset / non-``true``) lowercases every ref for backward-compat
        with existing LibV2 chunks. When ``TRAINFORGE_PRESERVE_LO_CASE=true``
        refs pass through with their source casing (still stripped and
        week-prefix-folded — ``WEEK_PREFIX_RE`` has ``re.IGNORECASE``).

        Enabling case preservation does NOT make the corpus case-consistent:
        the downstream ``valid_outcome_ids`` sites and ``align_chunks.py``
        still lowercase, so cross-artifact joins must compare case-folded.
        """
        preserve_case = (
            os.getenv("TRAINFORGE_PRESERVE_LO_CASE", "").lower() == "true"
        )

        def _normalize_one(raw: str) -> str:
            """Apply case policy + week-prefix stripping to a single ref."""
            base = raw.strip() if preserve_case else raw.lower().strip()
            # Strip week prefix (w01-, W01-, w02-, ...) to align with
            # course.json format. WEEK_PREFIX_RE is case-insensitive.
            return self.WEEK_PREFIX_RE.sub('', base)

        def _normalize(raw: Any) -> List[str]:
            """Normalize a raw reference into a list of references.

            Handles malformed comma-delimited strings like
            ``"co-01,co-02,co-03"`` (observed in subagent-emitted
            chunks) by splitting on comma and stripping each piece.
            Always returns a list — callers must extend, not append.
            """
            if raw is None:
                return []
            if not isinstance(raw, str):
                raw = str(raw)
            if "," in raw:
                parts = [p.strip() for p in raw.split(",")]
            else:
                parts = [raw]
            out: List[str] = []
            for p in parts:
                normed = _normalize_one(p)
                if normed:
                    out.append(normed)
            return out

        refs: List[str] = []

        def _extend(values: List[str]) -> None:
            for v in values:
                if v and v not in refs:
                    refs.append(v)

        # (1) Structured objective IDs from parser (JSON-LD or data-cf-*).
        for lo in item.get("learning_objectives", []):
            obj_id = lo.id if hasattr(lo, "id") else lo.get("id")
            if obj_id:
                _extend(_normalize(obj_id))

        # (2) Fallback: regex extraction from key_concepts when no
        # structured IDs were available. Preserves prior behavior of
        # returning-early when refs already populated from (1).
        if not refs:
            for concept in item.get("key_concepts", []):
                tag = normalize_tag(concept)
                if tag and self.OBJECTIVE_CODE_RE.match(tag) and tag not in refs:
                    refs.append(tag)

        # Merge activity and self-check objective references.
        # Prefer the section matching this chunk's heading; fall back to
        # the page-level union when no section matches (no-sections code
        # path in _chunk_content or heading drift).
        activity_refs: List[str] = []
        if section_heading:
            chunk_heading = re.sub(
                r'\s*\(part\s+\d+\)\s*$', '', section_heading
            ).lower()
            for section in item.get("sections", []):
                if section.heading.lower() == chunk_heading:
                    activity_refs = list(section.objective_refs)
                    break
        if not activity_refs:
            # Fallback to page-level refs harvested by the parser.
            activity_refs = list(item.get("objective_refs", []))

        for raw_ref in activity_refs:
            _extend(_normalize(raw_ref))

        return refs

    def _build_learning_outcome_source_refs(
        self,
        item: Dict[str, Any],
        lo_refs: List[str],
    ) -> Dict[str, List[str]]:
        """Build a per-chunk reverse map of
        ``{lo_id: [source_chunk_id, ...]}`` from
        ``item.courseforge_metadata.learningObjectives[].sourceReferences[]``.

        For each LO in ``lo_refs`` (this chunk's ``learning_outcome_refs``),
        walks the page-level JSON-LD ``learningObjectives[]`` looking for
        a matching ``id`` (case-insensitive) and harvests the
        ``sourceReferences[].sourceId`` strings so downstream consumers
        can resolve "which DART/Courseforge chunk(s) source TO-05?"
        without reloading ``synthesized_objectives.json``.

        Output key case follows the JSON-LD emit case (e.g. ``TO-05``,
        not the chunk's lowercased ``to-05``); chunk lookup is
        case-insensitive so a lowercased chunk ref can still match a
        canonical-cased LO id.

        Returns ``{}`` (NOT ``None``) when no matches — the call site
        gates on truthiness, so empty maps are elided from the chunk
        dict to preserve the back-compat absence-as-legacy contract.
        """
        if not lo_refs:
            return {}
        cf_meta = item.get("courseforge_metadata") or {}
        lo_entries = cf_meta.get("learningObjectives") or []
        if not lo_entries:
            return {}
        # Case-insensitive membership set for matching against JSON-LD ids.
        lo_ref_lookup = {ref.lower().strip() for ref in lo_refs if isinstance(ref, str) and ref.strip()}
        if not lo_ref_lookup:
            return {}
        reverse_map: Dict[str, List[str]] = {}
        for entry in lo_entries:
            if not isinstance(entry, dict):
                continue
            raw_id = entry.get("id") or entry.get("@id")
            if not isinstance(raw_id, str) or not raw_id.strip():
                continue
            emit_id = raw_id.strip()
            if emit_id.lower() not in lo_ref_lookup:
                continue
            refs = entry.get("sourceReferences") or []
            if not isinstance(refs, list):
                continue
            collected: List[str] = []
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                source_id = ref.get("sourceId")
                if not isinstance(source_id, str) or not source_id.strip():
                    continue
                collected.append(source_id.strip())
            if collected:
                # Multiple JSON-LD entries may share the same id (rare;
                # legacy duplication). Extend rather than overwrite so no
                # source-chunk-ID is silently dropped.
                if emit_id in reverse_map:
                    reverse_map[emit_id].extend(collected)
                else:
                    reverse_map[emit_id] = collected
        return reverse_map

    @staticmethod
    def _median_difficulty(blooms: List[str]) -> str:
        """Compute difficulty from median Bloom's weight of a week's objectives."""
        weights = sorted(BLOOM_WEIGHT[b] for b in blooms if b in BLOOM_WEIGHT)
        if not weights:
            return "intermediate"
        median = weights[len(weights) // 2]
        if median <= 2:
            return "foundational"
        if median <= 4:
            return "intermediate"
        return "advanced"

    @staticmethod
    def _cap_difficulty(difficulty: str) -> str:
        """Lower difficulty by one level (for introductory resource types)."""
        if difficulty == "advanced":
            return "intermediate"
        if difficulty == "intermediate":
            return "foundational"
        return "foundational"

    def _determine_difficulty(self, text: str, item: Dict[str, Any]) -> str:
        difficulty = None

        # First: check JSON-LD metadata for authoritative Bloom's levels
        cf_meta = item.get("courseforge_metadata")
        if cf_meta and cf_meta.get("learningObjectives"):
            for lo in cf_meta["learningObjectives"]:
                bl = lo.get("bloomLevel")
                if bl and bl in BLOOM_TO_DIFFICULTY:
                    difficulty = BLOOM_TO_DIFFICULTY[bl]
                    break

        # Second: use objectives file if we have week→bloom mapping
        if difficulty is None and self.objectives:
            week = item.get("week_num", 0)
            blooms = self.objectives.get("week_bloom_map", {}).get(week, [])
            if blooms:
                difficulty = self._median_difficulty(blooms)

        # Third: check learning objectives extracted from HTML
        if difficulty is None and item.get("learning_objectives"):
            for lo in item["learning_objectives"]:
                if lo.bloom_level and lo.bloom_level in BLOOM_TO_DIFFICULTY:
                    difficulty = BLOOM_TO_DIFFICULTY[lo.bloom_level]
                    break

        # Fallback: keyword heuristics
        if difficulty is None:
            text_lower = text.lower()
            if any(kw in text_lower for kw in ("basic", "introduction", "overview", "what is", "define")):
                difficulty = "foundational"
            elif any(kw in text_lower for kw in ("evaluate", "create", "design", "critique", "justify")):
                difficulty = "advanced"
            else:
                difficulty = "intermediate"

        # Cap difficulty for introductory resource types
        if item.get("resource_type") in INTRODUCTORY_RESOURCE_TYPES:
            difficulty = self._cap_difficulty(difficulty)

        return difficulty

    @staticmethod
    def _extract_plain_text(html: str) -> str:
        # canonical implementation lives in
        # ``Trainforge.chunker.helpers.extract_plain_text``. The wrapper
        # delegates so DART / Courseforge / Trainforge can converge on
        # one HTML-text extraction surface without breaking
        # any existing call site here.
        from Trainforge.chunker.helpers import extract_plain_text

        return extract_plain_text(html)

    @staticmethod
    def _extract_section_html(html: str, heading: str) -> str:
        """Return the HTML fragment for ``heading``, respecting section boundaries.

        The canonical implementation lives in
        ``Trainforge.chunker.helpers.extract_section_html`` with section-aware
        boundaries. This wrapper preserves the staticmethod surface so
        every existing call site — including the regression suite at
        ``Trainforge/tests/test_extract_section_html_boundary.py`` which
        invokes ``CourseProcessor._extract_section_html(...)`` directly
        via the class — keeps working without modification.
        """

        from Trainforge.chunker.helpers import extract_section_html

        return extract_section_html(html, heading)

    @staticmethod
    def _strip_assessment_feedback(html: str) -> str:
        """Remove answer feedback from quiz/self-check HTML before text extraction.

        The canonical implementation lives in
        ``Trainforge.chunker.helpers.strip_assessment_feedback``. Wrapper
        preserves the staticmethod surface for existing call sites.
        """

        from Trainforge.chunker.helpers import strip_assessment_feedback

        return strip_assessment_feedback(html)

    @staticmethod
    def _strip_feedback_from_text(text: str) -> str:
        """Remove residual feedback markers from plain text extraction.

        The canonical implementation lives in
        ``Trainforge.chunker.helpers.strip_feedback_from_text``. Wrapper
        preserves the staticmethod surface for existing call sites.
        """

        from Trainforge.chunker.helpers import strip_feedback_from_text

        return strip_feedback_from_text(text)

    @staticmethod
    def _split_by_sentences(text: str, target_words: int) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current: List[str] = []
        current_wc = 0

        for sentence in sentences:
            swc = len(sentence.split())
            if current_wc + swc > target_words and current:
                chunks.append(" ".join(current))
                current = [sentence]
                current_wc = swc
            else:
                current.append(sentence)
                current_wc += swc

        if current:
            chunks.append(" ".join(current))
        return chunks

    # ------------------------------------------------------------------
    # Stage 4: Write chunks
    # ------------------------------------------------------------------

    def _write_chunks(self, chunks: List[Dict[str, Any]]):
        jsonl_path = self.corpus_dir / "chunks.jsonl"
        json_path = self.corpus_dir / "chunks.json"

        # opt-in chunk validation against
        # schemas/knowledge/chunk_v4.schema.json. Gated by
        # TRAINFORGE_VALIDATE_CHUNKS=true for fail-closed behavior; default
        # is warn-log so existing pipelines don't break when the schema lands.
        strict = os.getenv("TRAINFORGE_VALIDATE_CHUNKS", "").lower() == "true"
        validation_errors: List[str] = []
        for i, chunk in enumerate(chunks):
            err = _validate_chunk(chunk)
            if err is None:
                continue
            chunk_id = chunk.get("id", f"<index {i}>")
            msg = f"Chunk {chunk_id}: {err}"
            if strict:
                validation_errors.append(msg)
            else:
                logger.warning("chunk_v4 validation: %s", msg)
        if validation_errors:
            preview = "; ".join(validation_errors[:5])
            suffix = " ..." if len(validation_errors) > 5 else ""
            raise ValueError(
                f"chunk_v4 validation failed for {len(validation_errors)} chunk(s): "
                f"{preview}{suffix}"
            )

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        # parity assertion. chunks.json must round-trip to the
        # exact same chunk list (ordered + content-equal) as chunks.jsonl.
        # Serialization parity protects against format-specific normalization
        # normalization re-emitted only the streaming format; we now read
        # both back from disk and fail loud on any divergence so future
        # patches can never silently regress one format.
        _assert_chunk_files_parity(jsonl_path, json_path)

        self.capture.log_decision(
            decision_type="chunk_serialization",
            decision=f"Write {len(chunks)} chunks to JSONL and JSON",
            rationale="JSONL format required for LibV2 streaming retrieval; JSON array for debugging and validation",
        )

    # ------------------------------------------------------------------
    # Stage 5: Generate metadata
    # ------------------------------------------------------------------

    def _auto_extract_topics(self) -> List[str]:
        """Extract topic tags from objectives chapter titles."""
        if not self.objectives:
            return []
        topics = []
        for ch in self.objectives.get("chapter_objectives", []):
            title = ch.get("chapter", "")
            # Strip "Week X-Y: " prefix and trailing preposition phrases
            cleaned = re.sub(r"^Week\s+\d+[-–]\d+:\s*", "", title)
            # Remove trailing prepositional phrases ("in Digital Environments", etc.)
            cleaned = re.sub(r"\s+(?:in|for)\s+.*$", "", cleaned, flags=re.IGNORECASE)
            if cleaned:
                tag = normalize_tag(cleaned)
                if tag and tag not in topics:
                    topics.append(tag)
        return topics

    def _auto_extract_subtopics(self, concept_graph: Dict[str, Any],
                                 exclude: List[str] = None, limit: int = 10) -> List[str]:
        """Extract subtopics from top concept graph nodes by frequency."""
        exclude_set = set(exclude or [])
        nodes = sorted(
            concept_graph.get("nodes", []),
            key=lambda n: n.get("frequency", 0),
            reverse=True,
        )
        subtopics = []
        for node in nodes:
            tag = node["id"]
            if tag not in exclude_set and tag not in subtopics:
                subtopics.append(tag)
            if len(subtopics) >= limit:
                break
        return subtopics

    def _generate_manifest(self, title: str,
                           concept_graph: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        description = ""
        if self.objectives:
            description = self.objectives.get("description", "")
        if not description:
            description = f"{title} - processed by Trainforge"

        # Detect section structure
        sections: List[str] = []
        if self.objectives:
            for ch in self.objectives.get("chapter_objectives", []):
                sections.append(ch.get("chapter", ""))

        # Auto-extract topics from objectives, subtopics from concept graph
        topics = self.topics or self._auto_extract_topics()
        subtopics = self._auto_extract_subtopics(
            concept_graph or {}, exclude=topics,
        )

        return {
            "course_id": self.course_code,
            "title": title,
            "description": description,
            "course_title": title,
            "sourceforge_version": "1.0",
            "chunk_schema_version": CHUNK_SCHEMA_VERSION,
            "export_timestamp": datetime.now().isoformat(),
            "source": {
                "type": "imscc",
                "path": str(self.imscc_path),
                "lms": "courseforge",
                "version": "1.3",
            },
            "classification": {
                "division": self.division,
                "primary_domain": self.domain,
                "secondary_domains": self.secondary_domains,
                "subdomains": self.subdomains,
                "topics": topics,
                "subtopics": subtopics,
            },
            "structure": {
                "total_sections": self.stats["sections_processed"],
                "sections": sections,
                "items_per_section": (
                    self.stats["modules_processed"] // max(self.stats["sections_processed"], 1)
                ),
            },
            "pedagogy": self._build_pedagogy_summary(),
            "processing": {
                "pipeline": "trainforge",
                "version": "1.0",
                "processed_date": datetime.now().isoformat(),
                "chunk_strategy": "pedagogical-units",
                "target_chunk_size": self.TARGET_CHUNK_SIZE,
                "chunk_schema_version": CHUNK_SCHEMA_VERSION,
            },
            "statistics": {
                "chunks": self.stats["total_chunks"],
                "total_words": self.stats["total_words"],
                "total_tokens": self.stats["total_tokens_estimate"],
                "concepts": len(self._all_concept_tags),
            },
        }

    def _generate_corpus_stats(self) -> Dict[str, Any]:
        total = self.stats["total_chunks"]
        stats: Dict[str, Any] = {
            "total_chunks": total,
            "total_words": self.stats["total_words"],
            "total_tokens_estimate": self.stats["total_tokens_estimate"],
            "avg_words_per_chunk": self.stats["total_words"] / total if total else 0,
            "chunk_type_distribution": dict(self.stats["chunk_types"]),
            "difficulty_distribution": dict(self.stats["difficulty_distribution"]),
            "modules_processed": self.stats["modules_processed"],
            "quizzes_processed": self.stats["quizzes_processed"],
            "sections_processed": self.stats["sections_processed"],
            "generated_at": datetime.now().isoformat(),
        }
        # Honest IRT difficulty-calibration scaffold: add the deterministic
        # difficulty-distribution descriptor + emit one difficulty_calibration
        # decision event per run. No-op + byte-identical when the flag is off.
        self._maybe_add_difficulty_descriptor(stats)
        return stats

    def _maybe_add_difficulty_descriptor(self, stats: Dict[str, Any]) -> None:
        """Add difficulty_distribution_descriptor + emit one decision event.

        Gated by TRAINFORGE_IRT_DIFFICULTY_SCAFFOLD; flag-off path untouched.
        """
        try:
            from lib.assessment.irt_difficulty import (
                DEFAULT_MIN_RESPONSES,
                describe_difficulty_distribution,
                irt_scaffold_enabled,
            )
            if not irt_scaffold_enabled():
                return
            counts = dict(self.stats["difficulty_distribution"])
            calibrated_map = self._get_irt_calibrated_map()
            total = self.stats["total_chunks"] or 0
            calibrated_fraction = (len(calibrated_map) / total) if total else 0.0
            descriptor = describe_difficulty_distribution(counts, calibrated_fraction)
            stats["difficulty_distribution_descriptor"] = descriptor
            try:
                self.capture.log_decision(
                    decision_type="difficulty_calibration",
                    decision=f"provenance={descriptor['provenance']}",
                    rationale=(
                        f"IRT difficulty scaffold: chunks_tagged={total}, "
                        f"calibrated_fraction={calibrated_fraction:.4f}, "
                        f"calibrated_item_count={len(calibrated_map)}, "
                        f"response_file_present={self._irt_response_file_present}, "
                        f"min_responses={DEFAULT_MIN_RESPONSES}, "
                        f"modal_level={descriptor['modal_level']}."
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("difficulty_calibration decision capture failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 — scaffold must never break stats
            logger.warning("IRT difficulty descriptor build failed: %s", exc)

    def _generate_concept_graph(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the domain concept co-occurrence graph.

        v0.1.0 semantics: nodes are unique concept tags that appear in 2+
        chunks, filtered to *exclude* pedagogy verbs and course-logistics
        tags. Edges carry ``relation_type = "co-occurs"`` — the only type
        produced today. A typed extractor (prerequisite / is-a / related-to)
        is reserved for v1.0 (see VERSIONING.md §4) and will write
        ``concept_graph_semantic.json`` alongside this file.
        """
        return self._build_tag_graph(
            chunks,
            exclude_tags=PEDAGOGY_TAG_SET | LOGISTICS_TAG_SET,
            graph_kind="concept",
        )

    def _build_misconceptions_for_graph(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Derive misconception entities from chunks for the semantic graph.

        The ``misconception-of`` inference rule expects a list of misconception
        dicts with stable ``id`` (``mc_[0-9a-f]{16}``) + optional ``concept_id``.
        Chunks carry misconceptions on their ``misconceptions`` list (populated
        from JSON-LD ``misconceptions[]`` during ``_chunk_content``).

        Concept-routing precedence:
          1. Explicit ``concept_id`` on the JSON-LD misconception entry.
          2. Token-overlap match against the chunk's ``concept_tags`` —
             pick the tag whose surface form (slug split on hyphens)
             shares the most tokens with the misconception's statement.
             Ties break by tag-list order.
          3. First concept tag (legacy fallback).

        The token-overlap path handles authored misconceptions without an
        explicit ``concept_id``. In that case the legacy first-tag heuristic
        can select an unrelated concept solely because of list order.
        Token-overlap closes that gap without requiring author-side changes.
        """
        from Trainforge.rag.typed_edge_inference import _make_concept_id

        course_id = getattr(self, "course_code", "") or ""
        entities: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for chunk in chunks:
            raw = chunk.get("misconceptions") or []
            if not raw:
                continue
            tags = [t for t in (chunk.get("concept_tags") or []) if t]
            for entry in raw:
                if isinstance(entry, dict):
                    statement = (entry.get("misconception") or "").strip()
                    correction = (entry.get("correction") or "").strip()
                    explicit_cid = (entry.get("concept_id") or "").strip() or None
                    # Bloom level (canonicalized lowercase in the
                    # html_content_parser misconception normalizer) now
                    # participates in the seed so Bloom-distinct
                    # misconceptions emit distinct IDs. Breaking change: old
                    # corpora re-chunked under this wave will see new
                    # misconception IDs (documented below).
                    # mirror the ``.strip().lower()`` normalization
                    # from ``preference_factory._misconception_id`` so the
                    # two hash call sites stay lock-step even when a direct
                    # chunk-construction path bypasses the html_content_parser
                    # normalizer and supplies mixed-case Bloom input.
                    bloom_level = (entry.get("bloom_level") or "").strip().lower()
                    cognitive_domain = (entry.get("cognitive_domain") or "").strip()
                elif isinstance(entry, str):
                    statement = entry.strip()
                    correction = ""
                    explicit_cid = None
                    bloom_level = ""
                    cognitive_domain = ""
                else:
                    continue
                if not statement:
                    continue
                # Content-hash ID per misconception.schema.json.
                # seed extended with bloom_level so two misconceptions
                # that share statement + correction text but target different
                # Bloom cognitive demands (e.g., apply-level vs analyze-level
                # misreading of the same concept) emit distinct IDs.
        # The bloom-less path keeps the two-field seed for stable IDs and
        # avoids adding an empty trailing field.
                # routed through the canonical helper so this site,
                # ``preference_factory._misconception_id``, and
                # ``pedagogy_graph_builder._mc_id`` are byte-equivalent.
                from lib.ontology.misconception_id import canonical_mc_id
                mc_id = canonical_mc_id(statement, correction, bloom_level)
                if mc_id in seen:
                    continue
                seen.add(mc_id)
                entity: Dict[str, Any] = {
                    "id": mc_id,
                    "misconception": statement,
                    "correction": correction or statement,
                }
                if bloom_level:
                    entity["bloom_level"] = bloom_level
                if cognitive_domain:
                    entity["cognitive_domain"] = cognitive_domain
                concept_id: Optional[str] = explicit_cid
                if not concept_id and tags:
                    # token-overlap match — pick the tag whose
                    # slug-derived tokens most overlap the misconception
                    # statement and returns the first tag when no tokens overlap.
                    routed_tag = _route_misconception_to_tag(statement, tags)
                    if routed_tag:
                        concept_id = _make_concept_id(routed_tag, course_id)
                if concept_id:
                    entity["concept_id"] = concept_id
                entities.append(entity)
        return entities

    def _build_questions_for_graph(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Derive question entities from assessment-item chunks.

        The ``assesses`` inference rule expects a list of question dicts with
        ``id`` + ``objective_id`` (and optional ``source_chunk_id``). For
        every chunk classified as an ``assessment_item`` that carries
        ``learning_outcome_refs``, emit one question entity per referenced
        objective so the rule can materialise ``question→LO`` edges. The
        The chunk ID doubles as ``source_chunk_id`` so
        ``TRAINFORGE_SOURCE_PROVENANCE`` can resolve evidence refs.
        """
        questions: List[Dict[str, Any]] = []
        for chunk in chunks:
            if chunk.get("chunk_type") != "assessment_item":
                continue
            chunk_id = chunk.get("id")
            if not chunk_id:
                continue
            refs = chunk.get("learning_outcome_refs") or []
            for ref in refs:
                if not ref:
                    continue
                # Deterministic question ID keyed off (chunk_id, objective).
                q_id = f"q_{chunk_id}_{ref}"
                questions.append({
                    "id": q_id,
                    "objective_id": ref,
                    "source_chunk_id": chunk_id,
                })
        return questions

    def _generate_semantic_concept_graph(
        self,
        chunks: List[Dict[str, Any]],
        course: Optional[Dict[str, Any]],
        concept_graph: Dict[str, Any],
        parsed_items: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build the typed-edge concept graph alongside ``concept_graph.json``.

        See ``Trainforge.rag.typed_edge_inference`` for rule details and
        precedence. The LLM path is opt-in via ``self.typed_edges_llm`` and
        has a deterministic fallback (no callable wired → rules only).

        The ``misconceptions=`` and ``questions=`` kwargs are derived from
        the chunk corpus so the ``misconception-of`` and ``assesses`` rules
        can fire. Passing ``None`` for either leaves those rule emitters
        inert.

        ``objectives_metadata`` is built from the parsed JSON-LD
        ``learningObjectives[]`` across every page so the
        ``targets_concept_from_lo`` rule can materialize
        ``targetedConcepts[]`` as typed ``targets-concept`` edges; it fires
        on empty input otherwise.

        Upstream-consumption short-circuit: when
        ``self.concept_graph_path`` points at a ``kind == "concept_semantic"``
        file (the genuine semantic graph emitted by the
        ``textbook_to_course::concept_extraction`` workflow phase), load and
        return it verbatim rather than re-invoking ``build_semantic_graph``.
        This mirrors the ``_generate_pedagogy_graph`` ST-13 short-circuit
        and keeps the ``concept_graph_sha256`` chain coherent end-to-end
        (the bytes the phase wrote equal the bytes the manifest validator
        re-hashes). Fail-soft: a missing / corrupt / wrong-kind file falls
        through to the in-process ``build_semantic_graph`` build so legacy
        corpora — and any stale phase-output handoff — degrade gracefully.
        """
        from Trainforge.rag import named_graph_writer
        from Trainforge.rag.typed_edge_inference import (
            build_semantic_graph,
            build_semantic_graph_with_dataset,
        )

        # short-circuit on an upstream ``kind: "concept_semantic"``
        # graph. Only the semantic kind is adopted here — a ``kind:
        # "pedagogy"`` file at the same path falls through (the
        # disambiguation guard) so the pedagogy artifact is never
        # mis-served as the semantic graph.
        upstream_path = getattr(self, "concept_graph_path", None)
        if upstream_path is not None:
            upstream_path = Path(upstream_path)
            if upstream_path.exists() and upstream_path.is_file():
                try:
                    upstream_graph = json.loads(
                        upstream_path.read_text(encoding="utf-8")
                    )
                except Exception as exc:  # noqa: BLE001 — fail-soft on parse error
                    logger.warning(
                        "Failed to load upstream semantic concept "
                        "graph at %s (%s); falling through to in-process "
                        "build_semantic_graph.",
                        upstream_path,
                        exc,
                    )
                else:
                    if (
                        isinstance(upstream_graph, dict)
                        and upstream_graph.get("kind") == "concept_semantic"
                    ):
                        logger.info(
                            "Consuming upstream semantic concept "
                            "graph from %s (nodes=%d, edges=%d); skipping "
                            "in-process build_semantic_graph rebuild.",
                            upstream_path,
                            len(upstream_graph.get("nodes") or []),
                            len(upstream_graph.get("edges") or []),
                        )
                        return upstream_graph
                    logger.warning(
                        "Upstream graph at %s is not a kind="
                        "'concept_semantic' dict (kind=%r); falling "
                        "through to in-process build_semantic_graph.",
                        upstream_path,
                        upstream_graph.get("kind")
                        if isinstance(upstream_graph, dict)
                        else type(upstream_graph).__name__,
                    )
            else:
                logger.warning(
                    "concept_graph_path %s does not exist or is "
                    "not a file; falling through to in-process "
                    "build_semantic_graph.",
                    upstream_path,
                )

        llm_callable = None
        if self.typed_edges_llm:
            # Placeholder hook — a future Trainforge LLM provider plugs in
            # here. Current behavior: log a non-decision and fall back to
            # rule-only output, keeping the flag semantically valid without
            # shipping a live LLM call path.
            try:
                self.capture.log_non_decision(
                    decision_type="typed_edge_inference",
                    default_value="rule_based_only",
                    rationale=(
                        "typed_edges_llm flag is on but no LLM callable is "
                        "wired into the Trainforge runtime yet; deterministic "
                        "rule-based output used."
                    ),
                )
            except Exception:  # pragma: no cover — capture is best-effort
                pass

        misconceptions = self._build_misconceptions_for_graph(chunks)
        questions = self._build_questions_for_graph(chunks)
        objectives_metadata = self._build_objectives_metadata_for_graph(
            parsed_items or []
        )

        # When TRAINFORGE_EMIT_TRIG is on, additionally compose an
        # rdflib.Dataset of per-rule named graphs and write a sibling
        # concept_graph_semantic.trig file. JSON output is byte-identical
        # whether the flag is on or off — the named-graph emit is purely
        # additive provenance metadata.
        if named_graph_writer.EMIT_TRIG:
            json_dict, dataset = build_semantic_graph_with_dataset(
                chunks=chunks,
                course=course,
                concept_graph=concept_graph,
                llm_enabled=self.typed_edges_llm and llm_callable is not None,
                llm_callable=llm_callable,
                decision_capture=self.capture,
                misconceptions=misconceptions or None,
                questions=questions or None,
                objectives_metadata=objectives_metadata or None,
                emit_trig=True,
            )
            # Stash the dataset for the metadata writer to serialise
            # alongside concept_graph_semantic.json.
            self._semantic_graph_trig_dataset = dataset
            return json_dict

        return build_semantic_graph(
            chunks=chunks,
            course=course,
            concept_graph=concept_graph,
            llm_enabled=self.typed_edges_llm and llm_callable is not None,
            llm_callable=llm_callable,
            decision_capture=self.capture,
            misconceptions=misconceptions or None,
            questions=questions or None,
            objectives_metadata=objectives_metadata or None,
        )

    def _build_objectives_metadata_for_graph(
        self, parsed_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Derive ``objectives_metadata`` for semantic graph construction.

        The ``targets_concept_from_lo`` rule expects a list of LO
        dicts shaped like Courseforge's JSON-LD ``learningObjectives[]``
        emit — each entry at minimum carrying ``id`` and an optional
        ``targetedConcepts[]`` (list of ``{concept, bloomLevel}`` dicts).
        The rule lowercases LO IDs itself and validates the bloom level
        against the canonical 6-value enum.

        We iterate every parsed page and prefer the raw JSON-LD payload
        (``courseforge_metadata.learningObjectives``) so the shape lands
        on the rule exactly as emitted. When no JSON-LD is available
        (legacy corpora / non-Courseforge IMSCC) we reconstruct the shape
        from ``html_content_parser.LearningObjective.targeted_concepts``
        — which is already a snake_case list — by translating back to
        camelCase for the rule.

        Deduplicated by LO ID so a page appearing twice (or cross-page
        duplicates) doesn't inflate the edge count downstream — the rule
        itself dedups by (lo_id, concept_id) inside each LO, but a clean
        input list also avoids log spam.

        Two-pass dedup keeps JSON-LD entries authoritative before reconstructed
        per item, which meant an early item's dataclass fallback (typically
        with empty ``targetedConcepts``) could shadow a later item's richer
        JSON-LD payload for the same LO. The docstring claimed JSON-LD was
        preferred but the per-item interleaving broke that guarantee for
        mixed corpora (legacy pages + JSON-LD pages sharing LO IDs). Now
        we do Path 1 across *all* items first, then Path 2 fills any LOs
        still absent — the "JSON-LD preferred" promise actually holds.

        Within Path 1, the first rich JSON-LD payload encountered for a
        given ``id`` wins; duplicate JSON-LD entries for the same id on
        later items are dropped (no deep-merge). Likewise within Path 2.
        This keeps dedup deterministic and list-order-driven; if a future
        wave needs cross-page concept-union it should be a new helper, not
        a silent behavior change here.
        """
        by_id: Dict[str, Dict[str, Any]] = {}

        # Pass 1: direct JSON-LD payload across every item — preferred
        # because it's the exact emit shape the rule expects.
        for item in parsed_items:
            cf_meta = item.get("courseforge_metadata") or {}
            for raw_lo in cf_meta.get("learningObjectives") or []:
                if not isinstance(raw_lo, dict):
                    continue
                lo_id = raw_lo.get("id")
                if not isinstance(lo_id, str) or not lo_id:
                    continue
                if lo_id in by_id:
                    continue
                # Shallow copy so we don't mutate the parsed item.
                by_id[lo_id] = dict(raw_lo)

        # Pass 2: reconstruct from parsed LearningObjective dataclass for
        # LOs not covered by Pass 1 (legacy corpora / non-Courseforge IMSCC).
        for item in parsed_items:
            for parsed_lo in item.get("learning_objectives") or []:
                # Dataclass or dict — support both.
                lo_id = getattr(parsed_lo, "id", None)
                if lo_id is None and isinstance(parsed_lo, dict):
                    lo_id = parsed_lo.get("id")
                if not isinstance(lo_id, str) or not lo_id:
                    continue
                if lo_id in by_id:
                    continue
                targeted = getattr(parsed_lo, "targeted_concepts", None)
                if targeted is None and isinstance(parsed_lo, dict):
                    targeted = parsed_lo.get("targeted_concepts")
                targeted = targeted or []
                # Back-translate snake_case → camelCase for the rule.
                rule_shape_targets = [
                    {
                        "concept": t.get("concept"),
                        "bloomLevel": t.get("bloom_level"),
                    }
                    for t in targeted
                    if isinstance(t, dict)
                ]
                by_id[lo_id] = {
                    "id": lo_id,
                    "targetedConcepts": rule_shape_targets,
                }
        return list(by_id.values())

    def _generate_pedagogy_graph(
        self,
        chunks: List[Dict[str, Any]],
        concept_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the typed pedagogical graph.

        Runs ``Trainforge.pedagogy_graph_builder.build_pedagogy_graph`` at
        emit time so fresh archives carry the full relation set —
        ``derived_from_objective`` / ``concept_supports_outcome`` /
        ``assessment_validates_outcome`` / ``chunk_at_difficulty``, plus
        ``teaches`` / ``assesses`` / ``practices`` / ``exemplifies`` /
        ``prerequisite_of`` / ``interferes_with`` /
        ``belongs_to_module`` / ``supports_outcome`` /
        ``at_bloom_level`` / ``follows``. A bare co-occurrence graph over
        pedagogy/logistics tags is not a substitute: on real corpora it
        degenerates to 1 node / 0 edges.

        Inputs:

        * ``chunks``        — same chunk list emitted in Stage 4. The
                              builder reads ``concept_tags`` /
                              ``learning_outcome_refs`` /
                              ``misconceptions`` / ``chunk_type`` /
                              ``difficulty`` / ``source.module_id`` /
                              ``source.item_path``.
        * ``concept_graph`` — concept_graph.json shape, used to source
                              the ``concept_classes`` map
                              classifier output stamped on every node).
                              When omitted the builder treats every
                              concept as DomainConcept-default
                              (permissive mode); current processing always
                              passes it from the caller.

        Objectives source: prefers ``self.objectives`` (already loaded
        by the constructor, including synthesized
        fallback). When unavailable we still call the builder with an
        empty objectives dict so the four DifficultyLevel + six
        BloomLevel typed nodes always emit (the builder degrades
        gracefully).

        Fail-soft: any exception inside the builder is logged and the
        method returns an empty graph shell so the rest of metadata
        emit proceeds. Mirrors the ``semantic_graph`` fail-soft pattern.

        When ``self.concept_graph_path`` is set, the
        upstream ``concept_extraction`` workflow phase ran and wrote
        ``LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json``),
        load the pre-built pedagogy graph from disk instead of
        re-invoking ``build_pedagogy_graph``. The upstream phase
        runs the same builder, so the rebuild here is purely
        redundant work — and re-running it inside the IMSCC-driven
        path can produce a slightly different graph because the
        chunk projection differs (DART staging sections vs. v4
        IMSCC chunks). Reading the upstream artifact pins both
        callers to the same graph. Falls through to the in-process
        build path on (a) ``concept_graph_path is None`` (legacy
        corpora), (b) the path doesn't exist, or (c) the file is
        unreadable — so a stale phase-output handoff degrades
        gracefully rather than crashing the run.

        The ``concept_extraction`` phase emits
        a genuine ``kind: "concept_semantic"`` graph at that path, NOT a
        pedagogy graph. The short-circuit therefore adopts the upstream
        file ONLY when ``kind == "pedagogy"``; a semantic-kind file falls
        through to the in-process ``build_pedagogy_graph`` so the
        now-semantic artifact is never mis-adopted into the pedagogy slot.
        """
        from datetime import datetime as _dt

        # short-circuit on upstream pedagogy graph.
        # only a ``kind: "pedagogy"`` file is adopted here.
        upstream_path = getattr(self, "concept_graph_path", None)
        if upstream_path is not None:
            upstream_path = Path(upstream_path)
            if upstream_path.exists() and upstream_path.is_file():
                try:
                    upstream_graph = json.loads(
                        upstream_path.read_text(encoding="utf-8")
                    )
                except Exception as exc:  # noqa: BLE001 — fail-soft on parse error
                    logger.warning(
                        "Failed to load upstream pedagogy "
                        "graph at %s (%s); falling through to in-process "
                        "build.",
                        upstream_path,
                        exc,
                    )
                else:
                    if not isinstance(upstream_graph, dict):
                        logger.warning(
                            "Upstream pedagogy graph at %s "
                            "is not a dict (%s); falling through to "
                            "in-process build_pedagogy_graph.",
                            upstream_path,
                            type(upstream_graph).__name__,
                        )
                    elif upstream_graph.get("kind") == "pedagogy":
                        logger.info(
                            "Consuming upstream pedagogy "
                            "graph from %s (nodes=%d, edges=%d); skipping "
                            "in-process build_pedagogy_graph rebuild.",
                            upstream_path,
                            len(upstream_graph.get("nodes") or []),
                            len(upstream_graph.get("edges") or []),
                        )
                        return upstream_graph
                    else:
                        logger.warning(
                            "Upstream graph at %s is not a "
                            "kind='pedagogy' dict (kind=%r); falling "
                            "through to in-process build_pedagogy_graph.",
                            upstream_path,
                            upstream_graph.get("kind"),
                        )
            else:
                logger.warning(
                    "concept_graph_path %s does not exist "
                    "or is not a file; falling through to in-process "
                    "build.",
                    upstream_path,
                )

        try:
            from Trainforge.pedagogy_graph_builder import build_pedagogy_graph
        except Exception as exc:  # pragma: no cover — import-failure guard
            logger.warning(
                "pedagogy_graph_builder import failed; emitting "
                "empty graph. Cause: %s",
                exc,
            )
            return {
                "kind": "pedagogy",
                "nodes": [],
                "edges": [],
                "generated_at": _dt.now().isoformat(),
                "stats": {
                    "node_count": 0,
                    "edge_count": 0,
                    "nodes_by_class": {},
                    "edges_by_relation": {},
                },
            }

        # Source concept_classes from the concept_graph nodes. Each
        # node carries a ``class`` field (wiring inside
        # ``_build_tag_graph``). The builder's ``prerequisite_of`` /
        # ``interferes_with`` / ``concept_supports_outcome`` filters
        # consult this map to drop pedagogical / assessment-option /
        # low-signal endpoints. ``concept_graph`` may be None when an
        # Callers may omit the concept graph; treat that as
        # permissive-mode (every concept defaults to DomainConcept).
        concept_classes: Dict[str, str] = {}
        if isinstance(concept_graph, dict):
            for node in concept_graph.get("nodes") or []:
                if not isinstance(node, dict):
                    continue
                raw_id = node.get("id")
                klass = node.get("class")
                if not isinstance(raw_id, str) or not raw_id:
                    continue
                if not isinstance(klass, str) or not klass:
                    continue
                # Strip the ``course_id:`` prefix when SCOPE_CONCEPT_IDS
                # is on so the builder sees the bare slug it was
                # designed against (mirrors the helper in
                # scripts/archive/wave75_classify_concept_graph.py). Also key
                # by the full ID so the builder's lookups land
                # regardless of scoping mode.
                slug = raw_id.split(":", 1)[-1]
                concept_classes[slug] = klass
                concept_classes[raw_id] = klass

        # ``self.objectives`` may be missing (legacy
        # ``CourseProcessor.__new__`` callers in existing tests bypass
        # the constructor) or None; treat both as empty so the builder
        # still emits BloomLevel + DifficultyLevel typed nodes.
        raw_objectives = getattr(self, "objectives", None)
        objectives = raw_objectives if isinstance(raw_objectives, dict) else {}

        try:
            return build_pedagogy_graph(
                chunks,
                objectives=objectives,
                course_id=getattr(self, "course_code", None) or None,
                concept_classes=concept_classes or None,
            )
        except Exception as exc:  # noqa: BLE001 — fail-soft on builder error
            logger.warning(
                "build_pedagogy_graph failed; emitting empty "
                "graph. Cause: %s",
                exc,
            )
            return {
                "kind": "pedagogy",
                "nodes": [],
                "edges": [],
                "generated_at": _dt.now().isoformat(),
                "stats": {
                    "node_count": 0,
                    "edge_count": 0,
                    "nodes_by_class": {},
                    "edges_by_relation": {},
                },
            }

    def _build_tag_graph(
        self,
        chunks: List[Dict[str, Any]],
        *,
        include_tags: Optional[Set[str]] = None,
        exclude_tags: Optional[Set[str]] = None,
        graph_kind: str = "concept",
    ) -> Dict[str, Any]:
        """Build the co-occurrence concept graph from chunk ``concept_tags``.

        This method delegates to
        ``lib.ontology.cooccurrence_graph.build_cooccurrence_graph`` — the
        instance-free helper that is the single source of truth for the
        co-occurrence build. The IMSCC path (this method, via
        ``_generate_concept_graph``) and the DART
        ``textbook_to_course::concept_extraction`` phase
        (``MCP/tools/pipeline_tools.py::_run_concept_extraction``) both call
        the same helper, so they emit byte-equivalent graphs (modulo the
        wall-clock ``generated_at``) from identical chunk inputs.

        ``course_code`` may be unset when the graph builder is called on a
        bare processor (e.g. unit tests using ``__new__``); fall back to an
        empty string → the helper treats empty as "no course_id" and emits
        flat slugs even under ``TRAINFORGE_SCOPE_CONCEPT_IDS``.
        """
        from lib.ontology.cooccurrence_graph import build_cooccurrence_graph

        course_id = getattr(self, "course_code", "") or ""
        return build_cooccurrence_graph(
            chunks,
            course_id,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            graph_kind=graph_kind,
        )

    def _generate_quality_report(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(chunks) or 1

        in_range = sum(1 for c in chunks if self.MIN_CHUNK_SIZE <= c["word_count"] <= self.MAX_CHUNK_SIZE)
        size_compliance = in_range / total

        with_tags = sum(1 for c in chunks if len(c.get("concept_tags", [])) >= 2)
        tag_coverage = with_tags / total

        # Structural integrity: chunk HTML must parse with balanced tags.
        balance_violations = [
            {"chunk_id": c["id"], "unclosed_tags": self._unclosed_tags(c.get("html", ""))}
            for c in chunks
            if not self._html_is_well_formed(c.get("html", ""))
        ]
        well_formed = total - len(balance_violations)
        html_preservation = well_formed / total

        with_bloom = sum(1 for c in chunks if c.get("bloom_level"))
        bloom_coverage = with_bloom / total

        # Referential integrity: count only refs that resolve to course.json IDs.
        valid_ids = self._valid_outcome_ids or set()
        lo_coverage = self._resolving_lo_coverage(chunks, valid_ids)
        broken_refs = self._collect_broken_refs(chunks, valid_ids)
        # Reverse coverage: which declared outcomes have ZERO resolving chunks?
        # This is the symmetric complement of learning_outcome_coverage and
        # catches content-generation gaps that the chunk-ratio metric misses.
        # case-insensitive comparison, skipping None/empty refs.
        referenced_ids: Set[str] = set()
        for c in chunks:
            for r in c.get("learning_outcome_refs", []) or []:
                if r is None:
                    continue
                r_str = r if isinstance(r, str) else str(r)
                if not r_str.strip():
                    continue
                lowered = r_str.lower()
                if lowered in valid_ids:
                    referenced_ids.add(lowered)
        uncovered_outcomes = sorted(valid_ids - referenced_ids)
        outcome_reverse_coverage = (
            (len(valid_ids) - len(uncovered_outcomes)) / len(valid_ids)
            if valid_ids else 1.0
        )

        # Content sanity: boilerplate contamination + factual flags + follows_chunk scope.
        footer_rate = contamination_rate(chunks, self._boilerplate_spans) if self._boilerplate_spans else 0.0
        boundary_violations = self._follows_chunk_violations(chunks)
        factual_flags = list(self._factual_flags)

        # ------------------------------------------------------------------
        # Flow metrics (METRICS_SEMANTIC_VERSION 4). Surface silent metadata
        # drops between parser -> chunk that current coverage metrics don't
        # reveal. See docs/operations/flow-metrics.md for full methodology.
        # ------------------------------------------------------------------
        flow_metrics, flow_methodology, flow_integrity = self._compute_flow_metrics(chunks)

        overall = (size_compliance * 0.25 + tag_coverage * 0.2 +
                   html_preservation * 0.2 + bloom_coverage * 0.2 + lo_coverage * 0.15)

        issues: List[str] = []
        recommendations: List[str] = []

        if size_compliance < 0.8:
            issues.append("Chunk size compliance below 80%")
            recommendations.append("Review chunking thresholds")
        if tag_coverage < 0.7:
            issues.append("Concept tag coverage below 70%")
            recommendations.append("Enhance concept extraction")
        if bloom_coverage < 0.9:
            issues.append(f"Bloom level coverage {bloom_coverage:.0%} — below 90% threshold")
        if lo_coverage < 0.8:
            issues.append(f"Learning outcome coverage {lo_coverage:.0%} — below 80% threshold")
        if valid_ids and outcome_reverse_coverage < 0.9:
            issues.append(
                f"{len(uncovered_outcomes)} learning outcomes have zero resolving chunks: "
                + ", ".join(uncovered_outcomes)
            )
        if html_preservation < 1.0:
            issues.append(f"HTML balance violations in {len(balance_violations)} chunks")
        if footer_rate > 0.05:
            issues.append(f"Footer contamination rate {footer_rate:.0%} — above 5% threshold")
        if broken_refs:
            issues.append(f"{len(broken_refs)} unresolvable learning_outcome_refs")
        if boundary_violations:
            issues.append(f"{len(boundary_violations)} follows_chunk cross-lesson links")
        if factual_flags:
            issues.append(f"{len(factual_flags)} factual-claim flags")
        if not issues:
            recommendations.append("Corpus meets all quality thresholds")

        metrics_block: Dict[str, Any] = {
            "chunk_size_compliance": round(size_compliance, 3),
            "concept_tag_coverage": round(tag_coverage, 3),
            "html_preservation_rate": round(html_preservation, 3),
            "bloom_level_coverage": round(bloom_coverage, 3),
            "learning_outcome_coverage": round(lo_coverage, 3),
            "outcome_reverse_coverage": round(outcome_reverse_coverage, 3),
            "footer_contamination_rate": round(footer_rate, 3),
            "follows_chunk_boundary_violations": len(boundary_violations),
            "avg_chunk_size_words": round(self.stats["total_words"] / total, 1),
        }
        metrics_block.update(flow_metrics)

        # package_completeness — flat mean of the five
        # enrichment coverage fractions. Surfaced as its own top-level key
        # (NOT inside `metrics`, NOT weighted into `overall_quality_score`)
        # so a consumer can read one number without cross-referencing five.
        package_completeness_components = (
            round(bloom_coverage, 3),
            flow_metrics.get("content_type_label_coverage", 0.0),
            flow_metrics.get("key_terms_coverage", 0.0),
            flow_metrics.get("misconceptions_present_rate", 0.0),
            flow_metrics.get("interactive_components_rate", 0.0),
        )
        package_completeness = round(
            sum(package_completeness_components) / len(package_completeness_components),
            3,
        )

        methodology_block: Dict[str, str] = {
            "html_preservation_rate": (
                "Fraction of chunks whose HTML parses with balanced open/close tags "
                "(stdlib html.parser.HTMLParser). Self-closing and void elements are "
                "not counted as needing close tags."
            ),
            "learning_outcome_coverage": (
                "Fraction of chunks that reference at least one outcome ID that "
                "resolves to course.json (referential integrity, not field presence)."
            ),
            "outcome_reverse_coverage": (
                "Fraction of declared course.json outcomes that have at least one "
                "chunk referencing them (catches content-generation gaps where whole "
                "outcomes are orphaned, which the chunk-ratio coverage misses)."
            ),
            "footer_contamination_rate": (
                "Fraction of chunks whose text still contains a detected corpus-wide "
                "repeated n-gram (likely footer/template-chrome that escaped stripping)."
            ),
            "follows_chunk_boundary_violations": (
                "Count of non-null follows_chunk links that cross lesson boundaries."
            ),
            "package_completeness": (
                "Flat mean of bloom_level_coverage, content_type_label_coverage, "
                "key_terms_coverage, misconceptions_present_rate, and "
                "interactive_components_rate. Answers: of the metadata this "
                "package claims to provide, how much actually landed. Not a "
                "weighted quality score — a flat completeness indicator. "
                "Emitted at top level (sibling of overall_quality_score), NOT "
                "inside `metrics`, and NOT weighted into overall_quality_score."
            ),
        }
        methodology_block.update(flow_methodology)

        integrity_block: Dict[str, Any] = {
            "broken_refs": broken_refs,
            "html_balance_violations": balance_violations,
            "follows_chunk_boundary_violations": boundary_violations,
            "factual_inconsistency_flags": factual_flags,
            "uncovered_outcomes": uncovered_outcomes,
        }
        integrity_block.update(flow_integrity)

        return {
            "metrics_semantic_version": METRICS_SEMANTIC_VERSION,
            "overall_quality_score": round(overall, 3),
            "package_completeness": package_completeness,
            "metrics": metrics_block,
            "methodology": methodology_block,
            "integrity": integrity_block,
            "validation": {"passed": overall >= 0.75 and not broken_refs, "issues": issues},
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Flow metrics (METRICS_SEMANTIC_VERSION 4)
    # ------------------------------------------------------------------

    def _compute_flow_metrics(
        self, chunks: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, List[str]]]:
        """Compute the five flow metrics that surface silent metadata drops.

        Returns ``(metrics, methodology, integrity)`` — three dicts to be
        merged into the corresponding blocks in ``_generate_quality_report``.

        Every metric is a ratio in ``[0.0, 1.0]`` with an ``int/int`` numerator
        and denominator so a failing flow is distinguishable from an absent
        upstream (denominator=0 ⇒ ratio=0.0 and the methodology string calls
        out the caveat).

        See ``docs/operations/flow-metrics.md`` for the full explanation of
        what each metric catches and how to read its value.
        """
        total = len(chunks) or 1
        chunk_total = len(chunks)  # real zero-aware total

        # 1. content_type_label_coverage
        with_label = sum(1 for c in chunks if c.get("content_type_label"))
        content_type_label_coverage = with_label / total

        # 2. key_terms_coverage
        with_key_terms = sum(1 for c in chunks if c.get("key_terms"))
        key_terms_coverage = with_key_terms / total

        # 3. key_terms_with_definitions_rate
        total_key_terms = 0
        terms_with_def = 0
        chunks_with_empty_definitions: List[str] = []
        for c in chunks:
            kts = c.get("key_terms") or []
            missing_def_in_this_chunk = False
            for kt in kts:
                if not isinstance(kt, dict):
                    continue
                total_key_terms += 1
                if (kt.get("definition") or "").strip():
                    terms_with_def += 1
                else:
                    missing_def_in_this_chunk = True
            if missing_def_in_this_chunk:
                chunks_with_empty_definitions.append(c["id"])
        if total_key_terms > 0:
            key_terms_with_definitions_rate = terms_with_def / total_key_terms
        else:
            key_terms_with_definitions_rate = 0.0

        # 4. misconceptions_present_rate
        # Denominator: chunks whose parent page had ≥1 misconception in JSON-LD.
        # This threading is populated in _chunk_content. When _chunk_content
        # was bypassed (e.g. unit tests that call _generate_quality_report
        # directly with hand-built chunks) the set is empty — in that case
        # we fall back to all chunks as the denominator so the metric
        # still reports something sensible.
        pages_with_mis = getattr(self, "_pages_with_misconceptions", None) or set()
        if pages_with_mis:
            eligible = [
                c for c in chunks
                if (c.get("source") or {}).get("lesson_id") in pages_with_mis
            ]
            mis_denom_label = "pages_with_json_ld_misconceptions"
        else:
            eligible = list(chunks)
            mis_denom_label = "all_chunks_fallback"
        mis_denom = len(eligible) or 1
        chunks_missing_misconceptions: List[str] = []
        chunks_with_mis = 0
        for c in eligible:
            if c.get("misconceptions"):
                chunks_with_mis += 1
            else:
                chunks_missing_misconceptions.append(c["id"])
        misconceptions_present_rate = chunks_with_mis / mis_denom if eligible else 0.0

        # 5. interactive_components_rate
        # Interactive components are NOT threaded onto chunks today (they live
        # on parsed_items only — see FOLLOWUP-WORKER-B-1). We fall back to
        # regex-detecting the same COMPONENT_PATTERNS the parser uses against
        # each chunk's own HTML, so the metric still reports flow without
        # requiring a chunk-schema change.
        from Trainforge.parsers.html_content_parser import HTMLContentParser
        patterns = HTMLContentParser.COMPONENT_PATTERNS
        compiled = [re.compile(p, re.IGNORECASE) for p in patterns.values()]
        with_component = 0
        for c in chunks:
            html = c.get("html", "") or ""
            if any(rx.search(html) for rx in compiled):
                with_component += 1
        interactive_components_rate = with_component / total

        metrics: Dict[str, float] = {
            "content_type_label_coverage": round(content_type_label_coverage, 3),
            "key_terms_coverage": round(key_terms_coverage, 3),
            "key_terms_with_definitions_rate": round(key_terms_with_definitions_rate, 3),
            "misconceptions_present_rate": round(misconceptions_present_rate, 3),
            "interactive_components_rate": round(interactive_components_rate, 3),
        }

        methodology: Dict[str, str] = {
            "content_type_label_coverage": (
                "Fraction of chunks carrying a non-empty content_type_label "
                "(e.g. explanation, example, procedure). Catches silent drops "
                "of JSON-LD / data-cf-content-type metadata between the parser "
                "and _create_chunk."
            ),
            "key_terms_coverage": (
                "Fraction of chunks with at least one key_terms entry. Catches "
                "silent drops of JSON-LD keyTerms / data-cf-key-terms between "
                "the parser and _create_chunk."
            ),
            "key_terms_with_definitions_rate": (
                "Across every key_terms entry on every chunk, the fraction "
                "whose definition field is non-empty. Denominator is the total "
                "count of key_terms entries across all chunks, not the chunk "
                "count. Catches the fallback path where data-cf-key-terms "
                "yields term strings but no definitions."
            ),
            "misconceptions_present_rate": (
                "Fraction of chunks carrying at least one misconception entry. "
                f"Denominator: {mis_denom_label}. When the parser found any "
                "misconceptions in the JSON-LD, the denominator is the chunks "
                "from those pages; when no misconceptions were found anywhere, "
                "the denominator falls back to all chunks and the metric is 0.0."
            ),
            "interactive_components_rate": (
                "Fraction of chunks whose HTML matches one of the parser's "
                "COMPONENT_PATTERNS (flip-card, accordion, tabs, callout, "
                "knowledge-check, activity-card). Interactive components are "
                "not yet threaded onto chunks as a first-class field — this "
                "regex fallback is intentional (FOLLOWUP-WORKER-B-1) and will "
                "be revisited when chunk-schema provenance exposes it directly."
            ),
        }

        integrity: Dict[str, List[str]] = {
            "chunks_with_empty_definitions": chunks_with_empty_definitions,
            "chunks_missing_misconceptions": chunks_missing_misconceptions,
        }

        # Silence unused-variable hint when chunks is empty.
        _ = chunk_total

        return metrics, methodology, integrity

    # ------------------------------------------------------------------
    # Integrity helpers (used by quality report + tests)
    # ------------------------------------------------------------------

    @staticmethod
    def _html_is_well_formed(html: str) -> bool:
        """True iff ``html`` is balanced (every opened non-void tag closes in order).

        Empty or whitespace-only HTML
        is ``True`` (well-formed by vacuity) rather than ``False``.
        Returning False here would conflate "no HTML to check" with
        "balance violation" and inflate the ``html_balance_violations``
        count in ``quality_report.json``. Empty-html (text-only chunks
        where the renderer dropped HTML) deserves its own metric, not
        miscategorization as unbalanced.
        """
        if not html or not html.strip():
            return True
        return _BalanceChecker.check(html)

    @staticmethod
    def _unclosed_tags(html: str) -> List[str]:
        if not html:
            return []
        return _BalanceChecker.unclosed(html)

    @staticmethod
    def _collect_broken_refs(
        chunks: List[Dict[str, Any]],
        valid_outcome_ids: Set[str],
    ) -> List[Dict[str, str]]:
        """List learning_outcome_refs that don't resolve.

        Policy:
          1. Comparison is case-insensitive. ``valid_outcome_ids`` is
             expected to be a lowercase set; chunk refs are lowercased
             before lookup. Mirrors the lowercase emit in
             ``_build_course_json`` and matches LibV2's case-insensitive
             join in ``retrieval_scoring.py``.
          2. ``None`` and empty-string refs are skipped silently — they
             carry no information, so they aren't ``broken``, just absent.
          3. Non-string refs are coerced via ``str()``; the exact
             malformed value (e.g. comma-joined ``"co-01,co-02"``) is
             preserved verbatim in the report so a reviewer can see the
             original input shape.
        """
        broken: List[Dict[str, str]] = []
        for c in chunks:
            for ref in c.get("learning_outcome_refs", []):
                if ref is None:
                    continue
                ref_str = ref if isinstance(ref, str) else str(ref)
                if not ref_str.strip():
                    continue
                if ref_str.lower() not in valid_outcome_ids:
                    broken.append({"chunk_id": c["id"], "ref": ref_str})
        return broken

    @staticmethod
    def _resolving_lo_coverage(
        chunks: List[Dict[str, Any]],
        valid_outcome_ids: Set[str],
    ) -> float:
        """Fraction of chunks with at least one ref that resolves.

        Comparison is case-insensitive and mirrors the policy in
        :meth:`_collect_broken_refs`.
        """
        total = len(chunks) or 1

        def _resolves(c: Dict[str, Any]) -> bool:
            for ref in c.get("learning_outcome_refs", []) or []:
                if ref is None:
                    continue
                ref_str = ref if isinstance(ref, str) else str(ref)
                if ref_str.lower() in valid_outcome_ids:
                    return True
            return False

        resolving = sum(1 for c in chunks if _resolves(c))
        return resolving / total

    @staticmethod
    def _follows_chunk_violations(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        by_id = {c["id"]: c for c in chunks}
        violations: List[Dict[str, str]] = []
        for c in chunks:
            follows = c.get("follows_chunk")
            if not follows:
                continue
            prev = by_id.get(follows)
            if prev is None:
                violations.append({"chunk_id": c["id"], "follows_chunk": follows, "reason": "dangling"})
                continue
            if c.get("source", {}).get("lesson_id") != prev.get("source", {}).get("lesson_id"):
                violations.append({
                    "chunk_id": c["id"],
                    "follows_chunk": follows,
                    "reason": "cross_lesson",
                })
        return violations

    # ------------------------------------------------------------------
    # Pre-chunking helpers (called from process())
    # ------------------------------------------------------------------

    def _detect_corpus_boilerplate(self, parsed_items: List[Dict[str, Any]]) -> List[str]:
        """Run N-gram frequency sweep across every page's raw HTML to find
        repeated spans (footers / template chrome) worth stripping.

        Returns the list of span strings; an empty list when the corpus is
        too small or no candidate exceeds the min-doc-frac threshold.
        """
        docs = [item.get("raw_html", "") for item in parsed_items if item.get("raw_html")]
        if len(docs) < 3:
            return []
        # Operate on plain text so we don't match span-containing tag noise.
        plain_docs = [self._extract_plain_text(d) for d in docs]
        spans = detect_repeated_ngrams(
            plain_docs,
            n=self._boilerplate_config.min_ngram_tokens,
            min_doc_frac=self._boilerplate_config.min_doc_frac,
        )
        if spans:
            self.capture.log_decision(
                decision_type="boilerplate_strip",
                decision=f"Detected {len(spans)} repeated span(s); will strip before chunking",
                rationale=(
                    "Corpus-wide n-gram frequency above threshold indicates "
                    "template chrome or footer contamination that would otherwise "
                    "bleed into every chunk's embedding."
                ),
            )
        return spans

    def _build_valid_outcome_ids(self) -> Set[str]:
        """Collect every outcome ID the chunks are allowed to reference.

        Course-level IDs (``co-*``, ``to-*``) are always included. Week-scoped
        IDs (``w01-co-*``) are included when the objectives file carries a
        ``week_scoped_ids`` list per outcome — this is the dual-emission
        contract from §2.1. Legacy objective files without ``week_scoped_ids``
        yield a set that only resolves flat IDs; chunks that reference
        week-scoped forms will surface as broken_refs in the quality report.

        Supports both objective-file schemas:
          (a) ``terminal_outcomes`` + ``component_objectives``
              (the flat shape produced by ``_emit_objectives_artifact``).
          (b) ``terminal_objectives`` +
              ``chapter_objectives`` (chapter is either a flat list of LO
              dicts or a nested ``[{chapter, objectives:[...]}]`` shape).

        Falls back to ``course.json::learning_outcomes[].id`` when no
        objectives file is loaded — covers archives that pre-date the
        objectives.json emit.

        Comparison is case-insensitive: every ID is lowercased before being
        added to the set, mirroring the lowercase emit in
        ``_build_course_json``.
        """
        ids: Set[str] = set()

        terminal_list = []
        component_list = []
        if self.objectives:
            terminal_list = (
                self.objectives.get("terminal_objectives")
                or self.objectives.get("terminal_outcomes")
                or []
            )
            component_list = (
                self.objectives.get("chapter_objectives")
                or self.objectives.get("component_objectives")
                or []
            )

        for to in terminal_list:
            if not isinstance(to, dict):
                continue
            obj_id = (to.get("id") or "").lower()
            if obj_id:
                ids.add(obj_id)
            for ws in to.get("week_scoped_ids", []) or []:
                if ws:
                    ids.add(str(ws).lower())

        for ch in component_list:
            if not isinstance(ch, dict):
                continue
            # Nested shape: {"chapter": "...", "objectives": [{...}, ...]}
            # Flat shape:   {"id": "co-01", "parent_terminal": "to-01", ...}
            if "objectives" in ch and isinstance(ch.get("objectives"), list):
                inner = ch["objectives"]
            else:
                inner = [ch]
            for obj in inner:
                if not isinstance(obj, dict):
                    continue
                obj_id = (obj.get("id") or "").lower()
                if obj_id:
                    ids.add(obj_id)
                for ws in obj.get("week_scoped_ids", []) or []:
                    if ws:
                        ids.add(str(ws).lower())

        # pre-objectives-emit archives shipped only
        # course.json. Resolve refs against its flat learning_outcomes[]
        # so the broken-refs check doesn't false-positive on every chunk.
        if not ids:
            try:
                course_json_path = self.output_dir / "course.json"
                if course_json_path.exists():
                    with open(course_json_path, encoding="utf-8") as f:
                        course = json.load(f)
                    for lo in course.get("learning_outcomes", []) or []:
                        if isinstance(lo, dict):
                            lo_id = (lo.get("id") or "").lower()
                            if lo_id:
                                ids.add(lo_id)
            except (OSError, json.JSONDecodeError, AttributeError):
                pass

        return ids

    def _assert_integrity(self, report: Dict[str, Any]) -> None:
        """When strict_mode is on, refuse to write metadata if integrity fails.

        Fired from :meth:`_write_metadata` before any file write. Violates:
        broken_refs non-empty, follows_chunk boundary violations non-empty,
        or html_balance_violations rate above 5%.
        """
        if not self.strict_mode:
            return
        integrity = report.get("integrity", {})
        broken = integrity.get("broken_refs", [])
        boundary = integrity.get("follows_chunk_boundary_violations", [])
        html_bad = integrity.get("html_balance_violations", [])
        total = max(self.stats.get("total_chunks", 0), 1)
        html_rate = len(html_bad) / total

        reasons: List[str] = []
        if broken:
            reasons.append(f"{len(broken)} unresolvable learning_outcome_refs")
        if boundary:
            reasons.append(f"{len(boundary)} cross-lesson follows_chunk links")
        if html_rate > 0.05:
            reasons.append(
                f"html_balance_violations rate {html_rate:.0%} > 5% threshold"
            )
        if reasons:
            raise PipelineIntegrityError(
                "strict_mode is on and core integrity invariants failed: "
                + "; ".join(reasons)
                + ". Disable strict_mode to produce a non-final artifact."
            )

    def _generate_enrichment_trace_report(
        self, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Group chunks by ``_metadata_trace``
        value per enrichment field and compute counts + percentages.

        Emitted alongside ``quality_report.json`` as ``metadata_trace_report.json``.

        Each section of the output answers "of chunks where this field landed
        / didn't land, which code path / hypothesis produced that outcome?"
        """
        from collections import Counter as _Counter

        fields = ("content_type_label", "key_terms", "bloom_level", "misconceptions")
        total = len(chunks) or 1
        per_field: Dict[str, Dict[str, Any]] = {}

        for field in fields:
            counter: _Counter = _Counter()
            for c in chunks:
                trace = c.get("_metadata_trace") or {}
                counter[trace.get(field, "none")] += 1
            rows = []
            for trace_value, count in sorted(counter.items(), key=lambda kv: -kv[1]):
                rows.append({
                    "trace": trace_value,
                    "count": count,
                    "pct": round(count / total, 3),
                    "hypothesis": _HYPOTHESIS_BY_TRACE.get(trace_value, "n/a"),
                })
            # Aggregate: how many chunks got this field populated?
            populated = sum(
                cnt for tv, cnt in counter.items() if not tv.startswith("none")
            )
            per_field[field] = {
                "populated_count": populated,
                "populated_pct": round(populated / total, 3),
                "by_trace": rows,
            }

        return {
            "course_code": self.course_code,
            "total_chunks": len(chunks),
            "generated_at": datetime.now().isoformat(),
            "fields": per_field,
            "hypotheses_reference": {
                "H1": "heading-normalisation drift between Courseforge emit + Trainforge consume",
                "H2": "JSON-LD sections genuinely absent on the page",
                "H3": "content_type_label short-circuit at _extract_section_metadata gate — JSON-LD supplies contentType but not keyTerms, so data-cf-* fallback never runs",
                "H4": "no-sections code path — chunk heading equals page title, JSON-LD sections keyed by section heading → no match",
                "H5": "JSON-LD script tag present but JSON parse failed; chunker treats as absent",
            },
        }

    def _build_pedagogy_summary(
        self,
        chunks: Optional[List[Dict[str, Any]]] = None,
        pedagogy_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a pedagogy model grounded in the actual chunk set.

        Emits module_sequence (order + per-module stats), bloom_progression
        (per-module Bloom distribution), and prerequisite_chain (concepts
        referenced as prereqs after being first introduced earlier). Falls
        back to just the top-level keys when chunks aren't provided so
        older call sites don't break.

        When ``pedagogy_graph`` is supplied, ``prerequisite_chain`` is
        populated from the graph's ``prerequisite_of`` edges instead of the
        chunk ``prereq_concepts`` field. Those are two computation paths over
        the same data with no link between them, and the chunk field is
        frequently empty while the graph carries a full prereq edge set — so
        reading the chunk field yields an empty chain on a healthy graph. The
        graph path is also more accurate (chunk co-occurrence + strict
        later-week filter). It replaces the chunk-field logic when available.
        """
        from lib.ontology.concept_id import strip_concept_prefix
        summary: Dict[str, Any] = {
            "instructional_approach": "competency-based",
            "learning_theory": "constructivism",
            "engagement_patterns": ["interactive-scenarios", "formative-assessment"],
        }
        if self.objectives and self.objectives.get("bloom_distribution"):
            summary["bloom_coverage"] = self.objectives["bloom_distribution"]

        if not chunks:
            return summary

        # --- module_sequence + bloom_progression --------------------------------
        module_meta: Dict[str, Dict[str, Any]] = {}
        module_order: List[str] = []
        def bloom_zero() -> Dict[str, int]:
            return {
                "remember": 0,
                "understand": 0,
                "apply": 0,
                "analyze": 0,
                "evaluate": 0,
                "create": 0,
            }

        for chunk in chunks:
            src = chunk.get("source") or {}
            module_id = src.get("module_id")
            if not module_id:
                continue
            if module_id not in module_meta:
                module_order.append(module_id)
                week_match = re.search(r"week[_\-\s]?(\d+)", module_id, re.IGNORECASE)
                module_meta[module_id] = {
                    "module_id": module_id,
                    "module_title": src.get("module_title", ""),
                    "week_num": int(week_match.group(1)) if week_match else 0,
                    "chunk_count": 0,
                    "outcome_refs_covered": set(),
                    "bloom_counts": bloom_zero(),
                    "first_seen": len(module_order),
                }
            meta = module_meta[module_id]
            meta["chunk_count"] += 1
            meta["outcome_refs_covered"].update(chunk.get("learning_outcome_refs", []))
            bloom = chunk.get("bloom_level")
            if bloom in meta["bloom_counts"]:
                meta["bloom_counts"][bloom] += 1

        # Deterministic order: by week_num, then by first-seen position.
        module_order.sort(key=lambda m: (module_meta[m]["week_num"], module_meta[m]["first_seen"]))

        module_sequence = []
        bloom_progression: Dict[str, Dict[str, int]] = {}
        for mid in module_order:
            meta = module_meta[mid]
            module_sequence.append({
                "module_id": mid,
                "module_title": meta["module_title"],
                "week_num": meta["week_num"],
                "chunk_count": meta["chunk_count"],
                "outcome_refs_covered": sorted(meta["outcome_refs_covered"]),
            })
            bloom_progression[mid] = meta["bloom_counts"]

        summary["module_sequence"] = module_sequence
        summary["bloom_progression"] = bloom_progression

        # --- prerequisite_chain + prerequisite_violations ------------------------
        prerequisite_chain: List[Dict[str, Any]] = []
        prerequisite_violations: List[Dict[str, Any]] = []

        if pedagogy_graph is not None:
            # read directly from the graph's prerequisite_of
            # edges. Each edge represents a co-occurrence pair where the
            # source concept lives in an earlier module than the target.
            # Strip the ``concept:`` prefix to keep the chain payload's
            # concept slugs flat (matches the legacy chunk-based path).
            for edge in pedagogy_graph.get("edges", []) or []:
                if not isinstance(edge, dict):
                    continue
                if edge.get("relation_type") != "prerequisite_of":
                    continue
                src_id = edge.get("source") or ""
                tgt_id = edge.get("target") or ""
                if not src_id or not tgt_id:
                    continue
                prerequisite_chain.append({
                    "concept": strip_concept_prefix(src_id),
                    "required_for": strip_concept_prefix(tgt_id),
                    "confidence": edge.get("confidence", 1),
                })
            # Deterministic order for byte-stable output across runs.
            prerequisite_chain.sort(
                key=lambda r: (r["concept"], r["required_for"])
            )
        else:
            # Legacy path: derive from chunk concept_tags vs prereq_concepts.
            # For each concept tag, record earliest (module_idx, chunk_id)
            # where it appears in concept_tags (definition site) vs
            # prereq_concepts (use site). Valid chain: first use in module
            # index > first definition's module index.
            module_idx = {mid: i for i, mid in enumerate(module_order)}
            first_def: Dict[str, Tuple[int, str, str]] = {}
            first_use: Dict[str, Tuple[int, str, str]] = {}
            for chunk in chunks:
                src = chunk.get("source") or {}
                mid = src.get("module_id")
                if mid not in module_idx:
                    continue
                idx = module_idx[mid]
                cid = chunk["id"]
                for tag in chunk.get("concept_tags", []) or []:
                    if tag not in first_def or idx < first_def[tag][0]:
                        first_def[tag] = (idx, mid, cid)
                for tag in chunk.get("prereq_concepts", []) or []:
                    if tag not in first_use or idx < first_use[tag][0]:
                        first_use[tag] = (idx, mid, cid)

            for tag in sorted(set(first_def) & set(first_use)):
                def_idx, def_mod, def_chunk = first_def[tag]
                use_idx, use_mod, use_chunk = first_use[tag]
                record = {
                    "concept": tag,
                    "defined_in": {"module_id": def_mod, "chunk_id": def_chunk},
                    "first_used_in": {"module_id": use_mod, "chunk_id": use_chunk},
                }
                if use_idx > def_idx:
                    prerequisite_chain.append(record)
                elif use_idx < def_idx:
                    prerequisite_violations.append(record)

        summary["prerequisite_chain"] = prerequisite_chain
        summary["prerequisite_violations"] = prerequisite_violations
        return summary

    def _build_course_json(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Build course.json with structured learning outcomes for LibV2.

        The result validates against
        ``schemas/knowledge/course.schema.json`` before being returned.
        Schema violations are logged as warnings (best-effort) — the
        canonical shape is still emitted.

        Always materialize course.json. When
        ``self.objectives`` is ``None`` (neither ``objectives_path``
        kwarg nor the synthesized-objectives sidecar was available),
        we now emit a valid shell:

            {"course_code": ..., "title": ...,
             "learning_outcomes": [], "note": "..."}

        so LibV2 archival always lands a file + downstream joins
        (``LibV2/tools/libv2/retrieval_scoring.py::load_course_outcomes``)
        have something to look at instead of a ``FileNotFoundError``.
        The ``note`` field is optional per the course schema
        (``additionalProperties: true``) so validation still passes.
        """
        outcomes: List[Dict[str, Any]] = []
        note: Optional[str] = None

        if self.objectives:
            for to in self.objectives.get("terminal_objectives", []):
                outcomes.append({
                    "id": to["id"].lower(),
                    "statement": to["statement"],
                    "bloom_level": (to.get("bloomLevel") or to.get("bloom_level") or "understand"),
                    "hierarchy_level": "terminal",
                })

            # chapter_objectives can come in two shapes —
            #   (a) nested: [{"chapter": "Week 1", "objectives": [{...}, ...]}]
            #   (b) flat:   [{"id": "co-01", "parent_to": "to-01", ...}, ...]
            # The planning agent emits the flat
            # form; Trainforge fixtures may use the nested form. Supporting
            # both ensures component objectives propagate into
            # the LibV2 archive's course.json.
            for ch in self.objectives.get("chapter_objectives", []):
                if isinstance(ch, dict) and "objectives" in ch:
                    inner = ch.get("objectives") or []
                else:
                    inner = [ch]
                for obj in inner:
                    if not isinstance(obj, dict) or "id" not in obj:
                        continue
                    outcomes.append({
                        "id": obj["id"].lower(),
                        "statement": obj.get("statement") or obj.get("text") or "",
                        "bloom_level": (
                            obj.get("bloomLevel")
                            or obj.get("bloom_level")
                            or "understand"
                        ),
                        "hierarchy_level": "chapter",
                        # emit a discriminator so downstream
                        # consumers can split terminal vs component
                        # without re-deriving from the ID prefix.
                        "type": "component",
                    })
        else:
            note = (
                "No learning objectives were supplied or synthesized "
                "for this course. Downstream retrieval/validation may "
                "be degraded."
            )

        course_data: Dict[str, Any] = {
            "course_code": self.course_code,
            "title": manifest.get("title", ""),
            "learning_outcomes": outcomes,
        }
        if note is not None:
            course_data["note"] = note

        # best-effort schema validation against the canonical
        # course.schema.json. We don't hard-fail here because the schema
        # is advisory (a soft guard against drift) — a hard failure
        # would block every pipeline run whose objectives file predates
        # the schema. Errors log at WARNING so drift is observable.
        try:
            from pathlib import Path as _Path

            import jsonschema  # type: ignore
            schema_path = (
                _Path(__file__).resolve().parent.parent
                / "schemas" / "knowledge" / "course.schema.json"
            )
            if schema_path.exists():
                with open(schema_path, encoding="utf-8") as _f:
                    schema = json.load(_f)
                try:
                    jsonschema.validate(course_data, schema)
                except jsonschema.ValidationError as exc:
                    logger.warning(
                        "course.json drifted from course.schema.json: %s",
                        exc.message,
                    )
        except ImportError:
            # jsonschema optional dep — skip silently.
            pass
        except Exception as exc:  # noqa: BLE001 - defensive
            logger.debug("course.schema.json validation skipped: %s", exc)

        return course_data

    def _build_objectives_json(self) -> Optional[Dict[str, Any]]:
        """Build the objectives.json hierarchy sidecar.

        Carries the full TO-/CO- hierarchy synthesized by Courseforge's
        ``plan_course_structure`` phase so downstream chunk
        ``learning_outcome_refs`` can resolve against ALL outcomes
        (not just the terminal ones declared on course.json).

        Returns ``None`` when no objectives are available. Otherwise
        returns a dict matching ``schemas/knowledge/objectives_v1.schema.json``.

        Schema validation: best-effort. Drift logs at WARNING; the
        canonical shape is still emitted so consumers can rely on the
        file structure regardless of jsonschema availability.
        """
        if not self.objectives:
            return None

        preserve_case = (
            os.getenv("TRAINFORGE_PRESERVE_LO_CASE", "").lower() == "true"
        )

        def _id(raw: str) -> str:
            return raw.strip() if preserve_case else raw.lower().strip()

        terminal_outcomes: List[Dict[str, Any]] = []
        for to in self.objectives.get("terminal_objectives", []):
            if not isinstance(to, dict) or "id" not in to:
                continue
            entry: Dict[str, Any] = {
                "id": _id(to["id"]),
                "statement": to.get("statement") or to.get("text") or "",
            }
            if to.get("bloom_level") or to.get("bloomLevel"):
                entry["bloom_level"] = (
                    to.get("bloom_level") or to.get("bloomLevel")
                )
            if to.get("bloom_verb"):
                entry["bloom_verb"] = to["bloom_verb"]
            if to.get("cognitive_domain"):
                entry["cognitive_domain"] = to["cognitive_domain"]
            if to.get("weeks"):
                entry["weeks"] = list(to["weeks"])
            terminal_outcomes.append(entry)

        component_objectives: List[Dict[str, Any]] = []
        for ch in self.objectives.get("chapter_objectives", []):
            # Same dual-shape handling as _build_course_json.
            if isinstance(ch, dict) and "objectives" in ch:
                inner = ch.get("objectives") or []
            else:
                inner = [ch]
            for obj in inner:
                if not isinstance(obj, dict) or "id" not in obj:
                    continue
                entry = {
                    "id": _id(obj["id"]),
                    "statement": obj.get("statement") or obj.get("text") or "",
                }
                parent = obj.get("parent_to") or obj.get("parent_terminal")
                if parent:
                    entry["parent_terminal"] = _id(parent)
                if obj.get("bloom_level") or obj.get("bloomLevel"):
                    entry["bloom_level"] = (
                        obj.get("bloom_level") or obj.get("bloomLevel")
                    )
                if obj.get("bloom_verb"):
                    entry["bloom_verb"] = obj["bloom_verb"]
                if obj.get("cognitive_domain"):
                    entry["cognitive_domain"] = obj["cognitive_domain"]
                if obj.get("week") is not None:
                    entry["week"] = obj["week"]
                if obj.get("source_refs"):
                    # Preserve strings and copy structured references. The
                    # ``list(obj["source_refs"])`` cast destroys structured
                    # ``{ref, chunk_ids[]}`` entries by shallow-copying the
                    # outer list while keeping the inner dict references —
                    # which is correct for legacy List[str] but not for
                    # the structured arm. Per-entry dispatch:
                    # strings pass through unchanged; dicts are deep-
                    # copied so a downstream mutator can't accidentally
                    # cross-contaminate two objective entries that share
                    # a ref dict by reference.
                    preserved_refs: List[Any] = []
                    for ref in obj["source_refs"]:
                        if isinstance(ref, str):
                            preserved_refs.append(ref)
                        elif isinstance(ref, dict):
                            ref_field = ref.get("ref")
                            chunk_ids = ref.get("chunk_ids")
                            if isinstance(chunk_ids, list):
                                chunk_ids = list(chunk_ids)
                            preserved_refs.append(
                                {"ref": ref_field, "chunk_ids": chunk_ids}
                            )
                        else:
                            # Mixed-shape entries are theoretically
                            # rejected by the schema's oneOf; stay
                            # defensive here.
                            preserved_refs.append(ref)
                    entry["source_refs"] = preserved_refs
                component_objectives.append(entry)

        objectives_data: Dict[str, Any] = {
            "schema_version": "v1",
            "course_code": self.course_code,
            "terminal_outcomes": terminal_outcomes,
            "component_objectives": component_objectives,
            "objective_count": {
                "terminal": len(terminal_outcomes),
                "component": len(component_objectives),
            },
        }

        # Best-effort schema validation (same pattern as _build_course_json).
        try:
            from pathlib import Path as _Path

            import jsonschema  # type: ignore
            schema_path = (
                _Path(__file__).resolve().parent.parent
                / "schemas" / "knowledge" / "objectives_v1.schema.json"
            )
            if schema_path.exists():
                with open(schema_path, encoding="utf-8") as _f:
                    schema = json.load(_f)
                try:
                    jsonschema.validate(objectives_data, schema)
                except jsonschema.ValidationError as exc:
                    logger.warning(
                        "objectives.json drifted from objectives_v1.schema.json: %s",
                        exc.message,
                    )
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001 - defensive
            logger.debug("objectives_v1.schema.json validation skipped: %s", exc)

        return objectives_data

    # ------------------------------------------------------------------
    # Stage 6: Write metadata
    # ------------------------------------------------------------------

    def _write_metadata(
        self,
        manifest: Dict[str, Any],
        corpus_stats: Dict[str, Any],
        concept_graph: Dict[str, Any],
        quality_report: Dict[str, Any],
        pedagogy_graph: Optional[Dict[str, Any]] = None,
        semantic_graph: Optional[Dict[str, Any]] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
    ):
        # Strict-mode gate: refuse to write an artifact whose quality report
        # shows integrity violations. Disabled by default for v0.1.x; flipped
        # on in the follow-up PR (see VERSIONING.md §1.6 severity trigger).
        self._assert_integrity(quality_report)

        def _write(path: Path, data: Dict[str, Any]):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        _write(self.output_dir / "manifest.json", manifest)

        # course.json — structured learning outcomes for LibV2 validator.
        # Gap 4: always write course.json (including the
        # empty-learning_outcomes shell with a ``note`` field) so LibV2
        # archival always lands a file and downstream retrieval joins
        # have something to look at. legacy this was gated on
        # ``self.objectives`` being truthy, so pipeline runs that
        # auto-synthesized objectives without threading the path in
        # never emitted course.json.
        course_data = self._build_course_json(manifest)
        _write(self.output_dir / "course.json", course_data)

        # emit objectives.json sidecar with the full TO-/CO-
        # hierarchy. course.json declares the flattened
        # learning_outcomes for LibV2 retrieval; objectives.json is the
        # structured source of truth that keeps the
        # ``terminal_outcomes[]`` / ``component_objectives[]`` split
        # explicit (with parent_terminal back-pointers, source_refs,
        # week assignments, and the canonical schema_version=v1
        # discriminator). LibV2's importer copies it alongside
        # course.json so chunk learning_outcome_refs can resolve
        # against the full set.
        objectives_data = self._build_objectives_json()
        if objectives_data is not None:
            _write(self.output_dir / "objectives.json", objectives_data)

        _write(self.corpus_dir / "corpus_stats.json", corpus_stats)
        _write(self.graph_dir / "concept_graph.json", concept_graph)
        if pedagogy_graph is not None:
            _write(self.graph_dir / "pedagogy_graph.json", pedagogy_graph)
        if semantic_graph is not None:
            _write(self.graph_dir / "concept_graph_semantic.json", semantic_graph)
            # additionally emit a TriG file with per-rule named
            # graphs when TRAINFORGE_EMIT_TRIG is on. Default off → this
            # branch never fires for legacy consumers.
            trig_dataset = getattr(self, "_semantic_graph_trig_dataset", None)
            if trig_dataset is not None:
                try:
                    from Trainforge.rag import named_graph_writer
                    trig_path = self.graph_dir / "concept_graph_semantic.trig"
                    trig_path.write_text(
                        named_graph_writer.serialize_trig(trig_dataset),
                        encoding="utf-8",
                    )
                except Exception as exc:  # pragma: no cover — defensive
                    print(
                        f"[warn] TRAINFORGE_EMIT_TRIG: failed to write "
                        f"concept_graph_semantic.trig: {exc}"
                    )
        _write(self.quality_dir / "quality_report.json", quality_report)

        # Pedagogy model (full: module sequence, bloom progression, prereq chain).
        # thread pedagogy_graph so prerequisite_chain populates from
        # the graph's prerequisite_of edges instead of the empty-by-default
        # chunk.prereq_concepts field. This prevents a populated graph from
        # producing an empty prerequisite chain in the pedagogy summary.
        pedagogy = self._build_pedagogy_summary(
            chunks=chunks, pedagogy_graph=pedagogy_graph
        )
        _write(self.pedagogy_dir / "pedagogy_model.json", pedagogy)

        # Emit enrichment provenance alongside quality_report.json.
        if chunks:
            trace_report = self._generate_enrichment_trace_report(chunks)
            _write(self.quality_dir / "metadata_trace_report.json", trace_report)

        # Training specs
        training_specs = {
            "format": "instruction-following",
            "target_models": _resolve_target_models(),
            "training_objectives": [
                f"{self.domain}_instruction",
                f"{self.domain}_reasoning",
            ],
            "statistics": {
                "total_tokens": self.stats["total_tokens_estimate"],
            },
        }
        _write(self.training_specs_dir / "dataset_config.json", training_specs)

        # IMPORT_SUMMARY.md
        self._write_import_summary(manifest, corpus_stats, quality_report)

    def _write_import_summary(
        self, manifest: Dict[str, Any], stats: Dict[str, Any], quality: Dict[str, Any]
    ):
        lines = [
            f"# Import Summary: {manifest['title']}",
            "",
            f"**Course Code:** {self.course_code}",
            f"**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Division:** {self.division} | **Domain:** {self.domain}",
            "",
            "## Corpus Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Chunks | {stats['total_chunks']} |",
            f"| Total Words | {stats['total_words']:,} |",
            f"| Total Tokens (est) | {stats['total_tokens_estimate']:,} |",
            f"| Avg Words/Chunk | {stats['avg_words_per_chunk']:.1f} |",
            f"| Sections | {stats['sections_processed']} |",
            f"| Modules | {stats['modules_processed']} |",
            f"| Quizzes | {stats['quizzes_processed']} |",
            "",
            "## Chunk Type Distribution",
            "",
        ]
        for ctype, count in sorted(stats.get("chunk_type_distribution", {}).items()):
            lines.append(f"- **{ctype}**: {count}")

        lines.extend([
            "",
            "## Difficulty Distribution",
            "",
        ])
        for diff, count in sorted(stats.get("difficulty_distribution", {}).items()):
            lines.append(f"- **{diff}**: {count}")

        lines.extend([
            "",
            "## Quality",
            "",
            f"- Overall Score: **{quality['overall_quality_score']:.3f}**",
            f"- Passed: **{quality['validation']['passed']}**",
            "",
            "Ready for LibV2 import.",
        ])

        with open(self.output_dir / "IMPORT_SUMMARY.md", "w") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_directories(self):
        for d in [self.corpus_dir, self.graph_dir, self.training_specs_dir,
                  self.pedagogy_dir, self.quality_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def prune_output_after_import(
    output_dir: Path,
    course_code: str,
    libv2_slug: Optional[str],
    libv2_target_path: Optional[Path],
    libv2_root: Path,
) -> Optional[Path]:
    """Drop everything in ``output_dir`` and leave only ``IMPORT_RECEIPT.json``.

    The caller is responsible for verifying that the LibV2
    import actually succeeded; this helper unconditionally prunes when
    invoked. Returns the path to the receipt on success, ``None`` when
    the prune was refused (sanity guard).

    Sanity guard: never prune if ``output_dir`` IS the LibV2 root or
    sits inside it — that would corrupt the freshly imported archive.

    The output dir itself is preserved (callers may have it open / cd'd);
    only its contents are dropped before the receipt is written.
    """
    import shutil

    output_dir = Path(output_dir).resolve()
    libv2_root = Path(libv2_root).resolve()
    try:
        output_dir.relative_to(libv2_root)
        inside_libv2 = True
    except ValueError:
        inside_libv2 = False
    if output_dir == libv2_root or inside_libv2:
        print(
            f"[Prune] Refusing to prune {output_dir} because it is the "
            f"LibV2 root or sits inside it."
        )
        return None

    chunks_imported = 0
    chunks_path = output_dir / "chunks.jsonl"
    if chunks_path.exists():
        try:
            with chunks_path.open("r", encoding="utf-8") as fh:
                chunks_imported = sum(1 for line in fh if line.strip())
        except Exception:
            chunks_imported = 0

    receipt = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "course_code": course_code,
        "libv2_slug": libv2_slug,
        "libv2_path": str(libv2_target_path) if libv2_target_path else None,
        "chunks_imported": chunks_imported,
    }

    if output_dir.exists():
        for entry in output_dir.iterdir():
            try:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            except Exception as exc:
                print(f"[Prune] Skipped {entry}: {exc}")
    output_dir.mkdir(parents=True, exist_ok=True)

    receipt_path = output_dir / "IMPORT_RECEIPT.json"
    with receipt_path.open("w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(
        f"[Prune] Pruned {output_dir}; left {receipt_path.name} with "
        f"{chunks_imported} chunks_imported recorded."
    )
    return receipt_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Process a Courseforge IMSCC into a Trainforge RAG corpus",
    )
    p.add_argument("--imscc", required=True, help="Path to .imscc file")
    p.add_argument("--course-code", required=True, help="Course code for this package")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--objectives", help="Path to objectives JSON (optional)")
    # Classification flags accept None sentinels so the
    # CourseProcessor can distinguish "user didn't pass this flag" (use
    # course_metadata.json stub if present) from "user explicitly set this"
    # (override the stub). --domain is no longer required at the argparse
    # layer; main() enforces that either the stub or --domain supplies a
    # primary domain before the processor starts.
    p.add_argument(
        "--division",
        default=None,
        choices=["STEM", "ARTS"],
        help="Division (overrides course_metadata.json stub when provided)",
    )
    p.add_argument(
        "--domain",
        default=None,
        help=(
            "Primary domain (overrides course_metadata.json stub when "
            "provided; required when no stub is present)"
        ),
    )
    p.add_argument(
        "--subdomain",
        action="append",
        default=None,
        help="Subdomain (repeatable; overrides stub when provided)",
    )
    p.add_argument("--secondary-domain", action="append", default=[], help="Secondary domain (repeatable)")
    p.add_argument(
        "--topic",
        action="append",
        default=None,
        help="Topic (repeatable; overrides stub when provided)",
    )
    p.add_argument("--align", action="store_true",
                   help="Run alignment stage after processing (prereq_concepts, teaching_role, learning_outcome_refs)")
    p.add_argument("--llm-provider", default="mock", choices=["mock", "anthropic", "together", "local"],
                   help=(
                       "LLM provider for the legacy --align direct-classification path "
                       "(default: mock). For the license-clean teaching-role surface, "
                       "prefer setting CURRICULUM_ALIGNMENT_PROVIDER=local instead — "
                       "that route is wired through Trainforge.align_chunks.main() "
                       "and honours the same LOCAL_SYNTHESIS_* / TOGETHER_* env vars "
                       "as synthesis."
                   ))
    p.add_argument("--import-to-libv2", action="store_true", help="Import into LibV2 after processing")
    p.add_argument(
        "--prune-after-import",
        action="store_true",
        help=(
            "After a successful --import-to-libv2, drop the contents of "
            "--output and leave only IMPORT_RECEIPT.json. Default off so "
            "current behavior (output dir kept verbatim) is preserved. "
            "Ignored with a warning when --import-to-libv2 is not set."
        ),
    )
    p.add_argument("--synthesize", action="store_true",
                   help="Synthesize SFT/DPO training pairs from chunks after base processing.")
    p.add_argument("--synthesis-provider", default="mock",
                   choices=["mock", "anthropic", "claude_session", "together", "local"],
                   help=(
                       "Provider for training-pair synthesis (default: mock). "
                       "License-clean choices: 'local' (Apache 2.0 Qwen via Ollama, "
                       "reads LOCAL_SYNTHESIS_BASE_URL/MODEL) or 'together' (Together "
                       "AI OSS, reads TOGETHER_API_KEY/MODEL). 'anthropic' and "
                       "'claude_session' produce ToS-restricted outputs — see "
                       "docs/LICENSING.md."
                   ))
    p.add_argument("--synthesis-seed", type=int, default=17,
                   help="Base deterministic seed for training-pair synthesis (default: 17).")
    p.add_argument(
        "--typed-edges-llm",
        action="store_true",
        help=(
            "Enable the optional LLM escalation pass for the typed-edge "
            "concept graph. OFF by default — the rule-based path is "
            "deterministic and byte-identical across runs."
        ),
    )
    p.add_argument(
        "--benchmark-retrieval",
        action="store_true",
        help=(
            "After processing, run the recall@k retrieval benchmark "
            "(BM25 over text vs summary vs retrieval_text) and write "
            "quality/retrieval_benchmark.json."
        ),
    )
    p.add_argument(
        "--concept-graph-path",
        default=None,
        help=(
            "Path to a pre-built pedagogy graph emitted "
            "upstream by the ``concept_extraction`` workflow phase "
            "(LibV2/courses/<slug>/concept_graph/concept_graph_semantic.json). "
            "When supplied AND readable, ``_generate_pedagogy_graph`` loads "
            "this graph instead of re-invoking ``build_pedagogy_graph`` "
            "in-process. Falls through to the in-process build when "
            "absent / unreadable (legacy corpora preserved unchanged)."
        ),
    )
    p.add_argument(
        "--imscc-chunks-path",
        default=None,
        help=(
            "Path to a pre-built IMSCC chunkset emitted "
            "upstream by the ``imscc_chunking`` workflow phase "
            "(LibV2/courses/<slug>/imscc_chunks/chunks.jsonl). When "
            "supplied AND readable, ``process()`` short-circuits the "
            "in-process ``_chunk_content`` call and consumes this "
            "chunkset directly. Falls through to the in-process build "
            "when absent / unreadable (legacy callers preserved "
            "unchanged)."
        ),
    )
    return p


def main():
    args = build_parser().parse_args()

    # Require either a course_metadata.json stub or a
    # --domain CLI flag. The processor can boot with defaults for legacy
    # pipelines that set --division but not --domain, but an empty primary
    # domain is a misconfiguration worth catching before Stage 1.
    if args.domain is None:
        imscc_path = Path(args.imscc)
        has_stub = (imscc_path.parent / "course_metadata.json").exists()
        if not has_stub and imscc_path.exists():
            try:
                with zipfile.ZipFile(imscc_path, "r") as z:
                    has_stub = "course_metadata.json" in z.namelist()
            except Exception:
                pass
        if not has_stub:
            sys.stderr.write(
                "error: --domain is required when no course_metadata.json "
                "stub is present at the IMSCC path or its parent directory.\n"
            )
            sys.exit(2)

    processor = CourseProcessor(
        imscc_path=args.imscc,
        output_dir=args.output,
        course_code=args.course_code,
        division=args.division,
        domain=args.domain,
        subdomains=args.subdomain,
        secondary_domains=args.secondary_domain,
        topics=args.topic,
        objectives_path=args.objectives,
        typed_edges_llm=getattr(args, "typed_edges_llm", False),
        concept_graph_path=getattr(args, "concept_graph_path", None),
        imscc_chunks_path=getattr(args, "imscc_chunks_path", None),
    )

    result = processor.process()

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Course: {result['title']}")
    print(f"Output: {result['output_dir']}")
    print(f"Chunks: {result['stats']['total_chunks']}")
    print(f"Words:  {result['stats']['total_words']:,}")
    print(f"Tokens: {result['stats']['total_tokens_estimate']:,}")
    print("\nChunk types:")
    for ct, count in result["stats"]["chunk_types"].items():
        print(f"  {ct}: {count}")
    print("\nDifficulty:")
    for d, count in result["stats"]["difficulty_distribution"].items():
        print(f"  {d}: {count}")

    # Optional alignment stage
    if args.align:
        print("\n[Alignment] Running alignment stage...")
        # env-var-first model resolution; mirrors
        # Trainforge/align_chunks.py::_resolve_align_model
        # so a single TRAINFORGE_ALIGN_CHUNKS_MODEL env var controls
        # both the standalone CLI and the embedded process_course path.
        from Trainforge.align_chunks import _resolve_align_model
        from Trainforge.align_chunks import main as align_main
        align_args = argparse.Namespace(
            corpus=args.output,
            objectives=args.objectives,
            fields="prereq_concepts,teaching_role,learning_outcome_refs",
            llm_provider=args.llm_provider,
            llm_model=_resolve_align_model(),
            dry_run=False,
            verbose=False,
        )
        align_main(align_args)

    # Optional training-pair synthesis stage
    if args.synthesize:
        print("\n[Synthesis] Running training-pair synthesis stage...")
        from Trainforge.synthesis.synthesize_training import run_synthesis
        try:
            synth_stats = run_synthesis(
                corpus_dir=Path(args.output),
                course_code=args.course_code,
                provider=args.synthesis_provider,
                seed=args.synthesis_seed,
            )
            print(f"[Synthesis] Emitted {synth_stats.instruction_pairs_emitted} "
                  f"instruction pairs, {synth_stats.preference_pairs_emitted} preference pairs "
                  f"from {synth_stats.chunks_eligible}/{synth_stats.chunks_total} eligible chunks.")
        except Exception as e:
            print(f"[Synthesis] Failed: {e}")

    # Optional LibV2 import
    libv2_import_succeeded = False
    libv2_slug: Optional[str] = None
    libv2_target_path: Optional[Path] = None
    if args.import_to_libv2:
        print("\n[LibV2] Importing into LibV2...")
        try:
            from LibV2.tools.libv2.importer import import_course as do_import

            # Use processor-resolved fields so stub-driven classification
            # flows into LibV2 import when no CLI flags were set.
            slug = do_import(
                source_dir=Path(args.output),
                repo_root=PROJECT_ROOT / "LibV2",
                division=processor.division,
                domain=processor.domain,
                subdomains=processor.subdomains if processor.subdomains else None,
                topics=processor.topics if processor.topics else None,
                secondary_domains=processor.secondary_domains if processor.secondary_domains else None,
                imscc_path=Path(args.imscc),
                strict_validation=False,
            )
            print(f"[LibV2] Imported as: {slug}")
            print(f"[LibV2] Location: LibV2/courses/{slug}/")
            libv2_import_succeeded = True
            libv2_slug = slug
            libv2_target_path = (PROJECT_ROOT / "LibV2" / "courses" / slug).resolve()
        except Exception as e:
            print(f"[LibV2] Import failed: {e}")
            print("[LibV2] You can import manually later with:")
            print(
                f"  python -m LibV2.tools.libv2.cli import {args.output} "
                f"--domain {processor.domain} --division {processor.division}"
            )

    # prune --output after a successful LibV2 import so the
    # output dir doesn't sit on disk as a duplicate of the LibV2 archive.
    # Default OFF — current behavior (full output dir kept) is unchanged
    # unless the caller explicitly opts in.
    if args.prune_after_import:
        if not args.import_to_libv2:
            print(
                "[Prune] Warning: --prune-after-import has no effect without "
                "--import-to-libv2; ignoring."
            )
        elif not libv2_import_succeeded:
            print(
                "[Prune] LibV2 import did not succeed; preserving --output "
                "dir verbatim (no pruning)."
            )
        else:
            prune_output_after_import(
                output_dir=Path(args.output),
                course_code=args.course_code,
                libv2_slug=libv2_slug,
                libv2_target_path=libv2_target_path,
                libv2_root=PROJECT_ROOT / "LibV2",
            )

    # Optional: retrieval benchmark over the freshly regenerated corpus.
    # Measures whether the per-chunk summary improves BM25 recall@k over
    # the raw text baseline. Written to quality/retrieval_benchmark.json.
    if args.benchmark_retrieval:
        print("\n[Benchmark] Running retrieval benchmark...")
        try:
            from Trainforge.rag.retrieval_benchmark import write_benchmark

            out_path, bench = write_benchmark(Path(args.output))
            print(f"[Benchmark] Wrote {out_path}")
            for variant, scores in bench.get("variants", {}).items():
                summary_line = ", ".join(
                    f"{k}={v:.3f}" for k, v in sorted(scores.items())
                )
                print(f"[Benchmark]   {variant}: {summary_line}")
        except Exception as e:
            print(f"[Benchmark] Failed: {e}")

    print("\nDone!")
    return result


if __name__ == "__main__":
    main()

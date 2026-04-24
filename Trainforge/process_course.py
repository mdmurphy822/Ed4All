#!/usr/bin/env python3
"""
Generic Course Corpus Pipeline

Processes any Courseforge IMSCC package into a Sourceforge-compatible
RAG corpus for LibV2 import.

Usage:
    python -m Trainforge.process_course \
        --imscc path/to/course.imscc \
        --course-code SAMPLE_101 \
        --division ARTS --domain education --subdomain instructional-design \
        --output Trainforge/output/sample_101

    # With objectives file for Bloom's-based difficulty mapping:
    python -m Trainforge.process_course \
        --imscc path/to/course.imscc \
        --objectives path/to/objectives.json \
        --course-code SAMPLE_101 \
        --division ARTS --domain education \
        --output Trainforge/output/sample_101 \
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.decision_capture import DecisionCapture
from lib.ontology.slugs import canonical_slug
from Trainforge.generators import summary_factory
from Trainforge.parsers.html_content_parser import HTMLContentParser, HTMLTextExtractor
from Trainforge.parsers.xpath_walker import (
    find_body_xpath,
    find_section_container_xpath,
    resolve_xpath,
)
from Trainforge.rag.boilerplate_detector import (
    BoilerplateConfig,
    contamination_rate,
    detect_repeated_ngrams,
    strip_boilerplate,
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
#     interactive_components_rate. See docs/metrics/flow-metrics.md. (Worker B)
# v5: adds top-level `package_completeness` aggregate — a flat mean of the
#     five enrichment coverage fractions. Answers "of the metadata this
#     package claims to provide, how much actually landed." NOT inside
#     `metrics`; NOT weighted into `overall_quality_score`. Separate
#     top-level key so consumers can read one honest number without
#     cross-referencing five metrics. (Worker P)
METRICS_SEMANTIC_VERSION = 5

# Chunk schema version. Bumped by the first of Workers B / D / E to touch
# chunk shape (ADR-001 Contract 1). v4 adds:
#   - `summary` (Worker D): 2–3 sentence extractive summary per chunk.
#   - `retrieval_text` (Worker D, optional): summary + " " + key_terms_joined.
#   - `schema_version` (all workers): stamped on every chunk.
#   - `source.html_xpath` and `source.char_span` (Worker E): audit-trail
#     provenance stamped on every chunk.
# The string also lands on manifest.json as `chunk_schema_version`. One bump
# per release train; see ADR-001 Contract 1 and docs/contributing/workers.md
# for the rebase protocol.
CHUNK_SCHEMA_VERSION = "v4"


# Worker N (REC-ID-01): opt-in content-hash chunk IDs. When
# TRAINFORGE_CONTENT_HASH_IDS=true, chunk IDs are derived from
# sha256(text + source_locator + schema_version) so re-chunking the same
# source produces identical IDs; this keeps edge-evidence references that
# quote chunk IDs stable across re-runs. Default remains position-based for
# backward compatibility with already-ingested LibV2 courses.
USE_CONTENT_HASH_IDS = os.getenv("TRAINFORGE_CONTENT_HASH_IDS", "").lower() == "true"


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


# Worker I (REC-CTR-01): opt-in chunk validation against chunk_v4.schema.json.
# The schema plus its $ref store (Worker F's taxonomies) is cached after first
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

    Wave 74 fix: prefer the modern ``referencing`` library (jsonschema
    4.18+) over the deprecated ``RefResolver``. Under certain load
    orders / resolver-stack pushes, ``RefResolver`` fails to resolve
    inline ``#/$defs/Source`` with
    ``_RefResolutionError: Unresolvable JSON pointer: '$defs/Source'``
    after descending into an external ``$ref`` (symptom observed in
    today's ``RDF_SHACL_KG`` pipeline run). The ``referencing``-based
    resolver keeps the base-URI stack honest and resolves both inline
    and external refs deterministically. We fall back to ``RefResolver``
    only when ``referencing`` is missing, preserving backward compat
    for environments still on pre-4.18 jsonschema.

    Returns None if jsonschema is unavailable or the schema file cannot
    be loaded — caller treats that as "hook disabled" and the pipeline
    proceeds without validation.
    """
    global _CHUNK_VALIDATOR, _CHUNK_SCHEMA_LOAD_FAILED
    if _CHUNK_VALIDATOR is not None:
        return _CHUNK_VALIDATOR
    if _CHUNK_SCHEMA_LOAD_FAILED:
        return None
    try:
        import jsonschema  # noqa: F401
        from jsonschema import Draft202012Validator
    except ImportError:
        _CHUNK_SCHEMA_LOAD_FAILED = True
        return None
    schemas_root = PROJECT_ROOT / "schemas"
    schema_path = schemas_root / "knowledge" / "chunk_v4.schema.json"
    if not schema_path.exists():
        _CHUNK_SCHEMA_LOAD_FAILED = True
        return None
    try:
        with open(schema_path) as f:
            schema = json.load(f)
        # Collect every local schema keyed by its $id for offline ref
        # resolution (Worker F taxonomies, source_reference, etc.).
        id_to_schema: Dict[str, Dict[str, Any]] = {}
        for p in schemas_root.rglob("*.json"):
            try:
                with open(p) as f:
                    s = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            sid = s.get("$id")
            if sid:
                id_to_schema[sid] = s

        # Prefer the modern `referencing` library (jsonschema 4.18+).
        # Falls back to the deprecated `RefResolver` only when
        # `referencing` is unavailable.
        try:
            from referencing import Registry, Resource
            from referencing.jsonschema import DRAFT202012

            resources = [
                (sid, Resource.from_contents(s, default_specification=DRAFT202012))
                for sid, s in id_to_schema.items()
            ]
            registry = Registry().with_resources(resources)
            _CHUNK_VALIDATOR = Draft202012Validator(schema, registry=registry)
        except ImportError:
            from jsonschema import RefResolver  # type: ignore

            resolver = RefResolver.from_schema(schema, store=dict(id_to_schema))
            _CHUNK_VALIDATOR = Draft202012Validator(schema, resolver=resolver)
    except Exception:
        _CHUNK_SCHEMA_LOAD_FAILED = True
        return None
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


# Worker M1 (§4.4a diagnostic): maps each _metadata_trace value to the
# VERSIONING.md §4.4a hypothesis it implicates. Removed by Worker M2.
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
    """
    with open(objectives_path) as f:
        data = json.load(f)

    week_bloom: Dict[int, List[str]] = defaultdict(list)

    for chapter in data.get("chapter_objectives", []):
        # Parse week range from chapter name like "Week 1-2: ..."
        chapter_name = chapter.get("chapter", "")
        week_match = re.search(r"[Ww]eek\s+(\d+)(?:\s*-\s*(\d+))?", chapter_name)
        if week_match:
            start = int(week_match.group(1))
            end = int(week_match.group(2)) if week_match.group(2) else start
            weeks = list(range(start, end + 1))
        else:
            weeks = []

        for obj in chapter.get("objectives", []):
            bloom = obj.get("bloomLevel", "").lower()
            if bloom:
                for w in weeks:
                    week_bloom[w].append(bloom)

    return {
        "terminal_objectives": data.get("terminal_objectives", []),
        "chapter_objectives": data.get("chapter_objectives", []),
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
# Concept tag normalization
# ---------------------------------------------------------------------------

def normalize_tag(raw: str) -> str:
    """Normalize a concept string to lowercase-hyphenated tag.

    Canonicalization delegates to the shared ``lib.ontology.slugs.canonical_slug``
    (REC-ID-03, Wave 4 Worker Q). The display-layer rules — truncating to 4
    tokens and rejecting tags whose first character isn't alphabetic — remain
    specific to Trainforge's LibV2 tag format and stay here.
    """
    tag = canonical_slug(raw)
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


# ---------------------------------------------------------------------------
# CourseProcessor
# ---------------------------------------------------------------------------

class CourseProcessor:
    """Generic processor that turns a Courseforge IMSCC into a Trainforge corpus."""

    TARGET_CHUNK_SIZE = 500
    MIN_CHUNK_SIZE = 100  # Courseforge pages can be short (overviews, summaries)
    MAX_CHUNK_SIZE = 800

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
    ):
        # When strict_mode is True the pipeline refuses to write a final
        # artifact whose quality_report shows any broken_refs, any cross-lesson
        # follows_chunk link, or html_balance_violations above 5%. See §1.5 of
        # VERSIONING.md.
        self.strict_mode = strict_mode
        # When typed_edges_llm is True, the typed-edge concept-graph builder
        # calls an LLM escalation callable for edges no rule covered. Off by
        # default — the default deterministic path is byte-identical across
        # runs (Worker F spec, ADR-001 Contract 3).
        self.typed_edges_llm = typed_edges_llm
        self.imscc_path = Path(imscc_path)
        self.output_dir = Path(output_dir)
        self.course_code = course_code

        # ------------------------------------------------------------------
        # Wave 2 REC-TAX-01: classification resolution.
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
        self.corpus_dir = self.output_dir / "corpus"
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
            # Wave 30 Gap 4: when no objectives_path is supplied, probe the
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
                        "Wave 30 Gap 4: auto-detected synthesized objectives at %s",
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
                    "Wave 30 Gap 4: failed to load objectives from %s: %s; "
                    "course.json will land as an empty-learning_outcomes shell",
                    resolved_objectives_path,
                    _obj_exc,
                )
                self.objectives = None
                self._objectives_source = "load_failed"

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
    # Classification stub loader (Wave 2 REC-TAX-01)
    # ------------------------------------------------------------------

    def _load_classification_stub(self) -> Optional[Dict[str, Any]]:
        """Locate and parse ``course_metadata.json``, if present.

        Searches (in order):
          1. Inside the IMSCC zip at root — forward-compat for when the
             packager starts bundling the stub (future Wave 2 worker).
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
        # Wave 69: retain parsed_items so the semantic graph stage can
        # reconstruct objectives_metadata (list of LO dicts shaped like
        # JSON-LD learningObjectives[]) for the Wave 66 targets_concept_from_lo
        # rule. Previously the call site passed objectives_metadata=None,
        # leaving the rule to fire on empty input.
        self._parsed_items = parsed_items

        # Pre-chunking: detect corpus-wide boilerplate (footers / template chrome)
        # and build the set of valid outcome IDs for referential-integrity checks.
        self._boilerplate_spans = self._detect_corpus_boilerplate(parsed_items)
        self._valid_outcome_ids = self._build_valid_outcome_ids()

        # Stage 3
        print("[3/6] Chunking content into pedagogical units...")
        chunks = self._chunk_content(parsed_items)

        # Stage 4
        print("[4/6] Writing chunks...")
        self._write_chunks(chunks)

        # Stage 5
        print("[5/6] Generating metadata...")
        concept_graph = self._generate_concept_graph(chunks)
        pedagogy_graph = self._generate_pedagogy_graph(chunks)
        manifest = self._generate_manifest(title, concept_graph=concept_graph)
        corpus_stats = self._generate_corpus_stats()
        quality_report = self._generate_quality_report(chunks)
        # Typed-edge concept graph (additive to concept_graph). Rule-based
        # by default; LLM escalation opt-in via self.typed_edges_llm.
        # Wave 30 Gap 4: always build course_data (the empty-LOs shell is
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

            # Worker M1 diagnostic (§4.4a H5 detection): if a JSON-LD
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
                # REC-JSL-03 (Wave 3, Worker M): page-level union of every
                # distinct data-cf-objective-ref on the page. Used as
                # fallback attachment when a chunk can't be mapped to a
                # specific section in _extract_objective_refs.
                "objective_refs": parsed.objective_refs,
                # Wave 10: page-level aggregated source_references (full
                # SourceReference dicts). Threaded into _create_chunk so
                # chunks carry source.source_references[] end-to-end.
                # Absence = pre-Wave-9 corpus; downstream treats as
                # "unknown", not an error.
                "source_references": parsed.source_references,
                # Worker M1 diagnostic flags (§4.4a H5 detection)
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

    def _chunk_content(self, parsed_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        chunk_counter = 1
        prefix = f"{self.course_code.lower()}_chunk_"
        prev_chunk_id: Optional[str] = None
        current_module_id: Optional[str] = None
        current_lesson_id: Optional[str] = None
        position_in_module = 0

        # Record which pages (by lesson_id = item["item_id"]) carried at least
        # one misconception in their parsed JSON-LD. This is the proper
        # denominator for misconceptions_present_rate: without it, pages that
        # never declared misconceptions in the first place dilute the
        # "silently dropped" signal we want to surface in quality_report.
        self._pages_with_misconceptions = {
            item["item_id"]
            for item in parsed_items
            if item.get("misconceptions")
        }

        for item in parsed_items:
            # Reset position counter when module changes
            if item["module_id"] != current_module_id:
                current_module_id = item["module_id"]
                position_in_module = 0

            # Break the follows_chunk chain at every lesson/module boundary so
            # downstream consumers can treat each lesson as its own pedagogical
            # sequence (see VERSIONING.md §3, defect #3).
            if item["item_id"] != current_lesson_id:
                current_lesson_id = item["item_id"]
                prev_chunk_id = None

            # Strip assessment feedback from quiz/self-check content
            raw_html = item["raw_html"]
            if item["resource_type"] == "quiz":
                raw_html = self._strip_assessment_feedback(raw_html)

            # Defensive boilerplate strip: even if Courseforge emits the footer
            # inside template-chrome, legacy packages may still embed it in body.
            if self._boilerplate_spans:
                raw_html, _ = strip_boilerplate(raw_html, self._boilerplate_spans)

            if not item["sections"]:
                # No sections — chunk the whole item as one piece
                text = self._extract_plain_text(raw_html)
                if item["resource_type"] == "quiz":
                    text = self._strip_feedback_from_text(text)
                if text.strip():
                    item_chunks = self._chunk_text_block(
                        text=text,
                        html=raw_html,
                        item=item,
                        heading=item["title"],
                        chunk_type=self._type_from_resource(item["resource_type"]),
                        prefix=prefix,
                        start_id=chunk_counter,
                        follows_chunk_id=prev_chunk_id,
                        position_in_module=position_in_module,
                    )
                    chunks.extend(item_chunks)
                    chunk_counter += len(item_chunks)
                    if item_chunks:
                        prev_chunk_id = item_chunks[-1]["id"]
                        position_in_module += len(item_chunks)
                continue

            # Merge adjacent small sections into larger pedagogical units
            merged = self._merge_small_sections(item["sections"])

            for heading, text, chunk_type, section_source_ids in merged:
                if not text.strip():
                    continue
                # Strip feedback from quiz section text (sections were parsed before HTML stripping)
                if item["resource_type"] == "quiz":
                    text = self._strip_feedback_from_text(text)
                # Sections were parsed before boilerplate detection, so strip here too.
                if self._boilerplate_spans:
                    text, _ = strip_boilerplate(text, self._boilerplate_spans)
                if not text.strip():
                    continue
                html_block = self._extract_section_html(raw_html, heading)
                item_chunks = self._chunk_text_block(
                    text=text,
                    html=html_block,
                    item=item,
                    heading=heading,
                    chunk_type=chunk_type,
                    prefix=prefix,
                    start_id=chunk_counter,
                    follows_chunk_id=prev_chunk_id,
                    position_in_module=position_in_module,
                    section_source_ids=section_source_ids,
                )
                chunks.extend(item_chunks)
                chunk_counter += len(item_chunks)
                if item_chunks:
                    prev_chunk_id = item_chunks[-1]["id"]
                    position_in_module += len(item_chunks)

        self.stats["total_chunks"] = len(chunks)
        print(f"  Generated {len(chunks)} chunks")
        return chunks

    # Wave 10: role-precedence ranking for merging source_references across
    # multiple sections that collapse into one chunk. Lower integer = stronger
    # (primary overrides contributing, contributing overrides corroborating).
    _SOURCE_ROLE_PRECEDENCE = {"primary": 0, "contributing": 1, "corroborating": 2}

    def _merge_section_source_ids(
        self, accumulated: List[str], section_source_ids: List[str]
    ) -> List[str]:
        """Union two sourceId-string lists, dedupe, preserve insertion order.

        Used while collapsing multiple sections into one chunk. The chunk's
        aggregate source_references[] must contain every distinct sourceId
        from every merged section; role-precedence is enforced downstream in
        ``_resolve_chunk_source_references`` when the strings are paired
        with their full SourceReference dicts.
        """
        seen = {sid for sid in accumulated}
        for sid in section_source_ids:
            if sid and sid not in seen:
                seen.add(sid)
                accumulated.append(sid)
        return accumulated

    def _merge_small_sections(
        self, sections
    ) -> List[Tuple[str, str, str, List[str]]]:
        """
        Merge adjacent sections that are below MIN_CHUNK_SIZE into combined blocks.

        Returns list of (heading, combined_text, chunk_type, merged_source_ids)
        tuples. ``merged_source_ids`` is the union of every section's
        ``data-cf-source-ids`` attribute (stringified) across all sections
        that collapsed into the same chunk (Wave 10); dedupe + insertion-
        order preserved so downstream role-precedence resolution stays
        deterministic.
        """
        merged: List[Tuple[str, str, str, List[str]]] = []
        buffer_heading = ""
        buffer_text = ""
        buffer_wc = 0
        buffer_type = "explanation"
        buffer_source_ids: List[str] = []

        for section in sections:
            section_type = self._type_from_heading(section.heading)
            section_src = list(getattr(section, "source_references", []) or [])

            if buffer_wc == 0:
                # Start a new buffer
                buffer_heading = section.heading
                buffer_text = section.content
                buffer_wc = section.word_count
                buffer_type = section_type
                buffer_source_ids = list(section_src)
            elif buffer_wc + section.word_count <= self.MAX_CHUNK_SIZE:
                # Merge into buffer
                buffer_text += "\n\n" + section.content
                buffer_wc += section.word_count
                # Keep the first heading but prefer non-trivial types
                if buffer_type == "explanation" and section_type != "explanation":
                    buffer_type = section_type
                self._merge_section_source_ids(buffer_source_ids, section_src)
            else:
                # Flush buffer and start new
                merged.append((buffer_heading, buffer_text, buffer_type, buffer_source_ids))
                buffer_heading = section.heading
                buffer_text = section.content
                buffer_wc = section.word_count
                buffer_type = section_type
                buffer_source_ids = list(section_src)

        # Flush remaining
        if buffer_text.strip():
            merged.append((buffer_heading, buffer_text, buffer_type, buffer_source_ids))

        return merged

    def _chunk_text_block(
        self, text: str, html: str, item: Dict[str, Any],
        heading: str, chunk_type: str, prefix: str, start_id: int,
        follows_chunk_id: Optional[str] = None,
        position_in_module: int = 0,
        section_source_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Split a text block into chunks of appropriate size.

        Each chunk carries provenance:
          - ``source.html_xpath``: absolute xpath to the container element
            whose descendant plain-text includes this chunk's text. For
            sectioned items this is the heading's parent (typically
            ``<main>``, ``<article>``, ``<section>``, or ``<body>``); for
            no-section items it is ``<body>`` itself.
          - ``source.char_span: [start, end]``: character offsets into the
            container's plain-text (the result of ``resolve_xpath``) where
            ``chunk.text`` begins and ends. Offsets are computed by
            ``str.find`` into the container text — the same path an
            auditor walks during round-trip recovery. For sentence-split
            blocks, sibling spans are disjoint and (modulo the single-
            space sentence joiner) contiguous, so the section is
            recoverable by concatenating slices in chunk-id order.

        See docs/compliance/audit-trail.md for the round-trip contract.
        """
        word_count = len(text.split())
        chunks: List[Dict[str, Any]] = []

        # Resolve the container xpath once per call. Heading's parent for
        # sectioned items (so descendant text includes the section body);
        # <body> for whole-page items.
        raw_html_for_xpath = item.get("raw_html", "") or html
        container_xpath: Optional[str] = None
        if heading and heading != item.get("title"):
            container_xpath = find_section_container_xpath(raw_html_for_xpath, heading)
        if not container_xpath:
            container_xpath = find_body_xpath(raw_html_for_xpath)

        # Resolve the container's plaintext once so we can compute char_span
        # by string search (the auditor's round-trip path).
        container_text = resolve_xpath(raw_html_for_xpath, container_xpath) or ""

        def _locate(needle: str, search_from: int = 0) -> List[int]:
            """Return [start, end] of ``needle`` in the container text.

            Falls back to a whitespace-normalised prefix search when the
            exact find fails (typical drift: SC canonicalisation, feedback
            strip). Never silently drops the provenance — if no anchor can
            be located at all, emit ``[search_from, search_from + len]``
            relative to the container text so sibling spans stay
            non-decreasing.
            """
            if container_text and needle:
                idx = container_text.find(needle, search_from)
                if idx >= 0:
                    return [idx, idx + len(needle)]
                # Whitespace-normalised prefix fallback: find the first
                # 8-word window of the needle in the collapsed container.
                collapsed_container = " ".join(container_text.split())
                collapsed_needle = " ".join(needle.split())
                prefix = " ".join(collapsed_needle.split()[:8])
                if prefix:
                    idx = collapsed_container.find(prefix, search_from)
                    if idx >= 0:
                        return [idx, idx + len(collapsed_needle)]
            return [search_from, search_from + len(needle)]

        # Worker N (REC-ID-01): resolve a stable per-source locator for
        # content-hash IDs. ``item_path`` is the IMSCC-relative HTML file
        # path and is stable across re-runs; fall back to module/lesson
        # composite if a parser variant ever omits it.
        source_locator = item.get("item_path") or f"{item['module_id']}/{item['item_id']}"

        if word_count <= self.MAX_CHUNK_SIZE:
            # Fits in one chunk.
            char_span = _locate(text, search_from=0)
            chunks.append(self._create_chunk(
                chunk_id=_generate_chunk_id(prefix, start_id, text, source_locator),
                text=text, html=html, item=item,
                section_heading=heading, chunk_type=chunk_type,
                follows_chunk_id=follows_chunk_id,
                position_in_module=position_in_module,
                html_xpath=container_xpath,
                char_span=char_span,
                section_source_ids=section_source_ids,
            ))
        else:
            # Split by sentences. Locate each sub_text independently,
            # anchored after the previous sibling's end so spans stay
            # disjoint and contiguous.
            sub_texts = self._split_by_sentences(text, self.TARGET_CHUNK_SIZE)
            prev_end = 0
            # Worker N (REC-ID-01): track the last emitted chunk id directly
            # rather than re-deriving it from position — under content-hash
            # mode the previous chunk's ID is only knowable once generated.
            last_chunk_id = follows_chunk_id
            for i, sub_text in enumerate(sub_texts):
                part_heading = f"{heading} (part {i + 1})" if len(sub_texts) > 1 else heading
                prev_id = last_chunk_id
                this_chunk_id = _generate_chunk_id(prefix, start_id + i, sub_text, source_locator)
                char_span = _locate(sub_text, search_from=prev_end)
                # Keep spans non-decreasing even if the locator fallback
                # collided with an earlier part's text.
                if char_span[0] < prev_end:
                    char_span = [prev_end, prev_end + (char_span[1] - char_span[0])]
                prev_end = char_span[1]
                chunks.append(self._create_chunk(
                    chunk_id=this_chunk_id,
                    text=sub_text, html="" if i > 0 else html, item=item,
                    section_heading=part_heading, chunk_type=chunk_type,
                    follows_chunk_id=prev_id,
                    position_in_module=position_in_module + i,
                    html_xpath=container_xpath,
                    char_span=char_span,
                    section_source_ids=section_source_ids,
                ))
                last_chunk_id = this_chunk_id

        return chunks

    def _create_chunk(
        self, chunk_id: str, text: str, html: str, item: Dict[str, Any],
        section_heading: str, chunk_type: str,
        follows_chunk_id: Optional[str] = None,
        position_in_module: int = 0,
        html_xpath: Optional[str] = None,
        char_span: Optional[List[int]] = None,
        section_source_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        words = text.split()
        word_count = len(words)
        tokens_estimate = int(word_count * 1.3)

        # Canonicalise WCAG SC references in prose before concept-tag
        # extraction so text-based detection sees the single canonical form.
        text = canonicalize_sc_references(text)

        concept_tags = self._extract_concept_tags(text, item)
        difficulty = self._determine_difficulty(text, item)

        source: Dict[str, Any] = {
            "course_id": self.course_code,
            "module_id": item["module_id"],
            "module_title": item["module_title"],
            "lesson_id": item["item_id"],
            "lesson_title": item["title"],
            "resource_type": item["resource_type"],
            "section_heading": section_heading,
            "position_in_module": position_in_module,
        }
        # Audit-trail provenance (Section 508 / ADA Title II). Every chunk
        # ties back to the source IMSCC HTML element it was derived from.
        # See docs/compliance/audit-trail.md for the round-trip contract.
        if html_xpath:
            source["html_xpath"] = html_xpath
        if char_span is not None:
            source["char_span"] = list(char_span)
        # Carry the IMSCC-relative path so auditors can open the source
        # file without walking imsmanifest.xml.
        if item.get("item_path"):
            source["item_path"] = item["item_path"]

        # Wave 10: fold DART source provenance into source.source_references[].
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
        # authoritative shape is missing (pre-Wave-9 corpora), the field is
        # omitted entirely — consumers treat absence as 'unknown'.
        resolved_refs = self._resolve_chunk_source_references(
            item=item,
            section_heading=section_heading,
            section_source_ids=section_source_ids or [],
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
            # REC-JSL-03 (Wave 3, Worker M): pass section_heading through
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
        bloom_level, content_type_label, key_terms, section_trace = self._extract_section_metadata(
            item, section_heading
        )
        bloom_source = "section_jsonld" if bloom_level else None

        # Merge structured JSON-LD keyTerms into concept_tags. These are the
        # highest-fidelity domain vocabulary Courseforge emits; leaving them
        # in chunk["key_terms"] only meant the concept graph missed them.
        for kt in key_terms or []:
            term = kt.get("term") if isinstance(kt, dict) else kt
            tag = normalize_tag(term or "")
            if not tag or len(tag) < 3 or tag in concept_tags:
                continue
            if (self.OBJECTIVE_CODE_RE.match(tag)
                    or self.WEEK_PREFIX_RE.match(tag)
                    or tag in self.NON_CONCEPT_TAGS):
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

        chunk["bloom_level"] = bloom_level
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

        # Wave 69: propagate Wave 57 targetedConcepts[] from LOs onto chunks
        # whose learning_outcome_refs cite those LOs. Each chunk entry is
        # {"concept": <slug>, "bloom_level": <canonical level>} — a Bloom-
        # qualified LO→concept binding that downstream consumers (retrieval,
        # training-synthesis, SHACL validation) can key off of without
        # re-walking the LO list. Deduplicated across LOs by (concept,
        # bloom_level); preserves the first-seen bloom level when the same
        # concept shows up under multiple Bloom levels across different LOs
        # (matches the Wave 66 rule's first-wins dedup policy).
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
        # Deterministic extractive summary — see Trainforge/generators/summary_factory.py.
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

        # Worker M1 (§4.4a diagnostic): temporary trace recording where each
        # enrichment field came from. Removed by the Worker M2 fix PR.
        chunk["_metadata_trace"] = {
            "content_type_label": section_trace.get("content_type_label", "none"),
            "key_terms": section_trace.get("key_terms", "none"),
            "bloom_level": bloom_source or (
                "section_jsonld" if bloom_level and section_trace.get("content_type_label") == "jsonld_section_match" else "none"
            ),
            "misconceptions": "jsonld_page_misconceptions" if chunk.get("misconceptions") else (
                "none_jsonld_parse_failed" if item.get("_jsonld_parse_failed") else "none"
            ),
        }

        # Stamp the chunk schema version on every chunk so downstream
        # readers can gate on capabilities without re-reading manifest.json.
        chunk["schema_version"] = CHUNK_SCHEMA_VERSION

        # REC-PRV-01 (Worker P Wave 4.1): stamp run_id + created_at on every
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

        self.stats["total_words"] += word_count
        self.stats["total_tokens_estimate"] += tokens_estimate
        self.stats["chunk_types"][chunk_type] += 1
        self.stats["difficulty_distribution"][difficulty] += 1
        self._all_concept_tags.update(concept_tags)

        return chunk

    def _resolve_chunk_source_references(
        self,
        *,
        item: Dict[str, Any],
        section_heading: str,
        section_source_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Wave 10: resolve the chunk's source_references[] array.

        Walks the precedence chain and returns a list of full
        SourceReference dicts (one per unique sourceId). Returns an empty
        list when no references are available (pre-Wave-9 corpus or a
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
        # full SourceReference dicts when Wave 9 input is present.
        for entry in item.get("source_references", []) or []:
            if isinstance(entry, dict):
                _add(entry)

        # 2. Section-level JSON-LD override: if a JSON-LD section matches
        # this chunk's heading and declares its own sourceReferences, add
        # them. The parser's _build_page_source_refs already merged these
        # into item["source_references"] so step 1 typically covers this,
        # but we re-walk here to ensure per-chunk specificity when
        # sections carry refs that aren't in the page-level set.
        chunk_heading_norm = re.sub(
            r'\s*\(part\s+\d+\)\s*$', '', section_heading or ''
        ).lower()
        cf_meta = item.get("courseforge_metadata") or {}
        for sec in cf_meta.get("sections", []) or []:
            if not isinstance(sec, dict):
                continue
            if sec.get("heading", "").lower() != chunk_heading_norm:
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
        self, item: Dict[str, Any], section_heading: str
    ) -> Tuple[Optional[str], Optional[str], List[Dict[str, str]], Dict[str, str]]:
        """Extract bloom_level, content_type_label, and key_terms for a section.

        Checks JSON-LD sections metadata first, then falls back to
        ContentSection data-cf-* attributes.

        Returns a 4-tuple: (bloom_level, content_type_label, key_terms, trace).
        ``trace`` is a Worker M1 diagnostic (VERSIONING.md §4.4a) naming the
        source path for each field. Values:

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
        trace: Dict[str, str] = {
            "content_type_label": "none",
            "key_terms": "none",
        }

        # Normalize heading: strip "(part N)" suffix added by _chunk_text_block
        # so multi-part chunks still match their JSON-LD / data-cf-* metadata.
        chunk_heading = re.sub(r'\s*\(part\s+\d+\)\s*$', '', section_heading).lower()

        # Signals for hypothesis discrimination (Worker M1 instrumentation).
        cf_meta = item.get("courseforge_metadata")
        jsonld_has_sections = bool(cf_meta and cf_meta.get("sections"))
        jsonld_parse_failed = bool(item.get("_jsonld_parse_failed"))
        section_match_found = False

        # Try JSON-LD sections metadata
        if jsonld_has_sections:
            for sec in cf_meta["sections"]:
                if sec.get("heading", "").lower() == chunk_heading:
                    section_match_found = True
                    sec_content_type = sec.get("contentType")
                    if sec_content_type:
                        content_type_label = sec_content_type
                        trace["content_type_label"] = "jsonld_section_match"
                    bloom_range = sec.get("bloomRange", [])
                    if bloom_range:
                        bloom_level = bloom_range[0] if isinstance(bloom_range, list) else bloom_range
                    for kt in sec.get("keyTerms", []):
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
        # NOTE: the original gate is ``if not content_type_label`` which is
        # exactly the H3 short-circuit — if JSON-LD provided contentType but
        # not keyTerms, the data-cf-* path never fills key_terms. Preserved
        # here verbatim so the diagnostic sees the current (un-fixed) behaviour.
        if not content_type_label:
            for section in item.get("sections", []):
                if section.heading.lower() == chunk_heading:
                    if section.content_type:
                        content_type_label = section.content_type
                        trace["content_type_label"] = "data_cf_fallback"
                    if section.key_terms:
                        key_terms = [{"term": t, "definition": ""} for t in section.key_terms]
                        trace["key_terms"] = "data_cf_fallback"
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

        return bloom_level, content_type_label, key_terms, trace

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
        mapping = {
            "quiz": "assessment_item",
            "overview": "overview",
            "summary": "summary",
            "discussion": "exercise",
            "application": "exercise",
        }
        return mapping.get(resource_type, "explanation")

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

    # Common educational concept patterns for text-based extraction
    CONCEPT_PATTERNS: Dict[str, List[str]] = {
        "learning-theory": ["learning theory", "learning theories"],
        "behaviorism": ["behaviorism", "behaviorist"],
        "cognitivism": ["cognitivism", "cognitivist", "information processing"],
        "constructivism": ["constructivism", "constructivist"],
        "connectivism": ["connectivism", "connectivist", "networked learning"],
        "instructional-design": ["instructional design"],
        "addie": ["addie model", "addie"],
        "backward-design": ["backward design", "understanding by design"],
        "cognitive-load": ["cognitive load", "intrinsic load", "extraneous load", "germane load"],
        "multimedia-learning": ["multimedia learning", "multimedia principle", "mayer"],
        "blooms-taxonomy": ["bloom's taxonomy", "blooms taxonomy", "bloom's", "higher-order thinking"],
        "assessment": ["assessment", "formative assessment", "summative assessment"],
        "rubric": ["rubric"],
        "accessibility": ["accessibility", "accessible", "wcag"],
        "udl": ["universal design for learning", "udl"],
        "oscqr": ["oscqr", "course quality"],
        "blended-learning": ["blended learning", "hybrid learning"],
        "online-learning": ["online learning", "online instruction", "distance learning"],
        "synchronous": ["synchronous"],
        "asynchronous": ["asynchronous"],
        "scaffolding": ["scaffolding", "zone of proximal development"],
        "engagement": ["student engagement", "learner engagement"],
        "community-of-inquiry": ["community of inquiry", "coi framework"],
        "alignment": ["constructive alignment", "learning objectives", "learning outcomes"],
        "feedback": ["feedback", "timely feedback"],
    }

    # Pattern for course/terminal/learning objective codes (CO-01, TO-08, LO-003, etc.)
    OBJECTIVE_CODE_RE = re.compile(r'^[a-z]{2}-\d{2,3}$')
    # Week prefix pattern (w01-, w02-) used by Courseforge JSON-LD but absent in course.json
    WEEK_PREFIX_RE = re.compile(r'^w\d{2}-', re.IGNORECASE)

    # Non-concept tags to filter out (generic metadata, not knowledge concepts)
    NON_CONCEPT_TAGS = {
        "estimated-time", "time", "minutes", "hours",
        # Bloom verbs (pedagogical intent, not domain concepts)
        "define", "list", "recall", "identify", "name", "state",
        "explain", "describe", "summarize", "interpret", "paraphrase",
        "apply", "demonstrate", "implement", "solve", "use", "execute",
        "analyze", "differentiate", "examine", "compare", "contrast", "organize",
        "evaluate", "assess", "critique", "judge", "justify", "argue",
        "create", "design", "develop", "construct", "produce", "formulate",
        # Course logistics
        "initial-post", "replies", "due", "guidelines",
        "correct", "incorrect", "submit", "deadline", "grading",
        "readings", "resources", "learning-objectives",
    }

    def _extract_concept_tags(self, text: str, item: Dict[str, Any]) -> List[str]:
        tags: List[str] = []

        # Key concepts from HTML parser (bold terms, definitions)
        for concept in item.get("key_concepts", []):
            tag = normalize_tag(concept)
            if not tag or len(tag) < 3:
                continue
            # Collapse known WCAG SC tag-form drift onto the canonical tag
            # before any filter or dedupe (§4.5 canonicalization).
            from Trainforge.rag.wcag_canonical_names import canonicalize_sc_tag
            tag = canonicalize_sc_tag(tag)
            if tag in tags:
                continue
            # Skip objective codes (co-01, to-01, w01-co-01) and non-concept tags
            if (self.OBJECTIVE_CODE_RE.match(tag)
                    or self.WEEK_PREFIX_RE.match(tag)
                    or tag in self.NON_CONCEPT_TAGS):
                continue
            tags.append(tag)

        # Text-based concept detection (pedagogy-only patterns).
        text_lower = text.lower()
        for tag, patterns in self.CONCEPT_PATTERNS.items():
            if tag not in tags and any(p in text_lower for p in patterns):
                tags.append(tag)

        # Per-course domain concept seeds. Pedagogy filter still applies
        # below since seeds are authored per course; a well-formed seed
        # list won't collide with NON_CONCEPT_TAGS, but we defend anyway.
        for canonical, patterns in self.domain_concept_seeds:
            if canonical in tags:
                continue
            if (self.OBJECTIVE_CODE_RE.match(canonical)
                    or self.WEEK_PREFIX_RE.match(canonical)
                    or canonical in self.NON_CONCEPT_TAGS):
                continue
            if any(p.search(text) for p in patterns):
                tags.append(canonical)

        return tags[:20]

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
          3. REC-JSL-03 (Wave 3): ``data-cf-objective-ref`` on
             ``.activity-card`` / ``.self-check`` elements. Preferred
             section-scoped (matching the chunk's heading) with page-level
             fallback when no section matches.

        Case policy: controlled by ``TRAINFORGE_PRESERVE_LO_CASE``. Default
        (unset / non-``true``) lowercases every ref for backward-compat
        with existing LibV2 chunks. When ``TRAINFORGE_PRESERVE_LO_CASE=true``
        refs pass through with their source casing (still stripped and
        week-prefix-folded — ``WEEK_PREFIX_RE`` has ``re.IGNORECASE``).

        Default will flip in Wave 4's structural migration; until then
        enabling case preservation means downstream ``valid_outcome_ids``
        sites (at L2561/2569/2783/2792) and ``align_chunks.py`` still
        lowercase, so cross-artifact joins need case-folded comparison.
        See ``plans/kg-quality-review-2026-04/worker-m-subplan.md`` §2.3.
        """
        preserve_case = (
            os.getenv("TRAINFORGE_PRESERVE_LO_CASE", "").lower() == "true"
        )

        def _normalize(raw: str) -> str:
            """Apply case policy + week-prefix stripping to a single ref."""
            base = raw.strip() if preserve_case else raw.lower().strip()
            # Strip week prefix (w01-, W01-, w02-, ...) to align with
            # course.json format. WEEK_PREFIX_RE is case-insensitive.
            return self.WEEK_PREFIX_RE.sub('', base)

        refs: List[str] = []

        # (1) Structured objective IDs from parser (JSON-LD or data-cf-*).
        for lo in item.get("learning_objectives", []):
            obj_id = lo.id if hasattr(lo, "id") else lo.get("id")
            if obj_id:
                normalized = _normalize(obj_id)
                if normalized and normalized not in refs:
                    refs.append(normalized)

        # (2) Fallback: regex extraction from key_concepts when no
        # structured IDs were available. Preserves prior behavior of
        # returning-early when refs already populated from (1).
        if not refs:
            for concept in item.get("key_concepts", []):
                tag = normalize_tag(concept)
                if tag and self.OBJECTIVE_CODE_RE.match(tag) and tag not in refs:
                    refs.append(tag)

        # (3) REC-JSL-03: merge in activity/self-check objective refs.
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
            normalized = _normalize(raw_ref)
            if normalized and normalized not in refs:
                refs.append(normalized)

        return refs

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
        extractor = HTMLTextExtractor()
        extractor.feed(html)
        return extractor.get_text()

    @staticmethod
    def _extract_section_html(html: str, heading: str) -> str:
        if not heading:
            return ""
        # Match from the opening <hN> tag containing this heading text
        # through to the next <hN> tag or end of string
        pattern = r"<h[1-6][^>]*>\s*" + re.escape(heading) + r".*?(?=<h[1-6]|$)"
        match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
        return match.group(0) if match else ""

    @staticmethod
    def _strip_assessment_feedback(html: str) -> str:
        """Remove answer feedback from quiz/self-check HTML before text extraction.

        Courseforge quizzes embed correct/incorrect feedback in
        <div class="sc-feedback"> blocks and data-correct attributes on labels.
        This strips that content so assessment chunks contain only question stems
        and answer options without revealing correctness.
        """
        # Remove feedback divs (Courseforge self-check pattern)
        cleaned = re.sub(
            r'<div\s+class="sc-feedback"[^>]*>.*?</div>',
            '', html, flags=re.DOTALL | re.IGNORECASE,
        )
        # Remove data-correct attributes from labels
        cleaned = re.sub(
            r'\s+data-correct="[^"]*"',
            '', cleaned,
        )
        return cleaned

    @staticmethod
    def _strip_feedback_from_text(text: str) -> str:
        """Remove residual feedback markers from plain text extraction.

        Handles both line-level and inline feedback patterns since the text
        extractor often concatenates feedback inline with answer options.
        """
        # Remove inline feedback: "Correct. <explanation>" or "Incorrect. <explanation>"
        # These appear after answer option text, running to the next answer option or end
        text = re.sub(
            r'\s*(?:Correct|Incorrect)\.\s+[^.]*(?:\.[^A-Z])*\.?',
            '', text,
        )
        # Also remove standalone lines
        lines = text.split('\n')
        filtered = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Correct.") or stripped.startswith("Incorrect."):
                continue
            filtered.append(line)
        return '\n'.join(filtered)

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

        # Worker I (REC-CTR-01): opt-in chunk validation against
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
        return {
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
        from JSON-LD ``misconceptions[]`` during ``_chunk_content``). We map
        each entry to a misconception entity whose ``concept_id`` is the
        chunk's first concept tag — the best available signal for which
        concept the misconception threatens without an explicit author-side
        declaration. When the chunk has no concept tags, the misconception
        is still emitted (so downstream consumers see it) but without
        ``concept_id`` — the rule skips entries lacking ``concept_id``, so
        those simply don't produce an edge.
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
            first_tag = tags[0] if tags else None
            for entry in raw:
                if isinstance(entry, dict):
                    statement = (entry.get("misconception") or "").strip()
                    correction = (entry.get("correction") or "").strip()
                    explicit_cid = (entry.get("concept_id") or "").strip() or None
                    # Wave 69: Bloom level (canonicalized lowercase in the
                    # html_content_parser misconception normalizer) now
                    # participates in the seed so Bloom-distinct
                    # misconceptions emit distinct IDs. Breaking change: old
                    # corpora re-chunked under this wave will see new
                    # misconception IDs (documented below).
                    bloom_level = (entry.get("bloom_level") or "").strip()
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
                # Wave 69: seed extended with bloom_level so two misconceptions
                # that share statement + correction text but target different
                # Bloom cognitive demands (e.g., apply-level vs analyze-level
                # misreading of the same concept) emit distinct IDs. Old
                # corpora without Wave 60 bloomLevel on misconceptions feed an
                # empty string here and keep the pre-Wave-69 hash stable *for
                # the bloom-less path* — but any misconception that now carries
                # a bloomLevel will hash differently than it did pre-wave.
                seed = f"{statement}|{correction}|{bloom_level}"
                mc_id = "mc_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
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
                if not concept_id and first_tag:
                    concept_id = _make_concept_id(first_tag, course_id)
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
        chunk ID doubles as ``source_chunk_id`` so Wave 11
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
        can fire. Both were previously always ``None`` at this call site,
        leaving those rule emitters inert.

        Wave 69: ``objectives_metadata`` is built from the parsed JSON-LD
        ``learningObjectives[]`` across every page so the Wave 66
        ``targets_concept_from_lo`` rule (which previously fired on empty
        input) can materialize the Wave 57 ``targetedConcepts[]`` as
        typed ``targets-concept`` edges.
        """
        from Trainforge.rag.typed_edge_inference import build_semantic_graph

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
        """Wave 69: derive ``objectives_metadata`` for build_semantic_graph.

        The Wave 66 rule ``targets_concept_from_lo`` expects a list of LO
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
        """
        by_id: Dict[str, Dict[str, Any]] = {}
        for item in parsed_items:
            # Path 1: direct JSON-LD payload (preferred — exact emit shape).
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

            # Path 2: reconstruct from parsed LearningObjective dataclass
            # when JSON-LD wasn't available or didn't include this LO.
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

    def _generate_pedagogy_graph(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mirror graph of pedagogical and logistics tags.

        Emitted so downstream consumers who want pedagogy signal don't have
        to re-derive it from the chunks, and so nothing is silently dropped
        when pedagogy tags show up in ``concept_tags``.
        """
        return self._build_tag_graph(
            chunks,
            include_tags=PEDAGOGY_TAG_SET | LOGISTICS_TAG_SET,
            graph_kind="pedagogy",
        )

    def _build_tag_graph(
        self,
        chunks: List[Dict[str, Any]],
        *,
        include_tags: Optional[Set[str]] = None,
        exclude_tags: Optional[Set[str]] = None,
        graph_kind: str = "concept",
    ) -> Dict[str, Any]:
        # REC-ID-02 (Wave 4, Worker O): opt-in course-scoped concept IDs.
        # When TRAINFORGE_SCOPE_CONCEPT_IDS=true, node IDs and edge endpoints
        # are emitted as ``f"{course_id}:{slug}"`` and each node carries a
        # ``course_id`` field. Default off → legacy flat-slug behaviour.
        from Trainforge.rag.typed_edge_inference import (
            SCOPE_CONCEPT_IDS,
            _make_concept_id,
        )

        # ``course_code`` may be unset when the graph builder is called on
        # a bare processor (e.g. unit tests using ``__new__``); fall back
        # to an empty string → ``_make_concept_id`` treats empty as "no
        # course_id" and emits flat slugs even under flag-on.
        course_id = getattr(self, "course_code", "") or ""
        tag_frequency: Dict[str, int] = defaultdict(int)
        co_occurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        # REC-LNK-01 (Wave 5.1, Worker S): inverted index from concept node_id
        # to the set of chunk IDs that reference the concept. Always-on
        # additive behaviour (no env var). Stable across re-chunks only when
        # TRAINFORGE_CONTENT_HASH_IDS=true (Worker N's flag); position-based
        # IDs invalidate entries on re-chunk. Using a set per-node avoids
        # duplicate chunk IDs when a chunk lists the same tag twice; sorted
        # to a list at emit time for deterministic output.
        #
        # Note: len(occurrences) counts DISTINCT chunks referencing the
        # concept, which may be less than ``frequency`` (frequency counts
        # total tag mentions — a chunk that lists a tag twice counts twice).
        concept_to_chunks: Dict[str, set] = defaultdict(set)
        # Wave 10: lookup from chunk id → source.source_references[] (if
        # any). Used to copy the first occurrence's refs onto the emitted
        # concept node as ``source_refs[]``. Same additive-optional pattern
        # as occurrences[]: absence = 'unknown' (pre-Wave-9 chunk) → the
        # node is emitted without ``source_refs``.
        chunk_source_refs: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            cid = chunk.get("id")
            if not cid:
                continue
            src = chunk.get("source") or {}
            refs = src.get("source_references") if isinstance(src, dict) else None
            if isinstance(refs, list) and refs:
                chunk_source_refs[cid] = refs

        def _accept(tag: str) -> bool:
            if include_tags is not None and tag not in include_tags:
                return False
            if exclude_tags is not None and tag in exclude_tags:
                return False
            return True

        for chunk in chunks:
            chunk_id = chunk.get("id")
            tags = [t for t in chunk.get("concept_tags", []) if _accept(t)]
            for tag in tags:
                tag_frequency[tag] += 1
                if chunk_id:
                    # Key the inverted index by the SAME node_id the emit
                    # loop below will produce — using _make_concept_id so
                    # the occurrences[] keys align with node["id"] under
                    # either flag state of TRAINFORGE_SCOPE_CONCEPT_IDS.
                    concept_to_chunks[_make_concept_id(tag, course_id)].add(chunk_id)
            for i, a in enumerate(tags):
                for b in tags[i + 1:]:
                    key = tuple(sorted([a, b]))
                    co_occurrence[key] += 1

        sorted_tags = sorted(tag_frequency.items(), key=lambda x: -x[1])
        nodes: List[Dict[str, Any]] = []
        for tag, freq in sorted_tags:
            if freq < 2:
                continue
            node_id = _make_concept_id(tag, course_id)
            # Label stays human-readable (no course_id prefix) regardless
            # of scoping mode; only ``id`` is composite when the flag is on.
            node: Dict[str, Any] = {
                "id": node_id,
                "label": tag.replace("-", " ").title(),
                "frequency": freq,
            }
            if SCOPE_CONCEPT_IDS and course_id:
                node["course_id"] = course_id
            # REC-LNK-01: attach sorted occurrences[] back-reference.
            # Sort is ASCII-ASC on chunk ID string — deterministic across
            # runs, cross-platform stable. Only emit when non-empty so
            # nodes whose tag wasn't present on any chunk with a resolvable
            # chunk_id stay legacy-shaped.
            occurrences = concept_to_chunks.get(node_id)
            if occurrences:
                sorted_occurrences = sorted(occurrences)
                node["occurrences"] = sorted_occurrences
                # Wave 10: populate source_refs[] from occurrences[0] (the
                # first chunk by sorted-ID ordering). Copy the full
                # SourceReference dicts verbatim so the node carries the
                # same authoritative roles as the underlying chunk. Only
                # emit when non-empty — pre-Wave-9 corpora produce empty
                # chunk source_references and therefore empty node
                # source_refs → field omitted.
                first_chunk_refs = chunk_source_refs.get(sorted_occurrences[0])
                if first_chunk_refs:
                    node["source_refs"] = [dict(r) for r in first_chunk_refs]
            nodes.append(node)
        node_ids = {n["id"] for n in nodes}

        edges = []
        for (a, b), weight in co_occurrence.items():
            scoped_a = _make_concept_id(a, course_id)
            scoped_b = _make_concept_id(b, course_id)
            if scoped_a in node_ids and scoped_b in node_ids:
                edges.append({
                    "source": scoped_a,
                    "target": scoped_b,
                    "weight": weight,
                    "relation_type": "co-occurs",
                })

        return {
            "kind": graph_kind,
            "nodes": nodes,
            "edges": edges,
            "generated_at": datetime.now().isoformat(),
        }

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
        referenced_ids = {
            r for c in chunks for r in c.get("learning_outcome_refs", [])
            if r in valid_ids
        }
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
        # reveal. See docs/metrics/flow-metrics.md for full methodology.
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

        # Worker P (v5): package_completeness — flat mean of the five
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

        See ``docs/metrics/flow-metrics.md`` for the full explanation of
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
        # requiring a chunk-schema change (that's Worker E's territory).
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
                "be revisited once Worker E lands chunk-schema provenance."
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
        """True iff ``html`` has non-empty content and every opened tag is closed."""
        if not html or not html.strip():
            return False
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
        broken: List[Dict[str, str]] = []
        for c in chunks:
            for ref in c.get("learning_outcome_refs", []):
                if ref not in valid_outcome_ids:
                    broken.append({"chunk_id": c["id"], "ref": ref})
        return broken

    @staticmethod
    def _resolving_lo_coverage(
        chunks: List[Dict[str, Any]],
        valid_outcome_ids: Set[str],
    ) -> float:
        total = len(chunks) or 1
        resolving = sum(
            1 for c in chunks
            if any(r in valid_outcome_ids for r in c.get("learning_outcome_refs", []))
        )
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
        """
        ids: Set[str] = set()
        if not self.objectives:
            return ids
        for to in self.objectives.get("terminal_objectives", []):
            obj_id = (to.get("id") or "").lower()
            if obj_id:
                ids.add(obj_id)
            for ws in to.get("week_scoped_ids", []) or []:
                if ws:
                    ids.add(ws.lower())
        for ch in self.objectives.get("chapter_objectives", []):
            for obj in ch.get("objectives", []):
                obj_id = (obj.get("id") or "").lower()
                if obj_id:
                    ids.add(obj_id)
                for ws in obj.get("week_scoped_ids", []) or []:
                    if ws:
                        ids.add(ws.lower())
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
        """Worker M1 (§4.4a diagnostic): group chunks by ``_metadata_trace``
        value per enrichment field and compute counts + percentages.

        Emitted alongside ``quality_report.json`` as ``metadata_trace_report.json``.
        Deleted by the Worker M2 fix PR once the root cause is addressed.

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
        self, chunks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Build a pedagogy model grounded in the actual chunk set.

        Emits module_sequence (order + per-module stats), bloom_progression
        (per-module Bloom distribution), and prerequisite_chain (concepts
        referenced as prereqs after being first introduced earlier). Falls
        back to just the top-level keys when chunks aren't provided so
        older call sites don't break.
        """
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
        bloom_zero = lambda: {
            "remember": 0, "understand": 0, "apply": 0,
            "analyze": 0, "evaluate": 0, "create": 0,
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
        # For each concept tag, record earliest (module_idx, chunk_id) where it
        # appears in concept_tags (definition site) vs prereq_concepts (use site).
        # Valid chain: first use in module index > first definition's module index.
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

        prerequisite_chain = []
        prerequisite_violations = []
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

        Wave 24: result validates against
        ``schemas/knowledge/course.schema.json`` before being returned.
        Schema violations are logged as warnings (best-effort) — the
        canonical shape is still emitted.

        Wave 30 Gap 4: guarantee course.json materialisation. When
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

            for ch in self.objectives.get("chapter_objectives", []):
                for obj in ch.get("objectives", []):
                    outcomes.append({
                        "id": obj["id"].lower(),
                        "statement": obj["statement"],
                        "bloom_level": (obj.get("bloomLevel") or obj.get("bloom_level") or "understand"),
                        "hierarchy_level": "chapter",
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

        # Wave 24: best-effort schema validation against the canonical
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
        # Wave 30 Gap 4: always write course.json (including the
        # empty-learning_outcomes shell with a ``note`` field) so LibV2
        # archival always lands a file and downstream retrieval joins
        # have something to look at. Pre-Wave-30 this was gated on
        # ``self.objectives`` being truthy, so pipeline runs that
        # auto-synthesized objectives without threading the path in
        # never emitted course.json.
        course_data = self._build_course_json(manifest)
        _write(self.output_dir / "course.json", course_data)

        _write(self.corpus_dir / "corpus_stats.json", corpus_stats)
        _write(self.graph_dir / "concept_graph.json", concept_graph)
        if pedagogy_graph is not None:
            _write(self.graph_dir / "pedagogy_graph.json", pedagogy_graph)
        if semantic_graph is not None:
            _write(self.graph_dir / "concept_graph_semantic.json", semantic_graph)
        _write(self.quality_dir / "quality_report.json", quality_report)

        # Pedagogy model (full: module sequence, bloom progression, prereq chain)
        pedagogy = self._build_pedagogy_summary(chunks=chunks)
        _write(self.pedagogy_dir / "pedagogy_model.json", pedagogy)

        # Worker M1 (§4.4a diagnostic): enrichment-trace report alongside
        # quality_report.json. Removed by the Worker M2 fix PR.
        if chunks:
            trace_report = self._generate_enrichment_trace_report(chunks)
            _write(self.quality_dir / "metadata_trace_report.json", trace_report)

        # Training specs
        training_specs = {
            "format": "instruction-following",
            "target_models": ["claude-opus-4-6", "claude-sonnet-4-6"],
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

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Process a Courseforge IMSCC into a Trainforge RAG corpus",
    )
    p.add_argument("--imscc", required=True, help="Path to .imscc file")
    p.add_argument("--course-code", required=True, help="Course code (e.g. SAMPLE_101)")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--objectives", help="Path to objectives JSON (optional)")
    # Wave 2 REC-TAX-01: classification flags accept None sentinels so the
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
    p.add_argument("--llm-provider", default="mock", choices=["mock", "anthropic"],
                   help="LLM provider for alignment stage (default: mock)")
    p.add_argument("--import-to-libv2", action="store_true", help="Import into LibV2 after processing")
    p.add_argument("--synthesize", action="store_true",
                   help="Synthesize SFT/DPO training pairs from chunks after base processing (Worker C).")
    p.add_argument("--synthesis-provider", default="mock", choices=["mock", "anthropic"],
                   help="Provider for training-pair synthesis (default: mock).")
    p.add_argument("--synthesis-seed", type=int, default=17,
                   help="Base deterministic seed for training-pair synthesis (default: 17).")
    p.add_argument(
        "--typed-edges-llm",
        action="store_true",
        help=(
            "Enable the optional LLM escalation pass for the typed-edge "
            "concept graph. OFF by default — the rule-based path is "
            "deterministic and byte-identical across runs (Worker F spec)."
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
    return p


def main():
    args = build_parser().parse_args()

    # Wave 2 REC-TAX-01: require either a course_metadata.json stub OR a
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
        from Trainforge.align_chunks import main as align_main
        align_args = argparse.Namespace(
            corpus=args.output,
            objectives=args.objectives,
            fields="prereq_concepts,teaching_role,learning_outcome_refs",
            llm_provider=args.llm_provider,
            llm_model="claude-haiku-4-5-20251001",
            dry_run=False,
            verbose=False,
        )
        align_main(align_args)

    # Optional training-pair synthesis stage (Worker C)
    if args.synthesize:
        print("\n[Synthesis] Running training-pair synthesis stage...")
        from Trainforge.synthesize_training import run_synthesis
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
    if args.import_to_libv2:
        print("\n[LibV2] Importing into LibV2...")
        try:
            from LibV2.tools.libv2.importer import import_course as do_import

            # Use processor-resolved fields so stub-driven classification
            # flows into LibV2 import when no CLI flags were set (Wave 2
            # REC-TAX-01).
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
        except Exception as e:
            print(f"[LibV2] Import failed: {e}")
            print("[LibV2] You can import manually later with:")
            print(
                f"  python -m LibV2.tools.libv2.cli import {args.output} "
                f"--domain {processor.domain} --division {processor.division}"
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

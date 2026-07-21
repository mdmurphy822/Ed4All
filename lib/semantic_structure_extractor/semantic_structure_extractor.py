"""
Semantic Structure Extractor

Main module that extracts complete semantic structure from HTML or Markdown content.
Combines heading hierarchy parsing and content block classification to produce
a structured representation of content suitable for presentation generation.

Supports:
- HTML input (SemantiK-processed or generic)
- Markdown input with YAML front matter
- Content profiling (difficulty, concepts)
- Concept graph building
- Presentation schema transformation

Output conforms to schemas/presentation/presentation_schema.json or
textbook_structure.schema.json based on extraction method used.
"""

import difflib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag

from lib.ontology.taxonomy import (
    get_lexicon_apparatus_names,
    strip_leading_ordinal,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SemantiK structure-fidelity guards (Package 1 + 3) — master gate.
#
# Wraps the article-path chapter/section assembly guards (continuation-article
# merge, headingless-wrapper grouping, noncontent/EOC/numbered-apparatus
# heading filtering, post-build sanity diagnostics) AND the Package-3
# ordinal-strip apparatus demotion in ``_is_eoc_section_heading``. Default OFF:
# flag-off is byte-identical to the pre-guard extractor (snapshot-pinned).
# ---------------------------------------------------------------------------
_STRUCTURE_EXTRACT_GUARDS_ENV = "ED4ALL_STRUCTURE_EXTRACT_GUARDS"


def _structure_extract_guards_enabled() -> bool:
    """Whether the Package 1/3 extractor guards are enabled (default OFF)."""
    raw = os.environ.get(_STRUCTURE_EXTRACT_GUARDS_ENV)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Lazily-cached lowercased lexicon apparatus display names, consulted (after
# an ordinal strip) by the guards-gated arm of ``_is_eoc_section_heading`` so a
# numbered banner "10.3 Review Exercises" / "1.4 Practice Test" demotes via the
# data-driven lexicon rather than a hardcoded name.
_LEXICON_APPARATUS_LOWER_CACHE: Optional[frozenset] = None


def _lexicon_apparatus_names_lower() -> frozenset:
    global _LEXICON_APPARATUS_LOWER_CACHE
    if _LEXICON_APPARATUS_LOWER_CACHE is None:
        try:
            _LEXICON_APPARATUS_LOWER_CACHE = frozenset(
                n.strip().lower()
                for n in get_lexicon_apparatus_names()
                if n and n.strip()
            )
        except Exception:  # pragma: no cover - defensive; lexicon always loads
            _LEXICON_APPARATUS_LOWER_CACHE = frozenset()
    return _LEXICON_APPARATUS_LOWER_CACHE


# ---------------------------------------------------------------------------
# Package 2 — outline-anchored section alignment (satellite of the Package-1/3
# guards). Harvests the document's OWN declared ``N.M Title`` structure (the
# chapter-outline zone + ``<nav class="toc">``, INCLUDING fused outline
# paragraphs), aligns body/outline/answer-key heading occurrences against it,
# and regroups sections/chapters by the declared ordinal spine rather than by
# the OCR-inflated article boundaries. Default ON when the Package-1 guards are
# on (opt-out) — it is a REFINEMENT of the guarded path, only ever reached from
# ``_build_chapters_from_articles_guarded``. Undeclared corpora (no harvestable
# outline) fall through UNCHANGED to the Package-1 guarded behavior.
# ---------------------------------------------------------------------------
_STRUCTURE_OUTLINE_ANCHOR_ENV = "ED4ALL_STRUCTURE_OUTLINE_ANCHOR"


def _outline_anchor_enabled() -> bool:
    """Whether Package-2 outline anchoring is enabled.

    Default ON (opt-out): only ``_build_chapters_from_articles_guarded`` calls
    this, so the guards master gate already had to be on to get here. An
    explicit falsey ``ED4ALL_STRUCTURE_OUTLINE_ANCHOR`` reverts to the
    Package-1 guarded behavior (byte-identical to guards-on / anchor-off).
    """
    raw = os.environ.get(_STRUCTURE_OUTLINE_ANCHOR_ENV)
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ``N.M`` section-ordinal shapes. ``_NM_HEADING_RE`` anchors a heading that
# OPENS with an ordinal ("1.4 Multiply and Divide Integers"); ``_NM_SPLIT_RE``
# finds every ordinal inside a run so a FUSED outline paragraph
# ("1.2 Use the Language of Algebra 1.3 Add and Subtract Integers") recovers
# BOTH entries.
_NM_HEADING_RE = re.compile(r"^\s*(\d+\.\d+)\s+(\S.*)$")
_NM_SPLIT_RE = re.compile(r"\b(\d+\.\d+)\b")

# Review / answer-key furniture words. An article whose TITLE carries one of
# these opens the document's trailing review/answer-key zone (OpenStax prints
# the per-section openers as reprints there — see the plan's "1.2/1.3 opener"
# forensics). Shape-based, publisher-neutral; the vocabulary is generic
# academic-apparatus wording, not a publisher name.
_REVIEW_ZONE_RE = re.compile(
    r"\b(review|key terms|key concepts|practice test|cumulative)\b",
    re.IGNORECASE,
)

# A declared ``N.M`` whose MINOR component exceeds this is almost certainly an
# example / figure reference ("EXAMPLE 1.58") leaking into a ``<nav class="toc">``
# or fused paragraph, not a real section ordinal. Pure-shape guard (section
# minors are small contiguous integers); heading-derived entries are exempt
# because a council-typed ``<h3>`` ordinal is trustworthy.
_MAX_SECTION_MINOR = 30

# Fuzzy title-match floor (difflib ratio, OCR tolerance). A body heading whose
# ordinal-stripped, casefolded, whitespace-collapsed text scores at least this
# against a declared entry's title is the same section.
_OUTLINE_TITLE_MATCH_RATIO = 0.80

# Zone preference for the surviving occurrence of a declared section: a real
# body opener beats an outline stub beats an answer-key reprint.
_ZONE_RANK = {"body": 0, "outline": 1, "answer_key": 2}

# Package 2b — multi-source ordinal-UNION harvest. Title-donor priority
# (priority-wins, replacing first-seen-wins): a real body non-apparatus heading
# beats an answer-key heading beats an outline-zone/nav fused split beats a
# fused body-paragraph split. Lower rank == higher priority (won by ``min``).
_TITLE_TIER_BODY_HEADING = 1
_TITLE_TIER_ANSWER_KEY_HEADING = 2
_TITLE_TIER_OUTLINE_FUSED = 3
_TITLE_TIER_BODY_FUSED = 4

# Contiguity belt (extra guard on top of the load-bearing structural-admission
# rule): the accepted minors within one major must form a near-contiguous 1..K
# run. A minor separated from the running kept-run by more than this gap is a
# stray structural false positive (e.g. a "6.15 exercises" cross-reference
# leaking a banner) and is pruned along with the rest of its tail. Generous
# enough to tolerate a single genuinely-missing per-section opener.
_OUTLINE_MINOR_GAP_MAX = 3


def _normalize_outline_text(text: str) -> str:
    """Casefold + whitespace-collapse for fuzzy title comparison."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _ordinal_sort_key(ordinal: str) -> Tuple[int, ...]:
    """Numeric sort key for an ``N.M`` ordinal so ``1.10`` sorts after ``1.9``."""
    try:
        return tuple(int(p) for p in ordinal.split("."))
    except ValueError:  # pragma: no cover - ordinals are regex-validated
        return (0,)


def _split_fused_outline_entries(text: str) -> List[Tuple[str, str]]:
    """Recover ``(ordinal, title)`` pairs from a run of fused outline text.

    Each ordinal owns the text up to the NEXT ordinal, so
    ``"1.2 Use the Language of Algebra 1.3 Add and Subtract Integers"`` yields
    ``[("1.2", "Use the Language of Algebra"), ("1.3", "Add and Subtract
    Integers")]``. Example-number leaks (minor > ``_MAX_SECTION_MINOR``) are
    dropped as a pure-shape guard against ``EXAMPLE N.NN`` references.
    """
    matches = list(_NM_SPLIT_RE.finditer(text))
    out: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        ordinal = m.group(1)
        try:
            if int(ordinal.split(".")[1]) > _MAX_SECTION_MINOR:
                continue
        except (ValueError, IndexError):  # pragma: no cover
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((ordinal, text[start:end].strip()))
    return out


# TOC-like heading texts that should NOT be promoted to chapter titles
# on their own. The SemantiK converter emits many "Contents" h2s when page
# chrome wraps every printed page; if we hand one of those to the
# course planner we end up with a chapter named "Contents" and every
# real chapter demoted to a section. Case-insensitive exact match.
_TOC_HEADING_TEXTS = frozenset({
    "contents",
    "table of contents",
    "toc",
    "index",
})


def _is_toc_heading(text: Optional[str]) -> bool:
    """Whether a heading text is a table-of-contents artifact."""
    if not text:
        return False
    return text.strip().lower() in _TOC_HEADING_TEXTS


# Front-matter / acknowledgments heading texts that must never become a
# chapter or section node. Case-insensitive exact match (after stripping
# a trailing colon). These poison the course planner with chapters like
# "Preface" / "About the Authors". "Foundations" is DELIBERATELY absent —
# it's a legitimate single-word math chapter title (often all-caps
# "FOUNDATIONS") and is distinguished from donor-list lines below.
_FRONT_MATTER_HEADING_TEXTS = frozenset({
    "preface",
    "acknowledgment",
    "acknowledgments",
    "acknowledgement",
    "acknowledgements",
    "about the author",
    "about the authors",
    "dedication",
    "copyright",
    "foreword",
})

# End-of-chapter (EOC) exercise / review / drill section headings that
# OpenStax-style textbooks emit AFTER the teaching content of a section or
# chapter. These are practice/drill material, not primary content —
# promoting one to a content page produced a spurious "Practice Makes
# Perfect" course page (week_01_content_12) in a live run. Two match modes
# keep the predicate high-precision:
#
#   * _EOC_EXACT — matched only as a normalized EXACT heading. Used for
#     short phrases whose words could legitly PREFIX real content
#     ("Practice Testing as a Study Strategy"), so a prefix rule would
#     over-reject.
#   * _EOC_PREFIX — matched as an exact heading OR as a "<phrase><sep>..."
#     prefix (OpenStax appends the section topic, e.g. "Review Exercises:
#     Add Whole Numbers"). Reserved for multi-word phrases so distinctive
#     that a real content title never starts with them. "practice makes
#     perfect" is NOT a prefix of "Practice Problems in Real Analysis", so
#     genuine chapter titles survive.
#
# DELIBERATELY EXCLUDED — the glossary/summary/metacognition family
# ("Self Check", "Key Terms", "Key Concepts", "Chapter Summary"). This
# predicate is SHARED with lib/chunk_heading_sanity.py, whose contract is
# "NEVER demote a real heading" and whose regression suite treats
# "Self Check" as a legitimate chunk section heading. Those headings label
# real glossary/summary CHUNKS in the corpus, so filtering them here would
# corrupt the chunk display heading (not just the topic→content-page list).
# The EOC-exercise headings below are drill banners neither path wants and
# are consistent with chunk_heading_sanity's existing exercise-banner rule.
_EOC_EXACT = frozenset({
    "practice test",       # OpenStax per-chapter practice test
    "everyday math",       # OpenStax applied-exercise category
})
_EOC_PREFIX = (
    "practice makes perfect",   # OpenStax per-section exercise block
    "chapter review exercises", # (checked before "review exercises")
    "section exercises",
    "review exercises",
    "additional practice",
    "writing exercises",
)


# Package 3 — bare per-section drill banners the numbered form ("1.4
# Exercises") yields after the ordinal strip. Generic-academic apparatus
# words (not a publisher name); consulted ONLY on the guards-gated ordinal
# arm below, so the shared ``_is_noncontent_heading`` / chunk-heading-sanity
# path stays byte-identical when the guards flag is off.
_EOC_NUMBERED_BARE = frozenset({"exercises", "exercise"})


def _eoc_core_match(normalized: str) -> bool:
    """Existing exact/prefix EOC match against ``_EOC_EXACT`` / ``_EOC_PREFIX``."""
    if normalized in _EOC_EXACT:
        return True
    for phrase in _EOC_PREFIX:
        if normalized == phrase:
            return True
        # Prefix form: the phrase must be followed by a NON-alphanumeric
        # boundary (space / colon / dash), so "Review Exercises: Solve ..."
        # matches but "Reviewing Exercises" and "Practice Makes Perfection"
        # (next char is a letter) do not.
        if (
            normalized.startswith(phrase)
            and not normalized[len(phrase):len(phrase) + 1].isalnum()
        ):
            return True
    return False


def _eoc_banner_regex() -> "re.Pattern":
    """Regex matching an apparatus-BANNER ordinal ("N.M Exercises" shape).

    Package 2b source (e): a per-section drill banner is STRUCTURAL evidence
    that its ordinal names a real section (every real section prints one), even
    though the banner itself donates no title. The apparatus-phrase alternation
    is drawn ONLY from the existing generic bare-exercises words, the core EOC
    exact/prefix phrases, and the data-driven lexicon apparatus display names —
    never a publisher-specific vocabulary. Longest phrases first so a two-word
    banner ("review exercises") wins over its one-word suffix.
    """
    phrases = (
        set(_EOC_NUMBERED_BARE)
        | set(_EOC_EXACT)
        | set(_EOC_PREFIX)
        | set(_lexicon_apparatus_names_lower())
    )
    alts = "|".join(
        re.escape(p)
        for p in sorted((p for p in phrases if p), key=len, reverse=True)
    )
    return re.compile(r"\b(\d+\.\d+)\s+(?:" + alts + r")\b", re.IGNORECASE)


def _is_eoc_section_heading(normalized: str) -> bool:
    """Whether ``normalized`` (already lowercased, whitespace-collapsed,
    trailing-colon-stripped) is an end-of-chapter exercise/review/summary
    section heading. Exact match for the short glossary/summary phrases;
    exact-or-space-delimited-prefix for the distinctive exercise phrases.

    Package 3 (guards-gated): a numbered per-section drill banner ("1.4
    Exercises", "10.3 Review Exercises") carries a leading ordinal that
    defeats the exact/prefix match. When ``ED4ALL_STRUCTURE_EXTRACT_GUARDS``
    is on, the leading ordinal is stripped and the bare name is re-matched
    against the core EOC set, the generic bare-exercises words, and the
    data-driven lexicon apparatus display names. Flag-off is byte-identical.
    """
    if _eoc_core_match(normalized):
        return True
    if not _structure_extract_guards_enabled():
        return False
    stripped = strip_leading_ordinal(normalized).strip()
    if not stripped or stripped == normalized:
        # No leading ordinal was present -> nothing new to match.
        return False
    if stripped in _EOC_NUMBERED_BARE:
        return True
    if _eoc_core_match(stripped):
        return True
    return stripped in _lexicon_apparatus_names_lower()


# Circled-letter answer markers (U+24D0..U+24D4 = ⓐⓑⓒⓓⓔ) and the
# circled-digit block (U+2460..U+2473 = ①..⑳). OpenStax answer-key tables
# render exercise answers with these glyphs; a heading carrying one is an
# answer-key fragment, never a content title.
_CIRCLED_ANSWER_MARKERS = (
    "ⓐⓑⓒⓓⓔ"          # ⓐⓑⓒⓓⓔ
    "①②③④⑤"          # ①②③④⑤
    "⑥⑦⑧⑨⑩"          # ⑥⑦⑧⑨⑩
)

# "N. " answer-sequence pattern (e.g. "13. 5,846,103 14. 1,458,398"):
# a number followed by a period and a space. Answer keys emit many of
# these; a real title emits at most one (e.g. "1.2 Add Whole Numbers"
# does NOT match because there's no space after the inner period).
_ANSWER_SEQUENCE_RE = re.compile(r"\b\d+\.\s")

# A heading whose non-space characters are >= this fraction digits or
# punctuation is treated as an answer-key numeric fragment. 0.60 leaves
# generous headroom for legitimate titles that merely contain a number
# (e.g. "1.2 Add Whole Numbers" is ~0.13 digit/punct by this measure).
_NUMERIC_HEADING_RATIO = 0.60

# Minimum count of "N. " answer-sequence matches before a heading is
# classed as an answer key on that signal alone. One match is common in
# real titles ("Section 2. Foo"); three+ in one heading is an answer run.
_ANSWER_SEQUENCE_MIN_COUNT = 3

# A donor/foundation list line ("Bill & Melinda Gates Foundation
# National Science Foundation") is distinguished from the math chapter
# title "Foundations" by (a) containing the whole word "Foundation"/
# "Fund" AND (b) being a multi-word phrase with several capitalized
# tokens (a list of proper names). This minimum capitalized-token count
# keeps the single-word "Foundations" title safe.
_DONOR_MIN_CAPITALIZED_TOKENS = 4
_DONOR_FUNDING_RE = re.compile(r"\b(?:foundation|fund)s?\b", re.IGNORECASE)

# WS4 §4 — structure-collapse early warning. When a whole textbook collapses
# into a SINGLE chapter carrying more than this many sections (the 141:1
# collapse signature, typically from answer-key HTML headings being absorbed
# into one chapter), the Stage-1 draft terminal objectives under-represent the
# back of the book. Parity with the provider-side
# ``_COLLAPSE_SECTION_THRESHOLD`` (WS4 §2). Strict ``>`` boundary: exactly 40
# sections does NOT trip the warning; 41 does.
_STRUCTURE_COLLAPSE_SECTION_THRESHOLD = 40


def _is_noncontent_heading(text: Optional[str]) -> bool:
    """Whether a heading is non-content noise that must NOT become a
    chapter/section node.

    Rejects three families of contamination observed in OpenStax-style
    textbook HTML:

    1. Answer-key / numeric-answer fragments — heading text that is
       mostly digits/punctuation, OR carries a circled-answer marker
       (ⓐⓑⓒⓓⓔ / circled digits), OR matches the "N. <number>"
       answer-sequence pattern several times.
    2. Front-matter / acknowledgments — exact-ish matches for "Preface",
       "Acknowledgments", "About the Authors", "Dedication", etc. Plus
       end-of-chapter exercise/review/drill section headings (OpenStax
       EOC family: "Practice Makes Perfect", "Review Exercises", "Practice
       Test", "Everyday Math", ...) matched exact/prefix. The glossary/
       summary family ("Key Terms", "Chapter Summary", "Self Check") is
       deliberately NOT filtered — this predicate is shared with the
       chunk-heading-sanity path, which treats those as real headings.
    3. Donor / foundation list lines — a phrase naming several capitalized
       proper-name tokens that mentions "Foundation"/"Fund". The
       single-word math chapter title "Foundations" is preserved.

    Fail-safe by design: the ratio / count thresholds are deliberately
    generous so a legitimate title containing a number ("1.2 Add Whole
    Numbers", "Section 3.5") survives. The caller is responsible for
    falling back to unfiltered output if EVERYTHING is filtered.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False

    # (1a) Circled-answer markers — unambiguous answer-key glyphs.
    if any(ch in _CIRCLED_ANSWER_MARKERS for ch in stripped):
        return True

    # (1b) Mostly-numeric: fraction of non-space chars that are digits or
    # punctuation. Pure answer rows ("78 41. 900 42. 800") sit near 1.0.
    non_space = [ch for ch in stripped if not ch.isspace()]
    if non_space:
        digit_punct = sum(
            1 for ch in non_space if ch.isdigit() or not ch.isalnum()
        )
        if digit_punct / len(non_space) >= _NUMERIC_HEADING_RATIO:
            return True

    # (1c) Repeated "N. " answer-sequence runs.
    if len(_ANSWER_SEQUENCE_RE.findall(stripped)) >= _ANSWER_SEQUENCE_MIN_COUNT:
        return True

    # (2) Front-matter / acknowledgments — exact match (drop trailing
    # colon, collapse whitespace). Case-insensitive.
    normalized = re.sub(r"\s+", " ", stripped).rstrip(":").strip().lower()
    if normalized in _FRONT_MATTER_HEADING_TEXTS:
        return True

    # (2b) End-of-chapter exercise / review / summary section headings
    # (OpenStax EOC family: "Practice Makes Perfect", "Review Exercises",
    # "Key Terms", "Chapter Summary", ...). Drill/recap material, never a
    # primary content page. High-precision exact/prefix match only.
    if _is_eoc_section_heading(normalized):
        return True

    # (3) Donor / foundation list line. Requires the funding keyword AND
    # several capitalized tokens (a list of proper names) so the bare
    # math chapter title "Foundations" / "FOUNDATIONS" is never caught.
    if _DONOR_FUNDING_RE.search(stripped):
        tokens = stripped.split()
        capitalized = sum(
            1 for tok in tokens
            if tok[:1].isalpha() and tok[:1].isupper()
        )
        # An all-caps single word ("FOUNDATIONS") has one token; a donor
        # list ("Bill & Melinda Gates Foundation National Science
        # Foundation") has many. Require both several capitalized tokens
        # AND more than one whitespace-separated word.
        if len(tokens) > 1 and capitalized >= _DONOR_MIN_CAPITALIZED_TOKENS:
            return True

    return False

from .analysis.concept_graph import ConceptGraphBuilder
from .analysis.content_profiler import ContentProfiler
from .core.content_block_classifier import (
    BlockType,
    ContentBlock,
    ContentBlockClassifier,
)
from .core.heading_parser import HeadingHierarchy, HeadingNode, HeadingParser

# Import extended modules
from .formats.markdown_parser import MarkdownParser, detect_format
from .transformers.presentation_transformer import PresentationTransformer


@dataclass
class ExtractedProcedure:
    """A step-by-step procedure extracted from content."""
    name: str
    steps: List[str]
    context: str
    chapter_id: str
    section_id: Optional[str] = None


@dataclass
class ExtractedExample:
    """An example or case study extracted from content."""
    title: Optional[str]
    content: str
    related_concept: Optional[str]
    chapter_id: str
    section_id: Optional[str] = None


@dataclass
class ReviewQuestion:
    """A review question extracted from content."""
    question: str
    chapter_id: str
    section_id: Optional[str] = None
    bloom_level: Optional[str] = None


@dataclass
class SectionStructure:
    """Structured representation of a section."""
    id: str
    heading_level: int
    heading_text: str
    heading_id: Optional[str]
    content_blocks: List[ContentBlock]
    subsections: List['SectionStructure'] = field(default_factory=list)
    # Three-stage textbook synthesis (plan §1): the FULL inter-heading
    # prose for this section, captured between the section's own heading
    # boundary and the next sibling/parent heading. Distinct from the
    # sparse ``content_blocks`` — that stays a structural classification
    # surface; ``section_text`` is the verbatim prose the LLM synthesis
    # stages read. Empty string when no prose was captured.
    section_text: str = ""
    # Package 2 (outline-anchored) provenance: which zone the surviving
    # occurrence of this declared section was drawn from —
    # ``body`` / ``outline`` / ``answer_key`` / ``declared_only``. ``None`` on
    # the legacy / Package-1 paths (field stays absent from the serialized
    # dict so those outputs are byte-stable).
    matched_zone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "headingLevel": self.heading_level,
            "headingText": self.heading_text,
            "headingId": self.heading_id,
            "contentBlocks": [b.to_dict() for b in self.content_blocks],
            "subsections": [s.to_dict() for s in self.subsections]
        }
        # Additive: only emit ``section_text`` when prose was captured so
        # legacy fixtures without it stay byte-stable when prose is absent.
        if self.section_text:
            result["section_text"] = self.section_text
        # Additive: only emit ``matchedZone`` on the outline-anchored path.
        if self.matched_zone:
            result["matchedZone"] = self.matched_zone
        return result


@dataclass
class ChapterStructure:
    """Structured representation of a chapter."""
    id: str
    heading_level: int
    heading_text: str
    heading_id: Optional[str]
    explicit_objectives: List[Dict[str, str]]
    content_blocks: List[ContentBlock]
    sections: List[SectionStructure]
    # Three-stage textbook synthesis (plan §1): the FULL inter-heading
    # prose for this chapter — every paragraph / list / table-text
    # between this chapter's heading boundary and the next chapter
    # heading, including the prose nested under its sections. This is
    # the chapter-text source Stages 1/2/3 read from
    # ``textbook_structure.json::chapters[].chapter_text``. ``content_blocks``
    # semantics are LEFT UNCHANGED — ``chapter_text`` is purely additive.
    chapter_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Title sanitizer (SEMANTIK_TITLE_SANITIZE, default OFF →
        # byte-identical). A scanned-textbook chapter <h1>/<title> is OCR'd
        # WITH the running-header + page number + first content word fused in
        # ("Chapter 1 Foundations 83 ✓ Solution"). The structure guards recover
        # the real chapter node but not a clean title; strip the furniture here
        # (CHAPTER heading only — sections are untouched). Best-effort.
        _heading_text = self.heading_text
        try:
            from lib.textbook_title_sanitize import (
                sanitize_running_header_title as _san_title,
                title_sanitize_enabled as _san_enabled,
            )
            if _san_enabled():
                _heading_text = _san_title(self.heading_text)
        except Exception:  # noqa: BLE001 — sanitizer is best-effort
            _heading_text = self.heading_text
        result = {
            "id": self.id,
            "headingLevel": self.heading_level,
            "headingText": _heading_text,
            "headingId": self.heading_id,
            "explicitObjectives": self.explicit_objectives,
            "contentBlocks": [b.to_dict() for b in self.content_blocks],
            "sections": [s.to_dict() for s in self.sections]
        }
        # Additive: only emit ``chapter_text`` when prose was captured so
        # a legacy fixture comparing the structural surface stays
        # byte-stable when no prose is present.
        if self.chapter_text:
            result["chapter_text"] = self.chapter_text
        return result


class SemanticStructureExtractor:
    """
    Extracts complete semantic structure from SemantiK-processed HTML.

    Uses HeadingParser and ContentBlockClassifier to build a hierarchical
    representation of textbook content suitable for learning objective extraction.
    """

    # Bloom's taxonomy verb patterns for question analysis. These are
    # regex alternations (not plain verb lists), so migrating to
    # schemas/taxonomies/bloom_verbs.json requires a pattern-schema
    # layer. See the canonical tracking TODO at
    # `lib/semantic_structure_extractor/analysis/content_profiler.py`
    # — Wave 28f deduped the TODO to a single site.
    BLOOM_PATTERNS = {
        'remember': [
            r'\b(define|list|recall|identify|name|state|label|match|recognize)\b',
        ],
        'understand': [
            r'\b(explain|describe|summarize|classify|compare|interpret|discuss)\b',
        ],
        'apply': [
            r'\b(demonstrate|implement|solve|use|execute|apply|compute|calculate)\b',
        ],
        'analyze': [
            r'\b(analyze|differentiate|examine|distinguish|organize|compare.*contrast)\b',
        ],
        'evaluate': [
            r'\b(evaluate|assess|critique|justify|judge|argue|defend)\b',
        ],
        'create': [
            r'\b(create|design|construct|develop|formulate|compose|plan)\b',
        ],
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the extractor.

        Args:
            config_path: Optional path to configuration file
        """
        self.heading_parser = HeadingParser()
        self.block_classifier = ContentBlockClassifier()
        self.config = self._load_config(config_path)

        # Initialize new modules
        self.markdown_parser = MarkdownParser(self.config.get('markdown_parsing', {}))
        self.content_profiler = ContentProfiler(self.config)
        self.concept_builder = ConceptGraphBuilder(self.config)
        self.presentation_transformer = PresentationTransformer(self.config)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "chapter_heading_levels": [1, 2],
            "section_heading_levels": [2, 3, 4],
            "subsection_heading_levels": [4, 5, 6],
            "min_procedure_steps": 2,
            "min_example_words": 20
        }

        if config_path:
            path = Path(config_path)
            if path.exists():
                with open(path) as f:
                    loaded = json.load(f)
                    default_config.update(loaded)

        return default_config

    def extract(self, html_content: str, source_path: str = "") -> Dict[str, Any]:
        """
        Extract semantic structure from HTML content.

        Args:
            html_content: The HTML string to process
            source_path: Path to the source file (for metadata)

        Returns:
            Dictionary conforming to textbook_structure.schema.json
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract heading hierarchy
        hierarchy = self.heading_parser.parse(html_content)

        # Extract document info
        document_info = self._extract_document_info(soup, source_path)

        # Build chapter structure
        chapters = self._build_chapter_structure(soup, hierarchy)

        # Extract concepts
        extracted_concepts = self._extract_all_concepts(chapters)

        # Extract review questions
        review_questions = self._extract_review_questions(soup, chapters)

        result: Dict[str, Any] = {
            "documentInfo": document_info,
            "tableOfContents": hierarchy.to_toc(),
            "chapters": [ch.to_dict() for ch in chapters],
            "extractedConcepts": extracted_concepts,
            "reviewQuestions": [
                {
                    "question": q.question,
                    "chapterId": q.chapter_id,
                    "sectionId": q.section_id,
                    "bloomLevel": q.bloom_level
                }
                for q in review_questions
            ]
        }
        self._attach_structure_diagnostics(result, chapters)
        return result

    def extract_file(self, file_path: str, format: str = "auto") -> Dict[str, Any]:
        """
        Extract semantic structure from a file (HTML or Markdown).

        Args:
            file_path: Path to the file
            format: Format hint ("auto", "html", "markdown")

        Returns:
            Dictionary conforming to textbook_structure.schema.json
        """
        path = Path(file_path)
        with open(path, encoding='utf-8') as f:
            content = f.read()

        # Auto-detect format if needed
        if format == "auto":
            if path.suffix.lower() in ['.md', '.markdown']:
                format = "markdown"
            elif path.suffix.lower() in ['.html', '.htm']:
                format = "html"
            else:
                format = detect_format(content)

        return self.extract(content, str(path), format=format)

    def extract(self, content: str, source_path: str = "", format: str = "auto") -> Dict[str, Any]:  # noqa: F811
        """
        Extract semantic structure from content (HTML or Markdown).

        Args:
            content: The content string to process
            source_path: Path to the source file (for metadata)
            format: Format hint ("auto", "html", "markdown")

        Returns:
            Dictionary conforming to textbook_structure.schema.json
        """
        # Auto-detect format
        if format == "auto":
            format = detect_format(content)

        if format == "markdown":
            return self._extract_from_markdown(content, source_path)
        else:
            return self._extract_from_html(content, source_path)

    def _extract_from_markdown(self, content: str, source_path: str = "") -> Dict[str, Any]:
        """Extract semantic structure from Markdown content."""
        doc = self.markdown_parser.parse(content, source_path)
        result = doc.to_dict()

        # Add extraction metadata
        result['documentInfo']['extractionTimestamp'] = datetime.now().isoformat()
        result['documentInfo']['sourcePath'] = source_path
        result['documentInfo']['sourceFormat'] = 'markdown'

        return result

    def _extract_from_html(self, html_content: str, source_path: str = "") -> Dict[str, Any]:
        """Extract semantic structure from HTML content (original method)."""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract heading hierarchy
        hierarchy = self.heading_parser.parse(html_content)

        # Extract document info
        document_info = self._extract_document_info(soup, source_path)

        # Build chapter structure
        chapters = self._build_chapter_structure(soup, hierarchy)

        # Extract concepts
        extracted_concepts = self._extract_all_concepts(chapters)

        # Extract review questions
        review_questions = self._extract_review_questions(soup, chapters)

        result: Dict[str, Any] = {
            "documentInfo": document_info,
            "tableOfContents": hierarchy.to_toc(),
            "chapters": [ch.to_dict() for ch in chapters],
            "extractedConcepts": extracted_concepts,
            "reviewQuestions": [
                {
                    "question": q.question,
                    "chapterId": q.chapter_id,
                    "sectionId": q.section_id,
                    "bloomLevel": q.bloom_level
                }
                for q in review_questions
            ]
        }
        self._attach_structure_diagnostics(result, chapters)
        return result

    def extract_with_profiling(
        self,
        content: str,
        source_path: str = "",
        format: str = "auto"
    ) -> Dict[str, Any]:
        """
        Extract semantic structure with content profiling.

        Adds difficulty assessment, concept extraction, and concept graph.

        Args:
            content: Content to extract from
            source_path: Source file path
            format: Format hint

        Returns:
            Dictionary with semantic structure plus profiling data
        """
        # Get base extraction
        structure = self.extract(content, source_path, format)

        # Profile content
        profiles = self._profile_all_content(structure)
        structure['contentProfiles'] = profiles

        # Build concept graph
        concept_graph = self.concept_builder.build_graph(structure)
        structure['conceptGraph'] = concept_graph.to_dict()

        # Detect pedagogical pattern
        pattern = self.content_profiler.detect_pedagogical_pattern(
            structure.get('chapters', [])
        )
        structure['pedagogicalPattern'] = pattern.value

        return structure

    def extract_for_presentation(
        self,
        content: str,
        source_path: str = "",
        format: str = "auto"
    ) -> Dict[str, Any]:
        """
        Extract and transform content directly to presentation schema format.

        This is the primary method for the presentation generation pipeline.

        Args:
            content: Content to extract from
            source_path: Source file path
            format: Format hint

        Returns:
            Dictionary conforming to schemas/presentation/presentation_schema.json
        """
        # Get profiled extraction
        structure = self.extract_with_profiling(content, source_path, format)

        # Transform to presentation format
        concept_graph = structure.get('conceptGraph', {})
        presentation = self.presentation_transformer.transform(
            structure,
            concept_graph
        )

        return presentation

    def _profile_all_content(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Profile all content in the structure."""
        profiles = {
            'sections': {},
            'aggregate': None,
            'difficultyDistribution': {
                'beginner': 0,
                'intermediate': 0,
                'advanced': 0
            }
        }

        all_profiles = []

        for chapter in structure.get('chapters', []):
            section_profile = self.content_profiler.profile_section(chapter)
            profiles['sections'][chapter.get('id', '')] = section_profile.to_dict()

            if section_profile.aggregate_profile:
                all_profiles.append(section_profile.aggregate_profile)

                # Track difficulty distribution
                level = section_profile.aggregate_profile.difficulty_level.value
                profiles['difficultyDistribution'][level] = (
                    profiles['difficultyDistribution'].get(level, 0) + 1
                )

        # Create overall aggregate
        if all_profiles:
            profiles['aggregate'] = self.content_profiler._aggregate_profiles(
                all_profiles, 'document'
            ).to_dict()

        return profiles

    def _extract_document_info(self, soup: BeautifulSoup, source_path: str) -> Dict[str, Any]:
        """Extract document metadata."""
        # Get title
        title = ""
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text(strip=True)
        else:
            title_elem = soup.find('title')
            if title_elem:
                title = title_elem.get_text(strip=True)

        # Get metadata from meta tags
        authors = []
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta:
            authors = [author_meta.get('content', '')]

        description = ""
        desc_meta = soup.find('meta', attrs={'name': 'description'})
        if desc_meta:
            description = desc_meta.get('content', '')

        keywords = []
        keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_meta:
            keywords = [k.strip() for k in keywords_meta.get('content', '').split(',')]

        # Determine source format
        source_format = self._detect_source_format(soup)

        return {
            "title": title,
            "sourcePath": source_path,
            "sourceFormat": source_format,
            "extractionTimestamp": datetime.now().isoformat(),
            "metadata": {
                "authors": authors,
                "description": description,
                "keywords": keywords,
                "language": soup.find('html').get('lang', 'en') if soup.find('html') else 'en'
            }
        }

    def _detect_source_format(self, soup: BeautifulSoup) -> str:
        """Detect the source format of the HTML."""
        # Check for SemantiK markers (skip-link + specific ARIA landmarks).
        if soup.find('a', class_='skip-link'):
            main = soup.find('main', attrs={'role': 'main'})
            if main:
                # Legacy-compat source-format marker (persisted read value).
                return 'dart_html'

        return 'generic_html'

    def _build_chapter_structure(
        self,
        soup: BeautifulSoup,
        hierarchy: HeadingHierarchy
    ) -> List[ChapterStructure]:
        """Build chapter structure from heading hierarchy.

        First look for ``<article role="doc-chapter">`` wrappers emitted by
        the SemantiK converter. When present, each article becomes a chapter
        with its inner ``<h2>`` as the title and inner ``<section>`` wrappers
        as sections. Falls back to the plain ``<h2>`` grouping heuristic when
        no doc-chapter articles exist (generic third-party HTML).

        When both primary paths produce trivial output (0 chapters, or
        chapters that are all TOC artifacts, or a single chapter with no
        sections but the DOM has many ``<h2>``/``<h3>`` headings), synthesize
        chapters from the raw heading hierarchy. This handles third-party HTML
        that lacks ``<section>`` wrappers and doc-chapter articles but still
        carries a rich heading structure (W3C specs are the canonical
        example).
        """
        # Reset per-extraction guard diagnostics (populated only on the
        # guarded article path below).
        self._structure_diagnostics = None

        # Wave 19 primary path: DPUB-ARIA doc-chapter articles.
        doc_chapter_articles = soup.find_all(
            'article', attrs={'role': 'doc-chapter'}
        )
        primary_chapters: List[ChapterStructure] = []
        if doc_chapter_articles:
            if _structure_extract_guards_enabled():
                primary_chapters = self._build_chapters_from_articles_guarded(
                    soup, doc_chapter_articles
                )
            else:
                for idx, article in enumerate(doc_chapter_articles, start=1):
                    chapter = self._build_chapter_from_article(
                        soup, article, idx
                    )
                    primary_chapters.append(chapter)

        if not primary_chapters:
            # Legacy heading-hierarchy path.
            chapter_counter = 0

            # Find h1 or h2 headings that represent chapters
            chapter_levels = self.config.get('chapter_heading_levels', [1, 2])

            for root_node in hierarchy.root_nodes:
                # Process h1 as document title, h2s as chapters
                if root_node.level == 1:
                    # Process children of h1 as chapters
                    for child_id in root_node.children:
                        child_node = hierarchy.get_node(child_id)
                        if (
                            child_node
                            and child_node.level in chapter_levels
                            and not _is_noncontent_heading(child_node.text)
                        ):
                            chapter_counter += 1
                            chapter = self._build_chapter(
                                soup, hierarchy, child_node, chapter_counter
                            )
                            primary_chapters.append(chapter)
                elif (
                    root_node.level in chapter_levels
                    and not _is_noncontent_heading(root_node.text)
                ):
                    chapter_counter += 1
                    chapter = self._build_chapter(
                        soup, hierarchy, root_node, chapter_counter
                    )
                    primary_chapters.append(chapter)

        # Wave 74 Session 3: heading-hierarchy fallback.
        # Fires when the primary paths degenerate to trivial output but
        # the DOM still carries meaningful h2/h3 structure.
        if self._primary_output_is_trivial(soup, primary_chapters):
            fallback = self._build_chapters_from_headings(soup)
            if fallback:
                source_path = self._document_source_hint(soup)
                logger.warning(
                    "SemanticStructureExtractor: primary extraction paths "
                    "produced trivial output (%d chapter(s)); falling back "
                    "to heading-hierarchy synthesis and emitted %d "
                    "chapter(s). source=%s",
                    len(primary_chapters),
                    len(fallback),
                    source_path or "<inline>",
                )
                self._populate_chapter_text(soup, fallback)
                self._warn_if_structure_collapsed(fallback)
                return fallback

        self._populate_chapter_text(soup, primary_chapters)
        self._warn_if_structure_collapsed(primary_chapters)
        return primary_chapters

    # ------------------------------------------------------------------
    # WS4 §4 — structure-collapse early warning
    # ------------------------------------------------------------------

    def _structure_collapse_suspected(
        self,
        chapters: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Detect the single-chapter / many-section collapse signature.

        Returns a diagnostics dict when the extractor produced exactly one
        chapter carrying more than ``_STRUCTURE_COLLAPSE_SECTION_THRESHOLD``
        sections (the 141:1 signature where Stage-1 draft terminal objectives
        under-represent the back of the book); ``None`` otherwise. Pure /
        side-effect free so both the warning and the
        ``structureDiagnostics`` output key derive from the same check.
        """
        chapter_count = len(chapters)
        section_count = sum(len(c.sections) for c in chapters)
        if (
            chapter_count == 1
            and section_count > _STRUCTURE_COLLAPSE_SECTION_THRESHOLD
        ):
            return {
                "suspected": True,
                "chapter_count": chapter_count,
                "section_count": section_count,
                "ratio": section_count,
                "threshold": _STRUCTURE_COLLAPSE_SECTION_THRESHOLD,
            }
        return None

    def _attach_structure_diagnostics(
        self,
        result: Dict[str, Any],
        chapters: List[Any],
    ) -> None:
        """Merge the collapse-suspected signal (all paths) and the Package-1
        guard counters (guarded article path only) into
        ``result["structureDiagnostics"]``.

        Byte-stable when guards are off: ``_structure_diagnostics`` is reset to
        ``None`` at the top of ``_build_chapter_structure`` and only the
        guarded article path populates it, so a legacy run emits exactly the
        prior ``{"structureCollapseSuspected": ...}`` shape (or no key)."""
        diagnostics: Dict[str, Any] = {}
        collapse = self._structure_collapse_suspected(chapters)
        if collapse:
            diagnostics["structureCollapseSuspected"] = collapse
        guard = getattr(self, "_structure_diagnostics", None)
        if guard:
            diagnostics["guards"] = guard
        if diagnostics:
            result["structureDiagnostics"] = diagnostics

    def _warn_if_structure_collapsed(self, chapters: List[Any]) -> None:
        """Emit a loud WARNING when ``_structure_collapse_suspected`` fires."""
        collapse = self._structure_collapse_suspected(chapters)
        if collapse:
            logger.warning(
                "SemanticStructureExtractor: STRUCTURE_COLLAPSE_SUSPECTED — "
                "%d section(s) under a single chapter (threshold %d). Stage-1 "
                "draft terminal objectives will under-represent the back of "
                "the book.",
                collapse["section_count"],
                collapse["threshold"],
            )

    # ------------------------------------------------------------------
    # Three-stage textbook synthesis — inter-heading prose capture
    # ------------------------------------------------------------------

    def _populate_chapter_text(
        self,
        soup: BeautifulSoup,
        chapters: List[ChapterStructure],
    ) -> None:
        """Capture full inter-heading prose into ``chapter_text`` /
        ``section_text`` on every chapter / section dict.

        Plan §1 / §9: the three-stage LLM synthesis architecture needs
        the FULL chapter prose, not the sparse structural
        ``content_blocks`` (which carry ~1.9 KB total per chapter on a
        real single-file algebra textbook — a sliver of the real 10-30 pages of
        prose). This method walks the document's headings in order and,
        for each chapter / section, accumulates the verbatim text of
        every element appearing AFTER that heading and BEFORE the next
        heading at the same-or-shallower level.

        Strategy — purely deterministic, provider-independent:

        1. Collect every ``h1``..``h6`` in document order.
        2. For each heading, the span of "owned" content runs from the
           heading to the next heading of an equal-or-shallower level.
        3. A chapter's ``chapter_text`` is the concatenation of the
           prose owned directly by its heading PLUS the prose owned by
           every descendant section/subsection heading — i.e. the whole
           chapter scope. A section's ``section_text`` covers just its
           own scope.

        ``content_blocks`` is untouched — this only sets the additive
        ``chapter_text`` / ``section_text`` fields.
        """
        if not chapters:
            return

        container = soup.find('main') or soup.find('body') or soup
        if container is None:
            return

        headings: List[Tag] = [
            t for t in container.find_all(
                ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
            )
            if t.get_text(strip=True)
        ]
        if not headings:
            return

        # heading element identity -> its directly-owned prose (between
        # this heading and the next equal-or-shallower heading).
        owned_prose: Dict[int, str] = {}
        # heading element identity -> (level, document index).
        heading_meta: Dict[int, tuple] = {}
        for idx, tag in enumerate(headings):
            try:
                level = int(tag.name.lstrip('h'))
            except (ValueError, AttributeError):
                level = 6
            heading_meta[id(tag)] = (level, idx)
            owned_prose[id(tag)] = self._collect_heading_prose(
                tag, headings, idx
            )

        # Resolve which heading each chapter/section maps to. The
        # heading-hierarchy walk consumes headings in the same document
        # order chapters/sections are built, so a stable two-pointer
        # match on (heading_text, level) is reliable. Track a consumed
        # set so duplicate heading texts resolve to distinct elements.
        consumed: set = set()

        def _match_heading(
            heading_text: str,
            heading_id: Optional[str],
        ) -> Optional[Tag]:
            # Prefer an exact id match — unambiguous.
            if heading_id:
                for tag in headings:
                    if id(tag) in consumed:
                        continue
                    if tag.get('id') == heading_id:
                        consumed.add(id(tag))
                        return tag
            target = (heading_text or "").strip()
            for tag in headings:
                if id(tag) in consumed:
                    continue
                if tag.get_text(strip=True) == target:
                    consumed.add(id(tag))
                    return tag
            return None

        def _scope_text(start_tag: Tag) -> str:
            """All prose owned by ``start_tag`` and its deeper-level
            descendant headings, up to the next equal-or-shallower
            heading."""
            start_level, start_idx = heading_meta[id(start_tag)]
            parts: List[str] = []
            owned = owned_prose.get(id(start_tag), "")
            if owned:
                parts.append(owned)
            for tag in headings[start_idx + 1:]:
                lvl, _ = heading_meta[id(tag)]
                if lvl <= start_level:
                    break
                deeper = owned_prose.get(id(tag), "")
                if deeper:
                    parts.append(deeper)
            return "\n\n".join(parts).strip()

        def _walk_sections(sections: List[SectionStructure]) -> None:
            for section in sections:
                tag = _match_heading(
                    section.heading_text, section.heading_id
                )
                if tag is not None:
                    section.section_text = _scope_text(tag)
                _walk_sections(section.subsections)

        for chapter in chapters:
            tag = _match_heading(chapter.heading_text, chapter.heading_id)
            if tag is not None:
                chapter.chapter_text = _scope_text(tag)
            _walk_sections(chapter.sections)

    @staticmethod
    def _collect_heading_prose(
        heading: Tag,
        headings: List[Tag],
        heading_idx: int,
    ) -> str:
        """Return the verbatim prose owned directly by one heading.

        "Owned" = every text-bearing element appearing after ``heading``
        in document order and before the next heading (of ANY level).
        The walk uses ``next_element`` traversal and stops at the next
        heading element so nested-section prose is attributed to that
        section's own heading rather than double-counted here.
        """
        next_heading = (
            headings[heading_idx + 1]
            if heading_idx + 1 < len(headings)
            else None
        )
        next_heading_ids = {id(h) for h in headings}
        collected: List[str] = []
        node = heading.next_element
        while node is not None:
            if node is next_heading:
                break
            if isinstance(node, Tag) and id(node) in next_heading_ids:
                # Reached some other heading — stop (defensive; the
                # next_heading short-circuit normally fires first).
                break
            if isinstance(node, Tag) and node.name in (
                'p', 'li', 'dd', 'dt', 'blockquote', 'pre',
                'caption', 'figcaption', 'th', 'td',
            ):
                text = node.get_text(" ", strip=True)
                if text:
                    collected.append(text)
            node = node.next_element
        # De-dup consecutive repeats (a <li> inside a <ul> is visited
        # only once via next_element, but nested inline tags can echo).
        seen: set = set()
        deduped: List[str] = []
        for chunk in collected:
            if chunk in seen:
                continue
            seen.add(chunk)
            deduped.append(chunk)
        return "\n".join(deduped).strip()

    # ------------------------------------------------------------------
    # Wave 74 Session 3: heading-hierarchy fallback
    # ------------------------------------------------------------------

    def _primary_output_is_trivial(
        self,
        soup: BeautifulSoup,
        chapters: List[ChapterStructure],
    ) -> bool:
        """Whether the primary extraction paths produced trivial output.

        Trivial means one of:

        * Zero chapters.
        * All chapter titles are TOC artifacts (``Contents``, ``Index``,
          etc.) — the extractor caught the TOC h2 and missed the real
          chapter headings that follow it as siblings.
        * Zero chapters with non-empty sections AND the raw DOM carries
          at least three h2/h3 headings that aren't TOC artifacts.
          This covers specs like rdf11-primer (1 TOC h2, 13 real h3s)
          and the W3C family more broadly.
        """
        if not chapters:
            return True

        non_toc_chapters = [
            c for c in chapters if not _is_toc_heading(c.heading_text)
        ]
        if not non_toc_chapters:
            return True

        chapters_with_sections = [
            c for c in chapters if c.sections
        ]
        if chapters_with_sections:
            return False

        # Count real (non-TOC) h2/h3 headings in the DOM. If there's
        # a genuine hierarchy lurking, the primary paths missed it.
        real_heading_count = 0
        for tag in soup.find_all(['h2', 'h3']):
            text = tag.get_text(strip=True)
            if text and not _is_toc_heading(text):
                real_heading_count += 1
                if real_heading_count >= 3:
                    return True
        return False

    def _document_source_hint(self, soup: BeautifulSoup) -> Optional[str]:
        """Best-effort source identifier for log messages."""
        title = soup.find('title')
        if title:
            text = title.get_text(strip=True)
            if text:
                return text
        h1 = soup.find('h1')
        if h1:
            text = h1.get_text(strip=True)
            if text:
                return text
        return None

    def _build_chapters_from_headings(
        self,
        soup: BeautifulSoup,
    ) -> List[ChapterStructure]:
        """Synthesize chapter/section hierarchy from raw heading levels.

        Strategy:

        1. Walk every ``h1``..``h6`` in document order inside ``<main>``
           (falling back to ``<body>`` then the whole soup).
        2. Drop TOC artifacts (``Contents``, ``Table of Contents``).
        3. Pick the "chapter level" as the shallowest heading level
           that has at least two non-TOC entries. If the only real
           heading level is h3 (e.g., rdf11-primer), h3s become
           chapters; if h2 and h3 both exist with real content, h2s
           become chapters and h3s become sections.
        4. Content blocks between two consecutive headings attach to
           the most recent open heading's chapter/section.
        5. ``data-semantik-*`` attributes on individual content elements
           carry through via ``ContentBlockClassifier._classify_element``.
        """
        container = soup.find('main') or soup.find('body') or soup
        if container is None:
            return []

        # Collect every heading in document order, filtering TOC noise.
        # Keep an UNFILTERED copy so we can fail safe back to it if the
        # content-heading filter removes everything (never emit empty).
        unfiltered_headings: List[Tag] = []
        all_headings: List[Tag] = []
        dropped_noncontent = 0
        for tag in container.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            # SEMANTIK_BOX_TITLE_HEADINGS — a presentational callout box title is
            # never a section (anti-re-poisoning); skip it unconditionally.
            if tag.has_attr('data-semantik-box-title'):
                continue
            text = tag.get_text(strip=True)
            if not text:
                continue
            if _is_toc_heading(text):
                continue
            unfiltered_headings.append(tag)
            # Drop answer-key / front-matter / donor-list headings so they
            # never become chapter/section nodes (the OpenStax
            # contamination bug). Conservative predicate — see
            # ``_is_noncontent_heading``.
            if _is_noncontent_heading(text):
                dropped_noncontent += 1
                continue
            all_headings.append(tag)

        # Fail-safe, NOT fail-closed-to-empty: if the content-heading
        # filter removed every real heading, the predicate was too
        # aggressive for this document — revert to the unfiltered set and
        # warn rather than emitting a structure-less course.
        if not all_headings and unfiltered_headings:
            logger.warning(
                "SemanticStructureExtractor: content-heading filter removed "
                "all %d heading(s); reverting to unfiltered heading set to "
                "avoid emitting an empty structure.",
                dropped_noncontent,
            )
            all_headings = unfiltered_headings

        if not all_headings:
            return []

        # Figure out which level acts as chapter vs section.
        level_counts: Dict[int, int] = {}
        for tag in all_headings:
            try:
                lv = int(tag.name.lstrip('h'))
            except ValueError:
                continue
            level_counts[lv] = level_counts.get(lv, 0) + 1

        # Chapter level: shallowest heading level with >= 1 entry,
        # preferring levels with multiple entries when present. h1 is
        # skipped as chapter-level when it appears exactly once (it's
        # the document title).
        sorted_levels = sorted(level_counts.keys())
        chapter_level: Optional[int] = None
        skip_solo_h1 = False
        for lv in sorted_levels:
            if lv == 1 and level_counts[lv] < 2:
                # Treat a solo h1 as the document title, not a chapter.
                skip_solo_h1 = True
                continue
            chapter_level = lv
            break
        if chapter_level is None:
            # Only a single h1 exists — promote it to a chapter anyway
            # so we at least emit one meaningful entry.
            chapter_level = sorted_levels[0]
            skip_solo_h1 = False

        section_level = chapter_level + 1
        subsection_level = chapter_level + 2

        # Walk the full descendants stream of the container. Maintain
        # a "current chapter / section / subsection" pointer and attach
        # any non-heading ContentBlock-yielding element to the deepest
        # open target.
        chapters: List[ChapterStructure] = []
        current_chapter: Optional[ChapterStructure] = None
        current_section: Optional[SectionStructure] = None
        current_subsection: Optional[SectionStructure] = None
        chapter_counter = 0
        section_counter = 0
        subsection_counter = 0
        classifier = ContentBlockClassifier()
        heading_set = set(id(h) for h in all_headings)

        # Track elements we've already processed to avoid double-counting
        # when a parent tag emits both itself and its children through
        # the descendant iterator.
        consumed: set = set()

        def _walk(node: Tag) -> None:
            nonlocal current_chapter, current_section, current_subsection
            nonlocal chapter_counter, section_counter, subsection_counter

            for child in node.children:
                if not isinstance(child, Tag):
                    continue
                if id(child) in consumed:
                    continue
                name = child.name.lower() if child.name else ''

                # Heading — open a new chapter/section/subsection.
                if name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                    if id(child) not in heading_set:
                        # Filtered (TOC artifact) — ignore.
                        continue
                    try:
                        lv = int(name.lstrip('h'))
                    except ValueError:
                        continue
                    heading_text = child.get_text(strip=True)
                    heading_id = child.get('id')

                    # Skip a solo h1 that's serving as the document title
                    # when chapter_level is deeper (e.g., chapter_level==3
                    # because h2 only had TOC entries).
                    if skip_solo_h1 and lv == 1:
                        consumed.add(id(child))
                        continue

                    if lv <= chapter_level:
                        chapter_counter += 1
                        current_chapter = ChapterStructure(
                            id=f"ch{chapter_counter}",
                            heading_level=lv,
                            heading_text=heading_text,
                            heading_id=heading_id,
                            explicit_objectives=[],
                            content_blocks=[],
                            sections=[],
                        )
                        chapters.append(current_chapter)
                        current_section = None
                        current_subsection = None
                        section_counter = 0
                        subsection_counter = 0
                    elif lv == section_level:
                        # Ensure there's a chapter to attach to.
                        if current_chapter is None:
                            chapter_counter += 1
                            current_chapter = ChapterStructure(
                                id=f"ch{chapter_counter}",
                                heading_level=chapter_level,
                                heading_text=heading_text,
                                heading_id=heading_id,
                                explicit_objectives=[],
                                content_blocks=[],
                                sections=[],
                            )
                            chapters.append(current_chapter)
                        section_counter += 1
                        current_section = SectionStructure(
                            id=f"{current_chapter.id}_s{section_counter}",
                            heading_level=lv,
                            heading_text=heading_text,
                            heading_id=heading_id,
                            content_blocks=[],
                            subsections=[],
                        )
                        current_chapter.sections.append(current_section)
                        current_subsection = None
                        subsection_counter = 0
                    elif lv >= subsection_level:
                        # Ensure a section exists; synthesize if needed.
                        if current_chapter is None:
                            chapter_counter += 1
                            current_chapter = ChapterStructure(
                                id=f"ch{chapter_counter}",
                                heading_level=chapter_level,
                                heading_text=heading_text,
                                heading_id=heading_id,
                                explicit_objectives=[],
                                content_blocks=[],
                                sections=[],
                            )
                            chapters.append(current_chapter)
                        if current_section is None:
                            section_counter += 1
                            current_section = SectionStructure(
                                id=f"{current_chapter.id}_s{section_counter}",
                                heading_level=section_level,
                                heading_text=heading_text,
                                heading_id=None,
                                content_blocks=[],
                                subsections=[],
                            )
                            current_chapter.sections.append(current_section)
                        subsection_counter += 1
                        current_subsection = SectionStructure(
                            id=(
                                f"{current_section.id}_sub{subsection_counter}"
                            ),
                            heading_level=lv,
                            heading_text=heading_text,
                            heading_id=heading_id,
                            content_blocks=[],
                            subsections=[],
                        )
                        current_section.subsections.append(current_subsection)
                    # Mark the heading as consumed — we don't want to
                    # reclassify it as a ContentBlock.
                    consumed.add(id(child))
                    continue

                # Non-heading leaf-like element — try to classify as a
                # content block and attach to the deepest open target.
                if name in (
                    'p', 'ul', 'ol', 'dl', 'pre', 'code', 'blockquote',
                    'table', 'figure', 'img', 'aside', 'div',
                ):
                    # Skip elements that contain nested headings — we
                    # want to recurse into them so the headings land in
                    # the right chapter/section.
                    has_nested_heading = any(
                        id(h) in heading_set
                        for h in child.find_all(
                            ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
                        )
                    )
                    if has_nested_heading:
                        _walk(child)
                        continue

                    block = classifier._classify_element(child)
                    if block is None:
                        continue
                    if current_subsection is not None:
                        current_subsection.content_blocks.append(block)
                    elif current_section is not None:
                        current_section.content_blocks.append(block)
                    elif current_chapter is not None:
                        current_chapter.content_blocks.append(block)
                    consumed.add(id(child))
                    continue

                # Structural wrappers (section/article/header/nav/main/
                # body) — recurse so we find nested headings.
                if name in (
                    'section', 'article', 'header', 'footer', 'nav',
                    'main', 'body', 'html', 'div',
                ):
                    _walk(child)

        _walk(container)

        # Drop any chapter whose only heading_text is a TOC artifact
        # AND that has no sections / content — belt and braces.
        chapters = [
            c for c in chapters
            if not (
                _is_toc_heading(c.heading_text)
                and not c.sections
                and not c.content_blocks
            )
        ]

        return chapters

    def _build_chapter_from_article(
        self,
        soup: BeautifulSoup,
        article: Tag,
        chapter_num: int,
    ) -> ChapterStructure:
        """Build a chapter from a ``<article role="doc-chapter">`` wrapper.

        The SemantiK converter emits every chapter as a standalone article
        with the chapter heading inside a ``<header>``
        block. We prefer the ``id`` attribute on the article itself
        (``chap-{N}``) for the chapter id; falling back to a synthesized
        ``ch{N}`` identifier when the article lacks an explicit id.
        """
        chapter_id = str(article.get('id') or f'ch{chapter_num}').strip()

        # Title: the first <h2> or <h1> inside the article (Wave 13 uses h2).
        heading_tag = article.find(['h1', 'h2'])
        heading_text = None
        heading_id = None
        heading_level = 2
        if heading_tag:
            heading_text = heading_tag.get_text(strip=True) or None
            heading_id = heading_tag.get('id')
            try:
                heading_level = int(heading_tag.name.lstrip('h'))
            except ValueError:
                heading_level = 2
        if not heading_text:
            heading_text = article.get('aria-label') or f'Chapter {chapter_num}'

        # Explicit objectives: reuse the existing helper on the article.
        explicit_objectives = self._extract_explicit_objectives(article)

        # Content blocks that appear directly in the article, before any
        # nested <section>. Treat the article like a chapter's own
        # section_elem for _extract_chapter_content.
        class _ArticleLike:
            """Duck-typed shim so ``_extract_chapter_content`` walks the
            article exactly like a ``<section>`` root.
            """
            def __init__(self, elem):
                self._elem = elem

            @property
            def children(self):
                return self._elem.children

        content_blocks = self._extract_chapter_content(
            _ArticleLike(article), None
        )

        # Build sections from every top-level <section> child inside the
        # article. The heading hierarchy isn't consulted here — Wave 13's
        # chapter article wraps its own section tree, so we walk the DOM
        # directly.
        sections: List[SectionStructure] = []
        sec_counter = 0
        for child in article.find_all('section', recursive=False):
            sec_counter += 1
            sections.append(
                self._build_section_from_element(
                    soup, child, chapter_id, sec_counter,
                )
            )
        # When sections don't live as direct children (common — Wave 13
        # emits the chapter article and lets the assembler sibling the
        # section blocks), also pull any <section> following the article
        # until the next <article role="doc-chapter"> or document end.
        if not sections:
            sibling = article.next_sibling
            while sibling is not None:
                if isinstance(sibling, Tag):
                    if (
                        sibling.name == 'article'
                        and sibling.get('role') == 'doc-chapter'
                    ):
                        break
                    if sibling.name == 'section':
                        sec_counter += 1
                        sections.append(
                            self._build_section_from_element(
                                soup, sibling, chapter_id, sec_counter,
                            )
                        )
                sibling = sibling.next_sibling

        return ChapterStructure(
            id=chapter_id,
            heading_level=heading_level,
            heading_text=heading_text,
            heading_id=heading_id,
            explicit_objectives=explicit_objectives,
            content_blocks=content_blocks,
            sections=sections,
        )

    # ------------------------------------------------------------------
    # Package 1 — guarded article-path assembly.
    #
    # Only reached when ``ED4ALL_STRUCTURE_EXTRACT_GUARDS`` is truthy.
    # Fixes the three chapter/section over-emission layers the legacy
    # article path let through on per-block-wrapped scan corpora:
    #   1a. a continuation article (no visible content h1/h2 — its title
    #       lives in an aria-hidden div per lib/semantik/adapter.py:291-305)
    #       merges its body into the PREVIOUS chapter instead of minting a
    #       ``Chapter {n}`` fallback. The first article never merges.
    #   1b. a ``<section>`` wrapper with no heading element does NOT mint a
    #       section — its blocks group under the preceding heading-bearing
    #       section (or the chapter's implicit lead ``content_blocks``).
    #   1c. a heading failing ``_is_noncontent_heading`` (which folds in
    #       ``_is_eoc_section_heading`` + Package-3 ordinal normalization)
    #       does not create a boundary; its content regroups like 1b.
    # ------------------------------------------------------------------

    def _article_content_heading(self, article: Tag) -> Optional[Tag]:
        """Return the article's visible chapter-title h1/h2, else ``None``.

        Mirrors the legacy title probe (``article.find(['h1','h2'])``) but
        additionally rejects a heading that fails ``_is_noncontent_heading``
        (1c) — a chapter whose only heading is answer-key / EOC / numbered-
        apparatus noise has no real title and is treated as a continuation.
        """
        heading_tag = article.find(['h1', 'h2'])
        if heading_tag is None:
            return None
        text = heading_tag.get_text(strip=True)
        if not text or _is_noncontent_heading(text):
            return None
        return heading_tag

    def _group_sections_from_iter(
        self,
        soup: BeautifulSoup,
        section_elems: List[Tag],
        parent_id: str,
        lead_content_blocks: List[ContentBlock],
        sec_start: int,
        diag: Dict[str, int],
    ) -> tuple:
        """Group a run of ``<section>`` wrappers into real sections (1b/1c).

        A wrapper with a content heading becomes a section; a headingless or
        noncontent-heading wrapper has its classified blocks appended to the
        preceding real section (or ``lead_content_blocks`` if none yet), so
        NOTHING is dropped — only regrouped. Returns ``(sections, counter)``.
        """
        sections: List[SectionStructure] = []
        sec_counter = sec_start
        for elem in section_elems:
            heading_tag = elem.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            heading_text = heading_tag.get_text(strip=True) if heading_tag else ''
            is_content_heading = bool(heading_text) and not _is_noncontent_heading(
                heading_text
            )
            if is_content_heading:
                sec_counter += 1
                sections.append(
                    self._build_section_from_element(
                        soup, elem, parent_id, sec_counter,
                        guarded=True, diag=diag,
                    )
                )
                continue
            # 1b / 1c — regroup. classify_section returns None for nested
            # <section> children (containers), so no double-count against a
            # parent classify_section.
            blocks = self.block_classifier.classify_section(elem)
            target = sections[-1].content_blocks if sections else lead_content_blocks
            target.extend(blocks)
            if heading_tag is not None:
                diag["apparatus_demoted"] += 1
            else:
                diag["headingless_wrappers_grouped"] += 1
        return sections, sec_counter

    def _collect_article_body(
        self,
        soup: BeautifulSoup,
        article: Tag,
        chapter_id: str,
        sec_start: int,
        diag: Dict[str, int],
    ) -> tuple:
        """Extract ``(content_blocks, sections, counter)`` from an article
        with the 1b/1c grouping applied. ``content_blocks`` is the article's
        implicit lead section (blocks before the first ``<section>``)."""
        content_blocks = self._extract_chapter_content(article, None)
        direct = article.find_all('section', recursive=False)
        if direct:
            section_elems = direct
        else:
            # Sections emitted as SIBLINGS after the article (Wave 13 shape),
            # up to the next doc-chapter article or document end.
            section_elems = []
            sibling = article.next_sibling
            while sibling is not None:
                if isinstance(sibling, Tag):
                    if (
                        sibling.name == 'article'
                        and sibling.get('role') == 'doc-chapter'
                    ):
                        break
                    if sibling.name == 'section':
                        section_elems.append(sibling)
                sibling = sibling.next_sibling
        sections, sec_counter = self._group_sections_from_iter(
            soup, section_elems, chapter_id, content_blocks, sec_start, diag
        )
        return content_blocks, sections, sec_counter

    def _build_chapter_from_article_guarded(
        self,
        soup: BeautifulSoup,
        article: Tag,
        chapter_num: int,
        heading_tag: Optional[Tag],
        diag: Dict[str, int],
    ) -> ChapterStructure:
        """Guarded counterpart of ``_build_chapter_from_article`` (1b/1c)."""
        chapter_id = str(article.get('id') or f'ch{chapter_num}').strip()
        heading_text = None
        heading_id = None
        heading_level = 2
        if heading_tag is not None:
            heading_text = heading_tag.get_text(strip=True) or None
            heading_id = heading_tag.get('id')
            try:
                heading_level = int(heading_tag.name.lstrip('h'))
            except ValueError:
                heading_level = 2
        if not heading_text:
            heading_text = article.get('aria-label') or f'Chapter {chapter_num}'

        explicit_objectives = self._extract_explicit_objectives(article)
        content_blocks, sections, _ = self._collect_article_body(
            soup, article, chapter_id, 0, diag
        )
        return ChapterStructure(
            id=chapter_id,
            heading_level=heading_level,
            heading_text=heading_text,
            heading_id=heading_id,
            explicit_objectives=explicit_objectives,
            content_blocks=content_blocks,
            sections=sections,
        )

    def _declared_section_estimate(self, soup: BeautifulSoup) -> int:
        """Cheap declared-section count: distinct ``N.M`` ordinals in a
        chapter-outline / ToC zone, if one exists. ``0`` when no such zone
        (undeclared corpora fall through with no sanity warning)."""
        zones: List[Tag] = []
        zones.extend(
            soup.find_all(attrs={"data-dart-block-id": "chapter-outline"})
        )
        zones.extend(soup.find_all("nav", class_="toc"))
        if not zones:
            return 0
        ordinals: set = set()
        for zone in zones:
            for m in re.findall(r"\b\d+\.\d+\b", zone.get_text(" ", strip=True)):
                ordinals.add(m)
        return len(ordinals)

    def _build_chapters_from_articles_guarded(
        self,
        soup: BeautifulSoup,
        articles: List[Tag],
    ) -> List[ChapterStructure]:
        """Guarded article path (1a merge + 1b/1c grouping + 1d diagnostics)."""
        diag: Dict[str, int] = {
            "sections_built": 0,
            "headingless_wrappers_grouped": 0,
            "continuations_merged": 0,
            "apparatus_demoted": 0,
            "declared_section_estimate": self._declared_section_estimate(soup),
        }
        chapters: List[ChapterStructure] = []
        chapter_ordinal = 0
        for article in articles:
            heading_tag = self._article_content_heading(article)
            # 1a — a continuation (no content h1/h2) merges into the previous
            # chapter. The FIRST article never merges (nothing precedes it).
            if heading_tag is None and chapters:
                prev = chapters[-1]
                content_blocks, sections, _ = self._collect_article_body(
                    soup, article, prev.id, len(prev.sections), diag
                )
                prev.content_blocks.extend(content_blocks)
                prev.sections.extend(sections)
                diag["continuations_merged"] += 1
                continue
            chapter_ordinal += 1
            chapters.append(
                self._build_chapter_from_article_guarded(
                    soup, article, chapter_ordinal, heading_tag, diag
                )
            )

        diag["sections_built"] = sum(len(c.sections) for c in chapters)
        self._structure_diagnostics = diag

        # Package 2 — outline-anchored realignment. Refines the Package-1
        # chapters onto the document's declared ``N.M`` spine when the book
        # declares its own structure. Falls through UNCHANGED (returns None)
        # for undeclared corpora, so the wide-net contract holds.
        if _outline_anchor_enabled():
            anchored = self._build_chapters_outline_anchored(soup, articles, diag)
            if anchored is not None:
                chapters = anchored

        # 1d — post-build sanity: warn (never fail) when the built section
        # count dwarfs the outline-declared estimate (>3x), the per-block-
        # wrapper / apparatus-banner inflation signature.
        est = diag["declared_section_estimate"]
        if est > 0 and diag["sections_built"] > 3 * est:
            logger.warning(
                "SemanticStructureExtractor: STRUCTURE_SECTION_OVERCOUNT — built "
                "%d section(s) vs ~%d declared in the chapter-outline/ToC zone "
                "(>3x). Likely per-block-wrapper / apparatus-banner inflation.",
                diag["sections_built"],
                est,
            )
        return chapters

    # ------------------------------------------------------------------
    # Package 2 — outline-anchored section alignment.
    # ------------------------------------------------------------------

    def _outline_zone_articles(self, soup: BeautifulSoup) -> List[Tag]:
        """Articles that CONTAIN a ``chapter-outline`` block (the outline zone).

        The converter stamps the outline block with
        ``data-dart-block-id="chapter-outline"`` (legacy read attribute); its
        enclosing ``<article role="doc-chapter">`` is the declared-structure
        zone whose heading sections + fused paragraphs are the cleanest entry
        source.
        """
        articles: List[Tag] = []
        seen: set = set()
        for blk in soup.find_all(attrs={"data-dart-block-id": "chapter-outline"}):
            art = blk.find_parent("article", attrs={"role": "doc-chapter"})
            if art is not None and id(art) not in seen:
                seen.add(id(art))
                articles.append(art)
        return articles

    @staticmethod
    def _prune_outline_contiguity(
        entries: Dict[str, str],
        title_sources: Dict[str, str],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Contiguity belt — keep only the near-contiguous 1..K run per major,
        then drop whole spurious majors.

        Stage 1 (within a major): sort the accepted minors and keep the dense
        run from the smallest minor upward, dropping a minor (and its tail) once
        it is more than ``_OUTLINE_MINOR_GAP_MAX`` beyond the previous kept
        minor. Never drops the opener, so a real chapter can only lose a stray
        far-outlier, never its body.

        Stage 2 (across majors): a real chapter opens at minor 1, so keep the
        DOMINANT major (most retained sections) plus any other major whose run
        opens at minor 1. A lone cross-reference leaking a banner from another
        chapter (e.g. a "7.3 Exercises" mention inside chapter 1) survives
        neither test and is dropped. Fail-safe: the dominant major is always
        kept, so a real single-chapter file can never be emptied.
        """
        by_major: Dict[int, List[int]] = {}
        for ordinal in entries:
            major, minor = ordinal.split(".")
            by_major.setdefault(int(major), []).append(int(minor))

        # Stage 1 — per-major contiguity.
        kept_minors: Dict[int, List[int]] = {}
        for major, minors in by_major.items():
            run_prev: Optional[int] = None
            kept: List[int] = []
            for minor in sorted(set(minors)):
                if run_prev is None or minor - run_prev <= _OUTLINE_MINOR_GAP_MAX:
                    kept.append(minor)
                    run_prev = minor
                else:
                    break
            kept_minors[major] = kept

        # Stage 2 — cross-major spurious-major drop.
        if kept_minors:
            dominant = max(kept_minors, key=lambda m: (len(kept_minors[m]), -m))
            keep_majors = {
                m
                for m, mins in kept_minors.items()
                if m == dominant or (mins and mins[0] == 1)
            }
        else:  # pragma: no cover - entries non-empty by construction
            keep_majors = set()

        keep = {
            f"{major}.{minor}"
            for major, mins in kept_minors.items()
            if major in keep_majors
            for minor in mins
        }
        pruned_entries = {o: entries[o] for o in entries if o in keep}
        pruned_sources = {o: title_sources[o] for o in entries if o in keep}
        return pruned_entries, pruned_sources

    def _harvest_outline(
        self,
        soup: BeautifulSoup,
    ) -> Tuple[Dict[str, str], List[Tag], Dict[str, str]]:
        """Package 2b — multi-source ordinal-UNION harvest of declared sections.

        LOAD-BEARING ADMISSION RULE: an ordinal is admitted as a declared
        section iff it has STRUCTURAL evidence — it opens a heading element
        ((d) numbered headings document-wide, existing ``_NM_HEADING_RE``
        shape) or forms an apparatus banner ((e) "N.M Exercises" shape). An
        ordinal seen ONLY in fused raw text (outline paragraphs, nav text, body
        paragraphs) is a TITLE DONOR, never an admitter — this kills the
        figure-caption / Try-It / Example ordinal leaks, which all enter via
        raw-text splits.

        Title backfill is priority-wins (not first-seen): body non-apparatus
        heading > answer-key heading > outline-zone/nav fused split > fused
        body-paragraph split. A heading whose tail is bare apparatus
        ("N.M Exercises") admits the ORDINAL but donates NO title. Two shape
        belts guard admission: the minor <= ``_MAX_SECTION_MINOR`` ceiling and
        the per-major near-contiguity prune.

        Returns ``(entries, outline_zone_articles, title_sources)``. ``entries``
        is empty when the book declares no structural ordinal spine (wide-net
        fall-through, byte-identical to the guards-only path).
        """
        outline_articles = self._outline_zone_articles(soup)
        outline_ids = {id(a) for a in outline_articles}
        articles = soup.find_all("article", attrs={"role": "doc-chapter"})

        # Zone map — mirrors ``_collect_outline_occurrences``'s trailing-review
        # latch: outline-zone articles are ``outline``; once an article TITLE
        # trips the review/answer-key furniture predicate every following
        # article is ``answer_key``; everything else is ``body``.
        zone_of: Dict[int, str] = {}
        in_review = False
        for article in articles:
            th = article.find(["h1", "h2"])
            t = th.get_text(strip=True) if th else ""
            if _REVIEW_ZONE_RE.search(t) or _is_noncontent_heading(t):
                in_review = True
            if id(article) in outline_ids:
                zone_of[id(article)] = "outline"
            elif in_review:
                zone_of[id(article)] = "answer_key"
            else:
                zone_of[id(article)] = "body"

        # ordinal -> {"structural": bool, "titles": {tier: (source_label, title)}}
        cand: Dict[str, Dict[str, Any]] = {}

        def _cand(ordinal: str) -> Optional[Dict[str, Any]]:
            try:
                if int(ordinal.split(".")[1]) > _MAX_SECTION_MINOR:
                    return None
            except (ValueError, IndexError):  # pragma: no cover - regex-validated
                return None
            return cand.setdefault(ordinal, {"structural": False, "titles": {}})

        def _clean_title(title: str) -> Optional[str]:
            norm = re.sub(r"\s+", " ", title or "").strip()
            if len(norm) < 3 or not any(c.isalpha() for c in norm):
                return None
            low = norm.lower()
            if (
                low in _EOC_NUMBERED_BARE
                or _eoc_core_match(low)
                or low in _lexicon_apparatus_names_lower()
                or _is_noncontent_heading(norm)
            ):
                return None
            return norm

        def _donate(ordinal: str, tier: int, source_label: str, title: str) -> None:
            c = _cand(ordinal)
            if c is None:
                return
            clean = _clean_title(title)
            if clean and tier not in c["titles"]:
                c["titles"][tier] = (source_label, clean)

        def _admit(ordinal: str) -> None:
            c = _cand(ordinal)
            if c is not None:
                c["structural"] = True

        # (d) numbered headings document-wide — the ordinal must OPEN the
        # heading. Opening a heading element IS the structural evidence.
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            # SEMANTIK_BOX_TITLE_HEADINGS — presentational callout titles never
            # supply a section ordinal (anti-re-poisoning); skip unconditionally.
            if heading.has_attr('data-semantik-box-title'):
                continue
            text = re.sub(r"\s+", " ", heading.get_text(" ", strip=True))
            m = _NM_HEADING_RE.match(text)
            if not m:
                continue
            ordinal, tail = m.group(1), m.group(2).strip()
            _admit(ordinal)
            art = heading.find_parent("article", attrs={"role": "doc-chapter"})
            zone = (
                zone_of.get(id(art), "no-article") if art is not None else "no-article"
            )
            low = tail.lower().rstrip(":")
            first = low.split()[0] if low.split() else ""
            apparatus = (
                low in _EOC_NUMBERED_BARE
                or _eoc_core_match(low)
                or low in _lexicon_apparatus_names_lower()
                or first in _EOC_NUMBERED_BARE
                or not any(c.isalpha() for c in tail)
            )
            if apparatus:
                continue  # admits the ordinal but donates no title
            if zone in ("body", "no-article"):
                _donate(ordinal, _TITLE_TIER_BODY_HEADING, "body_heading", tail)
            elif zone == "answer_key":
                _donate(
                    ordinal, _TITLE_TIER_ANSWER_KEY_HEADING, "answer_key_heading", tail
                )
            else:
                _donate(ordinal, _TITLE_TIER_OUTLINE_FUSED, "outline_heading", tail)

        # (e) apparatus-banner ordinals anywhere in text ("N.M Exercises").
        banner_re = _eoc_banner_regex()
        for m in banner_re.finditer(soup.get_text(" ", strip=True)):
            _admit(m.group(1))

        # Title donors (raw-text splits) — NEVER admit, only backfill titles.
        #   (b) outline-zone fused / demoted paragraphs
        for article in outline_articles:
            for sec in article.find_all("section", recursive=False):
                if sec.find(["h1", "h2", "h3", "h4", "h5", "h6"]) is not None:
                    continue  # heading sections already harvested by (d)
                for ordinal, title in _split_fused_outline_entries(
                    sec.get_text(" ", strip=True)
                ):
                    _donate(ordinal, _TITLE_TIER_OUTLINE_FUSED, "outline_fused", title)
        #   (c) ``<nav class="toc">`` fused text
        nav = soup.find("nav", class_="toc")
        if nav is not None:
            for ordinal, title in _split_fused_outline_entries(
                nav.get_text(" ", strip=True)
            ):
                _donate(ordinal, _TITLE_TIER_OUTLINE_FUSED, "nav_fused", title)
        #   fused body-paragraph splits (last-resort tier) — only a paragraph
        #   that OPENS with an ordinal AND fuses several (a printed-outline
        #   reprint), never a single-ordinal mid-body sentence.
        for para in soup.find_all(["p", "li", "div"]):
            text = re.sub(r"\s+", " ", para.get_text(" ", strip=True))
            if not _NM_HEADING_RE.match(text):
                continue
            fused = _split_fused_outline_entries(text)
            if len(fused) < 2:
                continue
            for ordinal, title in fused:
                _donate(ordinal, _TITLE_TIER_BODY_FUSED, "body_fused", title)

        # Assemble admitted entries with priority-wins titles + provenance.
        entries: Dict[str, str] = {}
        title_sources: Dict[str, str] = {}
        for ordinal, c in cand.items():
            if not c["structural"]:
                continue
            source_label = "apparatus_only"
            title = ""
            for tier in sorted(c["titles"]):
                source_label, title = c["titles"][tier]
                break
            entries[ordinal] = title
            title_sources[ordinal] = source_label

        entries, title_sources = self._prune_outline_contiguity(
            entries, title_sources
        )

        ordered = sorted(entries, key=_ordinal_sort_key)
        entries = {o: entries[o] for o in ordered}
        title_sources = {o: title_sources[o] for o in ordered}
        return entries, outline_articles, title_sources

    @staticmethod
    def _article_section_elems(article: Tag) -> List[Tag]:
        """Section wrappers belonging to an article (direct children, else the
        trailing siblings up to the next doc-chapter article — the Wave-13
        sibling shape). Mirrors ``_collect_article_body``'s section discovery."""
        direct = article.find_all("section", recursive=False)
        if direct:
            return direct
        section_elems: List[Tag] = []
        sibling = article.next_sibling
        while sibling is not None:
            if isinstance(sibling, Tag):
                if (
                    sibling.name == "article"
                    and sibling.get("role") == "doc-chapter"
                ):
                    break
                if sibling.name == "section":
                    section_elems.append(sibling)
            sibling = sibling.next_sibling
        return section_elems

    def _collect_outline_occurrences(
        self,
        articles: List[Tag],
        outline_article_ids: set,
        entries: Dict[str, str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Find every heading-bearing section whose ``N.M`` heading matches a
        declared entry, tagged with its zone.

        Zones (document order, trailing-review latches): a section inside an
        outline-zone article is ``outline``; once an article TITLE trips the
        review/answer-key furniture predicate every following section is
        ``answer_key`` (OpenStax reprints openers there); everything else is
        ``body``. A match requires the ordinal AND a fuzzy title score >=
        ``_OUTLINE_TITLE_MATCH_RATIO`` (OCR tolerance).
        """
        occurrences: Dict[str, List[Dict[str, Any]]] = {o: [] for o in entries}
        in_review = False
        for article in articles:
            title_heading = article.find(["h1", "h2"])
            title = title_heading.get_text(strip=True) if title_heading else ""
            if _REVIEW_ZONE_RE.search(title) or _is_noncontent_heading(title):
                in_review = True
            if id(article) in outline_article_ids:
                zone = "outline"
            elif in_review:
                zone = "answer_key"
            else:
                zone = "body"
            for elem in self._article_section_elems(article):
                heading = elem.find(["h1", "h2", "h3", "h4", "h5", "h6"])
                if heading is None:
                    continue
                heading_text = re.sub(
                    r"\s+", " ", heading.get_text(" ", strip=True)
                )
                m = _NM_HEADING_RE.match(heading_text)
                if m:
                    ordinal, title_text = m.group(1), m.group(2)
                    if ordinal not in entries or not entries[ordinal]:
                        continue
                    ratio = difflib.SequenceMatcher(
                        None,
                        _normalize_outline_text(title_text),
                        _normalize_outline_text(entries[ordinal]),
                    ).ratio()
                    if ratio < _OUTLINE_TITLE_MATCH_RATIO:
                        continue
                    matched_ordinal = ordinal
                else:
                    # Cheap extension (spec item 4): an UNNUMBERED heading whose
                    # whole normalized text fuzzy-matches a declared entry title
                    # (>= _OUTLINE_TITLE_MATCH_RATIO) counts as a body occurrence
                    # — recovers an opener whose leading ordinal OCR-dropped.
                    norm_heading = _normalize_outline_text(heading_text)
                    if not norm_heading:
                        continue
                    matched_ordinal = None
                    best = _OUTLINE_TITLE_MATCH_RATIO
                    for cand_ord, cand_title in entries.items():
                        if not cand_title:
                            continue
                        r = difflib.SequenceMatcher(
                            None,
                            norm_heading,
                            _normalize_outline_text(cand_title),
                        ).ratio()
                        if r >= best:
                            best = r
                            matched_ordinal = cand_ord
                    if matched_ordinal is None:
                        continue
                occurrences[matched_ordinal].append(
                    {
                        "zone": zone,
                        "elem": elem,
                        "heading_id": heading.get("id"),
                        "text_mass": len(elem.get_text(" ", strip=True)),
                    }
                )
        return occurrences

    def _build_chapters_outline_anchored(
        self,
        soup: BeautifulSoup,
        articles: List[Tag],
        diag: Dict[str, Any],
    ) -> Optional[List[ChapterStructure]]:
        """Regroup the article stream onto the declared ``N.M`` outline spine.

        Returns ``None`` (fall through to the Package-1 chapters) when the book
        declares no harvestable outline. Otherwise emits ONE section per
        declared entry, grouped into chapters by the ordinal's major number,
        with every non-matching heading + orphan block demoted (never dropped)
        to the nearest surviving section or the chapter lead.
        """
        entries, outline_articles, title_sources = self._harvest_outline(soup)
        if not entries:
            return None

        outline_article_ids = {id(a) for a in outline_articles}
        occurrences = self._collect_outline_occurrences(
            articles, outline_article_ids, entries
        )

        # Declared chapter title: the file h1 when a single major is declared
        # (its real title, e.g. "Foundations"); a synthesized "Chapter N"
        # otherwise. Ordinals grouped by major -> one chapter each.
        h1 = soup.find("h1")
        h1_text = h1.get_text(strip=True) if h1 else ""
        majors = sorted({int(o.split(".")[0]) for o in entries})
        single_major = len(majors) == 1

        def _chapter_title(major: int) -> str:
            if single_major and h1_text and not _is_noncontent_heading(h1_text):
                return h1_text
            return f"Chapter {major}"

        chapters_by_major: Dict[int, ChapterStructure] = {}
        for major in majors:
            chapters_by_major[major] = ChapterStructure(
                id=f"ch{major}",
                heading_level=2,
                heading_text=_chapter_title(major),
                heading_id=None,
                explicit_objectives=[],
                content_blocks=[],
                sections=[],
            )

        # One SectionStructure per declared entry; pick the surviving
        # occurrence (best zone), map its element for the content walk.
        chosen_by_elem: Dict[int, SectionStructure] = {}
        unmatched_declared: List[Dict[str, str]] = []
        matched_sections = 0
        section_counters: Dict[int, int] = {m: 0 for m in majors}

        for ordinal in sorted(entries, key=_ordinal_sort_key):
            major = int(ordinal.split(".")[0])
            chapter = chapters_by_major[major]
            occ_list = occurrences.get(ordinal, [])
            chosen = None
            if occ_list:
                # Best zone first (body > outline > answer_key); among same-zone
                # candidates prefer the element with the greatest descendant
                # text mass (a real opener >> an outline stub) — spec item 5.
                best_rank = min(_ZONE_RANK[o["zone"]] for o in occ_list)
                same_zone = [
                    o for o in occ_list if _ZONE_RANK[o["zone"]] == best_rank
                ]
                chosen = max(same_zone, key=lambda o: o.get("text_mass", 0))
            best_zone = chosen["zone"] if chosen else "declared_only"

            section_counters[major] += 1
            section = SectionStructure(
                id=f"{chapter.id}_s{section_counters[major]}",
                heading_level=3,
                heading_text=" ".join(
                    part for part in (ordinal, entries[ordinal]) if part
                ),
                heading_id=chosen["heading_id"] if chosen else None,
                content_blocks=[],
                subsections=[],
            )
            section.matched_zone = best_zone  # provenance (serialized below)
            chapter.sections.append(section)

            if chosen is not None:
                matched_sections += 1
                chosen_by_elem[id(chosen["elem"])] = section
            if best_zone != "body":
                # Loss signal: no real body opener. The re-conversion work list
                # for packages 4+5 (includes outline-only + answer-key-only +
                # declared-only entries).
                unmatched_declared.append(
                    {
                        "ordinal": ordinal,
                        "title": entries[ordinal],
                        "found_zone": best_zone,
                    }
                )

        # Content walk (document order): fill each surviving section with its
        # chosen element's blocks; demote everything else to the current
        # surviving section (or the chapter lead when nothing is open yet).
        first_chapter = chapters_by_major[majors[0]]
        current_section: Optional[SectionStructure] = None
        demoted_headings = 0
        for article in articles:
            lead_blocks = self._extract_chapter_content(article, None)
            target = (
                current_section.content_blocks
                if current_section is not None
                else first_chapter.content_blocks
            )
            target.extend(lead_blocks)
            for elem in self._article_section_elems(article):
                blocks = self.block_classifier.classify_section(elem)
                section = chosen_by_elem.get(id(elem))
                if section is not None:
                    section.content_blocks.extend(blocks)
                    current_section = section
                    continue
                heading = elem.find(["h1", "h2", "h3", "h4", "h5", "h6"])
                if heading is not None and heading.get_text(strip=True):
                    demoted_headings += 1
                target = (
                    current_section.content_blocks
                    if current_section is not None
                    else first_chapter.content_blocks
                )
                target.extend(blocks)

        chapters = [chapters_by_major[m] for m in majors]

        # Diagnostics — improve the (previously fused-blind) declared estimate
        # and record the outline-anchor counters + loss list.
        diag["declared_section_estimate"] = len(entries)
        diag["sections_built"] = sum(len(c.sections) for c in chapters)
        diag["outline_anchor"] = {
            "declared_entries": len(entries),
            "matched_sections": matched_sections,
            "demoted_headings": demoted_headings,
            "unmatched_declared": unmatched_declared,
            "title_sources": title_sources,
        }

        if unmatched_declared:
            logger.warning(
                "SemanticStructureExtractor: STRUCTURE_DECLARED_SECTION_MISSING "
                "— %d declared section(s) have no body opener (outline-only / "
                "answer-key reprint / absent): %s. These are the re-conversion "
                "work list.",
                len(unmatched_declared),
                ", ".join(u["ordinal"] for u in unmatched_declared),
            )
        return chapters

    def _build_section_from_element(
        self,
        soup: BeautifulSoup,
        section_elem: Tag,
        parent_id: str,
        section_num: int,
        guarded: bool = False,
        diag: Optional[Dict[str, int]] = None,
    ) -> SectionStructure:
        """Build a ``SectionStructure`` directly from a DOM ``<section>``.

        SemantiK output emits flat ``<section>`` wrappers rather
        than nesting them under article children, so we read heading
        info off the section itself.

        Package 1 (``guarded=True``): nested ``<section>`` children run
        through the 1b/1c grouping too, so a headingless/noncontent-heading
        nested wrapper regroups instead of minting an empty subsection.
        """
        section_id = f"{parent_id}_s{section_num}"
        heading_tag = section_elem.find(
            ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        )
        heading_text = (
            heading_tag.get_text(strip=True) if heading_tag else ''
        ) or ''
        heading_id = heading_tag.get('id') if heading_tag else None
        try:
            heading_level = int(heading_tag.name.lstrip('h')) if heading_tag else 3
        except ValueError:
            heading_level = 3

        content_blocks = self.block_classifier.classify_section(section_elem)
        # Nested <section> children become subsections.
        subsections: List[SectionStructure] = []
        nested_elems = section_elem.find_all('section', recursive=False)
        if guarded:
            subsections, _ = self._group_sections_from_iter(
                soup, nested_elems, section_id, content_blocks, 0,
                diag if diag is not None else {},
            )
        else:
            sub_counter = 0
            for nested in nested_elems:
                sub_counter += 1
                subsections.append(
                    self._build_section_from_element(
                        soup, nested, section_id, sub_counter,
                    )
                )

        return SectionStructure(
            id=section_id,
            heading_level=heading_level,
            heading_text=heading_text,
            heading_id=heading_id,
            content_blocks=content_blocks,
            subsections=subsections,
        )

    def _build_chapter(
        self,
        soup: BeautifulSoup,
        hierarchy: HeadingHierarchy,
        node: HeadingNode,
        chapter_num: int
    ) -> ChapterStructure:
        """Build a single chapter structure."""
        chapter_id = f"ch{chapter_num}"

        # Get the section element for this heading
        section_elem = node.section_element
        if not section_elem and node.element_id:
            heading = soup.find(id=node.element_id)
            if heading:
                section_elem = heading.find_parent('section')

        # Extract explicit objectives if present
        explicit_objectives = self._extract_explicit_objectives(section_elem)

        # Extract content blocks for this chapter (before subsections)
        content_blocks = self._extract_chapter_content(section_elem, node)

        # Build section structure for children
        sections = []
        section_counter = 0
        for child_id in node.children:
            child_node = hierarchy.get_node(child_id)
            if child_node and not _is_noncontent_heading(child_node.text):
                section_counter += 1
                section = self._build_section(
                    soup, hierarchy, child_node,
                    chapter_id, section_counter
                )
                sections.append(section)

        return ChapterStructure(
            id=chapter_id,
            heading_level=node.level,
            heading_text=node.text,
            heading_id=node.element_id,
            explicit_objectives=explicit_objectives,
            content_blocks=content_blocks,
            sections=sections
        )

    def _build_section(
        self,
        soup: BeautifulSoup,
        hierarchy: HeadingHierarchy,
        node: HeadingNode,
        parent_id: str,
        section_num: int
    ) -> SectionStructure:
        """Build a section structure."""
        section_id = f"{parent_id}_s{section_num}"

        # Get section element
        section_elem = node.section_element
        if not section_elem and node.element_id:
            heading = soup.find(id=node.element_id)
            if heading:
                section_elem = heading.find_parent('section')

        # Extract content blocks
        content_blocks = []
        if section_elem:
            content_blocks = self.block_classifier.classify_section(section_elem)

        # Build subsections
        subsections = []
        subsection_counter = 0
        for child_id in node.children:
            child_node = hierarchy.get_node(child_id)
            if child_node:
                subsection_counter += 1
                subsection = self._build_section(
                    soup, hierarchy, child_node,
                    section_id, subsection_counter
                )
                subsections.append(subsection)

        return SectionStructure(
            id=section_id,
            heading_level=node.level,
            heading_text=node.text,
            heading_id=node.element_id,
            content_blocks=content_blocks,
            subsections=subsections
        )

    def _extract_chapter_content(
        self,
        section_elem: Optional[Tag],
        node: HeadingNode
    ) -> List[ContentBlock]:
        """Extract content blocks that belong directly to a chapter (not in subsections)."""
        if not section_elem:
            return []

        # Find content that appears before the first subsection
        content_blocks = []
        classifier = ContentBlockClassifier()

        for child in section_elem.children:
            if isinstance(child, Tag):
                # Stop at subsections
                if child.name == 'section':
                    break

                # Skip the heading itself
                if child.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    continue

                block = classifier._classify_element(child)
                if block:
                    content_blocks.append(block)

        return content_blocks

    def _extract_explicit_objectives(self, section_elem: Optional[Tag]) -> List[Dict[str, str]]:
        """Extract explicitly stated learning objectives from a section."""
        if not section_elem:
            return []

        objectives = []

        # Look for objectives section
        objectives_section = section_elem.find(
            'section',
            attrs={'aria-labelledby': lambda x: x and 'objective' in x.lower()}
        )

        if not objectives_section:
            # Look for heading with "objectives" or "learning objectives"
            for heading in section_elem.find_all(['h2', 'h3', 'h4']):
                if heading.has_attr('data-semantik-box-title'):
                    continue
                if 'objective' in heading.get_text().lower():
                    objectives_section = heading.find_parent('section') or heading.parent
                    break

        if objectives_section:
            # Find the list of objectives
            obj_list = objectives_section.find(['ul', 'ol'])
            if obj_list:
                for li in obj_list.find_all('li'):
                    objectives.append({
                        "text": li.get_text(strip=True),
                        "source": "objectives_section"
                    })
        else:
            # Look for patterns like "After completing this chapter, you will be able to:"
            text = section_elem.get_text()
            patterns = [
                r'(?:After|Upon|By the end)[^:]+:\s*([^.]+\.(?:\s*[^.]+\.)*)',
                r'(?:you will be able to|students will|learners will)[^:]*:\s*([^.]+\.(?:\s*[^.]+\.)*)',
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Split by common delimiters
                    obj_text = match.group(1)
                    for obj in re.split(r'[;•\n]', obj_text):
                        obj = obj.strip()
                        if obj and len(obj) > 10:
                            objectives.append({
                                "text": obj,
                                "source": "inline"
                            })

        return objectives

    def _extract_all_concepts(self, chapters: List[ChapterStructure]) -> Dict[str, Any]:
        """Extract all concepts from chapters."""
        all_definitions = []
        all_key_terms = []
        all_procedures = []
        all_examples = []

        for chapter in chapters:
            # Extract from chapter content blocks
            self._extract_concepts_from_blocks(
                chapter.content_blocks,
                chapter.id,
                None,
                all_definitions,
                all_key_terms,
                all_procedures,
                all_examples
            )

            # Extract from sections
            for section in chapter.sections:
                self._extract_concepts_from_section(
                    section,
                    chapter.id,
                    all_definitions,
                    all_key_terms,
                    all_procedures,
                    all_examples
                )

        return {
            "definitions": all_definitions,
            "keyTerms": all_key_terms,
            "procedures": all_procedures,
            "examples": all_examples
        }

    def _extract_concepts_from_section(
        self,
        section: SectionStructure,
        chapter_id: str,
        all_definitions: List,
        all_key_terms: List,
        all_procedures: List,
        all_examples: List
    ) -> None:
        """Recursively extract concepts from a section."""
        self._extract_concepts_from_blocks(
            section.content_blocks,
            chapter_id,
            section.id,
            all_definitions,
            all_key_terms,
            all_procedures,
            all_examples
        )

        for subsection in section.subsections:
            self._extract_concepts_from_section(
                subsection,
                chapter_id,
                all_definitions,
                all_key_terms,
                all_procedures,
                all_examples
            )

    def _extract_concepts_from_blocks(
        self,
        blocks: List[ContentBlock],
        chapter_id: str,
        section_id: Optional[str],
        all_definitions: List,
        all_key_terms: List,
        all_procedures: List,
        all_examples: List
    ) -> None:
        """Extract concepts from a list of content blocks."""
        for block in blocks:
            # Add definitions
            for defn in block.definitions:
                all_definitions.append({
                    "term": defn.term,
                    "definition": defn.definition,
                    "sourceType": defn.source_type,
                    "chapterId": chapter_id,
                    "sectionId": section_id
                })

            # Add key terms
            for term in block.key_terms:
                all_key_terms.append({
                    "term": term.term,
                    "context": term.context,
                    "emphasisType": term.emphasis_type,
                    "chapterId": chapter_id,
                    "sectionId": section_id
                })

            # Check for procedures (ordered lists with multiple steps)
            if block.block_type == BlockType.LIST_ORDERED:
                min_steps = self.config.get('min_procedure_steps', 2)
                if len(block.list_items) >= min_steps:
                    # Check if it looks like a procedure
                    if self._looks_like_procedure(block.list_items):
                        all_procedures.append({
                            "name": self._infer_procedure_name(block),
                            "steps": block.list_items,
                            "context": "",
                            "chapterId": chapter_id,
                            "sectionId": section_id
                        })

            # Check for examples
            if block.block_type == BlockType.EXAMPLE:
                min_words = self.config.get('min_example_words', 20)
                if block.word_count >= min_words:
                    all_examples.append({
                        "title": None,
                        "content": block.content,
                        "relatedConcept": None,
                        "chapterId": chapter_id,
                        "sectionId": section_id
                    })

    def _looks_like_procedure(self, items: List[str]) -> bool:
        """Determine if a list looks like a procedure."""
        # Check for action verbs at start of items
        action_patterns = [
            r'^(click|select|enter|type|open|close|save|create|delete|configure|set|add|remove)',
            r'^(first|next|then|finally|after|before)',
            r'^\d+[.)]\s*',
        ]

        action_count = 0
        for item in items:
            for pattern in action_patterns:
                if re.match(pattern, item.lower()):
                    action_count += 1
                    break

        return action_count >= len(items) / 2

    def _infer_procedure_name(self, block: ContentBlock) -> str:
        """Infer a name for a procedure from its context."""
        # Try to find a preceding heading or strong text
        return "Procedure"

    def _extract_review_questions(
        self,
        soup: BeautifulSoup,
        chapters: List[ChapterStructure]
    ) -> List[ReviewQuestion]:
        """Extract review questions from the document."""
        questions = []

        # Look for review sections
        review_sections = soup.find_all(
            'section',
            attrs={'aria-labelledby': lambda x: x and any(
                term in x.lower() for term in ['review', 'question', 'quiz', 'assessment']
            )}
        )

        for review_section in review_sections:
            # Find the parent chapter
            chapter_id = self._find_parent_chapter_id(review_section, chapters)

            # Extract questions from ordered list
            for ol in review_section.find_all('ol'):
                for li in ol.find_all('li'):
                    question_text = li.get_text(strip=True)
                    bloom_level = self._infer_bloom_level(question_text)

                    questions.append(ReviewQuestion(
                        question=question_text,
                        chapter_id=chapter_id,
                        section_id=None,
                        bloom_level=bloom_level
                    ))

        return questions

    def _find_parent_chapter_id(
        self,
        element: Tag,
        chapters: List[ChapterStructure]
    ) -> str:
        """Find the chapter ID that contains an element."""
        # Simple heuristic: find the nearest h2 ancestor
        parent = element
        while parent:
            h2 = parent.find_previous('h2')
            if h2:
                h2_text = h2.get_text(strip=True).lower()
                for chapter in chapters:
                    if chapter.heading_text.lower() in h2_text or h2_text in chapter.heading_text.lower():
                        return chapter.id
            parent = parent.parent

        return chapters[0].id if chapters else "ch1"

    def _infer_bloom_level(self, question_text: str) -> Optional[str]:
        """Infer Bloom's taxonomy level from question text."""
        question_lower = question_text.lower()

        for level, patterns in self.BLOOM_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return level

        return None


def extract_textbook_structure(file_path: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to extract textbook structure from a file.

    Args:
        file_path: Path to the HTML or Markdown file
        config_path: Optional path to configuration file

    Returns:
        Dictionary conforming to textbook_structure.schema.json
    """
    extractor = SemanticStructureExtractor(config_path)
    return extractor.extract_file(file_path)


def extract_for_presentation(file_path: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to extract and transform to presentation format.

    Args:
        file_path: Path to the HTML or Markdown file
        config_path: Optional path to configuration file

    Returns:
        Dictionary conforming to presentation_schema.json
    """
    extractor = SemanticStructureExtractor(config_path)
    path = Path(file_path)
    with open(path, encoding='utf-8') as f:
        content = f.read()
    return extractor.extract_for_presentation(content, str(path))


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract semantic structure from HTML or Markdown content'
    )
    parser.add_argument('input_file', help='Path to the HTML or Markdown file')
    parser.add_argument(
        '-c', '--config',
        help='Path to configuration file',
        default=None
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: stdout)',
        default=None
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty print JSON output'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['auto', 'html', 'markdown'],
        default='auto',
        help='Input format (default: auto-detect)'
    )
    parser.add_argument(
        '-m', '--mode',
        choices=['basic', 'profiled', 'presentation'],
        default='basic',
        help='Extraction mode: basic, profiled (with concept graph), or presentation (full transform)'
    )

    args = parser.parse_args()

    extractor = SemanticStructureExtractor(args.config)

    # Read input file
    path = Path(args.input_file)
    with open(path, encoding='utf-8') as f:
        content = f.read()

    # Extract based on mode
    if args.mode == 'presentation':
        result = extractor.extract_for_presentation(content, str(path), args.format)
    elif args.mode == 'profiled':
        result = extractor.extract_with_profiling(content, str(path), args.format)
    else:
        result = extractor.extract(content, str(path), args.format)

    # Output
    indent = 2 if args.pretty else None
    output = json.dumps(result, indent=indent, ensure_ascii=False)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Output written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
